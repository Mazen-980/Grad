// Cow Detection dashboard front end. Boxes are drawn on a canvas over the
// original video from per-frame detections produced by the Python backend.
"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const chart = $("chart"), cctx = chart.getContext("2d");

const S = {
  videos: [], videoPath: null, meta: null,
  frames: [], times: [],          // processed frames (sorted by time)
  tracks: new Map(),              // id -> {id, first_t, last_t, frames, conf_sum}
  settings: null, summary: null, source: null,  // "job" | "run" | null
  jobId: null, poll: null, previewTimer: null, runDir: null,
  minTrackFrames: 5, activeTrack: null,
  zones: null, zoneScope: null,   // zones of the selected video (zones.py format)
  edit: null,                     // zone editor state {zones: [...], type, points: [...]}
};
const ZONE_COL = { pen: "rgba(90,200,90,", feed: "rgba(255,165,0,", water: "rgba(0,140,255," };
const BEH = ["eating", "drinking", "lying", "walking", "standing"];

// ---------------------------------------------------------------- helpers
const fmt = (t) => {
  if (!isFinite(t)) return "--:--.-";
  const tenths = Math.round(t * 10), m = Math.floor(tenths / 600), s = (tenths - m * 600) / 10;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
};
const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}
function banner(msg) { const b = $("banner"); b.textContent = msg || ""; b.classList.toggle("hidden", !msg); }

// Same colour as the Python exporter: OpenCV HSV(h=(id*47)%180, s=200, v=255).
function idColor(id) {
  if (id == null) return "rgb(255,200,0)";
  const h = ((id * 47) % 180) * 2, s = 200 / 255, v = 1;
  const f = (n) => { const k = (n + h / 60) % 6; return v - v * s * Math.max(0, Math.min(k, 4 - k, 1)); };
  return `rgb(${Math.round(f(5) * 255)},${Math.round(f(3) * 255)},${Math.round(f(1) * 255)})`;
}

// ---------------------------------------------------------------- data
function resetData() {
  S.frames = []; S.times = []; S.tracks = new Map(); S.summary = null; S.source = null; S.runDir = null;
  S.activeTrack = null;
}
function addFrames(frames) {
  for (const fr of frames) {
    S.frames.push(fr); S.times.push(fr.t);
    for (const [id, conf] of fr.d) {
      if (id == null) continue;
      let tr = S.tracks.get(id);
      if (!tr) { tr = { id, first_t: fr.t, last_t: fr.t, frames: 0, conf_sum: 0, raws: new Set(), beh: {} }; S.tracks.set(id, tr); }
      tr.last_t = fr.t; tr.frames++; tr.conf_sum += conf;
      if (fr.b) { const k = fr.d.findIndex((x) => x[0] === id); const b = fr.b[k]; if (b) tr.beh[b] = (tr.beh[b] || 0) + 1; }
      const d = fr.d.find((x) => x[0] === id); if (d && d.length > 6 && d[6] != null) tr.raws.add(d[6]);
    }
  }
}
function frameAt(t) {
  const T = S.times; if (!T.length) return null;
  const fps = (S.meta && S.meta.fps) || 25, stride = (S.settings && S.settings.stride) || 1;
  const target = t + 0.5 / fps;
  let lo = 0, hi = T.length - 1, ans = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (T[mid] <= target) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
  if (ans < 0) return null;
  if (t - T[ans] > (stride + 1.5) / fps) return null;  // not processed (yet) at this time
  return S.frames[ans];
}

// ---------------------------------------------------------------- overlay
function videoRect() {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  const vw = video.videoWidth || (S.meta && S.meta.width) || 16, vh = video.videoHeight || (S.meta && S.meta.height) || 9;
  const sc = Math.min(cw / vw, ch / vh);
  return { x: (cw - vw * sc) / 2, y: (ch - vh * sc) / 2, sc, vw, vh };
}
function toCanvas() {
  // video-pixel (original resolution) -> canvas CSS pixels
  const r = videoRect();
  const sx = r.sc * (r.vw / ((S.meta && S.meta.width) || r.vw)), sy = r.sc * (r.vh / ((S.meta && S.meta.height) || r.vh));
  return { r, sx, sy, f: (x, y) => [r.x + x * sx, r.y + y * sy], inv: (X, Y) => [(X - r.x) / sx, (Y - r.y) / sy] };
}
function zoneScale(z) {
  const [iw, ih] = (z && z.image_size) || [0, 0];
  const w = (S.meta && S.meta.width) || iw, h = (S.meta && S.meta.height) || ih;
  return [iw ? w / iw : 1, ih ? h / ih : 1];
}
function drawZones(c, zones, extra) {
  if (!zones) return;
  const T = toCanvas(), [kx, ky] = zoneScale(zones);
  for (const z of zones.zones || []) {
    if (!z.points.length) continue;
    c.beginPath();
    z.points.forEach(([x, y], i) => { const [X, Y] = T.f(x * kx, y * ky); i ? c.lineTo(X, Y) : c.moveTo(X, Y); });
    c.closePath();
    const col = ZONE_COL[z.type] || "rgba(255,255,255,";
    if (z.type !== "pen") { c.fillStyle = col + "0.18)"; c.fill(); }
    c.strokeStyle = col + "0.9)"; c.lineWidth = 2; c.setLineDash(z.type === "pen" ? [6, 4] : []); c.stroke(); c.setLineDash([]);
    const [lx, ly] = T.f(z.points[0][0] * kx, z.points[0][1] * ky);
    c.font = "600 11px -apple-system, sans-serif"; c.fillStyle = col + "1)"; c.fillText(z.name || z.type, lx + 4, ly + 4);
  }
  if (extra && extra.points.length) {  // shape being drawn in the editor
    c.beginPath();
    extra.points.forEach(([x, y], i) => { const [X, Y] = T.f(x, y); i ? c.lineTo(X, Y) : c.moveTo(X, Y); });
    c.strokeStyle = (ZONE_COL[extra.type] || "rgba(255,255,255,") + "1)"; c.lineWidth = 2; c.stroke();
    for (const [x, y] of extra.points) { const [X, Y] = T.f(x, y); c.fillStyle = "#fff"; c.fillRect(X - 3, Y - 3, 6, 6); }
  }
}
function drawOverlay(fr) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if ($("tZones").checked && !S.edit) drawZones(ctx, S.zones);
  if (!fr || !$("tOverlay").checked) return;
  const r = videoRect();
  // The metadata is in the original video's pixel space; the browser preview has the same frames.
  const sx = r.sc * (r.vw / ((S.meta && S.meta.width) || r.vw)), sy = r.sc * (r.vh / ((S.meta && S.meta.height) || r.vh));
  ctx.font = "600 12px -apple-system, Segoe UI, sans-serif"; ctx.textBaseline = "top";
  for (const [k, [id, conf, x1, y1, x2, y2]] of fr.d.entries()) {
    const X = r.x + x1 * sx, Y = r.y + y1 * sy, W = (x2 - x1) * sx, H = (y2 - y1) * sy;
    const col = idColor(id), active = S.activeTrack != null && S.activeTrack === id;
    ctx.lineWidth = active ? 4 : 2; ctx.strokeStyle = col; ctx.strokeRect(X, Y, W, H);
    if ($("tLabels").checked) {
      const beh = fr.b && fr.b[k] && fr.b[k] !== "standing" ? ` · ${fr.b[k]}` : "";
      const label = `${id != null ? "ID " + id : "cow"} · ${Math.round(conf * 100)}%${beh}`;
      const tw = ctx.measureText(label).width + 8, ly = Math.max(0, Y - 17);
      ctx.fillStyle = col; ctx.fillRect(X, ly, tw, 17);
      ctx.fillStyle = "#151515"; ctx.fillText(label, X + 4, ly + 2.5);
    }
  }
}

// ---------------------------------------------------------------- stats
let lastStatsT = -1;
function updateStats(force) {
  const t = video.currentTime || 0;
  const fr = frameAt(t);
  drawOverlay(fr);
  drawChart();
  if (!force && Math.abs(t - lastStatsT) < 0.03) return;
  lastStatsT = t;
  const dur = video.duration || (S.meta && S.meta.duration) || 0;
  $("timeLbl").textContent = `${fmt(t)} / ${fmt(dur)}`;
  $("sPos").textContent = `${fmt(t)} / ${fmt(dur)}  (frame ${Math.round(t * ((S.meta && S.meta.fps) || 25))})`;
  if (!document.activeElement || document.activeElement !== $("seek")) $("seek").value = dur ? Math.round((t / dur) * 1000) : 0;

  const has = S.frames.length > 0;
  const badge = $("frameBadge");
  if (has && !fr) { badge.textContent = "not processed at this time"; badge.classList.remove("hidden"); }
  else if (fr) { badge.textContent = `frame ${fr.f}`; badge.classList.remove("hidden"); }
  else badge.classList.add("hidden");

  $("sVisible").textContent = fr ? fr.n : "–";
  $("sConf").textContent = fr && fr.d.length ? pct(fr.d.reduce((a, d) => a + d[1], 0) / fr.d.length) : "–";
  if (fr && fr.b) {
    $("sBeh").innerHTML = BEH.map((b) => `<span class="chip ${b}">${b}<b>${fr.b.filter((x) => x === b).length}</b></span>`).join("");
  } else {
    $("sBeh").textContent = has ? (S.summary && S.summary.behaviour === null ? "not computed for this run" : "–") : "–";
  }
  const tracking = S.settings && S.settings.tracker !== "none";
  if (has && tracking) {
    let upto = 0, stable = 0;
    for (const tr of S.tracks.values()) { if (tr.first_t <= t + 1e-6) upto++; if (tr.frames >= S.minTrackFrames) stable++; }
    $("sUnique").textContent = `${upto} / ${S.tracks.size}`;
    const rl = S.summary && S.summary.relink;
    $("sStable").textContent = `so far / whole run · ${stable} stable (≥${S.minTrackFrames} frames)` +
      (rl ? ` · ${rl.raw_ids} before re-linking${rl.switch_splits && rl.switch_splits.length ? `, ${rl.switch_splits.length} switch(es) split` : ""}` : "");
  } else {
    $("sUnique").textContent = has ? "off" : "–";
    $("sStable").textContent = has ? "tracking disabled for this run" : "";
  }
  // Rows of the table that are visible at the playhead get highlighted.
  const ids = new Set(fr ? fr.d.map((d) => d[0]) : []);
  for (const row of $("tracksTbl").tBodies[0].rows) row.classList.toggle("active", ids.has(+row.dataset.id));
}
function updateRunStats(st) {
  // st: from job status or summary.stats
  if (!st) { $("sConfAll").textContent = ""; $("sFps").textContent = "–"; $("sInfer").textContent = ""; return; }
  $("sConfAll").textContent = st.avg_conf != null ? `whole run avg ${pct(st.avg_conf)}` : "";
  const fps = st.proc_fps ?? st.processing_fps, ms = st.infer_ms ?? st.avg_inference_ms;
  $("sFps").textContent = fps ? `${fps.toFixed(1)} fps` : "–";
  const dev = S.meta && S.meta.device ? ` on ${S.meta.device}` : "";
  $("sInfer").textContent = ms ? `${ms.toFixed(0)} ms / frame model time${dev}` : "";
}
function loop() { updateStats(false); requestAnimationFrame(loop); }

// ---------------------------------------------------------------- chart
function drawChart() {
  const dpr = window.devicePixelRatio || 1, w = chart.clientWidth, h = chart.clientHeight;
  if (chart.width !== Math.round(w * dpr) || chart.height !== Math.round(h * dpr)) { chart.width = Math.round(w * dpr); chart.height = Math.round(h * dpr); }
  cctx.setTransform(dpr, 0, 0, dpr, 0, 0); cctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const C = (n) => css.getPropertyValue(n).trim();
  const padL = 30, padR = 8, padT = 8, padB = 20, W = w - padL - padR, H = h - padT - padB;
  const dur = video.duration || (S.meta && S.meta.duration) || 1;
  const maxN = Math.max(1, ...S.frames.map((f) => f.n));
  const yMax = Math.ceil((maxN + 1) / 4) * 4;
  cctx.font = "11px -apple-system, sans-serif"; cctx.fillStyle = C("--muted"); cctx.strokeStyle = C("--line"); cctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4, y = padT + H - (v / yMax) * H;
    cctx.beginPath(); cctx.moveTo(padL, y); cctx.lineTo(padL + W, y); cctx.stroke();
    cctx.textAlign = "right"; cctx.textBaseline = "middle"; cctx.fillText(Math.round(v), padL - 6, y);
  }
  cctx.textAlign = "center"; cctx.textBaseline = "top";
  for (let i = 0; i <= 5; i++) cctx.fillText(fmt((dur * i) / 5).replace(/\.\d$/, ""), padL + (W * i) / 5, padT + H + 5);
  if (!S.frames.length) {
    cctx.fillText("No detection data yet", padL + W / 2, padT + H / 2 - 6);
  } else {
    // Bucket per horizontal pixel (max count) so long videos stay readable.
    const buckets = new Array(Math.max(1, Math.floor(W))).fill(-1);
    for (const f of S.frames) {
      const b = Math.min(buckets.length - 1, Math.floor((f.t / dur) * buckets.length));
      if (f.n > buckets[b]) buckets[b] = f.n;
    }
    const X = (i) => padL + i, Y = (n) => padT + H - (n / yMax) * H;
    cctx.beginPath(); let started = false, lastX = padL;
    buckets.forEach((n, i) => { if (n < 0) return; if (!started) { cctx.moveTo(X(i), Y(n)); started = true; } else cctx.lineTo(X(i), Y(n)); lastX = X(i); });
    cctx.strokeStyle = C("--chart"); cctx.lineWidth = 1.5; cctx.stroke();
    cctx.lineTo(lastX, padT + H); cctx.lineTo(padL, padT + H); cctx.closePath(); cctx.fillStyle = C("--chart-fill"); cctx.fill();
  }
  const px = padL + (Math.min(video.currentTime || 0, dur) / dur) * W;
  cctx.strokeStyle = C("--playhead"); cctx.lineWidth = 1.5; cctx.beginPath(); cctx.moveTo(px, padT); cctx.lineTo(px, padT + H); cctx.stroke();
}
chart.addEventListener("click", (e) => {
  const r = chart.getBoundingClientRect(), padL = 30, padR = 8, W = r.width - padL - padR;
  const dur = video.duration || (S.meta && S.meta.duration); if (!dur) return;
  video.currentTime = Math.max(0, Math.min(dur, ((e.clientX - r.left - padL) / W) * dur));
});

// ---------------------------------------------------------------- results table / summary
function renderTracks() {
  const tb = $("tracksTbl").tBodies[0]; tb.innerHTML = "";
  const tracks = [...S.tracks.values()].sort((a, b) => a.id - b.id);
  if (!tracks.length) {
    const msg = !S.frames.length ? "No results yet." : (S.settings && S.settings.tracker === "none") ? "Tracking was disabled for this run." : "No cows detected.";
    tb.innerHTML = `<tr><td colspan="10">${msg}</td></tr>`; return;
  }
  for (const tr of tracks) {
    const row = tb.insertRow(); row.dataset.id = tr.id;
    if (tr.frames < S.minTrackFrames) row.classList.add("short");
    row.innerHTML = `<td><span class="swatch" style="background:${idColor(tr.id)}"></span>${tr.id}</td>
      <td>${fmt(tr.first_t)}</td><td>${fmt(tr.last_t)}</td><td>${tr.frames}</td><td>${pct(tr.conf_sum / tr.frames)}</td>
      <td>${tr.raws.size > 1 ? [...tr.raws].sort((a, b) => a - b).join(" + ") : (tr.raws.size ? "–" : "")}</td>
      ${["eating", "drinking", "lying"].map((b) => `<td>${tr.beh[b] ? fmt(tr.beh[b] * sampleDt()) : "–"}</td>`).join("")}
      <td><button class="btn">Show</button></td>`;
    row.querySelector("button").onclick = () => { S.activeTrack = tr.id; video.pause(); video.currentTime = tr.first_t + 0.001; };
  }
}
function sampleDt() { return ((S.settings && S.settings.stride) || 1) / ((S.meta && S.meta.fps) || 25); }
function renderBehaviour() {
  const el = $("behRow");
  if (!S.frames.length || !S.frames.some((f) => f.b)) { el.innerHTML = ""; return; }
  const tot = {}; for (const f of S.frames) for (const b of f.b || []) tot[b] = (tot[b] || 0) + 1;
  const dt = sampleDt();
  el.innerHTML = "<span>Behaviour, cow-minutes:</span>" + BEH.map((b) => `<span class="chip ${b}">${b}<b>${((tot[b] || 0) * dt / 60).toFixed(1)}</b></span>`).join("") +
    `<span class="hint">${S.summary && S.summary.behaviour && !S.summary.behaviour.zones ? "no zones for this camera: eating/drinking off" : ""}</span>`;
}
function renderSummary(stats) {
  const el = $("summaryRow");
  if (!stats || !S.frames.length) { el.innerHTML = ""; return; }
  const counts = S.frames.map((f) => f.n);
  const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
  const withCows = counts.filter((c) => c > 0).length;
  el.innerHTML = `<span>Processed frames <b>${S.frames.length}</b></span>
    <span>Frames with cows <b>${withCows}</b></span>
    <span>Max visible <b>${Math.max(...counts)}</b></span>
    <span>Mean visible <b>${mean.toFixed(2)}</b></span>
    <span>Detections <b>${counts.reduce((a, b) => a + b, 0)}</b></span>`;
  if (withCows === 0) el.innerHTML += `<span><b>No cows were detected in this video at the chosen threshold.</b></span>`;
}
function renderDownloads(files) {
  const el = $("downloads"); el.innerHTML = "";
  if (!files) return;
  const labels = { detections_csv: "Detections CSV", frame_counts_csv: "Counts CSV", annotated_video: "Annotated MP4", detections_json: "JSON" };
  for (const [k, p] of Object.entries(files)) {
    if (!labels[k]) continue;
    const a = document.createElement("a");
    a.href = "/" + p.split("/").map(encodeURIComponent).join("/"); a.download = ""; a.className = "btn"; a.textContent = "⬇ " + labels[k];
    el.appendChild(a);
  }
  if (S.runDir) {
    const a = document.createElement("a"); a.className = "btn ghost"; a.textContent = "summary.json";
    a.href = "/" + (S.runDir + "/summary.json").split("/").map(encodeURIComponent).join("/"); a.target = "_blank";
    el.appendChild(a);
  }
}
function setSourceInfo() {
  const pill = $("srcPill"), note = $("sourceNote");
  if (!S.frames.length && S.source !== "job") { pill.textContent = "no data"; note.textContent = "No detections loaded for this video yet — run detection or pick a previous run."; return; }
  const st = S.settings || {};
  pill.textContent = S.source === "job" ? "live run" : "saved run";
  note.textContent = `Overlay: real ${st.model} detections · conf ≥ ${st.conf} · IoU ${st.iou} · ${st.imgsz}px · every ${st.stride} frame(s) · tracker ${st.tracker}.`;
}

// ---------------------------------------------------------------- runs
async function loadRuns(selectLatest) {
  const runs = await api(`/api/runs?video=${encodeURIComponent(S.videoPath)}`);
  const sel = $("runSelect"); sel.innerHTML = "";
  const none = new Option(runs.length ? "Previous runs…" : "No saved runs", ""); sel.add(none);
  for (const r of runs) {
    const s = r.settings || {}, st = r.stats || {};
    sel.add(new Option(`${(r.created || r.run_id).replace("T", " ")} · ${s.model} · conf ${s.conf} · ${r.status} · max ${st.max_visible}`, r.run_dir));
  }
  if (S.runDir) sel.value = S.runDir;
  if (selectLatest && runs.length) { sel.value = runs[0].run_dir; await loadRun(runs[0].run_dir); }
}
async function loadRun(runDir) {
  if (!runDir) return;
  const { summary, frames } = await api(`/api/run?run_dir=${encodeURIComponent(runDir)}`);
  resetData();
  S.source = "run"; S.runDir = runDir; S.summary = summary; S.settings = summary.settings;
  S.minTrackFrames = summary.settings.min_track_frames || 5;
  S.meta = { ...(S.meta || {}), ...summary.meta };
  addFrames(frames);
  updateRunStats(summary.stats); renderTracks(); renderSummary(summary.stats); renderBehaviour(); renderDownloads(summary.files); setSourceInfo();
  updateStats(true);
}
$("runSelect").onchange = (e) => { if (S.source === "job" && S.poll) return; loadRun(e.target.value); };

// ---------------------------------------------------------------- video selection
async function selectVideo(path) {
  S.videoPath = path; resetData(); updateRunStats(null); renderTracks(); renderSummary(null); renderBehaviour(); renderDownloads(null);
  closeEditor(); loadZones();
  const v = S.videos.find((x) => x.path === path);
  S.meta = v ? { ...v } : null;
  $("sFile").textContent = v ? v.name : "–";
  $("sMeta").textContent = v && v.readable ? `${v.width}×${v.height} · ${v.fps} fps · ${fmt(v.duration)} · ${v.codec} · ${v.size_mb} MB` : "";
  const msg = $("stageMsg");
  if (v && !v.readable) {
    msg.textContent = `Cannot read this file: ${v.error}`; msg.classList.remove("hidden");
    video.removeAttribute("src"); video.load(); $("bStart").disabled = true; setSourceInfo(); return;
  }
  $("bStart").disabled = !!S.poll;
  const transcode = v && !["h264", "avc1", "vp80", "vp90", "av01"].includes(v.codec);
  msg.textContent = transcode ? `Preparing browser preview (${v.codec} → H.264, one-time)…` : "Loading video…";
  msg.classList.remove("hidden");
  video.src = `/api/stream?path=${encodeURIComponent(path)}`;
  setSourceInfo();
  const active = await api("/api/active");
  if (active.id && active.video === path) attachJob(active.id);
  else await loadRuns(true);
}
video.addEventListener("loadeddata", () => { $("stageMsg").classList.add("hidden"); updateStats(true); });
video.addEventListener("error", () => {
  const m = $("stageMsg"); m.textContent = "This browser cannot play the video (detection still works; see the live preview / annotated export)."; m.classList.remove("hidden");
});
video.addEventListener("play", () => ($("bPlay").textContent = "❚❚ Pause"));
video.addEventListener("pause", () => ($("bPlay").textContent = "▶ Play"));
$("bPlay").onclick = () => (video.paused ? video.play() : video.pause());
const step = (dir) => { video.pause(); video.currentTime = Math.max(0, video.currentTime + dir / ((S.meta && S.meta.fps) || 25)); };
$("bBack").onclick = () => step(-1); $("bFwd").onclick = () => step(1);
$("seek").oninput = (e) => { if (video.duration) video.currentTime = (e.target.value / 1000) * video.duration; };
$("rate").onchange = (e) => (video.playbackRate = +e.target.value);
$("tOverlay").onchange = $("tLabels").onchange = $("tZones").onchange = () => updateStats(true);
$("videoSelect").onchange = (e) => selectVideo(e.target.value);

// ---------------------------------------------------------------- zones + editor
async function loadZones() {
  S.zones = null; S.zoneScope = null;
  if (!S.videoPath) return;
  try { const r = await api(`/api/zones?video=${encodeURIComponent(S.videoPath)}`); S.zones = r.zones; S.zoneScope = r.scope; } catch (_) {}
  updateStats(true);
}
const zc = $("zoneCanvas"), zctx = zc.getContext("2d");
function drawEditor() {
  if (!S.edit) return;
  const dpr = window.devicePixelRatio || 1, cw = zc.clientWidth, ch = zc.clientHeight;
  if (zc.width !== Math.round(cw * dpr)) { zc.width = Math.round(cw * dpr); zc.height = Math.round(ch * dpr); }
  zctx.setTransform(dpr, 0, 0, dpr, 0, 0); zctx.clearRect(0, 0, cw, ch);
  drawZones(zctx, { image_size: [S.meta.width, S.meta.height], zones: S.edit.zones }, { type: $("zType").value, points: S.edit.points });
  const n = (t) => S.edit.zones.filter((z) => z.type === t).length;
  $("zStatus").textContent = `Pen: ${n("pen")} · Feed troughs: ${n("feed")} · Water troughs: ${n("water")}` +
    (S.edit.points.length ? ` · drawing ${S.edit.points.length} point(s)` : "") +
    (S.zoneScope ? ` · currently saved for this ${S.zoneScope === "folder" ? "camera folder" : "video"}` : " · not saved yet");
}
function openEditor() {
  if (!S.meta || !S.meta.width) return;
  video.pause();
  const [kx, ky] = zoneScale(S.zones);
  S.edit = { zones: ((S.zones && S.zones.zones) || []).map((z) => ({ ...z, points: z.points.map(([x, y]) => [x * kx, y * ky]) })), points: [] };
  $("zoneEditor").classList.remove("hidden"); zc.classList.remove("hidden"); drawEditor(); updateStats(true);
}
function closeEditor() { S.edit = null; $("zoneEditor").classList.add("hidden"); zc.classList.add("hidden"); updateStats(true); }
zc.addEventListener("click", (e) => {
  if (!S.edit) return;
  const b = zc.getBoundingClientRect(), [x, y] = toCanvas().inv(e.clientX - b.left, e.clientY - b.top);
  if (x < 0 || y < 0 || x > S.meta.width || y > S.meta.height) return;
  S.edit.points.push([Math.round(x), Math.round(y)]); drawEditor();
});
$("bEditZones").onclick = () => (S.edit ? closeEditor() : openEditor());
$("zType").onchange = drawEditor;
$("zUndo").onclick = () => { if (S.edit) { S.edit.points.pop(); drawEditor(); } };
$("zFinish").onclick = () => {
  if (!S.edit || S.edit.points.length < 3) { $("zStatus").textContent = "A zone needs at least 3 corners."; return; }
  const t = $("zType").value;
  if (t === "pen") S.edit.zones = S.edit.zones.filter((z) => z.type !== "pen");  // one pen outline
  const name = { pen: "Pen", feed: "Feed trough", water: "Water trough" }[t];
  S.edit.zones.push({ type: t, name, points: S.edit.points }); S.edit.points = []; drawEditor();
};
$("zClearType").onclick = () => { if (S.edit) { const t = $("zType").value; S.edit.zones = S.edit.zones.filter((z) => z.type !== t); drawEditor(); } };
$("zCancel").onclick = closeEditor;
async function saveZones(scope) {
  const data = { image_size: [S.meta.width, S.meta.height], zones: S.edit.zones };
  try {
    await api("/api/zones", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ video: S.videoPath, scope, data }) });
    await loadZones(); closeEditor();
    banner(`Zones saved for this ${scope === "folder" ? "camera folder" : "video"}. Run detection again to apply them to behaviours and counts.`);
  } catch (e) { $("zStatus").textContent = "Save failed: " + e.message; }
}
$("zSaveFolder").onclick = () => saveZones("folder");
$("zSaveVideo").onclick = () => saveZones("video");
window.addEventListener("resize", drawEditor);

// ---------------------------------------------------------------- processing jobs
function readSettings() {
  return {
    video: S.videoPath, model: $("pModel").value, tracker: $("pTracker").value,
    conf: +$("pConf").value, iou: +$("pIou").value, imgsz: +$("pImgsz").value,
    stride: Math.max(1, +$("pStride").value || 1), export_video: $("pExport").checked,
    track_memory_s: +$("pMemory").value || 8, relink: $("pRelink").checked,
    max_seconds: +$("pMax").value || null, start_seconds: +$("pStart").value || 0,
    behaviours: $("pBehaviour").checked,
  };
}
$("pConf").oninput = (e) => ($("oConf").textContent = (+e.target.value).toFixed(2));
$("pIou").oninput = (e) => ($("oIou").textContent = (+e.target.value).toFixed(2));
$("bStart").onclick = async () => {
  try {
    banner("");
    const { id } = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(readSettings()) });
    attachJob(id);
  } catch (e) { banner("Could not start: " + e.message); }
};
$("bStop").onclick = async () => { if (S.jobId) await api(`/api/jobs/${S.jobId}/stop`, { method: "POST" }); };

function attachJob(id) {
  resetData(); S.jobId = id; S.source = "job";
  $("bStart").disabled = true; $("bStop").disabled = false; $("livePrevWrap").classList.remove("hidden");
  $("jobPill").textContent = "starting"; $("jobPill").classList.remove("err");
  let tableTick = 0;
  const tick = async () => {
    let st;
    try { st = await api(`/api/jobs/${id}?since=${S.frames.length}`); } catch (e) { banner(e.message); return finishJob(); }
    if (st.since === S.frames.length) addFrames(st.frames);
    S.settings = st.settings; S.minTrackFrames = st.settings.min_track_frames || 5;
    S.meta = { ...(S.meta || {}), ...st.meta };
    const phase = { relinking: "Re-linking broken IDs", exporting: "Rendering annotated video", saving: "Saving results" }[st.status];
    if (phase) {
      $("progBar").style.width = `${(st.phase_progress * 100).toFixed(1)}%`;
      $("progMeta").textContent = `${phase}… ${(st.phase_progress * 100).toFixed(0)}% · detection finished at ${st.proc_fps.toFixed(1)} fps`;
      $("jobPill").textContent = st.status; updateRunStats(st);
      S.poll = setTimeout(tick, 500); return;
    }
    $("progBar").style.width = `${(st.progress * 100).toFixed(1)}%`;
    const eta = st.proc_fps && st.total_frames ? (st.total_frames - st.frames_read) / ((st.frames_read / Math.max(st.elapsed, 0.1)) || 1) : null;
    $("progMeta").textContent = `${(st.progress * 100).toFixed(1)}% · frame ${st.frames_read}/${st.total_frames || "?"} · ${st.proc_fps.toFixed(1)} fps` + (eta != null && st.status === "running" ? ` · ETA ${fmt(eta)}` : "");
    $("jobPill").textContent = st.status;
    updateRunStats(st); setSourceInfo();
    if (++tableTick % 4 === 1) { renderTracks(); renderSummary(st); }
    if (["done", "stopped", "error"].includes(st.status)) {
      if (st.status === "error") { $("jobPill").classList.add("err"); banner("Processing failed: " + st.error); }
      finishJob();
      if (st.status !== "error" || S.frames.length) {
        S.runDir = st.run_dir;
        try { await loadRuns(false); await loadRun(st.run_dir); $("runSelect").value = st.run_dir; }
        catch (e) { banner("Run finished but its saved results could not be loaded: " + e.message); }
      }
      return;
    }
    S.poll = setTimeout(tick, 500);
  };
  S.previewTimer = setInterval(() => { $("livePrev").src = `/api/jobs/${id}/preview.jpg?ts=${Date.now()}`; }, 400);
  tick();
}
function finishJob() {
  clearTimeout(S.poll); clearInterval(S.previewTimer); S.poll = null; S.previewTimer = null;
  $("bStart").disabled = false; $("bStop").disabled = true;
  setTimeout(() => $("livePrevWrap").classList.add("hidden"), 1500);
}

// ---------------------------------------------------------------- init
(async function init() {
  try {
    const [vids, cfg] = await Promise.all([api("/api/videos"), api("/api/config")]);
    S.videos = vids;
    for (const m of cfg.models) $("pModel").add(new Option(m, m));
    $("pModel").value = cfg.default_model;
    const sel = $("videoSelect");
    if (!vids.length) { banner("No video files found in the project folder."); return; }
    for (const v of vids) sel.add(new Option(`${v.path}${v.readable ? `  (${fmt(v.duration)})` : "  — unreadable"}${v.runs ? `  · ${v.runs} run(s)` : ""}`, v.path));
    const active = await api("/api/active");
    const first = active.id ? active.video : (vids.find((v) => v.runs) || vids.find((v) => v.readable) || vids[0]).path;
    sel.value = first;
    await selectVideo(first);
  } catch (e) { banner("Failed to load: " + e.message); }
  requestAnimationFrame(loop);
})();
