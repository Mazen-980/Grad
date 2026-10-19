"""Video discovery, loading, frame-by-frame processing and export.

Original videos are never modified. Everything produced here goes to outputs/:
    outputs/<video-stem>/<run-id>/annotated.mp4     video with boxes, IDs, confidences
    outputs/<video-stem>/<run-id>/detections.csv    one row per detected cow per processed frame
    outputs/<video-stem>/<run-id>/frame_counts.csv  visible cow count per processed frame
    outputs/<video-stem>/<run-id>/detections.json   per-frame boxes used by the dashboard overlay
    outputs/<video-stem>/<run-id>/summary.json      settings + statistics + per-track table
    outputs/previews/                               H.264 copies of videos browsers can't play (e.g. HEVC)

CLI usage (no dashboard):
    .venv/bin/python video_processor.py "videos/clip.mp4" --model yolo11m.pt --stride 2
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from behaviour import BEHAVIOURS, label_behaviours, track_behaviour_seconds
from detector import DEFAULT_MODEL, CowDetector, Detection
from posture import PostureClassifier, crop_box
from relink import apply_mapping, build_tracklets, compute_embeddings, relink, split_switches
from tracker import TrackRegistry, tracker_config_path
from zones import ZoneSet, load_zones

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
PREVIEW_DIR = OUTPUT_DIR / "previews"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".ts"}
SKIP_DIRS = {".venv", "venv", "outputs", "models", "node_modules", "__pycache__", "static", "trackers", "zones", "data"}
BROWSER_CODECS = {"h264", "avc1", "vp80", "vp90", "av01"}


# ----------------------------------------------------------------------------- discovery

def _is_skipped(rel: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1])


def extract_archives(root: Path = PROJECT_DIR) -> List[str]:
    """Extract videos from top-level .zip files into videos/<zip-stem>/ (once; zips are kept)."""
    extracted = []
    existing = {p.name for p in (root / "videos").rglob("*")} if (root / "videos").exists() else set()
    for z in sorted(root.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                members = [m for m in zf.namelist()
                           if Path(m).suffix.lower() in VIDEO_EXTS and not Path(m).name.startswith(".")]
                todo = [m for m in members if Path(m).name not in existing]
                if not todo:
                    continue
                dest = root / "videos" / re.sub(r"[^\w.\-]+", "_", z.stem)
                dest.mkdir(parents=True, exist_ok=True)
                for m in todo:
                    target = dest / Path(m).name  # flatten; avoids path traversal from archive names
                    with zf.open(m) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(str(target.relative_to(root)))
        except zipfile.BadZipFile:
            continue
    return extracted


def discover_videos(root: Path = PROJECT_DIR) -> List[dict]:
    """All video files under the project (excluding outputs/, .venv/, ...)."""
    vids = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not _is_skipped(rel):
            vids.append({"path": str(rel), "name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)})
    return vids


def _fourcc_str(v: float) -> str:
    i = int(v)
    return "".join(chr((i >> 8 * k) & 0xFF) for k in range(4)).strip().lower()


def probe_video(path: Path) -> dict:
    """Read basic metadata; marks the file unreadable instead of raising."""
    info = {"path": str(path), "name": path.name, "readable": False, "error": None}
    if not path.is_file():
        info["error"] = "File not found"
        return info
    if path.suffix.lower() not in VIDEO_EXTS:
        info["error"] = f"Unsupported format '{path.suffix}'"
        return info
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            info["error"] = "OpenCV could not open this file (corrupt or unsupported codec)"
            return info
        ok, frame = cap.read()
        if not ok or frame is None:
            info["error"] = "No decodable frames"
            return info
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if 1 <= fps <= 240 else 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info.update(readable=True, width=frame.shape[1], height=frame.shape[0], fps=round(fps, 3),
                    frames=n, duration=round(n / fps, 3) if n else None,
                    codec=_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)))
        return info
    finally:
        cap.release()


def ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def playable_path(path: Path) -> Path:
    """Return a browser-playable version of the video (transcodes HEVC etc. once, cached)."""
    info = probe_video(path)
    if info.get("codec") in BROWSER_CODECS or path.suffix.lower() == ".webm":
        return path
    exe = ffmpeg_exe()
    if not exe:
        return path
    st = path.stat()
    key = hashlib.md5(f"{path.resolve()}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:12]
    out = PREVIEW_DIR / f"{path.stem[:40]}_{key}.mp4"
    if not out.exists():
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        # Preview only: cap the width at 1280 px (the dashboard overlay rescales boxes to the
        # preview size). Prefer the macOS hardware encoder; long 3 MP CCTV recordings would
        # otherwise take a very long time with libx264.
        scale = ["-vf", "scale='min(1280,iw)':-2"]
        encoders = [["-c:v", "h264_videotoolbox", "-b:v", "1200k"],
                    ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]]
        for enc in encoders:
            r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(path), *scale, *enc, "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", "-an", str(tmp)])
            if r.returncode == 0:
                break
        else:
            raise RuntimeError("ffmpeg could not create a browser preview")
        tmp.rename(out)
    return out


# ----------------------------------------------------------------------------- drawing

def id_color(track_id: Optional[int]):
    if track_id is None:
        return (0, 200, 255)
    h = (track_id * 47) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


ZONE_COLORS = {"pen": (90, 200, 90), "feed": (0, 165, 255), "water": (255, 140, 0)}


def draw_zones(img: np.ndarray, zones: Optional[ZoneSet]) -> None:
    if zones is None:
        return
    th = max(1, round(img.shape[0] / 500))
    for t, polys in zones.polys.items():
        for p in polys:
            cv2.polylines(img, [p.astype(np.int32)], True, ZONE_COLORS[t], th, cv2.LINE_AA)


def annotate(frame: np.ndarray, dets: List[Detection], show_count: bool = True,
             behaviours: Optional[List[str]] = None, zones: Optional[ZoneSet] = None) -> np.ndarray:
    out = frame.copy()
    draw_zones(out, zones)
    h = out.shape[0]
    th = max(1, round(h / 400))
    fs = max(0.45, h / 1400)
    for k, d in enumerate(dets):
        c = id_color(d.track_id)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, c, th + 1)
        label = (f"ID {d.track_id} " if d.track_id is not None else "cow ") + f"{d.confidence * 100:.0f}%"
        if behaviours and k < len(behaviours):
            label += f" {behaviours[k]}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y = max(p1[1], tht + 6)
        cv2.rectangle(out, (p1[0], y - tht - 6), (p1[0] + tw + 6, y), c, -1)
        cv2.putText(out, label, (p1[0] + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th, cv2.LINE_AA)
    if show_count:
        txt = f"Cows visible: {len(dets)}"
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, th + 1)
        cv2.rectangle(out, (8, 8), (24 + tw, 24 + tht), (25, 25, 25), -1)
        cv2.putText(out, txt, (16, 16 + tht), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, (255, 255, 255), th + 1, cv2.LINE_AA)
    return out


class AnnotatedWriter:
    """Writes H.264 via ffmpeg (browser/QuickTime friendly); falls back to OpenCV mp4v."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.path, self.gen, self.cv = path, None, None
        try:
            import imageio_ffmpeg
            self.gen = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24",
                                                   quality=7, macro_block_size=2,
                                                   output_params=["-movflags", "+faststart"])
            self.gen.send(None)
        except Exception:
            self.gen = None
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray):
        if self.gen is not None:
            self.gen.send(np.ascontiguousarray(frame))
        else:
            self.cv.write(frame)

    def close(self):
        if self.gen is not None:
            self.gen.close()
        if self.cv is not None:
            self.cv.release()


# ----------------------------------------------------------------------------- processing

@dataclass
class ProcessingSettings:
    model: str = DEFAULT_MODEL
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 960
    stride: int = 1            # run the model on every Nth frame
    tracker: str = "botsort"   # botsort | bytetrack | none
    track_memory_s: float = 8.0  # how long the tracker remembers a hidden cow
    relink: bool = True        # offline re-linking of broken tracks (see relink.py)
    device: str = "auto"
    export_video: bool = True
    min_track_frames: int = 5
    max_seconds: Optional[float] = None  # process only N seconds (quick tests)
    start_seconds: float = 0.0  # start processing this far into the video
    behaviours: bool = True    # zones (pen/feed/water) + posture model -> behaviour labels


def _fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _slug(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s).strip("_")[:60] or "video"


class VideoJob:
    """Runs detection (+ tracking) over a whole video. Use .start() for a background thread."""

    def __init__(self, video_path: str, settings: ProcessingSettings,
                 on_progress: Optional[Callable[["VideoJob"], None]] = None):
        self.video_rel = video_path
        self.video = (PROJECT_DIR / video_path) if not Path(video_path).is_absolute() else Path(video_path)
        self.settings = settings
        self.on_progress = on_progress
        self.id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + _slug(Path(settings.model).stem)
        self.run_dir = OUTPUT_DIR / _slug(self.video.stem) / self.id
        self.status = "queued"  # queued | loading | running | relinking | exporting | saving | done | stopped | error
        self.error: Optional[str] = None
        self.final_status: Optional[str] = None
        self.meta: dict = {}
        self.frames: List[dict] = []  # {"f", "t", "n", "d": [[id, conf, x1, y1, x2, y2], ...]}
        self.registry = TrackRegistry(settings.min_track_frames)
        self.frames_read = 0      # frames read since start_frame
        self.total_frames = 0
        self.start_frame = 0
        self.end_frame = 0
        self.proc_fps = 0.0
        self.infer_ms = 0.0
        self.started = None
        self.elapsed = 0.0
        self.preview_jpeg: Optional[bytes] = None
        self.phase_progress = 0.0  # progress of the post-processing phases
        self.relink_report: Optional[dict] = None
        self.aux: Dict[int, List[list]] = {}   # frame -> [[feed_frac, water_frac, p_lying], ...]
        self.zones: Optional[ZoneSet] = None
        self.behaviour_totals: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    # -- control
    def start(self):
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    @property
    def progress(self) -> float:
        return min(1.0, self.frames_read / self.total_frames) if self.total_frames else 0.0

    def status_dict(self, since: int = 0) -> dict:
        with self.lock:
            new = self.frames[since:]
            confs = [d[1] for fr in self.frames for d in fr["d"]]
            return {"id": self.id, "video": self.video_rel, "status": self.status, "error": self.error,
                    "progress": round(self.progress, 4), "frames_read": self.frames_read,
                    "total_frames": self.total_frames, "processed": len(self.frames),
                    "proc_fps": round(self.proc_fps, 2), "infer_ms": round(self.infer_ms, 1),
                    "phase_progress": round(self.phase_progress, 3), "relink": self.relink_report,
                    "elapsed": round(self.elapsed, 1), "unique_ids": self.registry.unique_ids,
                    "stable_ids": self.registry.stable_ids,
                    "avg_conf": round(float(np.mean(confs)), 4) if confs else None,
                    "settings": asdict(self.settings), "meta": self.meta, "run_dir": self._rel(self.run_dir),
                    "since": since, "frames": new}

    @staticmethod
    def _rel(p: Path) -> str:
        return str(p.relative_to(PROJECT_DIR))

    # -- main loop
    def run(self):
        s = self.settings
        final = "error"
        try:
            self.status = "loading"
            self.meta = probe_video(self.video)
            if not self.meta.get("readable"):
                raise RuntimeError(f"Cannot read video: {self.meta.get('error')}")
            fps = self.meta["fps"]
            n_all = self.meta["frames"] or 0
            self.start_frame = max(0, min(int(s.start_seconds * fps), max(n_all - 1, 0)))
            self.total_frames = max(n_all - self.start_frame, 0)
            if s.max_seconds:
                self.total_frames = min(self.total_frames or 10**9, int(s.max_seconds * fps))
            self.end_frame = self.start_frame + self.total_frames

            # Tracker memory is given in seconds; the tracker counts updates (= processed frames).
            buffer_updates = max(30, int(round(s.track_memory_s * fps / max(1, s.stride))))
            det = CowDetector(s.model, s.device, s.conf, s.iou, s.imgsz,
                              tracker_config_path(s.tracker, s.conf, buffer_updates))
            self.meta.update(device=det.device, cow_classes=list(det.class_names.values()))
            posture = None
            if s.behaviours:
                zdata, zscope = load_zones(self.video_rel)
                if zdata:
                    self.zones = ZoneSet(zdata, (self.meta["width"], self.meta["height"]))
                    self.meta["zones"] = {"scope": zscope, "data": zdata}
                if PostureClassifier.available():
                    posture = PostureClassifier(device=s.device)
                self.meta["posture_model"] = posture is not None
            self.run_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(self.video))
            if self.start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.status, self.started = "running", time.time()
            infer_times: List[float] = []
            idx = self.start_frame - 1
            while not self._stop.is_set():
                idx += 1
                if s.max_seconds and idx >= self.end_frame:
                    break
                if (idx - self.start_frame) % max(1, s.stride):
                    if not cap.grab():
                        break
                    self.frames_read = idx + 1 - self.start_frame
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                self.frames_read = idx + 1 - self.start_frame
                t = idx / fps
                res = det.process(frame)
                infer_times.append(res.inference_ms)
                if self.zones is not None and self.zones.has_pen:  # ignore cows outside this pen
                    res.detections = [d for d in res.detections if self.zones.in_pen((d.x1, d.y1, d.x2, d.y2))]
                if s.behaviours:
                    boxes = [(d.x1, d.y1, d.x2, d.y2) for d in res.detections]
                    pl: List[Optional[float]] = [None] * len(boxes)
                    if posture is not None and boxes:
                        crops = [crop_box(frame, b) for b in boxes]
                        ok_i = [i for i, c in enumerate(crops) if c is not None]
                        for i, p in zip(ok_i, posture.p_lying([crops[i] for i in ok_i])):
                            pl[i] = round(p, 3)
                    z = self.zones
                    def occluded(i):
                        # feet point hidden behind another cow whose box reaches lower (it stands in front):
                        # the visible box is cut short, so its height says nothing about posture
                        fx, fy = (boxes[i][0] + boxes[i][2]) / 2, boxes[i][3]
                        return any(j != i and o[0] <= fx <= o[2] and o[1] <= fy < o[3] - 5
                                   for j, o in enumerate(boxes))

                    def ratio(i):
                        if not z or occluded(i):
                            return None
                        r = z.height_ratio(boxes[i])
                        return None if r is None else round(r, 3)

                    self.aux[idx] = [[round(z.at_zone(b, "feed"), 3) if z else 0.0,
                                      round(z.at_zone(b, "water"), 3) if z else 0.0, pl[i], ratio(i)]
                                     for i, b in enumerate(boxes)]
                rec = {"f": idx, "t": round(t, 4), "n": res.count, "d": [d.as_list() for d in res.detections]}
                with self.lock:
                    self.frames.append(rec)
                    self.registry.update(idx, t, res.detections)
                self.elapsed = time.time() - self.started
                self.proc_fps = len(self.frames) / max(self.elapsed, 1e-6)
                self.infer_ms = float(np.mean(infer_times[-50:]))
                if len(self.frames) % 3 == 1:
                    small = annotate(frame, res.detections, zones=self.zones)
                    scale = 640 / small.shape[1]
                    small = cv2.resize(small, (640, int(small.shape[0] * scale)))
                    self.preview_jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                if self.on_progress:
                    self.on_progress(self)
            cap.release()
            if not self.total_frames or self.frames_read > self.total_frames:
                self.total_frames = self.frames_read
            self.elapsed = time.time() - self.started
            if self._stop.is_set():
                final = "stopped"
            else:
                if s.relink and s.tracker != "none" and self.frames:
                    self._relink(fps)
                if s.behaviours and self.frames and (self.zones is not None or posture is not None):
                    with self.lock:
                        self.behaviour_totals = label_behaviours(self.frames, self.aux, self.zones is not None)
                if s.export_video and self.frames:
                    self._export_video(fps)
                final = "stopped" if self._stop.is_set() else "done"
        except Exception as e:  # surface errors to the UI instead of crashing the server
            final, self.error = "error", f"{type(e).__name__}: {e}"
        # Only report done/stopped/error once all files are written, so clients can load them.
        self.status, self.final_status = "saving", final
        try:
            if self.frames or final != "error":
                self.write_outputs()
        except Exception as e:
            self.error = (self.error or "") + f" | export failed: {e}"
        finally:
            self.status = final

    def _relink(self, fps: float):
        """Join broken tracks of the same cow (relink.py) and renumber IDs 1..N."""
        self.status, self.phase_progress = "relinking", 0.0
        tracklets = build_tracklets(self.frames)
        compute_embeddings(self.video, tracklets, device=self.settings.device,
                           progress=lambda p: setattr(self, "phase_progress", p))
        tracklets, splits = split_switches(self.frames, tracklets)
        mapping, report = relink(self.frames, tracklets, fps)
        report["switch_splits"] = splits
        with self.lock:
            apply_mapping(self.frames, mapping)
            self.registry = TrackRegistry.from_frames(self.frames, self.settings.min_track_frames)
        self.relink_report = report

    def _export_video(self, fps: float):
        """Render the annotated MP4 after re-linking so it shows the final IDs.
        With stride > 1 the most recent processed result is held on skipped frames."""
        self.status, self.phase_progress = "exporting", 0.0
        writer = AnnotatedWriter(self.run_dir / "annotated.mp4", self.meta["width"], self.meta["height"], fps)
        cap = cv2.VideoCapture(str(self.video))
        if self.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        recs, j, dets, labels = self.frames, -1, [], None
        try:
            for idx in range(self.start_frame, self.start_frame + self.frames_read):
                if self._stop.is_set():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                while j + 1 < len(recs) and recs[j + 1]["f"] <= idx:
                    j += 1
                    dets = [Detection(d[2], d[3], d[4], d[5], d[1], d[0]) for d in recs[j]["d"]]
                    labels = recs[j].get("b")
                writer.write(annotate(frame, dets, behaviours=labels if j >= 0 else None, zones=self.zones))
                self.phase_progress = (idx + 1 - self.start_frame) / max(self.frames_read, 1)
        finally:
            cap.release()
            writer.close()

    # -- export
    def summary(self) -> dict:
        counts = [fr["n"] for fr in self.frames]
        confs = [d[1] for fr in self.frames for d in fr["d"]]
        files = {k: self._rel(self.run_dir / f) for k, f in
                 [("detections_csv", "detections.csv"), ("frame_counts_csv", "frame_counts.csv"),
                  ("detections_json", "detections.json"), ("annotated_video", "annotated.mp4")]
                 if (self.run_dir / f).exists() or k != "annotated_video"}
        return {
            "run_id": self.id, "video": self.video_rel, "status": self.final_status or self.status, "error": self.error,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "detection_source": "real model inference (Ultralytics YOLO)",
            "settings": asdict(self.settings), "meta": self.meta,
            "stats": {
                "frames_read": self.frames_read, "frames_processed": len(self.frames),
                "frames_with_cows": sum(1 for c in counts if c > 0),
                "max_visible": max(counts) if counts else 0,
                "mean_visible": round(float(np.mean(counts)), 3) if counts else 0,
                "median_visible": float(np.median(counts)) if counts else 0,
                "total_detections": len(confs),
                "avg_conf": round(float(np.mean(confs)), 4) if confs else None,
                "unique_ids": self.registry.unique_ids, "stable_ids": self.registry.stable_ids,
                "unique_ids_before_relink": (self.relink_report or {}).get("raw_ids", self.registry.unique_ids),
                "processing_fps": round(self.proc_fps, 2), "avg_inference_ms": round(self.infer_ms, 1),
                "elapsed_s": round(self.elapsed, 1),
            },
            "relink": self.relink_report,
            "behaviour": self._behaviour_summary(),
            "tracks": self._tracks_with_behaviour(),
            "files": files,
        }

    def _sample_dt(self) -> float:
        return max(1, self.settings.stride) / max(self.meta.get("fps") or 25.0, 1e-6)

    def _behaviour_summary(self) -> Optional[dict]:
        if self.behaviour_totals is None:
            return None
        dt_ = self._sample_dt()
        return {"cow_seconds": {b: round(n * dt_, 1) for b, n in self.behaviour_totals.items()},
                "zones": bool(self.zones), "posture_model": bool(self.meta.get("posture_model")),
                "note": "eating/drinking = standing in the feed/water zone for a few seconds; "
                        "lying needs the posture model (models/posture_cls.pt)."}

    def _tracks_with_behaviour(self) -> List[dict]:
        tracks = self.registry.to_list()
        if self.behaviour_totals is None:
            return tracks
        secs = track_behaviour_seconds(self.frames, self._sample_dt())
        for t in tracks:
            t["behaviour_s"] = secs.get(t["id"], {})
        return tracks

    def write_outputs(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "detections.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "frame", "timestamp_s", "timestamp", "track_id", "confidence",
                        "x1", "y1", "x2", "y2", "width", "height", "raw_track_id", "behaviour"])
            for fr in self.frames:
                labels = fr.get("b") or [""] * len(fr["d"])
                for d, lab in zip(fr["d"], labels):
                    tid, c, x1, y1, x2, y2 = d[:6]
                    raw = d[6] if len(d) > 6 else tid
                    w.writerow([self.video.name, fr["f"], fr["t"], _fmt_ts(fr["t"]), "" if tid is None else tid,
                                c, x1, y1, x2, y2, round(x2 - x1, 1), round(y2 - y1, 1), "" if raw is None else raw,
                                lab])
        with open(self.run_dir / "frame_counts.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "timestamp_s", "timestamp", "visible_cows", *BEHAVIOURS])
            for fr in self.frames:
                b = fr.get("b") or []
                w.writerow([fr["f"], fr["t"], _fmt_ts(fr["t"]), fr["n"], *[b.count(k) for k in BEHAVIOURS]])
        (self.run_dir / "detections.json").write_text(json.dumps({"frames": self.frames}, separators=(",", ":")))
        (self.run_dir / "summary.json").write_text(json.dumps(self.summary(), indent=2))


def list_runs(video_rel: Optional[str] = None) -> List[dict]:
    runs = []
    for p in sorted(OUTPUT_DIR.glob("*/*/summary.json"), reverse=True):
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue
        if video_rel and s.get("video") != video_rel:
            continue
        s["run_dir"] = str(p.parent.relative_to(PROJECT_DIR))
        runs.append(s)
    return runs


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Detect and track cows in a video (offline).")
    ap.add_argument("video", nargs="?", help="Path to a video. Omit to list available videos.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack", "none"])
    ap.add_argument("--track-memory", type=float, default=8.0, help="Seconds a hidden cow keeps its ID")
    ap.add_argument("--no-relink", action="store_true", help="Skip offline re-linking of broken tracks")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-video", action="store_true", help="Skip annotated video export")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--start-seconds", type=float, default=0.0)
    ap.add_argument("--no-behaviour", action="store_true", help="Skip zones/posture/behaviour labelling")
    a = ap.parse_args()

    extract_archives()
    if not a.video:
        for v in discover_videos():
            info = probe_video(PROJECT_DIR / v["path"])
            print(f"{v['path']:<75} {info.get('width')}x{info.get('height')} {info.get('fps')}fps "
                  f"{info.get('duration')}s {info.get('codec')} {'' if info['readable'] else info['error']}")
        return
    s = ProcessingSettings(a.model, a.conf, a.iou, a.imgsz, a.stride, a.tracker, a.track_memory, not a.no_relink,
                           a.device, not a.no_video, max_seconds=a.max_seconds, behaviours=not a.no_behaviour,
                           start_seconds=a.start_seconds)
    last = [0.0]

    def prog(job: VideoJob):
        if time.time() - last[0] > 2:
            last[0] = time.time()
            print(f"  {job.progress * 100:5.1f}%  {job.proc_fps:5.1f} fps  visible={job.frames[-1]['n']}  "
                  f"unique_ids={job.registry.unique_ids}", flush=True)

    job = VideoJob(a.video, s, prog)
    job.run()
    print(json.dumps(job.summary()["stats"], indent=2))
    if job.relink_report:
        print("relink:", job.relink_report)
    if job.behaviour_totals is not None:
        print("behaviour (cow-seconds):", job.summary()["behaviour"]["cow_seconds"])
    print("status:", job.status, job.error or "", "->", job.run_dir)


if __name__ == "__main__":
    main()
