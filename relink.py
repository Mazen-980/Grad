"""Offline track re-linking: give a cow back its ID after its track was broken.

The online tracker only remembers a lost cow for `track_buffer` frames. When a
cow is hidden longer (fence rails, other cows, panning camera) or flickers below
the detection threshold, it comes back with a new ID. Because videos are
processed offline, we can afterwards join those fragments ("tracklets"):

* Appearance: each tracklet gets an embedding = mean over sampled crops of
  [ImageNet YOLO11-cls embedding ‖ HSV colour histogram].
* Self-calibration per video: two tracklets visible in the same frame are
  certainly different cows. Their similarities give the "different cow"
  distribution for *this* video, and the merge threshold is set at its
  `neg_quantile` (default 99.5%), so roughly <0.5% of different-cow pairs
  would pass. Generic features rate all cows as similar (0.85-0.9), so a
  fixed threshold does not work across videos.
* Spatial cue: a cow that re-appears where it disappeared (box IoU >= 0.5,
  short gap) needs a lower threshold (`spatial_quantile`), which handles
  lying cows that flicker on fixed CCTV cameras.
* Switch splitting (before linking): the tracker sometimes jumps from one cow
  to another (e.g. while the camera pans), so one ID covers two animals. Each
  tracklet is sampled along its length and the before/after similarity is
  computed at every possible cut. The threshold is placed between the typical
  different-cow similarity (median over co-occurring tracklets) and the typical
  within-track self-similarity of this video (median over tracklets), weighted
  60% toward the latter. The tracklet is cut where the box jumps most, and
  only if the box really jumps there (1 - IoU >= `min_jump`): on the farm
  videos, false splits came from occlusion (appearance changes, box stays
  put, 1-IoU <= 0.09) while real switches had 1-IoU >= 0.26.
* Constraints: tracklets that co-occur are never merged, groups may not
  overlap in time, and a merge must agree with every member of both groups
  (average linkage), so A~B~C chains with A!=C are avoided.

This reduces ID changes but cannot guarantee identity; look-alike cows
(e.g. all-black animals) are only merged when they re-appear in place.
Permanent IDs across videos need identification against a registered herd.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import MODELS_DIR, pick_device

EMBED_MODEL = "yolo11s-cls.pt"


@dataclass
class Tracklet:
    tid: int                 # segment id (== raw tracker id unless the track was split)
    raw: Optional[int] = None  # original tracker id
    frames: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    crop_feats: Optional[np.ndarray] = None
    crop_frames: List[int] = field(default_factory=list)  # frame index of each crop feature
    emb: Optional[np.ndarray] = None

    @property
    def start(self): return self.frames[0]

    @property
    def end(self): return self.frames[-1]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def build_tracklets(frames: List[dict]) -> Dict[int, Tracklet]:
    tr: Dict[int, Tracklet] = {}
    for fr in frames:
        for d in fr["d"]:
            tid = d[0]
            if tid is None:
                continue
            t = tr.setdefault(tid, Tracklet(tid, d[6] if len(d) > 6 else tid))
            t.frames.append(fr["f"]); t.times.append(fr["t"]); t.boxes.append(tuple(d[2:6])); t.confs.append(d[1])
    return tr


def _sample_indices(t: Tracklet, k: int) -> List[int]:
    n = len(t.frames)
    if n <= k:
        return list(range(n))
    return [int(c[np.argmax([t.confs[i] for i in c])]) for c in np.array_split(np.arange(n), k) if len(c)]


def _load_embedder():
    from ultralytics import YOLO
    import os
    MODELS_DIR.mkdir(exist_ok=True)
    cwd = os.getcwd()
    try:  # Ultralytics downloads into cwd
        os.chdir(MODELS_DIR)
        return YOLO(EMBED_MODEL)
    finally:
        os.chdir(cwd)


def _hsv_hist(c: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(c, (128, 128)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 6], [0, 180, 0, 256, 0, 256]).ravel()
    return np.sqrt(h / max(h.sum(), 1e-9))


def compute_embeddings(video_path: Path, tracklets: Dict[int, Tracklet], per_track: int = 16,
                       device: str = "auto", progress=None) -> None:
    """Crop samples of every tracklet from the video and embed them."""
    model = _load_embedder()
    dev = pick_device(device)
    wanted: Dict[int, List[Tuple[int, Tuple]]] = {}
    for t in tracklets.values():
        for i in _sample_indices(t, per_track):
            wanted.setdefault(t.frames[i], []).append((t.tid, t.boxes[i]))
    crops: Dict[int, List[np.ndarray]] = {tid: [] for tid in tracklets}
    crop_frames: Dict[int, List[int]] = {tid: [] for tid in tracklets}
    cap = cv2.VideoCapture(str(video_path))
    last = max(wanted) if wanted else -1
    first = min(wanted) if wanted else 0
    if first > 0:  # runs that start partway through a video
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    idx = first - 1
    while idx < last:
        idx += 1
        if idx not in wanted:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        for tid, (x1, y1, x2, y2) in wanted[idx]:
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                crops[tid].append(frame[y1:y2, x1:x2])
                crop_frames[tid].append(idx)
        if progress and idx % 25 == 0:
            progress(idx / max(last, 1))
    cap.release()

    for tid, cs in crops.items():
        if not cs:
            continue
        cnn = model.embed([cv2.resize(c, (224, 224)) for c in cs], imgsz=224, device=dev, verbose=False)
        cnn = _norm(np.stack([x.cpu().numpy().ravel() for x in cnn]))
        hsv = _norm(np.stack([_hsv_hist(c) for c in cs]))
        feats = _norm(np.hstack([cnn, hsv]))
        tracklets[tid].crop_feats = feats
        tracklets[tid].crop_frames = crop_frames[tid]
        tracklets[tid].emb = _norm(feats.mean(0))


def _negative_sims(frames: List[dict], tracklets: Dict[int, Tracklet]) -> np.ndarray:
    E = {i: t.emb for i, t in tracklets.items() if t.emb is not None}
    return np.array([E[a] @ E[b] for a, b in _cooccurring(frames) if a in E and b in E])


def _best_cut(F: np.ndarray, min_side: int) -> Tuple[Optional[int], float]:
    best_k, best_sim = None, 1.0
    for k in range(min_side, len(F) - min_side + 1):
        sim = float(_norm(F[:k].mean(0)) @ _norm(F[k:].mean(0)))
        if sim < best_sim:
            best_k, best_sim = k, sim
    return best_k, best_sim


def split_switches(frames: List[dict], tracklets: Dict[int, Tracklet], min_side: int = 3,
                   pos_weight: float = 0.6, min_jump: float = 0.2) -> Tuple[Dict[int, Tracklet], List[dict]]:
    """Cut tracklets whose appearance changes from one cow to another (tracker ID switch)."""
    neg = _negative_sims(frames, tracklets)
    cuts = {tid: _best_cut(t.crop_feats, min_side) for tid, t in tracklets.items()
            if t.crop_feats is not None and len(t.crop_feats) >= 2 * min_side}
    if len(neg) < 20 or len(cuts) < 4:
        return tracklets, []
    neg_med = float(np.median(neg))                          # two different cows
    self_med = float(np.median([c[1] for c in cuts.values()]))  # most tracklets are one cow
    if self_med <= neg_med:
        return tracklets, []
    thr = neg_med + pos_weight * (self_med - neg_med)
    out: Dict[int, Tracklet] = {}
    splits = []
    next_id = max(tracklets) + 1 if tracklets else 1
    for tid, t in tracklets.items():
        F = t.crop_feats
        best_k, best_sim = cuts.get(tid, (None, 1.0))
        if best_k is None or best_sim >= thr:
            out[tid] = t
            continue
        # The switch happened between crop best_k-1 and best_k: cut where the box jumps most.
        lo, hi = t.crop_frames[best_k - 1], t.crop_frames[best_k]
        obs = [i for i, f in enumerate(t.frames) if lo <= f <= hi]
        cut = obs[-1]
        jump = -1.0
        for a, b in zip(obs, obs[1:]):
            j = 1.0 - _iou(t.boxes[a], t.boxes[b])
            if j > jump:
                jump, cut = j, b
        if jump < min_jump or cut <= 0:  # appearance changed but the box stayed put: occlusion, not a switch
            out[tid] = t
            continue
        left = Tracklet(tid, t.raw, t.frames[:cut], t.times[:cut], t.boxes[:cut], t.confs[:cut])
        right = Tracklet(next_id, t.raw, t.frames[cut:], t.times[cut:], t.boxes[cut:], t.confs[cut:])
        left.crop_feats, left.crop_frames = F[:best_k], t.crop_frames[:best_k]
        right.crop_feats, right.crop_frames = F[best_k:], t.crop_frames[best_k:]
        for part in (left, right):
            part.emb = _norm(part.crop_feats.mean(0))
            out[part.tid] = part
        splits.append({"raw_id": t.raw, "at_frame": t.frames[cut], "at_s": round(t.times[cut], 2),
                       "similarity": round(best_sim, 3), "threshold": round(thr, 3), "box_jump": round(jump, 2),
                       "new_segment": next_id})
        next_id += 1
    if splits:  # write segment ids into the frames so co-occurrence and mapping use them
        seg_of = {}
        for t in out.values():
            for f in t.frames:
                seg_of[(f, t.raw)] = t.tid
        for fr in frames:
            for d in fr["d"]:
                if len(d) == 6:
                    d.append(d[0])
                if d[6] is not None:
                    d[0] = seg_of.get((fr["f"], d[6]), d[0])
    return out, splits


def _cooccurring(frames: List[dict]) -> set:
    co = set()
    for fr in frames:
        p = [d[0] for d in fr["d"] if d[0] is not None]
        co.update((min(a, b), max(a, b)) for a, b in itertools.combinations(p, 2))
    return co


def relink(frames: List[dict], tracklets: Dict[int, Tracklet], fps: float, max_gap_s: float = 60.0,
           neg_quantile: float = 0.995, spatial_quantile: float = 0.95, spatial_iou: float = 0.5,
           spatial_gap_s: float = 15.0, min_frames: int = 3) -> Tuple[Dict[int, int], dict]:
    """Return ({raw_id: linked_id}, report). Linked IDs are renumbered 1..N by first appearance."""
    ids = sorted(tracklets)
    usable = [i for i in ids if tracklets[i].emb is not None and len(tracklets[i].frames) >= min_frames]
    co = _cooccurring(frames)
    E = {i: tracklets[i].emb for i in usable}

    neg = np.array([E[a] @ E[b] for a, b in co if a in E and b in E])
    if len(neg) >= 20:
        thr = float(np.quantile(neg, neg_quantile))
        thr_sp = float(np.quantile(neg, spatial_quantile))
    else:  # too little evidence to calibrate; be conservative
        thr, thr_sp = 0.97, 0.93
    thr_sp = min(thr_sp, thr)

    parent = {i: i for i in ids}
    members = {i: [i] for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cands = []
    max_gap, sp_gap = max_gap_s * fps, spatial_gap_s * fps
    for a, b in itertools.permutations(usable, 2):
        A, B = tracklets[a], tracklets[b]
        gap = B.start - A.end
        if gap <= 0 or gap > max_gap or (min(a, b), max(a, b)) in co:
            continue
        sim = float(E[a] @ E[b])
        spatial = _iou(A.boxes[-1], B.boxes[0]) >= spatial_iou and gap <= sp_gap
        need = thr_sp if spatial else thr
        if sim >= need:
            cands.append((sim - need + (0.05 if spatial else 0.0), a, b, need))
    cands.sort(reverse=True)

    def spans_overlap(ga, gb):
        return any(tracklets[x].start <= tracklets[y].end and tracklets[y].start <= tracklets[x].end
                   for x in ga for y in gb)

    merges = 0
    for _, a, b, need in cands:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        ga, gb = members[ra], members[rb]
        if spans_overlap(ga, gb) or any((min(x, y), max(x, y)) in co for x in ga for y in gb):
            continue
        cross = [E[x] @ E[y] for x in ga for y in gb if x in E and y in E]
        if not cross or float(np.mean(cross)) < need:  # average linkage: agree with the whole group
            continue
        parent[rb] = ra
        members[ra] = ga + gb
        del members[rb]
        merges += 1

    roots = sorted({find(i) for i in ids}, key=lambda r: min(tracklets[i].start for i in members[r]))
    new_id = {r: k + 1 for k, r in enumerate(roots)}
    mapping = {i: new_id[find(i)] for i in ids}
    report = {"raw_ids": len(ids), "linked_ids": len(roots), "merges": merges,
              "threshold": round(thr, 4), "spatial_threshold": round(thr_sp, 4), "negative_pairs": int(len(neg))}
    return mapping, report


def apply_mapping(frames: List[dict], mapping: Dict[int, int]) -> None:
    """Rewrite IDs (segment id -> linked id) in place; the raw tracker ID stays as the 7th element."""
    for fr in frames:
        for d in fr["d"]:
            if len(d) == 6:
                d.append(d[0])
            if d[0] is not None:
                d[0] = mapping.get(d[0], d[0])
