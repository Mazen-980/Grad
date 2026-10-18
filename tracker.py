"""Multi-object tracking configuration and per-video track bookkeeping.

The association itself is done by Ultralytics' ByteTrack / BoT-SORT
implementation (called from detector.CowDetector). This module:
  * writes tracker YAML files tuned for slow-moving, often-occluded cattle;
  * keeps statistics per track ID (first/last seen, confidence, frame count).

Note: tracking IDs are NOT identities. An ID can change when a cow is
occluded (fence rails, other cows), leaves and re-enters the frame, or
when the camera pans quickly. "Unique tracked IDs" therefore over-counts
the real number of animals; "stable tracks" (IDs seen in at least
`min_track_frames` processed frames) is a less noisy estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

TRACKER_DIR = Path(__file__).resolve().parent / "trackers"

TRACKERS = ("botsort", "bytetrack")


def build_tracker_yaml(name: str, conf: float, buffer_updates: int) -> str:
    """Tracker settings tuned on the farm videos in this project.

    * The detector runs at a low confidence (0.1) when tracking, so the tracker
      can keep existing cows alive through weak frames (ByteTrack's second
      association stage). Only boxes >= `conf` are drawn and counted.
    * `track_high_thresh`/`new_track_thresh` follow the user's confidence, so
      low-score boxes never *start* a new ID.
    * `track_buffer` is the number of tracker updates a lost cow is remembered
      (computed from seconds, fps and frame stride by the caller).
    * BoT-SORT adds camera-motion compensation (handheld clips, PTZ cameras)
      and ReID appearance matching using the detector's own features.
    Measured on 4 clips here: unique IDs 19->12, 58->45, 81->59, 52->34
    versus plain ByteTrack with a 0.35 detection cut-off.
    """
    common = (f"track_high_thresh: {conf:.3f}\ntrack_low_thresh: 0.1\n"
              f"new_track_thresh: {min(conf + 0.1, 0.9):.3f}\ntrack_buffer: {int(buffer_updates)}\n"
              "match_thresh: 0.8\nfuse_score: True\n")
    if name == "bytetrack":
        return "tracker_type: bytetrack\n" + common
    if name == "botsort":
        return ("tracker_type: botsort\n" + common + "gmc_method: sparseOptFlow\nproximity_thresh: 0.5\n"
                "appearance_thresh: 0.25\nwith_reid: True\nmodel: auto\n")
    raise ValueError(f"Unknown tracker '{name}'. Options: none, {', '.join(TRACKERS)}")


def tracker_config_path(name: str, conf: float = 0.35, buffer_updates: int = 150) -> Optional[str]:
    """Write the YAML for this run and return its path, or None for 'none' (detection only)."""
    if not name or name == "none":
        return None
    text = build_tracker_yaml(name, conf, buffer_updates)
    TRACKER_DIR.mkdir(exist_ok=True)
    path = TRACKER_DIR / f"cow_{name}_c{int(conf * 100)}_b{int(buffer_updates)}.yaml"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return str(path)


@dataclass
class TrackInfo:
    track_id: int
    first_t: float
    last_t: float
    first_frame: int
    last_frame: int
    frames: int = 0
    conf_sum: float = 0.0
    max_conf: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.track_id, "first_t": round(self.first_t, 3), "last_t": round(self.last_t, 3),
                "first_frame": self.first_frame, "last_frame": self.last_frame, "frames": self.frames,
                "mean_conf": round(self.conf_sum / max(self.frames, 1), 4), "max_conf": round(self.max_conf, 4)}


@dataclass
class TrackRegistry:
    min_track_frames: int = 5
    tracks: Dict[int, TrackInfo] = field(default_factory=dict)

    def update(self, frame_idx: int, t: float, detections) -> None:
        for d in detections:
            if d.track_id is None:
                continue
            tr = self.tracks.get(d.track_id)
            if tr is None:
                tr = self.tracks[d.track_id] = TrackInfo(d.track_id, t, t, frame_idx, frame_idx)
            tr.last_t, tr.last_frame = t, frame_idx
            tr.frames += 1
            tr.conf_sum += d.confidence
            tr.max_conf = max(tr.max_conf, d.confidence)

    @classmethod
    def from_frames(cls, frames: List[dict], min_track_frames: int = 5) -> "TrackRegistry":
        """Rebuild statistics from per-frame records (used after re-linking changed the IDs)."""
        from types import SimpleNamespace
        reg = cls(min_track_frames)
        for fr in frames:
            reg.update(fr["f"], fr["t"], [SimpleNamespace(track_id=d[0], confidence=d[1]) for d in fr["d"]])
        return reg

    @property
    def unique_ids(self) -> int:
        return len(self.tracks)

    @property
    def stable_ids(self) -> int:
        return sum(1 for t in self.tracks.values() if t.frames >= self.min_track_frames)

    def to_list(self) -> List[dict]:
        return [t.to_dict() for t in sorted(self.tracks.values(), key=lambda t: t.track_id)]
