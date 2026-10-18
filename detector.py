"""Cow detection logic (real model inference via Ultralytics YOLO).

Every box this module returns comes from a neural-network forward pass.
There is no simulated / random fallback: if the model cannot be loaded or
does not contain a cow-like class, a clear error is raised instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

# Class names that we accept as "cow" in either COCO or a custom livestock model.
COW_CLASS_NAMES = {"cow", "cows", "cattle", "bull", "calf", "heifer", "dairy cow", "beef cattle"}

# Pretrained weights that Ultralytics can download automatically (COCO-80; class 19 = "cow").
KNOWN_PRETRAINED = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
DEFAULT_MODEL = "yolo11m.pt"


def pick_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_models() -> List[str]:
    """Local weight files in models/ plus the known downloadable pretrained ones."""
    local = sorted(p.name for p in MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
    return local + [m for m in KNOWN_PRETRAINED if m not in local]


def resolve_model_path(name: str) -> str:
    p = Path(name)
    if p.is_file():
        return str(p)
    local = MODELS_DIR / p.name
    if local.is_file():
        return str(local)
    # Not present yet: let Ultralytics download it into models/.
    MODELS_DIR.mkdir(exist_ok=True)
    return str(local)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: Optional[int] = None
    class_name: str = "cow"

    def as_list(self) -> list:
        return [self.track_id, round(self.confidence, 4), round(self.x1, 1), round(self.y1, 1),
                round(self.x2, 1), round(self.y2, 1)]


@dataclass
class FrameResult:
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0  # pre + inference + post, as reported by Ultralytics

    @property
    def count(self) -> int:
        return len(self.detections)


class CowDetector:
    """Wraps a YOLO model, restricted to cow classes, with optional tracking."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto", conf: float = 0.35,
                 iou: float = 0.5, imgsz: int = 960, tracker_cfg: Optional[str] = None):
        from ultralytics import YOLO

        self.model_name = model_name
        self.device = pick_device(device)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.tracker_cfg = tracker_cfg  # None -> detection only (no IDs)

        path = resolve_model_path(model_name)
        cwd = os.getcwd()
        try:  # Ultralytics downloads relative to cwd; keep weights inside models/
            os.chdir(MODELS_DIR)
            self.model = YOLO(Path(path).name if Path(path).parent == MODELS_DIR else path)
        finally:
            os.chdir(cwd)

        names = self.model.names
        self.cow_class_ids = [i for i, n in names.items() if str(n).strip().lower() in COW_CLASS_NAMES]
        if not self.cow_class_ids:
            raise ValueError(
                f"Model '{model_name}' has no cow/cattle class (classes: {list(names.values())[:20]}...). "
                "Use a COCO model (class 'cow') or a livestock model fine-tuned with a 'cow'/'cattle' class.")
        self.class_names = {i: names[i] for i in self.cow_class_ids}

    # When tracking, weak boxes (>= this) are passed to the tracker so it can keep
    # existing IDs alive through occlusion; only boxes >= self.conf are reported.
    TRACK_FEED_CONF = 0.1

    def _run(self, frame: np.ndarray):
        kwargs = dict(iou=self.iou, imgsz=self.imgsz, classes=self.cow_class_ids, device=self.device, verbose=False)
        if self.tracker_cfg:
            return self.model.track(frame, persist=True, tracker=self.tracker_cfg,
                                    conf=min(self.TRACK_FEED_CONF, self.conf), **kwargs)[0]
        return self.model.predict(frame, conf=self.conf, **kwargs)[0]

    def process(self, frame: np.ndarray) -> FrameResult:
        r = self._run(frame)
        boxes = r.boxes
        dets: List[Detection] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(confs)
            for (x1, y1, x2, y2), c, k, tid in zip(xyxy, confs, cls, ids):
                if c < self.conf:  # weak box only used to sustain a track
                    continue
                dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c),
                                      None if tid is None else int(tid), self.model.names[int(k)]))
        ms = sum(v for v in (r.speed or {}).values() if v is not None)
        return FrameResult(dets, ms)
