"""Lying / standing classifier for cow crops.

Uses an Ultralytics YOLO11 classification model trained on crops from this
project's own videos (see train_posture.py and the labelling page). If no
trained model exists, posture is reported as unknown - never guessed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from detector import MODELS_DIR, pick_device

POSTURE_MODEL = MODELS_DIR / "posture_cls.pt"
CROP_PAD = 0.08  # extra context around the box (fraction of width/height)


def crop_box(frame: np.ndarray, box, pad: float = CROP_PAD) -> Optional[np.ndarray]:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    x2, y2 = int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return frame[y1:y2, x1:x2]


class PostureClassifier:
    def __init__(self, model_path: Path = POSTURE_MODEL, device: str = "auto"):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = pick_device(device)
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        if "lying" not in names.values():
            raise ValueError(f"Posture model must have a 'lying' class, got {list(names.values())}")
        self.lying_idx = [k for k, v in names.items() if v == "lying"][0]
        self.names = names

    @staticmethod
    def available(model_path: Path = POSTURE_MODEL) -> bool:
        return Path(model_path).is_file()

    def p_lying(self, crops: List[np.ndarray]) -> List[float]:
        """Probability that each crop shows a lying cow."""
        if not crops:
            return []
        res = self.model.predict(crops, imgsz=224, device=self.device, verbose=False)
        return [float(r.probs.data[self.lying_idx].cpu()) for r in res]
