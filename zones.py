"""Zones for fixed cameras: the pen outline, feed troughs and water troughs.

Zones are polygons in the video's own pixel coordinates, stored as JSON in zones/:
    zones/<video folder>.json      default for every video in that folder (same camera)
    zones/<video file>.json        override for one video (e.g. after the camera was moved)

    {"image_size": [2304, 1296],
     "zones": [{"type": "pen",   "name": "Pen",          "points": [[x, y], ...]},
               {"type": "feed",  "name": "Feed trough",  "points": [...]},
               {"type": "water", "name": "Water trough", "points": [...]}]}

* pen   - only cows whose feet (bottom-centre of the box) are inside are counted and
          get behaviours; cows in neighbouring pens are ignored.
* feed  - a cow that stays with part of its body in this zone counts as eating.
* water - same, drinking.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
ZONES_DIR = PROJECT_DIR / "zones"
ZONE_TYPES = ("pen", "feed", "water")


def _key(rel: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", rel.strip("/")).strip("_") or "root"


def zone_file(video_rel: str, scope: str = "video") -> Path:
    """scope='video' -> override for this file; scope='folder' -> default for its folder."""
    rel = str(Path(video_rel).parent) if scope == "folder" else video_rel
    return ZONES_DIR / f"{_key(rel)}.json"


def load_zones(video_rel: str) -> Tuple[Optional[dict], Optional[str]]:
    """Return (zones, scope) where scope is 'video', 'folder' or None."""
    for scope in ("video", "folder"):
        p = zone_file(video_rel, scope)
        if p.is_file():
            try:
                return json.loads(p.read_text()), scope
            except Exception:
                continue
    return None, None


def save_zones(video_rel: str, data: dict, scope: str = "folder") -> Path:
    clean = {"image_size": [int(v) for v in data.get("image_size", [0, 0])], "zones": []}
    if data.get("standing_height"):
        clean["standing_height"] = [float(v) for v in data["standing_height"]][:2]
    for z in data.get("zones", []):
        if z.get("type") not in ZONE_TYPES or len(z.get("points", [])) < 3:
            continue
        clean["zones"].append({"type": z["type"], "name": str(z.get("name") or z["type"])[:40],
                               "points": [[round(float(x), 1), round(float(y), 1)] for x, y in z["points"]]})
    ZONES_DIR.mkdir(exist_ok=True)
    p = zone_file(video_rel, scope)
    p.write_text(json.dumps(clean, indent=2))
    return p


def delete_zones(video_rel: str, scope: str) -> bool:
    p = zone_file(video_rel, scope)
    if p.is_file():
        p.unlink()
        return True
    return False


class ZoneSet:
    """Fast geometric tests against the zones of one video."""

    def __init__(self, data: dict, frame_size: Tuple[int, int]):
        w, h = frame_size
        iw, ih = data.get("image_size") or [w, h]
        sx, sy = (w / iw if iw else 1.0), (h / ih if ih else 1.0)
        self.polys: Dict[str, List[np.ndarray]] = {t: [] for t in ZONE_TYPES}
        for z in data.get("zones", []):
            pts = np.array([[x * sx, y * sy] for x, y in z["points"]], np.float32)
            self.polys[z["type"]].append(pts)
        # Rasterised masks make "fraction of the box inside a zone" cheap and exact enough.
        self.size = (w, h)
        self.masks = {}
        for t, ps in self.polys.items():
            if ps:
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [p.astype(np.int32) for p in ps], 1)
                self.masks[t] = m
        self.integrals = {t: cv2.integral(m) for t, m in self.masks.items()}
        # Camera calibration: expected box height of a STANDING cow whose feet are at image row y
        # (height = a*y + b, in the calibration image's pixels). Used for the lying rule.
        sh = data.get("standing_height")
        # h = a*y + b in calibration pixels -> in this video: h = a*y + b*sy (y and h both scale by sy)
        self.standing_height = (float(sh[0]), float(sh[1]) * sy) if sh else None

    def height_ratio(self, box) -> Optional[float]:
        """Box height / expected standing height at the same image row. Lying cows ~0.5-0.65,
        standing ~1.0. None if uncalibrated or the box touches the image border (cut-off cow)."""
        if not self.standing_height:
            return None
        w, h = self.size
        m = 4
        if box[0] <= m or box[2] >= w - m or box[3] >= h - m:
            return None
        if box[3] > 0.85 * h:  # right under the camera the wide-angle lens squashes heights
            return None
        a, b = self.standing_height
        exp = a * box[3] + b
        return float((box[3] - box[1]) / exp) if exp > 20 else None

    @property
    def has_pen(self) -> bool:
        return bool(self.polys["pen"])

    def in_pen(self, box) -> bool:
        """Feet point (bottom-centre) inside the pen; True when no pen is defined."""
        if not self.has_pen:
            return True
        x, y = (box[0] + box[2]) / 2, min(box[3], self.size[1] - 1)
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in self.polys["pen"])

    def feet_distance(self, box, zone_type: str) -> float:
        """Pixel distance from the cow's feet (bottom-centre) to the nearest zone of this type (0 if inside)."""
        x, y = float((box[0] + box[2]) / 2), float(box[3])
        best = float("inf")
        for p in self.polys.get(zone_type, []):
            d = cv2.pointPolygonTest(p, (x, y), True)  # >0 inside, <0 outside
            best = min(best, max(0.0, -d))
        return best

    def at_zone(self, box, zone_type: str, min_frac: float = 0.12, reach: float = 0.6) -> float:
        """Box fraction inside the zone, but only if the cow is actually standing at it:
        its feet must be within `reach` x box height of the zone. Cows further away that merely
        overlap the zone in the camera's perspective score 0."""
        frac = self.fraction(box, zone_type)
        if frac < min_frac:
            return frac
        h = max(1.0, box[3] - box[1])
        return frac if self.feet_distance(box, zone_type) <= reach * h else 0.0

    def fraction(self, box, zone_type: str) -> float:
        """Fraction of the box area that lies inside zones of this type (0..1)."""
        ii = self.integrals.get(zone_type)
        if ii is None:
            return 0.0
        w, h = self.size
        x1, y1 = int(max(0, min(w, box[0]))), int(max(0, min(h, box[1])))
        x2, y2 = int(max(0, min(w, box[2]))), int(max(0, min(h, box[3])))
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0
        inside = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
        return float(inside) / area
