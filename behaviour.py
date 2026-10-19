"""Behaviour labels per cow per frame: lying, eating, drinking, walking, standing.

Inputs per detection (collected while processing a video):
    feed  - fraction of the box inside a feed-trough zone   (zones.py)
    water - fraction of the box inside a water-trough zone  (zones.py)
    lying - probability of lying from the posture model     (posture.py), or None
    ratio - box height / expected standing height at that image row (zones.py camera
            calibration), or None

Rules, evaluated over each cow's track (after re-linking), in priority order:
    lying     smoothed P(lying) >= 0.5 over the last ~3 s (posture model), or, without a model,
              rolling median height ratio < LYING_RATIO (lying cows measured 0.53-0.66 on the
              15 Oct camera, standing 1.0-1.16; lying with the head raised ~0.85 is missed)
    drinking  in a water zone (>= ZONE_FRAC) for >= MIN_DRINK_S, not walking
    eating    in a feed zone  (>= ZONE_FRAC) for >= MIN_EAT_S,  not walking
    walking   box centre moves faster than WALK_SPEED body-heights per second
    standing  otherwise

Eating/drinking mean "standing at the trough", the usual proxy for feeding and
drinking time; the camera cannot see whether the cow is actually swallowing.
Walking is only computed for fixed cameras (videos that have zones).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

BEHAVIOURS = ("lying", "eating", "drinking", "walking", "standing")
ZONE_FRAC = 0.12      # min fraction of the box inside the trough zone
MIN_EAT_S = 3.0       # dwell time before "at feed trough" counts as eating
MIN_DRINK_S = 2.0
WALK_SPEED = 0.25     # body heights per second
LYING_WINDOW_S = 3.0
LYING_RATIO = 0.72
GAP_S = 1.5           # tolerated gaps (missed detections / brief exits) inside a dwell


def _dwell_mask(times: np.ndarray, inside: np.ndarray, min_s: float) -> np.ndarray:
    """True where the cow is inside and the surrounding run of 'inside' lasts >= min_s."""
    out = np.zeros(len(times), bool)
    start = None
    last_in = None
    for i, (t, ins) in enumerate(zip(times, inside)):
        if ins:
            if start is None or (last_in is not None and t - times[last_in] > GAP_S):
                start = i
            last_in = i
            if times[i] - times[start] >= min_s:
                out[start:i + 1] = inside[start:i + 1]
        # outside samples do not reset the run unless the gap grows beyond GAP_S
    return out


def label_behaviours(frames: List[dict], aux: Dict[int, List[list]], fixed_camera: bool) -> Dict[str, int]:
    """Write frame['b'] (one label per detection, aligned with frame['d']).

    aux[frame_index] = [[feed, water, p_lying_or_None, height_ratio_or_None], ...] aligned with frame['d'].
    Returns total seconds-weighted counts per behaviour (in processed samples).
    """
    per_track = defaultdict(list)  # id -> [(frame_pos, det_pos, t, box, feed, water, plying)]
    for fi, fr in enumerate(frames):
        a = aux.get(fr["f"], [])
        fr["b"] = ["standing"] * len(fr["d"])
        for di, d in enumerate(fr["d"]):
            feed, water, pl, hr = (list(a[di]) + [None])[:4] if di < len(a) else (0.0, 0.0, None, None)
            per_track[d[0]].append((fi, di, fr["t"], d[2:6], feed, water, pl, hr))

    for tid, obs in per_track.items():
        t = np.array([o[2] for o in obs])
        boxes = np.array([o[3] for o in obs], float)
        feed = np.array([o[4] for o in obs]) >= ZONE_FRAC
        water = np.array([o[5] for o in obs]) >= ZONE_FRAC
        pl = np.array([np.nan if o[6] is None else o[6] for o in obs])
        hr = np.array([np.nan if o[7] is None else o[7] for o in obs])

        # walking: centre speed over ~1 s, normalised by box height
        walking = np.zeros(len(obs), bool)
        if fixed_camera and len(obs) > 1:
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            hgt = np.maximum(boxes[:, 3] - boxes[:, 1], 1)
            j = 0
            for i in range(len(obs)):
                while t[i] - t[j] > 1.0:
                    j += 1
                if i > j and t[i] - t[j] > 0.3:
                    v = np.hypot(cx[i] - cx[j], cy[i] - cy[j]) / (t[i] - t[j]) / np.median(hgt[j:i + 1])
                    walking[i] = v > WALK_SPEED

        # lying: rolling mean of P(lying) over the previous LYING_WINDOW_S
        lying = np.zeros(len(obs), bool)
        use_model = not np.all(np.isnan(pl))
        if use_model or not np.all(np.isnan(hr)):
            j = 0
            for i in range(len(obs)):
                while t[i] - t[j] > LYING_WINDOW_S:
                    j += 1
                if use_model:
                    w = pl[j:i + 1]
                    w = w[~np.isnan(w)]
                    lying[i] = len(w) > 0 and float(w.mean()) >= 0.5
                else:
                    w = hr[j:i + 1]
                    w = w[~np.isnan(w)]
                    lying[i] = len(w) > 0 and float(np.median(w)) < LYING_RATIO

        pl_missing = np.all(np.isnan(pl))  # height-ratio rule (no posture model) needs the trough guard
        drinking = _dwell_mask(t, water, MIN_DRINK_S) & ~walking
        eating = _dwell_mask(t, feed, MIN_EAT_S) & ~walking

        for k, o in enumerate(obs):
            # cows with their head in the feed trough are standing, whatever their box looks like
            lab = ("lying" if lying[k] and not walking[k] and not (feed[k] and pl_missing) else "drinking" if drinking[k] else "eating" if eating[k]
                   else "walking" if walking[k] else "standing")
            frames[o[0]]["b"][o[1]] = lab

    totals = {b: 0 for b in BEHAVIOURS}
    for fr in frames:
        for lab in fr.get("b", []):
            totals[lab] += 1
    return totals


def track_behaviour_seconds(frames: List[dict], sample_dt: float) -> Dict[int, Dict[str, float]]:
    """Seconds per behaviour per track id (each processed sample represents sample_dt seconds)."""
    out: Dict[int, Dict[str, float]] = defaultdict(lambda: {b: 0.0 for b in BEHAVIOURS})
    for fr in frames:
        for d, lab in zip(fr["d"], fr.get("b", [])):
            if d[0] is not None:
                out[d[0]][lab] += sample_dt
    return {k: {b: round(v, 1) for b, v in d.items()} for k, d in out.items()}
