"""Posture (lying / standing) dataset tools and training.

    # 1. collect cow crops from videos (detector boxes, inside the pen if zones exist)
    .venv/bin/python train_posture.py extract videos/15-10-first_day --per-video 120

    # 2. label them in the dashboard (http://127.0.0.1:8000/static/label.html)
    #    or edit data/posture/labels.csv  (crop,label,labeler) with label in {lying, standing, skip}

    # 3. train + evaluate (writes models/posture_cls.pt and data/posture/report.json)
    .venv/bin/python train_posture.py train --epochs 40

The split is by video chunk + time block, so near-duplicate crops of the same cow
in consecutive frames never end up in both train and validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "posture"
CROPS_DIR = DATA_DIR / "crops"
LABELS_CSV = DATA_DIR / "labels.csv"
CLASSES = ("lying", "standing")


def read_labels() -> dict:
    if not LABELS_CSV.is_file():
        return {}
    with open(LABELS_CSV) as f:
        return {r["crop"]: r for r in csv.DictReader(f)}


def write_labels(rows: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "label", "labeler"])
        w.writeheader()
        for k in sorted(rows):
            w.writerow({"crop": k, "label": rows[k]["label"], "labeler": rows[k].get("labeler", "")})


def extract(folder: str, per_video: int, min_side: int, seed: int, only=None) -> None:
    from detector import CowDetector
    from posture import crop_box
    from video_processor import probe_video
    from zones import ZoneSet, load_zones

    random.seed(seed)
    det = CowDetector("yolo11m.pt", conf=0.45, imgsz=960)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    vids = [v for v in sorted(Path(PROJECT_DIR / folder).glob("*.mp4")) if not only or v.stem in only]
    n_total = 0
    for v in vids:
        rel = str(v.relative_to(PROJECT_DIR))
        info = probe_video(v)
        if not info.get("readable") or not info.get("frames"):
            continue
        zdata, _ = load_zones(rel)
        zs = ZoneSet(zdata, (info["width"], info["height"])) if zdata else None
        cap = cv2.VideoCapture(str(v))
        n_frames = info["frames"]
        n_samples = max(1, per_video // 6)
        got = 0
        for q in sorted(random.sample(range(n_frames), min(n_samples, n_frames))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, q)
            ok, fr = cap.read()
            if not ok:
                continue
            for i, d in enumerate(det.process(fr).detections):
                b = (d.x1, d.y1, d.x2, d.y2)
                if zs is not None and zs.has_pen and not zs.in_pen(b):
                    continue
                if min(d.x2 - d.x1, d.y2 - d.y1) < min_side:
                    continue
                c = crop_box(fr, b)
                if c is None:
                    continue
                name = f"{v.stem}_f{q:07d}_{i:02d}.jpg"
                cv2.imwrite(str(CROPS_DIR / name), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
                got += 1
        cap.release()
        n_total += got
        print(f"{rel}: {got} crops")
    print(f"total {n_total} crops in {CROPS_DIR}")


def _block(name: str) -> str:
    # crops named <video>_f<frame>_<i>.jpg -> group by video + 5-minute block at 15 fps
    stem, f, _ = name.rsplit("_", 2)
    return f"{stem}_{int(f[1:]) // (15 * 300)}"


def train(epochs: int, val_frac: float, seed: int) -> None:
    from ultralytics import YOLO

    from detector import MODELS_DIR, pick_device

    labels = {k: r for k, r in read_labels().items() if r["label"] in CLASSES and (CROPS_DIR / k).is_file()}
    counts = {c: sum(r["label"] == c for r in labels.values()) for c in CLASSES}
    print("labelled:", counts)
    if min(counts.values()) < 15:
        raise SystemExit("Need at least 15 labelled crops per class before training.")

    # group-wise split
    random.seed(seed)
    groups = sorted({_block(k) for k in labels})
    random.shuffle(groups)
    val_groups, n_val = set(), 0
    for g in groups:
        if n_val >= val_frac * len(labels):
            break
        val_groups.add(g)
        n_val += sum(_block(k) == g for k in labels)

    ds = DATA_DIR / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    for split in ("train", "val"):
        for c in CLASSES:
            (ds / split / c).mkdir(parents=True, exist_ok=True)
    for k, r in labels.items():
        split = "val" if _block(k) in val_groups else "train"
        shutil.copy(CROPS_DIR / k, ds / split / r["label"] / k)

    import os
    MODELS_DIR.mkdir(exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(MODELS_DIR)
        model = YOLO("yolo11n-cls.pt")
    finally:
        os.chdir(cwd)
    model.train(data=str(ds), epochs=epochs, imgsz=224, batch=32, device=pick_device(),
                project=str(DATA_DIR / "runs"), name="posture", exist_ok=True, verbose=False, plots=False,
                fliplr=0.5, hsv_v=0.4, degrees=5)
    best = DATA_DIR / "runs" / "posture" / "weights" / "best.pt"

    # evaluate on the held-out groups
    m = YOLO(str(best))
    names = {int(k): v for k, v in m.names.items()}
    conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for c in CLASSES:
        files = sorted((ds / "val" / c).glob("*.jpg"))
        for i in range(0, len(files), 64):
            for r in m.predict([str(f) for f in files[i:i + 64]], imgsz=224, verbose=False):
                conf[c][names[int(r.probs.top1)]] += 1
    tot = sum(sum(v.values()) for v in conf.values())
    acc = sum(conf[c][c] for c in CLASSES) / max(tot, 1)
    recall = {c: conf[c][c] / max(sum(conf[c].values()), 1) for c in CLASSES}
    report = {"val_accuracy": round(acc, 4), "recall": {k: round(v, 4) for k, v in recall.items()},
              "confusion (true -> predicted)": conf, "train_counts": counts, "val_size": tot,
              "split": "by video chunk + 5-minute block"}
    (DATA_DIR / "report.json").write_text(json.dumps(report, indent=2))
    shutil.copy(best, MODELS_DIR / "posture_cls.pt")
    print(json.dumps(report, indent=2))
    print("saved", MODELS_DIR / "posture_cls.pt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("folder")
    e.add_argument("--per-video", type=int, default=120)
    e.add_argument("--min-side", type=int, default=80)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--only", default="", help="comma-separated video stems to use (default: all)")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--val-frac", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.folder, a.per_video, a.min_side, a.seed, [x for x in a.only.split(",") if x])
    else:
        train(a.epochs, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
