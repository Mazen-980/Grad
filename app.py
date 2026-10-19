"""Cow Detection dashboard — entry point.

Run:
    .venv/bin/python app.py            # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import DEFAULT_MODEL, list_models, pick_device
from zones import ZONE_TYPES, delete_zones, load_zones, save_zones
from video_processor import (OUTPUT_DIR, PROJECT_DIR, ProcessingSettings, VideoJob, discover_videos,
                             extract_archives, list_runs, playable_path, probe_video)

STATIC_DIR = PROJECT_DIR / "static"
app = FastAPI(title="Cow Detection System")
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

JOBS: Dict[str, VideoJob] = {}
ACTIVE = ("queued", "loading", "running", "relinking", "exporting", "saving")
_probe_cache: Dict[str, dict] = {}
_preview_lock = threading.Lock()


def _video_file(rel: str) -> Path:
    """Only allow videos that discovery found (prevents path traversal)."""
    if rel not in {v["path"] for v in discover_videos()}:
        raise HTTPException(404, f"Unknown video: {rel}")
    return PROJECT_DIR / rel


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/videos")
def videos():
    out = []
    for v in discover_videos():
        p = PROJECT_DIR / v["path"]
        key = f"{v['path']}|{p.stat().st_mtime}"
        if key not in _probe_cache:
            _probe_cache[key] = probe_video(p)
        info = _probe_cache[key]
        out.append({**v, **{k: info.get(k) for k in
                            ("readable", "error", "width", "height", "fps", "frames", "duration", "codec")},
                    "runs": len(list_runs(v["path"]))})
    return out


@app.get("/api/config")
def config():
    return {"models": list_models(), "default_model": DEFAULT_MODEL, "device": pick_device(),
            "defaults": ProcessingSettings().__dict__}


@app.get("/api/stream")
def stream(path: str = Query(...)):
    src = _video_file(path)
    try:
        with _preview_lock:  # one transcode at a time
            playable = playable_path(src)
    except Exception as e:
        raise HTTPException(500, f"Could not prepare a browser-playable preview: {e}")
    return FileResponse(playable, media_type="video/mp4")


class JobRequest(BaseModel):
    video: str
    model: str = DEFAULT_MODEL
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 960
    stride: int = 1
    tracker: str = "botsort"
    track_memory_s: float = 8.0
    relink: bool = True
    export_video: bool = True
    max_seconds: Optional[float] = None
    start_seconds: float = 0.0
    behaviours: bool = True


@app.post("/api/jobs")
def start_job(req: JobRequest):
    _video_file(req.video)
    if any(j.status in ACTIVE for j in JOBS.values()):
        raise HTTPException(409, "A video is already being processed. Stop it first.")
    if not (0.01 <= req.conf <= 0.99 and 0.05 <= req.iou <= 0.95):
        raise HTTPException(400, "conf must be in [0.01, 0.99] and iou in [0.05, 0.95]")
    s = ProcessingSettings(model=req.model, conf=req.conf, iou=req.iou, imgsz=int(req.imgsz) // 32 * 32 or 640,
                           stride=max(1, int(req.stride)), tracker=req.tracker,
                           track_memory_s=min(max(float(req.track_memory_s), 1.0), 120.0), relink=req.relink,
                           export_video=req.export_video,
                           max_seconds=req.max_seconds or None, start_seconds=max(0.0, req.start_seconds or 0.0),
                           behaviours=req.behaviours)
    job = VideoJob(req.video, s).start()
    JOBS[job.id] = job
    return {"id": job.id}


def _job(job_id: str) -> VideoJob:
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0):
    return JSONResponse(_job(job_id).status_dict(since))


@app.post("/api/jobs/{job_id}/stop")
def job_stop(job_id: str):
    _job(job_id).stop()
    return {"ok": True}


@app.get("/api/jobs/{job_id}/preview.jpg")
def job_preview(job_id: str):
    jpg = _job(job_id).preview_jpeg
    if not jpg:
        return Response(status_code=204)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/zones")
def get_zones(video: str):
    _video_file(video)
    data, scope = load_zones(video)
    return {"zones": data, "scope": scope, "types": ZONE_TYPES}


class ZonesRequest(BaseModel):
    video: str
    scope: str = "folder"  # folder = every video from this camera folder; video = only this file
    data: dict


@app.put("/api/zones")
def put_zones(req: ZonesRequest):
    _video_file(req.video)
    if req.scope not in ("folder", "video"):
        raise HTTPException(400, "scope must be 'folder' or 'video'")
    old, _ = load_zones(req.video)
    data = dict(req.data)
    if old and old.get("standing_height") and "standing_height" not in data:
        data["standing_height"] = old["standing_height"]  # keep the camera calibration
    p = save_zones(req.video, data, req.scope)
    return {"ok": True, "file": str(p.relative_to(PROJECT_DIR))}


@app.delete("/api/zones")
def del_zones(video: str, scope: str):
    _video_file(video)
    return {"deleted": delete_zones(video, scope)}


@app.get("/api/active")
def active_job():
    for j in JOBS.values():
        if j.status in ACTIVE:
            return {"id": j.id, "video": j.video_rel}
    return {"id": None}


@app.get("/api/runs")
def runs(video: Optional[str] = None):
    return [{k: r.get(k) for k in ("run_id", "run_dir", "video", "status", "created", "settings", "stats", "files")}
            for r in list_runs(video)]


@app.get("/api/run")
def run_detail(run_dir: str):
    d = (PROJECT_DIR / run_dir).resolve()
    if OUTPUT_DIR.resolve() not in d.parents or not (d / "summary.json").exists():
        raise HTTPException(404, "Unknown run")
    summary = json.loads((d / "summary.json").read_text())
    frames = json.loads((d / "detections.json").read_text())["frames"] if (d / "detections.json").exists() else []
    summary["run_dir"] = run_dir
    return {"summary": summary, "frames": frames}


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    new = extract_archives()
    if new:
        print("Extracted from zip archives:", *new, sep="\n  ")
    print(f"Cow Detection dashboard -> http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
