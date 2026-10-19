# Cow Detection System — video prototype

Detects, tracks and counts cows in the farm videos in this folder, with a web dashboard
for playback, live statistics, results review and exports.

**All boxes, counts and confidences come from real model inference** (Ultralytics YOLO11,
COCO-pretrained, class 19 `cow`). Nothing in the app is simulated or randomly generated.

## Setup

```bash
cd ~/Downloads/Grad
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights are downloaded automatically into `models/` the first time they are used
(`yolo11s/m/x.pt` are already there).

## Run the dashboard

```bash
.venv/bin/python app.py
```

Open <http://127.0.0.1:8000>. Pick a video at the top, press **Start detection**, and watch
the progress bar and live inference preview. When the run finishes, play the video:
boxes, IDs and counts are drawn in sync with play, pause, seek and frame-step. Earlier runs
for each video are listed in the **Detection results** dropdown.

## Command-line processing (no UI)

```bash
.venv/bin/python video_processor.py                                   # list videos + metadata
.venv/bin/python video_processor.py "WhatsApp Video 2026-09-29 at 20.49.57 (3).mp4"
.venv/bin/python video_processor.py "videos/from_zip_8.55.23/WhatsApp Video 2026-09-29 at 8.53.45 PM.mp4" --stride 2 --conf 0.4 --track-memory 12
```

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | FastAPI server + REST API for the dashboard (entry point) |
| `static/` | Dashboard front end: HTML5 video + canvas overlay, chart, tables |
| `detector.py` | `CowDetector`: YOLO inference limited to cow classes; verifies that the model has one |
| `tracker.py` | BoT-SORT/ByteTrack configs tuned for cattle + per-track statistics |
| `relink.py` | Offline ID repair: splits ID switches, re-links broken tracks of the same cow |
| `video_processor.py` | Video discovery, zip extraction, probing, frame loop, annotation, CSV/JSON/MP4 export, CLI |
| `videos/` | Videos extracted from the two WhatsApp `.zip` files (the zips are left untouched) |
| `models/` | YOLO weights |
| `trackers/` | Generated tracker YAML files |
| `outputs/<video>/<run>/` | `annotated.mp4`, `detections.csv`, `frame_counts.csv`, `detections.json`, `summary.json` |
| `outputs/previews/` | H.264 copies of HEVC videos, used only for browser playback |

Original video files are never modified.

## Settings

| Setting | Default | Notes |
|---|---|---|
| Model | `yolo11m.pt` | `s` is faster; `x` is slower and only marginally better on this footage |
| Confidence | 0.35 | Below ~0.3, background objects (e.g. a parked motorbike) start appearing as cows |
| IoU (NMS) | 0.5 | Raise it in dense feeding lines where cows legitimately overlap a lot |
| Image size | 960 | 1280 helps with small, distant cows in the 1080p CCTV footage |
| Every Nth frame | 1 | Use 2–4 for the 60 fps phone clips; the overlay holds the last result |
| Tracker | BoT-SORT + ReID | Camera-motion compensation + appearance matching |
| ID memory | 8 s | How long a hidden cow keeps its ID |
| Re-link | on | Offline split/re-link pass after detection (see below) |

## Counting semantics

* **Cows visible now**: boxes in the current frame after NMS. Each cow is counted at most once per frame.
* **Unique track IDs**: distinct tracker IDs seen so far / in the whole run. **This is not a head count.**
  IDs change when a cow is hidden (fence rails, other cows), leaves and re-enters, or when a handheld camera
  pans. "Stable" counts only IDs seen in ≥ 5 processed frames, which is a less noisy figure.

## Keeping IDs fixed

A tracker gives each cow a *number*, not an *identity*. IDs break when a cow is hidden longer
than the tracker remembers, when its detection flickers, or when the camera pans. Within one video,
four steps reduce that:

1. **Weak boxes keep tracks alive.** When tracking, the detector runs at confidence 0.1. Boxes below
   your threshold are used only to keep existing IDs alive (ByteTrack's second matching stage), never
   to start new IDs or to be counted.
2. **BoT-SORT + ReID** with camera-motion compensation and an 8 s memory (`--track-memory`).
3. **Split ID switches** (`relink.py`). If a track starts as one cow and becomes another (appearance
   change **and** a box jump), it is cut in two.
4. **Re-link fragments** (`relink.py`). Pieces are joined when they never appear in the same frame and
   look alike. Similarity uses a CNN embedding plus a colour histogram, and the threshold is calibrated
   per video from pairs that are certainly different cows (visible together), so at most ~0.5% of
   different-cow pairs would pass. Re-appearing in the same place lowers the bar.

Measured on clips from this folder (unique IDs; lower is better):

| Clip | max visible | before | after 1–2 | after 3–4 |
|---|---|---|---|---|
| Trough, handheld, 21 s | 6 | 19 | 12 | 13* |
| CCTV, 60 s (camera pans for the first 20 s) | 15 | 58 | 45 | 43 |
| Feeding line, handheld, 15 s | 13–16 | 81 | 59 | – |
| Alley, handheld, 20 s | 11–12 | 52 | 34 | – |

\*Up by one because a real Holstein→black-cow switch was split (one cow with two numbers is a
smaller error than two cows sharing one). On the CCTV clip, during the still-camera period
(20–60 s), 23 IDs cover at most 15 visible cows.

The CSV keeps both `track_id` (final) and `raw_track_id` (from the tracker). `summary.json → relink`
lists every merge threshold and every split.

**Limits:** look-alike cows (all-black animals) that swap places while hidden can still be confused,
and a handheld camera that sweeps back and forth over the herd will always produce several IDs per cow.

**For truly permanent IDs** (the same cow keeps the same number across videos and days), tracking is
not enough; you need *identification* against a registered herd list:
* **Ear-tag reading.** Many cows here wear yellow ear tags. Train a tag detector plus OCR, use it on
  close fixed cameras (feeders, water trough), and assign the tag number to the track.
* **Coat-pattern re-identification** for Holsteins. Collect a few labelled crops per cow, train a
  metric-learning model (e.g. a ResNet/ViT embedding with triplet loss), and match each track against
  the herd gallery. This does not work well for solid black cows; use tags for those.
* **Fixed cameras with a counting line** at gates or feed alleys, instead of handheld video.

## Behaviours (eating, drinking, lying, walking)

Behaviours are computed per cow after detection, tracking and re-linking (`behaviour.py`). They use the
camera's **zones** (`zones.py`, saved in `zones/`), drawn once per camera in the dashboard
(**Edit zones** → draw → **Save for this camera**):

| Zone | Use |
|---|---|
| Pen | Only cows whose feet are inside are counted (removes the neighbouring pen) |
| Feed trough | Eating = standing at the trough (body in the zone **and** feet close to it) for ≥ 3 s, not walking |
| Water trough | Drinking = same, for ≥ 2 s |

* **Lying**: box height compared with the expected height of a *standing* cow at the same image row
  (`standing_height` in the zones file, fitted from ~14,000 boxes of this camera). Lying cows measure
  0.53–0.66 of standing height and standing cows 1.0–1.16; lying = rolling median below 0.72. Skipped for
  cows cut off at the frame edge, hidden behind another cow, at the feed trough, or right under the camera.
* **Walking**: box centre moving faster than 0.25 body-heights per second (fixed cameras only).
* Output: label on every box (dashboard + annotated MP4), `behaviour` column in `detections.csv`,
  per-frame counts in `frame_counts.csv`, seconds per behaviour per cow in `summary.json`.

Tested on the 15 Oct camera (`videos/15-10-first_day`, zones already set):

| Clip | Result |
|---|---|
| 17:30 feeding (2 min) | Cows detected at the feed wall → eating; 1.3% of cow-time falsely "lying" (cows partly hidden in the crowd) |
| 19:59–20:09 night (10 min) | Detected lying cows correctly flagged (ratio 0.53–0.66); a cow lying with its head raised (~0.85) is missed |

**Limitations:**
* The general-purpose detector misses or merges cows in the packed feeding line (once 2 of ~8 found),
  and misses many lying cows at night (at 20:06 none of ~6 detected). Behaviours can only be given to
  detected cows, so eating and lying time are **under-counted**.
* "Eating"/"drinking" mean *at the trough*; the camera cannot see swallowing.
* The water-trough zone was placed from the night footage (the trough is washed out by sun in the day); check it.

**Next step for accuracy:** fine-tune the detector on frames from this camera (crowded feeding line,
lying cows at night). `train_posture.py` + `posture.py` are ready for a lying/standing classifier, but
this footage has too few lying cows (essentially one individual) to train one honestly; record nights
or midday rest periods first. 1,500 candidate crops are in `data/posture/crops/`.
