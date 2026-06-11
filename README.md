---
title: RoadX Traffic Enforcement
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RoadX — AI Traffic Enforcement System

RoadX is an AI-powered traffic enforcement system that analyzes dashcam or CCTV footage to detect violations, extract vehicle details, and generate challans automatically.

---

## Key Features

- Live RTSP stream and video file processing
- No helmet detection (Sec 129 MV Act — Rs. 1,000)
- Triple riding detection (Sec 128 MV Act — Rs. 1,000)
- Wrong-way driving detection (Sec 184 MV Act — Rs. 5,000)
- Indian license plate recognition (97.6% precision)
- Repeat-offender fine multiplier (1× / 2× / 3×)
- PDF challan generation with QR payment link
- Email notifications with retry logic (3 attempts)
- Login-protected admin dashboard with live MJPEG feed
- Public citizen portal — no login required
- Analytics page with 5 Chart.js charts
- CSV export, mark-as-paid, multi-camera support

---

## Project Structure

```
app.py              — Main Flask app + detection pipeline
violation_engine.py — No-helmet, triple-riding, wrong-way logic
challan.py          — PDF challan generation (ReportLab)
notifications.py    — Email / WhatsApp notification system
vahan.py            — Vehicle owner lookup (mock; real API drop-in ready)
plate_ocr.py        — Advanced Indian plate OCR (used by detect_video.py)
detect.py           — Batch image detection script
detect_video.py     — Batch video processing script
evaluate_model.py   — Model evaluation (best.pt + Plate.pt)
templates/          — Flask HTML templates
static/             — Screenshots, challans
videos/             — Input video files
models/             — YOLOv8 weights (tracked with Git LFS)
```

---

## Setup

```bash
git clone https://github.com/i-am-Ankush/RoadX-traffic-enforcement-system.git
cd RoadX-traffic-enforcement-system

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root (never commit this):

```
SECRET_KEY=your-random-secret-here
ADMIN_PASSWORD=your-admin-password

# Optional — email notifications
GMAIL_ADDRESS=your@gmail.com
GMAIL_APP_PASS=your-app-password

CITIZEN_EMAIL=citizen@example.com
ADMIN_EMAIL=admin@example.com

# Optional — Vahan API
VAHAN_API_KEY=

# Demo owner details (shown when plate not in mock DB)
DEMO_NAME=Demo Owner
DEMO_PHONE=+910000000000
DEMO_EMAIL=demo@example.com
```

---

## Usage

### Web dashboard (recommended)

```bash
python app.py
```

Open `http://localhost:5001/citizen` — public citizen portal  
Open `http://localhost:5001/login` — admin login (traffic police)

From the admin dashboard, select a video file or enter an RTSP URL and click **▶ ADD**.

### Batch video processing

```bash
python detect_video.py
```

Processes all `.mp4 / .avi / .mov / .mkv` files in the `videos/` folder.  
Annotated output saved to `video_results/`.

### Batch image detection

```bash
python detect.py
```

Processes all images in `images/`. Results saved to `results/`.

### Model evaluation

```bash
python evaluate_model.py
```

Runs YOLO `val()` on held-out test splits for `best.pt` and `Plate.pt`.  
Results saved to `results/model_eval/`.

---

## Models

| Model | Type | Purpose | mAP\@0.5 |
|---|---|---|---|
| `yolov8s.pt` | COCO pretrained | Vehicle + person detection, ByteTrack | — |
| `best.pt` | Custom trained | Helmet / no-helmet classification | 76.5% |
| `Plate.pt` | Custom trained | Indian license plate localisation | 95.9% avg |

Models are tracked with Git LFS — install with `git lfs install` before cloning.

---

## Performance (Apple M4 CPU)

| Step | Latency |
|---|---|
| Traffic + helmet detection | ~185ms/frame |
| Plate OCR | ~43ms/crop |
| Full pipeline | ~235ms → ~4-6 FPS |
| PDF + email (background thread) | 3-7s — does not block the feed |

**Multi-camera note:** On CPU, each additional camera reduces per-camera FPS proportionally due to the GIL. GPU deployment (e.g. Hugging Face Spaces) gives 5×+ improvement.

---

## Docker

```bash
docker-compose up
```

Add videos to `./videos/`, set env vars in `.env`. Challans and screenshots persist in `./static/`.

---

## Disclaimer

Educational project. Not for production law enforcement use.