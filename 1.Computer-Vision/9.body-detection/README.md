# Body & Hand Motion Detection

A webcam app that does 3 things at once:
1. Tracks your hand and fingers in real time.
2. Recognizes common objects in view (person, phone, bottle, etc.).
3. Notices when *anything* moves, even things it doesn't recognize.

Built to run smoothly on a laptop **without a gaming GPU** (i5 11th-gen, 16GB
RAM, MX350 2GB VRAM) — everything here is light enough to run on CPU alone.

## What is this project, really?

Think of it like giving your webcam 3 different pairs of glasses, worn at
the same time:

- **Glasses #1 — Hand glasses.** Only looks for hands. Once it finds one, it
  marks 21 specific points on it (knuckles, fingertips, wrist) like sticking
  dots on a skeleton. This is how it "knows" where your fingers are and how
  fast they're moving.
- **Glasses #2 — Object glasses.** Scans the whole picture and asks "have I
  seen something like this before?" for ~80 common object categories (person,
  cup, phone, scissors...). If yes, draws a box around it with a label.
- **Glasses #3 — Movement glasses.** Doesn't know *what* anything is. Just
  compares "what the room looked like a second ago" vs "now" and flags
  whatever changed. This catches things the other two glasses don't even
  have a name for — like someone waving a pencil.

All 3 run on every webcam frame, and their results get drawn on top of each
other onto one video window.

## The models behind it (and which ones are "deep learning")

This is a good project to see the difference between **deep learning** (a
neural network that learned from millions of examples) and **classic
computer vision** (a fixed mathematical rule, no learning involved):

| Glasses | Technique | Is it deep learning? | How it actually works |
|---|---|---|---|
| Hand tracking | **MediaPipe HandLandmarker** | **Yes** | A neural network trained on huge numbers of hand photos. It learned to output the (x, y) position of 21 hand landmarks for any hand image. Like a model that learned to predict "where are the freckles on this face," except for hand joints. |
| Object detection | **YOLOv8n** ("You Only Look Once", nano size) | **Yes** | A convolutional neural network (CNN) trained on the COCO dataset (~330,000 labeled photos). It slides over the image and, in one single pass, predicts "is there an object here, and which of the 80 categories is it?" The "nano" version is the smallest YOLOv8 size — fewer layers, less accurate but much faster, made for weak hardware. |
| Motion detection | **MOG2 background subtraction** | **No — classic CV, no learning** | It builds a running statistical average of "what the background normally looks like" pixel-by-pixel, frame after frame. Anything that suddenly doesn't match that average (a pixel got way brighter/darker than its recent history) is flagged as "foreground" = motion. No training, no dataset, just stats (a Gaussian Mixture Model per pixel) — that's literally what "MOG" stands for. |

Why mix a non-deep-learning method (MOG2) into a deep learning project?
Because it's a free safety net — it's nearly zero CPU cost and catches
movement that the trained models were never taught to recognize (a pencil
isn't a "thing" YOLO knows, but MOG2 doesn't care what it is, only that it
moved).

## Why these specific models (not bigger/fancier ones)

| Need | Choice | Why this one |
|---|---|---|
| Hand/finger landmarks | **MediaPipe Hands** | Built specifically for hands, runs fast on CPU alone, and Google already trained it — no need to train your own model. |
| Object detection | **YOLOv8n (nano)** | The smallest YOLOv8 variant. Bigger YOLO versions (s/m/l/x) are more accurate but need more compute than this laptop's GPU (2GB VRAM) comfortably handles. |
| Motion detection | **MOG2 (OpenCV built-in)** | Practically free computationally — no neural network involved at all. |
| Video I/O / drawing | **OpenCV** | The standard library for reading webcam frames and drawing boxes/text on them. |

We deliberately skipped heavier options (full-body pose models, OpenPose,
bigger YOLO sizes) — they'd be more accurate but would make the laptop choppy
for little real benefit here.

## Honest limitations

- YOLO's 80 categories do **not** include "pen"/"pencil" — it'll often miss
  it or guess wrong, because it was never trained on that category at all.
  (This is why the motion-detection "glasses" exist as backup — it still
  notices the movement even if it can't name the object.)
- To make YOLO recognize a *specific* object like your own pen, you'd need to
  collect photos of it and retrain ("fine-tune") the model — a good follow-up
  project, not needed to get started.
- The MX350 GPU *can* run YOLOv8n if you install `torch` with CUDA, but
  laptop GPU driver/CUDA mismatches are common and annoying — CPU mode is the
  safe default here and is still fast enough.

## How to Run

1. **Open a terminal in this folder** (`9.body-detection/`).

2. **Create + activate a virtual environment** (recommended, keeps deps isolated):

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run it:**

   ```bash
   python main.py
   ```

   First run needs internet once — it auto-downloads `yolov8n.pt` (~6MB,
   ultralytics) and `models/hand_landmarker.task` (~7.6MB, Google). After
   that it works fully offline.

5. A window titled **"Body Detection - Hands + Objects"** opens using your
   default webcam. Move your hand to see the skeleton + speed readout; point
   objects (phone, bottle, cup, etc.) at the camera to see boxes + labels;
   any movement at all triggers the orange "MOTION DETECTED" boxes.

6. Press **`q`** (with the window focused) to quit.

> Note: newer `mediapipe` (0.10.30+, required for Python 3.13) dropped the old
> `mp.solutions.hands` API in favor of the Tasks API (`HandLandmarker`), which
> needs that `.task` model file instead of being bundled in the pip package.
> Already handled for you — no manual step needed.

### Troubleshooting

- **Webcam doesn't open / black window** — try changing `CAM_INDEX` in
  `main.py` (0, 1, 2...) if you have multiple cameras, or close other apps
  using the webcam (Zoom, Teams, browser tabs).
- **Laggy / low FPS** — raise `DETECT_EVERY_N_FRAMES` or lower
  `FRAME_WIDTH`/`FRAME_HEIGHT` in `main.py` (see Tuning knobs below).
- **`mediapipe` install fails** — make sure you're on Python 3.9–3.13 (check
  with `python --version`); mediapipe doesn't ship wheels for very new or
  very old Python versions.

## Project structure

```
9.body-detection/
├── main.py                  # webcam loop, combines all 3 "glasses"
├── src/
│   ├── hand_tracker.py       # deep learning: MediaPipe HandLandmarker
│   ├── object_detector.py    # deep learning: YOLOv8n
│   └── motion_detector.py    # classic CV: MOG2 background subtraction
├── requirements.txt
└── README.md
```

## Tuning knobs (in `main.py`)

- `DETECT_EVERY_N_FRAMES` — raise it (e.g. 10) if FPS feels low, lower it (e.g. 2, current default) for snappier object labels.
- `FRAME_WIDTH` / `FRAME_HEIGHT` — default 1280x720 for a bigger window; drop to 640x480 if FPS suffers. Console prints the actual resolution your webcam gave (some cams ignore the request and fall back to a lower native res).
- `YOLO_DEVICE` — `"cpu"` (default, safe) or `"0"` to try MX350 GPU if you have `torch`+CUDA set up.
- `YOLO_CONF` — default 0.25 (catches more objects, incl. smaller/farther ones like a phone). Raise toward 0.5+ if you get too many false-positive boxes.
- `MOTION_MIN_AREA` — default 800px. Raise it if small noise/lighting flicker triggers false "MOTION DETECTED"; lower it to catch subtler movement.

## Next steps / ideas

- Add gesture recognition (e.g. detect "pinch", "fist", "open palm") using
  the landmark positions already being tracked — good next exercise once
  you're comfortable with this baseline.
- Fine-tune YOLOv8n on a small custom dataset (e.g. via Roboflow) if you want
  it to recognize specific objects like a particular pen or tool.
- Swap in MediaPipe Pose (full body skeleton) alongside hands if you want
  body posture, not just hands — same CPU-friendly tradeoff applies.
