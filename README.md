# 🧩 Pinch Puzzle — Solve a Photo Puzzle With Your Hands

A browser-based photo puzzle you play using hand gestures instead of a mouse — pinch to capture a photo of yourself, then pinch-drag the tiles to solve it.

---

## 🔎 Overview

Pinch Puzzle turns your webcam into the only input device you need. Instead of clicking and dragging with a mouse, you use real hand gestures — a two-hand pinch to snap a photo, a pinch-and-drag to swap tiles, and a closed fist to reshuffle.

What makes it interesting is that there's no upload step and no backend: all hand tracking happens live in the browser using MediaPipe's Hand Landmarker model, and the puzzle itself is a **free-swap** puzzle rather than a classic sliding puzzle — there's no empty slot, so *any* tile can be dropped on *any* other tile.

The project ships in two forms that share the same gestures and rules:
- a **web version** (`docs/index.html`) — open it in a browser, nothing to install
- a **desktop version** (`desktop/pinch_puzzle.py`) — runs locally with Python + OpenCV

---

## ✨ Features

- **Two-hand pinch capture** — hold up two hands, pinch both at once, and it snaps a photo after a brief hold
- **Free-swap puzzle** — pinch any tile and drag it onto any other tile to swap them (no blank slot required)
- **Fist-to-reshuffle** — hold a closed fist briefly to scramble the board again
- **Live solve timer** — starts on your first move, stops the moment the puzzle is solved
- **Leaderboard** — save your name and time after solving; best times are kept per name (a faster run updates your existing entry instead of adding a duplicate). Shared globally across everyone who plays if a Firebase backend is configured (see below); otherwise stored locally in your browser.
- **Adjustable grid size** — 3×3, 4×4, or 5×5
- **Adjustable pinch sensitivity** — tune the pinch threshold to match your camera/lighting
- **Debug HUD** — an on-screen readout of hand detection, pinch ratio, and fist-curl count, for diagnosing gesture issues
- **Full keyboard/mouse fallback** — the whole game is playable without gestures if tracking is unreliable

---

## 🎮 How to Play

1. **Capture** — show two hands to the camera and pinch (thumb + index touching) on *both* hands at the same time. Hold briefly and it captures a photo automatically.
2. **Solve** — pinch near a tile to pick it up, drag your hand to any other tile, and release to swap the two.
3. **Reshuffle** — make a closed fist and hold it for under a second.
4. **Win** — once every tile is back in its original position, a "Solved" screen shows your time.
5. **Save your score** — type your name and confirm to add it to the leaderboard.

**Camera permission is required** — the browser will prompt you to allow camera access before the game starts.

### Fallback controls (mouse/keyboard)

| Input | Action |
|---|---|
| `1` / `2` / `3` | Grid size 3×3 / 4×4 / 5×5 |
| `Space` | Manual capture |
| Click/tap one tile, then another | Swap those two tiles |
| `r` | Reshuffle |
| `c` | Back to camera |
| `d` | Toggle debug HUD |
| `+` / `-` | Tune pinch sensitivity |

---

## 🛠️ Technologies Used

**Web version**
- HTML5 Canvas (2D rendering)
- Vanilla JavaScript (ES modules)
- [MediaPipe Tasks Vision — Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (loaded via CDN, runs client-side)
- `localStorage` for the local (per-browser) leaderboard
- [Cloud Firestore](https://firebase.google.com/docs/firestore) REST API (`fetch`, no SDK) for the optional shared/global leaderboard
- Google Fonts (Space Grotesk, IBM Plex Mono)

**Desktop version**
- Python
- [OpenCV](https://opencv.org/) (`opencv-python`) — camera capture and rendering
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) (`mediapipe`) — hand landmark detection
- NumPy

No backend server or database is used in either version.

---

## ⚙️ How It Works

1. The webcam feed is read every frame (via `getUserMedia` in the browser, or OpenCV's `VideoCapture` on desktop).
2. Each frame is passed to the MediaPipe Hand Landmarker, which returns 21 hand landmarks per detected hand.
3. Two custom gesture checks run on those landmarks:
   - **Pinch ratio** — distance between the thumb tip and index tip, normalized by hand size
   - **Fist detection** — checks whether at least 3 of 4 fingers are curled in toward the wrist
4. Gesture state drives the app's state machine: `LIVE` (capture) → `PUZZLE` (solve) → `SOLVED` → `NAME_ENTRY` → `LEADERBOARD`.
5. On capture, the current frame is cropped to a square and sliced into a grid of tiles. The puzzle board is a shuffled array mapping each grid position to a tile.
6. During solving, a pinch picks up whichever tile is under the fingertip; releasing the pinch swaps that tile with whatever tile is under the fingertip at that moment.

```
Webcam Feed
     │
     ▼
MediaPipe Hand Landmarker  ── 21 landmarks per hand
     │
     ▼
Gesture Logic (pinch ratio / fist curl)
     │
     ▼
App State Machine:  LIVE → PUZZLE → SOLVED → NAME_ENTRY → LEADERBOARD
     │
     ▼
Canvas Rendering (video + tiles + HUD)
```

---

## 📁 Project Structure

```
pinch-puzzle/
├── README.md              # you are here
├── LICENSE                # MIT license
├── docs/
│   └── index.html          # web version — single-file app (HTML + CSS + JS)
├── desktop/
│   ├── pinch_puzzle.py     # desktop version (OpenCV + MediaPipe)
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # setup notes specific to the desktop version
└── assets/
    └── demo.mp4          # gameplay demo video
```

---

## 🚀 Installation and Setup

### Web version

No installation needed — it's a single static HTML file.

```bash
git clone https://github.com/<your-username>/pinch-puzzle.git
cd pinch-puzzle/docs
```

Then open `index.html` directly in a browser, or serve it locally:

```bash
python -m http.server 8000
```

Open **http://localhost:8000** in your browser.

> A local server (rather than opening the file directly) is recommended, since some browsers restrict camera access on `file://` URLs.

### Desktop version

```bash
git clone https://github.com/<your-username>/pinch-puzzle.git
cd pinch-puzzle/desktop

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

python pinch_puzzle.py
```

A window will open showing your webcam feed with the game overlaid.

---

## 📷 Camera/Webcam Requirements

- A webcam and **camera permission** are required for both versions.
- The web version needs **HTTPS** (or `localhost`) to access the camera — this is a browser security requirement. Once deployed (e.g. via GitHub Pages), HTTPS is provided automatically.
- Hand tracking runs entirely client-side in the browser — no video is uploaded anywhere.
- If gesture detection is unreliable due to lighting or camera quality, the mouse/keyboard fallback controls (see above) let you play without gestures.

---

## 🌐 Live Demo

Coming soon.

## 🖼️ Screenshots / Demo

### 🎥 Gameplay Demo

[▶️ Watch the Pinch Puzzle Demo](./assets/demo.mp4)

The demo shows:
- Two-hand pinch photo capture
- Pinch-and-drag tile swapping
- Puzzle solving
- Solved screen
- Leaderboard

## 🌍 Shared Leaderboard Setup (optional)

By default the leaderboard is stored in `localStorage`, so it's local to
each browser. To make it shared across everyone who plays:

1. Create a free [Firebase](https://console.firebase.google.com) project and enable **Firestore Database**.
2. Paste the security rules from [`docs/firestore.rules`](./docs/firestore.rules) into Firestore's Rules tab.
3. Copy your Firebase **Project ID** into `FIREBASE_PROJECT_ID` near the top of `docs/index.html`.
4. Redeploy — the leaderboard screen automatically switches to the shared board.

If the shared board is unreachable for any reason, the game falls back to
showing local scores rather than breaking.

---

## 🔭 Future Improvements

- Packaged standalone executable for the desktop version (no Python install required)
- Additional gesture customization options exposed in the UI
- Mobile browser support testing and refinement

---

