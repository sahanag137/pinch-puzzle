"""
Webcam Pinch Puzzle v3 (free drag-and-drop, no blank slot required)
=====================================================================
PHASE 1: CAPTURE
  - Show two hands to the camera ("form a frame")
  - Pinch BOTH hands (thumb+index touching on each hand) and hold briefly -> SNAP

PHASE 2: SOLVE
  - Pinch near a tile to pick it up, drag it over ANY other tile, release to
    SWAP the two tiles. Every slot is always filled -- there is no blank tile
    and no adjacency requirement. Drop wherever you like, every frame updates.
  - Hold a closed fist for a moment -> reshuffle (reset)
  - A timer runs from your first move until the puzzle is solved

ON SOLVE
  - "PUZZLE SOLVED!" screen with your time
  - Type your name, press ENTER to save it to the leaderboard
  - Leaderboard screen shows best times, sorted fastest first

Keyboard is still fully supported as a fallback (mouse click / keys), in
case gesture detection is unreliable on your webcam/lighting:
  1 / 2 / 3   -> grid size 3x3 / 4x4 / 5x5   (LIVE screen)
  SPACE       -> manual capture              (LIVE screen)
  click,click -> click one tile then another to swap them (PUZZLE screen)
  r           -> manual reshuffle            (PUZZLE / SOLVED screen)
  c           -> back to camera              (PUZZLE / SOLVED / LEADERBOARD)
  ENTER       -> confirm name / continue     (NAME_ENTRY / LEADERBOARD)
  +/-         -> tune pinch sensitivity      (LIVE / PUZZLE screen)
  d           -> toggle debug HUD            (PUZZLE screen)
  ESC / q     -> quit

Install dependencies first:
    pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import json
import os

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720
PINCH_ON_THRESHOLD = 0.55     # thumb-tip<->index-tip distance / hand size. Tune with +/-.
PINCH_HOLD_SECONDS = 0.4      # how long a two-hand pinch must be held to SNAP
CAPTURE_COOLDOWN = 2.0
FIST_HOLD_SECONDS = 0.8       # how long a fist must be held to trigger reshuffle
DEFAULT_GRID = 3
CANVAS_SIZE = 600
LEADERBOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.json")
LEADERBOARD_MAX = 10
MAX_HAND_LOST_SECONDS = 2.0   # safety net: force-drop only if hand tracking is lost this long mid-drag

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# --------------------------------------------------------------------------
# Hand tracking helpers
# --------------------------------------------------------------------------
class HandTracker:
    def __init__(self, max_hands=2):
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.hands.process(rgb)

    def close(self):
        self.hands.close()


def pinch_ratio(landmarks):
    thumb, index = landmarks[4], landmarks[8]
    wrist, middle_mcp = landmarks[0], landmarks[9]
    d = math.hypot(thumb.x - index.x, thumb.y - index.y)
    hand_size = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y) + 1e-6
    return d / hand_size


def pinch_point_norm(landmarks):
    thumb, index = landmarks[4], landmarks[8]
    return (thumb.x + index.x) / 2, (thumb.y + index.y) / 2


def is_fist(landmarks):
    """Curled fingers (index/middle/ring/pinky) are closer to the wrist than
    their base knuckle -> fist. Needs at least 3 of 4 curled to count.
    Returns (is_fist, curled_count) so the debug HUD can show live counts."""
    wrist = landmarks[0]
    finger_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]  # (tip, mcp)
    curled = 0
    for tip_i, mcp_i in finger_pairs:
        tip, mcp = landmarks[tip_i], landmarks[mcp_i]
        d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        d_mcp = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
        if d_tip < d_mcp * 1.05:
            curled += 1
    return curled >= 3, curled


# --------------------------------------------------------------------------
# Puzzle logic (free-swap: every slot always has a real tile, no blank)
# --------------------------------------------------------------------------
class Puzzle:
    def __init__(self, image_bgr, grid):
        self.grid = grid
        self.tiles = self._slice_image(image_bgr, grid)
        self.board = list(range(grid * grid))
        self.shuffle()

    def _slice_image(self, image_bgr, grid):
        h, w = image_bgr.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        square = image_bgr[y0:y0 + side, x0:x0 + side]
        tile_size = side // grid
        square = cv2.resize(square, (tile_size * grid, tile_size * grid))
        tiles = {}
        idx = 0
        for r in range(grid):
            for c in range(grid):
                tiles[idx] = square[r * tile_size:(r + 1) * tile_size,
                                     c * tile_size:(c + 1) * tile_size]
                idx += 1
        return tiles

    def shuffle(self):
        """Randomly permute every tile. No blank, no adjacency constraint.
        Guaranteed not to land already-solved."""
        solved = list(range(self.grid * self.grid))
        while True:
            random.shuffle(self.board)
            if self.board != solved:
                break

    def swap(self, pos_a, pos_b):
        """Swap whatever tiles sit at pos_a and pos_b. Always legal (as long
        as both positions are valid and different) -- this is the whole point
        of the free-drag redesign: drop a tile on ANY other tile to swap."""
        n = len(self.board)
        if pos_a is None or pos_b is None or pos_a == pos_b:
            return False
        if not (0 <= pos_a < n) or not (0 <= pos_b < n):
            return False
        self.board[pos_a], self.board[pos_b] = self.board[pos_b], self.board[pos_a]
        return True

    def is_solved(self):
        return self.board == list(range(self.grid * self.grid))

    def render(self, canvas_size, dragging_pos=None):
        grid = self.grid
        ts = canvas_size // grid
        canvas = np.zeros((ts * grid, ts * grid, 3), dtype=np.uint8)
        for pos, tile_id in enumerate(self.board):
            r, c = divmod(pos, grid)
            y0, x0 = r * ts, c * ts
            if pos == dragging_pos:
                # tile is being dragged -> leave its slot looking "lifted"
                # (dim placeholder); the floating copy follows the fingertip,
                # drawn separately in render_puzzle().
                cv2.rectangle(canvas, (x0, y0), (x0 + ts, y0 + ts), (40, 40, 40), -1)
                cv2.rectangle(canvas, (x0, y0), (x0 + ts, y0 + ts), (0, 255, 255), 2)
                continue
            patch = cv2.resize(self.tiles[tile_id], (ts, ts))
            canvas[y0:y0 + ts, x0:x0 + ts] = patch
            cv2.rectangle(canvas, (x0, y0), (x0 + ts, y0 + ts), (20, 20, 20), 2)
        return canvas, ts


# --------------------------------------------------------------------------
# Leaderboard
# --------------------------------------------------------------------------
def load_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        return []
    try:
        with open(LEADERBOARD_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_score(name, seconds, grid):
    name = (name or "PLAYER").strip() or "PLAYER"
    seconds = round(seconds, 1)
    board = load_leaderboard()
    # if this name is already on the board, update their best time instead
    # of adding a duplicate row -- only if the new run is actually faster
    existing = next((e for e in board if e["name"].strip().lower() == name.lower()), None)
    if existing:
        if seconds < existing["time"]:
            existing["time"] = seconds
            existing["grid"] = f"{grid}x{grid}"
    else:
        board.append({"name": name, "time": seconds, "grid": f"{grid}x{grid}"})
    board.sort(key=lambda e: e["time"])
    board = board[:LEADERBOARD_MAX]
    try:
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(board, f, indent=2)
    except Exception:
        pass
    return board


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.tracker = HandTracker(max_hands=2)

        self.state = "LIVE"  # LIVE | PUZZLE | SOLVED | NAME_ENTRY | LEADERBOARD
        self.grid = DEFAULT_GRID
        self.puzzle = None
        self.pinch_threshold = PINCH_ON_THRESHOLD

        # capture (two-hand pinch)
        self.two_hand_pinch_start = None
        self.last_capture_time = 0.0

        # solve phase gesture state
        self.held_pos = None          # board position currently "picked up"
        self.was_pinching = False     # only updated on frames where a hand IS detected
        self.pinch_start_time = None
        self.last_hand_seen_time = None  # for the hand-lost safety net, not hold duration
        self.fist_start_time = None
        self.cursor_norm = None       # (x, y) normalized, for drawing + hit-testing
        self.last_cursor_px = None    # last known on-canvas pixel of the pinch point
        self.debug_hud = False
        self.last_ratio = None
        self.last_curl_count = None
        self.feedback_text = ""
        self.feedback_until = 0.0
        self.feedback_color = (0, 255, 0)

        # mouse fallback (click one tile, click another -> swap)
        self.mouse_held_pos = None

        # timer
        self.solve_start_time = None
        self.solve_end_time = None

        self.name_buffer = ""

        self.window = "Webcam Pinch Puzzle"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)

        self.tile_px = None
        self.puzzle_offset = (0, 0)
        self._last_frame_snapshot = None

    # ---- mouse fallback: click a tile, click another -> swap ----
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.state != "PUZZLE" or not self.puzzle or not self.tile_px:
            return
        ox, oy = self.puzzle_offset
        pos = self._pixel_to_pos(x - ox, y - 70 - oy)  # -70 header, then box offset within the video
        if pos is None:
            return
        if self.mouse_held_pos is None:
            self.mouse_held_pos = pos
            self._set_feedback("Picked up (click destination)", (0, 200, 255), 2.0)
        else:
            self._attempt_move(self.mouse_held_pos, pos)
            self.mouse_held_pos = None

    def _pixel_to_pos(self, x, y):
        grid = self.puzzle.grid
        board_px = self.tile_px * grid
        if x < 0 or y < 0 or x >= board_px or y >= board_px:
            return None
        c, r = x // self.tile_px, y // self.tile_px
        return r * grid + c

    def _set_feedback(self, text, color, seconds=1.0):
        self.feedback_text = text
        self.feedback_color = color
        self.feedback_until = time.time() + seconds

    def _attempt_move(self, from_pos, to_pos):
        """Swap the tile at from_pos with whatever is at to_pos. Free swap --
        always succeeds as long as both positions are valid and different."""
        if self.solve_start_time is None:
            self.solve_start_time = time.time()
        moved = self.puzzle.swap(from_pos, to_pos)
        if moved:
            self._set_feedback("Swapped!", (0, 255, 0), 0.6)
            if self.puzzle.is_solved():
                self.solve_end_time = time.time()
                self.state = "SOLVED"
        else:
            self._set_feedback("Drop missed the grid", (0, 0, 255), 1.0)
        return moved

    # ---- shared hand processing ----
    def read_hands(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        frame = cv2.flip(frame, 1)
        results = self.tracker.process(frame)
        return frame, results

    # ---- LIVE / capture phase ----
    def render_live(self):
        raw_frame, results = self.read_hands()
        if raw_frame is None:
            return None
        clean_frame = raw_frame.copy()   # untouched -> used for the actual capture
        frame = raw_frame                # this one gets all the overlay drawing
        h, w = frame.shape[:2]

        pinched_hands = 0
        if results and results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                ratio = pinch_ratio(lm.landmark)
                px, py = pinch_point_norm(lm.landmark)
                cx, cy = int(px * w), int(py * h)
                is_pinch = ratio < self.pinch_threshold
                pinched_hands += int(is_pinch)
                color = (0, 255, 0) if is_pinch else (0, 200, 255)
                cv2.circle(frame, (cx, cy), 14, color, 3)

        both_pinching = pinched_hands >= 2
        num_hands = len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0

        # header
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "PHASE 1: CAPTURE", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "1. Form a frame with two hands", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "2. Pinch both hands to SNAP", (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"hands: {num_hands}/2  grid:{self.grid}x{self.grid}(1/2/3)  SPACE=manual  +/- sens={self.pinch_threshold:.2f}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # progress ring while both hands pinch
        now = time.time()
        if now - self.last_capture_time < CAPTURE_COOLDOWN:
            self.two_hand_pinch_start = None
        elif both_pinching:
            if self.two_hand_pinch_start is None:
                self.two_hand_pinch_start = now
            elapsed = now - self.two_hand_pinch_start
            frac = min(elapsed / PINCH_HOLD_SECONDS, 1.0)
            cv2.ellipse(frame, (w // 2, h // 2), (60, 60), -90, 0, int(360 * frac), (0, 255, 0), 6)
            if elapsed >= PINCH_HOLD_SECONDS:
                self.two_hand_pinch_start = None
                self.last_capture_time = now
                self.do_capture(clean_frame)
        else:
            self.two_hand_pinch_start = None

        self._last_frame_snapshot = clean_frame
        return frame

    def do_capture(self, frame):
        self.puzzle = Puzzle(frame.copy(), self.grid)
        self.tile_px = CANVAS_SIZE // self.grid
        self.held_pos = None
        self.mouse_held_pos = None
        self.was_pinching = False
        self.pinch_start_time = None
        self.last_hand_seen_time = None
        self.fist_start_time = None
        self.last_cursor_px = None
        self.solve_start_time = None
        self.solve_end_time = None
        self.feedback_text = ""
        self.state = "PUZZLE"

    # ---- PUZZLE / solve phase ----
    def render_puzzle(self):
        frame, results = self.read_hands()
        hand_present = bool(results and results.multi_hand_landmarks)
        pinching = False
        fisting = False
        self.cursor_norm = None
        hand_landmarks = None

        if frame is not None and hand_present:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            ratio = pinch_ratio(lm)
            self.last_ratio = ratio
            fisting_raw, curl_count = is_fist(lm)
            self.last_curl_count = curl_count
            # A full fist ALWAYS wins over pinch: a closed fist naturally
            # brings the thumb close to the curled fingers too, which can
            # trip the pinch-ratio check. If we let pinch win, a genuine
            # fist can silently fail to register. So: fist first, and only
            # treat it as a pinch if the hand isn't fully closed.
            if fisting_raw:
                fisting = True
                pinching = False
            else:
                fisting = False
                pinching = ratio < self.pinch_threshold
            if not fisting:
                self.cursor_norm = pinch_point_norm(lm)

        # --- full-size live video as the background ---
        if frame is not None:
            canvas = frame.copy()
        else:
            canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        h, w = canvas.shape[:2]

        # --- puzzle box centered on top of the video ---
        box_size = min(w, h) - 160
        ox, oy = (w - box_size) // 2, (h - box_size) // 2
        self.puzzle_offset = (ox, oy)

        tiles_layer, ts = self.puzzle.render(box_size, dragging_pos=self.held_pos)
        self.tile_px = ts
        grid = self.puzzle.grid
        for pos in range(grid * grid):
            r, c = divmod(pos, grid)
            y0, x0 = oy + r * ts, ox + c * ts
            canvas[y0:y0 + ts, x0:x0 + ts] = tiles_layer[r * ts:(r + 1) * ts, c * ts:(c + 1) * ts]
        cv2.rectangle(canvas, (ox, oy), (ox + box_size, oy + box_size), (0, 255, 180), 2)

        # highlight the mouse-fallback "picked up" tile too, if any
        if self.mouse_held_pos is not None:
            r, c = divmod(self.mouse_held_pos, grid)
            y0, x0 = oy + r * ts, ox + c * ts
            cv2.rectangle(canvas, (x0, y0), (x0 + ts, y0 + ts), (255, 0, 255), 3)

        # --- hand skeleton drawn on top ---
        if hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        cursor_px = None
        if self.cursor_norm:
            cursor_px = (int(self.cursor_norm[0] * w), int(self.cursor_norm[1] * h))
            self.last_cursor_px = cursor_px
            color = (0, 255, 0) if pinching else (0, 200, 255)
            cv2.circle(canvas, cursor_px, 14, color, 3)

        # current grid position under the fingertip (recomputed every frame,
        # this is what makes the drop location live/continuous)
        pos_under_cursor = None
        cursor_for_hit_test = cursor_px if cursor_px is not None else self.last_cursor_px
        if cursor_for_hit_test is not None:
            pos_under_cursor = self._pixel_to_pos(cursor_for_hit_test[0] - ox, cursor_for_hit_test[1] - oy)

        # --- FLOATING DRAGGED TILE: follows the fingertip every frame ---
        if self.held_pos is not None:
            drag_center = cursor_px if cursor_px is not None else self.last_cursor_px
            if drag_center is not None:
                tile_id = self.puzzle.board[self.held_pos]
                scale = 1.12
                big = cv2.resize(self.puzzle.tiles[tile_id], (int(ts * scale), int(ts * scale)))
                bh, bw = big.shape[:2]
                tx0, ty0 = drag_center[0] - bw // 2, drag_center[1] - bh // 2
                src_x0, src_y0 = max(0, -tx0), max(0, -ty0)
                dst_x0, dst_y0 = max(0, tx0), max(0, ty0)
                dst_x1 = min(canvas.shape[1], tx0 + bw)
                dst_y1 = min(canvas.shape[0], ty0 + bh)
                if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                    seg = big[src_y0:src_y0 + (dst_y1 - dst_y0), src_x0:src_x0 + (dst_x1 - dst_x0)]
                    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = seg
                    # green outline over a valid drop target, red if off the grid
                    outline = (0, 255, 0) if pos_under_cursor is not None else (0, 0, 255)
                    cv2.rectangle(canvas, (dst_x0, dst_y0), (dst_x1, dst_y1), outline, 3)
                # also ring whichever slot is currently under the fingertip
                if pos_under_cursor is not None:
                    r, c = divmod(pos_under_cursor, grid)
                    y0t, x0t = oy + r * ts, ox + c * ts
                    cv2.rectangle(canvas, (x0t, y0t), (x0t + ts, y0t + ts), (0, 255, 0), 3)

        # --- fist-to-reset ---
        now = time.time()
        if fisting:
            if self.fist_start_time is None:
                self.fist_start_time = now
            elapsed_fist = now - self.fist_start_time
            frac = min(elapsed_fist / FIST_HOLD_SECONDS, 1.0)
            if hand_landmarks is not None:
                wrist = hand_landmarks.landmark[0]
                fx, fy = int(wrist.x * w), int(wrist.y * h)
            else:
                fx, fy = (w // 2, h // 2)
            cv2.ellipse(canvas, (fx, fy), (40, 40), -90, 0, int(360 * frac), (0, 0, 255), 5)
            if elapsed_fist >= FIST_HOLD_SECONDS:
                self.fist_start_time = None
                self.held_pos = None
                self.pinch_start_time = None
                self.puzzle.shuffle()
                self.solve_start_time = None
                self.solve_end_time = None
                self._set_feedback("Reshuffled", (0, 200, 255), 1.0)
        else:
            self.fist_start_time = None

        # --- pinch pick up / drag / drop -----------------------------------
        # Only trust pinch-state *transitions* on frames where a hand was
        # actually detected -- a single frame of lost tracking must not be
        # read as "pinch released", or the drop silently misfires.
        if hand_present:
            if pinching and not self.was_pinching:
                # pinch just started -> pick up whatever tile is under the fingertip
                if pos_under_cursor is not None:
                    self.held_pos = pos_under_cursor
                    self.pinch_start_time = now
            elif (not pinching) and self.was_pinching:
                # pinch just released -> drop onto whatever is under the
                # fingertip RIGHT NOW (not where it started)
                if self.held_pos is not None:
                    if pos_under_cursor is not None:
                        self._attempt_move(self.held_pos, pos_under_cursor)
                    else:
                        self._set_feedback("Dropped outside grid - cancelled", (0, 0, 255), 1.2)
                    self.held_pos = None
                    self.pinch_start_time = None
            self.was_pinching = pinching
        # else: hand not detected this frame -> leave was_pinching/held_pos untouched

        # safety net: force-drop ONLY if your hand has genuinely vanished from
        # tracking for a while mid-drag (left the frame, camera hiccup, etc).
        # A real pinch you're still holding -- however long -- must never be
        # auto-dropped; the tile only drops when you actually release.
        if hand_present:
            self.last_hand_seen_time = now
        if self.held_pos is not None and self.last_hand_seen_time is not None \
                and now - self.last_hand_seen_time > MAX_HAND_LOST_SECONDS:
            if pos_under_cursor is not None:
                self._attempt_move(self.held_pos, pos_under_cursor)
            else:
                self._set_feedback("Hand lost - drag cancelled", (0, 0, 255), 1.2)
            self.held_pos = None
            self.pinch_start_time = None
            self.was_pinching = False

        # timer
        if self.solve_start_time is not None:
            elapsed = (self.solve_end_time or now) - self.solve_start_time
        else:
            elapsed = 0.0

        header = np.zeros((70, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "PHASE 2: SOLVE", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, "Pinch a tile, drag it over ANY other tile, release to swap",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(header, format_time(elapsed), (canvas.shape[1] - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, "click,click=fallback  r=reshuffle  c=new photo  d=debug", (10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        if self.debug_hud:
            r_txt = f"{self.last_ratio:.2f}" if self.last_ratio is not None else "--"
            cv2.putText(canvas, f"hand:{hand_present} pinch:{pinching} ratio:{r_txt} thr:{self.pinch_threshold:.2f} held:{self.held_pos} target:{pos_under_cursor} fist_curl:{getattr(self,'last_curl_count','-')}/4",
                        (10, canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Use Index+Thumb Pinch", (canvas.shape[1] - 230, canvas.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # transient feedback banner ("Swapped!" / "Dropped outside grid" / etc.)
        if self.feedback_text and now < self.feedback_until:
            text = self.feedback_text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            bx = (canvas.shape[1] - tw) // 2
            cv2.rectangle(canvas, (bx - 12, 10), (bx + tw + 12, 10 + th + 20), (0, 0, 0), -1)
            cv2.putText(canvas, text, (bx, 10 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        self.feedback_color, 2, cv2.LINE_AA)

        return np.vstack([header, canvas])

    # ---- SOLVED / NAME_ENTRY / LEADERBOARD screens ----
    def render_solved(self):
        canvas, _ = self.puzzle.render(CANVAS_SIZE)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (0, 180, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
        elapsed = (self.solve_end_time or time.time()) - (self.solve_start_time or time.time())
        cv2.putText(canvas, "PUZZLE SOLVED!", (canvas.shape[1]//2 - 190, canvas.shape[0]//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"Time: {format_time(elapsed)}", (canvas.shape[1]//2 - 90, canvas.shape[0]//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        header = np.zeros((50, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "Press ENTER to save your score  |  r: reshuffle  c: new photo",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return np.vstack([header, canvas])

    def render_name_entry(self):
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        cv2.putText(canvas, "Enter your name for the leaderboard:", (30, CANVAS_SIZE // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        box_text = self.name_buffer + "_"
        cv2.putText(canvas, box_text, (30, CANVAS_SIZE // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Type letters, BACKSPACE to edit, ENTER to save", (30, CANVAS_SIZE // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        header = np.zeros((50, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "NAME ENTRY", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([header, canvas])

    def render_leaderboard(self):
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        board = load_leaderboard()
        cv2.putText(canvas, "LEADERBOARD", (CANVAS_SIZE//2 - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y = 100
        if not board:
            cv2.putText(canvas, "No scores yet.", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        for i, entry in enumerate(board):
            line = f"#{i+1:<2} {entry['name'][:12]:<12} {format_time(entry['time']):>6}  {entry['grid']}"
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(canvas, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y += 36
        header = np.zeros((50, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, "ENTER / c: back to game   ESC: quit", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([header, canvas])

    # ---- main loop ----
    def run(self):
        if not self.cap.isOpened():
            print("Could not open webcam. Check CAM_INDEX / camera permissions.")
            return

        while True:
            if self.state == "LIVE":
                frame = self.render_live()
                if frame is None:
                    break
                cv2.imshow(self.window, frame)
            elif self.state == "PUZZLE":
                cv2.imshow(self.window, self.render_puzzle())
            elif self.state == "SOLVED":
                cv2.imshow(self.window, self.render_solved())
            elif self.state == "NAME_ENTRY":
                cv2.imshow(self.window, self.render_name_entry())
            elif self.state == "LEADERBOARD":
                cv2.imshow(self.window, self.render_leaderboard())

            key = cv2.waitKey(1) & 0xFF
            if key == 255:  # no key pressed
                continue

            if key in (27,) or (key == ord('q') and self.state != "NAME_ENTRY"):
                break

            if self.state == "LIVE":
                if key in (ord('1'), ord('2'), ord('3')):
                    self.grid = int(chr(key)) + 2
                elif key == ord(' ') and self._last_frame_snapshot is not None:
                    self.do_capture(self._last_frame_snapshot)
                elif key in (ord('+'), ord('=')):
                    self.pinch_threshold = min(1.5, self.pinch_threshold + 0.05)
                elif key == ord('-'):
                    self.pinch_threshold = max(0.05, self.pinch_threshold - 0.05)

            elif self.state == "PUZZLE":
                if key == ord('r'):
                    self.puzzle.shuffle()
                    self.solve_start_time = None
                    self.solve_end_time = None
                    self.held_pos = None
                    self.mouse_held_pos = None
                elif key == ord('c'):
                    self.state = "LIVE"
                    self.puzzle = None
                elif key == ord('d'):
                    self.debug_hud = not self.debug_hud
                elif key in (ord('+'), ord('=')):
                    self.pinch_threshold = min(1.5, self.pinch_threshold + 0.05)
                elif key == ord('-'):
                    self.pinch_threshold = max(0.05, self.pinch_threshold - 0.05)

            elif self.state == "SOLVED":
                if key == 13:  # ENTER
                    self.name_buffer = ""
                    self.state = "NAME_ENTRY"
                elif key == ord('r'):
                    self.puzzle.shuffle()
                    self.solve_start_time = None
                    self.solve_end_time = None
                    self.state = "PUZZLE"
                elif key == ord('c'):
                    self.state = "LIVE"
                    self.puzzle = None

            elif self.state == "NAME_ENTRY":
                if key == 13:  # ENTER
                    elapsed = (self.solve_end_time or time.time()) - (self.solve_start_time or time.time())
                    save_score(self.name_buffer.strip() or "PLAYER", elapsed, self.grid)
                    self.state = "LEADERBOARD"
                elif key == 8:  # BACKSPACE
                    self.name_buffer = self.name_buffer[:-1]
                elif 32 <= key <= 126 and len(self.name_buffer) < 14:
                    self.name_buffer += chr(key)

            elif self.state == "LEADERBOARD":
                if key in (13, ord('c')):
                    self.state = "LIVE"
                    self.puzzle = None

        self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    App().run()
