"""
drive_with_detector.py  –  Autonomous Driver v2 for SpeedTrials2D
==================================================================
Complete rewrite based on diagnostic data from live gameplay.

Core strategy (simple but effective):
  1. ALWAYS full throttle (acceleration = 1.0) every frame.
  2. Detect all tokens visible on the road.
  3. For each token, score it:  green=+3, red=-5, yellow=-3.
  4. Project all token scores into a LEFT/CENTRE/RIGHT "cost map".
     Tokens near frame centre threaten the car's current path.
  5. Pick the lane with the BEST score (highest positive, or least negative).
  6. If the best lane differs from current lane → tap steer toward it.
  7. Steering uses fast taps with a short cooldown.

Key improvements over v1:
  - Uses raw pixel-space (no flawed lane-offset math)
  - Reacts to tokens at y >= 230  (v1 ignored everything above y=300)
  - Filters out persistent road markers (static red blobs at fixed positions)
  - Much faster tap cycle (0.18 s tap + 0.40 s cooldown)

All boilerplate (RTTask, sockets, simulator launch, display loop, etc.)
lives in ``rtse_framework.py``.
"""

import struct
import threading
import time
import cv2

import rtse_framework as fw
from rtse_framework import (
    TaskPriority, RTTask,
    shared_data, data_lock, display_queue,
    setup_cameras, setup_control_server,
    launch_simulator, run_main_loop, shutdown,
)
from detector import detect_and_annotate, detect_objects

# ─────────────────────────────────────────────────────────────────────────────
# Autonomous Control Parameters
# ─────────────────────────────────────────────────────────────────────────────

# Frame geometry (640×480)
FRAME_W  = 640
FRAME_H  = 480
FRAME_CX = FRAME_W // 2   # 320 — car is always here

# Road boundaries at screen bottom
ROAD_LEFT  = 80
ROAD_RIGHT = 560

# Token reaction zone — react to tokens in this vertical band
REACT_Y_MIN = 225   # top of reaction zone  (tokens first appear ~y=240)
REACT_Y_MAX = 430   # bottom of reaction zone (above the car)

# Lane model — divide the visible road into 3 zones: LEFT / CENTRE / RIGHT
ZONE_HALF_W = 70    # pixels — a token within ±70px of centre is "CENTRE"

# Token scoring
SCORE = {
    'green':  +3.0,   # collect
    'red':    -5.0,   # highest avoidance priority
    'yellow': -3.0,   # avoid
}

# Rear-camera trailing-car avoidance
TRAILING_MIN_AREA_FRAC   = 0.004  # ignore tiny red blobs/noise
TRAILING_CLOSE_Y_FRAC    = 0.58   # close objects appear in the lower rear view
TRAILING_CLOSE_AREA_FRAC = 0.012  # large enough to be collision-relevant
TRAILING_MEMORY_SEC      = 0.90   # keep evading through brief detection dropouts
TRAILING_LANE_PENALTY    = -12.0  # stronger than a close red token penalty


def _urgency_weight(y: float) -> float:
    """Weight factor for a token at screen-y. Range ~ [0.3, 2.0]."""
    t = (y - REACT_Y_MIN) / max(1, REACT_Y_MAX - REACT_Y_MIN)
    return 0.3 + 1.7 * t


# Static blob filter — ignore red detections that don't move between frames
STATIC_FRAMES_THRESH = 8    # if seen at same spot this many frames, suppress
STATIC_DIST_THRESH   = 10   # pixels — within this distance counts as "same spot"

# Steering tap parameters
TAP_STRENGTH = 1.0    # full steer
TAP_DURATION = 0.18   # seconds — shorter = snappier
TAP_COOLDOWN = 0.40   # seconds — shorter = more agile

# ─────────────────────────────────────────────────────────────────────────────
# Static Blob Tracker
# ─────────────────────────────────────────────────────────────────────────────
class StaticBlobTracker:
    """Tracks blob positions across frames to identify static road markers."""

    def __init__(self):
        self._blobs: dict = {}  # (bucket_x, bucket_y) → consecutive_count

    def update_and_filter(self, tokens: list) -> list:
        """Return tokens with static blobs removed."""
        bucket_size = STATIC_DIST_THRESH
        seen        = set()
        filtered    = []

        for tok in tokens:
            x, y, w, h = tok['bbox']
            cx = x + w // 2
            cy = y + h // 2
            bk = (cx // bucket_size, cy // bucket_size)
            seen.add(bk)

            count            = self._blobs.get(bk, 0) + 1
            self._blobs[bk]  = count

            if count < STATIC_FRAMES_THRESH:
                filtered.append(tok)
            # else: suppressed as static road marker

        # Decay blobs not seen this frame
        for bk in list(self._blobs):
            if bk not in seen:
                self._blobs[bk] = max(0, self._blobs[bk] - 2)
                if self._blobs[bk] <= 0:
                    del self._blobs[bk]

        return filtered


_static_tracker = StaticBlobTracker()

# Rear hazard state is owned by the processing task, so it does not need a lock.
_rear_hazard = {
    'active_until': 0.0,
    'lane': None,
    'bbox': None,
    'confidence': 0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Steering State Machine
# ─────────────────────────────────────────────────────────────────────────────
class _SteerState:
    IDLE     = 'idle'
    TAPPING  = 'tapping'
    COOLDOWN = 'cooldown'


_steer = {
    'state': _SteerState.IDLE,
    'value': 0.0,
    'tap_t': 0.0,
    'cd_t':  0.0,
    'lane':  1,    # 0=left, 1=centre, 2=right
}
_steer_lock = threading.Lock()


def _tick_steering(desired_lane: int) -> float:
    """
    Advance the steering state machine.

    Args:
        desired_lane: 0 (left), 1 (centre), 2 (right).

    Returns:
        Steering value to output this frame.
    """
    now = time.time()

    with _steer_lock:
        st = _steer

        if st['state'] == _SteerState.TAPPING:
            if now - st['tap_t'] >= TAP_DURATION:
                st['state'] = _SteerState.COOLDOWN
                st['value'] = 0.0
                st['cd_t']  = now
            return st['value']

        if st['state'] == _SteerState.COOLDOWN:
            if now - st['cd_t'] >= TAP_COOLDOWN:
                st['state'] = _SteerState.IDLE
            return 0.0

        # IDLE — decide whether to tap
        cur_lane = st['lane']
        if desired_lane == cur_lane:
            st['value'] = 0.0
            return 0.0

        direction = +1 if desired_lane > cur_lane else -1
        new_lane  = max(0, min(2, cur_lane + direction))
        st['lane']  = new_lane
        st['state'] = _SteerState.TAPPING
        st['value'] = TAP_STRENGTH * direction
        st['tap_t'] = now
        return st['value']


# ─────────────────────────────────────────────────────────────────────────────
# Core Decision Logic
# ─────────────────────────────────────────────────────────────────────────────
def _lane_from_x(x_pos: float, frame_w: int) -> int:
    """Map a camera x-coordinate into the shared LEFT/CENTRE/RIGHT lane model."""
    road_left = ROAD_LEFT * frame_w / FRAME_W
    road_right = ROAD_RIGHT * frame_w / FRAME_W
    lane_w = max(1.0, (road_right - road_left) / 3.0)
    lane = int((x_pos - road_left) / lane_w)
    return max(0, min(2, lane))


def detect_trailing_car(back_frame):
    """
    Detect a collision-relevant rear car and return the lane to avoid.

    The simulator's trailing-car event is satisfied by switching lanes before
    impact. We use the existing red car detector on the back camera, then keep
    the hazard alive briefly so steering does not flicker when a frame is missed.
    """
    now = time.time()

    if back_frame is not None:
        results = detect_objects(back_frame)
        car_bbox = results.get('car')

        if car_bbox is not None:
            x, y, w, h = car_bbox
            frame_h, frame_w = back_frame.shape[:2]
            area_frac = (w * h) / max(1.0, frame_w * frame_h)
            cy_frac = (y + h / 2) / max(1.0, frame_h)
            touches_bottom = (y + h) >= frame_h - 4

            is_close = (
                not touches_bottom
                and area_frac >= TRAILING_MIN_AREA_FRAC
                and (cy_frac >= TRAILING_CLOSE_Y_FRAC or area_frac >= TRAILING_CLOSE_AREA_FRAC)
            )

            if is_close:
                lane = _lane_from_x(x + w / 2, frame_w)
                area_score = min(1.0, area_frac / TRAILING_CLOSE_AREA_FRAC)
                y_score = max(0.0, (cy_frac - TRAILING_CLOSE_Y_FRAC) / (1.0 - TRAILING_CLOSE_Y_FRAC))
                confidence = min(1.0, 0.7 * area_score + 0.3 * y_score)

                _rear_hazard.update({
                    'active_until': now + TRAILING_MEMORY_SEC,
                    'lane': lane,
                    'bbox': car_bbox,
                    'confidence': confidence,
                })

    active = _rear_hazard['lane'] is not None and _rear_hazard['active_until'] > now
    return {
        'active': active,
        'lane': _rear_hazard['lane'] if active else None,
        'bbox': _rear_hazard['bbox'] if active else None,
        'confidence': _rear_hazard['confidence'] if active else 0.0,
    }


def apply_rear_hazard(scores: list, desired_lane: int, rear_hazard: dict) -> tuple[int, list]:
    """Penalize the threatened lane and choose a non-threatened lane while active."""
    if not rear_hazard['active'] or rear_hazard['lane'] is None:
        return desired_lane, scores

    scores = scores.copy()
    blocked_lane = rear_hazard['lane']
    scores[blocked_lane] += TRAILING_LANE_PENALTY

    candidates = [i for i in range(3) if i != blocked_lane]
    desired_lane = max(candidates, key=lambda i: scores[i])
    return desired_lane, scores


def decide_lane(tokens: list, frame_w: int) -> tuple[int, list]:
    """
    Score each of three zones (left / centre / right) based on visible tokens.

    Returns:
        (desired_lane: 0/1/2, zone_scores: list[float])
    """
    scores = [0.0, 0.15, 0.0]   # small centre bias: prefer staying centred
    cx     = frame_w // 2

    for tok in tokens:
        x, y, w, h = tok['bbox']
        tok_cx = x + w / 2
        tok_cy = y + h / 2

        if tok_cy < REACT_Y_MIN or tok_cy > REACT_Y_MAX:
            continue

        score          = SCORE.get(tok['color'], 0)
        weighted_score = score * _urgency_weight(tok_cy)

        # Perspective-adjusted zone width
        vp_y      = 270
        scale     = max(0.1, (tok_cy - vp_y) / (FRAME_H - vp_y))
        zone_half = ZONE_HALF_W * scale
        offset    = tok_cx - cx

        if abs(offset) < zone_half:
            scores[1] += weighted_score   # CENTRE
        elif offset < -zone_half:
            scores[0] += weighted_score   # LEFT
        else:
            scores[2] += weighted_score   # RIGHT

    best_lane = max(range(3), key=lambda i: scores[i])
    return best_lane, scores


# ─────────────────────────────────────────────────────────────────────────────
# HUD Overlay
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, steer, scores, cur_lane, desired_lane, state, tokens, rear_hazard):
    h, w = frame.shape[:2]
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # State panel (top-right, avoids game UI)
    sc = {
        _SteerState.IDLE:     (0, 200,   0),
        _SteerState.TAPPING:  (0, 120, 255),
        _SteerState.COOLDOWN: (0, 200, 255),
    }
    px, py = w - 170, 100
    panel_h = 92
    cv2.rectangle(frame, (px, py), (w - 4, py + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (px, py), (w - 4, py + panel_h), (80, 80, 80), 1)
    rear_label = 'CLEAR'
    rear_color = (140, 220, 140)
    if rear_hazard['active']:
        rear_label = 'L C R'.split()[rear_hazard['lane']]
        rear_color = (0, 120, 255)
    lines = [
        (f"State: {state.upper()}",                        sc.get(state, (180, 180, 180))),
        (f"Steer: {steer:+.1f}",                           (200, 200, 200)),
        (f"Lane:  {'L C R'.split()[cur_lane]}",            (200, 200, 200)),
        (f"Rear:  {rear_label}",                           rear_color),
        (f"Scores: {scores[0]:+.1f} {scores[1]:+.1f} {scores[2]:+.1f}", (180, 180, 180)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (px + 4, py + 16 + i * 17), FONT, 0.38, col, 1, cv2.LINE_AA)

    # Lane bar (bottom centre)
    bar_w, bar_h = 150, 14
    bx     = (w - bar_w) // 2
    by     = h - 20
    cell_w = bar_w // 3
    cv2.rectangle(frame, (bx - 2, by - 2), (bx + bar_w + 2, by + bar_h + 2), (0, 0, 0), -1)
    for i, label in enumerate(['L', 'C', 'R']):
        cx1   = bx + i * cell_w
        color = (0, 180, 255) if i == cur_lane else (60, 60, 60)
        if i == desired_lane and i != cur_lane:
            color = (0, 255, 100)
        cv2.rectangle(frame, (cx1, by), (cx1 + cell_w - 1, by + bar_h), color, -1)
        cv2.putText(frame, label, (cx1 + cell_w // 2 - 4, by + 11), FONT, 0.35, (255, 255, 255), 1)
    cv2.rectangle(frame, (bx - 2, by - 2), (bx + bar_w + 2, by + bar_h + 2), (120, 120, 120), 1)

    # Zone dividers on road
    cv2.line(frame, (FRAME_CX - ZONE_HALF_W, REACT_Y_MIN), (FRAME_CX - ZONE_HALF_W, REACT_Y_MAX), (80, 80, 80), 1)
    cv2.line(frame, (FRAME_CX + ZONE_HALF_W, REACT_Y_MIN), (FRAME_CX + ZONE_HALF_W, REACT_Y_MAX), (80, 80, 80), 1)

    # Reaction zone boundary
    cv2.line(frame, (ROAD_LEFT, REACT_Y_MIN), (ROAD_RIGHT, REACT_Y_MIN), (50, 50, 50), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Task Implementations
# ─────────────────────────────────────────────────────────────────────────────

def read_front_camera_task():
    fw.read_single_camera(fw.front_camera_sock, 'latest_front_frame',
                          push_display=False)


def read_back_camera_task():
    fw.read_single_camera(fw.back_camera_sock, 'latest_back_frame',
                          window_name="Back Camera", push_display=True)


def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']
    if front_frame is None:
        return

    # 1. Detect tokens
    annotated, results = detect_and_annotate(front_frame)

    # 2. Filter static blobs (persistent road markers)
    tokens = _static_tracker.update_and_filter(results['tokens'])

    # 3. Decide best lane
    desired_lane, scores = decide_lane(tokens, front_frame.shape[1])
    rear_hazard = detect_trailing_car(back_frame)
    desired_lane, scores = apply_rear_hazard(scores, desired_lane, rear_hazard)

    # 4. Advance steering state machine
    steer_out = _tick_steering(desired_lane)

    # 5. Write controls — ALWAYS accelerating
    with data_lock:
        shared_data['steering_input']     = steer_out
        shared_data['acceleration_input'] = 1.0

    # 6. Draw HUD and push to display
    with _steer_lock:
        cur_lane = _steer['lane']
        state    = _steer['state']

    annotated = draw_hud(annotated, steer_out, scores, cur_lane, desired_lane, state, tokens, rear_hazard)
    try:
        display_queue.put_nowait(("Autonomous View", cv2.resize(annotated, (640, 480))))
    except Exception:
        pass


def send_controls_task():
    """Reads steering/acceleration from shared_data and sends to control socket."""
    if fw.control_conn is None:
        return

    with data_lock:
        steer = shared_data.get('steering_input',     0.0)
        accel = shared_data.get('acceleration_input', 1.0)

    try:
        fw.control_conn.sendall(struct.pack('ff', steer, accel))
    except Exception as e:
        print(f"Control send error: {e}")
        fw.control_conn = None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Initializing RTSE Autonomous Driver v2...")

    sim = launch_simulator()

    try:
        threading.Thread(target=setup_control_server, daemon=True).start()
        threading.Thread(target=setup_cameras,        daemon=True).start()

        print("\n--- Starting Real-Time Tasks ---\n")

        t_front = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH,   execute_func=read_front_camera_task)
        t_back  = RTTask("ReadBackCamera",  period=0.005, priority=TaskPriority.HIGH,   execute_func=read_back_camera_task)
        t_proc  = RTTask("Processing",      period=0.020, priority=TaskPriority.MEDIUM, execute_func=processing_task)   # ~50 fps
        t_ctrl  = RTTask("SendControls",    period=0.005, priority=TaskPriority.HIGH,   execute_func=send_controls_task)

        for t in [t_front, t_back, t_proc, t_ctrl]:
            t.start()

        run_main_loop()  # blocks until Ctrl+C

        for t in [t_front, t_back, t_proc, t_ctrl]:
            t.join()

    finally:
        shutdown(sim)
