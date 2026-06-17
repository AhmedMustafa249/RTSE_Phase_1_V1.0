import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes
import os
import csv
from datetime import datetime

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

NUM_LANES = 5
START_LANE = 2  # assume the car starts centered

# Reaction zone: vertical band (fraction of frame height) where tokens are
# close enough to react to
REACT_Y_MIN_FRAC = 0.45
REACT_Y_MAX_FRAC = 0.90

# A token counts as "in our lane" if its center x falls within this fraction
# of the frame width around the center (one lane's worth of width)
LANE_HALF_WIDTH_FRAC = 0.10

# Hazard HSV ranges (red wraps around hue 0, so two ranges)
HAZARD_HSV_RANGES = {
    'red': [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([165, 120, 70]), np.array([180, 255, 255])),
    ],
    'yellow': [
        (np.array([18, 100, 100]), np.array([35, 255, 255])),
    ],
}
TOKEN_MIN_AREA = 80

# Static-blob suppression: a hazard blob that stays in the same bucket for
# this many consecutive frames is treated as a fixed road marker / UI
# element, not a real token, and ignored.
STATIC_DIST_THRESH = 10     # px — bucket size for grouping blob positions
STATIC_FRAMES_THRESH = 8    # consecutive sightings before a blob is suppressed

# Steering tap timing, in wall-clock seconds (not cycle counts) so behavior
# stays correct even if the processing task's actual period drifts under load
TAP_DURATION = 0.10      # seconds of active steer
COOLDOWN_DURATION = 0.20  # seconds of no new taps after one completes

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_front_frame_timestamp': None,
    'latest_front_frame_seq': 0,
    'latest_back_frame': None,
    'latest_back_frame_timestamp': None,
    'latest_back_frame_seq': 0,
    'steering_input': 0.0,
    'acceleration_input': 1.0,
    'lane_index': START_LANE,
}
data_lock = threading.Lock()
is_running = True

# Diagnostics are opt-in so normal competition behavior is unchanged.
# Set RTSE_DIAG = True before running the script to write diag_auto output.
RTSE_DIAG = True
DIAG_ENABLED = RTSE_DIAG
DIAG_DIR = os.path.join('diag_auto', datetime.now().strftime('%Y%m%d-%H%M%S'))
DIAG_HEARTBEAT_INTERVAL = 1.0
DIAG_FRAME_SAVE_LIMIT = 300

_diag_lock = threading.Lock()
_diag_file = None
_diag_writer = None
_diag_frame_count = 0
_diag_last_heartbeat = 0.0


def _diag_init():
    global _diag_file, _diag_writer
    if not DIAG_ENABLED or _diag_writer is not None:
        return

    os.makedirs(DIAG_DIR, exist_ok=True)
    _diag_file = open(os.path.join(DIAG_DIR, 'events.csv'), 'w', newline='')
    _diag_writer = csv.DictWriter(_diag_file, fieldnames=[
        'time',
        'event',
        'frame_seq',
        'frame_age_ms',
        'state_before',
        'state_after',
        'lane_before',
        'lane_after',
        'direction',
        'target_lane',
        'steering_output',
        'raw_boxes',
        'real_boxes',
        'suppressed_boxes',
        'in_lane',
        'zone_pixels',
        'mask_pixels',
        'processing_ms',
        'saved_frame',
        'note',
    ])
    _diag_writer.writeheader()
    _diag_file.flush()
    print(f"[DIAG] Writing auto-drive diagnostics to {DIAG_DIR}")


def _diag_box_text(boxes):
    return ';'.join(f'{x}:{y}:{w}:{h}' for (x, y, w, h) in boxes)


def _draw_diagnostic_frame(frame, analysis, event):
    out = frame.copy()
    y_min = analysis['y_min']
    y_max = analysis['y_max']
    cx = analysis['cx']
    lane_half_width = analysis['lane_half_width']

    cv2.line(out, (0, y_min), (out.shape[1] - 1, y_min), (0, 255, 255), 1)
    cv2.line(out, (0, y_max), (out.shape[1] - 1, y_max), (0, 255, 255), 1)
    cv2.line(out, (cx - lane_half_width, y_min), (cx - lane_half_width, y_max), (255, 255, 0), 1)
    cv2.line(out, (cx + lane_half_width, y_min), (cx + lane_half_width, y_max), (255, 255, 0), 1)

    for (x, y, w, h) in analysis['raw_boxes']:
        cv2.rectangle(out, (x, y), (x + w, y + h), (140, 140, 140), 1)
    for (x, y, w, h) in analysis['suppressed_boxes']:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
    for (x, y, w, h) in analysis['real_boxes']:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
    if analysis['in_lane_box'] is not None:
        x, y, w, h = analysis['in_lane_box']
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)

    lines = [
        f"event={event}",
        f"direction={analysis['direction']} in_lane={analysis['in_lane_box'] is not None}",
        f"raw={len(analysis['raw_boxes'])} real={len(analysis['real_boxes'])} static={len(analysis['suppressed_boxes'])}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(out, text, (8, 22 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _diag_record(event, frame=None, analysis=None, frame_seq=None,
                 frame_timestamp=None, state_before='', state_after='',
                 lane_before='', lane_after='', target_lane='', steering_output='',
                 processing_ms='', note='', save_frame=False):
    global _diag_frame_count, _diag_last_heartbeat
    if not DIAG_ENABLED:
        return

    now = time.time()
    if event == 'heartbeat' and now - _diag_last_heartbeat < DIAG_HEARTBEAT_INTERVAL:
        return
    if event == 'heartbeat':
        _diag_last_heartbeat = now

    with _diag_lock:
        _diag_init()
        saved_frame = ''
        if save_frame and frame is not None and _diag_frame_count < DIAG_FRAME_SAVE_LIMIT:
            filename = f'{_diag_frame_count:05d}_{event}.png'
            path = os.path.join(DIAG_DIR, filename)
            if analysis is not None:
                debug_frame = _draw_diagnostic_frame(frame, analysis, event)
            else:
                debug_frame = frame
            cv2.imwrite(path, debug_frame)
            saved_frame = filename
            _diag_frame_count += 1

        frame_age_ms = ''
        if frame_timestamp is not None:
            frame_age_ms = f'{(now - frame_timestamp) * 1000.0:.1f}'

        row = {
            'time': f'{now:.6f}',
            'event': event,
            'frame_seq': frame_seq if frame_seq is not None else '',
            'frame_age_ms': frame_age_ms,
            'state_before': state_before,
            'state_after': state_after,
            'lane_before': lane_before,
            'lane_after': lane_after,
            'direction': analysis['direction'] if analysis is not None else '',
            'target_lane': target_lane,
            'steering_output': steering_output,
            'raw_boxes': _diag_box_text(analysis['raw_boxes']) if analysis is not None else '',
            'real_boxes': _diag_box_text(analysis['real_boxes']) if analysis is not None else '',
            'suppressed_boxes': _diag_box_text(analysis['suppressed_boxes']) if analysis is not None else '',
            'in_lane': analysis['in_lane_box'] is not None if analysis is not None else '',
            'zone_pixels': analysis['zone_mask_pixels'] if analysis is not None else '',
            'mask_pixels': analysis['hazard_mask_pixels'] if analysis is not None else '',
            'processing_ms': processing_ms,
            'saved_frame': saved_frame,
            'note': note,
        }
        _diag_writer.writerow(row)
        _diag_file.flush()


def _diag_close():
    global _diag_file, _diag_writer
    if _diag_file is not None:
        _diag_file.flush()
        _diag_file.close()
    _diag_file = None
    _diag_writer = None

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False

    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass

        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass

        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")

    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Camera Reading
# ---------------------------------------------------------
def read_single_camera(sock, window_name, data_key):
    if sock is None:
        return
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return

        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet

        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes

        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet

            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes

        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                timestamp_key = f'{data_key}_timestamp'
                seq_key = f'{data_key}_seq'
                with data_lock:
                    shared_data[data_key] = frame
                    shared_data[timestamp_key] = time.time()
                    shared_data[seq_key] = shared_data.get(seq_key, 0) + 1

                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)

    except Exception:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

# ---------------------------------------------------------
# Static Blob Tracker
# ---------------------------------------------------------
class StaticBlobTracker:
    """Tracks blob positions across frames to suppress fixed false-positives
    (road markers, UI elements) that would otherwise look like a persistent
    hazard sitting in the same spot every cycle."""

    def __init__(self):
        self._blobs = {}  # (bucket_x, bucket_y) -> consecutive_count

    def filter_real(self, boxes, return_debug=False):
        """Given a list of (x, y, w, h), return the ones that are NOT static."""
        bucket_size = STATIC_DIST_THRESH
        seen = set()
        real = []
        suppressed = []
        counts = {}

        for (x, y, w, h) in boxes:
            cx, cy = x + w // 2, y + h // 2
            bk = (cx // bucket_size, cy // bucket_size)
            seen.add(bk)

            count = self._blobs.get(bk, 0) + 1
            self._blobs[bk] = count
            counts[(x, y, w, h)] = count

            if count < STATIC_FRAMES_THRESH:
                real.append((x, y, w, h))
            else:
                suppressed.append((x, y, w, h))
            # else: suppressed as a static road marker

        # Decay buckets not seen this frame so a marker that scrolls away
        # doesn't permanently poison that bucket
        for bk in list(self._blobs):
            if bk not in seen:
                self._blobs[bk] = max(0, self._blobs[bk] - 2)
                if self._blobs[bk] <= 0:
                    del self._blobs[bk]

        if return_debug:
            return real, suppressed, counts
        return real


_static_tracker = StaticBlobTracker()

# ---------------------------------------------------------
# Hazard Detection
# ---------------------------------------------------------
def analyze_hazard(frame, tracker=None):
    """
    Look for red/yellow tokens in the reaction zone that are in our lane.
    Returns detector metadata including direction:
    -1 (steer left), +1 (steer right), or 0 (no hazard ahead).
    """
    if tracker is None:
        tracker = _static_tracker

    h, w = frame.shape[:2]
    y_min = int(h * REACT_Y_MIN_FRAC)
    y_max = int(h * REACT_Y_MAX_FRAC)
    cx = w // 2
    lane_half_width = int(w * LANE_HALF_WIDTH_FRAC)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hazard_mask = np.zeros((h, w), dtype=np.uint8)
    for ranges in HAZARD_HSV_RANGES.values():
        for lo, hi in ranges:
            hazard_mask = cv2.bitwise_or(hazard_mask, cv2.inRange(hsv, lo, hi))

    # Restrict to the reaction zone
    zone_mask = np.zeros_like(hazard_mask)
    zone_mask[y_min:y_max, :] = hazard_mask[y_min:y_max, :]

    contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        if cv2.contourArea(c) < TOKEN_MIN_AREA:
            continue
        boxes.append(cv2.boundingRect(c))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    real_boxes, suppressed_boxes, static_counts = tracker.filter_real(boxes, return_debug=True)

    in_lane = False
    token_cx = cx
    in_lane_box = None
    for (x, y, bw, bh) in real_boxes:
        bcx = x + bw // 2
        if abs(bcx - cx) <= lane_half_width:
            in_lane = True
            token_cx = bcx
            in_lane_box = (x, y, bw, bh)
            break

    if not in_lane:
        direction = 0
    else:
        # Token is in our lane: dodge away from its exact position.
        # If it's left-of-center (or dead center), go right; otherwise go left.
        direction = 1 if token_cx <= cx else -1

    return {
        'direction': direction,
        'shape': (h, w),
        'y_min': y_min,
        'y_max': y_max,
        'cx': cx,
        'lane_half_width': lane_half_width,
        'hazard_mask_pixels': int(cv2.countNonZero(hazard_mask)),
        'zone_mask_pixels': int(cv2.countNonZero(zone_mask)),
        'raw_boxes': boxes,
        'real_boxes': real_boxes,
        'suppressed_boxes': suppressed_boxes,
        'static_counts': static_counts,
        'in_lane_box': in_lane_box,
        'token_cx': token_cx if in_lane else None,
    }


def find_hazard_direction(frame):
    """
    Look for red/yellow tokens in the reaction zone that are in our lane.
    Returns -1 (steer left), +1 (steer right), or 0 (no hazard ahead).
    """
    return analyze_hazard(frame)['direction']

# ---------------------------------------------------------
# Steering State Machine
# ---------------------------------------------------------
steer_state = 'IDLE'  # IDLE -> TAPPING -> COOLDOWN -> IDLE
steer_direction = 0
steer_deadline = 0.0  # time.time() value at which the current state ends

def processing_task():
    global steer_state, steer_direction, steer_deadline

    process_start = time.time()
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        lane_index = shared_data['lane_index']
        frame_timestamp = shared_data['latest_front_frame_timestamp']
        frame_seq = shared_data['latest_front_frame_seq']

    now = time.time()
    new_steering = 0.0
    state_before = steer_state
    lane_before = lane_index
    lane_after = lane_index
    state_after = steer_state
    analysis = None
    target_lane = ''
    diag_event = 'heartbeat'
    diag_note = ''
    save_diag_frame = False

    if steer_state == 'IDLE':
        if front_frame is not None:
            analysis = analyze_hazard(front_frame)
            direction = analysis['direction']
            if direction != 0:
                target_lane = lane_index + direction
                if 0 <= target_lane < NUM_LANES:
                    steer_direction = direction
                    steer_state = 'TAPPING'
                    steer_deadline = now + TAP_DURATION
                    diag_event = 'tap_start'
                    save_diag_frame = True
                else:
                    diag_event = 'tap_blocked'
                    diag_note = 'target lane out of bounds'
                    save_diag_frame = True
            elif analysis['raw_boxes'] or analysis['suppressed_boxes']:
                diag_event = 'hazard_seen_no_tap'
                save_diag_frame = True
        new_steering = 0.0

    elif steer_state == 'TAPPING':
        new_steering = float(steer_direction)
        if now >= steer_deadline:
            with data_lock:
                shared_data['lane_index'] += steer_direction
                lane_after = shared_data['lane_index']
            steer_state = 'COOLDOWN'
            steer_deadline = now + COOLDOWN_DURATION
            diag_event = 'tap_end'

    elif steer_state == 'COOLDOWN':
        new_steering = 0.0
        if now >= steer_deadline:
            steer_state = 'IDLE'
            diag_event = 'cooldown_end'

    with data_lock:
        shared_data['steering_input'] = new_steering
        shared_data['acceleration_input'] = 1.0
        lane_after = shared_data['lane_index']

    state_after = steer_state
    processing_ms = f'{(time.time() - process_start) * 1000.0:.2f}'
    _diag_record(
        diag_event,
        frame=front_frame,
        analysis=analysis,
        frame_seq=frame_seq,
        frame_timestamp=frame_timestamp,
        state_before=state_before,
        state_after=state_after,
        lane_before=lane_before,
        lane_after=lane_after,
        target_lane=target_lane,
        steering_output=f'{new_steering:.2f}',
        processing_ms=processing_ms,
        note=diag_note,
        save_frame=save_diag_frame,
    )

def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    with data_lock:
        steering_input = shared_data['steering_input']
        acceleration_input = shared_data['acceleration_input']

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None

# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Auto Drive...")

    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()

    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")

    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)

    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()

    try:
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()

    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    _diag_close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
