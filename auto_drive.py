import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

NUM_LANES = 5
START_LANE = 3  # assume the car starts centered

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

# Event detection constants
LOW_BRIGHTNESS_MEAN_V = 90          # Value channel threshold for low brightness
LOW_BRIGHTNESS_DARK_PIXEL_FRAC = 0.30
POLICE_RED_TOKEN_AREA = 100         # area threshold for a visible police red token
POLICE_EVENT_TIMEOUT = 2.0          # seconds to remain in police event state

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
    'latest_back_frame': None,
    'steering_input': 0.0,
    'acceleration_input': 1.0,
    'lane_index': START_LANE,
    'lights_on': False,
    'low_brightness': False,
    'police_event': False,
}
data_lock = threading.Lock()
is_running = True

police_event_deadline = 0.0

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
                with data_lock:
                    shared_data[data_key] = frame

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

    def filter_real(self, boxes):
        """Given a list of (x, y, w, h), return the ones that are NOT static."""
        bucket_size = STATIC_DIST_THRESH
        seen = set()
        real = []

        for (x, y, w, h) in boxes:
            cx, cy = x + w // 2, y + h // 2
            bk = (cx // bucket_size, cy // bucket_size)
            seen.add(bk)

            count = self._blobs.get(bk, 0) + 1
            self._blobs[bk] = count

            if count < STATIC_FRAMES_THRESH:
                real.append((x, y, w, h))
            # else: suppressed as a static road marker

        # Decay buckets not seen this frame so a marker that scrolls away
        # doesn't permanently poison that bucket
        for bk in list(self._blobs):
            if bk not in seen:
                self._blobs[bk] = max(0, self._blobs[bk] - 2)
                if self._blobs[bk] <= 0:
                    del self._blobs[bk]

        return real


_static_tracker = StaticBlobTracker()

# ---------------------------------------------------------
# Hazard Detection
# ---------------------------------------------------------
def detect_low_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark_pixels = np.count_nonzero(v < LOW_BRIGHTNESS_MEAN_V)
    return (dark_pixels / v.size) >= LOW_BRIGHTNESS_DARK_PIXEL_FRAC


def find_police_red_token(frame):
    h, w = frame.shape[:2]
    y_min = int(h * REACT_Y_MIN_FRAC)
    y_max = int(h * REACT_Y_MAX_FRAC)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in HAZARD_HSV_RANGES['red']:
        red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, lo, hi))

    zone_mask = np.zeros_like(red_mask)
    zone_mask[y_min:y_max, :] = red_mask[y_min:y_max, :]

    contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < POLICE_RED_TOKEN_AREA:
            continue
        boxes.append(cv2.boundingRect(c))

    return boxes


def find_hazard_direction(frame, police_mode=False):
    """
    Look for red/yellow tokens in the reaction zone that are in our lane.
    Returns -1 (steer left), +1 (steer right), or 0 (no hazard ahead).
    """
    h, w = frame.shape[:2]
    y_min = int(h * REACT_Y_MIN_FRAC)
    y_max = int(h * REACT_Y_MAX_FRAC)
    cx = w // 2
    lane_half_width = int(w * LANE_HALF_WIDTH_FRAC)

    if police_mode:
        red_boxes = find_police_red_token(frame)
        if red_boxes:
            closest = min(red_boxes, key=lambda b: b[1] + b[3])
            token_cx = closest[0] + closest[2] // 2
            if abs(token_cx - cx) <= lane_half_width:
                return 0
            return -1 if token_cx < cx else 1

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

    boxes = _static_tracker.filter_real(boxes)

    in_lane = False
    token_cx = cx
    for (x, y, bw, bh) in boxes:
        bcx = x + bw // 2
        if abs(bcx - cx) <= lane_half_width:
            in_lane = True
            token_cx = bcx
            break

    if not in_lane:
        return 0

    # Token is in our lane: dodge away from its exact position.
    # If it's left-of-center (or dead center), go right; otherwise go left.
    return 1 if token_cx <= cx else -1

# ---------------------------------------------------------
# Steering State Machine
# ---------------------------------------------------------
steer_state = 'IDLE'  # IDLE -> TAPPING -> COOLDOWN -> IDLE
steer_direction = 0
steer_deadline = 0.0  # time.time() value at which the current state ends

def processing_task():
    global steer_state, steer_direction, steer_deadline, police_event_deadline

    with data_lock:
        front_frame = shared_data['latest_front_frame']
        lane_index = shared_data['lane_index']
        lights_on = shared_data['lights_on']
        police_event = shared_data['police_event']

    now = time.time()
    new_steering = 0.0
    low_brightness = False
    next_police_event = police_event

    if front_frame is not None:
        low_brightness = detect_low_brightness(front_frame)
        red_boxes = find_police_red_token(front_frame)
        if red_boxes:
            police_event_deadline = now + POLICE_EVENT_TIMEOUT
            next_police_event = True
        elif now >= police_event_deadline:
            next_police_event = False

    if steer_state == 'IDLE':
        if front_frame is not None:
            direction = find_hazard_direction(front_frame, police_mode=next_police_event)
            if direction != 0:
                target_lane = lane_index + direction
                if 0 <= target_lane < NUM_LANES:
                    steer_direction = direction
                    steer_state = 'TAPPING'
                    steer_deadline = now + TAP_DURATION
        new_steering = 0.0

    elif steer_state == 'TAPPING':
        new_steering = float(steer_direction)
        if now >= steer_deadline:
            with data_lock:
                shared_data['lane_index'] += steer_direction
            steer_state = 'COOLDOWN'
            steer_deadline = now + COOLDOWN_DURATION

    elif steer_state == 'COOLDOWN':
        new_steering = 0.0
        if now >= steer_deadline:
            steer_state = 'IDLE'

    with data_lock:
        shared_data['steering_input'] = new_steering
        shared_data['acceleration_input'] = 1.0
        shared_data['low_brightness'] = low_brightness
        shared_data['police_event'] = next_police_event
        # If low brightness is detected, we flag headlights requested.
        # Actual headlight control is not exposed through the current two-float
        # control protocol, so this is a placeholder for future support.
        shared_data['lights_on'] = lights_on or low_brightness

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
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
