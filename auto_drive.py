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
START_LANE = 2  # corrected: middle lane of 0..4 is index 2

# Reaction zone: vertical band (fraction of frame height) where tokens are
# close enough to react to
REACT_Y_MIN_FRAC = 0.45
REACT_Y_MAX_FRAC = 0.90

# A token counts as "in our lane" if its center x falls within this fraction
# of the frame width around the center (one lane's worth of width)
LANE_HALF_WIDTH_FRAC = 0.10

# Token HSV ranges
TOKEN_HSV_RANGES = {
    'green': [
        (np.array([35, 50, 50]), np.array([85, 255, 255])),
    ],
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
    'latest_back_frame': None,
    'steering_input': 0.0,
    'acceleration_input': 1.0,
    'lane_index': START_LANE,
}
data_lock = threading.Lock()
is_running = True

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
# Lane Scoring & Hazard Avoidance
# ---------------------------------------------------------
def find_best_lane(frame, lane_index):
    """
    Evaluates all 5 lanes based on green tokens (attraction) and red/yellow tokens (avoidance).
    Returns the target lane index (0 to 4) that has the highest score and a safe path.
    """
    h, w = frame.shape[:2]
    y_min = int(h * REACT_Y_MIN_FRAC)
    y_max = int(h * REACT_Y_MAX_FRAC)
    cx = w // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Detect red/yellow hazards
    hazard_mask = np.zeros((h, w), dtype=np.uint8)
    for ranges in TOKEN_HSV_RANGES['red'] + TOKEN_HSV_RANGES['yellow']:
        hazard_mask = cv2.bitwise_or(hazard_mask, cv2.inRange(hsv, ranges[0], ranges[1]))

    # 2. Detect green tokens
    green_mask = np.zeros((h, w), dtype=np.uint8)
    for ranges in TOKEN_HSV_RANGES['green']:
        green_mask = cv2.bitwise_or(green_mask, cv2.inRange(hsv, ranges[0], ranges[1]))

    # Crop to reaction zone
    zone_hazard = np.zeros_like(hazard_mask)
    zone_hazard[y_min:y_max, :] = hazard_mask[y_min:y_max, :]

    zone_green = np.zeros_like(green_mask)
    zone_green[y_min:y_max, :] = green_mask[y_min:y_max, :]

    # Find hazard boxes
    contours_hazard, _ = cv2.findContours(zone_hazard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hazard_boxes = []
    for c in contours_hazard:
        if cv2.contourArea(c) < TOKEN_MIN_AREA:
            continue
        hazard_boxes.append(cv2.boundingRect(c))

    # Apply static blob suppression to hazards
    hazard_boxes = _static_tracker.filter_real(hazard_boxes)

    # Find green token boxes
    contours_green, _ = cv2.findContours(zone_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_boxes = []
    for c in contours_green:
        if cv2.contourArea(c) < TOKEN_MIN_AREA:
            continue
        green_boxes.append(cv2.boundingRect(c))

    # Initialize lane scores (0..4)
    scores = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Apply lane preferences: prefer current lane and prefer center of the road
    for l in range(5):
        scores[l] += -0.1 * abs(l - lane_index)
        scores[l] += -0.05 * abs(l - 2)

    # Map hazards to lanes and penalize
    # Lane width is approximately 96 pixels.
    # relative lane = round((bcx - 320) / 96.0)
    for (x, y, bw, bh) in hazard_boxes:
        bcx = x + bw // 2
        rel_lane = int(round((bcx - cx) / 96.0))
        abs_lane = lane_index + rel_lane
        if 0 <= abs_lane < 5:
            scores[abs_lane] -= 100.0

    # Map green tokens to lanes and reward
    for (x, y, bw, bh) in green_boxes:
        bcx = x + bw // 2
        rel_lane = int(round((bcx - cx) / 96.0))
        abs_lane = lane_index + rel_lane
        if 0 <= abs_lane < 5:
            scores[abs_lane] += 15.0

    # Path safety evaluation: find the best lane
    best_lane = lane_index
    max_score = -999999.0

    for l in range(5):
        # Determine if there is a safe path to lane l
        # The path to lane l is safe if all lanes we must switch into are free of hazards (score > -50)
        path_safe = True
        step = 1 if l > lane_index else -1
        # Check intermediate lanes (excluding current, including target l)
        for check_l in range(lane_index + step, l + step, step):
            if check_l < 0 or check_l >= 5:
                path_safe = False
                break
            if scores[check_l] < -50.0:
                path_safe = False
                break

        if path_safe:
            if scores[l] > max_score:
                max_score = scores[l]
                best_lane = l

    return best_lane

# ---------------------------------------------------------
# Steering State Machine
# ---------------------------------------------------------
steer_state = 'IDLE'  # IDLE -> TAPPING -> COOLDOWN -> IDLE
steer_direction = 0
steer_deadline = 0.0  # time.time() value at which the current state ends

def processing_task():
    global steer_state, steer_direction, steer_deadline

    with data_lock:
        front_frame = shared_data['latest_front_frame']
        lane_index = shared_data['lane_index']

    now = time.time()
    new_steering = 0.0

    if steer_state == 'IDLE':
        if front_frame is not None:
            best_lane = find_best_lane(front_frame, lane_index)
            if best_lane != lane_index:
                direction = 1 if best_lane > lane_index else -1
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
                shared_data['lane_index'] = max(0, min(4, shared_data['lane_index'] + steer_direction))
            steer_state = 'COOLDOWN'
            steer_deadline = now + COOLDOWN_DURATION

    elif steer_state == 'COOLDOWN':
        new_steering = 0.0
        if now >= steer_deadline:
            steer_state = 'IDLE'

    with data_lock:
        shared_data['steering_input'] = new_steering
        shared_data['acceleration_input'] = 1.0

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
