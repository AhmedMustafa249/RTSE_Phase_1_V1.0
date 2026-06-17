"""
rtse_framework.py  –  Shared RTOS Infrastructure for SpeedTrials2D
===================================================================
This module contains all the boilerplate that was previously copy-pasted
across every driver script:

  • Network constants (camera / control ports)
  • TaskPriority / RTTask  (real-time scheduling framework)
  • Shared state (shared_data, data_lock, is_running, display_queue)
  • setup_cameras()       – connects to front & back camera sockets
  • setup_control_server() – accepts the Unity control client
  • read_single_camera()  – drains the socket buffer, decodes the frame
  • launch_simulator()    – launches the Linux x86_64 binary if present
  • run_main_loop()       – runs the OpenCV display loop on the main thread
  • shutdown()            – closes sockets, destroys windows, stops sim

Usage pattern
-------------
    from rtse_framework import (
        TaskPriority, RTTask,
        shared_data, data_lock, is_running, display_queue,
        setup_cameras, setup_control_server,
        read_single_camera,
        launch_simulator, run_main_loop, shutdown,
        front_camera_sock, back_camera_sock, control_conn,
    )
"""

import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes
import sys
import os
import subprocess
import queue

# ─────────────────────────────────────────────────────────────────────────────
# Network Configuration
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_HOST       = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT  = 8082
CONTROL_HOST      = '127.0.0.1'
CONTROL_PORT      = 8081

# ─────────────────────────────────────────────────────────────────────────────
# Shared State  (module-level so all tasks share one reference)
# ─────────────────────────────────────────────────────────────────────────────
shared_data: dict = {
    'latest_front_frame': None,
    'latest_back_frame':  None,
    'steering_input':     0.0,
    'acceleration_input': 0.0,
}

data_lock     = threading.Lock()
is_running    = True                          # set False to stop all tasks
display_queue = queue.Queue(maxsize=8)        # (window_name, frame) tuples

# ─────────────────────────────────────────────────────────────────────────────
# Socket handles  (populated by setup_cameras / setup_control_server)
# ─────────────────────────────────────────────────────────────────────────────
front_camera_sock = None
back_camera_sock  = None
control_conn      = None

# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Scheduling Framework
# ─────────────────────────────────────────────────────────────────────────────
class TaskPriority:
    """Logical priority levels for RTTask."""
    HIGH   = 1
    MEDIUM = 2
    LOW    = 3


class RTTask(threading.Thread):
    """
    Real-Time Task:
      - Runs ``execute_func`` in a loop with a fixed ``period`` (seconds).
      - Sets OS-level thread priority where supported (Windows only; gracefully
        ignored on Linux/macOS).
      - Marked as daemon so it does not block process exit.
    """
    _PRIORITY_MAP = {
        TaskPriority.HIGH:   2,
        TaskPriority.MEDIUM: 0,
        TaskPriority.LOW:   -2,
    }

    def __init__(self, name: str, period: float, priority: int, execute_func):
        super().__init__()
        self.name         = name
        self.period       = period
        self.priority     = priority
        self.execute_func = execute_func
        self.daemon       = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")

        # Attempt Windows-style thread priority (no-op on Linux)
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(
                handle, self._PRIORITY_MAP.get(self.priority, 0)
            )
        except Exception:
            pass

        while is_running:
            t0 = time.time()
            self.execute_func()
            sleep_time = self.period - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)


# ─────────────────────────────────────────────────────────────────────────────
# Network Setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_cameras():
    """
    Blocking: connects to both camera sockets. Retries every second until both
    are connected or ``is_running`` becomes False.
    Intended to be run in a daemon thread.
    """
    global front_camera_sock, back_camera_sock
    print("Connecting to Cameras...")
    fok = bok = False

    while is_running and not (fok and bok):
        if not fok:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                fok = True
            except Exception:
                pass

        if not bok:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                bok = True
            except Exception:
                pass

        if not (fok and bok):
            time.sleep(1)


def setup_control_server():
    """
    Blocking: binds a TCP server on CONTROL_PORT and waits for the Unity game
    client to connect. Sets ``control_conn`` once connected.
    Intended to be run in a daemon thread.
    """
    global control_conn
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((CONTROL_HOST, CONTROL_PORT))
    srv.listen()
    srv.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")

    while is_running:
        try:
            conn, addr = srv.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Camera Frame Reader
# ─────────────────────────────────────────────────────────────────────────────
def read_single_camera(sock, data_key: str,
                        window_name: str | None = None,
                        push_display: bool = True):
    """
    Reads and decodes the latest JPEG frame from a camera socket.

    The protocol sends a 4-byte little-endian length header followed by the
    raw JPEG bytes. This function drains the socket to always use the freshest
    available frame.

    Args:
        sock:         The connected camera socket (may be None).
        data_key:     Key in ``shared_data`` where the decoded frame is stored.
        window_name:  cv2.imshow window name (used only when push_display=True).
        push_display: If True, pushes a 640×480 copy to ``display_queue``.
    """
    if sock is None:
        return

    try:
        latest = None
        sock.settimeout(None)

        # Read the first frame
        lb = sock.recv(4)
        if not lb:
            return
        il = int.from_bytes(lb, 'little')
        r  = b''
        while len(r) < il and is_running:
            p = sock.recv(il - len(r))
            if not p:
                break
            r += p
        if len(r) == il:
            latest = r

        # Drain any queued frames — keep only the newest
        while is_running:
            rd, _, _ = select.select([sock], [], [], 0.0)
            if not rd:
                break
            sock.settimeout(1.0)
            lb = sock.recv(4)
            if not lb:
                return
            il = int.from_bytes(lb, 'little')
            r  = b''
            while len(r) < il and is_running:
                p = sock.recv(il - len(r))
                if not p:
                    break
                r += p
            if len(r) == il:
                latest = r

        if latest is not None:
            arr   = np.frombuffer(latest, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame

                if push_display and window_name:
                    try:
                        display_queue.put_nowait(
                            (window_name, cv2.resize(frame, (640, 480)))
                        )
                    except queue.Full:
                        pass

    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Simulator Launcher
# ─────────────────────────────────────────────────────────────────────────────
def launch_simulator() -> subprocess.Popen | None:
    """
    On Linux, locates and launches ``SpeedTrials2D-linux.x86_64`` from the
    same directory as this module.

    Returns the ``Popen`` handle, or ``None`` if not on Linux / not found.
    """
    if not sys.platform.startswith('linux'):
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    executable  = os.path.join(script_dir, 'SpeedTrials2D-linux.x86_64')

    if not os.path.exists(executable):
        print(f"Simulator executable not found at: {executable}")
        return None

    try:
        os.chmod(executable, os.stat(executable).st_mode | 0o111)
        print(f"Launching simulator: {executable}")
        return subprocess.Popen(
            [executable],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to launch simulator: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Display Loop  (must run on the main thread for cv2.imshow on Linux)
# ─────────────────────────────────────────────────────────────────────────────
def run_main_loop():
    """
    Drains ``display_queue`` and calls ``cv2.imshow`` on the main thread.
    Blocks until a KeyboardInterrupt is received, then sets ``is_running=False``.
    """
    global is_running
    try:
        while is_running:
            try:
                while True:
                    window_name, frame = display_queue.get_nowait()
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)
            except queue.Empty:
                pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False


# ─────────────────────────────────────────────────────────────────────────────
# Graceful Shutdown
# ─────────────────────────────────────────────────────────────────────────────
def shutdown(simulator_process: subprocess.Popen | None = None):
    """
    Closes all sockets, destroys OpenCV windows, and terminates the simulator.

    Args:
        simulator_process: The handle returned by ``launch_simulator()``.
    """
    for sock in (front_camera_sock, back_camera_sock, control_conn):
        if sock:
            try:
                sock.close()
            except Exception:
                pass

    cv2.destroyAllWindows()

    if simulator_process is not None:
        print("Terminating simulator process...")
        simulator_process.terminate()
        try:
            simulator_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print("Simulator did not terminate in time. Killing it...")
            simulator_process.kill()
            simulator_process.wait()

    print("System terminated cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: standard camera task wrappers
# ─────────────────────────────────────────────────────────────────────────────
def read_front_camera_task(push_display: bool = False):
    """Task wrapper: reads the front camera socket."""
    read_single_camera(
        front_camera_sock, 'latest_front_frame',
        window_name="Front Camera", push_display=push_display
    )


def read_back_camera_task(push_display: bool = True):
    """Task wrapper: reads the back camera socket."""
    read_single_camera(
        back_camera_sock, 'latest_back_frame',
        window_name="Back Camera", push_display=push_display
    )


def send_controls_task(steering: float = 0.0, acceleration: float = 1.0):
    """
    Sends a fixed steering + acceleration command to the control socket.
    Use the shared_data version in drive scripts for dynamic control.
    """
    global control_conn
    if control_conn is None:
        return
    try:
        control_conn.sendall(struct.pack('ff', steering, acceleration))
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None
