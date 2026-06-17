"""
diag_run.py  –  Diagnostic Frame Capture
==========================================
Runs the game, saves EVERY frame where a token is detected plus periodic
frames without tokens, so we can inspect what the detector is actually seeing.

Saved files per frame index:
  diag_frames/raw_NNNN.png    – raw camera frame
  diag_frames/ann_NNNN.png    – annotated frame (from detect_and_annotate)
  diag_frames/masks_NNNN.png  – colour-channel debug image (G/R/Y masks)

All boilerplate (RTTask, sockets, simulator launch, display loop, etc.)
lives in ``rtse_framework.py``.
"""

import os
import struct
import threading
import time
import cv2
import numpy as np

import rtse_framework as fw
from rtse_framework import (
    TaskPriority, RTTask,
    shared_data, data_lock, display_queue,
    setup_cameras, setup_control_server,
    launch_simulator, run_main_loop, shutdown,
)
from detector import detect_and_annotate

# ─────────────────────────────────────────────────────────────────────────────
# Output Directory
# ─────────────────────────────────────────────────────────────────────────────
DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_frames')
os.makedirs(DIAG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic State
# ─────────────────────────────────────────────────────────────────────────────
frame_idx   = [0]
last_save   = [0.0]
token_saves = [0]

# ─────────────────────────────────────────────────────────────────────────────
# Task Implementations
# ─────────────────────────────────────────────────────────────────────────────

def read_front_camera_task():
    fw.read_single_camera(fw.front_camera_sock, 'latest_front_frame',
                          push_display=False)


def read_back_camera_task():
    fw.read_single_camera(fw.back_camera_sock, 'latest_back_frame',
                          push_display=False)


def diag_task():
    """Runs detector on the latest front frame and saves diagnostic images."""
    with data_lock:
        f = shared_data['latest_front_frame']
    if f is None:
        return

    annotated, results = detect_and_annotate(f)
    n_tok = len(results['tokens'])
    now   = time.time()

    # Save if tokens were detected OR every 3 seconds (periodic heartbeat)
    if n_tok > 0 or (now - last_save[0] > 3.0):
        idx = frame_idx[0]

        # Raw + annotated frames
        cv2.imwrite(os.path.join(DIAG_DIR, f'raw_{idx:04d}.png'), f)
        cv2.imwrite(os.path.join(DIAG_DIR, f'ann_{idx:04d}.png'), annotated)

        # Colour-mask debug image (green / red / yellow channels)
        hsv   = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        g_mask = cv2.inRange(hsv, np.array([45,  80,  80]),  np.array([85, 255, 255]))
        r1     = cv2.inRange(hsv, np.array([0,  120, 120]),  np.array([10, 255, 255]))
        r2     = cv2.inRange(hsv, np.array([165,120, 120]),  np.array([180,255, 255]))
        r_mask = cv2.bitwise_or(r1, r2)
        y_mask = cv2.inRange(hsv, np.array([18, 100, 100]),  np.array([35, 255, 255]))

        debug = np.zeros_like(f)
        debug[:, :, 1] = g_mask   # green channel → green tokens
        debug[:, :, 2] = r_mask   # red   channel → red tokens
        debug[:, :, 0] = y_mask   # blue  channel → yellow tokens
        cv2.imwrite(os.path.join(DIAG_DIR, f'masks_{idx:04d}.png'), debug)

        if n_tok > 0:
            token_saves[0] += 1
            print(f'[{idx:04d}] TOKENS={n_tok}: {[(t["color"], t["bbox"]) for t in results["tokens"]]}')
        else:
            print(f'[{idx:04d}] no tokens (periodic save)')

        frame_idx[0] += 1
        last_save[0]  = now

    # Push annotated preview to display queue
    try:
        display_queue.put_nowait(("Diag", cv2.resize(annotated, (640, 480))))
    except Exception:
        pass


def send_controls_task():
    """Keeps the car driving straight at full throttle during diagnostics."""
    if fw.control_conn is None:
        return
    try:
        fw.control_conn.sendall(struct.pack('ff', 0.0, 1.0))
    except Exception:
        fw.control_conn = None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== DIAGNOSTIC RUN ===")

    sim = launch_simulator()

    try:
        threading.Thread(target=setup_control_server, daemon=True).start()
        threading.Thread(target=setup_cameras,        daemon=True).start()

        t_front = RTTask("FrontCam", period=0.005, priority=TaskPriority.HIGH,   execute_func=read_front_camera_task)
        t_back  = RTTask("BackCam",  period=0.005, priority=TaskPriority.HIGH,   execute_func=read_back_camera_task)
        t_diag  = RTTask("Diag",     period=0.033, priority=TaskPriority.MEDIUM, execute_func=diag_task)
        t_ctrl  = RTTask("Ctrl",     period=0.005, priority=TaskPriority.HIGH,   execute_func=send_controls_task)

        for t in [t_front, t_back, t_diag, t_ctrl]:
            t.start()

        run_main_loop()  # blocks until Ctrl+C
        print(f"\nDone. Token frames saved: {token_saves[0]}, total frames: {frame_idx[0]}")

        for t in [t_front, t_back, t_diag, t_ctrl]:
            t.join()

    finally:
        shutdown(sim)
