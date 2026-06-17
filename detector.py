"""
detector.py  –  Real-Time Object Detector for SpeedTrials2D
============================================================
Detects and annotates every frame with:
  • The ego car          (red bounding box + label)
  • Tokens               (green / red / yellow – colored box + label)
  • Lane lines           (Hough segments overlaid in cyan)
  • Lane corners         (intersection points in magenta – used for depth later)

Public API
----------
    annotated = detect_and_annotate(frame)   →  BGR frame with all overlays

Internal results are also returned via detect_objects(frame) as a dict:
    {
        'car':          (x, y, w, h) or None,
        'tokens':       [{'bbox':(x,y,w,h), 'color':'green'|'red'|'yellow'}, ...],
        'lane_lines':   [(x1,y1,x2,y2), ...],
        'lane_corners': [(x, y), ...],
    }
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────
#  Tunable constants  (adjust here, not scattered in code)
# ─────────────────────────────────────────────────────────

# Road region: ignore the top N% (sky + UI text) for most detections
ROAD_START_FRAC = 0.40          # road starts at ~40 % from top (tokens appear ~y=225)

# ── Car ──────────────────────────────────────────────────
# The ego car body is bright red  (HSV hue ≈ 0 or 175-180, high saturation)
CAR_HSV_RANGES = [
    (np.array([0,   180, 100]), np.array([8,  255, 255])),   # pure red
    (np.array([172, 180,  80]), np.array([180, 255, 255])),  # magenta-red
]
CAR_MIN_AREA   = 800   # px²  – ignore tiny blobs
# Car is always in the bottom third of the frame
CAR_Y_FRAC     = 0.60

# ── Tokens ───────────────────────────────────────────────
TOKEN_MIN_AREA    = 80     # px²
TOKEN_MAX_AREA    = 15000  # px²  – tokens can be large spheres up close
TOKEN_MAX_ASPECT  = 2.0    # width/height – tokens are roughly square
TOKEN_EDGE_MARGIN = 55     # px  – ignore blobs this close to left/right edge
# HSV ranges per colour
TOKEN_RANGES = {
    'green':  [(np.array([45,  80, 80]),  np.array([85,  255, 255]))],
    'red':    [(np.array([0,   120, 120]), np.array([10,  255, 255])),
               (np.array([165, 120, 120]), np.array([180, 255, 255]))],
    'yellow': [(np.array([18,  100, 100]), np.array([35,  255, 255]))],
}

# ── Lane lines ───────────────────────────────────────────
LANE_WHITE_THRESH  = 175   # minimum brightness for a lane-dash pixel
LANE_HOUGH_RHO     = 1
LANE_HOUGH_THETA   = np.pi / 180
LANE_HOUGH_THRESH  = 30
LANE_HOUGH_MIN_LEN = 20
LANE_HOUGH_MAX_GAP = 15
# Accept only lines that are roughly vertical (angle from vertical < 40°)
LANE_MAX_ANGLE_DEG = 65

# ── Colours for drawing ───────────────────────────────────
COLOR_CAR         = (0,   0,   255)   # red box
COLOR_TOKEN_GREEN = (0,   220,  50)
COLOR_TOKEN_RED   = (0,    50, 220)
COLOR_TOKEN_YEL   = (0,   210, 255)
COLOR_LANE        = (255, 220,   0)   # cyan-yellow
COLOR_CORNER      = (255,  50, 255)   # magenta
FONT              = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def _hsv_mask(hsv, ranges):
    """Union of multiple HSV range masks."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def _largest_contour_bbox(mask, min_area=50):
    """Return (x,y,w,h) of the largest contour above min_area, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    biggest = max(valid, key=cv2.contourArea)
    return cv2.boundingRect(biggest)


def _line_angle_deg(x1, y1, x2, y2):
    """Angle of line from vertical (0 = perfectly vertical)."""
    dx, dy = x2 - x1, y2 - y1
    if dy == 0:
        return 90.0
    return abs(np.degrees(np.arctan2(abs(dx), abs(dy))))


def _segment_intersection(s1, s2):
    """
    Find the intersection point of two infinite lines defined by segments.
    Returns (x, y) as ints or None if lines are parallel.
    """
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = int(x1 + t * (x2 - x1))
    iy = int(y1 + t * (y2 - y1))
    return (ix, iy)


# ─────────────────────────────────────────────────────────
#  Detection functions
# ─────────────────────────────────────────────────────────

def detect_car(frame, hsv):
    """
    Detect the ego car.
    Returns (x, y, w, h) bounding box or None.
    """
    h, w = frame.shape[:2]
    car_y = int(h * CAR_Y_FRAC)

    # Build red mask, restrict to bottom portion of frame
    mask = _hsv_mask(hsv, CAR_HSV_RANGES)
    mask[:car_y, :] = 0   # blank out everything above car zone

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Find the largest blob that is plausibly the car
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= CAR_MIN_AREA]
    if not valid:
        return None

    # The car is the largest red blob at the bottom-centre
    biggest = max(valid, key=cv2.contourArea)
    return cv2.boundingRect(biggest)


def detect_tokens(frame, hsv, car_bbox):
    """
    Detect green / red / yellow tokens on the road.
    Returns list of dicts: [{'bbox': (x,y,w,h), 'color': str}, ...]
    """
    h, w = frame.shape[:2]
    road_y = int(h * ROAD_START_FRAC)
    tokens = []

    for color, ranges in TOKEN_RANGES.items():
        mask = _hsv_mask(hsv, ranges)
        # Ignore the sky / UI area
        mask[:road_y, :] = 0

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < TOKEN_MIN_AREA or area > TOKEN_MAX_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)

            # Tokens are roughly square — reject very elongated blobs (road borders)
            aspect = bw / (bh + 1e-6)
            if aspect > TOKEN_MAX_ASPECT or aspect < 1.0 / TOKEN_MAX_ASPECT:
                continue

            # Reject blobs touching the left/right frame edges (road border artefacts)
            if x < TOKEN_EDGE_MARGIN or (x + bw) > (w - TOKEN_EDGE_MARGIN):
                continue

            # Skip if it heavily overlaps with the car bbox
            if car_bbox is not None:
                cx, cy, cw, ch = car_bbox
                overlap_x = max(0, min(x+bw, cx+cw) - max(x, cx))
                overlap_y = max(0, min(y+bh, cy+ch) - max(y, cy))
                if overlap_x * overlap_y > 0.5 * area:
                    continue

            tokens.append({'bbox': (x, y, bw, bh), 'color': color})

    return tokens


def detect_lane_lines(frame):
    """
    Detect dashed white lane markings using Hough transform.
    Returns list of (x1, y1, x2, y2) tuples (absolute frame coords).
    """
    h, w = frame.shape[:2]
    road_y = int(h * ROAD_START_FRAC)

    # Work only on the road region
    road_crop = frame[road_y:, :]
    gray      = cv2.cvtColor(road_crop, cv2.COLOR_BGR2GRAY)

    # Isolate bright lane dashes
    _, white_mask = cv2.threshold(gray, LANE_WHITE_THRESH, 255, cv2.THRESH_BINARY)

    # Erode to remove road surface noise, then dilate to reconnect dashes
    k3 = np.ones((3, 3), np.uint8)
    white_mask = cv2.erode(white_mask,  k3, iterations=1)
    white_mask = cv2.dilate(white_mask, k3, iterations=2)

    edges = cv2.Canny(white_mask, 50, 150)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=LANE_HOUGH_RHO,
        theta=LANE_HOUGH_THETA,
        threshold=LANE_HOUGH_THRESH,
        minLineLength=LANE_HOUGH_MIN_LEN,
        maxLineGap=LANE_HOUGH_MAX_GAP,
    )

    lines = []
    if raw_lines is not None:
        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            if _line_angle_deg(x1, y1, x2, y2) <= LANE_MAX_ANGLE_DEG:
                # Convert back to full-frame coordinates
                lines.append((x1, y1 + road_y, x2, y2 + road_y))

    return lines


def detect_lane_corners(lane_lines, frame_shape):
    """
    Find lane corner points = intersections of left-side and right-side lane lines.

    Strategy:
      1. Split detected lines into 'left' (x-centroid < frame_centre)
         and 'right' (x-centroid >= frame_centre) groups.
      2. Fit one representative line per group via least-squares.
      3. Compute their intersection → the perspective vanishing / corner point.
      4. Also report the bottom endpoints of each fitted line (near-corners).

    Returns list of (x, y) points (may be empty if insufficient lines).
    """
    h, w = frame_shape[:2]
    cx = w // 2

    left_pts  = []
    right_pts = []

    for (x1, y1, x2, y2) in lane_lines:
        mid_x = (x1 + x2) / 2
        if mid_x < cx:
            left_pts.extend([(x1, y1), (x2, y2)])
        else:
            right_pts.extend([(x1, y1), (x2, y2)])

    def fit_line(pts):
        """Return flat 4-element fitLine array, or None."""
        if len(pts) < 2:
            return None
        arr = np.array(pts, dtype=np.float32)
        return cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    def line_to_segment(fit, y_bottom, y_top):
        """Convert fitLine output to two endpoints at given y values."""
        vx = float(fit[0])
        vy = float(fit[1])
        x0 = float(fit[2])
        y0 = float(fit[3])
        if abs(vy) < 1e-6:
            return None
        t1 = (y_bottom - y0) / vy
        t2 = (y_top    - y0) / vy
        return (int(x0 + t1 * vx), int(y_bottom),
                int(x0 + t2 * vx), int(y_top))

    left_fit  = fit_line(left_pts)
    right_fit = fit_line(right_pts)

    corners = []
    y_bottom = h - 1
    y_top    = int(h * ROAD_START_FRAC)

    left_seg  = line_to_segment(left_fit,  y_bottom, y_top) if left_fit  is not None else None
    right_seg = line_to_segment(right_fit, y_bottom, y_top) if right_fit is not None else None

    # Bottom-of-frame near-corners (where lane edges meet the car's horizon)
    if left_seg:
        corners.append((left_seg[0],  left_seg[1]))   # bottom-left corner
    if right_seg:
        corners.append((right_seg[0], right_seg[1]))  # bottom-right corner

    # Vanishing / far corner (where left and right converge)
    if left_seg and right_seg:
        vp = _segment_intersection(left_seg, right_seg)
        if vp is not None:
            vx, vy = vp
            # Only keep if it's within a reasonable region (upper road area)
            if 0 <= vx <= w and y_top - 80 <= vy <= y_top + 80:
                corners.append(vp)

    return corners, left_seg, right_seg


# ─────────────────────────────────────────────────────────
#  Main public API
# ─────────────────────────────────────────────────────────

def detect_objects(frame):
    """
    Run all detectors on a BGR frame.

    Returns
    -------
    dict with keys:
        'car'          : (x,y,w,h) or None
        'tokens'       : [{'bbox':(x,y,w,h), 'color': str}, ...]
        'lane_lines'   : [(x1,y1,x2,y2), ...]
        'lane_corners' : [(x,y), ...]
        '_left_seg'    : fitted left  lane segment (internal, for drawing)
        '_right_seg'   : fitted right lane segment (internal, for drawing)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    car     = detect_car(frame, hsv)
    tokens  = detect_tokens(frame, hsv, car)
    lines   = detect_lane_lines(frame)
    corners, left_seg, right_seg = detect_lane_corners(lines, frame.shape)

    return {
        'car':          car,
        'tokens':       tokens,
        'lane_lines':   lines,
        'lane_corners': corners,
        '_left_seg':    left_seg,
        '_right_seg':   right_seg,
    }


def detect_and_annotate(frame):
    """
    Detect all objects and draw annotations on a copy of the frame.

    Returns the annotated BGR frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    results = detect_objects(frame)

    # ── Lane lines (raw Hough segments) ──────────────────
    for (x1, y1, x2, y2) in results['lane_lines']:
        cv2.line(out, (x1, y1), (x2, y2), COLOR_LANE, 2, cv2.LINE_AA)

    # ── Fitted lane boundary lines ────────────────────────
    for seg, side in [(results['_left_seg'], 'L'), (results['_right_seg'], 'R')]:
        if seg:
            x1, y1, x2, y2 = seg
            cv2.line(out, (x1, y1), (x2, y2), (0, 180, 255), 2, cv2.LINE_AA)

    # ── Lane corners ─────────────────────────────────────
    for i, (px, py) in enumerate(results['lane_corners']):
        cv2.circle(out, (px, py), 8, COLOR_CORNER, -1)
        cv2.circle(out, (px, py), 9, (255, 255, 255), 1)
        cv2.putText(out, f'C{i}', (px + 10, py - 6),
                    FONT, 0.45, COLOR_CORNER, 1, cv2.LINE_AA)

    # ── Tokens ───────────────────────────────────────────
    token_draw_color = {
        'green':  COLOR_TOKEN_GREEN,
        'red':    COLOR_TOKEN_RED,
        'yellow': COLOR_TOKEN_YEL,
    }
    for tok in results['tokens']:
        x, y, bw, bh = tok['bbox']
        col   = tok['color']
        dcol  = token_draw_color.get(col, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), dcol, 2)
        # Filled label background
        label = col.upper()
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), dcol, -1)
        cv2.putText(out, label, (x + 2, y - 4),
                    FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Car ──────────────────────────────────────────────
    if results['car']:
        x, y, bw, bh = results['car']
        cv2.rectangle(out, (x, y), (x + bw, y + bh), COLOR_CAR, 2)
        label = 'EGO CAR'
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 4, y), COLOR_CAR, -1)
        cv2.putText(out, label, (x + 2, y - 5),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── HUD: detection summary ────────────────────────────
    n_tok  = len(results['tokens'])
    n_lane = len(results['lane_lines'])
    n_corn = len(results['lane_corners'])
    car_detected = results['car'] is not None

    hud_lines = [
        f"Car:     {'YES' if car_detected else 'NO'}",
        f"Tokens:  {n_tok}",
        f"Lanes:   {n_lane} segs",
        f"Corners: {n_corn}",
    ]
    hud_x, hud_y = 8, h - 10 - (len(hud_lines) * 18)
    bg_h = len(hud_lines) * 18 + 6
    cv2.rectangle(out, (hud_x - 4, hud_y - 4),
                  (hud_x + 140, hud_y + bg_h), (0, 0, 0), -1)
    cv2.rectangle(out, (hud_x - 4, hud_y - 4),
                  (hud_x + 140, hud_y + bg_h), (80, 80, 80), 1)
    for i, line in enumerate(hud_lines):
        cv2.putText(out, line, (hud_x, hud_y + i * 18 + 14),
                    FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return out, results
