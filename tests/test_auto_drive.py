import unittest

import cv2
import numpy as np

import auto_drive


FRAME_W = 640
FRAME_H = 480


def make_frame(x, y=300, color=(0, 0, 255)):
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    cv2.circle(frame, (x, y), 14, color, -1)
    return frame


class HazardDetectionTests(unittest.TestCase):
    def test_center_lane_red_hazard_returns_dodge_direction(self):
        tracker = auto_drive.StaticBlobTracker()
        analysis = auto_drive.analyze_hazard(make_frame(300), tracker=tracker)

        self.assertEqual(analysis['direction'], 1)
        self.assertIsNotNone(analysis['in_lane_box'])
        self.assertEqual(len(analysis['real_boxes']), 1)

    def test_adjacent_lane_red_hazard_does_not_trigger(self):
        tracker = auto_drive.StaticBlobTracker()
        analysis = auto_drive.analyze_hazard(make_frame(220), tracker=tracker)

        self.assertEqual(analysis['direction'], 0)
        self.assertIsNone(analysis['in_lane_box'])
        self.assertEqual(len(analysis['real_boxes']), 1)

    def test_yellow_hazard_is_treated_as_hazard(self):
        tracker = auto_drive.StaticBlobTracker()
        analysis = auto_drive.analyze_hazard(make_frame(340, color=(0, 255, 255)), tracker=tracker)

        self.assertEqual(analysis['direction'], -1)
        self.assertIsNotNone(analysis['in_lane_box'])

    def test_static_blob_is_suppressed_after_threshold(self):
        tracker = auto_drive.StaticBlobTracker()
        analysis = None

        for _ in range(auto_drive.STATIC_FRAMES_THRESH):
            analysis = auto_drive.analyze_hazard(make_frame(320), tracker=tracker)

        self.assertEqual(analysis['direction'], 0)
        self.assertEqual(analysis['real_boxes'], [])
        self.assertEqual(len(analysis['suppressed_boxes']), 1)

    def test_moving_token_is_not_suppressed_as_static(self):
        tracker = auto_drive.StaticBlobTracker()
        analysis = None

        for step in range(auto_drive.STATIC_FRAMES_THRESH):
            x = 276 + step * (auto_drive.STATIC_DIST_THRESH + 2)
            analysis = auto_drive.analyze_hazard(make_frame(x), tracker=tracker)

        self.assertNotEqual(analysis['real_boxes'], [])
        self.assertEqual(analysis['suppressed_boxes'], [])


if __name__ == '__main__':
    unittest.main()
