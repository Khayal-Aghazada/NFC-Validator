from typing import Optional

import cv2
import numpy as np


def validate_roi(roi: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x, y, rw, rh = roi

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))

    return x, y, rw, rh


def select_roi(frame: np.ndarray, window_name: str = "Select ROI") -> Optional[tuple[int, int, int, int]]:
    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None

    return validate_roi((int(x), int(y), int(w), int(h)), frame.shape)