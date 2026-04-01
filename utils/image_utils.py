from typing import Optional

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    if width is None and height is None:
        return frame

    h, w = frame.shape[:2]

    if width is not None and height is not None:
        return cv2.resize(frame, (width, height))

    if width is not None:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(frame, (width, new_h))

    ratio = height / h
    new_w = int(w * ratio)
    return cv2.resize(frame, (new_w, height))


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y:y + h, x:x + w].copy()


def to_hsv(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    gray = to_gray(frame)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def safe_contour_bbox(contour) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(w), int(h)