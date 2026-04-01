from typing import Any

import cv2
import numpy as np

from config import (
    GREEN_LOWER,
    GREEN_UPPER,
    RED1_LOWER,
    RED1_UPPER,
    RED2_LOWER,
    RED2_UPPER,
    LED_MIN_AREA,
    LED_DOMINANCE_RATIO,
)
from detectors.base_detector import BaseDetector
from utils.image_utils import to_hsv, safe_contour_bbox


class LEDDetector(BaseDetector):
    def __init__(self) -> None:
        self.kernel = np.ones((3, 3), np.uint8)

    def _create_mask(self, hsv: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _largest_blob(self, mask: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < LED_MIN_AREA:
            return 0.0, None

        bbox = safe_contour_bbox(largest)
        return area, bbox

    def detect(self, roi_frame: np.ndarray) -> dict[str, Any]:
        hsv = to_hsv(roi_frame)

        green_mask = self._create_mask(hsv, GREEN_LOWER, GREEN_UPPER)
        red_mask1 = self._create_mask(hsv, RED1_LOWER, RED1_UPPER)
        red_mask2 = self._create_mask(hsv, RED2_LOWER, RED2_UPPER)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_area, green_bbox = self._largest_blob(green_mask)
        red_area, red_bbox = self._largest_blob(red_mask)

        label = "NO_SIGNAL"
        confidence = 0.0
        debug: dict[str, Any] = {
            "green_bbox": green_bbox,
            "red_bbox": red_bbox,
        }

        if green_area >= LED_MIN_AREA and green_area > red_area * LED_DOMINANCE_RATIO:
            label = "GREEN"
            confidence = min(1.0, green_area / 1000.0)
        elif red_area >= LED_MIN_AREA and red_area > green_area * LED_DOMINANCE_RATIO:
            label = "RED"
            confidence = min(1.0, red_area / 1000.0)
        elif green_area >= LED_MIN_AREA and red_area >= LED_MIN_AREA:
            if green_area >= red_area:
                label = "GREEN"
                confidence = 0.5
            else:
                label = "RED"
                confidence = 0.5

        return {
            "label": label,
            "confidence": float(confidence),
            "details": {
                "green_area": int(green_area),
                "red_area": int(red_area),
            },
            "debug": debug,
        }