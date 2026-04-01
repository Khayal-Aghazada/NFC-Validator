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
    MULTI_LED_SLOTS,
    MULTI_LED_MIN_BRIGHT_PIXELS,
    MULTI_LED_BRIGHT_THRESHOLD,
    MULTI_LED_PASS_MIN_ON,
    MULTI_LED_FAIL_RED_REQUIRED,
)
from detectors.base_detector import BaseDetector
from utils.image_utils import to_hsv


class MultiLEDDetector(BaseDetector):
    def __init__(self) -> None:
        self.kernel = np.ones((3, 3), np.uint8)

    def _slot_boxes(self, roi_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = roi_frame.shape[:2]
        slot_w = w // MULTI_LED_SLOTS
        boxes = []

        for i in range(MULTI_LED_SLOTS):
            x = i * slot_w
            current_w = slot_w if i < MULTI_LED_SLOTS - 1 else w - x
            boxes.append((x, 0, current_w, h))

        return boxes

    def _classify_slot(self, slot: np.ndarray) -> str:
        gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
        bright_pixels = int(np.sum(gray > MULTI_LED_BRIGHT_THRESHOLD))

        hsv = to_hsv(slot)

        green_mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
        red_mask1 = cv2.inRange(hsv, np.array(RED1_LOWER), np.array(RED1_UPPER))
        red_mask2 = cv2.inRange(hsv, np.array(RED2_LOWER), np.array(RED2_UPPER))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_pixels = int(np.sum(green_mask > 0))
        red_pixels = int(np.sum(red_mask > 0))

        if red_pixels > MULTI_LED_MIN_BRIGHT_PIXELS:
            return "RED"
        if green_pixels > MULTI_LED_MIN_BRIGHT_PIXELS:
            return "GREEN"
        if bright_pixels > MULTI_LED_MIN_BRIGHT_PIXELS:
            return "ON"

        return "OFF"

    def detect(self, roi_frame: np.ndarray) -> dict[str, Any]:
        boxes = self._slot_boxes(roi_frame)
        states: list[str] = []

        for x, y, w, h in boxes:
            slot = roi_frame[y:y + h, x:x + w]
            states.append(self._classify_slot(slot))

        green_count = states.count("GREEN")
        red_count = states.count("RED")
        on_count = states.count("ON") + green_count + red_count

        label = "NO_SIGNAL"
        confidence = 0.0

        if MULTI_LED_FAIL_RED_REQUIRED and red_count >= 1:
            label = "PATTERN_FAIL"
            confidence = min(1.0, red_count / MULTI_LED_SLOTS + 0.4)
        elif on_count >= MULTI_LED_PASS_MIN_ON:
            label = "PATTERN_PASS"
            confidence = min(1.0, on_count / MULTI_LED_SLOTS)
        elif on_count > 0:
            label = "PARTIAL"
            confidence = on_count / MULTI_LED_SLOTS

        return {
            "label": label,
            "confidence": float(confidence),
            "details": {
                "led_states": states,
                "green_count": green_count,
                "red_count": red_count,
                "on_count": on_count,
            },
            "debug": {
                "led_boxes": boxes,
            },
        }