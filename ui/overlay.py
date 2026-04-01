from typing import Any

import cv2
import numpy as np

from config import RESULT_BANNER_HEIGHT
from utils.draw_utils import draw_rectangle, draw_status_label, draw_text_box


class OverlayRenderer:
    def draw(self, frame: np.ndarray, ui_state: dict[str, Any]) -> np.ndarray:
        output = frame.copy()

        roi = ui_state.get("roi")
        if roi:
            self._draw_roi(output, roi)

        self._draw_header(output, ui_state)
        self._draw_detector_debug(output, ui_state)

        result = ui_state.get("result", "PENDING")
        if result in {"PASS", "FAIL", "UNKNOWN"}:
            self._draw_result_banner(output, result)

        return output

    def _draw_roi(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        x, y, w, h = roi
        draw_rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0))
        draw_text_box(frame, "ROI", (x, max(25, y)))

    def _draw_header(self, frame: np.ndarray, ui_state: dict[str, Any]) -> None:
        lines = [
            f"Mode: {ui_state.get('mode', '-')}",
            f"Observed: {ui_state.get('observation_label', '-')}",
            f"Stable: {ui_state.get('stable_label', '-')}",
            f"Vote ratio: {ui_state.get('vote_ratio', 0.0):.2f}",
            f"Result: {ui_state.get('result', '-')}",
            f"Reason: {ui_state.get('reason', '-')}",
        ]

        fps = ui_state.get("fps")
        if fps is not None:
            lines.append(f"FPS: {fps:.2f}")

        y = 30
        for line in lines:
            draw_text_box(frame, line, (10, y))
            y += 28

    def _draw_detector_debug(self, frame: np.ndarray, ui_state: dict[str, Any]) -> None:
        debug = ui_state.get("debug", {})
        details = ui_state.get("details", {})
        roi = ui_state.get("roi")

        if roi is None:
            return

        rx, ry, _, _ = roi

        if "green_bbox" in debug and debug["green_bbox"] is not None:
            x, y, w, h = debug["green_bbox"]
            draw_rectangle(frame, (rx + x, ry + y), (rx + x + w, ry + y + h), (0, 255, 0))

        if "red_bbox" in debug and debug["red_bbox"] is not None:
            x, y, w, h = debug["red_bbox"]
            draw_rectangle(frame, (rx + x, ry + y), (rx + x + w, ry + y + h), (0, 0, 255))

        if "led_boxes" in debug:
            states = details.get("led_states", [])
            for i, box in enumerate(debug["led_boxes"]):
                x, y, w, h = box
                draw_rectangle(frame, (rx + x, ry + y), (rx + x + w, ry + y + h), (200, 200, 0))
                label = states[i] if i < len(states) else "?"
                draw_text_box(frame, label, (rx + x + 2, ry + y + 20))

        if "ocr_boxes" in debug:
            for box in debug["ocr_boxes"]:
                try:
                    pts = np.array(box, dtype=np.int32)
                    pts[:, 0] += rx
                    pts[:, 1] += ry
                    cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
                except Exception:
                    pass

        extra_y = frame.shape[0] - 90
        if details:
            for key, value in list(details.items())[:3]:
                draw_text_box(frame, f"{key}: {value}", (10, extra_y))
                extra_y += 28

    def _draw_result_banner(self, frame: np.ndarray, result: str) -> None:
        h, w = frame.shape[:2]
        color_map = {
            "PASS": (0, 180, 0),
            "FAIL": (0, 0, 220),
            "UNKNOWN": (140, 140, 140),
        }
        color = color_map.get(result, (50, 50, 50))

        cv2.rectangle(frame, (0, h - RESULT_BANNER_HEIGHT), (w, h), color, -1)
        cv2.putText(
            frame,
            result,
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )