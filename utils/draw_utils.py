import cv2
import numpy as np

from config import FONT, FONT_SCALE, FONT_THICKNESS, LINE_THICKNESS


def draw_rectangle(frame: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, pt1, pt2, color, LINE_THICKNESS)


def draw_text_box(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x, y = org
    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 8, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x + 4, y - 4), FONT, FONT_SCALE, text_color, FONT_THICKNESS, cv2.LINE_AA)


def draw_status_label(frame: np.ndarray, text: str, org: tuple[int, int], status: str) -> None:
    color_map = {
        "PASS": (0, 180, 0),
        "FAIL": (0, 0, 220),
        "PENDING": (0, 200, 255),
        "UNKNOWN": (140, 140, 140),
    }
    bg = color_map.get(status, (50, 50, 50))
    draw_text_box(frame, text, org, text_color=(255, 255, 255), bg_color=bg)