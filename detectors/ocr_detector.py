from typing import Any

import cv2
import numpy as np

from config import OCR_LANGUAGES, OCR_MIN_TEXT_CONFIDENCE, OCR_PASS_WORDS, OCR_FAIL_WORDS
from detectors.base_detector import BaseDetector
from utils.image_utils import preprocess_for_ocr

try:
    import easyocr
except ImportError:
    easyocr = None


class OCRDetector(BaseDetector):
    def __init__(self) -> None:
        self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False) if easyocr is not None else None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.upper().strip().split())

    def _match_keywords(self, normalized_text: str) -> tuple[str, float]:
        for word in OCR_PASS_WORDS:
            if word in normalized_text:
                return word, 1.0

        for word in OCR_FAIL_WORDS:
            if word in normalized_text:
                return word, 1.0

        return "", 0.0

    def detect(self, roi_frame: np.ndarray) -> dict[str, Any]:
        if self.reader is None:
            return {
                "label": "NO_TEXT",
                "confidence": 0.0,
                "details": {
                    "raw_text": "",
                    "error": "easyocr is not installed",
                },
                "debug": {
                    "ocr_boxes": [],
                },
            }

        processed = preprocess_for_ocr(roi_frame)
        results = self.reader.readtext(processed)

        best_text = ""
        best_conf = 0.0
        boxes = []

        for item in results:
            box, text, conf = item
            norm = self._normalize_text(text)
            boxes.append(box)

            if conf >= OCR_MIN_TEXT_CONFIDENCE and conf > best_conf:
                best_text = norm
                best_conf = float(conf)

        label = "NO_TEXT"
        confidence = float(best_conf)

        matched_word, matched_conf = self._match_keywords(best_text)
        if matched_word:
            if matched_word in OCR_PASS_WORDS:
                label = "APPROVED"
            elif matched_word in OCR_FAIL_WORDS:
                label = "DECLINED"
            confidence = max(confidence, matched_conf)

        return {
            "label": label,
            "confidence": float(confidence),
            "details": {
                "raw_text": best_text,
            },
            "debug": {
                "ocr_boxes": boxes,
            },
        }