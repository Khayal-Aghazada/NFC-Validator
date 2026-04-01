import time
from typing import Any

from config import (
    SAVE_FINAL_FRAME,
    SAVE_ROI_CROP,
    SAVE_VIDEO,
    WINDOW_SIZE,
    STABLE_VOTE_THRESHOLD,
    MIN_CONSECUTIVE_FRAMES,
)
from core.decision_engine import DecisionEngine
from core.evidence_manager import EvidenceManager
from core.recorder import VideoRecorder
from core.stabilizer import Stabilizer
from detectors.led_detector import LEDDetector
from detectors.multi_led_detector import MultiLEDDetector
from detectors.ocr_detector import OCRDetector
from ui.overlay import OverlayRenderer
from utils.image_utils import crop_roi


class NFCValidator:
    def __init__(self, mode: str, roi: tuple[int, int, int, int], fps: float) -> None:
        self.mode = mode.upper()
        self.roi = roi
        self.fps = fps if fps and fps > 1 else 20.0

        self.detector = self._build_detector(self.mode)
        self.stabilizer = Stabilizer(
            window_size=WINDOW_SIZE,
            stable_threshold=STABLE_VOTE_THRESHOLD,
            min_consecutive=MIN_CONSECUTIVE_FRAMES,
        )
        self.decision_engine = DecisionEngine()
        self.evidence_manager = EvidenceManager()
        self.overlay = OverlayRenderer()

        self.run_id = self.evidence_manager.create_run_id()
        self.recorder = VideoRecorder()
        self.start_time = time.time()

        self.finalized = False
        self.final_result = "PENDING"
        self.final_reason = "Waiting for stable signal"
        self.saved_paths: dict[str, str] | None = None

    def _build_detector(self, mode: str):
        if mode == "LED":
            return LEDDetector()
        if mode == "MULTI_LED":
            return MultiLEDDetector()
        if mode == "OCR":
            return OCRDetector()
        raise ValueError(f"Unsupported mode: {mode}")

    def start_recording(self, frame_width: int, frame_height: int) -> None:
        if SAVE_VIDEO:
            self.recorder.start(frame_width, frame_height, self.fps, self.run_id)

    def process_frame(self, frame, frame_index: int) -> dict[str, Any]:
        roi_frame = crop_roi(frame, self.roi)

        observation = self.detector.detect(roi_frame)
        stable_state = self.stabilizer.update(observation)

        elapsed = time.time() - self.start_time
        decision = self.decision_engine.evaluate(self.mode, stable_state, elapsed)

        result = decision["result"]
        reason = decision["reason"]

        ui_state = {
            "mode": self.mode,
            "roi": self.roi,
            "observation_label": observation.get("label", "-"),
            "stable_label": stable_state.get("stable_label", "-"),
            "vote_ratio": stable_state.get("vote_ratio", 0.0),
            "result": result,
            "reason": reason,
            "fps": self.fps,
            "details": observation.get("details", {}),
            "debug": observation.get("debug", {}),
        }

        annotated_frame = self.overlay.draw(frame, ui_state)

        if self.recorder.is_active():
            self.recorder.write(annotated_frame)

        if not self.finalized and result in {"PASS", "FAIL", "UNKNOWN"}:
            self.finalized = True
            self.final_result = result
            self.final_reason = reason

            video_path = self.recorder.stop() if self.recorder.is_active() else None

            self.saved_paths = self.evidence_manager.save_all(
                frame=annotated_frame,
                roi_frame=roi_frame,
                mode=self.mode,
                result=result,
                reason=reason,
                roi=self.roi,
                observation=observation,
                stable_state=stable_state,
                video_path=video_path,
            )

        return {
            "annotated_frame": annotated_frame,
            "observation": observation,
            "stable_state": stable_state,
            "decision": decision,
            "finalized": self.finalized,
            "saved_paths": self.saved_paths,
        }