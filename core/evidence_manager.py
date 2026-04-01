from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import FRAMES_DIR, LOGS_DIR, ROIS_DIR
from utils.json_utils import ensure_dir, save_json
from utils.time_utils import get_run_id, get_timestamp_str


class EvidenceManager:
    def __init__(self) -> None:
        ensure_dir(FRAMES_DIR)
        ensure_dir(LOGS_DIR)
        ensure_dir(ROIS_DIR)

    def create_run_id(self) -> str:
        return get_run_id()

    def save_frame(self, frame: np.ndarray, run_id: str) -> Path:
        path = FRAMES_DIR / f"{run_id}.png"
        cv2.imwrite(str(path), frame)
        return path

    def save_roi(self, roi_frame: np.ndarray, run_id: str) -> Path:
        path = ROIS_DIR / f"{run_id}.png"
        cv2.imwrite(str(path), roi_frame)
        return path

    def save_log(self, data: dict[str, Any], run_id: str) -> Path:
        path = LOGS_DIR / f"{run_id}.json"
        save_json(data, path)
        return path

    def save_all(
        self,
        frame: np.ndarray,
        roi_frame: np.ndarray,
        mode: str,
        result: str,
        reason: str,
        roi: tuple[int, int, int, int],
        observation: dict[str, Any],
        stable_state: dict[str, Any],
        video_path: str | None = None,
    ) -> dict[str, str]:
        run_id = self.create_run_id()

        frame_path = self.save_frame(frame, run_id)
        roi_path = self.save_roi(roi_frame, run_id)

        log_data = {
            "run_id": run_id,
            "timestamp": get_timestamp_str(),
            "mode": mode,
            "result": result,
            "reason": reason,
            "roi": {
                "x": roi[0],
                "y": roi[1],
                "w": roi[2],
                "h": roi[3],
            },
            "observation": observation,
            "stable_state": stable_state,
            "frame_path": str(frame_path),
            "roi_path": str(roi_path),
            "video_path": video_path,
        }

        log_path = self.save_log(log_data, run_id)

        return {
            "run_id": run_id,
            "frame_path": str(frame_path),
            "roi_path": str(roi_path),
            "log_path": str(log_path),
        }