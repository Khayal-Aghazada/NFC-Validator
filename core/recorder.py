from pathlib import Path

import cv2

from config import VIDEOS_DIR
from utils.json_utils import ensure_dir


class VideoRecorder:
    def __init__(self) -> None:
        ensure_dir(VIDEOS_DIR)
        self.writer: cv2.VideoWriter | None = None
        self.output_path: Path | None = None

    def start(self, frame_width: int, frame_height: int, fps: float, run_id: str) -> None:
        self.output_path = VIDEOS_DIR / f"{run_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        safe_fps = fps if fps and fps > 1 else 20.0
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, safe_fps, (frame_width, frame_height))

    def write(self, frame) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self) -> str | None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return str(self.output_path) if self.output_path else None

    def is_active(self) -> bool:
        return self.writer is not None