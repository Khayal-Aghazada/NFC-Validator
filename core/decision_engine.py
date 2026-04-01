from typing import Any

from config import MAX_ANALYSIS_SECONDS


class DecisionEngine:
    def __init__(self, timeout_seconds: float = MAX_ANALYSIS_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds

    def _map_label_to_result(self, mode: str, label: str) -> tuple[str, str]:
        mode = mode.upper()
        label = label.upper()

        if mode == "LED":
            if label == "GREEN":
                return "PASS", "Stable GREEN LED detected"
            if label == "RED":
                return "FAIL", "Stable RED LED detected"

        elif mode == "MULTI_LED":
            if label == "PATTERN_PASS":
                return "PASS", "Stable pass LED pattern detected"
            if label == "PATTERN_FAIL":
                return "FAIL", "Stable fail LED pattern detected"

        elif mode == "OCR":
            if label == "APPROVED":
                return "PASS", "Stable pass text detected"
            if label == "DECLINED":
                return "FAIL", "Stable fail text detected"

        return "PENDING", "No final stable result yet"

    def evaluate(self, mode: str, stable_state: dict[str, Any], elapsed_time: float) -> dict[str, str]:
        stable_label = stable_state.get("stable_label")
        is_stable = stable_state.get("is_stable", False)

        if is_stable and stable_label:
            result, reason = self._map_label_to_result(mode, stable_label)
            if result in {"PASS", "FAIL"}:
                return {"result": result, "reason": reason}

        if elapsed_time >= self.timeout_seconds:
            return {"result": "UNKNOWN", "reason": "Timeout reached without stable final signal"}

        return {"result": "PENDING", "reason": "Waiting for stable signal"}