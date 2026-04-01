from collections import Counter, deque
from typing import Any


class Stabilizer:
    def __init__(self, window_size: int, stable_threshold: float, min_consecutive: int) -> None:
        self.window_size = window_size
        self.stable_threshold = stable_threshold
        self.min_consecutive = min_consecutive
        self.history: deque[str] = deque(maxlen=window_size)

    def reset(self) -> None:
        self.history.clear()

    def _max_consecutive(self, label: str) -> int:
        best = 0
        current = 0

        for item in self.history:
            if item == label:
                current += 1
                best = max(best, current)
            else:
                current = 0

        return best

    def update(self, observation: dict[str, Any]) -> dict[str, Any]:
        label = observation.get("label", "NO_SIGNAL")
        self.history.append(label)

        if not self.history:
            return {
                "stable_label": None,
                "vote_ratio": 0.0,
                "counts": {},
                "is_stable": False,
                "history": [],
                "max_consecutive": 0,
            }

        counts = Counter(self.history)
        stable_label, count = counts.most_common(1)[0]
        vote_ratio = count / len(self.history)
        max_consecutive = self._max_consecutive(stable_label)

        is_stable = (
            vote_ratio >= self.stable_threshold
            and max_consecutive >= self.min_consecutive
        )

        return {
            "stable_label": stable_label,
            "vote_ratio": float(vote_ratio),
            "counts": dict(counts),
            "is_stable": bool(is_stable),
            "history": list(self.history),
            "max_consecutive": int(max_consecutive),
        }