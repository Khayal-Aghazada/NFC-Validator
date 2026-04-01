from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, roi_frame: np.ndarray) -> dict[str, Any]:
        """
        Returns:
        {
            "label": str,
            "confidence": float,
            "details": dict,
            "debug": dict
        }
        """
        raise NotImplementedError