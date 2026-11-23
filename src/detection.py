from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConeDetection:
    position: np.ndarray
    covariance: np.ndarray
    source: str
    timestamp: float
    cone_class: Optional[str] = None

    def __post_init__(self):
        self.position = np.asarray(
            self.position, dtype=float
        )

        self.covariance = np.asarray(
            self.covariance, dtype=float
        )

        if self.position.shape != (2,):
            raise ValueError("Position must contain [x, y]")

        if self.covariance.shape != (2, 2):
            raise ValueError("Covariance must be a 2x2 matrix")