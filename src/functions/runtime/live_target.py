"""Thread-safe live morph target for online control."""

from __future__ import annotations

from threading import Lock

import numpy as np

class LiveTargetState:
    """Thread-safe latest Crazyflow target from hand recognition."""

    def __init__(self, initial_target: np.ndarray):
        self._lock = Lock()
        self._target = np.asarray(initial_target, dtype=np.float32).copy()
        self.mode = 1
        self.open_alpha = 1.0

    def get(self) -> np.ndarray:
        with self._lock:
            return self._target.copy()

    def set(self, target: np.ndarray, mode: int, open_alpha: float) -> None:
        with self._lock:
            self._target = np.asarray(target, dtype=np.float32).copy()
            self.mode = int(mode)
            self.open_alpha = float(open_alpha)
