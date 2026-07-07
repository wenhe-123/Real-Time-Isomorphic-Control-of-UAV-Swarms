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
        """Return a thread-safe copy of the latest Crazyflow target.

        Returns:
            Target positions in sim meters, shape ``(N, 3)``.
        """
        with self._lock:
            return self._target.copy()

    def set(self, target: np.ndarray, mode: int, open_alpha: float) -> None:
        """Publish a new morph target and associated mode/open state.

        Args:
            target: Crazyflow positions in sim meters, shape ``(N, 3)``.
            mode: Active morph mode index (1–5).
            open_alpha: Openness in ``[0, 1]`` (0 closed, 1 open plane).
        """
        with self._lock:
            self._target = np.asarray(target, dtype=np.float32).copy()
            self.mode = int(mode)
            self.open_alpha = float(open_alpha)
