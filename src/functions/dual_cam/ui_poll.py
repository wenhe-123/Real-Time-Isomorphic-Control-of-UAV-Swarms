"""OpenCV key polling helpers for the online control loop."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def poll_cv_key(*, cv_poll_key: Any, imshow: bool, window: str, frame: np.ndarray) -> int:
    """Poll for a keyboard key from OpenCV or an injected callback.

    Args:
        cv_poll_key: Optional callable returning a raw key code (used when not imshow).
        imshow: If True, display ``frame`` and call ``cv2.waitKey(1)``.
        window: OpenCV window name for ``imshow``.
        frame: BGR image to display when ``imshow`` is True.

    Returns:
        Key code 0–255; 255 means no key (OpenCV ``waitKey`` convention).
    """
    if imshow:
        cv2.imshow(window, frame)
        try:
            return int(cv2.waitKey(1) & 0xFF)
        except Exception:
            return 255
    if cv_poll_key is not None:
        try:
            return int(cv_poll_key() & 0xFF)
        except Exception:
            return 255
    return 255
