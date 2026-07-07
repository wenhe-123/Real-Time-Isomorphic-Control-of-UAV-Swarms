"""Capture and MediaPipe: Orbbec frame unpack, safe get_capture, VIDEO-mode detection, webcam BGR normalization."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import mediapipe as mp


def safe_get_capture(k4a, warn_prefix: str = "get_capture"):
    """Call ``k4a.get_capture()`` and return None on failure.

    Args:
        k4a: Orbbec/K4A device wrapper.
        warn_prefix: Prefix for warning log messages.

    Returns:
        Capture object from the device, or ``None`` on exception.
    """
    try:
        return k4a.get_capture()
    except Exception as exc:
        print(f"[WARN] {warn_prefix} failed: {exc}")
        return None


def capture_orbbec_frame(capture) -> Optional[Tuple]:
    """Unpack color and depth from an Orbbec capture.

    Args:
        capture: Device capture object with ``color`` and ``depth`` fields.

    Returns:
        Tuple ``(frame_bgr, depth_raw, capture)``, or ``None`` when color is missing.
    """
    if capture is None or capture.color is None:
        return None
    color = capture.color
    if color.ndim == 3 and color.shape[2] == 4:
        frame = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    else:
        frame = color
    return frame, capture.depth, capture


def get_aligned_depth(capture, frame, enabled: bool):
    """Return color-aligned depth when transformed depth matches color resolution.

    Args:
        capture: Device capture object with ``transformed_depth``.
        frame: BGR color frame used for shape validation.
        enabled: If False, returns ``None`` immediately.

    Returns:
        Transformed depth array matching ``frame`` shape, or ``None``.
    """
    if not enabled:
        return None
    try:
        td = capture.transformed_depth
        if td is not None and td.size > 0 and td.shape[0] == frame.shape[0] and td.shape[1] == frame.shape[1]:
            return td
        if td is not None:
            print(f"[WARN] transformed_depth shape {td.shape} != color {frame.shape}; ignoring aligned depth")
    except Exception as exc:
        print(f"[WARN] transformed_depth failed: {exc}")
    return None


def make_mp_image_from_bgr(frame_bgr):
    """Wrap a BGR OpenCV frame as a MediaPipe SRGB ``Image``.

    Args:
        frame_bgr: BGR uint8 array.

    Returns:
        MediaPipe ``Image`` ready for ``detect_for_video``.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_for_video_safe(landmarker, mp_image, t_ms: int, warn_prefix: str = "detect_for_video"):
    """Run MediaPipe VIDEO-mode detection and return None on failure.

    Args:
        landmarker: MediaPipe hand landmarker instance.
        mp_image: MediaPipe ``Image`` input.
        t_ms: Frame timestamp in milliseconds.
        warn_prefix: Prefix for warning log messages.

    Returns:
        MediaPipe hand landmarker result, or ``None`` on exception.
    """
    try:
        return landmarker.detect_for_video(mp_image, int(t_ms))
    except Exception as exc:
        print(f"[WARN] {warn_prefix} failed: {exc}")
        return None

