"""Capture and MediaPipe: Orbbec frame unpack, safe get_capture, VIDEO-mode detection, webcam BGR normalization."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import mediapipe as mp


def safe_get_capture(k4a, warn_prefix: str = "get_capture"):
    try:
        return k4a.get_capture()
    except Exception as exc:
        print(f"[WARN] {warn_prefix} failed: {exc}")
        return None


def capture_orbbec_frame(capture) -> Optional[Tuple]:
    if capture is None or capture.color is None:
        return None
    color = capture.color
    if color.ndim == 3 and color.shape[2] == 4:
        frame = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    else:
        frame = color
    return frame, capture.depth, capture


def get_aligned_depth(capture, frame, enabled: bool):
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


def normalize_webcam_bgr(frame_web):
    if frame_web is None:
        return None
    if frame_web.ndim == 2:
        return cv2.cvtColor(frame_web, cv2.COLOR_GRAY2BGR)
    if frame_web.ndim == 3 and frame_web.shape[2] == 4:
        return cv2.cvtColor(frame_web, cv2.COLOR_BGRA2BGR)
    return frame_web


def make_mp_image_from_bgr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_for_video_safe(landmarker, mp_image, t_ms: int, warn_prefix: str = "detect_for_video"):
    try:
        return landmarker.detect_for_video(mp_image, int(t_ms))
    except Exception as exc:
        print(f"[WARN] {warn_prefix} failed: {exc}")
        return None

