"""MediaPipe result helpers: left/right hand index, world points in mm, visibility summaries."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def hand_label(result, hand_idx: int) -> str:
    if result.handedness and hand_idx < len(result.handedness):
        return result.handedness[hand_idx][0].category_name
    return "?"


def find_hand_index_by_side(result, side: str) -> Optional[int]:
    """side: 'right' or 'left' (case-insensitive)."""
    if not result.hand_landmarks or not result.handedness:
        return None
    side = side.lower()
    for idx in range(len(result.hand_landmarks)):
        if idx < len(result.handedness):
            name = result.handedness[idx][0].category_name.lower()
            if name == side:
                return idx
    return None


def resolve_webcam_left_hand_index(
    result,
    *,
    prefer_hand_idx: Optional[int] = None,
    invert_handedness: bool = False,
) -> Optional[int]:
    """Pick the physical-left hand on USB webcam MP results.

    Handedness labels are often wrong on a side-mounted USB cam; fall back to
    ``prefer_hand_idx`` (Orbbec mode-hand index) or the sole detected hand.
    """
    if not result or not getattr(result, "hand_landmarks", None):
        return None
    n = len(result.hand_landmarks)
    side = "right" if bool(invert_handedness) else "left"
    w_idx = find_hand_index_by_side(result, side)
    if w_idx is None and prefer_hand_idx is not None and int(prefer_hand_idx) < n:
        w_idx = int(prefer_hand_idx)
    if w_idx is None and n == 1:
        w_idx = 0
    return w_idx


def find_left_right_indices(result, *, invert_handedness: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """Return (idx_left, idx_right) using MediaPipe handedness labels.

    Args:
        invert_handedness: if True, swap left/right interpretation (for mirrored inputs).
    """
    if invert_handedness:
        idx_left = find_hand_index_by_side(result, "right")
        idx_right = find_hand_index_by_side(result, "left")
        return idx_left, idx_right
    idx_left = find_hand_index_by_side(result, "left")
    idx_right = find_hand_index_by_side(result, "right")
    return idx_left, idx_right


def resolve_mode_open_hand_indices(
    result,
    *,
    swap_mp_hands: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """Assign mode (MP Left) vs open (MP Right) using MediaPipe handedness only.

    One visible hand: Left → mode, Right → open (the other channel holds last value).
    ``swap_mp_hands`` inverts label interpretation (mirror / ``orbbec_hand_swap``).
    """
    if not result or not getattr(result, "hand_landmarks", None):
        return None, None
    return find_left_right_indices(result, invert_handedness=bool(swap_mp_hands))


def extract_world_points_mm_result(result, hand_idx: int):
    """21×(x,y,z) mm from MediaPipe world landmarks."""
    if not result.hand_world_landmarks or hand_idx >= len(result.hand_world_landmarks):
        return None
    wlm = result.hand_world_landmarks[hand_idx]
    pts = []
    for w in wlm:
        pts.append(
            (
                float(w.x * 1000.0),
                float(-w.y * 1000.0),
                float(-w.z * 1000.0),
            )
        )
    return pts


def extract_image_plane_points_mm_result(result, hand_idx: int):
    """21×(x,y,z) pseudo-mm from 2D normalized landmarks (z from MP relative depth).

    Better for USB-webcam finger-tier mode classify than ``hand_world_landmarks``,
    which can disagree with visible 2D extension when depth is unavailable.
    """
    if not result.hand_landmarks or hand_idx >= len(result.hand_landmarks):
        return None
    hlm = result.hand_landmarks[hand_idx]
    pts = []
    for lm in hlm:
        pts.append(
            (
                float(lm.x * 1000.0),
                float(-lm.y * 1000.0),
                float(-lm.z * 1000.0),
            )
        )
    return pts


def extract_landmark_visibilities(result, hand_idx: int = 0) -> Optional[np.ndarray]:
    """Per-joint confidence in [0,1]: visibility, else presence, else 1."""
    if not result.hand_landmarks or hand_idx >= len(result.hand_landmarks):
        return None
    hlm = result.hand_landmarks[hand_idx]
    out = np.ones(21, dtype=np.float64)
    for i, lm in enumerate(hlm):
        v = getattr(lm, "visibility", None)
        if v is None:
            v = getattr(lm, "presence", None)
        if v is not None:
            out[i] = float(np.clip(float(v), 0.0, 1.0))
    return out


def summarize_mp_visibility(vis: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """Mean and min over 21 joints for quick per-view confidence summary."""
    if vis is None or vis.size < 1:
        return None
    return {
        "mean": float(np.mean(vis)),
        "min": float(np.min(vis)),
    }


def confidence_color_bgr(mean_vis: float) -> Tuple[int, int, int]:
    """BGR color for confidence overlays."""
    if mean_vis >= 0.72:
        return (60, 220, 80)
    if mean_vis >= 0.45:
        return (80, 200, 255)
    return (100, 120, 255)


def orbbec_resolve_swap_mp_hands(*, hand_swap: str, flip_horizontal: bool, use_orbbec: bool) -> bool:
    """Whether to swap MediaPipe left/right indices for mode vs open (Orbbec path only)."""
    if not use_orbbec:
        return False
    hs = str(hand_swap).strip().lower()
    if hs not in ("auto", "on", "off"):
        hs = "auto"
    if hs == "auto":
        return bool(flip_horizontal)
    return hs == "on"

