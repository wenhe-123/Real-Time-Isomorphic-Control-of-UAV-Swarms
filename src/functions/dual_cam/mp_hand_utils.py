"""MediaPipe result helpers: left/right hand index, world points in mm, visibility summaries."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def find_hand_index_by_side(result, side: str) -> Optional[int]:
    """Find the hand index whose MediaPipe handedness label matches ``side``.

    Args:
        result: MediaPipe hand landmarker result.
        side: ``"left"`` or ``"right"`` (case-insensitive).

    Returns:
        Hand index, or ``None`` when no matching hand is detected.
    """
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
    """Pick the physical-left hand on USB webcam MediaPipe results.

    Handedness labels are often wrong on a side-mounted USB cam; fall back to
    ``prefer_hand_idx`` (Orbbec mode-hand index) or the sole detected hand.

    Args:
        result: MediaPipe hand landmarker result.
        prefer_hand_idx: Orbbec left-hand index to use when handedness is ambiguous.
        invert_handedness: If True, treat MediaPipe ``"Right"`` as physical left.

    Returns:
        Webcam hand index for the physical left hand, or ``None``.
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
    """Return left and right hand indices from MediaPipe handedness labels.

    Args:
        result: MediaPipe hand landmarker result.
        invert_handedness: If True, swap left/right interpretation (mirrored input).

    Returns:
        Tuple ``(idx_left, idx_right)``; either entry may be ``None``.
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

    Args:
        result: MediaPipe hand landmarker result.
        swap_mp_hands: If True, invert left/right label interpretation.

    Returns:
        Tuple ``(idx_mode, idx_open)``; either entry may be ``None``.
    """
    if not result or not getattr(result, "hand_landmarks", None):
        return None, None
    return find_left_right_indices(result, invert_handedness=bool(swap_mp_hands))


def extract_world_points_mm_result(result, hand_idx: int):
    """Extract 21 world landmarks in depth-camera millimeters.

    Args:
        result: MediaPipe hand landmarker result.
        hand_idx: Index into ``result.hand_world_landmarks``.

    Returns:
        List of 21 ``(x, y, z)`` tuples in mm, or ``None`` when unavailable.
    """
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
    """Extract 21 pseudo-mm points from 2D normalized image landmarks.

    Z comes from MediaPipe relative depth. Better for USB-webcam finger-tier mode
    classify than ``hand_world_landmarks``, which can disagree with visible 2D
    extension when depth is unavailable.

    Args:
        result: MediaPipe hand landmarker result.
        hand_idx: Index into ``result.hand_landmarks``.

    Returns:
        List of 21 ``(x, y, z)`` tuples in pseudo-mm, or ``None`` when unavailable.
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
    """Return per-joint visibility scores for one hand.

    Args:
        result: MediaPipe hand landmarker result.
        hand_idx: Index into ``result.hand_landmarks``.

    Returns:
        Array of shape ``(21,)`` with values in ``[0, 1]`` (visibility, else
        presence, else 1.0), or ``None`` when landmarks are unavailable.
    """
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


def orbbec_resolve_swap_mp_hands(*, hand_swap: str, flip_horizontal: bool, use_orbbec: bool) -> bool:
    """Decide whether to swap MediaPipe left/right for mode vs open (Orbbec only).

    Args:
        hand_swap: ``"auto"``, ``"on"``, or ``"off"``; ``"auto"`` follows flip.
        flip_horizontal: Whether the Orbbec color frame is horizontally flipped.
        use_orbbec: If False, always returns False (webcam-only path).

    Returns:
        True when left/right MediaPipe indices should be swapped.
    """
    if not use_orbbec:
        return False
    hs = str(hand_swap).strip().lower()
    if hs not in ("auto", "on", "off"):
        hs = "auto"
    if hs == "auto":
        return bool(flip_horizontal)
    return hs == "on"

