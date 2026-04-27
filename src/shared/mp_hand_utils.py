"""MediaPipe result helpers: left/right hand index, world points in mm, visibility summaries."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def _wrist_x_norm(result, hand_idx: int) -> Optional[float]:
    """Normalized wrist x in [0,1] from 2D landmarks; returns None if unavailable."""
    if not result.hand_landmarks or hand_idx >= len(result.hand_landmarks):
        return None
    hlm = result.hand_landmarks[hand_idx]
    if not hlm:
        return None
    wrist = hlm[0]
    x = getattr(wrist, "x", None)
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def hand_indices_sorted_by_image_x(result) -> List[int]:
    """Return detected hand indices sorted left→right by 2D wrist x position.

    This is often more stable than MediaPipe handedness on mirrored cameras or cross-view setups.
    """
    if not result.hand_landmarks:
        return []
    xs: List[Tuple[float, int]] = []
    for i in range(len(result.hand_landmarks)):
        x = _wrist_x_norm(result, i)
        if x is None or not np.isfinite(float(x)):
            continue
        xs.append((float(x), int(i)))
    xs.sort(key=lambda t: t[0])
    return [i for _, i in xs]

