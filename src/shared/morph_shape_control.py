"""In-mode shape control: index MCP–tip segment (normalized by palm scale) → shape_t ∈ [0,1], smoothed (ε₁,ε₂) display."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from shared.mode_gesture_utils import palm_center_and_scale
from shared.morph_lp_plot import mode_epsilon_pair
from shared.morph_renderers import reset_lp_scatter_inertia
from shared.topology_utils import clamp01

INDEX_TIP_ID = 8
INDEX_MCP_ID = 5

# Smaller → same finger motion moves p more (more sensitive). Tuned for palm-scale norm distance.
LEFT_SHAPE_RANGE_HALF_NORM = 0.22

LEFT_SHAPE_EMA_ALPHA = 0.24

MODE_P_DISPLAY_BASE_ALPHA = 0.22
MODE_P_DISPLAY_SWITCH_ALPHA = 0.07
# Legacy reference; switch window length is ``fps * TARGET_TRANSITION_SEC`` (see ``switch_blend_frames`` on state).
MODE_P_SWITCH_BLEND_FRAMES = 60
MODE_P_MAX_STEP = 0.10
MODE_P_MAX_STEP_SWITCH = 0.045

# Wall-clock target for ε display after a mode change (pair-specific step scaling aims for ~this duration).
TARGET_TRANSITION_SEC = 2.0
DEFAULT_FPS = 30.0

# Measured transition lengths (seconds) from tuning runs; symmetric (min,max) keys.
PAIR_MEASURED_SEC: Dict[Tuple[int, int], float] = {
    (1, 2): 8.0,
    (1, 3): 8.0,
    (1, 4): 7.0,
    (1, 5): 7.0,
    (2, 3): 6.0,
    (2, 4): 5.0,
    (2, 5): 7.0,
    (3, 4): 3.0,
    (3, 5): 5.0,
    (4, 5): 6.0,
}


def pair_measured_sec(mode_a: int, mode_b: int) -> float:
    a, b = int(mode_a), int(mode_b)
    if a == b:
        return 1.0
    lo, hi = min(a, b), max(a, b)
    return float(PAIR_MEASURED_SEC.get((lo, hi), 6.0))


def target_switch_frames(fps: Optional[float]) -> int:
    f = float(fps) if fps is not None and float(fps) > 0 else DEFAULT_FPS
    return int(max(12, round(f * TARGET_TRANSITION_SEC)))


def _fallback_wrist_mcp_scale(
    hand_points: Sequence[Tuple[float, float, float]],
    wrist_id: int,
    mcp_ids: Sequence[int],
) -> Optional[float]:
    """Mean ‖wrist − MCP‖ when palm_center_and_scale fails (e.g. NaN on some palm joints)."""
    if wrist_id >= len(hand_points):
        return None
    w = np.asarray(hand_points[wrist_id], dtype=float).reshape(3)
    if np.any(np.isnan(w)):
        return None
    dists: list[float] = []
    for i in mcp_ids:
        if i >= len(hand_points):
            continue
        p = np.asarray(hand_points[i], dtype=float).reshape(3)
        if np.any(np.isnan(p)):
            continue
        dists.append(float(np.linalg.norm(p - w)))
    if not dists:
        return None
    return float(np.mean(dists)) + 1e-6


def index_mcp_tip_segment_norm(
    hand_points: Sequence[Tuple[float, float, float]],
    *,
    wrist_id: int,
    mcp_ids: Sequence[int],
) -> Optional[float]:
    """‖index_tip − index_MCP‖ / palm scale (same normalization as mode-gesture tip distances)."""
    if len(hand_points) <= max(INDEX_TIP_ID, INDEX_MCP_ID):
        return None
    p_t = np.asarray(hand_points[INDEX_TIP_ID], dtype=float).reshape(3)
    p_m = np.asarray(hand_points[INDEX_MCP_ID], dtype=float).reshape(3)
    if np.any(np.isnan(p_t)) or np.any(np.isnan(p_m)):
        return None
    seg = float(np.linalg.norm(p_t - p_m))
    _pc, scale = palm_center_and_scale(hand_points, wrist_id, mcp_ids)
    if _pc is None or float(scale) <= 0.0:
        fb = _fallback_wrist_mcp_scale(hand_points, wrist_id, mcp_ids)
        if fb is None:
            return None
        scale = fb
    return float(seg / float(scale))


@dataclass
class LpShapePipelineState:
    """Mutable state for in-mode shape_t and smoothed (ε₁,ε₂) display (Orbbec / webcam pipelines)."""

    left_shape_t_ema: Optional[float] = None
    left_ref_dist_by_mode: Dict[int, float] = field(default_factory=dict)
    mode_prev_for_ref: int = 1
    prev_morph_mode_p: int = 1
    frames_since_mode_switch: int = MODE_P_SWITCH_BLEND_FRAMES
    epsilon_pair_display: Optional[Tuple[float, float]] = None
    last_scatter_mode: int = 1
    # Last (prev → active) morph mode transition: scale ≈ measured_sec/2 so ε settles in ~TARGET_TRANSITION_SEC.
    transition_scale: float = 1.0
    switch_blend_frames: int = field(default_factory=lambda: target_switch_frames(None))


def advance_lp_shape_p(
    dist_norm: Optional[float],
    active_morph_mode: int,
    state: LpShapePipelineState,
    *,
    fps: Optional[float] = None,
) -> None:
    """Update shape_t EMA, per-mode refs, and epsilon_pair_display. Calls reset_lp_scatter_inertia on mode change."""
    active = int(active_morph_mode)
    if active != state.last_scatter_mode:
        reset_lp_scatter_inertia()
    state.last_scatter_mode = active
    prev = state.prev_morph_mode_p
    if active != prev:
        measured = pair_measured_sec(prev, active)
        # Faster pairs (small measured) need smaller steps; slow measured pairs need larger steps to hit ~2s target.
        state.transition_scale = float(measured / TARGET_TRANSITION_SEC)
        state.transition_scale = float(np.clip(state.transition_scale, 0.35, 6.0))
        state.switch_blend_frames = target_switch_frames(fps)
        state.frames_since_mode_switch = 0
    else:
        state.frames_since_mode_switch += 1
    state.prev_morph_mode_p = active

    if dist_norm is not None:
        if active != state.mode_prev_for_ref or active not in state.left_ref_dist_by_mode:
            if state.left_shape_t_ema is None:
                state.left_ref_dist_by_mode[active] = float(dist_norm)
            else:
                half = max(1e-6, float(LEFT_SHAPE_RANGE_HALF_NORM))
                ref = float(dist_norm) - 2.0 * half * (float(state.left_shape_t_ema) - 0.5)
                state.left_ref_dist_by_mode[active] = float(ref)
            state.mode_prev_for_ref = active

    left_shape_t_raw = shape_t_from_reference_distance(
        dist_norm,
        state.left_ref_dist_by_mode.get(active),
        half_range=LEFT_SHAPE_RANGE_HALF_NORM,
    )
    if left_shape_t_raw is not None:
        if state.left_shape_t_ema is None:
            state.left_shape_t_ema = float(left_shape_t_raw)
        else:
            state.left_shape_t_ema = (
                LEFT_SHAPE_EMA_ALPHA * float(left_shape_t_raw)
                + (1.0 - LEFT_SHAPE_EMA_ALPHA) * float(state.left_shape_t_ema)
            )

    e1_target, e2_target = mode_epsilon_pair(active, state.left_shape_t_ema)
    state.epsilon_pair_display = step_epsilon_pair_display(
        e1_target,
        e2_target,
        state.epsilon_pair_display,
        frames_since_mode_switch=state.frames_since_mode_switch,
        transition_scale=state.transition_scale,
        switch_blend_frames=state.switch_blend_frames,
    )


def shape_t_from_reference_distance(
    current: Optional[float],
    ref: Optional[float],
    *,
    half_range: float = LEFT_SHAPE_RANGE_HALF_NORM,
) -> Optional[float]:
    """Map (current − ref) / half to t around 0.5; clamp to [0, 1]."""
    if current is None or ref is None:
        return None
    half = max(1e-6, float(half_range))
    t = 0.5 + 0.5 * ((float(current) - float(ref)) / half)
    return float(clamp01(t))


def step_mode_p_display(
    p_target: float,
    p_prev: Optional[float],
    *,
    frames_since_mode_switch: int,
    transition_scale: float = 1.0,
    switch_blend_frames: int = MODE_P_SWITCH_BLEND_FRAMES,
) -> float:
    """EMA + slew-rate limit on one channel (ε₁ or ε₂; avoids visible jumps)."""
    if p_prev is None:
        return float(p_target)
    blend_n = max(1, int(switch_blend_frames))
    in_switch = int(frames_since_mode_switch) < blend_n
    ts = float(transition_scale)
    if in_switch:
        alpha = float(
            min(0.52, float(MODE_P_DISPLAY_SWITCH_ALPHA) * ts)
        )
        max_step = float(
            np.clip(
                float(MODE_P_MAX_STEP_SWITCH) * ts,
                0.008,
                0.55,
            )
        )
    else:
        alpha = float(MODE_P_DISPLAY_BASE_ALPHA)
        max_step = float(MODE_P_MAX_STEP)
    p_ema = float(alpha * float(p_target) + (1.0 - alpha) * float(p_prev))
    dp = float(p_ema) - float(p_prev)
    dp = float(np.clip(dp, -float(max_step), float(max_step)))
    return float(p_prev) + dp


def step_epsilon_pair_display(
    e1_target: float,
    e2_target: float,
    e_prev: Optional[Tuple[float, float]],
    *,
    frames_since_mode_switch: int,
    transition_scale: float = 1.0,
    switch_blend_frames: int = MODE_P_SWITCH_BLEND_FRAMES,
) -> Tuple[float, float]:
    if e_prev is None:
        return float(e1_target), float(e2_target)
    e1p, e2p = e_prev
    e1o = step_mode_p_display(
        e1_target,
        e1p,
        frames_since_mode_switch=frames_since_mode_switch,
        transition_scale=transition_scale,
        switch_blend_frames=switch_blend_frames,
    )
    e2o = step_mode_p_display(
        e2_target,
        e2p,
        frames_since_mode_switch=frames_since_mode_switch,
        transition_scale=transition_scale,
        switch_blend_frames=switch_blend_frames,
    )
    return e1o, e2o
