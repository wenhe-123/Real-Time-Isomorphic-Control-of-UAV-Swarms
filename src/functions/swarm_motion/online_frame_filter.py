"""Target EMA, spacing, axswarm filter, open-jump handling per frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from functions.swarm_motion.axswarm_runtime import AxswarmSafetyFilter
from functions.swarm_motion.spacing_guard import closest_pair, enforce_min_separation


@dataclass
class TargetFilterResult:
    filter_src: np.ndarray
    safe_target: np.ndarray
    control_target: np.ndarray
    cmd_target: np.ndarray


def filter_online_targets(
    *,
    raw_target: np.ndarray,
    raw_target_filt: np.ndarray,
    raw_target_ema: float,
    min_separation_m: float,
    axswarm_rt: AxswarmSafetyFilter | None,
    gesture_control_enabled: bool,
    prev_gesture_control_enabled: bool,
    prev_cmd_target: np.ndarray,
    elapsed: float,
    open_out: float | None,
    open_jump_reset: float,
    prev_open_for_snap: float | None,
    spacing_audit_every: int,
    frame_idx: int,
    left_pose_runtime_armed: bool,
    left_pose_state: Any,
    swarm_workspace: Any,
    morph_targets_before_left_m: np.ndarray,
) -> tuple[TargetFilterResult, np.ndarray, float | None, bool]:
    """Return filtered targets and updated raw_target_filt / prev_open / prev_gesture flags."""
    if raw_target_ema > 0.0:
        b = raw_target_ema
        raw_target_filt = b * raw_target + (1.0 - b) * raw_target_filt
        filter_src = raw_target_filt
    else:
        filter_src = raw_target
    if axswarm_rt is not None:
        safe_target = np.asarray(filter_src, dtype=np.float32)
    else:
        safe_target = enforce_min_separation(filter_src, min_separation_m, iters=10)
    if (
        spacing_audit_every > 0
        and open_out is not None
        and float(open_out) < 0.32
        and (frame_idx % spacing_audit_every) == 0
    ):
        d_raw, pri, prj = closest_pair(raw_target)
        d_safe, _, _ = closest_pair(safe_target)
        env = float(min_separation_m)
        print(
            f"[spacing live f={frame_idx}] open={float(open_out):.2f} "
            f"pre_filter={d_raw:.3f}m pair=({pri},{prj}) "
            f"post_enforce={d_safe:.3f}m "
            f"(min_sep={env:.2f}m axswarm_env≈{env:.2f}m)"
        )
    if gesture_control_enabled and not prev_gesture_control_enabled:
        if axswarm_rt is not None:
            axswarm_rt.mark_armed(float(elapsed))
            _aw = float(axswarm_rt.arm_warmup_s)
            _ax_msg = (
                f" Axswarm MPC after {_aw:.1f}s."
                if _aw > 1e-6
                else " Axswarm safety filter active."
            )
        else:
            _ax_msg = ""
        print(f"Gesture armed.{_ax_msg}")
    prev_gesture_control_enabled = bool(gesture_control_enabled)

    control_target = safe_target
    if axswarm_rt is not None and gesture_control_enabled:
        control_target = axswarm_rt.safety_filter_targets(
            elapsed,
            filter_src,
            track_pos=np.asarray(prev_cmd_target, dtype=np.float32),
        )
    if (
        open_jump_reset > 0.0
        and gesture_control_enabled
        and open_out is not None
        and prev_open_for_snap is not None
        and abs(float(open_out) - float(prev_open_for_snap)) >= open_jump_reset
    ):
        if axswarm_rt is not None:
            axswarm_rt.enter_recover(float(elapsed))
        if (
            swarm_workspace.enabled
            and swarm_workspace.armed
            and left_pose_runtime_armed
            and not left_pose_state.is_unwinding()
        ):
            _rearm_xyz = np.asarray(prev_cmd_target, dtype=np.float64)
            swarm_workspace.arm(
                morph_targets_before_left_m,
                sim_xyz=_rearm_xyz,
                fit_contains=False,
            )
            print(
                "Swarm workspace re-armed after open jump: "
                f"{swarm_workspace.format_bounds()}"
            )
    if open_out is not None:
        prev_open_for_snap = float(open_out)

    cmd_target = np.asarray(control_target, dtype=np.float32)
    return (
        TargetFilterResult(
            filter_src=np.asarray(filter_src, dtype=np.float32),
            safe_target=np.asarray(safe_target, dtype=np.float32),
            control_target=np.asarray(control_target, dtype=np.float32),
            cmd_target=cmd_target,
        ),
        raw_target_filt,
        prev_open_for_snap,
        prev_gesture_control_enabled,
    )
