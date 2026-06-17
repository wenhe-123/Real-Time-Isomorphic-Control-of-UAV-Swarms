"""Target EMA, spacing, axswarm filter, open-jump handling per frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions.mode_switch.online_frame_gesture import GestureFrameResult
from functions.runtime.online_boot import OnlineBoot
from functions.runtime.online_runtime_config import OnlineRuntimeConfig
from functions.swarm_motion.spacing_guard import closest_pair, enforce_min_separation


@dataclass
class TargetFilterResult:
    filter_src: np.ndarray
    safe_target: np.ndarray
    control_target: np.ndarray
    cmd_target: np.ndarray


def filter_online_targets(
    *,
    boot: OnlineBoot,
    cfg: OnlineRuntimeConfig,
    gest: GestureFrameResult,
    raw_target: np.ndarray,
    morph_targets_before_left_m: np.ndarray,
    elapsed: float,
    track_pos: np.ndarray | None,
) -> tuple[TargetFilterResult, np.ndarray, float | None, bool]:
    """Return filtered targets and updated raw_target_filt / prev_open / prev_gesture flags."""
    raw_target_filt = boot.raw_target_filt
    raw_target_ema = cfg.raw_target_ema
    min_separation_m = cfg.min_separation_m
    axswarm_rt = boot.axswarm_rt
    gesture_control_enabled = boot.gesture_control_enabled
    prev_gesture_control_enabled = boot.prev_gesture_control_enabled
    prev_cmd_target = boot.prev_cmd_target
    open_out = gest.open_out
    open_jump_reset = cfg.open_jump_reset
    prev_open_for_snap = boot.prev_open_for_snap
    spacing_audit_every = cfg.spacing_audit_every
    frame_idx = boot.frame_idx
    left_pose_runtime_armed = boot.left_pose_runtime_armed
    left_pose_state = boot.left_pose_state
    swarm_workspace = boot.swarm_workspace

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
            _pos = (
                np.asarray(track_pos, dtype=np.float32)
                if track_pos is not None
                else np.asarray(prev_cmd_target, dtype=np.float32)
            )
            _vel = np.zeros((axswarm_rt.n_drones, 3), dtype=np.float32)
            axswarm_rt.sync_gesture(_pos, _vel)
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
    if axswarm_rt is not None:
        _track = (
            np.asarray(track_pos, dtype=np.float32)
            if track_pos is not None
            else np.asarray(prev_cmd_target, dtype=np.float32)
        )
        control_target = axswarm_rt.safety_filter_targets(
            elapsed,
            filter_src,
            track_pos=_track,
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
