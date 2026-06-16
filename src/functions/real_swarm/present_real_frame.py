"""Orbbec HUD + real Crazyflie setpoints (no MuJoCo)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from debug.online_control_debug import draw_drone_target_debug_hud, print_center_trace
from functions.real_swarm.executor import RealSwarmExecutor
from functions.swarm_motion.swarm_workspace_box import SwarmWorkspaceBox


def present_real_online_frame(
    *,
    frame: np.ndarray,
    real_executor: RealSwarmExecutor,
    cmd_target: np.ndarray,
    safe_target: np.ndarray,
    raw_target: np.ndarray,
    filter_src: np.ndarray,
    mode_state: Any,
    right_state: Any,
    left_pose_state: Any,
    left_pose_dbg: str,
    left_pose_runtime_armed: bool,
    axswarm_rt: Any,
    swarm_workspace: SwarmWorkspaceBox,
    gesture_control_enabled: bool,
    gesture_control_enabled_box: list,
    just_gesture_armed: bool,
    mode_raw: Any,
    open_out: float | None,
    tier_count: int,
    frame_idx: int,
    elapsed: float,
    center_trace: bool,
    center_trace_every: int,
    center_trace_prev: dict[str, float | None],
    debug_drone_targets_every: int,
    min_separation_m: float,
    led_every_n: int,
    section: Callable[[str], None] | None = None,
) -> None:
    """Draw HUD and stream ``cmd_target`` to physical drones."""

    def _sec(name: str) -> None:
        if section is not None:
            section(name)

    if debug_drone_targets_every > 0 and (frame_idx % debug_drone_targets_every) == 0:
        real_pos = real_executor.get_positions_for_debug()
        draw_drone_target_debug_hud(
            frame,
            frame_idx=frame_idx,
            mode_state=mode_state,
            right_state=right_state,
            open_out=open_out,
            min_separation_m=float(min_separation_m),
            raw_target=raw_target,
            filter_src=filter_src,
            cmd_target=cmd_target,
            sim=None,
        )
        if real_pos is not None:
            print(
                f"[debug real_pos] frame={frame_idx} "
                f"centroid_z={float(np.mean(real_pos[:, 2])):.3f}m",
                flush=True,
            )
    _sec("debug_hud")

    real_executor.track_frame(
        cmd_target,
        gesture_enabled=bool(gesture_control_enabled),
        just_armed=bool(just_gesture_armed),
        morph_mode=int(mode_state.morph_mode),
        led_every_n=int(led_every_n),
        frame_idx=int(frame_idx),
    )
    _sec("real_cmd")

    if bool(center_trace):
        print_center_trace(
            elapsed=float(elapsed),
            frame_idx=frame_idx,
            center_trace_every=center_trace_every,
            center_trace_prev=center_trace_prev,
            raw_target=raw_target,
            cmd_target=cmd_target,
            safe_target=safe_target,
            left_pose_state=left_pose_state,
            swarm_workspace=swarm_workspace,
        )

    if left_pose_state.enabled:
        if left_pose_state.is_unwinding():
            pose_hint = f" | L-move:restore{left_pose_dbg}"
        elif left_pose_runtime_armed:
            pose_hint = f" | L-move:ON{left_pose_dbg}"
        else:
            pose_hint = " | L-move:[0]"
    else:
        pose_hint = ""
    ax_hint = f" | {axswarm_rt.status_line()}" if axswarm_rt is not None else ""
    cv2.putText(
        frame,
        f"REAL {'ARMED' if gesture_control_enabled_box[0] else 'HOLD - press SPACE'} "
        f"M{mode_state.morph_mode} raw:{mode_raw} open:{open_out if open_out is not None else '-'} "
        f"tier:{tier_count if tier_count >= 0 else '-'}{pose_hint}{ax_hint}",
        (16, frame.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    _sec("overlay_hud")
