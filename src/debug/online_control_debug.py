"""Debug overlays and trace printing for online_control."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from functions.mode_switch.modes_runtime import ModeState, RightHandState
from functions.swarm_motion.left_hand_swarm_pose import LeftSwarmPoseState
from functions.swarm_motion.target_debug import debug_print_drone_targets


def draw_drone_target_debug_hud(
    frame: np.ndarray,
    *,
    frame_idx: int,
    mode_state: ModeState,
    right_state: RightHandState,
    open_out: float | None,
    min_separation_m: float,
    raw_target: np.ndarray,
    filter_src: np.ndarray,
    cmd_target: np.ndarray,
    sim: Any,
) -> None:
    """Print drone target chain to stdout and draw a one-line HUD on the Orbbec frame."""
    snap = getattr(right_state, "snap_state", None)
    dbg_kw = dict(
        frame_idx=frame_idx,
        morph_mode=int(mode_state.morph_mode),
        open_v=open_out,
        min_separation_m=float(min_separation_m),
        snap_state=str(snap) if snap is not None else None,
    )
    morph_snap = np.asarray(raw_target, dtype=np.float64).copy()
    debug_print_drone_targets(morph_snap, label="morph_raw", **dbg_kw)
    pre_snap = np.asarray(filter_src, dtype=np.float64).copy()
    debug_print_drone_targets(pre_snap, label="pre_filter", compare_to=morph_snap, **dbg_kw)
    cmd_snapshot = np.asarray(cmd_target, dtype=np.float64).copy()
    d_cmd, pi_cmd, pj_cmd = debug_print_drone_targets(
        cmd_snapshot, label="cmd_target", compare_to=pre_snap, **dbg_kw
    )
    try:
        sim_pos = np.asarray(sim.data.states.pos[0], dtype=np.float64)
        debug_print_drone_targets(sim_pos, label="sim_pos", compare_to=cmd_snapshot, **dbg_kw)
    except Exception as exc:
        print(f"[debug sim_pos] frame={frame_idx} unavailable: {exc}", flush=True)
    hud = f"drone dbg min=({pi_cmd},{pj_cmd}) {d_cmd:.3f}m"
    if open_out is not None:
        hud += f" open={float(open_out):.2f}"
    if snap is not None:
        hud += f" snap={snap}"
    cv2.putText(
        frame,
        hud,
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )


def print_center_trace(
    *,
    elapsed: float,
    frame_idx: int,
    center_trace_every: int,
    center_trace_prev: dict[str, float | None],
    raw_target: np.ndarray,
    cmd_target: np.ndarray,
    safe_target: np.ndarray,
    left_pose_state: LeftSwarmPoseState,
    swarm_workspace: Any,
) -> None:
    if frame_idx % max(1, int(center_trace_every)) != 0:
        return
    raw_center = np.mean(np.asarray(raw_target, dtype=np.float64), axis=0)
    smooth_center = np.mean(np.asarray(cmd_target, dtype=np.float64), axis=0)
    safe_center = np.mean(np.asarray(safe_target, dtype=np.float64), axis=0)
    track_world = np.asarray(cmd_target, dtype=np.float64)
    swarm_track_center = np.mean(track_world[:, :3], axis=0)
    track_smooth_err = swarm_track_center - smooth_center
    hand_cam_mm = np.asarray(
        getattr(left_pose_state, "last_palm_center_mm", np.zeros(3, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(3)
    hand_world_rel = np.asarray(left_pose_state.ema_offset, dtype=np.float64).reshape(3)
    flags: list[str] = []
    if bool(getattr(left_pose_state, "last_depth_outlier", False)):
        flags.append("depth-outlier")
    pz = center_trace_prev["hand_z"]
    if pz is not None and abs(float(hand_cam_mm[2]) - float(pz)) >= 80.0:
        flags.append(f"hand_dZ={float(hand_cam_mm[2]) - float(pz):+.0f}mm")
    rz_prev = center_trace_prev["raw_z"]
    if (
        pz is not None
        and rz_prev is not None
        and abs(float(hand_cam_mm[2]) - float(pz)) >= 80.0
        and abs(float(raw_center[2]) - float(rz_prev)) >= 0.12
    ):
        flags.append("DEPTH→TARGET")
    center_trace_prev["hand_z"] = float(hand_cam_mm[2])
    center_trace_prev["raw_z"] = float(raw_center[2])
    center_trace_prev["safe_z"] = float(safe_center[2])
    center_trace_prev["smooth_z"] = float(smooth_center[2])
    ws_tag = "ws=off"
    if swarm_workspace.enabled and swarm_workspace.armed:
        ws_tag = f"ws={swarm_workspace.mode}"
        if swarm_workspace.blocked:
            ws_tag += ",blocked"
    flag_s = (" " + " ".join(flags)) if flags else ""
    print(
        "[center-trace] "
        f"t={float(elapsed):7.3f}s {ws_tag}{flag_s}\n"
        f"  hand_cam_mm=({hand_cam_mm[0]:+7.1f},{hand_cam_mm[1]:+7.1f},{hand_cam_mm[2]:+7.1f}) "
        f"hand_rel_m=({hand_world_rel[0]:+6.3f},{hand_world_rel[1]:+6.3f},{hand_world_rel[2]:+6.3f})\n"
        f"  raw_center=({raw_center[0]:+6.3f},{raw_center[1]:+6.3f},{raw_center[2]:+6.3f}) "
        f"safe_center=({safe_center[0]:+6.3f},{safe_center[1]:+6.3f},{safe_center[2]:+6.3f}) "
        f"smooth_center=({smooth_center[0]:+6.3f},{smooth_center[1]:+6.3f},{smooth_center[2]:+6.3f})\n"
        f"  track_center=({swarm_track_center[0]:+6.3f},{swarm_track_center[1]:+6.3f},{swarm_track_center[2]:+6.3f}) "
        f"track-smooth=({track_smooth_err[0]:+6.3f},{track_smooth_err[1]:+6.3f},{track_smooth_err[2]:+6.3f})"
    )
