"""Stdout debug for left-hand swarm rigid pose (yaml: left_pose_debug)."""

from __future__ import annotations

import numpy as np

from functions.swarm_motion.left_hand_swarm_pose import LeftSwarmPoseState, rotvec_to_R


def _vec3_s(v: np.ndarray) -> str:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return f"({a[0]:+.1f},{a[1]:+.1f},{a[2]:+.1f})"


def _format_rotvec(rv: np.ndarray) -> str:
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-7:
        return "0"
    ax = rv / ang
    return f"{np.degrees(ang):+.1f}° @({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"


def format_middle_y_sign_hud(dbg) -> str:
    """One-line +Y sign summary for Matplotlib overlay."""
    if dbg is None:
        return ""
    flip = " FLIP" if bool(getattr(dbg, "flipped", False)) else ""
    segs = getattr(dbg, "seg_axial_mm", {}) or {}
    seg_s = " ".join(f"{k}:{v:+.0f}" for k, v in segs.items())
    ax = getattr(dbg, "axial_mm", None)
    ax_s = "n/a" if ax is None else f"{float(ax):+.1f}"
    return (
        f"+Y {getattr(dbg, 'reason', '')} anchor={getattr(dbg, 'anchor_used', '')}{flip}\n"
        f"axial={ax_s}mm [{seg_s}] dot(prev)={getattr(dbg, 'dot_out_prev', None)}\n"
        f"Z wrist={getattr(dbg, 'wrist_z_mm', None)} mcp={getattr(dbg, 'mcp_z_mm', None)}"
        f" pip={getattr(dbg, 'pip_z_mm', None)} tip={getattr(dbg, 'tip_z_mm', None)}"
    )


def print_left_swarm_pose_debug(
    state: LeftSwarmPoseState,
    *,
    frame_idx: int,
    axis_sign: tuple[float, float, float],
    trans_scale: float,
) -> None:
    """Print palm center (camera mm), palm/world deltas, and pose R."""
    del trans_scale
    if not state.initialized:
        return
    ref_w = np.asarray(state.ref_wrist_mm, dtype=np.float64).reshape(3)
    ref_pc = np.asarray(state.ref_palm_center, dtype=np.float64).reshape(3)
    pc = np.asarray(state.last_palm_center_mm, dtype=np.float64).reshape(3)
    w = np.asarray(state.last_wrist_mm, dtype=np.float64).reshape(3)
    mcp = np.asarray(state.last_mcp_center_mm, dtype=np.float64).reshape(3)
    mcp_n = int(state.last_mcp_valid_count)
    dc = np.asarray(state.last_delta_cam_mm, dtype=np.float64).reshape(3)
    dc_arm = np.asarray(state.last_delta_cam_arm_mm, dtype=np.float64).reshape(3)
    dh_raw = np.asarray(state.last_delta_h_raw_m, dtype=np.float64).reshape(3)
    dh_cmd = np.asarray(state.last_delta_h_world, dtype=np.float64).reshape(3)
    off = np.asarray(state.ema_offset, dtype=np.float64).reshape(3)
    rv_pose = np.asarray(state.last_rv_pose_world, dtype=np.float64).reshape(3)
    rv_cmd = np.asarray(state.last_rv_cmd_world, dtype=np.float64).reshape(3)
    R = rotvec_to_R(state.ema_rotvec)
    ang = float(np.degrees(np.linalg.norm(state.ema_rotvec)))
    sign = np.asarray(axis_sign, dtype=np.float64).reshape(3)
    wx, wy, wz = float(dh_raw[0]), float(dh_raw[1]), float(dh_raw[2])
    trans_tag = "palm center cam→world"
    hold_s = ""
    if state.last_pose_rejected:
        hold_s = f" hold reason={state.last_reject_reason}"
    sep = "=" * 72
    print(sep, flush=True)
    print(
        f"[left-pose debug] frame={int(frame_idx)} rigid{hold_s}"
        f"  axis_sign={tuple(float(x) for x in sign)}"
        f"  motion={state.last_axis_motion}"
        f"  rot_src={state.last_rot_source}"
        f"  dual_src={state.last_dual_rot_source}"
        f" vis_min={state.last_dual_vis_min:.2f}"
        f"/th={state.last_dual_vis_thresh:.2f}"
        f"  blend(wT={state.last_trans_blend_w:.2f},wR={state.last_rot_blend_w:.2f})",
        flush=True,
    )
    wrist_delta = w - ref_w if np.all(np.isfinite(w)) and np.all(np.isfinite(ref_w)) else dc
    print(
        f"  wrist (cam mm)  arm={_vec3_s(ref_w)}  now={_vec3_s(w)}  "
        f"Δframe={_vec3_s(wrist_delta)} |Δ|={np.linalg.norm(wrist_delta):.1f}mm  "
        f"(debug only, not used for trans)",
        flush=True,
    )
    print(
        f"  palm center (cam mm)  arm={_vec3_s(ref_pc)}  now={_vec3_s(pc)}  "
        f"Δframe={_vec3_s(dc)} |Δ|={np.linalg.norm(dc):.1f}mm  "
        f"Δarm={_vec3_s(dc_arm)} |Δarm|={np.linalg.norm(dc_arm):.1f}mm  dz={dc[2]:+.1f}",
        flush=True,
    )
    print(
        f"  origin parts (cam mm) wrist={_vec3_s(w)}  roots_mean(n={mcp_n}/5)={_vec3_s(mcp)}  on_plane={_vec3_s(pc)}",
        flush=True,
    )
    if state.last_palm_center_color_px is not None:
        px = state.last_palm_center_color_px
        print(f"  origin 2D centroid (color px) u,v=({int(px[0])},{int(px[1])})", flush=True)
    if state.last_depth_outlier:
        print("  depth=hold (using previous Z)", flush=True)
    rv_c = np.asarray(state.last_rv_cam_world, dtype=np.float64).reshape(3)
    print(f"  rv_cam={_format_rotvec(rv_c)}", flush=True)
    ydbg = getattr(state, "last_middle_y_sign_debug", None)
    if ydbg is not None:
        from debug.middle_y_sign_debug import print_middle_y_sign_debug

        print_middle_y_sign_debug(ydbg, frame_idx=int(frame_idx), force=True)
    print(
        f"  trans {trans_tag} arm_mm={_vec3_s(dc_arm)}  → world m (X,Y,Z)=({wx:+.4f},{wy:+.4f},{wz:+.4f})  "
        f"cmd={_vec3_s(dh_cmd)}",
        flush=True,
    )
    print(
        f"  world cmd Δ (snap×blend) = {_vec3_s(dh_cmd)}  ema_offset T={_vec3_s(off)}  |T|={np.linalg.norm(off):.3f}m",
        flush=True,
    )
    print(
        f"  rot  rv_pose(deg·ax)={_format_rotvec(rv_pose)}  rv_cmd={_format_rotvec(rv_cmd)}  "
        f"accum_angle≈{ang:.1f}°",
        flush=True,
    )
    print(
        f"  R_pose columns Xw={_vec3_s(R[:, 0])} Yw={_vec3_s(R[:, 1])} Zw={_vec3_s(R[:, 2])}",
        flush=True,
    )
    print(sep, flush=True)
