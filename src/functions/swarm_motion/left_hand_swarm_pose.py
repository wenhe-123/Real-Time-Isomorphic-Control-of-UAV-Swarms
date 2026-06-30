"""Left-hand 6DoF rigid control for the swarm (core update loop)."""

from __future__ import annotations

import time

import numpy as np

from functions.swarm_motion.left_palm_geom import (
    filter_palm_center_depth_mm,
    palm_center_components_mm,
    palm_orthonormal_basis,
)
from functions.swarm_motion.left_pose_tuning import LeftPoseSensorInput, LeftPoseTuning
from functions.swarm_motion.left_rigid_math import (
    R_to_rotvec,
    _reject_noisy_pose_frame,
    _resolve_cam_world_mats,
    _rigid_target_from_hand,
    _smooth_rigid_pose,
    axis_locked_trans_rot_blend_weights,
    palm_cam_rotvec_from_basis_delta,
    rotvec_to_R,
    sanitize_palm_rotvec_apply,
    scale_rotation_matrix,
    sync_left_swarm_pose_output,
)
from functions.swarm_motion.left_swarm_pose_state import LeftSwarmPoseState

def hand_points_to_matrix(pts) -> np.ndarray | None:
    if pts is None:
        return None
    if isinstance(pts, np.ndarray) and pts.dtype == object:
        rows = []
        for p in pts:
            v = np.asarray(p, dtype=np.float64).ravel()
            if v.size < 3:
                return None
            rows.append(v[:3])
        h = np.stack(rows, axis=0)
    else:
        h = np.asarray(pts, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 21 or h.shape[1] < 3:
        return None
    return h


def mp_hand_visibility_scores(result, hand_idx: int) -> tuple[float, float]:
    """Return (mean, min) per-joint visibility/presence in [0,1] for a hand index."""
    from functions.dual_cam.mp_hand_utils import extract_landmark_visibilities

    vis = extract_landmark_visibilities(result, hand_idx)
    if vis is None:
        return 0.0, 0.0
    return float(np.mean(vis)), float(np.min(vis))


def _clear_frozen_cam_to_sim(state: LeftSwarmPoseState) -> None:
    state.frozen_M_rot = None
    state.frozen_M_trans = None
    state.frozen_cam_preset = ""
    state.ref_swarm_targets = None
    state.ref_basis_image = None


def _clear_track_prev(state: LeftSwarmPoseState) -> None:
    state.prev_palm_mm = None
    state.prev_rot_basis = None
    state.prev_rot_source = "depth"


def _decay_pose_on_track_loss(
    state: LeftSwarmPoseState, lost_decay: float
) -> tuple[np.ndarray, np.ndarray]:
    _clear_track_prev(state)
    ld = float(np.clip(lost_decay, 0.0, 1.0))
    if state.initialized:
        ld = 1.0
    if ld < 1.0:
        state.ema_offset *= ld
        state.ema_rotvec *= ld
    return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)


def _pose_return(state: LeftSwarmPoseState) -> tuple[np.ndarray, np.ndarray]:
    return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)


def update_left_swarm_pose(
    state: LeftSwarmPoseState,
    *,
    sensor: LeftPoseSensorInput,
    tuning: LeftPoseTuning,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid follow: direct arm-relative (offset, R) with frame reject + step caps."""
    if not state.enabled:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    rot_scale = tuning.rot_scale * float(sensor.plane_rot_mul)
    step_cap = tuning.max_rot_rad if tuning.max_rot_rad > 0.0 else np.deg2rad(28.0)
    sign = np.asarray(tuning.axis_sign, dtype=np.float64).reshape(3)

    now = time.monotonic()
    if float(state.unwind_end_t) > 0.0:
        if now >= float(state.unwind_end_t):
            state.unwind_end_t = 0.0
            state.unwind_duration = 0.0
            state.initialized = False
            state.ema_offset[:] = 0.0
            state.ema_rotvec[:] = 0.0
            _clear_track_prev(state)
            _clear_frozen_cam_to_sim(state)
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        u = float(
            np.clip(
                (now - (state.unwind_end_t - state.unwind_duration))
                / max(state.unwind_duration, 1e-9),
                0.0,
                1.0,
            )
        )
        s = u * u * (3.0 - 2.0 * u)
        a = 1.0 - s
        state.ema_offset[:] = state.unwind_off0 * a
        state.ema_rotvec[:] = state.unwind_rv0 * a
        return _pose_return(state)

    h = hand_points_to_matrix(sensor.pts_l_pose_mm)
    if h is None:
        return _decay_pose_on_track_loss(state, tuning.lost_decay)

    if sensor.force_reset:
        state.filtered_palm_mm = None
        state.last_depth_outlier = False
        state.last_depth_outlier_prev = False

    ref_b = state.ref_basis if state.initialized and not sensor.force_reset else None
    out = palm_orthonormal_basis(
        h,
        palm_basis=sensor.palm_basis,
        ref_basis=ref_b,
        palm_center_override=sensor.palm_center_depth_mm,
    )
    if sensor.palm_center_color_px is not None:
        state.last_palm_center_color_px = sensor.palm_center_color_px
    if out is None:
        return _decay_pose_on_track_loss(state, tuning.lost_decay)

    palm_center, B_depth = out
    palm_center = np.asarray(palm_center, dtype=np.float64).reshape(3)
    B_depth = np.asarray(B_depth, dtype=np.float64).reshape(3, 3)

    meas_depth_mm: float | None = None
    if sensor.palm_center_depth_mm is not None:
        pc_ext = np.asarray(sensor.palm_center_depth_mm, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(pc_ext)) and float(pc_ext[2]) > 0.0:
            meas_depth_mm = float(pc_ext[2])
    palm_center, _depth_reliable = filter_palm_center_depth_mm(
        palm_center,
        state,
        z_outlier_mm=tuning.palm_depth_outlier_z_mm,
        z_outlier_lateral_ratio=tuning.palm_depth_outlier_lat_ratio,
        ema_alpha=tuning.palm_center_depth_ema,
        color_px=sensor.palm_center_color_px,
        calibration=sensor.palm_calib,
        frame_h=int(sensor.palm_frame_h),
        frame_w=int(sensor.palm_frame_w),
        depth_aligned=sensor.palm_depth_aligned,
        depth_raw=sensor.palm_depth_raw,
        depth_patch_r=int(sensor.palm_depth_patch_r),
        measured_depth_mm=meas_depth_mm,
    )
    if not np.all(np.isfinite(palm_center)):
        return _decay_pose_on_track_loss(state, tuning.lost_decay)

    if sensor.force_reset or not state.initialized:
        if not state.reset_to_current(
            h,
            palm_basis=sensor.palm_basis,
            sim_from_cam=sensor.arm_sim_from_cam if sensor.force_reset else None,
            sim_trans_from_cam=sensor.arm_sim_trans_from_cam if sensor.force_reset else None,
            cam_preset_label=str(sensor.arm_cam_preset_label) if sensor.force_reset else "",
            ref_swarm_targets=sensor.ref_swarm_xyz if sensor.force_reset else None,
            ref_basis_image=sensor.ref_basis_image if sensor.force_reset else None,
            palm_center_override=palm_center,
            palm_pose=(palm_center, B_depth),
        ):
            return _decay_pose_on_track_loss(state, tuning.lost_decay)
        state.last_palm_center_color_px = sensor.palm_center_color_px
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    ref_pc = np.asarray(state.ref_palm_center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(ref_pc)):
        ref_pc = palm_center
    delta_cam_arm = palm_center - ref_pc
    if state.prev_palm_mm is not None:
        delta_cam = palm_center - np.asarray(state.prev_palm_mm, dtype=np.float64).reshape(3)
    else:
        delta_cam = np.zeros(3, dtype=np.float64)

    w_now, mcp_now, mcp_n = palm_center_components_mm(h)
    state.last_palm_center_mm = palm_center.copy()
    if w_now is not None:
        state.last_wrist_mm = np.asarray(w_now, dtype=np.float64).reshape(3).copy()
    if mcp_now is not None:
        state.last_mcp_center_mm = np.asarray(mcp_now, dtype=np.float64).reshape(3).copy()
    state.last_mcp_valid_count = int(mcp_n)
    state.last_delta_cam_mm = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    state.last_delta_cam_arm_mm = np.asarray(delta_cam_arm, dtype=np.float64).reshape(3).copy()

    rejected, _reject_reason = _reject_noisy_pose_frame(
        delta_cam=delta_cam,
        mcp_valid=int(mcp_n),
        depth_hold=state.last_depth_outlier,
        depth_outlier_prev=state.last_depth_outlier_prev,
    )

    Mc_rot, Mc_trans = _resolve_cam_world_mats(
        state,
        cam_delta_to_world=sensor.cam_delta_to_world,
        cam_translation_to_world=sensor.cam_translation_to_world,
    )

    ref_b_depth = np.asarray(state.ref_basis, dtype=np.float64).reshape(3, 3)
    B = B_depth
    ref_b_rot = ref_b_depth
    rot_source = "depth"
    rv_world_override = None
    if sensor.B_rot is not None and state.ref_basis_image is not None:
        B_img = np.asarray(sensor.B_rot, dtype=np.float64).reshape(3, 3)
        ref_img = np.asarray(state.ref_basis_image, dtype=np.float64).reshape(3, 3)
        if np.all(np.isfinite(B_img)) and np.all(np.isfinite(ref_img)):
            # Low Orbbec visibility: full 2D palm basis for rotation (no depth/image rotvec mix).
            rv_world_override = palm_world_rotvec_from_basis_delta(
                Mc_rot, B_img, ref_img, axis_sign=sign
            )
            B = B_img
            ref_b_rot = ref_img
            rot_source = "webcam"

    off_tgt, R_tgt, off_raw, rv_world = _rigid_target_from_hand(
        delta_cam_arm=delta_cam_arm,
        B=B,
        ref_b_rot=ref_b_rot,
        rv_world_override=rv_world_override,
        Mc_rot=Mc_rot,
        Mc_trans=Mc_trans,
        sign=sign,
        trans_scale=tuning.trans_scale,
        rot_scale=rot_scale,
        rot_gain=tuning.rot_gain,
        rot_world_z_scale=tuning.rot_world_z_scale,
        trans_deadzone_m=tuning.axis_trans_deadzone_m,
        rot_deadzone_rad=tuning.axis_rot_deadzone_rad,
    )
    rv_cam = palm_cam_rotvec_from_basis_delta(B, ref_b_rot)
    rv_apply = sanitize_palm_rotvec_apply(
        rv_world,
        prev_basis=state.prev_rot_basis if state.prev_rot_source == rot_source else None,
        B_current=B,
        Mc_rot=Mc_rot,
        delta_cam_mm=delta_cam,
        max_step_rad=step_cap,
    )
    if float(np.linalg.norm(rv_apply - rv_world)) > 1e-9:
        R_world = rotvec_to_R(rv_apply)
        zsc = tuning.rot_world_z_scale
        if zsc != 1.0 and float(np.linalg.norm(rv_apply)) >= 1e-9:
            rv_z = rv_apply.copy()
            rv_z[2] *= zsc
            R_world = rotvec_to_R(rv_z)
        R_tgt = scale_rotation_matrix(R_world, scale=rot_scale, gain=tuning.rot_gain)
        if float(np.linalg.norm(R_to_rotvec(R_tgt))) < tuning.axis_rot_deadzone_rad:
            R_tgt = np.eye(3, dtype=np.float64)

    motion, w_rot, w_trans = axis_locked_trans_rot_blend_weights(
        off_tgt,
        rv_apply,
        trans_on_m=tuning.axis_trans_on_m,
        rot_on_rad=max(tuning.axis_rot_on_rad, tuning.rot_gate_rad),
        rv_cam_rad=rv_cam,
        delta_cam_mm=delta_cam,
        delta_trans_mm=delta_cam_arm,
        secondary_frac=tuning.axis_trans_rot_coupling,
    )
    rv_apply_norm = float(np.linalg.norm(rv_apply))
    off_tgt_norm = float(np.linalg.norm(off_tgt))
    if rv_apply_norm >= tuning.rot_gate_rad:
        w_rot = 1.0
        if off_tgt_norm >= tuning.axis_trans_on_m:
            motion = "rigid"
    else:
        w_rot = 0.0
    if motion == "rotate" and tuning.rot_trans_tau_mm > 0.0:
        if float(np.linalg.norm(delta_cam_arm)) < tuning.rot_trans_tau_mm:
            w_trans = 0.0
    if rejected and _reject_reason == "jump":
        motion = "hold_jump"
        w_trans = 0.0
        w_rot = 0.0
        off_tgt = np.asarray(state.ema_offset, dtype=np.float64).reshape(3).copy()
        R_tgt = rotvec_to_R(state.ema_rotvec)
    else:
        off_tgt = off_tgt * w_trans
        R_tgt = scale_rotation_matrix(R_tgt, scale=w_rot, gain=1.0)
    state.last_axis_motion = motion
    state.last_rot_source = rot_source
    state.last_rot_blend_w = w_rot
    state.last_trans_blend_w = w_trans
    state.last_delta_h_raw_m = np.asarray(off_raw * w_trans, dtype=np.float64).reshape(3).copy()
    state.last_rv_cam_world = np.asarray(rv_cam, dtype=np.float64).reshape(3).copy()
    state.last_rv_pose_world = np.asarray(rv_world, dtype=np.float64).reshape(3).copy()
    state.last_delta_h_world = np.asarray(off_tgt, dtype=np.float64).reshape(3).copy()
    state.last_rv_cmd_world = np.asarray(R_to_rotvec(R_tgt), dtype=np.float64).reshape(3).copy()

    if rejected:
        state.last_pose_rejected = True
        state.last_reject_reason = _reject_reason
    else:
        state.last_pose_rejected = False
        state.last_reject_reason = ""

    off_out, R_out = _smooth_rigid_pose(
        np.asarray(state.ema_offset, dtype=np.float64).reshape(3),
        rotvec_to_R(state.ema_rotvec),
        off_tgt,
        R_tgt,
        max_step_rad=step_cap,
        max_offset_m=tuning.max_offset_m,
        max_trans_step_m=0.055,
    )
    sync_left_swarm_pose_output(state, off_out, R_out)

    state.last_depth_outlier_prev = state.last_depth_outlier
    if not rejected or _reject_reason != "jump":
        state.prev_palm_mm = palm_center.copy()
        state.prev_rot_basis = B.copy()
        state.prev_rot_source = rot_source

    return _pose_return(state)


def apply_rigid_to_targets(
    targets: np.ndarray,
    offset: np.ndarray,
    R: np.ndarray,
    *,
    pivot_ref_m: np.ndarray | None = None,
) -> np.ndarray:
    """Apply palm-centered rigid: translate by ``offset``, rotate about ``pivot_ref + offset``.

    ``pivot_ref_m`` is the formation centroid frozen at arm (sim m). With palm translation
    ``offset``, the rotation pivot tracks the palm center in world space instead of the live
    per-frame target centroid (which drifts with morph).
    """
    t = np.asarray(targets, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] < 3:
        return np.asarray(targets, dtype=np.float32)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    off = np.asarray(offset, dtype=np.float64).reshape(3)
    p = t[:, :3]
    c0 = (
        np.asarray(pivot_ref_m, dtype=np.float64).reshape(3)
        if pivot_ref_m is not None
        else np.mean(p, axis=0)
    )
    pivot = c0 + off
    p_trans = p + off.reshape(1, 3)
    out = (R @ (p_trans - pivot.reshape(1, 3)).T).T + pivot.reshape(1, 3)
    t2 = t.copy()
    t2[:, :3] = out
    return t2.astype(np.float32)


# Historical import paths — config / geometry / math live in sibling modules.
from functions.swarm_motion.left_pose_config import (  # noqa: E402
    DEFAULT_LEFT_PALM_BASIS,
    LEFT_CAM_PRESET_ROT,
    LEFT_PALM_BASIS_PRESETS,
    build_sim_from_cam_matrices,
    left_cam_preset_rotation,
    make_cam_translation_matrix,
    palm_basis_pair_indices,
)
from functions.swarm_motion.left_palm_geom import (  # noqa: E402
    left_hand_pose_matrix_depth_mm,
    palm_basis_from_mp_image_plane,
    palm_center_color_px_from_landmarks,
    palm_frame_origin_mm,
    palm_orthonormal_basis,
    palm_plane_fit_mm,
)
from functions.swarm_motion.left_rigid_math import (  # noqa: E402
    R_to_quat,
    R_to_rotvec,
    quat_to_R,
    rotvec_to_R,
)

__all__ = [
    "DEFAULT_LEFT_PALM_BASIS",
    "LEFT_CAM_PRESET_ROT",
    "LEFT_PALM_BASIS_PRESETS",
    "LeftSwarmPoseState",
    "R_to_quat",
    "R_to_rotvec",
    "apply_rigid_to_targets",
    "build_sim_from_cam_matrices",
    "hand_points_to_matrix",
    "left_cam_preset_rotation",
    "left_hand_pose_matrix_depth_mm",
    "make_cam_translation_matrix",
    "mp_hand_visibility_scores",
    "palm_basis_from_mp_image_plane",
    "palm_basis_pair_indices",
    "palm_center_color_px_from_landmarks",
    "palm_frame_origin_mm",
    "palm_orthonormal_basis",
    "palm_plane_fit_mm",
    "quat_to_R",
    "rotvec_to_R",
    "update_left_swarm_pose",
]
