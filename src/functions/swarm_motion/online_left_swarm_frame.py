"""Per-frame left-hand rigid motion + palm overlay for online control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from functions.display_sim.left_pose_frame_viz import draw_left_pose_frame_overlay
from functions.display_sim.orbbec_hand import DEPTH_MEDIAN_PATCH_RADIUS
from functions.dual_cam.dual_view_utils import draw_hand_webcam
from functions.dual_cam.left_hand_rotation_dual import resolve_dual_left_rotation
from functions.dual_cam.online_frame_capture import OrbbecCaptureFrame
from functions.mode_switch.online_frame_gesture import GestureFrameResult
from functions.runtime.online_boot import OnlineBoot
from functions.runtime.online_runtime_config import OnlineRuntimeConfig, OnlineWebcamState
from functions.swarm_motion.left_hand_swarm_pose import (
    LeftSwarmPoseState,
    apply_rigid_to_targets,
    build_sim_from_cam_matrices,
    mp_hand_visibility_scores,
    print_left_swarm_pose_debug,
    rotvec_to_R,
    swarm_base_targets,
    sync_left_swarm_pose_output,
    update_left_swarm_pose,
)
from functions.swarm_motion.swarm_workspace_box import SwarmWorkspaceBox
from functions.runtime.online_defaults import _WCAM_PREVIEW_WINDOW


@dataclass
class LeftSwarmFrameResult:
    raw_target: np.ndarray
    left_swarm_off: np.ndarray | None
    left_swarm_R: np.ndarray | None
    left_pose_dbg: str
    viz_B_rot: Any
    webcam_frame_idx: int
    armed_this_frame: bool = False


def clamp_workspace_targets(swarm_workspace: SwarmWorkspaceBox, raw_target: np.ndarray) -> np.ndarray:
    if swarm_workspace.enabled and swarm_workspace.armed:
        return swarm_workspace.clamp_targets(raw_target).astype(np.float32, copy=False)
    return raw_target


def apply_left_swarm_frame(
    *,
    boot: OnlineBoot,
    cfg: OnlineRuntimeConfig,
    cap: OrbbecCaptureFrame,
    gest: GestureFrameResult,
    raw_target: np.ndarray,
    morph_targets_before_left_m: np.ndarray,
    webcam: OnlineWebcamState,
    section: Callable[[str], None] | None = None,
) -> LeftSwarmFrameResult:
    def _sec(name: str) -> None:
        if section is not None:
            section(name)

    frame = cap.frame
    result = cap.result
    depth_aligned = cap.depth_aligned
    depth_raw = cap.depth_raw
    frame_idx = boot.frame_idx
    webcam_frame_idx = webcam.frame_idx
    idx_l = gest.idx_l
    pts_l_pose_mm = gest.pts_l_pose_mm
    palm_center_depth_mm = gest.palm_center_depth_mm
    palm_center_color_px = gest.palm_center_color_px
    mph = gest.mph
    mpw = gest.mpw
    orbbec_vis_min_now = gest.orbbec_vis_min_now
    prefetch_B = gest.prefetch_B
    prefetch_res = gest.prefetch_res
    prefetch_wfr = gest.prefetch_wfr
    left_pose_state = boot.left_pose_state
    left_pose_runtime_armed = boot.left_pose_runtime_armed
    left_pose_reset_req = boot.left_pose_reset_req
    left_pose_reset_req_box = boot.left_pose_reset_req_box
    mode_state = boot.mode_state
    right_state = boot.right_state
    swarm_workspace = boot.swarm_workspace
    prev_cmd_target = boot.prev_cmd_target
    left_use_camera_at_arm = boot.left_use_camera_at_arm
    left_cam_preset = boot.left_cam_preset
    left_cam_y_to_world_z = cfg.left_cam_y_to_world_z
    left_rot_pivot_key = boot.left_rot_pivot_key
    left_dual_webcam_rot_eff = boot.left_dual_webcam_rot_eff
    left_rot_webcam_vis_thresh = boot.left_rot_webcam_vis_thresh
    show_webcam_preview = cfg.show_webcam_preview
    webcam_cap = webcam.cap
    webcam_landmarker = webcam.landmarker
    mp_input_scale = cfg.mp_input_scale
    left_M_rot = boot.left_M_rot
    left_M_trans = boot.left_M_trans
    left_trans_scale = cfg.left_trans_scale
    left_rot_scale = cfg.left_rot_scale
    left_plane_rot_scale_mul = cfg.left_plane_rot_scale_mul
    left_trans_ema = cfg.left_trans_ema
    left_rot_ema = cfg.left_rot_ema
    left_max_offset_m = cfg.left_max_offset_m
    left_max_rot_rad = cfg.left_max_rot_rad
    left_axis_sign = cfg.left_axis_sign
    left_lost_decay = cfg.left_lost_decay
    left_rot_gate_rad = cfg.left_rot_gate_rad
    left_rot_gain = cfg.left_rot_gain
    left_rot_trans_tau_mm = cfg.left_rot_trans_tau_mm
    left_rot_world_z_scale = cfg.left_rot_world_z_scale
    left_palm_basis = boot.left_palm_basis
    left_axis_trans_deadzone_m = cfg.left_axis_trans_deadzone_m
    left_axis_rot_deadzone_rad = cfg.left_axis_rot_deadzone_rad
    left_axis_trans_on_m = cfg.left_axis_trans_on_m
    left_axis_rot_on_rad = cfg.left_axis_rot_on_rad
    left_axis_trans_rot_coupling = cfg.left_axis_trans_rot_coupling
    left_palm_depth_outlier_z_mm = cfg.left_palm_depth_outlier_z_mm
    left_palm_depth_outlier_lat_ratio = cfg.left_palm_depth_outlier_lat_ratio
    left_palm_center_depth_ema = cfg.left_palm_center_depth_ema
    left_pose_debug = cfg.left_pose_debug
    left_pose_debug_every = cfg.left_pose_debug_every
    left_pose_frame_viz = cfg.left_pose_frame_viz
    left_pose_frame_viz_every = cfg.left_pose_frame_viz_every
    left_axis_rot_on_rad_viz = cfg.left_axis_rot_on_rad
    calib = boot.calib
    mode_vis_min = cfg.mode_vis_min
    fps = cfg.fps

    morph_targets_before_left_m = np.asarray(morph_targets_before_left_m, dtype=np.float32)
    raw_target = np.asarray(raw_target, dtype=np.float32)
    left_swarm_off: np.ndarray | None = None
    left_swarm_R: np.ndarray | None = None
    left_pose_dbg = ""
    _viz_B_rot = None
    _armed_this_frame = False
    if left_pose_state.enabled and (left_pose_runtime_armed or left_pose_state.is_unwinding()):
        _plane_rot_mul = (
            float(left_plane_rot_scale_mul)
            if right_state.snap_state == "plane"
            else 1.0
        )
        _rot_gate_eff = float(left_rot_gate_rad)
        _do_arm = bool(left_pose_reset_req and left_pose_runtime_armed)
        _armed_this_frame = bool(_do_arm)
        _arm_M_rot = None
        _arm_M_trans = None
        if left_use_camera_at_arm and _do_arm:
            _arm_M_rot, _arm_M_trans = build_sim_from_cam_matrices(
                left_cam_preset,
                image_y_to_world_z=float(left_cam_y_to_world_z),
            )
        _arm_ref_drone = None
        if _do_arm:
            _arm_ref_drone = np.asarray(morph_targets_before_left_m[:, :3], dtype=np.float64)
        _dual_rot, webcam_frame_idx = resolve_dual_left_rotation(
            enabled=bool(left_dual_webcam_rot_eff),
            orbbec_result=result,
            orbbec_idx_l=idx_l,
            do_arm=_do_arm,
            palm_basis=left_palm_basis,
            vis_thresh=float(left_rot_webcam_vis_thresh),
            webcam_cap=webcam_cap,
            webcam_landmarker=webcam_landmarker,
            fps=float(fps),
            webcam_frame_idx=webcam_frame_idx,
            mp_input_scale=float(mp_input_scale),
            prefetch_B=prefetch_B,
            prefetch_result=prefetch_res,
            prefetch_frame_bgr=prefetch_wfr,
        )
        _sec("left_dual_rot")
        if idx_l is not None and result is not None:
            _, _vis_orbbec = mp_hand_visibility_scores(result, int(idx_l))
            left_pose_state.last_orbbec_vis_min = float(_vis_orbbec)
        _B_rot = _dual_rot.B_rot
        _viz_B_rot = _B_rot
        _arm_ref_img = _dual_rot.arm_ref_img
        _rot_dbg = _dual_rot.rot_dbg
        left_pose_state.last_dual_rot_source = str(_dual_rot.rot_source)
        left_pose_state.last_dual_vis_min = float(_dual_rot.vis_min)
        left_pose_state.last_dual_vis_thresh = float(left_rot_webcam_vis_thresh)
        if bool(show_webcam_preview) and _dual_rot.webcam_frame_bgr is not None:
            _wdisp = _dual_rot.webcam_frame_bgr.copy()
            if _dual_rot.webcam_result is not None:
                _wdisp, _ = draw_hand_webcam(_wdisp, _dual_rot.webcam_result)
            _wc = (60, 220, 80) if _dual_rot.rot_source == "depth" else (80, 200, 255)
            cv2.putText(
                _wdisp,
                f"rot={_dual_rot.rot_source} vis_min={_dual_rot.vis_min:.2f} "
                f"thresh={left_rot_webcam_vis_thresh:.2f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _wc,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(_WCAM_PREVIEW_WINDOW, _wdisp)
        _left_pose_hold = (
            idx_l is None
            or (
                orbbec_vis_min_now is not None
                and float(orbbec_vis_min_now) < float(mode_vis_min)
            )
        )
        if _left_pose_hold:
            off = np.asarray(left_pose_state.ema_offset, dtype=np.float64).copy()
            R_pose = rotvec_to_R(left_pose_state.ema_rotvec)
            _sec("left_pose_hold")
        else:
            off, R_pose = update_left_swarm_pose(
                pts_l_pose_mm,
                left_pose_state,
                trans_scale=float(left_trans_scale),
                rot_scale=float(left_rot_scale) * _plane_rot_mul,
                trans_ema=float(left_trans_ema),
                rot_ema=float(left_rot_ema),
                max_offset_m=float(left_max_offset_m),
                max_rot_rad=float(left_max_rot_rad),
                axis_sign=tuple(left_axis_sign),
                hand_lost_decay=float(left_lost_decay),
                force_reset=_do_arm,
                cam_delta_to_world=left_M_rot,
                cam_translation_to_world=left_M_trans,
                arm_sim_from_cam=_arm_M_rot,
                arm_sim_trans_from_cam=_arm_M_trans,
                arm_cam_preset_label=str(left_cam_preset) if _arm_M_rot is not None else "",
                ref_drone_xyz=_arm_ref_drone,
                ref_swarm_xyz=morph_targets_before_left_m if _do_arm else None,
                ref_basis_image=_arm_ref_img,
                B_rot=_B_rot,
                rot_gate_rad=float(_rot_gate_eff),
                rot_gain=float(left_rot_gain),
                rot_trans_tau_mm=float(left_rot_trans_tau_mm),
                rot_world_z_scale=float(left_rot_world_z_scale),
                palm_basis=left_palm_basis,
                trans_deadzone_m=float(left_axis_trans_deadzone_m),
                rot_deadzone_rad=float(left_axis_rot_deadzone_rad),
                trans_on_m=float(left_axis_trans_on_m),
                rot_on_rad=float(left_axis_rot_on_rad),
                trans_rot_coupling=float(left_axis_trans_rot_coupling),
                palm_center_mm=palm_center_depth_mm,
                palm_center_color_px=palm_center_color_px,
                palm_depth_outlier_z_mm=float(left_palm_depth_outlier_z_mm),
                palm_depth_outlier_lateral_ratio=float(left_palm_depth_outlier_lat_ratio),
                palm_center_depth_ema=float(left_palm_center_depth_ema),
                palm_depth_patch_r=int(DEPTH_MEDIAN_PATCH_RADIUS),
                palm_calib=calib,
                palm_frame_h=int(frame.shape[0]),
                palm_frame_w=int(frame.shape[1]),
                palm_depth_aligned=depth_aligned,
                palm_depth_raw=depth_raw,
            )
            _sec("left_pose_update")
        if _do_arm and left_use_camera_at_arm and left_pose_state.frozen_cam_preset:
            print(
                "Left swarm armed: depth-camera palm-center translation + palm basis rotation; "
                f"frozen cam→sim preset={left_pose_state.frozen_cam_preset!r}."
            )
        if (
            _do_arm
            and swarm_workspace.enabled
            and left_pose_runtime_armed
            and not left_pose_state.is_unwinding()
        ):
            _arm_xyz = np.asarray(prev_cmd_target, dtype=np.float64)
            swarm_workspace.arm(
                morph_targets_before_left_m,
                sim_xyz=_arm_xyz,
                fit_contains=False,
            )
            print(f"Swarm workspace armed: {swarm_workspace.format_bounds()}")
        if left_pose_state.is_unwinding() and swarm_workspace.armed:
            swarm_workspace.disarm()
        left_pose_reset_req_box[0] = False
        _base_targets = swarm_base_targets(left_pose_state, morph_targets_before_left_m)
        if (
            swarm_workspace.enabled
            and swarm_workspace.armed
            and left_pose_runtime_armed
            and not left_pose_state.is_unwinding()
        ):
            off, R_pose, raw_target, _box_blocked, _box_msg = (
                swarm_workspace.guard_rigid_motion(
                    _base_targets,
                    off,
                    R_pose,
                    ref_drone_xyz=left_pose_state.ref_drone_xyz,
                    pivot=left_rot_pivot_key,
                )
            )
            if _box_msg:
                print(f"[swarm workspace] {_box_msg}")
            sync_left_swarm_pose_output(left_pose_state, off, R_pose)
        else:
            raw_target = apply_rigid_to_targets(
                _base_targets,
                off,
                R_pose,
                ref_drone_xyz=left_pose_state.ref_drone_xyz,
                pivot=left_rot_pivot_key,
            )
        _sec("left_apply_rigid")
        if swarm_workspace.enabled and swarm_workspace.armed:
            raw_target = swarm_workspace.clamp_targets(raw_target).astype(
                np.float32, copy=False
            )
            _sec("left_workspace_clamp")
        left_swarm_off = np.asarray(off, dtype=np.float64).copy()
        left_swarm_R = np.asarray(R_pose, dtype=np.float64).copy()
        _dc = getattr(left_pose_state, "last_delta_cam_mm", None)
        _dh0 = getattr(left_pose_state, "last_delta_h_raw_m", None)
        _pan_mm = float(np.linalg.norm(_dc)) if _dc is not None else 0.0
        _raw_m = float(np.linalg.norm(_dh0)) if _dh0 is not None else 0.0
        left_pose_dbg = (
            f" d={float(np.linalg.norm(off)):.2f}m"
            f" pan={_pan_mm:.0f}mm raw={_raw_m:.3f}m{_rot_dbg}"
        )
        _dbg_every = max(1, int(left_pose_debug_every))
        if (
            bool(left_pose_debug)
            and left_pose_state.enabled
            and (left_pose_runtime_armed or left_pose_state.is_unwinding())
            and left_pose_state.initialized
            and (frame_idx % _dbg_every) == 0
        ):
            print_left_swarm_pose_debug(
                left_pose_state,
                frame_idx=int(frame_idx),
                axis_sign=tuple(left_axis_sign),
                trans_scale=float(left_trans_scale),
            )
            _sec("left_debug_print")

    _viz_every = max(1, int(left_pose_frame_viz_every))
    if bool(left_pose_frame_viz) and left_pose_state.enabled and (frame_idx % _viz_every) == 0:
        draw_left_pose_frame_overlay(
            frame,
            calibration=calib,
            pts_l_pose_mm=pts_l_pose_mm,
            result=result,
            idx_l=idx_l,
            left_pose_state=left_pose_state,
            left_runtime_armed=bool(left_pose_runtime_armed),
            B_rot=_viz_B_rot,
            R_pose=left_swarm_R,
            off_m=left_swarm_off,
            palm_basis=left_palm_basis,
            use_depth_projection=calib is not None,
            palm_center_color_px=palm_center_color_px,
            mp_frame_h=mph,
            mp_frame_w=mpw,
            motion="rigid",
            rv_pose_world=np.asarray(left_pose_state.last_rv_pose_world, dtype=np.float64).copy()
            if left_pose_runtime_armed or left_pose_state.is_unwinding()
            else None,
            rv_cmd_world=np.asarray(left_pose_state.last_rv_cmd_world, dtype=np.float64).copy()
            if left_pose_runtime_armed or left_pose_state.is_unwinding()
            else None,
            pose_rotate_rad=float(left_axis_rot_on_rad_viz),
        )
        _sec("left_pose_viz")
    return LeftSwarmFrameResult(
        raw_target=raw_target,
        left_swarm_off=left_swarm_off,
        left_swarm_R=left_swarm_R,
        left_pose_dbg=left_pose_dbg,
        viz_B_rot=_viz_B_rot,
        webcam_frame_idx=webcam_frame_idx,
        armed_this_frame=bool(_armed_this_frame),
    )
