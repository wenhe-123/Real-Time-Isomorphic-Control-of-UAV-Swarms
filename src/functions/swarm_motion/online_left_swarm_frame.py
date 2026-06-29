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
from functions.runtime.online_defaults import ONLINE_DEFAULTS
from functions.runtime.online_runtime_config import OnlineWebcamState
from debug.left_swarm_pose_debug import print_left_swarm_pose_debug
from functions.swarm_motion.left_hand_swarm_pose import (
    apply_rigid_to_targets,
    build_sim_from_cam_matrices,
    mp_hand_visibility_scores,
    rotvec_to_R,
    update_left_swarm_pose,
)
from functions.swarm_motion.left_pose_tuning import LeftPoseSensorInput


@dataclass
class LeftSwarmFrameResult:
    raw_target: np.ndarray
    left_swarm_off: np.ndarray | None
    left_swarm_R: np.ndarray | None
    left_pose_dbg: str
    viz_B_rot: Any
    webcam_frame_idx: int
    armed_this_frame: bool = False


def apply_left_swarm_frame(
    *,
    boot: OnlineBoot,
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

    tuning = boot.left_pose_tuning
    left_pose_state = boot.left_pose_state
    morph_targets_before_left_m = np.asarray(morph_targets_before_left_m, dtype=np.float32)
    raw_target = np.asarray(raw_target, dtype=np.float32)
    left_swarm_off: np.ndarray | None = None
    left_swarm_R: np.ndarray | None = None
    left_pose_dbg = ""
    _viz_B_rot = None
    _armed_this_frame = False
    webcam_frame_idx = webcam.frame_idx
    if left_pose_state.enabled and (
        boot.left_pose_runtime_armed or left_pose_state.is_unwinding()
    ):
        _plane_rot_mul = (
            float(tuning.plane_rot_scale_mul)
            if boot.right_state.snap_state == "plane"
            else 1.0
        )
        _do_arm = bool(boot.left_pose_reset_req and boot.left_pose_runtime_armed)
        _armed_this_frame = bool(_do_arm)
        _arm_M_rot = None
        _arm_M_trans = None
        if boot.left_use_camera_at_arm and _do_arm:
            _arm_M_rot, _arm_M_trans = build_sim_from_cam_matrices(
                boot.left_cam_preset,
                image_y_to_world_z=float(tuning.cam_y_to_world_z),
            )
        _dual_rot, webcam_frame_idx = resolve_dual_left_rotation(
            enabled=bool(boot.left_dual_webcam_rot_eff),
            orbbec_result=cap.result,
            orbbec_idx_l=gest.idx_l,
            do_arm=_do_arm,
            palm_basis=boot.left_palm_basis,
            vis_thresh=float(boot.left_rot_webcam_vis_thresh),
            webcam_cap=webcam.cap,
            webcam_landmarker=webcam.landmarker,
            fps=float(tuning.fps),
            webcam_frame_idx=webcam_frame_idx,
            mp_input_scale=float(tuning.mp_input_scale),
            prefetch_B=gest.prefetch_B,
            prefetch_result=gest.prefetch_res,
            prefetch_frame_bgr=gest.prefetch_wfr,
        )
        _sec("left_dual_rot")
        if gest.idx_l is not None and cap.result is not None:
            _, _vis_orbbec = mp_hand_visibility_scores(cap.result, int(gest.idx_l))
            left_pose_state.last_orbbec_vis_min = float(_vis_orbbec)
        _B_rot = _dual_rot.B_rot
        _viz_B_rot = _B_rot
        _arm_ref_img = _dual_rot.arm_ref_img
        _rot_dbg = _dual_rot.rot_dbg
        left_pose_state.last_dual_rot_source = str(_dual_rot.rot_source)
        left_pose_state.last_dual_vis_min = float(_dual_rot.vis_min)
        left_pose_state.last_dual_vis_thresh = float(boot.left_rot_webcam_vis_thresh)
        if tuning.show_webcam_preview and _dual_rot.webcam_frame_bgr is not None:
            _wdisp = _dual_rot.webcam_frame_bgr.copy()
            if _dual_rot.webcam_result is not None:
                _wdisp, _ = draw_hand_webcam(_wdisp, _dual_rot.webcam_result)
            _wc = (60, 220, 80) if _dual_rot.rot_source == "depth" else (80, 200, 255)
            cv2.putText(
                _wdisp,
                f"rot={_dual_rot.rot_source} vis_min={_dual_rot.vis_min:.2f} "
                f"thresh={boot.left_rot_webcam_vis_thresh:.2f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _wc,
                2,
                cv2.LINE_AA,
            )
            _wdisp = cv2.flip(_wdisp, 1)
            cv2.imshow(ONLINE_DEFAULTS.display.wcam_preview_window, _wdisp)
        _left_pose_hold = gest.idx_l is None or (
            gest.orbbec_vis_min_now is not None
            and float(gest.orbbec_vis_min_now) < float(tuning.mode_vis_min)
        )
        if _left_pose_hold:
            off = np.asarray(left_pose_state.ema_offset, dtype=np.float64).copy()
            R_pose = rotvec_to_R(left_pose_state.ema_rotvec)
            _sec("left_pose_hold")
        else:
            off, R_pose = update_left_swarm_pose(
                left_pose_state,
                sensor=LeftPoseSensorInput(
                    pts_l_pose_mm=gest.pts_l_pose_mm,
                    palm_center_depth_mm=gest.palm_center_depth_mm,
                    palm_center_color_px=gest.palm_center_color_px,
                    palm_calib=boot.calib,
                    palm_frame_h=int(cap.frame.shape[0]),
                    palm_frame_w=int(cap.frame.shape[1]),
                    palm_depth_aligned=cap.depth_aligned,
                    palm_depth_raw=cap.depth_raw,
                    palm_depth_patch_r=int(DEPTH_MEDIAN_PATCH_RADIUS),
                    B_rot=_B_rot,
                    cam_delta_to_world=boot.left_M_rot,
                    cam_translation_to_world=boot.left_M_trans,
                    arm_sim_from_cam=_arm_M_rot,
                    arm_sim_trans_from_cam=_arm_M_trans,
                    arm_cam_preset_label=str(boot.left_cam_preset) if _arm_M_rot is not None else "",
                    ref_swarm_xyz=morph_targets_before_left_m if _do_arm else None,
                    ref_basis_image=_arm_ref_img,
                    palm_basis=boot.left_palm_basis,
                    force_reset=_do_arm,
                    plane_rot_mul=_plane_rot_mul,
                ),
                tuning=tuning,
            )
            _sec("left_pose_update")
        if _do_arm and boot.left_use_camera_at_arm and left_pose_state.frozen_cam_preset:
            print(
                "Left swarm armed: depth-camera palm-center translation + palm basis rotation; "
                f"frozen cam→sim preset={left_pose_state.frozen_cam_preset!r}."
            )
        boot.left_pose_reset_req_box[0] = False
        raw_target = apply_rigid_to_targets(
            morph_targets_before_left_m,
            off,
            R_pose,
        )
        _sec("left_apply_rigid")
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
        _dbg_every = max(1, int(tuning.pose_debug_every))
        if (
            tuning.pose_debug
            and left_pose_state.enabled
            and (boot.left_pose_runtime_armed or left_pose_state.is_unwinding())
            and left_pose_state.initialized
            and (boot.frame_idx % _dbg_every) == 0
        ):
            print_left_swarm_pose_debug(
                left_pose_state,
                frame_idx=int(boot.frame_idx),
                axis_sign=tuple(tuning.axis_sign),
                trans_scale=float(tuning.trans_scale),
            )
            _sec("left_debug_print")

    _viz_every = max(1, int(tuning.pose_frame_viz_every))
    if tuning.pose_frame_viz and left_pose_state.enabled and (boot.frame_idx % _viz_every) == 0:
        draw_left_pose_frame_overlay(
            cap.frame,
            calibration=boot.calib,
            pts_l_pose_mm=gest.pts_l_pose_mm,
            result=cap.result,
            idx_l=gest.idx_l,
            left_pose_state=left_pose_state,
            left_runtime_armed=bool(boot.left_pose_runtime_armed),
            B_rot=_viz_B_rot,
            R_pose=left_swarm_R,
            off_m=left_swarm_off,
            palm_basis=boot.left_palm_basis,
            use_depth_projection=boot.calib is not None,
            palm_center_color_px=gest.palm_center_color_px,
            mp_frame_h=gest.mph,
            mp_frame_w=gest.mpw,
            motion="rigid",
            rv_pose_world=np.asarray(left_pose_state.last_rv_pose_world, dtype=np.float64).copy()
            if boot.left_pose_runtime_armed or left_pose_state.is_unwinding()
            else None,
            rv_cmd_world=np.asarray(left_pose_state.last_rv_cmd_world, dtype=np.float64).copy()
            if boot.left_pose_runtime_armed or left_pose_state.is_unwinding()
            else None,
            pose_rotate_rad=float(tuning.axis_rot_on_rad),
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
