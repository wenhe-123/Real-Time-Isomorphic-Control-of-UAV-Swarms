"""Per-frame mode/open gesture + left pose source for online control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from functions.display_sim.orbbec_hand import DEPTH_MEDIAN_PATCH_RADIUS
from functions.dual_cam.left_hand_rotation_dual import poll_webcam_dual_cache
from functions.dual_cam.mp_hand_utils import (
    extract_landmark_visibilities,
    extract_world_points_mm_result,
    resolve_mode_open_hand_indices,
)
from functions.mode_switch.hand_constants import MCP_IDS, THUMB_TIP_ID, WRIST_ID
from functions.mode_switch.modes_runtime import process_left_mode, process_right_open
from functions.mode_switch.morph_shape_control import advance_lp_shape_p, index_mcp_tip_segment_norm
from functions.dual_cam.online_frame_capture import OrbbecCaptureFrame
from functions.runtime.online_boot import OnlineBoot
from functions.runtime.online_runtime_config import OnlineRuntimeConfig, OnlineWebcamState
from functions.swarm_motion.left_hand_swarm_pose import (
    left_hand_pose_matrix_depth_mm,
    mp_hand_visibility_scores,
    palm_center_color_px_from_landmarks,
)


@dataclass
class GestureFrameResult:
    idx_l: int | None
    idx_r: int | None
    hands_3d: list | None
    hands_3d_all: list
    open_out: float | None
    mode_raw: Any
    tier_count: int
    pts_l: Any
    pts_l_pose_mm: Any
    palm_center_depth_mm: Any
    palm_center_color_px: Any
    mph: int
    mpw: int
    orbbec_vis_min_now: float | None
    prefetch_B: Any
    prefetch_res: Any
    prefetch_wfr: Any
    webcam_frame_idx: int


def process_online_gesture_frame(
    *,
    boot: OnlineBoot,
    cfg: OnlineRuntimeConfig,
    cap: OrbbecCaptureFrame,
    webcam: OnlineWebcamState,
    section: Callable[[str], None] | None = None,
) -> GestureFrameResult:
    """Process mode, open, pose, and LP shape for one online control frame.

    Resolves hand indices, polls dual webcam cache when enabled, updates morph
    mode and open state, and computes depth pose for left-swarm control.

    Args:
        boot: Initialized ``OnlineBoot`` with mode/pose state.
        cfg: Per-run ``OnlineRuntimeConfig``.
        cap: Orbbec capture payload from ``grab_orbbec_mp_frame``.
        webcam: USB webcam state (cap, landmarker, cache).
        section: Optional profiling callback ``section(name)``.

    Returns:
        ``GestureFrameResult`` with indices, landmarks, mode/open outputs, and
        webcam prefetch data for the render step.
    """
    def _sec(name: str) -> None:
        if section is not None:
            section(name)

    frame = cap.frame
    mp_frame = cap.mp_frame
    result = cap.result
    hands_3d_all = cap.hands_3d_all
    depth_aligned = cap.depth_aligned
    depth_raw = cap.depth_raw
    frame_idx = boot.frame_idx
    webcam_frame_idx = webcam.frame_idx

    idx_l, idx_r = resolve_mode_open_hand_indices(
        result,
        swap_mp_hands=bool(boot.orbbec_swap_mp_hands),
    )
    _sec("gesture_indices")
    pts_l = (
        hands_3d_all[idx_l]
        if idx_l is not None and idx_l < len(hands_3d_all)
        else None
    )
    orbbec_vis_min_now: float | None = None
    orbbec_thumb_vis: float | None = None
    if idx_l is not None and result is not None:
        _, orbbec_vis_min_now = mp_hand_visibility_scores(result, int(idx_l))
        vis_arr = extract_landmark_visibilities(result, int(idx_l))
        if vis_arr is not None and vis_arr.size > int(THUMB_TIP_ID):
            orbbec_thumb_vis = float(vis_arr[int(THUMB_TIP_ID)])
    _sec("gesture_vis")
    _left_rotating = False
    _hold_morph_mode = False
    if (
        orbbec_vis_min_now is not None
        and float(cfg.mode_vis_min) > 0.0
        and float(orbbec_vis_min_now) < float(cfg.mode_vis_min)
    ):
        _hold_morph_mode = True
    _dual_mode_assist = bool(boot.left_dual_webcam_rot_eff)
    prefetch_B = None
    prefetch_res = None
    prefetch_wfr = None
    prefetch_widx: int | None = None
    if boot.left_dual_webcam_rot_eff and webcam.cap is not None and webcam.landmarker is not None:
        (
            prefetch_B,
            prefetch_res,
            prefetch_wfr,
            prefetch_widx,
            webcam_frame_idx,
        ) = poll_webcam_dual_cache(
            webcam_cap=webcam.cap,
            webcam_landmarker=webcam.landmarker,
            cache=webcam.rot_cache,
            frame_idx=frame_idx,
            stride=webcam.rot_stride,
            show_preview=bool(cfg.show_webcam_preview),
            orbbec_vis_min=orbbec_vis_min_now,
            rot_vis_thresh=float(boot.left_rot_webcam_vis_thresh),
            mode_vis_min=float(cfg.mode_vis_min),
            rotating=bool(_left_rotating),
            dual_mode_assist=_dual_mode_assist,
            dual_rot_always=True,
            orbbec_thumb_vis=orbbec_thumb_vis,
            fps=float(cfg.fps),
            mp_input_scale=float(cfg.mp_input_scale),
            palm_basis=boot.left_palm_basis,
            prefer_hand_idx=idx_l,
            webcam_frame_idx=webcam_frame_idx,
        )
    _sec("gesture_webcam_prefetch")
    mode_raw, tier_count = process_left_mode(
        hands_3d_all,
        idx_l,
        boot.mode_state,
        mp_result=result,
        mode_vis_min=float(cfg.mode_vis_min),
        hold_mode=_hold_morph_mode,
        debounce_frames=int(cfg.pipe.mode_debounce_frames),
        webcam_mp_result=prefetch_res,
        webcam_idx_left=prefetch_widx,
        dual_mode_assist=_dual_mode_assist,
        rotating=bool(_left_rotating),
        orbbec_thumb_vis=orbbec_thumb_vis,
    )
    _sec("gesture_mode")
    hands_3d, open_out = process_right_open(
        hands_3d_all,
        idx_r,
        boot.right_state,
        mp_result=result,
        open_vis_min=float(cfg.open_vis_min),
    )
    _sec("gesture_open")
    mph, mpw = int(mp_frame.shape[0]), int(mp_frame.shape[1])
    palm_center_depth_mm = None
    palm_center_color_px = None
    if boot.calib is not None and idx_l is not None:
        _pose_depth = left_hand_pose_matrix_depth_mm(
            result,
            idx_l,
            int(frame.shape[0]),
            int(frame.shape[1]),
            mph,
            mpw,
            calibration=boot.calib,
            depth_aligned=depth_aligned,
            depth_raw=depth_raw,
            patch_r=int(DEPTH_MEDIAN_PATCH_RADIUS),
            palm_basis=boot.left_palm_basis,
        )
        if _pose_depth is not None:
            pts_l_pose_mm, palm_center_depth_mm = _pose_depth
        else:
            pts_l_pose_mm = None
        palm_center_color_px = palm_center_color_px_from_landmarks(
            result,
            idx_l,
            int(frame.shape[0]),
            int(frame.shape[1]),
            mph,
            mpw,
        )
        _sec("gesture_depth_pose")
    else:
        pts_l_pose_mm = (
            extract_world_points_mm_result(result, idx_l) if idx_l is not None else None
        )
        _sec("gesture_mp_world_pose")
    dist_norm = (
        index_mcp_tip_segment_norm(pts_l, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)
        if pts_l is not None
        else None
    )
    advance_lp_shape_p(dist_norm, int(boot.mode_state.morph_mode), boot.lp_shape)
    _sec("gesture_lp_shape")
    return GestureFrameResult(
        idx_l=idx_l,
        idx_r=idx_r,
        hands_3d=hands_3d,
        hands_3d_all=hands_3d_all,
        open_out=open_out,
        mode_raw=mode_raw,
        tier_count=tier_count,
        pts_l=pts_l,
        pts_l_pose_mm=pts_l_pose_mm,
        palm_center_depth_mm=palm_center_depth_mm,
        palm_center_color_px=palm_center_color_px,
        mph=mph,
        mpw=mpw,
        orbbec_vis_min_now=orbbec_vis_min_now,
        prefetch_B=prefetch_B,
        prefetch_res=prefetch_res,
        prefetch_wfr=prefetch_wfr,
        webcam_frame_idx=webcam_frame_idx,
    )
