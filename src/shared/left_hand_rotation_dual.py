"""Dual rotation: Orbbec depth for translation; webcam 2D palm when MP visibility is low."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

import runtime.hand_tracking_orbbec as ob
from shared.left_hand_swarm_pose import mp_hand_visibility_scores, palm_basis_from_mp_image_plane


@dataclass
class DualRotationFrame:
    """Outputs for one frame of left-hand rotation source selection."""

    B_rot: np.ndarray | None = None
    rot_dbg: str = ""
    rot_source: str = "depth"
    vis_mean: float = 1.0
    vis_min: float = 1.0
    arm_ref_img: np.ndarray | None = None
    webcam_frame_bgr: np.ndarray | None = None
    webcam_result: Any = None


def _mp_image_from_bgr(bgr: np.ndarray, mp_module=mp) -> mp.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)


def _detect_webcam_hand(
    webcam_cap,
    webcam_landmarker,
    *,
    fps: float,
    webcam_frame_idx: int,
    mp_input_scale: float,
    palm_basis: str,
    prefer_hand_idx: int | None,
) -> tuple[np.ndarray | None, Any | None, np.ndarray | None, int]:
    """Read USB webcam, run MediaPipe, return (B_rot, result, frame_bgr, new_frame_idx)."""
    if webcam_cap is None or webcam_landmarker is None:
        return None, None, None, webcam_frame_idx
    ok_w, wfr = webcam_cap.read()
    if not ok_w or wfr is None or wfr.size == 0:
        return None, None, None, webcam_frame_idx
    wmp = wfr
    if 0.0 < mp_input_scale < 1.0:
        h0, w0 = wfr.shape[:2]
        sw = max(64, int(round(w0 * mp_input_scale)))
        sh = max(48, int(round(h0 * mp_input_scale)))
        wmp = cv2.resize(wfr, (sw, sh), interpolation=cv2.INTER_LINEAR)
    w_ms = int(webcam_frame_idx * (1000 / max(float(fps), 1.0)))
    w_res = ob.detect_for_video_safe(
        webcam_landmarker,
        _mp_image_from_bgr(wmp),
        w_ms,
        warn_prefix="left_hand_rotation_dual webcam",
    )
    idx = webcam_frame_idx + 1
    if w_res is None:
        return None, None, wfr, idx
    w_idx = ob.find_hand_index_by_side(w_res, "left")
    if w_idx is None and prefer_hand_idx is not None and prefer_hand_idx < len(w_res.hand_landmarks):
        w_idx = int(prefer_hand_idx)
    if w_idx is None:
        return None, w_res, wfr, idx
    B = palm_basis_from_mp_image_plane(w_res, w_idx, palm_basis=palm_basis)
    return B, w_res, wfr, idx


def resolve_dual_left_rotation(
    *,
    enabled: bool,
    orbbec_result,
    orbbec_idx_l: int | None,
    do_arm: bool,
    palm_basis: str,
    vis_thresh: float,
    webcam_cap,
    webcam_landmarker,
    fps: float,
    webcam_frame_idx: int,
    mp_input_scale: float,
    prefetch_B: np.ndarray | None = None,
    prefetch_result: Any = None,
    prefetch_frame_bgr: np.ndarray | None = None,
) -> tuple[DualRotationFrame, int]:
    """Pick palm basis for rotation; translation must still use depth wrist elsewhere."""
    out = DualRotationFrame()
    if not enabled or orbbec_idx_l is None or orbbec_result is None:
        return out, webcam_frame_idx
    out.vis_mean, out.vis_min = mp_hand_visibility_scores(orbbec_result, orbbec_idx_l)
    use_wcam = float(out.vis_min) < float(vis_thresh)
    if prefetch_frame_bgr is not None:
        out.webcam_frame_bgr = prefetch_frame_bgr
        out.webcam_result = prefetch_result

    if do_arm:
        if use_wcam:
            B = prefetch_B
            wres = prefetch_result
            wfr = prefetch_frame_bgr
            if B is None:
                B, wres, wfr, webcam_frame_idx = _detect_webcam_hand(
                    webcam_cap,
                    webcam_landmarker,
                    fps=fps,
                    webcam_frame_idx=webcam_frame_idx,
                    mp_input_scale=mp_input_scale,
                    palm_basis=palm_basis,
                    prefer_hand_idx=orbbec_idx_l,
                )
            out.webcam_frame_bgr = wfr
            out.webcam_result = wres
            if B is not None:
                out.arm_ref_img = B
                out.rot_source = "webcam"
        if out.arm_ref_img is None:
            out.arm_ref_img = palm_basis_from_mp_image_plane(
                orbbec_result, orbbec_idx_l, palm_basis=palm_basis
            )
            out.rot_source = "orbbec2d"
        return out, webcam_frame_idx

    if use_wcam and webcam_landmarker is not None and webcam_cap is not None:
        B = prefetch_B
        wres = prefetch_result
        wfr = prefetch_frame_bgr
        if B is None:
            B, wres, wfr, webcam_frame_idx = _detect_webcam_hand(
                webcam_cap,
                webcam_landmarker,
                fps=fps,
                webcam_frame_idx=webcam_frame_idx,
                mp_input_scale=mp_input_scale,
                palm_basis=palm_basis,
                prefer_hand_idx=orbbec_idx_l,
            )
        out.webcam_frame_bgr = wfr
        out.webcam_result = wres
        if B is not None:
            out.B_rot = B
            out.rot_dbg = " rot:wcam"
            out.rot_source = "webcam"
            return out, webcam_frame_idx
    if use_wcam:
        B = palm_basis_from_mp_image_plane(orbbec_result, orbbec_idx_l, palm_basis=palm_basis)
        if B is not None:
            out.B_rot = B
            out.rot_dbg = " rot:orbbec2d"
            out.rot_source = "orbbec2d"
    else:
        out.rot_source = "depth"
    return out, webcam_frame_idx
