"""Dual rotation: Orbbec depth for translation; USB webcam 2D palm for rotation (always when enabled)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from functions.dual_cam.stream_runtime_utils import detect_for_video_safe
from functions.mode_switch.dual_mode_fusion import should_poll_webcam_for_dual
from functions.swarm_motion.left_hand_swarm_pose import mp_hand_visibility_scores, palm_basis_from_mp_image_plane
from functions.dual_cam.mp_hand_utils import resolve_webcam_left_hand_index


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


def detect_webcam_hand(
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
    w_res = detect_for_video_safe(
        webcam_landmarker,
        _mp_image_from_bgr(wmp),
        w_ms,
        warn_prefix="left_hand_rotation_dual webcam",
    )
    idx = webcam_frame_idx + 1
    if w_res is None:
        return None, None, wfr, idx
    w_idx = resolve_webcam_left_hand_index(w_res, prefer_hand_idx=prefer_hand_idx)
    if w_idx is None:
        return None, w_res, wfr, idx
    B = palm_basis_from_mp_image_plane(w_res, w_idx, palm_basis=palm_basis)
    return B, w_res, wfr, idx


def _resolve_webcam_palm_basis(
    *,
    webcam_cap,
    webcam_landmarker,
    fps: float,
    webcam_frame_idx: int,
    mp_input_scale: float,
    palm_basis: str,
    prefer_hand_idx: int | None,
    prefetch_B: np.ndarray | None,
    prefetch_result: Any | None,
    prefetch_frame_bgr: np.ndarray | None,
) -> tuple[np.ndarray | None, Any | None, np.ndarray | None, int]:
    """Use prefetched webcam basis or detect live."""
    B = prefetch_B
    wres = prefetch_result
    wfr = prefetch_frame_bgr
    if B is None and webcam_landmarker is not None and webcam_cap is not None:
        B, wres, wfr, webcam_frame_idx = detect_webcam_hand(
            webcam_cap,
            webcam_landmarker,
            fps=fps,
            webcam_frame_idx=webcam_frame_idx,
            mp_input_scale=mp_input_scale,
            palm_basis=palm_basis,
            prefer_hand_idx=prefer_hand_idx,
        )
    return B, wres, wfr, webcam_frame_idx


def poll_webcam_dual_cache(
    *,
    webcam_cap,
    webcam_landmarker,
    cache: dict,
    frame_idx: int,
    stride: int,
    show_preview: bool,
    orbbec_vis_min: float | None,
    rot_vis_thresh: float,
    mode_vis_min: float,
    rotating: bool,
    dual_mode_assist: bool = False,
    dual_rot_always: bool = False,
    orbbec_thumb_vis: float | None = None,
    fps: float = 30.0,
    mp_input_scale: float = 1.0,
    palm_basis: str = "middle_thumb",
    prefer_hand_idx: int | None = None,
    webcam_frame_idx: int = 0,
) -> tuple[np.ndarray | None, Any | None, np.ndarray | None, int | None, int]:
    """Read USB webcam when dual rotation is on, preview is on, or Orbbec visibility is low."""
    if webcam_cap is None or webcam_landmarker is None:
        return None, None, None, None, webcam_frame_idx
    if not should_poll_webcam_for_dual(
        orbbec_vis_min=orbbec_vis_min,
        rot_vis_thresh=float(rot_vis_thresh),
        mode_vis_min=float(mode_vis_min),
        rotating=bool(rotating),
        show_preview=bool(show_preview),
        dual_mode_assist=bool(dual_mode_assist),
        dual_rot_always=bool(dual_rot_always),
        orbbec_thumb_vis=orbbec_thumb_vis,
    ):
        return None, None, None, None, webcam_frame_idx
    stride_n = max(1, int(stride))
    due = (int(frame_idx) % stride_n) == 0
    if not due and cache.get("res") is not None:
        return (
            cache.get("B"),
            cache.get("res"),
            cache.get("fr"),
            cache.get("idx"),
            webcam_frame_idx,
        )
    b, res, fr, webcam_frame_idx = detect_webcam_hand(
        webcam_cap,
        webcam_landmarker,
        fps=float(fps),
        webcam_frame_idx=webcam_frame_idx,
        mp_input_scale=float(mp_input_scale),
        palm_basis=str(palm_basis),
        prefer_hand_idx=prefer_hand_idx,
    )
    w_idx = None
    if res is not None:
        w_idx = resolve_webcam_left_hand_index(res, prefer_hand_idx=prefer_hand_idx)
    if res is not None:
        cache["B"] = b
        cache["res"] = res
        cache["fr"] = fr
        cache["idx"] = w_idx
    return b, res, fr, w_idx, webcam_frame_idx


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
    prefetch_result: Any | None = None,
    prefetch_frame_bgr: np.ndarray | None = None,
) -> tuple[DualRotationFrame, int]:
    """Pick palm basis for rotation; translation must still use depth wrist elsewhere."""
    del vis_thresh  # kept for API; rotation always uses webcam when enabled
    out = DualRotationFrame()
    if not enabled or orbbec_idx_l is None or orbbec_result is None:
        return out, webcam_frame_idx
    out.vis_mean, out.vis_min = mp_hand_visibility_scores(orbbec_result, orbbec_idx_l)

    B, wres, wfr, webcam_frame_idx = _resolve_webcam_palm_basis(
        webcam_cap=webcam_cap,
        webcam_landmarker=webcam_landmarker,
        fps=fps,
        webcam_frame_idx=webcam_frame_idx,
        mp_input_scale=mp_input_scale,
        palm_basis=palm_basis,
        prefer_hand_idx=orbbec_idx_l,
        prefetch_B=prefetch_B,
        prefetch_result=prefetch_result,
        prefetch_frame_bgr=prefetch_frame_bgr,
    )
    out.webcam_frame_bgr = wfr
    out.webcam_result = wres

    if do_arm:
        if B is not None:
            out.arm_ref_img = B
            out.rot_source = "webcam"
        return out, webcam_frame_idx

    if B is not None:
        out.B_rot = B
        out.rot_dbg = " rot:wcam"
        out.rot_source = "webcam"
    return out, webcam_frame_idx
