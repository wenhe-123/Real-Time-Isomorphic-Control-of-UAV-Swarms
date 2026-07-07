"""Dual rotation: Orbbec depth for translation; webcam 2D palm when MP visibility is low."""

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
    """Read USB webcam, run MediaPipe, and compute palm rotation basis.

    Args:
        webcam_cap: OpenCV ``VideoCapture`` for the USB camera.
        webcam_landmarker: MediaPipe hand landmarker in VIDEO mode.
        fps: Nominal frame rate for timestamping.
        webcam_frame_idx: Monotonic frame counter for MediaPipe timestamps.
        mp_input_scale: Downscale factor for MediaPipe input (``(0, 1)``).
        palm_basis: Palm basis variant passed to ``palm_basis_from_mp_image_plane``.
        prefer_hand_idx: Orbbec left-hand index when handedness is ambiguous.

    Returns:
        Tuple ``(B_rot, result, frame_bgr, new_frame_idx)`` where ``B_rot`` is the
        3×3 palm basis or ``None``, ``result`` is the MediaPipe output,
        ``frame_bgr`` is the full-resolution BGR frame, and ``new_frame_idx`` is
        the incremented frame counter.
    """
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
    """Poll USB webcam when rotation assist or low Orbbec visibility requires it.

    Reads on stride boundaries and caches results between polls. Skips polling
    when Orbbec visibility is good and dual assist is not needed.

    Args:
        webcam_cap: OpenCV ``VideoCapture`` for the USB camera.
        webcam_landmarker: MediaPipe hand landmarker in VIDEO mode.
        cache: Mutable dict storing ``"B"``, ``"res"``, ``"fr"``, ``"idx"`` keys.
        frame_idx: Current Orbbec frame index (controls poll stride).
        stride: Poll every N Orbbec frames.
        show_preview: Force polling when dual-view preview is shown.
        orbbec_vis_min: Minimum Orbbec joint visibility this frame.
        rot_vis_thresh: Visibility threshold to switch rotation to webcam.
        mode_vis_min: Visibility threshold for mode-classify webcam assist.
        rotating: Whether left-hand rotation is currently active.
        dual_mode_assist: Enable webcam mode fusion when Orbbec is occluded.
        dual_rot_always: Always poll webcam regardless of visibility.
        orbbec_thumb_vis: Thumb-tip visibility on Orbbec (occlusion hint).
        fps: Nominal frame rate for MediaPipe timestamps.
        mp_input_scale: Downscale factor for MediaPipe input.
        palm_basis: Palm basis variant for rotation.
        prefer_hand_idx: Orbbec left-hand index for handedness fallback.
        webcam_frame_idx: Monotonic USB frame counter.

    Returns:
        Tuple ``(B, result, frame_bgr, w_idx, new_webcam_frame_idx)``; entries
        are ``None`` when polling is skipped or detection fails.
    """
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
        # Do not feed stale webcam landmarks into per-frame mode fusion. When Orbbec
        # visibility is good, the cached webcam result only adds CPU work downstream.
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
    prefetch_result: Any = None,
    prefetch_frame_bgr: np.ndarray | None = None,
) -> tuple[DualRotationFrame, int]:
    """Select palm basis for left-hand rotation (depth vs USB webcam).

    Translation must still use depth wrist elsewhere; this only resolves rotation.

    Args:
        enabled: Whether dual-webcam rotation assist is active.
        orbbec_result: MediaPipe result from the Orbbec color frame.
        orbbec_idx_l: Left-hand index in the Orbbec result.
        do_arm: If True, populate ``arm_ref_img`` instead of ``B_rot``.
        palm_basis: Palm basis variant string.
        vis_thresh: Switch to webcam when Orbbec ``vis_min`` falls below this.
        webcam_cap: OpenCV ``VideoCapture`` for the USB camera.
        webcam_landmarker: MediaPipe hand landmarker for the USB camera.
        fps: Nominal frame rate for MediaPipe timestamps.
        webcam_frame_idx: Monotonic USB frame counter.
        mp_input_scale: Downscale factor for MediaPipe input.
        prefetch_B: Pre-fetched palm basis from ``poll_webcam_dual_cache``.
        prefetch_result: Pre-fetched MediaPipe result from cache.
        prefetch_frame_bgr: Pre-fetched USB BGR frame from cache.

    Returns:
        Tuple ``(DualRotationFrame, new_webcam_frame_idx)`` with rotation source,
        debug text, visibility scores, and optional webcam preview data.
    """
    out = DualRotationFrame()
    if not enabled or orbbec_idx_l is None or orbbec_result is None:
        return out, webcam_frame_idx
    out.vis_mean, out.vis_min = mp_hand_visibility_scores(orbbec_result, orbbec_idx_l)
    # Webcam basis is for low Orbbec visibility only — do not tie rotation to prefetch cache.
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
                B, wres, wfr, webcam_frame_idx = detect_webcam_hand(
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
        return out, webcam_frame_idx

    if use_wcam and webcam_landmarker is not None and webcam_cap is not None:
        B = prefetch_B
        wres = prefetch_result
        wfr = prefetch_frame_bgr
        if B is None:
            B, wres, wfr, webcam_frame_idx = detect_webcam_hand(
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
    return out, webcam_frame_idx
