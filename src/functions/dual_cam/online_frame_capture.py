"""Orbbec capture + flip + MediaPipe detect for one online control frame."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from functions.display_sim.orbbec_hand import (
    DEPTH_MAX_DELTA_FROM_WRIST_MM,
    DEPTH_MEDIAN_MAX_DELTA_MM,
    DEPTH_MEDIAN_PATCH_RADIUS,
    HAND_3D_SOURCE_MP,
    HAND_FRAME_SCALED,
    POINT_EMA_ALPHA,
    draw_hand,
)
from functions.dual_cam.stream_runtime_utils import (
    capture_orbbec_frame,
    detect_for_video_safe,
    get_aligned_depth,
    make_mp_image_from_bgr,
    safe_get_capture,
)


@dataclass
class OrbbecCaptureFrame:
    frame: np.ndarray
    depth_raw: Any
    depth_aligned: Any
    capture: Any
    mp_frame: np.ndarray
    result: Any
    hands_3d_all: list
    ema_3d: Any


def grab_orbbec_mp_frame(
    *,
    k4a: Any,
    landmarker: Any,
    frame_idx: int,
    t_ms: int,
    fps: float,
    mp_detect_every: int,
    mp_input_scale: float,
    orbbec_flip_horizontal: bool,
    orbbec_use_transformed_depth: bool,
    use_depth_fusion: bool,
    pipe: Any,
    calib: Any,
    draw_hand_debug: bool,
    cached_mp_result: Any,
    cached_hands_3d_all: list,
    ema_3d: Any,
    orbbec_flip_depth_warned: bool,
    section: Callable[[str], None] | None = None,
) -> tuple[OrbbecCaptureFrame | None, np.ndarray | None, bool]:
    """Return (payload, poll_frame, flip_depth_warned). poll_frame is set when MP detect fails."""
    capture = safe_get_capture(k4a, warn_prefix="online_control get_capture")
    got = capture_orbbec_frame(capture)
    if got is None:
        return None, None, orbbec_flip_depth_warned
    frame, depth_raw, capture = got
    depth_aligned = get_aligned_depth(capture, frame, bool(orbbec_use_transformed_depth))
    if bool(orbbec_flip_horizontal):
        fh, fw = int(frame.shape[0]), int(frame.shape[1])
        frame = cv2.flip(frame, 1)
        if depth_aligned is not None and depth_aligned.shape[:2] == (fh, fw):
            depth_aligned = cv2.flip(depth_aligned, 1)
        elif depth_raw is not None and depth_raw.shape[:2] == (fh, fw):
            depth_raw = cv2.flip(depth_raw, 1)
        elif not orbbec_flip_depth_warned:
            print(
                "[WARN] orbbec_flip_horizontal: depth resolution != color; flipped color only. "
                "Try disabling flip or ensure transformed_depth matches color."
            )
            orbbec_flip_depth_warned = True
    mp_frame = frame
    if 0.0 < mp_input_scale < 1.0:
        h0, w0 = frame.shape[:2]
        sw = max(64, int(round(w0 * mp_input_scale)))
        sh = max(48, int(round(h0 * mp_input_scale)))
        mp_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if section is not None:
        section("capture")

    run_mp = cached_mp_result is None or (frame_idx % mp_detect_every) == 0
    if run_mp:
        mp_image = make_mp_image_from_bgr(mp_frame)
        result = detect_for_video_safe(
            landmarker,
            mp_image,
            t_ms,
            warn_prefix="online_control detect_for_video",
        )
        if result is None:
            return None, frame, orbbec_flip_depth_warned
        if section is not None:
            section("mp_detect")
        depth_raw_for_draw = depth_raw if use_depth_fusion else None
        depth_aligned_for_draw = depth_aligned if use_depth_fusion else None
        calib_for_draw = calib if use_depth_fusion else None
        fusion_w = float(pipe.depth_fusion_weight) if use_depth_fusion else 0.0
        frame, hands_3d_all, ema_3d = draw_hand(
            frame,
            result,
            depth_raw=depth_raw_for_draw,
            depth_aligned=depth_aligned_for_draw,
            print_depth=False,
            calibration=calib_for_draw,
            fusion_weight=fusion_w,
            ema_alpha=POINT_EMA_ALPHA,
            ema_points=ema_3d,
            depth_patch_radius=(DEPTH_MEDIAN_PATCH_RADIUS if use_depth_fusion else 0),
            hand_frame=HAND_FRAME_SCALED,
            filter_depth_outliers=use_depth_fusion,
            depth_max_delta_mm=DEPTH_MAX_DELTA_FROM_WRIST_MM,
            depth_median_max_delta_mm=DEPTH_MEDIAN_MAX_DELTA_MM,
            hand_3d_source=HAND_3D_SOURCE_MP,
            depth_unproject_rigid_T=None,
            draw_skeleton=bool(draw_hand_debug),
        )
        if section is not None:
            section("draw_hand_dbg" if draw_hand_debug else "hand_3d")
    else:
        result = cached_mp_result
        hands_3d_all = cached_hands_3d_all
        if section is not None:
            section("mp_detect")
            section("draw_hand_dbg" if draw_hand_debug else "hand_3d")

    return (
        OrbbecCaptureFrame(
            frame=frame,
            depth_raw=depth_raw,
            depth_aligned=depth_aligned,
            capture=capture,
            mp_frame=mp_frame,
            result=result,
            hands_3d_all=hands_3d_all,
            ema_3d=ema_3d,
        ),
        None,
        orbbec_flip_depth_warned,
    )
