"""Orbbec capture + flip + MediaPipe detect for one online control frame."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from functions.runtime.online_boot import OnlineBoot
    from functions.runtime.online_runtime_config import OnlineRuntimeConfig


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


@dataclass
class CaptureFrameInput:
    """Per-frame Orbbec capture + MediaPipe detect inputs."""

    boot: OnlineBoot
    cfg: OnlineRuntimeConfig
    landmarker: Any
    frame_idx: int
    t_ms: int
    cached_mp_result: Any
    cached_hands_3d_all: list
    ema_3d: Any
    section: Callable[[str], None] | None = None


def grab_orbbec_mp_frame(
    inp: CaptureFrameInput,
) -> tuple[OrbbecCaptureFrame | None, np.ndarray | None, bool]:
    """Return (payload, poll_frame, flip_depth_warned). poll_frame is set when MP detect fails."""
    boot = inp.boot
    cfg = inp.cfg
    section = inp.section
    orbbec_flip_depth_warned = bool(boot.orbbec_flip_depth_warned)
    capture = safe_get_capture(boot.k4a, warn_prefix="online_control get_capture")
    got = capture_orbbec_frame(capture)
    if got is None:
        return None, None, orbbec_flip_depth_warned
    frame, depth_raw, capture = got
    depth_aligned = get_aligned_depth(
        capture, frame, bool(cfg.orbbec_use_transformed_depth)
    )
    if bool(cfg.orbbec_flip_horizontal):
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
    mp_input_scale = float(cfg.mp_input_scale)
    if 0.0 < mp_input_scale < 1.0:
        h0, w0 = frame.shape[:2]
        sw = max(64, int(round(w0 * mp_input_scale)))
        sh = max(48, int(round(h0 * mp_input_scale)))
        mp_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if section is not None:
        section("capture")

    run_mp = inp.cached_mp_result is None or (inp.frame_idx % cfg.mp_detect_every) == 0
    if run_mp:
        mp_image = make_mp_image_from_bgr(mp_frame)
        result = detect_for_video_safe(
            inp.landmarker,
            mp_image,
            inp.t_ms,
            warn_prefix="online_control detect_for_video",
        )
        if result is None:
            return None, frame, orbbec_flip_depth_warned
        if section is not None:
            section("mp_detect")
        use_depth_fusion = bool(boot.use_depth_fusion)
        depth_raw_for_draw = depth_raw if use_depth_fusion else None
        depth_aligned_for_draw = depth_aligned if use_depth_fusion else None
        calib_for_draw = boot.calib if use_depth_fusion else None
        fusion_w = float(boot.pipe.depth_fusion_weight) if use_depth_fusion else 0.0
        ema_3d = inp.ema_3d
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
            draw_skeleton=bool(cfg.draw_hand_debug),
        )
        if section is not None:
            section("draw_hand_dbg" if cfg.draw_hand_debug else "hand_3d")
    else:
        result = inp.cached_mp_result
        hands_3d_all = inp.cached_hands_3d_all
        ema_3d = inp.ema_3d
        if section is not None:
            section("mp_detect")
            section("draw_hand_dbg" if cfg.draw_hand_debug else "hand_3d")

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
