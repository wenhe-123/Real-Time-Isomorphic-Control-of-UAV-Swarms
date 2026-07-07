"""Crop Orbbec color preview to the depth-camera field of view (display only)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from functions.display_sim.depth_fusion_utils import (
    project_depth_cam_mm_to_color_px,
    unproject_depth_pixel_to_depth_camera_mm,
)

# Femto Bolt NFOV unbinned (default ``depth_mode=2`` in online_boot).
_DEPTH_HFOV_DEG = 75.0
_DEPTH_VFOV_DEG = 65.0
# RGB 720p 16:9 (default ``color_resolution=1``).
_RGB_HFOV_DEG = 80.0
_RGB_VFOV_DEG = 51.0


def _half_tan_deg(fov_deg: float) -> float:
    return float(math.tan(math.radians(float(fov_deg)) * 0.5))


def analytical_depth_fov_crop_rect(
    color_w: int,
    color_h: int,
    *,
    depth_hfov_deg: float = _DEPTH_HFOV_DEG,
    depth_vfov_deg: float = _DEPTH_VFOV_DEG,
    rgb_hfov_deg: float = _RGB_HFOV_DEG,
    rgb_vfov_deg: float = _RGB_VFOV_DEG,
) -> tuple[int, int, int, int]:
    """Compute a center crop on the color image that approximates the depth-camera FOV.

    Uses nominal horizontal/vertical FOV ratios when calibration is unavailable.

    Args:
        color_w: Color image width in pixels.
        color_h: Color image height in pixels.
        depth_hfov_deg: Depth camera horizontal field of view in degrees.
        depth_vfov_deg: Depth camera vertical field of view in degrees.
        rgb_hfov_deg: RGB camera horizontal field of view in degrees.
        rgb_vfov_deg: RGB camera vertical field of view in degrees.

    Returns:
        Crop rectangle ``(x0, y0, x1, y1)`` in color pixel coordinates.
    """
    cw = max(int(color_w), 1)
    ch = max(int(color_h), 1)
    crop_w = int(round(cw * _half_tan_deg(depth_hfov_deg) / max(_half_tan_deg(rgb_hfov_deg), 1e-9)))
    crop_h = int(round(ch * _half_tan_deg(depth_vfov_deg) / max(_half_tan_deg(rgb_vfov_deg), 1e-9)))
    crop_w = int(np.clip(crop_w, 1, cw))
    crop_h = int(np.clip(crop_h, 1, ch))
    x0 = (cw - crop_w) // 2
    y0 = (ch - crop_h) // 2
    return x0, y0, x0 + crop_w, y0 + crop_h


def compute_depth_fov_crop_rect(
    calibration,
    *,
    color_w: int,
    color_h: int,
    depth_w: int,
    depth_h: int,
    reference_depth_mm: float = 1000.0,
) -> tuple[int, int, int, int]:
    """Map depth-image corners at ``reference_depth_mm`` into color pixels.

    Falls back to :func:`analytical_depth_fov_crop_rect` when calibration or corner mapping fails.

    Args:
        calibration: PyK4A calibration object, or ``None``.
        color_w: Color image width in pixels.
        color_h: Color image height in pixels.
        depth_w: Depth image width in pixels.
        depth_h: Depth image height in pixels.
        reference_depth_mm: Depth in mm at which depth corners are unprojected.

    Returns:
        Crop rectangle ``(x0, y0, x1, y1)`` in color pixel coordinates.
    """
    cw = max(int(color_w), 1)
    ch = max(int(color_h), 1)
    dw = max(int(depth_w), 1)
    dh = max(int(depth_h), 1)
    if calibration is None:
        return analytical_depth_fov_crop_rect(cw, ch)

    uvs: list[tuple[float, float]] = []
    for xd, yd in (
        (0.0, 0.0),
        (float(dw - 1), 0.0),
        (float(dw - 1), float(dh - 1)),
        (0.0, float(dh - 1)),
    ):
        p = unproject_depth_pixel_to_depth_camera_mm(calibration, xd, yd, float(reference_depth_mm))
        if p is None:
            continue
        uv = project_depth_cam_mm_to_color_px(calibration, p)
        if uv is None:
            continue
        uvs.append(uv)

    if len(uvs) < 4:
        return analytical_depth_fov_crop_rect(cw, ch)

    xs = [float(u) for u, _ in uvs]
    ys = [float(v) for _, v in uvs]
    x0 = int(np.clip(math.floor(min(xs)), 0, cw - 1))
    y0 = int(np.clip(math.floor(min(ys)), 0, ch - 1))
    x1 = int(np.clip(math.ceil(max(xs)) + 1, x0 + 1, cw))
    y1 = int(np.clip(math.ceil(max(ys)) + 1, y0 + 1, ch))
    return x0, y0, x1, y1


def crop_orbbec_display_frame(
    frame: np.ndarray,
    crop_rect: tuple[int, int, int, int] | None,
    *,
    draw_border: bool = True,
) -> np.ndarray:
    """Return a display copy cropped to ``crop_rect``.

    Args:
        frame: Source BGR or RGB image.
        crop_rect: Inclusive-exclusive crop ``(x0, y0, x1, y1)``, or ``None`` to return
            ``frame`` unchanged.
        draw_border: When ``True``, draw a labeled border on the cropped output.

    Returns:
        Cropped image copy, or the original ``frame`` when ``crop_rect`` is ``None``.
    """
    if crop_rect is None:
        return frame
    x0, y0, x1, y1 = (int(v) for v in crop_rect)
    h, w = frame.shape[:2]
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y1 = int(np.clip(y1, y0 + 1, h))
    out = np.asarray(frame[y0:y1, x0:x1], dtype=frame.dtype).copy()
    if draw_border and out.size > 0:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 220, 160), 1, cv2.LINE_AA)
        cv2.putText(
            out,
            "depth FOV",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 160),
            1,
            cv2.LINE_AA,
        )
    return out
