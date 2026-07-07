"""2D debug overlays: hand palm frame (depth / image plane)."""

from __future__ import annotations

import cv2
import numpy as np

from functions.display_sim.depth_fusion_utils import project_depth_cam_mm_to_color_px
from functions.swarm_motion.left_hand_swarm_pose import (
    palm_basis_from_mp_image_plane,
    palm_center_color_px_from_landmarks,
    palm_orthonormal_basis,
    palm_plane_fit_mm,
)


def _clip_uv(u: float, v: float, w: int, h: int) -> tuple[int, int] | None:
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    ui, vi = int(round(float(u))), int(round(float(v)))
    if ui < -40 or vi < -40 or ui >= w + 40 or vi >= h + 40:
        return None
    return ui, vi


def _draw_axis_arrow(
    frame: np.ndarray,
    o: tuple[int, int],
    tip: tuple[int, int],
    color: tuple[int, int, int],
    *,
    label: str = "",
    thickness: int = 2,
) -> None:
    cv2.arrowedLine(frame, o, tip, color, thickness, tipLength=0.22, line_type=cv2.LINE_AA)
    if label:
        cv2.putText(
            frame,
            label,
            (tip[0] + 4, tip[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_depth_cam_basis_on_bgr(
    frame: np.ndarray,
    calibration,
    origin_mm: np.ndarray,
    basis: np.ndarray,
    *,
    axis_len_mm: float = 55.0,
    prefix: str = "",
    thickness: int = 2,
    origin_color_px: tuple[int, int] | None = None,
    axis_labels: tuple[str, str, str] | None = None,
) -> None:
    """Draw RGB = XYZ columns of ``basis`` at ``origin_mm`` projected onto the color image.

    Args:
        frame: BGR image modified in place.
        calibration: PyK4A calibration for depth-to-color projection.
        origin_mm: Palm origin in depth-camera mm, shape ``(3,)``.
        basis: Orthonormal palm basis, shape ``(3, 3)``; columns are axis directions.
        axis_len_mm: Length of each axis arrow in millimeters.
        prefix: Optional prefix prepended to axis labels.
        thickness: OpenCV line thickness for axis arrows.
        origin_color_px: Fixed color-pixel origin ``(u, v)``; computed from ``origin_mm`` when
            ``None``.
        axis_labels: Tuple of three axis label strings.

    Returns:
        None.
    """
    if calibration is None or frame is None:
        return
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if origin_color_px is not None:
        ou = _clip_uv(float(origin_color_px[0]), float(origin_color_px[1]), w, h)
    else:
        o = project_depth_cam_mm_to_color_px(calibration, origin_mm)
        if o is None:
            return
        ou = _clip_uv(o[0], o[1], w, h)
    if ou is None:
        return
    B = np.asarray(basis, dtype=np.float64).reshape(3, 3)
    origin = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # BGR: X,Y,Z
    labels = axis_labels if axis_labels is not None else ("X", "Y", "Z")
    L = float(max(axis_len_mm, 8.0))
    for i in range(3):
        tip_mm = origin + B[:, i] * L
        t = project_depth_cam_mm_to_color_px(calibration, tip_mm)
        if t is None:
            continue
        tu = _clip_uv(t[0], t[1], w, h)
        if tu is None:
            continue
        lab = f"{prefix}{labels[i]}" if prefix else labels[i]
        _draw_axis_arrow(frame, ou, tu, colors[i], label=lab, thickness=thickness)
    cv2.circle(frame, ou, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, ou, 4, (40, 40, 40), 1, cv2.LINE_AA)


def draw_palm_plane_inliers_on_bgr(
    frame: np.ndarray,
    calibration,
    inlier_points_mm: np.ndarray,
    *,
    hull_color: tuple[int, int, int] = (200, 180, 60),
) -> None:
    """Draw the convex hull of palm-plane inlier landmarks on the color image.

    Args:
        frame: BGR image modified in place.
        calibration: PyK4A calibration for depth-to-color projection.
        inlier_points_mm: Inlier landmarks in depth-camera mm, shape ``(N, 3)``.
        hull_color: BGR color for inlier dots and hull outline.

    Returns:
        None.
    """
    if calibration is None or frame is None:
        return
    P = np.asarray(inlier_points_mm, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < 3:
        return
    uv: list[tuple[int, int]] = []
    h, w = int(frame.shape[0]), int(frame.shape[1])
    for p in P:
        q = project_depth_cam_mm_to_color_px(calibration, p)
        if q is None:
            continue
        c = _clip_uv(q[0], q[1], w, h)
        if c is not None:
            uv.append(c)
            cv2.circle(frame, c, 3, hull_color, -1, cv2.LINE_AA)
    if len(uv) >= 3:
        hull = cv2.convexHull(np.asarray(uv, dtype=np.int32))
        cv2.polylines(frame, [hull], True, hull_color, 1, cv2.LINE_AA)


_PALM_AXIS_LABELS = ("X thumb", "Y mid", "Z palm")


def draw_image_plane_basis_on_bgr(
    frame: np.ndarray,
    result,
    hand_idx: int,
    basis: np.ndarray,
    *,
    axis_len_norm: float = 0.12,
    prefix: str = "",
    thickness: int = 2,
    origin_color_px: tuple[int, int] | None = None,
    mp_frame_h: int | None = None,
    mp_frame_w: int | None = None,
    axis_labels: tuple[str, str, str] | None = None,
) -> None:
    """Draw a palm orthonormal basis using normalized MediaPipe image coordinates.

    Used as a webcam / low-depth fallback when depth projection is unavailable.

    Args:
        frame: BGR image modified in place.
        result: MediaPipe ``HandLandmarkerResult``.
        hand_idx: Index into ``result.hand_landmarks``.
        basis: Orthonormal basis, shape ``(3, 3)``; columns are axis directions in image space.
        axis_len_norm: Axis length as a fraction of ``min(frame_w, frame_h)``.
        prefix: Optional prefix prepended to axis labels.
        thickness: OpenCV line thickness for axis arrows.
        origin_color_px: Fixed color-pixel origin ``(u, v)``; computed from palm center when
            ``None``.
        mp_frame_h: MediaPipe frame height used for landmark scaling, or ``None`` for ``frame`` height.
        mp_frame_w: MediaPipe frame width used for landmark scaling, or ``None`` for ``frame`` width.
        axis_labels: Tuple of three axis label strings.

    Returns:
        None.
    """
    if result is None or not getattr(result, "hand_landmarks", None):
        return
    if hand_idx < 0 or hand_idx >= len(result.hand_landmarks):
        return
    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    if origin_color_px is not None:
        o = (int(origin_color_px[0]), int(origin_color_px[1]))
    else:
        mph = int(mp_frame_h) if mp_frame_h is not None else fh
        mpw = int(mp_frame_w) if mp_frame_w is not None else fw
        px = palm_center_color_px_from_landmarks(result, hand_idx, fh, fw, mph, mpw)
        if px is not None:
            o = px
        else:
            hlm = result.hand_landmarks[hand_idx]
            if len(hlm) < 1:
                return
            w0 = np.array([float(hlm[0].x), float(hlm[0].y), 0.0], dtype=np.float64)
            o = (int(round(w0[0] * fw)), int(round(w0[1] * fh)))
    B = np.asarray(basis, dtype=np.float64).reshape(3, 3)
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    labels = axis_labels if axis_labels is not None else ("X", "Y", "Z")
    L = float(max(axis_len_norm, 0.04)) * float(min(fw, fh))
    for i in range(3):
        d = B[:, i] * L
        tip = (int(round(o[0] + d[0])), int(round(o[1] + d[1])))
        lab = f"{prefix}{labels[i]}" if prefix else labels[i]
        _draw_axis_arrow(frame, o, tip, colors[i], label=lab, thickness=thickness)
    cv2.circle(frame, o, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, o, 4, (40, 40, 40), 1, cv2.LINE_AA)


def draw_left_pose_frame_overlay(
    frame: np.ndarray,
    *,
    calibration,
    pts_l_pose_mm: np.ndarray | None,
    result,
    idx_l: int | None,
    left_pose_state,
    left_runtime_armed: bool,
    B_rot: np.ndarray | None,
    R_pose: np.ndarray | None,
    off_m: np.ndarray | None,
    palm_basis: str,
    use_depth_projection: bool,
    motion: str = "none",
    rv_pose_world: np.ndarray | None = None,
    rv_cmd_world: np.ndarray | None = None,
    pose_rotate_rad: float = 0.008,
    palm_center_color_px: tuple[int, int] | None = None,
    mp_frame_h: int | None = None,
    mp_frame_w: int | None = None,
) -> None:
    """Draw palm-frame overlay with press-0 reference and live basis at the palm origin.

    Axes are drawn in the image plane so they stay aligned with the visible hand.

    Args:
        frame: BGR image modified in place.
        calibration: PyK4A calibration for depth projection, or ``None``.
        pts_l_pose_mm: Left-hand landmarks in depth-camera mm, shape ``(21, 3)``, or ``None``.
        result: MediaPipe ``HandLandmarkerResult``, or ``None``.
        idx_l: Index of the left hand in ``result``, or ``None``.
        left_pose_state: Runtime state for L-move palm pose (reference basis, arm flag).
        left_runtime_armed: When ``True``, L-move control is actively armed.
        B_rot: Unused; retained for call-site compatibility.
        R_pose: Unused; retained for call-site compatibility.
        off_m: Unused; retained for call-site compatibility.
        palm_basis: Palm basis mode string passed to :func:`palm_orthonormal_basis`.
        use_depth_projection: When ``True``, draw depth-camera basis and inlier hull.
        motion: Unused; retained for call-site compatibility.
        rv_pose_world: Unused; retained for call-site compatibility.
        rv_cmd_world: Unused; retained for call-site compatibility.
        pose_rotate_rad: Unused; retained for call-site compatibility.
        palm_center_color_px: Fixed color-pixel palm origin ``(u, v)``, or ``None``.
        mp_frame_h: MediaPipe frame height for landmark scaling, or ``None``.
        mp_frame_w: MediaPipe frame width for landmark scaling, or ``None``.

    Returns:
        None.
    """
    del B_rot, R_pose, off_m, motion, rv_pose_world, rv_cmd_world, pose_rotate_rad
    if frame is None:
        return
    armed = bool(left_runtime_armed or left_pose_state.is_unwinding())
    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    mph = int(mp_frame_h) if mp_frame_h is not None else fh
    mpw = int(mp_frame_w) if mp_frame_w is not None else fw

    origin_px = palm_center_color_px
    if origin_px is None and result is not None and idx_l is not None:
        origin_px = palm_center_color_px_from_landmarks(result, idx_l, fh, fw, mph, mpw)

    h = np.asarray(pts_l_pose_mm, dtype=np.float64) if pts_l_pose_mm is not None else None
    if h is not None and use_depth_projection and calibration is not None:
        fit = palm_plane_fit_mm(h)
        if fit is not None:
            _, _, inliers = fit
            draw_palm_plane_inliers_on_bgr(frame, calibration, inliers)

    if h is None:
        return

    ref_b = left_pose_state.ref_basis if left_pose_state.initialized and armed else None
    pc_override = None
    if left_pose_state.initialized:
        pc_override = np.asarray(left_pose_state.last_palm_center_mm, dtype=np.float64).reshape(3)
    basis_kw: dict = {"palm_basis": palm_basis}
    if pc_override is not None and np.all(np.isfinite(pc_override)):
        basis_kw["palm_center_override"] = pc_override
    if ref_b is not None:
        basis_kw["ref_basis"] = ref_b

    out = palm_orthonormal_basis(h, **basis_kw)
    if out is None:
        return
    _, B_now = out

    B_draw = B_now
    if result is not None and idx_l is not None:
        B_img = palm_basis_from_mp_image_plane(result, idx_l, palm_basis=palm_basis)
        if B_img is not None:
            B_draw = B_img

    if result is not None and idx_l is not None:
        if armed and left_pose_state.initialized:
            ref_B = left_pose_state.ref_basis_image
            if ref_B is None:
                ref_B = left_pose_state.ref_basis
            ref_origin_px = origin_px
            if ref_origin_px is None and calibration is not None:
                ref_mm = np.asarray(left_pose_state.ref_palm_center, dtype=np.float64).reshape(3)
                if np.all(np.isfinite(ref_mm)):
                    q = project_depth_cam_mm_to_color_px(calibration, ref_mm)
                    if q is not None:
                        c = _clip_uv(q[0], q[1], fw, fh)
                        if c is not None:
                            ref_origin_px = c
            draw_image_plane_basis_on_bgr(
                frame,
                result,
                idx_l,
                ref_B,
                prefix="0 ",
                thickness=1,
                origin_color_px=ref_origin_px,
                mp_frame_h=mp_frame_h,
                mp_frame_w=mp_frame_w,
                axis_labels=_PALM_AXIS_LABELS,
            )
        draw_image_plane_basis_on_bgr(
            frame,
            result,
            idx_l,
            B_draw,
            prefix="",
            thickness=2,
            origin_color_px=origin_px,
            mp_frame_h=mp_frame_h,
            mp_frame_w=mp_frame_w,
            axis_labels=_PALM_AXIS_LABELS,
        )
    elif use_depth_projection and calibration is not None:
        origin_mm, _ = out
        if armed and left_pose_state.initialized:
            ref_mm = np.asarray(left_pose_state.ref_palm_center, dtype=np.float64).reshape(3)
            ref_B = left_pose_state.ref_basis_image
            if ref_B is None:
                ref_B = left_pose_state.ref_basis
            draw_depth_cam_basis_on_bgr(
                frame,
                calibration,
                ref_mm,
                ref_B,
                axis_len_mm=50.0,
                prefix="0 ",
                thickness=1,
                origin_color_px=origin_px,
                axis_labels=_PALM_AXIS_LABELS,
            )
        draw_depth_cam_basis_on_bgr(
            frame,
            calibration,
            origin_mm,
            B_now,
            axis_len_mm=62.0,
            prefix="",
            thickness=2,
            origin_color_px=origin_px,
            axis_labels=_PALM_AXIS_LABELS,
        )

    if armed:
        cv2.putText(
            frame,
            "0=arm  live=palm origin (wrist+5 MCPs on plane)",
            (12, fh - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
