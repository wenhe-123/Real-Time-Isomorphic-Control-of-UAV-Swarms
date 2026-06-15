"""2D overlays: hand palm frame (depth / image) and swarm centroid frame (sim world)."""

from __future__ import annotations

import cv2
import numpy as np

from functions.display_sim.depth_fusion_utils import project_depth_cam_mm_to_color_px
from functions.swarm_motion.left_hand_swarm_pose import (
    palm_basis_from_mp_image_plane,
    palm_center_color_px_from_landmarks,
    palm_orthonormal_basis,
    palm_plane_fit_mm,
    rotvec_to_R,
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
    """Draw RGB = XYZ columns of ``basis`` (3×3) at ``origin_mm`` in depth camera mm."""
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
    """Draw convex hull of palm-plane inlier landmarks (filtered wrist/MCP/tips)."""
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
    """Draw palm basis on normalized MP image coords (webcam / low-vis fallback)."""
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


def _format_rotvec(rv: np.ndarray) -> str:
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-7:
        return "rot 0"
    ax = rv / ang
    return f"rot {np.degrees(ang):+.1f}° ax({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"


def _draw_R_axes_at(
    frame: np.ndarray,
    cx: int,
    cy: int,
    R: np.ndarray,
    scale: float,
    colors: tuple[tuple[int, int, int], ...],
    labels: tuple[str, ...],
    *,
    thickness: int = 2,
    tip_len: float = 1.0,
) -> None:
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    for i in range(3):
        v = Rm[:, i]
        tip = (int(cx + v[0] * scale * tip_len), int(cy - v[1] * scale * tip_len))
        _draw_axis_arrow(frame, (cx, cy), tip, colors[i], label=labels[i], thickness=thickness)


def draw_swarm_pose_panel_on_bgr(
    frame: np.ndarray,
    *,
    R_pose: np.ndarray | None,
    off_m: np.ndarray | None,
    motion: str = "none",
    rv_pose_world: np.ndarray | None = None,
    rv_cmd_world: np.ndarray | None = None,
    pose_rotate_rad: float = 0.008,
    armed: bool = False,
    x0: int = 12,
    y0: int = 0,
    panel_w: int = 248,
    panel_h: int = 198,
) -> None:
    """Bottom-left: **solid** = cumulative swarm R (only grows when motion=rotate).

    **Dashed** = palm pose Δ this frame (hand turning but maybe classified as translate).
    """
    if frame is None:
        return
    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    if y0 <= 0:
        y0 = max(12, fh - panel_h - 12)
    x1, y1 = min(fw - 4, x0 + panel_w), min(fh - 4, y0 + panel_h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (16, 16, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 200, 255), 1, cv2.LINE_AA)
    cx, cy = x0 + 58, y0 + panel_h // 2 + 8
    scale = float(min(panel_w, panel_h)) * 0.38
    title = "swarm pose (world)" if armed else "swarm (idle)"
    cv2.putText(
        frame,
        title + "  Z=up Y=fwd X=lat",
        (x0 + 8, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    mot = str(motion).strip() or "none"
    cv2.putText(
        frame,
        f"motion: {mot}",
        (x0 + 8, y0 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 255, 180) if mot == "translate" else (255, 200, 140) if mot == "rotate" else (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    if not armed or R_pose is None:
        cv2.circle(frame, (cx, cy), 4, (180, 180, 180), -1, cv2.LINE_AA)
        return
    R = np.asarray(R_pose, dtype=np.float64).reshape(3, 3)
    ang_from_i = float(np.degrees(np.arccos(np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0))))
    cv2.putText(
        frame,
        f"accum R angle vs arm: {ang_from_i:.1f} deg",
        (x0 + 8, y0 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    # Solid: cumulative orientation applied to swarm (changes only when motion=rotate).
    _draw_R_axes_at(
        frame,
        cx,
        cy,
        R,
        scale,
        ((70, 70, 255), (70, 255, 70), (255, 90, 90)),
        ("Xw", "Yw", "Zw"),
        thickness=3,
    )
    # Dashed: palm pose delta this frame (often non-zero even when motion=translate).
    if rv_pose_world is not None:
        rp = float(np.linalg.norm(np.asarray(rv_pose_world, dtype=np.float64).reshape(3)))
        if rp > 1e-5:
            R_step = rotvec_to_R(np.asarray(rv_pose_world, dtype=np.float64).reshape(3))
            _draw_R_axes_at(
                frame,
                cx,
                cy,
                R_step,
                scale,
                ((120, 120, 255), (120, 255, 120), (255, 180, 120)),
                ("px", "py", "pz"),
                thickness=1,
                tip_len=0.72,
            )
            cv2.putText(
                frame,
                f"|rv_pose|={rp:.4f}  rot if >{float(pose_rotate_rad):.4f}",
                (x0 + 8, y0 + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 200, 120),
                1,
                cv2.LINE_AA,
            )
    if rv_cmd_world is not None:
        rc = float(np.linalg.norm(np.asarray(rv_cmd_world, dtype=np.float64).reshape(3)))
        cv2.putText(
            frame,
            f"|rv_cmd|={rc:.4f} (applied to swarm)",
            (x0 + 8, y0 + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (180, 255, 200) if rc > 1e-5 else (140, 140, 140),
            1,
            cv2.LINE_AA,
        )
    if off_m is not None:
        off = np.asarray(off_m, dtype=np.float64).reshape(3)
        nxy = float(np.hypot(off[0], off[1]))
        if nxy > 1e-5:
            d = off[:2] / nxy
            tip = (int(cx + d[0] * scale * 0.9), int(cy - d[1] * scale * 0.9))
            _draw_axis_arrow(frame, (cx, cy), tip, (80, 220, 255), label="T", thickness=2)
        off_s = f"T=({off[0]:+.2f},{off[1]:+.2f},{off[2]:+.2f})m |T|={float(np.linalg.norm(off)):.2f}"
        cv2.putText(
            frame,
            off_s,
            (x0 + 8, y1 - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (200, 230, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "solid=swarm accum  dashed=palm pose/frame",
        (x0 + 8, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )


def draw_swarm_world_axes_at_palm_on_bgr(
    frame: np.ndarray,
    calibration,
    palm_center_mm: np.ndarray,
    R_pose: np.ndarray,
    M_cam_to_world: np.ndarray,
    *,
    axis_len_mm: float = 75.0,
    off_m: np.ndarray | None = None,
    origin_color_px: tuple[int, int] | None = None,
) -> None:
    """Project swarm world axes (+ translation) onto the image at palm depth (frozen cam→world at arm)."""
    if calibration is None or frame is None:
        return
    M = np.asarray(M_cam_to_world, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R_pose, dtype=np.float64).reshape(3, 3)
    o = np.asarray(palm_center_mm, dtype=np.float64).reshape(3)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if origin_color_px is not None:
        ou_i = _clip_uv(float(origin_color_px[0]), float(origin_color_px[1]), w, h)
    else:
        ou = project_depth_cam_mm_to_color_px(calibration, o)
        if ou is None:
            return
        ou_i = _clip_uv(ou[0], ou[1], w, h)
    if ou_i is None:
        return
    L = float(max(axis_len_mm, 20.0))
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    labels = ("Sx", "Sy", "Sz")
    for i in range(3):
        v_cam = M.T @ R[:, i]
        nv = float(np.linalg.norm(v_cam))
        if nv < 1e-9:
            continue
        tip_mm = o + (v_cam / nv) * L
        t = project_depth_cam_mm_to_color_px(calibration, tip_mm)
        if t is None:
            continue
        tu = _clip_uv(t[0], t[1], w, h)
        if tu is None:
            continue
        _draw_axis_arrow(frame, ou_i, tu, colors[i], label=labels[i], thickness=3)
    if off_m is not None:
        v_cam = M.T @ np.asarray(off_m, dtype=np.float64).reshape(3)
        nv = float(np.linalg.norm(v_cam))
        if nv > 1e-6:
            tip_mm = o + (v_cam / nv) * L * 0.85
            t = project_depth_cam_mm_to_color_px(calibration, tip_mm)
            if t is not None:
                tu = _clip_uv(t[0], t[1], w, h)
                if tu is not None:
                    _draw_axis_arrow(frame, ou_i, tu, (255, 220, 80), label="T", thickness=2)


def draw_swarm_centroid_frame_inset(
    frame: np.ndarray,
    *,
    R_pose: np.ndarray | None,
    off_m: np.ndarray | None,
    armed: bool,
    panel_size: int = 148,
    margin: int = 12,
) -> None:
    """Top-right inset: world/sim axes at formation centroid (after L-move rotation)."""
    if frame is None:
        return
    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    ps = int(max(96, min(panel_size, min(fw, fh) // 3)))
    x0 = fw - ps - margin
    y0 = margin
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + ps, y0 + ps), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + ps, y0 + ps), (180, 180, 180), 1, cv2.LINE_AA)
    cx, cy = x0 + ps // 2, y0 + ps // 2
    scale = float(ps) * 0.32
    title = "swarm@centroid" if armed else "swarm (idle)"
    cv2.putText(
        frame,
        title,
        (x0 + 6, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if not armed or R_pose is None:
        cv2.circle(frame, (cx, cy), 3, (200, 200, 200), -1, cv2.LINE_AA)
        return
    R = np.asarray(R_pose, dtype=np.float64).reshape(3, 3)
    # World axes after applied formation rotation (columns of R).
    colors = ((80, 80, 255), (80, 255, 80), (255, 80, 80))
    labels = ("Xw", "Yw", "Zw")
    for i in range(3):
        v = R[:, i]
        tip = (int(cx + v[0] * scale), int(cy - v[1] * scale))
        _draw_axis_arrow(frame, (cx, cy), tip, colors[i], label=labels[i], thickness=2)
    if off_m is not None:
        off = np.asarray(off_m, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(off[:2]))
        if n > 1e-4:
            d = off[:2] / n
            tip = (int(cx + d[0] * scale * 0.85), int(cy - d[1] * scale * 0.85))
            _draw_axis_arrow(frame, (cx, cy), tip, (255, 220, 80), label="T", thickness=1)


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
    """Palm frame overlay: press-0 + live basis at palm-plane origin (wrist + 5 MCPs).

    Axes are drawn in the image plane so they stay aligned with the visible hand.
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
