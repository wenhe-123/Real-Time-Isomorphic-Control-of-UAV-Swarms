"""Legacy axis_locked / full control-style gates (backup tests only)."""

from __future__ import annotations

import numpy as np

from backup.tests.left_hand_swarm_pose_test_api import (
    palm_components_in_camera_mm,
    palm_vector_palm_to_world,
)
from functions.swarm_motion.left_hand_swarm_pose import PALM_AXIS_TO_WORLD_PERM, R_to_rotvec

def uses_palm_world_axis_embedding(palm_basis: str) -> bool:
    return str(palm_basis).strip().lower() in ("middle_thumb", "middle_y", "middle")

def lateral_dom_keep_only_dx(
    delta_cam: np.ndarray,
    *,
    lateral_dom_ratio: float,
    lateral_dom_min_mm: float,
) -> np.ndarray:
    """When ``|dx|`` dominates ``hypot(dy,dz)``, use only ``[dx,0,0]`` for translation (camera mm).

    Reduces **image up/down** and **near/far** leak when the user moves mainly **left/right**
    in the depth frame (Orbbec ``+X`` right, ``+Y`` down, ``+Z`` forward).
    """
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    r = float(lateral_dom_ratio)
    if r <= 0.0:
        return d
    dx = float(d[0])
    if abs(dx) < float(lateral_dom_min_mm):
        return d
    dy, dz = float(d[1]), float(d[2])
    h = float(np.hypot(dy, dz))
    if abs(dx) >= r * max(h, 1e-9):
        return np.array([dx, 0.0, 0.0], dtype=np.float64)
    return d

def axis_locked_gated_cam_delta_mm(
    delta_cam: np.ndarray,
    *,
    lateral_min_dx_mm: float = 4.0,
    lateral_strip_dz_ratio: float = 0.72,
    forward_min_dz_mm: float = 5.0,
    forward_dom_ratio: float = 1.22,
    forward_xy_small_mm: float = 6.0,
    vertical_min_dy_mm: float = 4.0,
    depth_strong_min_dz_mm: float = 28.0,
    depth_strong_dom_ratio: float = 1.35,
    lateral_over_depth_margin: float = 1.35,
) -> np.ndarray:
    """Gated wrist delta (camera mm) before ``M_trans`` → world.

    **Lateral (dx) and image vertical (dy) are checked before near/far (dz)** so a combined
    pan does not lose up/down when depth also moves (common with Orbbec palm motion).
    """
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    ax, ay, az = abs(float(d[0])), abs(float(d[1])), abs(float(d[2]))
    if ax >= float(lateral_min_dx_mm) and ax >= 1.06 * max(ay, az, 1e-9):
        if az >= float(forward_min_dz_mm) and az >= float(lateral_over_depth_margin) * max(
            ax, 1e-9
        ):
            return np.array([0.0, 0.0, d[2]], dtype=np.float64)
        return np.array([d[0], 0.0, 0.0], dtype=np.float64)
    if ay >= float(vertical_min_dy_mm) and ay >= 1.06 * max(ax, az, 1e-9):
        return np.array([0.0, d[1], 0.0], dtype=np.float64)
    if ax >= float(lateral_min_dx_mm) and ax >= float(lateral_strip_dz_ratio) * max(az, 1e-9):
        return np.array([d[0], 0.0, 0.0], dtype=np.float64)
    xy_small = max(ax, ay) < float(forward_xy_small_mm)
    if az >= float(depth_strong_min_dz_mm) and az >= float(depth_strong_dom_ratio) * max(
        ax, ay, 1e-9
    ):
        return np.array([0.0, 0.0, d[2]], dtype=np.float64)
    if az >= float(forward_min_dz_mm) and (
        xy_small or az >= float(forward_dom_ratio) * max(ax, ay, 1e-9)
    ):
        return np.array([0.0, 0.0, d[2]], dtype=np.float64)
    order = np.argsort(np.array([ax, ay, az], dtype=np.float64))
    i_max = int(order[2])
    mins = (
        float(lateral_min_dx_mm),
        float(vertical_min_dy_mm),
        float(forward_min_dz_mm),
    )
    if (i_max == 0 and ax >= mins[0]) or (i_max == 1 and ay >= mins[1]) or (i_max == 2 and az >= mins[2]):
        out = np.zeros(3, dtype=np.float64)
        out[i_max] = d[i_max]
        return out
    return np.zeros(3, dtype=np.float64)

def palm_translation_components_mm(
    delta_cam: np.ndarray,
    B_proj: np.ndarray,
    *,
    scale_delta_mm: float = 0.0,
    span_ref_mm: float = 0.0,
    scale_min_mm: float = 8.0,
    scale_min_rel: float = 0.08,
    scale_gain: float = 1.1,
    scale_max_per_frame_mm: float = 35.0,
    lateral_min_mm: float = 3.0,
    vertical_min_mm: float = 3.0,
    forward_min_mm: float = 5.0,
) -> np.ndarray:
    """Palm-frame mm for translation; project with **current** palm basis ``B_proj``."""
    dc = np.asarray(delta_cam, dtype=np.float64).reshape(3)
    frame_mm = float(np.linalg.norm(dc))
    cdx, cdy, cdz = abs(float(dc[0])), abs(float(dc[1])), abs(float(dc[2]))
    raw = palm_components_in_camera_mm(delta_cam, B_proj)
    sd = float(scale_delta_mm)
    if (
        frame_mm >= 2.0
        and frame_mm < 18.0
        and abs(sd) <= float(scale_max_per_frame_mm)
        and cdx < 14.0
        and cdy < 14.0
    ):
        fwd = forward_palm_component_from_scale_only(
            sd,
            span_ref_mm=span_ref_mm,
            min_mm=scale_min_mm,
            min_rel=scale_min_rel,
            gain=scale_gain,
        )
        if fwd is not None:
            return fwd
    gated = axis_locked_gated_palm_components(
        delta_cam,
        B_proj,
        lateral_min_mm=float(lateral_min_mm),
        vertical_min_mm=float(vertical_min_mm),
        forward_min_mm=float(forward_min_mm),
    )
    if float(np.linalg.norm(gated)) >= 1e-6:
        return gated
    ax, ay, az = abs(float(raw[0])), abs(float(raw[1])), abs(float(raw[2]))
    pick = int(np.argmax([ax, ay, az]))
    mins = (float(lateral_min_mm), float(vertical_min_mm), float(scale_min_mm) * 0.5)
    if [ax, ay, az][pick] >= mins[pick] * 0.55:
        out = np.zeros(3, dtype=np.float64)
        out[pick] = raw[pick]
        return out
    return gated

def strip_dz_when_abs_dx_dominates_dz(
    delta_cam: np.ndarray,
    *,
    ratio: float,
    min_abs_dx_mm: float,
) -> np.ndarray:
    """When ``|dx|`` exceeds ``ratio * |dz|`` (and ``|dx|`` is at least ``min_abs_dx_mm``), zero ``dz``.

    Slow lateral pans have **small per-frame** ``hypot(dx,dy)`` that can sit below ``min_hypot_xy``
    in :func:`strip_dz_when_xy_motion_dominates`, while ``dz`` noise is similar magnitude — this rule
    still removes spurious depth for mostly-X motion. ``ratio`` 0 disables.
    """
    r = float(ratio)
    if r <= 0.0:
        return np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    dx, dz = float(d[0]), float(d[2])
    if abs(dx) < float(min_abs_dx_mm):
        return d
    if abs(dx) >= r * max(abs(dz), 1e-9):
        d[2] = 0.0
    return d

def _snap_vector_keep_principal(v: np.ndarray, *, min_ratio: float) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).reshape(3)
    abs_o = np.abs(out)
    if float(np.max(abs_o)) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    order = np.argsort(abs_o)
    i_max = int(order[2])
    i_mid = int(order[1])
    v_max = float(abs_o[i_max])
    v_mid = float(abs_o[i_mid])
    ratio = float(max(1.0, min_ratio))
    if v_max < ratio * max(v_mid, 1e-9):
        return np.zeros(3, dtype=np.float64)
    keep = np.zeros(3, dtype=np.float64)
    keep[i_max] = out[i_max]
    return keep

def palm_components_to_world_m(
    comp_palm: np.ndarray,
    *,
    trans_scale: float,
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Palm-frame mm components (x thumb, y fingertip, z=X×Y) → world m."""
    world_mm = palm_vector_palm_to_world(np.asarray(comp_palm, dtype=np.float64).reshape(3))
    sign = np.asarray(axis_sign, dtype=np.float64).reshape(3)
    return float(trans_scale) * world_mm * sign

def forward_palm_component_from_scale_only(
    scale_delta_mm: float,
    *,
    span_ref_mm: float,
    min_mm: float = 8.0,
    min_rel: float = 0.08,
    gain: float = 1.1,
) -> np.ndarray | None:
    """Forward/back as **only** palm Z from hand span change; never wrist depth."""
    if not scale_forward_triggered(
        scale_delta_mm, span_ref_mm=span_ref_mm, min_mm=min_mm, min_rel=min_rel
    ):
        return None
    sd = float(scale_delta_mm) * float(max(0.5, gain))
    mag = max(abs(sd), float(min_mm) * 0.85)
    return np.array([0.0, 0.0, float(np.sign(sd) if abs(sd) > 1e-9 else 1.0) * mag], dtype=np.float64)

def effective_cam_delta_for_translation(
    delta_cam: np.ndarray,
    *,
    depth_dom_ratio: float,
    depth_dom_min_mm: float,
) -> np.ndarray:
    """Return wrist delta in depth-camera mm for **translation** only.

    When ``depth_dom_ratio > 0`` and ``|dz|`` is both at least ``depth_dom_min_mm`` and at least
    ``depth_dom_ratio * hypot(dx, dy)``, return ``[0, 0, dz]`` so only optical-axis motion maps
    to world (suppresses bogus ``dx, dy`` when the hand moves mainly toward/away from the camera
    and grows in the image). Otherwise return the full ``delta_cam`` copy.
    """
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    r = float(depth_dom_ratio)
    if r <= 0.0:
        return d
    dz = float(d[2])
    if abs(dz) < float(depth_dom_min_mm):
        return d
    dx, dy = float(d[0]), float(d[1])
    h = float(np.hypot(dx, dy))
    if h < 1e-9:
        return np.array([0.0, 0.0, dz], dtype=np.float64)
    if abs(dz) >= r * h:
        return np.array([0.0, 0.0, dz], dtype=np.float64)
    return d

def world_rotvec_from_palm_basis_delta(
    B_current: np.ndarray,
    B_arm: np.ndarray,
) -> np.ndarray:
    """Rotation from palm basis change; components map to world X/Y/Z (thumb/forward/up)."""
    Bc = np.asarray(B_current, dtype=np.float64).reshape(3, 3)
    Ba = np.asarray(B_arm, dtype=np.float64).reshape(3, 3)
    R_cam = Bc @ Ba.T
    R_palm = Bc.T @ R_cam @ Bc
    return palm_vector_palm_to_world(np.asarray(R_to_rotvec(R_palm), dtype=np.float64).reshape(3))

def strip_dz_when_xy_motion_dominates(
    delta_cam: np.ndarray,
    *,
    ratio: float,
    min_hypot_xy_mm: float,
) -> np.ndarray:
    """When ``hypot(dx,dy)`` clearly exceeds ``|dz|``, zero camera ``dz`` for translation only.

    Under ``fwd_y``, ``dz`` maps mostly to world ``Y`` (forward/back). Lateral pans still produce
    noisy depth at the wrist; stripping ``dz`` when image-plane motion dominates keeps **left/right**
    hand motion from driving large **near/far** targets. Intentional in-out keeps large ``|dz|``
    relative to ``hypot(dx,dy)`` so ``dz`` is retained. ``ratio`` 0 disables.
    """
    r = float(ratio)
    if r <= 0.0:
        return np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    hxy = float(np.hypot(dx, dy))
    if hxy < float(min_hypot_xy_mm):
        return d
    if hxy >= r * max(abs(dz), 1e-9):
        d[2] = 0.0
    return d

def hand_scale_toward_camera_mm(span_now: float, span_ref: float) -> float:
    """Signed mm proxy for near/far: span increase → toward camera (positive palm Z)."""
    if span_ref < 1e-6:
        return 0.0
    return float(span_now) - float(span_ref)

def axis_locked_gated_palm_components(
    delta_cam: np.ndarray,
    B: np.ndarray,
    *,
    lateral_min_mm: float = 4.0,
    vertical_min_mm: float = 4.0,
    forward_min_mm: float = 4.0,
) -> np.ndarray:
    """One dominant palm axis per frame: thumb X, fingertip Y, or Z (fwd/back)."""
    d = palm_components_in_camera_mm(delta_cam, B)
    ax, ay, az = abs(float(d[0])), abs(float(d[1])), abs(float(d[2]))
    if az >= float(forward_min_mm) and az >= 1.22 * max(ax, ay, 1e-9):
        return np.array([0.0, 0.0, d[2]], dtype=np.float64)
    if ax >= float(lateral_min_mm) and ax >= 1.05 * max(ay, az, 1e-9):
        return np.array([d[0], 0.0, 0.0], dtype=np.float64)
    if ay >= float(vertical_min_mm) and ay >= 1.05 * max(ax, az, 1e-9):
        return np.array([0.0, d[1], 0.0], dtype=np.float64)
    if ax >= float(lateral_min_mm) or ay >= float(vertical_min_mm) or az >= float(forward_min_mm):
        order = sorted(
            ((ax, 0, d[0]), (ay, 1, d[1]), (az, 2, d[2])),
            key=lambda t: t[0],
            reverse=True,
        )
        i = int(order[0][1])
        out = np.zeros(3, dtype=np.float64)
        out[i] = order[0][2]
        return out
    return np.zeros(3, dtype=np.float64)

def translation_plane_dominates_depth(
    delta_trans: np.ndarray,
    *,
    planar_ratio: float,
    planar_min_mm: float,
) -> bool:
    """Return True if ``hypot(dx,dy)`` clearly exceeds ``|dz|`` (camera mm after translation gates).

    Used to **suppress palm rotation** for that frame when the wrist delta looks like a **2D pan**
    rather than moving mainly in depth (reduces spurious formation tilt in plane morph mode).

    ``|dz|`` is lower-bounded by a small millimetre floor so that after ``dz`` stripping (≈0) we do
    not treat the ratio as infinite; pan dominance still requires ``hypot(dx,dy)`` to clear the
    ratio against that floor.
    """
    r = float(planar_ratio)
    if r <= 0.0:
        return False
    d = np.asarray(delta_trans, dtype=np.float64).reshape(3)
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    h = float(np.hypot(dx, dy))
    if h < float(planar_min_mm):
        return False
    dz_den = max(abs(dz), 1.05)
    return h >= r * dz_den

def snap_world_vector_to_principal_axis(
    v: np.ndarray,
    *,
    deadzone: float = 0.0,
    min_ratio: float = 1.0,
) -> np.ndarray:
    """Keep only the largest-magnitude world/sim component (±X, ±Y, or ±Z).

    Suppresses hand jitter that would otherwise leak into multiple axes at once.
    ``min_ratio``: require ``|v_max| >= min_ratio * |v_2nd|`` or return zero (reject ambiguous motion).
    """
    out = np.asarray(v, dtype=np.float64).reshape(3).copy()
    dz = float(max(0.0, deadzone))
    if dz > 0.0:
        out[np.abs(out) < dz] = 0.0
    return _snap_vector_keep_principal(out, min_ratio=float(min_ratio))

def palm_optical_alignment_cos(B: np.ndarray) -> float:
    """Return ``|n · e_z|`` for unit ``n`` = palm third column of ``B`` (Gram–Schmidt ``e3``).

    Near **1** means the palm plane is nearly parallel to the image plane (palm facing / parallel
    to the camera), so in-plane twist is mostly about camera **+Z** (optical axis).
    """
    n = np.asarray(B[:, 2], dtype=np.float64).reshape(3)
    ln = float(np.linalg.norm(n))
    if ln < 1e-9:
        return 0.0
    n /= ln
    return abs(float(n[2]))

def cam_delta_palm_to_world_m(
    delta_cam: np.ndarray,
    B: np.ndarray,
    *,
    trans_scale: float,
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Wrist delta (camera mm) → world m via palm-axis projection."""
    comp = palm_components_in_camera_mm(delta_cam, B)
    return palm_components_to_world_m(comp, trans_scale=trans_scale, axis_sign=axis_sign)

def scale_forward_triggered(
    scale_delta_mm: float,
    *,
    span_ref_mm: float,
    min_mm: float = 8.0,
    min_rel: float = 0.08,
) -> bool:
    """True only when the whole hand is clearly larger/smaller than at arm (push/pull)."""
    sd = abs(float(scale_delta_mm))
    if sd < float(min_mm):
        return False
    ref = float(span_ref_mm)
    if ref > 1e-6 and sd / ref < float(min_rel):
        return False
    return True

def vertical_dom_zero_optical_z(
    delta_cam: np.ndarray,
    *,
    vertical_dom_ratio: float,
    vertical_dom_min_mm: float,
    optical_preserve_ratio: float = 0.0,
    optical_preserve_min_mm: float = 0.0,
) -> np.ndarray:
    """When image-plane ``|dy|`` dominates ``hypot(dx,dz)``, use only ``[0, dy, 0]`` for translation.

    Mirrors :func:`lateral_dom_keep_only_dx`: vertical pans keep camera ``dy`` (→ world altitude
    via ``image_y_to_world_z``) and drop spurious ``dx`` / ``dz`` depth noise that otherwise maps to
    world ``X`` / ``Y`` under ``fwd_y``.

    When ``optical_preserve_ratio > 0`` and ``|dz|`` clearly dominates ``hypot(dx,dy)`` (and
    ``|dz| >= optical_preserve_min_mm``), return ``delta_cam`` unchanged so **near/far** motion is
    not stripped by vertical coupling (common when ``dy`` grows from perspective during in-out).
    """
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
    ro = float(optical_preserve_ratio)
    mo = float(optical_preserve_min_mm)
    if ro > 0.0:
        dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
        if abs(dz) >= float(mo) and abs(dz) >= ro * float(np.hypot(dx, dy)):
            return d
    r = float(vertical_dom_ratio)
    if r <= 0.0:
        return d
    dy = float(d[1])
    if abs(dy) < float(vertical_dom_min_mm):
        return d
    dx, dz = float(d[0]), float(d[2])
    h = float(np.hypot(dx, dz))
    if abs(dy) >= r * max(h, 1e-9):
        return np.array([0.0, dy, 0.0], dtype=np.float64)
    return d

