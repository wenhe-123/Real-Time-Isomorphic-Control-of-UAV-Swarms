"""Left-hand 6DoF (translation + rotation) driving a rigid transform on the whole swarm target.

Translation needs **metric wrist motion** (depth unprojection at 2D landmarks). MediaPipe
``hand_world_landmarks`` are wrist-centric, so they do **not** move when the hand pans — use
`left_hand_pose_matrix_depth_mm` on Orbbec.

Orbbec Femto Bolt depth frame (doc): **+X right, +Y down, +Z forward** into the scene.

**``camera_at_arm`` (online_control default):** on press ``0``, ``ref_wrist`` + ``ref_basis`` and a
``cam→sim`` matrix are frozen for that session. Motion is ``wrist − ref`` in camera mm, then
``M_frozen @ delta`` into simulation coordinates — an absolute reference, not a moving global map.

Presets map camera mm deltas into Crazyflow world (``Z`` = altitude). Default ``camera`` uses
identity (sim axes = depth camera at arm). ``fwd_y`` sends **near/far** mostly to
**world ``Y``** and **image left/right** to **world ``X``**; image **up/down** can map to **world ``Z``**
via ``make_cam_translation_matrix(..., image_y_to_world_z)``. Use ``legacy`` for the older ``-dz→world X`` map.

With preset ``fwd_y``, **vertical-dominant** gating can keep only ``[0,dy,0]`` when ``|dy|`` dominates; optional
``optical_preserve_*`` skips that strip when ``|dz|`` dominates ``hypot(dx,dy)`` (optional tuning).

Rotation (default pipeline): ``R_cam = B @ ref_basis.T`` in the depth-camera frame, then
``ω_world = R_to_rotvec(M @ R_cam @ M.T)`` (same ``M`` as translation base). Smoothed via axis–angle EMA;
optional ``rot_world_z_scale`` scales only the world ``Z`` component of ``ω_world``.

**Palm facing camera (optional, CLI):** ``--left-rot-palm-face-twist-world-y`` enables a path where,
under ``fwd_y`` + depth, aligned palm + optical-axis twist uses ``ω_world = [0, θ, 0]`` with
``θ = atan2(R_cam[1,0], R_cam[0,0])``. **Off by default** (noisy during translation). When enabled,
that branch skips several rotation dampers; it is **not** used in the same frame as image-plane pan
(``translation_plane_dominates_depth`` for ``delta_trans``).

**Coordinates (Orbbec Femto / K4A depth camera, right-handed):** ``+X`` image right, ``+Y`` image down,
``+Z`` optical axis into the scene (forward). Crazyflow targets use **world Z = up** (altitude).
Preset ``fwd_y`` uses orthonormal ``M`` with ``(Δx,Δy,Δz)_cam → (ΔX,ΔY,ΔZ)_world`` given by
``[ΔX,ΔY,ΔZ]^T = M @ [Δx,Δy,Δz]^T`` for rotation; translation uses ``M_trans`` (same first two rows as ``M``,
third row scaled by ``image_y_to_world_z``). Concretely for ``fwd_y``:
``ΔX≈Δx``, ``ΔY≈Δz`` (near/far → world horizontal “into scene”), ``ΔZ≈−image_y_to_world_z·Δy`` (image up/down → altitude).
``axis_sign`` multiplies ``(ΔX,ΔY,ΔZ)`` after ``M_trans``. Use ``--left-flip-x|y|z`` if a world axis feels inverted.

Optional: ``rot_trans_tau_mm`` scales rotation by ``exp(-Δwrist_mm/tau)`` per frame (default off
in online_control). ``rot_world_z_scale`` damps the **world vertical** component of the axis–angle
vector (reduces in-plane “spin” of the formation while keeping more tilt from XY components).

When ``|dz|`` clearly dominates ``hypot(dx,dy)``, ``depth_dom_ratio`` / ``depth_dom_min_mm`` can
zero ``dx,dy`` for **translation** only (perspective / scale in image); rotation still uses the full
palm delta.

``strip_dz_when_xy_motion_dominates`` / ``strip_dz_when_abs_dx_dominates_dz`` (``trans_*_strip_dz_*``):
    run **after** vertical gating and **before** ``depth_dom`` / ``lateral_dom``, stripping spurious ``dz``
    on lateral pans (``fwd_y`` maps ``dz`` → world ``Y``).

``lateral_dom_ratio`` / ``lateral_dom_min_mm``: when ``|dx|`` dominates ``hypot(dy,dz)``, translation
uses only ``[dx,0,0]`` (strong horizontal-only pan).

Default ``palm_basis=middle_thumb``: **+Y** = middle finger/MCP → wrist, **+X** = thumb lateral
(in the plane ⊥ Y), **+Z** = X×Y (right-hand rule, flips naturally for palm/back). Palm vectors map to world X/Y/Z via
:data:`PALM_AXIS_TO_WORLD_PERM`; rotation uses ``ω = R_to_rotvec(M @ ΔR @ M.T)`` with the frozen
``M`` at arm. When the wrist moves quickly but intrinsic palm rotation is below
``rot_coex_max_angle_rad``, ``rot_coex_trans_min_mm`` can zero rotation for that frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# MediaPipe landmark indices
IDX_WRIST = 0
IDX_THUMB_MCP = 2
IDX_THUMB_TIP = 4
IDX_INDEX_MCP = 5
IDX_INDEX_TIP = 8
IDX_MIDDLE_MCP = 9
IDX_MIDDLE_TIP = 12
IDX_RING_MCP = 13
IDX_RING_TIP = 16
IDX_PINKY_TIP = 20

# Fingertip indices for wrist-relative Kabsch rotation (decoupled from wrist translation).
FINGERTIP_TIP_INDICES: tuple[int, ...] = (
    IDX_INDEX_TIP,
    IDX_MIDDLE_TIP,
    IDX_RING_TIP,
    IDX_PINKY_TIP,
)
_MIN_FINGERTIP_REL_MM = 8.0

# Orbbec depth camera: +Z into scene; toward camera ≈ −Z_cam.
CAM_OPTICAL_TOWARD_CAMERA = np.array([0.0, 0.0, -1.0], dtype=np.float64)

# Palm (x,y,z) → simulation world (X,Y,Z): thumb=lateral, palm Z=forward, middle-to-wrist=vertical.
PALM_AXIS_TO_WORLD_PERM = (0, 2, 1)  # palm x→world X, palm z→world Y, palm y→world Z

# Palm frame: Gram–Schmidt on (p_first − wrist), (p_second − wrist). Wider span reduces
# spurious rotation when index/middle MCPs are close in depth noise.
LEFT_PALM_BASIS_PRESETS: dict[str, tuple[int, int]] = {
    # Default: X/Y in screen plane; Y is middle→wrist, X thumb side; Z flips for palm/back.
    "middle_thumb": (IDX_MIDDLE_MCP, IDX_THUMB_MCP),
    "index_middle": (IDX_MIDDLE_MCP, IDX_INDEX_MCP),
    "index_ring": (IDX_RING_MCP, IDX_INDEX_MCP),
    "middle_ring": (IDX_RING_MCP, IDX_MIDDLE_MCP),
}
DEFAULT_LEFT_PALM_BASIS = "middle_thumb"


def palm_basis_pair_indices(preset: str) -> tuple[int, int]:
    key = str(preset).strip().lower()
    if key not in LEFT_PALM_BASIS_PRESETS:
        keys = ", ".join(sorted(LEFT_PALM_BASIS_PRESETS))
        raise ValueError(f"unknown left palm basis {preset!r}; choose one of: {keys}")
    a, b = LEFT_PALM_BASIS_PRESETS[key]
    return int(a), int(b)


# Prefer tip−wrist if segment length exceeds this (mm); else MCP−wrist.
_MIN_TIP_EDGE_MM = 12.0

# --- Depth camera (mm) → simulation world (m) for left-swarm rigid -----------------
# Femto Bolt / K4A depth camera: +X right, +Y down, +Z forward (Orbbec documentation).
# Each preset is an orthonormal 3×3 ``M`` with det(M)=+1 used for translation (after row scaling)
# and for rotation ``R_world = M @ R_cam @ M.T``.
#
# Camera mm from ``unproject_to_depth_cam_mm``: +X right, +Y down, +Z forward (Femto / K4A).
# World: Z-up. See module docstring for fwd_y row meanings.

LEFT_CAM_PRESET_ROT: dict[str, np.ndarray] = {
    # Depth-camera axes = simulation axes (X right, Y down, Z forward). Use with ``camera_at_arm``.
    "camera": np.eye(3, dtype=np.float64),
    # Older iso_swarm mapping: −optical Z → world X, +cam X → world Y, −cam Y → world Z.
    "legacy": np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    # Optical Z (depth) → world Y “into scene”; optical X (image right) → world X; optical Y → −world Z.
    "fwd_y": np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    # Negate depth column vs ``legacy`` (try if near/far feels inverted).
    "flip_depth": np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
}

# Default matches ``fwd_y`` (see module doc).
DEFAULT_CAM_DELTA_TO_WORLD = LEFT_CAM_PRESET_ROT["fwd_y"].copy()
LEGACY_CAM_DELTA_TO_WORLD = LEFT_CAM_PRESET_ROT["legacy"].copy()


def build_sim_from_cam_matrices(
    preset: str,
    *,
    image_y_to_world_z: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(M_rot, M_trans)`` embedding depth-camera mm into simulation coordinates."""
    M_rot = left_cam_preset_rotation(preset)
    M_trans = make_cam_translation_matrix(M_rot, image_y_to_world_z=float(image_y_to_world_z))
    return M_rot, M_trans


def left_cam_preset_rotation(preset: str) -> np.ndarray:
    """Return a copy of the orthonormal rotation for ``preset`` (``camera`` | ``legacy`` | ``fwd_y`` | ``flip_depth``)."""
    key = str(preset).strip().lower()
    if key not in LEFT_CAM_PRESET_ROT:
        keys = ", ".join(sorted(LEFT_CAM_PRESET_ROT))
        raise ValueError(f"unknown left-cam preset {preset!r}; choose one of: {keys}")
    return np.asarray(LEFT_CAM_PRESET_ROT[key], dtype=np.float64).copy()


def make_cam_translation_matrix(M_rot: np.ndarray, *, image_y_to_world_z: float = 0.0) -> np.ndarray:
    """World mm delta from camera mm delta: ``M_trans @ v`` with same XY rows as ``M_rot``.

    World-Z row is ``M_rot[2,:] * image_y_to_world_z`` so ``0`` disables altitude from camera Y
    (avoids near/far coupling into ``wz``). ``1`` restores full third row like rotation.
    """
    M = np.asarray(M_rot, dtype=np.float64).reshape(3, 3).copy()
    s = float(np.clip(image_y_to_world_z, 0.0, 1.0))
    M[2, :] *= s
    return M


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


def snap_cam_mm_to_principal_axis(
    v: np.ndarray,
    *,
    deadzone_xyz_mm: tuple[float, float, float] = (5.0, 5.0, 14.0),
    min_ratio: float = 1.2,
) -> np.ndarray:
    """Snap wrist delta in **depth camera mm** before ``M_trans`` (dz deadzone usually largest).

    Camera frame: +X right, +Y down, +Z forward. Rejects depth noise stealing lateral pans.
    """
    out = np.asarray(v, dtype=np.float64).reshape(3).copy()
    dz = np.asarray(deadzone_xyz_mm, dtype=np.float64).reshape(3)
    for i in range(3):
        if abs(float(out[i])) < float(dz[i]):
            out[i] = 0.0
    return _snap_vector_keep_principal(out, min_ratio=float(min_ratio))


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


_HAND_SPAN_LANDMARKS: tuple[int, ...] = (
    IDX_THUMB_MCP,
    IDX_THUMB_TIP,
    IDX_INDEX_MCP,
    IDX_INDEX_TIP,
    IDX_MIDDLE_MCP,
    IDX_MIDDLE_TIP,
    IDX_RING_MCP,
    IDX_RING_TIP,
    IDX_PINKY_TIP,
)


def palm_center_mm(h: np.ndarray) -> np.ndarray | None:
    """Palm centroid in depth camera mm (wrist + MCPs); translation origin, not the wrist alone."""
    pts: list[np.ndarray] = []
    for idx in (IDX_WRIST, IDX_THUMB_MCP, IDX_INDEX_MCP, IDX_MIDDLE_MCP, IDX_RING_MCP):
        p = np.asarray(h[int(idx), :3], dtype=np.float64).reshape(3)
        if np.all(np.isfinite(p)):
            pts.append(p)
    if len(pts) < 3:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def hand_palm_span_mm(h: np.ndarray, wrist: np.ndarray | None = None) -> float:
    """Mean wrist→landmark distance (mm); grows when the hand moves closer (appears larger)."""
    w = np.asarray(h[IDX_WRIST, :3], dtype=np.float64).reshape(3) if wrist is None else np.asarray(wrist).reshape(3)
    dists: list[float] = []
    for idx in _HAND_SPAN_LANDMARKS:
        v = np.asarray(h[int(idx), :3], dtype=np.float64).reshape(3) - w
        n = float(np.linalg.norm(v))
        if n >= 8.0:
            dists.append(n)
    if not dists:
        return 0.0
    return float(np.mean(dists))


def hand_scale_toward_camera_mm(span_now: float, span_ref: float) -> float:
    """Signed mm proxy for near/far: span increase → toward camera (positive palm Z)."""
    if span_ref < 1e-6:
        return 0.0
    return float(span_now) - float(span_ref)


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


def axis_locked_gated_palm_components(
    delta_cam: np.ndarray,
    B: np.ndarray,
    *,
    lateral_min_mm: float = 4.0,
    vertical_min_mm: float = 4.0,
    forward_min_mm: float = 4.0,
) -> np.ndarray:
    """One dominant palm axis per frame: thumb X, middle→wrist Y, or Z (fwd/back)."""
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


def uses_palm_world_axis_embedding(palm_basis: str) -> bool:
    return str(palm_basis).strip().lower() in ("middle_thumb", "middle_y", "middle")


def palm_orientation_in_world(Mc_rot: np.ndarray, B_cam: np.ndarray) -> np.ndarray:
    """Map palm basis (columns in camera mm directions) to world orientation: ``Q = M @ B``."""
    M = np.asarray(Mc_rot, dtype=np.float64).reshape(3, 3)
    B = np.asarray(B_cam, dtype=np.float64).reshape(3, 3)
    return M @ B


def world_rotvec_from_palm_delta(
    Mc_rot: np.ndarray,
    B_current: np.ndarray,
    B_arm: np.ndarray,
) -> np.ndarray:
    """World rotation from palm basis change: ``ω = R_to_rotvec(M @ (B B_arm^T) @ M.T)``."""
    M = np.asarray(Mc_rot, dtype=np.float64).reshape(3, 3)
    R_cam = np.asarray(B_current, dtype=np.float64).reshape(3, 3) @ np.asarray(
        B_arm, dtype=np.float64
    ).reshape(3, 3).T
    R_world = M @ R_cam @ M.T
    return np.asarray(R_to_rotvec(R_world), dtype=np.float64).reshape(3)


def palm_world_rotvec_from_basis_delta(
    Mc_rot: np.ndarray | None,
    B_current: np.ndarray,
    B_arm: np.ndarray,
) -> np.ndarray:
    """Palm ΔR → world rotvec; uses frozen ``M`` when available."""
    if Mc_rot is not None:
        return world_rotvec_from_palm_delta(Mc_rot, B_current, B_arm)
    return world_rotvec_from_palm_basis_delta(B_current, B_arm)


def palm_pose_change_rad(
    B_current: np.ndarray,
    B_arm: np.ndarray,
    *,
    Mc_rot: np.ndarray | None = None,
) -> float:
    """Palm rotation angle (rad) vs arm-time basis."""
    rv = palm_world_rotvec_from_basis_delta(Mc_rot, B_current, B_arm)
    return float(np.linalg.norm(rv))


def palm_cam_rotvec_from_basis_delta(B_current: np.ndarray, B_arm: np.ndarray) -> np.ndarray:
    """Intrinsic palm rotation in camera frame (for classify; less false rot on 3D translation)."""
    R = np.asarray(B_current, dtype=np.float64).reshape(3, 3) @ np.asarray(B_arm, dtype=np.float64).reshape(
        3, 3
    ).T
    rv = np.asarray(R_to_rotvec(R), dtype=np.float64).reshape(3)
    if float(np.linalg.norm(rv)) > np.deg2rad(150.0):
        return np.zeros(3, dtype=np.float64)
    return rv


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


def palm_center_translation_components(
    delta_cam: np.ndarray,
    B: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Legacy name: ``B`` is the projection basis (use arm ``ref_basis`` for translation)."""
    return palm_translation_components_mm(delta_cam, B, **kwargs)


def axis_locked_trans_rot_blend_weights(
    delta_world_m: np.ndarray,
    rv_world_rad: np.ndarray,
    *,
    trans_on_m: float,
    rot_on_rad: float,
    rv_cam_rad: np.ndarray | None = None,
    delta_cam_mm: np.ndarray | None = None,
    delta_trans_mm: np.ndarray | None = None,
    secondary_frac: float = 0.30,
    none_below: float = 0.22,
    rot_noise_rad: float = 0.10,
) -> tuple[str, float, float]:
    """Compare normalized translation vs rotation strength; return primary motion + blend weights.

    Rotation score uses **camera-frame** palm twist (``rv_cam``), not ``M @ R @ M.T`` (inflates on
  pans). Translation score uses world m **and** raw palm-center mm so metric scale is not lost.
    """
    rv_w = np.asarray(rv_world_rad, dtype=np.float64).reshape(3)
    rv_c = (
        np.asarray(rv_cam_rad, dtype=np.float64).reshape(3)
        if rv_cam_rad is not None
        else rv_w
    )
    if float(np.linalg.norm(rv_c)) < 1e-9 and float(np.linalg.norm(rv_w)) >= np.deg2rad(35.0):
        # Palm/back transitions can be near 180 deg; the camera-frame classifier suppresses
        # those to avoid old axis flips, but the world pose is still the correct display target.
        rv_c = rv_w
    tw = axis_locked_trans_metric_world_m(
        delta_world_m,
        rv_c,
        trans_on_m=float(trans_on_m),
        rot_on_rad=float(rot_on_rad),
        ignore_world_y_when_rotating=False,
    )
    t_on = float(max(trans_on_m, 1e-6))
    t_on_mm = t_on * 1000.0
    ts = float(tw / t_on)
    pan_mm = 0.0
    if delta_cam_mm is not None:
        dc = np.asarray(delta_cam_mm, dtype=np.float64).reshape(3)
        pan_mm = float(np.linalg.norm(dc))
        ts = max(ts, pan_mm / max(t_on_mm, 1e-6))
    if delta_trans_mm is not None:
        dt = np.asarray(delta_trans_mm, dtype=np.float64).reshape(3)
        gt = float(np.linalg.norm(dt))
        if gt >= 2.5:
            ts = max(ts, gt / max(t_on_mm, 1e-6))
    r_on = float(max(rot_on_rad, 1e-6))
    rv_n = float(np.linalg.norm(rv_c))
    rs = float(rv_n / r_on)
    rn = float(max(rot_noise_rad, 1e-6))
    if rv_n < rn:
        rs = 0.0
    elif rv_n < np.deg2rad(22.0):
        rs *= (rv_n / np.deg2rad(22.0)) ** 2
    if delta_cam_mm is not None:
        if pan_mm >= 18.0:
            rs *= max(0.15, 18.0 / pan_mm)
    if delta_trans_mm is not None:
        gt = float(np.linalg.norm(np.asarray(delta_trans_mm, dtype=np.float64).reshape(3)))
        if gt >= 4.0 and rv_n < np.deg2rad(20.0):
            rs *= max(0.12, 4.0 / max(gt, 4.0))
    nb = float(max(0.15, none_below))
    tw_n = float(np.linalg.norm(np.asarray(delta_world_m, dtype=np.float64).reshape(3)))
    if tw_n < trans_on_m * 0.35 and rv_n < rn * 1.5:
        return "none", 0.0, 0.0
    if ts < nb and rs < nb:
        return "none", 0.0, 0.0
    if rv_n >= np.deg2rad(55.0) and pan_mm <= 24.0 and rs >= nb:
        # The hand can be held away from the press-0 origin while the user rotates it.
        # In that case absolute offset is large, but the current-frame palm motion is small:
        # treat it as pose control and freeze translation to avoid formation swinging.
        return "rotate", 1.0, 0.0
    sec = float(np.clip(secondary_frac, 0.0, 0.65))
    st = max(ts, 1e-9)
    sr = max(rs, 1e-9)
    if delta_trans_mm is not None and float(np.linalg.norm(np.asarray(delta_trans_mm).reshape(3))) >= 3.0:
        if rv_n < np.deg2rad(16.0) and st >= sr * 0.55:
            sr = sr * 0.35
    if sr >= st:
        w_rot = 1.0
        w_trans = sec * min(1.0, ts / sr)
        if ts >= nb:
            w_trans = max(w_trans, 0.2 * min(1.0, ts / nb))
        return "rotate", w_rot, w_trans
    w_trans = 1.0
    w_rot = sec * min(1.0, sr / st)
    if rs >= nb:
        w_rot = max(w_rot, 0.2 * min(1.0, rs / nb))
    return "translate", w_rot, w_trans


def classify_pose_small_translate_vs_rotate(
    rv_pose_world_rad: np.ndarray,
    delta_world_m: np.ndarray,
    *,
    pose_rotate_rad: float,
    trans_on_m: float,
    rv_cam_rad: np.ndarray | None = None,
    rot_excl_ratio: float = 1.25,
    trans_excl_ratio: float = 1.15,
    secondary_frac: float = 0.30,
) -> str:
    """Primary motion label (``rotate`` | ``translate`` | ``none``) from dominance comparison."""
    del rot_excl_ratio, trans_excl_ratio
    mot, _, _ = axis_locked_trans_rot_blend_weights(
        delta_world_m,
        rv_pose_world_rad,
        trans_on_m=float(trans_on_m),
        rot_on_rad=float(pose_rotate_rad),
        rv_cam_rad=rv_cam_rad,
        secondary_frac=float(secondary_frac),
    )
    return mot


def axis_locked_trans_metric_world_m(
    delta_world_m: np.ndarray,
    rv_intrinsic_rad: np.ndarray,
    *,
    trans_on_m: float = 0.009,
    rot_on_rad: float = 0.011,
    ignore_world_y_when_rotating: bool = True,
) -> float:
    """Translation score for classify — drop world Y when palm is already twisting (stops rot→fwd/back leak)."""
    d = np.asarray(delta_world_m, dtype=np.float64).reshape(3).copy()
    r = float(np.linalg.norm(np.asarray(rv_intrinsic_rad, dtype=np.float64).reshape(3)))
    if bool(ignore_world_y_when_rotating) and r >= 0.72 * float(max(rot_on_rad, 1e-9)):
        d[1] = 0.0
    tw = float(np.linalg.norm(d))
    ay, by = abs(float(d[1])), max(abs(float(d[0])), abs(float(d[2])))
    if ay >= float(max(trans_on_m, 1e-6)) * 0.65 and ay >= 1.15 * by:
        tw = max(tw, ay)
    return tw


def classify_axis_locked_motion(
    trans_world_m: np.ndarray,
    rot_world_rad: np.ndarray,
    *,
    trans_on_m: float = 0.009,
    rot_on_rad: float = 0.011,
    rot_excl_ratio: float = 1.32,
    trans_excl_ratio: float = 1.85,
    rv_intrinsic_for_trans: np.ndarray | None = None,
    trans_wins_rot_below_rad: float = 0.95,
) -> str:
    """``translate`` | ``rotate`` | ``none`` — mutually exclusive; rotation uses palm angle in world frame."""
    rv_i = (
        np.asarray(rot_world_rad, dtype=np.float64).reshape(3)
        if rv_intrinsic_for_trans is None
        else np.asarray(rv_intrinsic_for_trans, dtype=np.float64).reshape(3)
    )
    tw = axis_locked_trans_metric_world_m(
        trans_world_m,
        rv_i,
        trans_on_m=float(trans_on_m),
        rot_on_rad=float(rot_on_rad),
    )
    rw = float(np.linalg.norm(np.asarray(rot_world_rad, dtype=np.float64).reshape(3)))
    t_on = float(max(trans_on_m, 1e-6))
    r_on = float(max(rot_on_rad, 1e-6))
    re = float(max(1.05, rot_excl_ratio))
    te = float(max(1.05, trans_excl_ratio))
    ts = tw / t_on
    rs = rw / r_on
    if rw < float(trans_wins_rot_below_rad) * r_on and ts >= 1.0:
        return "translate"
    if rs >= 1.0 and rs >= re * ts:
        return "rotate"
    if ts >= 1.0 and ts >= te * rs:
        return "translate"
    return "none"


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


def rotvec_to_R(v: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector (axis * angle) -> 3x3."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = v / theta
    x, y, z = float(k[0]), float(k[1]), float(k[2])
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), angle in [0, pi]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = float(np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(c))
    rx = float(R[2, 1] - R[1, 2])
    ry = float(R[0, 2] - R[2, 0])
    rz = float(R[1, 0] - R[0, 1])
    v = np.array([rx, ry, rz], dtype=np.float64)
    s = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if s < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-2:
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]], dtype=np.float64)
        k = int(np.argmax(diag))
        axis = np.zeros(3, dtype=np.float64)
        axis[k] = np.sqrt(max(diag[k] + 1.0, 0.0) * 0.5)
        j, m = (k + 1) % 3, (k + 2) % 3
        axis[j] = R[j, k] / max(2.0 * axis[k], 1e-9)
        axis[m] = R[m, k] / max(2.0 * axis[k], 1e-9)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        return axis * theta
    axis = v / (2.0 * np.sin(theta))
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    return axis * theta


def left_hand_pose_matrix_depth_mm(
    result,
    idx_l: int,
    frame_h: int,
    frame_w: int,
    mp_h: int,
    mp_w: int,
    *,
    calibration,
    depth_aligned,
    depth_raw,
    patch_r: int,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
):
    """Wrist + index/middle MCP in **depth-camera mm** (real translation in space).

    MediaPipe ``hand_world_landmarks`` are wrist-centric (wrist does not move when the hand
    translates), so global shift must come from depth unprojection at 2D landmarks mapped
    from the MediaPipe input size to the color frame size.
    """
    if (
        result is None
        or calibration is None
        or idx_l is None
        or not getattr(result, "hand_landmarks", None)
        or idx_l >= len(result.hand_landmarks)
    ):
        return None

    from shared.depth_fusion_utils import read_depth_mm_at_landmark, unproject_to_depth_cam_mm

    hlm = result.hand_landmarks[idx_l]
    if len(hlm) < 21:
        return None

    fh = max(int(frame_h), 1)
    fw = max(int(frame_w), 1)
    mh = max(int(mp_h), 1)
    mw = max(int(mp_w), 1)

    def joint_mm(jidx: int) -> np.ndarray | None:
        lm = hlm[jidx]
        u_mp = float(lm.x) * float(mw)
        v_mp = float(lm.y) * float(mh)
        xc = int(np.clip(round(u_mp * (fw / float(mw))), 0, fw - 1))
        yc = int(np.clip(round(v_mp * (fh / float(mh))), 0, fh - 1))
        d = read_depth_mm_at_landmark(xc, yc, fh, fw, depth_aligned, depth_raw, int(patch_r))
        p = unproject_to_depth_cam_mm(calibration, xc, yc, d, fh, fw, depth_aligned, depth_raw)
        if p is None:
            return None
        return np.asarray(p, dtype=np.float64)

    key = str(palm_basis).strip().lower()
    ia, ib = palm_basis_pair_indices(palm_basis)
    wrist = joint_mm(IDX_WRIST)
    p_ia = joint_mm(ia)
    p_ib = joint_mm(ib)
    t_i = joint_mm(IDX_INDEX_TIP)
    t_m = joint_mm(IDX_MIDDLE_TIP)
    if wrist is None or p_ia is None or p_ib is None:
        return None
    out = np.zeros((21, 3), dtype=np.float64)
    out[IDX_WRIST] = wrist
    out[ia] = p_ia
    out[ib] = p_ib
    if key in ("middle_thumb", "middle_y", "middle"):
        p_thumb = joint_mm(IDX_THUMB_MCP)
        if p_thumb is not None:
            out[IDX_THUMB_MCP] = p_thumb
    for j in (IDX_INDEX_MCP, IDX_MIDDLE_MCP, IDX_RING_MCP, IDX_THUMB_MCP):
        if j == ia or j == ib:
            continue
        pj = joint_mm(j)
        if pj is not None:
            out[j] = pj
    if t_i is not None:
        out[IDX_INDEX_TIP] = t_i
    if t_m is not None:
        out[IDX_MIDDLE_TIP] = t_m
    return out


def hand_points_to_matrix(pts) -> np.ndarray | None:
    if pts is None:
        return None
    if isinstance(pts, np.ndarray) and pts.dtype == object:
        rows = []
        for p in pts:
            v = np.asarray(p, dtype=np.float64).ravel()
            if v.size < 3:
                return None
            rows.append(v[:3])
        h = np.stack(rows, axis=0)
    else:
        h = np.asarray(pts, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 21 or h.shape[1] < 3:
        return None
    return h


def mp_hand_visibility_scores(result, hand_idx: int) -> tuple[float, float]:
    """Return (mean, min) per-joint visibility/presence in [0,1] for a hand index."""
    from shared.mp_hand_utils import extract_landmark_visibilities

    vis = extract_landmark_visibilities(result, hand_idx)
    if vis is None:
        return 0.0, 0.0
    return float(np.mean(vis)), float(np.min(vis))


def palm_basis_from_mp_image_plane(
    result,
    hand_idx: int,
    *,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
) -> np.ndarray | None:
    """Palm frame from 2D landmarks in depth-camera axes (+X right, +Y down, +Z forward).

    Used when depth MCPs are occluded but the hand is still visible in a webcam / color view.
    Wrist and two MCPs lie in the image plane (z=0); ``e3 = e1×e2`` points along +Z_cam.
    """
    if (
        result is None
        or not getattr(result, "hand_landmarks", None)
        or hand_idx is None
        or hand_idx >= len(result.hand_landmarks)
    ):
        return None
    hlm = result.hand_landmarks[hand_idx]
    if len(hlm) < 21:
        return None
    def xy(j: int) -> np.ndarray:
        lm = hlm[j]
        return np.array([float(lm.x), float(lm.y), 0.0], dtype=np.float64)

    key = str(palm_basis).strip().lower()
    if key in ("middle_thumb", "middle_y", "middle"):
        h2 = np.zeros((21, 3), dtype=np.float64)
        for j in range(21):
            h2[j] = xy(j)
        out = palm_orthonormal_basis_middle_y_thumb_x(h2)
        return None if out is None else out[1]

    ia, ib = palm_basis_pair_indices(palm_basis)
    wrist = xy(IDX_WRIST)
    v1 = xy(ia) - wrist
    v2 = xy(ib) - wrist
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    e1 = v1 / n1
    v2o = v2 - float(np.dot(v2, e1)) * e1
    ne2 = float(np.linalg.norm(v2o))
    if ne2 < 1e-6:
        return None
    e2 = v2o / ne2
    e3 = np.cross(e1, e2)
    ne3 = float(np.linalg.norm(e3))
    if ne3 < 1e-6:
        return None
    e3 = e3 / ne3
    B = np.stack([e1, e2, e3], axis=1)
    h2 = np.zeros((21, 3), dtype=np.float64)
    for j in range(21):
        h2[j] = xy(j)
    return _enforce_thumb_positive_x(B, h2, wrist)


def orthonormal_basis_from_landmark_pair(
    h: np.ndarray, ia: int, ib: int, *, min_edge_mm: float = 1e-9
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (wrist, B) from two landmarks minus wrist (camera mm)."""
    wrist = h[IDX_WRIST, :3]
    v1 = h[int(ia), :3] - wrist
    v2 = h[int(ib), :3] - wrist
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < float(min_edge_mm) or n2 < float(min_edge_mm):
        return None
    e1 = v1 / n1
    e2 = v2 - float(np.dot(v2, e1)) * e1
    ne2 = float(np.linalg.norm(e2))
    if ne2 < float(min_edge_mm):
        return None
    e2 = e2 / ne2
    e3 = np.cross(e1, e2)
    ne3 = float(np.linalg.norm(e3))
    if ne3 < float(min_edge_mm):
        return None
    e3 = e3 / ne3
    B = np.stack([e1, e2, e3], axis=1)
    return wrist, B


def _segment_axis(h: np.ndarray, wrist: np.ndarray, mcp_idx: int, tip_idx: int) -> np.ndarray | None:
    """Unit vector wrist → landmark; prefer tip when both segments are well observed."""
    v_mcp = np.asarray(h[int(mcp_idx), :3], dtype=np.float64).reshape(3) - wrist
    v_tip = np.asarray(h[int(tip_idx), :3], dtype=np.float64).reshape(3) - wrist
    n_mcp, n_tip = float(np.linalg.norm(v_mcp)), float(np.linalg.norm(v_tip))
    v = v_mcp
    if n_tip >= 15.0 and n_mcp >= 15.0 and 0.45 < (n_tip / max(n_mcp, 1e-9)) < 2.2:
        v = v_tip
    nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        return None
    return v / nv


def _middle_finger_axis(h: np.ndarray, wrist: np.ndarray) -> np.ndarray | None:
    """+Y palm: middle finger/MCP → wrist (tip when reliable), stable for palm/back."""
    axis = _segment_axis(h, wrist, IDX_MIDDLE_MCP, IDX_MIDDLE_TIP)
    return None if axis is None else -axis


def _thumb_lateral_axis(h: np.ndarray, wrist: np.ndarray) -> np.ndarray | None:
    """+X palm: thumb direction (tip when reliable)."""
    return _segment_axis(h, wrist, IDX_THUMB_MCP, IDX_THUMB_TIP)


def palm_components_in_camera_mm(delta_cam: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Project camera-mm wrist delta onto palm columns (x=thumb, y=middle→wrist, z=X×Y)."""
    Bc = np.asarray(B, dtype=np.float64).reshape(3, 3)
    d = np.asarray(delta_cam, dtype=np.float64).reshape(3)
    return np.array([float(np.dot(d, Bc[:, i])) for i in range(3)], dtype=np.float64)


def palm_vector_palm_to_world(v: np.ndarray) -> np.ndarray:
    """Map palm-frame vector (x,y,z) to world/sim (X,Y,Z) via :data:`PALM_AXIS_TO_WORLD_PERM`."""
    p = np.asarray(v, dtype=np.float64).reshape(3)
    wx, wy, wz = PALM_AXIS_TO_WORLD_PERM
    return np.array([float(p[wx]), float(p[wy]), float(p[wz])], dtype=np.float64)


def palm_components_to_world_m(
    comp_palm: np.ndarray,
    *,
    trans_scale: float,
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Palm-frame mm components (x thumb, y middle→wrist, z=X×Y) → world m."""
    world_mm = palm_vector_palm_to_world(np.asarray(comp_palm, dtype=np.float64).reshape(3))
    sign = np.asarray(axis_sign, dtype=np.float64).reshape(3)
    return float(trans_scale) * world_mm * sign


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


def _enforce_thumb_positive_x(B: np.ndarray, h: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Palm +X toward thumb; keep +Y fixed and recompute +Z = X×Y (right-handed)."""
    out = np.asarray(B, dtype=np.float64).reshape(3, 3).copy()
    thumb = _thumb_lateral_axis(h, wrist)
    if thumb is None:
        thumb = np.asarray(h[IDX_THUMB_MCP, :3], dtype=np.float64).reshape(3) - wrist
    if float(np.dot(out[:, 0], thumb)) < 0.0:
        out[:, 0] *= -1.0
    ez = np.cross(out[:, 0], out[:, 1])
    nez = float(np.linalg.norm(ez))
    if nez > 1e-9:
        out[:, 2] = ez / nez
    return out


def _project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Remove component along ``plane_normal``."""
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return np.asarray(v, dtype=np.float64).reshape(3).copy()
    n = n / nn
    return np.asarray(v, dtype=np.float64).reshape(3) - float(np.dot(v, n)) * n


def _build_palm_basis_middle_y_thumb_x(
    ey: np.ndarray,
    h: np.ndarray,
    wrist: np.ndarray,
) -> np.ndarray | None:
    """Orthonormal palm basis (camera mm): **+Y** middle→wrist, **+X** thumb lateral, **+Z** = X×Y.

    ``+Y`` is never flipped to chase camera/ref continuity; palm vs back is represented by
    the resulting ``+Z`` sign. ``+X`` is chosen on the thumb side.
    """
    ey_u = np.asarray(ey, dtype=np.float64).reshape(3)
    ney = float(np.linalg.norm(ey_u))
    if ney < 1e-9:
        return None
    ey_u = ey_u / ney
    thumb = _thumb_lateral_axis(h, wrist)
    if thumb is None:
        thumb = np.asarray(h[IDX_THUMB_MCP, :3], dtype=np.float64).reshape(3) - wrist
    ex = _project_onto_plane(thumb, ey_u)
    nex = float(np.linalg.norm(ex))
    if nex < 1e-6:
        index = np.asarray(h[IDX_INDEX_MCP, :3], dtype=np.float64).reshape(3) - wrist
        ex = _project_onto_plane(index, ey_u)
        nex = float(np.linalg.norm(ex))
    if nex < 1e-9:
        return None
    ex = ex / nex
    ez = np.cross(ex, ey_u)
    nez = float(np.linalg.norm(ez))
    if nez < 1e-9:
        return None
    ez = ez / nez
    B = np.stack([ex, ey_u, ez], axis=1)
    return _enforce_thumb_positive_x(B, h, wrist)


def align_palm_basis_to_reference(
    B: np.ndarray,
    B_ref: np.ndarray,
    h: np.ndarray,
    wrist: np.ndarray,
) -> np.ndarray:
    """Rebuild with physical +Y preserved; palm/back changes appear as +Z changes."""
    del B_ref
    ey = np.asarray(B[:, 1], dtype=np.float64).reshape(3).copy()
    rebuilt = _build_palm_basis_middle_y_thumb_x(ey, h, wrist)
    if rebuilt is not None:
        return rebuilt
    return _enforce_thumb_positive_x(B, h, wrist)


def palm_orthonormal_basis_middle_y_thumb_x(
    h: np.ndarray,
    *,
    ref_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Palm frame (camera mm): **+Y** middle→wrist, **+X** thumb lateral, **+Z** = X×Y."""
    wrist = np.asarray(h[IDX_WRIST, :3], dtype=np.float64).reshape(3)
    ey = _middle_finger_axis(h, wrist)
    if ey is None:
        return None
    B = _build_palm_basis_middle_y_thumb_x(ey, h, wrist)
    if B is None:
        return None
    if ref_basis is not None:
        B = align_palm_basis_to_reference(B, ref_basis, h, wrist)
    pc = palm_center_mm(h)
    if pc is None:
        pc = wrist
    return np.asarray(pc, dtype=np.float64).reshape(3), B


def palm_orthonormal_basis(
    h: np.ndarray,
    *,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
    ref_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (palm_center_mm, B) with B columns = palm X (thumb), Y (middle→wrist), Z."""
    key = str(palm_basis).strip().lower()
    if key in ("middle_thumb", "middle_y", "middle"):
        return palm_orthonormal_basis_middle_y_thumb_x(h, ref_basis=ref_basis)
    ia, ib = palm_basis_pair_indices(palm_basis)
    out = orthonormal_basis_from_landmark_pair(h, ia, ib)
    if out is None:
        return None
    origin, B = out
    if ref_basis is not None:
        wrist = np.asarray(h[IDX_WRIST, :3], dtype=np.float64).reshape(3)
        B = align_palm_basis_to_reference(B, ref_basis, h, wrist)
    pc = palm_center_mm(h)
    if pc is None:
        pc = origin
    return np.asarray(pc, dtype=np.float64).reshape(3), B


def fingertip_orthonormal_basis(h: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Palm-style basis from index + middle fingertip (better for optical-axis twist)."""
    return orthonormal_basis_from_landmark_pair(
        h, IDX_INDEX_TIP, IDX_MIDDLE_TIP, min_edge_mm=_MIN_FINGERTIP_REL_MM
    )


@dataclass
class LeftSwarmPoseState:
    """Tracks reference palm frame and outputs smoothed world offset + full rotation."""

    enabled: bool = True
    initialized: bool = False
    ref_wrist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ref_basis: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    ema_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ema_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Smooth return-to-morph-frame after disarm (seconds); 0 = not unwinding
    unwind_end_t: float = 0.0
    unwind_duration: float = 0.0
    unwind_off0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    unwind_rv0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Previous palm center (camera mm), used for velocity/glitch checks.
    prev_wrist: np.ndarray | None = None
    #: Palm basis at previous frame (3×3), kept for legacy/full incremental helpers.
    prev_basis: np.ndarray | None = None
    #: Hand span (mm) at previous frame for incremental push/pull scale.
    prev_hand_span_mm: float = 0.0
    #: When set (``camera_at_arm``), maps gated camera mm → sim m; frozen at press-0, not updated per frame.
    frozen_M_rot: np.ndarray | None = None
    frozen_M_trans: np.ndarray | None = None
    frozen_cam_preset: str = ""
    #: Per-drone morph XYZ at arm (sim m) for ``per_drone`` rotation pivot (self-rotation).
    ref_drone_xyz: np.ndarray | None = None
    #: 2D/webcam palm basis at arm when dual-rotation fallback is enabled.
    ref_basis_image: np.ndarray | None = None
    #: Fingertip orthonormal basis at arm (3×3); legacy, optional.
    ref_fingertip_basis: np.ndarray | None = None
    #: Mean wrist→landmark span (mm) at arm; detects push/pull via apparent hand size.
    ref_hand_span_mm: float = 0.0
    #: Palm centroid (camera mm) at arm — translation origin.
    ref_palm_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Last ``axis_locked`` motion class: ``translate`` | ``rotate`` | ``none`` (HUD/debug).
    last_axis_motion: str = "none"
    #: Palm basis ΔR this frame (world rad); for HUD — may differ from applied ``rv_cmd``.
    last_rv_pose_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_rv_cmd_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_h_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: Blend weights last frame (0–1): rotation / translation kept after dominance split.
    last_rot_blend_w: float = 1.0
    last_trans_blend_w: float = 1.0
    last_delta_cam_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_cam_arm_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_h_raw_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_palm_center_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_delta_trans_palm_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_rv_cam_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_trans_cam_gated: bool = False

    def reset_to_current(
        self,
        h: np.ndarray,
        *,
        palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
        sim_from_cam: np.ndarray | None = None,
        sim_trans_from_cam: np.ndarray | None = None,
        cam_preset_label: str = "",
        ref_drone_xyz: np.ndarray | None = None,
        ref_basis_image: np.ndarray | None = None,
    ) -> bool:
        out = palm_orthonormal_basis(h, palm_basis=palm_basis)
        if out is None:
            return False
        origin, B = out
        wrist = np.asarray(h[IDX_WRIST, :3], dtype=np.float64).reshape(3)
        B = _enforce_thumb_positive_x(B, h, wrist)
        pc = palm_center_mm(h)
        if pc is None:
            pc = np.asarray(origin, dtype=np.float64).reshape(3)
        self.ref_palm_center = np.asarray(pc, dtype=np.float64).reshape(3).copy()
        self.ref_wrist = np.asarray(origin, dtype=np.float64).reshape(3).copy()
        self.ref_basis = B.copy()
        self.initialized = True
        self.ema_offset[:] = 0.0
        self.ema_rotvec[:] = 0.0
        self.prev_wrist = self.ref_palm_center.copy()
        self.prev_basis = B.copy()
        self.prev_hand_span_mm = float(self.ref_hand_span_mm)
        if sim_from_cam is not None:
            self.frozen_M_rot = np.asarray(sim_from_cam, dtype=np.float64).reshape(3, 3).copy()
            if sim_trans_from_cam is not None:
                self.frozen_M_trans = np.asarray(sim_trans_from_cam, dtype=np.float64).reshape(3, 3).copy()
            else:
                self.frozen_M_trans = self.frozen_M_rot.copy()
            self.frozen_cam_preset = str(cam_preset_label)
        else:
            self.frozen_M_rot = None
            self.frozen_M_trans = None
            self.frozen_cam_preset = ""
        if ref_drone_xyz is not None:
            rd = np.asarray(ref_drone_xyz, dtype=np.float64)
            if rd.ndim == 2 and rd.shape[1] >= 3:
                self.ref_drone_xyz = rd[:, :3].copy()
        else:
            self.ref_drone_xyz = None
        if ref_basis_image is not None:
            self.ref_basis_image = np.asarray(ref_basis_image, dtype=np.float64).reshape(3, 3).copy()
        else:
            self.ref_basis_image = None
        out_ft = fingertip_orthonormal_basis(h)
        self.ref_fingertip_basis = out_ft[1].copy() if out_ft is not None else None
        self.ref_hand_span_mm = float(hand_palm_span_mm(h, wrist))
        return True

    def is_unwinding(self) -> bool:
        return float(self.unwind_end_t) > 0.0 and time.monotonic() < float(self.unwind_end_t)

    def begin_unwind(self, duration_s: float) -> None:
        """Fade rigid offset/rotation to identity over ``duration_s`` (smoothstep)."""
        d = float(max(duration_s, 1e-3))
        self.unwind_off0 = np.asarray(self.ema_offset, dtype=np.float64).copy()
        self.unwind_rv0 = np.asarray(self.ema_rotvec, dtype=np.float64).copy()
        self.unwind_duration = d
        self.unwind_end_t = time.monotonic() + d

    def cancel_unwind(self) -> None:
        """Abort smooth restore (e.g. user re-arms); clears offset like disarm."""
        self.unwind_end_t = 0.0
        self.unwind_duration = 0.0
        self.unwind_off0[:] = 0.0
        self.unwind_rv0[:] = 0.0
        self.initialized = False
        self.ema_offset[:] = 0.0
        self.ema_rotvec[:] = 0.0
        self.prev_wrist = None
        self.prev_basis = None
        self.prev_hand_span_mm = 0.0
        self.frozen_M_rot = None
        self.frozen_M_trans = None
        self.frozen_cam_preset = ""
        self.ref_drone_xyz = None
        self.ref_basis_image = None
        self.ref_fingertip_basis = None


def _clear_frozen_cam_to_sim(state: LeftSwarmPoseState) -> None:
    state.frozen_M_rot = None
    state.frozen_M_trans = None
    state.frozen_cam_preset = ""
    state.ref_drone_xyz = None
    state.ref_basis_image = None
    state.ref_fingertip_basis = None


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


def disarm_left_swarm_pose(state: LeftSwarmPoseState) -> None:
    """Exit move mode: drop reference so the next arm captures a fresh baseline."""
    state.unwind_end_t = 0.0
    state.unwind_duration = 0.0
    state.unwind_off0[:] = 0.0
    state.unwind_rv0[:] = 0.0
    state.initialized = False
    state.ema_offset[:] = 0.0
    state.ema_rotvec[:] = 0.0
    state.prev_wrist = None
    state.prev_basis = None
    state.prev_hand_span_mm = 0.0
    _clear_frozen_cam_to_sim(state)


def update_left_swarm_pose(
    pts_l,
    state: LeftSwarmPoseState,
    *,
    trans_scale: float,
    rot_scale: float,
    trans_ema: float,
    rot_ema: float,
    max_offset_m: float,
    max_rot_rad: float,
    axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hand_lost_decay: float = 0.92,
    force_reset: bool = False,
    cam_delta_to_world: np.ndarray | None = None,
    cam_translation_to_world: np.ndarray | None = None,
    rot_gate_rad: float = 0.11,
    yaw_min_horiz_norm: float = 0.28,
    rot_gain: float = 1.0,
    rot_trans_tau_mm: float = 0.0,
    rot_world_z_scale: float = 1.0,
    depth_dom_ratio: float = 0.0,
    depth_dom_min_mm: float = 0.0,
    palm_basis: str = DEFAULT_LEFT_PALM_BASIS,
    rot_coex_trans_min_mm: float = 0.0,
    rot_coex_max_angle_rad: float = 0.0,
    vertical_dom_ratio: float = 0.0,
    vertical_dom_min_mm: float = 0.0,
    trans_boost_optical: float = 1.0,
    rot_planar_dom_ratio: float = 0.0,
    rot_planar_dom_min_mm: float = 0.0,
    lateral_dom_ratio: float = 0.0,
    lateral_dom_min_mm: float = 0.0,
    trans_xy_dom_strip_dz_ratio: float = 0.0,
    trans_xy_dom_strip_dz_min_h_mm: float = 0.0,
    trans_dx_dom_strip_dz_ratio: float = 0.0,
    trans_dx_dom_strip_dz_min_dx_mm: float = 0.0,
    vertical_optical_preserve_ratio: float = 0.0,
    vertical_optical_preserve_min_mm: float = 0.0,
    rot_palm_face_twist_world_y: bool = False,
    rot_palm_face_cos_align_min: float = 0.78,
    rot_palm_face_twist_dom_ratio: float = 1.0,
    rot_palm_face_twist_min_rad: float = 0.0,
    rot_palm_face_twist_world_y_sign: float = 1.0,
    arm_sim_from_cam: np.ndarray | None = None,
    arm_sim_trans_from_cam: np.ndarray | None = None,
    arm_cam_preset_label: str = "",
    ref_drone_xyz: np.ndarray | None = None,
    ref_basis_image: np.ndarray | None = None,
    B_rot: np.ndarray | None = None,
    rot_ref_basis: np.ndarray | None = None,
    control_style: str = "full",
    axis_trans_deadzone_m: float = 0.018,
    axis_rot_deadzone_rad: float = 0.055,
    axis_cam_deadzone_xyz_mm: tuple[float, float, float] = (5.0, 5.0, 14.0),
    axis_cam_snap_min_ratio: float = 1.2,
    axis_trans_on_m: float = 0.009,
    axis_rot_on_rad: float = 0.011,
    axis_rot_excl_ratio: float = 1.32,
    axis_trans_excl_ratio: float = 1.85,
    axis_rot_boost: float = 2.0,
    axis_forward_boost: float = 1.38,
    axis_lateral_min_dx_mm: float = 7.0,
    axis_forward_min_dz_mm: float = 4.0,
    axis_forward_xy_small_mm: float = 10.0,
    axis_scale_strong_min_mm: float = 7.0,
    axis_scale_strong_rel: float = 0.07,
    axis_scale_gain: float = 1.1,
    axis_depth_strong_min_dz_mm: float = 9.0,
    axis_depth_strong_dom_ratio: float = 0.38,
    axis_rot_hold_frac: float = 0.0,
    axis_rot_trans_ema_decay: float = 0.15,
    axis_trans_rot_coupling: float = 0.30,
    axis_rot_step_rad: float = 0.32,
    axis_palm_face_cos_min: float = 0.72,
    axis_palm_face_min_gesture_rad: float = 0.032,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (offset_m, R_3x3) in simulation/world frame to apply to all target points.

    ``cam_delta_to_world`` (3×3, orthonormal): palm rotation ``B @ ref_basis.T`` is mapped by
    ``M @ R_cam @ M.T``; rotation command uses ``ω = R_to_rotvec(M @ R_cam @ M.T)``, unless
    ``rot_palm_face_twist_world_y`` applies (``fwd_y`` + depth frame): when the palm is nearly
    parallel to the image plane and ``R_cam`` is dominated by twist about camera ``+Z``, use
    ``ω_world = [0, θ, 0]`` with ``θ = atan2(R_cam[1,0], R_cam[0,0])`` (in-plane twist → rotation
    about world **Y** under ``fwd_y``). That angle is folded into ``ema_rotvec`` like other rotation.

    ``cam_translation_to_world`` (3×3): wrist delta mm → world m for **translation** only; if
    ``None``, uses ``cam_delta_to_world`` (same as legacy).

    ``yaw_min_horiz_norm`` is unused (kept for CLI compatibility).

    ``rot_trans_tau_mm``: if >0, rotation command is multiplied by ``exp(-Δwrist_mm / tau)`` (0=off).

    ``rot_world_z_scale``: multiplier on the world axis–angle **Z** component after ``ω`` is formed.

    ``depth_dom_ratio``: if ``>0`` and ``|dz| >= depth_dom_min_mm`` and ``|dz| >= ratio * hypot(dx,dy)``,
    translation uses only ``[0,0,dz]`` in camera mm (``0`` = disabled).

    ``palm_basis``: ``index_middle`` | ``index_ring`` | ``middle_ring`` — two MCPs minus wrist
    define the palm frame (default ``index_ring`` for a wider baseline than index/middle).

    ``rot_coex_trans_min_mm``: if ``>0`` and wrist step (mm/frame) ≥ this and intrinsic
    ``‖R_to_rotvec(R_cam)‖ < rot_coex_max_angle_rad``, rotation command is zeroed for that frame
    (``rot_coex_max_angle_rad`` should be >0 when coex is used).

    ``vertical_dom_ratio`` / ``vertical_dom_min_mm``: when ``|dy|`` dominates ``hypot(dx,dz)``,
    translation uses only ``[0, dy, 0]`` in camera mm (pairs with ``image_y_to_world_z``).

    ``vertical_optical_preserve_*``: when ``|dz|`` dominates ``hypot(dx,dy)``, skip vertical ``dz``
    stripping (keeps near/far translation when perspective adds ``dy``).

    ``trans_boost_optical``: multiplier on camera ``dz`` (after dom gates) before ``M_trans``.

    ``trans_xy_dom_strip_dz_*`` / ``trans_dx_dom_strip_dz_*``: strip ``dz`` **before** ``depth_dom``
    (see module docstring). ``trans_dx_*`` catches slow lateral motion where per-frame ``hypot(dx,dy)``
    is small but ``|dx|`` still exceeds ``|dz|`` noise.

    ``lateral_dom_ratio`` / ``lateral_dom_min_mm``: when ``|dx|`` dominates ``hypot(dy,dz)`` in
    camera mm (after ``depth_dom``), translation uses only ``[dx,0,0]``.

    ``rot_planar_dom_ratio`` / ``rot_planar_dom_min_mm``: when image-plane ``hypot(dx,dy)`` dominates
    ``|dz|`` (with a small ``|dz|`` floor in the test), zero palm rotation for that frame (``ratio`` 0 = off).
    Skipped when the palm-face twist branch fired.

    ``rot_palm_face_twist_world_y``: optional palm–image-plane + optical twist path (see module
    docstring). **Skipped** when ``translation_plane_dominates_depth`` is true for the same frame’s
    ``delta_trans`` (image-plane pan: avoids twist noise while translating). Set
    ``rot_palm_face_twist_dom_ratio`` to ``0`` to disable the twist test (only alignment).
    ``rot_palm_face_twist_world_y_sign``: ``±1`` on ``θ`` when the palm-face path is enabled.
    That branch skips ``rot_planar_dom_*``, ``rot_coex_*``, ``rot_trans_tau_mm``, and ``rot_gate_rad``
    for that frame (per-frame ``θ`` can be small; smoothing remains via ``rot_ema`` / ``max_rot_rad``).

    ``control_style``: ``axis_locked`` — press-0 sets the reference palm center + pose
    (``ema_*`` cleared, frozen cam→sim). Each frame uses the current hand pose **relative to that
    reference**; commanded motion snaps to **world ±X/±Y/±Z**. ``full`` keeps the legacy multi-axis blend.
    """
    del yaw_min_horiz_norm  # API compat
    axis_locked = str(control_style).strip().lower() in ("axis_locked", "axis", "principal")
    if not state.enabled:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    now = time.monotonic()
    if float(state.unwind_end_t) > 0.0:
        if now >= float(state.unwind_end_t):
            state.unwind_end_t = 0.0
            state.unwind_duration = 0.0
            state.initialized = False
            state.ema_offset[:] = 0.0
            state.ema_rotvec[:] = 0.0
            state.prev_wrist = None
            state.prev_basis = None
            state.prev_hand_span_mm = 0.0
            _clear_frozen_cam_to_sim(state)
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        u = float(
            np.clip(
                (now - (state.unwind_end_t - state.unwind_duration)) / max(state.unwind_duration, 1e-9),
                0.0,
                1.0,
            )
        )
        s = u * u * (3.0 - 2.0 * u)
        a = 1.0 - s
        off = state.unwind_off0 * a
        rv = state.unwind_rv0 * a
        state.ema_offset[:] = off
        state.ema_rotvec[:] = rv
        return off.astype(np.float64), rotvec_to_R(rv)

    h = hand_points_to_matrix(pts_l)
    sign = np.asarray(axis_sign, dtype=np.float64).reshape(3)

    if force_reset and h is not None:
        state.reset_to_current(
            h,
            palm_basis=palm_basis,
            sim_from_cam=arm_sim_from_cam,
            sim_trans_from_cam=arm_sim_trans_from_cam,
            cam_preset_label=str(arm_cam_preset_label),
            ref_drone_xyz=ref_drone_xyz,
            ref_basis_image=ref_basis_image,
        )

    if h is None:
        state.prev_wrist = None
        state.prev_basis = None
        state.prev_hand_span_mm = 0.0
        ld = float(np.clip(hand_lost_decay, 0.0, 1.0))
        state.ema_offset *= ld
        state.ema_rotvec *= ld
        return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)

    ref_b = state.ref_basis if state.initialized else None
    out = palm_orthonormal_basis(h, palm_basis=palm_basis, ref_basis=ref_b)
    if out is None:
        state.prev_wrist = None
        state.prev_basis = None
        state.prev_hand_span_mm = 0.0
        ld = float(np.clip(hand_lost_decay, 0.0, 1.0))
        state.ema_offset *= ld
        state.ema_rotvec *= ld
        return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)

    palm_center, B_depth = out
    wrist = np.asarray(h[IDX_WRIST, :3], dtype=np.float64).reshape(3)
    B = np.asarray(B_rot, dtype=np.float64).reshape(3, 3) if B_rot is not None else B_depth
    if not state.initialized:
        if not state.reset_to_current(h, palm_basis=palm_basis):
            state.prev_wrist = None
            state.prev_basis = None
            state.prev_hand_span_mm = 0.0
            ld = float(np.clip(hand_lost_decay, 0.0, 1.0))
            state.ema_offset *= ld
            state.ema_rotvec *= ld
            return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    delta_cam_arm = np.asarray(palm_center, dtype=np.float64).reshape(3) - state.ref_palm_center
    if state.prev_wrist is not None:
        delta_cam = np.asarray(palm_center, dtype=np.float64).reshape(3) - state.prev_wrist
    else:
        delta_cam = np.zeros(3, dtype=np.float64)
    _max_frame_mm = 140.0
    _jump_resync_mm = 95.0
    _abs_resync_mm = 850.0
    _dn = float(np.linalg.norm(delta_cam))
    _dan = float(np.linalg.norm(delta_cam_arm))
    if _dn > _jump_resync_mm or _dan > _abs_resync_mm:
        state.prev_wrist = np.asarray(palm_center, dtype=np.float64).reshape(3).copy()
        state.prev_basis = np.asarray(B_depth, dtype=np.float64).reshape(3, 3).copy()
        state.prev_hand_span_mm = float(hand_palm_span_mm(h, wrist))
        state.last_palm_center_mm = np.asarray(palm_center, dtype=np.float64).reshape(3).copy()
        state.last_delta_cam_mm = np.zeros(3, dtype=np.float64)
        state.last_delta_cam_arm_mm = np.asarray(delta_cam_arm, dtype=np.float64).reshape(3).copy()
        state.last_delta_trans_palm_mm = np.zeros(3, dtype=np.float64)
        state.last_delta_h_raw_m = np.zeros(3, dtype=np.float64)
        state.last_delta_h_world = np.zeros(3, dtype=np.float64)
        state.last_rv_cmd_world = np.zeros(3, dtype=np.float64)
        state.last_axis_motion = "none"
        state.last_rot_blend_w = 0.0
        state.last_trans_blend_w = 0.0
        return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)
    if _dn > _max_frame_mm:
        delta_cam = delta_cam * (_max_frame_mm / _dn)
    state.last_delta_cam_arm_mm = np.asarray(delta_cam_arm, dtype=np.float64).reshape(3).copy()
    B_prev = (
        np.asarray(state.prev_basis, dtype=np.float64).reshape(3, 3)
        if state.prev_basis is not None
        else state.ref_basis
    )
    delta_cam_control = delta_cam_arm if axis_locked else delta_cam
    B_motion_ref = state.ref_basis if axis_locked else B_prev
    B_rot_ref = B_motion_ref
    if B_rot is not None and state.ref_basis_image is not None:
        B_rot_ref = state.ref_basis_image
    palm_world_embed = uses_palm_world_axis_embedding(palm_basis)
    if state.frozen_M_rot is not None:
        Mc_rot = np.asarray(state.frozen_M_rot, dtype=np.float64).reshape(3, 3)
        if state.frozen_M_trans is not None:
            Mc_trans = np.asarray(state.frozen_M_trans, dtype=np.float64).reshape(3, 3)
        else:
            Mc_trans = Mc_rot
    else:
        M_rot = cam_delta_to_world
        Mc_rot = np.asarray(M_rot, dtype=np.float64).reshape(3, 3) if M_rot is not None else None
        M_tr = cam_translation_to_world
        if M_tr is not None:
            Mc_trans = np.asarray(M_tr, dtype=np.float64).reshape(3, 3)
        else:
            Mc_trans = Mc_rot

    if axis_locked:
        if palm_world_embed and Mc_trans is not None:
            state.last_trans_cam_gated = True
            delta_trans = axis_locked_gated_cam_delta_mm(
                delta_cam_control,
                lateral_min_dx_mm=float(axis_lateral_min_dx_mm),
                forward_min_dz_mm=float(axis_forward_min_dz_mm),
                forward_xy_small_mm=float(axis_forward_xy_small_mm),
                vertical_min_dy_mm=max(5.0, float(axis_lateral_min_dx_mm) * 0.85),
                depth_strong_min_dz_mm=float(axis_depth_strong_min_dz_mm),
                depth_strong_dom_ratio=float(axis_depth_strong_dom_ratio),
            )
        elif palm_world_embed:
            state.last_trans_cam_gated = False
            span_now = float(hand_palm_span_mm(h, wrist))
            if state.ref_hand_span_mm > 1e-6:
                scale_delta = float(np.clip(span_now - state.ref_hand_span_mm, -35.0, 35.0))
            else:
                scale_delta = hand_scale_toward_camera_mm(span_now, float(state.ref_hand_span_mm))
            delta_trans = palm_translation_components_mm(
                delta_cam_control,
                B_depth,
                scale_delta_mm=float(scale_delta),
                span_ref_mm=float(state.ref_hand_span_mm),
                scale_min_mm=float(axis_scale_strong_min_mm),
                scale_min_rel=float(axis_scale_strong_rel),
                scale_gain=float(axis_scale_gain),
                lateral_min_mm=min(4.0, float(axis_lateral_min_dx_mm) * 0.55),
                vertical_min_mm=3.0,
            )
        else:
            delta_trans = axis_locked_gated_cam_delta_mm(
                delta_cam_control,
                lateral_min_dx_mm=float(axis_lateral_min_dx_mm),
                forward_min_dz_mm=float(axis_forward_min_dz_mm),
                depth_strong_min_dz_mm=float(axis_depth_strong_min_dz_mm),
                depth_strong_dom_ratio=float(axis_depth_strong_dom_ratio),
            )
            if abs(float(delta_trans[2])) > 1e-9 and abs(float(delta_trans[0])) < 1e-6:
                delta_trans = np.asarray(delta_trans, dtype=np.float64).reshape(3).copy()
                delta_trans[2] *= float(max(1.0, axis_forward_boost))
    else:
        d_plane = vertical_dom_zero_optical_z(
            delta_cam,
            vertical_dom_ratio=float(vertical_dom_ratio),
            vertical_dom_min_mm=float(vertical_dom_min_mm),
            optical_preserve_ratio=float(vertical_optical_preserve_ratio),
            optical_preserve_min_mm=float(vertical_optical_preserve_min_mm),
        )
        d_work = strip_dz_when_xy_motion_dominates(
            d_plane,
            ratio=float(trans_xy_dom_strip_dz_ratio),
            min_hypot_xy_mm=float(trans_xy_dom_strip_dz_min_h_mm),
        )
        d_work = strip_dz_when_abs_dx_dominates_dz(
            d_work,
            ratio=float(trans_dx_dom_strip_dz_ratio),
            min_abs_dx_mm=float(trans_dx_dom_strip_dz_min_dx_mm),
        )
        delta_trans = effective_cam_delta_for_translation(
            d_work,
            depth_dom_ratio=float(depth_dom_ratio),
            depth_dom_min_mm=float(depth_dom_min_mm),
        )
        delta_trans = lateral_dom_keep_only_dx(
            delta_trans,
            lateral_dom_ratio=float(lateral_dom_ratio),
            lateral_dom_min_mm=float(lateral_dom_min_mm),
        )
        boost = float(trans_boost_optical)
        if boost > 0.0 and boost != 1.0:
            delta_trans = np.asarray(delta_trans, dtype=np.float64).reshape(3).copy()
            delta_trans[2] *= boost

    rv_pose_world = np.zeros(3, dtype=np.float64)
    rv_cam = np.zeros(3, dtype=np.float64)
    if palm_world_embed and state.initialized:
        rv_cam = palm_cam_rotvec_from_basis_delta(B, B_rot_ref)
        rv_pose_world = palm_world_rotvec_from_basis_delta(Mc_rot, B, B_rot_ref)

    state.last_palm_center_mm = np.asarray(palm_center, dtype=np.float64).reshape(3).copy()
    state.last_delta_trans_palm_mm = np.asarray(delta_trans, dtype=np.float64).reshape(3).copy()

    if axis_locked and palm_world_embed and Mc_trans is not None:
        delta_h_raw = float(trans_scale) * (Mc_trans @ np.asarray(delta_trans, dtype=np.float64).reshape(3)) * sign
    elif axis_locked and palm_world_embed:
        delta_h_raw = palm_components_to_world_m(
            delta_trans,
            trans_scale=float(trans_scale),
            axis_sign=tuple(float(s) for s in sign.reshape(3)),
        )
    elif Mc_trans is not None:
        delta_h_raw = float(trans_scale) * (Mc_trans @ delta_trans) * sign
    else:
        delta_h_raw = delta_cam_control * sign * float(trans_scale)

    vel_mm = 0.0
    if state.prev_wrist is not None:
        vel_mm = float(np.linalg.norm(np.asarray(palm_center).reshape(3) - state.prev_wrist))

    planar_trans_pan = False
    rp_chk = float(rot_planar_dom_ratio)
    if rp_chk > 0.0 and translation_plane_dominates_depth(
        delta_trans,
        planar_ratio=rp_chk,
        planar_min_mm=float(rot_planar_dom_min_mm),
    ):
        planar_trans_pan = True

    ref_b = state.ref_basis
    if rot_ref_basis is not None:
        ref_b = np.asarray(rot_ref_basis, dtype=np.float64).reshape(3, 3)
    elif B_rot is not None and state.ref_basis_image is not None:
        ref_b = state.ref_basis_image
    R_cam = B @ ref_b.T
    used_palm_face_twist = False
    rpf = float(rot_palm_face_twist_dom_ratio)
    if (
        (not axis_locked)
        and bool(rot_palm_face_twist_world_y)
        and (not planar_trans_pan)
        and Mc_rot is not None
        and palm_optical_alignment_cos(B) >= float(rot_palm_face_cos_align_min)
    ):
        wc = np.asarray(R_to_rotvec(R_cam), dtype=np.float64).reshape(3)
        wlab = float(np.linalg.norm(wc))
        if wlab >= float(rot_palm_face_twist_min_rad):
            wxy = float(np.hypot(wc[0], wc[1]))
            twist_ok = rpf <= 0.0 or abs(float(wc[2])) >= rpf * max(wxy, 1e-9)
            if twist_ok:
                th = float(np.arctan2(float(R_cam[1, 0]), float(R_cam[0, 0])))
                sgy = float(rot_palm_face_twist_world_y_sign)
                if not np.isfinite(sgy) or sgy == 0.0:
                    sgy = 1.0
                rv_intrinsic = np.array([0.0, float(sgy) * th, 0.0], dtype=np.float64)
                used_palm_face_twist = True
    if not used_palm_face_twist:
        if Mc_rot is not None:
            R_apply = Mc_rot @ R_cam @ Mc_rot.T
        else:
            R_apply = R_cam
        rv_intrinsic = np.asarray(R_to_rotvec(R_apply), dtype=np.float64).reshape(3).copy()
    rp = float(rot_planar_dom_ratio)
    if (
        not used_palm_face_twist
        and rp > 0.0
        and translation_plane_dominates_depth(
            delta_trans,
            planar_ratio=rp,
            planar_min_mm=float(rot_planar_dom_min_mm),
        )
    ):
        rv_intrinsic = np.zeros(3, dtype=np.float64)
    cmin = float(rot_coex_trans_min_mm)
    cmax = float(rot_coex_max_angle_rad)
    if (
        not used_palm_face_twist
        and cmin > 0.0
        and cmax > 0.0
        and state.prev_wrist is not None
    ):
        if vel_mm >= cmin and float(np.linalg.norm(rv_intrinsic)) < cmax:
            rv_intrinsic = np.zeros(3, dtype=np.float64)

    rv_cmd_raw = rv_intrinsic * float(rot_scale) * float(max(0.0, rot_gain))
    zsc = float(rot_world_z_scale)
    if zsc != 1.0 and not axis_locked:
        rv_cmd_raw = np.asarray(rv_cmd_raw, dtype=np.float64)
        rv_cmd_raw[2] *= zsc

    span_now = float(hand_palm_span_mm(h, wrist)) if state.initialized else 0.0
    state.prev_wrist = np.asarray(palm_center, dtype=np.float64).reshape(3).copy()
    state.prev_basis = np.asarray(B_depth, dtype=np.float64).reshape(3, 3).copy()
    state.prev_hand_span_mm = span_now
    tau_mm = float(rot_trans_tau_mm)
    if tau_mm > 0.0 and not used_palm_face_twist and not axis_locked:
        rv_cmd_raw = rv_cmd_raw * float(np.exp(-vel_mm / tau_mm))

    delta_h = np.asarray(delta_h_raw, dtype=np.float64).reshape(3)
    rv_cmd = np.asarray(rv_cmd_raw, dtype=np.float64).reshape(3)
    axis_motion = "none"
    w_rot_blend = 1.0
    w_trans_blend = 1.0
    if axis_locked:
        if palm_world_embed:
            rv_for_class = rv_cam
            rv_apply = np.asarray(rv_pose_world, dtype=np.float64).reshape(3).copy()
        else:
            rv_apply = palm_world_rotvec_from_basis_delta(Mc_rot, B_depth, state.ref_basis)
            rv_for_class = rv_apply
        motion, w_rot_blend, w_trans_blend = axis_locked_trans_rot_blend_weights(
            delta_h_raw,
            rv_apply,
            trans_on_m=float(axis_trans_on_m),
            rot_on_rad=float(axis_rot_on_rad),
            rv_cam_rad=rv_for_class,
            delta_cam_mm=delta_cam,
            delta_trans_mm=delta_trans,
            secondary_frac=float(axis_trans_rot_coupling),
        )
        axis_motion = motion
        state.last_delta_cam_mm = np.asarray(delta_cam, dtype=np.float64).reshape(3).copy()
        state.last_delta_h_raw_m = np.asarray(delta_h_raw, dtype=np.float64).reshape(3).copy()
        state.last_rv_cam_world = np.asarray(rv_cam, dtype=np.float64).reshape(3).copy()
        rv_cmd = np.zeros(3, dtype=np.float64)
        delta_h = np.zeros(3, dtype=np.float64)
        trans_dz = float(axis_trans_deadzone_m) * 0.55 / max(float(w_trans_blend), 0.35)
        rot_dz = float(axis_rot_deadzone_rad) * 0.65 / max(float(w_rot_blend), 0.35)
        if w_rot_blend > 1e-6:
            # In rotate mode, follow the full 3D palm pose. Only translation stays axis-locked.
            # Snapping rotation to one world axis made rv_pose correct while the swarm display looked wrong.
            rv_full = (
                rv_apply
                * float(rot_scale)
                * float(max(0.0, rot_gain))
                * float(w_rot_blend)
            )
            if float(np.linalg.norm(rv_full)) >= rot_dz:
                rv_cmd = rv_full
        if w_trans_blend > 1e-6:
            delta_snapped = snap_world_vector_to_principal_axis(
                delta_h_raw,
                deadzone=trans_dz,
                min_ratio=float(axis_cam_snap_min_ratio),
            )
            delta_h = delta_snapped * float(w_trans_blend)
        state.last_axis_motion = motion
        state.last_rot_blend_w = float(w_rot_blend)
        state.last_trans_blend_w = float(w_trans_blend)
        state.last_rv_pose_world = np.asarray(rv_pose_world, dtype=np.float64).reshape(3).copy()
        state.last_delta_h_world = np.asarray(delta_h, dtype=np.float64).reshape(3).copy()
    elif axis_locked:
        state.last_rv_pose_world[:] = 0.0
        state.last_rv_cmd_world[:] = 0.0
        state.last_delta_h_world[:] = 0.0
    rg = float(max(0.0, rot_gate_rad))
    if (
        not axis_locked
        and rg > 0.0
        and not used_palm_face_twist
        and float(np.linalg.norm(rv_cmd)) < rg
    ):
        rv_cmd = np.zeros(3, dtype=np.float64)
    rn = float(np.linalg.norm(rv_cmd))
    if max_rot_rad > 0 and rn > max_rot_rad:
        rv_cmd = rv_cmd * (max_rot_rad / max(rn, 1e-9))
    if axis_locked:
        state.last_rv_cmd_world = np.asarray(rv_cmd, dtype=np.float64).reshape(3).copy()

    te = float(np.clip(trans_ema, 0.0, 1.0))
    re = float(np.clip(rot_ema, 0.0, 1.0))
    if axis_locked and axis_motion == "rotate" and w_trans_blend <= 1e-6:
        te = 0.0
    state.ema_offset = (1.0 - te) * state.ema_offset + te * delta_h
    state.ema_rotvec = (1.0 - re) * state.ema_rotvec + re * rv_cmd

    rn2 = float(np.linalg.norm(state.ema_rotvec))
    if max_rot_rad > 0 and rn2 > max_rot_rad:
        state.ema_rotvec *= max_rot_rad / max(rn2, 1e-9)

    on = float(np.linalg.norm(state.ema_offset))
    if max_offset_m > 0 and on > max_offset_m:
        state.ema_offset *= max_offset_m / on

    return state.ema_offset.astype(np.float64), rotvec_to_R(state.ema_rotvec)


def _vec3_s(v: np.ndarray) -> str:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return f"({a[0]:+.1f},{a[1]:+.1f},{a[2]:+.1f})"


def print_left_swarm_pose_debug(
    state: LeftSwarmPoseState,
    *,
    frame_idx: int = 0,
    axis_sign: tuple[float, float, float] = (1.0, 1.0, -1.0),
    trans_scale: float = 0.012,
) -> None:
    """Stdout debug: palm center (camera mm), palm/world deltas, pose R, blend weights."""
    if not state.initialized:
        return
    ref_pc = np.asarray(state.ref_palm_center, dtype=np.float64).reshape(3)
    pc = np.asarray(state.last_palm_center_mm, dtype=np.float64).reshape(3)
    dc = np.asarray(state.last_delta_cam_mm, dtype=np.float64).reshape(3)
    dc_arm = np.asarray(state.last_delta_cam_arm_mm, dtype=np.float64).reshape(3)
    dp = np.asarray(state.last_delta_trans_palm_mm, dtype=np.float64).reshape(3)
    dh_raw = np.asarray(state.last_delta_h_raw_m, dtype=np.float64).reshape(3)
    dh_cmd = np.asarray(state.last_delta_h_world, dtype=np.float64).reshape(3)
    off = np.asarray(state.ema_offset, dtype=np.float64).reshape(3)
    rv_pose = np.asarray(state.last_rv_pose_world, dtype=np.float64).reshape(3)
    rv_cmd = np.asarray(state.last_rv_cmd_world, dtype=np.float64).reshape(3)
    R = rotvec_to_R(state.ema_rotvec)
    ang = float(np.degrees(np.linalg.norm(state.ema_rotvec)))
    mot = str(state.last_axis_motion)
    wr = float(state.last_rot_blend_w)
    wt = float(state.last_trans_blend_w)
    sign = np.asarray(axis_sign, dtype=np.float64).reshape(3)
    wx, wy, wz = float(dh_raw[0]), float(dh_raw[1]), float(dh_raw[2])
    trans_tag = "cam(dx,dy,dz)" if bool(state.last_trans_cam_gated) else "palm(x,y,z)"
    sep = "=" * 72
    print(sep, flush=True)
    print(
        f"[left-pose debug] frame={int(frame_idx)} motion={mot} "
        f"blend R×{wr:.0%} T×{wt:.0%}  axis_sign={tuple(float(x) for x in sign)}",
        flush=True,
    )
    print(
        f"  palm center (cam mm)  arm={_vec3_s(ref_pc)}  now={_vec3_s(pc)}  "
        f"Δframe={_vec3_s(dc)} |Δ|={np.linalg.norm(dc):.1f}mm  "
        f"Δarm={_vec3_s(dc_arm)} |Δarm|={np.linalg.norm(dc_arm):.1f}mm  dz={dc[2]:+.1f}",
        flush=True,
    )
    rv_c = np.asarray(state.last_rv_cam_world, dtype=np.float64).reshape(3)
    print(f"  rv_cam(classify)={_format_rotvec(rv_c)}", flush=True)
    print(
        f"  trans {trans_tag}={_vec3_s(dp)}  → world m (X,Y,Z)=({wx:+.4f},{wy:+.4f},{wz:+.4f})  "
        f"raw={_vec3_s(dh_raw)}",
        flush=True,
    )
    print(
        f"  world cmd Δ (snap×blend) = {_vec3_s(dh_cmd)}  ema_offset T={_vec3_s(off)}  |T|={np.linalg.norm(off):.3f}m",
        flush=True,
    )
    print(
        f"  rot  rv_pose(deg·ax)={_format_rotvec(rv_pose)}  rv_cmd={_format_rotvec(rv_cmd)}  "
        f"accum_angle≈{ang:.1f}°",
        flush=True,
    )
    print(
        f"  R_pose columns Xw={_vec3_s(R[:, 0])} Yw={_vec3_s(R[:, 1])} Zw={_vec3_s(R[:, 2])}",
        flush=True,
    )
    print(sep, flush=True)


def _format_rotvec(rv: np.ndarray) -> str:
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-7:
        return "0"
    ax = rv / ang
    return f"{np.degrees(ang):+.1f}° @({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"


def apply_rigid_to_targets(
    targets: np.ndarray,
    offset: np.ndarray,
    R: np.ndarray,
    *,
    ref_drone_xyz: np.ndarray | None = None,
    pivot: str = "per_drone",
) -> np.ndarray:
    """Apply world translation ``offset`` and rotation ``R``.

    ``pivot="centroid"`` (default): ``p' = R @ (p - c) + c + off`` with ``c = mean(p)`` — the
    formation **整体自转** about its current centroid (rigid spin in place + world ``off``).

    ``pivot="per_drone"``: ``p' = ref_i + R @ (p - ref_i) + off`` — each drone pivots about its
    own arm-time slot instead of a shared center.
    """
    t = np.asarray(targets, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] < 3:
        return np.asarray(targets, dtype=np.float32)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    off = np.asarray(offset, dtype=np.float64).reshape(3)
    p = t[:, :3]
    key = str(pivot).strip().lower()
    if key == "per_drone" and ref_drone_xyz is not None:
        ref = np.asarray(ref_drone_xyz, dtype=np.float64)
        if ref.ndim == 2 and ref.shape[0] == p.shape[0] and ref.shape[1] >= 3:
            rel = p - ref[:, :3]
            out = ref[:, :3] + (R @ rel.T).T + off
        else:
            key = "centroid"
    if key != "per_drone":
        c = np.mean(p, axis=0)
        rel = p - c
        out = (R @ rel.T).T + c + off
    t2 = t.copy()
    t2[:, :3] = out
    return t2.astype(np.float32)
