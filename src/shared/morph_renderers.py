"""Matplotlib superellipsoid morph: parametric surface + fixed labeled sample points.

Open parameter role in this module:
- ``open_alpha=0``: fully bulged 3D superellipsoid.
- ``open_alpha=1``: flat plane state.
- During transition, points are not re-sampled; fixed IDs are mapped continuously.
"""

from __future__ import annotations

from itertools import permutations
from typing import Optional, Tuple

import numpy as np

from shared.morph_geometry import morph_plane_extent_radius
from shared.topology_utils import clamp01

# Fixed sample set for the whole process (same indices, only mapped to current surface).
_FIXED_SURFACE_UNIT_POINTS: Optional[np.ndarray] = None
_FIXED_SURFACE_COUNT: int = 0
_FIXED_PLANE_UNIT_POINTS: Optional[np.ndarray] = None
_OPEN1_PLANE_XY_STATE: Optional[np.ndarray] = None
_OPEN1_START_XY_STATE: Optional[np.ndarray] = None
_OPEN1_TRANSITION_FRAMES: int = 45  # ~1.5 s @30 fps
_OPEN1_FRAME_IDX: int = 0
_DENSE_SURFACE_UNIT_CANDIDATES: Optional[np.ndarray] = None
_RELAX_CACHE_KEY: Optional[Tuple[float, ...]] = None
_RELAX_CACHE_VALUE: Optional[np.ndarray] = None
_PLANE_CACHE_KEY: Optional[Tuple[float, ...]] = None
_PLANE_CACHE_VALUE: Optional[np.ndarray] = None
_SURFACE_CANDIDATE_CACHE_KEY: Optional[Tuple[float, ...]] = None
_SURFACE_CANDIDATE_CACHE_U: Optional[np.ndarray] = None
_SURFACE_CANDIDATE_CACHE_P: Optional[np.ndarray] = None
_SURFACE_CANDIDATE_CACHE_W: Optional[np.ndarray] = None
_AXIS6_OCTANT_GROUPS: Optional[Tuple[Tuple[int, ...], ...]] = None
_RING_LAYOUT_PREV_N: Optional[int] = None
_RING_LAYOUT_PREV_MODE: Optional[int] = None
_RING_LAYOUT_PREV_POINTS: Optional[np.ndarray] = None

# Test-tunable controls for local octant-triangle optimization on non-pole points.
# Intentionally large right now to strongly enforce similar spacing.
_TRI_MIN_SEP_SCALE = 1.18
_TRI_MAX_SEP_SCALE = 1.28
_TRI_PAIR_BALANCE_GAIN = 0.060
_TRI_MIN_SEP_BARRIER_GAIN = 0.180
_TRI_SPRING_GAIN = 0.28
_TRI_MAX_ANGLE_DEG = 28.0
_TRI_CVT_GAIN = 0.42


def _is_sphere_like_shape(epsilon1: float, epsilon2: float, close: float) -> bool:
    """Sphere mode should stay on the original minimal-energy directions for any open."""
    e1 = float(epsilon1)
    e2 = float(epsilon2)
    _ = float(close)
    return abs(e1 - 1.0) <= 0.14 and abs(e2 - 1.0) <= 0.14


def _spow(v, exp):
    return np.sign(v) * np.power(np.abs(v), exp)


def _superellipse_dense_xyz(
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
    n_eta: int,
    n_omega: int,
) -> np.ndarray:
    eta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(n_eta))
    omg = np.linspace(-np.pi, np.pi, int(n_omega))
    ETA, OMG = np.meshgrid(eta, omg, indexing="ij")
    CE = np.cos(ETA)
    SE = np.sin(ETA)
    CO = np.cos(OMG)
    SO = np.sin(OMG)
    X = R * _spow(CE, epsilon1) * _spow(CO, epsilon2)
    Y = R * _spow(CE, epsilon1) * _spow(SO, epsilon2)
    Z = close * R * _spow(SE, epsilon1)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(float)


def _seed_axis6() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-9)
    return x / n


def _minimal_energy_disk_points(n: int, *, steps: int = 1400, lr: float = 0.012) -> np.ndarray:
    """2D minimal-energy points in unit disk; first 6 are fixed angular anchors.

    Strategy:
    1) Initialize movable points with equal-area sunflower disk samples (already fairly uniform).
    2) Run Coulomb-like repulsion in 2D with a soft boundary barrier.
    3) Re-project to disk each step (projected gradient descent with annealed step size).
    """
    n = int(max(6, n))
    th = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    axis6 = np.stack([np.cos(th), np.sin(th)], axis=1).astype(float)
    if n == 6:
        return axis6.copy()

    m = n - 6
    k = np.arange(m, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    # Equal-area radial law in unit disk.
    r = np.sqrt((k + 0.5) / (m + 0.5))
    a = golden * k
    rest = np.stack([r * np.cos(a), r * np.sin(a)], axis=1).astype(float)
    pts = np.vstack([axis6, rest]).astype(float)

    movable = np.arange(6, n, dtype=int)
    n_steps = int(max(1, steps))
    for it in range(n_steps):
        g = np.zeros_like(pts)
        for i in movable:
            d = pts[i] - pts
            r2 = np.sum(d * d, axis=1) + 1e-6
            inv_r3 = 1.0 / np.power(r2, 1.5)
            inv_r3[i] = 0.0
            g[i] = np.sum(d * inv_r3[:, None], axis=0)
            # Soft disk-boundary barrier: grows near radius 1 and pushes inward.
            ri = float(np.linalg.norm(pts[i]))
            if ri > 0.0:
                margin = max(1e-4, 1.0 - ri)
                g[i] += -0.015 * (pts[i] / ri) / (margin * margin)
        # Annealed step keeps early exploration but stabilizes late-stage uniformity.
        step = float(lr) * (0.22 + 0.78 * (1.0 - float(it) / float(n_steps)))
        pts[movable] += step * g[movable]
        # Robust in-place projection to unit disk (avoid chained indexing copy bug).
        mv = pts[movable]
        rn = np.linalg.norm(mv, axis=1, keepdims=True)
        scale = np.maximum(1.0, rn)
        mv = mv / np.maximum(scale, 1e-9)
        # Clip any residual numerical outliers.
        mv = np.clip(mv, -1.0, 1.0)
        pts[movable] = mv

    return pts


def _uniform_disk_targets(n: int) -> np.ndarray:
    """Legacy fallback: unconstrained near-uniform disk targets."""
    n = int(max(1, n))
    k = np.arange(n, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt((k + 0.5) / (n + 0.5))
    a = golden * k
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1).astype(float)


def _superellipse_plane_boundary_radius(theta: np.ndarray, epsilon2: float) -> np.ndarray:
    e2 = max(1e-3, float(epsilon2))
    c = np.abs(np.cos(theta))
    s = np.abs(np.sin(theta))
    q = 2.0 / e2
    denom = np.power(c, q) + np.power(s, q)
    return 1.0 / np.power(np.maximum(denom, 1e-9), 0.5 * e2)


def _project_xy_inside_superellipse(
    xy: np.ndarray,
    *,
    R: float,
    epsilon2: float,
) -> np.ndarray:
    out = np.asarray(xy, dtype=float).copy()
    if out.size == 0:
        return out
    Rf = max(1e-6, float(R))
    e2 = max(1e-3, float(epsilon2))
    q = 2.0 / e2
    sx = np.power(np.abs(out[:, 0]) / Rf, q)
    sy = np.power(np.abs(out[:, 1]) / Rf, q)
    s = sx + sy
    scale = np.ones_like(s)
    outside = s > 1.0
    scale[outside] = np.power(s[outside], -0.5 * e2)
    out *= scale[:, None]
    return out


def _build_current_plane_xy_targets(
    u_base: np.ndarray,
    *,
    R: float,
    epsilon1: float,
    epsilon2: float,
) -> np.ndarray:
    """Uniform points inside the current open=1 superellipse plane.

    Strategy:
    1) Generate a dense uniform disk set and warp it to current superellipse boundary.
    2) Farthest-point sample ``n`` well-spread candidates.
    3) Assign sampled points back to fixed IDs with sphere-based directional hints
       (keeps relative ring/azimuth structure while remaining uniformly spread).
    """
    u0 = np.asarray(u_base, dtype=float).reshape(-1, 3)
    n = u0.shape[0]
    if n <= 0:
        return np.zeros((0, 2), dtype=float)

    key = (
        float(n),
        round(float(R), 3),
        round(float(epsilon1), 3),
        round(float(epsilon2), 3),
    )
    global _PLANE_CACHE_KEY, _PLANE_CACHE_VALUE
    if _PLANE_CACHE_KEY == key and _PLANE_CACHE_VALUE is not None:
        return _PLANE_CACHE_VALUE.copy()

    dense_n = max(1200, 18 * n)
    disk = _uniform_disk_targets(dense_n)
    theta = np.arctan2(disk[:, 1], disk[:, 0])
    rr = np.linalg.norm(disk, axis=1)
    rb = _superellipse_plane_boundary_radius(theta, float(epsilon2))
    cand = np.stack([rr * rb * np.cos(theta), rr * rb * np.sin(theta)], axis=1).astype(float)

    # Farthest-point sampling on current plane to get a well-spread subset.
    selected = []
    used = np.zeros(dense_n, dtype=bool)
    seed_targets = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    n_seed = min(4, n)
    for i in range(n_seed):
        d = np.sum((cand - seed_targets[i][None, :]) ** 2, axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        selected.append(j)

    min_d2 = np.full(dense_n, np.inf, dtype=float)
    if selected:
        seed_xy = cand[np.asarray(selected, dtype=int)]
        min_d2 = np.min(np.sum((cand[:, None, :] - seed_xy[None, :, :]) ** 2, axis=2), axis=1)
        min_d2[used] = -np.inf

    while len(selected) < n:
        j = int(np.argmax(min_d2))
        used[j] = True
        selected.append(j)
        d2_new = np.sum((cand - cand[j][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[used] = -np.inf

    sel_xy = cand[np.asarray(selected, dtype=int)]

    # Assign spread points back to fixed IDs with directional consistency.
    out = np.zeros((n, 2), dtype=float)
    assign_used = np.zeros(n, dtype=bool)
    xy0 = u0[:, :2]
    r0 = np.linalg.norm(xy0, axis=1)
    theta0 = np.arctan2(xy0[:, 1], xy0[:, 0])
    rb0 = _superellipse_plane_boundary_radius(theta0, float(epsilon2))
    hint = np.zeros((n, 2), dtype=float)
    nz = r0 > 1e-9
    hint[nz] = (xy0[nz] / r0[nz, None]) * (r0[nz] * rb0[nz])[:, None]

    priority = [0, 1, 2, 3] + [i for i in range(n) if i not in {0, 1, 2, 3}]
    for idx in priority:
        d = np.sum((sel_xy - hint[idx][None, :]) ** 2, axis=1)
        d[assign_used] = np.inf
        j = int(np.argmin(d))
        assign_used[j] = True
        out[idx] = sel_xy[j]

    out *= float(R)
    _PLANE_CACHE_KEY = key
    _PLANE_CACHE_VALUE = out.copy()
    return out


def _build_plane_targets_from_sphere(u3: np.ndarray) -> np.ndarray:
    """Build open=1 planar targets using CVT/Lloyd-like uniform sampling in a disk."""
    n = int(u3.shape[0])
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    rng = np.random.default_rng(20260421)

    # Init centers by sunflower (good starting guess).
    k = np.arange(n, dtype=float)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    r0 = np.sqrt((k + 0.5) / (n + 0.5))
    a0 = golden * k
    centers = np.stack([r0 * np.cos(a0), r0 * np.sin(a0)], axis=1).astype(float)

    # Monte-Carlo Lloyd in disk: assign uniform disk samples to nearest center, move to centroid.
    m = max(12000, 900 * n)
    rr = np.sqrt(rng.uniform(0.0, 1.0, size=(m,)))
    aa = rng.uniform(0.0, 2.0 * np.pi, size=(m,))
    samp = np.stack([rr * np.cos(aa), rr * np.sin(aa)], axis=1).astype(float)

    for _ in range(18):
        d2 = np.sum((samp[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        lab = np.argmin(d2, axis=1)
        new_c = centers.copy()
        for i in range(n):
            pts_i = samp[lab == i]
            if pts_i.shape[0] > 0:
                new_c[i] = np.mean(pts_i, axis=0)
        # Project to disk if needed.
        rn = np.linalg.norm(new_c, axis=1, keepdims=True)
        scale = np.maximum(1.0, rn)
        centers = new_c / np.maximum(scale, 1e-9)

    return centers


def _minimal_energy_points_direct(n: int, *, steps: int = 180, lr: float = 0.02) -> np.ndarray:
    """Original sphere sampler: first 6 axis seeds + Coulomb-like relaxation."""
    n = int(max(6, n))
    axis6 = _seed_axis6()
    if n == 6:
        return axis6.copy()

    rng = np.random.default_rng(20260421)
    rest = rng.normal(size=(n - 6, 3)).astype(float)
    rest = _normalize_rows(rest)
    pts = np.vstack([axis6, rest]).astype(float)

    movable = np.arange(6, n, dtype=int)
    for _ in range(int(max(1, steps))):
        g = np.zeros_like(pts)
        for i in movable:
            d = pts[i] - pts
            r2 = np.sum(d * d, axis=1) + 1e-6
            inv_r3 = 1.0 / np.power(r2, 1.5)
            inv_r3[i] = 0.0
            g[i] = np.sum(d * inv_r3[:, None], axis=0)
        pts[movable] += float(lr) * g[movable]
        pts[movable] = _normalize_rows(pts[movable])

    return pts


def _axis6_octant_faces(axis6_pts: np.ndarray) -> list[tuple[int, int, int]]:
    """Eight non-overlapping spherical triangles formed by the 6 poles."""
    p = np.asarray(axis6_pts, dtype=float).reshape(-1, 3)
    if p.shape[0] < 6:
        return []
    faces_raw = [
        (0, 2, 4),
        (0, 2, 5),
        (0, 3, 4),
        (0, 3, 5),
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (1, 3, 5),
    ]
    faces = []
    for i, j, k in faces_raw:
        face = [i, j, k]
        normal = np.cross(p[j] - p[i], p[k] - p[i])
        centroid = np.mean(p[face], axis=0)
        if float(np.dot(normal, centroid)) < 0.0:
            face = [i, k, j]
        faces.append(tuple(face))
    return faces


def _triangle_planar_frame(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a spherical triangle to a local 2D plane."""
    center = _normalize_rows((a + b + c)[None, :])[0]
    t1 = b - np.dot(b, center) * center
    if float(np.linalg.norm(t1)) < 1e-9:
        t1 = a - np.dot(a, center) * center
    t1 = _normalize_rows(t1[None, :])[0]
    t2 = _normalize_rows(np.cross(center, t1)[None, :])[0]
    basis = np.stack([t1, t2], axis=0)
    tri2 = np.stack([basis @ a, basis @ b, basis @ c], axis=0).astype(float)
    return tri2, basis


def _triangle_barycentric_2d(tri2: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    a = tri2[0]
    b = tri2[1]
    c = tri2[2]
    v0 = b - a
    v1 = c - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    denom = max(d00 * d11 - d01 * d01, 1e-12)
    v2 = pts2 - a[None, :]
    d20 = np.sum(v2 * v0[None, :], axis=1)
    d21 = np.sum(v2 * v1[None, :], axis=1)
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2
    return np.stack([w0, w1, w2], axis=1)


def _project_points_inside_triangle_2d(tri2: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    bary = _triangle_barycentric_2d(tri2, pts2)
    bary = np.maximum(bary, 1e-4)
    bary /= np.sum(bary, axis=1, keepdims=True)
    return bary @ tri2


def _triangle_edge_normals_2d(tri2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(tri2, axis=0)
    mids = []
    normals = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        e = tri2[j] - tri2[i]
        n = np.array([-e[1], e[0]], dtype=float)
        nn = max(float(np.linalg.norm(n)), 1e-12)
        n = n / nn
        mid = 0.5 * (tri2[i] + tri2[j])
        if float(np.dot(centroid - mid, n)) < 0.0:
            n = -n
        mids.append(mid)
        normals.append(n)
    return np.asarray(mids, dtype=float), np.asarray(normals, dtype=float)


def _triangle_init_points_2d(tri2: np.ndarray, n_target: int) -> np.ndarray:
    """Deterministic low-discrepancy init inside one planar triangle."""
    target = int(max(0, n_target))
    if target <= 0:
        return np.zeros((0, 2), dtype=float)
    phi = 0.6180339887498949
    pts = []
    for k in range(target):
        u1 = (k + 0.5) / float(target)
        u2 = np.mod((k + 0.5) * phi, 1.0)
        r1 = np.sqrt(u1)
        w0 = 1.0 - r1
        w1 = r1 * (1.0 - u2)
        w2 = r1 * u2
        pts.append(w0 * tri2[0] + w1 * tri2[1] + w2 * tri2[2])
    return np.asarray(pts, dtype=float)


def _triangle_uniform_candidates_2d(tri2: np.ndarray, n_samples: int) -> np.ndarray:
    """Uniform area samples inside one planar triangle for local CVT steps."""
    m = int(max(1, n_samples))
    phi = 0.6180339887498949
    pts = []
    for k in range(m):
        u1 = (k + 0.5) / float(m)
        u2 = np.mod((k + 0.5) * phi, 1.0)
        r1 = np.sqrt(u1)
        w0 = 1.0 - r1
        w1 = r1 * (1.0 - u2)
        w2 = r1 * u2
        pts.append(w0 * tri2[0] + w1 * tri2[1] + w2 * tri2[2])
    return np.asarray(pts, dtype=float)


def _triangle_area_2d(tri2: np.ndarray) -> float:
    v0 = tri2[1] - tri2[0]
    v1 = tri2[2] - tri2[0]
    return 0.5 * abs(float(v0[0] * v1[1] - v0[1] * v1[0]))


def _slerp_rows(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    aa = _normalize_rows(np.asarray(a, dtype=float))
    bb = _normalize_rows(np.asarray(b, dtype=float))
    tt = np.asarray(t, dtype=float).reshape(-1, 1)
    dot_ab = np.clip(np.sum(aa * bb, axis=1), -1.0, 1.0)
    theta = np.arccos(dot_ab)
    out = np.empty_like(aa)
    small = theta < 1e-7
    if np.any(small):
        out[small] = _normalize_rows((1.0 - tt[small]) * aa[small] + tt[small] * bb[small])
    if np.any(~small):
        th = theta[~small]
        tsub = tt[~small]
        denom = np.sin(th)
        s0 = np.sin((1.0 - tsub[:, 0]) * th) / denom
        s1 = np.sin(tsub[:, 0] * th) / denom
        out[~small] = _normalize_rows(s0[:, None] * aa[~small] + s1[:, None] * bb[~small])
    return out


def _apply_triangle_angle_limit(
    tri2: np.ndarray,
    basis: np.ndarray,
    pts2: np.ndarray,
    tri3: np.ndarray,
    ref_pts3: np.ndarray,
    max_angle_rad: float,
) -> np.ndarray:
    bary = _triangle_barycentric_2d(tri2, pts2)
    cand3 = _normalize_rows(
        bary[:, [0]] * tri3[0][None, :] + bary[:, [1]] * tri3[1][None, :] + bary[:, [2]] * tri3[2][None, :]
    )
    dot_rc = np.clip(np.sum(cand3 * ref_pts3, axis=1), -1.0, 1.0)
    ang = np.arccos(dot_rc)
    over = ang > float(max_angle_rad)
    if np.any(over):
        frac = np.clip(float(max_angle_rad) / np.maximum(ang[over], 1e-9), 0.0, 1.0)
        cand3[over] = _slerp_rows(ref_pts3[over], cand3[over], frac)
    pts2_limited = cand3 @ basis.T
    return _project_points_inside_triangle_2d(tri2, pts2_limited)


def _slerp_edge_points(a: np.ndarray, b: np.ndarray, n_samples: int = 15) -> np.ndarray:
    """Sample a spherical edge between two unit directions."""
    aa = _normalize_rows(np.asarray(a, dtype=float)[None, :])[0]
    bb = _normalize_rows(np.asarray(b, dtype=float)[None, :])[0]
    dot_ab = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    theta = float(np.arccos(dot_ab))
    t = np.linspace(0.0, 1.0, int(max(2, n_samples)), dtype=float)
    if theta < 1e-6:
        seg = (1.0 - t)[:, None] * aa[None, :] + t[:, None] * bb[None, :]
        return _normalize_rows(seg)
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    seg = s0[:, None] * aa[None, :] + s1[:, None] * bb[None, :]
    return _normalize_rows(seg)


def _project_unit_point_to_triangle_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri2, basis = _triangle_planar_frame(a, b, c)
    q2 = (basis @ np.asarray(q, dtype=float)).astype(float)
    bary = _triangle_barycentric_2d(tri2, q2[None, :])[0]
    return tri2, basis, bary


def _assign_points_to_anchor_faces(
    anchor_pts: np.ndarray,
    extra_pts: np.ndarray,
    faces: list[tuple[int, int, int]],
) -> list[list[int]]:
    groups = [[] for _ in faces]
    if extra_pts.size == 0 or not faces:
        return groups

    for local_idx, q in enumerate(np.asarray(extra_pts, dtype=float)):
        best_face = 0
        best_penalty = np.inf
        best_fit = np.inf
        for face_idx, (i, j, k) in enumerate(faces):
            tri2, basis, bary = _project_unit_point_to_triangle_2d(
                anchor_pts[i], anchor_pts[j], anchor_pts[k], q
            )
            penalty = float(np.sum(np.maximum(-bary, 0.0)))
            q_fit = bary[0] * anchor_pts[i] + bary[1] * anchor_pts[j] + bary[2] * anchor_pts[k]
            q_fit = _normalize_rows(q_fit[None, :])[0]
            fit = float(1.0 - np.clip(np.dot(q_fit, q), -1.0, 1.0))
            if penalty < best_penalty - 1e-9 or (abs(penalty - best_penalty) <= 1e-9 and fit < best_fit):
                best_face = face_idx
                best_penalty = penalty
                best_fit = fit
        groups[best_face].append(local_idx)
    return groups


def _get_axis6_octant_groups() -> list[list[int]]:
    """Fixed m1/sphere allocation of non-pole point counts across 8 octant regions."""
    global _AXIS6_OCTANT_GROUPS
    u = get_fixed_surface_points()
    if u.shape[0] <= 6:
        return []
    if _AXIS6_OCTANT_GROUPS is None:
        faces = _axis6_octant_faces(u[:6])
        groups = _assign_points_to_anchor_faces(u[:6], u[6:], faces)
        _AXIS6_OCTANT_GROUPS = tuple(tuple(int(v) for v in g) for g in groups)
    return [list(g) for g in _AXIS6_OCTANT_GROUPS]


def _project_xyz_to_current_shape(
    pts3: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
) -> np.ndarray:
    dirs = _normalize_rows(np.asarray(pts3, dtype=float).reshape(-1, 3))
    return _map_unit_points_to_superellipsoid(dirs, float(R), float(close), float(epsilon1), float(epsilon2))


def _apply_shape_angle_limit(
    pts3: np.ndarray,
    ref_dirs: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
    max_angle_rad: float,
) -> np.ndarray:
    cand_dirs = _normalize_rows(np.asarray(pts3, dtype=float).reshape(-1, 3))
    refs = _normalize_rows(np.asarray(ref_dirs, dtype=float).reshape(-1, 3))
    dot_rc = np.clip(np.sum(cand_dirs * refs, axis=1), -1.0, 1.0)
    ang = np.arccos(dot_rc)
    over = ang > float(max_angle_rad)
    if np.any(over):
        frac = np.clip(float(max_angle_rad) / np.maximum(ang[over], 1e-9), 0.0, 1.0)
        cand_dirs[over] = _slerp_rows(refs[over], cand_dirs[over], frac)
    return _map_unit_points_to_superellipsoid(
        cand_dirs,
        float(R),
        float(close),
        float(epsilon1),
        float(epsilon2),
    )


def _is_cylinder_like_epsilon(epsilon1: float, epsilon2: float) -> bool:
    """Heuristic match to mode2 base (very flat meridian, still round in XY)."""
    e1 = float(epsilon1)
    e2 = float(epsilon2)
    return e1 <= 0.18 and e2 >= 0.70


def _edge_corner_repel_gain(epsilon1: float, epsilon2: float) -> float:
    """Stronger for sharp superquad shapes; disabled for the cylinder mode."""
    if _is_cylinder_like_epsilon(epsilon1, epsilon2):
        return 0.0
    e1 = max(float(epsilon1), 1e-3)
    e2 = max(float(epsilon2), 1e-3)
    sharp1 = max(0.0, 1.0 / e1 - 1.0)
    sharp2 = max(0.0, 1.0 / e2 - 1.0)
    sharp = min(12.0, 0.55 * sharp1 + 0.45 * sharp2)
    return 0.0025 + 0.011 * sharp


def _superquad_edge_corner_grad_3d(
    p: np.ndarray,
    *,
    R: float,
    close: float,
) -> np.ndarray:
    """Push points away from edges/corners in normalized superquad coordinates."""
    Rf = max(1e-6, float(R))
    closef = max(1e-6, float(close))
    rz = closef * Rf
    xn = p[:, 0] / Rf
    yn = p[:, 1] / Rf
    zn = p[:, 2] / rz
    g = np.zeros_like(p)
    g[:, 0] = np.sign(p[:, 0]) * (np.abs(yn) + np.abs(zn)) / Rf
    g[:, 1] = np.sign(p[:, 1]) * (np.abs(xn) + np.abs(zn)) / Rf
    g[:, 2] = np.sign(p[:, 2]) * (np.abs(xn) + np.abs(yn)) / rz
    return g


def _optimize_existing_points_in_triangle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    pts_init: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
) -> np.ndarray:
    """Optimize already pure-mapped points inside one current-shape octant triangle."""
    init = np.asarray(pts_init, dtype=float).reshape(-1, 3)
    target = init.shape[0]
    if target <= 0:
        return np.zeros((0, 3), dtype=float)

    tri3 = np.stack([a, b, c], axis=0).astype(float)
    tri2, basis = _triangle_planar_frame(a, b, c)
    pts2 = (basis @ init.T).T.astype(float)
    pts2 = _project_points_inside_triangle_2d(tri2, pts2)
    pts2_ref = pts2.copy()
    mids, normals = _triangle_edge_normals_2d(tri2)
    tri_scale = max(
        float(np.linalg.norm(tri2[1] - tri2[0])),
        float(np.linalg.norm(tri2[2] - tri2[1])),
        float(np.linalg.norm(tri2[0] - tri2[2])),
        1e-3,
    )
    tri_area = max(_triangle_area_2d(tri2), 1e-9)
    sep_ref = np.sqrt(4.0 * tri_area / (np.sqrt(3.0) * max(float(target + 3), 1.0)))
    min_sep = float(_TRI_MIN_SEP_SCALE) * sep_ref
    max_sep = float(_TRI_MAX_SEP_SCALE) * sep_ref
    max_angle = np.deg2rad(float(_TRI_MAX_ANGLE_DEG))
    ref_dirs = _normalize_rows(init)
    cvt_cand = _triangle_uniform_candidates_2d(tri2, max(320, 120 * target))
    n_steps = int(90 + 10 * min(target, 8))
    corner_gain = _edge_corner_repel_gain(float(epsilon1), float(epsilon2))

    for it in range(n_steps):
        g = np.zeros_like(pts2)
        sites = np.vstack([tri2, pts2])
        d2_cand = np.sum((cvt_cand[:, None, :] - sites[None, :, :]) ** 2, axis=2)
        lab = np.argmin(d2_cand, axis=1)
        bary0 = _triangle_barycentric_2d(tri2, pts2)
        plane0 = bary0[:, [0]] * a[None, :] + bary0[:, [1]] * b[None, :] + bary0[:, [2]] * c[None, :]
        pts3_now = _project_xyz_to_current_shape(
            plane0,
            R=R,
            close=close,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
        )
        g3c = -corner_gain * _superquad_edge_corner_grad_3d(pts3_now, R=float(R), close=float(close))
        for i in range(target):
            d = pts2[i] - pts2
            r2 = np.sum(d * d, axis=1) + 1e-6 * tri_scale * tri_scale
            r = np.sqrt(r2)
            inv_r3 = 1.0 / np.power(r2, 1.5)
            inv_r3[i] = 0.0
            g[i] += np.sum(d * inv_r3[:, None], axis=0)
            g[i] += basis @ g3c[i]

            near = (r < min_sep) & (np.arange(target) != i)
            if np.any(near):
                barrier = (min_sep - r[near]) / np.maximum(r[near], 1e-6)
                g[i] += (
                    float(_TRI_MIN_SEP_BARRIER_GAIN)
                    * np.sum(d[near] * barrier[:, None], axis=0)
                    / max(min_sep * min_sep, 1e-6)
                )

            if target > 1:
                nn_count = min(4, target - 1)
                nn_idx = np.argsort(r)[1 : 1 + nn_count]
                if nn_idx.size > 0:
                    err = (r[nn_idx] - sep_ref) / max(sep_ref, 1e-6)
                    g[i] += -float(_TRI_PAIR_BALANCE_GAIN) * np.sum(
                        (err / np.maximum(r[nn_idx], 1e-6))[:, None] * d[nn_idx],
                        axis=0,
                    )
                    far = nn_idx[r[nn_idx] > max_sep]
                    if far.size > 0:
                        g[i] += 0.010 * np.sum(
                            d[far] / np.maximum(r[far], 1e-6)[:, None],
                            axis=0,
                        )

            g[i] += -float(_TRI_SPRING_GAIN) * (pts2[i] - pts2_ref[i]) / max(sep_ref * sep_ref, 1e-6)

            cell = cvt_cand[lab == (3 + i)]
            if cell.shape[0] > 0:
                centroid = np.mean(cell, axis=0)
                g[i] += float(_TRI_CVT_GAIN) * (centroid - pts2[i]) / max(sep_ref * sep_ref, 1e-6)

            for mid, normal in zip(mids, normals):
                dist = max(float(np.dot(pts2[i] - mid, normal)), 2.0e-3 * tri_scale)
                g[i] += 0.020 * normal / (dist * dist)

        step = 0.010 * tri_scale * (0.30 + 0.70 * (1.0 - float(it) / float(n_steps)))
        pts2 += step * g
        pts2 = _project_points_inside_triangle_2d(tri2, pts2)
        bary = _triangle_barycentric_2d(tri2, pts2)
        plane3 = bary[:, [0]] * a[None, :] + bary[:, [1]] * b[None, :] + bary[:, [2]] * c[None, :]
        pts3 = _project_xyz_to_current_shape(
            plane3,
            R=R,
            close=close,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
        )
        pts3 = _apply_shape_angle_limit(
            pts3,
            ref_dirs,
            R=R,
            close=close,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
            max_angle_rad=max_angle,
        )
        pts2 = (basis @ pts3.T).T.astype(float)
        pts2 = _project_points_inside_triangle_2d(tri2, pts2)

    bary = _triangle_barycentric_2d(tri2, pts2)
    plane3 = bary[:, [0]] * a[None, :] + bary[:, [1]] * b[None, :] + bary[:, [2]] * c[None, :]
    pts3 = _project_xyz_to_current_shape(
        plane3,
        R=R,
        close=close,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
    )
    pts3 = _apply_shape_angle_limit(
        pts3,
        ref_dirs,
        R=R,
        close=close,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        max_angle_rad=max_angle,
    )
    return pts3


def _post_optimize_extra_mapped_points(
    p_mapped: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
    morph_mode: int = 1,
) -> np.ndarray:
    """After pure mapping, optimize non-pole points inside fixed m1 octant regions."""
    p_in = np.asarray(p_mapped, dtype=float).reshape(-1, 3)
    n = p_in.shape[0]
    if n <= 6:
        return p_in.copy()

    out = p_in.copy()
    axis6 = out[:6].copy()
    extra = out[6:].copy()
    faces = _axis6_octant_faces(axis6)
    if not faces:
        return out

    if int(morph_mode) in (2, 3, 4, 5):
        return _mode3_layout_mapped_points_by_octant(
            p_in,
            R=float(R),
            close=float(close),
            epsilon1=float(epsilon1),
            epsilon2=float(epsilon2),
            morph_mode=int(morph_mode),
        )

    groups = _get_axis6_octant_groups()
    for face_indices, (i, j, k) in zip(groups, faces):
        if not face_indices:
            continue
        init_face = extra[np.asarray(face_indices, dtype=int)]
        opt_face = _optimize_existing_points_in_triangle(
            axis6[i],
            axis6[j],
            axis6[k],
            init_face,
            R=R,
            close=close,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
        )
        out[6 + np.asarray(face_indices, dtype=int)] = opt_face
    return out


def _mode3_layout_mapped_points_by_octant(
    p_mapped: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
    morph_mode: int = 3,
) -> np.ndarray:
    """Keep six poles fixed and arrange extras into layer rings.

    Rules:
    - mode2/3: n<=24 -> 3 layers, n>24 -> 5 layers.
    - mode4/5: always 5 layers.
    - Layer planes explicitly include: side-pole plane (z=0) and top/bottom-pole planes
      (z = +/-1 for layer definition).
    - Extra points are assigned by nearest layer plane from their sphere z position.
    - Points on each ring are distributed by equal mapped arc-length on the current
      superellipsoid cross section (shape-adaptive).
    """
    p = np.asarray(p_mapped, dtype=float).copy().reshape(-1, 3)
    n = p.shape[0]
    u0 = get_fixed_surface_points()
    if n != u0.shape[0] or n <= 6:
        return p

    n_ex = int(n - 6)
    global _RING_LAYOUT_PREV_N, _RING_LAYOUT_PREV_MODE, _RING_LAYOUT_PREV_POINTS
    if (
        _RING_LAYOUT_PREV_N == int(n)
        and _RING_LAYOUT_PREV_MODE == int(morph_mode)
        and _RING_LAYOUT_PREV_POINTS is not None
        and int(_RING_LAYOUT_PREV_POINTS.shape[0]) == n_ex
    ):
        anchor_prev = np.asarray(_RING_LAYOUT_PREV_POINTS, dtype=float).reshape(n_ex, 3).copy()
    else:
        anchor_prev = p[6:].copy()
    mm = int(morph_mode)
    if mm in (4, 5):
        n_layers = 5
    else:
        n_layers = 3 if int(n) <= 24 else 5
    if int(n) > 24:
        n_layers += 2
    n_layers = int(max(1, min(n_layers, n_ex)))

    u_ex = np.asarray(u0[6:], dtype=float).reshape(n_ex, 3)
    z_ex = u_ex[:, 2]
    z_top = 1.0
    # Build layer centers by equal spacing in *mapped* |Z| ratio, then invert back to
    # sphere-z centers. This avoids layer collapse after nonlinear |sin(eta)|^epsilon1.
    if n_layers <= 1:
        z_centers = np.array([0.0], dtype=float)
    elif (n_layers % 2) == 1:
        half = (n_layers - 1) // 2
        q = np.linspace(0.0, 1.0, half + 1)  # mapped |Z| ratio levels
        e1_eff = float(max(0.05, float(epsilon1)))
        s = np.power(np.clip(q, 0.0, 1.0), 1.0 / e1_eff)  # inverse of ^epsilon1
        pos = s[1:] if half > 0 else np.zeros((0,), dtype=float)
        z_centers = np.concatenate([-pos[::-1], np.array([0.0]), pos]).astype(float)
    else:
        # Fallback if layer count becomes even after min/max clamping.
        z_centers = np.linspace(-z_top, z_top, n_layers).astype(float)

    nearest = np.argmin(np.abs(z_ex[:, None] - z_centers[None, :]), axis=1)
    layer_ids: list[np.ndarray] = [np.flatnonzero(nearest == k).astype(int) for k in range(n_layers)]
    # For M3, keep the initial sphere-based layer counts. The top/bottom drawing
    # step below preserves four corners without moving overflow points to z=0.
    if mm in (4, 5) and n_layers >= 5:
        # Keep only fixed poles on top/bottom; extras should stay in middle layers.
        top_ids = layer_ids[-1]
        bot_ids = layer_ids[0]
        if int(top_ids.size) > 0:
            layer_ids[-2] = np.concatenate([layer_ids[-2], top_ids]).astype(int)
            layer_ids[-1] = np.zeros((0,), dtype=int)
        if int(bot_ids.size) > 0:
            layer_ids[1] = np.concatenate([layer_ids[1], bot_ids]).astype(int)
            layer_ids[0] = np.zeros((0,), dtype=int)
    # Rebalance empties by moving farthest z points from the largest layer.
    for k in range(n_layers):
        if mm in (4, 5) and n_layers >= 5 and (k == 0 or k == n_layers - 1):
            continue
        if int(layer_ids[k].size) > 0:
            continue
        donor = int(np.argmax([arr.size for arr in layer_ids]))
        if int(layer_ids[donor].size) <= 1:
            continue
        d_ids = layer_ids[donor]
        d_cost = np.abs(z_ex[d_ids] - z_centers[donor]) - np.abs(z_ex[d_ids] - z_centers[k])
        move_loc = int(np.argmax(d_cost))
        move_id = int(d_ids[move_loc])
        layer_ids[donor] = np.delete(d_ids, move_loc)
        layer_ids[k] = np.asarray([move_id], dtype=int)

    def _equal_arclen_ring_points(eta0: float, m: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
        if m <= 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
        n_dense = int(max(720, 120 * m))
        omg_dense = np.linspace(-np.pi, np.pi, n_dense, endpoint=False) + float(phase)
        eta_dense = np.full((n_dense,), float(eta0), dtype=float)
        p_dense = _map_angles_to_superellipsoid(
            eta_dense,
            omg_dense,
            R=float(R),
            close=float(close),
            epsilon1=float(epsilon1),
            epsilon2=float(epsilon2),
        )
        p_next = np.roll(p_dense, -1, axis=0)
        seg = np.linalg.norm(p_next - p_dense, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(cum[-1])
        if total < 1e-8:
            omg = np.linspace(-np.pi, np.pi, m, endpoint=False) + float(phase)
            eta = np.full((m,), float(eta0), dtype=float)
            return _map_angles_to_superellipsoid(
                eta, omg, R=float(R), close=float(close), epsilon1=float(epsilon1), epsilon2=float(epsilon2)
            ), omg
        targets = (np.arange(m, dtype=float) + 0.5) * (total / float(m))
        omg_ext = np.concatenate([omg_dense, [omg_dense[0] + 2.0 * np.pi]])
        out_omg = np.interp(targets, cum, omg_ext)
        eta = np.full((m,), float(eta0), dtype=float)
        out_pts = _map_angles_to_superellipsoid(
            eta,
            out_omg,
            R=float(R),
            close=float(close),
            epsilon1=float(epsilon1),
            epsilon2=float(epsilon2),
        )
        return out_pts, out_omg

    def _min_fixed_pole_distance(ring_pts: np.ndarray) -> float:
        if ring_pts.size == 0:
            return float("inf")
        d = ring_pts[:, None, :] - p[:6][None, :, :]
        return float(np.min(np.linalg.norm(d, axis=2)))

    def _choose_pole_safe_ring(
        eta0: float,
        m: int,
        phase0: float,
        *,
        zero_layer_xy: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate a ring so extra points do not land on the six fixed poles."""
        best_pts: np.ndarray | None = None
        best_omg: np.ndarray | None = None
        best_score = -float("inf")
        n_try = int(max(12, 2 * m))
        step = (2.0 * np.pi / max(float(m), 1.0)) / float(n_try)
        for k in range(n_try):
            phase_try = float(phase0) + float(k) * step
            eta_for_xy = 0.0 if zero_layer_xy else float(eta0)
            pts, omg = _equal_arclen_ring_points(eta_for_xy, m, phase_try)
            if zero_layer_xy:
                z_face = float(
                    _map_angles_to_superellipsoid(
                        np.array([float(eta0)], dtype=float),
                        np.array([0.0], dtype=float),
                        R=float(R),
                        close=float(close),
                        epsilon1=float(epsilon1),
                        epsilon2=float(epsilon2),
                    )[0, 2]
                )
                pts[:, 2] = z_face
            score = _min_fixed_pole_distance(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
                best_omg = omg
        if best_pts is None or best_omg is None:
            return _equal_arclen_ring_points(eta0, m, phase0)
        return best_pts, best_omg

    for lk, ids in enumerate(layer_ids):
        if ids.size == 0:
            continue
        zc = float(z_centers[lk])
        z_draw = float(np.clip(zc, -0.999999, 0.999999))
        z_draw = float(np.sign(z_draw) * min(abs(z_draw), 1.0 - 1.0e-4))
        eta0 = float(np.arcsin(z_draw))
        m = int(ids.size)

        anchors = anchor_prev[ids]

        phase = np.pi / float(2 * m)
        if mm == 3 and n_layers >= 3 and (lk == 0 or lk == n_layers - 1):
            # For M3 corner layers, lock phase so 4-point ring lands on diagonals.
            phase = 0.0
        else:
            if abs(zc) < 0.08:
                phase += np.pi / 4.0
            phase += (lk % 2) * (np.pi / max(3.0, float(m)))

        use_zero_layer_xy = mm in (2, 3) and n_layers >= 3 and (lk == 0 or lk == n_layers - 1)
        if mm == 3 and use_zero_layer_xy and m >= 4:
            # Hard-lock M3 top/bottom face corners. Extra points keep the same
            # sphere-based layer count and are spread between corners.
            n_dense_corner = 1440
            omg_dense = np.linspace(-np.pi, np.pi, n_dense_corner, endpoint=False)
            eta_mid = np.zeros((n_dense_corner,), dtype=float)
            p_mid = _map_angles_to_superellipsoid(
                eta_mid,
                omg_dense,
                R=float(R),
                close=float(close),
                epsilon1=float(epsilon1),
                epsilon2=float(epsilon2),
            )
            z_face = float(
                _map_angles_to_superellipsoid(
                    np.array([float(eta0)], dtype=float),
                    np.array([0.0], dtype=float),
                    R=float(R),
                    close=float(close),
                    epsilon1=float(epsilon1),
                    epsilon2=float(epsilon2),
                )[0, 2]
            )
            cxy = np.mean(p_mid[:, :2], axis=0)
            dx = p_mid[:, 0] - float(cxy[0])
            dy = p_mid[:, 1] - float(cxy[1])
            score = np.abs(dx * dy)
            quad_masks = (
                (dx >= 0.0) & (dy >= 0.0),
                (dx < 0.0) & (dy >= 0.0),
                (dx < 0.0) & (dy < 0.0),
                (dx >= 0.0) & (dy < 0.0),
            )
            pick_idx: list[int] = []
            used: set[int] = set()
            for qm in quad_masks:
                cand = np.flatnonzero(qm)
                if cand.size == 0:
                    continue
                ord_c = cand[np.argsort(score[cand])[::-1]]
                chosen = -1
                for ci in ord_c.tolist():
                    if int(ci) not in used:
                        chosen = int(ci)
                        break
                if chosen >= 0:
                    pick_idx.append(chosen)
                    used.add(chosen)
            if len(pick_idx) < 4:
                ord_all = np.argsort(score)[::-1]
                for ci in ord_all.tolist():
                    if int(ci) in used:
                        continue
                    pick_idx.append(int(ci))
                    used.add(int(ci))
                    if len(pick_idx) >= 4:
                        break
            pick_idx = pick_idx[:4]
            if m > 4:
                min_gap = int(max(2, n_dense_corner // (4 * max(m, 4))))

                def _circular_idx_dist(a: np.ndarray, b: int) -> np.ndarray:
                    raw = np.abs(a.astype(int) - int(b))
                    return np.minimum(raw, n_dense_corner - raw)

                while len(pick_idx) < m:
                    all_idx = np.arange(n_dense_corner, dtype=int)
                    allowed = np.ones((n_dense_corner,), dtype=bool)
                    for ui in pick_idx:
                        allowed &= _circular_idx_dist(all_idx, int(ui)) > min_gap
                    cand = all_idx[allowed]
                    if cand.size == 0:
                        cand = np.asarray([i for i in range(n_dense_corner) if int(i) not in used], dtype=int)
                    if cand.size == 0:
                        break
                    d_to_used = np.min(
                        np.stack([_circular_idx_dist(cand, int(ui)) for ui in pick_idx], axis=1),
                        axis=1,
                    )
                    chosen = int(cand[int(np.argmax(d_to_used))])
                    pick_idx.append(chosen)
                    used.add(chosen)
            ring_omg = np.asarray([float(omg_dense[i]) for i in pick_idx], dtype=float)
            ring_pts = np.asarray([p_mid[i] for i in pick_idx], dtype=float)
            ring_pts[:, 2] = z_face
        elif use_zero_layer_xy:
            # For M2/M3 outer layers, reuse the z=0 layer XY outline and only change Z.
            # This keeps top/bottom footprints aligned with the middle layer.
            ring_pts, ring_omg = _choose_pole_safe_ring(eta0, m, phase, zero_layer_xy=True)
        else:
            ring_pts, ring_omg = _choose_pole_safe_ring(eta0, m, phase, zero_layer_xy=False)
        ord_new = np.argsort(ring_omg)
        ring_pts = ring_pts[ord_new]

        if mm in (2, 3):
            # For M2/M3, assign directly by fixed spherical azimuth order on every layer.
            # This avoids visible center-out drift from historical nearest-neighbor matching.
            ids_sorted = np.asarray(
                sorted(ids.tolist(), key=lambda ii: float(np.arctan2(u_ex[int(ii), 1], u_ex[int(ii), 0]))),
                dtype=int,
            )
            m_assign = int(min(int(ids_sorted.size), int(ring_pts.shape[0])))
            for t in range(m_assign):
                p[6 + int(ids_sorted[t])] = ring_pts[int(t)]
            for t in range(m_assign, int(ids_sorted.size)):
                p[6 + int(ids_sorted[t])] = ring_pts[int(ring_pts.shape[0] - 1)]
        else:
            rem_ids = [int(v) for v in ids.tolist()]
            rem_new = list(range(m))
            while rem_ids and rem_new:
                best_id = -1
                best_j = -1
                best_cost = 1.0e100
                for idx_local in rem_ids:
                    a = anchor_prev[int(idx_local)]
                    for j in rem_new:
                        d = ring_pts[int(j)] - a
                        c = float(np.dot(d, d))
                        if c < best_cost:
                            best_cost = c
                            best_id = int(idx_local)
                            best_j = int(j)
                if best_id < 0 or best_j < 0:
                    break
                p[6 + best_id] = ring_pts[best_j]
                rem_ids.remove(best_id)
                rem_new.remove(best_j)
            for idx_local in rem_ids:
                p[6 + int(idx_local)] = ring_pts[0]
    _RING_LAYOUT_PREV_N = int(n)
    _RING_LAYOUT_PREV_MODE = int(morph_mode)
    _RING_LAYOUT_PREV_POINTS = p[6:].copy()
    return p


def _minimal_energy_points(n: int, *, steps: int = 180, lr: float = 0.02) -> np.ndarray:
    """Generate n sphere points: keep the original energy-minimal sphere unchanged."""
    return _minimal_energy_points_direct(n, steps=steps, lr=lr)


def _get_dense_surface_unit_candidates(min_count: int) -> np.ndarray:
    global _DENSE_SURFACE_UNIT_CANDIDATES
    need = int(max(512, min_count))
    if _DENSE_SURFACE_UNIT_CANDIDATES is None or _DENSE_SURFACE_UNIT_CANDIDATES.shape[0] < need:
        _DENSE_SURFACE_UNIT_CANDIDATES = _minimal_energy_points_direct(need, steps=260, lr=0.018)
    return _DENSE_SURFACE_UNIT_CANDIDATES[:need].copy()


def _get_area_weighted_surface_candidates(
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense current-surface candidates with approximate local area weights."""
    Rf = float(R)
    closef = float(close)
    e1 = float(epsilon1)
    e2 = float(epsilon2)
    n = int(max(6, n_points))
    n_eta = max(46, int(np.ceil(np.sqrt(30.0 * n))))
    n_omega = max(72, int(np.ceil(1.65 * n_eta)))
    key = (
        round(Rf, 3),
        round(closef, 3),
        round(e1, 3),
        round(e2, 3),
        float(n_eta),
        float(n_omega),
    )
    global _SURFACE_CANDIDATE_CACHE_KEY, _SURFACE_CANDIDATE_CACHE_U, _SURFACE_CANDIDATE_CACHE_P, _SURFACE_CANDIDATE_CACHE_W
    if (
        _SURFACE_CANDIDATE_CACHE_KEY == key
        and _SURFACE_CANDIDATE_CACHE_U is not None
        and _SURFACE_CANDIDATE_CACHE_P is not None
        and _SURFACE_CANDIDATE_CACHE_W is not None
    ):
        return (
            _SURFACE_CANDIDATE_CACHE_U.copy(),
            _SURFACE_CANDIDATE_CACHE_P.copy(),
            _SURFACE_CANDIDATE_CACHE_W.copy(),
        )

    eta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_eta)
    omg = np.linspace(-np.pi, np.pi, n_omega, endpoint=False)
    ETA, OMG = np.meshgrid(eta, omg, indexing="ij")
    CE = np.cos(ETA)
    SE = np.sin(ETA)
    CO = np.cos(OMG)
    SO = np.sin(OMG)

    X = Rf * _spow(CE, e1) * _spow(CO, e2)
    Y = Rf * _spow(CE, e1) * _spow(SO, e2)
    Z = closef * Rf * _spow(SE, e1)

    dX_deta = np.gradient(X, eta, axis=0)
    dY_deta = np.gradient(Y, eta, axis=0)
    dZ_deta = np.gradient(Z, eta, axis=0)
    dX_domg = np.gradient(X, omg, axis=1)
    dY_domg = np.gradient(Y, omg, axis=1)
    dZ_domg = np.gradient(Z, omg, axis=1)

    t_eta = np.stack([dX_deta, dY_deta, dZ_deta], axis=2)
    t_omg = np.stack([dX_domg, dY_domg, dZ_domg], axis=2)
    area = np.linalg.norm(np.cross(t_eta, t_omg), axis=2)
    area = np.maximum(area, 1e-9)

    U = np.stack([CE * CO, CE * SO, SE], axis=2)
    u_flat = U.reshape(-1, 3).astype(float)
    p_flat = np.stack([X, Y, Z], axis=2).reshape(-1, 3).astype(float)
    w_flat = area.reshape(-1).astype(float)

    keep = np.isfinite(w_flat) & np.all(np.isfinite(u_flat), axis=1) & np.all(np.isfinite(p_flat), axis=1)
    u_flat = u_flat[keep]
    p_flat = p_flat[keep]
    w_flat = w_flat[keep]

    _SURFACE_CANDIDATE_CACHE_KEY = key
    _SURFACE_CANDIDATE_CACHE_U = u_flat.copy()
    _SURFACE_CANDIDATE_CACHE_P = p_flat.copy()
    _SURFACE_CANDIDATE_CACHE_W = w_flat.copy()
    return u_flat, p_flat, w_flat


def init_fixed_surface_points(n_samples: int) -> np.ndarray:
    """Initialize fixed indexed sphere samples for this process."""
    global _FIXED_SURFACE_UNIT_POINTS, _FIXED_SURFACE_COUNT, _FIXED_PLANE_UNIT_POINTS, _OPEN1_PLANE_XY_STATE, _OPEN1_START_XY_STATE, _OPEN1_FRAME_IDX, _PLANE_CACHE_KEY, _PLANE_CACHE_VALUE, _RELAX_CACHE_KEY, _RELAX_CACHE_VALUE, _SURFACE_CANDIDATE_CACHE_KEY, _SURFACE_CANDIDATE_CACHE_U, _SURFACE_CANDIDATE_CACHE_P, _SURFACE_CANDIDATE_CACHE_W, _AXIS6_OCTANT_GROUPS, _RING_LAYOUT_PREV_N, _RING_LAYOUT_PREV_MODE, _RING_LAYOUT_PREV_POINTS
    n = int(max(6, n_samples))
    _FIXED_SURFACE_UNIT_POINTS = _minimal_energy_points(n)
    # For open=1 plane mapping, derive targets from sphere indexing with user-specified anchors.
    _FIXED_PLANE_UNIT_POINTS = _build_plane_targets_from_sphere(_FIXED_SURFACE_UNIT_POINTS)
    _OPEN1_PLANE_XY_STATE = None
    _OPEN1_START_XY_STATE = None
    _OPEN1_FRAME_IDX = 0
    _PLANE_CACHE_KEY = None
    _PLANE_CACHE_VALUE = None
    _RELAX_CACHE_KEY = None
    _RELAX_CACHE_VALUE = None
    _SURFACE_CANDIDATE_CACHE_KEY = None
    _SURFACE_CANDIDATE_CACHE_U = None
    _SURFACE_CANDIDATE_CACHE_P = None
    _SURFACE_CANDIDATE_CACHE_W = None
    _AXIS6_OCTANT_GROUPS = None
    _RING_LAYOUT_PREV_N = None
    _RING_LAYOUT_PREV_MODE = None
    _RING_LAYOUT_PREV_POINTS = None
    _FIXED_SURFACE_COUNT = int(n)
    return _FIXED_SURFACE_UNIT_POINTS.copy()


def prompt_and_init_fixed_surface_points(default_n: int = 24) -> int:
    raw = input(f"Enter n surface samples (>=6, default {int(default_n)}): ").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = int(default_n)
    else:
        n = int(default_n)
    n = int(max(6, n))
    init_fixed_surface_points(n)
    print(f"Fixed surface samples initialized: n={n} (first 6 are axis seeds).")
    return n


def get_fixed_surface_points() -> np.ndarray:
    global _FIXED_SURFACE_UNIT_POINTS
    if _FIXED_SURFACE_UNIT_POINTS is None:
        init_fixed_surface_points(24)
    return _FIXED_SURFACE_UNIT_POINTS.copy()


def get_fixed_surface_count() -> int:
    return int(get_fixed_surface_points().shape[0])


def get_fixed_plane_points() -> np.ndarray:
    global _FIXED_PLANE_UNIT_POINTS
    if _FIXED_PLANE_UNIT_POINTS is None:
        init_fixed_surface_points(24)
    return _FIXED_PLANE_UNIT_POINTS.copy()


def reset_lp_scatter_inertia() -> None:
    # Kept for compatibility with existing call sites.
    return


def _map_unit_points_to_superellipsoid(
    u_pts: np.ndarray,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
) -> np.ndarray:
    uu = np.asarray(u_pts, dtype=float)
    if uu.ndim == 1:
        uu = uu.reshape(1, -1)
    x = uu[:, 0]
    y = uu[:, 1]
    z = uu[:, 2]
    eta = np.arcsin(np.clip(z, -1.0, 1.0))
    omega = np.arctan2(y, x)
    CE = np.cos(eta)
    SE = np.sin(eta)
    CO = np.cos(omega)
    SO = np.sin(omega)
    X = float(R) * _spow(CE, float(epsilon1)) * _spow(CO, float(epsilon2))
    Y = float(R) * _spow(CE, float(epsilon1)) * _spow(SO, float(epsilon2))
    Z = float(close) * float(R) * _spow(SE, float(epsilon1))
    return np.stack([X, Y, Z], axis=1)


def _map_angles_to_superellipsoid(
    eta: np.ndarray,
    omega: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
) -> np.ndarray:
    ce = np.cos(eta)
    se = np.sin(eta)
    co = np.cos(omega)
    so = np.sin(omega)
    x = float(R) * _spow(ce, float(epsilon1)) * _spow(co, float(epsilon2))
    y = float(R) * _spow(ce, float(epsilon1)) * _spow(so, float(epsilon2))
    z = float(close) * float(R) * _spow(se, float(epsilon1))
    return np.stack([x, y, z], axis=1)


def _surface_area_density_from_angles(
    eta: np.ndarray,
    omega: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
    d_param: float,
) -> np.ndarray:
    """Local surface area factor ||dP/deta x dP/domega||."""
    p_eta_p = _map_angles_to_superellipsoid(
        eta + d_param,
        omega,
        R=R,
        close=close,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
    )
    p_eta_m = _map_angles_to_superellipsoid(
        eta - d_param,
        omega,
        R=R,
        close=close,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
    )
    p_omg_p = _map_angles_to_superellipsoid(
        eta,
        omega + d_param,
        R=R,
        close=close,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
    )
    p_omg_m = _map_angles_to_superellipsoid(
        eta,
        omega - d_param,
        R=R,
        close=close,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
    )
    t_eta = (p_eta_p - p_eta_m) / (2.0 * d_param)
    t_omg = (p_omg_p - p_omg_m) / (2.0 * d_param)
    return np.maximum(np.linalg.norm(np.cross(t_eta, t_omg), axis=1), 1e-9)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _face_soft_region_weights_from_points(
    pts: np.ndarray,
    *,
    radius_xy: float,
    radius_z: float,
    beta: float = 6.0,
) -> np.ndarray:
    """Soft membership to 6 signed faces from current mapped positions."""
    p = np.asarray(pts, dtype=float).reshape(-1, 3)
    rx = max(1e-6, float(radius_xy))
    rz = max(1e-6, float(radius_z))
    signed = np.stack(
        [
            p[:, 0] / rx,
            -p[:, 0] / rx,
            p[:, 1] / rx,
            -p[:, 1] / rx,
            p[:, 2] / rz,
            -p[:, 2] / rz,
        ],
        axis=1,
    )
    s = float(beta) * signed
    s -= np.max(s, axis=1, keepdims=True)
    w = np.exp(s)
    w_sum = np.sum(w, axis=1, keepdims=True)
    w_sum = np.maximum(w_sum, 1e-9)
    return w / w_sum


def _relax_unit_points_for_current_shape(
    u_base: np.ndarray,
    *,
    R: float,
    close: float,
    epsilon1: float,
    epsilon2: float,
    steps: int = 12,
    step0: float = 0.020,
    reg: float = 0.22,
) -> np.ndarray:
    """Keep the original mapping, then do small Coulomb slides on the surface.

    This preserves fixed-ID continuity much better than re-selecting candidate points.
    Each point starts from its original sphere->surface mapping, then moves only by
    a few tangential relaxation steps, with a spring pulling it back toward the
    original mapped location to avoid visible jumps.
    """
    u0 = np.asarray(u_base, dtype=float).reshape(-1, 3)
    if u0.shape[0] <= 1:
        return u0.copy()
    Rf = max(1e-6, float(R))
    closef = float(close)
    e1 = float(epsilon1)
    e2 = float(epsilon2)
    key = (
        float(u0.shape[0]),
        round(Rf, 3),
        round(closef, 3),
        round(e1, 3),
        round(e2, 3),
    )
    global _RELAX_CACHE_KEY, _RELAX_CACHE_VALUE
    if _RELAX_CACHE_KEY == key and _RELAX_CACHE_VALUE is not None:
        return _RELAX_CACHE_VALUE.copy()

    eta0 = np.arcsin(np.clip(u0[:, 2], -1.0, 1.0))
    omega0 = np.arctan2(u0[:, 1], u0[:, 0])
    eta = eta0.copy()
    omega = omega0.copy()
    sharp = max(0.0, 1.0 / max(min(e1, e2), 1e-3) - 1.0)
    cube_like = max(0.0, 1.0 / max(e1, 1e-3) - 1.0) * max(0.0, 1.0 / max(e2, 1e-3) - 1.0)
    cube_like_norm = min(cube_like / 16.0, 1.0)
    # Test mode: disable the spring pull-back to isolate whether it is the main
    # reason local spreading stalls near edges/corners.
    reg_eff = 0.0
    step_eff = float(step0) * (1.0 + 0.05 * min(sharp, 10.0))
    corner_push = 0.001 + 0.012 * cube_like_norm
    face_push = 0.80 + 0.35 * min(sharp / 8.0, 1.0)
    jacobian_push = 0.010 + 0.055 * min(sharp / 8.0, 1.0)
    p_ref = _map_angles_to_superellipsoid(
        eta0,
        omega0,
        R=Rf,
        close=closef,
        epsilon1=e1,
        epsilon2=e2,
    )
    d_param = 2.5e-3
    n_steps = int(min(24, max(1, round(float(steps) * (1.0 + 0.10 * sharp)))))
    n = u0.shape[0]
    n_fixed = min(6, n)
    movable = np.arange(n_fixed, n, dtype=int)
    rz = max(closef * Rf, 1e-6)

    for it in range(n_steps):
        p = _map_angles_to_superellipsoid(
            eta,
            omega,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
        )
        face_w = _face_soft_region_weights_from_points(
            p,
            radius_xy=Rf,
            radius_z=rz,
            beta=7.5,
        )
        g = np.zeros_like(p)
        for i in movable:
            d = p[i] - p
            r2 = np.sum(d * d, axis=1) + 1e-6
            inv_r3 = 1.0 / np.power(r2, 1.5)
            inv_r3[i] = 0.0
            g[i] = np.sum(d * inv_r3[:, None], axis=0)
            # Extra face-wise spreading in local 2D face coordinates. This targets
            # the observed failure mode where points crowd along face edges/rims.
            aff_x = face_w[i, 0] * face_w[:, 0] + face_w[i, 1] * face_w[:, 1]
            aff_y = face_w[i, 2] * face_w[:, 2] + face_w[i, 3] * face_w[:, 3]
            aff_z = face_w[i, 4] * face_w[:, 4] + face_w[i, 5] * face_w[:, 5]
            aff_x[i] = 0.0
            aff_y[i] = 0.0
            aff_z[i] = 0.0

            r2_yz = d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2] + 1e-6
            r2_xz = d[:, 0] * d[:, 0] + d[:, 2] * d[:, 2] + 1e-6
            r2_xy = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + 1e-6
            inv_yz3 = 1.0 / np.power(r2_yz, 1.5)
            inv_xz3 = 1.0 / np.power(r2_xz, 1.5)
            inv_xy3 = 1.0 / np.power(r2_xy, 1.5)
            inv_yz3[i] = 0.0
            inv_xz3[i] = 0.0
            inv_xy3[i] = 0.0

            g_face = np.zeros(3, dtype=float)
            g_face[1:] += np.sum(d[:, 1:] * (aff_x * inv_yz3)[:, None], axis=0)
            g_face[[0, 2]] += np.sum(d[:, [0, 2]] * (aff_y * inv_xz3)[:, None], axis=0)
            g_face[:2] += np.sum(d[:, :2] * (aff_z * inv_xy3)[:, None], axis=0)
            g[i] += float(face_push) * g_face

        g[movable] -= float(reg_eff) * (p[movable] - p_ref[movable]) / max(Rf * Rf, 1e-6)

        # Discourage corner occupancy: corners have multiple normalized coordinates large at once.
        xn = p[:, 0] / Rf
        yn = p[:, 1] / Rf
        zn = p[:, 2] / rz
        corner_grad = np.zeros_like(p)
        corner_grad[:, 0] = np.sign(p[:, 0]) * (np.abs(yn) + np.abs(zn)) / Rf
        corner_grad[:, 1] = np.sign(p[:, 1]) * (np.abs(xn) + np.abs(zn)) / Rf
        corner_grad[:, 2] = np.sign(p[:, 2]) * (np.abs(xn) + np.abs(yn)) / rz
        g[movable] += -float(corner_push) * corner_grad[movable]

        p_eta_p = _map_angles_to_superellipsoid(
            eta + d_param,
            omega,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
        )
        p_eta_m = _map_angles_to_superellipsoid(
            eta - d_param,
            omega,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
        )
        p_omg_p = _map_angles_to_superellipsoid(
            eta,
            omega + d_param,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
        )
        p_omg_m = _map_angles_to_superellipsoid(
            eta,
            omega - d_param,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
        )
        t_eta = (p_eta_p - p_eta_m) / (2.0 * d_param)
        t_omg = (p_omg_p - p_omg_m) / (2.0 * d_param)

        a11 = np.sum(t_eta * t_eta, axis=1) + 1e-9
        a22 = np.sum(t_omg * t_omg, axis=1) + 1e-9
        a12 = np.sum(t_eta * t_omg, axis=1)
        b1 = np.sum(g * t_eta, axis=1)
        b2 = np.sum(g * t_omg, axis=1)
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < 1e-9, 1e-9, det)
        d_eta = (a22 * b1 - a12 * b2) / det
        d_omg = (a11 * b2 - a12 * b1) / det

        # Jacobian compensation: move away from low-area-factor regions where the
        # Barr parameterization compresses many directions into small surface area
        # (cube corners / cylinder rims).
        j_eta_p = _surface_area_density_from_angles(
            eta + d_param,
            omega,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
            d_param=d_param,
        )
        j_eta_m = _surface_area_density_from_angles(
            eta - d_param,
            omega,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
            d_param=d_param,
        )
        j_omg_p = _surface_area_density_from_angles(
            eta,
            omega + d_param,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
            d_param=d_param,
        )
        j_omg_m = _surface_area_density_from_angles(
            eta,
            omega - d_param,
            R=Rf,
            close=closef,
            epsilon1=e1,
            epsilon2=e2,
            d_param=d_param,
        )
        grad_logj_eta = (
            np.log(np.maximum(j_eta_p, 1e-9)) - np.log(np.maximum(j_eta_m, 1e-9))
        ) / (2.0 * d_param)
        grad_logj_omg = (
            np.log(np.maximum(j_omg_p, 1e-9)) - np.log(np.maximum(j_omg_m, 1e-9))
        ) / (2.0 * d_param)
        d_eta[movable] += float(jacobian_push) * grad_logj_eta[movable]
        d_omg[movable] += float(jacobian_push) * grad_logj_omg[movable]

        step = float(step_eff) * (0.35 + 0.65 * (1.0 - float(it) / float(n_steps)))
        eta[movable] += step * d_eta[movable]
        omega[movable] += step * d_omg[movable]

        eta[movable] += 0.18 * float(reg_eff) * (eta0[movable] - eta[movable])
        omega[movable] += 0.18 * float(reg_eff) * _wrap_angle(omega0[movable] - omega[movable])

        eta = np.clip(eta, -0.5 * np.pi + 1e-4, 0.5 * np.pi - 1e-4)
        omega = _wrap_angle(omega)
        eta[:n_fixed] = eta0[:n_fixed]
        omega[:n_fixed] = omega0[:n_fixed]

    u_out = np.stack(
        [np.cos(eta) * np.cos(omega), np.cos(eta) * np.sin(omega), np.sin(eta)],
        axis=1,
    )
    u_out = _normalize_rows(u_out)
    u_out[:n_fixed] = u0[:n_fixed]
    _RELAX_CACHE_KEY = key
    _RELAX_CACHE_VALUE = u_out.copy()
    return u_out


def mapped_fixed_surface_points(
    *,
    radius: float,
    open_alpha: float,
    epsilon1: float,
    epsilon2: float,
    plane_radius_a: float,
    plane_radius_b: float,
    morph_mode: int = 1,
) -> np.ndarray:
    """Map fixed IDs to current shape.

    For high ``open_alpha`` (plane state), pure sphere->surface mapping can look uneven after Z-collapse.
    We therefore blend XY toward a fixed minimal-energy disk set, keeping ID order unchanged:
      XY = (1-open)*XY_superellipsoid + open*(R*XY_disk)
      Z  = (1-open)*Z_superellipsoid
    """
    open_a = float(clamp01(open_alpha))
    close = 1.0 - open_a
    R = morph_plane_extent_radius(radius, open_a, plane_radius_a, plane_radius_b)
    u = get_fixed_surface_points()
    # Keep pure sphere mapping for the active morph mode. Mode1 keeps only minimal-energy
    # directions; other modes can apply mode-specific post layout after the pure map.
    u_relaxed = u.copy()
    if (
        u_relaxed.shape[0] > 6
        and float(open_a) < 0.98
        and not _is_sphere_like_shape(float(epsilon1), float(epsilon2), float(close))
        and int(morph_mode) != 1
    ):
        p = _map_unit_points_to_superellipsoid(
            u_relaxed, R, close, float(epsilon1), float(epsilon2)
        )
        p = _post_optimize_extra_mapped_points(
            p,
            R=float(R),
            close=float(close),
            epsilon1=float(epsilon1),
            epsilon2=float(epsilon2),
            morph_mode=int(morph_mode),
        )
    else:
        p = _map_unit_points_to_superellipsoid(
            u_relaxed, R, close, float(epsilon1), float(epsilon2)
        )
    # open == 1: ignore superellipsoid mapping and transition (1-2s) to planar minimal-energy target.
    # open != 1: keep original superellipsoid logic untouched.
    global _OPEN1_PLANE_XY_STATE, _OPEN1_START_XY_STATE, _OPEN1_FRAME_IDX
    # In practice, open may hover near 1.0 due smoothing; use a practical threshold.
    if float(open_a) >= 0.98:
        tgt = _build_current_plane_xy_targets(
            u,
            R=float(R),
            epsilon1=float(epsilon1),
            epsilon2=float(epsilon2),
        )
        # Hard switch at open=1: avoid frame-to-frame drift/jitter.
        _OPEN1_PLANE_XY_STATE = tgt.copy()
        p[:, :2] = _OPEN1_PLANE_XY_STATE
        p[:, 2] = 0.0
    else:
        _OPEN1_PLANE_XY_STATE = None
        _OPEN1_START_XY_STATE = None
        _OPEN1_FRAME_IDX = 0
    return p


def draw_superellipsoid_morph_canonical(
    ax,
    radius,
    open_alpha,
    *,
    epsilon1: float,
    epsilon2: float,
    plane_radius_a: float,
    plane_radius_b: float,
    plane_grid_n: int,
    sample_scatter_s: float,
    sample_alpha: float,
    show_refs=True,
    mesh_n_eta: int = 40,
    mesh_n_omega: int = 52,
    scatter_use_inertia: bool = True,
    scatter_inertia_lambda: float = 0.65,
    morph_mode: int = 1,
):
    open_alpha = clamp01(open_alpha)
    close = 1.0 - open_alpha
    e1 = float(epsilon1)
    e2 = float(epsilon2)
    R = morph_plane_extent_radius(radius, open_alpha, plane_radius_a, plane_radius_b)

    n_eta = max(24, int(mesh_n_eta))
    n_omega = max(32, int(mesh_n_omega))
    eta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_eta)
    omg = np.linspace(-np.pi, np.pi, n_omega)
    ETA, OMG = np.meshgrid(eta, omg, indexing="ij")

    CE = np.cos(ETA)
    SE = np.sin(ETA)
    CO = np.cos(OMG)
    SO = np.sin(OMG)
    X = R * _spow(CE, e1) * _spow(CO, e2)
    Y = R * _spow(CE, e1) * _spow(SO, e2)
    Z = close * R * _spow(SE, e1)
    if show_refs:
        n_ref = 12
        xr = np.linspace(-R, R, n_ref)
        yr = np.linspace(-R, R, n_ref)
        Xr, Yr = np.meshgrid(xr, yr)
        ax.plot_wireframe(Xr, Yr, np.zeros_like(Xr), color="0.70", linewidth=0.3, alpha=0.5)

        Xs = R * np.cos(ETA) * np.cos(OMG)
        Ys = R * np.cos(ETA) * np.sin(OMG)
        Zs = close * R * np.sin(ETA)
        ax.plot_wireframe(Xs, Ys, Zs, color="0.85", linewidth=0.22, alpha=0.35)

    ax.plot_surface(X, Y, Z, color="tab:cyan", alpha=0.34, linewidth=0)
    ax.plot_wireframe(X, Y, Z, color="tab:blue", linewidth=0.26, alpha=0.50)

    u = get_fixed_surface_points()
    if (
        u.shape[0] >= 6
        and float(open_alpha) < 0.98
        and not _is_sphere_like_shape(e1, e2, close)
    ):
        faces6 = _axis6_octant_faces(u[:6])
        edge_set = set()
        for i, j, k in faces6:
            edge_set.add(tuple(sorted((i, j))))
            edge_set.add(tuple(sorted((j, k))))
            edge_set.add(tuple(sorted((k, i))))
        for i, j in sorted(edge_set):
            seg_u = _slerp_edge_points(u[i], u[j], n_samples=17)
            seg_p = _map_unit_points_to_superellipsoid(seg_u, R, close, e1, e2)
            ax.plot(
                seg_p[:, 0],
                seg_p[:, 1],
                seg_p[:, 2],
                color="tab:orange",
                linewidth=1.0,
                alpha=0.85,
            )

    p = mapped_fixed_surface_points(
        radius=float(radius),
        open_alpha=float(open_alpha),
        epsilon1=e1,
        epsilon2=e2,
        plane_radius_a=float(plane_radius_a),
        plane_radius_b=float(plane_radius_b),
        morph_mode=int(morph_mode),
    )
    ax.scatter(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        c="k",
        s=sample_scatter_s,
        alpha=sample_alpha,
    )

    for i, (px, py, pz) in enumerate(p):
        ax.text(px, py, pz, str(i), color="k", fontsize=6, alpha=0.9)
