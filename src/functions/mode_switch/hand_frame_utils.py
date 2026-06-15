"""Hand coordinate frames: wrist-centric normalization, palm-plane basis, fingertip curl metrics (scaled vs palm_plane 3D)."""

from __future__ import annotations

import numpy as np


def metric_hand_to_shape_normalized(points, *, wrist_id: int, mcp_ids, fingertip_ids):
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] < 21:
        return [tuple(float(x) for x in row) for row in arr]
    wrist = arr[wrist_id]
    if not np.all(np.isfinite(wrist)):
        return [tuple(float(x) for x in row) for row in arr]
    rel = arr - wrist
    mcp_norms = []
    for i in mcp_ids:
        if i < len(rel) and np.isfinite(rel[i]).all():
            mcp_norms.append(float(np.linalg.norm(rel[i])))
    if mcp_norms:
        s = float(np.mean(mcp_norms))
    elif np.isfinite(rel[9]).all():
        s = float(np.linalg.norm(rel[9]))
    else:
        ft = []
        for i in fingertip_ids:
            if i < len(rel) and np.isfinite(rel[i]).all():
                ft.append(float(np.linalg.norm(rel[i])))
        s = float(np.mean(ft)) if ft else 1.0
    s = max(s, 1e-3)
    out = rel / s
    return [tuple(float(x) for x in out[i]) for i in range(21)]


def palm_plane_basis_from_world(points, *, wrist_id: int, index_mcp_id: int = 5, middle_mcp_id: int = 9):
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] < 21:
        return None
    w = arr[wrist_id]
    pi = arr[index_mcp_id]
    pm = arr[middle_mcp_id]
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(pi)) or not np.all(np.isfinite(pm)):
        return None
    u = pm - w
    v = pi - w
    n = np.cross(u, v)
    ln = float(np.linalg.norm(n))
    if ln < 1e-6:
        return None
    e_z = n / ln
    lu = float(np.linalg.norm(u))
    if lu < 1e-6:
        return None
    e_x = u / lu
    e_y = np.cross(e_z, e_x)
    ly = float(np.linalg.norm(e_y))
    if ly < 1e-6:
        return None
    e_y = e_y / ly
    e_z = np.cross(e_x, e_y)
    lz = float(np.linalg.norm(e_z))
    if lz < 1e-6:
        return None
    e_z = e_z / lz
    R = np.stack([e_x, e_y, e_z], axis=1)
    return w, R


def metric_hand_to_palm_plane_normalized(points, *, wrist_id: int, mcp_ids):
    basis = palm_plane_basis_from_world(points, wrist_id=wrist_id)
    if basis is None:
        return metric_hand_to_shape_normalized(points, wrist_id=wrist_id, mcp_ids=mcp_ids, fingertip_ids=[4, 8, 12, 16, 20])
    w, R = basis
    arr = np.asarray(points, dtype=float)
    n = arr.shape[0]
    rel = np.zeros((n, 3), dtype=float)
    for k in range(n):
        p = arr[k]
        if not np.all(np.isfinite(p)):
            rel[k] = np.nan
        else:
            rel[k] = R.T @ (p - w)
    mcp_norms = []
    for mid in mcp_ids:
        if mid < n and np.isfinite(arr[mid]).all() and np.isfinite(w).all():
            mcp_norms.append(float(np.linalg.norm(arr[mid] - w)))
    if mcp_norms:
        s = float(np.mean(mcp_norms))
    elif n > 9 and np.isfinite(arr[9]).all():
        s = float(np.linalg.norm(arr[9] - w))
    else:
        s = 1.0
    s = max(s, 1e-3)
    out = rel / s
    return [tuple(float(x) for x in out[i]) for i in range(n)]


def palm_plane_curl_metrics(points_21, *, fingertip_ids_four):
    arr = np.asarray(points_21, dtype=float)
    if arr.shape[0] < 21:
        return None
    four = []
    zs = []
    for i in fingertip_ids_four:
        if i < len(arr) and np.isfinite(arr[i]).all():
            x, y, z = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])
            four.append(float(np.hypot(x, y)))
            zs.append(abs(z))
    thumb_r = None
    thumb_z = None
    if 4 < len(arr) and np.isfinite(arr[4]).all():
        x, y, z = float(arr[4, 0]), float(arr[4, 1]), float(arr[4, 2])
        thumb_r = float(np.hypot(x, y))
        thumb_z = abs(z)
    return {
        "mean_r_xy_four": float(np.mean(four)) if four else None,
        "mean_abs_z_four": float(np.mean(zs)) if zs else None,
        "thumb_r_xy": thumb_r,
        "thumb_abs_z": thumb_z,
    }

