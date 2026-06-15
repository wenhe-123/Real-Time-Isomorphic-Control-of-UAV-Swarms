"""Dual-view fusion: PCA-based geometry weights (planarity/isotropy) and visibility-weighted blend of two MP world hands."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _pca_eigenvalues_descending_mm(pts_21: List[Tuple[float, float, float]]) -> Optional[np.ndarray]:
    """PCA eigenvalues on finite points, sorted descending."""
    arr = np.asarray(pts_21, dtype=float)
    if arr.size != 63:
        return None
    mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    sub = arr[mask]
    if sub.shape[0] < 4:
        return None
    centered = sub - sub.mean(axis=0)
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return None
    ev = np.linalg.eigh(cov)[0]
    return np.sort(ev.astype(float))[::-1]


def _w_geom_from_planarity_isotropy(planarity: float, isotropy: float) -> float:
    ps = float(np.clip((planarity - 0.12) / (0.55 - 0.12), 0.0, 1.0))
    isos = float(np.clip((isotropy - 0.06) / (0.22 - 0.06), 0.0, 1.0))
    w = float(0.5 * ps + 0.5 * (1.0 - isos))
    return float(np.clip(w, 0.05, 1.0))


def _w_geom_from_eigenvalues(ev: np.ndarray) -> float:
    l0, l1, l2 = float(ev[0]), float(ev[1]), float(ev[2])
    ls = l0 + l1 + l2 + 1e-8
    planarity = (l1 - l2) / (l0 + 1e-8)
    isotropy = l2 / ls
    return _w_geom_from_planarity_isotropy(planarity, isotropy)


def geom_weight_from_eigen_analysis(
    pts_21: List[Tuple[float, float, float]],
    analyze_topology: Callable[[List[Tuple[float, float, float]]], Optional[Dict[str, Any]]],
) -> Tuple[float, np.ndarray]:
    """
    Scalar geometry confidence from PCA-like topology, with fallback eigen-analysis.
    Returns (w_geom, eigvals_descending); degenerate -> (0.15, nan x3).
    """
    ev_fb = _pca_eigenvalues_descending_mm(pts_21)
    an = analyze_topology(pts_21)
    if an is not None:
        p = float(an["planarity"])
        iso = float(an["isotropy"])
        w_geom = _w_geom_from_planarity_isotropy(p, iso)
        ev = np.linalg.eigh(np.cov(np.asarray(an["points"], dtype=float).T))[0]
        ev = np.sort(ev)[::-1]
        return w_geom, ev
    if ev_fb is not None:
        w_geom = _w_geom_from_eigenvalues(ev_fb)
        return w_geom, ev_fb
    return 0.15, np.full(3, np.nan)


def fuse_dual_views_weighted(
    pts_o: Optional[List[Tuple[float, float, float]]],
    pts_l: Optional[List[Tuple[float, float, float]]],
    vis_o: Optional[np.ndarray],
    vis_l: Optional[np.ndarray],
    w_geom_o: float,
    w_geom_l: float,
    *,
    conf_low: float,
    conf_high: float,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
    """
    Per joint: c_{v,k} = vis_{v,k} * w_geom_v.
    If any c >= conf_high, choose best view; else do weighted average.
    """
    fused: List[Tuple[float, float, float]] = []
    w_sum_o = 0.0
    w_sum_l = 0.0
    n_joint = 0
    n_high_pick = 0

    for k in range(21):
        cand: List[Tuple[str, np.ndarray, float]] = []
        for name, pts, vis, wg in (
            ("O", pts_o, vis_o, w_geom_o),
            ("L", pts_l, vis_l, w_geom_l),
        ):
            if pts is None or vis is None or not np.isfinite(float(wg)):
                continue
            p = pts[k]
            if p is None or len(p) < 3 or not np.isfinite(p[2]):
                continue
            c = float(vis[k]) * float(wg)
            if c < conf_low:
                continue
            cand.append((name, np.array(p, dtype=float), c))

        if not cand:
            fused.append((np.nan, np.nan, np.nan))
            continue

        hi = [t for t in cand if t[2] >= conf_high]
        if hi:
            best = max(hi, key=lambda t: t[2])
            fused.append((float(best[1][0]), float(best[1][1]), float(best[1][2])))
            n_high_pick += 1
            if best[0] == "O":
                w_sum_o += 1.0
            else:
                w_sum_l += 1.0
            n_joint += 1
            continue

        sw = sum(t[2] for t in cand)
        acc = sum(t[2] * t[1] for t in cand)
        p_f = acc / max(sw, 1e-12)
        fused.append((float(p_f[0]), float(p_f[1]), float(p_f[2])))
        for name, _vec, c in cand:
            frac = c / sw
            if name == "O":
                w_sum_o += frac
            else:
                w_sum_l += frac
        n_joint += 1

    if n_joint <= 0:
        wmo, wml = 0.0, 0.0
    else:
        wmo = float(w_sum_o / n_joint)
        wml = float(w_sum_l / n_joint)
    dbg = {
        "w_mean_o": wmo,
        "w_mean_l": wml,
        "n_joints_fused": n_joint,
        "n_high_exclusive": n_high_pick,
    }
    return fused, dbg

