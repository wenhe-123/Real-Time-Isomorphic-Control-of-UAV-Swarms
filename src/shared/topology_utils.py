"""Hand topology analysis: spread, planarity, isotropy, morph_alpha, topology labels, open remap helpers."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def topology_label_from_alpha(alpha: float, plane_thr: float = 0.67, sphere_thr: float = 0.33) -> str:
    a = float(alpha)
    if a > float(plane_thr):
        return "plane"
    if a < float(sphere_thr):
        return "sphere"
    return "intermediate"


def remap_open_display(alpha: float, lo: float, hi: float) -> float:
    return clamp01((float(alpha) - lo) / max(float(hi - lo), 1e-6))


def analyze_hand_topology_common(
    hand_points: Sequence[Sequence[float]],
    *,
    wrist_id: int,
    mcp_ids: Sequence[int],
    fingertip_ids: Sequence[int],
    open_gamma: float,
    label_fn: Optional[Callable[[float], str]] = None,
):
    """Compute PCA/topology metrics with configurable hand-index constants."""
    all_pts = np.array(hand_points, dtype=float)
    valid_all = ~np.isnan(all_pts[:, 2])
    if np.sum(valid_all) < 8:
        return None

    fit_ids = [i for i in range(len(hand_points)) if i != wrist_id]
    fit_pts = np.array(
        [hand_points[i] for i in fit_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    if fit_pts.shape[0] < 7:
        fit_pts = all_pts[valid_all]

    centroid = fit_pts.mean(axis=0)
    centered = fit_pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    normal = safe_normalize(eigvecs[:, 2])
    lamb_sum = float(np.sum(eigvals)) + 1e-8
    planarity = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-8))
    isotropy = float(eigvals[2] / lamb_sum)
    span_ratio = float(eigvals[0] / (eigvals[2] + 1e-8))

    palm_ids = [wrist_id] + list(mcp_ids)
    palm_pts = np.array(
        [hand_points[i] for i in palm_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    tip_pts = np.array(
        [hand_points[i] for i in fingertip_ids if i < len(hand_points) and not np.isnan(hand_points[i][2])],
        dtype=float,
    )
    if palm_pts.shape[0] > 0:
        palm_center = palm_pts.mean(axis=0)
    else:
        palm_center = centroid

    hand_scale = float(np.mean(np.linalg.norm(centered, axis=1))) + 1e-6
    if tip_pts.shape[0] > 0:
        tip_dist = np.linalg.norm(tip_pts - palm_center, axis=1)
        finger_spread = float(np.mean(tip_dist) / hand_scale)
    else:
        finger_spread = 0.0

    spread_score = clamp01((finger_spread - 1.00) / (1.65 - 1.00))
    planarity_score = clamp01((planarity - 0.12) / (0.55 - 0.12))
    isotropy_score = clamp01((isotropy - 0.06) / (0.22 - 0.06))
    alpha = clamp01(0.50 * spread_score + 0.35 * planarity_score + 0.15 * (1.0 - isotropy_score))
    alpha = clamp01(alpha ** float(open_gamma))

    if label_fn is None:
        topology = topology_label_from_alpha(alpha)
    else:
        topology = label_fn(alpha)

    radius = float(np.mean(np.linalg.norm(fit_pts - centroid, axis=1)))
    return {
        "centroid": centroid,
        "normal": normal,
        "eigvecs": eigvecs,
        "planarity": planarity,
        "isotropy": isotropy,
        "span_ratio": span_ratio,
        "finger_spread": finger_spread,
        "morph_alpha": alpha,
        "topology": topology,
        "radius": radius,
        "points": fit_pts,
    }

