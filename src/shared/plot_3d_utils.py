"""Matplotlib 3D axes: hand/topology subplot setup, limits, dual-axis finalize for hand+morph views."""

from __future__ import annotations

import numpy as np


def setup_hand_axis(ax_hand, title: str, *, shape_normalized: bool, hand_frame: str, palm_plane_frame: str):
    ax_hand.clear()
    ax_hand.set_title(title)
    if shape_normalized:
        if hand_frame == palm_plane_frame:
            ax_hand.set_xlabel("X (norm, along mid MCP)")
            ax_hand.set_ylabel("Y (norm, in palm)")
            ax_hand.set_zlabel("Z (norm, ⊥ palm)")
        else:
            ax_hand.set_xlabel("X (norm)")
            ax_hand.set_ylabel("Y (norm)")
            ax_hand.set_zlabel("Z (norm, rel. depth)")
    else:
        ax_hand.set_xlabel("X (mm)")
        ax_hand.set_ylabel("Y (mm)")
        ax_hand.set_zlabel("Z (mm)")


def setup_topology_axis(ax_topo, title: str):
    ax_topo.clear()
    ax_topo.set_title(title)
    ax_topo.set_xlabel("X (mm)")
    ax_topo.set_ylabel("Y (mm)")
    ax_topo.set_zlabel("Z (mm)")


def plot_hand_points_connections(ax_hand, arr: np.ndarray, valid_mask: np.ndarray, connections):
    pts = arr[valid_mask]
    ax_hand.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="r", s=20)
    for a, b in connections:
        if a < len(arr) and b < len(arr) and not np.isnan(arr[a, 2]) and not np.isnan(arr[b, 2]):
            ax_hand.plot(
                [arr[a, 0], arr[b, 0]],
                [arr[a, 1], arr[b, 1]],
                [arr[a, 2], arr[b, 2]],
                "b-",
                linewidth=1.2,
            )


def apply_hand_axis_limits(
    ax_hand,
    arr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    shape_normalized: bool,
    norm_axis_halflim: float,
    morph_axis_lim_mm: float,
):
    sub = arr[valid_mask]
    ctr_hand = sub.mean(axis=0)
    span = float(np.max(np.ptp(sub, axis=0)))
    if shape_normalized:
        half = min(float(norm_axis_halflim), max(0.35, 0.55 * span + 0.18))
    else:
        half = min(float(morph_axis_lim_mm), max(120.0, 0.55 * span + 90.0))
    ax_hand.set_xlim(ctr_hand[0] - half, ctr_hand[0] + half)
    ax_hand.set_ylim(ctr_hand[1] - half, ctr_hand[1] + half)
    ax_hand.set_zlim(ctr_hand[2] - half, ctr_hand[2] + half)


def finalize_dual_3d_axes(ax_hand, ax_topo, *, morph_axis_lim_mm: float):
    ax_hand.view_init(elev=20, azim=-70)
    ax_topo.view_init(elev=22, azim=-58)
    ax_hand.set_box_aspect((1.0, 1.0, 1.0))
    ax_topo.set_box_aspect((1.0, 1.0, 1.0))
    lim = float(morph_axis_lim_mm)
    ax_topo.set_xlim(-lim, lim)
    ax_topo.set_ylim(-lim, lim)
    ax_topo.set_zlim(-lim, lim)

