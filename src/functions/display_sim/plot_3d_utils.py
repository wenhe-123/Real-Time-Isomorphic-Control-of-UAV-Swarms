"""Matplotlib 3D axes: hand/topology subplot setup, limits, dual-axis finalize for hand+morph views."""

from __future__ import annotations

import numpy as np


def setup_hand_axis(ax_hand, title: str, *, shape_normalized: bool, hand_frame: str, palm_plane_frame: str):
    """Clear and label the hand 3D subplot axes.

    Args:
        ax_hand: Matplotlib 3D axis for the hand skeleton.
        title: Subplot title string.
        shape_normalized: When ``True``, use normalized axis labels.
        hand_frame: Active hand frame mode (scaled, palm-plane, or metric).
        palm_plane_frame: Constant string for palm-plane frame mode.

    Returns:
        None.
    """
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
    """Clear and label the morph topology 3D subplot axes.

    Args:
        ax_topo: Matplotlib 3D axis for morph topology.
        title: Subplot title string.

    Returns:
        None.
    """
    ax_topo.clear()
    ax_topo.set_title(title)
    ax_topo.set_xlabel("X (mm)")
    ax_topo.set_ylabel("Y (mm)")
    ax_topo.set_zlabel("Z (mm)")


def plot_hand_points_connections(ax_hand, arr: np.ndarray, valid_mask: np.ndarray, connections):
    """Scatter valid hand points and draw skeleton connections on a 3D axis.

    Args:
        ax_hand: Matplotlib 3D axis for the hand skeleton.
        arr: Hand keypoints, shape ``(21, 3)``.
        valid_mask: Boolean mask of finite keypoints to scatter.
        connections: Iterable of ``(a, b)`` skeleton edge index pairs.

    Returns:
        None.
    """
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
    """Set symmetric axis limits centered on the valid hand keypoints.

    Args:
        ax_hand: Matplotlib 3D axis for the hand skeleton.
        arr: Hand keypoints, shape ``(21, 3)``.
        valid_mask: Boolean mask of finite keypoints used for centering.
        shape_normalized: When ``True``, use normalized half-limit constants.
        norm_axis_halflim: Maximum half-span for normalized coordinates.
        morph_axis_lim_mm: Maximum half-span for metric mm coordinates.

    Returns:
        None.
    """
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
    """Apply consistent camera angles, aspect, and topology limits to dual 3D axes.

    Args:
        ax_hand: Matplotlib 3D axis for the hand skeleton, or ``None``.
        ax_topo: Matplotlib 3D axis for morph topology.
        morph_axis_lim_mm: Symmetric half-limit for topology axis bounds in mm.

    Returns:
        None.
    """
    if ax_hand is not None:
        ax_hand.view_init(elev=20, azim=-70)
    ax_topo.view_init(elev=22, azim=-58)
    if ax_hand is not None:
        ax_hand.set_box_aspect((1.0, 1.0, 1.0))
    ax_topo.set_box_aspect((1.0, 1.0, 1.0))
    lim = float(morph_axis_lim_mm)
    ax_topo.set_xlim(-lim, lim)
    ax_topo.set_ylim(-lim, lim)
    ax_topo.set_zlim(-lim, lim)

