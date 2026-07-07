"""Matplotlib debug overlay: morph-only vs L-move formation targets."""

from __future__ import annotations

import numpy as np

from functions.swarm_motion.left_hand_swarm_pose import R_to_rotvec


def _clear_formation_rigid_debug(ax_topo) -> None:
    for art in getattr(ax_topo, "_iso_swarm_debug_artists", None) or []:
        try:
            art.remove()
        except Exception:
            pass
    ax_topo._iso_swarm_debug_artists = []
    tx = getattr(ax_topo, "_iso_swarm_debug_text", None)
    if tx is not None:
        try:
            tx.remove()
        except Exception:
            pass
    ax_topo._iso_swarm_debug_text = None


def draw_formation_rigid_debug_on_topo(
    ax_topo,
    p_before_m: np.ndarray,
    p_after_m: np.ndarray,
    *,
    off_m: np.ndarray | None = None,
    R_pose: np.ndarray | None = None,
) -> None:
    """Overlay morph-only vs L-move targets on the topology axis for debug.

    Blue dots show morph-only targets before the left-hand rigid transform; magenta
    triangles show targets after :func:`apply_rigid_to_targets`. Used when
    ``--formation-rigid-3d-debug`` is enabled.

    Args:
        ax_topo: Matplotlib 3D topology axis modified in place.
        p_before_m: Morph-only drone targets in meters, shape ``(N, 3)``.
        p_after_m: Targets after L-move rigid transform, shape ``(N, 3)``.
        off_m: Swarm translation offset in meters, shape ``(3,)``, or ``None``.
        R_pose: Swarm rotation matrix, shape ``(3, 3)``, or ``None``.

    Returns:
        None.
    """
    _clear_formation_rigid_debug(ax_topo)
    pb = np.asarray(p_before_m, dtype=np.float64)
    pa = np.asarray(p_after_m, dtype=np.float64)
    if pb.ndim != 2 or pa.ndim != 2 or pb.shape != pa.shape or pb.shape[1] < 3:
        return
    c = np.mean(pb[:, :3], axis=0)
    d0 = pb[:, :3] - c
    d1 = pa[:, :3] - c
    r = max(
        float(np.max(np.linalg.norm(d0, axis=1))),
        float(np.max(np.linalg.norm(d1, axis=1))),
        0.12,
    )
    s = 85.0 / r
    q0 = d0 * s
    q1 = d1 * s
    art0 = ax_topo.scatter(
        q0[:, 0],
        q0[:, 1],
        q0[:, 2],
        c="tab:blue",
        s=22,
        alpha=0.88,
        depthshade=False,
        label="morph cmd",
    )
    art1 = ax_topo.scatter(
        q1[:, 0],
        q1[:, 1],
        q1[:, 2],
        c="tab:magenta",
        s=40,
        alpha=0.92,
        marker="^",
        depthshade=False,
        edgecolors="k",
        linewidths=0.45,
        label="+ L-move",
    )
    ax_topo._iso_swarm_debug_artists = [art0, art1]
    if off_m is not None and R_pose is not None:
        offv = np.asarray(off_m, dtype=np.float64).ravel()
        R = np.asarray(R_pose, dtype=np.float64).reshape(3, 3)
        rv = R_to_rotvec(R)
        ang = float(np.linalg.norm(rv))
        if ang > 1e-8:
            ax = rv / ang
            rot_s = f"rot≈{float(np.degrees(ang)):+.1f}°  n̂≈({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})"
        else:
            rot_s = "rot≈0°"
        t = (
            f"L-swarm: |Δ|={float(np.linalg.norm(offv)):.3f}m  "
            f"Δ=({offv[0]:+.2f},{offv[1]:+.2f},{offv[2]:+.2f})  {rot_s}"
        )
        try:
            tx = ax_topo.text2D(
                0.02,
                0.98,
                t,
                transform=ax_topo.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="tab:purple",
            )
            ax_topo._iso_swarm_debug_text = tx
        except Exception:
            ax_topo._iso_swarm_debug_text = None
