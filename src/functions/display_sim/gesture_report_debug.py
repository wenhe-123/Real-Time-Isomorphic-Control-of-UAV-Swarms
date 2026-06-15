"""Debug-only report figures: morph, hand, PCA, open/close landmarks, palm pose (separate windows)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from functions.display_sim.plot_3d_utils import (
    apply_hand_axis_limits,
    plot_hand_points_connections,
    setup_hand_axis,
)
from functions.mode_switch.hand_constants import HAND_CONNECTIONS, MCP_IDS, WRIST_ID
from functions.swarm_motion.left_hand_swarm_pose import (
    palm_frame_origin_mm,
    palm_orthonormal_basis,
    palm_plane_fit_mm,
)

_PCA_DISPLAY_RADIUS_MM = 95.0
_LM_DISPLAY_LIM_MM = 160.0
_PALM_AXIS_LEN_MM = 62.0

# (key, title, x, y)
_WINDOW_SPECS: tuple[tuple[str, str, int, int], ...] = (
    ("morph", "Report: Morph 3D", 40, 40),
    ("hand", "Report: Hand landmarks", 580, 40),
    ("pca", "Report: Hand PCA", 1120, 40),
    ("landmarks", "Report: Open vs Close landmarks", 40, 520),
    ("palm", "Report: Palm pose (cam mm)", 580, 520),
)
_FIG_SIZE = (5.4, 4.8)


@dataclass(frozen=True)
class ReportDebugPanels:
    """Which report Matplotlib windows to open (enable one at a time for screenshots)."""

    morph: bool = False
    hand: bool = False
    pca: bool = False
    landmarks: bool = False
    palm: bool = False

    def any_enabled(self) -> bool:
        return self.morph or self.hand or self.pca or self.landmarks or self.palm

    def extra_enabled(self) -> bool:
        return self.hand or self.pca or self.landmarks or self.palm

    @classmethod
    def all_on(cls) -> ReportDebugPanels:
        return cls(True, True, True, True, True)

    @classmethod
    def from_cli(
        cls,
        *,
        all_viz: bool,
        morph: bool,
        hand: bool,
        pca: bool,
        landmarks: bool,
        palm: bool,
    ) -> ReportDebugPanels:
        if all_viz:
            return cls.all_on()
        return cls(
            morph=bool(morph),
            hand=bool(hand),
            pca=bool(pca),
            landmarks=bool(landmarks),
            palm=bool(palm),
        )

    def enabled_labels(self) -> list[str]:
        out: list[str] = []
        if self.morph:
            out.append("morph")
        if self.hand:
            out.append("hand")
        if self.pca:
            out.append("pca")
        if self.landmarks:
            out.append("open/close")
        if self.palm:
            out.append("palm")
        return out


@dataclass
class ReportDebugFigures:
    panels: ReportDebugPanels
    fig_morph: Any | None = None
    ax_morph: Any | None = None
    fig_hand: Any | None = None
    ax_hand: Any | None = None
    fig_pca: Any | None = None
    ax_pca: Any | None = None
    fig_lm: Any | None = None
    ax_lm: Any | None = None
    fig_palm: Any | None = None
    ax_palm: Any | None = None


def _place_figure(fig, *, x: int, y: int, w: int = 540, h: int = 480) -> None:
    try:
        mgr = fig.canvas.manager
        win = getattr(mgr, "window", None) if mgr is not None else None
        if win is not None:
            if hasattr(win, "wm_geometry"):
                win.wm_geometry(f"{w}x{h}+{x}+{y}")
            elif hasattr(win, "setGeometry"):
                win.setGeometry(x, y, w, h)
    except Exception:
        pass


def _show_figure(fig) -> None:
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass
    try:
        fig.show()
    except Exception:
        plt.show(block=False)


def _raise_figure(fig) -> None:
    try:
        mgr = fig.canvas.manager
        win = getattr(mgr, "window", None) if mgr is not None else None
        if win is not None and hasattr(win, "lift"):
            win.lift()
        elif win is not None and hasattr(win, "raise_"):
            win.raise_()
    except Exception:
        pass


def init_report_debug_figures(panels: ReportDebugPanels) -> ReportDebugFigures:
    """Create only the selected Matplotlib report windows."""
    plt.ion()
    state = ReportDebugFigures(panels=panels)
    first_fig: Any | None = None
    for key, title, x, y in _WINDOW_SPECS:
        enabled = getattr(panels, key)
        if not enabled:
            continue
        fig = plt.figure(title, figsize=_FIG_SIZE)
        ax = fig.add_subplot(111, projection="3d")
        _place_figure(fig, x=x, y=y)
        _show_figure(fig)
        if first_fig is None:
            first_fig = fig
        if key == "morph":
            state.fig_morph, state.ax_morph = fig, ax
        elif key == "hand":
            state.fig_hand, state.ax_hand = fig, ax
        elif key == "pca":
            state.fig_pca, state.ax_pca = fig, ax
        elif key == "landmarks":
            state.fig_lm, state.ax_lm = fig, ax
        elif key == "palm":
            state.fig_palm, state.ax_palm = fig, ax
    if first_fig is not None:
        _raise_figure(first_fig)
        try:
            plt.pause(0.01)
        except Exception:
            pass
    return state


def close_report_debug_figures(state: ReportDebugFigures | None) -> None:
    if state is None:
        return
    for fig in (
        state.fig_morph,
        state.fig_hand,
        state.fig_pca,
        state.fig_lm,
        state.fig_palm,
    ):
        if fig is not None:
            plt.close(fig)


def refresh_report_debug_figures(state: ReportDebugFigures | None) -> None:
    if state is None:
        return
    for fig in (
        state.fig_morph,
        state.fig_hand,
        state.fig_pca,
        state.fig_lm,
        state.fig_palm,
    ):
        if fig is None:
            continue
        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            pass
    try:
        plt.pause(0.001)
    except Exception:
        pass
    for fig in (state.fig_morph, state.fig_hand, state.fig_pca, state.fig_lm, state.fig_palm):
        if fig is not None:
            _raise_figure(fig)
            break


def _center_scale_pts(pts: np.ndarray, *, display_radius: float) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    rel = pts - c
    r = float(np.max(np.linalg.norm(rel, axis=1))) + 1e-6
    s = float(display_radius) / r
    return c, rel * s


def _draw_thick_axis_line(ax, origin: np.ndarray, direction: np.ndarray, length: float, color: str, label: str) -> None:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    v = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return
    v = v / n
    p0 = o - v * length
    p1 = o + v * length
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=color,
        linewidth=4.0,
        solid_capstyle="round",
        label=label,
    )
    ax.scatter([p1[0]], [p1[1]], [p1[2]], c=color, s=55, depthshade=False)


def _draw_pca_plane_patch(ax, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray, half: float) -> None:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    u = np.asarray(e1, dtype=np.float64).reshape(3)
    v = np.asarray(e2, dtype=np.float64).reshape(3)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-8 or nv < 1e-8:
        return
    u, v = u / nu, v / nv
    g = np.linspace(-half, half, 4)
    X = o[0] + g[:, None] * u[0] + g[None, :] * v[0]
    Y = o[1] + g[:, None] * u[1] + g[None, :] * v[1]
    Z = o[2] + g[:, None] * u[2] + g[None, :] * v[2]
    ax.plot_surface(X, Y, Z, color="lightblue", alpha=0.22, linewidth=0)


def update_report_hand_figure(ax_hand, hand_points) -> None:
    """Hand skeleton + joint indices (normalized shape frame)."""
    ax_hand.clear()
    setup_hand_axis(ax_hand, "Hand landmarks + skeleton", shape_normalized=True, hand_frame="scaled", palm_plane_frame="palm_plane")
    arr = np.asarray(hand_points, dtype=float)
    if arr.size == 0:
        ax_hand.text2D(0.25, 0.5, "waiting for hand…", transform=ax_hand.transAxes)
        return
    valid = ~np.isnan(arr[:, 2])
    if not np.any(valid):
        ax_hand.text2D(0.25, 0.5, "no valid 3D joints", transform=ax_hand.transAxes)
        return
    plot_hand_points_connections(ax_hand, arr, valid, HAND_CONNECTIONS)
    for j in range(len(arr)):
        if not bool(valid[j]):
            continue
        ax_hand.text(
            float(arr[j, 0]),
            float(arr[j, 1]),
            float(arr[j, 2]),
            str(j),
            color="k",
            fontsize=7,
            alpha=0.9,
        )
    apply_hand_axis_limits(
        ax_hand,
        arr,
        valid,
        shape_normalized=True,
        norm_axis_halflim=0.55,
        morph_axis_lim_mm=120.0,
    )
    ax_hand.view_init(elev=22, azim=-68)
    ax_hand.set_box_aspect((1.0, 1.0, 1.0))


def update_report_pca_figure(ax_pca, analysis: dict) -> None:
    """Dedicated PCA view: centered landmarks, thick principal axes, PC1–PC2 plane."""
    ax_pca.clear()
    ax_pca.set_title("Hand PCA (topology fit, centered)")
    ax_pca.set_xlabel("X (display mm)")
    ax_pca.set_ylabel("Y (display mm)")
    ax_pca.set_zlabel("Z (display mm)")

    fit_pts = np.asarray(analysis["points"], dtype=np.float64)
    eigvecs = np.asarray(analysis["eigvecs"], dtype=np.float64).reshape(3, 3)
    eigvals = np.asarray(analysis["eigvals"], dtype=np.float64).reshape(3)
    _, pts = _center_scale_pts(fit_pts, display_radius=_PCA_DISPLAY_RADIUS_MM)

    ax_pca.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c="tab:orange",
        s=64,
        alpha=0.9,
        depthshade=False,
        edgecolors="k",
        linewidths=0.4,
        label="fit landmarks",
    )

    lam = np.maximum(eigvals, 1e-12)
    lam_norm = lam / (float(np.sum(lam)) + 1e-8)
    axis_colors = ("tab:red", "tab:green", "tab:blue")
    axis_names = ("λ1 (spread)", "λ2", "λ3 (normal)")
    origin = np.zeros(3)
    half_plane = _PCA_DISPLAY_RADIUS_MM * 0.92
    _draw_pca_plane_patch(ax_pca, origin, eigvecs[:, 0], eigvecs[:, 1], half_plane)

    for i in range(3):
        length = _PCA_DISPLAY_RADIUS_MM * (0.55 + 0.85 * float(lam_norm[i]))
        _draw_thick_axis_line(
            ax_pca,
            origin,
            eigvecs[:, i],
            length,
            axis_colors[i],
            f"{axis_names[i]} ({lam[i]:.2g})",
        )

    ax_pca.text2D(
        0.02,
        0.98,
        f"plan={analysis['planarity']:.2f}  iso={analysis['isotropy']:.2f}  "
        f"λ0/λ2={analysis['span_ratio']:.1f}",
        transform=ax_pca.transAxes,
        fontsize=9,
        va="top",
        color="tab:purple",
    )
    lim = _PCA_DISPLAY_RADIUS_MM * 1.15
    ax_pca.set_xlim(-lim, lim)
    ax_pca.set_ylim(-lim, lim)
    ax_pca.set_zlim(-lim, lim)
    ax_pca.view_init(elev=24, azim=-58)
    ax_pca.set_box_aspect((1.0, 1.0, 1.0))
    ax_pca.legend(loc="lower right", fontsize=7)


def update_report_landmark_cloud_figure(ax_lm, analysis: dict, *, open_out: float | None) -> None:
    """Open vs close vs current topology-fit landmark clouds (hand-centered mm)."""
    ax_lm.clear()
    ax_lm.set_title("Open / Close / Current landmark clouds")
    ax_lm.set_xlabel("X (hand mm)")
    ax_lm.set_ylabel("Y (hand mm)")
    ax_lm.set_zlabel("Z (hand mm)")

    fit_pts = np.asarray(analysis["points"], dtype=np.float64)
    if fit_pts.shape[0] < 4:
        ax_lm.text2D(0.2, 0.5, "topology unavailable", transform=ax_lm.transAxes)
        return
    c = fit_pts.mean(axis=0)
    rel = fit_pts - c
    r_hand = float(np.mean(np.linalg.norm(rel, axis=1))) + 1e-6

    open_pts = rel.copy()
    open_pts[:, 2] = 0.0
    open_pts[:, :2] *= 1.12

    close_r = max(0.32 * r_hand, 18.0)
    norms = np.linalg.norm(rel, axis=1, keepdims=True)
    close_pts = rel / np.maximum(norms, 1e-6) * close_r

    ax_lm.scatter(open_pts[:, 0], open_pts[:, 1], open_pts[:, 2], c="lime", s=52, alpha=0.65, depthshade=False, label="open template")
    ax_lm.scatter(close_pts[:, 0], close_pts[:, 1], close_pts[:, 2], c="crimson", s=52, alpha=0.65, depthshade=False, label="close template")
    ax_lm.scatter(
        rel[:, 0],
        rel[:, 1],
        rel[:, 2],
        c="tab:orange",
        s=58,
        alpha=0.85,
        depthshade=False,
        edgecolors="k",
        linewidths=0.35,
        label="current",
    )
    if open_out is not None:
        ax_lm.text2D(
            0.02,
            0.98,
            f"open_ctrl={float(open_out):.2f}  morph_α={analysis['morph_alpha']:.2f}",
            transform=ax_lm.transAxes,
            fontsize=9,
            va="top",
        )
    lim = _LM_DISPLAY_LIM_MM
    ax_lm.set_xlim(-lim, lim)
    ax_lm.set_ylim(-lim, lim)
    ax_lm.set_zlim(-lim, lim)
    ax_lm.view_init(elev=20, azim=-62)
    ax_lm.set_box_aspect((1.0, 1.0, 1.0))
    ax_lm.legend(loc="lower right", fontsize=7)


def _draw_basis_triad(ax, origin: np.ndarray, basis: np.ndarray, *, scale: float, dashed: bool) -> None:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    B = np.asarray(basis, dtype=np.float64).reshape(3, 3)
    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("X", "Y", "Z")
    ls = (0, (4, 3)) if dashed else "-"
    for i in range(3):
        tip = o + B[:, i] * float(scale)
        ax.plot(
            [o[0], tip[0]],
            [o[1], tip[1]],
            [o[2], tip[2]],
            color=colors[i],
            linewidth=3.0 if not dashed else 2.0,
            linestyle=ls,
            label=f"{'ref ' if dashed else ''}{labels[i]}",
        )


def update_report_palm_pose_figure(
    ax_palm,
    pts_l_pose_mm,
    *,
    left_pose_state: Any | None = None,
) -> None:
    """Palm plane + orthonormal basis in depth-camera mm."""
    ax_palm.clear()
    ax_palm.set_title("Palm pose (depth cam mm)")
    ax_palm.set_xlabel("X (mm)")
    ax_palm.set_ylabel("Y (mm)")
    ax_palm.set_zlabel("Z (mm)")

    if pts_l_pose_mm is None:
        ax_palm.text2D(0.3, 0.5, "no left-hand depth pose", transform=ax_palm.transAxes)
        return
    h = np.asarray(pts_l_pose_mm, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 21 or h.shape[1] < 3:
        ax_palm.text2D(0.25, 0.5, "invalid hand array", transform=ax_palm.transAxes)
        return

    origin = palm_frame_origin_mm(h)
    fit = palm_plane_fit_mm(h)
    basis_out = palm_orthonormal_basis(h, palm_center_override=origin)
    if basis_out is None:
        ax_palm.text2D(0.25, 0.5, "palm basis unavailable", transform=ax_palm.transAxes)
        return
    pc, basis = basis_out
    origin = np.asarray(pc if origin is None else origin, dtype=np.float64).reshape(3)

    palm_ids = [WRIST_ID] + list(MCP_IDS)
    palm_pts = np.array([h[i] for i in palm_ids if i < len(h)], dtype=np.float64)
    ax_palm.scatter(palm_pts[:, 0], palm_pts[:, 1], palm_pts[:, 2], c="tab:orange", s=42, alpha=0.9, depthshade=False, label="wrist+MCP")
    ax_palm.scatter([origin[0]], [origin[1]], [origin[2]], c="k", s=80, marker="*", depthshade=False, label="palm origin")

    if fit is not None:
        _n, _hp, _inliers = fit
        u = basis[:, 0]
        v = basis[:, 1]
        g = np.linspace(-0.55, 0.55, 5) * _PALM_AXIS_LEN_MM
        X = origin[0] + g[:, None] * u[0] + g[None, :] * v[0]
        Y = origin[1] + g[:, None] * u[1] + g[None, :] * v[1]
        Z = origin[2] + g[:, None] * u[2] + g[None, :] * v[2]
        ax_palm.plot_surface(X, Y, Z, color="wheat", alpha=0.35, linewidth=0)

    _draw_basis_triad(ax_palm, origin, basis, scale=_PALM_AXIS_LEN_MM, dashed=False)

    if left_pose_state is not None and bool(getattr(left_pose_state, "initialized", False)):
        ref_o = np.asarray(left_pose_state.ref_palm_center, dtype=np.float64).reshape(3)
        ref_b = np.asarray(left_pose_state.ref_basis, dtype=np.float64).reshape(3, 3)
        _draw_basis_triad(ax_palm, ref_o, ref_b, scale=_PALM_AXIS_LEN_MM * 0.92, dashed=True)
        ax_palm.scatter([ref_o[0]], [ref_o[1]], [ref_o[2]], c="0.45", s=55, marker="o", depthshade=False, label="ref origin")

    ctr = origin
    span = float(np.max(np.ptp(palm_pts, axis=0)))
    half = max(75.0, 0.65 * span + 40.0)
    ax_palm.set_xlim(ctr[0] - half, ctr[0] + half)
    ax_palm.set_ylim(ctr[1] - half, ctr[1] + half)
    ax_palm.set_zlim(ctr[2] - half, ctr[2] + half)
    ax_palm.view_init(elev=18, azim=-72)
    ax_palm.set_box_aspect((1.0, 1.0, 1.0))
    ax_palm.legend(loc="upper left", fontsize=7)


def update_report_debug_figures(
    state: ReportDebugFigures,
    *,
    hand_points,
    analysis: dict | None,
    open_out: float | None,
    pts_l_pose_mm,
    left_pose_state: Any | None,
) -> None:
    p = state.panels
    if p.hand and state.ax_hand is not None:
        update_report_hand_figure(state.ax_hand, hand_points if hand_points is not None else [])
    if p.pca and state.ax_pca is not None:
        if analysis is not None:
            update_report_pca_figure(state.ax_pca, analysis)
        else:
            state.ax_pca.clear()
            state.ax_pca.set_title("waiting for hand topology…")
    if p.landmarks and state.ax_lm is not None:
        if analysis is not None:
            update_report_landmark_cloud_figure(state.ax_lm, analysis, open_out=open_out)
        else:
            state.ax_lm.clear()
            state.ax_lm.set_title("waiting for hand topology…")
    if p.palm and state.ax_palm is not None:
        update_report_palm_pose_figure(state.ax_palm, pts_l_pose_mm, left_pose_state=left_pose_state)
    refresh_report_debug_figures(state)
