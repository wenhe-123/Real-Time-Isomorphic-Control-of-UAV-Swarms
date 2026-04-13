"""
平面（圆盘）↔ 轴对齐立方体表面：同一极坐标网格上，圆盘点与「从原点射向该方向的盒面出射点」做线性混合。
"""

from __future__ import annotations

import numpy as np

from .geometry_ray_aabb import ray_box_exit_from_origin

_DEFAULT_THETA_N = 28
_DEFAULT_RHO_N = 14


def _polar_mesh(radius: float, theta_n: int, rho_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, theta_n)
    rho = np.linspace(0.0, 1.0, rho_n)
    th, rr = np.meshgrid(theta, rho)
    x0 = radius * rr * np.cos(th)
    y0 = radius * rr * np.sin(th)
    z0 = np.zeros_like(x0)
    return x0, y0, z0, th, rr


def draw_plane_box_morph(
    ax,
    radius: float,
    morph_t: float,
    *,
    theta_n: int | None = None,
    rho_n: int | None = None,
    show_refs: bool = True,
) -> None:
    """
    morph_t ∈ [0, 1]: 0 = 平面圆盘 (z=0)，1 = 贴靠在 [-R,R]^3 盒表面上的目标点。

    与毯面球语义区分：此处 **morph_t=0 为平面**，**morph_t=1 为盒子**。
    """
    theta_n = int(theta_n if theta_n is not None else _DEFAULT_THETA_N)
    rho_n = int(rho_n if rho_n is not None else _DEFAULT_RHO_N)

    morph_t = float(max(0.0, min(1.0, morph_t)))
    R = float(radius)
    half = R

    x0, y0, z0, _, _ = _polar_mesh(R, theta_n, rho_n)
    zb = np.full_like(x0, max(R * 0.03, 0.5))
    dx = x0
    dy = y0
    dz = zb
    inv = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
    dx *= inv
    dy *= inv
    dz *= inv
    bx, by, bz, _ = ray_box_exit_from_origin(dx, dy, dz, half)

    x = (1.0 - morph_t) * x0 + morph_t * bx
    y = (1.0 - morph_t) * y0 + morph_t * by
    z = (1.0 - morph_t) * z0 + morph_t * bz

    if show_refs:
        _draw_box_wire(ax, half, color="0.75", alpha=0.45)

    ax.plot_surface(x, y, z, color="tab:cyan", alpha=0.42, linewidth=0.2, edgecolor="tab:blue", antialiased=True, shade=True)


def _draw_box_wire(ax, half: float, *, color: str = "0.7", alpha: float = 0.5) -> None:
    h = float(half)
    c = [
        [-h, -h, -h],
        [h, -h, -h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, h],
        [-h, h, h],
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        p0, p1 = c[a], c[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=0.8, alpha=alpha)
