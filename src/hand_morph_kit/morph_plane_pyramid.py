"""
平面（圆盘）↔ 方锥：底面为正方形 [-R,R]^2（z=0），锥顶 (0,0,H)。
目标点取圆盘 (xb,yb,0) 在底面上的投影截断到正方形后，沿「底面→锥顶」的竖直剖面高度场
z = H * (1 - max(|sx|,|sy|)/R) 的屋顶面（金字塔侧面与顶脊的组合近似，便于网格绘制）。
"""

from __future__ import annotations

import numpy as np

_DEFAULT_THETA_N = 28
_DEFAULT_RHO_N = 14


def _polar_mesh(radius: float, theta_n: int, rho_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, theta_n)
    rho = np.linspace(0.0, 1.0, rho_n)
    th, rr = np.meshgrid(theta, rho)
    x0 = radius * rr * np.cos(th)
    y0 = radius * rr * np.sin(th)
    z0 = np.zeros_like(x0)
    return x0, y0, z0


def draw_plane_pyramid_morph(
    ax,
    radius: float,
    morph_t: float,
    *,
    apex_height: float | None = None,
    theta_n: int | None = None,
    rho_n: int | None = None,
    show_refs: bool = True,
) -> None:
    """
    morph_t ∈ [0, 1]: 0 = 平面圆盘，1 = 金字塔目标表面。
    """
    theta_n = int(theta_n if theta_n is not None else _DEFAULT_THETA_N)
    rho_n = int(rho_n if rho_n is not None else _DEFAULT_RHO_N)

    morph_t = float(max(0.0, min(1.0, morph_t)))
    R = float(radius)
    H = float(apex_height if apex_height is not None else 1.2 * R)

    x0, y0, z0 = _polar_mesh(R, theta_n, rho_n)
    sx = np.clip(x0, -R, R)
    sy = np.clip(y0, -R, R)
    m = np.maximum(np.abs(sx), np.abs(sy)) / max(R, 1e-6)
    z1 = H * (1.0 - m)

    x = (1.0 - morph_t) * x0 + morph_t * sx
    y = (1.0 - morph_t) * y0 + morph_t * sy
    z = (1.0 - morph_t) * z0 + morph_t * z1

    if show_refs:
        _draw_pyramid_wire(ax, R, H, color="0.75", alpha=0.45)

    ax.plot_surface(x, y, z, color="tab:orange", alpha=0.42, linewidth=0.2, edgecolor="darkorange", antialiased=True, shade=True)


def _draw_pyramid_wire(ax, R: float, H: float, *, color: str, alpha: float) -> None:
    base = [
        [-R, -R, 0],
        [R, -R, 0],
        [R, R, 0],
        [-R, R, 0],
        [-R, -R, 0],
    ]
    arr = np.array(base, dtype=float)
    ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, linewidth=0.8, alpha=alpha)
    apex = np.array([0.0, 0.0, H])
    for i in range(4):
        ax.plot(
            [base[i][0], apex[0]],
            [base[i][1], apex[1]],
            [base[i][2], apex[2]],
            color=color,
            linewidth=0.8,
            alpha=alpha,
        )
