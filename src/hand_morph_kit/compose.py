"""
把「平面 → 球 / 盒 / 锥」画到同一个 Axes3D 上，参数统一为 morph_t：

- **morph_t = 0**：平面圆盘一侧（与球毯语义对齐时，球模式用 ``open_alpha = 1 - morph_t``）。
- **morph_t = 1**：目标形状（球 / 盒 / 锥）。

``sphere`` 与主脚本一致：``open_alpha=1`` 为平面，``open_alpha=0`` 为球。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

MorphKind = Literal["sphere", "box", "pyramid"]


def draw_morph(
    ax,
    kind: MorphKind,
    radius: float,
    morph_t: float,
    *,
    show_refs: bool = True,
    apex_height: float | None = None,
) -> None:
    """
    Parameters
    ----------
    morph_t
        0..1 插值进度；球模式内部转换为 ``open_alpha = 1 - morph_t``。
    apex_height
        仅 ``pyramid``：锥顶高度，默认约 ``1.2 * radius``。
    """
    t = float(np.clip(morph_t, 0.0, 1.0))
    if kind == "sphere":
        from .morph_plane_sphere import draw_plane_sphere_morph

        draw_plane_sphere_morph(ax, radius, open_alpha=1.0 - t, show_refs=show_refs)
    elif kind == "box":
        from .morph_plane_box import draw_plane_box_morph

        draw_plane_box_morph(ax, radius, t, show_refs=show_refs)
    elif kind == "pyramid":
        from .morph_plane_pyramid import draw_plane_pyramid_morph

        draw_plane_pyramid_morph(ax, radius, t, show_refs=show_refs, apex_height=apex_height)
    else:
        raise ValueError(f"unknown kind: {kind!r}")
