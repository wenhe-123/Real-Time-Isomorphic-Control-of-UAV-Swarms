"""
把手部 3D 点列与拓扑 ``morph_alpha`` 接到 ``compose.draw_morph`` 的示例映射。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .compose import MorphKind, draw_morph


def morph_alpha_from_hand(hand_points: list[tuple[float, float, float]]) -> float | None:
    """单帧：与主程序相同的拓扑标量；失败时返回 None。"""
    import hand_tracking_orbbec as hto

    a = hto.analyze_hand_topology(hand_points)
    if a is None:
        return None
    return float(a["morph_alpha"])


def suggested_morph_t(morph_alpha: float | None, *, kind: MorphKind) -> float:
    """
    将 [0,1] 的 morph_alpha（握拳→张开平面）映到各 morph 的 ``morph_t``。

    - **sphere**：``morph_t = 1 - morph_alpha``（张手→平面 disk，握拳→球）。
    - **box / pyramid**：默认 ``morph_t = morph_alpha``（可按实验反号）。
    """
    if morph_alpha is None:
        return 0.5
    m = float(np.clip(morph_alpha, 0.0, 1.0))
    if kind == "sphere":
        return 1.0 - m
    return m


def draw_morph_for_hand(
    ax,
    hand_points: list[tuple[float, float, float]],
    kind: MorphKind,
    radius: float = 200.0,
    *,
    override_t: float | None = None,
    show_refs: bool = True,
) -> dict[str, Any]:
    """
    一键：``analyze_hand_topology`` → ``suggested_morph_t`` → ``draw_morph``。

    若提供 ``override_t``，则跳过估计，直接用作 ``morph_t``。
    """
    alpha = morph_alpha_from_hand(hand_points)
    t = float(override_t) if override_t is not None else suggested_morph_t(alpha, kind=kind)
    draw_morph(ax, kind, radius, t, show_refs=show_refs)
    return {"morph_alpha": alpha, "morph_t": t, "kind": kind}
