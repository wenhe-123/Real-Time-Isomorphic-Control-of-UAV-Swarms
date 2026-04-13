"""
平面 ↔ 球（毯面 / blanket）— 直接委托 `hand_tracking_orbbec.draw_blanket_morph_canonical`。
不修改原文件，仅在此处做薄封装便于与其它 morph 模块并列导入。
"""

from __future__ import annotations


def draw_plane_sphere_morph(ax, radius: float, open_alpha: float, *, show_refs: bool = True) -> None:
    """
    open_alpha: 与主脚本一致 — 1 = 平面（张开），0 = 球（握拳侧）。
    """
    import hand_tracking_orbbec as hto

    hto.draw_blanket_morph_canonical(ax, radius, open_alpha, show_refs=show_refs)
