"""
小块组合：`平面↔球` 复用 `hand_tracking_orbbec`，`平面↔盒/锥` 在本包内实现。

仅使用 ``draw_morph`` / 盒锥几何时，可不装 mediapipe；用到 ``depth_gesture_facade`` 或 ``pipeline`` 时再加载主脚本。

示例::

    export PYTHONPATH=src
    python3 -c "from hand_morph_kit import draw_morph; ..."
"""

from __future__ import annotations

from .compose import MorphKind, draw_morph
from .morph_plane_box import draw_plane_box_morph
from .morph_plane_pyramid import draw_plane_pyramid_morph
from .morph_plane_sphere import draw_plane_sphere_morph
from .pipeline import draw_morph_for_hand, morph_alpha_from_hand, suggested_morph_t

__all__ = [
    "MorphKind",
    "draw_morph",
    "draw_plane_sphere_morph",
    "draw_plane_box_morph",
    "draw_plane_pyramid_morph",
    "draw_morph_for_hand",
    "morph_alpha_from_hand",
    "suggested_morph_t",
]


def __getattr__(name: str):
    """懒加载深度/手势门面（``draw_hand``, ``hto``, …）。"""
    if name in (
        "hto",
        "draw_hand",
        "analyze_hand_topology",
        "update_3d_plot",
        "HAND_FRAME_PALM_PLANE",
        "HAND_FRAME_SCALED",
        "HAND_FRAME_METRIC_MM",
        "HAND_3D_SOURCE_MP",
        "HAND_3D_SOURCE_FUSED",
    ):
        from . import depth_gesture_facade as dg

        return getattr(dg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | {"hto", "draw_hand", "analyze_hand_topology", "update_3d_plot"})
