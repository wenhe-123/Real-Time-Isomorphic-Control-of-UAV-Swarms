"""
深度相机 + MediaPipe：按需加载 `hand_tracking_orbbec`（避免仅画盒/锥时也依赖 mediapipe）。

使用前将 ``iso_swarm/src`` 加入 PYTHONPATH。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "hto",
    "draw_hand",
    "analyze_hand_topology",
    "update_3d_plot",
    "draw_blanket_morph_canonical",
    "HAND_FRAME_PALM_PLANE",
    "HAND_FRAME_SCALED",
    "HAND_FRAME_METRIC_MM",
    "HAND_3D_SOURCE_MP",
    "HAND_3D_SOURCE_FUSED",
]


def __getattr__(name: str) -> Any:
    import hand_tracking_orbbec as m

    if name == "hto":
        return m
    return getattr(m, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
