#!/usr/bin/env python3
"""Print initial plane cmd_target (sim m) for drone indices 0..n-1.

Same path as ``online_control`` startup: ``fixed_morph_points(open=1)`` →
``normalize_morph_points_at_hover`` with CLI defaults (mode=1, radius_mm=50).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from functions.open_close.morph_renderers import init_fixed_surface_points
from functions.open_close.morph_world import ScaleConfig, fixed_morph_points, normalize_morph_points_at_hover
from functions.runtime.online_defaults import ONLINE_DEFAULTS


def main() -> None:
    n = int(sys.argv[1])
    if n < 8:
        raise SystemExit("n must be >= 8")
    init_fixed_surface_points(n)
    pts_mm = fixed_morph_points(n, 50.0, 1, 1.0, None)
    scale = ScaleConfig(
        xy_radius=3.0,
        hover_z=float(ONLINE_DEFAULTS.prearm.prearm_hover_z),
        z_amplitude=0.35,
        reference_xy_extent_mm=100.0,
        reference_z_extent_mm=100.0,
        morph_world_scale=float(ONLINE_DEFAULTS.morph.morph_world_scale),
    )
    tgt = normalize_morph_points_at_hover(pts_mm, scale)
    for i, p in enumerate(tgt):
        print(f"{i} {float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}")


if __name__ == "__main__":
    main()
