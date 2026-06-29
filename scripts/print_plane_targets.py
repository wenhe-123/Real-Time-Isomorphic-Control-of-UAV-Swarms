#!/usr/bin/env python3
"""Print hover plane targets: sim frame and room (mocap) frame.

Same morph path as ``online_control`` startup: ``fixed_morph_points(open=1)`` →
``normalize_morph_points_at_hover``. Room coords use ``[frame]`` from
``config/drones.toml`` (``origin``, ``scale``, ``yaw_deg``).

Usage::

    python scripts/print_plane_targets.py 8
    python scripts/print_plane_targets.py 8 config/drones.toml
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from functions.open_close.morph_renderers import init_fixed_surface_points
from functions.open_close.morph_world import ScaleConfig, fixed_morph_points, normalize_morph_points_at_hover
from functions.real_swarm.swarm_config import load_drones_config
from functions.runtime.online_defaults import ONLINE_DEFAULTS


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: print_plane_targets.py N [drones.toml]")
    n = int(sys.argv[1])
    if n < 8:
        raise SystemExit("N must be >= 8")
    drones_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _REPO / "config" / "drones.toml"
    _, mapping, _ = load_drones_config(drones_path)

    init_fixed_surface_points(n)
    pts_mm = fixed_morph_points(n, 50.0, 1, 1.0, None)
    scale = ScaleConfig(
        xy_radius=3.0,
        hover_z=float(ONLINE_DEFAULTS.prearm.prearm_hover_z),
        z_amplitude=0.55,
        reference_xy_extent_mm=100.0,
        reference_z_extent_mm=100.0,
        morph_world_scale=float(ONLINE_DEFAULTS.morph.morph_world_scale),
    )
    sim = normalize_morph_points_at_hover(pts_mm, scale)
    print(
        f"# frame from {drones_path}: origin={mapping.origin.tolist()} "
        f"scale={mapping.scale} yaw_deg={math.degrees(mapping.yaw_rad):.1f}"
    )
    print("# idx  sim_x   sim_y   sim_z   room_x  room_y  room_z")
    for i, p in enumerate(sim):
        r = mapping.sim_to_real(p)
        print(
            f"{i} "
            f"{float(p[0]):8.4f} {float(p[1]):8.4f} {float(p[2]):8.4f}  "
            f"{float(r[0]):8.4f} {float(r[1]):8.4f} {float(r[2]):8.4f}"
        )


if __name__ == "__main__":
    main()
