#!/usr/bin/env python3
"""Check morph→world spacing **before** axswarm / enforce_min_separation (esp. open=0).

Run from ``iso_swarm``::

    pixi shell
    PYTHONPATH=src python src/check_formation_spacing.py --n 24 --open 0
    PYTHONPATH=src python src/check_formation_spacing.py --n 8 --open 0 --morph-world-scale 1.2

Compares:
  - topo/mm closest pair (graphics)
  - world meters **before** ``enforce_min_separation`` (pre_filter)
  - world meters **after** enforce (post_enforce)
  - ``--min-separation-m`` and axswarm-style collision envelope (~0.30 m)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from functions.open_close.morph_world import ScaleConfig, fixed_morph_points, normalize_morph_points  # noqa: E402
from functions.swarm_motion.formation_spacing import audit_formation_spacing, closed_shell_scale_boost  # noqa: E402
from functions.swarm_motion.spacing_guard import closest_pair  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--mode", type=int, default=1, choices=range(1, 6))
    p.add_argument("--open", type=float, default=0.0, dest="open_alpha")
    p.add_argument("--radius-mm", type=float, default=50.0)
    p.add_argument("--xy-radius", type=float, default=3.0)
    p.add_argument("--z-center", type=float, default=1.4)
    p.add_argument("--z-min", type=float, default=1.05)
    p.add_argument("--z-max", type=float, default=2.25)
    p.add_argument("--reference-xy-extent-mm", type=float, default=100.0)
    p.add_argument("--min-separation-m", type=float, default=0.32)
    p.add_argument("--morph-world-scale", type=float, default=1.2)
    p.add_argument("--no-auto-closed-boost", action="store_true")
    args = p.parse_args()
    n = max(8, int(args.n))
    scale = ScaleConfig(
        xy_radius=float(args.xy_radius),
        z_center=float(args.z_center),
        z_amplitude=0.35,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        reference_xy_extent_mm=float(args.reference_xy_extent_mm),
        reference_z_extent_mm=float(args.reference_xy_extent_mm),
        z_mm_scale=1.0,
        morph_world_scale=float(args.morph_world_scale),
    )
    oa = float(args.open_alpha)
    mm = fixed_morph_points(n, float(args.radius_mm), int(args.mode), oa, None)
    boost = 1.0 if args.no_auto_closed_boost else closed_shell_scale_boost(
        n_drones=n,
        open_alpha=oa,
        min_separation_m=float(args.min_separation_m),
        user_scale=float(scale.morph_world_scale),
    )
    print(
        f"Config: n={n} mode={args.mode} open={oa:.2f} radius_mm={args.radius_mm} "
        f"morph_world_scale={scale.morph_world_scale} effective_boost={boost:.3f}"
    )
    raw = normalize_morph_points(
        mm,
        scale,
        n_drones=n,
        open_alpha=oa,
        min_separation_m=float(args.min_separation_m),
    )
    mm_ext = np.max(mm, 0) - np.min(mm, 0)
    w_ext = np.max(raw, 0) - np.min(raw, 0)
    mm_zy = float(mm_ext[2] / max(mm_ext[0], mm_ext[1], 1e-9))
    w_zy = float(w_ext[2] / max(w_ext[0], w_ext[1], 1e-9))
    print(
        f"[morph shape] topo mm z/xy={mm_zy:.3f} (1.0=sphere) → world z/xy={w_zy:.3f}"
    )
    if w_zy < 0.92:
        print(
            "  WARN: world formation is flattened — usually z clamp after scale; "
            "re-run after normalize fix or lower --morph-world-scale."
        )
    audit_formation_spacing(
        mm,
        raw,
        label="check_script",
        n_drones=n,
        open_alpha=oa,
        min_separation_m=float(args.min_separation_m),
        collision_envelope_m=float(args.min_separation_m),
    )
    if boost > 1.01:
        print(f"Tip: auto closed-shell boost applied inside normalize (~{boost:.2f}x).")
    elif closest_pair(raw)[0] < float(args.min_separation_m) - 0.03:
        need = float(args.min_separation_m) / max(closest_pair(raw)[0], 1e-6)
        print(f"Tip: try --morph-world-scale {min(1.85, need * 1.05):.2f}")


if __name__ == "__main__":
    main()
