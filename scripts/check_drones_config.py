#!/usr/bin/env python3
"""Validate config/drones.toml and config/settings.yaml without hardware."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from functions.real_swarm.swarm_config import load_drones_config  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO / "config" / "drones.toml"
    drones, mapping, opts = load_drones_config(path)
    print(f"OK: {len(drones)} drone(s) from {path}")
    print(
        f"  frame origin={mapping.origin.tolist()} scale={mapping.scale} "
        f"yaw={mapping.yaw_rad:.3f} rad"
    )
    print(f"  ctrl_freq={opts.ctrl_freq} Hz land_on_exit={opts.land_on_exit}")
    for key, entry in drones.items():
        print(f"  [{key}] {entry['uri']} home={entry['pos'].tolist()}")


if __name__ == "__main__":
    main()
