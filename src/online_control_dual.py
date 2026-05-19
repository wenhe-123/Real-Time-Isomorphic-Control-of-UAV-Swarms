"""Online Crazyflow + Orbbec depth (main) + optional USB webcam rotation when occluded.

Run from ``iso_swarm/src`` (pixi project root)::

    cd iso_swarm/src && pixi install && pixi run online-dual

This entry auto-installs pynput/keyboard if missing. Do not use a separate conda env
without ``pip install pynput keyboard``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shared.online_input_keys import try_install_hotkey_dependencies
from online_control import main as _online_main


def main() -> None:
    """Orbbec + dual rotation with performance-friendly defaults."""
    try_install_hotkey_dependencies()
    if len(sys.argv) <= 1:
        sys.argv = [
            sys.argv[0],
            "--input-backend",
            "orbbec",
            "--plot-every",
            "4",
            "--install-hotkey-deps",
        ]
    elif "--install-hotkey-deps" not in sys.argv:
        sys.argv = [sys.argv[0], "--install-hotkey-deps", *sys.argv[1:]]
    if "--input-backend" not in sys.argv:
        idx = 1
        sys.argv = [sys.argv[0], "--input-backend", "orbbec", *sys.argv[idx:]]
    _online_main()


if __name__ == "__main__":
    main()
