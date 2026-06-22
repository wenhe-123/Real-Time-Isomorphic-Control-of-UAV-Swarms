"""Primary entry: Orbbec hand control → real Crazyflie swarm (same pipeline as online_control_dual).

Run from ``iso_swarm`` (pixi)::

    cd /path/to/iso_swarm
    pixi install -e deploy
    pixi run -e deploy setup
    # edit config/drones.toml (active drones) and config/settings.yaml (radio.uri_base)
    pixi run -e deploy real-dual -- --drones-config config/drones.toml

Uses the same gesture / axswarm / left-hand pose pipeline as ``online_control_dual.py``,
but streams ``cmd_target`` to physical drones via cflib2 instead of Crazyflow MuJoCo.

Requires: ``cflib2``, ``drone-estimators``, and ROS 2 mocap (``pixi install -e deploy``).
Start mocap in a separate terminal: ``pixi run -e deploy mocap``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parent
for _path in (_REPO, _SRC):
    _ps = str(_path)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from functions.dual_cam.online_input_keys import try_install_hotkey_dependencies
from online_control import main as _online_main
from online_control_dual import (
    _argv_has_flag,
    _strip_argv_separator,
    inject_dual_gesture_argv,
)


def inject_real_dual_default_argv(argv: list[str]) -> list[str]:
    """Real entry: morph point-count 8 (not 24); physical count from drones.toml."""
    out = list(argv)
    if not _argv_has_flag(out, "--point-count"):
        out.extend(["--point-count", "8"])
    return out


def _require_drones_config(argv: list[str]) -> list[str]:
    if any(a in ("-h", "--help") for a in argv):
        return list(argv)
    out = list(argv)
    if not _argv_has_flag(out, "--drones-config"):
        default = Path("config/drones.toml")
        if default.is_file():
            out.extend(["--drones-config", str(default)])
        else:
            print(
                "Real-swarm mode requires --drones-config PATH.\n"
                "  edit config/drones.toml and config/settings.yaml (radio.uri_base)\n"
                f"  python {argv[0]} --drones-config config/drones.toml",
                file=sys.stderr,
            )
            raise SystemExit(2)
    return out


def main() -> None:
    sys.argv = _strip_argv_separator(sys.argv)
    try_install_hotkey_dependencies()
    try:
        import cflib2  # noqa: F401
    except ImportError as exc:
        print(
            "Real-swarm mode needs cflib2 (Linux only).\n"
            "  pixi install -e deploy\n"
            "  pixi run setup          # once: Orbbec SDK + pyk4a\n"
            "  pixi run -e deploy real-dual -- --drones-config config/drones.toml\n"
            f"({exc})",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    if len(sys.argv) <= 1:
        sys.argv = [sys.argv[0], "--install-hotkey-deps"]
    elif "--install-hotkey-deps" not in sys.argv:
        sys.argv = [sys.argv[0], "--install-hotkey-deps", *sys.argv[1:]]
    sys.argv = inject_real_dual_default_argv(sys.argv)
    sys.argv = inject_dual_gesture_argv(sys.argv)
    sys.argv = _require_drones_config(sys.argv)
    _online_main()


if __name__ == "__main__":
    main()
