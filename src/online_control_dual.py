"""Primary entry: Orbbec + optional USB webcam rotation + Crazyflow swarm control.

Implementation lives in ``online_control.py`` (Orbbec helpers in ``functions/display_sim/orbbec_hand.py``).

Run from ``iso_swarm`` (pixi)::

    cd /path/to/iso_swarm
    pixi shell
    python src/online_control_dual.py --mp-detect-every 1 --profile-frame
    python src/online_control_dual.py --debug-drone-pos
    python src/online_control_dual.py --debug-drone-pos-every 15

With pixi tasks, keep the task separator: ``pixi run online-dual -- --mp-detect-every 1``.

Defaults: Orbbec input, axswarm planner, ``--point-count 24``. Gesture targets
flow through axswarm to ``cmd_target``; Crazyflow tracks them via
``state_control`` + ``sim.step`` (Mellinger), then ``sim.render``.

``--point-count`` / interactive ``n`` must be >= 8 (default 24).

Spacing and motion limits come from ``config/axswarm_settings.yaml`` (authoritative).
Override the yaml path with ``--axswarm-settings`` if needed.
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


def _strip_argv_separator(argv: list[str]) -> list[str]:
    """Drop standalone ``--`` (pixi/npm style); not valid for argparse."""
    if len(argv) <= 1:
        return list(argv)
    return [argv[0], *[a for a in argv[1:] if a != "--"]]


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv[1:])


def inject_dual_default_argv(argv: list[str]) -> list[str]:
    """Dual entry defaults: 24 drones unless overridden."""
    out = list(argv)
    if not _argv_has_flag(out, "--point-count"):
        out.extend(["--point-count", "24"])
    return out


def inject_dual_gesture_argv(argv: list[str]) -> list[str]:
    """Left-hand gesture tuning defaults for dual entry (not axswarm yaml)."""
    out = list(argv)
    if not _argv_has_flag(out, "--left-trans-scale"):
        out.extend(["--left-trans-scale", "0.0075"])
    if not _argv_has_flag(out, "--left-trans-ema"):
        out.extend(["--left-trans-ema", "0.65"])
    if not _argv_has_flag(out, "--left-rot-ema"):
        out.extend(["--left-rot-ema", "0.55"])
    if not _argv_has_flag(out, "--left-axis-trans-deadzone-m"):
        out.extend(["--left-axis-trans-deadzone-m", "0.0015"])
    if not _argv_has_flag(out, "--left-axis-rot-deadzone-deg"):
        out.extend(["--left-axis-rot-deadzone-deg", "0.25"])
    if not _argv_has_flag(out, "--left-axis-trans-on-m"):
        out.extend(["--left-axis-trans-on-m", "0.004"])
    if not _argv_has_flag(out, "--left-axis-rot-on-deg"):
        out.extend(["--left-axis-rot-on-deg", "1.15"])
    if not _argv_has_flag(out, "--left-trans-rot-coupling"):
        out.extend(["--left-trans-rot-coupling", "0.50"])
    if not _argv_has_flag(out, "--left-rot-direct-follow"):
        out.append("--left-rot-direct-follow")
    return out


def main() -> None:
    """Orbbec + dual rotation with performance-friendly defaults."""
    sys.argv = _strip_argv_separator(sys.argv)
    try_install_hotkey_dependencies()
    if len(sys.argv) <= 1:
        sys.argv = [sys.argv[0], "--install-hotkey-deps"]
    elif "--install-hotkey-deps" not in sys.argv:
        sys.argv = [sys.argv[0], "--install-hotkey-deps", *sys.argv[1:]]
    sys.argv = inject_dual_default_argv(sys.argv)
    sys.argv = inject_dual_gesture_argv(sys.argv)
    _online_main()


if __name__ == "__main__":
    main()
