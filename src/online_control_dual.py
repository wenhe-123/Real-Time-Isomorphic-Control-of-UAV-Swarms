"""Primary entry: Orbbec + optional USB webcam rotation + Crazyflow swarm control.

Implementation lives in ``online_control.py`` (Orbbec helpers in ``functions/display_sim/orbbec_hand.py``).

Run from ``iso_swarm`` (pixi)::

    cd /path/to/iso_swarm
    pixi shell
    python src/online_control_dual.py --mp-detect-every 1 --profile-frame
    python src/online_control_dual.py --debug-drone-targets-every 15

With pixi tasks, keep the task separator: ``pixi run online-dual -- --mp-detect-every 1``.

Defaults: Orbbec input, ``--planner axswarm``, ``--point-count 24``. Gesture targets
flow through axswarm (if enabled) to ``cmd_target``; MuJoCo uses ``render_targets`` only
(no ``sim.step``). Use ``--planner direct`` for raw gesture chasing without axswarm MPC.

``--point-count`` / interactive ``n`` must be >= 8 (default 24).

With ``--planner axswarm`` (default here), spacing and motion limits are taken from
``axswarm-amswarm/params/settings.yaml`` (``vel_max``, collision envelope) unless you
override ``--min-separation-m`` or ``--raw-target-ema``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from functions.swarm_motion.axswarm_runtime import load_axswarm_motion_limits
from online_control import main as _online_main
from functions.dual_cam.online_input_keys import try_install_hotkey_dependencies

_ONLINE_MAX_SIM_SUBSTEPS = 160


def _strip_argv_separator(argv: list[str]) -> list[str]:
    """Drop standalone ``--`` (pixi/npm style); not valid for argparse."""
    if len(argv) <= 1:
        return list(argv)
    return [argv[0], *[a for a in argv[1:] if a != "--"]]


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv[1:])


def _argv_flag_value(argv: list[str], flag: str, default: int) -> int:
    if flag not in argv:
        return default
    i = argv.index(flag)
    if i + 1 >= len(argv):
        return default
    try:
        return int(float(argv[i + 1]))
    except ValueError:
        return default


def _planner_is_axswarm(argv: list[str]) -> bool:
    if "--planner" not in argv:
        return False
    i = argv.index("--planner")
    return i + 1 < len(argv) and str(argv[i + 1]).strip().lower() == "axswarm"


def inject_dual_default_argv(argv: list[str]) -> list[str]:
    """Dual entry defaults: axswarm safety filter + 24 drones unless overridden."""
    out = list(argv)
    if not _argv_has_flag(out, "--planner"):
        out.extend(["--planner", "axswarm"])
    if not _argv_has_flag(out, "--point-count"):
        out.extend(["--point-count", "24"])
    return out


def inject_axswarm_motion_argv(argv: list[str]) -> list[str]:
    """Apply yaml-aligned caps when dual runs with axswarm and flags are omitted."""
    if not _planner_is_axswarm(argv):
        return argv
    fps = _argv_flag_value(argv, "--fps", 30)
    limits = load_axswarm_motion_limits(
        outer_fps=fps,
        max_substeps=_ONLINE_MAX_SIM_SUBSTEPS,
    )
    out = list(argv)
    if not _argv_has_flag(out, "--raw-target-ema"):
        out.extend(["--raw-target-ema", "0"])
    if not _argv_has_flag(out, "--min-separation-m"):
        out.extend(["--min-separation-m", f"{limits.min_separation_m:.2f}"])
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
    sys.argv = inject_axswarm_motion_argv(sys.argv)
    _online_main()


if __name__ == "__main__":
    main()
