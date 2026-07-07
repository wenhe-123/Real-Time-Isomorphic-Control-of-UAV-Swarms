"""Emergency halt for physical Crazyflies on exit."""

from __future__ import annotations

from functions.runtime.online_boot import OnlineBoot


def halt_real_swarm_control(boot: OnlineBoot) -> None:
    """Immediately stop setpoints and send Crazyflie emergency stop.

    Args:
        boot: Boot bundle; no-op when real executor is missing or already halted.
    """
    ex = boot.real_executor
    if ex is None or ex.control_halted:
        return
    print("Real swarm: halting control (emergency stop)...", flush=True)
    try:
        ex.halt_control()
    except Exception as exc:
        print(f"[WARN] Real swarm halt failed: {exc}", flush=True)
