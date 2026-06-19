"""Crazyflie swarm driver: swarmGPT ``core/drone_swarm.py`` + iso_swarm online extras."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm_gpt.core.drone_swarm import DroneSwarm as _SwarmGPTDroneSwarm


def _load_swarmgpt_drone_swarm_module():
    """Load ``swarm_gpt/core/drone_swarm.py`` without importing ``swarm_gpt.core`` (LLM deps)."""
    name = "swarm_gpt.core.drone_swarm"
    if name in sys.modules:
        return sys.modules[name]
    import swarm_gpt

    path = Path(swarm_gpt.__file__).resolve().parent / "core" / "drone_swarm.py"
    if not path.is_file():
        raise ImportError(
            f"swarmGPT drone_swarm not found at {path}. "
            "Install deploy extras: pixi install -e deploy"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load swarmGPT module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_SwarmGPTDroneSwarm = _load_swarmgpt_drone_swarm_module().DroneSwarm


class DroneSwarm(_SwarmGPTDroneSwarm):
    """swarmGPT ``DroneSwarm`` with ROS 2 mocap and per-frame setpoint streaming."""

    def __init__(self, *args, **kwargs) -> None:
        self._rclpy_init_by_us = False
        import rclpy

        if not rclpy.ok():
            rclpy.init()
            self._rclpy_init_by_us = True
        try:
            # swarmGPT: lighthouse=False → mocap via ROS TF (motion_capture_tracking).
            super().__init__(*args, lighthouse=False, **kwargs)
        except BaseException:
            self._shutdown_rclpy_if_owned()
            raise

    def send_setpoint_tick(self, target: dict[str, list[float]]) -> None:
        """Send one low-level position setpoint per drone (non-blocking batch)."""
        self._validate_required_uris("pos", target)
        for uri, setpoint in target.items():
            if len(setpoint) != 4:
                raise ValueError(f"pos[{uri!r}] must contain exactly four elements.")

        async def _tick(uri: str) -> None:
            await self._change_commander_level(uri, "low")
            x, y, z, yaw = (float(v) for v in target[uri])
            await self._cf(uri).commander().send_setpoint_position(x, y, z, yaw)

        self._run(self._parallel_by_uri("Setpoint tick", self.uris, _tick, timeout=0.5))

    def close(self) -> None:
        super().close()
        self._shutdown_rclpy_if_owned()

    def _shutdown_rclpy_if_owned(self) -> None:
        if not getattr(self, "_rclpy_init_by_us", False):
            return
        import rclpy

        if rclpy.ok():
            rclpy.shutdown()
        self._rclpy_init_by_us = False
