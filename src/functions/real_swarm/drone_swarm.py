"""Crazyflie swarm driver: swarmGPT ``core/drone_swarm.py`` + iso_swarm online extras."""

from __future__ import annotations

from swarm_gpt.core.drone_swarm import DroneSwarm as _SwarmGPTDroneSwarm


class DroneSwarm(_SwarmGPTDroneSwarm):
    """swarmGPT ``DroneSwarm`` with ROS 2 mocap and per-frame setpoint streaming."""

    def __init__(self, *args, **kwargs) -> None:
        import rclpy

        # ROSConnector (created in parent __init__ when lighthouse=False) requires rclpy.ok()
        # in this process. Subprocess TF spinners call rclpy.init() on their own.
        if not rclpy.ok():
            rclpy.init()
        super().__init__(*args, lighthouse=False, **kwargs)

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
