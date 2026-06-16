"""Real Crazyflie swarm control (replaces Crazyflow sim in the online loop)."""

from functions.real_swarm.swarm_config import RealFrameMapping, load_drones_config

__all__ = ["RealFrameMapping", "RealSwarmExecutor", "load_drones_config"]


def __getattr__(name: str):
    if name == "RealSwarmExecutor":
        from functions.real_swarm.executor import RealSwarmExecutor

        return RealSwarmExecutor
    raise AttributeError(name)
