"""Load Crazyflie swarm layout and sim→room frame mapping from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RealFrameMapping:
    """Map gesture/sim targets (m) into the mocap room frame."""

    origin: np.ndarray
    scale: float
    yaw_rad: float

    def sim_to_real(self, sim_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(sim_xyz, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        out = self.origin + (pts @ rot.T) * float(self.scale)
        return out[0] if single else out

    def real_to_sim(self, real_xyz: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`sim_to_real` (room / mocap → gesture sim frame)."""
        pts = np.asarray(real_xyz, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        scale = float(self.scale)
        if abs(scale) < 1e-9:
            raise ValueError("frame.scale must be non-zero for real_to_sim")
        local = (pts - self.origin) / scale
        out = local @ rot
        return out[0] if single else out


@dataclass(frozen=True)
class RealSwarmOptions:
    ctrl_freq: float
    update_freq: float
    col_freq: float
    arm_goto_s: float
    land_on_exit: bool
    max_pos_error_m: float


def load_drones_config(path: Path) -> tuple[dict[str, dict], RealFrameMapping, RealSwarmOptions]:
    """Parse ``config/drones.toml`` (see ``drones.example.toml``)."""
    path = Path(path).expanduser().resolve()
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    swarm = raw.get("swarm", {})
    frame = raw.get("frame", {})
    entries = raw.get("drone", raw.get("drones", []))
    if not entries:
        raise ValueError(f"No [[drone]] entries in {path}")

    drones: dict[str, dict] = {}
    homes: list[np.ndarray] = []
    for entry in entries:
        drone_id = str(entry["id"])
        uri = str(entry["uri"])
        home = np.asarray(entry.get("home", entry.get("pos", [0.0, 0.0, 0.0])), dtype=np.float64)
        if home.shape != (3,):
            raise ValueError(f"drone {drone_id}: home must be [x,y,z]")
        drones[drone_id] = {"id": drone_id, "uri": uri, "pos": home}
        homes.append(home)

    # Stable index order: sort by numeric id when possible.
    def _sort_key(item: tuple[str, dict]) -> tuple:
        key, _ = item
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    drones = dict(sorted(drones.items(), key=_sort_key))
    home_mean = np.mean(np.stack(homes, axis=0), axis=0)

    if "origin" in frame:
        origin = np.asarray(frame["origin"], dtype=np.float64)
    else:
        origin = home_mean.copy()
    scale = float(frame.get("scale", 1.0))
    yaw_deg = float(frame.get("yaw_deg", 0.0))

    opts = RealSwarmOptions(
        ctrl_freq=float(swarm.get("ctrl_freq", 100.0)),
        update_freq=float(swarm.get("update_freq", 50.0)),
        col_freq=float(swarm.get("col_freq", 10.0)),
        arm_goto_s=float(swarm.get("arm_goto_s", 5.0)),
        land_on_exit=bool(swarm.get("land_on_exit", True)),
        max_pos_error_m=float(swarm.get("max_pos_error_m", 0.35)),
    )
    mapping = RealFrameMapping(
        origin=origin,
        scale=scale,
        yaw_rad=np.deg2rad(yaw_deg),
    )
    return drones, mapping, opts
