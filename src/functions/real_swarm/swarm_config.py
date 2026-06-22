"""Load Crazyflie swarm layout and sim→room frame mapping from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_TOP_LEVEL_TABLES = frozenset(
    {"active", "swarm", "frame", "radio", "drone", "drones", "settings_file"}
)


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
    land_on_exit: bool
    max_pos_error_m: float


def default_settings_path() -> Path:
    """Bundled ``config/settings.yaml`` (radio URI template for drones.toml)."""
    return Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


def _resolve_settings_path(drones_path: Path, raw: dict, settings_path: Path | None) -> Path:
    if settings_path is not None:
        return Path(settings_path).expanduser().resolve()
    settings_file = raw.get("settings_file")
    if settings_file:
        candidate = Path(settings_file)
        if not candidate.is_absolute():
            candidate = drones_path.parent / candidate
        return candidate.expanduser().resolve()
    sibling = drones_path.parent / "settings.yaml"
    if sibling.is_file():
        return sibling.resolve()
    return default_settings_path()


def _uri_base_from_settings(raw: dict, settings_path: Path) -> str:
    radio = raw.get("radio")
    if isinstance(radio, dict) and "uri_base" in radio:
        return str(radio["uri_base"])
    if not settings_path.is_file():
        raise ValueError(
            "SwarmGPT-style drones.toml needs radio.uri_base in a [radio] table or "
            f"config/settings.yaml (expected {settings_path})"
        )
    with open(settings_path, encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    try:
        return str(settings["radio"]["uri_base"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Missing radio.uri_base in {settings_path} (or add [radio] to drones.toml)"
        ) from exc


def _parse_home(entry: dict, *, label: str) -> np.ndarray:
    home = np.asarray(entry.get("home", entry.get("pos", [0.0, 0.0, 0.0])), dtype=np.float64)
    if home.shape != (3,):
        raise ValueError(f"{label}: home/pos must be [x,y,z]")
    return home


def _parse_active_drones(
    raw: dict,
    *,
    path: Path,
    uri_base: str,
) -> tuple[dict[str, dict], list[np.ndarray]]:
    active = raw.get("active")
    if not isinstance(active, list) or not active:
        raise ValueError(f"'active' must be a non-empty list in {path}")

    registry = {
        k: v
        for k, v in raw.items()
        if k not in _TOP_LEVEL_TABLES and isinstance(v, dict)
    }
    missing = [name for name in active if name not in registry]
    if missing:
        raise ValueError(f"Drones in 'active' not found in drone table: {missing}")

    addrs = [int(registry[name]["addr"]) for name in active]
    if len(addrs) != len(set(addrs)):
        raise ValueError(f"Duplicate addr values in active drones: {addrs}")

    drones: dict[str, dict] = {}
    homes: list[np.ndarray] = []
    for i, name in enumerate(active):
        entry = registry[name]
        addr = int(entry["addr"])
        channel = int(entry["channel"])
        uri = uri_base.format(channel=channel, addr=addr)
        home = _parse_home(entry, label=name)
        drone_id = str(i)
        drones[drone_id] = {"id": drone_id, "uri": uri, "pos": home}
        homes.append(home)
    return drones, homes


def _parse_explicit_drones(
    entries: list[dict],
    *,
    path: Path,
) -> tuple[dict[str, dict], list[np.ndarray]]:
    drones: dict[str, dict] = {}
    homes: list[np.ndarray] = []
    for entry in entries:
        drone_id = str(entry["id"])
        uri = str(entry["uri"])
        home = _parse_home(entry, label=f"drone {drone_id}")
        drones[drone_id] = {"id": drone_id, "uri": uri, "pos": home}
        homes.append(home)
    return drones, homes


def _sort_drones(drones: dict[str, dict]) -> dict[str, dict]:
    def _sort_key(item: tuple[str, dict]) -> tuple:
        key, _ = item
        try:
            return (0, int(key))
        except ValueError:
            return (1, key)

    return dict(sorted(drones.items(), key=_sort_key))


def load_drones_config(
    path: Path,
    *,
    settings_path: Path | None = None,
) -> tuple[dict[str, dict], RealFrameMapping, RealSwarmOptions]:
    """Parse a drones TOML file.

    Supports two layouts (swarmGPT-compatible):

    * **Active list** — ``active = ["cf11", ...]`` plus ``[cf11]`` tables with
      ``addr``, ``channel``, and ``pos``. URIs come from ``radio.uri_base`` in
      ``config/settings.yaml`` (or ``[radio]`` in the same TOML).
    * **Explicit URIs** — ``[[drone]]`` rows with ``id``, ``uri``, and ``home``.
    """
    path = Path(path).expanduser().resolve()
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    swarm = raw.get("swarm", {})
    frame = raw.get("frame", {})

    if "active" in raw:
        uri_base = _uri_base_from_settings(
            raw, _resolve_settings_path(path, raw, settings_path)
        )
        drones, homes = _parse_active_drones(raw, path=path, uri_base=uri_base)
    else:
        entries = raw.get("drone", raw.get("drones", []))
        if not entries:
            raise ValueError(
                f"No [[drone]] entries and no 'active' list in {path}"
            )
        drones, homes = _parse_explicit_drones(entries, path=path)

    drones = _sort_drones(drones)
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
        land_on_exit=bool(swarm.get("land_on_exit", True)),
        max_pos_error_m=float(swarm.get("max_pos_error_m", 0.35)),
    )
    mapping = RealFrameMapping(
        origin=origin,
        scale=scale,
        yaw_rad=np.deg2rad(yaw_deg),
    )
    return drones, mapping, opts
