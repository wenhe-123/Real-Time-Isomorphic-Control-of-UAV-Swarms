"""Record online_control trajectories to NPZ for offline axswarm replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TrajectoryRecorder:
    """Append per-camera-frame samples; call :meth:`save` once at shutdown."""

    def __init__(self, path: str | Path, *, meta: dict | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._meta = dict(meta or {})
        self._t: list[float] = []
        self._setpoint: list[np.ndarray] = []
        self._raw_target: list[np.ndarray] = []
        self._cmd_target: list[np.ndarray] = []
        self._sim_pos: list[np.ndarray] = []
        self._sim_vel: list[np.ndarray] = []
        self._gesture_armed: list[bool] = []

    def append(
        self,
        t_sec: float,
        *,
        setpoint: np.ndarray,
        raw_target: np.ndarray,
        cmd_target: np.ndarray,
        sim_pos: np.ndarray,
        sim_vel: np.ndarray,
        gesture_armed: bool,
    ) -> None:
        self._t.append(float(t_sec))
        self._setpoint.append(np.asarray(setpoint, dtype=np.float32))
        self._raw_target.append(np.asarray(raw_target, dtype=np.float32))
        self._cmd_target.append(np.asarray(cmd_target, dtype=np.float32))
        self._sim_pos.append(np.asarray(sim_pos, dtype=np.float32))
        self._sim_vel.append(np.asarray(sim_vel, dtype=np.float32))
        self._gesture_armed.append(bool(gesture_armed))

    @property
    def n_frames(self) -> int:
        return len(self._t)

    def save(self) -> Path:
        if not self._t:
            raise RuntimeError(f"No trajectory frames recorded; not writing {self.path}")

        self._meta["n_frames"] = len(self._t)
        if self._setpoint:
            self._meta["n_drones"] = int(self._setpoint[0].shape[0])

        np.savez_compressed(
            self.path,
            t=np.asarray(self._t, dtype=np.float64),
            setpoint=np.stack(self._setpoint, axis=0),
            raw_target=np.stack(self._raw_target, axis=0),
            cmd_target=np.stack(self._cmd_target, axis=0),
            sim_pos=np.stack(self._sim_pos, axis=0),
            sim_vel=np.stack(self._sim_vel, axis=0),
            gesture_armed=np.asarray(self._gesture_armed, dtype=np.bool_),
            meta_json=np.asarray(json.dumps(self._meta)),
        )
        return self.path


def load_trajectory_npz(path: str | Path) -> dict:
    """Load NPZ written by :class:`TrajectoryRecorder`."""
    with np.load(Path(path), allow_pickle=False) as data:
        out = {k: data[k] for k in data.files if k != "meta_json"}
        out["meta"] = json.loads(str(data["meta_json"]))
    return out
