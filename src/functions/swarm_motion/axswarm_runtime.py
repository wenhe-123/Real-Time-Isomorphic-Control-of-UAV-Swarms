"""Realtime axswarm planner bridge for iso_swarm.

Gesture supplies position setpoints; axswarm MPC returns ``u_pos``/``u_vel``
per ``config/axswarm_settings.yaml``. Crazyflow tracks those commands via
``state_control`` (pos + vel), matching the online slice of swarmGPT's
``simulate_axswarm`` loop (solve → step → consume ``u_pos[:, 0]`` / ``u_vel[:, 0]``).

Solver API: ``SolverData.init(setpoints, initial_states, ...)``, ``solve(states, data, settings)``.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import yaml

if TYPE_CHECKING:
    from crazyflow.sim import Sim


def default_axswarm_settings_path() -> Path:
    """Bundled ``config/axswarm_settings.yaml`` (pip ``.[sim]``; no submodule required)."""
    return Path(__file__).resolve().parents[3] / "config" / "axswarm_settings.yaml"


def resolve_axswarm_settings_path(settings_path: Path | None) -> Path:
    if settings_path is not None:
        return Path(settings_path)
    bundled = default_axswarm_settings_path()
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        "axswarm settings not found. Pass --axswarm-settings PATH or keep "
        "config/axswarm_settings.yaml in the project."
    )


def ensure_axswarm_import() -> Path:
    try:
        import axswarm
    except ImportError as exc:
        raise ImportError('axswarm package not found. Install sim deps: pip install -e ".[sim]"') from exc
    return Path(axswarm.__file__).resolve().parent.parent


def load_axswarm_yaml(settings_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with open(settings_path) as f:
        config = yaml.safe_load(f)
    settings_dict = dict(config["SolverSettings"])
    for k, v in settings_dict.items():
        if isinstance(v, list):
            settings_dict[k] = np.asarray(v)
    dynamics = {k: np.asarray(v) for k, v in config["Dynamics"].items()}
    return settings_dict, dynamics


def min_separation_from_envelope(collision_envelope: np.ndarray) -> float:
    env = np.asarray(collision_envelope, dtype=np.float64)
    return float(2.0 * np.max(env))


def load_axswarm_min_separation(
    *,
    settings_path: Path | None = None,
) -> float:
    """Read ``collision_envelope`` from yaml and convert it to center distance."""
    path = resolve_axswarm_settings_path(settings_path)
    settings_dict, _ = load_axswarm_yaml(path)
    env = np.asarray(settings_dict.get("collision_envelope", [0.15, 0.15, 0.15]))
    return min_separation_from_envelope(env)


@contextlib.contextmanager
def _axswarm_force_cpu_init():
    """Keep axswarm ``SolverData.init`` on CPU even if the package was patched for GPU."""
    import jax

    real_devices = jax.devices

    def _devices(platform: str | None = None):
        if platform == "gpu":
            return real_devices("cpu")
        return real_devices(platform)

    jax.devices = _devices  # type: ignore[method-assign]
    try:
        yield
    finally:
        jax.devices = real_devices  # type: ignore[method-assign]


@dataclass
class AxswarmControl:
    """MPC output consumed by Crazyflow ``state_control`` (first horizon step)."""

    pos: np.ndarray
    vel: np.ndarray
    updated: bool


@dataclass
class AxswarmPlanner:
    """Gesture setpoints -> axswarm MPC -> Crazyflow commands."""

    settings: Any
    dynamics: dict[str, np.ndarray]
    solver_data: Any
    n_drones: int
    mpc_hz: float
    min_separation_m: float
    _control_updated: bool
    _last_mpc_time: float
    _solve_count: int
    _fail_count: int
    _last_solve_ms: float
    _last_ok: bool

    @classmethod
    def create(
        cls,
        n_drones: int,
        *,
        settings_path: Path | None = None,
    ) -> AxswarmPlanner:
        if n_drones < 8:
            raise ValueError(f"axswarm planner requires n_drones >= 8, got {n_drones}")
        ensure_axswarm_import()
        from axswarm import SolverData, SolverSettings, solve  # noqa: WPS433

        resolved = resolve_axswarm_settings_path(settings_path)
        settings_dict, dynamics = load_axswarm_yaml(resolved)
        settings = SolverSettings(**settings_dict)
        mpc_hz = float(max(0.5, float(settings.freq)))
        min_sep = min_separation_from_envelope(np.asarray(settings.collision_envelope))
        filt = cls(
            settings=settings,
            dynamics=dynamics,
            solver_data=None,
            n_drones=n_drones,
            mpc_hz=mpc_hz,
            min_separation_m=min_sep,
            _control_updated=False,
            _last_mpc_time=-1e9,
            _solve_count=0,
            _fail_count=0,
            _last_solve_ms=0.0,
            _last_ok=False,
        )
        filt._solve_fn = solve
        filt._SolverData = SolverData
        return filt

    @classmethod
    def from_runtime_config(
        cls,
        n_drones: int,
        cfg: Any,
    ) -> AxswarmPlanner:
        return cls.create(
            n_drones,
            settings_path=Path(cfg.axswarm_settings) if cfg.axswarm_settings else None,
        )

    @property
    def mpc_period_s(self) -> float:
        return 1.0 / max(float(self.mpc_hz), 1e-6)

    @staticmethod
    def _lock_vertical_z(pts: np.ndarray, hold_z: float | None) -> np.ndarray:
        if hold_z is None:
            return np.asarray(pts, dtype=np.float32)
        out = np.asarray(pts, dtype=np.float32).copy()
        out[:, 2] = float(hold_z)
        return out

    def reset(self, initial_pos: np.ndarray, initial_vel: np.ndarray | None = None) -> None:
        pos = np.asarray(initial_pos, dtype=np.float32)
        if pos.shape != (self.n_drones, 3):
            raise ValueError(f"initial_pos must be ({self.n_drones}, 3), got {pos.shape}")
        vel = (
            np.zeros((self.n_drones, 3), dtype=np.float32)
            if initial_vel is None
            else np.asarray(initial_vel, dtype=np.float32)
        )
        states = np.concatenate([pos, vel], axis=-1)
        z = np.zeros((self.n_drones, 3), dtype=np.float32)
        with _axswarm_force_cpu_init():
            self.solver_data = self._SolverData.init(
                setpoints={"pos": pos.copy(), "vel": vel.copy(), "acc": z.copy()},
                initial_states=states,
                K=self.settings.K,
                N=self.settings.N,
                A=self.dynamics["A"],
                B=self.dynamics["B"],
                A_prime=self.dynamics["A_prime"],
                B_prime=self.dynamics["B_prime"],
                freq=self.settings.freq,
                smoothness_weight=self.settings.smoothness_weight,
                input_smoothness_weight=self.settings.input_smoothness_weight,
                input_continuity_weight=self.settings.input_continuity_weight,
            )
        self._control_updated = False
        self._last_mpc_time = -1e9
        self._last_ok = True

    def sync_gesture(self, gesture: np.ndarray, sim_vel: np.ndarray | None = None) -> None:
        """Re-init MPC state from current gesture (e.g. SPACE armed)."""
        self.reset(gesture, sim_vel)

    def _planned_cmd(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """First MPC input from solver state (no separate output cache)."""
        if self.solver_data is None:
            return None, None
        try:
            pos = np.asarray(self.solver_data.u_pos[:, 0], dtype=np.float32)
            vel = np.asarray(self.solver_data.u_vel[:, 0], dtype=np.float32)
        except Exception:
            return None, None
        if pos.shape != (self.n_drones, 3) or not np.all(np.isfinite(pos)):
            return None, None
        if vel.shape != (self.n_drones, 3) or not np.all(np.isfinite(vel)):
            vel = np.zeros_like(pos, dtype=np.float32)
        return pos, vel

    def _run_mpc(self, states: np.ndarray, gesture_setpoint: np.ndarray) -> bool:
        sp = np.asarray(gesture_setpoint, dtype=np.float32)
        zero_sp = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.solver_data = self.solver_data.replace(
            setpoints={"pos": sp, "vel": zero_sp, "acc": zero_sp},
        )
        t0 = time.perf_counter()
        success, _, self.solver_data = self._solve_fn(
            states, self.solver_data, self.settings
        )
        jax.block_until_ready(self.solver_data)
        self._last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_count += 1
        ok = bool(np.all(success))
        self._last_ok = ok
        # Match axswarm's amswarm example: solve, advance SolverData, then consume
        # the first input setpoint from the shifted plan.
        self.solver_data = self.solver_data.step(self.solver_data)
        planned, _ = self._planned_cmd()
        if planned is not None:
            self._control_updated = True
            if not ok:
                self._fail_count += 1
            return ok
        if not ok:
            self._fail_count += 1
            return False
        self._fail_count += 1
        self._last_ok = False
        return False

    def _track_state(
        self,
        *,
        sim: Sim | None,
        track_pos: np.ndarray | None,
        track_vel: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if track_pos is not None:
            pos = np.asarray(track_pos, dtype=np.float32)
            if pos.shape != (self.n_drones, 3):
                raise ValueError(f"track_pos must be ({self.n_drones}, 3), got {pos.shape}")
            if track_vel is None:
                vel = np.zeros((self.n_drones, 3), dtype=np.float32)
            else:
                vel = np.asarray(track_vel, dtype=np.float32)
                if vel.shape != (self.n_drones, 3):
                    raise ValueError(f"track_vel must be ({self.n_drones}, 3), got {vel.shape}")
        elif sim is not None:
            pos = np.asarray(sim.data.states.pos[0], dtype=np.float32)
            vel = np.asarray(sim.data.states.vel[0], dtype=np.float32)
        else:
            raise ValueError("plan_control requires sim or track_pos")
        return pos, vel

    def plan_control(
        self,
        elapsed_s: float,
        gesture_setpoint: np.ndarray,
        sim: Sim | None = None,
        *,
        track_pos: np.ndarray | None = None,
        track_vel: np.ndarray | None = None,
        hold_z: float | None = None,
    ) -> AxswarmControl:
        """Run MPC (when due) and return ``u_pos[:, 0]`` / ``u_vel[:, 0]`` for Crazyflow."""
        el = float(elapsed_s)
        sp = self._lock_vertical_z(np.asarray(gesture_setpoint, dtype=np.float32), hold_z)
        pos, vel = self._track_state(sim=sim, track_pos=track_pos, track_vel=track_vel)
        states = np.concatenate([pos, vel], axis=-1)
        self._control_updated = False

        due = (el - self._last_mpc_time) >= self.mpc_period_s - 1e-9
        if due:
            self._last_mpc_time = el
            self._run_mpc(states, sp)
        planned_pos, planned_vel = self._planned_cmd()
        out_pos = planned_pos if planned_pos is not None else np.asarray(pos, dtype=np.float32)
        out_vel = (
            planned_vel
            if planned_vel is not None
            else np.zeros((self.n_drones, 3), dtype=np.float32)
        )
        return AxswarmControl(
            pos=self._lock_vertical_z(out_pos, hold_z),
            vel=out_vel,
            updated=bool(self._control_updated),
        )

    def plan_targets(
        self,
        elapsed_s: float,
        gesture_setpoint: np.ndarray,
        sim: Sim | None = None,
        *,
        track_pos: np.ndarray | None = None,
        track_vel: np.ndarray | None = None,
        hold_z: float | None = None,
    ) -> np.ndarray:
        """Return axswarm's planned position target for the raw setpoint."""
        return self.plan_control(
            elapsed_s,
            gesture_setpoint,
            sim,
            track_pos=track_pos,
            track_vel=track_vel,
            hold_z=hold_z,
        ).pos

    def current_control(self) -> np.ndarray:
        """Crazyflow ``state_control`` slice: ``(n_drones, 6)`` pos + vel."""
        pos, vel = self._planned_cmd()
        if pos is None:
            pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        if vel is None:
            vel = np.zeros((self.n_drones, 3), dtype=np.float32)
        return np.concatenate([pos, vel], axis=-1)

    def current_control_velocity(self) -> np.ndarray:
        _, vel = self._planned_cmd()
        if vel is None:
            return np.zeros((self.n_drones, 3), dtype=np.float32)
        return vel

    def control_updated(self) -> bool:
        return bool(self._control_updated)

    def status_line(self) -> str:
        if self._solve_count == 0:
            return "axswarm: idle"
        ok_frac = 1.0 - self._fail_count / max(1, self._solve_count)
        state = "ok" if self._last_ok else "best_effort"
        return (
            f"axswarm:{self.mpc_hz:.0f}Hz "
            f"ok={ok_frac * 100:.0f}% "
            f"last={self._last_solve_ms:.0f}ms {state}"
        )

