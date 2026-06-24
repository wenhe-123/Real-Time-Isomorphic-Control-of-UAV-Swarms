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
    cmd_source: str = "unknown"
    plan_drift_m: float = 0.0
    mpc_due: bool = False


@dataclass
class AxswarmPlanner:
    """Gesture setpoints -> axswarm MPC -> Crazyflow commands."""

    settings: Any
    dynamics: dict[str, np.ndarray]
    solver_data: Any
    n_drones: int
    mpc_hz: float
    min_separation_m: float
    horizon_steps: int
    _control_updated: bool
    _last_mpc_time: float
    _horizon_pos: np.ndarray | None
    _horizon_vel: np.ndarray | None
    _horizon_anchor_s: float
    _solve_count: int
    _fail_count: int
    _last_solve_ms: float
    _last_ok: bool
    _last_solve_n_ok: int

    @classmethod
    def create(
        cls,
        n_drones: int,
        *,
        settings_path: Path | None = None,
        horizon_steps: int = 1,
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
        k = int(settings.K)
        m = max(1, min(int(horizon_steps), k))
        if int(horizon_steps) > k:
            raise ValueError(
                f"mpc_horizon_steps ({horizon_steps}) must be <= axswarm K ({k})"
            )
        filt = cls(
            settings=settings,
            dynamics=dynamics,
            solver_data=None,
            n_drones=n_drones,
            mpc_hz=mpc_hz,
            min_separation_m=min_sep,
            horizon_steps=m,
            _control_updated=False,
            _last_mpc_time=-1e9,
            _horizon_pos=None,
            _horizon_vel=None,
            _horizon_anchor_s=-1e9,
            _solve_count=0,
            _fail_count=0,
            _last_solve_ms=0.0,
            _last_ok=False,
            _last_solve_n_ok=0,
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
            horizon_steps=int(cfg.mpc_horizon_steps),
        )

    @property
    def mpc_period_s(self) -> float:
        return 1.0 / max(float(self.mpc_hz), 1e-6)

    @property
    def replan_period_s(self) -> float:
        """Wall time between MPC solves when streaming ``u_pos[:, 0:M]``."""
        return float(self.horizon_steps) * self.mpc_period_s

    def _clear_horizon_cache(self) -> None:
        self._horizon_pos = None
        self._horizon_vel = None
        self._horizon_anchor_s = -1e9

    def _cache_horizon(self, solved_data: Any) -> None:
        m = int(self.horizon_steps)
        try:
            pos = np.asarray(solved_data.u_pos[:, :m], dtype=np.float32)
            vel = np.asarray(solved_data.u_vel[:, :m], dtype=np.float32)
        except Exception:
            self._clear_horizon_cache()
            return
        if pos.shape != (self.n_drones, m, 3) or not np.all(np.isfinite(pos)):
            self._clear_horizon_cache()
            return
        if vel.shape != (self.n_drones, m, 3) or not np.all(np.isfinite(vel)):
            vel = np.zeros_like(pos, dtype=np.float32)
        self._horizon_pos = pos
        self._horizon_vel = vel

    def _horizon_step_index(self, elapsed_s: float) -> int:
        if self._horizon_pos is None or self.horizon_steps <= 1:
            return 0
        steps_elapsed = int((float(elapsed_s) - self._horizon_anchor_s) / self.mpc_period_s)
        return int(min(max(0, steps_elapsed), self.horizon_steps - 1))

    def _needs_replan(self, elapsed_s: float) -> bool:
        if self.solver_data is None:
            return True
        if self.horizon_steps <= 1:
            return (float(elapsed_s) - self._last_mpc_time) >= self.mpc_period_s - 1e-9
        if self._horizon_pos is None:
            return True
        steps_elapsed = int((float(elapsed_s) - self._horizon_anchor_s) / self.mpc_period_s)
        return steps_elapsed >= self.horizon_steps

    def _cmd_from_horizon(
        self, elapsed_s: float
    ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        if self.horizon_steps > 1 and self._horizon_pos is not None:
            idx = self._horizon_step_index(elapsed_s)
            return self._horizon_pos[:, idx], self._horizon_vel[:, idx], idx
        pos, vel = self._planned_cmd()
        return pos, vel, 0

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
        self._clear_horizon_cache()

    def sync_gesture(self, gesture: np.ndarray, sim_vel: np.ndarray | None = None) -> None:
        """Re-init MPC state from current gesture (e.g. SPACE armed)."""
        self.reset(gesture, sim_vel)

    def _planned_cmd(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """First MPC input from solver state (``u_pos[:, 0]`` / ``u_vel[:, 0]``)."""
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
        success, _, solved_data = self._solve_fn(
            states, self.solver_data, self.settings
        )
        jax.block_until_ready(solved_data)
        self._last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_count += 1
        success_arr = np.asarray(success, dtype=bool)
        n_ok = int(np.sum(success_arr))
        ok = n_ok == self.n_drones
        self._last_ok = ok
        self._last_solve_n_ok = n_ok
        self.solver_data = solved_data
        self._cache_horizon(solved_data)
        if not ok:
            self._fail_count += 1
            planned, _, _ = self._cmd_from_horizon(self._horizon_anchor_s)
            self._control_updated = planned is not None
            return ok
        if self.horizon_steps <= 1:
            # Match amswarm: solve → step → consume u_pos[:, 0] (only on full success).
            self.solver_data = self.solver_data.step(self.solver_data)
        planned, _, _ = self._cmd_from_horizon(self._horizon_anchor_s)
        self._control_updated = planned is not None
        return ok

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
        """Run MPC (when due) and return the active horizon command for Crazyflow."""
        el = float(elapsed_s)
        sp = self._lock_vertical_z(np.asarray(gesture_setpoint, dtype=np.float32), hold_z)
        pos, vel = self._track_state(sim=sim, track_pos=track_pos, track_vel=track_vel)
        states = np.concatenate([pos, vel], axis=-1)
        self._control_updated = False

        due = self._needs_replan(el)
        if due:
            self._last_mpc_time = el
            self._horizon_anchor_s = el
            self._run_mpc(states, sp)
        planned_pos, planned_vel, horizon_idx = self._cmd_from_horizon(el)
        plan_drift_m = 0.0
        if planned_pos is not None:
            plan_drift_m = float(np.max(np.linalg.norm(planned_pos - sp, axis=-1)))
            out_pos = planned_pos
            out_vel = (
                planned_vel
                if planned_vel is not None
                else np.zeros((self.n_drones, 3), dtype=np.float32)
            )
            if self.horizon_steps > 1:
                cmd_source = (
                    f"mpc:horizon[{horizon_idx}/{self.horizon_steps - 1}]"
                    if self._last_ok
                    else f"mpc:horizon[{horizon_idx}/{self.horizon_steps - 1}]:best_effort"
                )
            else:
                cmd_source = "mpc" if self._last_ok else "mpc:best_effort"
        else:
            out_pos = np.zeros((self.n_drones, 3), dtype=np.float32)
            out_vel = np.zeros((self.n_drones, 3), dtype=np.float32)
            cmd_source = "mpc:none"
        return AxswarmControl(
            pos=self._lock_vertical_z(out_pos, hold_z),
            vel=out_vel,
            updated=bool(self._control_updated),
            cmd_source=cmd_source,
            plan_drift_m=plan_drift_m,
            mpc_due=due,
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
        pos, vel, _ = self._cmd_from_horizon(self._horizon_anchor_s)
        if pos is None:
            pos = np.zeros((self.n_drones, 3), dtype=np.float32)
        if vel is None:
            vel = np.zeros((self.n_drones, 3), dtype=np.float32)
        return np.concatenate([pos, vel], axis=-1)

    def current_control_velocity(self) -> np.ndarray:
        _, vel, _ = self._cmd_from_horizon(self._horizon_anchor_s)
        if vel is None:
            return np.zeros((self.n_drones, 3), dtype=np.float32)
        return vel

    def control_updated(self) -> bool:
        return bool(self._control_updated)

    @property
    def last_solve_ok(self) -> bool:
        return bool(self._last_ok)

    @property
    def last_solve_n_ok(self) -> int:
        return int(self._last_solve_n_ok)

    def status_line(self) -> str:
        if self._solve_count == 0:
            return "axswarm: idle"
        ok_frac = 1.0 - self._fail_count / max(1, self._solve_count)
        state = "ok" if self._last_ok else "best_effort"
        horizon = (
            f" M={self.horizon_steps}"
            if self.horizon_steps > 1
            else ""
        )
        return (
            f"axswarm:{self.mpc_hz:.0f}Hz{horizon} "
            f"ok={ok_frac * 100:.0f}% "
            f"last={self._last_solve_ms:.0f}ms {state}"
        )

