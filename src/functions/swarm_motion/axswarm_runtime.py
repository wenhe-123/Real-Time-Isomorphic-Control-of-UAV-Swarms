"""Realtime axswarm safety filter for iso_swarm (does not modify the axswarm package).

Gesture supplies position/velocity/acceleration setpoints; axswarm MPC returns
collision-feasible plans per ``config/axswarm_settings.yaml``. Crazyflow tracks
the filtered targets.

Same API entry points as ``simulate.py``: ``SolverData``, ``SolverSettings``, ``solve``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from crazyflow.sim import Sim


def default_axswarm_settings_path() -> Path:
    """Bundled ``config/axswarm_settings.yaml`` (pip ``.[sim]``; no submodule required)."""
    return Path(__file__).resolve().parents[3] / "config" / "axswarm_settings.yaml"


def default_axswarm_project_root() -> Path:
    """Legacy submodule checkout path (optional dev override)."""
    here = Path(__file__).resolve()
    iso_swarm = here.parents[3]
    return iso_swarm / "submodules" / "axswarm"


def resolve_axswarm_settings_path(
    settings_path: Path | None,
    project_root: Path | None,
) -> Path:
    if settings_path is not None:
        return Path(settings_path)
    bundled = default_axswarm_settings_path()
    if bundled.is_file():
        return bundled
    root = ensure_axswarm_import(project_root)
    legacy = root / "params" / "settings.yaml"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        "axswarm settings not found. Pass --axswarm-settings PATH or install sim deps: "
        "pip install -e \".[sim]\""
    )


def _axswarm_root_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    iso_swarm = here.parents[3]
    roots = [
        iso_swarm / "submodules" / "axswarm",
        iso_swarm.parent / "axswarm-amswarm",
        iso_swarm / "axswarm-amswarm",
    ]
    try:
        import axswarm

        pkg_root = Path(axswarm.__file__).resolve().parent.parent
        if pkg_root not in roots:
            roots.insert(0, pkg_root)
    except ImportError:
        pass
    return roots


def ensure_axswarm_import(project_root: Path | None = None) -> Path:
    candidates = [Path(project_root)] if project_root is not None else _axswarm_root_candidates()
    tried: list[Path] = []
    for cand in candidates:
        root = cand.resolve()
        tried.append(root)
        if not (root / "axswarm" / "solve.py").is_file():
            continue
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
        return root
    hint = tried[0] if tried else default_axswarm_project_root()
    raise FileNotFoundError(
        f"axswarm package not found (tried: {', '.join(str(p) for p in tried)}). "
        f"Install with: pip install -e {hint}"
    )


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


def load_axswarm_yaml_limits(
    *,
    settings_path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Read ``collision_envelope`` and ``pos_min``/``pos_max`` from yaml (authoritative)."""
    path = resolve_axswarm_settings_path(settings_path, project_root)
    settings_dict, _ = load_axswarm_yaml(path)
    env = np.asarray(settings_dict.get("collision_envelope", [0.15, 0.15, 0.15]))
    min_sep = min_separation_from_envelope(env)
    pos_lo = np.asarray(settings_dict["pos_min"], dtype=np.float64)
    pos_hi = np.asarray(settings_dict["pos_max"], dtype=np.float64)
    return min_sep, pos_lo, pos_hi


@dataclass(frozen=True)
class AxswarmOnlineConfig:
    """Runtime fields for ``AxswarmSafetyFilter`` (yaml is authoritative for MPC limits)."""

    settings_path: Path | None
    project_root: Path | None
    max_solve_ms: float
    outer_fps: int

    @classmethod
    def from_runtime_config(cls, cfg: Any, scale: Any) -> AxswarmOnlineConfig:
        del scale
        return cls(
            settings_path=Path(cfg.axswarm_settings) if cfg.axswarm_settings else None,
            project_root=Path(cfg.axswarm_project_root) if cfg.axswarm_project_root else None,
            max_solve_ms=float(cfg.axswarm_max_solve_ms),
            outer_fps=int(cfg.fps),
        )


@dataclass
class AxswarmSafetyFilter:
    """Gesture setpoints → axswarm MPC → collision-feasible targets for Crazyflow."""

    settings: Any
    dynamics: dict[str, np.ndarray]
    solver_data: Any
    n_drones: int
    mpc_hz: float
    max_solve_ms: float
    min_separation_m: float
    outer_fps: float
    _last_safe: np.ndarray | None
    _last_safe_vel: np.ndarray | None
    _control_updated: bool
    _last_mpc_time: float
    _solve_count: int
    _fail_count: int
    _skip_count: int
    _last_solve_ms: float
    _last_ok: bool
    _armed_at: float
    arm_warmup_s: float
    _prev_gesture_sp: np.ndarray | None
    _prev_gesture_vel: np.ndarray | None
    _prev_gesture_t: float | None
    _pos_lo: np.ndarray
    _pos_hi: np.ndarray

    @classmethod
    def create(
        cls,
        n_drones: int,
        *,
        settings_path: Path | None = None,
        project_root: Path | None = None,
        max_solve_ms: float = 90.0,
        outer_fps: int = 30,
        arm_warmup_s: float = 0.0,
    ) -> AxswarmSafetyFilter:
        if n_drones > 16:
            max_solve_ms = max(float(max_solve_ms), 220.0)
        if n_drones < 8:
            raise ValueError(f"axswarm safety filter requires n_drones >= 8, got {n_drones}")
        ensure_axswarm_import(project_root)
        from axswarm import SolverData, SolverSettings, solve  # noqa: WPS433

        resolved = resolve_axswarm_settings_path(settings_path, project_root)
        settings_dict, dynamics = load_axswarm_yaml(resolved)
        pos_lo = np.asarray(settings_dict["pos_min"], dtype=np.float64)
        pos_hi = np.asarray(settings_dict["pos_max"], dtype=np.float64)
        settings = SolverSettings(**settings_dict)
        mpc_hz = float(max(0.5, float(settings.freq)))
        min_sep = min_separation_from_envelope(np.asarray(settings.collision_envelope))
        filt = cls(
            settings=settings,
            dynamics=dynamics,
            solver_data=None,
            n_drones=n_drones,
            mpc_hz=mpc_hz,
            max_solve_ms=float(max(10.0, max_solve_ms)),
            min_separation_m=min_sep,
            outer_fps=float(max(1, outer_fps)),
            _last_safe=None,
            _last_safe_vel=None,
            _control_updated=False,
            _last_mpc_time=-1e9,
            _solve_count=0,
            _fail_count=0,
            _skip_count=0,
            _last_solve_ms=0.0,
            _last_ok=False,
            _armed_at=-1e9,
            arm_warmup_s=float(max(0.0, arm_warmup_s)),
            _prev_gesture_sp=None,
            _prev_gesture_vel=None,
            _prev_gesture_t=None,
            _pos_lo=np.asarray(pos_lo, dtype=np.float32),
            _pos_hi=np.asarray(pos_hi, dtype=np.float32),
        )
        filt._solve_fn = solve
        filt._SolverData = SolverData
        return filt

    @classmethod
    def from_online_config(
        cls,
        n_drones: int,
        online: AxswarmOnlineConfig,
    ) -> AxswarmSafetyFilter:
        return cls.create(
            n_drones,
            settings_path=online.settings_path,
            project_root=online.project_root,
            max_solve_ms=online.max_solve_ms,
            outer_fps=online.outer_fps,
        )

    def _clamp_pos(self, pts: np.ndarray) -> np.ndarray:
        lo = np.asarray(self._pos_lo, dtype=np.float32)
        hi = np.asarray(self._pos_hi, dtype=np.float32)
        return np.clip(np.asarray(pts, dtype=np.float32), lo, hi).astype(np.float32)

    @property
    def mpc_period_s(self) -> float:
        return 1.0 / max(float(self.mpc_hz), 1e-6)

    def mark_armed(self, elapsed_s: float) -> None:
        """Start post-SPACE window; first MPC tick runs on the same frame when warmup is 0."""
        el = float(elapsed_s)
        self._armed_at = el
        self._last_mpc_time = el - self.mpc_period_s

    def _reset_gesture_kinematics(self) -> None:
        self._prev_gesture_sp = None
        self._prev_gesture_vel = None
        self._prev_gesture_t = None

    def _gesture_setpoint_kinematics(
        self, sp: np.ndarray, elapsed_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        sp = np.asarray(sp, dtype=np.float32)
        z = np.zeros((self.n_drones, 3), dtype=np.float32)
        if self._prev_gesture_sp is None or self._prev_gesture_t is None:
            self._prev_gesture_sp = sp.copy()
            self._prev_gesture_vel = z.copy()
            self._prev_gesture_t = float(elapsed_s)
            return z.copy(), z.copy()
        dt = max(float(elapsed_s) - float(self._prev_gesture_t), 1.0 / self.outer_fps)
        vel = ((sp - self._prev_gesture_sp) / dt).astype(np.float32)
        prev_vel = (
            self._prev_gesture_vel
            if self._prev_gesture_vel is not None
            else z
        )
        acc = ((vel - prev_vel) / dt).astype(np.float32)
        vmax = float(self.settings.vel_max)
        amax = float(self.settings.acc_max)
        vnorm = np.linalg.norm(vel, axis=-1, keepdims=True)
        vel = vel * np.minimum(1.0, vmax / np.maximum(vnorm, 1e-6))
        anorm = np.linalg.norm(acc, axis=-1, keepdims=True)
        acc = acc * np.minimum(1.0, amax / np.maximum(anorm, 1e-6))
        self._prev_gesture_sp = sp.copy()
        self._prev_gesture_vel = vel.copy()
        self._prev_gesture_t = float(elapsed_s)
        return vel, acc

    def _gesture_enforced(
        self, gesture: np.ndarray, *, hold_z: float | None = None
    ) -> np.ndarray:
        return self._lock_vertical_z(np.asarray(gesture, dtype=np.float32), hold_z)

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
        self._last_safe = pos.copy()
        self._last_safe_vel = vel.copy()
        self._control_updated = False
        self._last_mpc_time = -1e9
        self._last_ok = True
        self._reset_gesture_kinematics()

    def sync_gesture(self, gesture: np.ndarray, sim_vel: np.ndarray | None = None) -> None:
        """Re-init MPC state from current gesture (e.g. SPACE armed)."""
        self.reset(gesture, sim_vel)

    def _run_mpc(
        self, states: np.ndarray, gesture_setpoint: np.ndarray, elapsed_s: float
    ) -> bool:
        sp = np.asarray(gesture_setpoint, dtype=np.float32)
        vel_sp, acc_sp = self._gesture_setpoint_kinematics(sp, elapsed_s)
        self.solver_data = self.solver_data.replace(
            setpoints={"pos": sp, "vel": vel_sp, "acc": acc_sp},
        )
        t0 = time.perf_counter()
        success, _, self.solver_data = self._solve_fn(
            states, self.solver_data, self.settings
        )
        self._last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_count += 1
        ok = bool(np.all(success))
        self._last_ok = ok
        # Match axswarm's amswarm example: solve, advance SolverData, then consume
        # the first input setpoint from the shifted plan.
        self.solver_data = self.solver_data.step(self.solver_data)
        planned = np.asarray(self.solver_data.u_pos[:, 0], dtype=np.float32)
        planned_vel = np.asarray(self.solver_data.u_vel[:, 0], dtype=np.float32)
        if np.all(np.isfinite(planned)):
            self._last_safe = planned
            self._last_safe_vel = (
                planned_vel
                if np.all(np.isfinite(planned_vel))
                else np.zeros_like(planned, dtype=np.float32)
            )
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

    def safety_filter_targets(
        self,
        elapsed_s: float,
        gesture_setpoint: np.ndarray,
        sim: Sim | None = None,
        *,
        track_pos: np.ndarray | None = None,
        track_vel: np.ndarray | None = None,
        hold_z: float | None = None,
        snap_z_to_setpoint: bool = True,
    ) -> np.ndarray:
        """Return axswarm's filtered target for the raw setpoint."""
        el = float(elapsed_s)
        sp = self._lock_vertical_z(self._clamp_pos(gesture_setpoint), hold_z)
        g_enf = self._gesture_enforced(sp, hold_z=hold_z)

        if track_pos is not None:
            pos = np.asarray(track_pos, dtype=np.float32)
            if pos.shape != (self.n_drones, 3):
                raise ValueError(f"track_pos must be ({self.n_drones}, 3), got {pos.shape}")
            if track_vel is None:
                vel = np.zeros((self.n_drones, 3), dtype=np.float32)
            else:
                vel = np.asarray(track_vel, dtype=np.float32)
        elif sim is not None:
            pos = np.asarray(sim.data.states.pos[0], dtype=np.float32)
            vel = np.asarray(sim.data.states.vel[0], dtype=np.float32)
        else:
            raise ValueError("safety_filter_targets requires sim or track_pos")
        states = np.concatenate([pos, vel], axis=-1)
        self._control_updated = False

        due = (el - self._last_mpc_time) >= self.mpc_period_s - 1e-9
        if due:
            self._last_mpc_time = el
            self._run_mpc(states, sp, el)
        if self._last_safe is not None:
            out = np.asarray(self._last_safe, dtype=np.float32)
        else:
            out = np.asarray(pos, dtype=np.float32)

        out = self._lock_vertical_z(self._clamp_pos(out), hold_z)
        if hold_z is None and snap_z_to_setpoint:
            out[:, 2] = g_enf[:, 2]
        return out

    def current_control_velocity(self) -> np.ndarray:
        if self._last_safe_vel is None:
            return np.zeros((self.n_drones, 3), dtype=np.float32)
        return np.asarray(self._last_safe_vel, dtype=np.float32)

    def control_updated(self) -> bool:
        return bool(self._control_updated)

    def track_target_for(
        self,
        elapsed_s: float,
        gesture_setpoint: np.ndarray,
        sim: Sim | None = None,
        *,
        track_pos: np.ndarray | None = None,
        track_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Alias for :meth:`safety_filter_targets`."""
        return self.safety_filter_targets(
            elapsed_s,
            gesture_setpoint,
            sim,
            track_pos=track_pos,
            track_vel=track_vel,
        )

    def status_line(self) -> str:
        if self._solve_count == 0:
            return "axswarm-filter: idle"
        ok_frac = 1.0 - self._fail_count / max(1, self._solve_count)
        state = "ok" if self._last_ok else "best_effort"
        return (
            f"axswarm-filter:{self.mpc_hz:.0f}Hz "
            f"ok={ok_frac * 100:.0f}% "
            f"last={self._last_solve_ms:.0f}ms {state}"
        )


AxswarmRealtimePlanner = AxswarmSafetyFilter
