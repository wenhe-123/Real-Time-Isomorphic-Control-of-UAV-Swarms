"""Realtime axswarm **safety filter** for iso_swarm (does not modify the axswarm package).

Gesture supplies desired setpoints; axswarm MPC (soft position + collision constraints)
returns a feasible, collision-aware correction. Crazyflow still tracks the filtered targets.

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

from functions.swarm_motion.spacing_guard import (
    closest_pair,
    clamp_targets_step,
    conservative_degraded_target,
    enforce_min_separation,
)

if TYPE_CHECKING:
    from crazyflow.sim import Sim


def default_axswarm_project_root() -> Path:
    """``submodules/axswarm`` (see pyproject.toml)."""
    here = Path(__file__).resolve()
    iso_swarm = here.parents[3]
    return iso_swarm / "submodules" / "axswarm"


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


def load_axswarm_yaml(settings_path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    with open(settings_path) as f:
        config = yaml.safe_load(f)
    settings_dict = dict(config["SolverSettings"])
    for k, v in settings_dict.items():
        if isinstance(v, list):
            settings_dict[k] = np.asarray(v)
    dynamics = {k: np.asarray(v) for k, v in config["Dynamics"].items()}
    return settings_dict, dynamics


def adapt_settings_dict_for_online(
    settings_dict: dict,
    *,
    n_drones: int,
    min_separation_m: float,
    xy_radius_m: float,
    z_min_m: float,
    z_max_m: float,
    pad_m: float = 0.4,
    max_iters: int | None = None,
    pos_weight: float | None = None,
) -> dict:
    """Online safety filter: track gesture (soft), enforce collisions (hard/soft)."""
    r = float(max(xy_radius_m, 1.0))
    z0 = float(z_min_m)
    z1 = float(max(z_max_m, z0 + 0.5))
    settings_dict = dict(settings_dict)
    settings_dict["pos_min"] = np.array([-r - pad_m, -r - pad_m, max(0.05, z0 - pad_m)])
    settings_dict["pos_max"] = np.array([r + pad_m, r + pad_m, z1 + pad_m])
    settings_dict["vel_max"] = float(settings_dict.get("vel_max", 1.73))
    half_sep = max(0.08, 0.5 * float(min_separation_m))
    settings_dict["collision_envelope"] = np.array([half_sep, half_sep, half_sep])
    settings_dict["max_collisions"] = int(max(1, n_drones - 1))
    settings_dict["pos_constraints"] = "soft"
    if pos_weight is not None:
        settings_dict["pos_weight"] = float(pos_weight)
    else:
        settings_dict["pos_weight"] = float(max(100.0, settings_dict.get("pos_weight", 100.0)))
    if max_iters is not None:
        settings_dict["max_iters"] = int(max(5, max_iters))
    else:
        settings_dict["max_iters"] = int(min(40, max(12, 80 - 2 * n_drones)))
    return settings_dict


@dataclass(frozen=True)
class AxswarmMotionLimits:
    """Motion caps aligned with teacher ``params/settings.yaml``."""

    vel_max_m_s: float
    acc_max_m_s2: float
    mpc_freq_hz: float
    min_separation_m: float
    per_substep_cap_m: float
    per_outer_frame_cap_m: float
    target_alpha: float


def per_substep_target_cap_m(
    *,
    vel_max_m_s: float,
    sim_freq_hz: int,
    outer_fps: int,
    max_substeps: int,
) -> float:
    """Cap per physics substep so blended targets respect axswarm ``vel_max`` (m/s)."""
    del outer_fps, max_substeps  # Cap is per physics step, independent of camera frame grouping.
    return float(vel_max_m_s) / max(float(sim_freq_hz), 1.0)


def load_axswarm_motion_limits(
    *,
    settings_path: Path | None = None,
    project_root: Path | None = None,
    sim_freq_hz: int = 500,
    outer_fps: int = 30,
    max_substeps: int = 160,
    min_separation_m: float | None = None,
) -> AxswarmMotionLimits:
    """Read yaml limits and derive online Crazyflow caps (dual / integrated defaults)."""
    root = ensure_axswarm_import(project_root)
    if settings_path is None:
        settings_path = root / "params" / "settings.yaml"
    settings_dict, _ = load_axswarm_yaml(Path(settings_path))
    vel = float(settings_dict.get("vel_max", 1.73))
    acc = float(settings_dict.get("acc_max", 1.0))
    freq = float(settings_dict.get("freq", 8.0))
    env = np.asarray(settings_dict.get("collision_envelope", [0.15, 0.15, 0.15]), dtype=np.float64)
    sep = float(min_separation_m) if min_separation_m is not None else float(2.0 * np.max(env))
    per_step = per_substep_target_cap_m(
        vel_max_m_s=vel,
        sim_freq_hz=sim_freq_hz,
        outer_fps=outer_fps,
        max_substeps=max_substeps,
    )
    per_frame = vel / max(float(outer_fps), 1.0)
    # In axswarm mode the safety filter already returns per-frame, velocity-limited
    # targets, so the outer loop should track them closely instead of adding a
    # second slow EMA.
    alpha = 0.99
    return AxswarmMotionLimits(
        vel_max_m_s=vel,
        acc_max_m_s2=acc,
        mpc_freq_hz=freq,
        min_separation_m=sep,
        per_substep_cap_m=per_step,
        per_outer_frame_cap_m=per_frame,
        target_alpha=alpha,
    )


def clamp_plan_toward_gesture(
    planned: np.ndarray, gesture: np.ndarray, *, max_deviation_m: float
) -> np.ndarray:
    """Limit per-drone deviation from gesture so the filter does not override the hand."""
    p = np.asarray(planned, dtype=np.float32)
    g = np.asarray(gesture, dtype=np.float32)
    delta = p - g
    dist = np.linalg.norm(delta, axis=-1, keepdims=True)
    cap = float(max(0.01, max_deviation_m))
    scale = np.minimum(1.0, cap / np.maximum(dist, 1e-6))
    return (g + delta * scale).astype(np.float32)


@dataclass
class AxswarmSafetyFilter:
    """Gesture setpoints → axswarm MPC → collision-feasible targets for Crazyflow."""

    settings: Any
    dynamics: dict[str, np.ndarray]
    solver_data: Any
    n_drones: int
    mpc_hz: float
    max_deviation_m: float
    max_solve_ms: float
    min_separation_m: float
    fail_sep_mult: float
    fail_gesture_blend: float
    fail_gesture_blend_recover: float
    fail_frame_step_m: float
    recover_hold_s: float
    recover_step_frac: float
    gesture_creep_blend: float
    gesture_creep_blend_recover: float
    _last_safe: np.ndarray | None
    _last_mpc_time: float
    _solve_count: int
    _fail_count: int
    _skip_count: int
    _last_solve_ms: float
    _last_ok: bool
    _armed_at: float
    _recover_until_s: float
    arm_warmup_s: float

    @classmethod
    def create(
        cls,
        n_drones: int,
        *,
        min_separation_m: float,
        xy_radius_m: float,
        z_min_m: float,
        z_max_m: float,
        settings_path: Path | None = None,
        project_root: Path | None = None,
        max_iters: int | None = None,
        pos_weight: float | None = None,
        max_deviation_m: float = 0.2,
        max_solve_ms: float = 90.0,
        outer_fps: int = 30,
        fail_sep_mult: float = 1.3,
        fail_gesture_blend: float = 0.2,
        fail_gesture_blend_recover: float = 0.12,
        recover_hold_s: float = 0.7,
        recover_step_frac: float = 1.0,
        gesture_creep_blend: float = 0.90,
        gesture_creep_blend_recover: float = 0.35,
        arm_warmup_s: float = 0.0,
    ) -> AxswarmSafetyFilter:
        if n_drones > 16:
            max_solve_ms = max(float(max_solve_ms), 220.0)
        if n_drones < 8:
            raise ValueError(f"axswarm safety filter requires n_drones >= 8, got {n_drones}")
        root = ensure_axswarm_import(project_root)
        from axswarm import SolverData, SolverSettings, solve  # noqa: WPS433

        if settings_path is None:
            settings_path = root / "params" / "settings.yaml"
        settings_dict, dynamics = load_axswarm_yaml(Path(settings_path))
        settings_dict = adapt_settings_dict_for_online(
            settings_dict,
            n_drones=n_drones,
            min_separation_m=min_separation_m,
            xy_radius_m=xy_radius_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            max_iters=max_iters,
            pos_weight=pos_weight,
        )
        settings = SolverSettings(**settings_dict)
        mpc_hz = float(settings.freq)
        if n_drones > 16:
            mpc_hz = min(mpc_hz, 4.0)
        elif n_drones > 12:
            mpc_hz = min(mpc_hz, 6.0)
        vel_max = float(settings_dict["vel_max"])
        filt = cls(
            settings=settings,
            dynamics=dynamics,
            solver_data=None,
            n_drones=n_drones,
            mpc_hz=mpc_hz,
            max_deviation_m=float(max(0.02, max_deviation_m)),
            max_solve_ms=float(max(10.0, max_solve_ms)),
            min_separation_m=float(min_separation_m),
            fail_sep_mult=float(max(1.0, fail_sep_mult)),
            fail_gesture_blend=float(np.clip(fail_gesture_blend, 0.0, 1.0)),
            fail_gesture_blend_recover=float(np.clip(fail_gesture_blend_recover, 0.0, 1.0)),
            fail_frame_step_m=8.0 * vel_max / max(float(outer_fps), 1.0),
            recover_hold_s=float(max(0.5, recover_hold_s)),
            recover_step_frac=float(np.clip(recover_step_frac, 0.08, 1.0)),
            gesture_creep_blend=float(np.clip(gesture_creep_blend, 0.02, 0.95)),
            gesture_creep_blend_recover=float(np.clip(gesture_creep_blend_recover, 0.01, 0.7)),
            _last_safe=None,
            _last_mpc_time=-1e9,
            _solve_count=0,
            _fail_count=0,
            _skip_count=0,
            _last_solve_ms=0.0,
            _last_ok=False,
            _armed_at=-1e9,
            _recover_until_s=-1e9,
            arm_warmup_s=float(max(0.0, arm_warmup_s)),
        )
        filt._solve_fn = solve
        filt._SolverData = SolverData
        return filt

    @property
    def mpc_period_s(self) -> float:
        return 1.0 / max(float(self.mpc_hz), 1e-6)

    def mark_armed(self, elapsed_s: float) -> None:
        """Start post-SPACE window; first MPC tick runs on the same frame when warmup is 0."""
        el = float(elapsed_s)
        self._armed_at = el
        # Otherwise ``due`` stays false until one full ``mpc_period_s`` after SPACE.
        self._last_mpc_time = el - self.mpc_period_s

    def enter_recover(self, elapsed_s: float) -> None:
        """Slow creep + keep re-planning after MPC fail, tight sim spacing, or morph jump."""
        el = float(elapsed_s)
        # Extend only when not already in recover — avoid per-frame renew that locks slow mode.
        if el >= float(self._recover_until_s):
            self._recover_until_s = el + float(self.recover_hold_s)
        self._last_ok = False

    def in_recover_at(self, elapsed_s: float) -> bool:
        return float(elapsed_s) < float(self._recover_until_s)

    def _slow_creep_at(self, elapsed_s: float) -> bool:
        return self.in_recover_at(elapsed_s)

    def _gesture_enforced(self, gesture: np.ndarray) -> np.ndarray:
        return enforce_min_separation(
            gesture, float(self.min_separation_m) * 1.15, iters=10
        )

    def _frame_step_m(self, elapsed_s: float, *, warmup: bool) -> float:
        base = float(self.fail_frame_step_m)
        if warmup:
            return base
        if self._slow_creep_at(elapsed_s):
            return base * float(self.recover_step_frac)
        return base

    def _creep_toward(
        self,
        anchor: np.ndarray,
        gesture_enf: np.ndarray,
        sim_pos: np.ndarray,
        *,
        elapsed_s: float,
        warmup: bool,
        gesture_blend: float | None = None,
    ) -> np.ndarray:
        if gesture_blend is None:
            gb = (
                float(self.gesture_creep_blend_recover)
                if self._slow_creep_at(elapsed_s)
                else float(self.gesture_creep_blend)
            )
        else:
            gb = float(gesture_blend)
        if warmup:
            gb = max(gb, 0.90)
        tgt = ((1.0 - gb) * anchor + gb * gesture_enf).astype(np.float32)
        step = self._frame_step_m(elapsed_s, warmup=warmup)
        out = clamp_targets_step(sim_pos, tgt, step)
        sep_mult = float(self.fail_sep_mult) if self._slow_creep_at(elapsed_s) else 1.15
        return enforce_min_separation(
            out, float(self.min_separation_m) * sep_mult, iters=10
        )

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
            setpoints={"pos": pos.copy(), "vel": z.copy(), "acc": z.copy()},
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
        self._last_mpc_time = -1e9
        self._last_ok = True

    def sync_gesture(self, gesture: np.ndarray, sim_vel: np.ndarray | None = None) -> None:
        """Re-init MPC state from current gesture (e.g. SPACE armed)."""
        self.reset(gesture, sim_vel)

    def _clip_states_velocity(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32).copy()
        speed = np.linalg.norm(states[:, 3:], axis=-1, keepdims=True)
        vmax = float(self.settings.vel_max)
        states[:, 3:] *= np.minimum(1.0, vmax / np.maximum(speed, 1e-6))
        return states

    def _run_mpc(self, states: np.ndarray, gesture_setpoint: np.ndarray) -> bool:
        sp = np.asarray(gesture_setpoint, dtype=np.float32)
        z = np.zeros((self.n_drones, 3), dtype=np.float32)
        self.solver_data = self.solver_data.replace(
            setpoints={"pos": sp, "vel": z, "acc": z},
        )
        t0 = time.perf_counter()
        success, _, self.solver_data = self._solve_fn(
            self._clip_states_velocity(states), self.solver_data, self.settings
        )
        self._last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_count += 1
        ok = bool(np.all(success))
        self._last_ok = ok
        if not ok:
            self._fail_count += 1
            return False
        planned = np.asarray(self.solver_data.pos[:, 1], dtype=np.float32)
        if not np.all(np.isfinite(planned)):
            self._fail_count += 1
            self._last_ok = False
            return False
        safe = clamp_plan_toward_gesture(
            planned, sp, max_deviation_m=self.max_deviation_m
        )
        safe = enforce_min_separation(safe, self.min_separation_m, iters=8)
        self._last_safe = safe
        self.solver_data = self.solver_data.step(self.solver_data)
        return True

    def safety_filter_targets(
        self,
        elapsed_s: float,
        gesture_setpoint: np.ndarray,
        sim: Sim | None = None,
        *,
        track_pos: np.ndarray | None = None,
        track_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Track spacing-safe gesture targets; use MPC ticks as collision corrections."""
        el = float(elapsed_s)
        sp = np.asarray(gesture_setpoint, dtype=np.float32)
        sep_out = float(self.min_separation_m) * 1.15
        g_enf = self._gesture_enforced(sp)

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

        sim_min, _, _ = closest_pair(pos)
        warmup = (el - self._armed_at) < self.arm_warmup_s
        was_recover = self.in_recover_at(el)
        if not warmup and sim_min < float(self.min_separation_m) * 0.86:
            self.enter_recover(el)
            if self.solver_data is not None and not was_recover:
                self.sync_gesture(pos, vel)
        anchor = (
            np.asarray(self._last_safe, dtype=np.float32)
            if self._last_safe is not None
            else pos.copy()
        )
        if warmup:
            return enforce_min_separation(g_enf, sep_out, iters=6)

        step = self._frame_step_m(el, warmup=False)
        out = g_enf.copy()

        due = (el - self._last_mpc_time) >= self.mpc_period_s - 1e-9
        overloaded = self._last_solve_ms > self.max_solve_ms
        if due:
            self._last_mpc_time = el
            if not overloaded:
                if self._run_mpc(states, sp) and self._last_safe is not None:
                    cand = enforce_min_separation(self._last_safe, sep_out, iters=12)
                    plan_min, _, _ = closest_pair(cand)
                    if plan_min >= float(self.min_separation_m) * 0.97:
                        out = cand
                        if sim_min >= float(self.min_separation_m):
                            self._last_ok = True
                            self._recover_until_s = -1e9
                    else:
                        self.enter_recover(el)
                        self._last_ok = False
                        out = conservative_degraded_target(
                            g_enf,
                            anchor,
                            min_separation_m=self.min_separation_m,
                            sep_mult=self.fail_sep_mult,
                            gesture_blend=self.fail_gesture_blend_recover,
                            max_step_m=step,
                        )
                        out = clamp_targets_step(pos, out, step)
                else:
                    self.enter_recover(el)
                    gb = (
                        self.fail_gesture_blend_recover
                        if self._slow_creep_at(el)
                        else self.fail_gesture_blend
                    )
                    out = conservative_degraded_target(
                        g_enf,
                        anchor,
                        min_separation_m=self.min_separation_m,
                        sep_mult=self.fail_sep_mult,
                        gesture_blend=gb,
                        max_step_m=step,
                    )
                    out = clamp_targets_step(pos, out, step)
            else:
                self._skip_count += 1

        if self._slow_creep_at(el):
            out = clamp_targets_step(pos, out, step)

        if self._last_ok and sim_min >= float(self.min_separation_m) and el >= self._recover_until_s:
            self._recover_until_s = -1e9

        return enforce_min_separation(out, sep_out, iters=6)

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
        hold = "recover" if self._recover_until_s > -1e8 else "ok"
        if (self._fail_count > 0 or self._skip_count > 0) and not self._last_ok:
            hold = "recover"
        return (
            f"axswarm-filter:{self.mpc_hz:.0f}Hz "
            f"ok={ok_frac * 100:.0f}% "
            f"last={self._last_solve_ms:.0f}ms skip={self._skip_count} {hold}"
        )


# Backward-compatible name used during integration.
AxswarmRealtimePlanner = AxswarmSafetyFilter
