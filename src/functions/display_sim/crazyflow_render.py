"""Iso-Swarm Crazyflow visualization helpers (keeps upstream crazyflow unpatched)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from numpy.typing import NDArray

_PROFILE_SIM_RENDER = os.environ.get("ISO_SWARM_PROFILE_SIM_RENDER", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_PROFILE_SIM_RENDER_EVERY = max(1, int(os.environ.get("ISO_SWARM_PROFILE_SIM_RENDER_EVERY", "200")))

# Crazyflow 0.1+ uses free joints (7 qpos per drone), not mocap bodies.
_QPOS_ADR_CACHE: dict[int, np.ndarray] = {}


@dataclass
class SimRenderProfiler:
    """Per-call wall time for render helpers (perf_counter). Off unless ISO_SWARM_PROFILE_SIM_RENDER=1."""

    enabled: bool = False
    report_every: int = 200
    _depth: int = 0
    _active: bool = False
    _call_t0: float = 0.0
    _last: float = 0.0
    _totals: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _calls: int = 0

    def call_start(self) -> None:
        if not self.enabled:
            return
        self._depth += 1
        if self._depth > 1:
            return
        self._active = True
        self._call_t0 = self._last = time.perf_counter()

    def section(self, name: str) -> None:
        if not self.enabled or not self._active or self._depth != 1:
            return
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self._totals[name] = self._totals.get(name, 0.0) + dt
        self._counts[name] = self._counts.get(name, 0) + 1

    def call_end(self, *, label: str = "render_targets") -> None:
        if not self.enabled:
            return
        if self._depth > 0:
            self._depth -= 1
        if not self._active or self._depth > 0:
            return
        total = time.perf_counter() - self._call_t0
        self._totals["call_total"] = self._totals.get("call_total", 0.0) + total
        self._counts["call_total"] = self._counts.get("call_total", 0) + 1
        self._active = False
        self._calls += 1
        every = max(1, int(self.report_every))
        if (self._calls % every) != 0:
            return
        n = max(1, self._calls)
        parts: list[tuple[str, float]] = []
        for key, sec in self._totals.items():
            if key == "call_total":
                continue
            cnt = max(1, self._counts.get(key, 0))
            parts.append((key, (sec / cnt) * 1000.0))
        parts.sort(key=lambda x: x[1], reverse=True)
        avg_call_ms = (self._totals.get("call_total", 0.0) / n) * 1000.0
        detail = " ".join(f"{k}={v:.2f}ms" for k, v in parts[:14])
        print(
            f"[sim-render-profile {label} calls={n}] "
            f"avg_call≈{avg_call_ms:.2f}ms | {detail}"
        )
        self._totals.clear()
        self._counts.clear()
        self._calls = 0


_render_prof = SimRenderProfiler(enabled=_PROFILE_SIM_RENDER, report_every=_PROFILE_SIM_RENDER_EVERY)
if _render_prof.enabled:
    print(
        f"Sim render profiling ON (perf_counter); report every {_render_prof.report_every} calls."
    )


def _drone_qpos_adrs(sim: Sim) -> np.ndarray:
    """Start index in ``mj_data.qpos`` for each ``drone:i`` free joint."""
    key = (id(sim.mj_model), sim.n_drones)
    cached = _QPOS_ADR_CACHE.get(key)
    if cached is not None:
        return cached
    adrs = np.empty(sim.n_drones, dtype=np.int32)
    for i in range(sim.n_drones):
        body_id = sim.mj_model.body(f"drone:{i}").id
        jnt_id = int(sim.mj_model.body_jntadr[body_id])
        adrs[i] = int(sim.mj_model.jnt_qposadr[jnt_id])
    _QPOS_ADR_CACHE[key] = adrs
    return adrs


def _ensure_viewer(
    sim: Sim,
    *,
    mode: str | None,
    camera: int | str,
    cam_config: dict | None,
    width: int,
    height: int,
) -> None:
    if sim.viewer is not None:
        return
    if isinstance(camera, str):
        cam_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        assert cam_id > -1, f"Camera name '{camera}' not found in the model."
    elif isinstance(camera, int):
        cam_id = camera
        assert cam_id >= -1, f"camera id must be >=-1, was {cam_id}"
    else:
        raise TypeError("camera argument must be integer or string")
    sim.mj_model.vis.global_.offwidth = width
    sim.mj_model.vis.global_.offheight = height
    sim.viewer = MujocoRenderer(
        sim.mj_model,
        sim.mj_data,
        max_geom=sim.max_visual_geom,
        default_cam_config=cam_config,
        height=height,
        width=width,
        camera_id=cam_id,
    )
    if mode == "human" and cam_id > -1:
        sim.viewer.render(mode)
        sim.viewer.viewer.cam.fixedcamid = cam_id
        sim.viewer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED


def _write_targets_to_qpos(sim: Sim, pts: np.ndarray) -> None:
    """Viz-only: place drones at targets without JAX sync_sim2mjx."""
    adrs = _drone_qpos_adrs(sim)
    qpos = sim.mj_data.qpos
    for i, adr in enumerate(adrs):
        qpos[adr : adr + 3] = pts[i]
        qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)  # MuJoCo quat [w,x,y,z]


def render_targets(
    sim: Sim,
    targets: Any,
    *,
    mode: str | None = "human",
    camera: int | str = -1,
    cam_config: dict | None = None,
    width: int = 1920,
    height: int = 1080,
) -> NDArray | None:
    """Render drones at command targets (viz-only fast path; physics unchanged).

    Uses direct ``qpos`` writes + ``mj_forward`` — not ``sim.render()``, which runs a full
    JAX ``sync_sim2mjx`` (kinematics/collision) every frame and is much slower.

    Enable timing: ``ISO_SWARM_PROFILE_SIM_RENDER=1`` (optional ``ISO_SWARM_PROFILE_SIM_RENDER_EVERY=N``).
    """
    _render_prof.call_start()
    try:
        _ensure_viewer(
            sim,
            mode=mode,
            camera=camera,
            cam_config=cam_config,
            width=width,
            height=height,
        )
        _render_prof.section("viewer_init")

        pts = np.asarray(targets, dtype=np.float64)
        if pts.shape != (sim.n_drones, 3):
            raise ValueError(f"targets must have shape ({sim.n_drones}, 3), got {pts.shape}")
        _write_targets_to_qpos(sim, pts)
        _render_prof.section("target_qpos")

        mujoco.mj_forward(sim.mj_model, sim.mj_data)
        _render_prof.section("mj_forward")

        out = sim.viewer.render(mode)
        _render_prof.section("viewer_render")
        return out
    finally:
        _render_prof.call_end(label="render_targets")
