"""MuJoCo sim bridge: same HL takeoff/land + axswarm setpoint gating as real Crazyflies."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from functions.display_sim.crazyflow_render import step_sim_to_cmd
from functions.swarm_motion.prearm import PREARM_PRE_LAND_HOVER_S

if TYPE_CHECKING:
    from crazyflow.sim import Sim


class SimSwarmExecutor:
    """Drive Crazyflow sim with the same prearm / axswarm interface as ``RealSwarmExecutor``."""

    def __init__(
        self,
        *,
        sim: Sim,
        ground_layout: np.ndarray,
        ctrl_freq: float,
        max_sim_substeps: int,
    ):
        self.sim = sim
        self._ground_layout = np.asarray(ground_layout, dtype=np.float32)
        self.ctrl_freq = float(ctrl_freq)
        self._max_sim_substeps = max(1, int(max_sim_substeps))
        self.n_physical = int(sim.n_drones)
        self.physical_armed = False
        self._control_halted = True

    @property
    def control_halted(self) -> bool:
        return bool(self._control_halted)

    def get_sim_track_positions(
        self, morph_fallback: np.ndarray, n_morph: int
    ) -> np.ndarray | None:
        del morph_fallback, n_morph
        return np.asarray(self.sim.data.states.pos[0], dtype=np.float32)

    def pause_setpoints_for_hl(self) -> None:
        """Stop axswarm setpoint stream before HL maneuver (matches real swarm)."""
        self._control_halted = True

    def _blocking_move(self, target: np.ndarray, *, duration_s: float) -> None:
        start = np.asarray(self.sim.data.states.pos[0], dtype=np.float64).copy()
        goal = np.asarray(target, dtype=np.float64)
        n_steps = max(1, int(round(float(duration_s) * self.ctrl_freq)))
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            pos = start * (1.0 - alpha) + goal * alpha
            step_sim_to_cmd(
                self.sim,
                pos,
                outer_fps=int(round(self.ctrl_freq)),
                max_substeps=self._max_sim_substeps,
                control_hz=self.ctrl_freq,
            )

    def high_level_takeoff(self, height_m: float, *, duration_s: float = 3.0) -> None:
        """Block until vertical climb completes; then allow axswarm low-level setpoints."""
        self._control_halted = True
        h = float(height_m)
        dur = float(duration_s)
        print(
            f"Sim swarm: high-level takeoff +{h:.2f}m ({dur:.1f}s, no setpoint stream)...",
            flush=True,
        )
        target = np.asarray(self.sim.data.states.pos[0], dtype=np.float64).copy()
        target[:, 2] = h
        self._blocking_move(target, duration_s=dur)
        self._control_halted = False
        self.physical_armed = True

    def high_level_descend(
        self,
        distance_m: float,
        *,
        duration_s: float = 3.0,
        settle_s: float | None = None,
    ) -> None:
        """Block until in-place vertical descent completes (current XY, −Z)."""
        self._control_halted = True
        d = float(distance_m)
        dur = float(duration_s)
        hover_s = PREARM_PRE_LAND_HOVER_S if settle_s is None else float(settle_s)
        pos = np.asarray(self.sim.data.states.pos[0], dtype=np.float64).copy()
        target = pos.copy()
        target[:, 2] = pos[:, 2] - d
        print(
            f"Sim swarm: high-level descend −{d:.2f}m in place ({dur:.1f}s, no setpoint stream)...",
            flush=True,
        )
        self._blocking_move(target, duration_s=dur)
        if hover_s > 0.0:
            print(
                f"Sim swarm: holding {hover_s:.1f}s at descend target before land...",
                flush=True,
            )
            time.sleep(hover_s)

    def high_level_land(self, height_m: float = 0.0, *, duration_s: float = 3.0) -> None:
        """Block until ground land completes; keep setpoint stream off afterward."""
        del height_m
        self._control_halted = True
        dur = float(duration_s)
        z_ground = float(np.median(self._ground_layout[:, 2]))
        print(
            f"Sim swarm: high-level land (z≈{z_ground:.2f}m, {dur:.1f}s, no setpoint stream)...",
            flush=True,
        )
        target = np.asarray(self.sim.data.states.pos[0], dtype=np.float64).copy()
        target[:, :2] = self._ground_layout[:, :2]
        target[:, 2] = self._ground_layout[:, 2]
        self._blocking_move(target, duration_s=dur)
        self.physical_armed = False

    def track_frame(
        self,
        cmd_target: np.ndarray,
        *,
        control_hz: float,
        max_substeps: int,
        prearm_phase: str = "ground",
        prearm_vertical_leg: str = "climb",
        just_prearm_phase: bool = False,
        prearm_vertical_layout: np.ndarray | None = None,
    ) -> None:
        if self._control_halted:
            return
        cmd = np.asarray(cmd_target, dtype=np.float32)
        phase = str(prearm_phase)

        if just_prearm_phase:
            if phase == "vertical" and prearm_vertical_layout is not None:
                z_takeoff = float(np.median(prearm_vertical_layout[:, 2]))
                if str(prearm_vertical_leg) == "climb":
                    print(
                        f"Sim vertical takeoff → z={z_takeoff:.2f}m "
                        "(axswarm-planned stream, same as real)."
                    )
            elif phase == "formation":
                print(
                    "Sim hover formation: axswarm setpoint stream "
                    "(press 1 for in-place HL descend + land)."
                )

        step_sim_to_cmd(
            self.sim,
            cmd,
            outer_fps=int(round(control_hz)),
            max_substeps=max(1, int(max_substeps)),
            velocities=None,
            control_hz=float(control_hz),
        )
