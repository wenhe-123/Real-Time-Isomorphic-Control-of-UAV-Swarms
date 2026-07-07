"""Raw target -> axswarm filter per frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions.mode_switch.online_frame_gesture import GestureFrameResult
from functions.runtime.online_boot import OnlineBoot
from functions.runtime.online_runtime_config import OnlineRuntimeConfig
from functions.swarm_motion.spacing_guard import closest_pair


@dataclass
class TargetFilterResult:
    axswarm_input: np.ndarray
    cmd_target: np.ndarray
    cmd_velocity: np.ndarray
    control_updated: bool
    cmd_source: str = "unknown"
    plan_drift_m: float = 0.0
    mpc_due: bool = False


def filter_online_targets(
    *,
    boot: OnlineBoot,
    cfg: OnlineRuntimeConfig,
    gest: GestureFrameResult,
    raw_target: np.ndarray,
    morph_targets_before_left_m: np.ndarray,
    elapsed: float,
    track_pos: np.ndarray | None,
    track_vel: np.ndarray | None = None,
) -> tuple[TargetFilterResult, bool]:
    """Run axswarm MPC filtering on raw morph targets for one online frame.

    Args:
        boot: Online boot state (axswarm planner, gesture control flags).
        cfg: Runtime config (spacing audit interval, min separation).
        gest: Gesture frame result (open-hand alpha, visibility).
        raw_target: Unfiltered morph targets (sim m).
        morph_targets_before_left_m: Pre-left-pose targets (unused; reserved).
        elapsed: Monotonic elapsed time since session start (s).
        track_pos: Optional drone positions for MPC state tracking.
        track_vel: Optional drone velocities for MPC state tracking.

    Returns:
        ``(result, prev_gesture_control_enabled)`` — filter output and updated gesture-arm flag.
    """
    del morph_targets_before_left_m
    prev_gesture_control_enabled = boot.prev_gesture_control_enabled

    axswarm_input = np.asarray(raw_target, dtype=np.float32)
    if (
        cfg.spacing_audit_every > 0
        and gest.open_out is not None
        and float(gest.open_out) < 0.32
        and (boot.frame_idx % cfg.spacing_audit_every) == 0
    ):
        d_raw, pri, prj = closest_pair(raw_target)
        env = float(cfg.min_separation_m)
        print(
            f"[spacing live f={boot.frame_idx}] open={float(gest.open_out):.2f} "
            f"raw={d_raw:.3f}m pair=({pri},{prj}) "
            f"(min_sep={env:.2f}m axswarm_env≈{env:.2f}m)"
        )
    if boot.gesture_control_enabled and not prev_gesture_control_enabled:
        _pos = np.asarray(axswarm_input, dtype=np.float32)
        if track_pos is not None:
            _sim = np.asarray(track_pos, dtype=np.float32)
            _dz = float(np.mean(_pos[:, 2]) - np.mean(_sim[:, 2]))
            if abs(_dz) > 0.08:
                print(
                    f"Gesture armed: sync MPC to hover setpoint "
                    f"(sim z≈{float(np.mean(_sim[:, 2])):.2f}m, "
                    f"target z≈{float(np.mean(_pos[:, 2])):.2f}m, Δz={_dz:+.2f}m)."
                )
        _vel = np.zeros((boot.axswarm_rt.n_drones, 3), dtype=np.float32)
        boot.axswarm_rt.sync_gesture(_pos, _vel)
        print("Gesture armed. Axswarm active.")
    prev_gesture_control_enabled = bool(boot.gesture_control_enabled)

    _hold_z: float | None = None
    _prearm_phase = str(boot.prearm_phase)
    _vertical_leg = str(boot.prearm_vertical_leg)
    if not boot.gesture_control_enabled:
        if _prearm_phase == "ground":
            _hold_z = float(boot.ground_z)
        elif _prearm_phase == "vertical" and _vertical_leg == "climb":
            _hold_z = float(boot.prearm_takeoff_z)
    _mpc_track = track_pos
    if _mpc_track is None and boot.sim is None:
        _mpc_track = np.asarray(boot.cmd_target, dtype=np.float32)
    cmd = boot.axswarm_rt.plan_control(
        elapsed,
        axswarm_input,
        sim=boot.sim,
        track_pos=_mpc_track,
        track_vel=track_vel,
        hold_z=_hold_z,
    )
    return (
        TargetFilterResult(
            axswarm_input=np.asarray(axswarm_input, dtype=np.float32),
            cmd_target=np.asarray(cmd.pos, dtype=np.float32),
            cmd_velocity=np.asarray(cmd.vel, dtype=np.float32),
            control_updated=cmd.updated,
            cmd_source=str(cmd.cmd_source),
            plan_drift_m=float(cmd.plan_drift_m),
            mpc_due=bool(cmd.mpc_due),
        ),
        prev_gesture_control_enabled,
    )
