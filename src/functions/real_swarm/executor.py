"""Bridge filtered ``cmd_target`` arrays to physical Crazyflie setpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from swarm_gpt.core.drone_swarm import DroneSwarm

from functions.real_swarm.swarm_config import RealFrameMapping, RealSwarmOptions, load_drones_config
from functions.swarm_motion.prearm import PREARM_PRE_LAND_HOVER_S

logger = logging.getLogger(__name__)

_MODE_LED_COLORS = {
    1: np.array([255, 40, 40, 0], dtype=np.float64),
    2: np.array([255, 200, 40, 0], dtype=np.float64),
    3: np.array([40, 200, 255, 0], dtype=np.float64),
    4: np.array([180, 80, 255, 0], dtype=np.float64),
    5: np.array([80, 255, 120, 0], dtype=np.float64),
}


def _ensure_rclpy() -> None:
    """Initialize ROS 2 if needed for mocap (``DroneSwarm`` / ``ROSConnector``).

    Raises:
        ImportError: If ``rclpy`` is not installed.
    """
    import rclpy

    if not rclpy.ok():
        rclpy.init()


class RealSwarmExecutor:
    """Send online ``cmd_target`` poses to a connected Crazyflie swarm."""

    def __init__(
        self,
        *,
        config_path: Path,
        morph_point_count: int | None = None,
        dry_run: bool = False,
    ):
        """Connect Crazyflie swarm and load sim↔room frame mapping from TOML.

        Args:
            config_path: Path to drones layout TOML (``active`` or ``[[drone]]``).
            morph_point_count: Virtual morph sample count; must be ≥ physical drones.
            dry_run: When ``True``, skip radio and ROS mocap connect (pipeline only).

        Raises:
            ValueError: When ``morph_point_count`` is smaller than the physical count.
        """
        drones, mapping, opts = load_drones_config(config_path)
        n_physical = len(drones)
        if morph_point_count is not None and int(morph_point_count) < n_physical:
            raise ValueError(
                f"morph point-count ({morph_point_count}) must be >= physical drones ({n_physical})"
            )

        self.mapping = mapping
        self.opts = opts
        self.n_physical = n_physical
        self._uris = [d["uri"] for d in drones.values()]
        self._homes = {
            d["uri"]: np.asarray(d["pos"], dtype=np.float64) for d in drones.values()
        }
        self.physical_armed = False
        self._last_mode = 1
        self._mocap_hold_logged = False
        self._dry_run = bool(dry_run)
        self.swarm: DroneSwarm | None = None
        self._setpoint_period_s = 1.0 / max(float(opts.ctrl_freq), 1e-6)
        self._setpoint_next_mono = 0.0
        self._pending_sim_layout: np.ndarray | None = None
        self._control_halted = False
        self._closed = False

        print(
            f"{'[dry-run] ' if self._dry_run else ''}"
            f"Connecting {n_physical} Crazyflie(s) (morph={morph_point_count} virtual targets; "
            f"physical indices 0..{n_physical - 1}) mocap/ROS ..."
        )
        print(
            f"  Sim (0,0,0) → room {np.round(mapping.origin, 3)} m "
            f"(scale={mapping.scale}, yaw={np.rad2deg(mapping.yaw_rad):.1f}°)"
        )
        if self._dry_run:
            print(
                "Dry-run: skipping Crazyflie radio + ROS mocap connect "
                "(Orbbec + axswarm pipeline only)."
            )
            return

        _ensure_rclpy()
        self.swarm = DroneSwarm(
            drones,
            ctrl_freq=opts.ctrl_freq,
            update_freq=opts.update_freq,
            col_freq=opts.col_freq,
            lighthouse=False,
        )
        missing = self.swarm.missing_uris()
        if missing:
            logger.warning("Inactive URIs after connect: %s", missing)
        # No low-level setpoints until high-level takeoff (key 1).
        self._control_halted = True

    @property
    def ctrl_freq(self) -> float:
        """Return configured Crazyflie setpoint rate from TOML options.

        Returns:
            Control frequency in Hz.
        """
        return float(self.opts.ctrl_freq)

    def _physical_cmd(self, sim_layout: np.ndarray) -> np.ndarray:
        """Slice the first physical-drone rows from a virtual formation layout.

        Args:
            sim_layout: Sim-frame positions, shape ``(N, 3)``.

        Returns:
            First ``n_physical`` rows, shape ``(n_physical, 3)``.

        Raises:
            ValueError: If layout shape is invalid or too few rows.
        """
        pts = np.asarray(sim_layout, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"cmd_target must be (N,3), got {pts.shape}")
        if pts.shape[0] < self.n_physical:
            raise ValueError(
                f"cmd_target has {pts.shape[0]} points but {self.n_physical} physical drones need indices 0..{self.n_physical - 1}"
            )
        return pts[: self.n_physical]

    def _room_targets(self, sim_layout: np.ndarray) -> dict[str, list[float]]:
        """Map sim-frame layout to room-frame setpoints for each drone URI.

        Args:
            sim_layout: Sim-frame positions, shape ``(N, 3)``.

        Returns:
            Mapping ``{uri: [x, y, z, yaw]}`` with yaw fixed at ``0.0``, or empty
            when the swarm is not connected.
        """
        if self.swarm is None:
            return {}
        real = self.mapping.sim_to_real(self._physical_cmd(sim_layout))
        return {
            uri: [float(real[i, 0]), float(real[i, 1]), float(real[i, 2]), 0.0]
            for i, uri in enumerate(self._uris)
        }

    def get_sim_ground_layout(
        self,
        n_morph: int,
        *,
        plane_layout: np.ndarray,
        min_separation_m: float,
    ) -> np.ndarray:
        """Build sim ground layout: TOML homes for physical rows, plane morph for rest.

        Args:
            n_morph: Total virtual morph sample count.
            plane_layout: Plane morph XY layout at ground Z, shape ``(N, 3)``.
            min_separation_m: Unused; kept for caller API compatibility.

        Returns:
            Ground layout in sim meters, shape ``(n_morph, 3)``.

        Raises:
            ValueError: If ``plane_layout`` shape is invalid or too small.
        """
        del min_separation_m  # spacing comes from morph plane layout; arg kept for callers
        from functions.swarm_motion.prearm import plane_ground_layout

        n = int(n_morph)
        plane = np.asarray(plane_layout, dtype=np.float32)
        if plane.ndim != 2 or plane.shape[1] != 3:
            raise ValueError(f"plane_layout must be (N,3), got {plane.shape}")
        if plane.shape[0] < n:
            raise ValueError(f"plane_layout has {plane.shape[0]} rows but need {n}")

        homes = np.stack([self._homes[uri] for uri in self._uris], axis=0)
        sim_phys = self.mapping.real_to_sim(homes).astype(np.float32)
        z_ground = float(np.median(sim_phys[:, 2]))
        layout = plane_ground_layout(plane[:n], z_ground=z_ground)
        layout[: self.n_physical] = sim_phys[: self.n_physical]
        return layout

    def get_sim_track_positions(
        self, morph_fallback: np.ndarray, n_morph: int
    ) -> np.ndarray | None:
        """Fuse mocap poses with morph fallback for the full virtual formation.

        Args:
            morph_fallback: Sim-frame morph targets when mocap is unavailable,
                shape ``(N, 3)``.
            n_morph: Number of virtual rows to return.

        Returns:
            Sim-frame positions, shape ``(n_morph, 3)``, or ``None`` when mocap
            poses are missing for any active drone.

        Raises:
            ValueError: If ``morph_fallback`` shape is invalid or too small.
        """
        real_pos = self.get_positions_for_debug()
        if real_pos is None:
            return None
        fallback = np.asarray(morph_fallback, dtype=np.float32)
        if fallback.ndim != 2 or fallback.shape[1] != 3:
            raise ValueError(f"morph_fallback must be (N,3), got {fallback.shape}")
        n = int(n_morph)
        if fallback.shape[0] < n:
            raise ValueError(f"morph_fallback has {fallback.shape[0]} rows but need {n}")
        out = fallback[:n].copy()
        sim_phys = self.mapping.real_to_sim(real_pos).astype(np.float32)
        out[: self.n_physical] = sim_phys[: self.n_physical]
        return out

    def verify_near_sim_layout(self, sim_layout: np.ndarray) -> bool:
        """Check that active drones are near the mapped sim layout before arming.

        Args:
            sim_layout: Proposed arm layout in sim meters, shape ``(N, 3)``.

        Returns:
            ``True`` when every active drone is within ``max_pos_error_m`` of its
            mapped room pose (always ``True`` in dry-run mode).
        """
        if self._dry_run:
            return True
        real = self.mapping.sim_to_real(self._physical_cmd(sim_layout))
        ok = True
        for i, uri in enumerate(self._uris):
            if not self.swarm.is_active(uri):
                logger.warning("Drone %s inactive before arm", uri)
                ok = False
                continue
            try:
                obs = self.swarm.get_obs(uri)
                err = float(np.linalg.norm(obs["pos"] - real[i]))
            except Exception as exc:
                logger.warning("Could not read %s before arm: %s", uri, exc)
                ok = False
                continue
            if err > self.opts.max_pos_error_m:
                logger.warning(
                    "Drone %s far from arm layout: err=%.2fm (max %.2fm) obs=%s exp=%s",
                    uri,
                    err,
                    self.opts.max_pos_error_m,
                    np.round(obs["pos"], 3),
                    np.round(real[i], 3),
                )
                ok = False
        return ok

    def mocap_ok(self) -> bool:
        """Return whether every active drone has a fresh mocap pose.

        Returns:
            ``True`` when :meth:`get_positions_for_debug` succeeds for all URIs.
        """
        return self.get_positions_for_debug() is not None

    @property
    def control_halted(self) -> bool:
        """Return whether low-level setpoint streaming is paused.

        Returns:
            ``True`` during HL maneuvers, emergency stop, or before takeoff.
        """
        return bool(self._control_halted)

    def pause_setpoints_for_hl(self) -> None:
        """Stop the low-level setpoint stream before a high-level commander action."""
        self._control_halted = True
        self._pending_sim_layout = None

    def high_level_takeoff(self, height_m: float, *, duration_s: float = 3.0) -> None:
        """Run blocking high-level takeoff, then re-enable axswarm setpoints.

        Args:
            height_m: Target altitude gain in room meters.
            duration_s: HL commander maneuver duration in seconds.
        """
        if self._dry_run or self.swarm is None:
            self._control_halted = False
            self.physical_armed = True
            return
        self._control_halted = True
        self._pending_sim_layout = None
        h = float(height_m)
        dur = float(duration_s)
        print(
            f"Real swarm: high-level takeoff +{h:.2f}m ({dur:.1f}s, no setpoint stream)...",
            flush=True,
        )
        self.swarm.takeoff(height=h, duration=dur)
        self._control_halted = False
        self.physical_armed = True

    def high_level_descend(
        self,
        distance_m: float,
        *,
        duration_s: float = 3.0,
        settle_s: float | None = None,
    ) -> None:
        """Run blocking high-level in-place vertical descent at current XY.

        Args:
            distance_m: Descent distance in room meters (positive down).
            duration_s: HL ``goto`` maneuver duration in seconds.
            settle_s: Hover time at target before land; defaults to prearm constant.
        """
        if self._dry_run or self.swarm is None:
            self._control_halted = True
            return
        self._control_halted = True
        self._pending_sim_layout = None
        d = float(distance_m)
        dur = float(duration_s)
        hover_s = PREARM_PRE_LAND_HOVER_S if settle_s is None else float(settle_s)
        targets: dict[str, list[float]] = {}
        for uri in self._uris:
            if not self.swarm.is_active(uri):
                continue
            obs = self.swarm.get_obs(uri)
            pos = np.asarray(obs["pos"], dtype=np.float64)
            rpy = np.asarray(obs.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64)
            targets[uri] = [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]) - d,
                float(rpy[2]),
            ]
        if not targets:
            logger.warning("No active drones for high-level descend")
            return
        print(
            f"Real swarm: high-level descend −{d:.2f}m in place ({dur:.1f}s, no setpoint stream)...",
            flush=True,
        )
        self.swarm.goto(targets, duration=dur)
        if hover_s > 0.0:
            print(
                f"Real swarm: holding {hover_s:.1f}s at descend target before land...",
                flush=True,
            )
            time.sleep(hover_s)

    def high_level_land(self, height_m: float = 0.0, *, duration_s: float = 3.0) -> None:
        """Run blocking high-level land and keep setpoint stream halted afterward.

        Args:
            height_m: Target landing height in room meters.
            duration_s: HL land maneuver duration in seconds.
        """
        if self._dry_run or self.swarm is None:
            self._control_halted = True
            self.physical_armed = False
            return
        self._control_halted = True
        self._pending_sim_layout = None
        h = float(height_m)
        dur = float(duration_s)
        print(
            f"Real swarm: high-level land (z={h:.2f}m, {dur:.1f}s, no setpoint stream)...",
            flush=True,
        )
        self.swarm.land(height=h, duration=dur)
        self.physical_armed = False

    def halt_control(self) -> None:
        """Stop setpoint streaming and send Crazyflie emergency stop."""
        if self._control_halted:
            return
        self._control_halted = True
        self._pending_sim_layout = None
        if self._dry_run or self.swarm is None:
            return
        try:
            self.swarm.emergency_stop()
        except Exception as exc:
            logger.warning("Emergency stop failed: %s", exc)

    def send_sim_layout(self, sim_layout: np.ndarray, *, force: bool = False) -> bool:
        """Stream one throttled low-level position setpoint batch to the radio.

        Args:
            sim_layout: Sim-frame target layout, shape ``(N, 3)``.
            force: When ``True``, bypass the ``ctrl_freq`` throttle.

        Returns:
            ``True`` when a setpoint batch was sent; ``False`` when halted,
            throttled, or mocap is unavailable.
        """
        if self._control_halted:
            return False
        if self._dry_run or self.swarm is None:
            return False
        self._pending_sim_layout = np.asarray(sim_layout, dtype=np.float32)
        if not self.mocap_ok():
            if not self._mocap_hold_logged:
                logger.warning(
                    "Mocap unavailable — pausing setpoint stream until poses return."
                )
                self._mocap_hold_logged = True
            return False
        self._mocap_hold_logged = False

        now = time.monotonic()
        if not force and now < self._setpoint_next_mono:
            return False

        targets = self._room_targets(self._pending_sim_layout)
        if not targets:
            return False
        self.swarm.setpoint(targets)
        self.physical_armed = True
        self._setpoint_next_mono = now + self._setpoint_period_s
        return True

    def track_frame(
        self,
        cmd_target: np.ndarray,
        *,
        gesture_enabled: bool,
        just_armed: bool,
        morph_mode: int,
        led_every_n: int,
        frame_idx: int,
        prearm_phase: str = "ground",
        prearm_vertical_leg: str = "climb",
        just_prearm_phase: bool = False,
        prearm_vertical_layout: np.ndarray | None = None,
    ) -> None:
        """Apply one frame of real-swarm tracking, LEDs, and prearm phase logging.

        Args:
            cmd_target: Filtered axswarm command layout in sim meters.
            gesture_enabled: Whether morph gesture control is armed.
            just_armed: ``True`` on the frame gesture control was just enabled.
            morph_mode: Active morph mode for mode LED coloring.
            led_every_n: Apply mode LED every N frames (0 disables).
            frame_idx: Monotonic outer-loop frame counter.
            prearm_phase: Current prearm phase (``ground``, ``vertical``, etc.).
            prearm_vertical_leg: Vertical leg name (``climb`` or return leg).
            just_prearm_phase: ``True`` when prearm phase changed this frame.
            prearm_vertical_layout: Takeoff/hover layout for phase logging.
        """
        if self._control_halted:
            return
        cmd = np.asarray(cmd_target, dtype=np.float32)
        phase = str(prearm_phase)

        if just_prearm_phase:
            if phase == "vertical" and prearm_vertical_layout is not None:
                z_takeoff = float(
                    np.median(prearm_vertical_layout[: self.n_physical, 2])
                )
                if str(prearm_vertical_leg) == "climb":
                    print(
                        f"Real vertical takeoff → z={z_takeoff:.2f}m "
                        "(axswarm-planned stream, same as sim)."
                    )
                else:
                    print(
                        f"Real direct 3D return to vertical layout z≈{z_takeoff:.2f}m "
                        "(axswarm-planned stream)."
                    )
            elif phase == "formation":
                print(
                    "Real hover formation: axswarm setpoint stream "
                    "(press 1 for in-place HL descend + land)."
                )
            elif phase == "ground":
                z_from = (
                    float(np.median(prearm_vertical_layout[: self.n_physical, 2]))
                    if prearm_vertical_layout is not None
                    else float(np.median(cmd[: self.n_physical, 2]))
                )
                print(
                    f"Real axswarm-planned descent to ground (from z≈{z_from:.2f}m)."
                )

        if gesture_enabled and just_armed:
            if not self.verify_near_sim_layout(cmd):
                print(
                    "[WARN] Real swarm position check failed; streaming arm layout anyway. "
                    "Move drones near mapped hover poses or adjust config/frame.origin."
                )

        self.send_sim_layout(cmd)

        if led_every_n > 0 and (frame_idx % led_every_n) == 0:
            mode = int(morph_mode)
            if mode != self._last_mode:
                self._apply_mode_led(mode)
                self._last_mode = mode

    def _apply_mode_led(self, mode: int) -> None:
        if self.swarm is None:
            return
        color = _MODE_LED_COLORS.get(int(mode), np.zeros(4))
        top = {uri: color for uri in self._uris if self.swarm.is_active(uri)}
        if top:
            try:
                self.swarm.apply_colors(top, top)
            except Exception as exc:
                logger.warning("LED update failed: %s", exc)

    def close(self) -> None:
        """Halt control and release Crazyflie swarm resources."""
        if self._closed:
            return
        self._closed = True
        self.halt_control()
        if self.swarm is not None:
            self.swarm.close()

    def get_positions_for_debug(self) -> np.ndarray | None:
        """Read current mocap positions for all configured drone URIs.

        Returns:
            Room-frame positions, shape ``(n_physical, 3)``, or ``None`` when the
            swarm is disconnected or any active URI lacks a pose.
        """
        if self.swarm is None:
            return None
        rows = []
        for uri in self._uris:
            if not self.swarm.is_active(uri):
                return None
            try:
                rows.append(np.asarray(self.swarm.get_obs(uri)["pos"], dtype=np.float64))
            except Exception:
                return None
        return np.stack(rows, axis=0) if rows else None
