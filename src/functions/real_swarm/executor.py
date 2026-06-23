"""Bridge filtered ``cmd_target`` arrays to physical Crazyflie setpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from swarm_gpt.core.drone_swarm import DroneSwarm

from functions.real_swarm.swarm_config import RealFrameMapping, RealSwarmOptions, load_drones_config

logger = logging.getLogger(__name__)

_MODE_LED_COLORS = {
    1: np.array([255, 40, 40, 0], dtype=np.float64),
    2: np.array([255, 200, 40, 0], dtype=np.float64),
    3: np.array([40, 200, 255, 0], dtype=np.float64),
    4: np.array([180, 80, 255, 0], dtype=np.float64),
    5: np.array([80, 255, 120, 0], dtype=np.float64),
}


def _ensure_rclpy() -> None:
    """ROSConnector (mocap) requires ``rclpy.ok()`` in this process."""
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
        self._last_sim_phys_pos: np.ndarray | None = None
        self._last_sim_phys_time: float | None = None
        self._last_sim_phys_vel: np.ndarray | None = None
        self.swarm: DroneSwarm | None = None

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

    @property
    def ctrl_freq(self) -> float:
        return float(self.opts.ctrl_freq)

    def _physical_cmd(self, sim_layout: np.ndarray) -> np.ndarray:
        """First ``n_physical`` rows of the virtual formation → real drones."""
        pts = np.asarray(sim_layout, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"cmd_target must be (N,3), got {pts.shape}")
        if pts.shape[0] < self.n_physical:
            raise ValueError(
                f"cmd_target has {pts.shape[0]} points but {self.n_physical} physical drones need indices 0..{self.n_physical - 1}"
            )
        return pts[: self.n_physical]

    def _room_targets(self, sim_layout: np.ndarray) -> dict[str, list[float]]:
        if self.swarm is None:
            return {}
        real = self.mapping.sim_to_real(self._physical_cmd(sim_layout))
        return {
            uri: [float(real[i, 0]), float(real[i, 1]), float(real[i, 2]), 0.0]
            for i, uri in enumerate(self._uris)
            if self.swarm.is_active(uri)
        }

    def get_sim_ground_layout(self, n_morph: int, *, min_separation_m: float) -> np.ndarray:
        """TOML ``home`` poses → sim frame for physical drones; chessboard fill for virtual rows."""
        from functions.swarm_motion.prearm import sim_chessboard_ground_layout

        n = int(n_morph)
        homes = np.stack([self._homes[uri] for uri in self._uris], axis=0)
        sim_phys = self.mapping.real_to_sim(homes).astype(np.float32)
        z_ground = float(np.median(sim_phys[:, 2]))
        xy_half = max(0.5, float(np.max(np.abs(sim_phys[:, :2]))) + 0.6)
        layout = sim_chessboard_ground_layout(
            n,
            min_separation_m=float(min_separation_m),
            z_ground=z_ground,
            xy_half_extent_m=xy_half,
        )
        layout[: self.n_physical] = sim_phys
        return layout

    def _validate_morph_fallback(self, morph_fallback: np.ndarray, n_morph: int) -> np.ndarray:
        fallback = np.asarray(morph_fallback, dtype=np.float32)
        if fallback.ndim != 2 or fallback.shape[1] != 3:
            raise ValueError(f"morph_fallback must be (N,3), got {fallback.shape}")
        n = int(n_morph)
        if fallback.shape[0] < n:
            raise ValueError(f"morph_fallback has {fallback.shape[0]} rows but need {n}")
        return fallback[:n].copy()

    def _update_sim_velocity_cache(self, sim_phys: np.ndarray, now_s: float) -> np.ndarray:
        sim_phys = np.asarray(sim_phys, dtype=np.float32)
        if self._last_sim_phys_pos is None or self._last_sim_phys_time is None:
            vel = np.zeros_like(sim_phys, dtype=np.float32)
        else:
            dt = float(now_s - self._last_sim_phys_time)
            if dt <= 1e-3 or dt > 1.0:
                vel = np.zeros_like(sim_phys, dtype=np.float32)
            else:
                vel = (sim_phys - self._last_sim_phys_pos) / dt
                if self._last_sim_phys_vel is not None:
                    vel = 0.35 * vel + 0.65 * self._last_sim_phys_vel
                if not np.all(np.isfinite(vel)):
                    vel = np.zeros_like(sim_phys, dtype=np.float32)
        self._last_sim_phys_pos = sim_phys.copy()
        self._last_sim_phys_time = float(now_s)
        self._last_sim_phys_vel = vel.astype(np.float32, copy=True)
        return self._last_sim_phys_vel

    def get_sim_track_state(
        self, morph_fallback: np.ndarray, n_morph: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Mocap poses → sim-frame position/velocity rows for axswarm."""
        real_pos = self.get_positions_for_debug()
        if real_pos is None:
            return None
        pos = self._validate_morph_fallback(morph_fallback, n_morph)
        vel = np.zeros_like(pos, dtype=np.float32)
        sim_phys = self.mapping.real_to_sim(real_pos).astype(np.float32)
        sim_phys_vel = self._update_sim_velocity_cache(sim_phys, time.monotonic())
        pos[: self.n_physical] = sim_phys[: self.n_physical]
        vel[: self.n_physical] = sim_phys_vel[: self.n_physical]
        return pos, vel

    def get_sim_track_positions(
        self, morph_fallback: np.ndarray, n_morph: int
    ) -> np.ndarray | None:
        """Mocap poses (ROS TF) → sim frame for physical rows; morph fallback for virtual."""
        state = self.get_sim_track_state(morph_fallback, n_morph)
        return None if state is None else state[0]

    def verify_near_sim_layout(self, sim_layout: np.ndarray) -> bool:
        """Check drones are close to mapped sim layout before arming."""
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
        """True when every active drone has a fresh mocap pose."""
        return self.get_positions_for_debug() is not None

    def send_sim_layout(self, sim_layout: np.ndarray) -> None:
        if self._dry_run:
            return
        if not self.mocap_ok():
            if not self._mocap_hold_logged:
                logger.warning(
                    "Mocap unavailable — pausing setpoint stream until poses return."
                )
                self._mocap_hold_logged = True
            return
        self._mocap_hold_logged = False
        targets = self._room_targets(sim_layout)
        if targets:
            self.swarm.setpoint(targets)
            self.physical_armed = True

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
                        "(axswarm-planned stream). Press 1 for ground."
                    )
            elif phase == "formation":
                print(
                    "Real hover formation: direct 3D move to hover layout "
                    "(axswarm-planned stream; press 1 to shrink to vertical)."
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
        if self.swarm is not None:
            self.swarm.close()

    def get_positions_for_debug(self) -> np.ndarray | None:
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
