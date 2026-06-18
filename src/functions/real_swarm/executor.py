"""Bridge filtered ``cmd_target`` arrays to physical Crazyflie setpoints."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from functions.real_swarm.drone_swarm import DroneSwarm
from functions.real_swarm.swarm_config import RealFrameMapping, RealSwarmOptions, load_drones_config
from functions.swarm_motion.spacing_guard import enforce_min_separation_xy

logger = logging.getLogger(__name__)

_MODE_LED_COLORS = {
    1: np.array([255, 40, 40, 0], dtype=np.float64),
    2: np.array([255, 200, 40, 0], dtype=np.float64),
    3: np.array([40, 200, 255, 0], dtype=np.float64),
    4: np.array([180, 80, 255, 0], dtype=np.float64),
    5: np.array([80, 255, 120, 0], dtype=np.float64),
}


class RealSwarmExecutor:
    """Send online ``cmd_target`` poses to a connected Crazyflie swarm."""

    def __init__(
        self,
        *,
        config_path: Path,
        morph_point_count: int | None = None,
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
        self.morph_point_count = int(morph_point_count) if morph_point_count is not None else None
        self._drone_ids = list(drones.keys())
        self._uris = [drones[k]["uri"] for k in self._drone_ids]
        self._homes = {drones[k]["uri"]: np.asarray(drones[k]["pos"], dtype=np.float64) for k in self._drone_ids}
        self.physical_armed = False
        self._last_cmd: np.ndarray | None = None
        self._last_mode = 1

        morph_note = (
            f"morph={self.morph_point_count} virtual targets"
            if self.morph_point_count is not None
            else "morph virtual targets"
        )
        print(
            f"Connecting {n_physical} Crazyflie(s) ({morph_note}; "
            f"physical indices 0..{n_physical - 1}) mocap/ROS ..."
        )
        print(
            f"  Sim (0,0,0) → room {np.round(mapping.origin, 3)} m "
            f"(scale={mapping.scale}, yaw={np.rad2deg(mapping.yaw_rad):.1f}°)"
        )
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
        out = enforce_min_separation_xy(
            layout, float(min_separation_m), z_ground, iters=10
        )
        out[: self.n_physical] = sim_phys
        return out

    def get_sim_track_positions(
        self, morph_fallback: np.ndarray, n_morph: int
    ) -> np.ndarray | None:
        """Lighthouse poses → sim frame for physical rows; morph fallback for virtual."""
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
        """Check drones are close to mapped sim layout before arming."""
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

    def goto_sim_layout(self, sim_layout: np.ndarray, *, duration: float | None = None) -> None:
        real = self.mapping.sim_to_real(self._physical_cmd(sim_layout))
        targets = {
            uri: [float(real[i, 0]), float(real[i, 1]), float(real[i, 2]), 0.0]
            for i, uri in enumerate(self._uris)
            if self.swarm.is_active(uri)
        }
        if not targets:
            raise RuntimeError("No active drones for goto")
        dur = float(self.opts.arm_goto_s if duration is None else duration)
        print(f"Real swarm goto arm layout ({dur:.1f}s) ...")
        self.swarm.goto(targets, duration=dur)
        self.physical_armed = True

    def send_sim_layout(self, sim_layout: np.ndarray) -> None:
        real = self.mapping.sim_to_real(self._physical_cmd(sim_layout))
        targets = {
            uri: [float(real[i, 0]), float(real[i, 1]), float(real[i, 2]), 0.0]
            for i, uri in enumerate(self._uris)
            if self.swarm.is_active(uri)
        }
        if targets:
            self.swarm.send_setpoint_tick(targets)

    def track_frame(
        self,
        cmd_target: np.ndarray,
        *,
        gesture_enabled: bool,
        just_armed: bool,
        morph_mode: int,
        led_every_n: int,
        frame_idx: int,
        prearm_climb_enabled: bool = False,
    ) -> None:
        cmd = np.asarray(cmd_target, dtype=np.float32)
        self._last_cmd = cmd.copy()

        if gesture_enabled:
            if just_armed and not self.physical_armed:
                if not self.verify_near_sim_layout(cmd):
                    print(
                        "[WARN] Real swarm position check failed; still flying to arm layout. "
                        "Move drones near mapped hover poses or adjust config/frame.origin."
                    )
                self.goto_sim_layout(cmd)
            elif self.physical_armed:
                self.send_sim_layout(cmd)
        elif prearm_climb_enabled or self.physical_armed:
            if not self.physical_armed:
                print(
                    "Real layout stream: axswarm-filtered setpoints "
                    "(1 = ground ↔ hover, SPACE = gestures)."
                )
                self.physical_armed = True
            self.send_sim_layout(cmd)

        if led_every_n > 0 and (frame_idx % led_every_n) == 0:
            mode = int(morph_mode)
            if mode != self._last_mode:
                self._apply_mode_led(mode)
                self._last_mode = mode

    def _apply_mode_led(self, mode: int) -> None:
        color = _MODE_LED_COLORS.get(int(mode), np.zeros(4))
        top = {uri: color for uri in self._uris if self.swarm.is_active(uri)}
        if top:
            try:
                self.swarm.apply_colors(top, top)
            except Exception as exc:
                logger.warning("LED update failed: %s", exc)

    def land_and_close(self) -> None:
        if self.opts.land_on_exit and self.physical_armed:
            landing = {
                uri: [
                    float(self._homes[uri][0]),
                    float(self._homes[uri][1]),
                    float(max(self._homes[uri][2], 0.05)),
                    0.0,
                ]
                for uri in self._uris
                if self.swarm.is_active(uri)
            }
            if landing:
                print("Real swarm landing ...")
                try:
                    self.swarm.goto(landing, duration=2.5)
                except Exception as exc:
                    logger.warning("Landing goto failed: %s", exc)
        print("Closing real swarm connections ...")
        self.swarm.close()

    def close(self) -> None:
        self.land_and_close()

    def get_positions_for_debug(self) -> np.ndarray | None:
        rows = []
        for uri in self._uris:
            if not self.swarm.is_active(uri):
                return None
            try:
                rows.append(np.asarray(self.swarm.get_obs(uri)["pos"], dtype=np.float64))
            except Exception:
                return None
        return np.stack(rows, axis=0) if rows else None
