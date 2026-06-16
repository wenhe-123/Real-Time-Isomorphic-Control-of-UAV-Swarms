"""Axis-aligned workspace box for left-hand swarm rigid motion.

When enabled, a cube centered at press-0 limits **all** commanded targets (morph/open
formation and left-hand rigid motion). **freeze**: latch until the formation can move
without leaving the box. **clip** (default): scale rigid motion to fit instead of
freezing. Use :meth:`clamp_targets` so open-hand morph (α→1) does not outgrow the cube.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from functions.swarm_motion.left_hand_swarm_pose import R_to_rotvec, apply_rigid_to_targets, rotvec_to_R


def _as_xyz(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return p[:, :3]


@dataclass
class SwarmWorkspaceBox:
    """Cube workspace guard for rigid left-swarm targets."""

    size_m: float = 3.5
    wall_margin_m: float = 0.03
    #: Smaller margin to resume motion after a freeze (hysteresis vs ``wall_margin_m``).
    clear_margin_m: float = 0.015
    floor_z: float = 0.05
    #: ``clip``: scale motion to fit; ``freeze``: latch until all drones are clear.
    mode: str = "clip"
    armed: bool = False
    blocked: bool = False
    center: np.ndarray | None = None
    frozen_off: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    frozen_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    _prev_blocked: bool = False
    _prev_clipped: bool = False

    @property
    def enabled(self) -> bool:
        return float(self.size_m) > 0.0

    @property
    def clip_mode(self) -> bool:
        return str(self.mode).strip().lower() in ("clip", "scale", "partial")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.center is None:
            c = np.zeros(3, dtype=np.float64)
        else:
            c = np.asarray(self.center, dtype=np.float64).reshape(3)
        half = 0.5 * float(self.size_m)
        return c - half, c + half

    def arm(
        self,
        morph_targets: np.ndarray,
        *,
        sim_xyz: np.ndarray | None = None,
        fit_contains: bool = False,
    ) -> None:
        """Record press-0 reference; center on morph (+ optional sim).

        When ``fit_contains`` is True the cube edge may grow to wrap all points; default
        keeps the configured ``size_m`` (no auto-expand to 6–7 m when the formation is large).
        """
        pts = _as_xyz(morph_targets)
        if sim_xyz is not None:
            sim_pts = _as_xyz(sim_xyz)
            if sim_pts.shape[0] > 0:
                pts = np.vstack([pts, sim_pts]) if pts.shape[0] > 0 else sim_pts
        if pts.shape[0] == 0:
            self.disarm()
            return
        self.center = np.mean(pts, axis=0)
        if fit_contains:
            c = np.asarray(self.center, dtype=np.float64).reshape(3)
            extent = float(np.max(np.linalg.norm(pts - c.reshape(1, 3), axis=1)))
            need_half = extent + float(self.wall_margin_m) + 0.08
            min_edge = 2.0 * need_half
            if min_edge > float(self.size_m):
                self.size_m = float(min_edge)
        half = 0.5 * float(self.size_m)
        floor = float(max(self.floor_z, 0.0))
        min_center_z = floor + half
        if float(self.center[2]) < min_center_z:
            self.center = np.asarray(self.center, dtype=np.float64).copy()
            self.center[2] = min_center_z
        self.armed = True
        self.blocked = False
        self._prev_blocked = False
        self._prev_clipped = False
        self.frozen_off = np.zeros(3, dtype=np.float64)
        self.frozen_R = np.eye(3, dtype=np.float64)

    def clamp_targets(self, targets: np.ndarray) -> np.ndarray:
        """Clamp every drone XYZ inside the workspace (with wall margin)."""
        if not self.enabled or not self.armed or self.center is None:
            return np.asarray(targets, dtype=np.float64)
        pts = np.asarray(targets, dtype=np.float64).copy()
        if pts.ndim != 2 or pts.shape[1] < 3:
            return pts
        lo, hi = self.bounds()
        m = float(max(self.wall_margin_m, 0.0))
        lo_safe = lo + m
        hi_safe = hi - m
        pts[:, :3] = np.clip(pts[:, :3], lo_safe, hi_safe)
        return pts

    def centroid_inside(
        self,
        points: np.ndarray,
        *,
        margin_m: float | None = None,
    ) -> bool:
        """True if the formation centroid lies inside the box with margin."""
        if self.center is None:
            return True
        pts = _as_xyz(points)
        if pts.shape[0] == 0:
            return True
        c = np.mean(pts, axis=0)
        lo, hi = self.bounds()
        m = float(margin_m if margin_m is not None else self.clear_margin_m)
        m = max(m, 0.0)
        return bool(np.all(c > lo + m) and np.all(c < hi - m))

    def can_unblock(self, proposed: np.ndarray) -> bool:
        """Relaxed unblock: all inside walls, or none outside and centroid is clear."""
        if self.all_clear(proposed):
            return True
        if self.any_outside(proposed):
            return False
        return self.centroid_inside(proposed)

    def disarm(self) -> None:
        self.armed = False
        self.blocked = False
        self._prev_blocked = False
        self._prev_clipped = False
        self.center = None
        self.frozen_off = np.zeros(3, dtype=np.float64)
        self.frozen_R = np.eye(3, dtype=np.float64)

    def any_outside(self, points: np.ndarray) -> bool:
        pts = _as_xyz(points)
        if pts.shape[0] == 0 or self.center is None:
            return False
        lo, hi = self.bounds()
        return bool(np.any((pts < lo) | (pts > hi)))

    def any_at_wall(self, points: np.ndarray, *, margin_m: float | None = None) -> bool:
        pts = _as_xyz(points)
        if pts.shape[0] == 0 or self.center is None:
            return False
        lo, hi = self.bounds()
        m = float(max(margin_m if margin_m is not None else self.wall_margin_m, 0.0))
        at_lo = pts <= (lo + m)
        at_hi = pts >= (hi - m)
        return bool(np.any(at_lo | at_hi))

    def all_clear(self, points: np.ndarray) -> bool:
        pts = _as_xyz(points)
        if pts.shape[0] == 0 or self.center is None:
            return True
        lo, hi = self.bounds()
        m = float(max(self.clear_margin_m, 0.0))
        inside = (pts > (lo + m)) & (pts < (hi - m))
        return bool(np.all(inside))

    def _targets_ok(self, points: np.ndarray) -> bool:
        return not self.any_outside(points) and not self.any_at_wall(points)

    def _violation_score(self, points: np.ndarray) -> float:
        """How far targets penetrate the safe box; 0 means fully usable."""
        pts = _as_xyz(points)
        if pts.shape[0] == 0 or self.center is None:
            return 0.0
        lo, hi = self.bounds()
        m = float(max(self.wall_margin_m, 0.0))
        lo_safe = lo + m
        hi_safe = hi - m
        below = np.maximum(lo_safe.reshape(1, 3) - pts, 0.0)
        above = np.maximum(pts - hi_safe.reshape(1, 3), 0.0)
        return float(np.max(below + above))

    @staticmethod
    def _blend_rigid(
        off0: np.ndarray,
        R0: np.ndarray,
        off1: np.ndarray,
        R1: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        a = float(np.clip(alpha, 0.0, 1.0))
        off0 = np.asarray(off0, dtype=np.float64).reshape(3)
        off1 = np.asarray(off1, dtype=np.float64).reshape(3)
        R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
        R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
        off = off0 + a * (off1 - off0)
        rv0 = R_to_rotvec(R0)
        rv1 = R_to_rotvec(R1)
        rv = rv0 + a * (rv1 - rv0)
        return off, rotvec_to_R(rv)

    def _clip_rigid_motion(
        self,
        morph_targets: np.ndarray,
        off0: np.ndarray,
        R0: np.ndarray,
        off1: np.ndarray,
        R1: np.ndarray,
        *,
        ref_drone_xyz: np.ndarray | None,
        pivot: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return largest in-box fraction of motion from (off0,R0) toward (off1,R1)."""
        proposed = apply_rigid_to_targets(
            morph_targets,
            off1,
            R1,
            ref_drone_xyz=ref_drone_xyz,
            pivot=pivot,
        )
        if self._targets_ok(proposed):
            return (
                np.asarray(off1, dtype=np.float64).reshape(3).copy(),
                np.asarray(R1, dtype=np.float64).reshape(3, 3).copy(),
                proposed,
            )

        lo, hi = 0.0, 1.0
        best_off = np.asarray(off0, dtype=np.float64).reshape(3).copy()
        best_R = np.asarray(R0, dtype=np.float64).reshape(3, 3).copy()
        best_raw = apply_rigid_to_targets(
            morph_targets,
            best_off,
            best_R,
            ref_drone_xyz=ref_drone_xyz,
            pivot=pivot,
        )
        found_ok = self._targets_ok(best_raw)
        best_bad_off = best_off.copy()
        best_bad_R = best_R.copy()
        best_bad_raw = best_raw
        best_bad_score = self._violation_score(best_raw)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            off_m, R_m = self._blend_rigid(off0, R0, off1, R1, mid)
            t = apply_rigid_to_targets(
                morph_targets,
                off_m,
                R_m,
                ref_drone_xyz=ref_drone_xyz,
                pivot=pivot,
            )
            if self._targets_ok(t):
                lo = mid
                best_off, best_R, best_raw = off_m, R_m, t
                found_ok = True
            else:
                score = self._violation_score(t)
                if score < best_bad_score:
                    best_bad_off, best_bad_R, best_bad_raw = off_m, R_m, t
                    best_bad_score = score
                hi = mid
        if not found_ok and best_bad_score < self._violation_score(best_raw):
            return best_bad_off, best_bad_R, best_bad_raw
        return best_off, best_R, best_raw

    def guard_rigid_motion(
        self,
        morph_targets: np.ndarray,
        proposed_off: np.ndarray,
        proposed_R: np.ndarray,
        *,
        ref_drone_xyz: np.ndarray | None,
        pivot: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
        """Return accepted (off, R, raw_target, blocked, status_message)."""
        if not self.enabled or not self.armed:
            raw = apply_rigid_to_targets(
                morph_targets,
                proposed_off,
                proposed_R,
                ref_drone_xyz=ref_drone_xyz,
                pivot=pivot,
            )
            return (
                np.asarray(proposed_off, dtype=np.float64).reshape(3),
                np.asarray(proposed_R, dtype=np.float64).reshape(3, 3),
                raw,
                False,
                "",
            )

        off0 = np.asarray(self.frozen_off, dtype=np.float64).reshape(3)
        R0 = np.asarray(self.frozen_R, dtype=np.float64).reshape(3, 3)
        off1 = np.asarray(proposed_off, dtype=np.float64).reshape(3)
        R1 = np.asarray(proposed_R, dtype=np.float64).reshape(3, 3)

        if self.clip_mode:
            off_out, R_out, raw_out = self._clip_rigid_motion(
                morph_targets,
                off0,
                R0,
                off1,
                R1,
                ref_drone_xyz=ref_drone_xyz,
                pivot=pivot,
            )
            self.frozen_off = np.asarray(off_out, dtype=np.float64).reshape(3).copy()
            self.frozen_R = np.asarray(R_out, dtype=np.float64).reshape(3, 3).copy()
            self.blocked = False
            clipped = float(np.linalg.norm(off_out - off1)) > 1e-6 or not np.allclose(
                R_out, R1, atol=1e-6, rtol=1e-5
            )
            status = ""
            if clipped and not self._prev_clipped:
                status = "workspace clipped (partial motion toward wall)"
            self._prev_clipped = clipped
            return off_out, R_out, raw_out, False, status

        proposed = apply_rigid_to_targets(
            morph_targets,
            proposed_off,
            proposed_R,
            ref_drone_xyz=ref_drone_xyz,
            pivot=pivot,
        )
        frozen_targets = apply_rigid_to_targets(
            morph_targets,
            self.frozen_off,
            self.frozen_R,
            ref_drone_xyz=ref_drone_xyz,
            pivot=pivot,
        )

        status = ""
        if self.blocked:
            if self.can_unblock(proposed):
                self.blocked = False
                self.frozen_off = off1.copy()
                self.frozen_R = R1.copy()
                off_out = self.frozen_off
                R_out = self.frozen_R
                raw_out = proposed
                status = "workspace unblocked"
            else:
                off_out = self.frozen_off
                R_out = self.frozen_R
                raw_out = frozen_targets
                status = "workspace blocked (move all drones away from walls)"
        else:
            hit_wall = self.any_at_wall(proposed) or self.any_outside(proposed)
            if hit_wall:
                self.blocked = True
                off_out = self.frozen_off
                R_out = self.frozen_R
                raw_out = frozen_targets
                status = "workspace wall hit — motion frozen"
            else:
                self.frozen_off = off1.copy()
                self.frozen_R = R1.copy()
                off_out = self.frozen_off
                R_out = self.frozen_R
                raw_out = proposed

        edge_msg = ""
        if self.blocked != self._prev_blocked:
            edge_msg = status
            self._prev_blocked = self.blocked
        elif status and self.blocked:
            edge_msg = ""

        return off_out, R_out, raw_out, bool(self.blocked), edge_msg

    def format_bounds(self) -> str:
        if self.center is None:
            return "box=off"
        lo, hi = self.bounds()
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        return (
            f"box={float(self.size_m):.2f}m mode={self.mode!r} center=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) "
            f"lo=({lo[0]:+.2f},{lo[1]:+.2f},{lo[2]:+.2f}) "
            f"hi=({hi[0]:+.2f},{hi[1]:+.2f},{hi[2]:+.2f}) "
            f"floor_z={float(self.floor_z):.2f}"
        )

    def corner_points(self) -> np.ndarray | None:
        """Eight corners of the axis-aligned workspace cube (sim/world frame)."""
        if self.center is None:
            return None
        lo, hi = self.bounds()
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        return np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float64,
        )


# Bottom loop, top loop, then verticals (MuJoCo draw_line connects consecutive points only).
_BOX_WIREFRAME_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def draw_swarm_workspace_box_in_sim(
    sim,
    box: SwarmWorkspaceBox,
    *,
    line_size: float = 4.0,
) -> None:
    """Draw a red wireframe cube in Crazyflow for the armed workspace bounds.

    Must be called every frame immediately before ``render_targets(sim, ...)``.
    """
    if not box.enabled or not box.armed:
        return
    corners = box.corner_points()
    if corners is None:
        return
    from crazyflow.sim.visualize import draw_line

    rgba = (
        np.array([1.0, 0.25, 0.25, 1.0], dtype=np.float64)
        if box.blocked
        else np.array([1.0, 0.0, 0.0, 0.92], dtype=np.float64)
    )
    sz = float(max(line_size, 0.5))
    for i, j in _BOX_WIREFRAME_EDGES:
        draw_line(
            sim,
            corners[[i, j]],
            rgba=rgba,
            start_size=sz,
            end_size=sz,
        )
