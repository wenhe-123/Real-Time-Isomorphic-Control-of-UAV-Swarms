"""Record and replay hand vs swarm rigid pose trajectories after press-0."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from functions.swarm_motion.left_hand_swarm_pose import R_to_quat


def _as_flist(v, n: int) -> list[float]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size < n:
        raise ValueError(f"expected >= {n} values, got {a.size}")
    return [float(x) for x in a[:n]]


def _centroid_m(targets: np.ndarray) -> np.ndarray:
    t = np.asarray(targets, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    return np.mean(t[:, :3], axis=0)


def _quat_wxyz(R: np.ndarray) -> list[float]:
    q = R_to_quat(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return _as_flist(q, 4)


def _rot_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = float(np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(float(np.arccos(c))))


@dataclass
class RigidPoseTraceRecorder:
    """Capture hand command pose and swarm target pose while left-swarm is active."""

    out_path: Path
    sample_every: int = 1
    meta: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    _active: dict[str, Any] | None = field(default=None, repr=False)
    _last_saved_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.out_path = Path(self.out_path)
        self.sample_every = max(1, int(self.sample_every))

    @property
    def is_active(self) -> bool:
        return self._active is not None

    def on_arm(
        self,
        *,
        frame_idx: int,
        t_s: float,
        ref_wrist_mm: np.ndarray,
        ref_palm_mm: np.ndarray,
        ref_centroid_m: np.ndarray,
        ref_swarm_targets: np.ndarray | None = None,
    ) -> None:
        if self._active is not None:
            self._finish_active(frame_idx=frame_idx, t_s=t_s, reason="rearm")
        seg: dict[str, Any] = {
            "arm_frame": int(frame_idx),
            "arm_t_s": float(t_s),
            "end_frame": None,
            "end_t_s": None,
            "end_reason": None,
            "ref_wrist_mm": _as_flist(ref_wrist_mm, 3),
            "ref_palm_mm": _as_flist(ref_palm_mm, 3),
            "ref_centroid_m": _as_flist(ref_centroid_m, 3),
            "frames": [],
        }
        if ref_swarm_targets is not None:
            rs = np.asarray(ref_swarm_targets, dtype=np.float64)
            if rs.ndim == 2 and rs.shape[1] >= 3:
                seg["ref_swarm_centroid_m"] = _as_flist(_centroid_m(rs), 3)
        self._active = seg
        print(
            f"[rigid-pose-trace] segment start frame={frame_idx} t={t_s:.3f}s "
            f"ref_centroid=({seg['ref_centroid_m'][0]:+.3f},"
            f"{seg['ref_centroid_m'][1]:+.3f},{seg['ref_centroid_m'][2]:+.3f})",
            flush=True,
        )

    def sample(
        self,
        *,
        frame_idx: int,
        t_s: float,
        hand_off_m: np.ndarray | None,
        hand_R: np.ndarray | None,
        raw_target: np.ndarray,
        cmd_target: np.ndarray,
        left_pose_state: Any,
        phase: str = "armed",
    ) -> None:
        if self._active is None:
            return
        if int(frame_idx) % self.sample_every != 0:
            return

        off = (
            np.zeros(3, dtype=np.float64)
            if hand_off_m is None
            else np.asarray(hand_off_m, dtype=np.float64).reshape(3)
        )
        R = (
            np.eye(3, dtype=np.float64)
            if hand_R is None
            else np.asarray(hand_R, dtype=np.float64).reshape(3, 3)
        )
        raw_c = _centroid_m(raw_target)
        cmd_c = _centroid_m(cmd_target)
        ref_c = np.asarray(self._active["ref_centroid_m"], dtype=np.float64).reshape(3)
        swarm_T = cmd_c - ref_c

        st = left_pose_state
        palm = np.asarray(
            getattr(st, "last_palm_center_mm", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        wrist = np.asarray(
            getattr(st, "last_wrist_mm", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        ref_wrist = np.asarray(
            self._active.get("ref_wrist_mm", self._active.get("ref_palm_mm", np.zeros(3))),
            dtype=np.float64,
        ).reshape(3)
        ref_palm = np.asarray(self._active["ref_palm_mm"], dtype=np.float64).reshape(3)
        tgt_T = np.asarray(
            getattr(st, "last_delta_h_world", off),
            dtype=np.float64,
        ).reshape(3)
        raw_T = np.asarray(
            getattr(st, "last_delta_h_raw_m", tgt_T),
            dtype=np.float64,
        ).reshape(3)
        ema_T = np.asarray(getattr(st, "ema_offset", off), dtype=np.float64).reshape(3)
        applied_T = np.asarray(off, dtype=np.float64).reshape(3)

        self._active["frames"].append(
            {
                "frame": int(frame_idx),
                "t_s": float(t_s),
                "phase": str(phase),
                "hand": {
                    "T_m": _as_flist(ema_T, 3),
                    "applied_T_m": _as_flist(applied_T, 3),
                    "tgt_T_m": _as_flist(tgt_T, 3),
                    "raw_T_m": _as_flist(raw_T, 3),
                    "quat_wxyz": _quat_wxyz(R),
                    "rot_deg": _rot_angle_deg(R),
                    "palm_mm": _as_flist(palm, 3),
                    "wrist_mm": _as_flist(wrist, 3),
                    "wrist_delta_arm_mm": _as_flist(wrist - ref_wrist, 3),
                    "palm_delta_arm_mm": _as_flist(palm - ref_palm, 3),
                    "hold": bool(getattr(st, "last_pose_rejected", False)),
                    "hold_reason": str(getattr(st, "last_reject_reason", "") or ""),
                    "depth_hold": bool(getattr(st, "last_depth_outlier", False)),
                },
                "swarm": {
                    "raw_centroid_m": _as_flist(raw_c, 3),
                    "cmd_centroid_m": _as_flist(cmd_c, 3),
                    "T_rel_m": _as_flist(swarm_T, 3),
                    "quat_wxyz": _quat_wxyz(R),
                    "rot_deg": _rot_angle_deg(R),
                },
                "err": {
                    "T_tgt_minus_swarm_m": _as_flist(tgt_T - swarm_T, 3),
                    "T_tgt_minus_ema_m": _as_flist(tgt_T - ema_T, 3),
                    "T_ema_minus_swarm_m": _as_flist(ema_T - swarm_T, 3),
                    "T_applied_minus_swarm_m": _as_flist(applied_T - swarm_T, 3),
                    "T_norm_m": float(np.linalg.norm(tgt_T - swarm_T)),
                },
            }
        )

    def on_disarm(self, *, frame_idx: int, t_s: float, reason: str = "disarm") -> None:
        if self._active is None:
            return
        self._finish_active(frame_idx=frame_idx, t_s=t_s, reason=reason)

    def on_exit(self, *, frame_idx: int, t_s: float) -> None:
        if self._active is not None:
            self._finish_active(frame_idx=frame_idx, t_s=t_s, reason="exit")
        self.save()

    def _finish_active(self, *, frame_idx: int, t_s: float, reason: str) -> None:
        if self._active is None:
            return
        seg = self._active
        seg["end_frame"] = int(frame_idx)
        seg["end_t_s"] = float(t_s)
        seg["end_reason"] = str(reason)
        n = len(seg.get("frames", []))
        self.segments.append(seg)
        print(
            f"[rigid-pose-trace] segment end reason={reason} frames={n} "
            f"t={float(seg['arm_t_s']):.3f}..{float(t_s):.3f}s",
            flush=True,
        )
        self._active = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": "iso_swarm.rigid_pose_trace.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "meta": dict(self.meta),
            "segments": list(self.segments),
        }
        if self._active is not None:
            payload["active_segment"] = self._active
        return payload

    def save(self, path: Path | None = None) -> Path:
        out = Path(path) if path is not None else self.out_path
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        tmp.replace(out)
        self._last_saved_path = out
        n_seg = len(self.segments) + (1 if self._active is not None else 0)
        print(f"[rigid-pose-trace] saved {out} ({n_seg} segment(s))", flush=True)
        return out


def load_rigid_pose_trace(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "segments" not in data and "active_segment" in data:
        data["segments"] = [data["active_segment"]]
    return data


def default_trace_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path.cwd() / "logs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"rigid_pose_trace_{stamp}.json"


def tick_rigid_pose_trace(
    recorder: RigidPoseTraceRecorder | None,
    *,
    armed_this_frame: bool,
    runtime_armed: bool,
    unwinding: bool,
    frame_idx: int,
    t_s: float,
    hand_off_m: np.ndarray | None,
    hand_R: np.ndarray | None,
    raw_target: np.ndarray,
    cmd_target: np.ndarray,
    left_pose_state: Any,
    morph_targets_before_left_m: np.ndarray,
    prev_runtime_armed: bool,
) -> bool:
    """Update recorder for one main-loop frame. Returns current runtime_armed."""
    if recorder is None:
        return bool(runtime_armed)

    if armed_this_frame and left_pose_state.initialized:
        ref_c = _centroid_m(morph_targets_before_left_m)
        ref_wrist = np.asarray(left_pose_state.ref_wrist_mm, dtype=np.float64).reshape(3)
        ref_palm = np.asarray(left_pose_state.ref_palm_center, dtype=np.float64).reshape(3)
        ref_swarm = getattr(left_pose_state, "ref_swarm_targets", None)
        recorder.on_arm(
            frame_idx=int(frame_idx),
            t_s=float(t_s),
            ref_wrist_mm=ref_wrist,
            ref_palm_mm=ref_palm,
            ref_centroid_m=ref_c,
            ref_swarm_targets=ref_swarm,
        )

    recording = bool(runtime_armed or unwinding) and bool(left_pose_state.initialized)
    if recording:
        phase = "unwind" if unwinding else "armed"
        recorder.sample(
            frame_idx=int(frame_idx),
            t_s=float(t_s),
            hand_off_m=hand_off_m,
            hand_R=hand_R,
            raw_target=raw_target,
            cmd_target=cmd_target,
            left_pose_state=left_pose_state,
            phase=phase,
        )
    elif prev_runtime_armed and not runtime_armed and not unwinding:
        recorder.on_disarm(frame_idx=int(frame_idx), t_s=float(t_s), reason="disarm")

    return bool(runtime_armed)
