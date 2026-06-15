#!/usr/bin/env python3
"""Replay hand vs swarm rigid pose traces recorded by ``--rigid-pose-trace``.

Usage (from ``iso_swarm`` with pixi env)::

    python src/debug/replay_rigid_pose_trace.py logs/rigid_pose_trace_20260101_120000.json
    python src/debug/replay_rigid_pose_trace.py logs/rigid_pose_trace_*.json --segment 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from debug.rigid_pose_trace import load_rigid_pose_trace
from functions.swarm_motion.left_hand_swarm_pose import quat_to_R


def _segments(data: dict) -> list[dict]:
    segs = list(data.get("segments") or [])
    active = data.get("active_segment")
    if active and (not segs or segs[-1] is not active):
        segs.append(active)
    return [s for s in segs if s.get("frames")]


def _frame_arrays(seg: dict) -> dict[str, np.ndarray]:
    frames = seg["frames"]
    n = len(frames)
    t = np.array([float(f["t_s"]) for f in frames], dtype=np.float64)
    t0 = float(t[0]) if n else 0.0
    t_rel = t - t0

    hand_T = np.array([f["hand"]["T_m"] for f in frames], dtype=np.float64)
    hand_tgt_T = np.array(
        [f["hand"].get("tgt_T_m", f["hand"]["T_m"]) for f in frames],
        dtype=np.float64,
    )
    swarm_T = np.array([f["swarm"]["T_rel_m"] for f in frames], dtype=np.float64)
    err_T = np.array(
        [
            f["err"].get(
                "T_tgt_minus_swarm_m",
                f["err"].get("T_hand_minus_swarm_m", np.zeros(3)),
            )
            for f in frames
        ],
        dtype=np.float64,
    )
    hand_rot = np.array([float(f["hand"]["rot_deg"]) for f in frames], dtype=np.float64)
    swarm_rot = np.array([float(f["swarm"]["rot_deg"]) for f in frames], dtype=np.float64)
    hold = np.array([bool(f["hand"].get("hold", False)) for f in frames], dtype=bool)
    depth_hold = np.array([bool(f["hand"].get("depth_hold", False)) for f in frames], dtype=bool)

    hand_R = np.stack([quat_to_R(f["hand"]["quat_wxyz"]) for f in frames], axis=0)
    ref_c = np.asarray(seg["ref_centroid_m"], dtype=np.float64).reshape(3)
    cmd_c = np.array([f["swarm"]["cmd_centroid_m"] for f in frames], dtype=np.float64)
    raw_c = np.array([f["swarm"]["raw_centroid_m"] for f in frames], dtype=np.float64)

    return {
        "t_rel": t_rel,
        "hand_T": hand_T,
        "hand_tgt_T": hand_tgt_T,
        "swarm_T": swarm_T,
        "err_T": err_T,
        "hand_rot": hand_rot,
        "swarm_rot": swarm_rot,
        "hold": hold,
        "depth_hold": depth_hold,
        "hand_R": hand_R,
        "ref_c": ref_c,
        "cmd_c": cmd_c,
        "raw_c": raw_c,
        "n": n,
    }


def _draw_triad(ax, origin: np.ndarray, R: np.ndarray, scale: float, labels: tuple[str, str, str]) -> list:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    colors = ("#e74c3c", "#2ecc71", "#3498db")
    artists: list = []
    for i, (c, lab) in enumerate(zip(colors, labels)):
        d = R[:, i] * float(scale)
        artists.append(
            ax.quiver(
                o[0], o[1], o[2], d[0], d[1], d[2],
                color=c, arrow_length_ratio=0.15, linewidth=2,
            )
        )
        artists.append(
            ax.text(o[0] + d[0], o[1] + d[1], o[2] + d[2], lab, color=c, fontsize=8)
        )
    return artists


def replay_segment(seg: dict, *, speed: float = 1.0) -> None:
    arr = _frame_arrays(seg)
    n = int(arr["n"])
    if n == 0:
        print("Segment has no frames.", flush=True)
        return

    ref_c = arr["ref_c"]
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title(
        f"Rigid pose trace  arm_f={seg.get('arm_frame')}  n={n}  reason={seg.get('end_reason')}"
    )

    ax3d = fig.add_axes([0.05, 0.30, 0.42, 0.65], projection="3d")
    ax_tx = fig.add_axes([0.55, 0.72, 0.40, 0.22])
    ax_rot = fig.add_axes([0.55, 0.42, 0.40, 0.22])
    ax_err = fig.add_axes([0.55, 0.12, 0.40, 0.22])

    # Static paths
    hand_world = ref_c + arr["hand_tgt_T"]
    ema_world = ref_c + arr["hand_T"]
    swarm_world = arr["cmd_c"]
    ax3d.plot(hand_world[:, 0], hand_world[:, 1], hand_world[:, 2], "C0-", label="hand tgt", linewidth=1.8)
    ax3d.plot(ema_world[:, 0], ema_world[:, 1], ema_world[:, 2], "C0--", label="hand ema", linewidth=1.0, alpha=0.65)
    ax3d.plot(swarm_world[:, 0], swarm_world[:, 1], swarm_world[:, 2], "C1-", label="swarm cmd", linewidth=1.5)
    ax3d.scatter([ref_c[0]], [ref_c[1]], [ref_c[2]], c="k", s=40, label="arm ref")

    ax_tx.plot(arr["t_rel"], arr["hand_tgt_T"][:, 0], color="#1f77b4", linestyle="-", linewidth=1.6, label="hand tgt X")
    ax_tx.plot(arr["t_rel"], arr["hand_tgt_T"][:, 1], color="#1f77b4", linestyle="--", linewidth=1.6, label="hand tgt Y")
    ax_tx.plot(arr["t_rel"], arr["hand_tgt_T"][:, 2], color="#1f77b4", linestyle=":", linewidth=1.6, label="hand tgt Z")
    ax_tx.plot(arr["t_rel"], arr["hand_T"][:, 0], color="#aec7e8", linestyle="-", linewidth=1.0, alpha=0.9, label="hand ema X")
    ax_tx.plot(arr["t_rel"], arr["hand_T"][:, 1], color="#aec7e8", linestyle="--", linewidth=1.0, alpha=0.9, label="hand ema Y")
    ax_tx.plot(arr["t_rel"], arr["hand_T"][:, 2], color="#aec7e8", linestyle=":", linewidth=1.0, alpha=0.9, label="hand ema Z")
    ax_tx.plot(arr["t_rel"], arr["swarm_T"][:, 0], color="#ff7f0e", linestyle="-", linewidth=1.4, label="swarm X")
    ax_tx.plot(arr["t_rel"], arr["swarm_T"][:, 1], color="#ff7f0e", linestyle="--", linewidth=1.4, label="swarm Y")
    ax_tx.plot(arr["t_rel"], arr["swarm_T"][:, 2], color="#ff7f0e", linestyle=":", linewidth=1.4, label="swarm Z")
    for i in np.where(arr["hold"])[0]:
        ax_tx.axvline(arr["t_rel"][i], color="#e67e22", alpha=0.06)
    ax_tx.set_title("Translation (rel. arm centroid, m)")
    ax_tx.set_xlabel("t (s)")
    ax_tx.legend(fontsize=7, ncol=2, loc="upper right")
    ax_tx.grid(True, alpha=0.3)

    ax_rot.plot(arr["t_rel"], arr["hand_rot"], "C0-", label="hand |R| deg")
    ax_rot.plot(arr["t_rel"], arr["swarm_rot"], "C1--", label="swarm |R| deg")
    for i in np.where(arr["hold"])[0]:
        ax_rot.axvline(arr["t_rel"][i], color="#e67e22", alpha=0.08)
    for i in np.where(arr["depth_hold"])[0]:
        ax_rot.axvline(arr["t_rel"][i], color="#9b59b6", alpha=0.06)
    ax_rot.set_title("Rotation magnitude (deg); orange=hold, purple=depth_hold")
    ax_rot.set_xlabel("t (s)")
    ax_rot.legend(fontsize=8)
    ax_rot.grid(True, alpha=0.3)

    err_norm = np.linalg.norm(arr["err_T"], axis=1)
    ax_err.plot(arr["t_rel"], err_norm, "C3-", label="|T_tgt - T_swarm|")
    ax_err.set_title("Hand target vs swarm (|T_tgt - T_swarm|)")
    ax_err.set_xlabel("t (s)")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    # Dynamic artists
    hand_pt, = ax3d.plot([], [], [], "o", color="C0", markersize=8)
    swarm_pt, = ax3d.plot([], [], [], "o", color="C1", markersize=8)
    vline_tx = ax_tx.axvline(0.0, color="k", alpha=0.35)
    vline_rot = ax_rot.axvline(0.0, color="k", alpha=0.35)
    vline_err = ax_err.axvline(0.0, color="k", alpha=0.35)
    time_text = fig.text(0.05, 0.22, "", fontsize=10)

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="upper left", fontsize=8)

    all_pts = np.vstack([hand_world, swarm_world, ref_c.reshape(1, 3)])
    pad = max(0.15, float(np.ptp(all_pts)) * 0.15)
    mid = np.mean(all_pts, axis=0)
    ax3d.set_xlim(mid[0] - pad, mid[0] + pad)
    ax3d.set_ylim(mid[1] - pad, mid[1] + pad)
    ax3d.set_zlim(mid[2] - pad, mid[2] + pad)

    triad_scale = max(0.08, pad * 0.35)
    triad_artists: list = []

    ax_slider = fig.add_axes([0.12, 0.06, 0.55, 0.03])
    slider = Slider(ax_slider, "frame", 0, max(0, n - 1), valinit=0, valstep=1)
    ax_play = fig.add_axes([0.72, 0.055, 0.10, 0.04])
    btn_play = Button(ax_play, "Play")

    state = {"playing": False, "anim": None}

    def set_frame(i: int) -> None:
        i = int(np.clip(i, 0, n - 1))
        slider.set_val(i)
        hw = hand_world[i]
        ew = ema_world[i]
        sw = swarm_world[i]
        hand_pt.set_data([hw[0]], [hw[1]])
        hand_pt.set_3d_properties([hw[2]])
        swarm_pt.set_data([sw[0]], [sw[1]])
        swarm_pt.set_3d_properties([sw[2]])

        for a in triad_artists:
            try:
                a.remove()
            except Exception:
                pass
        triad_artists.clear()
        triad_artists.extend(_draw_triad(ax3d, hw, arr["hand_R"][i], triad_scale, ("Hx", "Hy", "Hz")))
        triad_artists.extend(_draw_triad(ax3d, sw, arr["hand_R"][i], triad_scale * 0.85, ("Sx", "Sy", "Sz")))

        tr = float(arr["t_rel"][i])
        vline_tx.set_xdata([tr, tr])
        vline_rot.set_xdata([tr, tr])
        vline_err.set_xdata([tr, tr])
        f = seg["frames"][i]
        flags = []
        if f["hand"].get("hold"):
            flags.append("hold")
        if f["hand"].get("depth_hold"):
            flags.append("depth_hold")
        flag_s = (" [" + ",".join(flags) + "]") if flags else ""
        time_text.set_text(
            f"t={tr:.3f}s  frame={f['frame']}  phase={f.get('phase','?')}{flag_s}\n"
            f"hand tgt T=({hw[0]:+.3f},{hw[1]:+.3f},{hw[2]:+.3f})  "
            f"hand ema T=({ew[0]:+.3f},{ew[1]:+.3f},{ew[2]:+.3f})  "
            f"swarm T=({sw[0]:+.3f},{sw[1]:+.3f},{sw[2]:+.3f})  "
            f"err={err_norm[i]:.4f}m"
        )
        fig.canvas.draw_idle()

    def on_slider(val) -> None:
        set_frame(int(val))

    def on_play(_event) -> None:
        state["playing"] = not state["playing"]
        btn_play.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"] and state["anim"] is not None:
            state["anim"].event_source.start()

    slider.on_changed(on_slider)

    def animate(_i):
        if not state["playing"]:
            return hand_pt, swarm_pt
        nxt = int(slider.val) + 1
        if nxt >= n:
            nxt = 0
        set_frame(nxt)
        return hand_pt, swarm_pt

    interval_ms = max(20, int(33 / max(speed, 0.1)))
    state["anim"] = FuncAnimation(fig, animate, interval=interval_ms, blit=False, cache_frame_data=False)
    set_frame(0)
    btn_play.on_clicked(on_play)
    plt.show()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay rigid hand/swarm pose trace JSON")
    ap.add_argument("trace", type=Path, help="Path to rigid_pose_trace_*.json")
    ap.add_argument("--segment", type=int, default=-1, help="Segment index (-1 = last)")
    ap.add_argument("--list", action="store_true", help="List segments and exit")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = ap.parse_args(argv)

    data = load_rigid_pose_trace(args.trace)
    segs = _segments(data)
    if not segs:
        print(f"No segments with frames in {args.trace}", flush=True)
        return 1

    if args.list:
        for i, s in enumerate(segs):
            nf = len(s.get("frames") or [])
            print(
                f"[{i}] arm_frame={s.get('arm_frame')} frames={nf} "
                f"t={s.get('arm_t_s')}..{s.get('end_t_s')} reason={s.get('end_reason')}"
            )
        return 0

    idx = int(args.segment)
    if idx < 0:
        idx = len(segs) + idx
    idx = int(np.clip(idx, 0, len(segs) - 1))
    print(f"Replaying segment {idx}/{len(segs)-1} from {args.trace}", flush=True)
    replay_segment(segs[idx], speed=float(args.speed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
