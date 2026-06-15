"""
Laptop webcam hand tracking with morph modes (standalone demo; not used by online_control_dual).

- **Left hand**: mode 1–5 (finger extension tiers).
- **Right hand**: open ↔ closed morph only.

Controls: q quit, p toggle 3D, s save matplotlib figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from functions.display_sim.common_utils import draw_hud, resolve_model_path
from backup.runtime.hand_draw_utils import draw_all_hands
from functions.mode_switch.modes_runtime import (
    ModeState,
    RightHandState,
    SnapVisualState,
    build_modes_hud_lines,
    update_mode_state as shared_update_mode_state,
    update_open_state as shared_update_open_state,
    update_snap_visual_state as shared_update_snap_visual_state,
)
from functions.mode_switch.hand_constants import HAND_CONNECTIONS
from functions.open_close.morph_lp_plot import (
    MORPH_LP_MESH_ETA,
    MORPH_LP_MESH_OMEGA,
    mode_epsilon_pair,
    update_3d_plot_lp,
)
from functions.dual_cam.mp_hand_utils import extract_world_points_mm_result, find_left_right_indices
from functions.mode_switch.topology_utils import clamp01, topology_label_from_alpha
from functions.mode_switch.webcam_mode_defaults import (
    ENABLE_3D_PLOT,
    EPSILON_TRANSITION_K,
    HAND_3D_SOURCE_MP,
    HAND_FRAME_PALM_PLANE,
    HAND_FRAME_SCALED,
    MORPH_AXIS_LIM_MM,
    NORM_AXIS_HALFLIM,
    HUD_METRIC_STEP,
    HUD_OPEN_STEP,
    HUD_UPDATE_EVERY_N_FRAMES,
    MODE_DEBOUNCE_FRAMES,
    PLOT_ADAPT_DOWN_FPS,
    PLOT_ADAPT_UP_FPS,
    PLOT_EVERY_N_FRAMES,
    PLOT_EVERY_N_MAX,
    PLANE_SNAP_OFF,
    PLANE_SNAP_ON,
    SNAP_HOLD_AFTER_RELEASE_FRAMES,
    SNAP_SHOW_AFTER_FRAMES,
    SPHERE_SNAP_OFF,
    SPHERE_SNAP_ON,
    TOPO_ALPHA_PLANE,
    TOPO_ALPHA_SPHERE,
    analyze_hand_topology,
    classify_mode_from_fingers,
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def lerp_eps(
    prev: Optional[Tuple[float, float]], target: Tuple[float, float], k: float
) -> Tuple[float, float]:
    if prev is None:
        return float(target[0]), float(target[1])
    kk = float(np.clip(k, 0.01, 1.0))
    e1 = float(prev[0] + kk * (float(target[0]) - prev[0]))
    e2 = float(prev[1] + kk * (float(target[1]) - prev[1]))
    return e1, e2


def update_3d_plot(
    ax_hand,
    ax_topo,
    hands_3d,
    morph_mode: int,
    morph_alpha_smoothed=None,
    control_label: str = "",
    *,
    shape_normalized: bool = False,
    hand_frame: str = HAND_FRAME_SCALED,
    hand_3d_source: str = HAND_3D_SOURCE_MP,
    mode_shape_t: Optional[float] = None,
    epsilon_pair_display: Optional[Tuple[float, float]] = None,
    analyze_hand_topology_fn_override=None,
    lp_show_refs: bool = True,
    show_sample_ids: bool = False,
    mesh_n_eta: int = MORPH_LP_MESH_ETA,
    mesh_n_omega: int = MORPH_LP_MESH_OMEGA,
):
    return update_3d_plot_lp(
        ax_hand,
        ax_topo,
        hands_3d,
        morph_mode=morph_mode,
        morph_alpha_smoothed=morph_alpha_smoothed,
        control_label=control_label,
        analyze_hand_topology_fn=(
            analyze_hand_topology if analyze_hand_topology_fn_override is None else analyze_hand_topology_fn_override
        ),
        clamp01_fn=clamp01,
        shape_normalized=shape_normalized,
        hand_frame=hand_frame,
        hand_3d_source=hand_3d_source,
        hand_frame_palm_plane=HAND_FRAME_PALM_PLANE,
        norm_axis_halflim=NORM_AXIS_HALFLIM,
        morph_axis_lim_mm=MORPH_AXIS_LIM_MM,
        hand_connections=HAND_CONNECTIONS,
        mode_shape_t=mode_shape_t,
        epsilon_pair_display=epsilon_pair_display,
        lp_show_refs=lp_show_refs,
        show_sample_ids=show_sample_ids,
        mesh_n_eta=mesh_n_eta,
        mesh_n_omega=mesh_n_omega,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Left hand → mode 1–5; Right hand → open/morph only",
    )
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument(
        "--plot-every",
        type=int,
        default=None,
        metavar="N",
        help="Refresh matplotlib 3D every N camera frames (default: %s). Smaller = smoother 3D, higher CPU."
        % (PLOT_EVERY_N_FRAMES,),
    )
    ap.add_argument(
        "--adaptive-plot-every",
        action="store_true",
        help="Auto-adjust 3D refresh interval based on FPS (may introduce visual discretization).",
    )
    ap.add_argument(
        "--no-3d-refs",
        action="store_true",
        help="Skip faint Lp reference wireframes (faster 3D draw).",
    )
    ap.add_argument(
        "--show-sample-ids",
        action="store_true",
        help="Draw sample point ID text in 3D (slower; off by default).",
    )
    ap.add_argument(
        "--camera-buffer",
        type=int,
        default=1,
        help="VideoCapture buffer size (1 reduces latency; may not work on all drivers).",
    )
    args = ap.parse_args()
    from functions.open_close.morph_renderers import prompt_and_init_fixed_surface_points

    prompt_and_init_fixed_surface_points()
    model_path = resolve_model_path(args.model, __file__)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(args.camera_buffer))
    except Exception:
        pass

    plot_every_n = int(args.plot_every) if args.plot_every is not None else int(PLOT_EVERY_N_FRAMES)
    plot_every_n = max(1, plot_every_n)
    adaptive_plot_every_n = int(plot_every_n)
    use_adaptive_plot_every = bool(args.adaptive_plot_every)
    lp_show_refs = not bool(args.no_3d_refs)

    with HandLandmarker.create_from_options(options) as landmarker:
        plt.ion()
        fig = plt.figure("Hand 3D Webcam")
        ax_hand = fig.add_subplot(121, projection="3d")
        ax_topo = fig.add_subplot(122, projection="3d")

        print(
            "Left hand = MODE (1–5). Right hand = OPEN / shape morph.  q=quit p=3D s=save"
        )

        try:
            frame_idx = 0
            mode_state = ModeState()
            right_state = RightHandState()
            snap_visual = SnapVisualState()
            hud_cache = {"open": None, "free": None, "plan": None, "iso": None, "spread": None, "text": None}
            eps_display: Optional[Tuple[float, float]] = None
            enable_3d = ENABLE_3D_PLOT
            perf_ema_dt: Optional[float] = None
            while True:
                frame_t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_ms = int(frame_idx * (1000 / 30))
                try:
                    result = landmarker.detect_for_video(mp_image, t_ms)
                except Exception as exc:
                    print(f"[WARN] detect_for_video: {exc}")
                    continue

                idx_L, idx_R = find_left_right_indices(result, invert_handedness=False)

                pts_L = (
                    extract_world_points_mm_result(result, idx_L) if idx_L is not None else None
                )
                mode_raw, tier_count = shared_update_mode_state(
                    pts_L,
                    mode_state=mode_state,
                    classify_mode_fn=classify_mode_from_fingers,
                    debounce_frames=MODE_DEBOUNCE_FRAMES,
                    mode_smooth=0.22,
                )
                active_mode = int(mode_state.morph_mode)

                hands_3d: List = []
                pts_R = None
                topo_right = None
                if idx_R is not None:
                    pts_R = extract_world_points_mm_result(result, idx_R)
                    if pts_R is not None:
                        topo_right = analyze_hand_topology(pts_R)
                        right_state.last_right_pts = list(pts_R)
                        hands_3d = [pts_R]
                else:
                    if right_state.last_right_pts is not None:
                        hands_3d = [right_state.last_right_pts]

                open_out = shared_update_open_state(
                    pts_R,
                    right_state=right_state,
                    analyze_topology_fn=analyze_hand_topology,
                    open_smooth=0.18,
                    plane_snap_on=PLANE_SNAP_ON,
                    plane_snap_off=PLANE_SNAP_OFF,
                    sphere_snap_on=SPHERE_SNAP_ON,
                    sphere_snap_off=SPHERE_SNAP_OFF,
                    topology_analysis=topo_right,
                )

                frame, _kp_map = draw_all_hands(
                    frame,
                    result,
                    mode_hand_idx=idx_L,
                    morph_hand_idx=idx_R,
                    morph_mode=mode_state.morph_mode,
                    open_value=open_out,
                    depth_map=None,
                    print_depth=False,
                )

                hint_parts = []
                if idx_L is None:
                    hint_parts.append("no LEFT (mode)")
                if idx_R is None:
                    hint_parts.append("no RIGHT (open frozen)")
                hint = "  |  ".join(hint_parts) if hint_parts else "L=mode  R=open"
                otxt = f"{open_out:.2f}" if open_out is not None else "-"
                cv2.putText(
                    frame,
                    f"M{mode_state.morph_mode} raw:{mode_raw}  open:{otxt}  "
                    f"tier:{tier_count if tier_count >= 0 else '-'}  "
                    f"{hint}"[:95],
                    (16, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                shared_update_snap_visual_state(
                    right_state.snap_state,
                    snap_visual_state=snap_visual,
                    snap_show_after_frames=SNAP_SHOW_AFTER_FRAMES,
                    snap_hold_after_release_frames=SNAP_HOLD_AFTER_RELEASE_FRAMES,
                )

                analyses = None
                if enable_3d and (frame_idx % adaptive_plot_every_n) == 0 and hands_3d:
                    eps_target = mode_epsilon_pair(int(mode_state.morph_mode), None)
                    eps_display = lerp_eps(eps_display, eps_target, EPSILON_TRANSITION_K)
                    analyses = update_3d_plot(
                        ax_hand,
                        ax_topo,
                        hands_3d,
                        morph_mode=mode_state.morph_mode,
                        morph_alpha_smoothed=open_out,
                        control_label="open+p",
                        mode_shape_t=None,
                        epsilon_pair_display=eps_display,
                        analyze_hand_topology_fn_override=(
                            (lambda _pts, _topo=topo_right: _topo) if topo_right is not None else None
                        ),
                        lp_show_refs=lp_show_refs,
                        show_sample_ids=bool(args.show_sample_ids),
                    )
                    try:
                        fig.canvas.flush_events()
                    except Exception:
                        pass
                    plt.pause(0.001)

                if analyses:
                    a0 = analyses[0]
                    open_disp = open_out if open_out is not None else a0["morph_alpha"]
                    free_disp = right_state.open_free_ema if right_state.open_free_ema is not None else a0["morph_alpha"]

                    need_refresh = (frame_idx % HUD_UPDATE_EVERY_N_FRAMES) == 0 or hud_cache["open"] is None
                    if not need_refresh:
                        if abs(float(open_disp) - float(hud_cache["open"])) > HUD_OPEN_STEP:
                            need_refresh = True
                        if abs(float(free_disp) - float(hud_cache["free"])) > HUD_OPEN_STEP:
                            need_refresh = True
                        if abs(float(a0["planarity"]) - float(hud_cache["plan"])) > HUD_METRIC_STEP:
                            need_refresh = True
                        if abs(float(a0["isotropy"]) - float(hud_cache["iso"])) > HUD_METRIC_STEP:
                            need_refresh = True
                        if abs(float(a0["finger_spread"]) - float(hud_cache["spread"])) > HUD_METRIC_STEP:
                            need_refresh = True

                    if need_refresh:
                        hud_cache["open"] = float(open_disp)
                        hud_cache["free"] = float(free_disp)
                        hud_cache["plan"] = float(a0["planarity"])
                        hud_cache["iso"] = float(a0["isotropy"])
                        hud_cache["spread"] = float(a0["finger_spread"])
                        topo_lbl = topology_label_from_alpha(
                            float(free_disp),
                            plane_thr=TOPO_ALPHA_PLANE,
                            sphere_thr=TOPO_ALPHA_SPHERE,
                        )
                        hud_cache["text"] = build_modes_hud_lines(
                            morph_mode=mode_state.morph_mode,
                            topo_label=topo_lbl,
                            open_disp=float(open_disp),
                            free_disp=float(free_disp),
                            spread=float(a0["finger_spread"]),
                            planarity=float(a0["planarity"]),
                            isotropy=float(a0["isotropy"]),
                        )

                    if frame_idx % 5 == 0:
                        open_v = float(open_out if open_out is not None else a0["morph_alpha"])
                        e1d, e2d = eps_display if eps_display is not None else mode_epsilon_pair(int(mode_state.morph_mode), None)
                        print(
                            f"mode={int(mode_state.morph_mode)} open={open_v:.3f} "
                            f"eps=({float(e1d):.3f},{float(e2d):.3f}) radius={float(a0['radius']):.1f} "
                            f"plan={float(a0['planarity']):.3f} iso={float(a0['isotropy']):.3f}"
                        )

                if hud_cache["text"] is not None:
                    draw_hud(frame, hud_cache["text"], origin=(16, 16))

                cv2.imshow("Hand Tracking Webcam Modes", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    out_name = f"hand_webcam_mode_frame_{frame_idx:06d}.png"
                    fig.savefig(out_name, dpi=150, bbox_inches="tight")
                    print(f"Saved: {out_name}")
                if key == ord("p"):
                    enable_3d = not enable_3d
                    print(f"3D plot: {enable_3d}")
                if key == ord("q"):
                    break

                dt = max(1e-5, float(time.perf_counter() - frame_t0))
                perf_ema_dt = dt if perf_ema_dt is None else (0.12 * dt + 0.88 * perf_ema_dt)
                if use_adaptive_plot_every and (frame_idx % 15) == 0:
                    fps_est = 1.0 / max(1e-5, float(perf_ema_dt))
                    if fps_est < PLOT_ADAPT_UP_FPS and adaptive_plot_every_n < PLOT_EVERY_N_MAX:
                        adaptive_plot_every_n += 1
                    elif fps_est > PLOT_ADAPT_DOWN_FPS and adaptive_plot_every_n > plot_every_n:
                        adaptive_plot_every_n -= 1

                frame_idx += 1
        finally:
            cap.release()
            plt.ioff()
            plt.close(fig)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
