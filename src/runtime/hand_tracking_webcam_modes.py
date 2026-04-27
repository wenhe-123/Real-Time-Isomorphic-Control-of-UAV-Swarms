"""
Laptop webcam hand tracking with three morph modes (test on webcam first).

- **Left hand**: selects **mode** 1–5 (index/middle/ring/pinky/thumb “how many up”; see
  classify_mode_from_fingers). Does not drive open/close.
- **Right hand**: only drives **open ↔ closed** shape (same topology / morph_alpha as
  hand_tracking_orbbec). Left never affects planarity, spread, or open.
- Both hands drawn; thicker skeleton on Left = mode, on Right = open/morph.

Modes:
  1–5 — Plane ↔ superellipsoid bulge; mode picks base (ε₁,ε₂); finger slide adjusts within range.

Controls: q quit, p toggle 3D, s save matplotlib figure.
Performance: 3D uses matplotlib and is heavy; use ``--plot-every 12`` or ``--no-3d-refs``, or press ``p`` to disable 3D for smoother camera preview.

If the left hand is missing, the last mode is kept. If the right hand is missing,
open/morph freezes at the last value until the right hand returns.
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from shared.common_utils import draw_hud, resolve_model_path
from shared.hand_constants import FINGERTIP_IDS, HAND_CONNECTIONS, MCP_IDS, WRIST_ID
from shared.hand_draw_utils import draw_all_hands, draw_single_hand
from shared.mode_gesture_utils import (
    classify_mode_from_fingers as _shared_classify_mode_from_fingers,
    palm_center_and_scale as _shared_palm_center_and_scale,
)
from shared.morph_lp_plot import (
    MORPH_LP_MESH_ETA,
    MORPH_LP_MESH_OMEGA,
    MORPH_PLANE_RADIUS_A,
    MORPH_PLANE_RADIUS_B,
    mode_epsilon_pair,
    update_3d_plot_lp,
)
from shared.morph_renderers import prompt_and_init_fixed_surface_points
from shared.morph_renderers import get_fixed_surface_count, mapped_fixed_surface_points
from shared.morph_shape_control import (
    LpShapePipelineState,
    advance_lp_shape_p,
    index_mcp_tip_segment_norm,
)
from shared.modes_runtime import (
    ModeState,
    RightHandState,
    SnapVisualState,
    build_modes_hud_lines,
    update_mode_state as shared_update_mode_state,
    update_open_state as shared_update_open_state,
    update_snap_visual_state as shared_update_snap_visual_state,
)
from shared.mp_hand_utils import extract_world_points_mm_result, find_left_right_indices
from shared.topology_utils import (
    analyze_hand_topology_common,
    clamp01,
    safe_normalize,
    topology_label_from_alpha,
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# MediaPipe exposes RunningMode (not VisionRunningMode) on tasks.vision
RunningMode = mp.tasks.vision.RunningMode

# Index / middle / ring / pinky / thumb — used for 1–5 mode gesture.
MODE_COUNT_TIP_IDS = [8, 12, 16, 20, 4]

OPEN_GAMMA = 1.8
TOPO_ALPHA_PLANE = 0.67
TOPO_ALPHA_SPHERE = 0.33
MORPH_AXIS_LIM_MM = 200.0
# Wrist-centered normalized plots (match hand_tracking_orbbec.py)
NORM_AXIS_HALFLIM = 1.35
HAND_3D_SOURCE_MP = "mp"
HAND_3D_SOURCE_FUSED = "fused"
HAND_FRAME_SCALED = "scaled"
HAND_FRAME_PALM_PLANE = "palm_plane"
HAND_FRAME_METRIC_MM = "metric_mm"

# 3D matplotlib is heavy; refresh less often for smoother camera window (override with --plot-every).
PLOT_EVERY_N_FRAMES = 8
ENABLE_3D_PLOT = True
# Lp mesh resolution defaults: shared.morph_lp_plot (MORPH_LP_MESH_ETA / MORPH_LP_MESH_OMEGA).

PLANE_SNAP_ON = 0.88
PLANE_SNAP_OFF = 0.82
SPHERE_SNAP_ON = 0.12
SPHERE_SNAP_OFF = 0.18

HUD_UPDATE_EVERY_N_FRAMES = 10
HUD_OPEN_STEP = 0.03
HUD_METRIC_STEP = 0.05

SNAP_SHOW_AFTER_FRAMES = 6
SNAP_HOLD_AFTER_RELEASE_FRAMES = 10

# Mode classification: normalized tip distances vs. hand scale (index/middle/ring/pinky/thumb)
MODE_EXTEND_MIN = 0.62  # below this max(dn): fist / no clear gesture → mode 1
# Tips within this gap of max(dn) count as "same tier extended" (fixes slightly bent finger).
# Shared classifier uses max(gap, 0.34*mx) plus pinky rescue; keep floor in sync.
MODE_TIER_GAP = 0.38
# Require raw mode this many consecutive frames before accepting (reduces flicker)
MODE_DEBOUNCE_FRAMES = 7

# Left-hand in-mode Lp: ‖index_tip−index_MCP‖ / palm scale → shape_t (see shared.morph_shape_control).

def _safe_normalize(v):
    return safe_normalize(v)


def _clamp01(x):
    return clamp01(x)


def palm_center_and_scale(hand_points: Sequence[Tuple[float, float, float]]):
    return _shared_palm_center_and_scale(hand_points, WRIST_ID, MCP_IDS)


def classify_mode_from_fingers(hand_points: Sequence[Tuple[float, float, float]]):
    return _shared_classify_mode_from_fingers(
        hand_points,
        mode_count_tip_ids=MODE_COUNT_TIP_IDS,
        mode_extend_min=MODE_EXTEND_MIN,
        mode_tier_gap=MODE_TIER_GAP,
        wrist_id=WRIST_ID,
        mcp_ids=MCP_IDS,
    )


def analyze_hand_topology(hand_points):
    return analyze_hand_topology_common(
        hand_points,
        wrist_id=WRIST_ID,
        mcp_ids=MCP_IDS,
        fingertip_ids=FINGERTIP_IDS,
        open_gamma=OPEN_GAMMA,
        label_fn=lambda a: topology_label_from_alpha(a, plane_thr=TOPO_ALPHA_PLANE, sphere_thr=TOPO_ALPHA_SPHERE),
    )


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
    lp_show_refs: bool = True,
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
        analyze_hand_topology_fn=analyze_hand_topology,
        clamp01_fn=_clamp01,
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
        help="Refresh matplotlib 3D every N camera frames (default: %s). Larger = smoother video, choppier 3D."
        % (PLOT_EVERY_N_FRAMES,),
    )
    ap.add_argument(
        "--no-3d-refs",
        action="store_true",
        help="Skip faint Lp reference wireframes (faster 3D draw).",
    )
    ap.add_argument(
        "--camera-buffer",
        type=int,
        default=1,
        help="VideoCapture buffer size (1 reduces latency; may not work on all drivers).",
    )
    args = ap.parse_args()
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
            enable_3d = ENABLE_3D_PLOT
            lp_shape = LpShapePipelineState()

            while True:
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
                dist_norm = (
                    index_mcp_tip_segment_norm(pts_L, wrist_id=WRIST_ID, mcp_ids=MCP_IDS)
                    if pts_L is not None
                    else None
                )

                # --- Mode: LEFT hand only (right hand ignored for mode) ---
                mode_raw, tier_count = shared_update_mode_state(
                    pts_L,
                    mode_state=mode_state,
                    classify_mode_fn=classify_mode_from_fingers,
                    debounce_frames=MODE_DEBOUNCE_FRAMES,
                    mode_smooth=0.22,
                )
                active_mode = int(mode_state.morph_mode)
                advance_lp_shape_p(dist_norm, active_mode, lp_shape)

                # --- Open / morph: RIGHT hand only (left hand ignored for open) ---
                hands_3d: List = []
                pts_R = None
                if idx_R is not None:
                    pts_R = extract_world_points_mm_result(result, idx_R)
                    if pts_R is not None:
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
                lshape_txt = f"{lp_shape.left_shape_t_ema:.2f}" if lp_shape.left_shape_t_ema is not None else "-"
                ref_v = lp_shape.left_ref_dist_by_mode.get(active_mode)
                ref_txt = f"{ref_v:.3f}" if ref_v is not None else "-"
                otxt = f"{open_out:.2f}" if open_out is not None else "-"
                cv2.putText(
                    frame,
                    f"M{mode_state.morph_mode} raw:{mode_raw}  open:{otxt}  "
                    f"tier:{tier_count if tier_count >= 0 else '-'}  "
                    f"Lshape:{lshape_txt}  ref:{ref_txt}  "
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
                if enable_3d and (frame_idx % plot_every_n) == 0 and hands_3d:
                    analyses = update_3d_plot(
                        ax_hand,
                        ax_topo,
                        hands_3d,
                        morph_mode=mode_state.morph_mode,
                        morph_alpha_smoothed=open_out,
                        control_label="open+p",
                        mode_shape_t=lp_shape.left_shape_t_ema,
                        epsilon_pair_display=lp_shape.epsilon_pair_display,
                        lp_show_refs=lp_show_refs,
                    )
                    # Let Qt/Tk process events without long sleep (3D draw is already the heavy part).
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
                        e1, e2 = mode_epsilon_pair(int(mode_state.morph_mode), lp_shape.left_shape_t_ema)
                        pts = mapped_fixed_surface_points(
                            radius=float(a0["radius"]),
                            open_alpha=open_v,
                            epsilon1=e1,
                            epsilon2=e2,
                            plane_radius_a=MORPH_PLANE_RADIUS_A,
                            plane_radius_b=MORPH_PLANE_RADIUS_B,
                            morph_mode=int(mode_state.morph_mode),
                        )
                        pts_txt = " ".join(
                            [f"{i}:({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for i, p in enumerate(pts)]
                        )
                        print(f"mode={int(mode_state.morph_mode)} n={get_fixed_surface_count()} points={pts_txt}")

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

                frame_idx += 1
        finally:
            cap.release()
            plt.ioff()
            plt.close(fig)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
