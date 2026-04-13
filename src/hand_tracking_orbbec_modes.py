"""
Three morph modes (same 3D morph logic as hand_tracking_webcam_modes.py), using the
RGB–D pipeline from hand_tracking_orbbec.py (pyk4a / Femto Bolt–style calibration).

- **Left hand**: mode 1/2/3 (index/middle/ring gesture).
- **Right hand**: open ↔ closed morph only.

Depth fusion, EMA, --hand-frame, --hand-3d, and open remap match hand_tracking_orbbec.py.
Mode 1/2/3 rendering uses `hand_tracking_webcam_modes.update_3d_plot`.

Controls: q quit, p toggle 3D plot, s save matplotlib figure.

Run from `iso_swarm/` or `iso_swarm/src/`: `python hand_tracking_orbbec_modes.py`
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from pyk4a import Config, FPS, PyK4A

import hand_tracking_orbbec as ob
import hand_tracking_webcam_modes as hm


def main():
    ap = argparse.ArgumentParser(
        description="Orbbec/K4A: Left→mode 1/2/3; Right→open/morph (depth-fused 3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Same depth/RGB options as hand_tracking_orbbec.py. "
            "See that script's --help for calibration and fusion notes."
        ),
    )
    # Reuse the same CLI as hand_tracking_orbbec.main (see that file for help text).
    ap.add_argument("--model", type=str, default=None, help="hand_landmarker.task path")
    ap.add_argument(
        "--depth-fusion",
        type=float,
        default=ob.DEPTH_FUSION_WEIGHT,
        help="0=MediaPipe world only, 1=depth unproject only",
    )
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=ob.POINT_EMA_ALPHA,
        help="EMA smoothing on fused keypoints (0..1)",
    )
    ap.add_argument(
        "--hand-frame",
        choices=(ob.HAND_FRAME_SCALED, ob.HAND_FRAME_PALM_PLANE, ob.HAND_FRAME_METRIC_MM),
        default=ob.HAND_FRAME_SCALED,
        help="3D coords: scaled | palm_plane | metric_mm",
    )
    ap.add_argument(
        "--no-shape-normalize",
        action="store_true",
        help="Same as --hand-frame metric_mm (unless palm_plane / metric_mm already set).",
    )
    ap.add_argument(
        "--no-depth-outlier-filter",
        action="store_true",
        help="Keep all per-joint depth samples.",
    )
    ap.add_argument(
        "--no-open-remap",
        action="store_true",
        help="Use raw morph_alpha for HUD (default: remap with shape_norm).",
    )
    ap.add_argument(
        "--open-remap-lo",
        type=float,
        default=None,
        metavar="X",
        help=f"Lower bound for open remap (default {ob.OPEN_REMAP_LO}).",
    )
    ap.add_argument(
        "--open-remap-hi",
        type=float,
        default=None,
        metavar="X",
        help=f"Upper bound for open remap (default {ob.OPEN_REMAP_HI}).",
    )
    ap.add_argument(
        "--depth-patch",
        type=int,
        default=ob.DEPTH_MEDIAN_PATCH_RADIUS,
        metavar="R",
        help="Median depth patch radius",
    )
    ap.add_argument(
        "--depth-max-delta-mm",
        type=float,
        default=None,
        metavar="D",
        help=f"Max |depth−wrist| per joint (default {ob.DEPTH_MAX_DELTA_FROM_WRIST_MM}).",
    )
    ap.add_argument(
        "--depth-median-max-delta-mm",
        type=float,
        default=None,
        metavar="D",
        help="Second pass vs palm median; 0 disables.",
    )
    ap.add_argument(
        "--use-transformed-depth",
        action="store_true",
        help="Use SDK depth→color alignment (often unsafe on Orbbec K4A wrapper).",
    )
    ap.add_argument(
        "--hand-3d",
        choices=(ob.HAND_3D_SOURCE_MP, ob.HAND_3D_SOURCE_FUSED),
        default=ob.HAND_3D_SOURCE_MP,
        help="3D plot source: mp | fused",
    )
    ap.add_argument(
        "--depth-unproject-rigid-npy",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional 4×4 .npy rigid transform after depth unproject.",
    )
    args = ap.parse_args()

    model_path = hm._resolve_model_path(args.model)
    fusion_w = float(np.clip(args.depth_fusion, 0.0, 1.0))
    ema_a = float(np.clip(args.ema_alpha, 0.0, 1.0))
    depth_patch_r = int(np.clip(args.depth_patch, 0, 15))
    if args.hand_frame != ob.HAND_FRAME_SCALED:
        hand_frame = args.hand_frame
    elif args.no_shape_normalize:
        hand_frame = ob.HAND_FRAME_METRIC_MM
    else:
        hand_frame = ob.HAND_FRAME_SCALED
    shape_norm = hand_frame in (ob.HAND_FRAME_SCALED, ob.HAND_FRAME_PALM_PLANE)
    depth_outlier_filter = not args.no_depth_outlier_filter
    if shape_norm and not args.no_open_remap:
        lo = ob.OPEN_REMAP_LO if args.open_remap_lo is None else args.open_remap_lo
        hi = ob.OPEN_REMAP_HI if args.open_remap_hi is None else args.open_remap_hi
        open_remap = (lo, hi) if hi > lo + 1e-6 else None
    else:
        open_remap = None

    depth_max_delta_mm = (
        ob.DEPTH_MAX_DELTA_FROM_WRIST_MM if args.depth_max_delta_mm is None else float(args.depth_max_delta_mm)
    )
    if args.depth_median_max_delta_mm is None:
        depth_median_max_delta_mm: float | None = ob.DEPTH_MEDIAN_MAX_DELTA_MM
    elif args.depth_median_max_delta_mm <= 0:
        depth_median_max_delta_mm = None
    else:
        depth_median_max_delta_mm = float(args.depth_median_max_delta_mm)

    depth_rigid_T = ob.load_depth_unproject_rigid_npy(args.depth_unproject_rigid_npy)

    options = hm.HandLandmarkerOptions(
        base_options=hm.BaseOptions(model_asset_path=model_path),
        running_mode=hm.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    k4a = PyK4A(
        Config(
            color_resolution=1,
            depth_mode=2,
            synchronized_images_only=False,
            camera_fps=FPS.FPS_30,
        )
    )

    with hm.HandLandmarker.create_from_options(options) as landmarker:
        plt.ion()
        fig = plt.figure("Hand 3D — Orbbec modes")
        ax_hand = fig.add_subplot(121, projection="3d")
        ax_topo = fig.add_subplot(122, projection="3d")

        k4a.start()
        calib = k4a.calibration
        print(
            "Depth-fused modes. "
            f"fusion={fusion_w:.2f} ema={ema_a:.2f} depth_patch={depth_patch_r}  "
            f"hand_frame={hand_frame}  shape_norm={shape_norm}  hand_3d={args.hand_3d}  "
            f"open_remap={open_remap}  aligned_depth={args.use_transformed_depth}  "
            f"depth_rigid_T={'on' if depth_rigid_T is not None else 'off'}  "
            "L=mode 1/2/3  R=open  q=quit  p=3D  s=save"
        )

        try:
            frame_idx = 0
            warned_fusion_linear_map = False
            ema_3d = None
            open_free_ema = None
            alpha_smooth = 0.18
            snap_state = None
            hud_cache = {
                "open": None,
                "free": None,
                "plan": None,
                "iso": None,
                "spread": None,
                "curl": None,
                "text": None,
            }
            snap_vis_state = None
            snap_stable_frames = 0
            snap_hold_frames = 0
            enable_3d = hm.ENABLE_3D_PLOT
            mode_raw = 1
            mode_ema = 1.0
            mode_smooth = 0.22
            morph_mode = 1
            mode_raw_prev: Optional[int] = None
            mode_stable_frames = 0
            last_right_pts: Optional[list] = None
            last_open_out: Optional[float] = None
            last_mode_raw = 1

            while True:
                try:
                    capture = k4a.get_capture()
                except Exception as exc:
                    print(f"[WARN] get_capture failed: {exc}")
                    continue
                if capture.color is None:
                    continue

                color = capture.color
                if color.ndim == 3 and color.shape[2] == 4:
                    frame = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                else:
                    frame = color

                depth_raw = capture.depth
                depth_aligned = None
                if args.use_transformed_depth:
                    try:
                        td = capture.transformed_depth
                        if (
                            td is not None
                            and td.size > 0
                            and td.shape[0] == frame.shape[0]
                            and td.shape[1] == frame.shape[1]
                        ):
                            depth_aligned = td
                        elif td is not None:
                            print(
                                f"[WARN] transformed_depth shape {td.shape} != color {frame.shape}; "
                                "ignoring aligned depth"
                            )
                    except Exception as exc:
                        print(f"[WARN] transformed_depth failed: {exc}")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t_ms = int(frame_idx * (1000 / 30))
                try:
                    result = landmarker.detect_for_video(mp_image, t_ms)
                except Exception as exc:
                    print(f"[WARN] detect_for_video: {exc}")
                    continue

                if (
                    fusion_w >= 0.99
                    and not args.use_transformed_depth
                    and args.hand_3d == ob.HAND_3D_SOURCE_FUSED
                    and not warned_fusion_linear_map
                ):
                    print(
                        "[INFO] depth-fusion≈1 + --hand-3d fused: linear RGB→depth map; "
                        "try --hand-3d mp or --depth-patch 3–4."
                    )
                    warned_fusion_linear_map = True

                frame, keypoints_3d, ema_3d = ob.draw_hand(
                    frame,
                    result,
                    depth_raw=depth_raw,
                    depth_aligned=depth_aligned,
                    print_depth=(frame_idx % 30 == 0),
                    calibration=calib,
                    fusion_weight=fusion_w,
                    ema_alpha=ema_a,
                    ema_points=ema_3d,
                    depth_patch_radius=depth_patch_r,
                    hand_frame=hand_frame,
                    filter_depth_outliers=depth_outlier_filter,
                    depth_max_delta_mm=depth_max_delta_mm,
                    depth_median_max_delta_mm=depth_median_max_delta_mm,
                    hand_3d_source=args.hand_3d,
                    depth_unproject_rigid_T=depth_rigid_T,
                    skip_wrist_labels=True,
                )

                idx_L = hm.find_hand_index_by_side(result, "left")
                idx_R = hm.find_hand_index_by_side(result, "right")

                tier_count = -1
                pts_L = None
                if idx_L is not None and idx_L < len(keypoints_3d):
                    pts_L = keypoints_3d[idx_L]
                if pts_L is not None:
                    mode_raw, tier_count, _dbg = hm.classify_mode_from_fingers(pts_L)
                    last_mode_raw = mode_raw
                    if mode_raw_prev is None:
                        mode_raw_prev = mode_raw
                        mode_stable_frames = hm.MODE_DEBOUNCE_FRAMES
                    elif mode_raw == mode_raw_prev:
                        mode_stable_frames += 1
                    else:
                        mode_raw_prev = mode_raw
                        mode_stable_frames = 0

                    if mode_stable_frames >= hm.MODE_DEBOUNCE_FRAMES:
                        mode_ema = mode_smooth * float(mode_raw) + (1.0 - mode_smooth) * float(mode_ema)
                        morph_mode = int(round(mode_ema))
                        morph_mode = max(1, min(3, morph_mode))
                else:
                    mode_raw = last_mode_raw

                hands_3d: List = []
                open_out: Optional[float] = None
                pts_R = None
                if idx_R is not None and idx_R < len(keypoints_3d):
                    pts_R = keypoints_3d[idx_R]
                    last_right_pts = list(pts_R)
                    hands_3d = [pts_R]

                    tmp = ob.analyze_hand_topology(pts_R)
                    if tmp is not None:
                        if open_free_ema is None:
                            open_free_ema = float(tmp["morph_alpha"])
                        else:
                            open_free_ema = (
                                alpha_smooth * float(tmp["morph_alpha"])
                                + (1.0 - alpha_smooth) * open_free_ema
                            )

                        open_free = float(open_free_ema)
                        if snap_state == "plane":
                            if open_free < ob.PLANE_SNAP_OFF:
                                snap_state = None
                        elif snap_state == "sphere":
                            if open_free > ob.SPHERE_SNAP_OFF:
                                snap_state = None
                        else:
                            if open_free > ob.PLANE_SNAP_ON:
                                snap_state = "plane"
                            elif open_free < ob.SPHERE_SNAP_ON:
                                snap_state = "sphere"

                        open_out = open_free
                        if snap_state == "plane":
                            open_out = 1.0
                        elif snap_state == "sphere":
                            open_out = 0.0
                        last_open_out = float(open_out)
                else:
                    if last_right_pts is not None:
                        hands_3d = [last_right_pts]
                    open_out = last_open_out

                wrist_lbl: dict = {}
                if idx_L is not None:
                    wrist_lbl[idx_L] = f"M{int(morph_mode)}"
                if idx_R is not None:
                    wrist_lbl[idx_R] = (
                        f"open {float(open_out):.2f}" if open_out is not None else "open —"
                    )
                if wrist_lbl:
                    ob.overlay_wrist_labels(frame, result, wrist_lbl)

                hint_parts = []
                if idx_L is None:
                    hint_parts.append("no LEFT (mode)")
                if idx_R is None:
                    hint_parts.append("no RIGHT (open frozen)")
                hint = "  |  ".join(hint_parts) if hint_parts else "L=mode  R=open"
                cv2.putText(
                    frame,
                    f"L→M{morph_mode} raw:{mode_raw}  R→open:{open_out if open_out is not None else '—'}  "
                    f"tier:{tier_count if tier_count >= 0 else '-'}  {hint}"[:95],
                    (16, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if snap_state is None:
                    snap_stable_frames = 0
                    if snap_vis_state is not None:
                        snap_hold_frames += 1
                        if snap_hold_frames >= ob.SNAP_HOLD_AFTER_RELEASE_FRAMES:
                            snap_vis_state = None
                            snap_hold_frames = 0
                else:
                    snap_hold_frames = 0
                    if snap_vis_state == snap_state:
                        snap_stable_frames = min(ob.SNAP_SHOW_AFTER_FRAMES, snap_stable_frames + 1)
                    else:
                        snap_stable_frames += 1
                        if snap_stable_frames >= ob.SNAP_SHOW_AFTER_FRAMES:
                            snap_vis_state = snap_state
                            snap_stable_frames = 0

                analyses = None
                if enable_3d and (frame_idx % hm.PLOT_EVERY_N_FRAMES) == 0 and hands_3d:
                    analyses = hm.update_3d_plot(
                        ax_hand,
                        ax_topo,
                        hands_3d,
                        morph_mode=morph_mode,
                        morph_alpha_smoothed=open_out,
                        control_label="Right→open",
                        shape_normalized=shape_norm,
                        hand_frame=hand_frame,
                        hand_3d_source=args.hand_3d,
                    )
                    plt.pause(0.0001)

                if analyses:
                    a0 = analyses[0]
                    topo_lbl = ob._topology_label_from_alpha(
                        float(open_free_ema) if open_free_ema is not None else float(a0["morph_alpha"])
                    )
                    open_disp = open_out if open_out is not None else a0["morph_alpha"]
                    free_disp = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                    if open_remap is not None:
                        lo_r, hi_r = open_remap
                        open_disp = ob._remap_open_display(open_disp, lo_r, hi_r)
                        free_disp = ob._remap_open_display(free_disp, lo_r, hi_r)

                    need_refresh = (frame_idx % hm.HUD_UPDATE_EVERY_N_FRAMES) == 0 or hud_cache["open"] is None
                    if not need_refresh:
                        if abs(float(open_disp) - float(hud_cache["open"])) > hm.HUD_OPEN_STEP:
                            need_refresh = True
                        if abs(float(free_disp) - float(hud_cache["free"])) > hm.HUD_OPEN_STEP:
                            need_refresh = True
                        if abs(float(a0["planarity"]) - float(hud_cache["plan"])) > hm.HUD_METRIC_STEP:
                            need_refresh = True
                        if abs(float(a0["isotropy"]) - float(hud_cache["iso"])) > hm.HUD_METRIC_STEP:
                            need_refresh = True
                        if abs(float(a0["finger_spread"]) - float(hud_cache["spread"])) > hm.HUD_METRIC_STEP:
                            need_refresh = True
                        if hand_frame == ob.HAND_FRAME_PALM_PLANE and hands_3d:
                            cm = ob._palm_plane_curl_metrics(hands_3d[0])
                            curl_s = None
                            if cm and cm.get("mean_r_xy_four") is not None:
                                tr = cm.get("thumb_r_xy")
                                thumb_s = f" thumb_r={tr:.2f}" if tr is not None else ""
                                curl_s = (
                                    f"curl r_xy4={cm['mean_r_xy_four']:.2f} |z|4={cm['mean_abs_z_four']:.2f}"
                                    f"{thumb_s}"
                                )
                            if curl_s != hud_cache.get("curl"):
                                need_refresh = True

                    if need_refresh:
                        hud_cache["open"] = float(open_disp)
                        hud_cache["free"] = float(free_disp)
                        hud_cache["plan"] = float(a0["planarity"])
                        hud_cache["iso"] = float(a0["isotropy"])
                        hud_cache["spread"] = float(a0["finger_spread"])
                        snap_txt = f"  SNAP:{snap_vis_state.upper()}" if snap_vis_state is not None else ""
                        lines = [
                            f"L→mode M{morph_mode}{snap_txt}  (1=sph 2=pyr 3=box)  |  Topo:{topo_lbl}",
                            f"open:{open_disp:.2f}  free:{free_disp:.2f}",
                            f"spread:{a0['finger_spread']:.2f}  plan:{a0['planarity']:.2f}  iso:{a0['isotropy']:.2f}",
                        ]
                        if hand_frame == ob.HAND_FRAME_PALM_PLANE and hands_3d:
                            cm = ob._palm_plane_curl_metrics(hands_3d[0])
                            if cm and cm.get("mean_r_xy_four") is not None:
                                tr = cm.get("thumb_r_xy")
                                thumb_s = f" thumb_r={tr:.2f}" if tr is not None else ""
                                curl_s = (
                                    f"curl r_xy4={cm['mean_r_xy_four']:.2f} |z|4={cm['mean_abs_z_four']:.2f}"
                                    f"{thumb_s}"
                                )
                                hud_cache["curl"] = curl_s
                                lines.append(curl_s)
                            else:
                                hud_cache["curl"] = None
                        else:
                            hud_cache["curl"] = None
                        hud_cache["text"] = lines

                    if frame_idx % 30 == 0:
                        out_v = open_out if open_out is not None else a0["morph_alpha"]
                        free_v = open_free_ema if open_free_ema is not None else a0["morph_alpha"]
                        if open_remap is not None:
                            lo_r, hi_r = open_remap
                            out_show = ob._remap_open_display(out_v, lo_r, hi_r)
                            free_show = ob._remap_open_display(free_v, lo_r, hi_r)
                            open_part = f"open={out_show:.3f} raw={out_v:.3f} free={free_show:.3f} raw_free={free_v:.3f}"
                        else:
                            open_part = f"open_out={out_v:.3f} free={free_v:.3f}"
                        curl_part = ""
                        if hand_frame == ob.HAND_FRAME_PALM_PLANE and hands_3d:
                            cm = ob._palm_plane_curl_metrics(hands_3d[0])
                            if cm and cm.get("mean_r_xy_four") is not None:
                                tr = cm.get("thumb_r_xy")
                                thumb_p = f" thumb_r={tr:.3f}" if tr is not None else ""
                                curl_part = (
                                    f" curl_rxy4={cm['mean_r_xy_four']:.3f} curl_z4={cm['mean_abs_z_four']:.3f}"
                                    f"{thumb_p}"
                                )
                        print(
                            f"M{morph_mode} raw={mode_raw} {open_part} "
                            f"spread={a0['finger_spread']:.3f} plan={a0['planarity']:.3f} iso={a0['isotropy']:.3f}"
                            f"{curl_part}"
                        )

                cv2.imshow("Hand Tracking Orbbec Modes", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    out_name = f"hand_orbbec_mode_frame_{frame_idx:06d}.png"
                    fig.savefig(out_name, dpi=150, bbox_inches="tight")
                    print(f"Saved: {out_name}")
                if key == ord("p"):
                    enable_3d = not enable_3d
                    print(f"3D plot: {enable_3d}")
                if key == ord("q"):
                    break

                if hud_cache["text"] is not None:
                    ob.draw_hud(frame, hud_cache["text"], origin=(16, 16))
                frame_idx += 1
        finally:
            k4a.stop()
            plt.ioff()
            plt.close(fig)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
