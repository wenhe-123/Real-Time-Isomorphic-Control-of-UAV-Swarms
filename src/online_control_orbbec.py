"""Online Crazyflow control using the Orbbec RGB-D hand tracking pipeline.

This mirrors ``online_control.py`` but replaces the webcam-only MediaPipe path
with the same depth-fused Orbbec path used by ``pipelines/orbbec_main.py``.
The drones hold the initial M1/open=1 target until ``s`` is pressed.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line
from online_control import (
    ScaleConfig,
    closest_pair,
    make_initial_live_target,
    normalize_morph_points,
)
from pipelines import orbbec_main as om
import runtime.hand_tracking_orbbec as ob
from runtime.hand_tracking_webcam_modes import (
    MORPH_PLANE_RADIUS_A,
    MORPH_PLANE_RADIUS_B,
    mode_epsilon_pair,
)
from shared.common_utils import resolve_model_path
from shared.hand_constants import MCP_IDS, WRIST_ID
from shared.modes_runtime import (
    ModeState,
    RightHandState,
    RuntimeState,
    SnapVisualState,
    draw_bottom_status,
    overlay_mode_open_wrist_labels,
    process_left_mode,
    process_right_open,
    update_hud_cache,
    update_snap_visual_state_for_modes,
)
from shared.morph_lp_plot import update_3d_plot_lp
from shared.morph_renderers import mapped_fixed_surface_points
from shared.morph_shape_control import (
    LpShapePipelineState,
    advance_lp_shape_p,
    index_mcp_tip_segment_norm,
)
from shared.mp_hand_utils import find_hand_index_by_side
from shared.stream_runtime_utils import (
    capture_orbbec_frame,
    get_aligned_depth,
    safe_get_capture,
)
from shared.topology_utils import clamp01


def _parse_combined_args() -> argparse.Namespace:
    parser = om.build_parser()
    parser.description = "Run online Crazyflow control from Orbbec morph targets."
    parser.add_argument("--point-count", type=int, default=24, help="Number of Crazyflow drones / morph points.")
    parser.add_argument("--mode", type=int, default=1, help="Initial morph mode while unarmed.")
    parser.add_argument("--open", type=float, default=1.0, dest="open_alpha", help="Initial open value while unarmed.")
    parser.add_argument("--shape-t", type=float, default=None, help="Initial mode-shape interpolation value.")
    parser.add_argument("--radius-mm", type=float, default=50.0, help="Initial morph radius in millimeters.")
    parser.add_argument("--duration", type=float, default=0.0, help="Run time in seconds; <=0 means run until q.")
    parser.add_argument("--fps", type=int, default=60, help="Crazyflow/render control loop rate.")
    parser.add_argument("--target-alpha", type=float, default=0.06, help="EMA smoothing for Crazyflow targets.")
    parser.add_argument("--xy-radius", type=float, default=1.20, help="Crazyflow XY workspace radius in meters.")
    parser.add_argument("--z-center", type=float, default=1.40, help="Crazyflow target z center in meters.")
    parser.add_argument("--z-amplitude", type=float, default=0.35, help="Crazyflow target z amplitude in meters.")
    parser.add_argument("--z-min", type=float, default=1.10, help="Minimum Crazyflow target altitude.")
    parser.add_argument("--z-max", type=float, default=2.10, help="Maximum Crazyflow target altitude.")
    parser.add_argument("--reference-xy-extent-mm", type=float, default=100.0)
    parser.add_argument("--reference-z-extent-mm", type=float, default=50.0)
    parser.add_argument("--plot-every", type=int, default=ob.PLOT_EVERY_N_FRAMES)
    parser.add_argument("--print-only", action="store_true", help="Only print the initial target and exit.")
    return parser.parse_args()


def _make_orbbec_config(args: argparse.Namespace) -> om.OrbbecModesConfig:
    model_path = resolve_model_path(args.model, __file__)
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
    depth_max_delta_mm = ob.DEPTH_MAX_DELTA_FROM_WRIST_MM if args.depth_max_delta_mm is None else float(args.depth_max_delta_mm)
    if args.depth_median_max_delta_mm is None:
        depth_median_max_delta_mm: float | None = ob.DEPTH_MEDIAN_MAX_DELTA_MM
    elif args.depth_median_max_delta_mm <= 0:
        depth_median_max_delta_mm = None
    else:
        depth_median_max_delta_mm = float(args.depth_median_max_delta_mm)
    return om.OrbbecModesConfig(
        model_path=model_path,
        fusion_w=fusion_w,
        ema_a=ema_a,
        depth_patch_r=depth_patch_r,
        hand_frame=hand_frame,
        shape_norm=shape_norm,
        depth_outlier_filter=depth_outlier_filter,
        open_remap=open_remap,
        depth_max_delta_mm=depth_max_delta_mm,
        depth_median_max_delta_mm=depth_median_max_delta_mm,
        depth_rigid_T=ob.load_depth_unproject_rigid_npy(args.depth_unproject_rigid_npy),
        use_transformed_depth=bool(args.use_transformed_depth),
        hand_3d=str(args.hand_3d),
    )


def _update_crazyflow_target(
    *,
    analyses,
    live_target,
    mode_state,
    open_out,
    lp_shape,
    scale,
) -> None:
    a0 = analyses[0]
    open_v = float(open_out if open_out is not None else a0["morph_alpha"])
    epsilon1, epsilon2 = mode_epsilon_pair(int(mode_state.morph_mode), lp_shape.left_shape_t_ema)
    points_mm = mapped_fixed_surface_points(
        radius=float(a0["radius"]),
        open_alpha=open_v,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        plane_radius_a=MORPH_PLANE_RADIUS_A,
        plane_radius_b=MORPH_PLANE_RADIUS_B,
        morph_mode=int(mode_state.morph_mode),
    )
    live_target.set(normalize_morph_points(points_mm, scale), mode=int(mode_state.morph_mode), open_alpha=open_v)


def run_integrated_orbbec_control(
    *,
    config: om.OrbbecModesConfig,
    live_target,
    point_count: int,
    duration: float,
    fps: int,
    target_alpha: float,
    plot_every_n: int,
    scale: ScaleConfig,
) -> None:
    landmarker = om.build_landmarker(config.model_path)
    k4a = om.build_k4a()
    n_worlds = 1
    n_drones = int(point_count)
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control=Control.state)
    sim.reset()

    first_target = live_target.get()
    first_target[:, 2] = np.maximum(first_target[:, 2], 1.10)
    smooth_target = first_target.copy()
    pos_buffer = deque(maxlen=int(max(2, min(8, 96 // max(n_drones, 1)))))
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(n_drones, 2)))
    zeros = jnp.zeros_like(sim.data.states.pos)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=jnp.asarray(first_target[None, :, :], device=sim.device),
            vel=zeros,
            ang_vel=zeros,
        )
    )

    runtime = RuntimeState(enable_3d=True)
    mode_state = ModeState()
    right_state = RightHandState()
    snap_visual = SnapVisualState()
    lp_shape = LpShapePipelineState()
    plot_every_n = max(1, int(plot_every_n))
    control_steps_per_frame = max(1, int(round(sim.freq / max(float(fps), 1.0))))

    plt.ion()
    fig = plt.figure("Online Control Orbbec + 3D")
    ax_hand = fig.add_subplot(121, projection="3d")
    ax_topo = fig.add_subplot(122, projection="3d")

    k4a.start()
    calib = k4a.calibration
    print(
        "Online Orbbec Crazyflow control. "
        f"n={n_drones} fusion={config.fusion_w:.2f} ema={config.ema_a:.2f} "
        f"hand_frame={config.hand_frame} hand_3d={config.hand_3d} "
        "L=mode R=open  s=arm  p=3D  q=quit"
    )
    print("Holding default target until s is pressed.")

    start_time = time.monotonic()
    last_status_second = -1
    gesture_control_enabled = False
    try:
        while True:
            elapsed = time.monotonic() - start_time
            if float(duration) > 0.0 and elapsed > float(duration):
                break

            capture = safe_get_capture(k4a)
            if capture is None:
                continue
            got = capture_orbbec_frame(capture)
            if got is None:
                continue
            frame, depth_raw, capture = got
            depth_aligned = get_aligned_depth(capture, frame, config.use_transformed_depth)
            result = om.detect_frame(landmarker, frame, runtime.frame_idx)
            if result is None:
                continue

            if (
                config.fusion_w >= 0.99
                and not config.use_transformed_depth
                and config.hand_3d == ob.HAND_3D_SOURCE_FUSED
                and not runtime.warned_fusion_linear_map
            ):
                print("[INFO] depth-fusion≈1 + --hand-3d fused: try --hand-3d mp or --depth-patch 3-4.")
                runtime.warned_fusion_linear_map = True

            frame, keypoints_3d, runtime.ema_3d = ob.draw_hand(
                frame,
                result,
                depth_raw=depth_raw,
                depth_aligned=depth_aligned,
                print_depth=(runtime.frame_idx % 30 == 0),
                calibration=calib,
                fusion_weight=config.fusion_w,
                ema_alpha=config.ema_a,
                ema_points=runtime.ema_3d,
                depth_patch_radius=config.depth_patch_r,
                hand_frame=config.hand_frame,
                filter_depth_outliers=config.depth_outlier_filter,
                depth_max_delta_mm=config.depth_max_delta_mm,
                depth_median_max_delta_mm=config.depth_median_max_delta_mm,
                hand_3d_source=config.hand_3d,
                depth_unproject_rigid_T=config.depth_rigid_T,
                skip_wrist_labels=True,
            )

            idx_left = find_hand_index_by_side(result, "left")
            idx_right = find_hand_index_by_side(result, "right")
            mode_raw, tier_count = process_left_mode(keypoints_3d, idx_left, mode_state)
            hands_3d, open_out = process_right_open(keypoints_3d, idx_right, right_state)

            pts_left = keypoints_3d[idx_left] if idx_left is not None and idx_left < len(keypoints_3d) else None
            dist_norm = index_mcp_tip_segment_norm(pts_left, wrist_id=WRIST_ID, mcp_ids=MCP_IDS) if pts_left is not None else None
            advance_lp_shape_p(dist_norm, int(mode_state.morph_mode), lp_shape)

            overlay_mode_open_wrist_labels(
                frame=frame,
                result=result,
                idx_left=idx_left,
                idx_right=idx_right,
                morph_mode=mode_state.morph_mode,
                open_out=open_out,
                overlay_wrist_labels_fn=ob.overlay_wrist_labels,
            )
            update_snap_visual_state_for_modes(right_state.snap_state, snap_visual)

            analyses = None
            if runtime.enable_3d and hands_3d and (runtime.frame_idx % plot_every_n) == 0:
                analyses = update_3d_plot_lp(
                    ax_hand,
                    ax_topo,
                    hands_3d,
                    morph_mode=mode_state.morph_mode,
                    morph_alpha_smoothed=open_out,
                    control_label="online orbbec",
                    analyze_hand_topology_fn=ob.analyze_hand_topology,
                    clamp01_fn=clamp01,
                    shape_normalized=config.shape_norm,
                    hand_frame=config.hand_frame,
                    hand_3d_source=config.hand_3d,
                    hand_frame_palm_plane=ob.HAND_FRAME_PALM_PLANE,
                    norm_axis_halflim=ob.NORM_AXIS_HALFLIM,
                    morph_axis_lim_mm=ob.MORPH_AXIS_LIM_MM,
                    mode_shape_t=lp_shape.left_shape_t_ema,
                    epsilon_pair_display=lp_shape.epsilon_pair_display,
                )
                plt.pause(0.001)

            if analyses:
                update_hud_cache(
                    runtime=runtime,
                    frame_idx=runtime.frame_idx,
                    analyses=analyses,
                    hands_3d=hands_3d,
                    hand_frame=config.hand_frame,
                    morph_mode=mode_state.morph_mode,
                    open_out=open_out,
                    open_free_ema=right_state.open_free_ema,
                    open_remap=config.open_remap,
                    snap_vis_state=snap_visual.snap_vis_state,
                )
                if gesture_control_enabled:
                    _update_crazyflow_target(
                        analyses=analyses,
                        live_target=live_target,
                        mode_state=mode_state,
                        open_out=open_out,
                        lp_shape=lp_shape,
                        scale=scale,
                    )

            raw_target = live_target.get()
            smooth_target = target_alpha * raw_target + (1.0 - target_alpha) * smooth_target
            cmd = np.zeros((n_worlds, n_drones, 13), dtype=np.float32)
            cmd[..., :3] = smooth_target
            cmd[..., 9] = 0.0
            sim.state_control(cmd)
            sim.step(control_steps_per_frame)

            if runtime.frame_idx % 2 == 0:
                pos_buffer.append(sim.data.states.pos[0].copy())
            if len(pos_buffer) > 1:
                lines = np.asarray(pos_buffer)
                for d in range(n_drones):
                    draw_line(sim, lines[:, d, :], rgba=colors[d].tolist(), start_size=0.5, end_size=2.0)
            sim.render()

            draw_bottom_status(frame, mode_state.morph_mode, mode_raw, tier_count, idx_left, idx_right, open_out)
            if runtime.hud_cache["text"] is not None:
                ob.draw_hud(frame, runtime.hud_cache["text"], origin=(16, 16))
            cv2.putText(
                frame,
                "ONLINE ORBBEC " + ("ARMED" if gesture_control_enabled else "HOLD DEFAULT - press s"),
                (16, frame.shape[0] - 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Online Control Orbbec", frame)

            status_second = int(elapsed)
            if status_second != last_status_second:
                last_status_second = status_second
                target_now = live_target.get()
                min_dist, min_i, min_j = closest_pair(target_now)
                open_txt = f"{float(open_out):.2f}" if open_out is not None else "-"
                print(
                    f"online_orbbec t={elapsed:.1f}s armed={'yes' if gesture_control_enabled else 'no'} "
                    f"mode={int(mode_state.morph_mode)} raw={mode_raw} open={open_txt} "
                    f"L={'yes' if idx_left is not None else 'no'} R={'yes' if idx_right is not None else 'no'} "
                    f"target_min=({min_i},{min_j}) {min_dist:.2f}m"
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                runtime.enable_3d = not runtime.enable_3d
                print(f"3D plot: {runtime.enable_3d}")
            if key == ord("s"):
                gesture_control_enabled = True
                print("Gesture control armed: Crazyflow targets now follow Orbbec mode/open recognition.")
            runtime.frame_idx += 1
    finally:
        landmarker.close()
        k4a.stop()
        sim.close()
        plt.ioff()
        plt.close(fig)
        cv2.destroyAllWindows()


def main() -> None:
    args = _parse_combined_args()
    scale = ScaleConfig(
        xy_radius=float(args.xy_radius),
        z_center=float(args.z_center),
        z_amplitude=float(args.z_amplitude),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        reference_xy_extent_mm=float(args.reference_xy_extent_mm),
        reference_z_extent_mm=float(args.reference_z_extent_mm),
    )
    live_target = make_initial_live_target(
        point_count=int(args.point_count),
        radius_mm=float(args.radius_mm),
        morph_mode=int(args.mode),
        open_alpha=float(args.open_alpha),
        shape_t=args.shape_t,
        scale=scale,
    )
    if args.print_only:
        return
    run_integrated_orbbec_control(
        config=_make_orbbec_config(args),
        live_target=live_target,
        point_count=int(args.point_count),
        duration=float(args.duration),
        fps=int(args.fps),
        target_alpha=float(args.target_alpha),
        plot_every_n=int(args.plot_every),
        scale=scale,
    )


if __name__ == "__main__":
    main()
