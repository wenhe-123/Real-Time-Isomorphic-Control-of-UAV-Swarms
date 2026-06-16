"""Online Crazyflow + Orbbec hand control (use ``online_control_dual.py`` by default).

Main loop only. Logic lives under ``functions/{mode_switch,open_close,swarm_motion,display_sim,dual_cam,runtime}/``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from functions.display_sim.common_utils import resolve_model_path
from functions.display_sim.frame_profiler import FrameSectionProfiler
from functions.display_sim.online_frame_present import present_online_frame
from functions.display_sim.gesture_report_debug import close_report_debug_figures
from functions.display_sim.online_plot import close_3d_plot, init_3d_plot
from functions.display_sim.online_plot_frame import update_online_plot_frame
from functions.display_sim.orbbec_hand import create_hand_landmarker
from functions.dual_cam.dual_view_utils import open_webcam_capture
from functions.dual_cam.online_frame_capture import grab_orbbec_mp_frame
from functions.dual_cam.online_input_keys import apply_online_control_key  # re-export
from functions.dual_cam.online_input_keys import (
    OnlineKeyQueue,
    format_hotkey_install_hint,
    process_online_control_keys,
    probe_global_hotkey_backends,
    try_install_hotkey_dependencies,
)
from functions.dual_cam.ui_poll import poll_cv_key
from functions.mode_switch.online_frame_gesture import process_online_gesture_frame
from functions.open_close.morph_renderers import (
    init_fixed_surface_points,
    prompt_and_init_fixed_surface_points,
)
from functions.open_close.morph_world import ScaleConfig
from functions.runtime.live_target import LiveTargetState
from functions.runtime.online_boot import (
    boot_online_control,
    make_key_poller,
    print_pipeline_mode,
    sync_armed_flags,
)
from functions.runtime.online_cli_args import build_online_control_parser, report_debug_panels_from_args
from functions.display_sim.gesture_report_debug import ReportDebugPanels
from functions.runtime.online_defaults import (
    _DEFAULT_LEFT_LOST_DECAY,
    _DEFAULT_LEFT_TRANS_SCALE_MM,
    _ONLINE_MP_INPUT_SCALE,
    _WCAM_PREVIEW_WINDOW,
)
from functions.runtime.online_targets import make_initial_live_target, update_live_target_from_state
from functions.runtime.pipeline_tuning import PipelineTuning, online_pipeline_defaults
from functions.swarm_motion.formation_spacing import lift_morph_to_hover_z
from functions.swarm_motion.online_frame_filter import filter_online_targets
from functions.swarm_motion.online_left_swarm_frame import apply_left_swarm_frame, clamp_workspace_targets
from functions.swarm_motion.prearm import complete_prearm_takeoff
from functions.swarm_motion.spacing_guard import closest_pair, enforce_min_separation

try:
    from debug.pipeline_tuning import resolve_pipeline_tuning
    from debug.rigid_pose_trace import RigidPoseTraceRecorder, default_trace_path, tick_rigid_pose_trace
except ImportError:
    resolve_pipeline_tuning = None  # type: ignore[misc, assignment]
    RigidPoseTraceRecorder = None  # type: ignore[misc, assignment]
    default_trace_path = None  # type: ignore[misc, assignment]
    tick_rigid_pose_trace = None  # type: ignore[misc, assignment]

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_UI_BACKEND", "GTK3")


def run_integrated_online_control(
    live_target: LiveTargetState,
    point_count: int,
    duration: float,
    fps: int,
    min_separation_m: float,
    model_path: str | None,
    plot_every_n: int,
    scale: ScaleConfig,
    *,
    trail_every_n: int = 0,
    led_every_n: int = 3,
    sim_render_every: int = 2,
    morph_radius_mm: float = 50.0,
    drone_model: str = "cf21B_500",
    max_sim_substeps: int = 160,
    imshow_every: int = 4,
    mp_detect_every: int = 1,
    raw_target_ema: float = 0.0,
    left_swarm_pose: bool = True,
    left_trans_scale: float = _DEFAULT_LEFT_TRANS_SCALE_MM,
    left_rot_scale: float = 1.0,
    left_trans_ema: float = 1.0,
    left_rot_ema: float = 1.0,
    left_max_offset_m: float = 1.35,
    left_max_rot_rad: float = 0.55,
    left_axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
    left_lost_decay: float = _DEFAULT_LEFT_LOST_DECAY,
    debug_drone_targets_every: int = 0,
    spacing_audit_every: int = 0,
    prearm_hover_z: float = 1.80,
    prearm_takeoff_z: float = 0.92,
    open_jump_reset: float = 0.34,
    left_unwind_s: float = 2.6,
    left_swarm_depth_frame_motion: bool = True,
    left_rot_gate_rad: float = 0.07,
    left_yaw_min_horiz: float = 0.17,
    left_rot_gain: float = 1.0,
    left_rot_trans_tau_mm: float = 32.0,
    left_rot_world_z_scale: float = 1.0,
    left_cam_y_to_world_z: float = 1.0,
    left_cam_preset: str = "fwd_y",
    left_world_frame: str = "camera_at_arm",
    left_axis_trans_deadzone_m: float = 0.004,
    left_axis_rot_deadzone_rad: float = 0.014,
    left_axis_trans_on_m: float = 0.004,
    left_axis_rot_on_rad: float = 0.020,
    left_axis_trans_rot_coupling: float = 0.50,
    install_hotkey_deps: bool = False,
    left_palm_basis: str = "middle_thumb",
    left_plane_rot_scale_mul: float = 1.0,
    left_rot_direct_follow: bool = True,
    left_rot_pivot: str = "centroid",
    left_dual_webcam_rot: bool = True,
    left_rot_webcam_vis_thresh: float = 0.42,
    left_rot_webcam_index: int = -1,
    orbbec_flip_horizontal: bool = False,
    orbbec_use_transformed_depth: bool = False,
    orbbec_hand_swap: str = "auto",
    formation_rigid_3d_debug: bool = False,
    left_pose_frame_viz: bool = False,
    left_pose_frame_viz_every: int = 10,
    left_pose_debug: bool = True,
    left_pose_debug_every: int = 2,
    webcam_rot_stride: int = 6,
    show_webcam_preview: bool = False,
    global_hotkeys: bool = True,
    mode_vis_min: float = 0.55,
    open_vis_min: float = 0.55,
    planner: str = "direct",
    axswarm_settings: str | None = None,
    axswarm_project_root: str | None = None,
    axswarm_max_iters: int | None = None,
    axswarm_max_solve_ms: float = 90.0,
    axswarm_max_deviation_m: float = 0.2,
    axswarm_pos_weight: float | None = None,
    profile_frame: bool = False,
    profile_every: int = 60,
    swarm_workspace_box_m: float = 3.5,
    swarm_workspace_wall_margin_m: float = 0.03,
    swarm_workspace_clear_margin_m: float = 0.015,
    swarm_workspace_mode: str = "clip",
    left_palm_depth_outlier_z_mm: float = 95.0,
    left_palm_depth_outlier_lat_ratio: float = 2.2,
    left_palm_center_depth_ema: float = 0.42,
    center_trace: bool = False,
    center_trace_every: int = 10,
    rigid_pose_trace: bool = False,
    rigid_pose_trace_out: str | None = None,
    rigid_pose_trace_every: int = 1,
    draw_hand_debug: bool = False,
    report_panels: ReportDebugPanels | None = None,
    mp_delegate: str = "cpu",
    pipeline_tuning: PipelineTuning | None = None,
    drones_config: str | None = None,
    real_lighthouse: bool | None = None,
) -> None:
    """Run MediaPipe, Matplotlib, OpenCV, and Crazyflow or real Crazyflie in the main thread."""
    resolved_model = resolve_model_path(model_path, __file__)
    mp_delegate_key = str(mp_delegate).strip().lower()
    pipe = pipeline_tuning or online_pipeline_defaults()
    print_pipeline_mode(pipe, draw_hand_debug=draw_hand_debug, mp_delegate_key=mp_delegate_key)

    plot_every_n = max(0, int(plot_every_n))
    trail_every_n = max(0, int(trail_every_n))
    led_every_n = max(1, int(led_every_n))
    sim_render_every = max(0, int(sim_render_every))
    imshow_every = max(1, int(imshow_every))
    mp_detect_every = max(1, int(mp_detect_every))
    raw_target_ema = float(max(0.0, min(raw_target_ema, 1.0)))
    debug_drone_targets_every = max(0, int(debug_drone_targets_every))
    center_trace_every = max(1, int(center_trace_every))

    boot = boot_online_control(
        live_target=live_target,
        point_count=point_count,
        fps=fps,
        min_separation_m=min_separation_m,
        scale=scale,
        pipe=pipe,
        drone_model=drone_model,
        prearm_hover_z=prearm_hover_z,
        prearm_takeoff_z=prearm_takeoff_z,
        planner=planner,
        axswarm_settings=axswarm_settings,
        axswarm_project_root=axswarm_project_root,
        axswarm_max_iters=axswarm_max_iters,
        axswarm_max_solve_ms=axswarm_max_solve_ms,
        axswarm_max_deviation_m=axswarm_max_deviation_m,
        axswarm_pos_weight=axswarm_pos_weight,
        max_sim_substeps=max_sim_substeps,
        plot_every_n=plot_every_n,
        debug_report_panels=report_panels,
        left_swarm_pose=left_swarm_pose,
        left_unwind_s=left_unwind_s,
        left_rot_direct_follow=left_rot_direct_follow,
        left_swarm_depth_frame_motion=left_swarm_depth_frame_motion,
        left_world_frame=left_world_frame,
        left_cam_preset=left_cam_preset,
        left_cam_y_to_world_z=left_cam_y_to_world_z,
        left_palm_basis=left_palm_basis,
        left_plane_rot_scale_mul=left_plane_rot_scale_mul,
        left_rot_pivot=left_rot_pivot,
        left_dual_webcam_rot=left_dual_webcam_rot,
        left_rot_webcam_vis_thresh=left_rot_webcam_vis_thresh,
        left_rot_scale=left_rot_scale,
        left_rot_gain=left_rot_gain,
        left_rot_gate_rad=left_rot_gate_rad,
        left_rot_trans_tau_mm=left_rot_trans_tau_mm,
        left_rot_world_z_scale=left_rot_world_z_scale,
        orbbec_flip_horizontal=orbbec_flip_horizontal,
        orbbec_use_transformed_depth=orbbec_use_transformed_depth,
        orbbec_hand_swap=orbbec_hand_swap,
        swarm_workspace_box_m=swarm_workspace_box_m,
        swarm_workspace_wall_margin_m=swarm_workspace_wall_margin_m,
        swarm_workspace_clear_margin_m=swarm_workspace_clear_margin_m,
        swarm_workspace_mode=swarm_workspace_mode,
        center_trace=center_trace,
        center_trace_every=center_trace_every,
        install_hotkey_deps=install_hotkey_deps,
        global_hotkeys=global_hotkeys,
        drones_config=drones_config,
        real_lighthouse=real_lighthouse,
    )
    _poll_keys = make_key_poller(boot, global_hotkeys=global_hotkeys)
    ocv_window_title = "Online Control Orbbec"

    rigid_pose_recorder = None
    rigid_pose_trace_prev_armed = False
    if bool(rigid_pose_trace) and RigidPoseTraceRecorder is not None:
        trace_out = (
            Path(rigid_pose_trace_out)
            if rigid_pose_trace_out
            else default_trace_path(Path.cwd() / "logs")
        )
        rigid_pose_recorder = RigidPoseTraceRecorder(
            out_path=trace_out,
            sample_every=max(1, int(rigid_pose_trace_every)),
            meta={
                "left_cam_preset": str(left_cam_preset),
                "left_world_frame": str(left_world_frame),
                "left_trans_scale": float(left_trans_scale),
            },
        )
        print(
            f"[rigid-pose-trace] recording enabled → {trace_out} "
            f"(every {max(1, int(rigid_pose_trace_every))} frame(s); replay with "
            f"python src/debug/replay_rigid_pose_trace.py {trace_out})",
            flush=True,
        )

    try:
        with create_hand_landmarker(resolved_model, delegate=mp_delegate_key) as landmarker:
            webcam_cap = None
            webcam_landmarker = None
            webcam_frame_idx = 0
            wcam_rot_stride = max(1, int(webcam_rot_stride))
            wcam_rot_cache: dict = {"B": None, "res": None, "fr": None, "idx": None}
            if boot.left_dual_webcam_rot_eff:
                try:
                    webcam_cap, _widx, _wb = open_webcam_capture(
                        int(left_rot_webcam_index), 0, 0, 8
                    )
                    webcam_landmarker = create_hand_landmarker(
                        resolved_model, delegate=mp_delegate_key
                    )
                    print(
                        f"Dual USB webcam index {_widx} ({_wb}): palm rotation when Orbbec "
                        f"visibility < {boot.left_rot_webcam_vis_thresh:.2f}; mode M1–M5 assist when "
                        f"visibility < {mode_vis_min:.2f} or while rotating; translation stays depth."
                    )
                except Exception as exc:
                    print(f"[WARN] Dual webcam rotation disabled: {exc}")
                    boot.left_dual_webcam_rot_eff = False
                    webcam_cap = None
                    if webcam_landmarker is not None:
                        try:
                            webcam_landmarker.close()
                        except Exception:
                            pass
                        webcam_landmarker = None
            try:
                cv2.namedWindow(ocv_window_title, cv2.WINDOW_NORMAL)
            except Exception:
                pass
            if bool(show_webcam_preview) and boot.left_dual_webcam_rot_eff:
                try:
                    cv2.namedWindow(_WCAM_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
                except Exception:
                    pass
            frame_prof = FrameSectionProfiler(
                enabled=bool(profile_frame),
                report_every=max(1, int(profile_every)),
            )
            if frame_prof.enabled:
                print(
                    f"Frame profiling ON (perf_counter); report every "
                    f"{frame_prof.report_every} completed frames."
                )
            if mp_detect_every > 1:
                print(
                    f"MediaPipe: detect every {mp_detect_every} frames "
                    f"(reuse landmarks on skip; Orbbec depth pose still each frame)."
                )
            if imshow_every > 1:
                print(
                    f"Orbbec preview: imshow every {imshow_every} frames "
                    f"(control loop still every frame; pollKey on off frames when available)."
                )
            cached_mp_result = None
            cached_hands_3d_all: list = []
            ema_3d = None
            cv_poll_key = getattr(cv2, "pollKey", None)
            left_pose_dbg = ""
            left_swarm_off = None
            left_swarm_R = None
            prev_gesture_armed = False

            while True:
                elapsed = time.monotonic() - boot.start_time
                if float(duration) > 0.0 and elapsed > float(duration):
                    break
                if _poll_keys():
                    break
                sync_armed_flags(boot)

                frame_prof.frame_start()
                t_ms = int(boot.frame_idx * (1000 / max(float(fps), 1.0)))
                cap, poll_frame, boot.orbbec_flip_depth_warned = grab_orbbec_mp_frame(
                    k4a=boot.k4a,
                    landmarker=landmarker,
                    frame_idx=boot.frame_idx,
                    t_ms=t_ms,
                    fps=float(fps),
                    mp_detect_every=mp_detect_every,
                    mp_input_scale=float(_ONLINE_MP_INPUT_SCALE),
                    orbbec_flip_horizontal=bool(orbbec_flip_horizontal),
                    orbbec_use_transformed_depth=bool(orbbec_use_transformed_depth),
                    use_depth_fusion=boot.use_depth_fusion,
                    pipe=boot.pipe,
                    calib=boot.calib,
                    draw_hand_debug=bool(draw_hand_debug),
                    cached_mp_result=cached_mp_result,
                    cached_hands_3d_all=cached_hands_3d_all,
                    ema_3d=ema_3d,
                    orbbec_flip_depth_warned=boot.orbbec_flip_depth_warned,
                    section=frame_prof.section,
                )
                if cap is None:
                    _pf = poll_frame if poll_frame is not None else np.zeros((1, 1, 3), np.uint8)
                    _ck = poll_cv_key(
                        cv_poll_key=cv_poll_key,
                        imshow=False,
                        window=ocv_window_title,
                        frame=_pf,
                    )
                    if _poll_keys(_ck):
                        break
                    frame_prof.cancel()
                    time.sleep(0.002 if poll_frame is not None else 0.004)
                    continue
                cached_mp_result = cap.result
                cached_hands_3d_all = cap.hands_3d_all
                ema_3d = cap.ema_3d

                gest = process_online_gesture_frame(
                    frame=cap.frame,
                    mp_frame=cap.mp_frame,
                    result=cap.result,
                    hands_3d_all=cap.hands_3d_all,
                    calib=boot.calib,
                    depth_aligned=cap.depth_aligned,
                    depth_raw=cap.depth_raw,
                    boot_mode_state=boot.mode_state,
                    boot_right_state=boot.right_state,
                    boot_lp_shape=boot.lp_shape,
                    boot_left_pose_state=boot.left_pose_state,
                    left_pose_runtime_armed=boot.left_pose_runtime_armed,
                    left_dual_webcam_rot_eff=boot.left_dual_webcam_rot_eff,
                    left_rot_webcam_vis_thresh=boot.left_rot_webcam_vis_thresh,
                    mode_vis_min=float(mode_vis_min),
                    open_vis_min=float(open_vis_min),
                    left_palm_basis=boot.left_palm_basis,
                    pipe=boot.pipe,
                    fps=float(fps),
                    mp_input_scale=float(_ONLINE_MP_INPUT_SCALE),
                    webcam_cap=webcam_cap,
                    webcam_landmarker=webcam_landmarker,
                    webcam_frame_idx=webcam_frame_idx,
                    wcam_rot_cache=wcam_rot_cache,
                    wcam_rot_stride=wcam_rot_stride,
                    show_webcam_preview=show_webcam_preview,
                    frame_idx=boot.frame_idx,
                    orbbec_swap_mp_hands=boot.orbbec_swap_mp_hands,
                    section=frame_prof.section,
                )
                webcam_frame_idx = gest.webcam_frame_idx
                frame_prof.section("gesture")
                frame_prof.section("left_pose_src")

                if boot.gesture_control_enabled:
                    update_live_target_from_state(
                        live_target=boot.live_target,
                        mode_state=boot.mode_state,
                        right_state=boot.right_state,
                        lp_shape=boot.lp_shape,
                        scale=boot.scale,
                        radius_mm=float(morph_radius_mm),
                        open_out=gest.open_out,
                        min_separation_m=float(min_separation_m),
                    )
                    raw_target = np.asarray(boot.live_target.get(), dtype=np.float32)
                else:
                    raw_target = enforce_min_separation(
                        lift_morph_to_hover_z(
                            np.asarray(boot.live_target.get(), dtype=np.float32),
                            float(boot.prearm_hover_z),
                        ),
                        float(min_separation_m),
                        iters=10,
                    )
                raw_target = clamp_workspace_targets(boot.swarm_workspace, raw_target)
                frame_prof.section("live_target")
                morph_targets_before_left_m = raw_target.copy()
                ls = apply_left_swarm_frame(
                    raw_target=raw_target,
                    morph_targets_before_left_m=morph_targets_before_left_m,
                    frame=cap.frame,
                    result=cap.result,
                    idx_l=gest.idx_l,
                    pts_l_pose_mm=gest.pts_l_pose_mm,
                    palm_center_depth_mm=gest.palm_center_depth_mm,
                    palm_center_color_px=gest.palm_center_color_px,
                    mph=gest.mph,
                    mpw=gest.mpw,
                    frame_idx=boot.frame_idx,
                    fps=float(fps),
                    mode_vis_min=float(mode_vis_min),
                    orbbec_vis_min_now=gest.orbbec_vis_min_now,
                    left_pose_state=boot.left_pose_state,
                    left_pose_runtime_armed=boot.left_pose_runtime_armed,
                    left_pose_reset_req=boot.left_pose_reset_req,
                    left_pose_reset_req_box=boot.left_pose_reset_req_box,
                    mode_state=boot.mode_state,
                    right_state=boot.right_state,
                    swarm_workspace=boot.swarm_workspace,
                    prev_cmd_target=boot.prev_cmd_target,
                    left_use_camera_at_arm=boot.left_use_camera_at_arm,
                    left_cam_preset=boot.left_cam_preset,
                    left_cam_y_to_world_z=left_cam_y_to_world_z,
                    left_rot_pivot_key=boot.left_rot_pivot_key,
                    left_dual_webcam_rot_eff=boot.left_dual_webcam_rot_eff,
                    left_rot_webcam_vis_thresh=boot.left_rot_webcam_vis_thresh,
                    show_webcam_preview=show_webcam_preview,
                    webcam_cap=webcam_cap,
                    webcam_landmarker=webcam_landmarker,
                    webcam_frame_idx=webcam_frame_idx,
                    mp_input_scale=float(_ONLINE_MP_INPUT_SCALE),
                    prefetch_B=gest.prefetch_B,
                    prefetch_res=gest.prefetch_res,
                    prefetch_wfr=gest.prefetch_wfr,
                    left_M_rot=boot.left_M_rot,
                    left_M_trans=boot.left_M_trans,
                    left_trans_scale=left_trans_scale,
                    left_rot_scale=left_rot_scale,
                    left_plane_rot_scale_mul=left_plane_rot_scale_mul,
                    left_trans_ema=left_trans_ema,
                    left_rot_ema=left_rot_ema,
                    left_max_offset_m=left_max_offset_m,
                    left_max_rot_rad=left_max_rot_rad,
                    left_axis_sign=left_axis_sign,
                    left_lost_decay=left_lost_decay,
                    left_rot_gate_rad=left_rot_gate_rad,
                    left_rot_gain=left_rot_gain,
                    left_rot_trans_tau_mm=left_rot_trans_tau_mm,
                    left_rot_world_z_scale=left_rot_world_z_scale,
                    left_palm_basis=boot.left_palm_basis,
                    left_axis_trans_deadzone_m=left_axis_trans_deadzone_m,
                    left_axis_rot_deadzone_rad=left_axis_rot_deadzone_rad,
                    left_axis_trans_on_m=left_axis_trans_on_m,
                    left_axis_rot_on_rad=left_axis_rot_on_rad,
                    left_axis_trans_rot_coupling=left_axis_trans_rot_coupling,
                    left_palm_depth_outlier_z_mm=left_palm_depth_outlier_z_mm,
                    left_palm_depth_outlier_lat_ratio=left_palm_depth_outlier_lat_ratio,
                    left_palm_center_depth_ema=left_palm_center_depth_ema,
                    left_pose_debug=left_pose_debug,
                    left_pose_debug_every=left_pose_debug_every,
                    left_pose_frame_viz=left_pose_frame_viz,
                    left_pose_frame_viz_every=left_pose_frame_viz_every,
                    left_axis_rot_on_rad_viz=left_axis_rot_on_rad,
                    calib=boot.calib,
                    depth_aligned=cap.depth_aligned,
                    depth_raw=cap.depth_raw,
                    section=frame_prof.section,
                )
                raw_target = ls.raw_target
                left_swarm_off = ls.left_swarm_off
                left_swarm_R = ls.left_swarm_R
                left_pose_dbg = ls.left_pose_dbg
                frame_prof.section("left_swarm")
                frame_prof.section("pose_viz")

                boot.plot_enabled = update_online_plot_frame(
                    plot_enabled=boot.plot_enabled,
                    plot_every_n=plot_every_n,
                    frame_idx=boot.frame_idx,
                    fig=boot.fig,
                    ax_hand=boot.ax_hand,
                    ax_topo=boot.ax_topo,
                    hands_3d=gest.hands_3d,
                    hands_3d_all=gest.hands_3d_all,
                    idx_l=gest.idx_l,
                    mode_state=boot.mode_state,
                    open_out=gest.open_out,
                    lp_shape=boot.lp_shape,
                    formation_rigid_3d_debug=formation_rigid_3d_debug,
                    left_pose_state=boot.left_pose_state,
                    left_pose_runtime_armed=boot.left_pose_runtime_armed,
                    morph_targets_before_left_m=morph_targets_before_left_m,
                    raw_target=raw_target,
                    left_swarm_R=left_swarm_R,
                    left_swarm_off=left_swarm_off,
                    report_panels=boot.extras.get("report_panels"),
                    report_debug_figs=boot.extras.get("report_debug_figs"),
                    pts_l_pose_mm=gest.pts_l_pose_mm,
                )
                frame_prof.section("plot3d")

                filt, boot.raw_target_filt, boot.prev_open_for_snap, boot.prev_gesture_control_enabled = (
                    filter_online_targets(
                        raw_target=raw_target,
                        raw_target_filt=boot.raw_target_filt,
                        raw_target_ema=raw_target_ema,
                        min_separation_m=min_separation_m,
                        axswarm_rt=boot.axswarm_rt,
                        gesture_control_enabled=boot.gesture_control_enabled,
                        prev_gesture_control_enabled=boot.prev_gesture_control_enabled,
                        prev_cmd_target=boot.prev_cmd_target,
                        elapsed=elapsed,
                        open_out=gest.open_out,
                        open_jump_reset=float(open_jump_reset),
                        prev_open_for_snap=boot.prev_open_for_snap,
                        spacing_audit_every=spacing_audit_every,
                        frame_idx=boot.frame_idx,
                        left_pose_runtime_armed=boot.left_pose_runtime_armed,
                        left_pose_state=boot.left_pose_state,
                        swarm_workspace=boot.swarm_workspace,
                        morph_targets_before_left_m=morph_targets_before_left_m,
                    )
                )
                frame_prof.section("target_filter")
                boot.cmd_target = filt.cmd_target
                boot.prev_cmd_target = boot.cmd_target.copy()
                just_gesture_armed = bool(boot.gesture_control_enabled) and not prev_gesture_armed
                prev_gesture_armed = bool(boot.gesture_control_enabled)

                if rigid_pose_recorder is not None and tick_rigid_pose_trace is not None:
                    rigid_pose_trace_prev_armed = tick_rigid_pose_trace(
                        rigid_pose_recorder,
                        armed_this_frame=bool(getattr(ls, "armed_this_frame", False)),
                        runtime_armed=bool(boot.left_pose_runtime_armed),
                        unwinding=bool(boot.left_pose_state.is_unwinding()),
                        frame_idx=int(boot.frame_idx),
                        t_s=float(elapsed),
                        hand_off_m=left_swarm_off,
                        hand_R=left_swarm_R,
                        raw_target=raw_target,
                        cmd_target=filt.cmd_target,
                        left_pose_state=boot.left_pose_state,
                        morph_targets_before_left_m=morph_targets_before_left_m,
                        prev_runtime_armed=bool(rigid_pose_trace_prev_armed),
                    )

                if boot.real_executor is not None:
                    from functions.real_swarm.present_real_frame import present_real_online_frame

                    present_real_online_frame(
                        frame=cap.frame,
                        real_executor=boot.real_executor,
                        cmd_target=filt.cmd_target,
                        safe_target=filt.safe_target,
                        raw_target=raw_target,
                        filter_src=filt.filter_src,
                        mode_state=boot.mode_state,
                        right_state=boot.right_state,
                        left_pose_state=boot.left_pose_state,
                        left_pose_dbg=left_pose_dbg,
                        left_pose_runtime_armed=boot.left_pose_runtime_armed,
                        axswarm_rt=boot.axswarm_rt,
                        swarm_workspace=boot.swarm_workspace,
                        gesture_control_enabled=boot.gesture_control_enabled,
                        gesture_control_enabled_box=boot.gesture_control_enabled_box,
                        just_gesture_armed=just_gesture_armed,
                        mode_raw=gest.mode_raw,
                        open_out=gest.open_out,
                        tier_count=gest.tier_count,
                        frame_idx=boot.frame_idx,
                        elapsed=elapsed,
                        center_trace=bool(center_trace),
                        center_trace_every=center_trace_every,
                        center_trace_prev=boot.center_trace_prev,
                        debug_drone_targets_every=debug_drone_targets_every,
                        min_separation_m=min_separation_m,
                        led_every_n=led_every_n,
                        section=frame_prof.section,
                    )
                else:
                    boot.render_enabled = present_online_frame(
                        frame=cap.frame,
                        sim=boot.sim,
                        cmd_target=filt.cmd_target,
                        safe_target=filt.safe_target,
                        raw_target=raw_target,
                        filter_src=filt.filter_src,
                        mode_state=boot.mode_state,
                        right_state=boot.right_state,
                        left_pose_state=boot.left_pose_state,
                        left_pose_dbg=left_pose_dbg,
                        left_pose_runtime_armed=boot.left_pose_runtime_armed,
                        axswarm_rt=boot.axswarm_rt,
                        swarm_workspace=boot.swarm_workspace,
                        gesture_control_enabled=boot.gesture_control_enabled,
                        gesture_control_enabled_box=boot.gesture_control_enabled_box,
                        mode_raw=gest.mode_raw,
                        open_out=gest.open_out,
                        tier_count=gest.tier_count,
                        frame_idx=boot.frame_idx,
                        elapsed=elapsed,
                        center_trace=bool(center_trace),
                        center_trace_every=center_trace_every,
                        center_trace_prev=boot.center_trace_prev,
                        debug_drone_targets_every=debug_drone_targets_every,
                        min_separation_m=min_separation_m,
                        led_every_n=led_every_n,
                        trail_every_n=trail_every_n,
                        sim_render_every=sim_render_every,
                        n_drones=boot.n_drones,
                        pos_buffer=boot.pos_buffer,
                        trail_rgba=boot.trail_rgba,
                        render_enabled=boot.render_enabled,
                        section=frame_prof.section,
                    )

                ui_key = poll_cv_key(
                    cv_poll_key=cv_poll_key,
                    imshow=(boot.frame_idx % imshow_every) == 0,
                    window=ocv_window_title,
                    frame=cap.frame,
                )
                if _poll_keys(ui_key):
                    break
                frame_prof.frame_end(boot.frame_idx)
                sync_armed_flags(boot)
                boot.frame_idx += 1

            if webcam_landmarker is not None:
                try:
                    webcam_landmarker.close()
                except Exception:
                    pass
            if webcam_cap is not None:
                webcam_cap.release()
            boot.key_queue.stop()
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user, stopping online control...")
    finally:
        if rigid_pose_recorder is not None:
            try:
                rigid_pose_recorder.on_exit(
                    frame_idx=int(getattr(boot, "frame_idx", 0)),
                    t_s=float(time.monotonic() - boot.start_time),
                )
            except Exception as exc:
                print(f"[rigid-pose-trace] save failed: {exc}", flush=True)
        boot.key_queue.stop()
        try:
            boot.k4a.stop()
        except Exception:
            pass
        if boot.real_executor is not None:
            try:
                boot.real_executor.land_and_close()
            except Exception as exc:
                print(f"[WARN] Real swarm shutdown failed: {exc}")
        elif boot.sim is not None:
            boot.sim.close()
        report_figs = boot.extras.get("report_debug_figs")
        if report_figs is not None:
            close_report_debug_figures(report_figs)
        else:
            close_3d_plot(boot.fig)
        cv2.destroyAllWindows()


__all__ = [
    "FrameSectionProfiler",
    "LiveTargetState",
    "OnlineKeyQueue",
    "PipelineTuning",
    "ScaleConfig",
    "apply_online_control_key",
    "closest_pair",
    "complete_prearm_takeoff",
    "format_hotkey_install_hint",
    "init_3d_plot",
    "main",
    "make_initial_live_target",
    "online_pipeline_defaults",
    "process_online_control_keys",
    "probe_global_hotkey_backends",
    "run_integrated_online_control",
    "try_install_hotkey_dependencies",
    "update_live_target_from_state",
]


def main() -> None:
    parser = build_online_control_parser()
    args = parser.parse_args()

    n_default = int(max(8, int(args.point_count)))
    if n_default < 8:
        parser.error("--point-count must be >= 8")
    if sys.stdin.isatty() and not args.drones_config:
        point_count = int(prompt_and_init_fixed_surface_points(default_n=n_default))
    else:
        init_fixed_surface_points(n_default)
        point_count = n_default
        if args.drones_config:
            from functions.real_swarm.swarm_config import load_drones_config

            _drones, _, _ = load_drones_config(args.drones_config)
            print(
                f"Fixed surface samples: morph n={point_count} (virtual formation); "
                f"physical Crazyflies n={len(_drones)} from {args.drones_config} "
                f"(receive cmd_target indices 0..{len(_drones) - 1})."
            )
        else:
            print(f"Fixed surface samples initialized (non-interactive): n={point_count}")
    box_m = float(args.swarm_workspace_box_m)
    wall_m = float(args.swarm_workspace_wall_margin_m)
    formation_cap_m = box_m - 2.0 * wall_m if box_m > 0.0 else None
    scale = ScaleConfig(
        xy_radius=float(args.xy_radius),
        z_center=float(args.z_center),
        z_amplitude=float(args.z_amplitude),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        reference_xy_extent_mm=float(args.reference_xy_extent_mm),
        reference_z_extent_mm=float(args.reference_z_extent_mm),
        z_mm_scale=float(args.z_mm_scale),
        morph_world_scale=float(args.morph_world_scale),
        formation_max_extent_m=formation_cap_m,
    )
    live_target = make_initial_live_target(
        point_count=point_count,
        radius_mm=float(args.radius_mm),
        morph_mode=int(args.mode),
        open_alpha=float(args.open_alpha),
        shape_t=args.shape_t,
        scale=scale,
    )
    if args.print_only:
        return

    left_trans_scale = args.left_trans_scale
    if left_trans_scale is None:
        left_trans_scale = _DEFAULT_LEFT_TRANS_SCALE_MM
    left_axis_sign = (
        -1.0 if args.left_flip_x else 1.0,
        -1.0 if args.left_flip_y else 1.0,
        -1.0 if args.left_flip_z else 1.0,
    )
    panels = report_debug_panels_from_args(args)
    if resolve_pipeline_tuning is None:
        pipeline = online_pipeline_defaults()
    else:
        pipeline = resolve_pipeline_tuning(
            debug_webcam_pipeline=bool(args.debug_webcam_pipeline),
            plot_every_cli=int(args.plot_every),
            debug_3d_plot=bool(args.debug_3d_plot)
            or bool(args.formation_rigid_3d_debug)
            or panels.any_enabled(),
        )
    if pipeline.plot_every_n > 0:
        print(f"Matplotlib 3D topo plot enabled (plot-every={pipeline.plot_every_n}).")
    if panels.any_enabled():
        print(
            f"Debug report panels: {', '.join(panels.enabled_labels())}. "
            "Tip: enable ONE panel per run for screenshots; use --plot-every 8+ to reduce lag."
        )

    run_integrated_online_control(
        live_target=live_target,
        point_count=point_count,
        duration=float(args.duration),
        fps=int(args.fps),
        min_separation_m=float(args.min_separation_m),
        model_path=args.model,
        plot_every_n=int(pipeline.plot_every_n),
        scale=scale,
        trail_every_n=int(args.trail_every),
        led_every_n=int(args.led_every),
        sim_render_every=int(args.sim_render_every),
        morph_radius_mm=float(args.radius_mm),
        drone_model=str(args.drone_model),
        max_sim_substeps=int(args.max_sim_substeps),
        imshow_every=int(args.imshow_every),
        mp_detect_every=int(args.mp_detect_every),
        raw_target_ema=float(args.raw_target_ema),
        left_swarm_pose=bool(args.left_swarm_pose),
        left_trans_scale=float(left_trans_scale),
        left_rot_scale=float(args.left_rot_scale),
        left_trans_ema=float(args.left_trans_ema),
        left_rot_ema=float(args.left_rot_ema),
        left_max_offset_m=float(args.left_max_offset_m),
        left_max_rot_rad=float(args.left_max_rot_rad),
        left_axis_sign=left_axis_sign,
        left_lost_decay=float(args.left_lost_decay),
        debug_drone_targets_every=int(args.debug_drone_targets_every),
        spacing_audit_every=int(args.spacing_audit_every),
        prearm_hover_z=float(args.prearm_hover_z),
        prearm_takeoff_z=float(args.prearm_takeoff_z),
        open_jump_reset=float(args.open_jump_reset),
        left_unwind_s=float(args.left_unwind_seconds),
        left_swarm_depth_frame_motion=bool(args.left_swarm_depth_frame_motion),
        left_rot_gate_rad=float(np.radians(float(args.left_rot_gate_deg))),
        left_yaw_min_horiz=float(args.left_yaw_min_horiz),
        left_rot_gain=float(args.left_rot_gain),
        left_rot_trans_tau_mm=float(args.left_rot_trans_tau_mm),
        left_rot_world_z_scale=float(args.left_rot_world_z_scale),
        left_cam_y_to_world_z=float(args.left_cam_y_to_world_z),
        left_cam_preset=str(args.left_cam_preset).strip().lower(),
        left_world_frame=str(args.left_world_frame).strip().lower(),
        left_axis_trans_deadzone_m=float(args.left_axis_trans_deadzone_m),
        left_axis_rot_deadzone_rad=float(np.radians(args.left_axis_rot_deadzone_deg)),
        left_axis_trans_on_m=float(args.left_axis_trans_on_m),
        left_axis_rot_on_rad=float(np.radians(args.left_axis_rot_on_deg)),
        left_axis_trans_rot_coupling=float(args.left_trans_rot_coupling),
        install_hotkey_deps=bool(args.install_hotkey_deps),
        left_palm_basis=str(args.left_palm_basis).strip().lower(),
        left_plane_rot_scale_mul=float(args.left_plane_rot_scale_mul),
        left_rot_direct_follow=bool(args.left_rot_direct_follow),
        left_rot_pivot=str(args.left_rot_pivot).strip().lower(),
        left_dual_webcam_rot=not bool(args.no_left_dual_webcam_rot),
        left_rot_webcam_vis_thresh=float(args.left_rot_webcam_vis_thresh),
        left_rot_webcam_index=int(args.left_rot_webcam_index),
        orbbec_flip_horizontal=bool(args.orbbec_flip_horizontal),
        orbbec_use_transformed_depth=bool(args.orbbec_use_transformed_depth),
        orbbec_hand_swap=str(args.orbbec_hand_swap).strip().lower(),
        formation_rigid_3d_debug=bool(args.formation_rigid_3d_debug),
        left_pose_frame_viz=bool(args.left_pose_frame_viz) or panels.palm,
        left_pose_frame_viz_every=int(args.left_pose_frame_viz_every),
        left_pose_debug=bool(args.left_pose_debug),
        left_pose_debug_every=int(args.left_pose_debug_every),
        webcam_rot_stride=int(args.webcam_rot_stride),
        show_webcam_preview=bool(args.show_webcam_preview),
        global_hotkeys=not bool(args.no_global_hotkeys),
        mode_vis_min=float(args.mode_vis_min),
        open_vis_min=float(args.open_vis_min),
        center_trace=bool(args.center_trace),
        center_trace_every=int(args.center_trace_every),
        rigid_pose_trace=bool(args.rigid_pose_trace),
        rigid_pose_trace_out=args.rigid_pose_trace_out,
        rigid_pose_trace_every=int(args.rigid_pose_trace_every),
        planner=str(args.planner),
        axswarm_settings=args.axswarm_settings,
        axswarm_project_root=args.axswarm_project_root,
        axswarm_max_iters=args.axswarm_max_iters,
        axswarm_max_deviation_m=float(args.axswarm_max_deviation_m),
        axswarm_pos_weight=args.axswarm_pos_weight,
        axswarm_max_solve_ms=float(args.axswarm_max_solve_ms),
        profile_frame=bool(args.profile_frame),
        profile_every=int(args.profile_every),
        swarm_workspace_box_m=float(args.swarm_workspace_box_m),
        swarm_workspace_wall_margin_m=float(args.swarm_workspace_wall_margin_m),
        swarm_workspace_clear_margin_m=float(args.swarm_workspace_clear_margin_m),
        swarm_workspace_mode=str(args.swarm_workspace_mode),
        left_palm_depth_outlier_z_mm=float(args.left_palm_depth_outlier_z_mm),
        left_palm_depth_outlier_lat_ratio=float(args.left_palm_depth_outlier_lat_ratio),
        left_palm_center_depth_ema=float(args.left_palm_center_depth_ema),
        draw_hand_debug=bool(args.draw_hand_debug) or panels.hand,
        report_panels=panels if panels.any_enabled() else None,
        mp_delegate=str(args.mp_delegate),
        pipeline_tuning=pipeline,
        drones_config=str(args.drones_config) if args.drones_config else None,
        real_lighthouse=args.real_lighthouse,
    )


if __name__ == "__main__":
    main()
