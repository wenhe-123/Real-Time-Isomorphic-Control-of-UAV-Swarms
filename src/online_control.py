"""Online Crazyflow + Orbbec hand control (use ``online_control_dual.py`` by default).

Main loop only. Logic lives under ``functions/{mode_switch,open_close,swarm_motion,display_sim,dual_cam,runtime}/``.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
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
from functions.runtime.online_runtime_config import (
    OnlineRuntimeConfig,
    OnlineWebcamState,
    build_online_runtime_config,
)
from functions.runtime.online_cli_args import build_online_control_parser, report_debug_panels_from_args
from functions.runtime.online_defaults import _WCAM_PREVIEW_WINDOW
from functions.runtime.online_targets import make_initial_live_target, update_live_target_from_state
from functions.runtime.pipeline_tuning import PipelineTuning, online_pipeline_defaults
from functions.display_sim.online_present_input import PresentFrameInput
from functions.swarm_motion.online_frame_filter import filter_online_targets
from functions.swarm_motion.online_left_swarm_frame import apply_left_swarm_frame, clamp_workspace_targets
from functions.swarm_motion.prearm import complete_prearm_takeoff
from functions.swarm_motion.spacing_guard import closest_pair, enforce_min_separation, enforce_min_separation_xy

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
    cfg: OnlineRuntimeConfig,
    *,
    duration: float = 0.0,
    model_path: str | None = None,
    mp_delegate: str = "cpu",
    rigid_pose_trace: bool = False,
    rigid_pose_trace_out: str | None = None,
    rigid_pose_trace_every: int = 1,
) -> None:
    """Run MediaPipe, Matplotlib, OpenCV, and Crazyflow or real Crazyflie in the main thread."""
    resolved_model = resolve_model_path(model_path, __file__)
    mp_delegate_key = str(mp_delegate).strip().lower()
    print_pipeline_mode(cfg.pipe, draw_hand_debug=cfg.draw_hand_debug, mp_delegate_key=mp_delegate_key)

    boot = boot_online_control(live_target=live_target, cfg=cfg)
    _poll_keys = make_key_poller(boot, global_hotkeys=cfg.global_hotkeys)
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
                "left_cam_preset": cfg.left_cam_preset,
                "left_world_frame": cfg.left_world_frame,
                "left_trans_scale": float(cfg.left_trans_scale),
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
            webcam = OnlineWebcamState(rot_stride=cfg.webcam_rot_stride)
            if boot.left_dual_webcam_rot_eff:
                try:
                    webcam.cap, _widx, _wb = open_webcam_capture(
                        cfg.left_rot_webcam_index, 0, 0, 8
                    )
                    webcam.landmarker = create_hand_landmarker(
                        resolved_model, delegate=mp_delegate_key
                    )
                    print(
                        f"Dual USB webcam index {_widx} ({_wb}): palm rotation when Orbbec "
                        f"visibility < {boot.left_rot_webcam_vis_thresh:.2f}; mode M1–M5 assist when "
                        f"visibility < {cfg.mode_vis_min:.2f} or while rotating; translation stays depth."
                    )
                except Exception as exc:
                    print(f"[WARN] Dual webcam rotation disabled: {exc}")
                    boot.left_dual_webcam_rot_eff = False
                    webcam.cap = None
                    if webcam.landmarker is not None:
                        try:
                            webcam.landmarker.close()
                        except Exception:
                            pass
                        webcam.landmarker = None
            try:
                cv2.namedWindow(ocv_window_title, cv2.WINDOW_NORMAL)
            except Exception:
                pass
            if cfg.show_webcam_preview and boot.left_dual_webcam_rot_eff:
                try:
                    cv2.namedWindow(_WCAM_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
                except Exception:
                    pass
            frame_prof = FrameSectionProfiler(
                enabled=cfg.profile_frame,
                report_every=max(1, int(cfg.profile_every)),
            )
            if frame_prof.enabled:
                print(
                    f"Frame profiling ON (perf_counter); report every "
                    f"{frame_prof.report_every} completed frames."
                )
            if cfg.mp_detect_every > 1:
                print(
                    f"MediaPipe: detect every {cfg.mp_detect_every} frames "
                    f"(reuse landmarks on skip; Orbbec depth pose still each frame)."
                )
            if cfg.imshow_every > 1:
                print(
                    f"Orbbec preview: imshow every {cfg.imshow_every} frames "
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
            boot.prev_prearm_climb_enabled = False

            while True:
                elapsed = time.monotonic() - boot.start_time
                if float(duration) > 0.0 and elapsed > float(duration):
                    break
                if _poll_keys():
                    break
                just_prearm_climb = bool(boot.prearm_climb_enabled_box[0]) and not boot.prev_prearm_climb_enabled
                just_prearm_descend = boot.prev_prearm_climb_enabled and not bool(
                    boot.prearm_climb_enabled_box[0]
                )
                sync_armed_flags(boot)

                frame_prof.frame_start()
                t_ms = int(boot.frame_idx * (1000 / max(float(cfg.fps), 1.0)))
                cap, poll_frame, boot.orbbec_flip_depth_warned = grab_orbbec_mp_frame(
                    k4a=boot.k4a,
                    landmarker=landmarker,
                    frame_idx=boot.frame_idx,
                    t_ms=t_ms,
                    fps=float(cfg.fps),
                    mp_detect_every=cfg.mp_detect_every,
                    mp_input_scale=float(cfg.mp_input_scale),
                    orbbec_flip_horizontal=cfg.orbbec_flip_horizontal,
                    orbbec_use_transformed_depth=cfg.orbbec_use_transformed_depth,
                    use_depth_fusion=boot.use_depth_fusion,
                    pipe=boot.pipe,
                    calib=boot.calib,
                    draw_hand_debug=cfg.draw_hand_debug,
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
                    boot=boot,
                    cfg=cfg,
                    cap=cap,
                    webcam=webcam,
                    section=frame_prof.section,
                )
                webcam.frame_idx = gest.webcam_frame_idx
                frame_prof.section("gesture")
                frame_prof.section("left_pose_src")

                if boot.gesture_control_enabled:
                    update_live_target_from_state(
                        live_target=boot.live_target,
                        mode_state=boot.mode_state,
                        right_state=boot.right_state,
                        lp_shape=boot.lp_shape,
                        scale=boot.scale,
                        radius_mm=float(cfg.morph_radius_mm),
                        open_out=gest.open_out,
                        min_separation_m=float(cfg.min_separation_m),
                    )
                    raw_target = np.asarray(boot.live_target.get(), dtype=np.float32)
                elif boot.prearm_climb_enabled:
                    raw_target = enforce_min_separation(
                        boot.prearm_hover_layout.copy(),
                        float(cfg.min_separation_m),
                        iters=10,
                    )
                else:
                    raw_target = enforce_min_separation_xy(
                        boot.ground_layout.copy(),
                        float(cfg.min_separation_m),
                        float(boot.ground_z),
                        iters=10,
                    )
                raw_target = clamp_workspace_targets(boot.swarm_workspace, raw_target)
                frame_prof.section("live_target")
                morph_targets_before_left_m = raw_target.copy()
                ls = apply_left_swarm_frame(
                    boot=boot,
                    cap=cap,
                    gest=gest,
                    raw_target=raw_target,
                    morph_targets_before_left_m=morph_targets_before_left_m,
                    webcam=webcam,
                    section=frame_prof.section,
                )
                raw_target = ls.raw_target
                left_swarm_off = ls.left_swarm_off
                left_swarm_R = ls.left_swarm_R
                left_pose_dbg = ls.left_pose_dbg
                frame_prof.section("left_swarm")
                frame_prof.section("pose_viz")

                webcam.frame_idx = ls.webcam_frame_idx
                boot.plot_enabled = update_online_plot_frame(
                    boot=boot,
                    cfg=cfg,
                    gest=gest,
                    raw_target=raw_target,
                    morph_targets_before_left_m=morph_targets_before_left_m,
                    left_swarm_R=left_swarm_R,
                    left_swarm_off=left_swarm_off,
                )
                frame_prof.section("plot3d")

                axswarm_track_pos: np.ndarray | None = None
                if boot.axswarm_rt is not None:
                    if boot.real_executor is not None:
                        axswarm_track_pos = boot.real_executor.get_sim_track_positions(
                            boot.prev_cmd_target,
                            boot.n_drones,
                        )
                    elif boot.sim is not None:
                        axswarm_track_pos = np.asarray(
                            boot.sim.data.states.pos[0], dtype=np.float32
                        )

                if (
                    (just_prearm_climb or just_prearm_descend)
                    and boot.axswarm_rt is not None
                ):
                    _layout_pos = (
                        axswarm_track_pos
                        if axswarm_track_pos is not None
                        else boot.prev_cmd_target
                    )
                    boot.axswarm_rt.sync_gesture(
                        np.asarray(_layout_pos, dtype=np.float32),
                        np.zeros((boot.n_drones, 3), dtype=np.float32),
                    )

                filt, boot.raw_target_filt, boot.prev_open_for_snap, boot.prev_gesture_control_enabled = (
                    filter_online_targets(
                        boot=boot,
                        cfg=cfg,
                        gest=gest,
                        raw_target=raw_target,
                        morph_targets_before_left_m=morph_targets_before_left_m,
                        elapsed=elapsed,
                        track_pos=axswarm_track_pos,
                    )
                )
                frame_prof.section("target_filter")
                boot.cmd_target = filt.cmd_target
                boot.prev_cmd_target = boot.cmd_target.copy()
                boot.prev_prearm_climb_enabled = bool(boot.prearm_climb_enabled)
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

                present_inp = PresentFrameInput(
                    frame=cap.frame,
                    boot=boot,
                    cfg=cfg,
                    gest=gest,
                    filt=filt,
                    raw_target=raw_target,
                    left_pose_dbg=left_pose_dbg,
                    elapsed=elapsed,
                    just_gesture_armed=just_gesture_armed,
                    section=frame_prof.section,
                )
                if boot.real_executor is not None:
                    from functions.real_swarm.present_real_frame import present_real_online_frame

                    present_real_online_frame(present_inp)
                else:
                    boot.render_enabled = present_online_frame(present_inp)

                ui_key = poll_cv_key(
                    cv_poll_key=cv_poll_key,
                    imshow=(boot.frame_idx % cfg.imshow_every) == 0,
                    window=ocv_window_title,
                    frame=cap.frame,
                )
                if _poll_keys(ui_key):
                    break
                frame_prof.frame_end(boot.frame_idx)
                sync_armed_flags(boot)
                boot.frame_idx += 1

            if webcam.landmarker is not None:
                try:
                    webcam.landmarker.close()
                except Exception:
                    pass
            if webcam.cap is not None:
                webcam.cap.release()
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
            with warnings.catch_warnings():
                try:
                    from glfw import GLFWError

                    warnings.filterwarnings("ignore", category=GLFWError)
                except ImportError:
                    pass
                warnings.filterwarnings("ignore", message=".*GLFW.*")
                try:
                    boot.sim.close()
                except Exception:
                    pass
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
    "OnlineRuntimeConfig",
    "PipelineTuning",
    "ScaleConfig",
    "apply_online_control_key",
    "build_online_runtime_config",
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

    cfg = build_online_runtime_config(
        args,
        point_count=point_count,
        scale=scale,
        pipeline=pipeline,
        panels=panels,
    )
    run_integrated_online_control(
        live_target=live_target,
        cfg=cfg,
        duration=float(args.duration),
        model_path=args.model,
        mp_delegate=str(args.mp_delegate),
        rigid_pose_trace=bool(args.rigid_pose_trace),
        rigid_pose_trace_out=args.rigid_pose_trace_out,
        rigid_pose_trace_every=int(args.rigid_pose_trace_every),
    )


if __name__ == "__main__":
    main()
