import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

try:
    from pyk4a import Config, FPS, PyK4A
except ImportError as exc:
    raise ImportError(
        "pyk4a is required. Install Azure/Orbbec K4A python bindings first."
    ) from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def depth_to_colormap(depth_u16: np.ndarray, max_depth_mm: int) -> np.ndarray:
    clipped = np.clip(depth_u16, 0, max_depth_mm).astype(np.uint16)
    depth_8u = cv2.convertScaleAbs(clipped, alpha=255.0 / max_depth_mm)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


def save_frame_data(
    index: int,
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    output_dir: Path,
    max_depth_mm: int,
) -> dict:
    rgb_path = output_dir / f"frame_{index:03d}_rgb.png"
    depth_png_path = output_dir / f"frame_{index:03d}_depth_raw.png"
    depth_npy_path = output_dir / f"frame_{index:03d}_depth.npy"
    merge_path = output_dir / f"frame_{index:03d}_rgb_depth_vis.png"

    depth_color = depth_to_colormap(depth_u16, max_depth_mm)
    merged = np.hstack((color_bgr, depth_color))

    cv2.imwrite(str(rgb_path), color_bgr)
    cv2.imwrite(str(depth_png_path), depth_u16)  # keep uint16 depth
    np.save(depth_npy_path, depth_u16)
    cv2.imwrite(str(merge_path), merged)

    valid_depth = depth_u16[depth_u16 > 0]
    center_depth = int(depth_u16[depth_u16.shape[0] // 2, depth_u16.shape[1] // 2])
    stats = {
        "frame_id": index,
        "rgb_path": rgb_path.name,
        "depth_png_path": depth_png_path.name,
        "depth_npy_path": depth_npy_path.name,
        "merge_path": merge_path.name,
        "height": int(depth_u16.shape[0]),
        "width": int(depth_u16.shape[1]),
        "valid_depth_count": int(valid_depth.size),
        "depth_min_mm": int(valid_depth.min()) if valid_depth.size else 0,
        "depth_max_mm": int(valid_depth.max()) if valid_depth.size else 0,
        "depth_mean_mm": float(valid_depth.mean()) if valid_depth.size else 0.0,
        "center_depth_mm": center_depth,
    }
    return stats


def write_stats_csv(rows: list[dict], output_csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture RGBD frames and export RGB/depth images with depth data."
    )
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to save.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rgbd_output",
        help="Directory to save rgb/depth files.",
    )
    parser.add_argument("--width", type=int, default=640, help="Unused in pyk4a mode.")
    parser.add_argument("--height", type=int, default=480, help="Unused in pyk4a mode.")
    parser.add_argument(
        "--fps", type=int, default=30, choices=[5, 15, 30], help="Camera FPS: 5/15/30."
    )
    parser.add_argument(
        "--color-resolution",
        type=int,
        default=1,
        help="K4A color resolution enum value (same style as test_orbbec.py).",
    )
    parser.add_argument(
        "--depth-mode",
        type=int,
        default=2,
        help="K4A depth mode enum value (same style as test_orbbec.py).",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show live preview windows while capturing.",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=int,
        default=3000,
        help="Max depth (mm) for depth visualization.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fps_map = {5: FPS.FPS_5, 15: FPS.FPS_15, 30: FPS.FPS_30}
    camera_fps = fps_map[args.fps]

    # Keep the same camera opening method as crazyflow/test_orbbec.py
    k4a = PyK4A(
        Config(
            color_resolution=args.color_resolution,
            depth_mode=args.depth_mode,
            synchronized_images_only=False,
            camera_fps=camera_fps,
        )
    )

    saved_rows = []
    try:
        k4a.start()
        print("Camera started.")
        print(f"Start capture, target frames: {args.frames}")
        i = 0
        while i < args.frames:
            capture = k4a.get_capture()

            if capture.color is None or capture.depth is None:
                print(f"[WARN] skip frame {i}, invalid frame.")
                continue

            color = capture.color
            depth_u16 = capture.depth

            # pyk4a color is usually BGRA; convert to BGR for OpenCV saving/visualization.
            if color.ndim == 3 and color.shape[2] == 4:
                color_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            else:
                color_bgr = color

            if (
                color_bgr.shape[1] != depth_u16.shape[1]
                or color_bgr.shape[0] != depth_u16.shape[0]
            ):
                color_bgr = cv2.resize(
                    color_bgr, (depth_u16.shape[1], depth_u16.shape[0]), interpolation=cv2.INTER_LINEAR
                )

            stats = save_frame_data(i, color_bgr, depth_u16, output_dir, args.max_depth_mm)
            saved_rows.append(stats)

            if args.show_preview:
                depth_vis = depth_to_colormap(depth_u16, args.max_depth_mm)
                cv2.imshow("color", color_bgr)
                cv2.imshow("depth", depth_vis)
                if cv2.waitKey(1) == 27:
                    print("ESC pressed, early stop.")
                    break

            print(
                f"[{i + 1}/{args.frames}] saved: "
                f"{stats['rgb_path']}, {stats['depth_png_path']}, {stats['depth_npy_path']}"
            )
            i += 1

        csv_path = output_dir / "frame_stats.csv"
        write_stats_csv(saved_rows, csv_path)
        print(f"Saved statistics: {csv_path}")
        print(f"All output directory: {output_dir.resolve()}")
    finally:
        k4a.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
