from pyk4a import PyK4A, Config
import cv2
import numpy as np

k4a = PyK4A(
    Config(
        color_resolution=1,
        depth_mode=2,
        synchronized_images_only=False,
    )
)

k4a.start()

print("Running... ESC to quit")

while True:
    capture = k4a.get_capture()

    if capture.depth is not None:
        depth = capture.depth

        print("min/max:", depth.min(), depth.max())

        depth_vis = (depth / 4000 * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("depth", depth_vis)

    if capture.color is not None:
        color = capture.color
        cv2.imshow("color", color)

    if cv2.waitKey(1) == 27:
        break

k4a.stop()
cv2.destroyAllWindows()