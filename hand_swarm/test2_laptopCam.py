import cv2
import mediapipe as mp
import numpy as np

# ===== 初始化 =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
draw = mp.solutions.drawing_utils

# ===== 读取RGB =====
color = cv2.imread("color.png")
color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

h, w, _ = color.shape

# ===== 读取Depth（关键：保持16bit）=====
depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)

print("Depth dtype:", depth.dtype)  # 应该是 uint16

# ===== MediaPipe检测 =====
results = hands.process(color_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:

        # 画点
        draw.draw_landmarks(color, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        print("\n=== Hand keypoints ===")

        # ===== 遍历21个关键点 =====
        for i, lm in enumerate(hand_landmarks.landmark):

            # 像素坐标
            x = int(lm.x * w)
            y = int(lm.y * h)

            # 防越界
            if x >= w or y >= h:
                continue

            # 深度值
            z = depth[y, x]

            print(f"id:{i:2d}  x:{x:4d} y:{y:4d} depth:{z}")

            # 可视化：标点
            cv2.circle(color, (x, y), 5, (0, 255, 0), -1)

# ===== 显示 =====
cv2.imshow("RGB + Keypoints", color)
cv2.waitKey(0)
cv2.destroyAllWindows()