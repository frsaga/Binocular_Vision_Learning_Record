import cv2
import os
from datetime import datetime

# 设置保存路径（与你之前的代码保持一致）
left_folder = "E:\\1DA2166"
right_folder = "E:\\2DA56"

# 确保保存目录存在
os.makedirs(left_folder, exist_ok=True)
os.makedirs(right_folder, exist_ok=True)

# 打开两个相机（默认左为0，右为1，可根据设备调整）
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# 检查相机是否打开成功
if not cap_left.isOpened() or not cap_right.isOpened():
    print("无法打开相机，请检查设备连接！")
    exit()

print("按下 's' 拍照并保存，按下 'q' 退出程序")

count = 0
while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not ret_l or not ret_r:
        print("无法读取图像帧")
        break

    # 显示图像
    cv2.imshow("Left Camera", frame_l)
    cv2.imshow("Right Camera", frame_r)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # 拍照保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_path = os.path.join(left_folder, f"left_{timestamp}.png")
        right_path = os.path.join(right_folder, f"right_{timestamp}.png")
        cv2.imwrite(left_path, frame_l)
        cv2.imwrite(right_path, frame_r)
        print(f"已保存：\n{left_path}\n{right_path}")
        count += 1

    elif key == ord('q'):
        break

# 释放资源
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
