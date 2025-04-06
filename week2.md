#  新平台搭建
![a0c2e9448e75351e1a663cb1156e4ea](https://github.com/user-attachments/assets/ce80ef07-f131-4f5d-a4c3-bbd5027360c1)
使用三脚架+玻纤板固定，固定基线=8cm

##  极线矫正
参考资料
https://blog.csdn.net/qq_42722197/article/details/118663803
为什么要做极线校正？

三维重建是通过双目立体匹配实现的如图1，通过匹配空间中点 在两个图像中的投影点，再根据三角关系得到P的Z值。
![844d8b8ff23768c854560eef2818d7e0](https://github.com/user-attachments/assets/71c2bec1-424e-44c8-a2ba-bffd7b3a545d)
![abb4368be7cec7bf1e469445433efd1f](https://github.com/user-attachments/assets/f6c6776a-3c37-4a0c-b93d-22e78139f39b)
我们双目相机拍摄的时候实际情况下如下图a,两个图像做匹配时如我们图中蓝色箭头指示的匹配点那样，需要在全图中进行查找。但是如果我们对相机进行校正，使得它们成像面平行且行对齐如下图b，匹配点在同一行。那么我们只需在同行上查找，大大节约时间。因此，极线校正目的是对两幅图像的二维匹配搜索变成一维，节省计算量，排除虚假匹配点
![ae266f86b27b59575e87191112e2bf41](https://github.com/user-attachments/assets/6ff0e242-84b2-436f-9572-4d7583fb66a5)

极线校正怎么实现？

1.三角测量原理
假设我们两台摄像机像平面精准位于同一平面上，且行对齐，两个光轴严格平行。
![784ac64eacb0928fc03697340ce397bf](https://github.com/user-attachments/assets/14212fc4-ac7e-41dc-aab3-0933e14da819)

转换矩阵Q求解
在原有的基础上添加极线矫正并输出图像的代码

```python
    # 拼接图像
    combined_img = np.hstack((rectified_img_l, rectified_img_r))

    # 画极线
    num_lines = 20  # 你可以调整极线数量
    step = combined_img.shape[0] // num_lines  # 根据图像高度均匀分布
    for i in range(num_lines):
        y = i * step
        cv2.line(combined_img, (0, y), (combined_img.shape[1], y), (0, 255, 0), 1)



    # 显示矫正后的图像
    cv2.imshow('Rectified Images with Epilines', combined_img)
    cv2.imwrite("stereo_epilines.png", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 加载并矫正示例图像
example_idx = 0
img_l = img_left[example_idx][0]
img_r = img_right[example_idx][0]
stereo_rectify_and_display_with_epilines(img_l, img_r, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T)

```

校正结果
![f4182580fda69ca55f11b0cd5e0f895](https://github.com/user-attachments/assets/5257eb21-c07e-4876-b562-8afd02374624)


##  相机实时取流以及拍照

![32e03fc3f1fdad8d0f7d7ffe522b732](https://github.com/user-attachments/assets/00fae35f-5d58-41a1-bfd2-ee6070c6bbc5)


```python
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

```
