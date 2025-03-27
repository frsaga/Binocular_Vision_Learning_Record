import cv2
import os
import numpy as np
import itertools
import yaml

# 定义文件夹路径
left_folder = "E:\\1DA2166"
right_folder = "E:\\2DA56"

# 获取图像文件列表并排序
left_images = sorted(os.listdir(left_folder))
right_images = sorted(os.listdir(right_folder))

# 确保左右相机图像数量一致
assert len(left_images) == len(right_images), "左右相机图像数量不一致"

# 加载两个摄像头图片文件夹并将里面的彩图转换为灰度图
def load_images(folder, images):
    img_list = []
    for img_name in images:
        img_path = os.path.join(folder, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_list.append((frame, gray))
        else:
            print(f"无法读取图像: {img_path}")
    return img_list



# 检测棋盘格角点
def get_corners(imgs, pattern_size):
    corners = []
    for frame, gray in imgs:
        ret, c = cv2.findChessboardCorners(gray, pattern_size)     #ret 表示是否成功找到棋盘格角点，c 是一个数组，包含了检测到的角点的坐标
        if not ret:
            print("未能检测到棋盘格角点")
            continue
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))     #cv2.cornerSubPix 函数用于提高棋盘格角点的精确度，对初始检测到的角点坐标 c 进行优化
        corners.append(c)      #将优化后的角点坐标 c 添加到 corners 列表中

        # 绘制角点并显示
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, pattern_size, c, ret)
        new_size = (1440, 1080)
        resized_img = cv2.resize(vis, new_size)
        cv2.imshow('Corners', resized_img)
        cv2.waitKey(150)

    return corners

# 相机标定
def calibrate_camera(object_points, corners, imgsize):
    cm_input = np.eye(3, dtype=np.float32)
    ret = cv2.calibrateCamera(object_points, corners, imgsize, cm_input, None)
    return ret

def save_calibration_to_yaml(file_path, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T, E, F):
    data = {
        'camera_matrix_left': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_l.flatten().tolist()
        },
        'dist_coeff_left': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_l.flatten().tolist()
        },
        'camera_matrix_right': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_r.flatten().tolist()
        },
        'dist_coeff_right': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_r.flatten().tolist()
        },
        'R': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': R.flatten().tolist()
        },
        'T': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': T.flatten().tolist()
        },
        'E': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': E.flatten().tolist()
        },
        'F': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': F.flatten().tolist()
        }
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Calibration parameters saved to {file_path}")



img_left = load_images(left_folder, left_images)      #img_left是个列表，存放左摄像头所有的灰度图片。
img_right = load_images(right_folder, right_images)
pattern_size = (11, 8)
corners_left = get_corners(img_left, pattern_size)       #corners_left的长度表示检测到棋盘格角点的图像数量。corners_left[i] 和 corners_right[i] 中存储了第 i 张图像检测到的棋盘格角点的二维坐标。
corners_right = get_corners(img_right, pattern_size)
cv2.destroyAllWindows()

# 断言，确保所有图像都检测到角点
assert len(corners_left) == len(img_left), "有图像未检测到左相机的角点"
assert len(corners_right) == len(img_right), "有图像未检测到右相机的角点"

# 准备标定所需数据
points = np.zeros((11 * 8, 3), dtype=np.float32)   #创建40 行 3 列的零矩阵，用于存储棋盘格的三维坐标点。棋盘格的大小是 8 行 5 列，40 个角点。数据类型为 np.float32，这是一张图的，因为一个角点对应一个三维坐标
points[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 20  #给这些点赋予实际的物理坐标，* 21 是因为每个棋盘格的大小为 21mm

object_points = [points] * len(corners_left)     #包含了所有图像中棋盘格的三维物理坐标点 points。这里假设所有图像中棋盘格的物理坐标是相同的，因此用 points 复制 len(corners_left) 次。
imgsize = img_left[0][1].shape[::-1]     #img_left[0] 是左相机图像列表中的第一张图像。img_left[0][1] 是该图像的灰度图像。shape[::-1] 取灰度图像的宽度和高度，并反转顺序，以符合 calibrateCamera 函数的要求。

print('开始左相机标定')
ret_l = calibrate_camera(object_points, corners_left, imgsize)    #object_points表示标定板上检测到的棋盘格角点的三维坐标；corners_left[i]表示棋盘格角点在图像中的二维坐标；imgsize表示图像大小
retval_l, cameraMatrix_l, distCoeffs_l, rvecs_l, tvecs_l = ret_l[:5]    #返回值里就包含了标定的参数

print('开始右相机标定')
ret_r = calibrate_camera(object_points, corners_right, imgsize)
retval_r, cameraMatrix_r, distCoeffs_r, rvecs_r, tvecs_r = ret_r[:5]

# 立体标定，得到左右相机的外参：旋转矩阵、平移矩阵、本质矩阵、基本矩阵
print('开始立体标定')
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
ret_stereo = cv2.stereoCalibrate(object_points, corners_left, corners_right,
                                 cameraMatrix_l, distCoeffs_l,
                                 cameraMatrix_r, distCoeffs_r,
                                 imgsize, criteria=criteria_stereo,
                                 flags=cv2.CALIB_FIX_INTRINSIC)
ret, _, _, _, _, R, T, E, F = ret_stereo

# 输出结果
print("左相机内参:\n", cameraMatrix_l)
print("左相机畸变系数:\n", distCoeffs_l)
print("右相机内参:\n", cameraMatrix_r)
print("右相机畸变系数:\n", distCoeffs_r)
print("旋转矩阵 R:\n", R)
print("平移向量 T:\n", T)
print("本质矩阵 E:\n", E)
print("基本矩阵 F:\n", F)
print("标定完成")

# 保存标定结果
save_calibration_to_yaml('calibration_parameters.yaml', cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T, E, F)


# 计算重投影误差
def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        total_points += len(imgpoints2)
    mean_error = total_error / total_points
    return mean_error

# 计算并打印左相机和右相机的重投影误差
print("左相机重投影误差: ", compute_reprojection_errors(object_points, corners_left, rvecs_l, tvecs_l, cameraMatrix_l, distCoeffs_l))
print("右相机重投影误差: ", compute_reprojection_errors(object_points, corners_right, rvecs_r, tvecs_r, cameraMatrix_r, distCoeffs_r))

# 立体矫正和显示
def stereo_rectify_and_display(img_l, img_r, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T):
    img_size = img_l.shape[:2][::-1]

    # 立体校正
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, img_size, R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix_l, distCoeffs_l, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix_r, distCoeffs_r, R2, P2, img_size, cv2.CV_32FC1)

    # 图像矫正
    rectified_img_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
    rectified_img_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)

    # 显示矫正后的图像
    combined_img = np.hstack((rectified_img_l, rectified_img_r))
    cv2.imshow('Rectified Images', combined_img)
    cv2.imwrite("stereo_jiaozheng.png",combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 加载并矫正示例图像
example_idx = 0
img_l = img_left[example_idx][0]
img_r = img_right[example_idx][0]
stereo_rectify_and_display(img_l, img_r, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T)
