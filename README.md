# ECJTU- ZEYU CHEN
## 记录双目视觉学习路线
### 更新在week n.md

# 双目视觉系统项目

---

## 1. 项目简介  
- **项目目标**：实现双目视觉系统，进行图像采集、极线校正与三维重建准备。  
- **硬件**：双目摄像头、三脚架、玻纤板（基线固定为 8cm）。  
- **软件**：OpenCV、C++ / Python 混合开发。

---

## 2. 平台搭建  
- 相机通过三脚架与玻纤板固定，确保两个相机视角尽可能平行。  
- 固定基线长度为 8cm，有助于后续的深度计算精度。  

---

## 3. 极线校正  
### 3.1 为什么需要极线校正？  
- 解决双目匹配搜索效率低与误匹配问题。  
- 校正后左右图像的匹配点在同一水平线上，匹配只需一维搜索。

### 3.2 实现步骤  
- 使用 `StereoRectify` 或 `Fusiello` 方法进行校正矩阵计算。  
- 生成变换矩阵 H1、H2 和投影矩阵 Q。  
- 使用 warp 方法将图像进行映射变换，得到校正图像。  

### 3.3 极线可视化  
- 拼接左右图像，绘制若干水平绿线用于辅助验证校正效果。  

---

## 4. 实时取流与图像采集  
- 使用 OpenCV 打开左右相机，实时读取图像流。  
- 按下 `s` 键保存当前帧图像，命名使用时间戳，自动保存在左右图像文件夹中。  
- 按下 `q` 键退出。  

---

## 5. 图像校正与保存  
- 使用摄像机内参与畸变参数对图像进行畸变校正。  
- 对图像对进行极线矫正，并保存校正结果图。  

---

## 6. 后续拓展建议  
- 加入视差图计算模块（如 SGBM 算法）。  
- 添加深度估计与三维点云可视化（Open3D、PCL）。  
- 增加 ROS 支持，或开发图形用户界面以提高交互体验。  

---

## 7. 参考资料  
- CSDN 极线校正原理博客  
- OpenCV 官方文档  
- GitHub 示例项目与论文链接

---

# 📙 Stereo Vision System README Outline (English)

---

## 1. Project Overview  
- **Goal**: Develop a stereo vision system for image acquisition, rectification, and 3D reconstruction preparation.  
- **Hardware**: Stereo cameras, tripod, fiberglass plate (baseline = 8 cm).  
- **Software**: OpenCV with C++ / Python hybrid development.

---

## 2. Platform Setup  
- Cameras are mounted using a tripod and fiberglass plate to ensure parallel alignment.  
- Fixed baseline of 8 cm improves depth estimation accuracy.  

---

## 3. Epipolar Rectification  
### 3.1 Why Rectification?  
- Reduces matching complexity from 2D to 1D, increasing speed and reducing false matches.  
- Matching points lie on the same horizontal line after rectification.

### 3.2 Implementation Steps  
- Use `StereoRectify` or `Fusiello` method to compute rectification matrices.  
- Generate transformation matrices H1, H2 and projection matrix Q.  
- Apply warp transformations to obtain rectified images.

### 3.3 Epipolar Line Visualization  
- Concatenate left and right images, draw horizontal green lines to verify alignment.

---

## 4. Real-time Streaming & Image Capture  
- Use OpenCV to open camera streams and read images in real-time.  
- Press `s` to save current frames with timestamped filenames to local folders.  
- Press `q` to exit the program.  

---

## 5. Image Rectification & Saving  
- Apply undistortion using camera intrinsics and distortion coefficients.  
- Perform stereo rectification and save the result images.  

---

## 6. Future Work  
- Add disparity map generation module (e.g., SGBM algorithm).  
- Add depth estimation and 3D point cloud visualization (Open3D, PCL).  
- Add ROS support or GUI for better user experience.  

---

## 7. References  
- Blog post on rectification principles (CSDN)  
- OpenCV official documentation  
- GitHub sample projects and academic papers  

