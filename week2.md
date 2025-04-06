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
bool Image::StereoRectifyImages(const Image& image1, const Image& image2, const Point3fArr& points1, const Point3fArr& points2, Image8U3& rectifiedImage1, Image8U3& rectifiedImage2, Image8U& mask1, Image8U& mask2, Matrix3x3& H, Matrix4x4& Q)
{
 ASSERT(image1.IsValid() && image2.IsValid());
 ASSERT(image1.GetSize() == image1.image.size() && image2.GetSize() ==        image2.image.size());
 ASSERT(points1.size() && points2.size());
 
 #if 0
 { // display projection pairs
  std::vector<Point2f> matches1, matches2;
  FOREACH(i, points1) {
   matches1.emplace_back(reinterpret_cast<const Point2f&>(points1[i]));
   matches2.emplace_back(reinterpret_cast<const Point2f&>(points2[i]));
  }
  RECTIFY::DrawMatches(const_cast<Image8U3&>(image1.image),       const_cast<Image8U3&>(image2.image), matches1, matches2);
 }
 #endif
 
 // compute rectification
 // 校正计算
 Matrix3x3 K1, K2, R1, R2;
 #if 0
 const REAL t(Camera::StereoRectify(image1.GetSize(), image1.camera, image2.GetSize(), image2.camera, R1, R2, K1, K2));
 #elif 1
 const REAL t(Camera::StereoRectifyFusiello(image1.GetSize(), image1.camera, image2.GetSize(), image2.camera, R1, R2, K1, K2));
 #else
 Pose pose;
 ComputeRelativePose(image1.camera.R, image1.camera.C, image2.camera.R, image2.camera.C, pose.R, pose.C);
 cv::Mat P1, P2;
 cv::stereoRectify(image1.camera.K, cv::noArray(), image2.camera.K, cv::noArray(), image1.GetSize(), pose.R, Vec3(pose.GetTranslation()), R1, R2, P1, P2, Q, 0/*cv::CALIB_ZERO_DISPARITY*/, -1);
 K1 = P1(cv::Rect(0,0,3,3));
 K2 = P2(cv::Rect(0,0,3,3));
 const Point3 _t(R2 * pose.GetTranslation());
 ASSERT((ISZERO(_t.x) || ISZERO(_t.y)) && ISZERO(_t.z));
 const REAL t(ISZERO(_t.x)?_t.y:_t.x);
 #if 0
 cv::Mat map1, map2;
 cv::initUndistortRectifyMap(image1.camera.K, cv::noArray(), R1, K1, image1.GetSize(), CV_16SC2, map1, map2);
 cv::remap(image1.image, rectifiedImage1, map1, map2, cv::INTER_CUBIC);
 cv::initUndistortRectifyMap(image2.camera.K, cv::noArray(), R2, K2, image1.GetSize(), CV_16SC2, map1, map2);
 cv::remap(image2.image, rectifiedImage2, map1, map2, cv::INTER_CUBIC);
 return;
 #endif
 #endif
 if (ISZERO(t))
  return false;
 
 // adjust rectified camera matrices such that the entire area common to both source images is contained in the rectified images
 // 调整校正后的相机矩阵，使两个源图像的公共区域都包含在校正后的图像中
 cv::Size size1(image1.GetSize()), size2(image2.GetSize());
 if (!points1.empty())
  Camera::SetStereoRectificationROI(points1, size1, image1.camera, points2, size2, image2.camera, R1, R2, K1, K2);
 ASSERT(size1 == size2);
 
 // compute rectification homography (from original to rectified image)
 // 计算校正的单应性矩阵（描述的是两个图像像素坐标的转换矩阵H[u,v,1]^t=[u',v',1]^t）(从原始图像到校正图像)
 const Matrix3x3 H1(K1 * R1 * image1.camera.GetInvK()); H = H1;
 const Matrix3x3 H2(K2 * R2 * image2.camera.GetInvK());
 
 #if 0
 { // display epipolar lines before and after rectification
  Pose pose;
  ComputeRelativePose(image1.camera.R, image1.camera.C, image2.camera.R, image2.camera.C, pose.R, pose.C);
  const Matrix3x3 F(CreateF(pose.R, pose.C, image1.camera.K, image2.camera.K));
  std::vector<Point2f> matches1, matches2;
  #if 1
  FOREACH(i, points1) {
   matches1.emplace_back(reinterpret_cast<const Point2f&>(points1[i]));
   matches2.emplace_back(reinterpret_cast<const Point2f&>(points2[i]));
  }
  #endif
  RECTIFY::DrawRectifiedImages(image1.image.clone(), image2.image.clone(), F, H1, H2, matches1, matches2);
 }
 #endif
 
 // rectify images (apply homographies)
 // 校正图像,就是利用单应性矩阵，把原图像每个像素坐标转换到校正的图像下。
 rectifiedImage1.create(size1);
 cv::warpPerspective(image1.image, rectifiedImage1, H1, rectifiedImage1.size());
 rectifiedImage2.create(size2);
 cv::warpPerspective(image2.image, rectifiedImage2, H2, rectifiedImage2.size());
 
 // mark valid regions covered by the rectified images
 // 标记正确图像覆盖的有效区域
 struct Compute {
  static void Mask(Image8U& mask, const cv::Size& sizeh, const cv::Size& size, const Matrix3x3& H) {
   mask.create(sizeh);
   mask.memset(0);
   std::vector<Point2f> corners(4);
   corners[0] = Point2f(0,0);
   corners[1] = Point2f((float)size.width,0);
   corners[2] = Point2f((float)size.width,(float)size.height);
   corners[3] = Point2f(0,(float)size.height);
   cv::perspectiveTransform(corners, corners, H);
   std::vector<std::vector<Point2i>> contours(1);
   for (int i=0; i<4; ++i)
    contours.front().emplace_back(ROUND2INT(corners[i]));
   cv::drawContours(mask, contours, 0, cv::Scalar(255), cv::FILLED);
  }
 };
 Compute::Mask(mask1, size1, image1.GetSize(), H1);
 Compute::Mask(mask2, size2, image2.GetSize(), H2);
 
 // from the formula that relates disparity to depth as z=B*f/d where B=-t and d=x_l-x_r
 // and the formula that converts the image projection from right to left x_r=K1*K2.inv()*x_l
 // compute the inverse projection matrix that transforms image coordinates in image 1 and its
 // corresponding disparity value to the 3D point in camera 1 coordinates as:
 // 根据depth=Bf/d的关系，计算投影矩阵Q将校正的视差图转到为校正的深度图。
 ASSERT(ISEQUAL(K1(1,1),K2(1,1)));
 Q = Matrix4x4::ZERO;
 //   Q * [x, y, disparity, 1] = [X, Y, Z, 1] * w
 ASSERT(ISEQUAL(K1(0,0),K2(0,0)) && ISZERO(K1(0,1)) && ISZERO(K2(0,1)));
 Q(0,0) = Q(1,1) = REAL(1);
 Q(0,3) = -K1(0,2);
 Q(1,3) = -K1(1,2);
 Q(2,3) =  K1(0,0);
 Q(3,2) = -REAL(1)/t;
 Q(3,3) =  (K1(0,2)-K2(0,2))/t;
 
 // compute Q that converts disparity from rectified to depth in original image
 // 计算将视差从校正到原始图像深度转换的Q值
 Matrix4x4 P(Matrix4x4::IDENTITY);
 cv::Mat(image1.camera.K*R1.t()).copyTo(cv::Mat(4,4,cv::DataType<Matrix4x4::Type>::type,P.val)(cv::Rect(0,0,3,3)));
 Q = P*Q;
 return true;
}
 ```

### 在原有程序的基础上添加极线矫正并输出图像的代码
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
