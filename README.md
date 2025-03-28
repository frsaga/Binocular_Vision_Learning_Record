# ECJTU-czy Binocular_Vision_Learning_Record

#### 双目相机标定理论知识

参考资料

[双目相机标定理论总结-CSDN博客](https://blog.csdn.net/h1527820835/article/details/124366040)

[【一文弄懂】张正友标定法-完整学习笔记-从原理到实战-CSDN博客](https://blog.csdn.net/hehedadaq/article/details/105763241)

常用术语


**径向畸变: Radial Distortion**

径向畸变是指由于相机镜头的几何特性（如透镜效应）导致的图像几何变形。常见的表现是图像中心附近的物体看起来不对称或扩展。径向畸变一般通过两个参数（k1,k2）来进行建模，通常有 **桶形畸变** 和 **枕形畸变**。

**切向畸变: Tangential Distortion**


切向畸变是由于相机镜头和传感器的对准问题引起的图像畸变。这种畸变使得图像呈现出扭曲的效果，通常影响到图像的边缘。
![img](https://i-blog.csdnimg.cn/blog_migrate/9d439d21c1380bcd3c208b01c88a6910.png)

**旋转矩阵: Rotation Matrices**

**平移向量: Translation Vectors**

**平均重投影误差: Mean Reprojection Error**

**重投影误差: Reprojection Errors**

**重投影点: Reprojected Points**



**内参矩阵**

![img](https://i-blog.csdnimg.cn/blog_migrate/6802e572d92d4201e74828d51a565121.png)

 (*U*,*V*,*W*)为在世界坐标系下一点的物理坐标， (*u*,*v*)为该点对应的在像素坐标系下的像素坐标， *Z*为尺度因子。

我们将矩阵：![img](https://i-blog.csdnimg.cn/blog_migrate/d37a0e780aa8bab6a69282d9df5ceadc.png)

称为相机的内参矩阵

####  双目相机平台搭建


![557db3a3ffc78a51d6504d524cc1ae7](https://github.com/user-attachments/assets/5e04624e-88e7-41a8-92b6-95fbc9886522)

使用海康机器人CS016相机进行简易双目平台搭建，使用双面胶固定，**基线Baseline**为10cm，相机参数如下

设定两台相机参数保持一致

镜头参数<img width="452" alt="b287f6ff61d31f9b43b0dbd764386d8" src="https://github.com/user-attachments/assets/067dfd7d-3a3f-4db0-b820-12d265450d4f" />

光圈f2.8

对焦距离为♾️



###  matlab标定

####  双目标定

使用Stereo Camera Calibrator App

分别导入两台相机各自拍摄的标定图片


<img width="390" alt="d933b6894a063cc7874010d32d2e95d" src="https://github.com/user-attachments/assets/39e11404-1639-474a-8e60-4faaae116e09" />


选定3 Coefficients （畸变系数）包含径向畸变、切向畸变参数

勾选**skew（非正交参数）**，**如果相机传感器是理想的，Skew 应该接近 0**，表示 x 轴和 y 轴是正交的

勾选**Tangential Distortion（切向畸变）**

切向畸变（Tangential Distortion）是由于 **镜头安装与图像传感器平面不完全平行**，导致直线看起来发生倾斜。


<img width="901" alt="c33a40a17a22436ee71c981518f26eb" src="https://github.com/user-attachments/assets/2b2cc69b-9df8-4a9c-bc2b-ded7dac04b53" />



将错误过多的图片进行删除，其余的图片重排之后重复筛选，直到柱状图所有图片组都接近1，得到标定结果之后导出参数Export


<img width="658" alt="1772b7d9b364c727d14a526469ea2dd" src="https://github.com/user-attachments/assets/0337994a-bf87-424a-bd20-64ff2c97707a" />



CameraParameters1与CameraParameters2为左右摄像头的单独标定参数

分别进入两个相机的文件夹IntrinsicMatrix存放的是摄像头的内参，只与摄像机的内部结构有关，需要先转置再使用。
RadialDistortion：径向畸变，摄像头由于光学透镜的特性使得成像存在着径向畸变，可由K1，K2，K3确定。
TangentialDistortion：切向畸变，由于装配方面的误差，传感器与光学镜头之间并非完全平行，因此成像存在切向畸变，可由两个参数P1，P2确定。


<img width="658" alt="a5254730fdd0905dec3353b94c36504" src="https://github.com/user-attachments/assets/d1861e4b-10be-4631-bc4a-f695ba1bfb2a" />


####  单目标定

打开matlab

导入图片add images -- calibrate


<img width="911" alt="97506daf7d8e0f4f8304bf6be834ef4" src="https://github.com/user-attachments/assets/ee8edf61-ba45-491c-b7ca-406466725d03" />

选择错误过多的图片删除 得到标定结果 


<img width="900" alt="515be9f958eb5eef53f2a7fa8e9bb3a" src="https://github.com/user-attachments/assets/45c7b00e-8dcd-4048-a5d9-72a9df2a5d56" />
<img width="900" alt="9aa7ca29e9d09fa32a6a4565a48995d" src="https://github.com/user-attachments/assets/278bd841-9459-4599-b108-ddd82ccff1ec" />

导出数据 Export Camera Parameters

RadiaDistortion为相机的畸变矩阵，Intrinsics为内参矩阵


<img width="658" alt="4a62f1631f067fde559493dd2ed28e4" src="https://github.com/user-attachments/assets/6a2ecc4d-bf1d-4734-863e-d8aeb9132ff6" />


<img width="657" alt="3bbc22e9ee4d4abd0c7a03e2e794d11" src="https://github.com/user-attachments/assets/389107c3-3fcc-424e-a5ba-dc3a2a9e5f01" />

###  基于OpenCV的相机标定

参考资料

[【摄像头标定】使用opencv进行双目摄像头的标定及矫正（python）_opencv双目相机标定-CSDN博客](https://blog.csdn.net/m0_71523511/article/details/139960845)

[一文详解双目相机标定理论 - 知乎](https://zhuanlan.zhihu.com/p/362018123)

运行环境：VScode python312

代码1 通过读取目标文件夹中的图片进行标定并生成矫正后的图像

简单修改参考代码的读取照片文件夹和输出参数格式后，运行出现如下错误，经过改错，发现有四张照片没有拍摄到全部格角点，故报错，通过删除这四张图片后，正常输出参数
<img width="823" alt="64d533840720057847e83aedc28e902" src="https://github.com/user-attachments/assets/c6502bbd-d30d-4f59-806f-4f8364f3c381" />



代码输出结果

```
---
camera_matrix_left:
  rows: 3
  cols: 3
  dt: d
  data:
    - 531.7200210313852
    - 0
    - 642.0170539101581
    - 0
    - 533.6471323984354
    - 420.4033045027399
    - 0
    - 0
    - 1
dist_coeff_left:
  rows: 1
  cols: 5
  dt: d
  data:
    - -0.1670007968198256
    - 0.04560028196221921
    - 0.0011938487550718078
    - -0.000866537907860316
    - -0.00805042100882671
camera_matrix_right:
  rows: 3
  cols: 3
  dt: d
  data:
    - 525.9058345430292
    - 0
    - 628.7761214904813
    - 0
    - 528.2078922687268
    - 381.8575789135264
    - 0
    - 0
    - 1
dist_coeff_right:
  rows: 1
  cols: 5
  dt: d
  data:
    - -0.15320688387351564
    - 0.03439886104586617
    - -0.0003732170677440928
    - -0.0024909528446780153
    - -0.005138400994014348
R:
  rows: 3
  cols: 3
  dt: d
  data:
    - 0.9999847004116569
    - -0.00041406631566505544
    - 0.005516112008926496
    - 0.0003183979929468572
    - 0.9998497209492369
    - 0.017333036100216304
    - -0.005522460079247196
    - -0.017331014592906722
    - 0.9998345554979852
T:
  rows: 3
  cols: 1
  dt: d
  data:
    - -55.849260376265015
    - 2.1715925432988743
    - 0.46949841441903933
E:
  rows: 3
  cols: 3
  dt: d
  data:
    - -0.012142020481601675
    - -0.5070637607007459
    - 2.1630954322858496
    - 0.1610659204031652
    - -0.9681187500627653
    - 55.84261022903612
    - -2.189341611238282
    - -55.83996821910631
    - -0.9800159939787676
F:
  rows: 3
  cols: 3
  dt: d
  data:
    - -2.4239149875305048e-8
    - -0.0000010085973649868748
    - 0.0027356495714066175
    - 3.2013501988129346e-7
    - -0.0000019172863951399893
    - 0.05961765359743852
    - -0.002405523166325036
    - -0.057046539240958545
    - 1

