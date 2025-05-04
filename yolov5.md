# yolo v5-lite学习记录

**前言**

选用lite模型的原因是为了适合树莓派5（Cortex-A76）硬件性能的本地部署选择的[网络模型](https://so.csdn.net/so/search?q=网络模型&spm=1001.2101.3001.7020)为 **YOLOv5** 模型的变种 **YOLOv5-Lite** 模型。**YOLOv5-Lite** 与 **YOLOv5** 相比虽然牺牲了部分网络模型精度，但是缺极大的提升了模型的***\*推理速度\****，该模型属性将更适合实战部署使用。同时也给训练设备没有CUDA加速提供一个参数参考

**参考资料**

[GitHub - ppogg/YOLOv5-Lite: 🍅🍅🍅YOLOv5-Lite: Evolved from yolov5 and the size of model is only 900+kb (int8) and 1.7M (fp16). Reach 15 FPS on the Raspberry Pi 4B~](https://github.com/ppogg/YOLOv5-Lite)

[基于树莓派4B的YOLOv5-Lite目标检测的移植与部署（含训练教程）-CSDN博客](https://blog.csdn.net/black_sneak/article/details/131374492)



## 1、yolov5概述



## 2、训练

数据集制作
常规的神经网络模型训练是需要收集到大量语义丰富的数据集进行训练的。但是考虑实际工程下可能仅需要对已知场地且固定实物进行目标检测追踪等任务，这个时候我们可以采取偷懒的下方作者使用的方法！

1、使用树莓派5的 Camera 直接在捕获需要识别目标物的图片信息（捕获期间转动待识别的目标物体）；

```python
import cv2
from threading import Thread
import uuid
import os
import time
count = 0
def image_collect(cap):
    global count
    while True:
        success, img = cap.read()
        if success:
            file_name = str(uuid.uuid4())+'.jpg'
            cv2.imwrite(os.path.join('images',file_name),img)
            count = count+1
            print("save %d %s"%(count,file_name))
        time.sleep(0.4)
 
if __name__ == "__main__":
    
    os.makedirs("images",exist_ok=True)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
 
    m_thread = Thread(target=image_collect, args=([cap]),daemon=True)
    
    while True:
 
        # 读取一帧图像
 
        success, img = cap.read()
 
        if not success:
 
            continue
 
        cv2.imshow("video",img)
 
        key =  cv2.waitKey(1) & 0xFF   
 
        # 按键 "q" 退出
        if key ==  ord('c'):
            m_thread.start()
            continue
        elif key ==  ord('q'):
            break
 
    cap.release() 
```

没有指明分辨率信息默认使用摄像头支持的最大分辨率，此处采集分辨率为1280*960

按动 **“c”** 开始采集待识别目标图像，按动 **“q”** 退出摄像头 **Camera** 的图片采集；

### 在采集完毕之后使用labelimg进行标注

via canda to create python3.9 virtual labelimg，lastest python version have some unknown error

```
conda create -n labelimg python=3.10
conda init
conda activate labelimg
labelimg
```

 Open dir (select the source photo flie, usually the image flie)

Change Save Dir(select the save flie,usually the label flie)

next change PascalVOC mode to yolo mode

and now you can use 'w' create RectBox, 'd'  Next image

all well done the labels saved to 'label' flie

we can directly use it via code ,access the label flie

### train the model

first create the mydata.yaml ,we need points to your dataset

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/

```

and in this dataset only one class 'RED'

```
train: E:/onedrive/Desktop/YOLOv5-Lite-master/dataphoto/train
val: E:/onedrive/Desktop/YOLOv5-Lite-master/dataphoto/val
nc: 1
names: ['RED']
```

in train.py also need to change variable because this pc not support CUDA 

in line461 include mydata.yaml and the epochs(highly epochs means more training)

```
python train.py --data data/mydata.yaml --weights weights/v5Lite-s.pt --device cpu --epochs 100
```

Set default device in code if not specified:

```
parser.add_argument('--device', default='cpu', help='device to use, e.g. 0 or cpu')
```

