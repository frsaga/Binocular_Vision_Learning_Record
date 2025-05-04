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
![7a5d70cd572320967ceaa266ebdd2c1](https://github.com/user-attachments/assets/215809bd-94f1-48af-976f-e489ca7b93c6)

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

按动 **“c”** 开始采集待识别目标图像，按动 **“q”** 退出摄像头 **Camera** 的图片采集；<img width="724" alt="4c628690c75c51472d828e156178899" src="https://github.com/user-attachments/assets/f3bbfff2-632b-45b6-888a-483fb2a47737" />


### 在采集完毕之后使用labelimg进行标注

via canda to create python3.9 virtual labelimg，lastest python version have some unknown error

```
conda create -n labelimg python=3.10
conda init
conda activate labelimg
labelimg
```
<img width="876" alt="d27e99048a55ddaff5b86e3f5b87c49" src="https://github.com/user-attachments/assets/9477f278-5b04-4a71-960f-1455109e4cf5" />

 Open dir (select the source photo flie, usually the image flie)

Change Save Dir(select the save flie,usually the label flie)

next change PascalVOC mode to yolo mode
<img width="877" alt="01aba2ba80de16b3c18410f78a86c71" src="https://github.com/user-attachments/assets/1d36e222-4b16-4146-b07a-09eaed1d4e6e" />

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
<img width="955" alt="aaed0024a1837a53210a017d3442590" src="https://github.com/user-attachments/assets/935b7cfa-8119-4a4e-98dc-d6535e31303e" />

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
#### 训练环境依赖
```
# base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow
PyYAML>=5.3.1
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.41.0
 
# logging -------------------------------------
tensorboard>=2.4.1
# wandb
 
# plotting ------------------------------------
seaborn>=0.11.0
pandas
 
# export --------------------------------------
# coremltools>=4.1
# onnx>=1.9.1
# scikit-learn==0.19.2  # for coreml quantization
 
# extras --------------------------------------
thop  # FLOPS computation
pycocotools>=2.0  # COCO mAP

```
<img width="955" alt="056521900d842a9c0a7506934b7f05d" src="https://github.com/user-attachments/assets/3e1c8a2c-9183-452a-bcb4-f500287763ce" />

now the code running
<img width="955" alt="367fd206deff7c9b906a190358e7710" src="https://github.com/user-attachments/assets/af9c5204-1e5b-4e1c-b5a6-21ba145d01d0" />

**after running we get some glie in \runs floder**
<img width="176" alt="3c544ae0f38ae7e9f7b545d090dcf0e" src="https://github.com/user-attachments/assets/572d43be-965d-41de-a6ec-d1d77e1147cd" />
we only use the last exp like exp18 floder
<img width="176" alt="4aa0c6b2d79de05df0f3c29c41db8f0" src="https://github.com/user-attachments/assets/159cc29a-a8ff-46b1-bf74-9fe49492da09" />
After successful training, you will find a folder named expx (where x is a number) inside the train folder under the runs directory in the current working directory.
Each expx folder contains data from the x-th training session, including:

the best model weights (best_weight)

the latest model weights (last_weight)

training results (result)
and other related information.

## Test ONNX model
first to export .onnx, use best.pt(Make sure you have trained a YOLOv5-Lite model and have a .pt)
<img width="955" alt="d0e70f246be21b9038e689c63e270eb" src="https://github.com/user-attachments/assets/bea376a0-8d63-48e0-9eb2-25715dfffe90" />

use the export.py to export best.onnx
** saved as best.onnx**
###  Load ONNX model and run on train images
```
import cv2
import numpy as np
import onnxruntime as ort
import time
 
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
 
def _make_grid( nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
 
def cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride):
    
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w/ stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)
 
        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs
 
 
 
def post_process_opencv(outputs,model_h,model_w,img_h,img_w,thred_nms,thred_cond):
    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)
 
    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas,conf,thred_cond,thred_nms)
    if len(ids)>0:
        return  np.array(areas)[ids],np.array(conf)[ids],cls_id[ids]
    else:
        return [],[],[]
def infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, [model_w,model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
 
    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
 
    # 输出坐标矫正
    outs = cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride)
 
    # 检测框计算
    img_h,img_w,_ = np.shape(img0)
    boxes,confs,ids = post_process_opencv(outs,model_h,model_w,img_h,img_w,thred_nms,thred_cond)
 
    return  boxes,confs,ids
 
 
 
 
if __name__ == "__main__":
 
    # 模型加载
    model_pb_path = "best.onnx"
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)
    
    # 标签字典
    dic_labels= {0:'drug',
            1:'glue',
            2:'prime'}
    
    # 模型参数
    model_h = 320
    model_w = 320
    nl = 3
    na = 3
    stride=[8.,16.,32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)
    
    video = 0
    cap = cv2.VideoCapture(video)
    flag_det = False
    while True:
        success, img0 = cap.read()
        if success:
            
            if flag_det:
                t1 = time.time()
                det_boxes,scores,ids = infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5)
                t2 = time.time()
            
                
                for box,score,id in zip(det_boxes,scores,ids):
                    label = '%s:%.2f'%(dic_labels[id],score)
            
                    plot_one_box(box.astype(np.int16), img0, color=(255,0,0), label=label, line_thickness=None)
                    
                str_FPS = "FPS: %.2f"%(1./(t2-t1))
                
                cv2.putText(img0,str_FPS,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
                
            
            cv2.imshow("video",img0)
        key=cv2.waitKey(1) & 0xFF    
        if key == ord('q'):
        
            break
        elif key & 0xFF == ord('s'):
            flag_det = not flag_det
            print(flag_det)
            
    cap.release() 
```
<img width="1261" alt="31036112370de30380cb356a358de6f" src="https://github.com/user-attachments/assets/192457b8-171b-49de-95f1-91aecfb70da9" />

