# 将自己的VOC数据集放到VOCdevkit/VOC2007中

# 编写 classes.json

> voc_annotation.py中可以规定训练集，验证集，测试集图片比例，yolo数据集默认不用测试数据集，所以只划分训练集和验证集即可

## classes.txt

```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```



# 先运行voc_annotation.py

> `voc_annotation.py`中可以更改数据集比例
>
> 会在 `VOCdevkit\VOC2007\ImageSets\Main` 中生成对应的 test.txt train.txt trainval.txt val.txt

# 再运行voc2yolo.py

> 根据 test.txt train.txt trainval.txt val.txt 移动文件到yolo下对应目录

# voc2coco.py会生成coco的json解图片

# 各数据集格式说明

> voc数据集格式
>
> x1 y1 x2 y2

```sh
VOCdevkit
    └── VOC2012
         ├── Annotations               所有的图像标注信息(XML文件)
         ├── ImageSets    
         │   ├── Action                人的行为动作图像信息
         │   ├── Layout                人的各个部位图像信息
         │   │
         │   ├── Main                  目标检测分类图像信息
         │   │     ├── train.txt       训练集(5717)
         │   │     ├── val.txt         验证集(5823)
         │   │     └── trainval.txt    训练集+验证集(11540)
         │   │
         │   └── Segmentation          目标分割图像信息
         │         ├── train.txt       训练集(1464)
         │         ├── val.txt         验证集(1449)
         │         └── trainval.txt    训练集+验证集(2913)
         │ 
         ├── JPEGImages                所有图像文件
         ├── SegmentationClass         语义分割png图（基于类别） 单通道图像
         └── SegmentationObject        实例分割png图（基于目标） 单通道图像
```

> yolo数据集格式(yolov5的coco128和霹雳吧啦Wz的yolo3为例)
>
> txt内容，每一行都是 `3 0.933536 0.486124 0.030408 0.154487`
>
> 是 label 中心横坐标与图像宽度比值 中心纵坐标与图像高度比值 bbox宽度与图像宽度比值 bbox高度与图像宽高比值

```sh
#-------------------------------------------#
# 	yolov5 v8的格式
#-------------------------------------------#
yaml:
    path: ../datasets/coco128   # dataset root dir
    train: images/train         # train images (relative to 'path') 128 images
    val: images/val             # val images (relative to 'path') 128 images
    test: images/test           # test images (optional)
dir:
    datasets
    ├── coco128
        ├── images
        │   ├── train   # 训练图片
        │   ├── val     # 验证图片
        │   └── test    # 测试图片
        └── labels
            ├── train   # 训练标签txt
            ├── val     # 验证标签txt
            └── test    # 测试标签txt

#-------------------------------------------#
# 	yolov5 v8另的一种图片目录格式
#-------------------------------------------#
yaml:
    path: ../datasets/coco128   # dataset root dir
    train: train/images         # train images (relative to 'path')
    val: val/images             # val images (relative to 'path')
    test: test/images           # test images (optional)
dir:
    datasets
    ├── coco128
        ├── train
        │   ├── images  # 训练图片
        │   └── labels  # 训练标签txt
        ├── val
        │   ├── images  # 验证图片
        │   └── labels  # 验证标签txt
        └── test
            ├── images  # 测试图片
            └── labels  # 测试标签txt
```

> coco数据集格式

```sh
data
├── coco
    ├── annotations
    │   ├── instances_train2017.json	训练图片信息
    │   └── instances_val2017.json		验证图片信息
    ├── train2017	训练图片
    └── val2017		验证图片
```
