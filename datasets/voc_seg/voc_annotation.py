import os
import random


# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 1  # 要设置为1，因为yolo数据集默认没有测试集合
train_percent = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = "VOCdevkit"


if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    jpgfilepath = os.path.join(VOCdevkit_path, "VOC2012/JPEGImages")
    saveBasePath = os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Segmentation")
    temp_jpg = os.listdir(jpgfilepath)
    total_jpg = []
    for xml in temp_jpg:
        if xml.endswith(".jpg"):
            total_jpg.append(xml)

    num = len(total_jpg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(saveBasePath, "trainval.txt"), "w")
    ftest = open(os.path.join(saveBasePath, "test.txt"), "w")
    ftrain = open(os.path.join(saveBasePath, "train.txt"), "w")
    fval = open(os.path.join(saveBasePath, "val.txt"), "w")

    for i in list:
        name = total_jpg[i][:-4] + "\n"
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")
