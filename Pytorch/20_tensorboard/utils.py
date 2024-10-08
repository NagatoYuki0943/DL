import os
import json
import pickle
import random

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_class_preds(
    net,
    images_dir: str,
    transform,
    num_plot: int = 5,  # 显示多少图片
    device="cpu",
):
    # 图片
    if not os.path.exists(images_dir):
        print("not found {} path, ignore add figure.".format(images_dir))
        return None

    # label
    label_path = os.path.join(images_dir, "label.txt")
    if not os.path.exists(label_path):
        print("not found {} file, ignore add figure".format(label_path))
        return None

    # 读取  类别索引和对应名称
    json_label_path = "./class_indices.json"
    assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    json_file = open(json_label_path, "r")
    # {"0": "daisy"}
    flower_class = json.load(json_file)
    # {"daisy": "0"}
    # class_indices = dict((v, k) for k, v in flower_class.items())
    class_indices = dict(zip(flower_class.values(), flower_class.keys()))

    # 读取 label文件 对应图片文件和真实标签
    label_info = []
    with open(label_path, "r") as rd:
        for line in rd.readlines():
            line = line.strip()  # 去除空格
            if len(line) > 0:  # 不是空行
                # [图片名称, 类别]
                split_info = [i for i in line.split(" ") if len(i) > 0]
                assert (
                    len(split_info) == 2
                ), "label format error, expect file_name and class_name"
                image_name, class_name = split_info  # 图片名称, 类别
                image_path = os.path.join(images_dir, image_name)  # 图片路径
                # 如果文件不存在，则跳过
                if not os.path.exists(image_path):
                    print("not found {}, skip.".format(image_path))
                    continue
                # 如果读取的类别不在给定的类别内，则跳过
                if class_name not in class_indices.keys():
                    print("unrecognized category {}, skip".format(class_name))
                    continue
                # 图片路径和label
                label_info.append([image_path, class_name])

    # 没有预测图片
    if len(label_info) == 0:
        return None

    # 图片数目大于检测数量,取前5张
    if len(label_info) > num_plot:
        label_info = label_info[:num_plot]
    # 图片数量
    num_imgs = len(label_info)

    # 读取多张图片和label
    images = []
    labels = []
    for img_path, class_name in label_info:
        # read img
        img = Image.open(img_path).convert("RGB")
        label_index = int(class_indices[class_name])

        # preprocessing
        img = transform(img)

        images.append(img)
        labels.append(label_index)

    # 图片拼接为batch
    images = torch.stack(images, dim=0).to(device)

    # 推理
    with torch.inference_mode():
        output = net(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # width, height
    fig = plt.figure(figsize=(num_imgs * 2.5, 3), dpi=100)
    for i in range(num_imgs):
        # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
        ax = fig.add_subplot(
            1, num_imgs, i + 1, xticks=[], yticks=[]
        )  # xticks 刻度信息设置为空

        # CHW -> HWC
        npimg = images[i].cpu().numpy().transpose(1, 2, 0)

        # 将图像还原至标准化之前
        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        print(npimg)
        plt.imshow(npimg.astype("uint8"))

        # 设置标签,正确绿色,否则红色
        title = "{}, {:.2f}%\n(label: {})".format(
            flower_class[str(preds[i])],  # predict class
            probs[i] * 100,  # predict probability
            flower_class[str(labels[i])],  # true class
        )
        ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))

    return fig
