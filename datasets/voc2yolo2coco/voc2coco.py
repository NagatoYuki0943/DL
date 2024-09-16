# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json
import shutil
from tqdm import tqdm


classes_path = "./classes.txt"


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, mode="r", encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


classes, _ = get_classes(classes_path)


# xml存放路径
voc2007xmls = "VOCdevkit/VOC2007/Annotations"
# train/val存放路径
txt_list = [
    "VOCdevkit/VOC2007/ImageSets/Main/train.txt",
    "VOCdevkit/VOC2007/ImageSets/Main/val.txt",
]
# train/val存放路径
json_name_list = ["instances_train2017.json", "instances_val2017.json"]
# save dir
save_dir = "coco"
os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "train2017"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "val2017"), exist_ok=True)


categories = []
for iind, cat in enumerate(classes):
    cate = {}
    cate["id"] = iind
    cate["name"] = cat
    cate["supercategory"] = cat
    categories.append(cate)


def get_image_info(xmlname, id):
    sig_xml_box = []
    with open(xmlname, mode="r", encoding="utf-8") as f:
        tree = ET.parse(f)
    root = tree.getroot()
    images = {}
    images["id"] = id
    for i in root:  # 遍历一级节点
        if i.tag == "filename":
            file_name = i.text  # 0001.jpg
            # print("image name: ", file_name)
            images["file_name"] = file_name
        if i.tag == "size":
            for j in i:
                if j.tag == "height":
                    height = j.text
                    images["height"] = int(height)
                if j.tag == "width":
                    width = j.text
                    images["width"] = int(width)
        if i.tag == "object":
            for j in i:
                if j.tag == "name":
                    cls_name = j.text
                cat_id = classes.index(cls_name)  # 找到列表中值的索引
                if j.tag == "bndbox":
                    bbox = []
                    xmin = 0
                    ymin = 0
                    xmax = 0
                    ymax = 0
                    for r in j:
                        if r.tag == "xmin":
                            xmin = eval(r.text)
                        if r.tag == "ymin":
                            ymin = eval(r.text)
                        if r.tag == "xmax":
                            xmax = eval(r.text)
                        if r.tag == "ymax":
                            ymax = eval(r.text)
                    bbox.append(xmin)
                    bbox.append(ymin)
                    bbox.append(xmax - xmin)
                    bbox.append(ymax - ymin)
                    bbox.append(id)  # 保存当前box对应的image_id
                    bbox.append(cat_id)
                    # anno area
                    bbox.append((xmax - xmin) * (ymax - ymin) - 10.0)  # bbox的ares
                    # coco中的ares数值是 < w*h 的, 因为它其实是按segmentation的面积算的,所以我-10.0一下...
                    sig_xml_box.append(bbox)
                    # print('bbox', xmin, ymin, xmax - xmin, ymax - ymin, 'id', id, 'cls_id', cat_id)
    # print ('sig_img_box', sig_xml_box)
    return images, sig_xml_box


def txt2list(txtfile):
    with open(txtfile, mode="r", encoding="utf-8") as f:
        l = f.read().splitlines()
    return l


def main():
    for txt, json_name in zip(txt_list, json_name_list):
        if "train" in txt:
            save_image_path = os.path.join(save_dir, "train2017")
            train_val = "train"
        else:
            save_image_path = os.path.join(save_dir, "val2017")
            train_val = "val"

        names = txt2list(txt)

        xmls = []
        for name in tqdm(names, desc="transfer {} file...".format(train_val)):
            # save xml path
            xmls.append(os.path.join(voc2007xmls, name + ".xml"))

            # move image
            ori_path = os.path.join("VOCdevkit/VOC2007/JPEGImages", name + ".jpg")
            dst_path = os.path.join(save_image_path, name + ".jpg")
            shutil.copy(ori_path, dst_path)

        ann_js = {}
        images = []
        bboxes = []
        for i_index, xml_file in enumerate(xmls):
            image, sig_xml_bbox = get_image_info(xml_file, i_index)
            images.append(image)
            bboxes.extend(sig_xml_bbox)
        ann_js["images"] = images  # images
        ann_js["categories"] = categories  # categories
        annotations = []
        for box_ind, box in enumerate(bboxes):
            anno = {}
            anno["id"] = box_ind
            anno["image_id"] = box[-3]
            anno["category_id"] = box[-2]
            anno["bbox"] = box[:-3]
            anno["area"] = box[-1]
            anno["iscrowd"] = 0
            annotations.append(anno)
        ann_js["annotations"] = annotations  # annotations

        json_path = os.path.join(save_dir, "annotations", json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                ann_js, f, indent=4, ensure_ascii=False
            )  # indent=4 更加美观显示 ensure_ascii=False 防止中文乱码
        print(f"generate {json_name} files")


if __name__ == "__main__":
    main()
