"""使用torchvision对图像随机调整"""

from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm


def get_transforms(resize=224):
    trans = [  # transforms.Resize(int(resize/0.875)),         # 224 / 0.875 = 256
        # transforms.RandomCrop((resize, resize)),
        transforms.RandomResizedCrop(resize),  # 随机缩小剪裁,输出为(resize, resize)
        transforms.RandomRotation(degrees=90),  # 随机旋转，-90到90度之间随机选
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        # transforms.RandomPerspective(p=0.5),          # 透视变换
        # transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.5), scale=(0.5, 1.5), shear=(-45, 45)),   # 随机仿射变化
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0
        ),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        # transforms.RandomGrayscale(0.025),            # 概率转换成灰度率，3通道就是R=G=B
    ]

    transform = transforms.Compose(trans)
    return transform


def trans_images(origin: str, new: str, times: int = 4):
    """随机增强图片

    Args:
        origin (str): 原始图片文件夹
        new (str):    保存图片文件夹
        times (int, optional): 重复次数. Defaults to 4.
    """
    if not os.path.exists(new):
        os.mkdir(new)

    transform = get_transforms(224)

    image_list = os.listdir(origin)
    image_list = [
        image for image in image_list if image.endswith(("jpg", "jpeg", "png", "bmp"))
    ]

    for i in range(times):
        for image in tqdm(image_list):
            origin_image_path = os.path.join(origin, image)
            new_image_path = os.path.join(new, f"trans_{i}_{image}")
            img = Image.open(origin_image_path)
            img = transform(img)
            img.save(new_image_path)


if __name__ == "__main__":
    ori = r"D:/ai/数据集/abnormal/溧阳科达利/面阵相机NGImage/3 mark/train/train_val/normal"
    dst = r"D:/ai/数据集/abnormal/溧阳科达利/面阵相机NGImage/3 mark/train/train_val/normal_trans"
    trans_images(ori, dst)
