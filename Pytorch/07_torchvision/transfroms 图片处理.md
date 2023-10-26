`http://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py`

`D:\AI\AI\04_Pytorch\code\4 new\08 torchvision\08.2 transform 图片处理\transform.py`

# 示例

> **归一化,减去均值(前面),除以标准差(后面)**
>
> 反求: **乘以标准差加上均值最后乘以255**

## 

## ImageFolder

```python
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import json


def get_data(data_dir, batch_size=8, resize=224, num_works=0):
    """_summary_

    Args:
        data_dir (str): 数据集目录
        batch_size (int, optional): batch. Defaults to 8.
        resize (int, optional): 图片缩放小. Defaults to 224.
        num_works (int, optional): 图片读取线程数. Defaults to 0.

    Returns:
        dataloaders: {'train':DataLoader, 'val':DataLoader}
        image_nums: {'train':num, 'val':num}
    """
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(resize*1.25),
            # transforms.RandomCrop((resize, resize)),
            transforms.RandomResizedCrop(resize),           # 随机缩小剪裁,输出为(resize, resize)
            transforms.RandomRotation(45),                  # 随机旋转，-45到45度之间随机选
            transforms.RandomHorizontalFlip(0.5),           # 随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(0.5),             # 随机垂直翻转
            transforms.RandomPerspective(),                 # 透视变换
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)), # 随机仿射变化
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(0.025),              # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),                          # 转化为tensor,并归一化
            transforms.Normalize([0.485, 0.456, 0.406],     # 减去均值(前面),除以标准差(后面)
                                [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(resize*1.25)),
            transforms.RandomCrop((resize, resize)),
            transforms.ToTensor(),                          # 转化为tensor,并归一化
            transforms.Normalize([0.485, 0.456, 0.406],     # 减去均值(前面),除以标准差(后面)
                                [0.229, 0.224, 0.225])
        ]),
    }

    datasets    = {x: ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(dataset=datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_works) for x in ['train', 'val']}
    image_nums  = {x: len(datasets[x]) for x in ['train', 'val']}
    print(image_nums)
    # {'train': 14034, 'val': 3000}

    # 将id和标签写入json列表
    with open('./class_to_idx.json', 'w', encoding='utf-8') as f:
        json.dump(datasets['train'].class_to_idx, f)
        print('class_to_idx.json write success.')

    return datasets, dataloaders, image_nums


if __name__ == "__main__":
    datasets, dataloaders, image_nums = get_data("./dataset/scenery", 8, 128)

    # 获取文件夹名字和标签字典
    class_to_idx = datasets['train'].class_to_idx
    print(class_to_idx)     # {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

    # 获取文件夹列表
    classes = datasets['train'].classes
    print(classes)          # ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    # 通过id找到文件夹
    print(classes[0])       # buildings

    # 读取json并切换key和value的位置
    with open('./class_to_idx.json', 'r', encoding='utf-8') as f:
        cls_to_idx = json.load(f)
    idx_to_cls = dict(zip(cls_to_idx.values(), cls_to_idx.keys()))
    print(idx_to_cls)       # {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
```

## Dataset

```python
from PIL import Image


def __getitem__(self, idx):
    # idx~[0~len(images)]
    # self.images, self.targets
    # img: 'C:\Ai\Scenery\data\scenery\\train/buildings\10006.jpg'
    # target: 0
    img, target = self.images[idx], self.targets[idx]

    img = Image.open(img).convert('RGB')


    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(resize*1.25),
            # transforms.RandomCrop((resize, resize)),
            transforms.RandomResizedCrop(resize),           # 随机缩小剪裁,输出为(resize, resize)
            transforms.RandomRotation(45),                  # 随机旋转，-45到45度之间随机选
            transforms.RandomHorizontalFlip(0.5),           # 随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(0.5),             # 随机垂直翻转
            transforms.RandomPerspective(),                 # 透视变换
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)), # 随机仿射变化
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(0.025),              # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),                          # 转化为tensor,并归一化
            transforms.Normalize([0.485, 0.456, 0.406],     # 减去均值(前面),除以标准差(后面)
                                [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(resize*1.25)),
            transforms.RandomCrop((resize, resize)),
            transforms.ToTensor(),                          # 转化为tensor,并归一化
            transforms.Normalize([0.485, 0.456, 0.406],     # 减去均值(前面),除以标准差(后面)
                                [0.229, 0.224, 0.225])
        ]),
    }
    transform = data_transforms[self.mode]

    img = transform(img)
    target = torch.tensor(target)

    return img, target
```

# 

# 对PIL.Image进行变换

## Compose(transforms) 组合

将多个`transform`组合起来使用。

`transforms`： 由`transform`构成的列表. 例子：

```python
transform = transforms.Compose([
     transforms.CenterCrop(10),
     transforms.ToTensor(),
 ])

img = transform(img)
```



## Resize(size: (int,tuple), interpolation=Image.BILINEAR) 缩放

Image做resize操作的，几乎都要用到。

> int or tuple 解释

- 这里输入可以是int，此时表示将输入图像的短边resize到这个int数，长边则根据对应比例调整，图像的长宽比不变。
- 如果输入是个(h,w)的序列，h和w都是int，则直接将输入图像resize到这个(h,w)尺寸，相当于force resize，所以一般最后图像的长宽比会变化，也就是图像内容被拉长或缩短。



## RandomHorizontalFlip() 随机水平翻转

随机水平翻转给定的`PIL.Image`,概率为`0.5`。即：一半的概率翻转，一半的概率不翻转。



## RandomVerticalFlip() 随机竖直翻转

随机水平翻转给定的`PIL.Image`,概率为`0.5`。即：一半的概率翻转，一半的概率不翻转。

## 

## RandomRotation(degrees, interpolation=InterpolationMode.NEAREST)

RandomRotation类是随机旋转输入图像，也比较常用，具体参数可以看注释，在F.rotate()中主要是调用PILImage的rotate方法。

```python
transforms.RandomRotation(15)
```

## 剪裁



### CenterCrop(size: (int,tuple)) 中心剪裁 中心点固定

将给定的`PIL.Image`进行中心切割，得到给定的`size`，`size`可以是`tuple`，`(target_height, target_width)`。`size`也可以是一个`Integer`，在这种情况下，切出来的图片的形状是正方形。

CenterCrop是以输入图的中心点为中心点做指定size的crop操作，一般数据增强不会采用这个，因为当size固定的时候，在相同输入图像的情况下，N次CenterCrop的结果都是一样的。注释里面说明了size为int和序列时候尺寸的定义。



### RandomCrop(size: (int,tuple), padding=0) 中心剪裁 中心点随机选取

切割中心点的位置随机选取。`size`可以是`tuple`也可以是`Integer`。

**比前面的CenterCrop，这个RandomCrop更常用，差别就在于crop时的中心点坐标是随机的，并不是输入图像的中心点坐标，因此基本上每次crop生成的图像都是有差异的。**



### RandomResizedCrop(size: sequence, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR)) 推荐使用

**相比 CenterCrop,RandomCrop,RandomSizedCrop更加随机,更推荐使用**

前面不管是CenterCrop还是RandomCrop，在crop的时候其尺寸是固定的，而这个类则是randomsize的crop。该类主要用到3个参数：size、scale和ratio，总的来讲就是先做crop（用到scale和ratio），再resize到指定尺寸（用到size）。做crop的时候，其中心点坐标和长宽是由get_params方法得到的，在get_params方法中主要用到两个参数：scale和ratio，首先在scale限定的数值范围内随机生成一个数，用这个数乘以输入图像的面积作为crop后图像的面积；然后在ratio限定的数值范围内随机生成一个数，表示长宽的比值，根据这两个值就可以得到crop图像的长宽了。至于crop图像的中心点坐标，也是类似RandomCrop类一样是随机生成的。

```python
# 随机缩小剪裁,输出为(224, 224)
transforms.RandomResizedCrop(resize),  				    
```



### FiveCrop(size: (sequence or int)) 一出5

FiveCrop类，顾名思义就是从一张输入图像中crop出5张指定size的图像，这5张图像包括4个角的图像和一个center crop的图像。

> 参数和例子

```python
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
            If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
```

### TenCrop(size: (sequence or int)) 一出10

TenCrop类和前面FiveCrop类类似，只不过在FiveCrop的基础上，再将输入图像进行水平或竖直翻转，然后再进行FiveCrop操作，这样一张输入图像就能得到10张crop结果。

> 参数和例子

```python
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
```



## 填充

### Pad(padding, fill=0) 填充

将给定的`PIL.Image`的所有边用给定的`pad value`填充。 `padding：`要填充多少像素 `fill：`用什么值填充 例子：

```python
from torchvision import transforms
from PIL import Image
padding_img = transforms.Pad(padding=10, fill=0)
img = Image.open('test.jpg')

print(type(img))
print(img.size)

padded_img=padding(img)
print(type(padded_img))
print(padded_img.size)
<class 'PIL.PngImagePlugin.PngImageFile'>
(10, 10)
<class 'PIL.Image.Image'>
(30, 30) #由于上下左右都要填充10个像素，所以填充后的size是(30,30)
```

## ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) 亮度,对比度等

主要是修改输入图像的4大参数值：brightness, contrast and saturation，hue，也就是亮度，对比度，饱和度和色度。可以根据注释来合理设置这4个参数。

> 类解释和参数解释

- brightness = (0, 1)
- contrast = (0, 1)
- saturation = (0, 1)
- hue = (0.5, 0.5) 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

```python
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
```

## Grayscale(num_output_channels=1) 转化为灰度图

> 参数

- num_output_channels (int): (1 or 3) number of channels desired for output image

> 返回值
>
> num_output_channels == 1 返回单通道
>
> num_output_channels == 3 返回RGB都相同的三个通道

```python
If ``num_output_channels == 1`` : returned image is single channel
If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b
```

### RandomGrayscale(p=0.1) 指定的概率转换灰度图

和前面的Grayscale类类似，只不过变成了按照指定的概率进行转换。

> 参数

- p (float): 转换为灰度的可能性

> 返回值
>
> 输入通道为1,返回就为1;3通道也是如此

```python
PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
with probability (1-p).
- If input image is 1 channel: grayscale version is 1 channel
- If input image is 3 channel: grayscale version is 3 channel with r == g == b
```

## LinearTransformation(transformation_matrix, mean_vector)

> 参数解释

- transformation_matrix (Tensor): tensor [D x D], D = C x H x W

- mean_vector (Tensor): tensor [D], D = C x H x W

  **用一个变换矩阵去乘输入图像得到输出结果。**

# interpolation介绍

```python
class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"
```

> 使用方法

```python
interpolation=InterpolationMode.BILINEAR
```

# Conversion Transforms

## ToTensor() 转为为tensor

Convert a `PIL Image` or `numpy.ndarray` to tensor的过程，在PyTorch中常用PIL库来读取图像数据，因此这个方法相当于搭建了PILImage和Tensor的桥梁。**另外要强调的是在做数据归一化之前必须要把PILImage转成Tensor，而其他resize或crop操作则不需要。**

把一个取值范围是`[0,255]`的`PIL.Image`或者`shape`为`(H,W,C)`的`numpy.ndarray`，转换成形状为`[C,H,W]`，取值范围是`[0,1.0]`的`torch.FloadTensor`

```python
data = np.random.randint(0, 255, size=300)
img = data.reshape(10,10,3)
print(img.shape)
img_tensor = transforms.ToTensor()(img) # 转换成tensor
print(img_tensor)
```

## ToPILImage()  转换成PIL.Image

从Tensor到PILImage的过程，和前面ToTensor类的相反的操作。

将`shape`为`(C,H,W)`的`Tensor`或`shape`为`(H,W,C)`的`numpy.ndarray`转换成`PIL.Image`，值不变。

# 通用变换

## Lambda(lambd) 自定义函数

使用`lambd`作为转换器。