# 训练集,验证集,测试集都需要标准化
## transforms.Normalize

- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]



```python
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(resize*1.25),
        # transforms.RandomCrop((resize, resize)),
        transforms.RandomResizedCrop(resize),  				    # 随机缩小剪裁,输出为(resize, resize)
        transforms.RandomRotation(45),						    # 随机旋转，-45到45度之间随机选
        transforms.RandomHorizontalFlip(p=0.5),                 # 随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),                   # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),# 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),                    # 概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],             # 均值，标准差
                            [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(int(resize*1.25)),
        transforms.RandomCrop((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
}
```



## 将标准化的数据还原

> 值 * std + mean

```python
def draw(image, target):
    '''
    画图
    '''
    # print(image)
    # print(target)

    # [c, h, w]
    image_numpy = image.cpu().detach().float().numpy()
    # print(image_numpy[0])

    # print(image_numpy[0].shape) # (224, 224)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(len(mean)): #反标准化
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]

    print(image_numpy[0])
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(c, h, w)变为(h, w, c)

    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(image_numpy)
    plt.axis('off')

    # 添加title
    plt.title(str(target.cpu().detach().item()))

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题
    plt.show()

```

