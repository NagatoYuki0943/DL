import cv2
import numpy as np


class ActivationsAndGradients:
    """Class for extracting activations and
    registering gradients from targeted intermediate layers"""

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        # -----------------------------#
        #   遍历指定的每一个网络
        # -----------------------------#
        for target_layer in target_layers:
            self.handles.append(
                # -----------------------------#
                #   注册正向传播的钩子函数
                #   保存输出
                # -----------------------------#
                target_layer.register_forward_hook(self.save_activation)
            )
            # -----------------------------#
            #   注册反向传播的钩子函数
            #   保存输出
            # -----------------------------#
            # 判断是为了兼容pytorch的不同版本
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, "register_full_backward_hook"):
                self.handles.append(
                    target_layer.register_full_backward_hook(self.save_gradient)
                )
            else:
                self.handles.append(
                    target_layer.register_backward_hook(self.save_gradient)
                )

    # -----------------------------#
    #   保存输出
    # -----------------------------#
    def save_activation(self, module, input, output):
        activation = output
        # -----------------------------------------------#
        #   reshape_transform transformer才会使用到
        # -----------------------------------------------#
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        # ----------------------------------#
        #   存入列表中,直接放到最后面
        #   引入正向传播数据从底层流向高层
        # ----------------------------------#
        self.activations.append(activation.cpu().detach())

    # -----------------------------#
    #   梯度信息
    # -----------------------------#
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        # ---------------------------------------------------------------------#
        #   保存grad的第一个输出,返回结果是一个元组,只有一个tensor所以获取第一个
        # ---------------------------------------------------------------------#
        grad = grad_output[0]
        # -----------------------------------------------#
        #   reshape_transform transformer才会使用到
        # -----------------------------------------------#
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        # -------------------------------------#
        #   梯度将数据添加到列表最前面
        #   因为梯度反向传播,新数据的浅层特征的
        # -------------------------------------#
        self.gradients = [grad.cpu().detach()] + self.gradients

    # ----------------------------------#
    #   正向传播过程
    # ----------------------------------#
    def __call__(self, x):
        # 清空记录的信息
        self.gradients = []
        self.activations = []
        # 允许模型会触发钩子函数
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self, model, target_layers, reshape_transform=None, use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        # ---------------------------------------------#
        #   捕获正向传播过程中的特征层A以及反向传播的A'
        # ---------------------------------------------#
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform
        )

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    # -------------------------------------------------------------------#
    #   求梯度权重,求宽高均值
    #   \alpha_k^c = \frac 1 Z \sum_i \sum_j \frac {∂y^c} {∂A_{ij}^k}
    # -------------------------------------------------------------------#
    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    # --------------------------------#
    #   计算loss
    # --------------------------------#
    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        # 遍历target列表,长度是图片数目
        for i in range(len(target_category)):
            # --------------------------------------------------#
            #   找对应类别的输出
            #   i: 第几张图片
            #   target_category[i]: 指定类别索引
            # --------------------------------------------------#
            loss = loss + output[i, target_category[i]]
        return loss

    # ---------------------------------------------------------------#
    #   针对每一层计算cam
    #   activations: 特征层
    #   grads: 对应的梯度
    #   L_{Grad−CAM}^c = ReLU(\sum_k a_k^c A^k) 不包含relu的部分
    # ---------------------------------------------------------------#
    def get_cam_image(self, activations, grads):
        # 求梯度权重,求宽高均值
        weights = self.get_cam_weights(grads)
        # 特征层*权重
        weighted_activations = weights * activations
        # 求和
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    # -------------------------------#
    #   计算每一层的cam
    # -------------------------------#
    def compute_cam_per_layer(self, input_tensor):
        # -------------------------------#
        #   获得正向传播的特征层
        # -------------------------------#
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        # -------------------------------#
        #   获取反向梯度信息
        # -------------------------------#
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # 得到图片宽高
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        # ----------------------------------#
        #   同时遍历特征层和对应的梯度信息
        # ----------------------------------#
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # ------------------------------------#
            #   针对每一层特征层和对应的梯度计算cam
            #   求梯度权重,求宽高均值
            # ------------------------------------#
            cam = self.get_cam_image(layer_activations, layer_grads)

            # ------------------------------------------------------#
            #   小于0的数值全部置为0,就是relu
            #   L_{Grad−CAM}^c = ReLU(\sum_k a_k^c A^k)  relu部分
            # ------------------------------------------------------#
            cam[cam < 0] = (
                0  # works like mute the min-max scale in the function of scale_cam_image
            )
            # 后处理部分
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    # ------------------------------------------------------#
    #   后处理部分
    #   减去最小值,除以最大值,缩放到图片大小,转换为彩色
    # ------------------------------------------------------#
    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    # ----------------------------------#
    #   正向传播过程
    # ----------------------------------#
    def __call__(self, input_tensor, target_category=None):
        """
        input_tensor:    输入数据
        target_category: 感兴趣的类别
        """

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # ---------------------------------------------#
        #   正向传播得到网络输出logits(未经过softmax)
        # ---------------------------------------------#
        output = self.activations_and_grads(input_tensor)

        # ---------------------------------------------------------------------#
        #   target为整数
        #   根据target生成长度和图片数量一致的target,这样可以一次性处理多张图片
        # ---------------------------------------------------------------------#
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        # ---------------------------------------------------------------------#
        #   如果没有给定类别,就指定概率最大的类别索引
        #   根据target生成长度和图片数量一致的target,这样可以一次性处理多张图片
        # ---------------------------------------------------------------------#
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert len(target_category) == input_tensor.size(0)

        # 清除梯度,反向传播
        self.model.zero_grad()
        # -------------------------------#
        #   计算loss
        # -------------------------------#
        loss = self.get_loss(output, target_category)
        # -------------------------------#
        #   反向传播,激活反向传播钩子
        # -------------------------------#
        loss.backward(retain_graph=True)

        # -----------------------------------------------------------------#
        # 计算每一个层的cam,返回列表
        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        # -----------------------------------------------------------------#
        cam_per_layer = self.compute_cam_per_layer(input_tensor)

        # ------------------------------------#
        #   所指定的所有层cam进行融合
        # ------------------------------------#
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}"
            )
            return True


# ------------------------------#
#   融合原图和cam并处理
# ------------------------------#
def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.    原图
    :param mask: The cam mask.                          cam
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    # ------------------------------------------------------#
    #   cam转化到0~022之间并转化为彩色图片 0为蓝色 255为红色
    # ------------------------------------------------------#
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # 再缩放到0~1之间
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    # 融合图片,除以最大值,乘以255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h : h + size]
    else:
        w = (new_w - size) // 2
        img = img[:, w : w + size]

    return img
