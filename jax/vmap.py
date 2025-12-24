# 在 PyTorch 中，如果你写了一个处理单张图片的逻辑，想要支持 Batch，通常需要考虑 x.unsqueeze(0) 或者在各种算子里调整 dim=1 或 dim=0。
# 但在 JAX 中，你只需要编写纯粹处理单张图片的逻辑，剩下的维度扩展交给 vmap。

import jax
import jax.numpy as jnp


# --- 第一步：编写只处理“单张图片”的函数 ---
# 假设输入 shape 是 (H, W, C)，完全不考虑 Batch 维度
def process_single_image(image, color_offset):
    # 模拟简单的图像处理：归一化 + 偏移
    image = image / 255.0
    processed = image + color_offset
    return jnp.clip(processed, 0.0, 1.0)

# --- 第二步：使用 vmap 瞬间支持 Batch ---
# in_axes 说明：
# image 参数在第 0 维进行展开（Batch 维度）
# color_offset 如果你想让每个 Batch 用同一个偏移，设为 None；
# 如果每个 Batch 有各自的偏移，也设为 0。
batch_process = jax.vmap(process_single_image, in_axes=(0, None))

if __name__ == "__main__":
    # 准备数据
    # 单张图片 (224, 224, 3)
    single_img = jnp.ones((224, 224, 3)) * 128

    # 一个 Batch 的图片 (8, 224, 224, 3)
    batch_img = jnp.ones((8, 224, 224, 3)) * 128
    offset = 0.1

    # 1. 处理单张图片
    res_single = process_single_image(single_img, offset)
    print(f"单图处理输出形状: {res_single.shape}") # (224, 224, 3)

    # 2. 处理一个 Batch（不需要修改 process_single_image 的任何代码）
    res_batch = batch_process(batch_img, offset)
    print(f"Batch 处理输出形状: {res_batch.shape}") # (8, 224, 224, 3)
