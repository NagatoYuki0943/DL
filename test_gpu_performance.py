import torch
import time
import pynvml


def test_fp_performance(
    num_warmup=5, num_iterations=1000, tensor_size=(1024, 1024), dtype=torch.float16
):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability(0)[0] >= 7:
        raise RuntimeError("当前设备不支持FP16/BF16格式,请使用支持FP16/BF16的显卡。")

    a = torch.randn(tensor_size, dtype=dtype, device="cuda")
    b = torch.randn(tensor_size, dtype=dtype, device="cuda")

    # 预热
    for _ in range(num_warmup):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    total_time = 0.0

    for _ in range(num_iterations):
        start_time = time.time()

        c = torch.matmul(a, b)

        torch.cuda.synchronize()

        end_time = time.time()

        total_time += end_time - start_time

    flops = 2 * tensor_size[0] * tensor_size[1] * tensor_size[0]
    average_time = total_time / num_iterations
    tflops = flops / average_time / 1e12

    return tflops


def test_bf16_performance(num_warmup=5, num_iterations=1000, tensor_size=(1024, 1024)):
    return test_fp_performance(num_warmup, num_iterations, tensor_size, torch.bfloat16)


if __name__ == "__main__":
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle).encode("utf-8")
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
        print(f"GPU型号: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.2f} MB")
        print(f"矩阵大小：(40960,40960)")

        print("测试BF16算力:")
        average_bf16_tflops = test_bf16_performance(
            num_warmup=5, num_iterations=50, tensor_size=(40960, 40960)
        )
        print(f"BF16算力平均值: {average_bf16_tflops:.2f} TFlops")

        print("测试FP16算力:")
        average_fp16_tflops = test_fp_performance(
            num_warmup=5,
            num_iterations=50,
            tensor_size=(40960, 40960),
            dtype=torch.float16,
        )
        print(f"FP16算力平均值: {average_fp16_tflops:.2f} TFlops")

        print("测试FP32算力:")
        print(f"矩阵大小：(20480,20480)")

        average_fp32_tflops = test_fp_performance(
            num_warmup=1,
            num_iterations=10,
            tensor_size=(20480, 20480),
            dtype=torch.float32,
        )
        print(f"FP32算力平均值: {average_fp32_tflops:.2f} TFlops")

    except RuntimeError as e:
        print(e)
    finally:
        pynvml.nvmlShutdown()
