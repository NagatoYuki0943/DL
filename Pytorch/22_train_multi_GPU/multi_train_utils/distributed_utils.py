import os

import torch
import torch.distributed as dist


# --------------------------------------------------------------#
#   初始化各进程环境
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_gpu_using_launch.py
# 上面的 --use_env 会传入环境变量 RANK WORLD_SIZE LOCAL_RANK
#   给args添加rank, world_size, gpu
# --------------------------------------------------------------#
def init_distributed_mode(args):
    # --------------------------------------------------------------#
    #   WORLD_SIZE: 所有机器中使用的进程数量,一个进程对应一个GPU
    #   RANK: 代表所有进程中的第几个进程, 0代表主进程
    #   LOCAL_RANK: 对应当前机器中第几个进程
    # --------------------------------------------------------------#
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    # --------------------------------------------------------------#
    #   指定当前设备使用的GPU
    # --------------------------------------------------------------#
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"  # 通信后端，nvidia GPU推荐使用NCCL
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    # --------------------------------------------------------------#
    #   创建进程组
    # --------------------------------------------------------------#
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    # --------------------------------------------------------------#
    #   等待所有GPU运行完毕
    # --------------------------------------------------------------#
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


# --------------------------------------------------------------#
#   求所有设备数据的均值
# --------------------------------------------------------------#
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    # 不同设备求和操作
    with torch.inference_mode():
        dist.all_reduce(value)
        # 返回均值
        if average:
            value /= world_size
        # 返回总和
        return value
