import onnx


def check_onnx(onnx_path):
    """检查onnx模型是否损坏
    Args:
        onnx_path (str): onnx模型路径
    """
    # 载入onnx模型
    model_ = onnx.load(onnx_path)
    # print(model_)
    # 检查IR是否良好
    try:
        onnx.checker.check_model(model_)
    except Exception:
        print(f"{onnx_path} Model incorrect/")
    else:
        print(f"{onnx_path} Model correct")
