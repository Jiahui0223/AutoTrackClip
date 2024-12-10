import torch
from torchvision.transforms import Normalize, Resize, Compose

def preprocess_frame(frame, resize=(224, 224)):
    """
    预处理单个帧以适配 SlowFast 输入格式。
    Args:
        frame (numpy.ndarray): 单个视频帧，形状为 [H, W, C]。
        resize (tuple): 帧的目标大小。
    Returns:
        torch.Tensor: 预处理后的帧，形状为 [3, H, W]。
    """
    # 转换帧为 torch 张量: [H, W, C] -> [C, H, W]
    frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 归一化到 [0, 1]

    # 定义 Resize 和 Normalize 转换
    transform = Compose([
        Resize(resize, antialias=True),  # 调整帧大小
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])  # 标准化
    ])

    # 应用变换
    processed_frame = transform(frame_tensor)  # [3, H, W]
    return processed_frame
