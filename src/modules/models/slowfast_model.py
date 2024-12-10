import torch
from pytorchvideo.models.hub import slowfast_r50
from src.modules.utils.alignment_utils import align_time_dimensions

def load_slowfast_model():
    """
    加载预训练的 SlowFast 模型。
    Returns:
        torch.nn.Module: SlowFast 模型实例。
    """
    model = slowfast_r50(pretrained=True)
    model.eval()  # 设置为评估模式
    print("SlowFast model loaded successfully!")
    return model

def prepare_slowfast_inputs(frames, alpha=4):
    """
    为 SlowFast 模型准备输入，确保 Slow 和 Fast 通道的时间维度正确对齐。
    Args:
        frames (torch.Tensor): 预处理后的视频张量，形状 [3, T, H, W]。
        alpha (int): Slow Pathway 的时间采样比例。
    Returns:
        list[torch.Tensor]: [slow_pathway, fast_pathway]。
    """
    if frames.ndim != 4 or frames.size(0) != 3:
        raise ValueError(f"Expected input shape [3, T, H, W], but got {frames.shape}")

    T = frames.size(1)  # 时间维度

    # Fast Pathway: 使用所有时间步
    fast_pathway = frames.unsqueeze(0)  # 添加批次维度 -> [1, 3, T, H, W]

    # Slow Pathway: 按比例采样时间步
    if T % alpha != 0:
        # 如果时间维度不能整除 alpha，裁剪到最近的可整除值
        T = (T // alpha) * alpha
        frames = frames[:, :T, :, :]
        print(f"Frames truncated to {T} time steps for alignment.")

    slow_pathway = frames[:, ::alpha, :, :].unsqueeze(0)  # 按 alpha 下采样 -> [1, 3, T/alpha, H, W]

    # 最终输出 Slow 和 Fast 通道
    return [slow_pathway, fast_pathway]
