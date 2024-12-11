import torch
from torchvision.transforms import Normalize, Resize, Compose
import numpy as np

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

def compute_audio_change_rate(audio_features):
    """
    计算音频特征的变化率。
    
    Args:
        audio_features (np.ndarray): 音频特征 (帧数, 特征维度)。
    
    Returns:
        np.ndarray: 变化率得分 (帧数,)。
    """
    diff = np.abs(np.diff(audio_features, axis=0))  # 计算相邻帧的差异
    change_rate = np.mean(diff, axis=1)  # 平均变化率
    return change_rate


def compute_combined_score(video_features, audio_features, video_weight=0.9, audio_weight=0.1):
    """
    结合视频和音频特征，计算综合得分。

    Args:
        video_features (np.ndarray): 视频特征，形状为 (1, T)。
        audio_features (np.ndarray): 音频特征，形状为 (T, Features)。
        video_weight (float): 视频得分权重。
        audio_weight (float): 音频得分权重。

    Returns:
        np.ndarray: 综合得分，形状为 (T,)。
    """
    # 确保视频特征的维度为 (T,)
    if video_features.ndim == 2 and video_features.shape[0] == 1:
        video_scores = video_features.squeeze(0)  # 转为 (T,)
    else:
        raise ValueError("Expected video_features shape (1, T), got {}".format(video_features.shape))

    # 计算音频的逐时间步得分
    audio_scores = audio_features.mean(axis=-1)  # 计算每个时间步的音频特征均值，形状为 (T,)

    # 检查时间步是否一致
    if video_scores.shape[0] != audio_scores.shape[0]:
        raise ValueError("Mismatched time steps: video ({}) vs audio ({})".format(
            video_scores.shape[0], audio_scores.shape[0]))

    # 加权组合视频和音频得分
    combined_scores = video_weight * video_scores + audio_weight * audio_scores
    return combined_scores



def detect_highlights(combined_scores, threshold_factor=0.5):
    """
    根据综合得分检测高光时刻。
    
    Args:
        combined_scores (np.ndarray): 综合得分 (帧数,)。
        threshold_factor (float): 动态阈值因子。
    
    Returns:
        np.ndarray: 高光时刻的索引。
    """
    mean_score = combined_scores.mean()
    std_score = combined_scores.std()
    threshold = mean_score + threshold_factor * std_score  # 动态阈值
    highlights = np.where(combined_scores > threshold)[0]  # 高光时刻索引
    return highlights, threshold

