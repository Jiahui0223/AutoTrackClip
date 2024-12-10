import cv2
import numpy as np
import torch
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

import cv2
import numpy as np
import torch
from src.modules.utils.preprocess_utils import preprocess_frame

def load_video(path):
    """加载视频文件并返回帧列表。

    Args:
        path (str): 视频文件路径。

    Returns:
        np.ndarray: 视频帧列表 (T x H x W x C)。
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色为 RGB
        frames.append(frame)
    cap.release()
    return np.array(frames)

def load_video_for_slowfast(path, frames_per_clip=32):
    """加载视频并对其进行预处理以适配 SlowFast 模型。

    Args:
        path (str): 视频文件路径。
        frames_per_clip (int): 每个剪辑的帧数。

    Returns:
        torch.Tensor: 处理后的视频张量 (C x T x H x W)。
    """
    frames = load_video(path)

    # 对视频帧进行变换
    transform = Compose([
        Lambda(lambda x: UniformTemporalSubsample(frames_per_clip)(torch.tensor(x).permute(3, 0, 1, 2))),
        Resize((224, 224)),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    video_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)  # 转换为 T x C x H x W
    return transform(video_tensor / 255.0)  # 归一化至 [0, 1]


def load_and_process_frames(video_path, target_frames=960, resize=(224, 224)):
    """
    加载视频并逐帧预处理。
    Args:
        video_path (str): 视频文件路径。
        target_frames (int): 目标采样帧数。
        resize (tuple): 帧的目标大小。
    Returns:
        torch.Tensor: 预处理后的视频张量，形状为 [C, T, H, W]。
    """
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换 BGR 到 RGB
            processed_frame = preprocess_frame(frame, resize)  # 逐帧预处理
            processed_frames.append(processed_frame)
        except Exception as e:
            print(f"跳过异常帧: {e}")
    
    cap.release()
    print(f"共处理帧数: {len(processed_frames)}")

    # 将帧堆叠为张量
    video_tensor = torch.stack(processed_frames, dim=1)  # [C, T, H, W]

    # 采样帧
    total_frames = video_tensor.shape[1]
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    sampled_video = video_tensor[:, indices, :, :]  # 采样后的张量
    print(f"采样后视频张量形状: {sampled_video.shape}")

    return sampled_video