import cv2
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch
import numpy as np

class BasicClipper:
    """
    基础视频剪辑器，用于根据高光时间段裁剪视频。
    """
    def clip(self, video_path, highlights, output_path):
        """
        根据高光时刻剪辑视频并保存输出。

        Args:
            video_path (str): 输入视频路径。
            highlights (list): 高光时间段列表，每个元素是 (start_time, end_time) 的元组。
            output_path (str): 输出视频路径。

        Returns:
            str: 保存的剪辑视频路径。
        """
        print(f"开始剪辑视频: {video_path}")
        print(f"高光时段: {highlights}")
        
        # 打开视频
        try:
            video = VideoFileClip(video_path)
        except Exception as e:
            raise ValueError(f"无法打开视频文件: {e}")
        
        # 裁剪高光片段
        clips = []
        for i, (start_time, end_time) in enumerate(highlights):
            try:
                clip = video.subclip(start_time, end_time)
                clips.append(clip)
                print(f"裁剪片段 {i}: {start_time}s - {end_time}s")
            except Exception as e:
                print(f"裁剪片段失败 ({start_time}-{end_time}): {e}")
                continue

        if not clips:
            raise ValueError("没有可用的高光片段，无法生成输出视频。")

        # 合并片段
        try:
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print(f"输出视频已保存到: {output_path}")
        except Exception as e:
            raise ValueError(f"视频合并失败: {e}")
        
        return output_path

    def convert_highlight_scores_to_intervals(self, highlight_scores, threshold, video_duration):
        """
        将高光分数转换为时间段。

        Args:
            highlight_scores (torch.Tensor or np.ndarray): 高光分数 (Batch, T)。
            threshold (float): 高光分数的阈值。
            video_duration (float): 视频总时长（秒）。

        Returns:
            list: 高光时间段列表，每个元素是 (start_time, end_time) 的元组。
        """
        # 如果输入是 PyTorch 张量，则转换为 NumPy 数组
        if isinstance(highlight_scores, torch.Tensor):
            scores = highlight_scores.squeeze(0).detach().cpu().numpy()  # (T,)
        elif isinstance(highlight_scores, np.ndarray):
            scores = highlight_scores.squeeze(0)  # 如果是 NumPy 数组，直接处理
        else:
            raise TypeError("highlight_scores 必须是 torch.Tensor 或 np.ndarray 类型")

        # 计算每个时间步对应的时间
        time_per_step = video_duration / len(scores)

        # 找到高于阈值的时间步索引
        highlight_indices = (scores > threshold).nonzero()[0]

        # 合并连续的时间步为时间段
        highlights = []
        start_idx = None
        for i, idx in enumerate(highlight_indices):
            if start_idx is None:
                start_idx = idx
            # 如果当前索引不连续或已到最后一个索引
            if i == len(highlight_indices) - 1 or highlight_indices[i + 1] != idx + 1:
                end_idx = idx
                start_time = start_idx * time_per_step
                end_time = (end_idx + 1) * time_per_step
                highlights.append((start_time, end_time))
                start_idx = None

        return highlights

    
    def get_video_duration(self, video_path):
        """
        使用 moviepy 获取视频总时长。
        
        Args:
            video_path (str): 视频文件路径。
        
        Returns:
            float: 视频总时长（单位：秒）。
        """
        try:
            video = VideoFileClip(video_path)
            duration = video.duration  # 获取视频总时长（秒）
            return duration
        except Exception as e:
            raise ValueError(f"无法获取视频时长: {e}")

    def convert_frame_indices_to_time_ranges(self, frame_indices, fps):
        """
        将帧索引转换为高光时间段 (start_time, end_time)。

        Args:
            frame_indices (list or tensor): 高光帧索引列表。
            fps (int): 视频的帧率。

        Returns:
            list: 高光时间段列表，每个元素是 (start_time, end_time) 的元组。
        """
        time_per_frame = 1 / fps
        frame_indices = sorted(frame_indices)  # 确保帧索引有序
        highlights = []

        # 合并连续帧为时间段
        start_frame = frame_indices[0]
        for i in range(1, len(frame_indices)):
            if frame_indices[i] != frame_indices[i - 1] + 1:
                # 如果当前帧不连续，结束当前时间段
                end_frame = frame_indices[i - 1]
                highlights.append((start_frame * time_per_frame, end_frame * time_per_frame))
                start_frame = frame_indices[i]
        # 添加最后一个时间段
        highlights.append((start_frame * time_per_frame, frame_indices[-1] * time_per_frame))

        return highlights
