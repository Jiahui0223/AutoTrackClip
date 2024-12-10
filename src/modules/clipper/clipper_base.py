import cv2
import os

from .clipper_base import ClipperBase


class BasicClipper(ClipperBase):
    """
    基础视频剪辑实现，按高光时间段裁剪视频并生成最终输出。
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
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open video file {video_path}")

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 临时保存剪辑片段
        temp_clips = []

        for i, (start_time, end_time) in enumerate(highlights):
            # 计算帧范围
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # 创建临时文件路径
            temp_clip_path = f"temp_clip_{i}.mp4"
            temp_clips.append(temp_clip_path)

            # 创建 VideoWriter
            out = cv2.VideoWriter(temp_clip_path, fourcc, fps, (frame_width, frame_height))

            # 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame

            while frame_index <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_index += 1

            out.release()

        cap.release()

        # 合并所有临时片段
        final_output_path = self._combine_clips(temp_clips, output_path)

        # 删除临时文件
        for temp_clip_path in temp_clips:
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)

        return final_output_path

    def _combine_clips(self, clip_paths, output_path):
        """
        合并多个视频片段为最终输出视频。

        Args:
            clip_paths (list): 临时剪辑片段路径列表。
            output_path (str): 最终输出视频路径。

        Returns:
            str: 保存的最终视频路径。
        """
        import moviepy.editor as mp

        # 加载所有剪辑片段
        clips = [mp.VideoFileClip(clip) for clip in clip_paths]

        # 合并为单个视频
        final_clip = mp.concatenate_videoclips(clips)

        # 写入最终输出
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return output_path
