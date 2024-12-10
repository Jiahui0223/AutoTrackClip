import cv2
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from deep_sort_realtime.deepsort_tracker import DeepSort


class PoseFeatureExtractor:
    def __init__(self, detr_model_name="facebook/detr-resnet-50", tracker_params=None):
        """
        初始化姿势特征提取器。

        Args:
            detr_model_name (str): DETR 模型名称。
            tracker_params (dict): DeepSORT 跟踪器参数。
        """
        # 初始化 DETR 模型和处理器
        self.processor = DetrImageProcessor.from_pretrained(detr_model_name, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(detr_model_name, revision="no_timm")

        # 初始化 DeepSORT 跟踪器
        self.tracker = DeepSort(**(tracker_params or {}))

    def extract_pose_features(self, video_path, confidence_threshold=0.9):
        """
        从视频中提取姿势特征。

        Args:
            video_path (str): 视频文件路径。
            confidence_threshold (float): 物体检测置信度阈值。

        Returns:
            dict: 包含对象 ID 和轨迹的姿势特征。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video at {video_path}")

        pose_features = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 将帧转换为 RGB 格式并处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # 准备输入并推理
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            # 获取检测结果
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]

            # 提取检测框
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detections.append(([box[0], box[1], box[2] - box[0], box[3] - box[1]], score.item(), label.item()))

            # 更新跟踪器
            tracks = self.tracker.update_tracks(detections, frame=frame)

            # 记录跟踪信息
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if track_id not in pose_features:
                    pose_features[track_id] = {"trajectory": [], "label": label.item()}

                pose_features[track_id]["trajectory"].append((center_x, center_y))

        cap.release()
        return pose_features
