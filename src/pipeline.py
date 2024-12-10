from src.modules.utils.video_utils import load_video, load_video_for_slowfast
from src.modules.utils.visualization import visualize_video_frames
from src.modules.models.slowfast_model import load_slowfast_model, prepare_slowfast_inputs

from src.modules.models.slowfast_model import load_slowfast_model, prepare_slowfast_inputs
from src.modules.models.debug_slowfast import DebugSlowFast

from src.modules.models.slowfast_model import load_slowfast_model, prepare_slowfast_inputs
from src.modules.models.highlight_detection import HighlightDetectionHead
from src.modules.utils.video_utils import load_and_process_frames


# from src.modules.utils.audio_utils import extract_audio_features, normalize_audio_features
from src.modules.utils.audio_utils import extract_audio_features_from_video, normalize_audio_features

from src.modules.utils.pose_utils import PoseFeatureExtractor
from src.modules.models.multimodal_model import MultimodalModel
import torch



import matplotlib.pyplot as plt
import torch


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def process_and_visualize_video(input_path: str):
    """加载视频、进行预处理并可视化结果。

    Args:
        input_path (str): 输入视频路径。
    """
    # 加载原始视频帧
    video_frames = load_video(input_path)

    # 可视化原始视频帧
    visualize_video_frames(video_frames)

    # 预处理以适配 SlowFast 模型
    processed_video = load_video_for_slowfast(input_path)

    print(f"Processed video shape: {processed_video.shape}")



def process_video(video_path):
    """
    加载和处理视频的完整流程。
    Args:
        video_path (str): 输入视频路径。
    """
    print(f"开始处理视频: {video_path}")
    preprocessed_video = load_and_process_frames(video_path)
    print(f"处理后视频张量形状: {preprocessed_video.shape}")
    return preprocessed_video




def process_video_with_model(video_path):
    """
    加载视频并使用 SlowFast 模型处理。
    """
    print(f"Processing video: {video_path}")
    
    # 加载并预处理视频
    preprocessed_video = load_and_process_frames(video_path)
    print(f"Preprocessed video shape: {preprocessed_video.shape}")

    # 加载模型
    model = load_slowfast_model()

    # 准备输入
    inputs = prepare_slowfast_inputs(preprocessed_video)
    print(f"Prepared inputs for SlowFast: Slow={inputs[0].shape}, Fast={inputs[1].shape}")

    # 模型推理
    with torch.no_grad():
        outputs = model(inputs)
    print("Model inference complete!")
    return outputs



def process_video_with_debug_model(video_path):
    """
    加载视频，使用 SlowFast 模型处理，并对输出进行调试。
    Args:
        video_path (str): 视频文件路径。
    """
    print(f"Processing video: {video_path}")

    # 加载并预处理视频
    preprocessed_video = load_and_process_frames(video_path)
    print(f"Preprocessed video shape: {preprocessed_video.shape}")

    # 准备输入
    inputs = prepare_slowfast_inputs(preprocessed_video)
    print(f"Prepared inputs for SlowFast: Slow={inputs[0].shape}, Fast={inputs[1].shape}")

    # 加载模型
    slowfast_model = load_slowfast_model()

    # 包装模型进行调试
    debug_model = DebugSlowFast(slowfast_model)

    # 模型推理
    with torch.no_grad():
        outputs = debug_model(inputs)
    print("Model inference complete!")
    return outputs


def process_video_for_highlight_detection(video_path):
    """
    使用 SlowFast 和 HighlightDetectionHead 处理视频并检测高光时刻。
    Args:
        video_path (str): 视频路径。
    """
    print(f"Processing video: {video_path}")

    # 加载视频并预处理
    preprocessed_video = load_and_process_frames(video_path)
    print(f"Preprocessed video shape: {preprocessed_video.shape}")


    # audio_features = extract_audio_features(video_path, feature_type='mfcc')
    # normalized_features = normalize_audio_features(audio_features)

    # 1.Video
    # 加载 SlowFast 模型
    slowfast_model = load_slowfast_model()

    debug_model = DebugSlowFast(slowfast_model)

    # 2.Audio
    print("AUDIO")

    # audio_features = extract_audio_features_from_video(video_path, feature_type='mfcc')
    # normalized_audio_features = normalize_audio_features(audio_features)
    # print(f"Normalized Audio Features Shape: {normalized_audio_features.shape}")

    # 3.Pose
    print("POSE")
    # pose_extractor = PoseFeatureExtractor()
    # pose_features = pose_extractor.extract_pose_features(video_path)
    # print(f"Pose features extracted: {pose_features}")


    # 准备输入
    slowfast_inputs = prepare_slowfast_inputs(preprocessed_video)

    # 获取 SlowFast 输出
    with torch.no_grad():
        # slowfast_outputs = slowfast_model(slowfast_inputs)
        slowfast_outputs = debug_model(slowfast_inputs)
    print(f"SlowFast outputs shape: {slowfast_outputs.shape}")

    # 加载 HighlightDetectionHead
    in_channels = slowfast_outputs.shape[1]
    hidden_dim = 512
    num_classes = 1
    highlight_head = HighlightDetectionHead(in_channels, hidden_dim, num_classes)
    #######

    # 获取高光检测结果
    highlight_scores = highlight_head(slowfast_outputs)  # 输出形状: [Batch, T, num_classes]
    highlight_scores = highlight_scores.squeeze(-1)  # 去掉 num_classes 维度，形状: [Batch, T]
    print(f"Highlight scores shape: {highlight_scores.shape}")

    # 处理结果 - 动态调整阈值
    mean_score = highlight_scores.mean().item()
    std_score = highlight_scores.std().item()
    highlight_threshold = mean_score + 0.5 * std_score  # 动态阈值: 平均值加半个标准差

    # 高光时间步判断
    highlights = (highlight_scores > highlight_threshold).nonzero(as_tuple=True)
    highlight_scores_np = highlight_scores.detach().cpu().numpy()  # 转换为 NumPy 数组

    print("Highlight scores for each time step:")
    print(highlight_scores_np)
    print(f"Dynamic threshold: {highlight_threshold}")
    print("Highlight time steps:", highlights)

    # # 可视化分数和高光时间步
    # plt.figure(figsize=(16, 8))

    # # 可视化高光分数曲线
    # plt.subplot(2, 1, 1)
    # plt.plot(highlight_scores_np[0], label="Highlight Scores")  # 只绘制 Batch 中第一个序列
    # plt.axhline(y=highlight_threshold, color='r', linestyle='--', label="Threshold")
    # plt.scatter(highlights[1], highlight_scores_np[0][highlights[1]], color='red', label="Highlights", zorder=5)
    # plt.xlabel("Time Steps")
    # plt.ylabel("Highlight Scores")
    # plt.title("Highlight Detection Scores Over Time")
    # plt.legend()
    # plt.grid(True)

    # # 可视化分数分布直方图
    # plt.subplot(2, 1, 2)
    # plt.hist(highlight_scores_np[0], bins=20, alpha=0.7, color='blue', label="Score Distribution")
    # plt.axvline(x=highlight_threshold, color='red', linestyle='--', label="Threshold")
    # plt.xlabel("Highlight Scores")
    # plt.ylabel("Frequency")
    # plt.title("Highlight Score Distribution")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    print("Highlight Scores NP:", highlight_scores_np)
    print("Highlight Scores NP Shape:", highlight_scores_np.shape)

    print("Processed Highlights:", highlights)


    return highlight_scores_np, highlights, highlight_threshold

