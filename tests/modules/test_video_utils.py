from src.modules.utils.video_utils import load_video, load_video_for_slowfast

def test_load_video():
    video_path = "assets/example.mp4"  # 确保示例视频存在
    frames = load_video(video_path)
    assert len(frames) > 0, "视频帧加载失败"
    assert frames[0].shape[-1] == 3, "帧应为 RGB 图像"

def test_load_video_for_slowfast():
    video_path = "assets/example.mp4"
    processed_video = load_video_for_slowfast(video_path)
    assert processed_video.shape[1] == 32, "视频片段的帧数应为 32"
