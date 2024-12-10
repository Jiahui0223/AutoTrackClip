import matplotlib.pyplot as plt

def visualize_video_frames(frames):
    """可视化视频的起始、中间和结束帧。

    Args:
        frames (np.ndarray): 视频帧列表 (T x H x W x C)。
    """
    if len(frames) == 0:
        print("No frames loaded.")
        return

    print(f"Total frames loaded: {len(frames)}")
    print(f"Frame shape: {frames[0].shape}")

    plt.figure(figsize=(10, 5))
    # 起始帧
    plt.subplot(1, 3, 1)
    plt.imshow(frames[0])
    plt.title("First Frame")
    plt.axis('off')

    # 中间帧
    middle_frame_idx = len(frames) // 2
    plt.subplot(1, 3, 2)
    plt.imshow(frames[middle_frame_idx])
    plt.title("Middle Frame")
    plt.axis('off')

    # 结束帧
    plt.subplot(1, 3, 3)
    plt.imshow(frames[-1])
    plt.title("Last Frame")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
