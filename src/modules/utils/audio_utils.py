import librosa
import numpy as np
import moviepy.editor as mp


def extract_audio_from_video(video_path, output_audio_path=None, sr=16000):
    """
    从视频文件中提取音频，并保存为音频文件（可选）。
    
    Args:
        video_path (str): 视频文件路径。
        output_audio_path (str): 保存提取音频的路径（如果为 None，不保存音频）。
        sr (int): 提取音频的采样率（默认 16000）。
    
    Returns:
        np.ndarray: 提取的音频数据。
    """
    print(f"Extracting audio from video: {video_path}")
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio_samples = audio.to_soundarray(fps=sr, nbytes=2).mean(axis=1)  # 转为单声道
    if output_audio_path:
        audio.write_audiofile(output_audio_path, fps=sr)
        print(f"Audio saved to: {output_audio_path}")
    return audio_samples


def extract_audio_features_from_video(video_path, sr=16000, n_mfcc=13, feature_type='mfcc'):
    """
    从视频文件中提取音频特征。
    
    Args:
        video_path (str): 视频文件路径。
        sr (int): 采样率（默认 16000）。
        n_mfcc (int): MFCC 特征数量（仅在 feature_type='mfcc' 时使用）。
        feature_type (str): 特征类型，可选值为 'mfcc'、'chroma'、'spectrogram'。
    
    Returns:
        np.ndarray: 提取的音频特征。
    """
    # 提取音频数据
    audio_samples = extract_audio_from_video(video_path, sr=sr)
    print(f"Audio samples extracted with shape: {audio_samples.shape}")

    # 提取音频特征
    if feature_type == 'mfcc':
        print(f"Extracting MFCC features with n_mfcc={n_mfcc}")
        features = librosa.feature.mfcc(y=audio_samples, sr=sr, n_mfcc=n_mfcc)
        return features.mean(axis=1)

    elif feature_type == 'chroma':
        print("Extracting Chroma features")
        chroma = librosa.feature.chroma_stft(y=audio_samples, sr=sr)
        return chroma.mean(axis=1)

    elif feature_type == 'spectrogram':
        print("Extracting Spectrogram features")
        spectrogram = librosa.feature.melspectrogram(y=audio_samples, sr=sr)
        return librosa.power_to_db(spectrogram).mean(axis=1)

    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


def normalize_audio_features(features):
    """
    对音频特征进行归一化。
    
    Args:
        features (np.ndarray): 原始音频特征。
    
    Returns:
        np.ndarray: 归一化后的音频特征。
    """
    mean = np.mean(features)
    std = np.std(features)
    return (features - mean) / (std + 1e-6)


# 测试函数（仅供开发调试使用）
if __name__ == "__main__":
    # 替换为您的视频文件路径
    video_path = "assets/example.mp4"

    # 提取音频特征
    mfcc_features = extract_audio_features_from_video(video_path, feature_type='mfcc')
    print(f"MFCC Features Shape: {mfcc_features.shape}")
    print(f"MFCC Features: {mfcc_features}")
