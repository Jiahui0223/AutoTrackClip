import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, video_dim, audio_dim, hidden_dim, output_dim):
        """
        多模态融合模型，将视频和音频特征结合并进行预测。

        Args:
            video_dim (int): 视频特征的输入维度。
            audio_dim (int): 音频特征的输入维度。
            hidden_dim (int): 隐藏层的维度。
            output_dim (int): 输出维度。
        """
        super(MultimodalModel, self).__init__()

        # 视频特征映射
        self.video_fc = nn.Linear(video_dim, hidden_dim)

        # 音频特征映射
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)

        # 融合后的全连接层
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, video_features, audio_features):
        """
        前向传播。
        
        Args:
            video_features (torch.Tensor): 视频特征 (Batch, T, Video_Dim)。
            audio_features (torch.Tensor): 音频特征 (Batch, T, Audio_Dim)。

        Returns:
            torch.Tensor: 模型输出 (Batch, T, Output_Dim)。
        """
        # 映射视频特征
        video_out = self.video_fc(video_features)  # (Batch, T, Hidden_Dim)

        # 映射音频特征
        audio_out = self.audio_fc(audio_features)  # (Batch, T, Hidden_Dim)

        # 融合视频和音频特征
        fused_features = torch.cat([video_out, audio_out], dim=-1)  # (Batch, T, Hidden_Dim * 2)

        # 输出预测结果
        output = self.fusion_fc(fused_features)  # (Batch, T, Output_Dim)
        return output
