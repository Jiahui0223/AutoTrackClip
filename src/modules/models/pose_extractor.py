import torch.nn as nn

class PoseFeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_joints, num_layers, hidden_dim):
        """
        姿态特征提取模块。
        Args:
            in_channels (int): 每个关节的输入特征维度。
            num_joints (int): 姿态中关节数量。
            num_layers (int): 线性层数量。
            hidden_dim (int): 隐藏层维度。
        """
        super(PoseFeatureExtractor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(num_joints * in_channels, hidden_dim))
            layers.append(nn.ReLU())
        self.pose_net = nn.Sequential(*layers)

    def forward(self, pose_feats):
        """
        Args:
            pose_feats (torch.Tensor): 姿态特征，形状为 [B, T, num_joints, in_channels]。
        Returns:
            torch.Tensor: 处理后的姿态特征，形状为 [B, T, hidden_dim]。
        """
        B, T, num_joints, in_channels = pose_feats.shape
        pose_feats = pose_feats.view(B, T, -1)  # 展平关节维度
        return self.pose_net(pose_feats)
