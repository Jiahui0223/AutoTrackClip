import torch
import torch.nn as nn
import torch.nn.functional as F


class HighlightDetectionHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes=1, dropout_rate=0.5):
        """
        高光检测头，使用3D卷积进行时间特征提取和分类。
        """
        super(HighlightDetectionHead, self).__init__()

        # 时间特征提取模块
        self.temporal_conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0)
        )
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        self.relu1 = nn.ReLU()

        self.temporal_conv2 = nn.Conv3d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0)
        )
        self.bn2 = nn.BatchNorm3d(hidden_dim)
        self.relu2 = nn.ReLU()

        # 残差连接
        self.residual = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=(1, 1, 1)
        )

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # 分类头
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播逻辑。
        Args:
            x (torch.Tensor): SlowFast 模型的输出，形状 [Batch, Channels, T, H, W]。
        Returns:
            torch.Tensor: 每个时间步的得分，形状 [Batch, T, num_classes]。
        """
        residual = self.residual(x)
        x = self.temporal_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.temporal_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x += residual  # 残差连接

        x = self.global_pool(x)  # 形状: [Batch, hidden_dim, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # 形状: [Batch, hidden_dim, T]

        x = x.permute(0, 2, 1)  # 形状: [Batch, T, hidden_dim]
        x = self.dropout(x)
        x = self.fc1(x)  # 形状: [Batch, T, 128]
        x = F.relu(x)
        x = self.fc2(x)  # 形状: [Batch, T, num_classes]

        return x
