import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: 解码器上采样后的引导特征 (batch, gate_channels, H_g, W_g)
        # x: 编码器跳跃特征 (batch, skip_channels, H_x, W_x)

        # 1. 通道变换
        g1 = self.W_g(g)  # (batch, inter_channels, H_g, W_g)
        x1 = self.W_x(x)  # (batch, inter_channels, H_x, W_x)
        # 2. 核心修改：将g1的尺寸插值到与x1完全一致（用双线性插值，保持通道数不变）
        # size=(H_x, W_x) 对应x1的空间尺寸
        g1 = F.interpolate(
            g1,
            size=(x1.size(2), x1.size(3)),  # 按x1的H和W对齐
            mode='bilinear',
            align_corners=True
        )

        # 3. 特征融合与激活（此时g1和x1尺寸完全匹配）
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # 4. 权重应用
        return x * psi