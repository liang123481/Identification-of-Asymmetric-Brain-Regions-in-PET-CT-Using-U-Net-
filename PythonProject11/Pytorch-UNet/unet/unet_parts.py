""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_gate import *

class DoubleConv(nn.Module):#两次卷积层
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),#直接修改输入张量（Tensor）的内存空间，将计算结果覆盖到输入上，不额外开辟新的内存。
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),#将输入的 mid_channels 个特征通道，通过 128 个 3×3 卷积核的计算，转换为 out_channels 个新的特征通道，同时通过填充保持尺寸不变，且不使用偏置以配合后续的批归一化。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)#输入的x是一个张量


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # 定义注意力门（参数：引导特征通道数，跳跃特征通道数，中间通道数）
        # 引导特征通道数 = in_channels//2（解码器上采样后的通道数）
        # 跳跃特征通道数 = in_channels//2（编码器输出的通道数）
        self.attention = AttentionGate(gate_channels=in_channels // 2, skip_channels=in_channels // 2,
                                       inter_channels=in_channels // 4)

    def forward(self, x1, x2):
        # x1: 解码器输出特征（引导特征）
        # x2: 编码器输出特征（跳跃连接特征）

        # 上采样解码器特征
        x1_up = self.up(x1)

        # 用注意力门加权编码器特征
        x2_att = self.attention(g=x1_up, x=x2)  # 核心：注意力加权

        # 对齐尺寸
        diffY = x2_att.size()[2] - x1_up.size()[2]
        diffX = x2_att.size()[3] - x1_up.size()[3]
        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # 拼接加权后的跳跃特征和解码器特征
        x = torch.cat([x2_att, x1_up], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
