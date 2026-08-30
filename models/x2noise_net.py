import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .unet3d import UNet3D


class XRayEncoder2D(nn.Module):
    """
    2D X光编码器
    对单张X光图像进行2D卷积特征提取
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, out_channels: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),

            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),

            nn.Conv2d(base_channels * 4, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 2D X光图像 [B, C, H, W]
        Returns:
            feat: 2D特征 [B, out_channels, H/4, W/4]
        """
        return self.encoder(x)


class X2NoiseNet(nn.Module):
    """
    X2NoiseNet: 双平面X光条件扩散模型

    核心思路：
    1. 双平面X光分别通过2D编码器提取特征
    2. 将2D特征升维为3D特征（沿投影方向复制）
    3. 两个视角的3D特征维度置换对齐后融合
    4. 融合特征与带噪CT拼接，送入3D UNet预测噪声
    """

    def __init__(self,
                 in_channels: int = 1,
                 cond_channels: int = 32,
                 base_channels: int = 32,
                 channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 time_emb_dim: int = 128,
                 use_attention: bool = True,
                 attention_levels: Tuple[int, ...] = (2, 3),
                 dropout: float = 0.1,
                 num_groups: int = 8,
                 volume_size: Tuple[int, int, int] = (128, 128, 128),
                 use_checkpoint: bool = False):
        """
        Args:
            in_channels: CT输入通道数
            cond_channels: 条件特征通道数
            base_channels: UNet基础通道数
            channel_mults: UNet通道倍数
            num_res_blocks: 每层ResNet块数
            time_emb_dim: 时间嵌入维度
            use_attention: 是否使用注意力
            attention_levels: 注意力层级
            dropout: dropout率
            num_groups: GroupNorm组数
            volume_size: 体数据尺寸 (D, H, W)
        """
        super().__init__()

        self.volume_size = volume_size
        self.cond_channels = cond_channels

        # 2D X光编码器（冠状面和矢状面各一个）
        self.coronal_encoder = XRayEncoder2D(1, base_channels // 2, cond_channels)
        self.sagittal_encoder = XRayEncoder2D(1, base_channels // 2, cond_channels)

        # 3D UNet（输入通道 = CT通道 + 条件通道）
        self.unet = UNet3D(
            in_channels=in_channels + cond_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            use_attention=use_attention,
            attention_levels=attention_levels,
            dropout=dropout,
            num_groups=num_groups,
            use_checkpoint=use_checkpoint
        )

        # 条件特征3D卷积融合
        self.cond_fusion = nn.Sequential(
            nn.Conv3d(cond_channels, cond_channels, 3, padding=1),
            nn.GroupNorm(num_groups, cond_channels),
            nn.SiLU(),
        )

    def _xray_to_3d_condition(self,
                              xray_coronal: torch.Tensor,
                              xray_sagittal: torch.Tensor,
                              volume_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        将双平面2D X光转换为3D条件特征

        Args:
            xray_coronal: 冠状面X光 [B, 1, D, W]
            xray_sagittal: 矢状面X光 [B, 1, D, H]
            volume_size: 目标体数据尺寸 (D, H, W)

        Returns:
            cond_3d: 3D条件特征 [B, cond_channels, D, H, W]
        """
        B = xray_coronal.shape[0]
        D, H, W = volume_size

        # 2D编码
        feat_coronal = self.coronal_encoder(xray_coronal)  # [B, C, D/4, W/4]
        feat_sagittal = self.sagittal_encoder(xray_sagittal)  # [B, C, D/4, H/4]

        # 上采样回原始尺寸
        feat_coronal = F.interpolate(feat_coronal, size=(D, W), mode='bilinear', align_corners=False)
        feat_sagittal = F.interpolate(feat_sagittal, size=(D, H), mode='bilinear', align_corners=False)

        # 升维：沿投影方向复制
        # 冠状面 -> 沿Y轴(H轴)复制
        cond_coronal = feat_coronal.unsqueeze(3).repeat(1, 1, 1, H, 1)  # [B, C, D, H, W]

        # 矢状面 -> 沿X轴(W轴)复制，然后维度置换对齐
        cond_sagittal = feat_sagittal.unsqueeze(4).repeat(1, 1, 1, 1, W)  # [B, C, D, H, W]

        # 特征融合（相加）
        cond_3d = cond_coronal + cond_sagittal

        # 3D卷积进一步融合
        cond_3d = self.cond_fusion(cond_3d)

        return cond_3d

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                xray_coronal: torch.Tensor,
                xray_sagittal: torch.Tensor) -> torch.Tensor:
        """
        前向传播：预测噪声

        Args:
            x_t: 带噪CT [B, 1, D, H, W]
            t: 时间步 [B]
            xray_coronal: 冠状面X光 [B, 1, D, W]
            xray_sagittal: 矢状面X光 [B, 1, D, H]

        Returns:
            noise_pred: 预测的噪声 [B, 1, D, H, W]
        """
        B, C, D, H, W = x_t.shape

        # 生成3D条件特征
        cond_3d = self._xray_to_3d_condition(xray_coronal, xray_sagittal, (D, H, W))

        # 拼接带噪CT和条件特征
        x_input = torch.cat([x_t, cond_3d], dim=1)  # [B, 1+cond_channels, D, H, W]

        # UNet预测噪声
        noise_pred = self.unet(x_input, t)

        return noise_pred
