import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def sinusoidal_position_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    正弦位置编码（用于时间步编码）

    Args:
        timesteps: 时间步张量 [B]
        dim: 编码维度

    Returns:
        emb: 位置编码 [B, dim]
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """时间步嵌入层"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间步 [B]
        Returns:
            emb: 时间嵌入 [B, dim]
        """
        emb = sinusoidal_position_embedding(t, self.dim)
        emb = self.mlp(emb)
        return emb


class GroupNorm3D(nn.Module):
    """3D Group Normalization"""

    def __init__(self, num_channels: int, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ResnetBlock3D(nn.Module):
    """
    3D ResNet块
    支持时间步嵌入注入
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 dropout: float = 0.1,
                 num_groups: int = 8):
        super().__init__()

        self.norm1 = GroupNorm3D(in_channels, num_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        # 时间步投影
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = GroupNorm3D(out_channels, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        # 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, D, H, W]
            t_emb: 时间步嵌入 [B, time_emb_dim]
        Returns:
            out: 输出特征 [B, out_channels, D, H, W]
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # 注入时间步
        t = self.time_proj(self.act(t_emb))
        h = h + t[:, :, None, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock3D(nn.Module):
    """
    3D自注意力块
    用于捕获长距离依赖关系
    """

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.norm = GroupNorm3D(channels, num_groups)
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, D, H, W]
        Returns:
            out: 输出特征 [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape

        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # 展平空间维度
        q = q.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        k = k.reshape(B, C, -1)  # [B, C, N]
        v = v.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        # 注意力计算
        attn = torch.bmm(q, k) * self.scale  # [B, N, N]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)  # [B, N, C]
        out = out.permute(0, 2, 1).reshape(B, C, D, H, W)

        out = self.proj_out(out)

        return x + out


class Downsample3D(nn.Module):
    """3D下采样（步长卷积）"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D上采样（最近邻插值 + 卷积）"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
