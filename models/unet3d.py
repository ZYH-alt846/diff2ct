import torch
import torch.nn as nn
from typing import Tuple, List

from .modules import (
    TimeEmbedding,
    ResnetBlock3D,
    AttentionBlock3D,
    Downsample3D,
    Upsample3D,
)


class UNet3D(nn.Module):
    """
    3D UNet 主干网络（带时间步嵌入）

    结构：编码器 -> 瓶颈层 -> 解码器 + 跳跃连接
    每个level的最终输出作为一个跳跃连接
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 time_emb_dim: int = 128,
                 use_attention: bool = True,
                 attention_levels: Tuple[int, ...] = (2, 3),
                 dropout: float = 0.1,
                 num_groups: int = 8):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = len(channel_mults)
        self.use_attention = use_attention
        self.attention_levels = attention_levels
        self.num_res_blocks = num_res_blocks

        # 时间步嵌入
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # 初始卷积
        self.init_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # ========== 编码器 ==========
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        current_channels = base_channels

        for level in range(self.num_levels):
            out_ch = base_channels * channel_mults[level]

            # 多个ResNet块
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock3D(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                    num_groups=num_groups
                ))
                current_channels = out_ch

            # 注意力块
            if use_attention and level in attention_levels:
                blocks.append(AttentionBlock3D(current_channels, num_groups))

            self.down_blocks.append(blocks)

            # 下采样（最后一层不下采样）
            if level < self.num_levels - 1:
                self.down_samples.append(Downsample3D(current_channels))
            else:
                self.down_samples.append(None)

        # ========== 瓶颈层 ==========
        self.mid_block1 = ResnetBlock3D(
            current_channels, current_channels, time_emb_dim, dropout, num_groups
        )
        self.mid_attn = AttentionBlock3D(current_channels, num_groups) if use_attention else nn.Identity()
        self.mid_block2 = ResnetBlock3D(
            current_channels, current_channels, time_emb_dim, dropout, num_groups
        )

        # ========== 解码器 ==========
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(self.num_levels)):
            out_ch = base_channels * channel_mults[level]

            # 上采样
            if level < self.num_levels - 1:
                self.up_samples.append(Upsample3D(current_channels))
            else:
                self.up_samples.append(None)

            # 多个ResNet块（第一个带跳跃连接，输入通道翻倍）
            blocks = nn.ModuleList()
            # 第一个block：拼接skip后输入
            blocks.append(ResnetBlock3D(
                in_channels=current_channels + out_ch,  # skip连接的通道数是out_ch
                out_channels=out_ch,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                num_groups=num_groups
            ))
            current_channels = out_ch
            # 剩余block
            for _ in range(num_res_blocks - 1):
                blocks.append(ResnetBlock3D(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                    num_groups=num_groups
                ))

            # 注意力块
            if use_attention and level in attention_levels:
                blocks.append(AttentionBlock3D(current_channels, num_groups))

            self.up_blocks.append(blocks)

        # 输出层
        self.out_norm = nn.GroupNorm(num_groups, current_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(current_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 时间步嵌入
        t_emb = self.time_embedding(t)

        # 初始卷积
        h = self.init_conv(x)

        # 编码器 + 保存跳跃连接（每个level的最终输出）
        skips = []

        for level in range(self.num_levels):
            # ResNet块 + 注意力
            for block in self.down_blocks[level]:
                if isinstance(block, AttentionBlock3D):
                    h = block(h)
                else:
                    h = block(h, t_emb)

            # 保存该level的输出作为skip
            skips.append(h)

            # 下采样
            if self.down_samples[level] is not None:
                h = self.down_samples[level](h)

        # 瓶颈层
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # 解码器
        for level in range(self.num_levels):
            # 上采样
            if self.up_samples[level] is not None:
                h = self.up_samples[level](h)

            # 取出对应level的skip
            skip = skips.pop()

            # ResNet块 + 跳跃连接
            for i, block in enumerate(self.up_blocks[level]):
                if isinstance(block, AttentionBlock3D):
                    h = block(h)
                elif i == 0:
                    # 第一个block拼接skip
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, t_emb)
                else:
                    h = block(h, t_emb)

        # 输出
        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)

        return out
