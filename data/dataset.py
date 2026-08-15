import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
import random

from .drr import DRRGenerator, load_ct_image, resample_volume, center_crop_or_pad


class SpineCTDataset(Dataset):
    """
    脊柱CT重建数据集

    加载3D CT数据，生成配对的双平面DRR（冠状面+矢状面）
    返回: {ct, xray_coronal, xray_sagittal}
    """

    def __init__(self,
                 data_root: str,
                 file_list: str,
                 volume_size: Tuple[int, int, int] = (128, 128, 128),
                 voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
                 ct_min: int = -1000,
                 ct_max: int = 4096,
                 norm_min: float = -1.0,
                 norm_max: float = 1.0,
                 use_augmentation: bool = True,
                 flip_prob: float = 0.5,
                 rotate_range: float = 10.0,
                 mode: str = "train"):
        """
        Args:
            data_root: 数据根目录
            file_list: 数据列表文件路径，每行一个文件名
            volume_size: 目标体数据尺寸 (D, H, W)
            voxel_spacing: 目标体素间距
            ct_min: CT值最小值（HU）
            ct_max: CT值最大值（HU）
            norm_min: 归一化最小值
            norm_max: 归一化最大值
            use_augmentation: 是否使用数据增强
            flip_prob: 翻转概率
            rotate_range: 旋转角度范围
            mode: 模式 (train / val / test)
        """
        super().__init__()

        self.data_root = data_root
        self.volume_size = volume_size
        self.voxel_spacing = voxel_spacing
        self.ct_min = ct_min
        self.ct_max = ct_max
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.use_augmentation = use_augmentation and (mode == "train")
        self.flip_prob = flip_prob
        self.rotate_range = rotate_range
        self.mode = mode

        # DRR生成器
        self.drr_generator = DRRGenerator(ct_min=ct_min, ct_max=ct_max)

        # 加载文件列表
        self.file_list = self._load_file_list(file_list)
        print(f"[{mode}] 加载 {len(self.file_list)} 个样本")

    def _load_file_list(self, file_list_path: str) -> list:
        """加载文件列表"""
        with open(file_list_path, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f.readlines() if line.strip()]
        return files

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """CT值归一化到 [norm_min, norm_max]"""
        volume = np.clip(volume, self.ct_min, self.ct_max)
        volume = (volume - self.ct_min) / (self.ct_max - self.ct_min)
        volume = volume * (self.norm_max - self.norm_min) + self.norm_min
        return volume

    def _denormalize(self, volume: np.ndarray) -> np.ndarray:
        """反归一化回HU值"""
        volume = (volume - self.norm_min) / (self.norm_max - self.norm_min)
        volume = volume * (self.ct_max - self.ct_min) + self.ct_min
        return volume

    def _augment(self, ct_volume: np.ndarray) -> np.ndarray:
        """数据增强：随机翻转、旋转等"""
        # 左右翻转（沿X轴）
        if random.random() < self.flip_prob:
            ct_volume = np.flip(ct_volume, axis=2).copy()

        # 上下翻转（沿Z轴）
        if random.random() < self.flip_prob:
            ct_volume = np.flip(ct_volume, axis=0).copy()

        return ct_volume

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            dict: {
                'ct': 3D CT体数据 [1, D, H, W],
                'xray_coronal': 冠状面DRR [1, D, W],
                'xray_sagittal': 矢状面DRR [1, D, H]
            }
        """
        # 加载CT
        filename = self.file_list[idx]
        ct_path = os.path.join(self.data_root, filename)

        ct_volume, original_spacing = load_ct_image(ct_path)

        # 重采样到目标体素间距
        ct_volume = resample_volume(ct_volume, original_spacing, self.voxel_spacing)

        # 中心裁剪/填充到固定尺寸
        ct_volume = center_crop_or_pad(ct_volume, self.volume_size)

        # 数据增强
        if self.use_augmentation:
            ct_volume = self._augment(ct_volume)

        # 生成双平面DRR
        coronal_drr, sagittal_drr = self.drr_generator.generate_biplanar(ct_volume)

        # 归一化CT
        ct_norm = self._normalize(ct_volume)

        # DRR归一化到 [-1, 1]
        coronal_norm = coronal_drr * 2.0 - 1.0
        sagittal_norm = sagittal_drr * 2.0 - 1.0

        # 转为Tensor，增加通道维度
        ct_tensor = torch.from_numpy(ct_norm).float().unsqueeze(0)  # [1, D, H, W]
        coronal_tensor = torch.from_numpy(coronal_norm).float().unsqueeze(0)  # [1, D, W]
        sagittal_tensor = torch.from_numpy(sagittal_norm).float().unsqueeze(0)  # [1, D, H]

        return {
            'ct': ct_tensor,
            'xray_coronal': coronal_tensor,
            'xray_sagittal': sagittal_tensor,
            'filename': filename
        }


def build_dataloader(cfg, mode: str = "train"):
    """
    构建数据加载器

    Args:
        cfg: 配置对象
        mode: 模式 (train / val / test)

    Returns:
        dataloader: PyTorch DataLoader
    """
    data_cfg = cfg.data

    if mode == "train":
        file_list = data_cfg.train_list
        batch_size = cfg.train.batch_size
        shuffle = True
        use_augmentation = data_cfg.use_augmentation
    elif mode == "val":
        file_list = data_cfg.val_list
        batch_size = cfg.train.batch_size
        shuffle = False
        use_augmentation = False
    elif mode == "test":
        file_list = data_cfg.test_list
        batch_size = 1
        shuffle = False
        use_augmentation = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dataset = SpineCTDataset(
        data_root=data_cfg.data_root,
        file_list=file_list,
        volume_size=data_cfg.volume_size,
        voxel_spacing=data_cfg.voxel_spacing,
        ct_min=data_cfg.ct_min,
        ct_max=data_cfg.ct_max,
        norm_min=data_cfg.norm_min,
        norm_max=data_cfg.norm_max,
        use_augmentation=use_augmentation,
        flip_prob=data_cfg.flip_prob,
        rotate_range=data_cfg.rotate_range,
        mode=mode
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows上设为0避免多进程问题
        pin_memory=True,
        drop_last=(mode == "train")
    )

    return dataloader
