import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from typing import Tuple


class DRRGenerator:
    """
    数字重建放射影像（DRR）生成器
    基于射线追踪的正交投影算法
    """

    def __init__(self, ct_min: int = -1000, ct_max: int = 4096):
        self.ct_min = ct_min
        self.ct_max = ct_max

    def generate_coronal(self, ct_volume: np.ndarray) -> np.ndarray:
        """
        生成冠状面DRR（沿前后轴/Y轴投影）

        Args:
            ct_volume: 3D CT数组，shape (D, H, W) 对应 (Z, Y, X)

        Returns:
            coronal_drr: 冠状面DRR，shape (D, W)
        """
        # 将HU值转换为衰减系数（近似线性映射）
        ct_clipped = np.clip(ct_volume, self.ct_min, self.ct_max)
        # 归一化到 [0, 1]
        ct_norm = (ct_clipped - self.ct_min) / (self.ct_max - self.ct_min)

        # 沿Y轴（轴1）积分求和，模拟X射线穿透
        drr = np.sum(ct_norm, axis=1)

        # 归一化到 [0, 1]
        drr = drr / (drr.max() + 1e-8)
        return drr

    def generate_sagittal(self, ct_volume: np.ndarray) -> np.ndarray:
        """
        生成矢状面DRR（沿左右轴/X轴投影）

        Args:
            ct_volume: 3D CT数组，shape (D, H, W)

        Returns:
            sagittal_drr: 矢状面DRR，shape (D, H)
        """
        ct_clipped = np.clip(ct_volume, self.ct_min, self.ct_max)
        ct_norm = (ct_clipped - self.ct_min) / (self.ct_max - self.ct_min)

        # 沿X轴（轴2）积分求和
        drr = np.sum(ct_norm, axis=2)

        drr = drr / (drr.max() + 1e-8)
        return drr

    def generate_axial(self, ct_volume: np.ndarray) -> np.ndarray:
        """
        生成轴向DRR（沿上下轴/Z轴投影）

        Args:
            ct_volume: 3D CT数组，shape (D, H, W)

        Returns:
            axial_drr: 轴向DRR，shape (H, W)
        """
        ct_clipped = np.clip(ct_volume, self.ct_min, self.ct_max)
        ct_norm = (ct_clipped - self.ct_min) / (self.ct_max - self.ct_min)

        # 沿Z轴（轴0）积分求和
        drr = np.sum(ct_norm, axis=0)

        drr = drr / (drr.max() + 1e-8)
        return drr

    def generate_biplanar(self, ct_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时生成冠状面和矢状面双平面DRR

        Returns:
            coronal_drr: 冠状面DRR
            sagittal_drr: 矢状面DRR
        """
        coronal = self.generate_coronal(ct_volume)
        sagittal = self.generate_sagittal(ct_volume)
        return coronal, sagittal


def load_ct_image(file_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    加载CT图像（NIfTI格式）

    Args:
        file_path: CT文件路径 (.nii / .nii.gz)

    Returns:
        ct_array: CT体数据数组
        spacing: 体素间距 (z, y, x)
    """
    sitk_image = sitk.ReadImage(file_path)
    ct_array = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()[::-1]  # (x, y, z) -> (z, y, x)
    return ct_array, spacing


def resample_volume(volume: np.ndarray,
                    original_spacing: Tuple[float, float, float],
                    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0)
                    ) -> np.ndarray:
    """
    将CT重采样到目标体素间距

    Args:
        volume: 原始CT体数据
        original_spacing: 原体素间距
        target_spacing: 目标体素间距

    Returns:
        resampled_volume: 重采样后的体数据
    """
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(original_spacing[::-1])  # (z,y,x) -> (x,y,z)

    # 计算新尺寸
    original_size = sitk_image.GetSize()
    new_size = [
        int(round(original_size[i] * original_spacing[::-1][i] / target_spacing[::-1][i]))
        for i in range(3)
    ]

    # 重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing[::-1])
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())

    resampled_image = resampler.Execute(sitk_image)
    resampled_volume = sitk.GetArrayFromImage(resampled_image)

    return resampled_volume


def center_crop_or_pad(volume: np.ndarray,
                       target_size: Tuple[int, int, int] = (128, 128, 128)
                       ) -> np.ndarray:
    """
    中心裁剪或填充到目标尺寸

    Args:
        volume: 输入体数据
        target_size: 目标尺寸 (D, H, W)

    Returns:
        result: 裁剪/填充后的体数据
    """
    result = np.zeros(target_size, dtype=volume.dtype)

    # 计算各维度的起始位置
    starts = []
    for i in range(3):
        if volume.shape[i] >= target_size[i]:
            start = (volume.shape[i] - target_size[i]) // 2
            starts.append(start)
        else:
            starts.append(0)

    # 计算各维度的截取长度
    crops = []
    for i in range(3):
        crop_len = min(volume.shape[i], target_size[i])
        crops.append(crop_len)

    # 目标位置
    target_starts = []
    for i in range(3):
        if volume.shape[i] < target_size[i]:
            start = (target_size[i] - volume.shape[i]) // 2
            target_starts.append(start)
        else:
            target_starts.append(0)

    # 执行裁剪/填充
    result[
        target_starts[0]:target_starts[0]+crops[0],
        target_starts[1]:target_starts[1]+crops[1],
        target_starts[2]:target_starts[2]+crops[2]
    ] = volume[
        starts[0]:starts[0]+crops[0],
        starts[1]:starts[1]+crops[1],
        starts[2]:starts[2]+crops[2]
    ]

    return result
