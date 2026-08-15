import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ProjectionLoss(nn.Module):
    """
    三轴正交投影一致性损失

    将3D体数据分别沿三个轴向投影，与对应投影计算损失，
    强制3D结构在三个正交视角下的几何一致性。
    """

    def __init__(self, loss_type: str = "mse", weight_axial: float = 1.0,
                 weight_coronal: float = 1.0, weight_sagittal: float = 1.0):
        """
        Args:
            loss_type: 损失类型 (mse / l1 / smooth_l1)
            weight_axial: 轴向投影损失权重
            weight_coronal: 冠状面投影损失权重
            weight_sagittal: 矢状面投影损失权重
        """
        super().__init__()
        self.loss_type = loss_type
        self.weights = {
            'axial': weight_axial,
            'coronal': weight_coronal,
            'sagittal': weight_sagittal,
        }

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算单视角投影损失"""
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, pred_volume: torch.Tensor, target_volume: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_volume: 预测体数据 [B, C, D, H, W]
            target_volume: 目标体数据 [B, C, D, H, W]

        Returns:
            total_loss: 总投影损失
            loss_dict: 各视角损失字典
        """
        # 轴向投影（沿Z轴/D轴求和）
        pred_axial = torch.sum(pred_volume, dim=2)
        target_axial = torch.sum(target_volume, dim=2)
        loss_axial = self._compute_loss(pred_axial, target_axial)

        # 冠状面投影（沿Y轴/H轴求和）
        pred_coronal = torch.sum(pred_volume, dim=3)
        target_coronal = torch.sum(target_volume, dim=3)
        loss_coronal = self._compute_loss(pred_coronal, target_coronal)

        # 矢状面投影（沿X轴/W轴求和）
        pred_sagittal = torch.sum(pred_volume, dim=4)
        target_sagittal = torch.sum(target_volume, dim=4)
        loss_sagittal = self._compute_loss(pred_sagittal, target_sagittal)

        # 加权求和
        total_loss = (
            self.weights['axial'] * loss_axial +
            self.weights['coronal'] * loss_coronal +
            self.weights['sagittal'] * loss_sagittal
        ) / 3.0

        loss_dict = {
            'proj_total': total_loss.item(),
            'proj_axial': loss_axial.item(),
            'proj_coronal': loss_coronal.item(),
            'proj_sagittal': loss_sagittal.item(),
        }

        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """
    组合损失：基础扩散损失 + 投影一致性损失
    """

    def __init__(self, projection_weight: float = 0.3, loss_type: str = "mse"):
        super().__init__()
        self.projection_weight = projection_weight
        self.projection_loss = ProjectionLoss(loss_type=loss_type)
        self.loss_type = loss_type

    def forward(self,
                noise_pred: torch.Tensor,
                noise_true: torch.Tensor,
                pred_x0: torch.Tensor = None,
                true_x0: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            noise_pred: 预测噪声
            noise_true: 真实噪声
            pred_x0: 预测的x0（用于投影损失，可选）
            true_x0: 真实的x0（用于投影损失，可选）

        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量字典
        """
        # 基础噪声损失
        if self.loss_type == "mse":
            noise_loss = F.mse_loss(noise_pred, noise_true)
        elif self.loss_type == "l1":
            noise_loss = F.l1_loss(noise_pred, noise_true)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss_dict = {
            'noise_loss': noise_loss.item(),
        }

        total_loss = noise_loss

        # 投影损失（如果提供了x0）
        if pred_x0 is not None and true_x0 is not None:
            proj_loss, proj_dict = self.projection_loss(pred_x0, true_x0)
            total_loss = total_loss + self.projection_weight * proj_loss
            loss_dict.update(proj_dict)

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
