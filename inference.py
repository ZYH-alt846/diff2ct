import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import SimpleITK as sitk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import cfg
from data.dataset import SpineCTDataset
from data.drr import DRRGenerator, load_ct_image, resample_volume, center_crop_or_pad
from models.x2noise_net import X2NoiseNet
from models.diffusion import GaussianDiffusion


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    # 构建模型
    model = X2NoiseNet(
        in_channels=cfg.model.in_channels,
        cond_channels=cfg.model.base_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        time_emb_dim=cfg.model.time_emb_dim,
        use_attention=cfg.model.use_attention,
        attention_levels=cfg.model.attention_levels,
        dropout=cfg.model.dropout,
        num_groups=cfg.model.num_groups,
        volume_size=cfg.data.volume_size,
    )

    diffusion = GaussianDiffusion(
        num_timesteps=cfg.diffusion.num_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        loss_type=cfg.diffusion.loss_type,
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    diffusion = diffusion.to(device)

    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, diffusion


def denormalize_ct(volume, ct_min=-1000, ct_max=4096, norm_min=-1.0, norm_max=1.0):
    """将归一化的CT值反归一化回HU值"""
    volume = (volume - norm_min) / (norm_max - norm_min)
    volume = volume * (ct_max - ct_min) + ct_min
    return volume


def save_as_nifti(volume, output_path, spacing=(2.0, 2.0, 2.0)):
    """保存为NIfTI格式"""
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(spacing[::-1])  # (z,y,x) -> (x,y,z)
    sitk.WriteImage(sitk_image, output_path)
    print(f"Saved: {output_path}")


@torch.no_grad()
def reconstruct(model, diffusion, xray_coronal, xray_sagittal, device,
                volume_size=(128, 128, 128), verbose=True):
    """
    从双平面X光重建CT

    Args:
        model: 训练好的模型
        diffusion: 扩散过程
        xray_coronal: 冠状面X光 [1, 1, D, W]
        xray_sagittal: 矢状面X光 [1, 1, D, H]
        device: 设备
        volume_size: 体数据尺寸
        verbose: 是否显示进度

    Returns:
        ct_recon: 重建的CT体数据 [D, H, W]
    """
    B = 1
    shape = (B, 1) + volume_size

    # 反向采样
    ct_recon = diffusion.p_sample_loop(
        model, shape, xray_coronal, xray_sagittal,
        device, clip_denoised=True, verbose=verbose
    )

    # 去掉batch和channel维度
    ct_recon = ct_recon.squeeze(0).squeeze(0).cpu().numpy()

    return ct_recon


def reconstruct_from_file(model, diffusion, ct_file_path, device, output_path=None):
    """
    从CT文件重建（用于测试，先用CT生成DRR再重建）

    Args:
        model: 模型
        diffusion: 扩散过程
        ct_file_path: 输入CT文件路径
        device: 设备
        output_path: 输出保存路径
    """
    # 加载CT
    ct_volume, original_spacing = load_ct_image(ct_file_path)

    # 预处理
    ct_volume = resample_volume(ct_volume, original_spacing, cfg.data.voxel_spacing)
    ct_volume = center_crop_or_pad(ct_volume, cfg.data.volume_size)

    # 生成DRR
    drr_gen = DRRGenerator(ct_min=cfg.data.ct_min, ct_max=cfg.data.ct_max)
    coronal_drr, sagittal_drr = drr_gen.generate_biplanar(ct_volume)

    # DRR归一化到 [-1, 1]
    coronal_norm = coronal_drr * 2.0 - 1.0
    sagittal_norm = sagittal_drr * 2.0 - 1.0

    # 转为Tensor
    xray_cor = torch.from_numpy(coronal_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    xray_sag = torch.from_numpy(sagittal_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    print(f"Input CT shape: {ct_volume.shape}")
    print(f"Coronal DRR shape: {coronal_drr.shape}")
    print(f"Sagittal DRR shape: {sagittal_drr.shape}")
    print("Reconstructing...")

    # 重建
    ct_recon = reconstruct(
        model, diffusion, xray_cor, xray_sag, device,
        cfg.data.volume_size, verbose=True
    )

    # 反归一化回HU值
    ct_recon_hu = denormalize_ct(
        ct_recon, cfg.data.ct_min, cfg.data.ct_max,
        cfg.data.norm_min, cfg.data.norm_max
    )

    # 保存
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_as_nifti(ct_recon_hu, output_path, cfg.data.voxel_spacing)

    return ct_recon_hu


def main():
    parser = argparse.ArgumentParser(description="Diff2CT Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入CT文件路径（用于生成DRR并重建）')
    parser.add_argument('--output', type=str, default=None, help='输出CT文件路径')
    parser.add_argument('--device', type=str, default=None, help='设备')
    args = parser.parse_args()

    # 设备
    device = torch.device(args.device or cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model, diffusion = load_model(args.checkpoint, device)

    # 输出路径
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"./outputs/recon_{base_name}.nii.gz"

    # 重建
    ct_recon = reconstruct_from_file(
        model, diffusion, args.input, device, args.output
    )

    print(f"\nReconstruction completed!")
    print(f"Output shape: {ct_recon.shape}")
    print(f"Output range: [{ct_recon.min():.1f}, {ct_recon.max():.1f}] HU")


if __name__ == "__main__":
    main()
