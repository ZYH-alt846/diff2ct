"""
测量一致性约束推理脚本（创新点）
在扩散采样完成后，将重建CT投影回X光空间，与输入X光对比，用梯度下降迭代优化，
使得重建结果的投影更接近输入X光，从而提升几何结构正确性。
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from data.dataset import SpineCTDataset
from models.x2noise_net import X2NoiseNet
from models.diffusion import GaussianDiffusion
from torch.utils.data import DataLoader


def project_to_xray(volume, mode='coronal'):
    """将3D体数据投影为2D X光（沿投影方向求均值）"""
    if mode == 'coronal':
        # 冠状面：沿H轴（dim=3）投影
        return torch.mean(volume, dim=3)
    elif mode == 'sagittal':
        # 矢状面：沿W轴（dim=4）投影
        return torch.mean(volume, dim=4)
    elif mode == 'axial':
        # 轴向：沿D轴（dim=2）投影
        return torch.mean(volume, dim=2)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")


def measurement_consistency_refine(model, diffusion, xray_cor, xray_sag,
                                    device, num_iter=20, lr=0.01,
                                    consistency_weight=1.0,
                                    prior_weight=0.1):
    """
    测量一致性优化：在扩散采样结果基础上，通过投影一致性约束迭代优化。

    Args:
        model: 扩散模型
        diffusion: 扩散过程
        xray_cor: 冠状面X光 [B, 1, D, W]
        xray_sag: 矢状面X光 [B, 1, D, H]
        device: 设备
        num_iter: 优化迭代次数
        lr: 学习率
        consistency_weight: 投影一致性损失权重
        prior_weight: 扩散先验损失权重（保持生成结果的自然性）

    Returns:
        refined: 优化后的CT体数据 [B, 1, D, H, W]
    """
    B, _, D, W = xray_cor.shape
    _, _, _, H = xray_sag.shape

    # Step 1: 标准扩散采样得到初始重建
    print("Step 1: Running diffusion sampling...")
    with torch.no_grad():
        x0_init = diffusion.p_sample_loop(
            model, (B, 1, D, H, W),
            xray_cor, xray_sag, device,
            clip_denoised=True, verbose=False
        )

    # Step 2: 测量一致性迭代优化
    print(f"Step 2: Measurement consistency refinement ({num_iter} iterations)...")
    x_refined = x0_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_refined], lr=lr)

    for i in range(num_iter):
        optimizer.zero_grad()

        # 投影一致性损失
        proj_cor = project_to_xray(x_refined, 'coronal')  # [B, 1, D, W]
        proj_sag = project_to_xray(x_refined, 'sagittal')  # [B, 1, D, H]

        loss_cor = F.mse_loss(proj_cor, xray_cor)
        loss_sag = F.mse_loss(proj_sag, xray_sag)
        loss_consistency = (loss_cor + loss_sag) / 2.0

        # 先验损失：保持与初始扩散结果的距离，防止偏离自然图像流形
        loss_prior = F.mse_loss(x_refined, x0_init)

        # 总损失
        loss = consistency_weight * loss_consistency + prior_weight * loss_prior

        loss.backward()
        optimizer.step()

        # 裁剪到合理范围
        with torch.no_grad():
            x_refined.clamp_(-1.0, 1.0)

        if (i + 1) % 5 == 0:
            print(f"  Iter {i+1}/{num_iter}: consistency={loss_consistency.item():.6f}, "
                  f"prior={loss_prior.item():.6f}, total={loss.item():.6f}")

    return x0_init.detach(), x_refined.detach()


def compute_metrics(pred, target):
    """计算评估指标"""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    mae = np.mean(np.abs(pred_np - target_np))

    mse = np.mean((pred_np - target_np) ** 2)
    psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 100.0  # range [-1,1] -> max 4

    # SSIM (simplified)
    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2
    mu_pred = np.mean(pred_np)
    mu_target = np.mean(target_np)
    sigma_pred = np.var(pred_np)
    sigma_target = np.var(target_np)
    sigma_cross = np.mean((pred_np - mu_pred) * (target_np - mu_target))
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

    return {'mae': mae, 'psnr': psnr, 'ssim': ssim}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_iter', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--consistency_weight', type=float, default=1.0)
    parser.add_argument('--prior_weight', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
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
        use_checkpoint=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=cfg.diffusion.num_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
    ).to(device)

    # 加载测试数据
    test_dataset = SpineCTDataset(
        data_root=cfg.data.data_root,
        file_list=cfg.data.test_list,
        volume_size=cfg.data.volume_size,
        voxel_spacing=cfg.data.voxel_spacing,
        ct_min=cfg.data.ct_min,
        ct_max=cfg.data.ct_max,
        norm_min=cfg.data.norm_min,
        norm_max=cfg.data.norm_max,
        use_augmentation=False,
        mode='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 评估
    results_baseline = []
    results_refined = []

    for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if idx >= args.num_samples:
            break

        ct_gt = batch['ct'].to(device)
        xray_cor = batch['xray_coronal'].to(device)
        xray_sag = batch['xray_sagittal'].to(device)

        # 测量一致性优化
        x0_baseline, x0_refined = measurement_consistency_refine(
            model, diffusion, xray_cor, xray_sag, device,
            num_iter=args.num_iter, lr=args.lr,
            consistency_weight=args.consistency_weight,
            prior_weight=args.prior_weight
        )

        # 计算指标
        metrics_baseline = compute_metrics(x0_baseline, ct_gt)
        metrics_refined = compute_metrics(x0_refined, ct_gt)

        results_baseline.append(metrics_baseline)
        results_refined.append(metrics_refined)

        print(f"Sample {idx}:")
        print(f"  Baseline - MAE: {metrics_baseline['mae']:.4f}, PSNR: {metrics_baseline['psnr']:.2f}, SSIM: {metrics_baseline['ssim']:.4f}")
        print(f"  Refined  - MAE: {metrics_refined['mae']:.4f}, PSNR: {metrics_refined['psnr']:.2f}, SSIM: {metrics_refined['ssim']:.4f}")

    # 汇总结果
    print("\n" + "="*60)
    print("FINAL RESULTS (Measurement Consistency Refinement)")
    print("="*60)

    for name, results in [("Baseline (Diffusion Only)", results_baseline),
                           ("Baseline + Measurement Consistency", results_refined)]:
        mae_mean = np.mean([r['mae'] for r in results])
        mae_std = np.std([r['mae'] for r in results])
        psnr_mean = np.mean([r['psnr'] for r in results])
        psnr_std = np.std([r['psnr'] for r in results])
        ssim_mean = np.mean([r['ssim'] for r in results])
        ssim_std = np.std([r['ssim'] for r in results])

        print(f"\n{name}:")
        print(f"  MAE:  {mae_mean:.4f} ± {mae_std:.4f}")
        print(f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
        print(f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")

    # 保存结果
    output_path = os.path.join(cfg.train.output_dir, 'measurement_consistency_results.txt')
    with open(output_path, 'w') as f:
        f.write("Measurement Consistency Refinement Results\n")
        f.write(f"num_iter={args.num_iter}, lr={args.lr}, "
                f"consistency_weight={args.consistency_weight}, prior_weight={args.prior_weight}\n\n")
        for name, results in [("Baseline", results_baseline), ("Refined", results_refined)]:
            f.write(f"{name}:\n")
            f.write(f"  MAE:  {np.mean([r['mae'] for r in results]):.4f} ± {np.std([r['mae'] for r in results]):.4f}\n")
            f.write(f"  PSNR: {np.mean([r['psnr'] for r in results]):.2f} ± {np.std([r['psnr'] for r in results]):.2f}\n")
            f.write(f"  SSIM: {np.mean([r['ssim'] for r in results]):.4f} ± {np.std([r['ssim'] for r in results]):.4f}\n\n")
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
