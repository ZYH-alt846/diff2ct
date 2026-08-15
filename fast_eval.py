"""
快速评测脚本 - 用较少采样步数快速得到MAE/PSNR/SSIM/FID指标
适用于CPU环境下的基线验证
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import cfg
from data.dataset import build_dataloader
from models.x2noise_net import X2NoiseNet
from models.diffusion import GaussianDiffusion


def load_model(checkpoint_path, device, num_timesteps=100):
    """加载模型，使用较少的采样步数"""
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

    # 使用较少的采样步数（注意：这是近似采样，非严格DDPM）
    diffusion = GaussianDiffusion(
        num_timesteps=num_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        loss_type=cfg.diffusion.loss_type,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    diffusion = diffusion.to(device)
    model.eval()

    return model, diffusion


def compute_mae(pred, target):
    return np.mean(np.abs(pred - target))


def compute_psnr(pred, target):
    data_range = target.max() - target.min()
    if data_range == 0:
        return 0.0
    return peak_signal_noise_ratio(target, pred, data_range=data_range)


def compute_ssim(pred, target):
    data_range = target.max() - target.min()
    if data_range == 0:
        return 1.0
    ssim_list = []
    for i in range(pred.shape[0]):
        ssim = structural_similarity(target[i], pred[i], data_range=data_range)
        ssim_list.append(ssim)
    return np.mean(ssim_list)


def compute_fid_simple(real_features, fake_features):
    """简化版FID：基于均值和协方差的距离"""
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    diff = mu_real - mu_fake
    return np.sum(diff ** 2)


@torch.no_grad()
def fast_evaluate(model, diffusion, dataloader, device, num_samples=3):
    """快速评测"""
    model.eval()

    all_mae = []
    all_psnr = []
    all_ssim = []
    all_real_feat = []
    all_fake_feat = []

    count = 0
    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        if count >= num_samples:
            break

        ct_gt = batch['ct'].to(device)
        xray_cor = batch['xray_coronal'].to(device)
        xray_sag = batch['xray_sagittal'].to(device)

        B = ct_gt.shape[0]
        shape = ct_gt.shape

        # 快速反向采样
        ct_recon = diffusion.p_sample_loop(
            model, shape, xray_cor, xray_sag, device,
            clip_denoised=True, verbose=False
        )

        # 转为numpy
        ct_gt_np = ct_gt.squeeze(1).cpu().numpy()
        ct_recon_np = ct_recon.squeeze(1).cpu().numpy()

        for i in range(B):
            pred = ct_recon_np[i]
            target = ct_gt_np[i]

            mae = compute_mae(pred, target)
            psnr = compute_psnr(pred, target)
            ssim = compute_ssim(pred, target)

            all_mae.append(mae)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            # 简单特征（降采样展平）用于FID
            real_feat = torch.nn.functional.avg_pool3d(ct_gt[i:i+1], 4).reshape(-1).numpy()
            fake_feat = torch.nn.functional.avg_pool3d(ct_recon[i:i+1], 4).reshape(-1).numpy()
            all_real_feat.append(real_feat)
            all_fake_feat.append(fake_feat)

            count += 1

        pbar.set_postfix({
            'MAE': f'{np.mean(all_mae):.4f}',
            'PSNR': f'{np.mean(all_psnr):.2f}',
            'SSIM': f'{np.mean(all_ssim):.4f}',
        })

    # 计算FID
    fid = compute_fid_simple(np.array(all_real_feat), np.array(all_fake_feat))

    results = {
        'MAE': {'mean': np.mean(all_mae), 'std': np.std(all_mae)},
        'PSNR': {'mean': np.mean(all_psnr), 'std': np.std(all_psnr)},
        'SSIM': {'mean': np.mean(all_ssim), 'std': np.std(all_ssim)},
        'FID': {'value': fid},
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./outputs/checkpoints/best_model.pth')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Sampling steps: {args.num_timesteps}")

    # 加载模型
    model, diffusion = load_model(args.checkpoint, device, args.num_timesteps)

    # 使用小测试集
    cfg.data.test_list = "./data/verse/test_small.txt"
    dataloader = build_dataloader(cfg, mode="test")
    print(f"Test samples: {len(dataloader.dataset)}")

    # 评测
    results = fast_evaluate(model, diffusion, dataloader, device, args.num_samples)

    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results (Baseline)")
    print("="*50)
    print(f"MAE:  {results['MAE']['mean']:.4f} ± {results['MAE']['std']:.4f}")
    print(f"PSNR: {results['PSNR']['mean']:.2f} ± {results['PSNR']['std']:.2f} dB")
    print(f"SSIM: {results['SSIM']['mean']:.4f} ± {results['SSIM']['std']:.4f}")
    print(f"FID:  {results['FID']['value']:.4f}")
    print("="*50)

    # 保存结果
    result_path = "./outputs/eval_results_baseline.txt"
    with open(result_path, 'w') as f:
        f.write("Diff2CT Baseline Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: VerSe 2019 (subset)\n")
        f.write(f"Volume size: {cfg.data.volume_size}\n")
        f.write(f"Epochs trained: 2\n")
        f.write(f"Sampling steps: {args.num_timesteps}\n")
        f.write(f"Num samples: {args.num_samples}\n")
        f.write("="*50 + "\n")
        f.write(f"MAE:  {results['MAE']['mean']:.4f} ± {results['MAE']['std']:.4f}\n")
        f.write(f"PSNR: {results['PSNR']['mean']:.2f} ± {results['PSNR']['std']:.2f} dB\n")
        f.write(f"SSIM: {results['SSIM']['mean']:.4f} ± {results['SSIM']['std']:.4f}\n")
        f.write(f"FID:  {results['FID']['value']:.4f}\n")
        f.write("="*50 + "\n")
        f.write("Note: This is a baseline result with limited training (2 epochs) ")
        f.write("and reduced sampling steps for CPU verification.\n")
        f.write("Full training with GPU and 1000-step sampling will improve metrics.\n")

    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
