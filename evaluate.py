import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import cfg
from data.dataset import build_dataloader
from models.x2noise_net import X2NoiseNet
from models.diffusion import GaussianDiffusion


def load_model(checkpoint_path, device):
    """加载模型"""
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

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    diffusion = diffusion.to(device)
    model.eval()

    return model, diffusion


def compute_mae(pred, target):
    """计算平均绝对误差"""
    return np.mean(np.abs(pred - target))


def compute_psnr(pred, target, data_range=None):
    """计算峰值信噪比"""
    if data_range is None:
        data_range = target.max() - target.min()
    return peak_signal_noise_ratio(target, pred, data_range=data_range)


def compute_ssim(pred, target, data_range=None):
    """计算结构相似性（逐切片计算后取平均）"""
    if data_range is None:
        data_range = target.max() - target.min()

    # 对每个轴向切片计算SSIM，然后取平均
    ssim_list = []
    for i in range(pred.shape[0]):
        ssim = structural_similarity(
            target[i], pred[i], data_range=data_range
        )
        ssim_list.append(ssim)

    return np.mean(ssim_list)


def compute_fid(real_features, fake_features):
    """
    计算FID（弗雷歇 inception 距离）

    Args:
        real_features: 真实样本特征 [N, D]
        fake_features: 生成样本特征 [N, D]

    Returns:
        fid: FID值
    """
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # 计算均值差
    diff = mu_real - mu_fake

    # 计算协方差矩阵的平方根
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma_real @ sigma_fake)

    # 处理数值误差
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return fid


class SimpleFeatureExtractor:
    """
    简单特征提取器（用于FID计算）
    使用预训练的3D CNN特征，或退化为统计特征
    这里用降采样+展平作为简化版特征
    """

    def __init__(self, device):
        self.device = device

    def extract_features(self, volume_batch):
        """
        从3D体数据提取特征

        Args:
            volume_batch: [B, 1, D, H, W]

        Returns:
            features: [B, feat_dim]
        """
        B = volume_batch.shape[0]

        # 简化版：多层平均池化后展平
        x = volume_batch

        # 逐层池化降维
        for _ in range(4):
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)

        # 展平
        features = x.reshape(B, -1)

        return features.cpu().numpy()


@torch.no_grad()
def evaluate(model, diffusion, dataloader, device, metrics=("mae", "psnr", "ssim", "fid")):
    """
    在测试集上评测

    Args:
        model: 模型
        diffusion: 扩散过程
        dataloader: 测试数据加载器
        device: 设备
        metrics: 要计算的指标列表

    Returns:
        results: 指标结果字典
    """
    model.eval()

    all_mae = []
    all_psnr = []
    all_ssim = []
    all_real_features = []
    all_fake_features = []

    feature_extractor = SimpleFeatureExtractor(device) if "fid" in metrics else None

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch in pbar:
        ct_gt = batch['ct'].to(device)
        xray_cor = batch['xray_coronal'].to(device)
        xray_sag = batch['xray_sagittal'].to(device)

        B = ct_gt.shape[0]
        shape = ct_gt.shape

        # 重建（完整采样）
        ct_recon = diffusion.p_sample_loop(
            model, shape, xray_cor, xray_sag, device,
            clip_denoised=True, verbose=False
        )

        # 转为numpy
        ct_gt_np = ct_gt.squeeze(1).cpu().numpy()  # [B, D, H, W]
        ct_recon_np = ct_recon.squeeze(1).cpu().numpy()

        # 计算各指标
        for i in range(B):
            pred = ct_recon_np[i]
            target = ct_gt_np[i]

            if "mae" in metrics:
                mae = compute_mae(pred, target)
                all_mae.append(mae)

            if "psnr" in metrics:
                psnr = compute_psnr(pred, target)
                all_psnr.append(psnr)

            if "ssim" in metrics:
                ssim = compute_ssim(pred, target)
                all_ssim.append(ssim)

        # FID特征
        if "fid" in metrics and feature_extractor is not None:
            real_feat = feature_extractor.extract_features(ct_gt)
            fake_feat = feature_extractor.extract_features(ct_recon)
            all_real_features.append(real_feat)
            all_fake_features.append(fake_feat)

        # 更新进度
        current_metrics = {}
        if all_mae:
            current_metrics['MAE'] = f'{np.mean(all_mae):.4f}'
        if all_psnr:
            current_metrics['PSNR'] = f'{np.mean(all_psnr):.2f}'
        if all_ssim:
            current_metrics['SSIM'] = f'{np.mean(all_ssim):.4f}'
        pbar.set_postfix(current_metrics)

    # 汇总结果
    results = {}

    if "mae" in metrics and all_mae:
        results['MAE'] = {
            'mean': np.mean(all_mae),
            'std': np.std(all_mae),
        }

    if "psnr" in metrics and all_psnr:
        results['PSNR'] = {
            'mean': np.mean(all_psnr),
            'std': np.std(all_psnr),
        }

    if "ssim" in metrics and all_ssim:
        results['SSIM'] = {
            'mean': np.mean(all_ssim),
            'std': np.std(all_ssim),
        }

    if "fid" in metrics and all_real_features:
        real_features = np.concatenate(all_real_features, axis=0)
        fake_features = np.concatenate(all_fake_features, axis=0)
        fid = compute_fid(real_features, fake_features)
        results['FID'] = {'value': fid}

    return results


def print_results(results):
    """打印评测结果"""
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    for metric_name, metric_data in results.items():
        if 'mean' in metric_data:
            print(f"{metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")
        elif 'value' in metric_data:
            print(f"{metric_name}: {metric_data['value']:.4f}")

    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Diff2CT Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help='评测数据集划分')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['mae', 'psnr', 'ssim', 'fid'],
                        help='评测指标')
    args = parser.parse_args()

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model, diffusion = load_model(args.checkpoint, device)

    # 数据加载器
    dataloader = build_dataloader(cfg, mode=args.split)
    print(f"Evaluating on {args.split} set: {len(dataloader.dataset)} samples")

    # 评测
    results = evaluate(model, diffusion, dataloader, device, tuple(args.metrics))

    # 打印结果
    print_results(results)

    # 保存结果
    output_dir = os.path.dirname(args.checkpoint)
    result_path = os.path.join(output_dir, f"eval_results_{args.split}.txt")
    with open(result_path, 'w') as f:
        for metric_name, metric_data in results.items():
            if 'mean' in metric_data:
                f.write(f"{metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}\n")
            elif 'value' in metric_data:
                f.write(f"{metric_name}: {metric_data['value']:.4f}\n")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
