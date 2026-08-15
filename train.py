import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import cfg
from data.dataset import build_dataloader
from models.x2noise_net import X2NoiseNet
from models.diffusion import GaussianDiffusion
from losses.projection_loss import ProjectionLoss


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg, device):
    """构建模型和扩散过程"""
    model = X2NoiseNet(
        in_channels=cfg.model.in_channels,
        cond_channels=cfg.model.base_channels,  # 条件通道数
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

    model = model.to(device)
    diffusion = diffusion.to(device)

    return model, diffusion


def build_optimizer(cfg, model):
    """构建优化器和学习率调度器"""
    if cfg.train.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")

    # 学习率调度器
    if cfg.train.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.num_epochs, eta_min=1e-6
        )
    elif cfg.train.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.train.num_epochs // 3, gamma=0.5
        )
    elif cfg.train.lr_scheduler == "constant":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.train.lr_scheduler}")

    return optimizer, scheduler


def train_one_epoch(model, diffusion, projection_loss, dataloader, optimizer,
                    device, epoch, scaler=None, writer=None, log_interval=10):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_noise_loss = 0.0
    total_proj_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        ct_gt = batch['ct'].to(device)
        xray_cor = batch['xray_coronal'].to(device)
        xray_sag = batch['xray_sagittal'].to(device)

        B = ct_gt.shape[0]

        # 随机采样时间步
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()

        # 混合精度训练
        if scaler is not None and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                # 前向加噪
                x_t, noise_true = diffusion.q_sample(ct_gt, t)

                # 预测噪声
                noise_pred = model(x_t, t, xray_cor, xray_sag)

                # 基础噪声损失
                noise_loss = torch.nn.functional.mse_loss(noise_pred, noise_true)

                # 投影损失（反推x0后计算）
                with torch.no_grad():
                    x0_pred = diffusion.predict_start_from_noise(x_t, t, noise_pred)

                proj_loss, proj_dict = projection_loss(x0_pred, ct_gt)

                # 总损失
                loss = noise_loss + cfg.diffusion.projection_loss_weight * proj_loss

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 梯度裁剪
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向加噪
            x_t, noise_true = diffusion.q_sample(ct_gt, t)

            # 预测噪声
            noise_pred = model(x_t, t, xray_cor, xray_sag)

            # 基础噪声损失
            noise_loss = torch.nn.functional.mse_loss(noise_pred, noise_true)

            # 投影损失
            with torch.no_grad():
                x0_pred = diffusion.predict_start_from_noise(x_t, t, noise_pred)

            proj_loss, proj_dict = projection_loss(x0_pred, ct_gt)

            # 总损失
            loss = noise_loss + cfg.diffusion.projection_loss_weight * proj_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

            optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_noise_loss += noise_loss.item()
        total_proj_loss += proj_loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'noise': f'{noise_loss.item():.4f}',
            'proj': f'{proj_loss.item():.4f}',
        })

        # TensorBoard日志
        if writer is not None and batch_idx % log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/total_loss', loss.item(), global_step)
            writer.add_scalar('Train/noise_loss', noise_loss.item(), global_step)
            writer.add_scalar('Train/proj_loss', proj_loss.item(), global_step)
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step)

    avg_loss = total_loss / num_batches
    avg_noise_loss = total_noise_loss / num_batches
    avg_proj_loss = total_proj_loss / num_batches

    return avg_loss, avg_noise_loss, avg_proj_loss


@torch.no_grad()
def validate(model, diffusion, dataloader, device, epoch, writer=None):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    for batch in pbar:
        ct_gt = batch['ct'].to(device)
        xray_cor = batch['xray_coronal'].to(device)
        xray_sag = batch['xray_sagittal'].to(device)

        B = ct_gt.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()

        # 前向加噪
        x_t, noise_true = diffusion.q_sample(ct_gt, t)

        # 预测噪声
        noise_pred = model(x_t, t, xray_cor, xray_sag)

        # 计算损失
        loss = torch.nn.functional.mse_loss(noise_pred, noise_true)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches

    if writer is not None:
        writer.add_scalar('Val/loss', avg_loss, epoch)

    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename="checkpoint.pth"):
    """保存模型检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    print(f"Checkpoint loaded: {checkpoint_path}, epoch {epoch}, loss {loss:.4f}")
    return epoch, loss


def main():
    parser = argparse.ArgumentParser(description="Train Diff2CT")
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(cfg.seed)

    # 设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.train.log_dir, exist_ok=True)

    # 数据加载器
    print("Building dataloaders...")
    train_loader = build_dataloader(cfg, mode="train")
    val_loader = build_dataloader(cfg, mode="val")

    # 构建模型
    print("Building model...")
    model, diffusion = build_model(cfg, device)

    # 投影损失
    projection_loss = ProjectionLoss(loss_type=cfg.diffusion.loss_type).to(device)

    # 优化器
    optimizer, scheduler = build_optimizer(cfg, model)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if (cfg.train.use_amp and device.type == 'cuda') else None

    # TensorBoard
    writer = SummaryWriter(cfg.train.log_dir)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume or cfg.train.resume:
        resume_path = args.resume or cfg.train.resume_checkpoint
        if os.path.exists(resume_path):
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, resume_path)
            start_epoch += 1

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # 训练循环
    print("Start training...")
    print(f"Total epochs: {cfg.train.num_epochs}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{cfg.train.num_epochs}")
        print(f"{'='*50}")

        # 训练
        train_loss, train_noise_loss, train_proj_loss = train_one_epoch(
            model, diffusion, projection_loss, train_loader, optimizer,
            device, epoch, scaler, writer, cfg.train.log_interval
        )

        # 学习率更新
        if scheduler is not None:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f} (noise: {train_noise_loss:.4f}, proj: {train_proj_loss:.4f})")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证
        if (epoch + 1) % cfg.train.val_interval == 0:
            val_loss = validate(model, diffusion, val_loader, device, epoch, writer)
            print(f"Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss,
                               cfg.train.checkpoint_dir, "best_model.pth")
                print(f"New best val loss: {best_val_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % cfg.train.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                          cfg.train.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")

    # 保存最终模型
    save_checkpoint(model, optimizer, cfg.train.num_epochs - 1, train_loss,
                  cfg.train.checkpoint_dir, "final_model.pth")

    writer.close()
    print("\nTraining completed!")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
