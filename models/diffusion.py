import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GaussianDiffusion:
    """
    高斯扩散过程

    实现DDPM的前向加噪和反向采样
    """

    def __init__(self,
                 num_timesteps: int = 1000,
                 beta_schedule: str = "linear",
                 beta_start: float = 1e-6,
                 beta_end: float = 1e-2,
                 loss_type: str = "mse"):
        """
        Args:
            num_timesteps: 扩散步数
            beta_schedule: beta调度方式 (linear / cosine)
            beta_start: beta起始值
            beta_end: beta结束值
            loss_type: 损失类型 (mse / l1)
        """
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type

        # 计算beta调度
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # 预计算各种系数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 注册为buffer（随模型移动到device）
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # 前向过程用
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()

        # 反向过程用
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).float()
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        ).float()
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).float()

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦beta调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def to(self, device):
        """移动到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """从系数数组中提取指定时间步的值，并reshape到匹配x的形状"""
        B = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(B, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散：x0 -> xt

        Args:
            x_start: 干净图像 x0 [B, C, D, H, W]
            t: 时间步 [B]
            noise: 可选的噪声

        Returns:
            x_t: 带噪图像
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor,
                                  t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                                 noise: torch.Tensor) -> torch.Tensor:
        """
        从xt和预测噪声反推x0
        """
        return (
            self._extract(self.sqrt_recip_alphas, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recip_alphas * self.sqrt_one_minus_alphas_cumprod,
                          t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t: torch.Tensor, t: torch.Tensor,
                        xray_coronal: torch.Tensor, xray_sagittal: torch.Tensor,
                        clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 p(x_{t-1} | x_t) 的均值和方差

        Args:
            model: 噪声预测网络
            x_t: 当前带噪图像
            t: 当前时间步
            xray_coronal: 冠状面X光条件
            xray_sagittal: 矢状面X光条件
            clip_denoised: 是否裁剪x0

        Returns:
            model_mean: 模型预测的均值
            posterior_variance: 后验方差
            model_log_variance: 后验对数方差
        """
        # 预测噪声
        noise_pred = model(x_t, t, xray_coronal, xray_sagittal)

        # 反推x0
        x_recon = self.predict_start_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)

        # 计算后验均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start=x_recon, x_t=x_t, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor,
                 xray_coronal: torch.Tensor, xray_sagittal: torch.Tensor,
                 clip_denoised: bool = True) -> torch.Tensor:
        """
        单步反向采样：x_t -> x_{t-1}
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, xray_coronal, xray_sagittal, clip_denoised
        )

        noise = torch.randn_like(x_t)
        # t=0时不加噪声
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))

        x_prev = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, model, shape: Tuple[int, ...],
                      xray_coronal: torch.Tensor, xray_sagittal: torch.Tensor,
                      device: torch.device,
                      clip_denoised: bool = True,
                      verbose: bool = False) -> torch.Tensor:
        """
        完整反向采样循环：从纯噪声生成图像

        Args:
            model: 噪声预测网络
            shape: 生成图像形状 [B, C, D, H, W]
            xray_coronal: 冠状面X光条件
            xray_sagittal: 矢状面X光条件
            device: 设备
            clip_denoised: 是否裁剪
            verbose: 是否显示进度

        Returns:
            x_0: 生成的清晰图像
        """
        B = shape[0]

        # 从纯噪声开始
        img = torch.randn(shape, device=device)

        # 逐步去噪
        from tqdm import tqdm
        timesteps = tqdm(range(self.num_timesteps - 1, -1, -1), desc="Sampling") if verbose \
                    else range(self.num_timesteps - 1, -1, -1)

        for i in timesteps:
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, xray_coronal, xray_sagittal, clip_denoised)

        return img

    def training_loss(self, model, x_start: torch.Tensor,
                      xray_coronal: torch.Tensor, xray_sagittal: torch.Tensor,
                      t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算训练损失

        Args:
            model: 噪声预测网络
            x_start: 干净图像 x0
            xray_coronal: 冠状面X光条件
            xray_sagittal: 矢状面X光条件
            t: 可选的时间步，若为None则随机采样

        Returns:
            loss: 损失值
        """
        B = x_start.shape[0]
        device = x_start.device

        # 随机采样时间步
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        # 前向加噪
        x_t, noise = self.q_sample(x_start, t)

        # 预测噪声
        noise_pred = model(x_t, t, xray_coronal, xray_sagittal)

        # 计算损失
        if self.loss_type == "mse":
            loss = F.mse_loss(noise_pred, noise)
        elif self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss
