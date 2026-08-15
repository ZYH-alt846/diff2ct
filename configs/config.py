import os
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DataConfig:
    """数据配置"""
    # 数据集路径
    data_root: str = "./data/verse"
    train_list: str = "./data/verse/train.txt"
    val_list: str = "./data/verse/val.txt"
    test_list: str = "./data/verse/test.txt"

    # 图像参数
    volume_size: Tuple[int, int, int] = (32, 32, 32)  # (D, H, W) - CPU训练用小尺寸
    voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # mm

    # CT值范围 (HU)
    ct_min: int = -1000
    ct_max: int = 4096

    # 归一化范围
    norm_min: float = -1.0
    norm_max: float = 1.0

    # 数据增强
    use_augmentation: bool = True
    flip_prob: float = 0.5
    rotate_range: float = 10.0  # 度
    scale_range: float = 0.1


@dataclass
class ModelConfig:
    """模型配置"""
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    time_emb_dim: int = 128
    use_attention: bool = True
    attention_levels: Tuple[int, ...] = (2, 3)
    dropout: float = 0.1
    norm_type: str = "group"  # group / batch
    num_groups: int = 8


@dataclass
class DiffusionConfig:
    """扩散过程配置"""
    num_timesteps: int = 1000
    beta_schedule: str = "linear"  # linear / cosine
    beta_start: float = 1e-6
    beta_end: float = 1e-2

    # 采样配置
    sample_timesteps: int = 1000  # 推理时的采样步数
    sample_strategy: str = "ddpm"  # ddpm / ddim

    # 损失权重
    loss_type: str = "mse"  # mse / l1
    projection_loss_weight: float = 0.3


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 1
    num_epochs: int = 2  # CPU快速跑通基线
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"  # cosine / step / constant
    warmup_epochs: int = 10
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    optimizer: str = "adam"  # adam / adamw
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # 保存与日志
    save_interval: int = 50
    val_interval: int = 1
    log_interval: int = 5
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    log_dir: str = "./outputs/logs"

    # 恢复训练
    resume: bool = False
    resume_checkpoint: str = ""

    # 混合精度训练
    use_amp: bool = False  # CPU不支持混合精度

    # 多GPU
    use_ddp: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [0])


@dataclass
class EvalConfig:
    """评测配置"""
    metrics: Tuple[str, ...] = ("mae", "psnr", "ssim", "fid")
    save_samples: bool = True
    sample_dir: str = "./outputs/samples"
    fid_feat_dim: int = 2048
    num_workers: int = 0  # Windows上设为0避免多进程问题


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    device: str = "cpu"


# 全局配置实例
cfg = Config()
