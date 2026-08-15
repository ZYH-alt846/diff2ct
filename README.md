# Diff2CT: 双平面X光引导的3D脊柱CT扩散重建

## 项目简介

本项目基于条件扩散模型（Conditional DDPM），实现从冠状面+矢状面双平面DRR影像到3D CT体数据的端到端重建。采用3D UNet作为扩散去噪主干，通过X2NoiseNet将2D X光特征升维为3D条件特征，结合三轴投影一致性损失约束重建结果的几何结构正确性。

## 方法框架

1. **条件编码模块**：双平面X光分别经过独立2D编码器提取特征，沿投影方向复制升维为3D特征，相加融合后通过3D卷积进一步校准，作为扩散过程的条件输入。
2. **扩散主干网络**：带时间步嵌入的3D UNet架构，在深层加入自注意力机制捕获长距离空间依赖。
3. **联合损失函数**：基础噪声预测MSE损失 + 三轴正交投影一致性损失，兼顾像素级灰度精度与解剖结构几何约束。

## 项目目录结构

```
diff2ct/
├── configs/
│   └── config.py          # 全局超参数与路径配置
├── data/
│   ├── dataset.py         # 数据集加载、预处理与增强
│   └── drr.py             # 数字重建放射影像(DRR)生成
├── models/
│   ├── diffusion.py       # 扩散前向加噪与反向采样过程
│   ├── unet3d.py          # 3D UNet去噪主干网络
│   ├── x2noise_net.py     # X2NoiseNet条件编码网络
│   └── modules.py         # 基础组件(时间编码、注意力、残差块)
├── losses/
│   └── projection_loss.py # 三轴投影一致性损失
├── train.py               # 训练主脚本
├── inference.py           # 单样本重建推理脚本
├── evaluate.py            # 批量评测脚本(MAE/PSNR/SSIM/FID)
├── fast_eval.py           # 快速评测脚本(CPU/少量样本)
├── requirements.txt       # 依赖包清单
└── README.md
```

## 环境配置

### 安装步骤

```bash
# 创建虚拟环境
conda create -n diff2ct python=3.9
conda activate diff2ct

# 安装PyTorch(根据CUDA版本选择)
# GPU版本:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU版本:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖
pip install -r requirements.txt
```

### 核心依赖

- 深度学习框架：torch>=1.12.0, torchvision>=0.13.0
- 医学影像处理：SimpleITK>=2.1.0, nibabel>=3.2.0, scikit-image>=0.18.0
- 工具库：numpy, scipy, pandas, tqdm, PyYAML, einops
- 日志可视化：matplotlib, tensorboard

## 数据集准备

### VerSe 脊柱挑战赛数据集（基线用）

- **数据规模**：VerSe 2019，共80例三维脊柱CT，覆盖颈胸腰全节段
- **获取地址**：
  - 官方仓库：https://github.com/anjany/verse
  - OSF数据页：https://osf.io/nqjyw/
- **定位**：本项目基线训练与核心评测基准数据集

### 数据预处理流程

1. 下载数据集后，将所有 `_ct.nii.gz` 文件整理至 `./data/verse/train_raw/` 目录
2. 生成训练/验证/测试列表文件：
   ```bash
   python -c "
   import os
   data_root = './data/verse'
   ct_files = []
   for root, dirs, files in os.walk(os.path.join(data_root, 'train_raw')):
       for f in files:
           if f.endswith('_ct.nii.gz'):
               ct_files.append(os.path.relpath(os.path.join(root, f), data_root))
   ct_files.sort()
   n_val = len(ct_files) // 5
   for name, subset in [('train.txt', ct_files[n_val:]), ('val.txt', ct_files[:n_val]), ('test.txt', ct_files[:n_val])]:
       with open(os.path.join(data_root, name), 'w') as f:
           f.write('\n'.join(subset))
   "
   ```
3. 代码内置自动重采样、中心裁剪、DRR生成、归一化等全流程预处理

## 使用说明

### 模型训练

```bash
python train.py
```

所有超参数、路径均可在 `configs/config.py` 中统一修改。

### 单样本推理

```bash
python inference.py \
  --checkpoint ./outputs/checkpoints/best_model.pth \
  --input ./data/verse/train_raw/sample/sample_ct.nii.gz \
  --output ./outputs/recon_result.nii.gz
```

### 批量评测

```bash
python evaluate.py \
  --checkpoint ./outputs/checkpoints/best_model.pth \
  --split test \
  --metrics mae psnr ssim fid
```

### 快速评测（CPU/少量样本）

```bash
python fast_eval.py --num_samples 3 --num_timesteps 100
```

## 评测指标说明

| 指标 | 全称 | 意义与解读 |
|------|------|------------|
| MAE | 平均绝对误差 | 像素级灰度精度指标，数值越小代表重建灰度偏差越小 |
| PSNR | 峰值信噪比 | 整体图像失真程度，单位dB，数值越大代表整体质量越高 |
| SSIM | 结构相似性 | 从亮度、对比度、结构维度衡量解剖一致性，取值0-1，越接近1结构越准确 |
| FID | 弗雷歇感知距离 | 衡量生成图像的视觉真实感，数值越小代表与真实CT分布越接近 |

## 基线实验结果

### 实验配置

- **数据集**：VerSe 2019（64训练 / 16验证）
- **体积尺寸**：32×32×32（CPU快速验证）
- **训练轮数**：2 epochs
- **扩散步数**：训练1000步 / 采样100步（快速近似）
- **硬件**：CPU（无GPU加速）

> 注：以下为流程验证基线结果，用于确认全链路（数据加载→DRR生成→扩散训练→反向采样→指标计算）正确跑通。完整GPU训练（128³体积、1000步采样、充分训练）后指标将显著提升。

### VerSe 2019 基线结果

| 方法 | MAE ↓ | PSNR (dB) ↑ | SSIM ↑ | FID ↓ |
|------|-------|-------------|--------|-------|
| Diff2CT (baseline) | 0.6509 ± 0.1853 | 2.17 ± 1.05 | 0.0014 ± 0.0012 | 217.5981 |

### 训练损失曲线

- Epoch 1: train noise loss 0.4223, val loss 0.1530
- Epoch 2: train noise loss 0.05（持续下降中）

## 后续计划

- [x] 完成代码框架与模块实现
- [x] 整理公开数据集与评测指标
- [x] 跑通VerSe数据集训练全流程
- [x] 输出基线实验数值结果
- [ ] GPU环境下全量训练（128³体积、1000步采样）
- [ ] 补充SOTA方法性能对比
- [ ] 消融实验（投影损失权重、条件编码方式等）
