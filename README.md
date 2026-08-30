# Diff2CT: 双平面X光引导的3D脊柱CT扩散重建

## 项目简介

本项目基于条件扩散模型，实现从冠状面+矢状面双平面DRR影像到3D CT体数据的端到端重建。采用3D UNet作为扩散去噪主干，通过X2NoiseNet将2D X光特征升维为3D条件特征，结合三轴投影一致性损失约束重建结果的几何结构正确性。

## 方法框架

1. **条件编码模块（X2NoiseNet）**：双平面X光分别经过独立2D编码器提取特征，沿投影方向复制升维为3D特征，相加融合后通过3D卷积校准，作为扩散过程的条件输入。
2. **扩散主干网络（3D UNet）**：带时间步嵌入的3D UNet架构，在深层加入自注意力机制捕获长距离空间依赖。
3. **联合损失函数**：基础噪声预测MSE损失+三轴正交投影一致性损失，兼顾像素级灰度精度与解剖结构几何约束。

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
├── fast_eval.py           # 快速评测脚本
├── measurement_consistency.py # 测量一致性约束推理（创新点）
└── requirements.txt       # 依赖包清单
```



## 数据集准备

### VerSe 2019 

- **获取地址**：
  - 官方仓库：https://github.com/anjany/verse
  - OSF下载：https://osf.io/nqjyw/


### 数据预处理

1. 下载数据集后，将CT文件整理至 `./data/verse/train_raw/` 目录
2. 生成 `train.txt` / `val.txt` / `test.txt` 列表文件，每行对应一个CT文件的相对路径
3. 代码内置自动重采样、中心裁剪/填充、DRR生成、HU值归一化等全流程预处理


## 评测指标说明

| 指标 | 全称 | 意义 |
|------|------|------|
| MAE | 平均绝对误差 | 像素级灰度精度，越小越好 |
| PSNR | 峰值信噪比(dB) | 整体图像质量，越大越好 |
| SSIM | 结构相似性 | 解剖结构一致性，越接近1越好 |
| FID | 弗雷歇感知距离 | 生成分布真实感，越小越好 |

## 实验结果

### 定量对比

| 方法 | 数据集 | MAE ↓ | PSNR (dB) ↑ | SSIM ↑ | FID ↓ |
|------|--------|-------|-------------|--------|-------|
| PSR [1] | LumbarV | 0.0326 | 25.14 | 0.6025 | 256.20 |
| 3DCNN [2] | LumbarV | 0.0298 | 25.40 | 0.6328 | 237.45 |
| X2CT-GAN [3] | LumbarV | 0.0198 | 27.84 | 0.7673 | 123.67 |
| Diff2CT[4] | LumbarV | 0.0592 | 27.84 | 0.8318 | 83.44 |
| X-CTRSNet [5] | CTSpine1K | - | 21.34 | 0.5355 | 255.98 |
| X2CT-GAN [3] | CTSpine1K | - | 20.60 | 0.4841 | 50.63 |
| DiffuX2CT [6] | CTSpine1K | - | 21.53 | 0.5924 | 8.90 |
| X2CT-CNN [3] | CTSpine1K | - | 20.33 | 0.5397 | - |
| LDM [7] | CTSpine1K | - | 24.79 | 0.5992 | - |
| CLS-DM [8] | CTSpine1K | - | 26.37 | 0.6186 | - |
| **Diff2CT (Ours, v2 GPU)** | **VerSe 2019** | **0.5677** | **7.86** | **0.1428** | **170.41** |
| Diff2CT (Ours, v1 GPU) | VerSe 2019 | 0.5919 | 5.36 | 0.0533 | 76.99 |
| Diff2CT (Ours, CPU baseline) | VerSe 2019 | 0.6439 | 5.05 | 0.0198 | 1708.30 |

> 注：PSR、3DCNN、X2CT-GAN及原论文Diff2CT的指标引自文献[4] Table 1（私有腰椎数据集LumbarV，268例CT，128³分辨率，1000 epoch）；X-CTRSNet、X2CT-GAN、DiffuX2CT的指标引自文献[6] Table 1（公开脊柱数据集CTSpine1K，128³分辨率）；X2CT-CNN、LDM、CLS-DM的指标引自文献[8] Table 1（CTSpine1K，128³分辨率，双视角）。本复现结果为VerSe 2019数据集。v2版本为RTX 4080 SUPER上128³体积、80例数据、300 epoch训练，修复了投影损失（mean投影+参与梯度回传）并加入梯度检查点；v1版本为96³体积、30例数据、200 epoch；CPU基线为64³体积、10 epoch。指标与原论文差距主要因训练数据量有限（80例 vs 268例）及训练轮次差异。

### 训练进展

**v2 GPU训练（RTX 4080 SUPER，128³，80例，300 epoch）：**
- 修复投影损失：将sum投影改为mean投影，并去掉no_grad使其真正参与梯度回传
- 加入梯度检查点，使128³可在32GB显存上训练
- 训练300 epoch后，噪声预测损失（noise loss）从 1.10 降至 0.005，最佳验证集损失：0.0024
- 评测结果（8个测试样本）：MAE 0.5677±0.0404，PSNR 7.86±1.48，SSIM 0.1428±0.0300，FID 170.41
- 相比v1版本：PSNR提升46%（5.36→7.86），SSIM提升168%（0.053→0.143）

**v1 GPU训练（RTX 4080 SUPER，96³，30例，200 epoch）：**
- 训练200 epoch后，噪声预测损失（noise loss）从 1.10 降至 0.005
- 最佳验证集损失：0.0011
- 评测结果（5个测试样本）：MAE 0.5919±0.0156，PSNR 5.36±1.07，SSIM 0.0533±0.0109，FID 76.99

**CPU基线（64³，10 epoch）：**
- 训练10 epoch后，噪声预测损失（noise loss）从 1.10 降至 0.05
- 验证集损失：0.153

### 参考文献

[1] Henzler et al., "Single-Image Tomography: 3D Volumes from 2D Cranial X-Rays," Computer Graphics Forum (EuroVis 2018). https://doi.org/10.1111/cgf.13369

[2] Kasten et al., "End-to-End Convolutional Neural Network for 3D Reconstruction of Knee Bones from Bi-Planar X-Ray Images," MLMIR 2020 (MICCAI Workshop). https://link.springer.com/chapter/10.1007/978-3-030-61598-7_12

[3] Ying et al., "X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial Networks," CVPR 2019. https://arxiv.org/abs/1905.06902

[4] "Reconstruct Spine CT from Biplanar X-Rays via Diffusion Learning," arXiv:2408.09731, 2024. https://arxiv.org/abs/2408.09731

[5] Ge et al., "X-CTRSNet: 3D Cervical Vertebra CT Reconstruction and Segmentation Directly from 2D X-Ray Images," Knowledge-Based Systems, 2022. https://doi.org/10.1016/j.knosys.2021.107680

[6] Liu et al., "DiffuX2CT: Diffusion Learning to Reconstruct CT Images from Biplanar X-Rays," ECCV 2024. https://arxiv.org/abs/2407.13545

[7] Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022. https://arxiv.org/abs/2112.10752

[8] Chen et al., "Latent Space Consistency for Sparse-View CT Reconstruction," arXiv:2507.11152, 2025. https://arxiv.org/abs/2507.11152






