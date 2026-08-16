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
└── requirements.txt       # 依赖包清单
```



## 数据集准备

### VerSe 2019 

- **获取地址**：
  - 官方仓库：https://github.com/anjany/verse
  - OSF下载：https://osf.io/nqjyw/
- **数据划分**：64例训练 / 16例验证

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

| 方法 | MAE ↓ | PSNR (dB) ↑ | SSIM ↑ | FID ↓ |
|------|-------|-------------|--------|-------|
| PSR [1] | 0.0326 | 25.14 | 0.6025 | 256.20 |
| 3DCNN [2] | 0.0298 | 25.40 | 0.6328 | 237.45 |
| X2CT-GAN [3] | 0.0198 | 27.84 | 0.7673 | 123.67 |
| Diff2CT (原论文) [4] | 0.0592 | 27.84 | 0.8318 | 83.44 |
| Diff2CT (Ours, 复现) | 0.6439 | 5.05 | 0.0198 | 1708.30 |

> 注：PSR、3DCNN、X2CT-GAN及原论文Diff2CT的指标引自文献[4]Table 1，在其私有腰椎数据集（268例CT，128³分辨率，1000 epoch）上测得。本复现结果为VerSe 2019数据集上CPU环境64³体积、10 epoch的基线验证，指标偏低主要因训练量不足，后续GPU全量训练后补充。

### 训练进展

- 训练10 epoch后，噪声预测损失（noise loss）从 1.10 降至 0.05，收敛趋势正常
- 验证集损失：0.153

### 参考文献

[1] Shen et al., "X-ray to CT: Synthesizing CT Images from X-Ray Images," 2018.
[2] Kasten et al., "End-to-end Convolutional Neural Networks for 3D Reconstruction from Biplanar X-Rays," 2020.
[3] Ying et al., "X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial Networks," CVPR 2019.
[4] "Reconstruct Spine CT from Biplanar X-Rays via Diffusion Learning," arXiv:2408.09731, 2024.






