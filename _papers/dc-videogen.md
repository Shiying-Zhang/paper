---
title: "DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder"
authors: "Junyu Chen, Wenkun He, Yuchao Gu, Yuyang Zhao, Jincheng Yu, Junsong Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Muyang Li, Haocheng Xi, Ligeng Zhu, Enze Xie, Song Han, Han Cai"
year: 2025
venue: "arXiv 2025"
tags: [视频生成, Transformer]
rating: 4
paper_url: "https://arxiv.org/abs/2509.25182"
code_url: ""
summary: "后训练加速框架：Deep Compression Video AE (DC-AE-V) 实现 384× 压缩率 + AE-Adapt-V 轻量适配，单卡 H100 生成 720p 视频加速 7.7×"
---

## 核心思想

视频生成模型的推理瓶颈在于：(i) 视频 VAE 压缩率不够高 → latent 空间过大；(ii) 更换压缩率更高的 VAE 需要从头训练 DiT。本文提出 **DC-VideoGen**：一个**后训练加速框架**，包含两个核心组件：

1. **DC-AE-V**：深度压缩视频自编码器，实现 32×/64× 空间 + 4× 时间压缩（总压缩率最高 **384×**）
2. **AE-Adapt-V**：轻量级适配策略，将预训练 DiT 迁移到新 VAE，成本仅为从头训练的 **1/230**

## 方法详解

### 1. 压缩率定义

$$\text{Compression Ratio} = \frac{3 \cdot f^2 \cdot t}{c} \tag{1}$$

其中 $f$ 为空间下采样倍率，$t$ 为时间下采样倍率，$c$ 为 latent 通道数。配置记为 $f^x t^y c^z$。

**典型配置对比**：

| VAE | 配置 | 压缩率 |
|-----|------|--------|
| Wan-2.1 VAE | f8t4c16 | 48 |
| LTX Video | f32t8c128 | 192 |
| **DC-AE-V** | **f64t4c128** | **384** |

压缩率越高 → latent 序列越短 → DiT 推理越快。

### 2. DC-AE-V：时序建模策略

核心问题：如何在视频 AE 中处理时间维度？

#### 2.1 三种时序设计对比

| 设计 | 信息流 | 重建质量 | 泛化性 |
|------|--------|---------|--------|
| **因果 (Causal)** | 单向（前→后） | 差 | 好（长视频泛化） |
| **非因果 (Non-causal)** | 双向 | 好 | 差（长视频退化） |
| **Chunk-Causal（本文）** | 块内双向 + 块间因果 | **好** | **好** |

#### 2.2 Chunk-Causal 设计

将视频切分为固定大小的 chunk，每个 chunk 内部使用双向注意力，chunk 之间使用因果注意力。消融实验表明 chunk size = 40 是重建质量与训练成本的最佳平衡点。

### 3. AE-Adapt-V：两阶段适配

**问题**：如何将预训练 DiT（在旧 VAE 的 latent 空间训练）适配到新 VAE 的 latent 空间，而不从头训练？

#### 3.1 Stage 1：Patch Embedder 对齐

$$\mathcal{L} = \text{MSE}(\mathbf{e}_n, \mathbf{e}_b') \tag{2}$$

其中：
- $\mathbf{e}_b \in \mathbb{R}^{H_b \times W_b \times D}$：基础模型（旧 VAE）的 patch embedding
- $\mathbf{e}_n \in \mathbb{R}^{H_n \times W_n \times D}$：新模型（新 VAE）的 patch embedding
- $\mathbf{e}_b' = \text{AvgPool}(\mathbf{e}_b)$：空间下采样到与新模型匹配的尺寸

**训练过程**：
1. 冻结基础 embedder，训练新 patch embedder 最小化公式 (2)
2. 然后用扩散损失微调输出头（最多 4K 步）

**直觉**：让新 VAE 的 latent 经过 patch embedding 后，与旧 VAE 的 patch embedding 在语义上对齐。

#### 3.2 Stage 2：LoRA 端到端微调

- **配置**：rank=256, alpha=512
- **可训练参数**：350.37M（vs 全微调 1418.90M）
- **关键发现**：LoRA 微调 **优于** 全微调

| 方法 | 可训练参数 | VBench |
|------|-----------|--------|
| 全微调 | 1418.90M | 79.81 |
| **LoRA** | **350.37M** | **84.48** |

**原因**：LoRA 更好地保留了预训练知识，全微调反而导致灾难性遗忘。

### 4. 对齐两阶段的必要性

直接微调（跳过 Stage 1）会导致训练不稳定——20K 步后退化为噪声。AE-Adapt-V 的 patch embedder 对齐提供了稳健的初始化。

## 实验结果

### 视频重建质量

| VAE | 压缩率 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|-----|--------|-------|-------|--------|------|
| Wan-2.1 VAE (f8t4c16) | 48 | — | — | — | — |
| LTX Video (f32t8c128) | 192 | — | — | — | 70.92 |
| **DC-AE-V (f64t4c128)** | **384** | **32.79** | **0.932** | **0.030** | **29.35** |

### 文本到视频生成 (720×1280)

| 模型 | VBench 总分↑ | 质量↑ | 语义↑ | 延迟 | 加速比 |
|------|------------|-------|-------|------|--------|
| Wan-2.1-14B | 83.73 | 85.41 | 76.00 | 27.52 min | 1× |
| Wan-2.1-1.3B | — | — | — | 5.76 min | 1× |
| **DC-VideoGen-14B** | **84.83** | **86.80** | **76.93** | **3.58 min** | **7.7×** |
| **DC-VideoGen-1.3B** | **84.63** | — | — | **0.70 min** | **8.2×** |

### 图像到视频生成 (720×1280)

| 模型 | VBench I2V↑ | 延迟 | 加速比 |
|------|------------|------|--------|
| Wan-2.1-14B | 86.86 | 27.88 min | 1× |
| **DC-VideoGen-14B** | **87.73** | **3.67 min** | **7.6×** |

### 分辨率缩放特性

| 分辨率 | 延迟 (DC-VideoGen-1.3B) | 加速比 |
|--------|----------------------|--------|
| 720p, 80 帧 | 0.70 min | 8.2× |
| 720p, 160 帧 | 1.99 min | 10.1× |
| 720p, 320 帧 | 6.03 min | 12.6× |
| 4K (2160×3840) | — | **14.8×** |

**关键观察**：分辨率/帧数越高，加速比越大——因为压缩率优势在更大 latent 上更显著。

### 训练效率

| 指标 | DC-VideoGen | 从头训练 |
|------|------------|---------|
| 适配成本 | **10 H100 GPU days** | 2300 H100 GPU days |
| 节省 | **230×** | — |

## 个人思考

1. **后训练加速**的思路非常实用：不需要从头训练巨大的 DiT，只需适配 VAE + LoRA 微调。
2. **Chunk-Causal**是优雅的折中：结合了因果（泛化性）和非因果（重建质量）的优点。
3. **LoRA > 全微调**的发现反直觉但可理解——预训练 DiT 的知识是宝贵的，全微调容易破坏。
4. **384× 压缩率**非常激进，但重建质量仍然很好（FVD 29.35），说明视频信号的冗余度极高。
5. **分辨率越高加速越大**的特性使得该方法在高分辨率/长视频场景下特别有价值——恰好是当前最需要加速的场景。
