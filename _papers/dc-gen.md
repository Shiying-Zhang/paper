---
title: "DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space"
authors: "Wenkun He, Yuchao Gu, Junyu Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Haocheng Xi, Muyang Li, Ligeng Zhu, Jincheng Yu, Junsong Chen, Enze Xie, Song Han, Han Cai"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, Transformer]
rating: 4
paper_url: "https://arxiv.org/abs/2509.25180"
code_url: "https://github.com/dc-ai-projects/DC-Gen"
summary: "后训练扩散加速：Embedding 对齐 + LoRA 微调将预训练 DiT 迁移到高压缩 AE，4K 生成加速 53×，训练成本降 520×"
---

## 核心思想

预训练扩散模型使用低压缩率自编码器（如 FLUX 用 8×），如果换成高压缩率 AE（32×/64×），latent 空间序列大幅缩短 → 推理大幅加速。但直接微调会导致灾难性遗忘。

**DC-Gen** 提出两阶段后训练方案：
1. **Embedding 对齐**：让新 AE 的 patch embedding 与旧 AE 对齐
2. **LoRA 端到端微调**：保留预训练知识的轻量微调

## 方法详解

### 1. Token 数公式

$$\text{Token Count} = \frac{H}{f \cdot p} \times \frac{W}{f \cdot p}$$

$f$ 为 AE 压缩率，$p$ 为 patch size。例如 4K 图像 (3840×2160)：
- FLUX-VAE (f=8, p=2)：$240 \times 135 = 32,400$ tokens
- DC-AE (f=32, p=2)：$60 \times 34 \approx 2,040$ tokens（**16× 减少**）

### 2. Stage 1：Embedding 对齐

$$\mathcal{L}_\text{mse} = \|\mathbf{e}_\varphi - \mathbf{e}'\|_2^2 \tag{1}$$

其中：
- $\mathbf{e}$：预训练模型（旧 AE）的 patch embedding
- $\mathbf{e}_\varphi$：新模型（新 AE）的 patch embedding
- $\mathbf{e}' = \text{AvgPool}(\mathbf{e})$：将旧 embedding 空间下采样到新尺寸

**训练过程**：
1. 冻结预训练 embedder，训练新 patch embedder 最小化 MSE
2. 联合微调输出头（5K-20K 步）

**必要性验证**：不做对齐直接微调 → DiT-XL FID 456.10（崩溃），SANA-1.6B FID 258.50（严重退化），FLUX FID 15.78（有伪影）。

### 3. Stage 2：LoRA 端到端微调

使用 Flow Matching 损失：

$$\mathcal{L}_\text{fm} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|\mathbf{v}_\theta(\mathbf{x}_t, c, t) - \mathbf{v}_t\|_2^2\right] \tag{2}$$

**LoRA 配置**：rank=256, alpha=256。

**LoRA vs 全微调**：

| 方法 | 可训练参数 | FID | CLIP Score |
|------|-----------|-----|-----------|
| 全微调 | 11.9B | 49.01 | 26.98 |
| **LoRA** | **1.1B** | **48.13** | **27.51** |

LoRA 更好地保留预训练语义知识。

### 4. 引导蒸馏模型的处理

对于已经过 CFG 蒸馏的模型（如 FLUX），标准 FM 损失不适用。本文推导修正的速度估计：

$$\mathbf{v}_\theta(\mathbf{x}_t, c, t) \approx \frac{1}{1+w}\left[\mathbf{v}_\eta(\mathbf{x}_t, c, t, w) + w \cdot \mathbf{v}_\eta(\mathbf{x}_t, \emptyset, t, w)\right] \tag{3-6}$$

修正后的目标：

$$\mathcal{L}_\text{fm}^\text{guide} = \mathbb{E}_{w,t,\mathbf{x}_0,\mathbf{x}_1}\left[\|\hat{\mathbf{v}}_\eta(\mathbf{x}_t, c, t, w) - \mathbf{v}_t\|_2^2\right] \tag{7}$$

**公式链条**：CFG 输出 (3) → 反解条件速度 (4-6) → 修正损失 (7)。

## 实验结果

### 类条件生成 (ImageNet)

| 模型 | 分辨率 | gFID↓ | 吞吐量 |
|------|--------|-------|--------|
| DiT-XL (基线) | 256 | 9.62 | 2.44 img/s |
| **DC-Gen-DiT-XL** | 256 | **8.01** | **12.00 img/s** |
| DiT-XL (基线) | 512 | 12.03 | 0.85 img/s |
| **DC-Gen-DiT-XL** | 512 | **8.21** | **4.03 img/s** |

4× 吞吐量提升 + 更好的 FID。

### 文本到图像 (MJHQ-30K, 1024×1024)

| 模型 | 吞吐量 (img/min) | CLIP | GenEval |
|------|-----------------|------|---------|
| SANA-1.6B (f32) | 110.70 | 29.01 | 0.82 |
| **DC-Gen-SANA-1.6B (f64)** | **435.68** | 28.91 | 0.82 |
| FLUX-Krea-12B (f8) | 16.82 | 27.93 | 0.69 |
| **DC-Gen-FLUX (f32)** | **69.37** | 27.94 | 0.72 |

### 高分辨率性能 (H100)

| 分辨率 | 基线延迟 | DC-Gen 延迟 | 加速比 |
|--------|---------|-----------|--------|
| 1K | 4.65s | 1.10s | 4.2× |
| 2K | 23.76s | 1.41s | 16.9× |
| **4K** | **213.81s** | **4.04s** | **52.9×** |

**+ SVDQuant (RTX 5090)**：4K 生成仅 **3.5 秒**，总加速 **138×**。

### 训练效率

| 指标 | DC-Gen-FLUX | 从头训练 |
|------|------------|---------|
| 训练成本 | **40 H100·天** | ~20,800 H100·天 |
| 节省 | **520×** | — |

## 个人思考

1. **后训练范式**的实用价值极高：任何预训练 DiT 都可以通过 DC-Gen 获得 4-53× 加速。
2. **Embedding 对齐**是稳健迁移的关键——跳过这步直接微调导致灾难性崩溃（FID 456）。
3. **4K 加速 53×**使得此前不可行的分辨率变得实用——从 3.5 分钟降到 4 秒。
4. **引导蒸馏模型的处理**是重要的工程贡献：FLUX 等预蒸馏模型需要修正损失才能正确微调。
5. **与量化的协同**（138× 总加速）展示了多种加速技术叠加的巨大潜力。
