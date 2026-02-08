---
title: "GenTron: Diffusion Transformers for Image and Video Generation"
authors: "Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, Juan-Manuel Perez-Rua"
year: 2023
venue: "arXiv 2023"
tags: [图像生成, 视频生成, Transformer]
rating: 4
paper_url: "https://arxiv.org/abs/2312.04557"
code_url: ""
summary: "系统探索 Transformer 替代 U-Net 做文本到图像/视频扩散生成：cross-attention 优于 adaLN 做文本条件，Motion-Free Guidance 解决视频质量退化"
---

## 核心思想

在扩散模型中，U-Net 一直是主流骨干网络。GenTron 系统探索了用 **Transformer（DiT）替代 U-Net** 做文本条件的图像和视频生成。

核心发现：
1. 文本条件下 **cross-attention 远优于 adaLN**（与类别条件的结论相反）
2. 模型从 900M 扩展到 3B+ 带来持续的质量提升
3. **Motion-Free Guidance (MFG)** 解决了 T2I→T2V 微调时的视频质量退化问题

## 背景知识：扩散模型基础

### 前向加噪过程

给定干净数据 $x_0$，逐步添加高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \mathbf{I})$$

其中 $\beta_t$ 是预定义的噪声调度。经过 $T$ 步后，$x_T$ 接近纯噪声。

### 反向去噪过程

训练一个网络 $\epsilon_\theta$ 预测噪声，反向逐步去噪：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，$z \sim \mathcal{N}(0, \mathbf{I})$。

### Latent Diffusion

不在像素空间操作，而是先用 VAE 编码到潜在空间（如 $256 \times 256$ 图像 → $32 \times 32 \times 4$ latent），在潜在空间做扩散。

### DiT 架构

将 latent tensor 切分为不重叠的 patch token（如 $2 \times 2$ patchification → $16 \times 16 = 256$ 个 token），通过 Transformer block 处理，最后解码回 latent 空间。

## 方法详解

### 1. 文本条件机制：adaLN vs Cross-Attention

**问题**：原始 DiT 用 adaLN（adaptive layer normalization）注入**类别**条件，但**文本**条件更复杂——如何最好地将文本信息融入 Transformer？

#### 1.1 adaLN 方案

将文本嵌入替换原来的 one-hot 类别嵌入，通过归一化层的缩放和偏移参数注入：

$$\text{adaLN}(h, \gamma, \beta) = \gamma \cdot \frac{h - \mu}{\sigma} + \beta$$

其中 $\gamma, \beta$ 由文本嵌入的 MLP 预测。

**局限**：adaLN 对所有空间位置施加**相同的**调制——文本说"左边是猫，右边是狗"时，无法区分空间位置。

#### 1.2 Cross-Attention 方案

图像特征作为 Query，文本嵌入作为 Key/Value：

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

$$Q = W_Q \cdot h_\text{image}, \quad K = W_K \cdot h_\text{text}, \quad V = W_V \cdot h_\text{text}$$

**优势**：每个图像 token 可以**选择性地关注**不同的文本 token → 实现空间敏感的文本条件。

#### 1.3 GenTron 的混合方案

- **时间步信息**：通过 adaLN 注入（全局调制足够）
- **文本信息**：通过 cross-attention 注入（需要空间选择性）

**实验验证**（T2I-CompBench 平均得分）：

| 文本编码器 | 注入方式 | 得分 |
|-----------|---------|------|
| CLIP-L | adaLN | 34.32 |
| CLIP-L | **cross-attn** | **47.84** |
| T5-XXL | cross-attn | 48.93 |
| CLIP + T5 | cross-attn | **49.13** |

Cross-attention 比 adaLN 高出 **13.5 分**——差距巨大。

### 2. 文本编码器选择

| 编码器 | 类型 | 优势 |
|--------|------|------|
| CLIP-L | 多模态对比学习 | 图文对齐好，但语义理解有限 |
| Flan-T5-XXL | 纯语言模型 | 深层语义理解强 |
| **CLIP + T5（组合）** | 两者互补 | **最优** |

T5 在属性绑定（颜色/形状）和空间关系上优于 CLIP，CLIP 在整体对齐上更好——两者互补。

### 3. 模型缩放

| 模型 | 深度 | 宽度 | MLP 宽度 | 参数量 |
|------|------|------|---------|--------|
| GenTron-XL/2 | 28 | 1152 | 4608 | 930M |
| GenTron-G/2 | 48 | 1664 | 6656 | **3.1B** |

缩放维度：层数（深度）+ patch 嵌入维度（宽度）+ MLP 隐藏层维度。

### 4. 文本到视频：时间建模

#### 4.1 时间自注意力

在每个 Transformer block 中插入轻量的**时间自注意力层**（在 cross-attention 之后、MLP 之前）：

```
# 空间注意力处理后的特征 x：shape (B*T, N, D)
x = rearrange(x, "(b t) n d → (b n) t d")  # 将帧维度展开
x = x + TempSelfAttn(LN(x))                  # 时间维度自注意力
x = rearrange(x, "(b n) t d → (b t) n d")  # 恢复空间维度
```

其中 $b$ = batch, $t$ = 帧数, $n$ = patch 数/帧, $d$ = 通道维度。

**初始化**：时间注意力的输出投影**零初始化** → 初始行为为恒等映射（通过残差连接）→ 初始时模型表现与纯图像模型一致。

#### 4.2 Motion-Free Guidance (MFG)

**问题**：从 T2I 模型微调到 T2V 后，**每帧的视觉质量显著下降**。原因：
1. 视频数据质量远低于图像数据（10M 视频 vs 2B+ 图像，存在运动模糊/水印）
2. 微调过程强调时间一致性，牺牲了空间质量

**核心思想**：将**运动**视为一种条件信号（类似文本条件），在推理时用 classifier-free guidance 控制运动强度。

**训练时**：以概率 $p_\text{motion-free}$ 禁用时间注意力（用恒等矩阵替换时间注意力掩码 → 每帧只看自己）。同时进行：
- 运动启用迭代：使用真实视频训练
- 运动禁用迭代：将图像重复 $T-1$ 次创建伪视频训练

**推理时**：三项引导公式：

$$\tilde{\epsilon}_\theta = \underbrace{\epsilon_\theta(x_t, \emptyset, \emptyset)}_{\text{无条件}} + \lambda_T \underbrace{[\epsilon_\theta(x_t, c_T, c_M) - \epsilon_\theta(x_t, \emptyset, c_M)]}_{\text{文本引导}} + \lambda_M \underbrace{[\epsilon_\theta(x_t, \emptyset, c_M) - \epsilon_\theta(x_t, \emptyset, \emptyset)]}_{\text{运动引导}}$$

- $c_T$：文本条件，$c_M$：运动条件（时间注意力启用/禁用）
- $\lambda_T = 7.5$（固定），$\lambda_M \in [1.0, 1.3]$（每样本调优）

**直觉**：$\lambda_M$ 控制"运动强度"——$\lambda_M$ 小 → 更接近静态图像质量；$\lambda_M$ 大 → 更多运动但可能质量下降。

## 实验结果

### T2I-CompBench

| 模型 | 颜色绑定 | 形状绑定 | 纹理绑定 | 空间关系 | 平均 |
|------|---------|---------|---------|---------|------|
| SD v1.4 | 37.65 | 35.76 | 41.56 | 12.46 | 31.50 |
| PixArt-α | 68.86 | 55.82 | 70.44 | 20.82 | 48.15 |
| **GenTron-G/2** | **76.74** | **57.00** | **71.50** | **20.98** | **49.99** |

颜色绑定比 PixArt-α 高 **7.88%**。

### 与主流模型对比

| 方法 | 参数量 | FID-30K↓ | CLIP↑ |
|------|--------|---------|-------|
| Imagen | 3.0B | 7.27 | 0.27 |
| SDXL | 2.6B | 17.82 | 0.329 |
| GenTron-XL/2 | 0.9B | 14.21 | 0.326 |
| **GenTron-G/2** | 3.1B | 14.53 | **0.335** |

### 人类评估（vs SDXL，100 提示 × 30 评估者）

| 维度 | GenTron 赢 | 平局 | SDXL 赢 |
|------|-----------|------|---------|
| 视觉质量 | **51.1%** | 19.8% | 29.1% |
| 文本对齐 | 42.3% | **42.9%** | 14.8% |

### 训练配置

- 低分辨率（256²）：batch 2048，500K 步
- 高分辨率（512²）：batch 784，300K 步
- 视频：34M 数据集，128 clips × 8 帧，4 FPS

## 个人思考

1. **"Cross-attention >> adaLN"** 的发现具有深远影响：后来的 PixArt-α、SD3 都采用了 cross-attention——GenTron 的实验提供了早期验证。
2. **Motion-Free Guidance** 的思路优雅：将运动视为可控条件而非固有属性——这使得同一模型可以在"高质量静态"和"有运动但略低质量"之间平滑插值。
3. **文本编码器组合**（CLIP + T5）成为后续模型的标准做法——各取所长比单一编码器更优。
4. **时间注意力零初始化**保证了平滑的 T2I→T2V 迁移——初始时不破坏预训练的图像生成能力。
5. **FID 的局限性**被明确指出：GenTron-G/2 的 FID（14.53）反而略差于 XL/2（14.21），但人类评估显著更好——FID 不是可靠的质量指标。
