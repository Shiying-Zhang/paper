---
title: "MeanFlow Transformers with Representation Autoencoders"
authors: "Zheyuan Hu, Chieh-Hsin Lai, Ge Wu, Yuki Mitsufuji, Stefano Ermon"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, 流匹配, 表示自编码器]
rating: 5
paper_url: "https://arxiv.org/abs/2511.13019"
code_url: "https://github.com/sony/mf-rae"
summary: "MeanFlow + RAE 的高效少步生成：DiT^DH 时间差编码 + CMT 中间训练 + MFD 有限差分蒸馏，1-step FID 2.03（ImageNet 256），训练成本降 6×，GFLOPS 降 38%"
---

## 核心思想

**MeanFlow（MF）** 可以实现 1-2 步生成，但在标准 SD-VAE 潜在空间上训练慢、依赖 CFG。**RAE**（Representation Autoencoder）用冻结的预训练编码器（如 DINO）替代 VAE，提供更好的潜在空间。

**MF-RAE** 将两者结合：
1. **DiT^DH** 架构：扩展 DiT 加入时间差编码 → 显式编码绝对时间和相对时间
2. **CMT 中间训练**：用教师 ODE 轨迹做固定回归 → 稳定初始化
3. **MFD 有限差分蒸馏**：用有限差分近似替代昂贵的 JVP 计算

结果：1-step FID **2.03**（ImageNet 256），训练成本仅 **~100 GPU-days**（降 6×），采样 GFLOPS 降 **38%**。

## 背景知识

### MeanFlow 基础

MeanFlow 学习 ODE 轨迹上的**平均速度场**：

$$h_\theta(z_t, t, s) \approx \frac{1}{t - s} \int_s^t v(z_u, u) du$$

即从时间 $t$ 到 $s$ 的平均传输速度 → 一步可以从噪声跳到干净样本。

### RAE vs VAE

| 方面 | SD-VAE | RAE |
|------|--------|-----|
| 编码器 | 学习的 CNN | **冻结的 DINO/SigLIP** |
| 潜在空间维度 | 低维（4 通道） | **高维（语义丰富）** |
| 解码器 | CNN | **ViT + DiT^DH** |
| CFG 依赖 | 严重（无 CFG FID 5.94） | **不需要（FID 2.03）** |

### 为什么 RAE 消除了 CFG 依赖

VAE 的低维潜在空间语义不足 → 需要 CFG 的条件-无条件差分信号来引导。RAE 的高维语义潜在空间本身就包含丰富的类别信息 → 不需要额外引导。

## 方法详解

### 1. DiT^DH 架构

扩展标准 DiT，加入**时间差编码**：

$$\text{conditioning} = \text{Embed}(\text{class}) + \text{Embed}(t) + \text{Embed}(t - s)$$

- $t$：当前噪声级别
- $t - s$：目标时间差
- 显式编码两个时间信息 → 模型知道"从哪里来"和"要跳多远"

### 2. 三阶段训练管道

#### Stage 1：预训练（78 H100 GPU-days）

在 RAE 潜在空间训练标准流匹配教师：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_t \mathbb{E}_{z_0, \epsilon} \left[w(t) \|v_\phi(z_t, t) - (\alpha'_t z_0 + \sigma'_t \epsilon)\|_2^2\right]$$

800 个 epoch。

#### Stage 2：CMT 中间训练（2.1 GPU-days）

用教师 ODE 求解器（Euler 16 步）生成固定轨迹点作为回归目标：

$$\mathcal{L}_{\text{CMT-MF}}(\theta) = \mathbb{E}_{i > j} \mathbb{E}_{z_T \sim p_{\text{prior}}} \left[\left\|h_\theta(\hat{z}_{t_i}, t_i, t_j) - \frac{\hat{z}_{t_i} - \hat{z}_{t_j}}{t_i - t_j}\right\|_2^2\right]$$

27K 迭代 → 提供轨迹一致的初始化。

#### Stage 3：后训练（21 GPU-days）

**MFD（MeanFlow Distillation）**：用有限差分近似传输导数：

$$\frac{d}{dt} h_\theta(z_t, t, s) \approx \frac{h_\theta(z_{t+\Delta t}, t+\Delta t, s) - h_\theta(z_{t-\Delta t}, t-\Delta t, s)}{2\Delta t}$$

其中 $z_{t \pm \Delta t} \approx z_t \pm \Delta t \cdot w(z_t, t)$（Euler 步），$\Delta t = 0.005$。

可选的 **MFT 引导**作为最后阶段的微调。

### 3. 偏差-方差权衡（Proposition 3.1）

混合速度 $w_\lambda = (1 - \lambda) \hat{v} + \lambda v_\phi$：

- **MFD**（$\lambda = 1$）：低方差但继承教师偏差
- **MFT**（$\lambda = 0$）：无偏但高方差

最优策略：**先 MFD 后 MFT** → 先稳定收敛，再去偏。

### 4. 训练总成本

| 阶段 | GPU-days | 说明 |
|------|---------|------|
| 预训练 | 78 | 标准 FM 教师 |
| CMT | 2.1 | 轨迹回归初始化 |
| 后训练 | 21 | MFD + 可选 MFT |
| **总计** | **~100** | **比 vanilla MF 的 600+ 低 6×** |

## 实验结果

### ImageNet 256×256

| 方法 | 1-step FID↓ | 2-step FID↓ | 参数量 |
|------|-----------|-----------|--------|
| MeanFlow | 3.43 | 2.20 | 676M |
| CMT w/ MF | 3.34 | — | 676M |
| AlphaFlow | 2.58 | 1.95 | 675M |
| **MF-RAE** | **2.03** | **1.89** | **841M** |

### GFLOPS 对比

| 方法 | 解码器 | 扩散 | 总计 | 降低 |
|------|--------|------|------|------|
| Vanilla DiT MF | 310 | 114 | 424 | — |
| **MF-RAE** | **106** | **157** | **263** | **38%** |

RAE 的轻量解码器大幅降低总 GFLOPS。

### 消融实验

| 算法 | 引导 | 架构 | 1-step FID |
|------|------|------|-----------|
| MFT | 有 | SD-VAE+DiT | 3.38 |
| MFD | 有 | SD-VAE+DiT | 3.15 |
| MFD | **无** | SD-VAE+DiT | **5.94** |
| MFT | **无** | RAE+DiT^DH | 2.81 |
| **MFD** | **无** | **RAE+DiT^DH** | **2.03** |

**关键发现**：SD-VAE 无 CFG 时 FID 从 3.15 暴涨到 5.94 → RAE 消除了 CFG 依赖。

### ImageNet 512×512

| 方法 | 1-step FID↓ | GFLOPS↓ | 训练成本 (GPU-days) |
|------|-----------|---------|-------------------|
| sCD | 2.28 | 2344 | 233 |
| CMT w/ ECD | 3.38 | 2344 | — |
| **MF-RAE** | **3.23** | **1051** | **17** |

训练成本仅 17 GPU-days（sCD 的 **1/14**），GFLOPS 降 **55%**。

### 初始化消融（ImageNet 512）

| 初始化 | 算法 | 1-step FID |
|--------|------|-----------|
| 随机 | MFT | 梯度爆炸 |
| FM | MFD | 梯度爆炸 |
| CMT | MFT only | 5.82 |
| CMT | MFD only | 3.95 |
| **CMT** | **MFD+MFT** | **3.23** |

CMT 初始化对稳定性**至关重要**——没有 CMT 直接训练会梯度爆炸。

## 个人思考

1. **RAE 消除 CFG 依赖**是最重要的实用贡献：CFG 需要两次模型前向传播 → 不需要 CFG 直接节省 50% 推理成本。
2. **三阶段管道（预训练→CMT→MFD+MFT）** 每步都有明确目的：FM 教师提供质量保证，CMT 提供稳定初始化，MFD 提供低方差蒸馏。
3. **有限差分替代 JVP** 是简洁的工程选择：JVP（Jacobian-vector product）计算复杂且内存消耗大，有限差分只需要额外两次前向传播。
4. **100 GPU-days 总训练成本**使高质量少步生成变得实际可行——之前 600+ GPU-days 的成本限制了研究和部署。
5. **DiT^DH 的时间差编码**看似小改动但效果显著——让模型显式区分"当前位置"和"跳跃距离"，而非隐式推断。
