---
title: "One-step Latent-free Image Generation with Pixel Mean Flows"
authors: "Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, Kaiming He"
year: 2025
venue: "ICML 2025"
tags: [图像生成, Transformer]
rating: 5
paper_url: "https://arxiv.org/abs/2601.22158"
code_url: ""
summary: "提出像素 MeanFlow (pMF)，分离预测空间与损失空间，实现单步无潜在空间图像生成，ImageNet 256×256 达到 FID 2.22"
---

## 核心思想

现代扩散/流模型有两个核心特征：(i) 多步采样，(ii) 在潜在空间中操作。本文同时去掉这两点，提出 **pixel MeanFlow (pMF)**，实现**单步 (1-NFE)** + **像素空间**直接生成。

核心指导原则：**将网络输出空间与损失空间分离**——网络直接预测像素空间的去噪图像（x-prediction），而损失在速度空间（v-loss）中定义。

![pMF 框架：平均速度场 u 对应噪声图像，而新定义的场 x 对应近似干净图像](https://arxiv.org/html/2601.22158v2/extracted_media/x1.png)

![可视化：追踪 ODE 轨迹的 z_t、u、x 量](https://arxiv.org/html/2601.22158v2/extracted_media/x2.png)

## 方法详解

### 1. 预备知识：流匹配

#### 1.1 线性插值调度

$$\mathbf{z}_t = (1-t)\mathbf{x} + t\boldsymbol{\varepsilon} \tag{1}$$

其中 $\mathbf{x} \sim p_\text{data}$，$\boldsymbol{\varepsilon} \sim p_\text{prior}$，$t \in [0,1]$。条件速度场为：

$$\mathbf{v} = \boldsymbol{\varepsilon} - \mathbf{x} \tag{2}$$

#### 1.2 流匹配损失

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\varepsilon}} \|\mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v}\|^2 \tag{3}$$

目标是学习边际速度 $\mathbf{v}(\mathbf{z}_t, t) \triangleq \mathbb{E}[\mathbf{v} | \mathbf{z}_t, t]$。

### 2. JiT 的 x-prediction 参数化

JiT 方法将速度预测转换为图像预测：

$$\mathbf{v}_\theta(\mathbf{z}_t, t) := \frac{1}{t}(\mathbf{z}_t - \mathbf{x}_\theta(\mathbf{z}_t, t)) \tag{4}$$

其中 $\mathbf{x}_\theta = \text{net}_\theta$ 为 ViT 直接输出。**直觉**：网络不预测高维噪声速度，而是预测位于低维图像流形上的去噪图像。

### 3. MeanFlow 平均速度场

#### 3.1 平均速度定义

$$\mathbf{u}(\mathbf{z}_t, r, t) \triangleq \frac{1}{t-r}\int_r^t \mathbf{v}(\mathbf{z}_\tau, \tau) \, d\tau \tag{5}$$

定义在两个时间步 $r$ 和 $t$ 之间，$0 \leq r \leq t \leq 1$。

#### 3.2 MeanFlow 恒等式

$$\mathbf{v}(\mathbf{z}_t, t) = \mathbf{u}(\mathbf{z}_t, r, t) + (t-r)\frac{d}{dt}\mathbf{u}(\mathbf{z}_t, r, t) \tag{6}$$

关联**瞬时速度 v** 与**平均速度 u 及其时间导数**。这是 MeanFlow 的核心恒等式。

#### 3.3 改进 MeanFlow (iMF) 的网络预测

$$\mathbf{V}_\theta \triangleq \mathbf{u}_\theta + (t-r) \cdot \text{JVP}_\text{sg} \tag{7}$$

其中 JVP 为雅可比向量积，sg 为 stop-gradient。

### 4. 广义去噪图像场（本文核心贡献）

#### 4.1 新场定义

$$\boxed{\mathbf{x}(\mathbf{z}_t, r, t) \triangleq \mathbf{z}_t - t \cdot \mathbf{u}(\mathbf{z}_t, r, t)} \tag{8}$$

这是本文引入的**关键新概念**：将平均速度 $\mathbf{u}$ 变换为去噪图像 $\mathbf{x}$。

**三个边界情况验证这个定义的合理性**：

**边界 I**（$r = t$）：退化为 JiT 的去噪预测

$$\mathbf{x}(\mathbf{z}_t, t, t) = \mathbf{z}_t - t \cdot \mathbf{v}(\mathbf{z}_t, t) \tag{9}$$

**边界 II**（$r = 0$）：对应 ODE 轨迹终点，服从数据分布

$$\mathbf{x}(\mathbf{z}_t, 0, t) = \mathbf{z}_0 \tag{10}$$

**一般情况**（$0 < r < t$）：经验观察表明 $\mathbf{x}$ 近似为干净或轻微模糊的图像。

**公式 (8) 与 (5)(6) 的联系**：公式 (8) 本质上是公式 (5) 的变量变换。通过 $\mathbf{u} = (\mathbf{z}_t - \mathbf{x})/t$，可以在 $\mathbf{x}$-空间和 $\mathbf{u}$-空间之间自由转换，而公式 (6) 保证了这种转换在 MeanFlow 恒等式下的一致性。

### 5. 像素 MeanFlow 网络参数化

#### 5.1 从 x-prediction 恢复 u

$$\mathbf{u}_\theta(\mathbf{z}_t, r, t) = \frac{1}{t}(\mathbf{z}_t - \mathbf{x}_\theta(\mathbf{z}_t, r, t)) \tag{11}$$

其中 $\mathbf{x}_\theta(\mathbf{z}_t, r, t) := \text{net}_\theta(\mathbf{z}_t, r, t)$。

**推导逻辑**：从公式 (8) 反解 → $\mathbf{u} = (\mathbf{z}_t - \mathbf{x})/t$ → 用网络 $\mathbf{x}_\theta$ 替代真实 $\mathbf{x}$ → 得到公式 (11)。

#### 5.2 pMF 优化目标

$$\mathcal{L}_\text{pMF} = \mathbb{E}_{t,r,\mathbf{x},\boldsymbol{\varepsilon}} \|\mathbf{V}_\theta - \mathbf{v}\|^2 \tag{12}$$

其中 $\mathbf{V}_\theta \triangleq \mathbf{u}_\theta + (t-r) \cdot \text{JVP}_\text{sg}$。

**完整计算链**：
1. 网络输出 $\mathbf{x}_\theta = \text{net}_\theta(\mathbf{z}_t, r, t)$ — **x-prediction**
2. 转换 $\mathbf{u}_\theta = (\mathbf{z}_t - \mathbf{x}_\theta)/t$ — 公式 (11)
3. 计算 JVP 得到 $d\mathbf{u}/dt$ — MeanFlow 恒等式的导数项
4. 合成 $\mathbf{V}_\theta = \mathbf{u}_\theta + (t-r) \cdot \text{sg}(d\mathbf{u}/dt)$ — 公式 (7)
5. 与真实速度 $\mathbf{v} = \boldsymbol{\varepsilon} - \mathbf{x}$ 计算 L2 损失 — **v-loss**

### 6. 感知损失

由于网络直接输出像素空间图像，可自然加入感知损失：

$$\mathcal{L} = \mathcal{L}_\text{pMF} + \lambda \mathcal{L}_\text{perc} \tag{13}$$

其中 $\mathcal{L}_\text{perc}$ 为 LPIPS 损失（VGG 或 ConvNeXt-V2），$\lambda$ 为权重，仅在 $t \leq t_\text{thr}$ 时应用。

**为什么 pMF 天然支持感知损失？** 传统 u-prediction 输出的是速度场（类似噪声），无法直接计算 LPIPS；而 x-prediction 输出的就是去噪图像，「所见即所得」，可直接送入感知网络。

### 7. 关键设计选择

#### 7.1 时间采样策略

在 $(r, t)$ 平面上采样，$0 \leq r \leq t \leq 1$。消融实验表明覆盖完整三角形区域至关重要：

| 采样方案 | FID |
|---------|-----|
| 仅 $r = t$ | 194.53 |
| 仅 $r = 0$ | 389.28 |
| $r \in \{0, t\}$ | 106.59 |
| **$0 \leq r \leq t$** | **3.53** |

#### 7.2 x-prediction vs u-prediction

| 分辨率 | Patch 维度 | x-prediction FID | u-prediction FID |
|--------|-----------|-----------------|-----------------|
| 64×64 | 48 | 3.80 | 3.82 |
| 256×256 | 768 | **9.56** | **164.89** |

高维观测空间中 u-prediction 完全崩溃，验证了**流形假设**：$\mathbf{x}$ 位于低维流形上，比 $\mathbf{u}$ 更易学习。

![Toy 实验：x-prediction 在各维度稳定，u-prediction 高维时崩溃](https://arxiv.org/html/2601.22158v2/extracted_media/figs/demo_final.png)

#### 7.3 优化器选择

Muon 优化器显著优于 Adam（FID 8.71 vs 11.86 @320 epochs）。

![Muon vs Adam 训练曲线](https://arxiv.org/html/2601.22158v2/extracted_media/x3.png)

## 实验结果

### ImageNet 256×256 主要结果

| 方法 | NFE | 空间 | 参数量 | GFLOPs | FID ↓ |
|------|-----|------|--------|--------|-------|
| **pMF-H/16** | **1** | **像素** | **956M** | **271** | **2.22** |
| **pMF-L/16** | **1** | **像素** | **410M** | **117** | **2.52** |
| iMF-XL/2 | 1 | 潜在 | 610M | 175 | 1.72 |
| StyleGAN-XL | 1 | 像素 | 166M | 1574 | 2.30 |
| SiT-XL/2 | 250×2 | 潜在 | 675M | 119 | 2.06 |

- **首个一步无潜在方法达到 FID 2.22**
- 计算量仅为 StyleGAN-XL 的 **1/6**（271 vs 1574 GFLOPs）
- 避免 VAE 解码器开销（256×256: 310 GFLOPs, 512×512: 1230 GFLOPs）

### ImageNet 512×512

| 方法 | FID ↓ |
|------|-------|
| **pMF-H/32** | **2.48** |
| DiT-XL/2 | 3.04 |
| SiT-XL/2 | 2.62 |

通过激进 patch 大小（32×32）保持与 256 相同计算量，同时避免 VAE 解码器 1230 GFLOPs 开销。

## 个人思考

1. **预测空间与损失空间分离**是本文最优雅的 insight：x-prediction 利用流形假设降低学习难度，v-loss 保证训练目标的数学正确性。
2. **流形假设的实验验证**非常有说服力：toy 实验和 256×256 消融都清晰表明高维 u-prediction 的灾难性失败。
3. **感知损失的自然融入**是 pixel-space 方法的独有优势——潜在空间方法需要先 VAE 解码才能计算 LPIPS。
4. **与 iMF 的关系**：pMF 可以视为 iMF 在像素空间的推广，通过公式 (8)(11) 建立了 x-space 和 u-space 之间的桥梁。
5. **512×512 的 patch=32 策略**很实用：证明了 x-prediction 可以处理极高维 patch（3072 维），而 u-prediction 在此维度早已崩溃。
