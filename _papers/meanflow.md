---
title: "Mean Flows for One-step Generative Modeling"
authors: "Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, Transformer]
rating: 5
paper_url: "https://arxiv.org/abs/2505.13447"
code_url: ""
summary: "引入平均速度概念替代瞬时速度，实现单步生成 FID 3.43，无需预训练/蒸馏/课程学习"
---

## 核心思想

Flow Matching 用**瞬时速度** $\mathbf{v}(\mathbf{z}_t, t)$ 描述流场，推理需要多步 ODE 积分。本文提出 **MeanFlow**：用**平均速度** $\mathbf{u}(\mathbf{z}_t, r, t)$ 替代瞬时速度，单步即可完成积分（因为平均速度直接给出位移）。

**关键创新**：推导出一个恒等式（MeanFlow Identity），将平均速度与瞬时速度联系起来，使得平均速度可以通过神经网络训练得到。

![MeanFlow 单步生成 vs 先前方法](https://arxiv.org/html/2505.13447v1/extracted_media/x1.png)

## 方法详解

### 1. Flow Matching 预备知识

#### 1.1 边际速度

$$\mathbf{v}(\mathbf{z}_t, t) \triangleq \mathbb{E}_{p_t(\mathbf{v}_t | \mathbf{z}_t)}[\mathbf{v}_t] \tag{1}$$

#### 1.2 采样 ODE

$$\frac{d}{dt}\mathbf{z}_t = \mathbf{v}(\mathbf{z}_t, t) \tag{2}$$

需要多步数值积分（如 Euler、Heun），计算量大。

### 2. 平均速度定义

$$\boxed{\mathbf{u}(\mathbf{z}_t, r, t) \triangleq \frac{1}{t-r}\int_r^t \mathbf{v}(\mathbf{z}_\tau, \tau) \, d\tau} \tag{3}$$

**含义**：从时间 $r$ 到 $t$ 的**平均速度** = 位移 / 时间间隔。

**为什么平均速度有用？** 知道 $\mathbf{u}(\mathbf{z}_t, 0, t)$ 后，一步即可采样：

$$\mathbf{z}_r = \mathbf{z}_t - (t-r) \cdot \mathbf{u}(\mathbf{z}_t, r, t) \tag{12}$$

取 $r=0, t=1$：$\mathbf{z}_0 = \mathbf{z}_1 - \mathbf{u}(\mathbf{z}_1, 0, 1)$。

![瞬时速度 v（切线方向）vs 平均速度 u（位移方向）](https://arxiv.org/html/2505.13447v1/extracted_media/x3.png)

### 3. MeanFlow 恒等式（核心推导）

#### 3.1 等式变换

将公式 (3) 改写为：

$$(t-r) \cdot \mathbf{u}(\mathbf{z}_t, r, t) = \int_r^t \mathbf{v}(\mathbf{z}_\tau, \tau) \, d\tau \tag{4}$$

对两边关于 $t$ 求导：

$$\mathbf{u} + (t-r)\frac{d}{dt}\mathbf{u} = \mathbf{v}(\mathbf{z}_t, t)$$

整理得到 **MeanFlow 恒等式**：

$$\boxed{\mathbf{u}(\mathbf{z}_t, r, t) = \mathbf{v}(\mathbf{z}_t, t) - (t-r)\frac{d}{dt}\mathbf{u}(\mathbf{z}_t, r, t)} \tag{6}$$

**含义**：平均速度 = 瞬时速度 − 时间间隔 × 平均速度的时间变化率。

#### 3.2 时间导数展开

$$\frac{d}{dt}\mathbf{u}(\mathbf{z}_t, r, t) = \mathbf{v}(\mathbf{z}_t, t) \cdot \partial_z \mathbf{u} + \partial_t \mathbf{u} \tag{8}$$

通过链式法则展开全导数。这个导数可以用**雅可比向量积 (JVP)** 高效计算。

### 4. 训练目标

用网络 $\mathbf{u}_\theta$ 逼近真实平均速度。将公式 (6) 中的未知量用网络替代：

$$\mathbf{u}_\text{tgt} = \mathbf{v}_t - (t-r)(\mathbf{v}_t \cdot \partial_z \mathbf{u}_\theta + \partial_t \mathbf{u}_\theta) \tag{公式 9-10 合并}$$

$$\boxed{\mathcal{L}(\theta) = \mathbb{E}\left[\|\mathbf{u}_\theta(\mathbf{z}_t, r, t) - \text{sg}(\mathbf{u}_\text{tgt})\|_2^2\right]} \tag{9}$$

其中 $\mathbf{v}_t = \boldsymbol{\varepsilon} - \mathbf{x}$ 为条件速度，sg 为 stop-gradient。

**公式链条**：
- 定义平均速度 (3) → 对 $t$ 求导 → MeanFlow 恒等式 (6)
- 恒等式 (6) 用网络替代 → 训练目标 (9)
- 导数通过 JVP 计算 (8)，额外计算开销约 20%

### 5. Classifier-Free Guidance

CFG 速度场：

$$\mathbf{v}^\text{cfg}(\mathbf{z}_t, t | \mathbf{c}) \triangleq \omega \, \mathbf{v}(\mathbf{z}_t, t | \mathbf{c}) + (1-\omega) \, \mathbf{v}(\mathbf{z}_t, t) \tag{13}$$

MeanFlow 恒等式自然扩展到 CFG 设置：

$$\mathbf{u}^\text{cfg} = \mathbf{v}^\text{cfg} - (t-r)\frac{d}{dt}\mathbf{u}^\text{cfg} \tag{15}$$

训练时的修改速度：

$$\tilde{\mathbf{v}}_t \triangleq \omega \, \mathbf{v}_t + (1-\omega) \, \mathbf{u}_\theta^\text{cfg}(\mathbf{z}_t, t, t) \tag{19}$$

将 CFG 融入训练，保持 1-NFE 采样。

### 6. 单步采样

$$\mathbf{z}_0 = \mathbf{z}_1 - \mathbf{u}_\theta(\mathbf{z}_1, 0, 1), \quad \mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

一次前向传播完成生成。

## 实验结果

### ImageNet 256×256 主要结果

| 方法 | NFE | FID ↓ |
|------|-----|-------|
| iCT-XL/2 | 1 | 34.24 |
| Shortcut-XL/2 | 1 | 10.60 |
| **MeanFlow-XL/2** | **1** | **3.43** |
| **MeanFlow-XL/2+** | **2** | **2.20** |
| DiT-XL/2 | 250 | 2.27 |

MeanFlow 单步 FID 3.43，比此前最佳（Shortcut 10.60）提升 **50-70%**。2-NFE 达到 2.20，与 250 步 DiT 相当。

### 模型缩放

| 模型 | 参数量 | 1-NFE FID ↓ |
|------|--------|------------|
| MeanFlow-B/2 | 131M | 6.17 |
| MeanFlow-M/2 | 308M | 5.01 |
| MeanFlow-L/2 | 459M | 3.84 |
| MeanFlow-XL/2 | 676M | 3.43 |

![MeanFlow 跨模型尺度的 FID](https://arxiv.org/html/2505.13447v1/extracted_media/x4.png)

### 关键消融

| 消融项 | FID |
|--------|-----|
| 最优 $r \neq t$ 比例 25% | 61.06 |
| $r \neq t$ 比例 0%（退化为 FM） | 328.91 |
| 错误的 JVP 切向量 | 268~329 |
| 正确的 JVP | **61.06** |

$r \neq t$ 的采样和正确的 JVP 计算都是关键。

## 个人思考

1. **平均速度的直觉**非常自然：知道了起点到终点的「平均速度」，一步就能到达目的地，无需沿路径逐步走。
2. **MeanFlow 恒等式**是全文的数学核心：巧妙地将不可直接训练的积分量转化为可用 JVP 计算的微分形式。
3. **与后续工作的关系**：MeanFlow 是 iMF（改进版本）和 pMF（像素空间扩展）的基础，三篇论文形成递进关系。
4. **CFG 的优雅融入**：将引导视为流场性质而非后处理技巧，保持 1-NFE。
5. **局限性**：训练目标依赖网络自身（通过 JVP），导致训练不稳定——这正是 iMF 要解决的问题。
