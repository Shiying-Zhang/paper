---
title: "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
authors: "Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, Kaiming He"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, Transformer]
rating: 5
paper_url: "https://arxiv.org/abs/2512.02012"
code_url: ""
summary: "针对 MeanFlow 的三大挑战提出改进：v-loss 重参数化、灵活 CFG 条件化、高效上下文条件化，ImageNet 256×256 单步 FID 1.72"
---

## 核心思想

原始 MeanFlow 虽然实现了单步生成，但存在三个关键挑战：(1) 训练目标依赖网络自身，导致不稳定；(2) CFG 引导尺度在训练时固定；(3) 条件处理参数量大。本文提出 **improved MeanFlow (iMF)**，逐一解决这些问题，将 FID 从 3.43 降至 **1.72**（相对提升 50%）。

![原始 MF vs 改进 MF：原始 MF 存在非标准回归问题，改进方法仅依赖 z_t](https://arxiv.org/html/2512.02012v1/extracted_media/x1.png)

## 方法详解

### 1. 预备知识

#### 1.1 线性时间表与条件速度

$$\mathbf{z}_t = (1-t)\mathbf{x} + t\mathbf{e}, \quad \mathbf{v}_c = \mathbf{e} - \mathbf{x}$$

其中 $\mathbf{x} \sim p_\text{data}$，$\mathbf{e} \sim p_\text{prior}$。边际速度为 $\mathbf{v}(\mathbf{z}_t, t) \triangleq \mathbb{E}[\mathbf{v}_c | \mathbf{z}_t]$。

#### 1.2 平均速度定义

$$\mathbf{u}(\mathbf{z}_t, r, t) \triangleq \frac{1}{t-r}\int_r^t \mathbf{v}(\mathbf{z}_\tau) \, d\tau \tag{3}$$

#### 1.3 MeanFlow 恒等式

$$\mathbf{u}(\mathbf{z}_t) = \mathbf{v}(\mathbf{z}_t) - (t-r)\frac{d}{dt}\mathbf{u}(\mathbf{z}_t) \tag{4}$$

时间导数通过 JVP 计算：

$$\frac{d}{dt}\mathbf{u}(\mathbf{z}_t) = \partial_z \mathbf{u}(\mathbf{z}_t) \cdot \mathbf{v}(\mathbf{z}_t) + \partial_t \mathbf{u}(\mathbf{z}_t) \triangleq \text{JVP}(\mathbf{u}; \mathbf{v}) \tag{5}$$

### 2. 挑战一：网络依赖的训练目标 → v-loss 重参数化

#### 2.1 原始 MF 的问题

原始 MF 的训练目标：

$$\mathbf{u}_\text{tgt} = (\mathbf{e} - \mathbf{x}) - (t-r) \cdot \text{JVP}(\mathbf{u}_\theta; \mathbf{e} - \mathbf{x}) \tag{6}$$

$$\mathcal{L}_\text{MF} = \mathbb{E}_{t,r,\mathbf{x},\mathbf{e}} \|\mathbf{u}_\theta - \text{sg}(\mathbf{u}_\text{tgt})\|^2 \tag{7}$$

**问题**：目标 $\mathbf{u}_\text{tgt}$ 中包含 $\mathbf{u}_\theta$ 本身（通过 JVP），这不是一个标准的回归问题。训练损失非单调递减，方差大。

#### 2.2 v-loss 重参数化（核心改进）

从 MeanFlow 恒等式 (4) 出发，重新整理为：

$$\mathbf{v}(\mathbf{z}_t) = \mathbf{u}(\mathbf{z}_t) + (t-r)\frac{d}{dt}\mathbf{u}(\mathbf{z}_t) \tag{8}$$

定义复合预测：

$$\boxed{\mathbf{V}_\theta(\mathbf{z}_t) \triangleq \mathbf{u}_\theta(\mathbf{z}_t) + (t-r) \cdot \text{JVP}_\text{sg}(\mathbf{u}_\theta; \mathbf{v}_\theta)} \tag{12}$$

v-loss：

$$\mathcal{L}_\text{v} = \mathbb{E}_{t,r,\mathbf{x},\mathbf{e}} \|\mathbf{V}_\theta - (\mathbf{e} - \mathbf{x})\|^2$$

**关键改进**：原始 MF 的 JVP 输入是条件速度 $(\mathbf{e} - \mathbf{x})$（依赖数据对），而 iMF 用 $\mathbf{v}_\theta$ 替代——$\mathbf{v}_\theta$ 只依赖 $\mathbf{z}_t$，不依赖具体的 $(\mathbf{x}, \mathbf{e})$ 对。

**$\mathbf{v}_\theta$ 的两种实现**：

- **边界条件**（推荐）：$\mathbf{v}_\theta(\mathbf{z}_t, t) \equiv \mathbf{u}_\theta(\mathbf{z}_t, t, t)$，无额外参数
- **辅助头**：从 $\mathbf{u}_\theta$ 的特征加一个轻量头预测 $\mathbf{v}_\theta$

**公式关系链**：
- 公式 (4) → 恒等式的等价变换 → 公式 (8)
- 公式 (8) 用网络替代 → 公式 (12)
- 公式 (12) 与真实速度 $(\mathbf{e} - \mathbf{x})$ 做回归 → 标准的 v-loss
- v-loss 消除了目标对网络本身的依赖 → 训练更稳定

![训练损失对比：原始 MF 非单调且方差大，iMF 单调递减且稳定](https://arxiv.org/html/2512.02012v1/extracted_media/x4.png)

### 3. 挑战二：固定引导尺度 → 灵活 CFG 条件化

#### 3.1 原始 MF 的固定 CFG

$$\mathbf{v}_\text{cfg}(\mathbf{z}_t | \mathbf{c}) = \omega \, \mathbf{v}(\mathbf{z}_t | \mathbf{c}) + (1 - \omega) \, \mathbf{v}(\mathbf{z}_t) \tag{13}$$

原始 MF 在训练时固定 $\omega$，模型只能在该 $\omega$ 下生成。

#### 3.2 改进：将 ω 作为条件变量

$$\mathbf{V}_\theta(\cdot | \mathbf{c}, \omega) \triangleq \mathbf{u}_\theta(\mathbf{z}_t | \mathbf{c}, \omega) + (t-r) \cdot \text{JVP}_\text{sg} \tag{15}$$

训练时随机采样 $\omega \in [1.0, 8.0]$，推理时可自由调整。

**进一步扩展**：支持 CFG 间隔 $[t_\text{min}, t_\text{max}]$，完整条件集 $\Omega = \{\omega, t_\text{min}, t_\text{max}\}$：

$$\mathbf{u}_\theta = \mathbf{u}_\theta(\mathbf{z}_t \mid r, t, \mathbf{c}, \Omega) \tag{16}$$

#### 3.3 CFG 目标计算

$$\mathbf{v}_\text{cfg} = (\mathbf{e} - \mathbf{x}) + \left(1 - \frac{1}{\omega}\right)\big(\mathbf{u}_\theta(\mathbf{z}_t | t, t, \mathbf{c}) - \mathbf{u}_\theta(\mathbf{z}_t | t, t, \varnothing)\big) \tag{17}$$

![最优 CFG 尺度随训练 epoch 和推理步数变化](https://arxiv.org/html/2512.02012v1/extracted_media/x5.png)

### 4. 挑战三：条件处理效率 → 上下文条件化

原始方法使用 adaLN-zero 处理条件，参数量大。iMF 改用**多 token 上下文条件化**：

- 每个条件类型生成多个 token（类别 8 个，$r, t, \Omega$ 各 4 个）
- 沿序列轴与图像 token 拼接
- **完全移除 adaLN-zero 层**

效果：参数从 133M 降至 **89M**（减少 33%），FID 同时改善（4.57 → 4.09）。

![上下文条件化架构：条件转为多 token 并与图像 token 拼接](https://arxiv.org/html/2512.02012v1/extracted_media/x6.png)

### 5. 单步采样

$$\mathbf{z}_0 = \mathbf{z}_1 - \mathbf{u}_\theta(\mathbf{z}_1)$$

其中 $(r, t) = (0, 1)$，$\mathbf{z}_1 \sim p_\text{prior}$。单次前向传播完成生成。

## 实验结果

### 逐步消融（ImageNet 256×256, 1-NFE）

| 改进 | FID ↓ |
|------|-------|
| 原始 MF | 6.17 |
| + v-loss（边界条件 $\mathbf{v}_\theta$） | 5.97 |
| + v-loss（辅助头 $\mathbf{v}_\theta$） | 5.68 |
| + $\omega$ 条件化 | 5.52 |
| + $\Omega$ 条件化 | 4.57 |
| + 上下文条件化（−33% 参数） | 4.09 |
| + 高级 Transformer | 3.82 |
| + 640 轮训练 | 3.39 |

![FID 训练曲线：逐步改进 6.17 → 3.39](https://arxiv.org/html/2512.02012v1/extracted_media/x7.png)

### 系统级结果

| 模型 | 参数量 | FID ↓ | IS ↑ |
|------|--------|-------|------|
| MF-XL/2 | 676M | 3.43 | 247.5 |
| **iMF-XL/2** | **610M** | **1.72** | **282.0** |

**相对改进 50%**（3.43 → 1.72），参数量还减少了 10%。

### 与其他 1-NFE 方法对比（从头训练，无蒸馏）

| 方法 | 参数量 | FID ↓ |
|------|--------|-------|
| iCT-XL/2 | 675M | 34.24 |
| Shortcut-XL/2 | 675M | 10.60 |
| MeanFlow-XL/2 | 676M | 3.43 |
| **iMF-XL/2** | **610M** | **1.72** |

iMF 以显著优势领先所有从头训练的单步方法。

### 2-NFE 结果

iMF-XL/2（2-NFE）: FID = **1.54**，进一步接近多步方法。

## 个人思考

1. **v-loss 重参数化**是核心贡献：将非标准回归问题转化为标准回归，训练稳定性大幅提升。这个 insight 也被 pMF 采用。
2. **灵活 CFG** 非常实用：单个模型训练后可以在推理时自由调整引导强度，省去了为不同 $\omega$ 分别训练的开销。
3. **上下文条件化**的「参数更少效果更好」现象值得关注：adaLN-zero 的逐层调制可能过度参数化了。
4. **与 pMF 的互补关系**：iMF 工作在潜在空间，pMF 将其扩展到像素空间。两者共享 v-loss 框架，但 pMF 额外引入了流形假设（x-prediction）。
5. **1.72 FID 的意义**：这是首个从头训练（无蒸馏）的单步方法突破 FID 2.0 的结果。
