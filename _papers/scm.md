---
title: "Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models"
authors: "Cheng Lu, Yang Song"
year: 2024
venue: "arXiv 2024"
tags: [扩散模型, 一致性模型, 图像生成]
rating: 5
paper_url: "https://arxiv.org/abs/2410.11081"
code_url: ""
summary: "统一 TrigFlow 框架 + 切线归一化/自适应加权/双重归一化等稳定化技术，首次将连续时间一致性模型扩展到 1.5B 参数，2 步采样 ImageNet 512 FID 1.88"
---

## 核心思想

**一致性模型（Consistency Model, CM）** 的目标是将扩散模型的多步采样压缩到 **1-2 步**，实现超快速生成。但此前连续时间 CM 的训练极不稳定，无法扩展到大规模。

本文提出 **TrigFlow** 统一框架 + 一系列稳定化技术，首次将连续时间 CM 扩展到 **1.5B 参数**，在 ImageNet 512×512 上仅用 **2 步采样**达到 FID **1.88**——接近最好的扩散模型（EDM2 的 1.85，需要 126 步）。

## 背景知识

### 什么是一致性模型

扩散模型的采样需要数十到数百步去噪，速度很慢。一致性模型的核心思想是：

**一致性函数** $f(\mathbf{x}_t, t)$：对于同一条概率流 ODE 轨迹上的所有点 $\mathbf{x}_t$，无论 $t$ 是多少，$f(\mathbf{x}_t, t)$ 都映射到同一个起点 $\mathbf{x}_0$。

$$f(\mathbf{x}_t, t) = \mathbf{x}_0, \quad \forall t \in [0, T]$$

有了这个函数，只需 **1 步**就能从噪声 $\mathbf{x}_T$ 直接得到干净图像：$\hat{\mathbf{x}}_0 = f(\mathbf{x}_T, T)$。

### 连续时间 vs 离散时间

| 方面 | 离散时间 CM | 连续时间 CM |
|------|-----------|-----------|
| 时间步 | 有限个离散点 $t_1, t_2, \ldots, t_N$ | 连续区间 $[0, T]$ |
| 训练目标 | 相邻时间步的一致性 | ODE 轨迹上的无穷小一致性 |
| 理论 | 有离散化误差 | 无离散化误差 |
| 实践 | 相对稳定 | **极不稳定**（本文要解决的问题） |

### 两种训练方式

- **一致性蒸馏（CD）**：从预训练扩散模型蒸馏而来
- **一致性训练（CT）**：从零开始训练，不需要预训练模型

## 方法详解

### 1. TrigFlow：三角函数统一框架

#### 1.1 扩散过程

给定干净数据 $\mathbf{x}_0$ 和噪声 $\mathbf{z} \sim \mathcal{N}(0, \sigma_d^2 \mathbf{I})$，TrigFlow 使用**三角函数**参数化：

$$\mathbf{x}_t = \cos(t) \cdot \mathbf{x}_0 + \sin(t) \cdot \mathbf{z}, \quad t \in [0, \pi/2]$$

- $t = 0$：$\mathbf{x}_0 = 1 \cdot \mathbf{x}_0 + 0 \cdot \mathbf{z}$ = 纯净数据
- $t = \pi/2$：$\mathbf{x}_{\pi/2} = 0 \cdot \mathbf{x}_0 + 1 \cdot \mathbf{z}$ = 纯噪声

**为什么用三角函数**：$\cos^2(t) + \sin^2(t) = 1$ 天然满足信噪比约束，大大简化了 EDM 和 Flow Matching 中复杂的噪声调度公式。

#### 1.2 概率流 ODE

$$\frac{d\mathbf{x}_t}{dt} = \sigma_d \mathbf{F}_\theta\left(\frac{\mathbf{x}_t}{\sigma_d}, c_\text{noise}(t)\right)$$

其中 $\mathbf{F}_\theta$ 是去噪网络，$c_\text{noise}(t)$ 是时间步编码。

#### 1.3 扩散模型训练目标

$$\mathcal{L}_\text{Diff}(\theta) = \mathbb{E}_{\mathbf{x}_0, \mathbf{z}, t}\left[\left\|\sigma_d \mathbf{F}_\theta\left(\frac{\mathbf{x}_t}{\sigma_d}, c_\text{noise}(t)\right) - \mathbf{v}_t\right\|^2\right]$$

其中**速度目标**为：

$$\mathbf{v}_t = \cos(t) \cdot \mathbf{z} - \sin(t) \cdot \mathbf{x}_0$$

这就是 $d\mathbf{x}_t / dt$ 的真值。

#### 1.4 一致性模型参数化

$$f_\theta(\mathbf{x}_t, t) = \cos(t) \cdot \mathbf{x}_t - \sin(t) \cdot \sigma_d \mathbf{F}_\theta\left(\frac{\mathbf{x}_t}{\sigma_d}, c_\text{noise}(t)\right)$$

**边界条件**自然满足：$f_\theta(\mathbf{x}_0, 0) = \cos(0) \cdot \mathbf{x}_0 - \sin(0) \cdot (\cdots) = \mathbf{x}_0$。

### 2. 稳定化技术

连续时间 CM 训练需要计算一致性函数的**时间导数** $df_{\theta^-}(\mathbf{x}_t, t)/dt$（其中 $\theta^-$ 是 EMA 参数）。这个导数展开为：

$$\frac{df_{\theta^-}}{dt} = -\cos(t)\left(\sigma_d \mathbf{F}_{\theta^-} - \frac{d\mathbf{x}_t}{dt}\right) - \sin(t)\left(\mathbf{x}_t + \sigma_d \frac{d\mathbf{F}_{\theta^-}}{dt}\right)$$

问题出在 $\sin(t) \cdot d\mathbf{F}_{\theta^-}/dt$ 这一项——它的数值可能**爆炸**。

#### 2.1 切线归一化（Tangent Normalization）

将时间导数归一化：

$$\frac{df_{\theta^-}/dt}{\|df_{\theta^-}/dt\| + c}, \quad c = 0.1$$

或者更简单地裁剪到 $[-1, 1]$。这**直接消除了梯度爆炸**。

#### 2.2 自适应加权函数

训练一个可学习的加权 $w_\phi(t)$：

$$\mathcal{L}_\text{sCM}(\theta, \phi) = \mathbb{E}_{\mathbf{x}_t, t}\left[\frac{e^{w_\phi(t)}}{D}\left\|\mathbf{F}_\theta - \mathbf{F}_{\theta^-} - \cos(t)\frac{df_{\theta^-}}{dt}\right\|^2 - w_\phi(t)\right]$$

$w_\phi(t)$ 会自动学会对不同时间步分配不同权重——噪声大的时间步需要更大权重。

#### 2.3 自适应双重归一化（Adaptive Double Normalization）

替换标准 AdaGN：

$$\text{标准}: \mathbf{y} = \text{norm}(\mathbf{x}) \odot \mathbf{s}(t) + \mathbf{b}(t)$$

$$\text{改进}: \mathbf{y} = \text{norm}(\mathbf{x}) \odot \text{pnorm}(\mathbf{s}(t)) + \text{pnorm}(\mathbf{b}(t))$$

其中 pnorm 是像素归一化，防止条件参数 $\mathbf{s}(t), \mathbf{b}(t)$ 数值过大。

#### 2.4 恒等时间变换

使用 $c_\text{noise}(t) = t$（直接用时间步），而非 EDM 的 $c_\text{noise}(t) \propto \log(\sigma_d \tan t)$。后者在 $t \to \pi/2$ 时 $\sin(t) \cdot \partial_t c_\text{noise}(t) = 1/\cos(t) \to \infty$。

#### 2.5 位置时间嵌入

用位置编码替代傅里叶嵌入。傅里叶嵌入的大频率参数会导致梯度剧烈振荡。

#### 2.6 切线预热（Tangent Warmup）

将 $\sin(t)$ 系数替换为 $r \cdot \sin(t)$，其中 $r$ 在前 10K 步从 0 线性增到 1。避免训练初期不稳定。

### 3. 训练方式

#### 3.1 一致性蒸馏（sCD）

从预训练扩散模型出发，利用扩散模型提供的 $d\mathbf{x}_t/dt$ 来训练 CM。

#### 3.2 一致性训练（sCT）

不需要预训练模型。用数据和噪声的无偏估计近似 $d\mathbf{x}_t/dt$：

$$\frac{d\mathbf{x}_t}{dt} \approx \cos(t) \cdot \mathbf{z} - \sin(t) \cdot \mathbf{x}_0 = \mathbf{v}_t$$

当 $\Delta t \to 0$ 时这个近似是精确的。

## 实验结果

### CIFAR-10 无条件生成

| 方法 | 采样步数 | FID |
|------|---------|-----|
| EDM | 35 | 2.01 |
| sCT（1 步） | 1 | 2.85 |
| **sCT（2 步）** | **2** | **2.06** |
| sCD（1 步） | 1 | 3.66 |
| sCD（2 步） | 2 | 2.52 |

2 步 sCT 几乎追平 35 步 EDM。

### ImageNet 64×64 类条件生成

| 方法 | 采样步数 | FID |
|------|---------|-----|
| EDM2 | 63 | 1.33 |
| sCT（1 步） | 1 | 2.04 |
| **sCT（2 步）** | **2** | **1.48** |
| sCD（2 步） | 2 | 1.66 |

### ImageNet 512×512 类条件生成

| 模型 | 参数量 | 采样步数 | FID |
|------|--------|---------|-----|
| EDM2-XL | 1.1B | 126 | 1.85 |
| **sCT-1.5B** | **1.5B** | **2** | **1.88** |
| sCT-1B | 1B | 2 | 2.08 |
| sCD-750M | 750M | 2 | 2.17 |

**1.5B 参数的 sCT 用 2 步达到 FID 1.88**——仅比 126 步的 EDM2 差 0.03。

### 连续时间 vs 离散时间

连续时间 CM 在所有离散化级别 $N$ 上都显著优于离散时间 CM。离散时间 CM 在 $N \approx 1024$ 时性能达到峰值然后下降（数值精度问题）。

### 消融实验

| 技术 | 效果 |
|------|------|
| 切线归一化 | 1 步和 2 步 FID 大幅改善 |
| 自适应加权 | 在切线归一化基础上进一步提升 |
| 位置嵌入替代傅里叶嵌入 | 消除时间导数不稳定 |
| 双重归一化 | 防止条件参数数值爆炸 |

## 个人思考

1. **TrigFlow 的优雅性**：用 $\cos/\sin$ 统一 EDM 和 Flow Matching，数学上极其简洁——所有信噪比关系都变成三角恒等式。
2. **"稳定化工程"的价值**：连续时间 CM 的理论优势早已知晓，但一直无法训练——本文通过 6 项工程改进使其成为现实，说明理论和工程同等重要。
3. **2 步 ≈ 126 步**：sCT-1.5B 的 2 步 FID 1.88 vs EDM2 的 126 步 FID 1.85，速度提升 63 倍代价仅 0.03 FID——对实际应用意义巨大。
4. **sCT > sCD 的意外发现**：从零训练的 sCT 反而优于从预训练模型蒸馏的 sCD——暗示蒸馏过程可能引入了不必要的约束。
5. **扩展性验证**：sCD 的 FID 与教师扩散模型保持恒定偏移，随模型增大同步改善——说明 CM 的扩展规律与扩散模型一致。
