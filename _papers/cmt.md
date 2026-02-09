---
title: "CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models"
authors: "Zheyuan Hu, Chieh-Hsin Lai, Yuki Mitsufuji, Stefano Ermon"
year: 2025
venue: "arXiv 2025"
tags: [扩散模型, 一致性模型, 训练效率]
rating: 4
paper_url: "https://arxiv.org/abs/2509.24526"
code_url: ""
summary: "扩散模型的中间训练阶段：用预训练教师的轨迹点做固定回归目标初始化一致性/流映射模型，2-step FID 1.97/1.32/1.84，训练成本降低 50-98%"
---

## 核心思想

一致性模型（CM）和流映射模型（MF）可以实现 1-2 步快速生成，但训练不稳定、收敛慢——因为它们依赖 **stop-gradient** 的自蒸馏目标（EMA 教师），梯度信号有偏。

**CMT（Consistency Mid-Training）** 提出在预训练和后训练之间插入一个**中间训练阶段**：
1. 用预训练扩散模型（教师）生成确定性 ODE 轨迹
2. **以轨迹上的点作为固定回归目标**训练学生模型
3. 学生获得**轨迹一致的初始化** → 后训练更快收敛

结果：2-step FID **1.97**（CIFAR-10）、**1.32**（ImageNet 64）、**1.84**（ImageNet 512），训练成本降低 **50-98%**。

## 背景知识

### 扩散模型的加速生成谱系

| 方法 | NFE | 思路 |
|------|-----|------|
| DDPM | ~1000 | 标准采样 |
| DDIM | ~50 | 确定性 ODE |
| DPM-Solver | ~10-20 | 高阶 ODE 求解器 |
| **一致性模型** | **1-2** | **学习 ODE 轨迹上的一步映射** |
| **流映射** | **1-2** | **学习时间步间的流映射** |

### 一致性模型的训练困难

一致性模型的目标是：对 ODE 轨迹上的任意两点 $x_{t_1}, x_{t_2}$，模型输出相同结果：

$$f_\theta(x_{t_1}, t_1) = f_\theta(x_{t_2}, t_2) = x_0$$

训练时用 **stop-gradient** 的 EMA 教师提供目标：

$$\mathcal{L} = d(f_\theta(x_{t_i}, t_i), \text{sg}[f_{\theta^-}(x_{t_{i-1}}, t_{i-1})])$$

**问题**：
- EMA 教师本身在学习中 → 目标不稳定
- stop-gradient 引入梯度偏差
- 需要精心设计时间采样、损失加权、EMA 衰减等超参数

### CMT 的灵感

类比 LLM 的三阶段训练：预训练 → **中间训练** → 后训练。CMT 是视觉生成领域的"中间训练"。

## 方法详解

### 1. 三阶段管道

| 阶段 | 目标 | 特点 |
|------|------|------|
| Stage 1：预训练 | 学习确定性 ODE 采样器 | 标准扩散模型训练 |
| **Stage 2：CMT** | **学习轨迹映射** | **固定回归目标，简单稳定** |
| Stage 3：后训练 | 精细化一致性/流映射 | 从 CMT 初始化出发 |

### 2. CMT 的核心：固定回归目标

#### 对于一致性模型

$$\mathcal{L}_{\text{CMT-CM}}(\theta) = \mathbb{E}_i \mathbb{E}_{x_T \sim p_{\text{prior}}} \left[d\left(f_\theta(\hat{x}_{t_i}, t_i), \hat{x}_{t_0}\right)\right]$$

- $\hat{x}_{t_i}$：教师 ODE 轨迹在时间 $t_i$ 的点（**固定，不随训练更新**）
- $\hat{x}_{t_0}$：轨迹终点（干净样本）
- $d$：距离函数（LPIPS 感知损失）

**与标准一致性训练的区别**：目标 $\hat{x}_{t_0}$ 是**预计算的固定值**，不是 EMA 教师的输出 → 消除了 stop-gradient 的不稳定性。

#### 对于流映射（Mean Flow）

$$\mathcal{L}_{\text{CMT-MF}}(\theta) = \mathbb{E}_{i > j} \mathbb{E}_{x_T \sim p_{\text{prior}}} \left[\left\|h_\theta(\hat{x}_{t_i}, t_i, t_j) - \frac{\hat{x}_{t_i} - \hat{x}_{t_j}}{t_i - t_j}\right\|_2^2\right]$$

学习轨迹上两点之间的**有限差分**（速度场）。

### 3. 理论分析（Theorem 5.1）

定义梯度偏差度量 $\mathcal{B}(\theta) = \|\nabla \mathcal{L}_{\text{oracle}} - \nabla \mathcal{L}_{\text{CM}}\|_2^2$：

| 初始化 | 梯度偏差 |
|--------|---------|
| **CMT** | $\mathcal{O}(\varepsilon + \Delta t^2)$ |
| 扩散模型 | $\mathcal{O}(\varepsilon + \Delta t^2 + \mathbb{E}_t[\sigma_t^2/\alpha_t^2]) +$ 额外项 |
| 随机初始化 | $\mathcal{O}(1)$ |

CMT 提供**最小的梯度偏差** → 后训练优化更稳定、更高效。

### 4. CMT 消除的复杂设计

| 被消除的设计 | 说明 |
|------------|------|
| Stop-gradient 目标 | 改为固定回归目标 |
| 自定义时间采样计划 | 不需要 |
| 手工损失加权 | 不需要 |
| 复杂超参数调优 | $\Delta t$ 退火、EMA 变体等均不需要 |

## 实验结果

### CIFAR-10 & ImageNet 64×64

| 数据集 | 方法 | 1-step FID | 2-step FID |
|--------|------|-----------|-----------|
| CIFAR-10 | iCT-deep | 2.51 | 2.24 |
| CIFAR-10 | IMM | 3.20 | 1.98 |
| CIFAR-10 | **CMT (w/ ECT)** | **2.74** | **1.97** |
| ImageNet 64 | sCT | 2.04 | 1.48 |
| ImageNet 64 | ECD | 2.24 | 1.50 |
| ImageNet 64 | **CMT (w/ ECD)** | **1.78** | **1.32** |

### AFHQv2 & FFHQ（51.2M 图像预算）

| 数据集 | 方法 | 1-step FID | 2-step FID |
|--------|------|-----------|-----------|
| AFHQv2 | ECT | 3.89 | 2.61 |
| AFHQv2 | **CMT (w/ ECT)** | **3.28** | **2.34** |
| FFHQ | ECT | 5.99 | 4.39 |
| FFHQ | **CMT (w/ ECT)** | **3.89** | **2.75** |

FFHQ 上改善尤为显著（4.39 → **2.75**，降低 37%）。

### ImageNet 512×512

| 方法 | 1-step FID | 2-step FID | 训练成本 (Mimgs) |
|------|-----------|-----------|----------------|
| sCD | 2.28 | 1.88 | 409.6 |
| AYF | 3.32 | 1.87 | 102.4 |
| ECD | 8.47 | 3.38 | 409.6 |
| **CMT (w/ ECD)** | **3.38** | **1.84** | **28.8** |

CMT 用 **28.8M 图像**达到 1.84 FID，sCD 用 **409.6M 图像**达到 1.88 FID → **93% 成本降低**且更优。

### ImageNet 256×256 训练时间

| 方法 | 预训练 | 中间训练 | 后训练 | 总时间 | FID |
|------|--------|---------|--------|--------|-----|
| MF-XL/2（从零） | 0 | 0 | 1520h | 1520h | 3.43 |
| **CMT-XL/2** | 38h | 135h | 587h | **760h** | **3.34** |

CMT 实现 **50% 时间减少**且 FID 更优。

## 个人思考

1. **"固定回归目标"** 是最简洁的解决方案：之前所有的一致性模型训练技巧（EMA 衰减、$\Delta t$ 退火、损失加权）都是为了对抗 stop-gradient 的不稳定性——CMT 直接消除了这个不稳定性来源。
2. **中间训练的灵感来自 LLM**：LLM 领域的"预训练→中间训练→微调"已被验证有效，CMT 首次将这个范式引入视觉生成 → 跨领域的思路迁移。
3. **93% 成本降低**同时达到 SOTA → 这不是微小的效率提升，而是数量级的变化——使大规模一致性模型训练变得实际可行。
4. **理论上的梯度偏差分析**清晰解释了"为什么 CMT 初始化比扩散模型初始化更好"——前者的偏差 $\mathcal{O}(\varepsilon + \Delta t^2)$ 不包含噪声项 $\sigma_t^2/\alpha_t^2$。
5. **教师质量要求低**：即使用较弱的教师采样器（如 MF-B/4），CMT 也能提供有效初始化 → 方法不依赖完美的教师。
