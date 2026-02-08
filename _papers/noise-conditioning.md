---
title: "Is Noise Conditioning Necessary for Denoising Generative Models?"
authors: "Qiao Sun, Zhicheng Jiang, Hanhong Zhao, Kaiming He"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, CNN]
rating: 4
paper_url: "https://arxiv.org/abs/2502.13129"
code_url: ""
summary: "挑战噪声条件化的必要性：大多数扩散模型去掉噪声级别输入后仍能工作甚至更好，并给出理论解释"
---

## 核心思想

扩散模型的一个基本假设是：网络**必须知道当前噪声级别 $t$** 才能正确去噪。本文挑战这一假设，灵感来自**盲图像去噪**（blind denoising）——不知道噪声水平也能去噪。

**核心发现**：大多数扩散模型去掉 $t$ 输入后，表现出「优雅退化」甚至**改善**。论文给出了理论解释并提出 uEDM 达到有竞争力的 FID。

![有噪声条件 vs 去掉噪声条件](https://arxiv.org/html/2502.13129v2/extracted_media/x1.png)

## 方法详解

### 1. 扩散模型基本公式

#### 1.1 噪声数据生成

$$\mathbf{z} = a(t)\mathbf{x} + b(t)\boldsymbol{\varepsilon} \tag{1}$$

其中 $a(t), b(t)$ 为调度函数。

#### 1.2 条件训练损失

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}, \boldsymbol{\varepsilon}, t}[w(t)\|\text{NN}_\theta(\mathbf{z} | t) - r(\mathbf{x}, \boldsymbol{\varepsilon}, t)\|^2] \tag{2}$$

其中 $r(\mathbf{x}, \boldsymbol{\varepsilon}, t) = c(t)\mathbf{x} + d(t)\boldsymbol{\varepsilon}$ 为回归目标。

### 2. 有效目标分析

#### 2.1 条件有效目标

网络实际学到的是对给定 $(\mathbf{z}, t)$ 的期望目标：

$$R(\mathbf{z} | t) = \mathbb{E}_{(\mathbf{x}, \boldsymbol{\varepsilon}) \sim p(\mathbf{x}, \boldsymbol{\varepsilon} | \mathbf{z}, t)}[r(\mathbf{x}, \boldsymbol{\varepsilon}, t)] \tag{6}$$

#### 2.2 无条件有效目标

去掉 $t$ 后，网络学到的是关于 $t$ 的边际化：

$$R(\mathbf{z}) = \mathbb{E}_{t \sim p(t | \mathbf{z})}[R(\mathbf{z} | t)] \tag{8}$$

**关键问题**：$R(\mathbf{z})$ 和 $R(\mathbf{z} | t)$ 差多少？取决于后验 $p(t | \mathbf{z})$ 的集中度。

等价的损失形式：

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\|\text{NN}_\theta(\mathbf{z}) - R(\mathbf{z})\|^2] \tag{7}$$

### 3. 后验集中性（核心理论）

#### 3.1 Statement 1：后验方差

$$\boxed{\text{Var}_{t \sim p(t | \mathbf{z})}[t] \approx \frac{t^{*2}}{2d}} \tag{9}$$

其中 $t^*$ 为真实噪声级别，$d$ 为数据维度。

**含义**：维度越高，后验越集中 → 网络即使不知道 $t$，也能从 $\mathbf{z}$ 本身**推断出**近似的噪声级别。

**条件**：$1/d \ll t^*$ 且 $1/d \ll 1-t^*$（即不在极端噪声级别）。

对于 CIFAR-10（$d = 3072$），方差 $\approx t^{*2}/6144$，非常小。

#### 3.2 Statement 2：目标误差

$$E(\mathbf{z}) := \mathbb{E}_{t \sim p(t | \mathbf{z})}[\|R(\mathbf{z} | t) - R(\mathbf{z})\|^2] \approx \frac{1}{2}(1 + \sigma_d^2) \tag{10-11}$$

**含义**：单步去噪误差相对于 $\|R(\mathbf{z})\|^2 \sim d$ 可忽略不计。

**公式联系**：Statement 1 说后验集中 → Statement 2 说因此有效目标的误差小 → 去掉 $t$ 不会造成大问题。

### 4. 累积误差界（采样过程）

采样迭代：

$$\mathbf{x}_{i+1} := \kappa_i \mathbf{x}_i + \eta_i \text{NN}_\theta(\mathbf{x}_i | t_i) + \zeta_i \tilde{\boldsymbol{\varepsilon}}_i \tag{4}$$

其中 $\kappa_i$ 为状态系数，$\eta_i$ 为网络输出缩放，$\zeta_i$ 为随机噪声注入。

#### Statement 3：累积误差界

$$\|\mathbf{x}_N - \mathbf{x}'_N\| \leq \sum_{i=0}^{N-1} A_i B_i \tag{12}$$

其中：
- $A_i = \prod_{j=i+1}^{N-1}(\kappa_j + |\eta_j| L_j)$：误差放大因子（与调度有关）
- $B_i = |\eta_i| \delta_i$：单步误差（与 Statement 2 有关）
- $L_i$：$R(\cdot | t_i)$ 的 Lipschitz 常数

**不同采样器的误差界**：

| 采样器 | 误差界量级 | 去掉 $t$ 后表现 |
|--------|----------|---------------|
| DDIM (ODE) | ~3×10⁶ | 灾难性崩溃 |
| EDM (Heun) | ~10³ | 适度退化 |
| FM (Euler) | ~10² | **改善** |

**为什么 DDIM 崩溃？** 其调度设计导致 $A_i$ 极大（误差剧烈放大），而 FM 的线性调度保持 $A_i$ 较小。

### 5. 随机性的帮助

增加采样过程的随机性（DDIM 的 $\lambda$ 参数从 0 到 1）在去掉噪声条件后一致改善 FID，暗示随机性可以**取消累积误差**。

### 6. uEDM：面向无条件的改进

标准 EDM 预条件器：$c_\text{skip}(t)\mathbf{z} + c_\text{out}(t) \cdot \text{NN}_\theta(c_\text{in}(t)\mathbf{z} | t)$

uEDM 修改：$c_\text{out}(t) = 1$（常量，不随 $t$ 变化）

**理由**：当网络不知道 $t$ 时，时间相关的缩放系数变得不合理。固定 $c_\text{out} = 1$ 让网络不需要建模时间依赖的缩放。

## 实验结果

### CIFAR-10 主要结果

| 模型 | 采样器 | NFE | 有 $t$ | 无 $t$ | 变化 |
|------|--------|-----|--------|--------|------|
| DDIM | ODE | 100 | 3.99 | 40.90 | 崩溃 |
| DDIM | SDE | 1000 | 3.18 | 5.41 | 退化 |
| EDM | Heun | 35 | 1.99 | 3.36 | 适度退化 |
| **FM (1-RF)** | **Euler** | **100** | **3.01** | **2.61** | **改善** |
| **FM (1-RF)** | **Heun** | **99** | **2.87** | **2.63** | **改善** |
| uEDM | Heun | 35 | 2.04 | **2.23** | 接近有条件 |

FM 去掉噪声条件后 FID **改善**了！

### 其他数据集

| 数据集 | 模型 | 有 $t$ | 无 $t$ |
|--------|------|--------|--------|
| ImageNet 32×32 | FM | 5.15 | **4.85** (改善) |
| FFHQ 64×64 | EDM | 2.64 | 3.59 (退化) |

### CFG 兼容性

SiT-B/2 在 ImageNet 256×256 上，去掉噪声条件后，各 CFG 尺度下 FID 几乎无退化。

## 个人思考

1. **挑战基本假设**的勇气值得赞赏：「噪声级别是否必要？」这个问题看似荒谬但答案出人意料。
2. **后验集中性**的理论分析非常漂亮：高维数据的祝福——维度越高，$t$ 越容易从 $\mathbf{z}$ 推断出来。
3. **采样器影响 > 去噪能力**：DDIM 崩溃不是因为去噪差，而是因为误差放大系数大。这提示我们采样器设计比去噪精度更重要。
4. **FM 的改善**最令人惊讶：可能是因为去掉 $t$ 后模型容量更多地分配给了去噪本身。
5. **对未来的启示**：如果噪声条件不是必要的，那么扩散模型的设计空间比我们想象的更大。
