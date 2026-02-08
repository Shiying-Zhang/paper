---
title: "KAN: Kolmogorov-Arnold Networks"
authors: "Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark"
year: 2024
venue: "arXiv 2024"
tags: [神经网络架构, 科学发现]
rating: 5
paper_url: "https://arxiv.org/abs/2404.19756"
code_url: "https://github.com/KindXiaoming/pykan"
summary: "基于 Kolmogorov-Arnold 表示定理的新型网络：可学习激活函数在边上而非节点上，B-spline 参数化实现 α=4 的缩放指数，兼具精度与可解释性"
---

## 核心思想

MLP 基于**万能逼近定理**（固定激活函数在节点上，可学习权重在边上），KAN 基于 **Kolmogorov-Arnold 表示定理**（可学习激活函数在边上，节点仅做求和）。KAN 用 B-spline 参数化边上的一元函数，在科学计算任务上实现更优的缩放律和可解释性。

## 方法详解

### 1. 数学基础：Kolmogorov-Arnold 表示定理

对任意连续函数 $f: [0,1]^n \to \mathbb{R}$，存在一元连续函数使得：

$$f(x_1, \cdots, x_n) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^n \phi_{q,p}(x_p)\right) \tag{2.1}$$

其中 $\phi_{q,p}: [0,1] \to \mathbb{R}$ 和 $\Phi_q: \mathbb{R} \to \mathbb{R}$ 都是一元函数。

**含义**：任何多元函数都可以分解为一元函数的**有限组合**——这是 Hilbert 第 13 问题的否定回答。

**局限**：原始定理中的 $\phi_{q,p}$ 可能不光滑（甚至是分形的），无法直接用于机器学习。

### 2. KAN 的泛化：从定理到网络

#### 2.1 网络定义

KAN 将公式 (2.1) 泛化为**任意深度和宽度**的网络。设网络形状为 $[n_0, n_1, \ldots, n_L]$，第 $l$ 层的变换矩阵 $\Phi_l$ 是一个 $n_{l+1} \times n_l$ 的**函数矩阵**：

$$(\Phi_l)_{j,i} = \phi_{l,j,i}, \quad \phi_{l,j,i}: \mathbb{R} \to \mathbb{R}$$

每个元素是一个可学习的一元函数（而非标量权重）。

#### 2.2 前向传播

$$x_{l+1,j} = \sum_{i=1}^{n_l} \phi_{l,j,i}(x_{l,i}) \tag{2.5}$$

**与 MLP 的关键对比**：
- **MLP**：$x_{l+1} = \sigma(W_l x_l + b_l)$——线性变换后接固定非线性
- **KAN**：每条边是一个**独立的可学习非线性函数**，节点仅做求和

#### 2.3 激活函数参数化

每个边上的函数用**残差 B-spline** 表示：

$$\phi(x) = w_b \cdot b(x) + w_s \cdot \text{spline}(x) \tag{2.10}$$

$$b(x) = \text{silu}(x) = \frac{x}{1 + e^{-x}} \tag{2.11}$$

$$\text{spline}(x) = \sum_i c_i B_i(x) \tag{2.12}$$

其中 $b(x)$ 是类似残差连接的基函数，$B_i(x)$ 是 $k$ 阶 B-spline 基函数，$c_i$ 是可学习系数。

**参数量**：对 $G$ 个网格区间、$k$ 阶 B-spline，每条边有 $G + k$ 个参数。总网络参数 $O(N^2 L (G+k))$，比 MLP 的 $O(N^2 L)$ 多 $G+k$ 倍，但 KAN 所需的 $N$ 通常远小于 MLP。

### 3. 逼近理论

#### 3.1 KAN 逼近定理（定理 2.1）

若目标函数具有光滑的组合结构 $f = f_{L-1} \circ \cdots \circ f_0$，则 KAN 的逼近误差：

$$\|f - (\Phi_{L-1}^G \circ \cdots \circ \Phi_0^G)\|_{C^m} \leq C \cdot G^{-(k+1-m)} \tag{定理 2.1}$$

$G$ 为网格大小，$k$ 为 B-spline 阶数，$m$ 为求导阶数。

**关键性质**：收敛速率 $\alpha = k+1$ **不依赖于输入维度**（当组合结构存在时）——这打破了维度诅咒。

#### 3.2 缩放律对比

| 网络类型 | 缩放指数 $\alpha$ |
|---------|-----------------|
| MLP (ReLU) | $\alpha \approx 1$ |
| 组合稀疏理论 | $\alpha = m/2$ |
| **KAN (k=3 B-spline)** | **$\alpha = 4$** |

### 4. 网格扩展

训练过程中可以动态**细化网格**而不丢失已学信息：

$$\{c'_j\} = \arg\min \mathbb{E}_x\left(\sum_j c'_j B'_j(x) - \sum_i c_i B_i(x)\right)^2 \tag{2.16}$$

新网格的 B-spline 系数通过最小二乘拟合旧函数来初始化。这实现了从粗到细的**渐进训练**：先在粗网格上快速学习大结构，再在细网格上精化细节。

### 5. 可解释性技术

#### 5.1 稀疏化

定义激活函数的 $L^1$ 范数：

$$|\phi|_1 = \frac{1}{N_p} \sum_{s=1}^{N_p} |\phi(x^{(s)})| \tag{2.17}$$

层级 $L^1$ 范数：$|\Phi_l|_1 = \sum_{i,j} |\phi_{l,j,i}|_1$

**熵正则化**（鼓励稀疏连接）：

$$S(\Phi_l) = -\sum_{i,j} \frac{|\phi_{l,j,i}|_1}{|\Phi_l|_1} \cdot \log\left(\frac{|\phi_{l,j,i}|_1}{|\Phi_l|_1}\right) \tag{2.19}$$

**总损失**：

$$\ell_\text{total} = \ell_\text{pred} + \lambda\left(\mu_1 \sum_l |\Phi_l|_1 + \mu_2 \sum_l S(\Phi_l)\right) \tag{2.20}$$

$L^1$ 项使不重要的边趋近零，熵项使重要性集中在少数边上。

#### 5.2 剪枝

节点重要性由输入输出边的最大 $L^1$ 范数衡量：

$$I_{l,i} = \max_k |\phi_{l-1,i,k}|_1, \quad O_{l,i} = \max_j |\phi_{l+1,j,i}|_1 \tag{2.21}$$

当 $I_{l,i} < \theta$ 且 $O_{l,i} < \theta$（$\theta = 10^{-2}$）时剪枝该节点。

#### 5.3 符号化

对剪枝后的 KAN，将每条边的 spline 拟合为仿射变换的符号函数：

$$y \approx c \cdot f(a \cdot x + b) + d$$

从预定义函数库（$\sin, \cos, \exp, \log, \sqrt{\cdot}$等）中选择拟合最好的。

## 实验结果

### 函数逼近

5 个测试函数上，KAN 的缩放指数 $\alpha \approx 4$，MLP $\alpha \approx 1$。例如 $f(x,y) = \exp(\sin(\pi x) + y^2)$：[2,1,1] KAN 随网格细化达到 $G^{-3} \sim G^{-4}$ 收敛。

### PDE 求解（Poisson 方程）

KAN 在 1D 和 2D Poisson 方程上优于同参数量 MLP，网格细化带来持续改善。

### 科学发现

- **纽结理论**：从数值数据中发现纽结不变量之间的关系
- **Anderson 局域化**：从数值模拟中恢复已知的缩放关系

### KAN vs MLP 总结

| 方面 | KAN | MLP |
|------|-----|-----|
| 激活位置 | 边（可学习） | 节点（固定） |
| 权重形式 | Spline 函数 | 线性矩阵 |
| 缩放指数 | $\alpha = 4$ | $\alpha \approx 1$ |
| 可解释性 | 高（可视化每条边） | 低 |
| 最佳场景 | 低维、有结构的科学问题 | 高维、大规模深度学习 |

## 个人思考

1. **数学基础的优雅**：从 Kolmogorov-Arnold 定理到实用网络的泛化路径清晰——保留核心思想（一元函数组合）但放松不光滑约束。
2. **$\alpha = 4$ 的缩放指数**意味着网格每细化一倍，误差减少 $2^4 = 16$ 倍——这是 MLP 无法匹配的精度优势。
3. **网格扩展**是训练效率的关键：先粗后细避免了从一开始就用细网格的高成本。
4. **可解释性流程**（稀疏化 → 剪枝 → 符号化）使 KAN 成为科学发现工具——自动从数据中提取数学公式。
5. **局限性诚实**：大规模深度学习（NLP、CV）上尚未验证，KAN 的优势集中在低维科学计算领域。
