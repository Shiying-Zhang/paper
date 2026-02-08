---
title: "Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation"
authors: "Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, Maneesh Agrawala, Ion Stoica, Kurt Keutzer, Song Han"
year: 2025
venue: "NeurIPS 2025"
tags: [视频生成, 注意力机制]
rating: 5
paper_url: "https://arxiv.org/abs/2506.19852"
code_url: "https://github.com/mit-han-lab/radial-attention"
summary: "发现时空能量衰减现象，设计 O(n log n) 静态稀疏注意力 mask，长视频生成 4× 加速 + 88% 稀疏率，质量几乎无损"
---

## 核心思想

本文发现视频 DiT 中的一个关键现象——**时空能量衰减**（Spatiotemporal Energy Decay）：注意力分数随空间和时间距离的增加而**指数衰减**，类似于物理信号衰减。基于此设计 **Radial Attention**：一种静态稀疏注意力机制，将复杂度从 $O(n^2)$ 降至 $O(n \log n)$。

## 方法详解

### 1. 能量衰减模型

对于位于帧 $i_0$、空间位置 $k_0$ 的 query token，其与帧 $j$、位置 $l$ 处 key token 的注意力分数上界为：

$$p_{js+l} \leq C_\text{rel} \cdot e^{-\alpha|j-i_0| - \beta|l-k_0|} \cdot p_{i_0 s + k_0} \tag{3}$$

其中 $\alpha$ 控制**时间衰减率**，$\beta$ 控制**空间衰减率**。

**含义**：距离越远的 token，注意力权重指数级递减 → 远处 token 可以安全地稀疏化。

### 2. 标准注意力与稀疏注意力

标准注意力：
$$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V \tag{1}$$

稀疏注意力：
$$\text{SparseAttention}(Q,K,V) = \text{softmax}((QK^T + M)/\sqrt{d})V \tag{2}$$

其中 $M \in \{-\infty, 0\}^{n \times n}$ 为稀疏 mask，$-\infty$ 表示屏蔽。

### 3. Radial Attention Mask 构造

4D mask $\tilde{M} \in \{-\infty, 0\}^{f \times f \times s \times s}$（$f$ 为帧数，$s$ 为每帧 token 数）：

$$\tilde{M}_{i,j,k,l} = \begin{cases} 0 & \text{if } 2^{\lfloor \log_2 \max(|i-j|,1) \rfloor} \leq s \text{ and } |k-l|+1 \leq \frac{s}{2^{\lfloor \log_2 \max(|i-j|,1) \rfloor}} \\ 0 & \text{if } |i-j| \bmod \lceil 2^{\lfloor \log_2 \max(|i-j|,1) \rfloor}/s \rceil = 0 \text{ and } k=l \\ -\infty & \text{otherwise} \end{cases} \tag{4}$$

**直觉理解**：

#### 3.1 时间维度密度衰减

帧 $i$ 与帧 $j$ 之间的计算密度：$(1/2)^{\lfloor \log_2(\max(|i-j|,1)) \rfloor}$

将注意力组织为对角带：
- **中心带**（band 0，相邻帧）：100% 计算密度
- **外层带**：指数递减——50%、25%、12.5%...

#### 3.2 空间维度密度衰减

帧 $i$ 与帧 $j$ 之间的对角宽度：$\lfloor s / 2^{\lfloor \log_2 \max(|i-j|,1) \rfloor} \rfloor$

当对角宽度 < 1 时，只保留满足采样条件的对角线。

### 4. 复杂度分析

mask 中零元素数量的上界：

$$\#\text{zeros} \leq 4s^2 f \cdot \lfloor \log_2 f \rfloor = 4sn(\log_2 n - \log_2 s) \tag{5-6}$$

达到 $O(n \log n)$ 复杂度。

三个组成部分：
1. 中心带 + sink：$4s^2 f$
2. 对角宽度 ≥ 1 的带：$4s^2 f \cdot \lfloor \log_2 s \rfloor$
3. 对角宽度 < 1 的带（子采样）：$(\lfloor \log_2 f \rfloor - \lfloor \log_2 s \rfloor) \cdot 4s^2 f$

### 5. 误差界

$\ell_1$ 注意力误差：

$$\|\tilde{p} - p\|_1 \leq C_\text{rel}\left[\frac{8 e^{-\beta(s/2+1)}}{(1-e^{-\alpha})(1-e^{-\beta})} + \frac{4(1+e^{-\beta})}{1-e^{-\beta}} \cdot \frac{e^{-\alpha(s+1)}}{1-e^{-\alpha}}\right] = O(C_\text{rel} \cdot e^{-\min(\beta/2, \alpha) \cdot s}) \tag{7}$$

**含义**：误差随帧数 $s$ 和衰减率**指数下降**——理论上保证质量。

### 6. 实现细节

- 前 2 个 DiT block 保持 dense attention（开销可忽略：2/40-60 blocks）
- 前 12 个去噪时间步使用 dense attention（warmup）
- 128×128 block-level 稀疏粒度（适配硬件）
- 训练用 FlashAttention-2 Block-Sparse，推理用 FlashInfer

### 7. LoRA 微调

- 对 Q, K, V, output projections 应用 LoRA (rank=128)
- 支持与现有 style LoRA 权重合并
- 实现高效的长度扩展（4× 更长视频）

## 实验结果

### 默认视频长度性能

| 模型 | 方法 | PSNR | SSIM | LPIPS↓ | 加速 |
|------|------|------|------|--------|------|
| HunyuanVideo (117帧) | Original | — | — | 0.141 | — |
| | STA | 26.7 | 0.866 | 0.167 | 2.29× |
| | SVG | 27.2 | 0.895 | 0.114 | 1.90× |
| | **Radial** | **27.3** | 0.886 | **0.114** | **1.88×** |

### 4× 长视频生成

| 模型 | 训练加速 | 推理加速 | Vision Reward | 稀疏率 |
|------|---------|---------|-------------|--------|
| HunyuanVideo (509帧) | 4.37× | 3.71× | 0.134 | **88.3%** |
| Mochi 1 (667帧) | 2.83× | 2.57× | 0.113 | 85.5% |
| Wan2.1-14B (161帧) | 1.93× | 2.01× | 0.145 | 73.6% |

### 消融实验

#### Dense Attention 层数

| Dense 层数 | Vision Reward |
|-----------|--------------|
| 0 | 0.139 |
| 1 | 0.156 |
| **2** | **0.163** |
| 3 | 0.157 |

2 层 dense 最优。

#### Attention MSE 对比

| 方法 | Attention MSE |
|------|-------------|
| STA | 1.5×10⁻² |
| SVG | 4.4×10⁻³ |
| **Radial** | **3.9×10⁻³** |

## 个人思考

1. **能量衰减的物理直觉**非常自然：视频中远处的时空信息确实应该贡献更少的注意力。
2. **静态 mask**是关键优势：SVG 需要运行时分析确定 mask，Radial 的 mask 是预定义的，消除了 profiling 开销且支持训练。
3. **$O(n \log n)$ 复杂度**的对数因子来自时间维度的二进制分带——优雅的数学结构。
4. **88% 稀疏率**意味着只计算 12% 的注意力，但质量几乎无损——视频注意力的冗余度惊人。
5. **与 LoRA 的配合**使得长度扩展变得实用：先用 Radial Attention 节省内存，再用 LoRA 适配新长度。
