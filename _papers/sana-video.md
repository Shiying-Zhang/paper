---
title: "SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer"
authors: "Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, et al."
year: 2025
venue: "arXiv 2025"
tags: [视频生成, Transformer]
rating: 5
paper_url: "https://arxiv.org/abs/2509.24695"
code_url: ""
summary: "用 O(N) 线性注意力替代 O(N²) 注意力 + O(D²) 常量内存 KV Cache，2B 参数实现 720p 视频生成，比 Wan-14B 快 53×，训练成本仅 MovieGen 的 1%"
---

## 核心思想

视频 DiT 的核心计算瓶颈是 $O(N^2)$ 的自注意力。本文提出 **SANA-Video**：用 **Block Linear Attention** 替代标准注意力，实现：

1. $O(N)$ 计算复杂度（推理加速 4×+）
2. $O(D^2)$ **常量内存 KV Cache**（不随序列长度增长）→ 支持分钟级长视频
3. 训练仅需 64 H100 GPU × 12 天（**MovieGen 的 1%**）

## 方法详解

### 1. 训练目标

$$\mathcal{L} = \mathbb{E}_{c, t, \mathbf{x}^0}\left[\|\mathbf{u}(\mathbf{x}^t | t, c; \theta) - \mathbf{v}(\mathbf{x})\|^2\right] \tag{1}$$

其中 $\mathbf{u}(\mathbf{x}^t | t, c; \theta)$ 为预测速度，$\mathbf{v}(\mathbf{x})$ 为目标速度。使用 **Rectified Flows** + SNR 采样器。

### 2. 线性注意力 + RoPE

#### 2.1 标准线性注意力

标准注意力：$O_i = \text{softmax}(Q_i K^T) V$，复杂度 $O(N^2 D)$。

线性注意力通过 kernel 函数 $\phi$ 近似：

$$O_i = \frac{\phi(Q_i) \left(\sum_{j=1}^N \phi(K_j)^T V_j\right)}{\phi(Q_i) \left(\sum_{j=1}^N \phi(K_j)^T\right)} \tag{基础}$$

将 $\sum \phi(K_j)^T V_j$ 预计算后，每个 token 的计算复杂度从 $O(ND)$ 降为 $O(D^2)$。

#### 2.2 RoPE 集成的关键设计

$$O_i = \frac{\text{RoPE}(\phi(Q_i)) \left(\sum_{j=1}^N \text{RoPE}(\phi(K_j))^T V_j\right)}{\phi(Q_i) \left(\sum_{j=1}^N \phi(K_j)^T\right)} \tag{2}$$

**两个关键设计选择**：
1. **RoPE 在 ReLU 之后应用**：如果 RoPE 在 $\phi$ 之前，激活函数会过滤掉位置信息
2. **分母中去掉 RoPE**：确保数值稳定性（RoPE 可能导致分母接近零）

#### 2.3 因果线性注意力——常量内存 KV Cache

$$O_i = \frac{\phi(Q_i) \left(\sum_{j=1}^{i-1} S_j + S_i\right)}{\phi(Q_i) \left(\sum_{j=1}^{i-1} \phi(K_j)^T + \phi(K_i)^T\right)} \tag{3}$$

其中 $S_j = \phi(K_j)^T V_j$ 为注意力状态。

**内存分析**：
- 累积状态：$\sum_{j=1}^{i-1} S_j \in \mathbb{R}^{D \times D}$
- 累积键：$\sum_{j=1}^{i-1} \phi(K_j)^T \in \mathbb{R}^{D \times 1}$
- **总内存**：$O(D^2)$，**与已处理的 token 数 $N$ 无关**

| 注意力类型 | 内存 | 第 N 个 token 计算 | N 个 token 总计算 |
|-----------|------|------------------|-----------------|
| 因果全注意力 | $O(ND)$ | $O(ND)$ | $O(N^2 D)$ |
| 因果局部注意力 | $O(WD)$ | $O(WD)$ | $O(NWD)$ |
| **因果线性注意力** | $O(D^2)$ | $O(D^2)$ | $O(ND^2)$ |

由于视频中 $N \gg W \gg D$，线性注意力在所有维度上占优。

### 3. 三阶段训练流程

#### Stage 1：VAE 适配

微调现有视频 VAE 到新架构：
- **480P**：Wan-VAE（f8t4c16，压缩率 16）
- **720P**：DCAE-V（f32t4c32，压缩率 128）
- 收敛：5-10K 训练步

#### Stage 2：继续预训练

从 SANA-1.6B 文本到图像模型初始化，添加：
- **3D RoPE**：旋转位置编码扩展到时空
- **1D 时序卷积**：在 Mix-FFN 中通过 shortcut 集成
- **零初始化**新层以保留预训练权重

训练进度：低分辨率（192P, 2.5s）→ 高分辨率（480P-720P, 5s）

#### Stage 3：自回归块训练

##### 3.1 单调递增 SNR 采样器

随机选择一个块，通过传播概率采样其余块的时间步，保证：

$$t_1 \leq t_2 \leq \cdots \leq t_N$$

**优势**：更小的采样空间 → 更快收敛。

##### 3.2 Self-Forcing 解决暴露偏差

训练时用 ground truth 条件，推理时用生成的内容条件——两者存在差距。Self-Forcing 在训练中生成最长 1 分钟视频，使用全局注意力（$O(D^2)$ 内存使之可行），改善时间一致性。

### 4. 块线性扩散推理算法

```
输入：去噪时间步 {t₁,...,t_T}，噪声调度器 Ψ，M 个块，模型 G_θ
输出：生成视频 X_θ

初始化：X_θ ← [], KV ← [None, None, None]

对每个块 i = 1 到 M:
    初始化 x^i_{t_T} ~ N(0, I)
    对每个步骤 j = T 到 1:
        ŷ⁰_i ← G_θ(x^i_{t_j}; t_j, KV)
        如果 j = 1:
            X_θ.append(ŷ⁰_i)
            更新 KV ← G^KV_θ(ŷ⁰_i; 0, KV)  // 累积 KV cache
        否则:
            采样 ε ~ N(0, I)
            x^i_{t_{j-1}} ← Ψ(ŷ⁰_i, ε, t_{j-1})
```

**KV Cache 三个组件**：
1. $\sum S$：累积注意力状态
2. $\sum \phi(K)^T$：累积键求和
3. Conv cache $f$：时序卷积的最后一帧

### 5. 统一 T2I/T2V/I2V

单一模型架构，通过条件变化实现多任务：
- **T2V**：标准噪声初始化
- **I2V**：第一帧噪声置零（图像条件）
- **T2I**：退化为单帧

## 实验结果

### 文本到视频性能

| 模型 | 参数 | 480P 延迟 | 加速 | VBench 总分↑ | 质量↑ | 语义↑ |
|------|------|----------|------|------------|-------|-------|
| Wan 2.1-14B | 14B | 484s | 1× | 83.69 | 85.59 | 76.11 |
| Wan 2.1-1.3B | 1.3B | 103s | 4.7× | 83.31 | 85.23 | 75.65 |
| **SANA-Video** | **2B** | **60s** | **8.0×** | **83.71** | 84.35 | **81.35** |

**720P**：SANA-Video 延迟 36s，比 Wan-14B (1897s) 快 **53×**。

### 图像到视频

| 模型 | VBench I2V 总分↑ | 语义/I2V↑ |
|------|-----------------|----------|
| Wan 2.1-14B | 86.86 | — |
| HunyuanVideo-I2V | 86.82 | — |
| **SANA-Video I2V** | **88.02** | **96.40** |

### 长视频生成

| 模型 | VBench 总分↑ | 质量↑ | 语义↑ |
|------|------------|-------|-------|
| CausVid | 81.20 | 84.05 | 69.80 |
| SkyReels-V2 | 82.67 | 84.70 | 74.53 |
| Self-Forcing | 84.31 | 85.07 | 81.28 |
| **SANA-Video** | 83.70 | 84.43 | 80.78 |

**实时长视频**：1 分钟 480P@16FPS 在 **35 秒**内生成（27 FPS 生成速度）。

### DCAE-V 视频自编码器

| 自编码器 | 压缩率 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|--------|-------|-------|--------|
| Wan2.1-VAE (f8t4c16) | 16 | 34.41 | 0.95 | 0.01 |
| Wan2.2-VAE (f16t4c48) | 21 | 35.61 | 0.96 | 0.01 |
| **DCAE-V (f32t4c32)** | **128** | 33.25 | 0.94 | 0.03 |

**鲁棒性测试**：添加高斯噪声 $\varepsilon=0.2$ 后，DCAE-V PSNR 29.34 优于 Wan2.2-VAE 的 25.94——更好地泛化到扩散模型的噪声输出。

### 训练成本对比

| 模型 | 训练成本 |
|------|---------|
| MovieGen | ~1200 H100·天 |
| OpenSora | ~120 H100·天 |
| **SANA-Video** | **12 天 × 64 H100 = 768 H100·天** |

### 边缘部署：NVFP4 量化

| 精度 | 720p 5s 视频延迟 | 加速比 |
|------|----------------|--------|
| BF16 | 71s | 1× |
| **NVFP4 (RTX 5090)** | **29s** | **2.4×** |

质量损失几乎不可感知。

### 消融实验

| 组件 | 影响 |
|------|------|
| 3D RoPE | 显著降低训练损失，改善局部特征关注 |
| 时序卷积 in Mix-FFN | 大幅改善运动性能 |
| 线性 vs 全注意力 | 480P: 2× 加速，720P: 4× 加速 |
| 单调递增 SNR 采样 | 更好的跨块一致性，更快收敛 |

## 应用场景

- **具身 AI**：机器人训练数据生成（AgiBot 数据集）
- **自动驾驶**：真实驾驶场景生成
- **游戏生成**：Minecraft 视频合成

## 个人思考

1. **线性注意力终于在视频生成中证明了自己**：$O(N^2) \to O(N)$ 的加速在视频的超长序列中尤为重要。
2. **$O(D^2)$ 常量内存 KV Cache** 是实现分钟级长视频的关键——全注意力的 $O(ND)$ 在长视频中根本不可行。
3. **RoPE 放在 ReLU 之后**的设计细节很重要：位置编码和 kernel 函数的交互需要仔细考虑。
4. **1% 的训练成本**达到可比效果，说明从 T2I 模型初始化 + 高效训练流程的巨大价值。
5. **统一 T2I/T2V/I2V**的单模型架构简洁优雅，降低了部署复杂度。
