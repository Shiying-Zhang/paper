---
title: "Guda: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning"
authors: "Naoki Murata, Yuhta Takida, Chieh-Hsin Lai, Toshimitsu Uesaka, Bac Nguyen, Stefano Ermon, Yuki Mitsufuji"
year: 2025
venue: "arXiv 2025"
tags: [扩散模型, 数据归因, 机器遗忘]
rating: 4
paper_url: "https://arxiv.org/abs/2601.22651"
code_url: ""
summary: "基于机器遗忘的扩散模型组级数据归因：用遗忘近似 Leave-One-Group-Out 反事实，ELBO 差值评分，比 LOGO 快 ~100×，Top-1 准确率 72.7%"
---

## 核心思想

当扩散模型生成了一张图像时，我们想知道：**训练数据中哪一组数据对这张图像的生成影响最大？**

这是一个**反事实问题**——"如果训练时去掉这组数据，模型的行为会怎样变化？"

**黄金标准**：Leave-One-Group-Out（LOGO）——每次去掉一组数据重新训练，直接对比。但 $N$ 组数据需要训练 $N+1$ 个模型 → 计算成本极高。

**Guda** 提出：用**机器遗忘**近似反事实重训练：
1. 不需要从头重新训练，只需要**遗忘**特定数据组
2. 用 ELBO 差值衡量遗忘前后的行为变化
3. 比 LOGO 快 **~100×**

## 背景知识

### 实例级 vs 组级归因

| 类型 | 问题 | 方法 |
|------|------|------|
| 实例级 | "哪张训练图最相关？" | 影响函数、TRAK |
| **组级** | **"哪组训练数据最相关？"** | **LOGO、Guda** |

**为什么不能简单求和？** 组的影响是**非线性的**——一组中的图像共享特征表示，简单累加实例级归因分数无法捕获这种**协同效应**。

### 什么是机器遗忘

让模型"忘记"特定训练数据的影响，**而不影响其他数据上的表现**：

$$\theta^{\text{ul}}_{-k} \approx \theta^{\text{logo}}_{-k}$$

- $\theta^{\text{ul}}_{-k}$：遗忘第 $k$ 组后的模型（便宜）
- $\theta^{\text{logo}}_{-k}$：去掉第 $k$ 组重新训练的模型（昂贵）

遗忘只需要 ~20 个 epoch，完全重训练需要 ~2400 个 epoch → **1/120 的计算量**。

### ELBO 与似然的关系

扩散模型的精确似然 $\log p_\theta(x_0)$ 难以计算，但**证据下界（ELBO）** 是可计算的近似：

$$\text{ELBO}(x_0 | c; \theta) \le \log p_\theta(x_0 | c)$$

ELBO 越高，模型对该样本的"生成倾向"越强。

## 方法详解

### 1. LOGO 归因（黄金标准）

$$\text{LOGOA}_k(x_0, c) = \text{ELBO}(x_0 | c; \theta^{\text{full}}) - \text{ELBO}(x_0 | c; \theta^{\text{logo}}_{-k})$$

含义：去掉第 $k$ 组数据后，模型对 $x_0$ 的生成能力下降了多少？下降越多 → 这组数据越重要。

### 2. Guda 归因（遗忘近似）

$$\text{GUDA}_k(x_0, c) = \text{ELBO}(x_0 | c; \theta^{\text{full}}) - \text{ELBO}(x_0 | c; \theta^{\text{ul}}_{-k})$$

用遗忘模型 $\theta^{\text{ul}}_{-k}$ 替代重训练模型 $\theta^{\text{logo}}_{-k}$。

### 3. 遗忘目标函数

$$\mathcal{L}_{\text{unlearn}} = \mathcal{L}_{\text{forget}} + \lambda_{\text{pres}} \cdot \mathcal{L}_{\text{preserve}}$$

#### 保持损失（防止灾难性遗忘）

$$\mathcal{L}_{\text{preserve}} = \mathbb{E}\left[\|\epsilon_\theta(x_t, t, c) - \epsilon_{\theta^{\text{full}}}(x_t, t, c)\|_2^2\right]$$

在**保留集**上做分数匹配 → 确保遗忘目标组时不破坏其他数据的生成能力。

### 4. Guda-U：无条件生成

基于 **ReTrack** 的遗忘损失：

$$\mathcal{L}_{\text{forget}}^{(U)} = \mathbb{E}\left[\|\epsilon_\theta(x_t, t) - \bar{\epsilon}_t(x_t)\|_2^2\right]$$

其中目标噪声用**重要性加权**的保留集样本构建：

$$\bar{\epsilon}_t(x_t) = \sum_r w_t(x_t; x_0^{(r)}) \cdot \frac{x_t - \sqrt{\bar{\alpha}_t} x_0^{(r)}}{\sigma_t}$$

$$w_t(x_t; x_0^{(r)}) \propto q_t(x_t | x_0^{(r)})$$

将去噪轨迹**重定向**到保留集的样本上 → 模型"忘记"目标组。

K 近邻截断提高计算效率。

### 5. Guda-C：条件文本到图像生成

**问题**：直接用 ReTrack 会失败——因为去掉风格 $k$ 后，包含风格 $k$ 的提示词在保留集中**没有对应内容**。

**解决方案**：加权风格选择（WSS）锚点——保留内容描述，替换风格描述：

$$\mathcal{L}_{\text{forget}}^{(C)} = \mathbb{E}\left[\|\epsilon_\theta(x_t, t, c_f) - \epsilon_{\theta^{\text{full}}}(x_t, t, c_a)\|_2^2\right]$$

其中 $c_a$ 从锚点分布 $\mathcal{A}_{\text{WSS}}(c_f)$ 中采样——用 CLIP 加权的风格替换。

## 实验结果

### CIFAR-10 无条件生成（10 类，2048 生成样本）

| 方法 | Top-1↑ | MRR↑ | NDCG@3↑ | 总时间 |
|------|--------|------|---------|--------|
| **Guda** | **0.727** | **0.798** | **0.677** | **2h 02m** |
| Guda (ESD) | 0.619 | 0.732 | 0.634 | 1h 33m |
| CLIPA | 0.662 | 0.755 | 0.646 | ~2m |
| DAS | 0.716 | 0.794 | 0.675 | 35h 24m |
| D-TRAK | 0.609 | 0.731 | 0.639 | 30h 30m |
| TRAK | 0.118 | 0.313 | 0.317 | 30h 58m |
| LOGOA (oracle) | — | — | — | **207h 47m** |

**关键发现**：
- Guda 比 LOGO 快 **~100×**（2h vs 208h）
- ReTrack 遗忘（0.727）优于 ESD 遗忘（0.619）
- 语义相似度方法 CLIPA（0.662）不够 → 反事实因果推理更有效
- 梯度方法（D-TRAK, TRAK）表现不佳

### UnlearnCanvas 条件文本到图像（16 种风格，SD 1.5）

| 方法 | Top-1↑ | MRR↑ | NDCG@3↑ | 总时间 |
|------|--------|------|---------|--------|
| **Guda** | **0.456** | **0.582** | **0.734** | **8h 54m** |
| CLIPA | 0.338 | 0.467 | 0.672 | ~2m |
| Wang et al. | 0.047 | 0.214 | 0.588 | 158h 33m |
| LOGOA (oracle) | — | — | — | 46h 08m |

### 计算成本分析

| 阶段 | LOGO | Guda |
|------|------|------|
| 预处理（重训练/遗忘） | 206h 52m | 1h 07m |
| 查询评估 | 55m | 55m |
| **总计** | **207h 47m** | **2h 02m** |

遗忘用 ~20 epoch vs 重训练 ~2400 epoch → **1/120 的训练步数**。

## 个人思考

1. **"遗忘 ≈ 反事实"** 是巧妙的洞察：不需要真的"从来没见过这些数据"，只需要"假装忘了" → 行为变化近似反事实变化。
2. **组级归因的非线性性**是关键理论贡献：简单累加实例级分数（D-TRAK Top-1 仅 0.609）远不如反事实方法（Guda 0.727）→ 证明了组效应的非可加性。
3. **WSS 锚点策略**解决了条件生成中的分布偏移问题——去掉一种风格后，该风格的提示词变成了"域外"输入，需要替换风格描述才能有意义地重定向。
4. **ELBO 差值评分**直接测量生成倾向变化，比嵌入空间相似度（CLIPA）更能捕获因果影响 → 因果 > 相关。
5. **实际应用**：可用于版权检测（"这张生成图像受了哪组训练数据的影响？"）和数据审计（"哪些数据贡献最大？"）。
