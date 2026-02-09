---
title: "reAR: Rethinking Visual Autoregressive Models via Generator-Tokenizer Consistency Regularization"
authors: "Qiyuan He, Yicong Li, Haotian Ye, Jinghao Wang, Xinyao Liao, Pheng-Ann Heng, Stefano Ermon, James Zou, Angela Yao"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, 自回归模型, 正则化]
rating: 4
paper_url: "https://arxiv.org/abs/2510.04450"
code_url: ""
summary: "视觉自回归模型的生成器-分词器一致性正则化：噪声上下文增强 + codebook 嵌入对齐，FID 3.02→1.86（461M 参数），即插即用兼容任意分词器"
---

## 核心思想

视觉自回归（AR）模型为什么不如扩散模型？本文发现关键瓶颈是**生成器-分词器不一致性**——AR 模型生成的 token 序列无法被分词器的解码器有效还原为图像。

两个不一致来源：
1. **暴露偏差放大**：训练时看真实 token，推理时看自己的预测 → 错误**级联放大**
2. **嵌入无感知**：AR 模型只优化离散 token ID，不知道 token 在嵌入空间中的关系

**reAR** 提出两个互补的正则化：
1. **噪声上下文正则化** → 训练时暴露于不完美输入 → 减少暴露偏差
2. **Codebook 嵌入正则化** → 让隐藏层特征与分词器嵌入对齐 → 建立嵌入感知

结果：FID **3.02 → 1.86**（461M 参数），即插即用。

## 背景知识

### 视觉 AR 生成的流程

1. **分词器编码**：图像 → 离散 token 序列（codebook 索引）
2. **AR 模型**：自回归预测下一个 token
3. **分词器解码**：token 序列 → codebook 嵌入 → 图像

### 为什么暴露偏差在视觉 AR 中更严重

| 模态 | 语言 | 视觉 |
|------|------|------|
| 错误传播 | 局部影响（一个错词） | **全局影响（一个错 token → 整张图结构错误）** |
| 错误类型 | 语义偏差 | **空间结构破坏** |
| 纠错能力 | 后续 token 可修正 | **后续 token 进一步偏离** |

### 嵌入无感知的问题

| 预测 | 与正确 token 的关系 | 解码效果 |
|------|-------------------|---------|
| Token A（正确） | — | 正确图像 |
| Token B（错误但嵌入接近 A） | 嵌入距离小 | **视觉差异小** |
| Token C（错误且嵌入远离 A） | 嵌入距离大 | **视觉差异大** |

标准 AR 训练无法区分 B 和 C——**都算"错误"** → 浪费了"近似正确"的信号。

## 方法详解

### 1. 噪声上下文正则化

训练时对输入 token 加随机噪声：

$$\tilde{x}_i = (1 - b_i) x_i + b_i u_i$$

- $b_i \sim \text{Bernoulli}(\epsilon)$：以概率 $\epsilon$ 替换
- $u_i \sim \text{Uniform}(\{1, \ldots, K\})$：随机 codebook 索引
- $\epsilon \sim U(0, f(t))$：噪声率随训练退火

**退火策略**：

$$f(t) = \max\left(0, 1 - \frac{4}{3}t\right)$$

训练初期噪声大（暴露鲁棒性）→ 训练后期噪声小（精确学习）。

**损失函数**：

$$\mathcal{L}'_{\text{AR}}(\theta; t) = -\mathbb{E}_{\tilde{x} \sim q_\epsilon(\cdot|x)} \sum_{i=1}^{N} \log p_\theta(x_i | \tilde{x}_{i-1}, \ldots, \tilde{x}_1)$$

注意：**目标仍是正确 token** $x_i$，只是**上下文被噪声污染** → 学习从不完美输入中预测正确输出。

### 2. Codebook 嵌入正则化

用可训练的 MLP $h_\phi$ 将 AR 模型的隐藏层特征对齐到分词器嵌入空间：

$$\mathcal{L}_{\text{re}}(\theta, \phi; t) = \sum_{i=1}^{N-1} \left[d(h_\phi^i(\mathbf{w}_\theta^l(\tilde{x})), z_{x_i}) + d(h_\phi^i(\mathbf{w}_\theta^{l'}(\tilde{x})), z_{x_{i+1}})\right]$$

- $\mathbf{w}_\theta^l$：**浅层**（$l=0$）特征 → 对齐当前 token 嵌入
- $\mathbf{w}_\theta^{l'}$：**深层**（$l'=15/18/18$）特征 → 对齐下一个 token 嵌入
- $z_{x_i}$：分词器 codebook 中的嵌入向量
- $d(\cdot, \cdot)$：余弦距离

**直觉**：浅层编码"当前看到什么"（→ 当前 token），深层编码"将要预测什么"（→ 下一个 token）。

### 3. 组合目标

$$\mathcal{L}_{\text{reAR}}(\theta, \phi; t) = \mathcal{L}'_{\text{AR}}(\theta; t) + \lambda \mathcal{L}_{\text{re}}(\theta, \phi; t)$$

$\lambda = 1$，对权重选择不敏感（AdamW 天然归一化梯度尺度）。

### 4. 架构

- DiT 风格因果 Transformer + AdaLN 条件化
- MLP 正则化头：2 层，2048 隐藏维度（仅增加 3-4M 参数）
- 模型变体：reAR-S（201M）、reAR-B（261M）、reAR-L（461M）

## 实验结果

### ImageNet 256×256 主要结果

| 模型 | 分词器 | 参数量 | FID↓ | IS↑ |
|------|--------|--------|------|------|
| AR-L（vanilla） | Patch-VQ | 461M | 3.02 | 256.2 |
| **reAR-L** | Patch-VQ | 461M | **1.86** | **316.9** |
| LlamaGen-XXL | Patch-VQ | 1.4B | 2.34 | 253.9 |
| RAR-XL | Patch-VQ | 955M | 1.50 | 306.9 |
| DiT-XL | Patch-VAE | 675M | 2.27 | 278.2 |
| MAR-H | Patch-VAE | 943M | 1.55 | 303.7 |
| VAR-d30 | VAR | 2.0B | 1.92 | 323.1 |

reAR-L（461M）FID 1.86 超过了 LlamaGen-XXL（1.4B，FID 2.34）→ 用 **1/3 参数**达到更好效果。

### 跨分词器泛化

| 方法 | 分词器 | 参数量 | FID↓ |
|------|--------|--------|------|
| AR-AliTok-B | AliTok | 177M | 1.50 |
| RAR-B-AliTok | AliTok | 177M | 1.52 |
| **reAR-B-AliTok** | AliTok | **177M** | **1.42** |

177M 参数 FID **1.42** → 与 675M 参数的扩散模型 REPA（1.42）持平。

### 消融实验

| 组件 | FID↓ |
|------|------|
| 无噪声 + 无嵌入正则化 | 2.12 |
| 噪声退火 + 无嵌入正则化 | 2.02 |
| 无噪声 + 嵌入正则化 | 2.18 |
| **噪声退火 + 嵌入正则化** | **2.00** |

两个组件的组合效果大于各自的简单相加。

### 噪声策略对比

| 策略 | FID↓ |
|------|------|
| $\epsilon = 0$（无噪声） | 2.12 |
| $\epsilon = 0.25$（固定） | 2.08 |
| $\epsilon = 0.5$（固定） | 3.15（不稳定） |
| $\epsilon \sim U(0, 1-t)$（线性退火） | 2.02 |
| **$\epsilon \sim U(0, \max(0, 1 - \frac{4}{3}t))$** | **2.00** |

退火策略优于固定噪声；噪声过大反而有害。

### 鲁棒性验证

10% 随机 token 噪声下：
- Vanilla AR：CTR 8.3
- **reAR：CTR 9.0（+8.4%）**

reAR 确实学到了从不完美输入中恢复的能力。

## 个人思考

1. **"生成器-分词器不一致性"** 是被忽视的核心问题：之前的工作关注模型架构、分词器设计、训练策略，但没人关注**两者之间的接口** → reAR 找到了正确的瓶颈。
2. **即插即用**是最大的实用优势：不需要改分词器、不需要改推理流程 → 直接在现有 AR 管道上加正则化即可。
3. **暴露偏差在视觉 AR 中比 NLP 严重得多**：NLP 中一个错词影响有限，但视觉中一个错 token 可能破坏整张图的空间结构。
4. **"浅层对齐当前 token、深层对齐下一个 token"** 的设计有清晰的理论直觉——CKA 分析证实了这种分工确实存在于 vanilla AR 中，reAR 只是**强化了这个自然模式**。
5. **177M 参数 FID 1.42** 挑战了"视觉 AR 需要巨大模型"的观念 → 正确的正则化可以用小模型达到大模型的效果。
