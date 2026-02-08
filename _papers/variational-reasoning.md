---
title: "Variational Reasoning for Language Models"
authors: "Xiangxin Zhou, Zichen Liu, Haonan Wang, Chao Du, Min Lin, Chongxuan Li, Liang Wang, Tianyu Pang"
year: 2025
venue: "arXiv 2025"
tags: [LLM推理, 强化学习]
rating: 5
paper_url: "https://arxiv.org/abs/2509.22637"
code_url: "https://github.com/sail-sg/variational-reasoning"
summary: "将思维链视为隐变量、用变分推断优化：ELBO + IWAE 多轨迹界 + 前向 KL 后验训练，统一解释 RFT/RL/GRPO 的偏差来源"
---

## 核心思想

将推理过程（思维链 $z$）视为**隐变量**，用**变分推断**框架优化。核心贡献：

1. 推导 ELBO 下界和 IWAE 多轨迹紧界
2. 用前向 KL 训练变分后验，避免模式坍缩
3. 统一解释 RFT、Binary-RL、GRPO 的偏差来源
4. 在多个推理基准上超越现有方法

## 方法详解

### 1. 概率框架

#### 1.1 符号定义

- $\pi_\theta(z, y | x)$：推理模型，生成思维链 $z$ 和答案 $y$
- $P_\theta(y | x) = \sum_z \pi_\theta(z, y | x)$：边际答案分布
- $q_\phi(z | x, y')$：变分后验，给定问题和答案提示 $y'$ 预测思维链
- $Y_x$：正确答案集合

#### 1.2 ELBO 推导

$$\log P_\theta(Y_x | x) \geq \mathbb{E}_{q_\phi(y')} \left[\mathbb{E}_{q_\phi(z|x,y')} [\log \pi_\theta(Y_x | x, z)] - D_\text{KL}(q_\phi(z|x,y') \| \pi_\theta(z|x))\right] \equiv \mathcal{L}_\text{ELBO} \tag{2}$$

**含义**：ELBO 由两部分组成：
- **重建项** $\mathbb{E}[\log \pi_\theta(Y_x | x, z)]$：思维链 $z$ 应导向正确答案
- **正则项** $D_\text{KL}(q_\phi \| \pi_\theta)$：变分后验不应偏离先验太远

**ELBO 与真实似然的关系**：

$$\mathcal{L}_\text{ELBO} = \log P_\theta(Y_x | x) - D_\text{KL}(q_\phi(z|x,y') \| P_\theta(z|x,Y_x)) \tag{4}$$

其中真实后验 $P_\theta(z|x,Y_x) = \pi_\theta(Y_x|x,z)\pi_\theta(z|x) / P_\theta(Y_x|x)$。

### 2. IWAE 多轨迹扩展

采样 $K$ 条思维链以收紧下界：

$$\mathcal{L}_\text{ELBO}^K = \mathbb{E}_{z_{1:K} \sim q_\phi}\left[\log \frac{1}{K} \sum_{k=1}^K \frac{\pi_\theta(z_k, Y_x | x)}{q_\phi(z_k | x, y')}\right] \tag{5}$$

**性质**：$\mathcal{L}_\text{ELBO}^K \leq \mathcal{L}_\text{ELBO}^{K+1} \leq \log P_\theta(Y_x | x)$，轨迹越多，界越紧。

#### 2.1 梯度估计

$$\nabla_\theta \mathcal{L}_\text{ELBO}^K = \mathbb{E}_{z_{1:K} \sim q_\phi}\left[\sum_k \tilde{\rho}_k \nabla_\theta \log \pi_\theta(z_k, Y_x | x)\right] \tag{6}$$

$$\tilde{\rho}_k = \frac{\rho_k}{\sum_j \rho_j}, \quad \rho_k = \frac{\pi_\theta(z_k, Y_x | x)}{q_\phi(z_k | x, y')} \tag{重要性权重}$$

**直觉**：每条思维链按其重要性权重 $\tilde{\rho}_k$ 加权贡献梯度——好的思维链（高似然、低后验概率）获得更大权重。

#### 2.2 权重估计

$$\rho_k^\text{est} = \left(\frac{\pi_\theta(z_k | x)}{q_\phi(z_k | x, y')}\right)^{1/|z_k|} \cdot \mathbb{E}_{y \sim \pi_\theta(y|x,z_k)}[\mathbf{1}(y \in Y_x)] \tag{8}$$

两个关键设计：
- **几何平均** $(1/|z_k|)$：对序列长度归一化，避免长序列的方差爆炸
- **准确率估计**：用采样准确率代替精确似然

**定理 1**：当 $\pi_\theta(Y_x|x,z) \geq 1/|Y_x|$ 时，准确率估计的最坏方差 ≤ 似然估计的最坏方差。

### 3. 前向 KL 训练变分后验

$$\nabla_\phi \mathcal{L}_\text{forward}^M \simeq \mathbb{E}_{z_{1:M} \sim \pi_\theta(z|x)}\left[\sum_m \tilde{w}_m \nabla_\phi \log q_\phi(z_m | x, y')\right] \tag{9}$$

$$\tilde{w}_m = \frac{w_m}{\sum_j w_j}, \quad w_m = \mathbb{E}_{y \sim \pi_\theta(y|x,z_m)}[\mathbf{1}(y \in Y_x)]$$

**为什么用前向 KL 而非反向 KL**：反向 KL 会导致模式坍缩——当后验 $q_\phi$ 训练不足时，它会集中在一个模式上忽略其他好的思维链。前向 KL 鼓励 $q_\phi$ 覆盖所有高质量思维链。

### 4. 统一解释现有方法

#### 4.1 RFT（拒绝采样微调）

$$\nabla_\theta \mathcal{L}_\text{RFT} \approx -P_\text{ref}(Y_x|x) \cdot \nabla_\theta D_\text{KL}(P_\text{ref}(z|x,Y_x) \| \pi_\theta(z|x)) \tag{10}$$

**偏差来源**：梯度被 $P_\text{ref}(Y_x|x)$（参考模型准确率）加权 → **偏向简单问题**。

#### 4.2 Binary-RL

$$\nabla_\theta \mathcal{L}_\text{bi-RL} \approx -P_\theta^{sg}(Y_x|x) \cdot \nabla_\theta D_\text{KL}(P_\theta^{sg}(z|x,Y_x) \| \pi_\theta(z|x)) \tag{11}$$

同样隐式地按准确率加权，偏向简单问题。

#### 4.3 GRPO

$$\nabla_\theta \mathcal{L}_\text{bi-GRPO} \approx -\sqrt{\frac{P_\theta^{sg}(Y_x|x)}{1 - P_\theta^{sg}(Y_x|x)}} \cdot \nabla_\theta D_\text{KL}(P_\theta^{sg}(z|x,Y_x) \| \pi_\theta(z|x)) \tag{12}$$

权重函数随准确率单调递增 → **仍然偏向简单问题**。

**变分推理的优势**：公式 (9) 中所有问题的权重更均匀，不会系统性偏向简单问题。

### 5. 训练算法

```
输入：初始模型 π_θ₀, 后验 q_ϕ, 数据集 {x, y*}
for t = 1 to T:
    # 更新后验 q_ϕ（前向 KL）
    for s = 1 to S_ϕ:
        采样 z₁:M ~ π_θ(z|x)
        计算 w_m = E[1(y ∈ Y_x)]（准确率）
        ϕ ← ϕ + η∑ w̃_m ∇ log q_ϕ(z_m|x,y')
    # 更新模型 π_θ（IWAE）
    for s = 1 to S_θ:
        采样 z₁:K ~ q_ϕ(z|x,y')
        计算 ρ_k^est（几何平均 + 准确率）
        θ ← θ + η∑ ρ̃_k ∇ log π_θ(z_k, Y_x|x)
return θ_T, ϕ_T
```

## 实验结果

### Qwen3-4B-Base

| 方法 | MATH500 | AIME24 | AIME25 | AMC23 | Overall |
|------|---------|--------|--------|-------|---------|
| Base | 45.30 | 4.79 | 5.73 | 27.73 | 21.38 |
| Bespoke-Stratos | 84.70 | 27.29 | 24.17 | 70.16 | 51.35 |
| **Ours-Acc** | **88.30** | **31.67** | **27.29** | **75.63** | **55.72** |

### Qwen3-8B-Base

| 方法 | MATH500 | AIME24 | AIME25 | LCB-M | LCB-H |
|------|---------|--------|--------|-------|-------|
| Bespoke-Stratos | 89.70 | 39.58 | 28.85 | 36.89 | 7.11 |
| **Ours-Acc** | **91.80** | **45.63** | **31.98** | **49.33** | **13.21** |

LiveCodeBench-Medium 提升 12.44pp，Hard 提升 6.10pp。

### Qwen2.5-32B-Instruct

| 方法 | MATH500 | AIME24 | AIME25 | GPQA-D |
|------|---------|--------|--------|--------|
| Bespoke-Stratos | 92.60 | 55.42 | 46.88 | 57.57 |
| RLT | 93.50 | 56.77 | 47.19 | 59.09 |
| **Ours-Acc** | **93.50** | **58.85** | **50.31** | **60.73** |

### 消融：答案提示 $y'$ 的效果

| 配置 | MATH500 | AIME24 | GPQA-D |
|------|---------|--------|--------|
| 无 $y'$ | 81.20 | 23.44 | 40.53 |
| **有 $y'$** | **88.30** | **31.67** | **45.33** |

去掉答案提示损失 7+ 百分点——$y'$ 防止了快捷推理。

## 个人思考

1. **将思维链视为隐变量**的概率框架非常优雅：ELBO、IWAE、变分后验这些经典工具自然地适用于推理优化。
2. **统一解释 RFT/RL/GRPO 的偏差**是重要的理论贡献：它们都隐式地偏向简单问题，而变分推理提供了更均匀的权重。
3. **前向 KL 训练后验**的动机清晰：反向 KL 会坍缩到单一模式，前向 KL 保证覆盖——这对多样化推理路径至关重要。
4. **几何平均归一化** $(\cdot)^{1/|z_k|}$ 是关键工程细节：长思维链的概率积极小，不归一化会导致方差爆炸。
5. **训练动态更稳定**（更少的梯度范数尖峰）——变分框架自带的正则效果。
