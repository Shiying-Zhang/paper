---
title: "VeriFree: Reinforcing General Reasoning without Verifiers"
authors: "Xiangxin Zhou, Zichen Liu, Anya Sims, Haonan Wang, Tianyu Pang, Chongxuan Li, Liang Wang, Min Lin, Chao Du"
year: 2025
venue: "arXiv 2025"
tags: [LLM推理, 强化学习]
rating: 4
paper_url: "https://arxiv.org/abs/2505.21493"
code_url: "https://github.com/sail-sg/VeriFree"
summary: "无需验证器的通用推理 RL：用模型对参考答案的置信度替代二元验证奖励，方差更低，MMLU-Pro 提升 16.3%"
---

## 核心思想

DeepSeek-R1-Zero 风格的 RL 训练依赖**验证器**（rule-based 或 model-based）提供奖励信号，但通用推理（如 MMLU、GPQA）没有可靠的验证器。

**VeriFree** 的核心思路：将验证器奖励替换为模型**对参考答案的生成置信度**——如果推理链 $z$ 导向高置信度的正确答案，那么 $z$ 就是好的推理。

## 方法详解

### 1. 从验证器目标到 VeriFree

#### 1.1 原始验证器目标

标准 RL 目标是最大化期望奖励：

$$\mathcal{J}(\theta) = \mathbb{E}_{z \sim \pi_\theta(z|x), y \sim \pi_\theta(y|x,z)}[\mathbf{1}(y \in Y_x)]$$

其策略梯度需要：(1) 采样推理链 $z$，(2) 采样答案 $y$，(3) 用验证器检查 $\mathbf{1}(y \in Y_x)$。步骤 (3) 在通用推理中不可行。

#### 1.2 VeriFree 变换

**关键假设**：$|Y_x| = 1$，即正确答案唯一，记为 $y^*$。

**边际化推导**：将答案采样解析积分掉：

$$\mathcal{J}(\theta) = \mathbb{E}_{z \sim \pi_\theta(z|x)}\left[\sum_y \pi_\theta(y|x,z) \cdot \mathbf{1}(y = y^*)\right] = \mathbb{E}_{z \sim \pi_\theta(z|x)}[\pi_\theta(y^*|x,z)]$$

不再需要采样 $y$ 再验证，直接计算模型对 $y^*$ 的**条件概率**作为连续奖励。

**等价性证明**：在 $|Y_x|=1$ 下，$\mathbb{E}_y[\mathbf{1}(y=y^*)] = \pi_\theta(y^*|x,z)$——精确等价，无近似。

### 2. 梯度估计

对 $\mathcal{J}(\theta)$ 求梯度，利用 $\nabla_\theta [\pi_\theta(z|x) \cdot \pi_\theta(y^*|x,z)] = \nabla_\theta \pi_\theta(z, y^*|x)$：

$$\nabla_\theta \mathcal{J} = \mathbb{E}_{z \sim \pi_\theta(z|x)}\left[\pi_\theta(y^*|x,z) \nabla_\theta \log \pi_\theta(z|x) + \nabla_\theta \log \pi_\theta(y^*|x,z) \cdot \pi_\theta(y^*|x,z)\right]$$

整理后得到两项：

$$\nabla_\theta \mathcal{L}_\text{VeriFree} = \underbrace{\mathbb{E}_z\left[\pi_\theta(y^*|x,z) \cdot \nabla_\theta \log \pi_\theta(z|x)\right]}_{\text{推理项：策略梯度，权重=答案置信度}} + \underbrace{\mathbb{E}_z\left[\pi_\theta(y^*|x,z) \cdot \nabla_\theta \log \pi_\theta(y^*|x,z)\right]}_{\text{答案项：监督学习，权重=答案置信度}} \tag{7}$$

**推理项的直觉**：高置信度的推理链（$\pi_\theta(y^*|x,z)$ 大）获得更大梯度权重——强化"想对了"的推理路径。

**答案项的直觉**：在好的推理链上加强对正确答案的生成能力——类似于条件 SFT，但权重由模型自身置信度决定（而非均匀加权）。

**与 JLB/LaTRO 的关键区别**：它们在答案项上使用均匀权重，可能强化与低质量推理链配对的答案——实际上是在不相关的推理上训练答案生成。

实际实现中结合 **RLOO**（Leave-One-Out）方差减少：

$$\text{baseline}_k = \frac{1}{K-1}\sum_{j \neq k} \pi_\theta(y^*|x,z_j)$$

以及**响应长度归一化**防止长回复被系统性低估。

### 3. 方差减少

**定理 1**：VeriFree 估计通过 Rao-Blackwellization（解析边际化答案采样）实现**严格更低**的方差——因为减少了一个随机变量。

### 4. Token 级实现细节

**关键工程问题**：推理链 $z$ 和答案 $y$ 的分界如何确定？

**错误做法**：文本级别切分（在生成文本中找 `<answer`）→ tokenization 不一致

**正确做法**：在 token 序列中找对应 `<answer`（不含 `>`）的 token 位置进行切分 → 保证采样和优化阶段的 tokenization 一致。

### 5. 与相关方法的对比

| 方法 | 奖励信号 | 答案项权重 | 目标 |
|------|---------|-----------|------|
| VeriFree | 概率（连续） | 按答案置信度加权 | 等价于原始目标 |
| JLB | log-概率 | 均匀权重 | 不同的下界 |
| LaTRO | 固定参考策略 | 均匀权重 | 不同的下界 |

**区别**：VeriFree 精确恢复原始目标，而 JLB/LaTRO 优化的是不同的（更松的）下界。

## 实验结果

### MMLU-Pro

| 模型 | 基线 | VeriFree | 提升 |
|------|------|---------|------|
| Qwen3-1.7B | 33.3% | **46.9%** | +13.6 |
| Qwen3-4B | 47.2% | **63.5%** | **+16.3** |
| Qwen3-8B | 59.8% | **67.2%** | +7.4 |

### SuperGPQA

| 模型 | 基线 | VeriFree | 提升 |
|------|------|---------|------|
| Qwen3-1.7B | 17.4% | **24.8%** | +7.4 |
| Qwen3-4B | 24.7% | **35.1%** | +10.4 |
| Qwen3-8B | 31.0% | **38.0%** | +7.0 |

### 训练效率

VeriFree 收敛更快、训练步数更少，且最终准确率高于 model-based verifier 方法。

### 置信度与准确率相关性

模型置信度与准确率改善的相关系数 $\rho = 0.82$——置信度是推理能力的良好代理。

### 消融实验（Qwen3-1.7B）

| 配置 | 效果 |
|------|------|
| 文本切分 vs **Token 切分** | Token 切分避免优化不稳定 |
| 去掉 RLOO | 最终准确率下降 >3%，提前收敛 |
| 加入等价类 | 小幅改善 ~2% |

### 跨域迁移

仅在**非数学数据**上训练仍能迁移到数学基准——VeriFree 学到的是**通用推理能力**而非特定任务模式。

## 个人思考

1. **"置信度即奖励"**的核心思路简洁有力：不需要外部验证器，模型自身对答案的确信程度就是奖励信号。
2. **方差证明**（Rao-Blackwellization）提供了理论保证：不只是启发式地替换奖励，而是证明了方差严格更低。
3. **Token 级切分的工程细节**看似微小但至关重要：tokenization 不一致会导致整个训练崩溃。
4. **跨域迁移**证明了方法的通用性：在非数学数据上训练也能提升数学推理，说明学到了通用的"思考能力"。
5. **局限性**：单一正确答案假设在开放式生成任务中可能不成立——如何扩展到多正确答案场景是未来方向。
