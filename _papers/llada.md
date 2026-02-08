---
title: "LLaDA: Large Language Diffusion with mAsking"
authors: "Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li"
year: 2025
venue: "arXiv 2025"
tags: [扩散模型, LLM]
rating: 5
paper_url: "https://arxiv.org/abs/2502.09992"
code_url: ""
summary: "首个 8B 规模掩码扩散语言模型，证明生成建模原则（而非自回归形式）才是 LLM 能力的根基，在推理任务上打破逆向诅咒"
---

## 核心思想

**核心论点**：LLM 的关键能力来自**生成建模原则**（最大化似然），而非自回归分解形式本身。掩码扩散模型同样满足生成建模原则，因此也能获得 LLM 的核心能力。

**LLaDA** 基于掩码扩散（Masked Diffusion）训练了 8B 参数的语言模型，在多个基准上与 LLaMA3 8B 竞争，并在**逆向推理**任务上展现出独特优势。

## 方法详解

### 1. 理论基础

#### 1.1 生成建模原则

所有生成模型的共同目标：

$$\max_\theta \mathbb{E}_{p_\text{data}(\mathbf{x})} \log p_\theta(\mathbf{x}) \iff \min_\theta D_\text{KL}(p_\text{data}(\mathbf{x}) \| p_\theta(\mathbf{x})) \tag{1}$$

自回归模型通过链式分解实现：

$$p_\theta(\mathbf{x}) = p_\theta(x^1) \prod_{i=2}^L p_\theta(x^i | x^1, \ldots, x^{i-1}) \tag{2}$$

**关键洞察**：公式 (1) 是本质，公式 (2) 只是一种特定实现。掩码扩散提供了另一种实现方式。

### 2. 掩码扩散前向过程

#### 2.1 独立掩码

每个 token 独立地以概率 $t$ 被替换为 mask token $[M]$：

$$q_{t|0}(x_t | x_0) = \prod_{i=1}^L q_{t|0}(x_t^i | x_0^i) \tag{7}$$

$$q_{t|0}(x_t^i | x_0^i) = \begin{cases} 1-t, & \text{if } x_t^i = x_0^i \\ t, & \text{if } x_t^i = [M] \end{cases} \tag{8}$$

当 $t=0$ 时，序列完全未掩码；$t=1$ 时，序列完全掩码。

#### 2.2 反向过程：从 $t$ 到 $s$ 的转移

从时间 $t$ 到 $s$（$s < t$）的反向转移核：

$$q_{s|t,0}(x_s^i | x_t^i, x_0^i) = \begin{cases} 1, & \text{if } x_t^i \neq [M] \text{ (未掩码 token 不变)} \\ \frac{1-s}{1-t} \cdot \delta(x_s^i, x_0^i) + \frac{s-t \cdot \delta(x_s^i,[M])}{1}, & \text{if } x_t^i = [M] \end{cases}$$

**直觉**：对于已掩码的 token，在时间步 $t \to s$ 中，有 $\frac{t-s}{t}$ 的概率被揭示（恢复为 $x_0^i$），有 $\frac{s}{t}$ 的概率保持掩码。

反向过程的条件分布具有**完全分解**结构——每个掩码位置的预测独立于其他掩码位置（给定未掩码 token）：

$$q_{0|t}(x_0^i | x_t) = p_\text{data}(x_0^i | x_t^{UM}) \tag{11}$$

其中 $x_t^{UM}$ 表示未掩码的 token。这是一个**时间无关**的参数化——预测仅依赖于哪些 token 被掩码，与具体时间步 $t$ 无关。

**为什么时间无关**：由公式 (8) 的结构决定——未掩码 token 在前向过程中保持不变，所以条件分布只取决于哪些位置是掩码的，不取决于 $t$ 的具体值。这极大简化了模型设计：网络不需要显式知道当前时间步。

### 3. 训练损失

#### 3.1 掩码扩散损失

$$\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\left[\frac{1}{t} \sum_{i=1}^L \mathbf{1}[x_t^i = M] \log p_\theta(x_0^i | x_t)\right] \tag{3}$$

**$1/t$ 加权的推导**：从连续时间 ELBO 出发，负对数似然的上界可以写为：

$$-\log p_\theta(x_0) \leq \int_0^1 \frac{1}{t} \mathbb{E}_{x_t}\left[\sum_i \mathbf{1}[x_t^i = M](-\log p_\theta(x_0^i | x_t))\right] dt$$

将积分离散化为采样 $t \sim \text{Uniform}(0,1)$ 即得公式 (3)。

**与 BERT/MaskGIT 的关键区别**：$1/t$ 加权项。当 $t$ 小（掩码少）时，每个掩码 token 的权重更大——直觉上，少量掩码意味着模型已有大量上下文，应该能更精确地预测，因此给予更高权重。当 $t$ 大（掩码多）时，模型缺乏上下文，单个 token 的预测误差影响较小。

**无此项的后果**：BERT 的 MLM 损失（$t$ 固定为 0.15，无 $1/t$ 权重）是启发式的，没有似然上界保证。

#### 3.2 似然上界

$$-\mathbb{E}_{p_\text{data}(x_0)}[\log p_\theta(x_0)] \leq \mathcal{L}(\theta) \tag{4}$$

这证明了掩码扩散损失是负对数似然的上界——最小化它等价于最大化似然，回到了公式 (1) 的生成建模原则。

**公式链条**：生成建模原则 (1) → 掩码扩散实现 (7-8) → 训练损失 (3) → 似然上界 (4) → 回到 (1)。整个理论是自洽的。

### 4. 监督微调 (SFT)

对 prompt $p_0$ 和 response $r_0$，仅在 response 的掩码位置计算损失：

$$\mathcal{L}_\text{SFT} = -\mathbb{E}_{t, p_0, r_0, r_t}\left[\frac{1}{t} \sum_{i=1}^{L'} \mathbf{1}[r_t^i = M] \log p_\theta(r_0^i | p_0, r_t)\right] \tag{5}$$

prompt 始终保持未掩码，只对 response 施加扩散过程。

### 5. 推理方法

#### 5.1 随机重掩码采样

```
输入：完全掩码的序列 x_1，步数 N
for t = 1 to 0, step = 1/N:
    s = t - 1/N
    预测所有掩码位置的 token
    随机重新掩码 s/t 比例的预测 token
return x_0
```

每步保留 $(1-s/t)$ 比例的最新预测，其余重新掩码等待下一步细化。

#### 5.2 低置信度重掩码

改进版本：不随机重掩码，而是选择预测**置信度最低**的 $\lfloor L(1-s) \rfloor$ 个 token 进行重掩码。高置信度预测优先保留。

### 6. 似然评估

$$-\mathbb{E}_{l, r_0, r_l}\left[\frac{L}{l} \sum_{i=1}^L \mathbf{1}[r_l^i = M] \log p_\theta(r_0^i | p_0, r_l)\right] \tag{6}$$

$l$ 从 $\{1, 2, \ldots, L\}$ 均匀采样。这允许在不生成的情况下评估给定文本的似然。

### 7. 架构

- Transformer 掩码预测器（**无因果掩码**，双向注意力）
- 标准多头注意力（非 GQA）
- RoPE 位置编码
- 与 LLaMA3 8B 的差异：FFN 维度略小（因双向注意力参数更多）

## 实验结果

### 预训练基准 (8B)

| 任务 | LLaDA 8B | LLaMA3 8B | LLaMA2 7B |
|------|----------|-----------|-----------|
| MMLU (5-shot) | 65.9 | 65.4 | 45.9 |
| GSM8K (4-shot) | **70.3** | 48.7 | 13.1 |
| Math (4-shot) | **31.4** | 16.0 | 4.3 |
| CMMLU (5-shot) | **69.9** | 50.7 | 32.5 |
| HumanEval | 35.4 | 34.8 | 12.8 |
| HumanEval-FIM | 73.8 | 73.3 | 26.9 |

数学和中文任务上显著领先。

### 指令微调基准

| 任务 | LLaDA 8B | LLaMA3 8B |
|------|----------|-----------|
| MMLU | 65.5 | 68.4 |
| GSM8K | 69.4 | 78.3 |
| ARC-C | **88.5** | 82.4 |
| HumanEval | 49.4 | 59.8 |

注意：LLaDA 仅用 SFT，LLaMA3 用了 SFT+RL 对齐。

### 逆向推理（古诗补全）

| 模型 | 正向 | 逆向 | 差距 |
|------|------|------|------|
| GPT-4o | 82.7% | 34.3% | 48.4pp |
| Qwen2.5 7B | 75.9% | 38.0% | 37.9pp |
| LLaDA 8B | 51.8% | 45.6% | **6.2pp** |

**LLaDA 有效打破了逆向诅咒**：正向和逆向性能几乎一致，而自回归模型正向远优于逆向。

### 训练配置

- 2.3 万亿 token，0.13M H800 GPU 小时
- 序列长度 4096
- SFT：450 万条指令-回复对，3 epochs

## 个人思考

1. **"生成建模原则 vs 自回归形式"** 的区分是深刻的理论贡献——将讨论从"自回归是否必要"提升到了更本质的层面。
2. **$1/t$ 加权**是与 MaskGIT 的关键区别：没有这个项就没有似然上界保证，从启发式方法变成了有理论基础的方法。
3. **逆向诅咒的消除**是双向建模的天然优势——自回归模型从左到右的固定顺序限制了逆向推理。
4. **无 RL 对齐**的限制是明显的：SFT-only 在 HumanEval 等任务上还有差距，未来加入 RL 应该能进一步提升。
5. **采样效率**是主要劣势：扩散需要多步采样（vs 自回归单次前向），但灵活的生成模式（双向、填充等）可能在特定场景补偿这一劣势。
