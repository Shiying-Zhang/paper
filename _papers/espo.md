---
title: "ESPO: Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective"
authors: "Jingyang Ou, Jiaqi Han, Minkai Xu, Shaoxuan Xu, Jianwen Xie, Stefano Ermon, Yi Wu, Chongxuan Li"
year: 2025
venue: "arXiv 2025"
tags: [扩散语言模型, 强化学习, 后训练]
rating: 5
paper_url: "https://arxiv.org/abs/2512.03759"
code_url: ""
summary: "扩散 LLM 的原则性 RL 训练：序列级策略优化 + ELBO 作似然代理 + k₂ KL 估计器，Countdown 提升 62pp，Sudoku 提升 70pp，解决 token 级分解的理论缺陷"
---

## 核心思想

将 RL（如 GRPO）应用于**扩散语言模型（dLLM）** 有一个根本问题：标准 RL 假设序列可以分解为条件 token 概率的乘积（自回归分解），但 dLLM 是**非自回归的**——它通过迭代去噪并行生成所有 token → **token 级条件概率无法定义**。

**ESPO** 提出**序列级**视角：
1. 将整个序列的生成视为**一个动作** → 避免 token 级分解
2. 用 **ELBO** 作为联合似然的可计算代理
3. 用 **$k_2$（二次）KL 估计器**替代不稳定的 $k_3$ 估计器

结果：Countdown **+62pp**（18.7% → 81.0%），Sudoku **+70pp**（15.7% → 86.0%）。

## 背景知识

### 自回归 vs 扩散语言模型的 RL 挑战

| 方面 | 自回归 LLM | 扩散 LLM |
|------|----------|---------|
| 生成方式 | 从左到右逐 token | 并行去噪 |
| 似然分解 | $\log p(y) = \sum_i \log p(y_i \| y_{<i})$ | **不可分解** |
| 重要性比率 | 直接计算 | **无法 token 级计算** |
| RL 算法 | GRPO 直接适用 | **需要重新设计** |

### 为什么 token 级分解在 dLLM 上失败

自回归模型中，$p(y_i | y_{<i})$ 只依赖**之前的** token（单向）。

但 dLLM 的去噪过程是**双向的**——每个 token 的预测依赖所有已揭示的 token，包括之前和之后的。因此：

$$\log p_\theta(y) \ne \sum_i \log p_\theta(y_i | y_{\text{unmasked}})$$

ELBO 的逐 token 分解项**缺乏条件概率的概率解释**。

### 之前的启发式方法

| 方法 | 做法 | 问题 |
|------|------|------|
| 平均场代理 | 假设 token 独立 | 忽略 token 间依赖 |
| Token 级 ELBO 分解 | 逐 token 计算 ELBO 项 | 缺乏理论依据，不稳定 |
| **ESPO** | **序列级 ELBO** | **有理论保证** |

## 方法详解

### 1. 序列级策略优化

**核心思想**：不在 token 级别计算重要性比率，而是在**整个序列级别**。

**序列级重要性比率**：

$$\rho_{\text{seq}}^{(i)} = \exp\left(\frac{1}{L}\left(\mathcal{L}_\theta(y^{(i)} | x) - \mathcal{L}_{\theta_{\text{old}}}(y^{(i)} | x)\right)\right)$$

- $\mathcal{L}_\theta$：ELBO
- 除以序列长度 $L$ 防止大 ELBO 差值导致的指数不稳定

### 2. ELBO 作为似然代理

**离散 ELBO 变体**（低方差）：

$$\mathcal{L}'_\theta(y | x) = \mathbb{E}_l\left[\frac{L}{l} \sum_i \mathbb{1}[y_l^i = M] \log p_\theta(y^i | y_l, x)\right]$$

- $l$：掩码数量
- $\mathbb{1}[y_l^i = M]$：仅对被掩码的位置计算
- $\frac{L}{l}$：归一化因子

**理论保证**：ELBO 是联合似然的变分下界 $\mathcal{L}_\theta \le \log \pi_\theta$，保持数学严格性。

### 3. 鲁棒 KL 散度估计

标准的 $k_3$ 估计器：

$$\hat{\text{KL}}_{k_3} = \exp\left(\mathcal{L}_\theta(y | x) - \mathcal{L}_{\text{ref}}(y | x)\right) - \left(\mathcal{L}_\theta(y | x) - \mathcal{L}_{\text{ref}}(y | x)\right) - 1$$

**问题**：指数项导致数值不稳定。

**$k_2$（二次）估计器**：

$$\hat{\text{KL}}_{k_2} = \frac{1}{2}\left(\mathcal{L}_\theta(y^{(i)} | x) - \mathcal{L}_{\text{ref}}(y^{(i)} | x)\right)^2$$

- 避免指数项 → 数值稳定
- 保持无偏梯度信号

### 4. 稳定化技术

- **每 token 归一化**：ELBO 差值除以 $L$ → 防止长序列的大差值
- **对称采样**：共享掩码模式 → 降低方差
- **耦合采样**：互补掩码 → 进一步降低方差

### 5. 计算效率

总 FLOPs = $2ND(K + 6\mu M)$

- $N$：参数量，$D$：序列长度
- $K$：采样步数，$\mu$：策略更新次数，$M$：MC 样本数
- 增加 $M$ 仅影响策略更新项 → 生成是计算瓶颈
- $M = 1 \to M = 4$：wall-clock 仅增加 47%

## 实验结果

### 数学推理（LLaDA-8B）

| 任务 | 基线 | ESPO | 提升 |
|------|------|------|------|
| GSM8K | 75.9% | **82.0%** | +6.1% |
| MATH500 | 37.0% | **39.5%** | +2.5% |

### 规划任务（最戏剧性的提升）

| 任务 | 基线 | ESPO | 提升 |
|------|------|------|------|
| Countdown | 18.7% | **81.0%** | **+62.3pp** |
| Sudoku | 15.7% | **86.0%** | **+70.3pp** |

### 代码生成（LLaDA-8B）

| 任务 | 基线 | ESPO | 提升 |
|------|------|------|------|
| HumanEval | 37.8% | **40.1%** | +2.3% |
| MBPP | 37.8% | **45.4%** | +7.6% |

### 关键消融：动作空间 × 似然近似

| 动作空间 | 似然近似 | Sudoku 成功率 |
|---------|---------|-------------|
| Token 级 | 平均场 | ~26%（差） |
| Token 级 | ELBO | 不稳定（坍缩） |
| 序列级 | 平均场 | 不足 |
| **序列级** | **ELBO (ESPO)** | **~93%（最优）** |

**序列级 + ELBO 是唯一稳定收敛的组合。**

### KL 估计器对比

| 估计器 | 行为 |
|--------|------|
| $k_1$ | 梯度信号为零 → 坍缩 |
| $k_3$ | 高度不稳定 → 性能停滞 |
| **$k_2$** | **平滑、稳定收敛** |

### Dream-7B 上的一致性

在不同模型上也展现一致的提升，尤其在规划任务上。

## 个人思考

1. **"序列即动作"** 的重新框架化是关键突破：不需要强行把 dLLM 塞进自回归的框架里——直接在序列级别操作更自然、更有理论保证。
2. **Countdown/Sudoku 的巨大提升**（+62/+70pp）说明 dLLM 在规划任务上有巨大的**未释放潜力**——基座模型的 18.7% 和 15.7% 可能是因为缺乏正确的训练信号。
3. **$k_2$ vs $k_3$ 的选择**看似技术细节，实际上决定了训练能否成功——$k_3$ 的指数项在 ELBO 差值大时会爆炸，$k_2$ 的二次形式天然有界。
4. **Token 级 ELBO 分解的理论缺陷**被严格论证：双向去噪上下文使条件概率缺乏概率解释 → 之前的启发式方法缺乏理论基础。
5. **计算效率分析**表明生成（$K$ 步采样）而非策略更新是瓶颈 → 增加 MC 样本数几乎免费 → 实际中应该多采样。
