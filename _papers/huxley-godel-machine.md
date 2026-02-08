---
title: "Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine"
authors: "Wenyi Wang, Piotr Piękos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, Jürgen Schmidhuber"
year: 2025
venue: "arXiv 2025"
tags: [LLM Agent, 自我改进]
rating: 5
paper_url: "https://arxiv.org/abs/2510.21614"
code_url: "https://github.com/metauto-ai/HGM"
summary: "发现元生产力-性能不匹配问题，提出 Clade-Metaproductivity (CMP) 指标指导 Agent 树搜索演化，SWE-bench 达人类水平 61.4%"
---

## 核心思想

本文发现了 **Metaproductivity-Performance Mismatch (MPM)**：Agent 的即时编码任务性能**不能**很好地预测其未来自我改进的潜力。基于此，提出 **HGM (Huxley-Gödel Machine)**：使用 **Clade-level 性能聚合**（CMP）指导 Agent 演化树搜索，在 SWE-bench 上达到人类水平。

**核心洞察**：评估一个 Agent 不仅看它自身的表现，还要看它**整个后代谱系**的表现——好的"祖先"不一定自己最强，但能产生最强的后代。

## 方法详解

### 1. 形式化：自我改进作为树搜索

#### 1.1 状态-动作框架

在迭代 $t$ 时，策略 $\pi$ 从复合动作中选择：

$$a_{t+1} \sim \pi(\cdot|\mathcal{T}_t), \quad \mathcal{A}_t = \mathcal{M}_t \cup V_t$$

其中：
- $\mathcal{M}_t = \{m_a : a \in \mathcal{T}_t\}$：修改动作（从现有 Agent 生成新 Agent）
- $V_t = \{v_a : a \in \mathcal{T}_t\}$：评估动作（测试某个 Agent）
- $\mathcal{T}_t$：当前 Agent 树

#### 1.2 目标

$$J(\pi) = \mathbb{E}[U(a_\text{final})]$$

其中 $a_\text{final} = \arg\max_{a \in \mathcal{T}_B} \text{Score}_\pi(a)$，$B$ 为计算预算。

### 2. Clade-Metaproductivity (CMP)——核心定义

#### 2.1 CMP 定义

$$\boxed{\text{CMP}_\pi(\mathcal{T}, a) = \mathbb{E}_{\mathcal{T}_B \sim p_\pi(\cdot|\mathcal{T}, a)}\left[\max_{a' \in C(\mathcal{T}_B, a)} U(a')\right]}$$

其中：
- $\mathcal{T}$：Agent 树
- $C(\mathcal{T}_B, a)$：以 Agent $a$ 为根的**世系子树**（clade）
- $U$：效用函数（任务性能）

**含义**：CMP 衡量的是"从 Agent $a$ 出发，其整个后代谱系中能达到的最佳性能的期望值"。

#### 2.2 CMP 实际估计器

$$\widehat{\text{CMP}}(a) = \frac{n_\text{success}^C(a)}{n_\text{success}^C(a) + n_\text{failure}^C(a)}$$

聚合整个世系子树中所有后代的成功/失败计数。

#### 2.3 定理 1（主要理论结果）

在假设 1 下（固定评估环境、可重复试验、已知有限预算、零成本证明），访问真实 CMP 预言机足以在编码 Agent 开发场景中实现 Gödel 机。

**证明思路**：CMP 在特定 POMDP 中退化为 Q 值函数 $Q_\pi$。

### 3. 复合策略分解

与 DGM/SICA 不同，HGM **解耦**三种策略：

| 策略 | 功能 | 指导指标 |
|------|------|---------|
| **选择策略** | 何时扩展 vs 评估 | 无限臂 bandit 准则 |
| **扩展策略** | 选哪个父代修改 | CMP（世系聚合） |
| **评估策略** | 选哪个 Agent 测试 | 个体性能（非世系） |

### 4. HGM 算法细节

#### 4.1 扩展策略：Thompson Sampling + 自适应调度

从 Beta 分布中采样选择父代：

$$a^* \sim \text{TS}\left(\left\{(\tau(1+n_\text{success}^C(a)), \tau(1+n_\text{failure}^C(a))) \mid a \in \mathcal{T}_t\right\}\right)$$

其中 $\tau = \tau(t)$ 为单调递增调度器——早期偏向探索，后期偏向利用。

#### 4.2 评估策略

基于**个体**（非世系）统计的 Thompson Sampling：

$$a^* \sim \text{TS}\left(\{(\tau(1+n_\text{success}(a)), \tau(1+n_\text{failure}(a))) \mid a \in \mathcal{T}_t\}\right)$$

每次评估选中 Agent 在**单个任务**上测试（非批量），减少计算浪费。

#### 4.3 选择策略：何时扩展 vs 评估

受无限臂 bandit (UCB-Air) 启发：

$$\text{扩展当且仅当 } N_t^\alpha \geq |\mathcal{T}_t|$$

其中 $N_t$ 为累计评估次数，$\alpha \in [0,1]$ 为加宽参数（经验设为 0.6）。

#### 4.4 最终 Agent 选择

返回使后验信念 $\epsilon$-分位数最大的 Agent：

$$\arg\max_{a \in \mathcal{T}_B} I_\epsilon(1+n_\text{success}(a), 1+n_\text{failure}(a))$$

其中 $I_\epsilon$ 为正则化不完全 Beta 函数。

#### 4.5 异步实现 (HGM-Async)

- 每个 CPU 独立执行迭代
- 使用最新数据更新
- 选择策略中考虑飞行中任务

### 5. MPM 问题验证

**与经验 CMP 的相关性对比**：

| 方法 | SWE-Verified-60 (加权) | Polyglot (加权) |
|------|----------------------|----------------|
| SICA | 0.444 | 0.274 |
| DGM | 0.285 | 0.383 |
| **HGM** | **0.778** | **0.626** |

HGM 的 CMP 估计器在 SWE-Verified-60 上达到 **1.75×** 更高的加权相关性。

## 实验结果

### 自我改进效率对比

| 方法 | SWE-Verified-60 准确率 | 时间 (小时) | Polyglot 准确率 | 时间 (小时) |
|------|---------------------|------------|---------------|------------|
| SICA | 50.0% | ∞ (失败) | 25.4% | 572 |
| DGM | 53.3% | 1231 | 27.1% | 2385 |
| **HGM** | **56.7%** | **517** | **30.5%** | **347** |

HGM 相比 DGM 实现 **2.38× 加速**，且最终性能更优 (+3.4%)。

### SWE-bench 人类水平

- **HGM 最佳 Agent**：61.4% 准确率（8000 次评估）
- 进入 SWE-bench Verified 排行榜 **Top-10**

### 迁移到 SWE-bench Lite

| Agent | Filtered | Standard |
|-------|----------|----------|
| 初始祖先 | 34.8% | 44.0% |
| SWE-agent + GPT-5-mini | 39.6% | 47.6% |
| **HGM 最佳 Agent** | **40.1%** | **49.0%** |

迁移到 GPT-5：HGM Agent 在 SWE-Lite Standard 上达到 **57.0%**，展示跨模型架构的鲁棒性。

## 个人思考

1. **MPM 的发现**是最重要的贡献：即时性能 ≠ 自改进潜力，这彻底改变了 Agent 选择的思路。
2. **世系思维 (Clade Thinking)**：评估一个 Agent 要看其"家族谱系"而非个人——这与生物进化中的群体适应度概念一致。
3. **三策略解耦**使得系统可以在不同维度独立优化，比 DGM 的单一选择机制更灵活。
4. **Thompson Sampling + Beta 分布**是优雅的选择：自然处理不确定性，且调度器 $\tau(t)$ 优雅地控制探索-利用权衡。
5. **与 DGM 的对比**：HGM 更快（2.38×）且更好，关键在于 CMP 比单纯性能+后代数更好地指导搜索。
