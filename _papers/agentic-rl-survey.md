---
title: "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey"
authors: "Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Heng Ji, Jun Wang, Shuicheng Yan, Philip Torr, et al."
year: 2025
venue: "Transactions on Machine Learning Research 2025"
tags: [综述, 智能体, 强化学习]
rating: 5
paper_url: "https://arxiv.org/abs/2509.02547"
code_url: ""
summary: "智能体 RL 综述：从 PBRFT（单步 MDP）到 Agentic RL（多步 POMDP）的范式转变，六大能力分类（规划/工具/记忆/推理/自改进/感知），涵盖 500+ 篇文献"
---

## 核心思想

传统 LLM 强化学习（RLHF/DPO）把 LLM 当作**被动的序列生成器** → 单步 MDP，一问一答。

**Agentic RL** 是范式转变：LLM 是**自主决策的智能体**，嵌入复杂动态环境 → 多步 POMDP，持续交互。

本综述综合 **500+ 篇文献**，建立了 Agentic RL 的理论基础和系统分类。

## 背景知识

### PBRFT vs Agentic RL 的理论区分

| 维度 | PBRFT（传统 RLHF） | Agentic RL |
|------|-------------------|-----------|
| 数学模型 | 退化单步 MDP | **POMDP**（部分可观测马尔可夫决策过程） |
| 时间跨度 | $T = 1$（单步） | **$T \gg 1$（多步）** |
| 状态转移 | 固定（提示→回复→终止） | **动态（环境反馈→新状态）** |
| 可观测性 | 完全可观测 | **部分可观测** |
| 奖励 | 即时偏好反馈 | **延迟、稀疏、环境奖励** |

**形式化**：

PBRFT：$\langle \mathcal{S}_{\text{trad}}, \mathcal{A}_{\text{trad}}, \mathcal{P}_{\text{trad}}, \mathcal{R}_{\text{trad}}, T=1, \gamma=1 \rangle$

Agentic RL：$\langle \mathcal{S}_{\text{agent}}, \mathcal{A}_{\text{agent}}, \mathcal{P}_{\text{agent}}, \mathcal{R}_{\text{agent}}, \gamma, \mathcal{O} \rangle$

## 六大智能体能力

### 1. 规划（Planning）

| 方法 | 思路 | 代表工作 |
|------|------|---------|
| 外部引导 | RL 训练价值函数引导经典搜索（MCTS） | RAP, LATS |
| 内部优化 | RL 直接优化 LLM 的规划策略 | VOYAGER, ETO |
| 前瞻性 | 将搜索过程内化为快思考 | — |

### 2. 工具使用（Tool Use）

**三阶段演化**：

| 阶段 | 方法 | 局限 |
|------|------|------|
| ReAct 风格 | 提示工程/SFT | 静态模式复制 |
| 工具集成 RL | RL 学习何时、如何调用工具 | 单步决策 |
| **长程 TIR** | **跨多轮推理的工具策略** | **信用分配困难** |

代表工作：ToolRL, ReTool, AutoTIR, VTool-R1, DeepEyes

**理论贡献**：Lin & Xu (2025) 证明工具集成推理**从理论上扩展了纯文本 RL 的能力边界**。

### 3. 记忆（Memory）

| 类别 | 方法 | 代表工作 |
|------|------|---------|
| RAG 风格 | 外部存储 + 学习检索策略 | Memory-R1, Prospect |
| Token 级 | 显式记忆 token + RL 管理 | MemAgent, MEM1, ReSum |
| 结构化记忆 | 知识图谱、层级表示 | Zep, A-MEM, Mem0 |

### 4. 自改进（Self-Improvement）

- 语言反思：RL 优化反思轨迹
- 参数内化：将反思转化为参数更新
- 迭代自训练：合成数据自我进化
- 元进化：反思能力本身的进化

### 5. 推理（Reasoning）

| 模式 | 特点 | 代表 |
|------|------|------|
| 快思考 | 直觉、高效、单次推理 | 标准 LLM |
| 慢思考 | 审慎、显式中间步骤 | o1/R1 风格 |
| **混合** | **自适应计算分配** | **前沿方向** |

### 6. 感知（Perception）

- 视觉锚定：主动视觉认知
- 工具驱动：搜索、缩放、裁剪
- 生成驱动：请求描述
- 音频模态

## RL 算法对比

### PPO 家族

VAPO, LitePPO, PF-PPO, VinePPO, PSGPO

### DPO 家族

β-DPO, SimPO, IPO, KTO, ORPO, Step-DPO, LCPO

### GRPO 家族（20+ 变体）

DAPO, GSPO, GMPO, ProRL, Dr.GRPO, Step-GRPO, SRPO, GRESO, StarPO, GHPO, ASPO, TreePO, EDGE-GRPO, CHORD, PAPO

**关键发现**：GRPO 的组相对优势估计比 PPO 的 critic 网络更具**样本效率**。

## 应用领域

| 领域 | 代表任务 |
|------|---------|
| 搜索/研究智能体 | 互联网搜索、深度研究 |
| 代码智能体 | 代码生成、迭代修复、自动化软件工程 |
| 数学智能体 | 非形式/形式推理 |
| GUI 智能体 | 图形界面操作 |
| 视觉智能体 | 图像、视频、3D 理解 |
| 具身智能体 | 导航、操作（VLA） |
| 多智能体系统 | 协调、端到端 MARL |

## 环境与框架

### 主要训练框架

| 框架 | 定位 |
|------|------|
| VeRL | 大规模 LLM RL |
| SkyRL | 长上下文 RL |
| ROLL | 通用 RL rollout |
| EARL | 动态并行度优化 |

### 商业系统

OpenAI o3, DeepSeek-R1, Kimi K2, Qwen QwQ-32B, Microsoft rStar2-Agent, Meituan LongCat

## 开放挑战

### 1. 可信度

- 安全漏洞：智能体系统的攻击面更大
- 幻觉：多步推理中的事实性保证
- 迎合性与奖励 hack

### 2. 扩展性

- 计算需求：智能体训练比 RLHF 贵得多
- 缩放定律：模型/数据/计算的最优配比
- 训练效率

### 3. 机制理解

- RL 在 LLM 中究竟改变了什么？
- 数学推理能力的涌现机制

### 4. 部署架构

- 护栏和安全模式
- 人在回路验证
- 多智能体通信协议

## 个人思考

1. **PBRFT → Agentic RL 的形式化**是本综述最重要的贡献：将直觉上的"从被动到主动"严格定义为"单步 MDP → 多步 POMDP"→ 为整个领域提供了统一的理论语言。
2. **GRPO 家族的 20+ 变体**反映了该领域的火热程度——但也意味着"哪种 GRPO 变体最好"可能取决于具体任务，缺乏统一的最优选择。
3. **工具集成推理的理论证明**（Lin & Xu, 2025）是里程碑：纯文本 RL 有理论能力上限，工具调用**从根本上扩展了可解决问题的边界**。
4. **奖励密度 vs 稀疏性**是 Agentic RL 的核心矛盾：过程奖励更有效但需要标注，结果奖励易获取但信用分配困难 → 如何自动生成过程奖励是关键开放问题。
5. **"快思考+慢思考"的自适应混合**是最令人期待的方向：不是所有问题都需要 CoT → 让模型自己决定"何时深思、何时直觉"是通往通用智能体的关键一步。
