---
title: "Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement"
authors: "Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, William Yang Wang"
year: 2024
venue: "arXiv 2024"
tags: [LLM Agent, 自我改进]
rating: 4
paper_url: "https://arxiv.org/abs/2410.04444"
code_url: "https://github.com/Arvid-pku/Godel_Agent"
summary: "提出自引用递归自我改进框架：LLM Agent 通过 monkey patching 在运行时修改自身代码，同时改进策略 π 和学习算法 I"
---

## 核心思想

传统 Agent 有固定策略或固定优化算法。本文提出 **Gödel Agent**：一个能够**自引用地修改自身代码**的 LLM Agent 框架，包括修改自己的策略 $\pi$ **和**学习算法 $I$。

**与 meta-learning 的区别**：Meta-learning Agent 只更新策略 $\pi$，但学习算法 $I$ 是固定的。Gödel Agent 两者都可以更新，实现真正的递归自我改进。

## 方法详解

### 1. 形式化框架

#### 1.1 Meta-Learning Agent（先前方法）

$$\pi_{t+1} = I(\pi_t, r_t), \quad r_t = U(\mathcal{E}, \pi_t) \tag{Meta}$$

$I$ 是固定的学习算法，只更新策略 $\pi$。

#### 1.2 Gödel Agent（本文）

$$\boxed{\pi_{t+1}, I_{t+1} = I_t(\pi_t, I_t, r_t, g)}, \quad r_t = U(\mathcal{E}, \pi_t) \tag{Gödel}$$

其中：
- $\mathcal{E} \in \mathcal{S}$：环境状态
- $\pi \in \Pi$：Agent 策略
- $U: \mathcal{S} \times \Pi \to \mathbb{R}$：效用函数
- $g \in \mathcal{G}$：高层优化目标
- $I_0 = (f_0, o_0)$：初始学习算法（决策函数 + 辅助组件）

**关键区别**：$I_t$ 以**自身**为输入并产生**自身的更新版本** $I_{t+1}$，实现自引用。

### 2. 初始配置

#### 2.1 初始策略 $\pi_0$
如 chain-of-thought prompting 等。

#### 2.2 动作集 $\mathcal{A}$

四个核心能力：
1. **`self_inspect`**：读取当前算法状态（Python 局部/全局变量）
2. **`interact`**：通过 $U$ 评估性能
3. **`self_update`**：修改 $(\pi_t, I_t)$ 生成 $(\pi_{t+1}, I_{t+1})$
4. **`continue_improve`**：递归调用决策函数 $f$（实现递归自我改进）

#### 2.3 决策函数 $f_0$
确定动作序列 $a_1, \ldots, a_n \in \mathcal{A}$。

### 3. 实现机制

#### 3.1 自感知（Self-Awareness）
运行时内存检查 Python 局部/全局变量，实现对 Agent 代码结构的内省。

#### 3.2 自修改（Self-Modification）
通过 **monkey patching** 动态生成代码并写入运行时内存。采用**递归函数设计**（而非传统迭代循环），使主循环本身也可以被更新。

#### 3.3 环境交互
验证集反馈提供性能指标用于决策。

### 4. 算法伪代码

```
输入：π₀, f₀, 目标 g, 环境 E, 效用函数 U, SELF_INSPECT
输出：优化后的 π, agent s

s ← SELF_INSPECT()
r ← U(E, π₀)
π, s ← SELF_IMPROVE(π, s, r, g)

函数 SELF_IMPROVE(E, π, s, r, g):
    a₁,...,aₙ ← f₀(π, s, r, g)
    对每个 aᵢ:
        π, s, r ← EXECUTE(E, π, s, r, aᵢ)
    返回 π, s

函数 EXECUTE(E, π, s, r, a):
    switch a.name:
        case self_state:  s ← SELF_INSPECT()
        case interact:    r ← U(E, π)
        case self_update: π, s ← a.code  // monkey patching
        case continue_improve: π, s ← SELF_IMPROVE(E, π, s, r, g)  // 递归!
    返回 π, s, r
```

**递归的关键作用**：`continue_improve` 调用 `SELF_IMPROVE`，而 `SELF_IMPROVE` 中的 $f_0$ 可能已被 `self_update` 修改 → 学习算法本身在进化。

### 5. 辅助机制

| 机制 | 描述 | 对 MGSM 的贡献 |
|------|------|---------------|
| **Thinking before acting** | 推迟推理提高决策质量 | −13.4%（去掉后） |
| **Error handling** | 从执行失败中恢复 | −14.8%（去掉后） |
| **Code execution** | Python/Bash 执行 | −7.1% |
| **External LLM** | 外部 LLM API 调用 | −3.8% |

思考和错误恢复是最关键的辅助机制。

## 实验结果

### 基准测试（GPT-3.5-turbo）

| 任务 | DROP (F1) | MGSM (Acc%) | MMLU (Acc%) | GPQA (Acc%) |
|------|-----------|-------------|-------------|-------------|
| **Gödel-base** | 80.9±0.8 | 64.2±3.4 | 70.9±3.1 | 34.9±3.3 |
| Meta Agent Search | 79.4±0.8 | 53.4±3.5 | 69.6±3.2 | 34.6±3.2 |
| **Gödel-free** | 90.5±1.8 | 90.6±2.0 | 87.9±2.2 | 55.7±3.1 |

- MGSM 改善最大（+10.8%）
- Gödel-free 自发调用 GPT-4o，展示了自主升级能力
- 收敛速度快于 Meta Agent Search

### 鲁棒性分析（100 次优化试验，MGSM）

- 意外终止：4%
- 临时性能下降：92%
- 最终优化失败：14%
- 通过错误处理成功恢复：~86%

### Game of 24 案例研究

**算法切换**：6 次 LLM 尝试失败后，Agent 自主将方法替换为搜索算法，达到 **100%** 准确率。

**持续精化**：迭代改进包括代码辅助验证（+10%）、额外数据尝试（+15%）、详细错误分析、并行化和冗余去除。

### 发现的策略

| 任务 | Agent 自主发现的策略 |
|------|-------------------|
| DROP | 多专家角色（推理 + 数学 + 历史）→ Self-Consistency 投票 |
| MGSM | 逐步验证 + CoT Self-Consistency |
| MMLU | Few-shot + Self-Consistency + 知识专家角色 |
| GPQA | 多样化 CoT-SC + 科学教授角色提示 |

## 个人思考

1. **自引用的优雅性**：$I_t$ 以自身为输入产生 $I_{t+1}$，真正实现了 Gödel 机的精神——系统审视并改进自身。
2. **递归 vs 迭代**的设计选择很关键：递归函数允许主循环被修改，而迭代循环中的循环体无法自我修改。
3. **Game of 24 的算法切换**最令人印象深刻：Agent 从 LLM prompting 自主切换到搜索算法，展示了真正的"工具发明"能力。
4. **局限性**：无法微调底层 LLM 权重；随着复杂度增长，系统可能越来越难以自我修改（"可理解性天花板"）。
5. **与 DGM/HGM 的关系**：Gödel Agent 是单 Agent 自改进，而 DGM/HGM 是群体演化——两种范式互补。
