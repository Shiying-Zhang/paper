---
title: "Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning"
authors: "Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, Sergey Levine"
year: 2024
venue: "arXiv 2024"
tags: [多模态, 强化学习, 决策智能体]
rating: 4
paper_url: "https://arxiv.org/abs/2405.10292"
code_url: ""
summary: "用 RL（PPO）微调 VLM 做多步决策智能体：CoT 推理 + 文本动作 → 环境执行 → 任务奖励优化，7B 模型超越 GPT-4V"
---

## 核心思想

大型视觉语言模型（VLM）擅长理解图像和对话，但能否用它们做**多步决策**（如玩游戏、操控机器人）？

本文提出用**强化学习（PPO）直接微调 VLM**，使其成为决策智能体：
1. VLM 输入观察图像 → 输出 **CoT 推理** + **文本动作**
2. 解析文本动作为环境可执行命令
3. 环境返回奖励 → 用 PPO 优化整个 VLM

**关键发现**：7B 模型经 RL 微调后，在多个任务上**超越 GPT-4V 和 Gemini**。

## 背景知识

### 为什么用 VLM 做决策

| 传统 RL 智能体 | VLM 智能体 |
|-------------|----------|
| 从零学习 | 利用预训练知识 |
| 需要百万步交互 | 可能只需少量微调 |
| 特定任务专用 | 语言驱动，通用性强 |
| 无法解释决策 | CoT 提供可解释推理 |

### 挑战：如何计算动作概率

RL（PPO）需要计算策略的**动作概率** $\pi_\theta(a_t | o_t)$。但 VLM 的输出是一段**自由文本**（包含 CoT + 动作），不是传统的离散动作。

核心问题：CoT 部分的 log-probability **远大于**动作部分，直接相加会让梯度被 CoT 主导。

## 方法详解

### 1. RL 形式化

将环境建模为 POMDP：
- **状态空间** $\mathcal{S} = \mathcal{O} \times \mathcal{V}^m$：RGB 图像 + 文本输入
- **动作空间** $\mathcal{V}^n$：文本输出
- **策略** $\pi_\theta: \mathcal{O} \times \mathcal{V}^m \to \mathcal{V}^n$：VLM 本身

### 2. 提示设计

VLM 的输入包含任务描述和合法动作空间。输出为 JSON 格式：

```json
{
    "thoughts": "我需要先移动到苹果旁边...",
    "action": "go to apple"
}
```

- `thoughts`：CoT 推理过程
- `action`：可执行的文本动作

### 3. 动作解析函数

$$f(\mathbf{v}^\text{out}) = \begin{cases} a, & \text{if "action": "a"} \in \mathbf{v}^\text{out} \\ \text{Unif}(\mathcal{A}), & \text{otherwise} \end{cases}$$

如果 VLM 无法生成合规格式 → 随机探索（保证训练不中断）。

### 4. 动作概率计算（核心创新）

VLM 输出文本 $\mathbf{v}^\text{out} = [\mathbf{v}^\text{tht}, \mathbf{v}^\text{act}]$（CoT + 动作）。朴素方法计算整个序列的 log-probability：

$$\log \pi_\theta(a_t | o_t) = \log \pi_\theta(\mathbf{v}^\text{tht}) + \log \pi_\theta(\mathbf{v}^\text{act} | \mathbf{v}^\text{tht})$$

**问题**：实验测量发现 CoT 的 log-probability 远大于动作：

| 任务 | CoT log-prob | 动作 log-prob |
|------|------------|-------------|
| NumberLine | -3.4 | 0.0 |
| EZPoints | -9.0 | 0.0 |
| Points24 | -37.6 | 0.0 |
| ALFWorld | -20.3 | -0.4 |

CoT 部分的绝对值是动作部分的 **10-90 倍**→ 梯度几乎完全来自 CoT → 动作优化被忽视。

**解决方案：缩放因子 $\lambda$**

$$\log \pi_\theta(a_t | o_t) \leftarrow \lambda \cdot \log \pi_\theta(\mathbf{v}^\text{tht} | o_t, \mathbf{v}^\text{in}) + \log \pi_\theta(\mathbf{v}^\text{act} | o_t, \mathbf{v}^\text{in}, \mathbf{v}^\text{tht})$$

$\lambda \in [0.2, 0.5]$ 衰减 CoT 的贡献，让动作梯度占据合理比例。

### 5. 完整训练流程（Algorithm 1）

```
for 每轮交互:
    1. 从观察 oₜ 构建输入提示 v^in_t = h(oₜ)
    2. VLM 生成文本 v^out_t = [v^tht_t, v^act_t]
    3. 解析提取合法动作 aₜ = f(v^out_t)
    4. 环境执行 aₜ → 返回奖励 rₜ 和下一状态 o_{t+1}
    5. 计算缩放后的 log π_θ(aₜ)
    6. 存入经验回放缓冲 Bₖ
for 每个 PPO 更新:
    从 Bₖ 采样 → 计算优势函数 → PPO 裁剪目标优化
```

### 6. 两阶段训练

**阶段 1：监督微调（SFT）**
在少量演示数据上微调 VLM，使其学会正确的输出格式（JSON + CoT）。

**阶段 2：RL 微调**
在环境中交互，用 PPO 和任务奖励优化。

## 评估任务

### Gym Cards 领域

| 任务 | 目标 | 动作空间 | 性质 |
|------|------|---------|------|
| NumberLine | 将值移到目标 | {"+", "-"} | 确定性 |
| EZPoints | 用 2 张牌凑 12 | {1-10, "+", "*", "="} | 确定性 |
| Points24 | 用 4 张牌凑 24 | 同上 | 确定性 |
| Blackjack | 不超 21 赢庄家 | {"hit", "stand"} | 随机性 |

### ALFWorld 领域

交互式虚拟家居环境，需要语义理解 + 多步规划：
- "去厨房找到苹果 → 拿起来 → 放到冰箱里"
- 需要导航、物体识别、操控

## 实验结果

### 主要结果

- RL 微调在**所有任务**上都带来改善
- 7B 模型经 RL 微调后**超越 GPT-4V 和 Gemini**
- 确定性任务和随机性任务都有提升

### CoT 的重要性

移除 CoT 推理导致**性能显著下降**——说明推理链对 RL 决策至关重要。

### 缩放因子 $\lambda$ 消融

| $\lambda$ 范围 | 效果 |
|---------------|------|
| $\lambda \to 1$（完整 CoT 权重） | 性能下降 |
| $\lambda \in [0.2, 0.5]$（最优范围） | **最佳性能** |
| $\lambda \to 0$（忽略 CoT） | 性能下降 |

$\lambda$ 与性能呈**倒 U 型**关系。

### 计算成本

约 30 小时/15K 环境步——限制了更大规模基准（如 Atari 需要 2M 步）的评估。

## 个人思考

1. **VLM + RL 的正确打开方式**：不是让 VLM 直接当传统 RL 智能体，而是利用其语言能力做 CoT 推理 + 用 RL 优化决策质量。
2. **$\lambda$ 缩放因子**是最关键的技术贡献：解决了"CoT 主导梯度"的根本问题——这个问题在所有用 RL 训练序列生成模型时都会出现。
3. **7B > GPT-4V** 说明**领域微调**的价值远超模型规模——通用大模型在特定决策任务上可能不如专精小模型。
4. **CoT 的双面性**：CoT 对决策有帮助（提供推理），但对梯度计算有害（主导概率）——$\lambda$ 优雅地平衡了两者。
5. **局限性诚实**：30 小时/15K 步的计算成本说明 VLM + RL 目前还不适合需要大量交互的任务——效率优化是关键。
