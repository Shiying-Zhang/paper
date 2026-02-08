---
title: "ThinkMerge: Think in Parallel, Answer as One"
authors: "Haonan Wang, Chao Du, Kenji Kawaguchi, Tianyu Pang"
year: 2025
venue: "arXiv 2025"
tags: [LLM推理, 解码策略]
rating: 4
paper_url: "https://arxiv.org/abs/2512.02874"
code_url: ""
summary: "无训练即插即用解码策略：并行生成 K 条推理链，在同步点合并 logits 生成统一答案，代码任务 pass@1 提升 5.77%"
---

## 核心思想

**ThinkMerge** 是一种**无训练的即插即用**解码策略：

1. 并行采样 $K$ 条独立的推理链（思维过程）
2. 在分隔符（如 `</think>`）处同步
3. **合并所有推理上下文的 logits** 生成统一答案

与多数投票（Majority Voting）不同，ThinkMerge 不生成 $K$ 个独立答案再投票，而是将 $K$ 条推理链的信息**融合**到一个答案中。

## 方法详解

### 1. 两阶段流水线

#### Stage 1：多样化推理

使用高温采样（0.5-0.7）独立生成 $K$ 条推理链 $R_1, \ldots, R_K$，直到遇到分隔符 `</think>`。

#### Stage 2：集成解码

在每个答案 token 位置 $i$，聚合所有推理上下文的 logits：

$$\bar{z}_i = \frac{1}{K} \sum_{k=1}^K z_i^{(k)}$$

$$\bar{P}_\theta(y_i | Q, R_{1:K}, y_{<i}) = \text{softmax}(\bar{z}_i)[y_i]$$

**关键设计**：选出的 token $y_i$ 反馈到**所有 $K$ 个上下文**中，确保后续解码的一致性。

### 2. 设计变体

| 变体 | 描述 |
|------|------|
| **Direct-Merge (A)** | 等所有推理链完成后集成 |
| **Early-Ready (B)** | $N$ 条中 $K$ 条完成即开始答案解码（降低延迟） |
| **Trimming (C)** | 去除重复后缀模式后重新填充 |
| **Shortest-KK (D)** | 从 $N$ 条中选最短 $K$ 条（对抗"过度思考"） |

### 3. 实现方式

- **两阶段 vLLM**：批量生成 $K$ 条推理链 → 左填充对齐 → 重新填充 KV cache → 联合解码
- **单步 Flex-Attention**：用注意力掩码屏蔽填充 token，一次前向完成

### 4. 为什么是 logit 合并而非概率合并

$$\text{Logit 合并}: \quad \bar{z}_i = \frac{1}{K}\sum_k z_i^{(k)} \xrightarrow{\text{softmax}} P$$

$$\text{概率合并}: \quad \bar{P} = \frac{1}{K}\sum_k \text{softmax}(z_i^{(k)})$$

Logit 合并更好，因为它与标准采样技术（top-k, top-p）兼容——对 logits 做 top-k 裁剪后再 softmax 比在概率上做更合理。

## 实验结果

### 封闭式任务（AIME 2025）

| 模型 | 基线 (MV) | ThinkMerge | 提升 |
|------|----------|-----------|------|
| Qwen3-14B (K=8) | 74.0% | **78.0%** | +4.0 |
| R1-Distill-Qwen-7B (K=2) | 40.7% | **41.3%** | +0.6 |

### 开放式代码（LiveCodeBench v5, Pass@1）

| 模型 | 基线 | ThinkMerge | 提升 |
|------|------|-----------|------|
| DeepCoder-14B | 55.32% | **61.09%** | **+5.77** |
| Qwen3-8B | 57.14% | **59.57%** | +2.43 |

**按难度分层**：

| 难度 | DeepCoder 提升 | Qwen3-8B 提升 |
|------|--------------|--------------|
| Easy | +0.09 | −1.00 |
| Medium | +7.62 | +3.43 |
| **Hard** | **+8.28** | **+7.58** |

难题获益最大，简单题已饱和。

### 深度研究 Agent（WebSailor）

| 模型 | 基准 | 基线 | ThinkMerge | 提升 |
|------|------|------|-----------|------|
| WebSailor-32B | XbenchDeepSearch | 50.4% | **57.6%** | +7.2 |
| WebSailor-7B | XbenchDeepSearch | 37.8% | **48.0%** | +10.2 |
| WebSailor-32B | GAIA | 46.64% | **51.46%** | +4.8 |

### Logit vs 概率合并

| 方法 | AIME (Qwen3-14B, K=8) |
|------|----------------------|
| **Logit 合并** | **78.0%** |
| 概率合并 | 69.4% |

Logit 合并显著优于概率合并（+8.6%）。

## 个人思考

1. **简单但有效**：无需训练、无需修改模型，仅改变解码策略就获得显著提升——实用价值极高。
2. **"融合优于投票"**的洞察：$K$ 条推理链各有信息优势，合并 logits 比选最优答案能利用更多信息。
3. **Hard 问题受益最大**合理：简单问题单条推理链就够，难题需要多视角信息融合。
4. **Shortest-KK 变体**对抗过度思考——最短推理链往往更直接准确，长链容易引入错误。
5. **局限性**：弱模型（3B）多条推理链可能都错 → "垃圾进垃圾出"，ThinkMerge 不能修复基础能力不足的问题。
