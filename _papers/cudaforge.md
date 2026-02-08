---
title: "CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization"
authors: "Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding"
year: 2025
venue: "arXiv 2025"
tags: [LLM Agent, CUDA优化]
rating: 4
paper_url: "https://arxiv.org/abs/2511.01884"
code_url: ""
summary: "双 Agent 框架 + Nsight Compute 硬件反馈，无需训练，CUDA kernel 平均 1.77× 加速，每 kernel 仅 $0.30/26.5 分钟"
---

## 核心思想

现有 LLM 生成 CUDA kernel 时缺乏**硬件级别反馈**，只能盲目优化。本文提出 **CudaForge**：一个**免训练双 Agent 框架**，模拟人类专家工作流——Coder 生成 kernel，Judge 使用 **Nsight Compute (NCU) profiler** 分析硬件瓶颈并提供针对性优化建议。

**核心创新**：集成外部硬件反馈（GPU 规格 + NCU 指标），使 Judge 能像人类专家一样识别性能瓶颈。

## 方法详解

### 1. 双 Agent 系统

#### 1.1 Coder Agent
- 接收任务需求和反馈，生成候选 CUDA kernel
- **轻量级记忆设计**：仅保留当前轮信息，不保留完整历史
- 使用 one-shot 示例提供结构指导

#### 1.2 Judge Agent
两种工作模式：
- **纠错模式 (Correction)**：识别编译/运行时错误
- **优化模式 (Optimization)**：分析硬件指标识别瓶颈

### 2. 迭代工作流

框架执行 $N$ 轮（默认 $N=10$），每轮流程：

```
1. Coder 生成/改进候选 kernel
2. 正确性测试（编译 + 执行，容差 1e-4）
3. 若正确 → Judge 用 NCU 分析并提供优化反馈
4. 若错误 → Judge 分析错误并提供纠正反馈
5. Coder 整合反馈进入下一轮
```

### 3. 硬件反馈集成——NCU 指标选择

#### 3.1 离线指标筛选（Algorithm 1-2）

**步骤 1——Kernel 采样**：每个代表性任务生成 100 个 kernel，选择速度差异最大的 10 个。

**步骤 2——每任务 Top-20 指标**：对每个任务的 10 个 kernel，计算 NCU 指标与运行时的 Pearson 相关系数，保留绝对值最大的 Top-20。

**步骤 3——跨任务稳定选择**：合并所有任务的 Top-20 列表，计算全局相关分数：

$$S_m = \frac{1}{n} \sum_i |r_{m,i}|$$

其中 $r_{m,i}$ 为指标 $m$ 在任务 $i$ 上与运行时的 Pearson 相关系数。选择 $S_m$ 超过 75 分位数的指标，最终得到 **24 个指标**。

#### 3.2 Judge 优化流程

每轮优化中：
1. Judge 用 NCU 对生成的 kernel 进行 profiling
2. 基于推理选择 3-4 个关键指标
3. 聚焦于每轮的**单一主导瓶颈**进行优化

### 4. 关键设计选择

#### 4.1 Coder-Judge 分离的必要性

| 配置 | 正确率 | 性能 |
|------|--------|------|
| 仅纠错 | 97.6% | 1.222× |
| 仅优化 | 88.4% | — |
| **Coder + Judge（完整）** | **97.6%** | **1.677×** |

角色分离减少了认知负担，优于单 Agent 方案。

#### 4.2 精选指标 vs 全量指标

| 配置 | Median 加速 | Fast1 | 成本 |
|------|-----------|-------|------|
| 全量 NCU 指标 | 1.280× | 80% | 高 |
| **24 精选指标** | **1.322×** | **84%** | 低 |

精选指标不仅更便宜，性能还更好——减少了噪声信息。

## 实验结果

### 主要结果（RTX 6000，250 KernelBench 任务）

| 方法 | 正确率 | Median 加速 | 75% 加速 | 平均性能 | Fast1 |
|------|--------|-----------|---------|---------|-------|
| OpenAI-o3 | 57.6% | 0.390× | 1.014× | 0.680× | 31.6% |
| o3-self-refine | 90.8% | 1.012× | 1.209× | 1.107× | 55.2% |
| **CudaForge** | **97.6%** | **1.107×** | **1.592×** | **1.677×** | **70.8%** |
| CudaForge (scaled) | 100% | 1.322× | 1.736× | 1.767× | 84.0% |
| CudaForge-Scaling Up | 100% | 1.317× | 1.777× | 2.265× | 92.0% |

### 成本效率

| 指标 | CudaForge | Agentic Baseline |
|------|-----------|-----------------|
| 计算时间/kernel | **26.5 分钟** | 60 分钟 |
| API 成本/kernel | **$0.30** | $5.00 |
| GPU | 单块 RTX 6000 | 6× H100 |

### 跨 GPU 泛化

| GPU | 架构 | 加速比 |
|-----|------|-------|
| RTX 6000 (Ada) | 数据中心 | 1.767× |
| A100 (Ampere) | 数据中心 | 1.841× |
| RTX 4090 (Ada) | 桌面 | 1.327× |
| RTX 3090 (Ampere) | 桌面 | 1.320× |

所有 GPU 上均达到 100% 正确率。

### 多模型兼容性

在 OpenAI-o3、GPT-5、Claude-Sonnet-4、GPT-OSS-120B、QwQ-32B 等组合下均保持有效性。

### 案例分析（CrossEntropyLoss）

展示类似专家的优化路径：
- **Round 2**：识别 23.7% warp stall → 用 warp-level shuffle 替代共享内存 reduction → 从 1.66× 提升到 2.42×
- **Rounds 6-7**：解决寄存器压力和冗余内存访问 → 最终 **3.762× 加速**

### 测试时缩放

| 轮数 | 加速比 |
|------|-------|
| 10 | 1.767× |
| 20 | 2.049× |
| 30 | 2.271× |

性能随迭代次数单调提升，体现测试时缩放效应。

## 个人思考

1. **硬件反馈的价值**不可替代：没有 NCU 指标，LLM 只能"猜测"瓶颈；有了 profiler 数据，优化变得有的放矢。
2. **24 精选指标 > 全量指标**很有启发：LLM 的注意力有限，精炼的信息比海量数据更有效。
3. **免训练**是巨大优势：无需昂贵的 RL 训练，纯靠提示工程 + 硬件反馈就能达到优秀效果。
4. **跨 GPU 泛化**说明优化策略具有通用性——不是 overfitting 到特定硬件。
5. **与人类专家工作流的对应**：Coder 像初级工程师写代码，Judge 像高级工程师用 profiler 做 code review——分工协作效率更高。
