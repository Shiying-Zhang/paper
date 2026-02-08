---
title: "NEO: From Pixels to Words – Towards Native Vision-Language Primitives at Scale"
authors: "Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, Ziwei Liu"
year: 2025
venue: "arXiv 2025"
tags: [多模态, Transformer]
rating: 4
paper_url: "https://arxiv.org/abs/2510.14979"
code_url: "https://github.com/EvolvingLMMs-Lab/NEO"
summary: "原生视觉语言模型：轻量卷积 patch embedding + Pre-Buffer 层 + Native-RoPE + 混合注意力掩码，仅 390M 图文数据从零训练视觉能力"
---

## 核心思想

现有多模态模型（如 LLaVA）是模块化拼接（CLIP 视觉编码器 + MLP + LLM），NEO 追求**原生统一**的视觉语言模型——视觉感知不依赖外部编码器，而是在共享的 Transformer 中从零学习。

三个设计原则：
1. 在共享语义空间中实现**像素-文字对齐**
2. **整合**视觉和语言模块的优势
3. 支持统一编码和推理的**原生跨模态特性**

## 方法详解

### 1. Patch 和 Word Embedding

图像通过轻量卷积层转为 token 序列（无需外部视觉编码器）：

$$\mathbf{x}_v = \text{Conv}_2(\text{GELU}(\text{Conv}_1(I)) + \text{PE})$$

- Conv1 步长 16，Conv2 步长 2 → 对应 32×32 的 patch
- $\mathbf{x}_v \in \mathbb{R}^{(h \times w) \times d}$：视觉 token
- PE：2D 正弦位置编码

文本：$\mathbf{x}_t = \text{Tokenizer}(T) \in \mathbb{R}^{n \times d}$

### 2. 多头原生注意力（MHNA）

扩展 Query/Key 的头维度以编码空间关系：

- 解耦高度 (H)、宽度 (W)、时间 (T) 三个维度
- 比标准 Transformer block 增加约 **10%** 参数
- H/W 通道的 K 权重**零初始化** → 渐进激活多模态能力，保留预训练 LLM 的行为

### 3. Native-RoPE

为不同维度设计**独立的基频率**：

$$\Theta_T = \{\beta_T^{-2k/d} \mid k \in [0, d/2)\}$$

$$\Theta_H = \{\beta_H^{-4i/d} \mid i \in [0, d/4)\}$$

$$\Theta_W = \{\beta_W^{-4j/d} \mid j \in [0, d/4)\}$$

频率参数：$\beta_T = 10^6$（时间），$\beta_H = \beta_W = 10^4$（空间）。

**索引分配策略**：
- **文本 token**：保留 T 索引，H/W 置零
- **图像 token**：T 恒定，每个位置有唯一 H/W
- **视频 token**：每帧递增 T
- **多模态**：T 索引跨模态连续递增

**与其他 RoPE 变体的对比**：

| 变体 | 性能 |
|------|------|
| 1D-RoPE | 39.1% |
| IL-RoPE | 39.5% |
| M-RoPE | 41.7% |
| MM-RoPE | 42.4% |
| Video-RoPE | 43.2% |
| **Native-RoPE** | **44.0%** |

Native-RoPE 比标准 1D-RoPE 提升 **4.9%**。

### 4. 混合注意力掩码

- **文本 token**：标准因果注意力（自回归）
- **图像 token**：双向注意力（类似视觉编码器）
- 通过 FlexAttention 高效实现

### 5. Pre-Buffer 架构

模型分为两部分：
- **Pre-Buffer**（前 12 层 / NEO-2.2B 或前 6 层 / NEO-9B）：包含原生多模态组件（MHNA + Native-RoPE + 混合掩码）
- **Post-LLM**：标准 LLM 层

**训练策略**：
- 预训练阶段：冻结 LLM 权重，仅训练 Pre-Buffer → 保护语言知识
- 中间训练后：解冻全部，统一端到端训练 → 网络自主分配编码/对齐/推理能力

### 6. 三阶段训练

#### Stage 1：预训练（345M 样本）

| 数据 | 规模 |
|------|------|
| LAION-400M（英文+中文） | 120M |
| COYO-700M | 150M |
| BLIP3o 长描述 | 20M |
| OpenImages 重标注 | 5M |
| LAION-COCO (OCR) | 30M |
| Wukong (OCR) | 20M |

配置：学习率 $8 \times 10^{-4}$，batch 2560，190K 步。文本:多模态比例 3:7。

#### Stage 2：中间训练（40M 样本）

- 全模型端到端训练
- 学习率 $4 \times 10^{-5}$，batch 1200，50K 步
- 分辨率 256²-2048²

#### Stage 3：SFT（4M 样本）

- 双语指令数据：VQA、对话、数学、知识推理、OCR、定位
- 学习率 $5 \times 10^{-5}$，batch 650，6K 步

## 实验结果

### NEO-2.2B vs 同规模模型

| 基准 | NEO-2.2B | Qwen2-VL | Mono-InternVL |
|------|----------|----------|---------------|
| MMMU | **48.6** | 41.1 | 33.7 |
| MMBench | **76.0** | 74.9 | 65.5 |
| MMVet | 49.6 | 49.5 | 40.1 |
| AI2D | **80.1** | 74.7 | 67.4 |
| DocVQA | 89.9 | 90.1 | 81.7 |
| ChartQA | **81.2** | 73.5 | 72.2 |

### NEO-9B vs 同规模模型

| 基准 | NEO-9B | Qwen2-VL-7B | InternVL2.5-7B |
|------|--------|-------------|----------------|
| MMMU | 54.6 | 54.1 | 56.0 |
| MMBench | 82.1 | 83.0 | 84.6 |
| MMStar | 62.4 | 60.7 | 64.4 |

### Pre-Buffer vs 外部视觉编码器

| 视觉编码器 | 相对性能 |
|-----------|---------|
| CLIP ViT-large | -2.4% |
| SigLIP-so400m | -1.7% |
| InternViT-300M | -2.5% |
| **NEO Pre-Buffer** | **基准** |

Pre-Buffer 优于所有外部编码器——说明原生训练的视觉能力超过了模块化拼接。

## 个人思考

1. **原生 vs 模块化**的路线之争：NEO 证明仅用 390M 图文数据就能从零训练出竞争性的视觉能力——不需要 CLIP。
2. **Native-RoPE 的 4.9% 提升**说明位置编码的模态感知设计至关重要——空间和时间需要不同的频率范围。
3. **Pre-Buffer 的可复用性**是工程亮点：一次预训练，后续研究只需微调 Post-LLM 部分。
4. **零初始化 H/W 通道**是渐进训练的关键——确保初始行为与纯 LLM 一致，避免灾难性遗忘。
5. **知识密集型任务仍有差距**（MMMU 54.6 vs InternVL2.5 56.0），可能需要更大规模的预训练数据。
