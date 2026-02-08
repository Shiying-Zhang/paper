---
title: "LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning"
authors: "Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, Chongxuan Li"
year: 2025
venue: "arXiv 2025"
tags: [扩散模型, 多模态]
rating: 4
paper_url: "https://arxiv.org/abs/2505.16933"
code_url: ""
summary: "首个纯扩散多模态语言模型：LLaDA-8B + SigLIP2 视觉编码器 + MLP 连接器，在知识推理任务上展现更优的数据扩展性"
---

## 核心思想

**LLaDA-V** 是首个**纯扩散驱动**的多模态语言模型——将 LLaDA（掩码扩散语言模型）与视觉编码器结合。尽管语言基座（LLaDA-8B）弱于 LLaMA3-8B，LLaDA-V 在多模态基准上表现竞争力，且在知识和推理任务上展现出**更优的数据扩展性**。

## 方法详解

### 1. 架构组件

| 组件 | 选择 |
|------|------|
| 视觉编码器 | SigLIP2 (so400m-patch14-384) |
| 语言模型 | LLaDA-8B-Instruct（掩码扩散模型） |
| 连接器 | 两层 MLP |
| 注意力机制 | 双向（无因果掩码） |

### 2. 掩码扩散前向过程

$$q_{t|0}(x_t | x_0) = \prod_i q_{t|0}(x_t^i | x_0^i) \tag{2}$$

$$q_{t|0}(x_t^i | x_0^i) = \begin{cases} \alpha_t = 1-t, & \text{if } x_t^i = x_0^i \\ 1-\alpha_t = t, & \text{if } x_t^i = [M] \end{cases}$$

每个 token 独立地以概率 $t$ 被替换为 mask。

### 3. 多轮对话训练目标

$$\mathcal{L} = -\mathbb{E}\left[\frac{1}{t} \sum_i \sum_j \mathbf{1}[r_t^{1,i} = [M] \wedge r_t^{2,j} = [M]] \log p_\theta(r_0^{1,i}, r_0^{2,j} | v, p_0^1, r_t^1, p_0^2, r_t^2)\right] \tag{1}$$

其中 $v$ 为视觉特征，$p_0^k$ 为第 $k$ 轮 prompt（始终未掩码），$r_t^k$ 为第 $k$ 轮 response（施加掩码）。

**关键**：双向注意力允许 response token 跨对话轮次相互关注——这是自回归模型中对话级因果掩码所不允许的。

### 4. 推理过程

1. 初始化 response 为全 $[M]$ token（$t=1$）
2. 对每个时间步 $t \to s$（$s < t$）：
   - 模型预测所有 $[M]$ 位置的 token
   - 使用**低置信度策略**重掩码 $s/t$ 比例的预测
   - 保留剩余 $1-s/t$ 的预测
3. 迭代直到 $t=0$，得到完整序列

### 5. 三阶段训练

#### Stage 1：语言-图像对齐

- 数据：LLaVA-Pretrain 558K 样本
- 冻结语言和视觉模型，仅训练 MLP 连接器
- 学习率：$10^{-3}$

#### Stage 2：视觉指令微调

- **Phase A**：单图像训练，MAmmoTH-VL 10M 样本
- **Phase B**：OneVision 训练，2M 多样化多模态样本
- 视觉编码器学习率 $2 \times 10^{-6}$，语言模型 $10^{-5}$

#### Stage 3：多模态推理增强

- 推理训练：VisualWebInstruct 900K QA 对
- 平衡推理：混合 VisualWebInstruct + MAmmoTH-VL
- 使用 `/think` 和 `/no_think` 标签控制推理模式

## 实验结果

### 知识与推理基准

| 基准 | LLaDA-V | LLaMA3-V | Qwen2-VL |
|------|---------|----------|----------|
| MMMU (val) | **48.6** | 45.4 | 54.1 |
| MMMU-Pro | **35.2** | 28.3 | 43.5 |
| MMStar | 60.1 | 56.5 | 60.7 |
| MMBench (en-dev) | **82.9** | 79.8 | — |
| MathVista | 59.7 | 62.1 | 58.2 |

### 文档与场景基准

| 基准 | LLaDA-V | LLaMA3-V | Qwen2-VL |
|------|---------|----------|----------|
| AI2D | 77.8 | 81.1 | 83.0 |
| ChartQA | **78.3** | 77.8 | 83.0 |
| DocVQA | 83.9 | 86.2 | — |
| InfoVQA | **66.3** | 58.9 | — |
| VideoMME | 56.1 | 55.8 | — |

### 数据扩展性

LLaDA-V 在数据量增加时表现出更优的扩展特性：

- **MMMU-Pro**：LLaDA-V 在 1M 样本时（26.01）已超过 LLaMA3-V 在 9M 样本时的性能
- 知识推理任务上优势明显；场景/文档任务上仍有差距

### 注意力机制消融

| 配置 | MMMU | MMBench | MuirBench |
|------|------|---------|-----------|
| 对话级因果 | 42.89 | 75.42 | 28.69 |
| **双向（无掩码）** | **44.67** | **76.71** | **33.88** |

双向注意力在 12 个基准中的 7 个上优于因果注意力。

## 个人思考

1. **纯扩散多模态模型的可行性**被首次证明：不需要自回归解码器也能做好视觉问答。
2. **数据扩展性优势**是最引人注目的发现——同样的数据量下，扩散模型比自回归模型学到更多，暗示扩散的双向建模更数据高效。
3. **双向注意力的多轮对话优势**合理：后面的回复可以关注前面的上下文，前面的上下文也可以被后面的信息"修正"。
4. **Stage 3 的 think/no_think 标签**是巧妙的工程设计——让模型在简单问题上直接回答，复杂问题上展开推理。
5. **局限性**：图像处理（分割-缩放-拼接）不如 Qwen2-VL 的原生动态分辨率高效，这是工程层面的改进空间。
