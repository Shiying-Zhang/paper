---
title: "RegionGPT: Towards Region Understanding Vision Language Model"
authors: "Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, Sifei Liu"
year: 2024
venue: "arXiv 2024"
tags: [多模态, 区域理解]
rating: 4
paper_url: "https://arxiv.org/abs/2403.02330"
code_url: ""
summary: "区域级视觉语言理解：反卷积特征细化 + Mask Pooling 区域编码 + GPT 辅助生成 87 词/区域的详细描述，mAP 70.0"
---

## 核心思想

现有 VLM 侧重全图理解，**区域级空间感知能力不足**。RegionGPT 通过三个创新解决：

1. **特征细化模块**：反卷积上采样提升空间分辨率
2. **任务引导指令**：将视觉任务转化为 VQA 格式
3. **GPT 辅助区域标注**：自动生成平均 87 词/区域的详细描述（现有数据集仅 8 词）

## 方法详解

### 1. 视觉编码

#### 1.1 基础编码

预训练 CLIP ViT-L（冻结）提取低分辨率特征图：

$$\mathbf{Z}_\text{LRes} = f(\mathbf{X}_v)$$

#### 1.2 特征细化模块

两层反卷积（步长 2）实现 **4× 上采样**：

$$\mathbf{Z}_\text{HRes} = g(\mathbf{Z}_\text{LRes})$$

**为什么需要上采样**：CLIP ViT 输出的特征图分辨率较低（如 24×24），对小区域的信息不足。4× 上采样到 96×96 后，小物体的区域特征更精确。

消融验证：

| 细化方法 | AP | APs（小物体）|
|---------|-----|------------|
| 无 | 57.7 | 42.7 |
| 双线性 16× | 60.9 | 52.8 |
| **反卷积 4×** | **66.8** | **51.1** |

反卷积优于简单双线性插值，小物体 AP 提升显著。

#### 1.3 区域特征提取

**Mask Pooling** 从高分辨率特征图提取区域特征：

$$\mathbf{Z}_r = \text{MaskPool}(\mathbf{Z}_\text{HRes}, \mathbf{X}_r)$$

$\mathbf{X}_r$ 为区域掩码。对图像级特征使用自适应池化：

$$\mathbf{Z}_v = \text{AdaPool}(\mathbf{Z}_\text{HRes}, (H, W))$$

### 2. 视觉-语言连接器

两层 MLP 将视觉特征投影到语言嵌入空间：

$$\mathbf{H}_v = h(\mathbf{Z}_v) \quad \text{（图像嵌入）}$$

$$\mathbf{H}_r = h(\mathbf{Z}_r) \quad \text{（区域嵌入）}$$

共享连接器确保图像和区域在同一语义空间中。

特殊 token `⟨region⟩` 作为占位符，推理时替换为实际的区域嵌入 $\mathbf{H}_r$。

### 3. 任务引导指令

将不同视觉任务统一为 VQA 格式。例如 COCO 分类：

> "What category name best describes the region represented by ⟨region⟩? Answer using COCO-80 category names."

| 指令模式 | mAP |
|---------|-----|
| 单轮（所有 RoI） | 70.0 |
| **多轮（所有 RoI）** | **73.8** |
| 单轮（单个 RoI） | 71.5 |

多轮对话模式允许模型利用前面区域的上下文改善后续预测。

### 4. GPT 辅助区域标注

两阶段流水线：

**Stage 1**：用 LLaVA 生成全局图像描述 GlobalCaption

**Stage 2**：对每个区域提示 GPT-4：

> "In the context of the entire image, ⟨GlobalCaption⟩, describe the ⟨ClassName⟩ in the close-up region in detail."

| 数据集 | 平均词数/区域 |
|--------|------------|
| ReferCOCOg | 8.46 |
| **RegionGPT 标注** | **87.14** |

**10× 更丰富的描述**。每个阶段使用 10 种提示变体增加多样性，CLIP 评分过滤低质量描述。

数据规模：V3Det 213K 图像，150 万区域标注。

### 5. 训练流程

#### 预训练

- 冻结视觉编码器和 LLM，训练特征细化模块 + MLP
- 数据：LAION-CC-SBU 558K + Visual Genome + ReferCOCOg + V3Det
- Batch 256，学习率 $10^{-3}$，1 epoch

#### 微调

- 解冻 LLM，继续更新细化模块和 MLP
- Batch 128，学习率 $2 \times 10^{-5}$
- 标准自回归损失，仅在回复部分计算

### 6. 对标注噪声的鲁棒性

| 输入类型 | mAP |
|---------|-----|
| 分割掩码 | 70.0 |
| **膨胀掩码** | **71.0** |
| 边界框 | 70.4 |
| 侵蚀掩码 | 68.2 |

膨胀掩码反而最好——稍大的区域包含更多上下文信息。

## 实验结果

### 区域分类（COCO 2017）

| 方法 | mAP | 准确率 |
|------|-----|--------|
| CLIP | 58.9 | — |
| LLaVA | — | 40.04 |
| Shikra | — | 53.91 |
| GPT4RoI | — | 64.01 |
| **RGPT** | **70.0** | **80.61** |

### 区域描述（RefCOCOg）

| 方法 | METEOR | CIDEr |
|------|--------|-------|
| GRIT | 15.2 | 71.6 |
| Kosmos-2 | 14.1 | 62.3 |
| **RGPT** | **16.9** | **109.9** |

CIDEr 提升 **53%**——得益于 87 词/区域的丰富训练数据。

### 指称表达理解（ReferCOCOg）

- Val: 86.44%，Test: **86.96%**
- 超过 MiniGPT-V2 (84.44%)

### 幻觉（POPE）

| 类别 | RGPT | Shikra | InstructBLIP |
|------|------|--------|-------------|
| 对抗 F1 | **84.50** | 82.49 | 77.32 |

区域级理解减少了幻觉。

## 个人思考

1. **反卷积细化**看似简单但效果显著：4× 上采样 + 小物体 AP 大幅提升——说明 VLM 的空间分辨率是被忽视的瓶颈。
2. **87 词 vs 8 词**的数据质量差距巨大：CIDEr +53% 说明详细的区域描述是提升区域理解的关键。
3. **膨胀掩码更好**的发现实用：实际应用中精确掩码难以获取，稍大的区域反而更鲁棒。
4. **多轮对话 > 单轮**说明区域理解需要全图上下文——孤立地看一个区域不如在全图语境中理解。
5. **数据流水线可扩展**：GPT-4 标注成本虽高，但自动化流水线使其可以扩展到任意数据集。
