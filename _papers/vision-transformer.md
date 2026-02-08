---
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: "Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al."
year: 2021
venue: "ICLR 2021"
tags: [Transformer, 图像分类]
rating: 5
paper_url: "https://arxiv.org/abs/2010.11929"
code_url: "https://github.com/google-research/vision_transformer"
summary: "首次将纯 Transformer 架构直接应用于图像分类任务，证明无需 CNN 也能达到 SOTA"
---

## 核心思想

ViT 将图像切分为固定大小的 patch（如 16x16），将每个 patch 线性映射为 token，然后直接输入标准 Transformer Encoder 进行分类。这一工作证明了在大规模数据集预训练的条件下，纯 Transformer 架构可以在图像分类任务上匹配甚至超越 CNN。

## 方法详解

1. **Patch Embedding**：将输入图像 $H \times W \times C$ 切分为 $N = HW/P^2$ 个 patch，每个 patch 展平后通过线性投影映射到 $D$ 维
2. **Position Embedding**：使用可学习的 1D 位置编码
3. **[CLS] Token**：在序列头部添加一个可学习的分类 token
4. **Transformer Encoder**：标准的多头自注意力 + FFN 结构
5. **分类头**：取 [CLS] token 的输出接 MLP Head

## 实验结果

- 在 JFT-300M 上预训练后，ViT-L/16 在 ImageNet 上达到 **87.76%** top-1 准确率
- 在中小规模数据集上，ViT 表现不如 CNN（缺乏归纳偏置）
- 预训练数据量越大，ViT 的优势越明显

## 个人思考

- ViT 的成功关键在于大规模预训练，这体现了 Transformer 的 scaling law
- Patch 大小是一个关键超参数，直接影响计算量和性能
- 后续 DeiT 证明了通过蒸馏可以在 ImageNet-1K 上高效训练 ViT
