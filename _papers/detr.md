---
title: "End-to-End Object Detection with Transformers (DETR)"
authors: "Nicolas Carion, Francisco Massa, Gabriel Synnaeve, et al."
year: 2020
venue: "ECCV 2020"
tags: [目标检测, Transformer]
rating: 4
paper_url: "https://arxiv.org/abs/2005.12872"
code_url: "https://github.com/facebookresearch/detr"
summary: "首个基于 Transformer 的端到端目标检测框架，去除了 NMS 和 anchor 等手工设计组件"
---

## 核心思想

DETR 提出了一种全新的目标检测范式：将目标检测建模为集合预测问题，使用 Transformer 的 encoder-decoder 架构直接输出固定数量的预测结果，并通过匈牙利算法进行二分图匹配计算损失。完全去除了传统检测器中的 anchor、NMS 等手工设计组件。

## 方法详解

1. **CNN Backbone**：使用 ResNet 提取特征图
2. **Transformer Encoder**：对特征图进行全局自注意力建模
3. **Transformer Decoder**：使用 N 个可学习的 object queries 与编码特征交互
4. **预测头**：每个 query 输出一个类别 + 边界框
5. **匈牙利匹配**：将预测与 GT 进行最优二分图匹配
6. **集合损失**：分类损失 + L1 损失 + GIoU 损失

## 实验结果

- 在 COCO 上达到与 Faster R-CNN 相当的性能（~42 AP）
- 在大物体检测上表现优异（得益于全局注意力）
- 小物体检测表现偏弱（后续 Deformable DETR 改进）
- 训练收敛慢（需要 500 epochs）

## 个人思考

- DETR 最大的贡献是简化了检测流程，让检测变成了一个优雅的集合预测问题
- 训练收敛慢是主要缺陷，Deformable DETR 通过可变形注意力大幅改善
- Object query 的物理含义值得深入研究，后续工作如 DAB-DETR 给出了更好的理解
