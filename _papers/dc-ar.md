---
title: "DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer"
authors: "Yecheng Wu, Junyu Chen, Zhuoyang Zhang, Enze Xie, Jincheng Yu, Junsong Chen, Jinyi Hu, Yao Lu, Song Han, Han Cai"
year: 2025
venue: "arXiv 2025"
tags: [图像生成, Transformer]
rating: 4
paper_url: "https://arxiv.org/abs/2507.04947"
code_url: ""
summary: "混合 tokenizer (DC-HT)：离散 token 捕获结构 + 连续残差 token 捕获细节，32× 压缩率下 12 步 MaskGIT 采样达 gFID 5.49"
---

## 核心思想

自回归图像生成面临效率与质量的权衡：MaskGIT 式的离散 token 方法采样快（~8 步）但质量受限于 VQ；MAR 式的连续 token 方法质量高但采样慢（~64 步）。本文提出 **DC-AR**：

- **DC-HT (Deep Compression Hybrid Tokenizer)**：结合离散和连续 token 路径的混合 tokenizer
- **混合生成框架**：先用离散 token 预测结构，再用残差 token 补充细节

## 方法详解

### 1. DC-HT Tokenizer 设计

#### 1.1 混合 Tokenization 流程

$$\mathbf{Z} = \text{Enc}(\mathbf{I}) \quad \text{（CNN 编码器）}$$

**离散路径**：$\mathbf{Z}_q = \text{Quant}(\mathbf{Z})$ → $\text{Dec}(\mathbf{Z}_q)$

**连续路径**：$\text{Dec}(\mathbf{Z})$（跳过量化）

**残差 token**：$\mathbf{Z}_r = \mathbf{Z} - \mathbf{Z}_q$

**关键设计**：保持 2D 空间对应（不像 1D tokenizer），支持跨分辨率泛化。

#### 1.2 三阶段适配训练

| 阶段 | 焦点 | 训练轮数 |
|------|------|---------|
| Stage 1：连续预热 | 初始化编码器 | ~10 epochs |
| Stage 2：离散学习 | 稳定 VQ codebook | ~40 epochs |
| Stage 3：交替微调 | 解码器鲁棒性 | ~10 epochs，50% 概率采样 |

效果：rFID 从 1.92 → **1.60**；离散 rFID 从 6.18 → **5.13**。

### 2. 生成器架构

#### 2.1 Transformer 组件
- **骨干**：PixArt-α（28 层，宽度 1152，634M 参数）
- **仅处理离散 token**：前向传播只涉及离散 token → 保持 MaskGIT 的采样效率
- 文本嵌入通过 cross-attention 集成（T5-base，109M 参数）

#### 2.2 Diffusion Head
- **6 层 MLP**（37M 参数）
- 在 Transformer hidden states 条件下预测残差 token
- 使用扩散损失优化

#### 2.3 训练损失

$$\mathcal{L} = \mathcal{L}_\text{CE}(\text{离散 token}) + \mathcal{L}_\text{Diff}(\text{残差 token})$$

交叉熵损失用于离散 token 预测，扩散损失用于残差 token 预测。

### 3. 推理流程

```
1. 初始化全部 mask 的离散 token
2. 渐进式 unmasking（12 个采样步）     ← MaskGIT 式
3. 提取最终 Transformer hidden states
4. Diffusion head 预测残差 token（20 步）← 扩散式
5. 组合：Z_final = Z_q + Z_r
6. 解码器产生输出图像
```

**关键设计选择**：只有离散 token 参与 Transformer 前向传播 → 保持 MaskGIT 的 ~8 步采样效率，而非 MAR 的 ~64 步。

### 4. 两阶段生成训练

针对 512×512 模型：
1. 在 256×256 上预训练（200K 步）
2. 在 512×512 上微调（50K 步）
3. 结果：GPU 时间减少 **1.9×**，同时质量更好

## 实验结果

### Tokenizer 性能

| 指标 | 分辨率 | DC-HT | TiTok | TexTok |
|------|--------|-------|-------|--------|
| rFID | 256×256 | 1.60 | 1.70 | 1.53 |
| rFID | 512×512 | **0.83** | — | 0.73 |
| PSNR | 256×256 | **21.50** | — | 20.10 |

DC-HT 在 PSNR/SSIM 上优于 TexTok，且支持跨分辨率泛化（1D tokenizer 不能）。

### 文本到图像生成 (MJHQ-30K)

| 模型 | 参数 | gFID↓ | 延迟 | 吞吐量 |
|------|------|-------|------|--------|
| **DC-AR** | **671M** | **5.49** | **0.4s** | **10.3 img/s** |
| Sana-0.6B | 590M | 5.81 | 0.9s | 6.7 img/s |
| SDXL | 2.6B | 8.71 | 3.3s | — |

比 Sana-0.6B **延迟降低 2×，吞吐量提升 1.5×**。

### GenEval 基准

| 模型 | 参数 | Overall↑ |
|------|------|----------|
| **DC-AR** | 671M | **0.69** |
| Show-o | 1.3B | 0.69 |
| LlamaGen | 775M | 0.58 |

### 消融实验

#### 混合 vs 纯离散

| 配置 | rFID | gFID | GenEval |
|------|------|------|---------|
| **DC-AR (混合)** | **1.60** | **5.50** | **0.69** |
| 纯离散 | 5.13 | 6.71 | 0.66 |

混合设计以 ~10% 额外开销换取显著质量提升。

#### 采样步数

- DC-AR 最优：**12 步**
- MAR-B 需要：64 步
- 离散主导的生成是效率关键

## 个人思考

1. **离散 token 负责结构、连续 token 负责细节**的分工非常直觉：VQ 天然捕获结构信息，残差补充被量化丢弃的细节。
2. **三阶段训练策略**解决了 32× 高压缩率下 VQ 不稳定的问题，rFID 从 1.92 降到 1.60。
3. **只让离散 token 参与 Transformer**是精妙的设计：保持了 MaskGIT 的采样效率，避免了 MAR 的 64 步瓶颈。
4. **2D 空间对应**使得单个 tokenizer 支持多分辨率——1D tokenizer 的致命弱点。
5. **预训练 + 微调策略**节省 1.9× GPU 时间且质量更好，说明低分辨率预训练提供了良好初始化。
