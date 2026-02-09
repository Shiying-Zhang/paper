---
title: "Energy Scaling Laws for Diffusion Models: Quantifying Compute and Carbon Emissions in Image Generation"
authors: "Aniketh Iyengar, Jiaqi Han, Boris Ruf, Vincent Grari, Marcin Detyniecki, Stefano Ermon"
year: 2025
venue: "arXiv 2025"
tags: [扩散模型, 能耗分析, 缩放定律]
rating: 3
paper_url: "https://arxiv.org/abs/2511.17031"
code_url: ""
summary: "扩散模型推理能耗的缩放定律：FLOPs 分解（文本编码 + 去噪 + 解码）+ 对数线性回归，跨架构 R² > 0.9，去噪占 90%+ 能耗，fp32 比 fp16 贵 ~7.4×"
---

## 核心思想

生成一张图像要消耗多少能量？随着扩散模型越来越大（从 865M 到 20B 参数），这个问题变得重要。

本文建立了扩散模型推理**能耗的缩放定律**：
1. 将推理分解为文本编码 + 迭代去噪 + 解码三个阶段
2. 去噪阶段占总 FLOPs 的 **90%+** → 是能耗主导因素
3. 建立对数线性回归模型：$\log(E) = \alpha \cdot \log(\text{FLOPs}) + \text{硬件/配置项}$
4. 跨架构（U-Net → Transformer）泛化 $R^2 > 0.9$

## 背景知识

### 为什么关注能耗

| 方面 | 说明 |
|------|------|
| 环境 | 一张高质量图像的能耗可达 LLM 推理的 **10×** |
| 成本 | 大规模部署时能耗直接转化为电费 |
| 监管 | 欧盟等地区开始要求 AI 系统的碳排放报告 |
| 优化 | 知道能耗构成才能针对性优化 |

### 影响能耗的因素

| 因素 | 影响 |
|------|------|
| 模型大小 | 865M → 20B 参数 |
| 分辨率 | 256² → 1024² |
| 采样步数 | 10 → 50 步 |
| 精度 | fp16 vs fp32 |
| CFG | 有无分类器引导（2× FLOPs） |
| GPU 型号 | A100 vs A4000 vs A6000 |

## 方法详解

### 1. FLOPs 分解

$$\text{FLOPs}_{\text{total}} = \text{FLOPs}_{\text{text}} + N_{\text{steps}} \times \text{FLOPs}_{\text{denoise}} + \text{FLOPs}_{\text{decode}}$$

去噪步骤被重复 $N_{\text{steps}}$ 次 → **主导总计算量**。

### 2. 去噪 FLOPs 估计（Kaplan 公式）

$$\text{FLOPs}_{\text{denoise}} \approx 2N + 2L \cdot d_{\text{attn}} \cdot n_{\text{layers}}$$

- $N$：参数量
- $L$：序列长度（与分辨率相关）
- $d_{\text{attn}}$：注意力维度
- $n_{\text{layers}}$：层数

### 3. 能耗回归模型

$$\log(E) = \log(A) + \alpha \cdot \log(\text{FLOPs} \times 2^{I_{\text{cfg}}}) + \beta_{\text{dtype}} \cdot I_{\text{dtype}} + \beta_{\text{gpu}} \cdot I_{\text{gpu}} + \beta_{\text{res}} \cdot \log\frac{HW}{256}$$

| 系数 | 含义 |
|------|------|
| $\alpha \approx 1$ | FLOPs 与能耗近似**线性**关系 |
| $\beta_{\text{dtype}} \approx 2.0$ | fp32 比 fp16 贵 $e^2 \approx 7.4$ 倍 |
| $\beta_{\text{gpu}}$ | GPU 型号的能效差异 |
| $\beta_{\text{res}}$ | 分辨率的超线性影响（注意力的二次复杂度） |

### 4. 测试模型

| 模型 | 参数量 | 架构 | 特点 |
|------|--------|------|------|
| SD 2 | 865M | U-Net | ResNet + 交叉注意力 |
| SD 3.5 | 8B | MMDiT | 38 层，双 CLIP+T5 |
| Flux.1 | 12B | 混合 MMDiT | 19+38 层，整流流 |
| Qwen-Image | 20B | MMDiT | 60 层，16×16 patch |

## 实验结果

### 模型级缩放

| 模型 | $R^2$ | $\alpha$ | $\beta_{\text{dtype}}$ |
|------|-------|---------|---------------------|
| Flux | 1.000 | 0.989 | 2.04 |
| SD 3.5 | 0.998 | 0.969 | 1.90 |
| Qwen | 0.994 | 0.992 | -0.306（仅 fp16） |

$\alpha \approx 1$ → **能耗与 FLOPs 近似线性**（计算受限而非内存受限）。

### 跨 GPU 泛化

| 模型 | GPU 组合 | $\alpha$ | $\beta_{\text{gpu}}$（A6000 开销） |
|------|---------|---------|--------------------------------|
| Flux | A100 + A6000 | 0.997 | 0.450 |
| SD 3.5 | A100 + A6000 | 0.989 | 0.308 |

### 典型能耗（100 张图，A100）

| 模型 | 256²/10步/fp16/无CFG | 1024²/50步/fp32/CFG |
|------|---------------------|---------------------|
| SD 2 | 3.20×10³ J | 6.26×10⁵ J |
| SD 3.5 | 1.45×10⁴ J | 5.33×10⁶ J |
| Flux | 2.95×10⁴ J | 1.21×10⁷ J |

最大配置比最小配置贵 **100-200×**。

### 跨架构泛化

用 MMDiT 模型训练的回归模型预测 U-Net（SD2）的能耗：相对排序准确度 $R, R_s > 0.9$。

## 个人思考

1. **$\alpha \approx 1$ 是核心发现**：说明扩散模型推理是**计算受限**的（而非内存受限）→ 降低 FLOPs 直接等比降低能耗。
2. **fp32 贵 7.4× 而质量提升微乎其微** → 实际部署应该始终使用 fp16/bf16。
3. **去噪占 90%+ 能耗** → 减少采样步数（一致性模型、蒸馏等）是最有效的节能手段。
4. **跨架构泛化**说明能耗主要由计算复杂度决定，而非架构细节 → 统一的能耗估计框架是可行的。
5. **实际意义**：可以在部署前预估能耗和碳排放，支持碳感知的推理参数选择（如根据电网碳强度动态调整分辨率/步数）。
