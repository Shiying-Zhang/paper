---
title: "Light-X: Generative 4D Video Rendering with Camera and Illumination Control"
authors: "Tianqi Liu, Zhaoxi Chen, Zihao Huang, Shaocong Xu, Saining Zhang, Chongjie Ye, Bohan Li, Zhiguo Cao, Wei Li, Hao Zhao, Ziwei Liu"
year: 2024
venue: "arXiv 2024"
tags: [视频生成, 3D视觉]
rating: 4
paper_url: "https://arxiv.org/abs/2512.05115"
code_url: ""
summary: "联合相机轨迹和光照控制的视频生成：动态点云解耦几何/运动，重光照帧解耦光照，Light-Syn 降质流水线合成训练数据"
---

## 核心思想

从单目视频出发，实现**同时控制相机轨迹和光照**的 4D 视频渲染。核心挑战是光照保真度和时序一致性的固有矛盾——逐帧重光照保证光照但破坏时序，传统方法保证时序但忽略光照。

**Light-X** 通过解耦策略解决：
1. **几何/运动**：通过动态点云投影控制相机
2. **光照**：通过稀疏重光照帧传递光照信息
3. **数据**：Light-Syn 降质流水线从野外视频合成训练对

## 方法详解

### 1. 相机-光照解耦

#### 1.1 相机控制：动态点云

源视频经深度估计得到动态点云序列：

$$\mathbf{P}_i = \Phi^{-1}(\mathbf{I}_i^s, \mathbf{D}_i^s; \mathbf{K})$$

$\Phi^{-1}$ 为逆透视投影，$\mathbf{K}$ 为相机内参。沿用户指定轨迹投影：

$$\mathbf{I}_i^p, \mathbf{M}_i^p = \Phi(\mathbf{R}_i \mathbf{P}_i + \mathbf{t}_i; \mathbf{K})$$

得到几何对齐的渲染 $\mathbf{V}^p$ 和可见性掩码 $\mathbf{V}^m$。

#### 1.2 光照控制：重光照点云

IC-Light 对源帧进行重光照，得到稀疏重光照视频（仅一帧有光照信息，其余为空）：

$$\hat{\mathbf{P}}_i = \Phi^{-1}(\hat{\mathbf{I}}_i^s, \mathbf{D}_i^s; \mathbf{K})$$

沿相同目标轨迹投影：

$$\hat{\mathbf{I}}_i^p, \hat{\mathbf{M}}_i^p = \Phi(\mathbf{R}_i \hat{\mathbf{P}}_i + \mathbf{t}_i; \mathbf{K})$$

### 2. 条件视频扩散架构

目标视频分布：

$$\mathbf{x} \sim p(\mathbf{x} \mid \mathbf{V}^s, \hat{\mathbf{V}}^s, \mathbf{V}^p, \hat{\mathbf{V}}^p, \mathbf{V}^m, \hat{\mathbf{V}}^m)$$

6 种条件信号：源视频、重光照视频、几何投影、重光照投影、两组可见性掩码。

**细粒度线索处理**：条件信号经 VAE 编码后与噪声拼接，通过 patchification 转为视觉 token $\mathcal{T}_\text{vision}$，与文本 token $\mathcal{T}_\text{text}$ 一起输入 DiT。

**全局光照控制模块**：Q-Former 从重光照帧提取光照信息到可学习的光照 token $\mathcal{T}_\text{illum}^{(0)}$，通过 cross-attention 注入：

$$\mathcal{T}'_\text{vision} = \text{CrossAttn}(Q = \mathcal{T}_\text{vision}, K = V = \mathcal{T}_\text{illum})$$

**软掩码机制**：

$$(\hat{\mathbf{V}}^p, \hat{\mathbf{V}}^m) = (\mathbf{V}_k, \alpha_k \mathbf{1}), \quad k \in \{\text{ref}, \text{hdr}\}$$

$\alpha_\text{ref} = 0.25$, $\alpha_\text{hdr} = 0.50$。通过不同的 $\alpha$ 值，单一模型适配文本、背景图、HDR map、参考图等多种条件。

### 3. Light-Syn 数据合成

**核心思路**：野外视频作为目标（高质量），通过降质变换生成输入，逆映射生成对齐的条件信号。

| 数据类型 | 来源 | 规模 |
|---------|------|------|
| 静态场景 | DL3DV + VGGT 深度/位姿 | 8K |
| 动态场景 | VDW + TrajectoryCrafter + LAV 降质 | 8K |
| AI 生成视频 | OpenVid-1M | 2K |
| **总计** | | **18K** |

## 实验结果

### 联合相机-光照控制

| 指标 | Light-X | 最佳基线 |
|------|---------|---------|
| FID↓ | **101.06** | 122.73 |
| 审美评分↑ | **0.623** | 0.596 |
| 运动保持↓ | **2.007** | 2.007 |
| CLIP↑ | **0.989** | 0.987 |

### 与真实视频对比

| 指标 | Light-X | TL-Free |
|------|---------|---------|
| PSNR↑ | **13.96** | 13.49 |
| SSIM↑ | **0.582** | 0.547 |
| LPIPS↓ | **0.378** | 0.418 |
| FVD↓ | **45.91** | 54.44 |

### 消融实验

| 配置 | FID↓ |
|------|------|
| **完整 Light-X** | **101.06** |
| 去除细粒度光照线索 | 143.02 |
| 去除全局光照控制 | 103.13 |
| 去除软掩码 | 148.51 |
| 去除静态数据 | 123.35 |

细粒度光照线索和软掩码影响最大。

## 个人思考

1. **解耦几何与光照**的思路清晰：点云负责"物体在哪"，重光照帧负责"光照是什么"——各司其职。
2. **Light-Syn 降质流水线**解决了训练数据瓶颈：无需多视角/多光照采集，从野外视频自动合成。
3. **软掩码**实现单模型多条件适配，是优雅的工程设计——不同 $\alpha$ 值对应不同条件强度。
4. **18K 训练数据**相对较少就取得了不错效果，说明条件信号的设计比数据量更重要。
5. **局限性**：极端相机运动（>60°）下点云会出现大面积遮挡，降质假设可能不成立。
