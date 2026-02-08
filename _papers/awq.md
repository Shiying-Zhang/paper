---
title: "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"
authors: "Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han"
year: 2023
venue: "MLSys 2024"
tags: [模型量化, LLM推理]
rating: 5
paper_url: "https://arxiv.org/abs/2306.00978"
code_url: "https://github.com/mit-han-lab/llm-awq"
summary: "发现仅 1% 关键权重（由激活分布决定）即可大幅降低量化误差，提出免训练的 per-channel 缩放方法，INT4 量化几乎无损"
---

## 核心思想

**核心发现**：LLM 中并非所有权重同等重要。保护仅 **1%** 的关键通道（由**激活分布**而非权重幅度决定）即可大幅减少量化误差。

**AWQ** 不使用混合精度（硬件不友好），而是通过 **per-channel 缩放**保护重要权重，保持统一量化的硬件效率。

## 方法详解

### 1. 核心观察

对 OPT-6.7B 的 INT3-g128 量化：

| 方法 | PPL |
|------|-----|
| RTN 基线 | 23.54 |
| 保留 1% 通道 FP16（按激活选择）| **11.39** |
| 保留 1% 通道 FP16（随机选择）| 24.23 |

**关键**：按**激活幅度**选择的 1% 通道远比随机选择重要得多。

### 2. 量化误差数学分析

#### 2.1 标准量化

$$Q(\mathbf{w}) = \Delta \cdot \text{Round}\left(\frac{\mathbf{w}}{\Delta}\right), \quad \Delta = \frac{\max(|\mathbf{w}|)}{2^{N-1}} \tag{1}$$

#### 2.2 缩放分析

对权重元素 $w$ 乘以缩放因子 $s > 1$：

$$Q(w \cdot s) \cdot \frac{x}{s} = \Delta' \cdot \text{Round}\left(\frac{ws}{\Delta'}\right) \cdot x \cdot \frac{1}{s}$$

量化误差：

$$\text{Err}(Q(w)x) = \Delta \cdot \text{RoundErr}\left(\frac{w}{\Delta}\right) \cdot x \tag{3}$$

缩放后的误差：

$$\text{Err}\left(Q(w \cdot s)\left(\frac{x}{s}\right)\right) = \Delta' \cdot \text{RoundErr}\left(\frac{ws}{\Delta'}\right) \cdot x \cdot \frac{1}{s} \tag{4}$$

误差比值：$\frac{\Delta'}{\Delta} \cdot \frac{1}{s}$

**在经验假设 $\Delta' \approx \Delta$（缩放一个元素不改变组最大值）下，误差比值 ≈ $1/s$**——缩放确实降低了关键通道的量化误差。

#### 2.3 缩放上限

| 缩放因子 $s$ | PPL |
|-------------|-----|
| $s = 1$ | 23.54 |
| $s = 1.5$ | **11.92** |
| $s = 2$ | 12.36 |

$s$ 太大会增大非关键通道的误差——存在权衡。

### 3. 激活感知缩放

#### 3.1 优化目标

$$\mathbf{s}^* = \arg\min_\mathbf{s} \mathcal{L}(\mathbf{s}) \tag{4}$$

$$\mathcal{L}(\mathbf{s}) = \|Q(\mathbf{W} \cdot \text{diag}(\mathbf{s}))(\text{diag}(\mathbf{s})^{-1} \cdot \mathbf{X}) - \mathbf{W}\mathbf{X}\| \tag{5}$$

最小化量化前后的输出差异。

#### 3.2 搜索空间参数化

$$\mathbf{s} = \mathbf{s}_X^\alpha, \quad \alpha^* = \arg\min_\alpha \mathcal{L}(\mathbf{s}_X^\alpha)$$

其中 $\mathbf{s}_X$ 为每通道的平均激活幅度，$\alpha \in [0,1]$。

- $\alpha = 0$：不缩放
- $\alpha = 1$：完全按激活幅度缩放
- 网格搜索 20 个值找最优 $\alpha$

#### 3.3 完整算法

```
1. 从小型校准集离线收集激活统计
2. 通过 per-channel 激活幅度识别关键通道
3. 网格搜索最优缩放指数 α
4. 应用 per-channel 缩放 + 权重裁剪
5. 统一量化所有权重（INT3/INT4）
6. 逆缩放融入前序算子
```

**关键优势**：
- 无需反向传播或重建
- 极少的校准数据（比 GPTQ 少 **10×**）
- 更好的跨域泛化
- 不过拟合校准分布

### 4. TinyChat 系统实现

#### 4.1 瓶颈分析

LLaMA-2-7B 在 RTX 4090 上：
- 上下文处理：200 token 需 10ms
- **生成**：20 token 需 310ms（**31× 慢**）
- 生成的算术强度：~1 FLOP/byte（内存受限）
- 4-bit 量化将权重内存减少 4×，算术强度从 1 提升到 ~4 FLOP/byte

#### 4.2 系统优化

**On-the-fly 权重反量化**：将反量化融入矩阵乘法 kernel，避免写回 DRAM。

**SIMD-aware 权重打包**（ARM NEON）：
- 128-bit SIMD：$w_0, w_{16}, w_1, w_{17}, \ldots$
- 实现 **1.2× 加速**

**Kernel 融合**：LayerNorm、QKV 投影、位置编码、KV cache 更新全部融合。

#### 4.3 部署速度

| 平台 | 模型 | FP16 | W4A16 | 加速 |
|------|------|------|-------|------|
| RTX 4090 | LLaMA-2-7B | 52 tok/s | 162 tok/s | 3.1× |
| RTX 4090 | LLaMA-2-13B | 28 tok/s | 87 tok/s | 3.1× |
| Jetson Orin | LLaMA-2-7B | — | — | 3.5× |
| RTX 4070 (8GB) | LLaMA-2-13B | OOM | 33 tok/s | ∞ |

## 实验结果

### 语言建模 (WikiText-2 PPL, INT4-g128)

| 模型 | FP16 | RTN | GPTQ | **AWQ** |
|------|------|-----|------|---------|
| LLaMA-7B | 5.68 | 5.96 | 6.22 | **5.78** |
| LLaMA-13B | 5.09 | 5.25 | 5.23 | **5.19** |
| LLaMA-65B | 3.53 | 3.67 | 3.66 | **3.62** |

### 跨域泛化 (OPT-6.7B INT3-g128)

| 校准→评估 | AWQ PPL | GPTQ PPL |
|-----------|---------|----------|
| PubMed→PubMed | 10.86 | 10.97 |
| PubMed→Enron | 11.36 | **14.87** |
| Enron→PubMed | 11.13 | **14.57** |

分布不匹配时，AWQ PPL 仅增加 0.5-0.6，GPTQ 增加 2.3-4.9——**AWQ 泛化性远优于 GPTQ**。

### 多模态模型 (VILA-7B/13B)

INT4 量化后在 11 个视觉语言基准上**几乎无损**（0.1-0.5% 差距）。

### 数学/编程任务

| 模型 | 任务 | FP16 | AWQ |
|------|------|------|-----|
| LLaMA-2-70B | GSM8K | 56.41 | **56.40** |
| CodeLLaMA-7B | MBPP pass@1 | 38.53 | **40.64** |

## 个人思考

1. **"激活决定权重重要性"**的洞察是核心贡献：权重幅度大不等于重要，激活幅度大的通道才是关键。
2. **Per-channel 缩放**的优雅之处在于：保护重要权重但保持统一量化精度 → 硬件友好。
3. **免训练 + 少数据**是巨大的实用优势：无需反向传播，校准数据比 GPTQ 少 10×。
4. **跨域泛化**是 AWQ 相比 GPTQ 的核心优势：GPTQ 容易过拟合校准分布，AWQ 不会。
5. **广泛采用**（HuggingFace, TensorRT-LLM, vLLM 等）证明了方法的实用性——简单、有效、通用。
