---
title: "SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions in Diffusion-Based Task Execution"
authors: "Zhixuan Liang, Yao Mu, Hengbo Ma, Masayoshi Tomizuka, Mingyu Ding, Ping Luo"
year: 2023
venue: "arXiv 2023"
tags: [机器人规划, 扩散模型]
rating: 4
paper_url: "https://arxiv.org/abs/2312.11598"
code_url: ""
summary: "可解释的分层规划：向量量化学习离散技能表示 + 技能条件扩散生成状态轨迹 + 逆动力学解码动作，组合任务成功率 25.2%"
---

## 核心思想

机器人执行长程多步任务需要**分层规划**。SkillDiffuser 将问题分为两层：

1. **高层技能抽象**：从视觉观察和语言指令中学习离散的、人类可理解的技能表示（向量量化）
2. **低层技能条件扩散**：用扩散模型生成与技能对齐的状态轨迹，再通过逆动力学解码为动作

## 方法详解

### 1. 高层技能抽象

#### 1.1 编码

视觉编码：$\Phi_\text{im}: I \to \mathcal{R}^I$，语言编码：$\Phi_\text{lang}: L \to \mathcal{R}^L$

技能码生成：

$$\tilde{z} = f(\Phi_\text{im}(i_t), \Phi_\text{lang}(l))$$

$f$ 是融合网络，将当前视觉观察和语言指令映射到连续技能表示 $\tilde{z}$。

#### 1.2 向量量化

$$z = q(\tilde{z}) = \arg\min_{z^k} \|\tilde{z} - z^k\|_2, \quad z^k \in \mathcal{C}$$

将连续的 $\tilde{z}$ 量化到 $K$ 个离散嵌入 $\{z^1, z^2, \ldots, z^K\}$ 组成的码本 $\mathcal{C}$ 中的最近邻。

**VQ 损失**：

$$\mathcal{L}_\text{VQ} = \mathbb{E}_\tau\left[\|q(\tilde{z}) - \tilde{z}\|_2^2\right]$$

使用 EMA 更新码本 + 承诺损失（commitment loss）。

**固定 horizon**：每个技能在多个时间步内保持不变，确保子任务层面的时间一致性。

#### 1.3 可解释性

每个离散技能码对应一个可解释的子任务（如"打开抽屉"、"拧水龙头"）。统计显示 17 种唯一技能覆盖了 75 条指令，平均每条指令使用 1.55 个技能——**技能被复用**。

### 2. 低层技能条件扩散规划

#### 2.1 扩散模型

U-Net 噪声预测器 $\boldsymbol{\varepsilon}_\theta(\cdot)$，技能通过 MLP $\Lambda$ 对齐后融入残差块：

$$y(\tau) = \Lambda(z) \quad \text{（技能嵌入）}$$

#### 2.2 Classifier-Free Guidance

$$\hat{\boldsymbol{\varepsilon}} = \boldsymbol{\varepsilon}_\theta(\tau^i, \emptyset, i) + \omega \left[\boldsymbol{\varepsilon}_\theta(\tau^i, y, i) - \boldsymbol{\varepsilon}_\theta(\tau^i, \emptyset, i)\right]$$

$\omega$ 为引导尺度，$\emptyset$ 为无条件（训练时以概率 $\beta$ 丢弃技能条件）。引导尺度越大，生成轨迹越忠实于技能。

#### 2.3 扩散损失

$$\mathcal{L}_\text{diff}(\theta) = \mathbb{E}_{i,\tau,\boldsymbol{\varepsilon}}\left[\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\tau^i, (1-\beta)y(\tau^i) + \beta\emptyset, i)\|^2\right]$$

### 3. 状态到动作：逆动力学

扩散模型输出状态轨迹 $\tilde{s}_t, \tilde{s}_{t+1}, \ldots$，通过逆动力学模型 $\Psi$ 解码为动作：

$$a_t = \Psi([\tilde{s}_t, \tilde{s}_{t+1}], i_t), \quad t = 0, \ldots, T-1$$

**优势**：状态空间规划与具体执行器解耦——更换机器人只需重新训练 $\Psi$，技能和规划模块可复用。

### 4. 训练损失

三部分联合优化：

| 损失 | 公式 | 作用 |
|------|------|------|
| 逆动力学 | $\mathcal{L}_\text{inv} = \mathbb{E}_\tau[\|a - \Psi(\tilde{s}, \tilde{s}', i)\|_2^2]$ | 状态→动作 |
| VQ | $\mathcal{L}_\text{VQ} = \mathbb{E}_\tau[\|q(\tilde{z}) - \tilde{z}\|_2^2]$ | 技能量化 |
| 扩散 | $\mathcal{L}_\text{diff}$ | 轨迹生成 |

总损失：$\mathcal{L}_\text{VQ} + \lambda \mathcal{L}_\text{diff}$，$\mathcal{L}_\text{inv}$ 用单独优化器。

## 实验结果

### LOReL Sawyer 数据集

| 方法 | 平均成功率 |
|------|----------|
| Random | 20% |
| LCBC | 29% |
| Lang DT | 15% |
| LISA | 40% |
| **SkillDiffuser** | **43±1.1%** |

### 组合任务

| 方法 | 成功率 |
|------|--------|
| Lang DT | 13.33% |
| LOReL | 18.18% |
| LISA | 20.89% |
| **SkillDiffuser** | **25.21±2.7%** |

比最佳基线提升 **4.3%**——在多步组合任务上优势更明显。

### 语言泛化

| 指令类型 | LISA | SkillDiffuser |
|---------|------|--------------|
| 见过的 | 40% | 43.65% |
| 未见名词 | 33.33% | 36.01% |
| 未见动词 | 30% | 36.70% |
| 未见名词+动词 | 20% | **42.02%** |
| 人类提供 | 27.35% | **40.16%** |

在未见语言组合上的泛化性远优于基线。

### 消融（Meta-World MT10）

| 配置 | 成功率 |
|------|--------|
| 无语言 + 无技能 + 无扩散 | 13.3% |
| +语言 + 技能（LISA） | 13.8% |
| +语言 + 扩散（无技能） | 16.7% |
| **+语言 + 技能 + 扩散** | **23.3%** |

三个组件缺一不可。

## 个人思考

1. **VQ 离散技能**实现可解释性：17 个技能覆盖 75 条指令，人类可以检查每个技能的含义。
2. **状态空间规划 + 逆动力学**的解耦设计优雅：规划不关心执行器，执行不关心高层意图。
3. **组合任务的提升更显著**（+4.3% vs 单步 +3%）验证了分层规划在长程任务上的优势。
4. **Classifier-free guidance**在机器人规划中的应用有效：引导尺度控制"遵循技能"的程度。
5. **绝对成功率仍然较低**（25%）——说明视觉-语言-动作的对齐仍是巨大挑战。
