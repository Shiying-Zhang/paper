---
title: "DexFlyWheel: A Scalable and Self-improving Data Generation Framework for Dexterous Manipulation"
authors: "Kefei Zhu, Fengshuo Bai, YuanHao Xiang, Yishuai Cai, Xinglin Chen, Ruochong Li, Xingtao Wang, Hao Dong, Yaodong Yang, Xiaopeng Fan, Yuanpei Chen"
year: 2024
venue: "arXiv 2024"
tags: [机器人操控, 模仿学习]
rating: 4
paper_url: "https://arxiv.org/abs/2509.23829"
code_url: ""
summary: "两阶段飞轮框架：从单个人类示范出发，通过模仿学习+残差强化学习+数据增强的自我改进循环，生成2000+高质量灵巧操控轨迹"
---

## 核心思想

灵巧操控数据稀缺是机器人学的核心瓶颈。本文提出 **DexFlyWheel**：一个自我改进的数据生成飞轮框架，从**单个人类示范**出发，通过迭代循环（模仿学习 → 残差 RL → 策略 rollout → 数据增强）不断扩大数据规模和多样性，最终生成 2000+ 轨迹覆盖 500+ 场景。

## 方法详解

### 1. 任务定义

建模为马尔可夫决策过程：

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \pi, \mathcal{T}, R, \gamma, \rho, G)$$

其中 $\mathcal{T}(s_{t+1}|s_t, a_t)$ 为转移分布，$R$ 为奖励函数，$\gamma$ 为折扣因子。

### 2. 阶段一：预热初始化

1. 通过 VR 遥操作（Apple Vision Pro）收集**单个**人类示范 $d_\text{seed}$
2. 应用多维增强模块：$\mathcal{D}_1 = \mathcal{A}_\text{EP}(d_\text{seed}; \mathcal{C}_1)$
3. 生成初始数据集，包含环境和空间变化

### 3. 阶段二：自我改进飞轮

每次迭代 $i \in \{1, 2, \ldots, n-1\}$ 包含四步：

#### 3.1 步骤一：基础策略训练（模仿学习）

- **架构**：扩散策略（DDIMScheduler，50 训练时间步，16 推理步）
- **输入状态**：$s_t = \{s_t^\text{vis}, s_t^\text{obj}, s_t^\text{prop}\}$
  - $s_t^\text{vis}$：224×224×3 相机图像
  - $s_t^\text{obj}$：13D 物体状态（6D 位姿 + 速度），双物体为 26D
  - $s_t^\text{prop}$：机器人本体感觉（单臂 69D，双臂 130D）
- **输出**：动作序列 $(a_t, a_{t+1}, \ldots, a_{t+H})$，$H=8$
- **训练**：200,000 epochs，batch size 256，lr 3.0e-4

#### 3.2 步骤二：残差策略训练（强化学习）

- **算法**：Soft Actor-Critic (SAC)
- **网络**：3 层 MLP [256, 256, 256]
- **组合策略**：

$$\boxed{\pi_\text{combined}(s) = \begin{cases} \pi_\text{base}(s) + \alpha \cdot \pi_\text{res}(s) & \text{概率 } \varepsilon \\ \pi_\text{base}(s) & \text{概率 } 1-\varepsilon \end{cases}} \tag{1}$$

其中 $\varepsilon$ 从 0 线性增到 1（步骤 1500 到 10000），实现渐进式探索。

**残差动作组合**：

$$\tilde{a}_t = a_t + \alpha \cdot \Delta a_t$$

$\alpha = 0.1$（缩放因子），确保残差是**微调**而非完全替换。

#### 3.3 步骤三：Rollout 收集

冻结组合策略生成轨迹：

$$\mathcal{D}_O^i = \{d_j = \{(s_t, a_t)\}_{t=0}^{T-1} \mid d_j \sim \pi_\text{combined}^i\}_{j=1}^K$$

基于成功率进行轨迹筛选。

#### 3.4 步骤四：数据增强

$$\mathcal{D}_{i+1} = \mathcal{A}_\text{EP}(\mathcal{D}_O^i; \mathcal{C}_\text{aug}^{i+1})$$

多维增强包括：
- **物体变化**：基于课程的渐进扩展
- **环境变化**：12 种不同光照/桌面条件
- **空间变化**：从源头附近开始，逐步向外扩展

### 4. 奖励函数

#### 4.1 抓取任务

$$r_\text{grasp} = \exp\left(-5 \cdot \max\left(\sum_i d_i - 0.05, 0\right)\right) + 100 \cdot \max(0.2 - |z_\text{target} - z_\text{current}|, -0.01) \tag{2}$$

#### 4.2 倒水任务

$$r_\text{pour} = 5.0 \cdot \mathbb{1}(\text{成功}) + 10 \cdot (r_\text{grasp\_dist} + r_\text{lift}) + 50 \cdot (r_\text{tilt} + r_\text{ball\_bowl}) \tag{3}$$

#### 4.3 举升任务

$$r_\text{lift} = r_\text{left\_grasp} + r_\text{right\_grasp} + r_\text{sync} + r_\text{lift\_height} - p_\theta \tag{4}$$

其中：
- $r_\text{sync} = 4 \cdot \exp(-5 \cdot \max(s_\text{sync} - 0.2, 0))$（同步奖励）
- $r_\text{lift\_height} = 10 \cdot \min(\max(\Delta z / 0.15, 0), 1)$（高度奖励）
- $p_\theta = \min(5.0, (\theta_\max - 30.0)/5.0) \cdot \mathbb{1}(\theta_\max > 30.0)$（倾斜惩罚）

### 5. 残差范数比 (RNR)

验证"微调"观察：

$$\text{RNR} = \frac{\|a_t^\text{res}\|}{\|a_t^\text{base}\| + \epsilon}, \quad \epsilon = 10^{-6}$$

平均 RNR ≈ 14-16%，确认残差是微妙调整而非彻底重规划。

## 实验结果

### 迭代进展

| 指标 | i=1 | i=2 | i=3 | 提升 |
|------|-----|-----|-----|------|
| 物体种类 | 1.0 | 6.8 | 20.0 | 20× |
| 环境种类 | 3.0 | 6.8 | 12.0 | 4× |
| 配置总数 | 9.5 | 322 | 2,040 | 214.7× |
| 轨迹数量 | 20 | 100 | 500 | 25× |
| T_OEP 成功率 | 16.5% | 43.9% | **81.9%** | +396% |

### 基线对比（T_OEP 测试集）

| 方法 | 抓取 | 倒水 | 举升 | 交接 | 平均 |
|------|------|------|------|------|------|
| Human Demo (默认) | 6.1% | 16.7% | 13.9% | 0.8% | 9.4% |
| DexMimicGen (增强) | 50.3% | 44.4% | 43.7% | 42.5% | 45.2% |
| **DexFlyWheel** | **90.0%** | **85.8%** | **79.4%** | **72.5%** | **81.9%** |

DexFlyWheel 比拥有 10× 数据优势的 DexMimicGen (Enhanced) 高出 **36.7 个百分点**。

### 时间效率

| 方法 | 每轨迹 | 500 条成功轨迹 |
|------|--------|--------------|
| 人类遥操作 | 60s | 12.5 小时 |
| DexMimicGen | 15s | 4.4 小时 |
| **DexFlyWheel** | **15s** | **2.4 小时** |

比人类遥操作快 **5.21×**。

### 真实世界部署

通过数字孪生迁移到双臂机器人：
- **举升任务**：78.3% 成功率
- **交接任务**：63.3% 成功率

### 消融实验

- **去掉残差策略**：性能下降最大（~40-50%）
- **去掉数据增强**：中等下降（~20-30%）
- **两者都去掉**：仅剩基线水平

### 扩展迭代分析

| 迭代 | 抓取 SR | 举升 SR |
|------|--------|--------|
| i=3 | 90.0% | 79.4% |
| i=4 | 92.5% | 82.1% |
| i=5 | 93.2% | 83.5% |

i=3 提供最佳的性能-成本权衡，后续收益递减。

## 个人思考

1. **飞轮效应**非常直观：更好的数据 → 更好的策略 → 更好的 rollout → 更好的数据，形成正反馈循环。
2. **残差 RL 的微调本质**（RNR ~15%）说明基础策略已经很好，RL 只是做精细调整——这比从头 RL 更稳定高效。
3. **单个示范到 2000+ 轨迹**的数据放大效率惊人，关键在于增强模块和课程式物体扩展。
4. **真实世界迁移**的成功率不错（63-78%），说明数字孪生 + 多样化训练数据的组合是有效的。
5. **局限性**：仍依赖手工设计的奖励函数；缺乏触觉反馈；Minor Adjustment 假设可能不适用于柔性物体操控。
