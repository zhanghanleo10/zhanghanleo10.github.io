---
layout: post
title: "MPK 论文解读：把 LLM 推理编译成一个 MegaKernel"
description: "梳理 MPK 论文的核心问题、SM 级 tGraph、编译器流水线、GPU 内 Worker/Scheduler 运行时，以及跨算子流水与计算通信重叠为何能够降低推理时延。"
date: 2026-07-27 16:20:00 +0800
category: "GPU · MegaKernel"
series: "MegaKernel"
tags:
  - MPK
  - MegaKernel
  - Persistent Kernel
  - GPU
  - LLM Inference
  - Compiler
  - tGraph
  - CUDA
reading_time: "约 18 分钟"
math: true
---

> 本文只分析论文 **MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs** 的设计与实验结论，暂不展开 Mirage 代码仓的具体实现。论文版本为 arXiv v2，最后修订于 2026 年 6 月 10 日。

论文与项目：

- [论文主页：arXiv 2512.22219](https://arxiv.org/abs/2512.22219)
- [论文 HTML](https://arxiv.org/html/2512.22219)
- [项目仓库：mirage-project/mirage](https://github.com/mirage-project/mirage)

---

## 1. 最短答案

MPK 要解决的核心问题是：

> **传统 LLM 推理以 Operator/Kernel 为调度边界，导致跨算子必须整体同步；MPK 将调度粒度下沉到单个 SM 可执行的 Task，并在一个常驻 MegaKernel 内通过 Event 驱动这些 Task。**

因此，它并不是简单地把很多 CUDA Kernel 拼成一个更大的 Kernel，而是重新定义了整条推理链路的执行方式：

```text
模型计算图
    ↓
每个算子拆成多个 SM 级 Task
    ↓
分析 Task 之间的局部数据依赖
    ↓
构造 Task / Event 组成的 tGraph
    ↓
GPU 内 Worker / Scheduler 持续调度
    ↓
计算、通信、Batch 与 KV Cache 更新连续执行
```

论文的三项关键收益来源是：

1. 减少逐算子的 Kernel Launch；
2. 支持跨 Task、跨算子的软件流水；
3. 将计算和通信以细粒度 Task 进行重叠。

![MPK 将 LLM 推理编译为 MegaKernel 的总体流程]({{ '/assets/images/mpk/mpk-overview.svg' | relative_url }})

图中最重要的变化不是最右侧的“只启动一次”，而是中间两步：**算子先被拆成 SM 级 Task，依赖再从算子级下降到 Task 级。** 后面的 GPU 常驻运行时，正是建立在这个细粒度表示之上。

---

## 2. 传统 kernel-per-operator 为什么限制性能

现有推理框架通常按下面的方式执行一个 Transformer Block：

```text
RMSNorm Kernel
      ↓
QKV GEMM Kernel
      ↓
Attention Kernel
      ↓
Output Projection Kernel
      ↓
AllReduce Kernel
      ↓
MLP Kernels
```

同一个 CUDA Stream 上，相邻 Kernel 之间存在隐式的 Kernel Barrier：后一个 Kernel 启动前，前一个 Kernel 的所有 Thread Block 必须完成。

这种执行模型很清晰，也很容易保证正确性，但会带来三个结构性问题。

### 2.1 全 Kernel Barrier 过于粗糙

假设 MatMul 的输出被划分为多个 Tile：

```text
MatMul Task 0 → 输出 Tile 0
MatMul Task 1 → 输出 Tile 1
MatMul Task 2 → 输出 Tile 2
...
```

后续 AllReduce 的某个 Task 可能只依赖其中一个 Tile。理论上，`MatMul Task 0` 完成后，对应的通信 Task 就可以开始；传统方式却必须等待整个 MatMul Kernel 的所有 Tile 都完成。

也就是说，真实依赖可能是：

```text
MM Tile 0 → AR Tile 0
MM Tile 1 → AR Tile 1
MM Tile 2 → AR Tile 2
```

但 Kernel Barrier 将它强制扩大成：

```text
所有 MM Tile 完成 → 所有 AR Tile 启动
```

这会直接损失计算与通信重叠机会。

### 2.2 软件流水只能停留在单个 Kernel 内部

现代 GPU 中，TMA、Tensor Core 和 CUDA Core 可以并行工作。一个优化良好的 GEMM Kernel 通常已经具备 Kernel 内流水：

```text
加载第 n+1 个 Tile
计算第 n 个 Tile
写回第 n-1 个 Tile
```

问题是，Kernel 边界会截断这条流水。即便当前算子的计算阶段已经接近结束，下一算子的预取也不能提前开始，因此会形成 Pipeline Bubble。

### 2.3 每个 Token 需要大量 Launch 与 CPU 参与

LLM Decode 会重复执行几十层模型。即使每个算子都已经由 cuBLAS、CUTLASS、FlashAttention 或 FlashInfer 高度优化，整条模型仍需要启动大量 Kernel。

论文以 Qwen3-8B 为例：传统 kernel-per-operator 执行每生成一个 Token 需要 **293 次 Kernel Launch**。在 B200 上，论文测得：

| 执行方式 | 单次 Launch 成本 | 每 Token 累积开销 |
| --- | ---: | ---: |
| Eager Launch | 约 3.8 μs | 约 1.1 ms |
| CUDA Graph Replay | 约 0.8 μs | 约 0.2 ms |

CUDA Graph 已经显著降低 Launch 成本，但它没有改变 Kernel 仍是执行和同步边界这一事实。

![传统 kernel-per-operator 与 MPK Task 级执行对比]({{ '/assets/images/mpk/mpk-vs-kernel-per-operator.svg' | relative_url }})

右侧 MPK 方式的关键是：不同 SM 可以同时处于 MatMul、Attention 或 AllReduce 的不同阶段。只要 Task 所依赖的数据已经准备好，就不必等待同一算子的所有其他 Task 完成。

---

## 3. MPK 的核心抽象：SM 级 tGraph

传统计算图的节点是 Operator：

```text
MatMul → Attention → Linear → AllReduce
```

MPK 的 `tGraph` 则将节点下沉为两种对象：

- **Task**：在单个 SM 上执行的一段计算或通信；
- **Event**：Task 之间的同步点。

一个 Task 有两类关系：

```text
dependent event
    ↓
   Task
    ↓
triggering event
```

当 Task 的依赖 Event 被激活时，它可以开始执行；Task 完成后，通知它所触发的 Event。一个 Event 收到所有前驱 Task 的完成通知后，才会被激活并释放下一批 Task。

### 3.1 为什么用 Task，而不是继续用 Kernel

每个 CUDA Thread Block 本来就会被调度到一个 SM 上。MPK 将这一级工作单元显式提升为编译器和运行时都能看见的 Task，从而获得三个能力：

1. 精确描述某个输出 Tile 由哪个 Task 生成；
2. 精确描述某个消费者 Task 读取哪些 Tile；
3. 只在真正存在数据重叠的位置建立依赖。

因此依赖不再是：

```text
Operator A 完成 → Operator B 开始
```

而是：

```text
A 的局部 Task 完成 → B 中真正依赖它的 Task 开始
```

### 3.2 tGraph 与 CUDA Graph 的区别

| 维度 | CUDA Graph | MPK tGraph |
| --- | --- | --- |
| 节点粒度 | Kernel / Memcpy 等 GPU 操作 | 单 SM 执行的 Task 与 Event |
| 依赖粒度 | Kernel 级 | Tile / Task 级 |
| 核心作用 | 降低一串 Kernel 的 Host Launch 成本 | 重构 Kernel 内部的执行、同步和调度 |
| 计算通信重叠 | 受 Kernel 边界约束 | 可按局部数据依赖重叠 |
| 跨算子流水 | 很有限 | tGraph 显式支持 |

可以把 tGraph 理解为比 CUDA Graph 更低一层的执行图：CUDA Graph 记录“启动哪些 Kernel”，tGraph 描述“哪些 SM 级工作现在可以执行”。

---

## 4. 编译器如何从模型图生成 tGraph

MPK Compiler 的主流程可以概括为五步：

```text
Operator Decomposition
        ↓
Dependency Analysis
        ↓
Event Fusion
        ↓
tGraph Normalization
        ↓
tGraph Linearization
```

### 4.1 Operator Decomposition：把算子拆成 SM 级任务

编译器按输出 Tensor 的维度对算子进行切分，每个 Task 计算互不重叠的一块输出。

以矩阵乘为例：

$$
C = A \times B, \quad C \in \mathbb{R}^{M \times N}
$$

可以沿 `M` 和 `N` 方向对 `C` 做二维切分：

```text
Task(0, 0) → C 的第 0 行块、第 0 列块
Task(0, 1) → C 的第 0 行块、第 1 列块
Task(1, 0) → C 的第 1 行块、第 0 列块
...
```

论文默认生成与 GPU SM 数量同量级的 Task，以便给运行时提供足够多的独立工作并改善负载均衡。

在 Qwen3-8B 的编译统计中：

```text
293 个 Operator
       ↓
13,867 个 Task
       ↓
平均每个 Operator 约 47.3 个 Task
```

这里体现了 MPK 的第一层价值：它不是把整张模型图机械塞进一个巨型函数，而是先构造一个拥有充分并行度的 SM 级任务集合。

### 4.2 Dependency Analysis：按 Tensor 区域判断真实依赖

对于共享 Tensor 的生产者和消费者，编译器枚举两侧 Task，并判断：

```text
生产 Task 写出的 Tensor 区域
              与
消费 Task 读取的 Tensor 区域
              是否重叠
```

只有区域重叠时，才插入 Event 依赖。

这一步决定了 MPK 是否真的能突破全 Kernel Barrier。若依赖分析仍停留在整个 Operator 级别，那么即使使用 MegaKernel，也只是在一个 Kernel 内模拟原来的串行执行。

### 4.3 Event Fusion：减少同步节点

直接对每一对生产者/消费者 Task 建 Event，会产生非常多的同步对象。MPK 使用两种融合：

- **Successor-set Fusion**：多个 Event 释放完全相同的一组消费者 Task，则合并；
- **Predecessor-set Fusion**：多个 Event 由完全相同的一组生产者 Task 触发，则合并。

论文统计显示，最终 tGraph 只保留约 1,142～2,366 个 Event，却编码了约 69,000～162,000 对 Task 依赖；Event 数量减少了约 **37～118 倍**。

### 4.4 tGraph Normalization：让 Task 描述保持定长

为了降低运行时判断和内存访问成本，MPK 希望每个 Task：

```text
最多依赖一个 Event
最多触发一个 Event
```

遇到复杂的 Fork/Join 时，编译器会插入空 Task 和辅助 Event，将图转换成规范形式。

这会让每个 Task 的运行时描述非常简单：只需记录一个 `dependent_event` 和一个 `trigger_event`，不需要保存可变长度的 Event 数组。

论文指出，真实模型通常是“深而不宽”的长链结构，Normalization 的额外开销在评测中始终低于 1%。

### 4.5 tGraph Linearization：把事件后继变成连续区间

即使一个 Task 只关联一个前驱和一个后继 Event，一个 Event 仍可能释放很多 Task。如果显式保存所有后继 Task ID，会消耗大量设备内存，并增加不规则访存。

MPK 通过 BFS 风格的线性化过程，让同一个 Event 所释放的 Task 在 Task 数组中连续排列：

```text
Event e
    ↓
Task[first_task_id : last_task_id]
```

于是 Event 只需要保存首尾索引，不必保存显式 Task 列表。论文中，该编码将后继关系的设备内存占用降低了：

- Dense 模型约 4.4～5.9 倍；
- Qwen3-30B-A3B MoE 模型约 15 倍。

---

## 5. GPU 内部运行时：Worker、Scheduler 与 Event

编译阶段生成 tGraph 后，还需要一个运行时持续执行这些 Task。MPK 将运行时直接放进 MegaKernel 内部，并把 SM 划分为两类角色。

### 5.1 Worker

每个 Worker 占用一个物理 SM，维护自己的 Task Queue，并持续执行：

```text
取出 Task
    ↓
检查 dependent event
    ↓
执行计算或通信
    ↓
通知 trigger event
```

Task 可以是矩阵乘、Attention、归一化，也可以是 NVSHMEM 数据传输或本地归约。运行时不再从调度角度区分“计算 Kernel”和“通信 Kernel”，它们都只是拥有不同 Device Function 的 Task。

### 5.2 Scheduler

Scheduler 以 Warp 为单位运行。论文评测中固定保留 4 个 SM、共 16 个 Scheduler Warp，其余 SM 作为 Worker。

Scheduler 的职责是：

1. 轮询已激活的 Event；
2. 找出该 Event 释放的连续 Task 区间；
3. 将 Task 分发给 Worker；
4. 在一轮推理结束时准备下一轮 Batch 和 KV Cache 元数据。

### 5.3 Event Counter

Event 本质上需要记录完成通知次数：

```text
多个前驱 Task
    │ atomic notification
    ▼
Event Counter
    │ 达到 required count
    ▼
释放后继 Task
```

这种事件驱动方式避免使用一个全局中央调度器。多个 Scheduler 使用局部状态分散式分发 Task，降低运行时同步开销。

![MPK GPU 内部 Worker、Scheduler 与 Event 协调流程]({{ '/assets/images/mpk/mpk-runtime.svg' | relative_url }})

这张图需要特别注意第 6 步：Scheduler 不只是调度同一轮 Forward 内的下一算子；当一轮 tGraph 结束后，它还可以整理请求、更新 KV Cache 元数据并启动下一 Token 的推理。因此论文讨论的是一套 **GPU 内持续运行的推理执行系统**，不只是一次性的算子融合。

---

## 6. JIT 与 AOT 混合 Task Launch

如果所有 Task 都等 Event 激活后再由 Scheduler 投递，能够动态负载均衡，但 Scheduler 会进入关键路径；如果所有 Task 都预先分配给 Worker，则调度开销低，但某些 Worker 可能提前拿到尚未 Ready 的 Task并阻塞。

MPK 因此结合两种模式：

### JIT Dispatch

```text
Event 激活
    ↓
Scheduler 动态选择 Worker
    ↓
立即执行 Ready Task
```

优点是动态负载均衡，适合存在运行时间波动的区域。

### AOT Dispatch

```text
运行前将 Task 预先放入 Worker Queue
    ↓
Worker 检查依赖 Event
    ↓
Ready 后直接执行
```

优点是减少 Scheduler 分发成本，适合经过同步边界后、负载较稳定的区域。

Worker 同时维护 JIT 和 AOT Queue，并优先执行已确定 Ready 的 JIT Task；没有 JIT 工作时再检查 AOT Task。这个混合策略在调度开销和负载均衡之间取得折中。

---

## 7. 为什么 MPK 会更快

论文消融实验将收益拆成了三个主要部分。

### 7.1 Kernel Launch Reduction

Qwen3-8B 的传统执行每 Token 启动 293 个 Kernel。MPK 用一个 MegaKernel 覆盖整条推理链路，避免这些重复 Launch。

更重要的是，论文中 GPU 内 Scheduler 只占总运行时间的约 **0.28%**。这说明它没有用高昂的设备端调度成本换取 Launch 减少。

### 7.2 Cross-task Software Pipelining

MPK 将 Task 进一步拆成：

```text
Pre-loading Phase
Compute Phase
```

当当前 Task 已完成自己的内存传输，并且还有足够 Shared Memory 时，Worker 可以在当前 Task 计算的同时，提前为下一个 Task 预取数据。

论文在 B200 上对 Qwen3-8B 最后一层 Linear 做消融：跨 Task 流水带来约 **1.2～1.3 倍**的 Task 加速。

为了支持这种机制，MPK 还引入 Paged Shared Memory：Shared Memory 被划分为固定大小 Page，Task 按需申请和释放，使两个相邻 Task 的不同阶段能够在同一个 Worker 上短暂共存。

### 7.3 Fine-grained Compute–Communication Overlap

在 Tensor Parallel 场景，Attention 和 MLP 后常跟 AllReduce。MPK 将集体通信拆成：

```text
跨 GPU 数据传输 Task
        +
本地归约 Task
```

这些通信 Task 与普通计算 Task 一起进入 tGraph，由同一套 Event 依赖控制。只要局部输入 Tile 已完成，就可以开始通信，不再等待整个前序算子。

论文在 4 张 H100、Qwen3-1.7B 上的消融显示，细粒度计算通信重叠将每轮延迟进一步降低约 **1.1 倍**。

---

## 8. 动态 LLM Serving 如何进入 MegaKernel

MegaKernel 容易给人一种“只能处理静态 Shape”的印象，但 LLM Serving 具有明显动态性：

- 请求会到达和结束；
- Batch Size 会变化；
- Prefill 与 Decode 工作量不同；
- KV Cache Page 需要分配和回收。

论文的处理方式包括：

### 8.1 将连续批处理和 KV Cache 更新放入 GPU

每轮 tGraph 开始时，Scheduler 执行一次准备逻辑：

```text
移除上一轮完成请求
        ↓
接纳新到达请求
        ↓
更新请求的 KV Cache Metadata
        ↓
开始新的推理迭代
```

KV Cache 元数据保存在 Device Memory 中，Attention Task 可以直接访问，从而减少 CPU 与 GPU 的往返同步。

### 8.2 为代表性 Batch Size 生成多个 tGraph

编译器为 1、2、4、8……等代表性 Batch Size 生成专门的 tGraph；运行时根据当前 Batch 选择合适版本。

因此 MPK 并不是完全依赖一个能够覆盖任意动态 Shape 的通用图，而是通过多版本特化，在编译优化和运行时灵活性之间平衡。

---

## 9. 实验结果应该怎样理解

论文评估覆盖 A100、H100 和 B200，模型包含 Dense 与 MoE 架构，并与 PyTorch、SGLang、vLLM 进行比较。

### 9.1 端到端结果

对于 Batch Size 为 1 的推理，MPK 相对最佳基线取得约 **1.0～1.7 倍**的性能提升。收益在以下场景更明显：

- 模型较小，每 Token 计算量相对低；
- GPU 更新、更快，固定调度开销占比更高；
- 对单 Batch、低时延 Decode 更敏感。

论文给出的一个具体例子是 Qwen3-8B 在 A100 上：

```text
vLLM / SGLang：约 14.5 ms / token
MPK：          约 12.5 ms / token
估算硬件下界：约 10 ms / token
```

### 9.2 多 GPU 结果

在最多 8 张 H100 的 Tensor Parallel 场景中：

- 相对 PyTorch，吞吐最高提升约 10 倍；
- 相对 SGLang 和 vLLM，提升约 1.1～1.4 倍。

这里的主要优势来自：

1. Page Allocation 和请求调度进入 MegaKernel；
2. 计算 Task 与通信 Task 异步重叠；
3. 消除 Kernel Barrier，支持跨 Task 流水。

### 9.3 编译阶段统计

| 模型 | Operator 数 | 平均 Tasks/Op | 最终 Event 数 | Event Fusion | Linearization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 229 | 35.6 | 1,870 | 37× | 4.4× |
| Qwen3-8B | 293 | 47.3 | 2,366 | 68× | 5.9× |
| Qwen3-30B-A3B | 533 | 32.2 | 1,142 | 118× | 15.0× |

这组数据说明：如果没有 Event Fusion 和 Linearization，SM 级依赖虽然能够表达更多并行性，但同步对象和图存储本身可能失控。编译器的图优化不是附属功能，而是让细粒度运行时可落地的关键。

---

## 10. 论文结果的适用边界

阅读性能数字时，需要同时看到实验条件和系统代价。

### 10.1 论文主要使用受控的离线批处理实验

端到端实验统一使用：

- Prompt Length：64；
- 输出长度：1024；
- BF16；
- 最大 Batch Size：1～16；
- 离线 Batched Inference。

这有利于稳定比较执行系统，但不等价于所有真实在线流量。突发到达、复杂采样、Prefix Cache、超长 Context 或更大的 Prefill 比例，可能改变收益组成。

### 10.2 Scheduler 会占用部分 SM

论文在所有 GPU 上保留 4 个 SM 运行 16 个 Scheduler Warp。对于大模型，4 个 SM 占比不高；对于极小模型或不同硬件，Worker/Scheduler 比例仍需要调优。

### 10.3 Register 资源由最重 Task 决定

Task 和 Shared Memory 可以按时间复用，但 MegaKernel 的每线程 Register 上限需要覆盖所有 Task 类型中的最大需求。若某个 Task Register 压力过高，可能影响整个 MegaKernel 的 Occupancy。

### 10.4 编译结果依赖推理配置

MPK Compiler 会针对以下信息进行特化：

- GPU 架构；
- 模型 Shape；
- Batch 代表值；
- 并行配置；
- Task Partition；
- 通信方式。

这种特化是获得高性能的重要原因，也意味着它不是完全无需编译成本和配置管理的通用运行时。

---

## 11. 我认为这篇论文最重要的三个观点

### 观点一：推理优化的边界从 Operator 下沉到了 SM Task

过去很多工作是在寻找“更快的 Attention Kernel”“更快的 GEMM Kernel”。MPK 的目标则是优化：

```text
整个模型中，不同 SM 在每个时刻应该执行什么
```

这是从算子优化转向执行系统优化。

### 观点二：计算与通信不再属于两套运行时

传统方案通常由 CUDA Kernel 负责计算、NCCL/NVSHMEM Kernel 负责通信、CPU/Stream 负责协调。MPK 把计算与通信统一成 Task，使重叠不再是某几个手工设计 Kernel 的特例，而成为全局 Task Schedule 的自然结果。

### 观点三：MegaKernel 的难点不是“融合”，而是“可调度”

把大量代码放进同一个 Kernel 并不自动等于高性能。真正困难的是：

- 怎样拆分任务；
- 怎样表达依赖；
- 怎样减少同步对象；
- 怎样保持设备端图表示紧凑；
- 怎样在不阻塞 Worker 的前提下动态分发任务；
- 怎样管理跨 Task 的 Shared Memory 和预取。

MPK 的贡献正是在编译器和运行时两侧同时解决这些问题。

---

## 12. 总结

MPK 可以概括成下面一句话：

> **将 Tensor Program 编译成 SM 级 Task/Event 图，并由 GPU 内的常驻并行运行时持续执行。**

它相对传统 kernel-per-operator 的根本变化包括：

```text
Kernel 级同步
    → Task 级同步

逐算子 Host Launch
    → GPU 内持续事件驱动

计算 Kernel + 通信 Kernel 分离
    → 计算/通信统一 Task 化

算子内部流水
    → 跨算子、跨 Task 流水
```

论文展示的最高 1.7 倍端到端收益并不是来自某一个神奇算子，而是来自整个执行模型的改变：减少 Launch、消除不必要的全局 Barrier、暴露跨算子并行，并把部分请求与 KV Cache 调度搬到 GPU 内部。

下一篇将继续进入 Mirage/MPK 代码仓，沿着：

```text
Python PersistentKernel
    → KNGraph / TBGraph
    → Task 注册与代码生成
    → task_graph.json
    → Worker / Scheduler Device Runtime
```

逐段对照论文中的设计如何落到实际代码。

---

## 参考资料

1. Xinhao Cheng et al. [MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://arxiv.org/abs/2512.22219), arXiv v2, 2026.
2. [MPK 论文 HTML 全文](https://arxiv.org/html/2512.22219).
3. [Mirage / MPK GitHub Repository](https://github.com/mirage-project/mirage).
