---
layout: post
title: "simpler Runtime 解读：PTO 体系中的 Task DAG、AICPU 调度与 AIC/AIV Worker"
description: "系统梳理 hw-native-sys/simpler 的定位与实现：从两层 Orchestrator/Scheduler、TensorMap 自动依赖、PTO2 Ring Buffer，到 Early Dispatch、sync_start、多卡层级与它和 MPK、TileRT 的关系。"
date: 2026-07-28 00:40:00 +0800
category: "NPU · MegaKernel"
series: "MegaKernel"
tags:
  - simpler
  - PTO
  - PyPTO
  - Ascend
  - AICPU
  - AICore
  - Task Graph
  - Runtime
  - MegaKernel
reading_time: "约 22 分钟"
math: false
---

> 本文分析的是 PTO 软件栈中的 **`simpler` Runtime**，不是大模型推理中的 Sampling/Sampler。检查日期为 **2026-07-28**，主要基于 `hw-native-sys/simpler`、`pypto` 和 `pypto-lib` 当前公开代码与文档；其中 capability survey 自身标注的代码快照为 `main @ e6a61f15`。

相关仓库：

- [`hw-native-sys/simpler`](https://github.com/hw-native-sys/simpler)
- [`hw-native-sys/pypto`](https://github.com/hw-native-sys/pypto)
- [`hw-native-sys/pypto-lib`](https://github.com/hw-native-sys/pypto-lib)
- [`hw-native-sys/pto-isa`](https://github.com/hw-native-sys/pto-isa)
- [`hw-native-sys/PTOAS`](https://github.com/hw-native-sys/PTOAS)

---

## 1. 最短答案

`simpler` 是 PTO 体系中的**运行时控制面**。

它负责把上层编译器生成或运行时提交的 Task 组织成 DAG，然后在 Host、AICPU 和 AICore 之间完成：

```text
任务提交
    ↓
根据 Tensor 读写关系推导依赖
    ↓
将 Ready Task 分发给合适的 AIC / AIV
    ↓
接收完成通知并释放下游 Task
    ↓
回收 Task Slot 和中间 Tensor 内存
```

它不负责 PTO Tile 指令本身、单个 Kernel 的计算逻辑、Tensor Graph 编译和 Serving 请求管理。这些职责分别属于 `pto-isa`、`PTOAS`、`PyPTO`、`pypto-lib` 和 `pypto-serving`。

从技术定位看：

> **simpler 是 PTO 体系里最接近 MPK Runtime 的组件。**

但它不是简单的 Ascend 版 MPK。它充分利用了 AICPU、AIC、AIV 的异构结构，并采用 TensorMap 自动依赖、Ring/Scope 生命周期和分层 Worker，而不是直接复刻 GPU 上的 Task/Event 模型。

---

## 2. simpler 在 PTO 软件栈中的位置

![simpler 在 PTO 软件栈中的位置]({{ '/assets/images/simpler/simpler-stack.svg' | relative_url }})

整个软件栈可以粗略理解为：

```text
pypto-serving
    请求、Batch、KV Cache、模型服务

pypto-lib
    Qwen3 / DeepSeek 等模型实现与 Kernel 组合

PyPTO
    Tensor Graph → Tile / Block / Execution Graph

simpler
    Task DAG、调度、Worker、生命周期、Host↔Device 协作

PTOAS / PTO ISA
    Tile IR 优化、代码生成、计算/搬运/同步/通信

Ascend Hardware
```

`simpler` 自己拥有的职责包括：

- DAG submission；
- Host/AICPU Scheduler；
- Worker 和设备生命周期；
- Host↔Device handshake；
- Profiling、Dump、PMU 和依赖图等 DFX。

因此讨论“PTO 能否形成类似 TileRT 的端到端执行系统”时，不能只看 PTO ISA。更完整的对应关系是：

```text
TileRT Whole-model Compiler  ≈  PyPTO
TileRT Runtime Engine        ≈  simpler
TileRT Tile Kernel Backend   ≈  PTOAS + PTO ISA
TileRT Model Implementation  ≈  pypto-lib
```

---

## 3. 为什么 simpler 是关键拼图

仅有高性能 Tile Kernel，并不能自动形成低延迟整模型系统。

即使已经具备：

```text
RMSNorm
QKV GEMM
RoPE
Paged Attention
AllReduce
SwiGLU
Top-K
```

仍然需要解决：

1. 哪些 Task 可以立即执行；
2. 哪些 Task 必须等待某个 Tensor 的生产者；
3. Task 应占用 AIC、一个 AIV、两个 AIV，还是完整 AIC+AIV Cluster；
4. 中间 Tensor 何时分配、何时回收；
5. Task 数量超过固定描述符容量时如何流式推进；
6. AICPU 如何把 Task 交给 AICore；
7. 多卡、子进程和 Host 任务如何纳入统一 DAG；
8. 如何记录运行时性能与依赖关系。

这正是 simpler 解决的问题。

```text
PTO ISA：Task 内部如何高效执行
simpler：Task 之间如何正确、高效地推进
```

---

## 4. simpler 实际有两层调度

simpler 不只有 AICPU Scheduler，而是至少存在两个运行时层级。

### 4.1 L3+：Host 层调度

在单机多卡或更高层级上：

```text
Python Orchestration Function
        ↓
C++ Orchestrator
        ↓
C++ Scheduler Thread
        ↓
ChipWorker / SubWorker / 下一级 Worker
```

Host 层负责：

- 将多个 ChipWorker 组织成 DAG；
- 定向调度指定设备；
- 调度 Python SubWorker；
- 通过进程和共享内存 Mailbox 分发任务；
- 递归组合 L3、L4 等 Worker。

### 4.2 L2：AICPU 层调度

进入单张 Ascend 设备后：

```text
AICPU Orchestrator
        ↓
AICPU Scheduler Threads
        ↓
AIC / AIV Worker
```

L2 负责芯片内部 Task DAG、Tensor 依赖、Ring、资源匹配和完成通知。

因此更准确的层次是：

```text
L3 Host Runtime：调度哪张卡或哪个下级 Worker
L2 AICPU Runtime：调度卡内哪个 AIC/AIV 执行哪个 Task
L0 AICore：执行具体 PTO Kernel
```

---

## 5. L2 的三程序模型

simpler 的单卡 Runtime 由三个独立编译的程序协作：

```text
Host Runtime .so
AICPU Runtime .so
AICore Program .o
```

### Host Runtime

Host 侧负责设备绑定、内存分配、数据拷贝、Callable 上传、AICPU/AICore Binary 加载、启动和结果回收。

### AICPU Runtime

AICPU 是 Task 控制面：

- Orchestrator 创建 Task；
- TensorMap 建立 Producer 依赖；
- Scheduler 判断 Task 是否 Ready；
- Scheduler 选择符合资源要求的 AIC/AIV；
- 完成后释放下游 Consumer。

### AICore Program

AIC/AIV 类似常驻 Worker：

```text
等待 Handshake / DATA_MAIN_BASE
    ↓
读取 PTO2DispatchPayload
    ↓
根据 kernel_id 找到函数地址
    ↓
执行 PTO Kernel
    ↓
写 ACK / FIN
    ↓
继续等待下一 Task
```

生产路径并不是每个细粒度 Task 都回到 Host 重新发起公开 CANN Kernel Launch。Host 启动通用 Executor 后，AICPU 通过共享内存和寄存器向 Core 投递函数与参数。

---

## 6. 两种 L2 Runtime

| 维度 | `host_build_graph` | `tensormap_and_ringbuffer` |
| --- | --- | --- |
| 图构建位置 | Host CPU | AICPU |
| Task 存储 | 固定 Task 数组 | Task Ring |
| 依赖 | 显式预构建 | TensorMap 自动推导 |
| 中间内存 | Host 侧规划 | GM Heap Ring |
| 构图与执行重叠 | 不支持 | 支持 |
| Streaming / Back-pressure | 不支持 | 支持 |
| 定位 | 开发和调试 | 生产路径 |

### `host_build_graph`

Host 先构建完整 DAG，再整体上传给设备。它更容易调试，但无法边构图边执行，也不适合特别大的动态任务流。

### `tensormap_and_ringbuffer`（TRB / PTO2）

生产模式把 Orchestration 放到 AICPU：

```text
AICPU 提交一批 Task
    ↓
Scheduler 开始执行
    ↓
旧 Task 完成和回收
    ↓
Orchestrator 继续提交新 Task
```

它支持流式构图、自动依赖、固定容量 Ring 复用、Scope 生命周期和 Back-pressure。

---

## 7. PTO2 / TRB 的核心结构

![simpler L2 生产运行时]({{ '/assets/images/simpler/simpler-l2-runtime.svg' | relative_url }})

### TaskRing

Task ID 单调递增，但物理槽位映射到有限 Ring：

```text
task_slot = task_id & (window_size - 1)
```

当活动任务接近窗口容量，Orchestrator 等待 Scheduler 推进 `last_task_alive`，从而形成背压。

### HeapRing

Task 可以从 GM Heap 创建中间输出。当所有 Consumer 和 Scope 引用释放后，HeapRing 才回收对应空间。

### DepPool

保存 Fanin/Fanout 边。Producer 完成后，Scheduler 增加 Consumer 的 `fanin_released`；全部 Fanin 满足后，Consumer 进入 Ready Queue。

### TensorMap

维护：

```text
Tensor Region → 当前 Producer Task
```

它让上层无需手工为大部分数据依赖构造 Event。

### Multi-Ring

TaskRing、HeapRing 和依赖资源会按 Scope 深度映射到最多四组独立 Ring，避免外层长生命周期 Task 阻塞内层短生命周期临时 Tensor 的 FIFO 回收。

---

## 8. TensorMap 如何推导依赖

| Tag | 查找上一 Producer | 注册当前 Task 为新 Producer |
| --- | --- | --- |
| `INPUT` | 是 | 否 |
| `OUTPUT` | 否 | 是 |
| `INOUT` | 是 | 是 |
| `OUTPUT_EXISTING` | 否 | 是 |
| `NO_DEP` | 否 | 否 |

例如：

```text
Task A: OUTPUT X
Task B: INPUT X
Task C: INPUT X
```

自动形成：

```text
       ┌──> Task B
Task A ┤
       └──> Task C
```

### 能自动表达的依赖

- RAW：Consumer 读取 Producer 输出；
- 当前 Producer 替换；
- 同一 Buffer 子区域的覆盖与重叠。

### 需要谨慎的依赖

TensorMap 不是完整的内存别名分析器：

- Writer 之间不一定自动形成想要的 WAW 顺序；
- Write-after-Read Anti-dependency 不会完整自动推导；
- 复杂覆盖应使用 `INOUT`、显式依赖或更准确的 Buffer 设计。

所以 TensorMap 更像**运行时 Producer Map**，而不是自动解决所有内存冲突。

---

## 9. Task 状态与生命周期

Host L3+ Runtime 的 Task 状态可以概括为：

```text
FREE → PENDING → READY → RUNNING → COMPLETED/FAILED → CONSUMED → FREE
```

这里必须区分：

```text
Task 执行完成 ≠ Task 资源可以立即回收
```

`COMPLETED` 仅表示 Kernel 已结束；`CONSUMED` 才表示所有下游和 Scope 引用均已释放，Task Slot 与中间 Buffer 可以回收。

### Scope

```cpp
PTO2_SCOPE() {
    A = submit(...)
    B = submit(A)
}
```

Scope 为 Task 增加生命周期引用。离开 Scope 后，若所有 Consumer 也完成，资源即可回收。

需要注意：`TaskOutputTensors` 及其引用不能逃逸出创建它的 Scope，否则 Ring Slot 复用后可能静默指向另一个 Task 的 Tensor 描述。

---

## 10. AICPU Scheduler 是资源感知的

| Resource Shape | 使用资源 | 典型任务 |
| --- | --- | --- |
| `AIC_ONLY` | 1 个 AIC | GEMM / Cube |
| `AIV_X1` | 1 个 AIV | 小型 Vector Task |
| `AIV_X2` | 2 个 AIV | 更宽 Vector Task |
| `AIC_AIV_X1` | AIC + 1 AIV | 混合流水 |
| `AIC_AIV_X2` | AIC + 2 AIV | 完整 Cluster 协同 |

Scheduler 先处理 Completion，再从按 Resource Shape 分类的 Ready Queue 中匹配空闲 AIC/AIV，并通过寄存器 Doorbell 启动 Core。

相较于 MPK 的统一 SM/CTA Worker，simpler 的资源模型更贴合 Ascend 的异构硬件。

---

## 11. Early Dispatch：提前准备 Consumer

普通路径是：

```text
Producer 完成
    ↓
Consumer 变 Ready
    ↓
Scheduler 查找空闲 Core
    ↓
分发 Consumer
```

Early Dispatch 允许尚未正式 Ready 的 Consumer 提前占用 Pending Slot 并准备 Payload：

```text
Consumer 依赖尚未释放
    ↓
提前准备 Core Slot 与 Payload
    ↓
Producer 完成
    ↓
Doorbell 立即启动 Consumer
```

它并不允许 Consumer 提前执行，只是提前完成资源准备，以减少 Producer 完成后的调度气泡。

为了避免下游提前占满所有 Core、导致上游剩余 Block 无法发布，Producer 必须先完成全部 Block 的发布，才可以向下游传播 Early Candidate。

---

## 12. `sync_start` 与 MIX Task

### `sync_start`

一个 SPMD Cohort 需要多个 Core 同时启动时，simpler 会：

1. 检查完整 Core 集合；
2. All-or-nothing 地准备全部 Core；
3. 等待全部 Payload 和依赖就绪；
4. 统一 Doorbell 启动。

它不会先启动部分 Core，再等待另一部分。

### MIX Task

一个逻辑 Task 可以同时使用：

```text
1 AIC + 1 AIV
```

或：

```text
1 AIC + 2 AIV
```

Scheduler 为每个 Core 构造 Subtask，只有全部必需 Subtask 完成，整个逻辑 Task 才算完成。这使 AIC 承担矩阵计算、AIV 承担预处理/后处理/归约成为可能。

这与 TileRT 的 Heterogeneous Worker 有相似之处，但硬件层级不同：TileRT 常强调 Warp/Block 专门化，simpler 强调 Ascend Cluster 内的 AIC/AIV 异构分工。

---

## 13. L3+ 分层 Runtime

simpler 定义 L0–L6：

```text
L6 Cluster
L5 SuperNode
L4 Pod
L3 Host
L2 Chip
L1 Die
L0 Core
```

当前成熟状态：

| 层级 | 状态 |
| --- | --- |
| L2 单卡 | 已实现并用于生产 Runtime |
| L3 单机多卡 | 已实现 |
| L4 本地递归 | 已实现 |
| L4 远程控制面 | Socket Simulation 已实现 |
| L4 远程硬件数据面 | 尚未闭环 |
| L5 / L6 | 复用代码路径，缺少充分验证 |

L3+ 由 Orchestrator、Scheduler 和 WorkerManager 组成。ChipWorker/SubWorker 通常在预先 Fork 的子进程中，通过共享内存 Mailbox 接收 Callable Digest、CallConfig 和 TaskArgs Blob。

### L3 调度限制

NEXT_LEVEL Task 是定向 FIFO：

- 不支持 Work Stealing；
- 不会自动迁移到其他兼容 Worker；
- 没有优先级和 Aging；
- Group FIFO 可能出现队头阻塞；
- Group Task 要求完整目标集合同时空闲。

因此上层编译器或 Runtime 需要明确知道 Task 应放到哪张卡。

---

## 14. 当前通信能力

| 能力 | 当前状态 |
| --- | --- |
| Counter Completion | A2/A3/A5 已实现并有 CI |
| A2/A3 SDMA | 已实现，运行时显式开启 |
| A5 SDMA | 当前默认未构建 |
| A5 URMA | 有实现，但公开路径被宏门控 |
| ROCE / CCU | 仅有枚举或名称 |
| HCCL | 主要用于初始化、Barrier 和销毁 |
| Collective 数据面 | 当前多由手写 AIV Kernel 完成 |
| 跨 Host L4 数据面 | 设计/模拟为主 |

所以 simpler 已经具备通信 Task 进入 DAG 的基本框架，但要达到 TileRT 式极致跨卡流水，通信 Backend 仍是明显短板。

---

## 15. simpler、MPK 与 TileRT

![simpler、MPK 与 TileRT 对比]({{ '/assets/images/simpler/simpler-mpk-tilert.svg' | relative_url }})

| 维度 | simpler | MPK | TileRT |
| --- | --- | --- | --- |
| 核心抽象 | TensorMap + Task DAG + Resource Shape | Task + Event 的 tGraph | 静态 Tile Pipeline |
| 调度位置 | Host + AICPU | GPU 内 Scheduler | 尽量前移到 AOT 编译期 |
| Worker | Chip/SubWorker；AIC/AIV | SM/CTA | Warp/Block/GPU 异构 Worker |
| 依赖生成 | 运行时 Tensor 读写推导 | 编译期显式 Event Graph | 编译期流水和 Barrier |
| 动态性 | 强 | 中到强 | 相对弱，偏特化 |
| 异构资源 | AIC/AIV/MIX | 统一 SM 模型为主 | 强调 Worker Specialization |
| 中间内存 | GM Heap Ring + Scope | Task Buffer/共享内存优化 | 强调 Register/Shared/L2 驻留 |
| 主要目标 | 通用 Ascend Task Runtime | 通用 MegaKernel Runtime | 固定模型极低 TPOT |

### simpler 与 MPK

可以直观对照：

```text
MPK Event Counter
    ≈ simpler Fanin/Fanout Counter

MPK TaskDesc
    ≈ PTO2TaskDescriptor + Payload

MPK Worker Queue
    ≈ Resource-shape Ready Queue

MPK GPU Scheduler
    ≈ simpler AICPU Scheduler
```

MPK 的图更偏编译期显式生成，利于 Event 融合和全局优化；simpler 的 TRB 可以在 AICPU 上根据 Tensor 参数动态推导依赖，更适合程序化动态图和流式提交。

### simpler 与 TileRT

相似点包括：

- AIC/AIV 异构 Worker；
- MIX Task；
- Early Dispatch；
- `sync_start` Cohort；
- 构图和执行重叠；
- PTO Kernel 内可进一步实现 Tile Pipeline。

但 simpler 本身是通用动态 Runtime，不会自动生成 TileRT 那种高度模型特化的整机流水计划。

---

## 16. simpler 还不是完整的 Persistent Model Engine

一个 simpler Run 内可以边构图、边执行、边回收，但它和永久驻留的 Token Engine 仍有区别。

### 外部 Run 仍是同步边界

```text
准备输入
启动/驱动 Runtime
等待完成
结果拷回
清理 Run 状态
```

### 多个 Run 不能真正重叠

L3 `submit()` 可以返回 `RunHandle`，但后续 Run 当前仍需要等待前一个 Run 的 Fence 和清理完成。

```text
同一 Run 内 Task Streaming：支持
多个 Run 并发执行：当前不足
```

### Serving 状态属于上层

simpler 不直接拥有 Request Queue、Continuous Batching、KV Block Allocator、Prefix Cache、Sampling、MTP 接受状态和 Token 生命周期，这些应由 `pypto-serving` 或模型 Runtime 管理。

---

## 17. 当前主要限制

### 默认整卡占用

L2 Run 默认使用设备全部可见 AIC/AIV，公开 `CallConfig` 没有通用的每次调用 `block_dim` 选择接口。这利于低延迟专用部署，但不利于同卡多模型和设备共享。

### AICPU 调度有固定成本

AICPU 需要读取 Descriptor、轮询 Core、写 MMIO Doorbell、处理 Completion、遍历 Fanout 和推进 Ring 水位。如果 Task 太小，控制开销会显著。

因此合理 Task 粒度应是：

```text
具有足够计算量，同时能够暴露跨算子并行的 Tile/Block Region
```

### Ring 配置可能造成死锁

若单个 Scope 内提交的 Task 数量超过 TaskRing 窗口，Orchestrator 可能等待空 Slot，而旧 Task 又等待 Scope End 释放引用，从而形成环形等待。

工程上应保证：

```text
task_window_size > 单个最大 Scope 的 Task 数量
```

### Runtime 隔离

由于当前 CANN AICPU Framework 会缓存首次加载的用户 AICPU `.so`，同一进程上下文中不应在同一设备上切换不同 simpler Runtime；需要独立进程或设备分区。

### 通信尚未闭环

单机多卡控制面已经存在，但部分异步 Engine 不可达，远程 L4 数据面未完成，通信与计算的自动全局流水仍需要编译器共同规划。

---

## 18. 对 PTO 达到 TileRT 效果的意义

有了 simpler 后，PTO 已经具备构建类似 TileRT 系统的大部分基础分层：

```text
PyPTO
    整图编译与 Task 划分

simpler
    Task DAG、AICPU 调度、Worker 和生命周期

PTOAS / PTO ISA
    Tile Kernel、同步、流水和通信

pypto-lib
    模型特化实现

pypto-serving
    请求、KV Cache、Batch 与服务接口
```

真正的差距不再是“PTO 没有 Runtime”，而是：

1. PyPTO 能否生成低气泡的整模型 Task Plan；
2. simpler 能否把多个 Token/Request 组织成长期运行状态机；
3. PTO Kernel 能否在 Task 内保持更强数据驻留；
4. 多卡通信 Backend 能否稳定进入流水；
5. 模型、拓扑和 Worker 角色能否自动或半自动特化。

> simpler 已补齐 PTO 体系最关键的运行时控制面，但要达到 TileRT 展示的极致端到端 TPOT，还需要 PyPTO、pypto-lib、pypto-serving 和通信栈共同闭环。

---

## 19. 推荐演进方向

### 编译器显式输出 Runtime Hint

PyPTO 可以向 simpler 输出：

- Resource Shape；
- Early-dispatch eligibility；
- `sync_start` Cohort；
- Task 优先级；
- Compatible Worker Set；
- Buffer 生命周期；
- 通信 Engine 偏好；
- Pipeline Region 边界。

### 两级任务粒度

不应把每条极小 Tile 指令都变成 simpler Task，更合理的是：

```text
simpler Task = 可动态调度的 Region
Region 内部 = PTO Tile Pipeline
```

例如：

```text
Task A：RMSNorm + QKV 局部流水
Task B：Paged Attention Region
Task C：Output Projection + AllReduce Region
```

全局由 simpler 动态调度，局部由 PTO Kernel 静态流水。

### 增加长期驻留 ModelSession

在 Run 之上增加：

```text
ModelSession
    ├── 常驻 Worker
    ├── 请求 Doorbell / Queue
    ├── KV Cache 状态
    ├── Token Iteration
    ├── Sampling / MTP
    └── Shutdown / Recovery
```

### 完善通信 Task

需要形成统一的 Compute Task、Comm Task、Async Completion 和 Early Dispatch，并让编译器可靠规划 Tile 发送、Consumer 提前占槽和通信 Buffer 生命周期。

---

## 20. 推荐阅读顺序

1. [`README.md`](https://github.com/hw-native-sys/simpler/blob/main/README.md)
2. [`docs/chip-level-arch.md`](https://github.com/hw-native-sys/simpler/blob/main/docs/chip-level-arch.md)
3. [`docs/hierarchical-level-runtime.md`](https://github.com/hw-native-sys/simpler/blob/main/docs/hierarchical-level-runtime.md)
4. [`docs/orchestrator.md`](https://github.com/hw-native-sys/simpler/blob/main/docs/orchestrator.md)
5. [`docs/scheduler.md`](https://github.com/hw-native-sys/simpler/blob/main/docs/scheduler.md)
6. [`docs/task-flow.md`](https://github.com/hw-native-sys/simpler/blob/main/docs/task-flow.md)
7. [`src/a2a3/docs/runtimes.md`](https://github.com/hw-native-sys/simpler/blob/main/src/a2a3/docs/runtimes.md)
8. [`RUNTIME_LOGIC.md`](https://github.com/hw-native-sys/simpler/blob/main/src/a2a3/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md)
9. [`docs/capability-survey.md`](https://github.com/hw-native-sys/simpler/blob/main/docs/capability-survey.md)

---

## 21. 最终总结

simpler 不是 PTO 的附属 Launcher，而是一个完整度已经较高的 Ascend Task Graph Runtime。

它的核心价值可以概括为：

1. **双层调度**：Host 层管理多卡/子任务，AICPU 层管理 AIC/AIV；
2. **自动依赖**：TensorMap 从 Tensor 读写关系构建 DAG；
3. **流式执行**：TaskRing、HeapRing、DepPool 支持边构图、边执行、边回收；
4. **异构资源**：AIC、AIV、MIX 和 `sync_start` 贴合 Ascend 硬件；
5. **低气泡能力**：Early Dispatch 提前准备 Consumer，减少依赖释放后的调度延迟。

它最准确的定位是：

> **Ascend 上的分层、异构、动态 Task Graph Runtime。**

与 MPK 相比，它更依赖 AICPU 和 TensorMap，异构资源表达更强；与 TileRT 相比，它更通用、更动态，但整模型静态流水、数据驻留和通信闭环仍不够激进。

长期最合理的组合是：

```text
PyPTO / simpler
    提供全局动态图和请求适配能力

PTO Kernel
    在稳定 Region 内采用 TileRT 风格静态流水

pypto-serving
    管理请求、KV Cache、Batch 和 Token 生命周期
```

这条路线既能保留 Serving 的动态性，也有机会把关键 Decode 路径压缩到接近硬件极限。
