---
layout: post
title: "PTO 核心软件栈：从 Tile ISA 到 NPU 推理服务"
description: "基于 hw-native-sys 的真实代码，梳理 pto-isa、PTOAS、simpler、pypto、pypto-lib 与 pypto-serving 的职责、接口、依赖关系和当前边界。"
date: 2026-07-26 18:00:00 +0800
category: "NPU · MegaKernel"
series: "MegaKernel"
tags:
  - NPU
  - MegaKernel
  - PTO
  - PyPTO
  - Ascend
  - AI Compiler
reading_time: "约 16 分钟"
math: false
---

> 本文检查了 [`hw-native-sys`](https://github.com/hw-native-sys) 组织的公开仓库，并与本地代码逐仓对照。检查时间为 **2026-07-26**。结论主要来自 README、目录结构、关键入口、构建配置和代码实现；没有在 Ascend 硬件上执行构建或性能测试。

## 1. 先说结论：它不是一个仓库，而是一条纵向软件栈

PTO 生态把 NPU 程序拆成六个核心部件：

```text
pypto-serving
  └─ 请求、连续批处理、KV Cache、权重与模型执行
          ↓
pypto-lib
  └─ Qwen / DeepSeek 等模型 Kernel 与 golden 验证
          ↓
pypto
  └─ Python DSL → 多级 IR / Pass → 两条代码生成路径
          ├─ InCore .pto → PTOAS → pto-isa
          └─ Orchestration C++ → simpler
```

从 MegaKernel 视角看，这条链最重要的价值是：**同一个高层程序可以同时描述设备内的 Tile 计算，以及设备间、核间和 Host/AICPU/AICore 之间的任务编排。**

但需要避免一个误解：这些仓库目前并不等于“把整个模型融合成一个单独二进制 Kernel”。它们更准确的定位，是为跨层融合和大粒度 NPU 程序提供编程、编译与执行基础。

| 层次 | 核心仓库 | 负责什么 | 不负责什么 |
| --- | --- | --- | --- |
| ISA | [`pto-isa`](https://github.com/hw-native-sys/pto-isa) | Tile 指令 API、分代实现、CPU 仿真 | MLIR、调度、模型 |
| Kernel 编译 | [`PTOAS`](https://github.com/hw-native-sys/PTOAS) | PTO MLIR 优化与 lowering | Tensor 级编译、任务运行时 |
| 任务运行时 | [`simpler`](https://github.com/hw-native-sys/simpler) | 任务图、内存、调度、Worker | DSL、模型 Kernel、服务 |
| 高层编译 | [`pypto`](https://github.com/hw-native-sys/pypto) | Python DSL、多级 IR、Pass、代码生成 | ISA 实现、服务层 |
| 模型 Kernel | [`pypto-lib`](https://github.com/hw-native-sys/pypto-lib) | 模型/算子实现与 golden | 编译器、运行时、权重 |
| Serving | [`pypto-serving`](https://github.com/hw-native-sys/pypto-serving) | HTTP、调度、KV Cache、加载与执行 | 底层 Kernel 和编译器实现 |

## 2. `pto-isa`：稳定的 Tile 指令边界

`pto-isa` 是最底层的编程接口。它用模板化 C++ 定义 PTO（Parallel Tile Operation）虚拟 ISA，并为不同 Ascend 代际提供具体实现。

### 主要目录

- `include/pto/common/`：Tile、Tensor、Shape、Stride、内存和事件等公共抽象；
- `include/pto/npu/`：面向不同 NPU 代际的指令实现；
- `include/pto/cpu/`：CPU Simulator；
- `include/pto/comm/`：点对点、信号与集合通信扩展；
- `include/pto/costmodel/`：指令成本模型；
- `kernels/`、`demos/`、`tests/`：手工 Kernel、Auto Mode、示例与跨平台验证；
- `docs/isa/`：90 余条指令的独立文档。

### 技术范围

它覆盖 load/store、vector、cube、reduction、transform、同步、跨核与通信指令，目标包括 Ascend A2/A3、A5，以及 x86_64/AArch64 CPU 仿真。

### 当前边界

- 它定义“单条 Tile 操作怎样写、怎样映射到硬件”，不决定整个模型怎样切图和调度。
- PTOAS 生成的 C++ 会调用这些头文件，simpler 构建 Kernel 时也使用固定版本的 ISA。
- Auto Mode 目前主要服务 CPU 仿真；真实硬件上的高性能路径仍大量依赖手工优化 Kernel。

## 3. `PTOAS`：把 PTO IR 降低为可编译的 Kernel C++

PTOAS 是基于 LLVM/MLIR 的 PTO Assembler & Optimizer。它解析 `.pto`，做语义验证、同步处理、内存规划和面向 Da Vinci/VPTO 的优化，最终生成调用 `pto-isa` 的 C++。

### 主要目录

- `include/PTO/`、`lib/PTO/IR/`：PTO Dialect、TableGen 与 IR；
- `lib/PTO/Transforms/`：优化和 lowering Pass；
- `tools/ptoas/`：主编译工具；
- `tools/ptobc/`：PTO bytecode 工具；
- `lib/Bindings/Python/`、`python/`：Python MLIR binding；
- `ptodsl/`、`tilelang-dsl/`：两套 Python/TileLang 前端；
- `test/`：PTO、C++、Python、VPTO 与上板回归。

### 与其他部件的接口

```text
pypto 生成 .pto
        ↓
PTOAS 解析、优化、lowering
        ↓
生成包含 pto-isa 调用的 C++
        ↓
设备工具链编译为 AICore Kernel
```

### 当前边界

- PTOAS 负责 PTO MLIR 层，不等于 PyPTO 的 Tensor 级多层编译器。
- 它强依赖专用 LLVM 21 VPTO 分支，不能直接当作标准 LLVM 环境下的通用 MLIR 工具。
- 仓库也容纳 PTODSL 与 TileLang DSL，但它们和 PyPTO 是不同前端边界。

## 4. `simpler`：让编译好的任务图在 Host、AICPU 和 AICore 上运行

`simpler` 是 PTO 的任务运行时。它接收编译完成的 orchestration 和 AICore Kernel，构建任务依赖图，管理内存与生命周期，并协调 Host、AICPU 和 AICore。

### 主要目录

- `src/common/`：任务类型、Orchestrator、Scheduler、WorkerManager、Ring、TensorMap 与 Scope；
- `src/{a2a3,a5}/platform/`：真实设备 `onboard` 和线程模拟 `sim` 后端；
- `src/{a2a3,a5}/runtime/`：任务图与 ring buffer 两类运行时；
- `python/simpler/`、`python/bindings/`：Python API 与 nanobind 扩展；
- `simpler_setup/`：runtime/kernel 构建、场景测试与 ISA pin；
- `examples/`、`tests/`、`docs/`：分代示例、单测、板测和架构文档。

### 技术范围

当前 L2 覆盖单芯片 Host/AICPU/AICore 三程序模型；L3 已有多芯片和子 Worker；本地递归 L4 与 socket 远端模拟也已有实现。

### 当前边界

- 它执行程序，不把 Python 模型编译成 Kernel。
- 它拥有任务 DAG、依赖推断、内存作用域、调度和设备后端，但不拥有 PyPTO IR 或模型实现。
- A2 RoCE、A3 HCCS、A5 UB 的生产 HCOMM profile 仍未完成；远端模拟已经存在，不代表对应硬件通信路径已经完备。

## 5. `pypto`：把高层程序拆成 InCore 与 Orchestration

`pypto` 是整条链的高层编译核心。Python DSL 程序先进入多级 IR，再经过 40 余个 Pass 逐步降级，最后分成两条输出：

```text
InCore 路径：Tile / Block IR → PTO MLIR → PTOAS → AICore Kernel

Orchestration 路径：任务与数据依赖 → C++ → simpler runtime
```

### 主要目录

- `python/pypto/language/`：Python DSL、装饰器、parser 与类型表面；
- `include/pypto/ir/`、`src/ir/`：多级 IR、算子、校验器和 Pass；
- `src/codegen/`：PTO、orchestration、distributed、tensor/array codegen；
- `src/backend/{910B,950,common}/`：A2/A3 与 A5 后端；
- `python/pypto/runtime/`：已编译程序和 L2/L3/分布式 runner；
- `runtime/`：固定到 simpler 的 Git 子模块；
- `examples/`、`tests/`、`docs/`：示例、编译单测与硬件系统测试。

### 当前边界

- 它拥有前端语言、多级 IR、Pass 和代码生成。
- 它不实现 PTO MLIR assembler、ISA 头文件或任务运行时核心；`runtime/` 是 simpler 子模块，不是另一份自研 runtime。
- 纯编译能力可以做 CPU 单测，但硬件执行依赖固定的 PTOAS、pto-isa、simpler 与 CANN 环境。

## 6. `pypto-lib`：真实模型 Kernel 与 golden 验证

如果说前四个仓库提供“造路能力”，`pypto-lib` 就是行驶在这条路上的真实工作负载。它用 `pypto.language` 编写算子和完整模型路径，同时用 PyTorch golden 校验数值。

### 主要目录

- `examples/`：matmul、softmax、RMSNorm、RoPE、融合和通信示例；
- `models/qwen3/`：Qwen3 14B 的 prefill/decode/采样，以及 32B 实现；
- `models/deepseek/`：V3.2、V4 Flash、V4 Pro 的 attention、MoE、MTP 等；
- `golden/`：TensorSpec、编译、运行、benchmark 与数值比对；
- `contract/`：模型和 Kernel 参数合同；
- `tests/`、`tools/`、`docs/`：回归、检查工具与调试文档。

### 当前边界

- 它包含可复用 Kernel、模型前向路径和验证工具，不包含编译器、运行时和 HTTP 服务。
- 它不保存模型权重，也不负责通用模型下载。
- 根目录没有独立 `pyproject.toml`，更接近源码模型库与验证工程，而不是通用 pip 包。
- 名为 `_draft.py` 的实现明确处于进行中状态，并被排除在 CI 外。

## 7. `pypto-serving`：把模型 Kernel 变成可调用的推理服务

`pypto-serving` 位于最上层，负责把 PyPTO Kernel 组织成 Ascend 上的本地 LLM 推理服务。当前重点覆盖 Qwen3-14B 与 DeepSeek V4，并提供 OpenAI 兼容的 completion/chat completion HTTP 接口。

### 主要目录

- `pypto_serving/cli/`：命令入口、参数与拓扑校验；
- `serving/engine/`：同步/异步引擎、replica 和请求生命周期；
- `serving/sched/`、`serving/memory/`：连续批处理和分页 KV Cache；
- `serving/server/`：FastAPI、流式返回与多进程 IPC；
- `model/qwen/`、`model/deepseek/`：模型执行、权重布局和加载；
- `worker/`：simpler Worker 封装；
- `platform/`：Meson、HiCR、TaskR 组成的 C++ 平台控制层；
- `pypto-lib/`：固定版本的模型 Kernel 子模块。

### 当前边界

- 服务层拥有 API、请求调度、KV Cache、tokenizer、权重加载与 profiling。
- 它不拥有底层模型 Kernel、PyPTO 编译器或 simpler 的运行时实现。
- 当前是 Qwen3-14B/DeepSeek V4 的专用推理栈，不能直接视为广泛模型兼容的 vLLM/SGLang 替代品。
- C++ `platform/` 仍是平台生命周期和控制通道的初始实现，不位于逐 token 热路径。

## 8. 版本 pin：真正能拼起来的是固定组合，不是所有仓库的最新 `main`

跨仓集成时，最容易犯的错是把六个仓库最新 `main` 直接放在一起。实际依赖通过子模块、提交号或发行版固定：

| 调用方 | 固定依赖 | 检查时的固定值 | 相对依赖仓 `main` |
| --- | --- | --- | --- |
| `pypto` | `simpler` 子模块 | `8cdb306` | 落后 53 个提交 |
| `simpler` | `pto-isa` | `83d0131` | 落后 234 个提交 |
| `pypto-serving` | `pypto-lib` 子模块 | `c364bbc` | 落后 7 个提交 |
| `pypto` | PTOAS 发行包 | `v0.48` | 不直接跟随 PTOAS `main` |

因此复现、调试和性能比较时，应先读取调用方记录的 pin。尤其是 pto-isa：`pypto` 会从 simpler 的 pin 派生 ISA 版本，用来保证编译期与运行期看到同一套指令定义。

## 9. 从问题出发，应该进入哪个仓库

| 要解决的问题 | 首先进入 | 通常联动 |
| --- | --- | --- |
| 新增或修改 Tile 指令 | `pto-isa` | `PTOAS`、`pypto` |
| 新增 PTO MLIR op / lowering | `PTOAS` | `pypto`、`pto-isa` |
| 修改 Tensor/Tile DSL、IR 或 Pass | `pypto` | `PTOAS`、`simpler` |
| 修改任务调度、内存生命周期、设备 runtime | `simpler` | `pypto` orchestration codegen |
| 编写和优化模型 Kernel | `pypto-lib` | `pypto`，必要时继续下沉 |
| 修改 API、batching、KV Cache、权重加载 | `pypto-serving` | `pypto-lib`、`simpler` |

## 10. 本地代码与 GitHub 的对照结果

本地目录中有上述六个核心仓库，以及设计资料仓 `pypto_top_level_documents`；GitHub 组织还多一个仅用于组织主页的 `.github` 仓。

检查时，七个同名本地仓库的 `HEAD` 和 Git 提交树都与 GitHub 一致。但“提交一致”不等于“本地可直接构建”：

1. `pypto`、`pypto-serving`、`PTOAS` 的子模块尚未初始化。
2. `pto-isa` 同时保存只在大小写上不同的三组文档文件，在 macOS 默认的大小写不敏感文件系统上产生工作树碰撞。
3. `pypto_top_level_documents` 中大量内容是 Draft 或提案，判断已实现能力时仍需回到上述代码仓和测试。

## 11. MegaKernel 视角下最值得继续追的三条线

### 11.1 编译器怎样决定 Kernel 与任务边界

重点跟踪 `pypto` 的 IR 和 Pass：哪些算子进入同一个 InCore 区域，哪些依赖被抬升为 orchestration，以及最终生成多少个 AICore Kernel。

### 11.2 运行时能否把细粒度启动开销压到足够低

重点跟踪 simpler 的 task DAG、ring buffer、scope 生命周期和 AICPU 调度。大粒度融合只有与稳定的数据驻留和低开销调度结合，才能形成系统收益。

### 11.3 模型路径能否暴露跨层优化空间

重点从 `pypto-lib` 的 Qwen/DeepSeek 实现向下追踪，判断 attention、MoE、通信与采样怎样穿过 PyPTO IR、PTOAS 和 pto-isa，而不是只在单个算子 benchmark 上看峰值。

---

这六个仓库的边界非常清楚：`pto-isa` 定义指令，PTOAS 编译设备内程序，simpler 执行任务图，pypto 连接高层 DSL 与两条后端路径，pypto-lib 提供模型 Kernel，pypto-serving 提供产品化推理入口。理解这条纵向链，比只盯住某一个“MegaKernel”文件更接近它真正的系统价值。
