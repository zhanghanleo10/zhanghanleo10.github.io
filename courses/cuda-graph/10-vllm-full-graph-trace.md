---
layout: post
title: "CUDA Graph 源码课程 10：只追踪一张真实 FULL Graph"
description: "沿 vLLM v0.22.1 的 capture_model、dummy run、ForwardContext 与 CUDAGraphWrapper，追完 capture 到 replay。"
date: 2026-08-16 10:00:00 +0800
category: "vLLM · CUDA Graph 源码课程"
series: "vLLM CUDA Graph 源码课程"
tags: [vLLM, CUDA Graph, Full Graph, CUDAGraphWrapper, Source Code]
reading_time: "约 20 分钟"
mermaid: true
permalink: /courses/cuda-graph/10-vllm-full-graph-trace/
---

上一课：[Graph-Safety Harness](/courses/cuda-graph/09-graph-safety-harness/) · [课程目录](/courses/cuda-graph/) · 下一课：[Dispatcher 与真实多流](/courses/cuda-graph/11-dispatcher-padding-multistream/)

> 源码基线：[vLLM v0.22.1 / `0decac0d`](https://github.com/vllm-project/vllm/commit/0decac0d96c42b49572498019f0a0e3600f50398)。本课中的 “FULL” 固定指 model backbone forward 的捕获边界。

这一课只做一件事：沿 vLLM 0.22.1 的真实源码，追踪一张 `FULL` CUDA Graph 从 capture 到 replay。先冻结边界：这里的 “Full” 是**完整覆盖模型 forward**，不是把 scheduler、输入整理、logits、sampler 和 Engine 状态更新全部塞进一张图。

## 一、Capture 调用链

主链可以压缩为：

```text
GPUModelRunner.capture_model
  → CudagraphDispatcher.get_capture_descs
  → GPUModelRunner._capture_cudagraphs
  → GPUModelRunner._warmup_and_capture
  → GPUModelRunner._dummy_run
  → set_forward_context(..., mode=FULL, descriptor=...)
  → self.model(...)
  → CUDAGraphWrapper.__call__（capture 分支）
```

`capture_model()` 先进入 `parallel_state.graph_capture()`。它创建独立 capture stream，并让该流等待原 current stream 上已经提交的初始化工作。TP、PP group 共享同一个 `GraphCaptureContext`，避免每个通信组各用一条互不相关的捕获流。

随后 `get_capture_descs()` 返回 `(runtime_mode, descriptors)`：源码顺序是 **PIECEWISE 在前、FULL 在后**；每种模式内部再按 `num_tokens`、`num_active_loras` 从大到小排序。大 shape 先捕获，是为了先建立足够大的 graph pool，后续小 shape 更可能复用已有内存，降低峰值和碎片；这不是 runtime 的路由优先级。

对每个 descriptor，`_warmup_and_capture()` 先以 `CUDAGraphMode.NONE` 跑若干次 `_dummy_run()`，把 lazy init、编译、autotune 和 workspace 扩容赶到图外；最后一次才以 `FULL` 进入 capture。

## 二、Dummy 不等于“数据随便填”

`_dummy_run()` 会按 descriptor 构造 padded token/request 几何，准备静态 `input_ids`、`positions`、block table、slot mapping 和 attention metadata，然后通过 `set_forward_context()` 把 runtime mode 与 `BatchDescriptor` 送到 wrapper。

Dummy token 的具体值通常不重要；但凡会改变执行拓扑、workspace 或参数解释的 metadata 都必须真实。例如：

- uniform decode 的 query length 决定是否走优化的 decode routine；
- `num_reqs` 会影响某些 attention scheduler metadata；
- LoRA 激活数量可能改变 kernel grid；
- encoder length、block table 宽度、spec-decode 几何可能影响 plan；
- dummy slot mapping 被填为 `-1`，让 KV 写 kernel 跳过虚构槽位，不能随便写进真实 KV。

所以正确表述是：**数值内容可以是假数据，物理形状和执行语义不能是假拓扑。**

## 三、真正的 capture 发生在哪里

模型已经被 `CUDAGraphWrapper` 包装。其 `__call__()` 从 `ForwardContext` 取出 mode 与 descriptor：如果 mode 是 `NONE`，或与当前 wrapper 的固定 mode 不一致，就直接调用底层 runnable；只有匹配时才进入该 wrapper 的图缓存。

源码没有一个独立 `GraphKey` 类。此处 entry 的映射是：

```python
self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry]
```

同一个 descriptor 位于不同 runtime-mode wrapper 中，就是不同的具体图实例。entry 首次出现时，wrapper：

1. 记录 Tensor 参数的 `data_ptr`；
2. 选择全局 graph pool；
3. 调用 `offloader.sync_prev_onload()` 清理 capture 外尚未完成的 copy；
4. 进入 `torch.cuda.graph(..., stream=current_stream())`；
5. 执行底层 `runnable(*args, **kwargs)`；
6. 调用 `offloader.join_after_forward()`，确保图内分叉出的 copy stream 在结束前 join；
7. 保存 graph，并保留指向 graph-pool 静态 output 的弱引用/句柄。

这说明 wrapper 管的是“选择哪个 entry、capture 还是 replay”；它**不负责**把任意动态输入复制到持久 buffer。源码注释明确把静态输入 staging 留给 ModelRunner，因为 wrapper 不知道各模型的动态 shape、metadata 和生命周期规则。

## 四、Runtime replay 调用链

真实请求走另一条链：

```text
GPUModelRunner.execute_model
  → _prepare_inputs / _build_attention_metadata
  → _determine_batch_execution_and_padding
  → CudagraphDispatcher.dispatch
  → set_forward_context(mode, BatchDescriptor)
  → _model_forward
  → CUDAGraphWrapper.__call__（replay 分支）
  → static output
  → hidden-state / logits / ExecuteModelState（图外）
  → sample_tokens（独立调用）
```

`_determine_batch_execution_and_padding()` 先识别 uniform decode、LoRA、cascade attention、encoder output 等条件，再调用 dispatcher。返回的 descriptor 已含 padded token 数；ModelRunner 据此准备固定容量的 Tensor 和 attention metadata。

wrapper 命中已有 entry 后，在 debug 模式检查输入地址与 capture 时完全一致，先 `sync_prev_onload()`，再调用 `entry.cudagraph.replay()`，最后返回静态 output。Graph 是提交模板，不是固定绑死的执行 Stream；PyTorch replay 会把图提交到调用时的 current stream。若 staging copy 也在该流，FIFO 已保证 copy → replay；若在别的流，必须显式建立 consumer-waits-producer。

replay 返回后，`execute_model()` 继续 hidden-state 选择、LM head/logits，并保存 `ExecuteModelState`；采样发生在随后独立的 `sample_tokens()` 中。因此“FULL Graph 覆盖整个服务循环”的说法在这个版本上是不成立的。

## 五、为什么会有嵌套 wrapper

`FULL_AND_PIECEWISE` 下，外层可以是 FULL wrapper，编译分区内部还存在 PIECEWISE wrapper。运行时命中 FULL 时，外层直接 replay 整个模型，内部 Python 不再执行；FULL miss 而 PIECEWISE hit 时，外层因 mode 不匹配而 pass-through，底层模型真实执行，内部 PIECEWISE wrapper 才各自 replay。

这个“模式不匹配就直通”不是冗余判断，而是让两层 wrapper 共存且只激活一层的关键。

## 建议观察的日志

为一张图至少记录：runtime mode、完整 descriptor、real/padded token 数、输入输出 `data_ptr`、graph pool id、capture 顺序、entry hit/miss、replay 次数以及 clear/destroy 时机。若只打印“用了 CUDA Graph”，无法诊断错 key、旧数据、输出覆盖和显存膨胀。

## 本课验收

1. Dummy input 哪些内容可以任意，哪些 metadata 必须反映真实 topology？
2. 为什么 capture 顺序是 PIECEWISE → FULL、large → small，而 dispatch 是 FULL → PIECEWISE → NONE？
3. 外层 FULL wrapper 收到 PIECEWISE mode 时为什么必须 pass-through？
4. 为什么 wrapper 不应独自负责 runtime input copy？
5. FULL replay 之后，logits 与 sampler 为什么仍可在图外？

能从日志把一次请求精确落到“哪个 wrapper 的哪个 descriptor entry”，并指出 capture 与 replay 两条链的汇合点，才算通过。

## 源码

- [GPUModelRunner：capture、dummy run 与 runtime forward](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/worker/gpu_model_runner.py)
- [CudagraphDispatcher：capture descriptors 与 runtime dispatch](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/cudagraph_dispatcher.py)
- [CUDAGraphWrapper：capture/replay 两个分支](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/compilation/cuda_graph.py)
- [ForwardContext 与 BatchDescriptor](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/forward_context.py)
- [parallel_state.graph_capture：TP/PP 捕获上下文](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/distributed/parallel_state.py)

上一课：[Graph-Safety Harness](/courses/cuda-graph/09-graph-safety-harness/) · 下一课：[Dispatcher 与真实多流](/courses/cuda-graph/11-dispatcher-padding-multistream/)
