---
layout: post
title: "vLLM 源码课程 20：超时不是一张网——TP RPC 300 秒与单卡 Hang 盲区"
description: "沿 EngineCore.step、MultiprocExecutor Future 与 worker monitor 追踪一次执行停滞如何变成 fail-stop，并辨清当前 V1 尚未接线的 iteration timeout。"
date: 2026-08-30
category: "vLLM 源码"
tags: [vLLM, EngineCore, MultiprocExecutor, Timeout, Fault Tolerance]
reading_time: "约 18 分钟"
mermaid: true
---

## 本篇在课程路线中的位置

第 19 章解释了 **EngineCore 已经死亡之后**，`EngineDeadError` 如何广播到全部请求，并让 `/health` 返回 503。本章向前追一层：如果 Worker 进程还在、GPU/NCCL 调用却不再返回，系统凭什么判定失败？

结论先说：基于 vLLM 主干 [`680e2177`](https://github.com/vllm-project/vllm/commit/680e2177e473ed8dfaa9773f7ead185b369cab46)，当前 V1 并不存在一张覆盖所有 backend 的“60 秒 Engine iteration watchdog”。真正接入执行路径的是 `MultiprocExecutor` 的 `execute_model` RPC deadline，默认 300 秒；进程 monitor 只检测进程退出；5 秒配置则是失败后的 shutdown grace。单进程 `UniProcExecutor` 对 alive-but-stalled GPU 仍存在盲区。

## 前置知识回顾

一次正常迭代已经形成事务边界：

`Scheduler.schedule → SchedulerOutput → Executor.execute_model → ModelRunnerOutput → Scheduler.update_from_output`

只有 `future.result()` 成功返回后，Scheduler 才提交 token、更新 request 状态并决定释放 KV。第 19 章又确认，fatal exception 一旦越出 `EngineCoreProc.run_engine`，Core 会发送 `ENGINE_CORE_DEAD`，前端把同一异常传播给所有 collector，并停止新请求 admission。

本章的问题是：**谁负责让“永远不返回”先变成一个 exception？**

## 本篇要回答的核心问题

1. `execute_model` 的 300 秒从哪里开始计时，在哪个线程抛出？
2. worker process monitor、RPC timeout 和 shutdown timeout 分别证明什么？
3. 超时后为什么只能 fail-stop，不能把同一 Scheduler 状态留在原进程里重试？
4. `VLLM_ENGINE_ITERATION_TIMEOUT_S=60` 在当前 V1 是否真的生效？

## 组件在全局架构中的位置

```mermaid
flowchart LR
    S["Scheduler.schedule"] --> O["SchedulerOutput"]
    O --> E["EngineCore.step"]
    E --> M["MultiprocExecutor.execute_model"]
    M --> B["rpc_broadcast_mq"]
    B --> W0["WorkerProc rank 0"]
    B --> W1["WorkerProc rank 1"]
    W0 --> R["response_mq"]
    W1 --> R
    R --> F["FutureWrapper.result"]
    F -->|success| C["Scheduler.update_from_output"]
    F -->|TimeoutError| D["EngineCoreProc._send_engine_dead"]
    D --> P["MPClient → all collectors → /health 503"]
```

`SchedulerOutput` 是 Host 侧 msgspec 对象，记录 request IDs、每请求本轮 token 数、block IDs 等调度元数据；它不是有固定 `shape/dtype/device` 的单一 Tensor。Worker 消费它后才在设备侧物化 `input_ids`、`positions`、`slot_mapping` 等 Tensor。它由 `EngineCore.step()` 暂时持有，并在 `ModelRunnerOutput` 成功返回后被 Scheduler 提交；发生 fatal timeout 时，本轮永远不会进入正常 commit。

## 完整调用链

当前 multiprocess 路径如下：

1. [`EngineCore.step`](https://github.com/vllm-project/vllm/blob/680e2177e473ed8dfaa9773f7ead185b369cab46/vllm/v1/engine/core.py) 调用 `scheduler.schedule()`，随后用 `non_block=True` 调 `model_executor.execute_model()`。
2. [`MultiprocExecutor.execute_model`](https://github.com/vllm-project/vllm/blob/680e2177e473ed8dfaa9773f7ead185b369cab46/vllm/v1/executor/multiproc_executor.py) 进入 `collective_rpc`，传入 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`，默认值在 [`envs.py`](https://github.com/vllm-project/vllm/blob/680e2177e473ed8dfaa9773f7ead185b369cab46/vllm/envs.py) 中为 300 秒。
3. `collective_rpc` 在发送前计算一次绝对 `deadline = monotonic() + timeout`，广播 RPC，并返回 `FutureWrapper`。
4. `EngineCore.step()` 调 `future.result()`；`FutureWrapper` 按 FIFO 等待此前 outstanding future，再执行响应读取 closure。
5. `get_response()` 对每个 response queue 使用同一个 deadline 的剩余时间。任一必要响应未按时到达，就抛出 `TimeoutError("RPC call to execute_model timed out.")`，而不是每个 rank 各等 300 秒。
6. exception 穿过 `log_error_detail`；这里会 dump 诊断信息，但不吞错。最外层 `EngineCoreProc.run_engine` 记录 fatal、发送 `ENGINE_CORE_DEAD` 并开始 shutdown。
7. 后续前端广播、collector 终止和 `/health` 503 复用第 19 章已经验证的 fail-stop 链。

## 关键类型、字段和状态生命周期

### `SchedulerOutput`：尚未提交的执行计划

创建于 `Scheduler.schedule()`，由 `EngineCore.step()` 传给 Executor。其关键条件是：Scheduler 已为本轮选择 request/token，并建立相应 KV allocation 视图；后置条件必须由 `ModelRunnerOutput` 驱动的 `update_from_output()` 完成。

超时时，不能假定设备什么都没做：某些 rank 可能已完成 KV write，另一些 rank 卡在 collective，Host 也可能只缺最终响应。因此本轮处于 **completion unknown**，既不能正常 commit，也不能在同一 engine 内无条件重放。

### `FutureWrapper`：响应所有权与 deadline 容器

`FutureWrapper` 保存响应 closure 和结果/异常状态。调用方不能在 `result(timeout=...)` 临时指定超时；timeout 在 `collective_rpc` 创建 closure 时就固定。它还按提交顺序 drain 旧 future，保护响应队列和 future 的对应关系，但会产生 head-of-line waiting：后一个已完成结果不能越过前一个未完成 RPC。

### Worker process 与资源释放

worker monitor 等待 process sentinel。它能证明“进程已经退出”，不能证明“进程还在推进”。检测到死亡后，它关闭 Executor 并把 `EXECUTOR_FAILED` 控制消息送入 EngineCore input queue。

shutdown 的 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` 默认 5 秒，只是给 Worker 自清理的宽限期；之后发 `SIGTERM`，再等 4 秒，仍存活才发 `SIGKILL`。这个 5 秒不是执行 hang detector。

## 逐函数源码解读

### `EngineCore.step()`：等待点就是事务闸门

简化后只有四步：

```text
scheduler_output = scheduler.schedule()
future = executor.execute_model(scheduler_output, non_block=True)
model_output = future.result()
scheduler.update_from_output(scheduler_output, model_output)
```

最重要的不是“用了 Future”，而是 commit 位于 `result()` 之后。异常退出时不会产生伪造的 token 终态。代价是：如果 backend 没有可用 deadline，Core busy loop 就会一直停在这里，ABORT queue、健康状态和后续调度也无法前进。

### `MultiprocExecutor.collective_rpc()`：单一绝对 deadline

deadline 在 RPC 入队之前生成，因此包含排队、Worker 执行和响应传输时间。所有 response queue 共用它，确保总等待上界接近 300 秒，而不是 `rank_count × 300 秒`。

这个 timeout 只说明“Host 没在期限内收到必要响应”，不说明是 CUDA kernel、NCCL collective、Python Worker、消息队列还是系统过载造成。它是 failure detector，不是 root-cause detector。

该参数最初由 [PR #19544](https://github.com/vllm-project/vllm/pull/19544) 配置化；更早的 [PR #18558](https://github.com/vllm-project/vllm/pull/18558) 把 TP timeout 从 40 秒提高到 300 秒，反映了典型权衡：阈值过短会把合法长执行误判为故障，过长则放大故障发现时间。

### `UniProcExecutor.collective_rpc()`：接口有 timeout，执行没有 deadline

[`UniProcExecutor`](https://github.com/vllm-project/vllm/blob/680e2177e473ed8dfaa9773f7ead185b369cab46/vllm/v1/executor/uniproc_executor.py) 直接在 Core 进程内调用 Worker 方法。函数签名接受 `timeout`，但没有计时或可抢占等待；`AsyncOutputFuture.result(timeout)` 也明确不实现 timeout。

同样，`envs.py` 声明了默认 60 秒的 `VLLM_ENGINE_ITERATION_TIMEOUT_S`，但在本次基线的 V1 runtime 搜索中，`EngineCore.step()`、`AsyncLLM` 和 V1 Executor 都没有读取它。**代码事实是：不能把这个环境变量当作当前 V1 单卡 hang 保护。**

## 具体示例与 shape/状态演算

假设 TP=2，有请求 A 正在 decode：prompt 已有 16 token，本轮只调度 1 个 token。

| 时刻 | rank 0 | rank 1 | Host/Scheduler |
| --- | --- | --- | --- |
| `t0` | 收到 RPC | 收到 RPC | `num_scheduled_tokens={A:1}`，尚未 commit |
| `t0+20ms` | 产生局部激活 | 卡在 collective/GPU | `FutureWrapper` 等必要响应 |
| `t0+300s` | 可能仍存活 | 进程也可能仍存活 | response dequeue 抛 `TimeoutError` |
| 随后 | Executor shutdown | grace→TERM→KILL | 不调用 `update_from_output`，Core 进入 dead latch |

设备实际输入会因模型而异，但 decode 主体通常把一个 token row 物化为类似 `input_ids:[1] int64 CUDA`、`positions:[1] int64 CUDA`，KV 写入位置由 `slot_mapping:[1] int64 CUDA` 指定。这里的 shape 只用于说明单 token decode，不是所有 model backend 的统一 ABI。

如果同样场景是 TP=1 `UniProcExecutor`，Core 直接陷入 Worker 调用。进程 sentinel 不触发，60 秒 iteration env 也未接线；因此 `/health` 可能仍无法观测 forward progress。后一句是由当前调用链推出的系统行为，不是仓库测试已经证明的 SLA。

## 为什么这样设计及替代方案

当前 RPC deadline 的优点是边界清晰：父进程无需理解 CUDA/NCCL 内部状态，只要求 Worker 在期限内回复；TP 多进程又天然具备可隔离、可强杀的执行单元。它不增加每 kernel heartbeat，也不会在正常迭代中引入设备同步。

替代方案有三类：

- **Core 内 Python timer**：实现简单，但 Python exception 无法可靠抢占卡死的 CUDA/NCCL 调用；在同进程继续复用 context 也缺少正确性证明。
- **独立 supervisor + progress counter/heartbeat**：可以覆盖 UniProc alive stall，并在超时后直接终止整个 engine process；代价是定义“真实进展”、跨编译/长 prefill 设置动态阈值，以及额外控制面与误杀风险。
- **backend-native kernel/collective timeout**：定位更精确、可更早失败，但 CUDA、NCCL、Ray、XPU/NPU 的机制不同，维护成本高，也未必能安全恢复 device context。

第一性原理约束是：检测阈值应高于合法最坏执行时间（长 prefill、首次 compile/capture、collective 抖动）的可信上界并留余量，同时小于业务可接受的故障发现与重建预算。仓库的 40→300 秒历史说明静态阈值无法同时对所有 workload 最优。

## 性能、并发、正确性与边界条件

- timeout 不会提升吞吐；它限制的是故障占用时间和请求无限悬挂风险。
- deadline 包含队列等待，因此极端拥塞也可能被判为执行失败；扩大阈值降低 false positive，却直接拉长 fail-stop latency。
- process monitor 的 callback 通过 input queue 进入 busy loop。若 busy loop 永久阻塞而 backend 没有 response deadline，callback 本身不能抢占它。
- timeout 后不做同进程 retry 是正确性选择：KV、RNG、collective 和 graph replay 可能已经部分推进，系统没有跨 rank transactional rollback。
- fail-stop 也不等于立即回收。真正的 device memory 清理由 Worker/Engine process teardown 完成；5 秒 grace、4 秒 TERM 等待和 KILL 属于另一阶段。
- Ray V2 monitor 的 5 秒 `ray.wait` 是 monitor 线程的轮询间隔，检测 actor `run()` ObjectRef 是否终结；它也不是“5 秒无进展即失败”。

## 测试证据与未覆盖风险

当前直接证据分三层：

1. [`tests/v1/executor/test_executor.py`](https://github.com/vllm-project/vllm/blob/680e2177e473ed8dfaa9773f7ead185b369cab46/tests/v1/executor/test_executor.py) 用 fake clock 验证 worker 在 shutdown grace 内退出不发 TERM，超过阈值会发 TERM。它验证的是关停升级，不是 RPC hang detection。
2. [`tests/distributed/test_multiproc_executor.py`](https://github.com/vllm-project/vllm/blob/680e2177e473ed8dfaa9773f7ead185b369cab46/tests/distributed/test_multiproc_executor.py) 覆盖 TP=2 collective RPC、callback 注册和 Worker 存活/关闭；没有让一个 alive Worker 吞掉 `execute_model` 响应。
3. [`tests/v1/fault_tolerance/test_fault_tolerance_e2e.py`](https://github.com/vllm-project/vllm/blob/680e2177e473ed8dfaa9773f7ead185b369cab46/tests/v1/fault_tolerance/test_fault_tolerance_e2e.py) 注入 `SIGKILL` 验证 fault-tolerance 模式的死亡检测；这不能替代默认路径的 silent hang 测试。

本次没有找到直接断言 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS → TimeoutError → ENGINE_CORE_DEAD → all collectors` 的测试，也没有 V1 测试证明 60 秒 iteration timeout 生效。最高风险是：TP=1 alive GPU hang、不同 Executor backend 的 deadline parity、超时期间已有 KV write 的清理，以及 Kubernetes liveness 在 `/health` 仍为 200 时无法重启实例。

## 与前后章节的连接

第 19 章从 fatal exception 向外讲到 HTTP；本章补上 Multiproc path 如何从“没有响应”制造 fatal exception。两章合起来才是：

`response deadline → TimeoutError → EngineCore dead → collector error → admission closed → /health 503`

下一章继续沿 fatal path 向诊断面下钻：`log_error_detail → dump_engine_exception → Scheduler/request/KV snapshot`，研究故障现场包含什么、什么时候写、哪些信息仍不足以区分 CUDA/NCCL/MQ 根因。

## 本篇结论、知识债、三个理解检查问题和下一章

本篇确认了四件事：Multiproc `execute_model` 使用默认 300 秒的共享绝对 deadline；process monitor 检测 death 而非 progress；5 秒是 shutdown grace；V1 的 60 秒 Engine iteration 配置当前没有接入这条执行链。超时只能证明 completion unknown，因此安全动作是终止 engine，而不是原地继续调度。

知识债：补齐 TP=1 独立 supervisor/progress watchdog；建立 alive Worker withheld-response 的跨层 E2E；统一 Multiproc、Ray、external launcher 的 deadline contract；量化 compile、长 prefill 与大 TP collective 的安全阈值分布；验证 timeout 后 Scheduler、KV、connector 与 CUDA context 随进程退出全部归零。

理解检查：

1. 为什么共享绝对 deadline 比“每个 rank 各等 300 秒”更符合故障预算？
2. Worker process sentinel 已经触发时，为什么阻塞在 `future.result()` 的 Core 仍可能需要 RPC timeout 才能前进？
3. 为什么 timeout 后不能仅 abort 当前请求并继续服务其他请求？

下一章：**fatal dump 的证据链——`log_error_detail`、`dump_engine_exception` 与 Scheduler/request/KV snapshot。**

## 课程账本增量

- 日期：2026-08-30
- 章节：20
- 源码基线：[`680e2177`](https://github.com/vllm-project/vllm/commit/680e2177e473ed8dfaa9773f7ead185b369cab46)
- 新覆盖：`EngineCore.step/step_with_batch_queue`、`MultiprocExecutor.execute_model/collective_rpc/get_response`、`FutureWrapper`、worker monitor、`_ensure_worker_termination`、`UniProcExecutor.collective_rpc`、`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`、`VLLM_ENGINE_ITERATION_TIMEOUT_S`、`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`
- 新不变量：RPC response timeout、process death detection、Engine fail-stop 和 shutdown escalation 是四个不同阶段；只有 active deadline 能把 alive stall 变成 exception；completion unknown 不允许同 engine 原地重试。
- 下一章：fatal dump 与可观测性证据链。
