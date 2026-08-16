# vLLM 源码课程账本

## 总体路线

`入口 API 与 Renderer/InputProcessor → 请求对象与跨进程边界 → EngineCore → Scheduler → KV Cache Manager → Executor/Worker → ModelRunner → Attention → Sampling → 分布式执行 → CUDA Graph/torch.compile → Serving → 测试、性能与故障诊断`

## 当前阶段

- 阶段 3：Scheduler 执行计划到 Executor/Worker/ModelRunner 设备状态边界。
- 当前主线：一次 offline `LLM.generate()` 请求已从 prompt 追到可执行的 KV 地址计划；已完成 `SchedulerOutput → RequestState/BlockTables/InputBuffers → slot_mapping`。下一章沿 `ModelRunnerOutput → Scheduler.update_from_output` 追踪 token/spec progress 的提交事务。

## 已完成章节

| 日期/章节 | 主题 | vLLM commit | 文章 |
| --- | --- | --- | --- |
| 2026-08-10 01 | `LLM.generate → Renderer → InputProcessor → EngineCoreRequest` | [`751f2ccd`](https://github.com/vllm-project/vllm/commit/751f2ccdd3b7ed2ba65dcb04dbd93187edc25b2d) | [请求边界]({{ '/articles/vllm-input-pipeline-request-boundary/' | relative_url }}) |
| 2026-08-10 02 | `SyncMPClient → Msgpack/ZMQ/Tensor IPC → EngineCoreProc.input_queue` | [`51562de5`](https://github.com/vllm-project/vllm/commit/51562de5ab16bacd821d7130187b6bebdd293f93) | [跨进程 wire protocol]({{ '/articles/vllm-enginecore-request-wire-protocol/' | relative_url }}) |
| 2026-08-10 03 | `ADD/ABORT 双队列 → output buffer/backpressure → EngineDeadError` | [`bd653607`](https://github.com/vllm-project/vllm/commit/bd6536071cec4dcd8cf91c0e2aa04aec83fc1c37) | [取消、背压与故障传播]({{ '/articles/vllm-enginecore-abort-backpressure-failure/' | relative_url }}) |
| 2026-08-13 04 | `EngineCore.add_request → Request.from_engine_core_request → Scheduler.add_request` | [`98f86b9c`](https://github.com/vllm-project/vllm/commit/98f86b9c02329200a0390aecfe598e27928cbf40) | [Scheduler admission]({{ '/articles/vllm-scheduler-request-admission/' | relative_url }}) |
| 2026-08-14 05 | `Scheduler.schedule waiting loop → prefix lookup → KVCacheManager.allocate_slots → SchedulerOutput` | [`827a2af8`](https://github.com/vllm-project/vllm/commit/827a2af806c4e4ea7bcc280f57f793e6a5fcc676) | [第一个物理 Batch]({{ '/articles/vllm-scheduler-first-physical-batch/' | relative_url }}) |
| 2026-08-15 06 | `running slot growth → victim selection → PREEMPTED → prefix lookup → resumed output` | [`615d4cfa`](https://github.com/vllm-project/vllm/commit/615d4cfadeb3d5ea1df248eb59aa128af5dbd441) | [Preemption 与重算闭环]({{ '/articles/vllm-scheduler-preemption-recompute-resume/' | relative_url }}) |
| 2026-08-16 07 | `SchedulerOutput → GPUModelRunner v2 → RequestState/BlockTables → slot_mapping` | [`fa9d67f7`](https://github.com/vllm-project/vllm/commit/fa9d67f7828e9bc105912ddf41dc384105732b1e) | [设备状态与 Slot Mapping]({{ '/articles/vllm-scheduler-output-modelrunner-device-state/' | relative_url }}) |

## 既有专题（课程前置资料）

- [v0.22.1 EngineCore 基类]({{ '/articles/vllm-enginecore-source-walkthrough/' | relative_url }})：旧版本逐函数专题，后续课程会基于当前 `main` 标注漂移。
- [v0.22.1 EngineCore 派生类]({{ '/articles/vllm-enginecore-subclasses-source-walkthrough/' | relative_url }})：EngineCoreProc/DP/Ray 专题。

## 已覆盖符号

- `LLM.generate`
- `OfflineInferenceMixin._add_completion_requests`
- `OfflineInferenceMixin._render_and_add_requests`
- `OfflineInferenceMixin._add_request`
- `OfflineInferenceMixin._run_engine`
- `BaseRenderer.render_cmpl`
- `LLMEngine.add_request`
- `LLMEngine.step`（仅前端返回边界）
- `InputProcessor.process_inputs`
- `InputProcessor.assign_request_id`
- `InputProcessor._validate_prompt_len`
- `EngineCoreRequest`
- `EngineCoreClient.make_client`
- `MPClient.__init__`
- `SyncMPClient._send_input`
- `MsgpackEncoder.encode/_encode_tensor`
- `MsgpackDecoder.decode/_decode_tensor`
- `TensorIpcSender/TensorIpcReceiver`
- `EngineCoreProc.process_input_sockets`
- `EngineCoreProc._process_input_queue`
- `EngineCoreProc._handle_client_request`
- `EngineCore._process_aborts_queue`
- `EngineCore.step/step_with_batch_queue`
- `EngineCoreProc.process_output_sockets`
- `EngineCoreProc._send_msg_tracking_payload`
- `EngineCoreProc._send_engine_dead`
- `BackgroundResources.validate_alive`
- `MPClient.ensure_alive/start_engine_core_monitor`
- `make_zmq_socket`
- `EngineCore.preprocess_add_request`
- `EngineCore.add_request`
- `Request.__init__`
- `Request.from_engine_core_request`
- `RequestStatus`
- `Scheduler.add_request`
- `Scheduler._enqueue_waiting_request`
- `Scheduler.finish_requests`
- `Scheduler._free_request`
- `EngineCoreRequestType`
- `Scheduler.schedule`（waiting flow）
- `Scheduler._get_local_prefix_cache_hit`
- `KVCacheManager.get_computed_blocks`
- `KVCacheManager.allocate_slots`
- `KVCacheManager.record_prefix_cache_stats`
- `KVCacheBlocks`
- `NewRequestData.from_request`
- `SchedulerOutput`
- `Scheduler.schedule`（running flow 与 preemption rollback）
- `Scheduler._preempt_request`
- `Scheduler._update_after_schedule`
- `Scheduler._make_cached_request_data`
- `CachedRequestData`
- `EngineCore.step`（SchedulerOutput 执行与返回主链）
- `Executor.execute_model`
- `WorkerWrapperBase.execute_model`
- `GPUWorker.execute_model`
- `GPUModelRunner.finish_requests`
- `GPUModelRunner.add_requests`
- `GPUModelRunner.update_requests`
- `GPUModelRunner.gather_batch_req_state`
- `GPUModelRunner.prepare_inputs`
- `GPUModelRunner.prepare_attn`
- `GPUModelRunner.execute_model`（v2）
- `RequestState.add_request/remove_request/apply_staged_writes`
- `BlockTables.append_block_ids/apply_staged_writes`
- `BlockTables.gather_block_tables/compute_slot_mappings`
- `InputBuffers`
- `InputBatch`

## 已确认不变量

1. Renderer 负责用户输入到 `EngineInput`；InputProcessor 负责 `EngineInput` 到 `EngineCoreRequest`。
2. `SamplingParams/PoolingParams` 在请求内 clone 后再验证和补默认值。
3. `external_req_id` 保留 API 身份；随机化后的 `request_id` 是 EngineCore 内部身份。
4. `EngineCoreRequest` 是 array-like msgspec wire struct，字段顺序具有协议意义。
5. offline 多请求先逐个加入 EngineCore，物理 batch 由 Scheduler 后续决定。
6. 批量加入中途失败时，前端 abort 已加入请求。
7. MP ADD wire 为 ROUTER identity、单字节 request type、msgpack payload 和可选 auxiliary frames。
8. 默认大 tensor 走 ZMQ auxiliary frame；`torch_shm` 才走共享内存/OOB queue，且当前限制为 spawn、单 DP/TP/PP rank。
9. EngineCore input IO thread完成 decode、multimodal cache 恢复、`Request` 构造和 grammar init，再交给 busy loop 的 `input_queue`。
10. `send_multipart(copy=False)` 保证 Python buffer 保活并避免 msgpack 聚合复制，但不承诺跨进程端到端零拷贝。
11. ABORT 同时进入 `aborts_queue` 和 `input_queue`：前者在 model output 提交前抢占，后者保持 ADD/ABORT 顺序；正确性依赖 Scheduler abort 幂等。
12. 普通与 batch-queue step 都必须在 `scheduler.update_from_output` 前处理执行期间到达的 abort，使最终 step 取消保持 `FINISHED_ABORTED`。
13. reusable msgpack `bytearray` 只有在首帧 `MessageTracker.done` 后才能复用；`max_reuse_bufs` 仅限制空闲池，不限制发送中的 `pending`。
14. ZMQ HWM=0 与无界 Python queues 优先吸收 backlog，不提供有界应用层背压；慢消费者可能将压力转化为内存和尾延迟。
15. EngineCore failure 通过 in-band `ENGINE_CORE_DEAD` sentinel 与本地进程 out-of-band monitor 两条路径汇聚到单向 `engine_dead` latch。
16. `Request` 在 input IO thread 构造，但 `Scheduler.requests/waiting/running` 只能由 busy loop 修改。
17. `Scheduler.add_request` 是 Host 侧逻辑 admission：计算 block hashes 并建立状态，但不分配 KV block。
18. `Scheduler.requests` 是 canonical registry；`waiting`、`skipped_waiting`、`running` 引用同一 `Request` 对象。
19. grammar、remote KV、streaming input 等异步依赖未满足时进入 `skipped_waiting`，仍属于活请求。
20. finished Request 可能因 connector 延迟释放而暂留 registry；存在于 `requests` 不等于仍可调度。
21. waiting request 的物理 admission 以 `KVCacheManager.allocate_slots()` 成功为界；prefix hit、token budget 与 sequence slot 均不是充分条件。
22. waiting allocation 失败时，请求不出队、状态不转 RUNNING、token/input budget 不扣减；FCFS 主路径在队首容量失败时停止继续接纳。
23. 首次入场时 `Request.num_computed_tokens` 只提交本地/外部 cache 已覆盖的进度，本 step 的 `num_scheduled_tokens` 只有在 ModelRunner 输出回到 Scheduler 后才成为已计算进度。
24. prefix cache 查询统计在成功 allocation 后记录；未调度的重试不计数，避免 cache hit rate 被重试次数放大。
25. `SchedulerOutput` 传递 Host 侧 request/block/token 计划；设备 tensor 和 slot mapping 属于 ModelRunner 边界。
26. running request 的 KV continuation 仍以 `KVCacheManager.allocate_slots()` 成功为物理边界；返回 `None` 会触发 victim 选择，而不是产生部分 allocation。
27. FCFS preemption 取 running 队尾；PRIORITY 取数值优先级最低、同优先级到达最晚的 running 请求；两者都不以重算 FLOPs 为目标函数。
28. victim 若已进入本轮 batch，必须同步撤销 token/input/encoder budget、spec metadata 与新 block 计划，保证 `SchedulerOutput` 不包含已重置请求。
29. preemption 释放请求对 KV blocks 的所有权并把 `num_computed_tokens` 清零，但保留 token 历史；恢复进度只能由重新确认的 local/external prefix hit 建立。
30. resumed request 必须替换 Worker/ModelRunner 中的旧 block IDs，不能按普通 cached request 追加；ModelRunner v2 用完整 `NewRequestData`，v1 用 `CachedRequestData.resumed_req_ids` 表达同一不变量。
31. async preemption 必须隔离旧 step 的 in-flight/stale output；队列状态变化与迟到 model output 共同构成正确性协议。
32. ModelRunner v2 的 request slot 是固定容量资源：从 `scheduled_new_reqs` 占用，到 `finished_req_ids/preempted_req_ids` 上报后释放；Scheduler admission 必须计入仍持有 Worker slot 的暂停请求。
33. v2 same-step preempt+resume 必须按 `finish_requests → add_requests` 顺序先 purge 再重建；new/resumed block IDs 用 overwrite，cached continuation 才能 append。
34. request-slot canonical block table 与 current-batch `input_block_tables` 是两层状态；`idx_mapping` 把稀疏 request slot gather 成紧凑 batch row。
35. `slot_mapping` 将 position 映射为物理 KV slot；`CP_SIZE=1` 时为 `block_id × block_size + block_offset`，DCP 下还必须按 interleave/rank 将非本地 token 写成 `PAD_SLOT_ID`。
36. `InputBuffers`、`input_block_tables`、`slot_mappings` 是持久设备 buffer；每 step 更新内容而不重建地址，是 full CUDA Graph 可复用的必要条件，但不是充分条件。
37. CUDA Graph padding 区必须从 actual token 尾部覆盖为 `PAD_SLOT_ID`，不能保留上一 batch 的有效 slot。

## 前置依赖与版本注意

- 本课程以每章记录的 `main` commit 为准，不把 v0.22.1 专题中的实现自动视为当前事实。
- 直接向 InputProcessor 传 raw prompt、向 LLMEngine 传 `EngineCoreRequest` 均处于 v0.18 移除迁移期。

## 尚未解释的知识债

- `AsyncMPClient`/DP client 的 engine identity 选择、跨 producer 顺序和线程安全边界。
- hybrid KV groups 下 per-group prefix blocks 如何收敛为统一 `num_computed_tokens`，以及 partial-tail/CoW 的正确性边界。
- watermark、`scheduler_reserve_full_isl` 与 async KV load `reserved_blocks` 的容量策略和配置取舍。
- `n>1` parallel sampling 的 parent/child ID 与前端合并。
- multimodal cache 在线程池下的并发契约，以及 PR #50896 的最终结果。
- Rust EngineCore client 与 Python `EngineCoreRequest`/aux frame 的双向协议兼容测试。
- `torch_shm` 中 payload 发送失败后的 orphan tensor 清理与可观测性。
- 慢/断连 output consumer 下 `pending` payload、ZMQ buffer 和 Python queue 的实际内存曲线，以及应该采用的容量/拒绝策略。
- 大量 waiting 请求的 Host metadata、prompt embedding 和 multimodal feature 内存曲线，以及服务层 admission 上限。
- `connector.on_new_request()` 抛异常时 queue/registry 的回滚原子性。
- 运行期间 EngineCore 被硬杀时，在途 `get_output`、后续 `add_request` 与各 DP/Ray manager 的一致故障语义。
- `ENGINE_CORE_DEAD` 在 5 秒 output-thread join、4 秒 socket linger 与 frontend shutdown 竞争下的交付边界。
- victim selection 未纳入已计算 token、模型结构或实际重算代价；长短 prompt 混合负载下的浪费、公平性和饥饿边界缺少策略级基准。
- async scheduling 与 PP/MTP/KVConnector 叠加时 stale output、drop mode 和多次连续 preemption 的端到端顺序性。
- `_compute_slot_mappings_kernel` 缺少覆盖跨 block 边界、resumed replace、graph padding、DCP interleave 与多 KV group 的直接单测。
- ModelRunner metadata preparation（staged write、gather、slot mapping）的实际 GPU 时间及其与 full/piecewise graph 边界尚未通过 trace 量化。
- UVA-backed `all_token_ids` 在长上下文、高并发下的 Host/Device 访问和 page-fault 成本尚未测量。
- 不同 executor backend 下 `SchedulerOutput` 向各 Worker 传播的一致性、序列化成本与部分 rank 失败语义尚未下钻。

## 下一批候选章节

1. 下一主线：`ModelRunnerOutput → Scheduler.update_from_output` 如何提交 token、修正 speculative progress 并释放 finished request。
2. 后续：Attention metadata 如何消费 block table、slot mapping、seq lens，并读写真实 KV cache tensor。
3. 协议专项回访：定义有界 backpressure，并建立 Python↔Rust 双向 golden fixtures。

## 第七篇知识图谱回顾

- 已打通：`LLM.generate → Renderer/InputProcessor → EngineCoreRequest wire → EngineCore queue/failure → Scheduler logical admission → KV physical admission/preemption → Worker device-state materialization`。
- 已闭合边界：用户请求已经能够被追踪为一次具体的 `input_ids/positions/block_tables/slot_mapping` 设备执行计划。
- 当前最大盲区：ModelRunner 输出何时成为 Scheduler 可提交 token；async/speculative progress 如何回滚；finished 状态如何同时释放 Scheduler registry、KV ownership 与 Worker slot。
- 后续路线调整：先完成返回事务，再进入 Attention metadata 和真实 KV cache tensor；暂不提前跳到 Sampling 或 CUDA Graph 优化。
