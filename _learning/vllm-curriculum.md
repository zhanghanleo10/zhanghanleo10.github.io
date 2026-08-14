# vLLM 源码课程账本

## 总体路线

`入口 API 与 Renderer/InputProcessor → 请求对象与跨进程边界 → EngineCore → Scheduler → KV Cache Manager → Executor/Worker → ModelRunner → Attention → Sampling → 分布式执行 → CUDA Graph/torch.compile → Serving → 测试、性能与故障诊断`

## 当前阶段

- 阶段 2：Scheduler 请求建档、waiting 状态与物理 batch 形成。
- 当前主线：一次 offline `LLM.generate()` 请求从 prompt 到 Scheduler 的完整生命周期；已完成逻辑 admission 与 `Scheduler.schedule()` 首次物理 batch 形成。下一章进入 running request 的 slot 追加与 preemption。

## 已完成章节

| 日期/章节 | 主题 | vLLM commit | 文章 |
| --- | --- | --- | --- |
| 2026-08-10 01 | `LLM.generate → Renderer → InputProcessor → EngineCoreRequest` | [`751f2ccd`](https://github.com/vllm-project/vllm/commit/751f2ccdd3b7ed2ba65dcb04dbd93187edc25b2d) | [请求边界]({{ '/articles/vllm-input-pipeline-request-boundary/' | relative_url }}) |
| 2026-08-10 02 | `SyncMPClient → Msgpack/ZMQ/Tensor IPC → EngineCoreProc.input_queue` | [`51562de5`](https://github.com/vllm-project/vllm/commit/51562de5ab16bacd821d7130187b6bebdd293f93) | [跨进程 wire protocol]({{ '/articles/vllm-enginecore-request-wire-protocol/' | relative_url }}) |
| 2026-08-10 03 | `ADD/ABORT 双队列 → output buffer/backpressure → EngineDeadError` | [`bd653607`](https://github.com/vllm-project/vllm/commit/bd6536071cec4dcd8cf91c0e2aa04aec83fc1c37) | [取消、背压与故障传播]({{ '/articles/vllm-enginecore-abort-backpressure-failure/' | relative_url }}) |
| 2026-08-13 04 | `EngineCore.add_request → Request.from_engine_core_request → Scheduler.add_request` | [`98f86b9c`](https://github.com/vllm-project/vllm/commit/98f86b9c02329200a0390aecfe598e27928cbf40) | [Scheduler admission]({{ '/articles/vllm-scheduler-request-admission/' | relative_url }}) |
| 2026-08-14 05 | `Scheduler.schedule waiting loop → prefix lookup → KVCacheManager.allocate_slots → SchedulerOutput` | [`827a2af8`](https://github.com/vllm-project/vllm/commit/827a2af806c4e4ea7bcc280f57f793e6a5fcc676) | [第一个物理 Batch]({{ '/articles/vllm-scheduler-first-physical-batch/' | relative_url }}) |

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

## 下一批候选章节

1. 下一主线：running request 的 slot 追加、preemption victim 选择与 `PREEMPTED → waiting → resumed` 状态闭环。
2. 后续：`SchedulerOutput` 到 ModelRunner 的 block table、slot mapping 与设备 input buffer。
3. 协议专项回访：定义有界 backpressure，并建立 Python↔Rust 双向 golden fixtures。

