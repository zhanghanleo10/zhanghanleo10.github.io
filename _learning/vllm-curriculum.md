# vLLM 源码课程账本

## 总体路线

`入口 API 与 Renderer/InputProcessor → 请求对象与跨进程边界 → EngineCore → Scheduler → KV Cache Manager → Executor/Worker → ModelRunner → Attention → Sampling → 分布式执行 → CUDA Graph/torch.compile → Serving → 测试、性能与故障诊断`

## 当前阶段

- 阶段 8：CUDA Graph/torch.compile，聚焦 graph key、动态 batch 归一化、固定地址与跨 rank replay 契约。
- 当前主线：已打通 local graph dispatch、DP mode/token envelope 共识、rank-local capture/replay 与 eager fallback；下一章进入 torch.compile 的切图边界。

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
| 2026-08-17 08 | `GPUModelRunner.sample_tokens → AsyncOutput → Scheduler.update_from_output → finish/free` | [`dc9ae4b8`](https://github.com/vllm-project/vllm/commit/dc9ae4b8ac2331991ad7091812ef82ece4f8fdc2) | [ModelRunnerOutput 返回事务]({{ '/articles/vllm-modelrunner-output-commit-transaction/' | relative_url }}) |
| 2026-08-18 09 | `BlockTables → Attention metadata → KV cache update → paged read` | [`c296851a`](https://github.com/vllm-project/vllm/commit/c296851a7d173fa89d2eefbca0243be42ae9b5e0) | [Attention 的三张地址表]({{ '/articles/vllm-attention-kv-write-paged-read-metadata/' | relative_url }}) |
| 2026-08-19 10 | `compute_logits → logits processors → Gumbel-max → logprobs → AsyncOutput` | [`f1178f3a`](https://github.com/vllm-project/vllm/commit/f1178f3a06fa30a0cc282376924210cedad08c44) | [一次 GPU Sampling 事务]({{ '/articles/vllm-gpu-sampling-logits-gumbel-logprobs/' | relative_url }}) |
| 2026-08-20 11 | `last PP rank sampling → PPHandler broadcast → generation filter → canonical output rank` | [`bf2866f8`](https://github.com/vllm-project/vllm/commit/bf2866f8bf5bb20628e2b93835be3c281a9b4ca4) | [PP Token 广播与状态收敛]({{ '/articles/vllm-pipeline-parallel-sampling-broadcast-state-convergence/' | relative_url }}) |
| 2026-08-21 12 | `SchedulerOutput fan-out → TP/PP data plane → canonical output/failure` | [`d29f7f5c`](https://github.com/vllm-project/vllm/commit/d29f7f5c9294be8e489dac34d45a939b95a06336) | [TP × PP Executor DAG]({{ '/articles/vllm-tp-pp-executor-dag-control-data-failure/' | relative_url }}) |
| 2026-08-22 13 | `local dispatch → DP mode/token consensus → ForwardContext → capture/replay` | [`8bdc70ec`](https://github.com/vllm-project/vllm/commit/8bdc70ec7b379279ec0152343239c2d50aced687) | [分布式 CUDA Graph Key 契约]({{ '/articles/vllm-distributed-cudagraph-key-padding-address-contract/' | relative_url }}) |

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
- `SamplerOutput`
- `AsyncOutput.__init__/get_output`
- `ModelRunnerOutput`
- `GPUModelRunner.sample_tokens`
- `Scheduler._update_after_schedule`
- `Scheduler.update_from_output`
- `Scheduler._update_request_with_output`
- `AsyncScheduler._update_request_with_output`
- `check_stop`
- `Scheduler._handle_stopped_request`
- `Scheduler.finish_requests`
- `Scheduler._free_request/_free_blocks/_free_request_blocks`
- `CommonAttentionMetadata`
- `build_attn_metadata`
- `DefaultModelState.prepare_attn`
- `FlashAttentionMetadataBuilder.build`
- `Attention.forward`
- `get_attention_context`
- `unified_kv_cache_update`
- `unified_attention_with_output`
- `FlashAttentionImpl.do_kv_cache_update`
- `FlashAttentionImpl.forward`
- `reshape_and_cache_flash`
- `GPUModelRunner.sample`
- `Sampler.add_request/apply_staged_writes`
- `Sampler.__call__`
- `Sampler.apply_sampling_params`
- `Sampler.sample`
- `SamplingStates`
- `gumbel_sample`
- `gumbel_block_argmax`
- `compute_topk_scores`
- `GPUModelRunner.execute_model/sample_tokens`（ModelRunner V2 PP path）
- `PPHandler`
- `PendingRecv`
- `compute_need_sampled_mask`
- `GPUModelRunner.update_pp_decode_requests`
- `GPUModelRunner.postprocess_sampled`
- `GPUWorker.execute_model`（PP IntermediateTensors send/recv）
- `MultiprocExecutor._get_output_rank`

- `Executor.get_class/execute_model/collective_rpc`
- `MultiprocExecutor.execute_model/sample_tokens/collective_rpc/_get_output_rank`
- `FutureWrapper`
- `WorkerProc.worker_busy_loop/enqueue_output/handle_output`
- `GPUWorker.execute_model`
- `AsyncIntermediateTensors`
- `RayDistributedExecutor._execute_dag/_compiled_ray_dag`
- `RayWorkerWrapper.execute_model_ray`
- `RayExecutorV2`
- `BatchDescriptor`
- `CudagraphDispatcher.initialize_cudagraph_keys/dispatch/get_capture_descs`
- `GPUModelRunner._prepare_inputs`（CUDA Graph dispatch 与 DP re-dispatch）
- `coordinate_batch_across_dp`
- `_run_ar/_post_process_cudagraph_mode/_post_process_dp_padding`
- `ForwardContext/set_forward_context`
- `CUDAGraphWrapper/CUDAGraphEntry`
- `GPUModelRunner.capture_model`

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
38. `Scheduler._update_after_schedule()` 会乐观增加 `num_computed_tokens` 与 `num_in_flight_tokens`；设备返回是对该乐观状态的 commit/rollback，而不是首次记录执行进度。
39. `AsyncOutput` 必须在 copy stream 完成前持有源 GPU tensor，并以 Event 作为 D2H 完成边界；`ModelRunnerOutput.sampled_token_ids` 进入 Scheduler 前已按每请求真实长度裁掉 padding。
40. speculative rejection 只回滚未接受 draft 对应的 computed progress；stale output 的 rejection 属于旧 request generation，不得再次作用于 preemption 后重置的计数。
41. 普通 async preemption 可按序交付迟到 token但不得污染重置计数；same-step resume/reset 或 connector handoff 的 `drop_stale_output` 必须丢弃旧 token，避免同一 position 被提交两次。
42. stop 判断必须逐 token 发生；spec batch 中途命中 EOS/stop/length 时，后续 token、logprobs 和 sampling mask 都必须按最终保留长度裁剪。
43. 前端收到 `finish_reason`、Scheduler 释放 KV ownership、connector 异步 job 释放额外引用、下一 SchedulerOutput 让 Worker purge request slot，是不同但有序的生命周期边界。
44. 异步消费者若寿命超过 Request，必须拥有 job/generation 级 identity 和资源引用；`req_id` 可在 preemption/resume 后复用，不能单独充当 ABA-safe 的工作单元身份。
45. Attention metadata 包含三个独立索引空间：`query_start_loc` 划分 packed query，`slot_mapping` 路由当前 K/V 写入，`block_table + seq_lens` 定义历史 KV 的 paged read 与有效边界。
46. split cache update backend 必须保证当前 K/V write 在 causal attention read 前可见；compiled/graph 模式依赖显式 dummy data dependency 保留隐藏副作用顺序。
47. graph padding 对 cache 必须无副作用：padding slot 为 `PAD_SLOT_ID`，padded block-table rows 不能残留上一 step 的有效物理页。
48. 写侧 slot metadata 与读侧 block table/seq lens 必须来自同一次 KV allocation；任一跨 request alias 都可能成为无异常的静默 KV 污染。
49. GPU v2 Sampler 的 temperature/top-k/top-p/min-p/seed 与 processor 子状态以稳定 request slot 为 owner；紧凑 batch 必须经 `idx_mapping/expanded_idx_mapping` 访问，不能以 row index 直接索引持久状态。
50. logits processor 的执行顺序属于采样语义：约束/bias/penalty/bad words/thinking budget 先于 temperature、min-p、top-k/top-p；被 mask 为 `-inf` 的 token 在后续变换和 Gumbel-max 中必须保持不可选。
51. native Gumbel 随机流由 request seed、logical token position 与 vocab lane 共同确定；`temperature=0` 必须跳过噪声并返回精确 argmax，batch row 重排不得成为显式 seed 的随机输入。
52. FlashInfer 等加速路径只有在能满足当前 mixed-batch 语义时才能使用；greedy、显式 per-request seed 或 processed logprobs 会触发 native fallback。
53. raw 与 processed logprobs 是不同的外部观察语义；`SamplerOutput` 的 GPU token 只有经过 AsyncOutput D2H 与 Scheduler commit 后才成为请求历史。
54. PP 非末 stage 只产生 `IntermediateTensors`；final hidden states、LM Head、logits 与 sampling 由 last PP stage 持有。
55. `PPHandler` 不广播整个 `SamplerOutput`；只广播设备状态收敛所需的 sampled IDs、`num_sampled` 与 `num_rejected`，而 EngineCore 只从 last-PP/TP0 output rank 接收 canonical `ModelRunnerOutput`。
56. deferred sampled output 必须以 request-slot generation 过滤；相同 `idx_mapping` 在 free/reuse 后不是同一生命周期，失配 row 必须映射为 `-1`。
57. sampled-token collective 必须在所有 PP ranks 上保持 skip 决策、调用顺序、shape/dtype 和 stream lifetime 对称；任一失配可能成为 collective hang 而非普通 Python 异常。
58. Multiproc 与 Ray V2 把同一 `SchedulerOutput` 控制计划按同一 RPC 顺序投递给所有 TP/PP workers；所有 layer owner 必须据此更新本地 request/KV state。
59. PP hidden-state 数据沿相同 TP lane 从前一 stage 异步发送到后一 stage；发送 buffer 在 handle 完成前不得复用，TP collective 与 PP P2P 共同构成不可拆分的 step。
60. `PCP=1` 时 canonical output rank 是 last PP stage 的 TP0；唯一的是 Scheduler 的模型 token commit，KV/EC connector sideband 仍可从多 rank 聚合。
61. 非输出 rank 的 Python 异常不保证直接进入 unique-reply RPC；故障可见性还依赖 collective 传播、execute timeout 或 worker process liveness monitor。
62. 任一 rank 部分失败后，未知完成度的 forward/KV side effect 不能作为部分成功提交；当前系统采用 Executor/Engine fail-stop，而非单-rank step recovery。
63. `BatchDescriptor` 是 graph replay 的最小显式 key：FULL 通常保留精确 `num_reqs/uniform`，PIECEWISE 可放宽这些字段；任何影响命令序列的状态都不能被错误省略。
64. DP ranks 的 runtime CUDA Graph mode 取最小值（`NONE < PIECEWISE < FULL`）；只要一个 rank 不能录图，全部 rank 必须采用兼容的更弱路径。
65. 同步 mode 非 NONE 时，DP graph token envelope 取各 rank 本地 padded token 数的最大值；ModelRunner 必须用该 mode 与 token 数重新 dispatch，并验证 descriptor 一致。
66. graph key 一致不是充分条件：持久输入 `data_ptr`、workspace 地址、collective 参与者/顺序和 padding 无副作用语义也必须稳定。
67. `CUDAGraphEntry` 是 rank-local 长寿命资源，与 compiled module/runner 同寿命而非 request 同寿命；未知 shape 应在 dispatch 层回退 eager，不能在 capture 关闭后偷偷新录图。

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
- async scheduling 的 stale/drop 基本协议已有源码与 fake-runner 测试；真实 PP×MTP×KVConnector 组合的故障注入、跨 rank 顺序和资源归零仍未覆盖。
- `_compute_slot_mappings_kernel` 缺少覆盖跨 block 边界、resumed replace、graph padding、DCP interleave 与多 KV group 的直接单测。
- ModelRunner metadata preparation（staged write、gather、slot mapping）的实际 GPU 时间及其与 full/piecewise graph 边界尚未通过 trace 量化。
- UVA-backed `all_token_ids` 在长上下文、高并发下的 Host/Device 访问和 page-fault 成本尚未测量。
- Multiproc/Ray V2 的控制面与 PP 数据面职责已下钻；其 fan-out 序列化成本、跨节点 MQ 流量和 batch-queue 峰值内存仍未量化。
- `Scheduler.update_from_output()` 每请求 Host commit loop 在 1K request batch 下的 CPU/P99 成本尚未 trace。
- 除 Mooncake store 外，各 KV/EC connector 的后台 job 是否都具备 generation-safe identity、独立资源引用和 Engine liveness hook，尚未系统审计。
- 缺少一条同时断言 `EngineCoreOutput` 终态唯一、Scheduler registry 删除、KV/connector ref 归零与 Worker slot purge 的跨层 finish transaction 测试。
- 缺少在 `torch.compile`/full CUDA Graph capture+replay 下直接断言 KV update 先于 paged read 的 CI guard。
- 缺少重复 slot、跨 request slot alias、缩 batch 后 stale padding slot 的负向测试。
- FlashAttention 等 backend 的 split/fused cache update、NHD/HND cache layout 与公共 metadata contract 尚未形成统一一致性测试。
- 多 KV cache group 与 DCP interleave 下，写侧 slot 与读侧 block table 对同一 allocation 的一致性尚未系统覆盖。
- 缺少显式 seed 请求跨 batch row 重排、request-slot 复用、preemption/resume 与 graph replay 的端到端 token-sequence 兼容测试。
- 现有 Gumbel 统计测试不保证 RNG 实现升级后的 seeded sequence 兼容；若合入 PR #51367 一类 RNG 变更，需要明确版本承诺与迁移策略。
- mixed batch 中任一请求需要 processor 就可能触发整批 FP32 logits copy；不同 processor mix、词表规模与并发度下的吞吐/P99 台阶尚未量化。
- native 与 FlashInfer sampling 在错误语义、分布、GPU backend 和 raw/processed logprobs 模式上的 parity 尚未建立完整 golden fixtures。
- speculative rejection 如何推进或回滚 RNG logical position，以及 stale output 对随机流的影响尚未下钻。
- `PPHandler` 缺少 generation filter、collective skip 对称与 stream/event 生命周期的直接单测；现有 PP×DP E2E 主要验证长度和存活性。
- 开放 PR #46994 指出的 spec sampled width 与 draft-token relay 问题、PR #52179 指出的 sync cadence 问题尚未合入；需要基于最终 post-image 重做 contract。
- Multiproc、Ray V2、legacy Ray 的主路径已对照；仍缺 `TP>1 × PP>1` 故障注入、非输出 rank exception aggregation、external launcher output ownership 与各 backend connector sideband 的统一 contract。
- `_post_process_cudagraph_mode` 缺少 DP mixed-mode 直接单测；尚无 `DP=2 × TP=2` 下 FULL/PIECEWISE/NONE 收敛与 collective trace 对称性的 CI guard。
- production 路径通常不检查 graph replay 输入 `data_ptr`；stale 地址、workspace resize 与多 stream 共享 graph pool 的 fail-fast 边界尚未建立。
- CUDA Graph padding 的额外 FLOPs、DP all-reduce 同步点、capture pool 显存与 P99 收益缺少按 prefill/decode 分层的实测。

## 下一批候选章节

1. 下一主线：`torch.compile` 切图边界——`splitting_ops → PiecewiseBackend → attention/custom op → CUDAGraphWrapper`。
2. Executor 回访：`TP>1 × PP>1` 故障注入、connector sideband 与 external launcher contract。
3. PP 回访：spec sampled width、draft-token relay、sync/async cadence 与 generation-safe CI。
4. Attention 回访：多 KV group、DCP 与 backend cache layout 的一致性测试。
5. 协议专项回访：定义有界 backpressure，并建立 Python↔Rust 双向 golden fixtures。

## 第七篇知识图谱回顾

- 已打通：`LLM.generate → Renderer/InputProcessor → EngineCoreRequest wire → EngineCore queue/failure → Scheduler logical admission → KV physical admission/preemption → Worker device-state materialization`。
- 已闭合边界：用户请求已经能够被追踪为一次具体的 `input_ids/positions/block_tables/slot_mapping` 设备执行计划。
- 当前最大盲区：ModelRunner 输出何时成为 Scheduler 可提交 token；async/speculative progress 如何回滚；finished 状态如何同时释放 Scheduler registry、KV ownership 与 Worker slot。
- 后续路线调整：先完成返回事务，再进入 Attention metadata 和真实 KV cache tensor；暂不提前跳到 Sampling 或 CUDA Graph 优化。
