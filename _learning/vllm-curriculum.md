# vLLM 源码课程账本

## 总体路线

`入口 API 与 Renderer/InputProcessor → 请求对象与跨进程边界 → EngineCore → Scheduler → KV Cache Manager → Executor/Worker → ModelRunner → Attention → Sampling → 分布式执行 → CUDA Graph/torch.compile → Serving → 测试、性能与故障诊断`

## 当前阶段

- 阶段 9：测试、性能与故障诊断，聚焦 fatal failure、timeout、资源回收和诊断证据链。
- 当前主线：第 22 章已把 fatal 正确性拆成 F1 故障检测、F2 请求收敛、F3 进程收敛、F4 逻辑资源收敛与 F5 设备资源收敛，并审计 ModelRunner exception、Worker death、RPC timeout 三组测试证据；下一章进入 shutdown ownership 与 acknowledgement。

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
| 2026-08-23 14 | `Dynamo full graph → FX partition → PiecewiseBackend/RangeEntry → CUDAGraphWrapper` | [`30b34171`](https://github.com/vllm-project/vllm/commit/30b34171b113887e0e08d7f6d06e2e5a5c33b9d2) | [torch.compile 切图边界]({{ '/articles/vllm-torch-compile-piecewise-splitting-boundaries/' | relative_url }}) |
| 2026-08-24 15 | `HTTP disconnect → AsyncLLM.generate → OutputProcessor tombstone → EngineCore ABORT → KV fence` | [`a7195188`](https://github.com/vllm-project/vllm/commit/a7195188a4b45dec40030467ec6b69b4f1283c8e) | [Serving 断连取消]({{ '/articles/vllm-serving-disconnect-abort-lifecycle/' | relative_url }}) |
| 2026-08-26 16 | `EngineCoreOutputs → AsyncMPClient queue → output_handler → collector coalescing → SSE` | [`a447955a`](https://github.com/vllm-project/vllm/commit/a447955acad919a6902d65f0af2d4f76c0335ed3) | [单槽输出合并]({{ '/articles/vllm-serving-output-coalescing-slow-consumer/' | relative_url }}) |
| 2026-08-27 17 | `P×n Core child → ParentRequest n-way → merge_async_iterators P-way → flattened choice index` | [`d1e5e66e`](https://github.com/vllm-project/vllm/commit/d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302) | [两级 Fan-in]({{ '/articles/vllm-serving-multi-prompt-parallel-sampling-fanin/' | relative_url }}) |
| 2026-08-28 18 | `HTTP 200 → partial SSE → error payload → [DONE] / cancel→abort` | [`6ec92bcb`](https://github.com/vllm-project/vllm/commit/6ec92bcbc8ef9af25cb4834ba3d922115792fbeb) | [HTTP 200 后的错误终态]({{ '/articles/vllm-serving-sse-error-terminal-after-http-200/' | relative_url }}) |
| 2026-08-29 19 | `Core failure → MPClient latch/queue → output_handler → all collectors → /health/watchdog` | [`99013d77`](https://github.com/vllm-project/vllm/commit/99013d77d332a2d21d7214b57fa495f2bad2b448) | [EngineCore 失效广播]({{ '/articles/vllm-enginecore-failure-broadcast-health-readiness/' | relative_url }}) |
| 2026-08-30 20 | `SchedulerOutput → Multiproc RPC deadline → TimeoutError → Core fail-stop → worker termination` | [`680e2177`](https://github.com/vllm-project/vllm/commit/680e2177e473ed8dfaa9773f7ead185b369cab46) | [TP RPC timeout 与单卡 hang 盲区]({{ '/articles/vllm-execute-model-timeout-hang-failstop-gap/' | relative_url }}) |
| 2026-08-31 21 | `SchedulerOutput → log_error_detail → prepare_object_to_dump/make_stats → logger → original exception` | [`c92b29a1`](https://github.com/vllm-project/vllm/commit/c92b29a1d40644da710209f862b1be0ebd5c2e74) | [Fatal Dump 证据与隐私边界]({{ '/articles/vllm-fatal-dump-scheduleroutput-evidence-privacy/' | relative_url }}) |
| 2026-09-01 22 | `ModelRunner exception / Worker death / RPC timeout → EngineCore fatal → frontend fanout / process shutdown / resource evidence` | [`8600db5d`](https://github.com/vllm-project/vllm/commit/8600db5dff18054f7a4314f6f8bba4259e3e2a98) | [Fatal 测试矩阵与资源归零]({{ '/articles/vllm-fatal-path-test-matrix-resource-zero/' | relative_url }}) |

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

- `TorchCompileWithNoGuardsWrapper.__init__/__call__`
- `support_torch_compile`
- `VllmBackend.__call__`
- `split_graph`
- `SplitItem`
- `should_split`
- `PiecewiseCompileInterpreter.call_module`
- `PiecewiseBackend.__init__/__call__`
- `RangeEntry`
- `wrap_with_cudagraph_if_needed`
- `generate_execution_code/compile_execution_fn`
- `CompilationConfig.set_splitting_ops_for_v1`
- `Attention.forward`（custom-op output allocation/mutation boundary）

- `completion.api_router.create_completion`
- `with_cancellation/listen_for_disconnect`
- `OpenAIServingCompletion._create_completion/completion_stream_generator`
- `merge_async_iterators`
- `AsyncLLM.add_request/_add_request/generate/abort`
- `RequestOutputCollector`
- `OutputProcessor.abort_requests/process_outputs`
- `InputProcessor.assign_request_id`（Serving internal ID 生命周期）
- `AsyncMPClient.abort_requests_async`
- `EngineCoreProc.process_input_sockets`（ABORT 双队列）
- `EngineCore._process_aborts_queue`
- `Scheduler.finish_requests/_free_request/_free_request_blocks/_drain_deferred_frees`（取消释放路径）

- `EngineCoreOutput/EngineCoreOutputs`（Serving Host 输出协议）
- `AsyncMPClient._ensure_output_queue_task/get_output_async`
- `AsyncLLM._run_output_handler/generate`（正常输出路径）
- `RequestState.make_request_output/_new_request_output/_new_completion_output`
- `OutputProcessor.process_outputs/propagate_error`（collector 投递与错误广播）
- `RequestOutputCollector.put/get/get_nowait`
- `RequestOutput.add`
- `OpenAIServingCompletion.completion_stream_generator`（SSE 序列化与流内错误）
- `SamplingParams.stream_interval`（请求级降频）

- `CompletionRequest.prompt/n`（多 prompt 与 parallel sampling 输入）
- `OpenAIServingCompletion._create_completion`（per-prompt generator 创建与 P-way fan-out）
- `AsyncLLM.add_request/_add_request`（`n>1` child admission）
- `ParentRequest.__init__/_get_child_sampling_params/get_child_info/get_outputs`
- `merge_async_iterators`（single fast path、P-way task map 与关闭）
- `OpenAIServingCompletion.completion_stream_generator`（二维 choice index 展平）

- `completion.api_router.create_completion`（header 前 JSONResponse 与 streaming 200 分界）
- `GenerateBaseServing._raise_if_error/_convert_generation_error_to_streaming_response`
- `GenerateBaseServing.create_streaming_error_response`
- `create_error_response/sanitize_message`
- `FinishReason.ERROR/GenerationError/EngineGenerateError/EngineDeadError`
- `RequestResponseMetadata.final_usage_info`
- `PrometheusStatLogger.counter_request_success`（`finished_reason` label）
- `OpenAIServingCompletion.completion_stream_generator`（error payload、`[DONE]` 与 cancellation 分界）

- `MPClient.start_engine_core_monitor/_format_exception/get_output_async`（fatal error 双路检测与规范化）
- `AsyncLLM._run_output_handler/is_running/errored/check_health/dead_error`
- `OutputProcessor.propagate_error`（全活跃请求异常广播）
- `RequestOutputCollector.put/get`（Exception 覆盖 pending output）
- `health`（EngineDeadError → HTTP 503）
- `watchdog_loop/terminate_if_errored`
- `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH`
- `engine_error_handler`（非 streaming fatal response 与即时终止检查）

- `EngineCore.step/step_with_batch_queue`（Executor future 等待与 commit 闸门）
- `MultiprocExecutor.execute_model/sample_tokens/collective_rpc/get_response`
- `FutureWrapper.result/_wait_for_response`
- `MultiprocExecutor.start_worker_monitor/_ensure_worker_termination/shutdown`
- `EngineCoreProc._send_engine_dead`（timeout fatal 收敛）
- `UniProcExecutor.collective_rpc/execute_model`
- `AsyncOutputFuture.result`
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
- `VLLM_ENGINE_ITERATION_TIMEOUT_S`
- `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`

- `EngineCore.log_error_detail`
- `dump_engine_exception/_dump_engine_exception`
- `prepare_object_to_dump`
- `NewRequestData.anon_repr/CachedRequestData.anon_repr`
- `Scheduler.make_stats`（fatal path 的聚合统计与 drain/reset 副作用）
- `WorkerProc._execute_worker_rpc`
- `MultiprocExecutor.execute_model/collective_rpc`
- `MultiprocExecutor._run_worker_monitor/register_failure_callback/shutdown`
- `BackgroundResources.validate_alive/_format_exception`
- `AsyncLLM.output_handler`（fatal fan-out）
- `OutputProcessor.propagate_error`
- `tests.v1.shutdown.test_forward_error.evil_forward/test_async_llm_model_error/test_llm_model_error`
- `wait_for_gpu_memory_to_clear`
- `test_ray_v2_executor_worker_death/test_ray_v2_executor_shutdown`
- `test_worker_kill_survivor_unhealthy_and_dead_rejects_retry`
- `tests.v1.executor.test_multiproc_executor_timeout`

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

68. `torch.compile(fullgraph=True)` 只保证 Dynamo 得到完整 FX graph；vLLM backend 仍可在其后按 `splitting_ops` 主动 partition。
69. FX-level split 必须使用 `keep_original_order=True`；KV update 与 attention 还需 dummy tensor dependency 保留隐藏写读顺序。
70. splitting region 不进入 `PiecewiseBackend`；attention 之间的 non-splitting regions 才被 Inductor 编译并包成 PIECEWISE `CUDAGraphWrapper`。
71. attention output 在 custom op 外分配，custom op 通过 mutation schema 原地写入；allocation、mutation 与后继消费的 partition ownership 必须一致。
72. `RangeEntry` 选择 Inductor runnable，`BatchDescriptor` 选择 CUDA Graph replay；两层 shape 路由解决不同问题，不能互相替代。
73. `PiecewiseBackend` 在构造期编译或加载全部 range；运行时 shape 越界必须失败，不能临时扩张编译/capture 集合。
74. splitting ops 的默认集合不仅含 attention，还在 FX early-partition 路径加入 KV-cache update，以隔离 string layer identity 并保住重复 layer artifact 复用。
75. tuple/getitem、`torch.Size` 和 empty allocation 不能任意穿越 AOT submodule 边界；splitter 必须重写或合并这些节点，同时保持 mutation 语义。
76. Streaming response 返回前由 `with_cancellation` 监听 `http.disconnect`；返回后由 `StreamingResponse` 关闭 stream iterator，取消必须沿 `completion_stream_generator → merge_async_iterators → AsyncLLM.generate` 传播。
77. `AsyncLLM.generate` 捕获 `CancelledError/GeneratorExit` 时必须以 collector 的 internal request ID 调用 `abort(internal=True)`；external ID 只用于公开 fan-out/批量身份。
78. Serving abort 先从 `OutputProcessor.request_states` 删除本地状态，再发远端 ABORT；迟到 `EngineCoreOutput` 因无 state 被忽略，不能重新进入已取消 stream。
79. ABORT 的 eager queue 保证执行期间到达的取消先于 `update_from_output`，ordered input queue 保证 ADD/ABORT 相对顺序；双路径正确性依赖 Scheduler finish 幂等。
80. 显式 external abort 可向仍存活消费者产生 `finish_reason=abort` 的 final output；disconnect 路径消费者已取消，二者共享 cleanup 机制但外部可观察语义不同。
81. 请求 terminal 与 KV block 物理复用分离：在途 GPU 写未越过 `processed_step_seq` 时，blocks 必须进入 `deferred_frees`，不能因 Host tombstone 提前复用。
82. `AsyncMPClient.outputs_queue` 没有 `maxsize`；socket reader 通过 `put_nowait` 与 output handler 解耦，因此 handler 落后会转化为 Host backlog，而非 Core 侧等待。
83. `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 只限制单次 event-loop 占用时间，不限制 `EngineCoreOutputs`、ZMQ 或 request payload 容量。
84. `RequestOutputCollector` 至多持有一个 pending 对象，但 DELTA 会扩展该对象的 text/token_ids/logprobs；对象数有界不能推出 bytes 有界。
85. DELTA 按 `CompletionOutput.index` 追加并采用最新终态 metadata；CUMULATIVE 按 index 替换最新 snapshot，同时保留其他 choice index。
86. `stream_interval` 的 request 值只能高于 engine default；它降低 Host/SSE 事件频率，不是慢消费者容量协议。
87. collector 的 exception put 覆盖 pending data，保证失败优先于未发送普通 chunk；SSE 已开始后错误只能作为流内事件收敛。
88. 当前输出路径优先隔离 GPU/Core 与单个网络消费者；若要严格有界，必须在无损、上游不停和有限内存三者中显式选择拒绝或背压策略。
89. 多 prompt × `n>1` 是两级 fan-in：每 prompt 的 n 个 child 先经 `ParentRequest`/共享 collector 收敛，外层 merge 只接收 P 个 prompt generator。
90. `n>1` child 是独立 `EngineCoreRequest`，其 `SamplingParams.n` 被改为 1；parent identity、child set 与 output aggregator 都是 frontend Host 状态，不是 Scheduler group。
91. streaming parent finished 以所有 child 首次终态后 `child_requests` 归零为准；重复 child 终态不得再次进入公开输出。
92. `merge_async_iterators` 对每个 active source 最多持有一个 pending `anext`，task/state 上界为 O(P)，但整个调用的 Core request/KV 规模仍为 O(P×n)。
93. SSE `choice.index = local_choice_index + prompt_idx × n`；公开 index 与到达顺序解耦，跨 prompt 交错顺序不属于稳定协议。
94. `FIRST_COMPLETED` 提供 ready-source 及时转发，不提供 round-robin、配额或稳定 tie-break；work-conserving 不等于严格公平。
95. outer source 异常使整个 Completion call 失败并触发其余 source 的 close；单 prompt fast path 也必须显式 `aclose` 底层 generator，才能把关闭传播到 `AsyncLLM.generate` abort。
96. `StreamingResponse` 一旦开始发送，HTTP status 已固定为 200；后续 `ErrorResponse.error.code=500` 是 SSE payload 字段，不能改写 transport status。
97. `FinishReason.ERROR` 必须在 Serving 边界转成 `GenerationError`；连接仍可写时，GenerationError 与普通 Exception 都以 error data frame 后接 `[DONE]` 收敛。
98. `CancelledError`/generator close 不进入普通 `except Exception`，因此 disconnect 不保证 error frame 或 `[DONE]`；其正确性由 abort、tombstone、Scheduler/KV 最终释放定义。
99. `[DONE]` 只表示 SSE iterator 不再产生 event，不蕴含生成成功；客户端必须把已见 error payload 作为最终失败状态。
100. normal stream 尾部才赋 `RequestResponseMetadata.final_usage_info`；error 提前跳转到 catch 时没有 canonical final usage，partial usage 不能自动视为完整计费事实。

101. Core process monitor 与 output channel exception 两条检测路径最终汇入单向 `engine_dead` latch 和 `EngineDeadError`；上层不依赖具体 ZMQ/EOF/exit-code 形态。
102. `output_handler` 的 fatal catch 把同一个异常广播给所有 active collectors；collector 中 Exception 覆盖 pending normal output，错误优先于未消费数据。
103. `EngineDeadError` 是共享 EngineCore 的不可恢复终态；`generate()` 不再发送 per-request ABORT，因为 client/server shutdown 承担全局资源回收。
104. `resources.engine_dead` 或 `output_handler.done()` 任一成立都会使 `errored=True`；后续请求在建立 frontend state 和发送 ADD 之前被拒绝。
105. `/health` 仅在 `check_health()` 正常返回时给 200，只把 `EngineDeadError` 映射为 503；render-only server 固定 200。它检查 fail-stop health/admission，不检查 forward progress。
106. server watchdog 每 5 秒轮询；默认在 `errored && !is_running` 时设置 `server.should_exit`。keep-alive 只禁止 server exit，不恢复请求 admission，也不把 `/health` 改回 200。
107. 官方 Kubernetes/Helm 示例把同一 `/health` 同时用于 readiness 与 liveness，因此流量摘除和容器重启语义被合并；keep-alive 若用于诊断，需要独立的进程 liveness 设计。
108. 当前 `/health` 无法发现进程/handler 存活但 GPU/NCCL 不再前进；PR #45453 的 progress endpoint 方案已关闭且未合入，不能视为现有功能。
109. Multiproc `execute_model` 在 RPC 创建时绑定一个共享绝对 deadline；所有必要 response queue 消耗同一预算，不能按 rank 累加完整 timeout。
110. RPC timeout 只证明 deadline 前未收到必要响应，不证明 GPU/NCCL/MQ 根因，也不保证设备工作已取消。
111. worker process monitor 检测进程退出，不检测 alive-but-stalled forward progress；shutdown grace/TERM/KILL 是检测之后的资源回收阶段。
112. `SchedulerOutput` 只有在 `future.result()` 成功后才进入 `Scheduler.update_from_output`；timeout 时 completion unknown，不能在同一 Engine 中安全原地重试。
113. 当前 V1 runtime 未消费 `VLLM_ENGINE_ITERATION_TIMEOUT_S`；UniProc 对 alive GPU hang 没有由该变量提供的 60 秒 watchdog，不能把配置声明误作生效契约。
114. fatal dump 位于 `future.result/sample_tokens` 与 `Scheduler.update_from_output` 之间；它记录 Host 执行计划，不能证明 device completion。
115. normal step 与 batch queue 都必须用触发异常的 matched `SchedulerOutput` dump，不能错配到更新的一轮计划。
116. `dump_engine_exception` 必须 exception-free，诊断失败不能覆盖原始 model execution exception；`BaseException` 不属于该 dump 契约。
117. token/Tensor 脱敏是字段级协议：New/Cached request token 改为长度、Tensor 只留 metadata，但 SchedulerOutput 其他嵌套字段仍可能保留 request/block/spec/connector 数据。
118. `Scheduler.make_stats()` 是聚合观测而非 registry/KV snapshot；且会 reset prefix stats、connector stats 并 drain eviction events。
119. 当前 dump 只写进程内 logger，没有持久化文件、原子提交、schema version 或跨 rank 一致性边界。
120. fatal evidence 只能证明 scheduler intent 与当时可读的 Host 聚合状态；SIGKILL/native crash、日志截断和设备内部进度仍是明确盲区。
121. fatal 正确性必须分为 F1 故障检测、F2 请求收敛、F3 进程收敛、F4 逻辑资源收敛和 F5 设备资源收敛；任一单层通过都不能替代其余层。
122. forward-error shutdown 测试直接证明 TP=1/2 下在途请求 fatal fan-out、new admission 拒绝和粗粒度 GPU memory 回落，但没有直接断言 Scheduler/KV/connector/Worker state 精确归零。
123. Worker monitor 通过 OS process sentinel 检测 death 并关闭整个 Worker group；alive-but-stalled Worker 只能由 RPC deadline 覆盖。
124. Worker 被 SIGKILL 后不能再提供自身 cleanup acknowledgement；应断言进程消失、parent-owned IPC/actor refs 清理和设备 baseline，而不是虚构 victim 侧逻辑回执。
125. multiprocess timeout 现有直接测试验证 deadline clamp、剩余时间与 FIFO drain 的单元语义；真实 withheld-response 到 ENGINE_CORE_DEAD、collector fan-out 和资源收敛仍无直接 E2E。
126. 进程退出或显存低于阈值只证明地址空间/设备资源粗粒度收敛，不能反证 KV refcount、connector job 与 Worker request state 走过正常逻辑释放路径。

## 前置依赖与版本注意

- 本课程以每章记录的 `main` commit 为准，不把 v0.22.1 专题中的实现自动视为当前事实。
- 直接向 InputProcessor 传 raw prompt、向 LLMEngine 传 `EngineCoreRequest` 均处于 v0.18 移除迁移期。

## 尚未解释的知识债

- `AsyncMPClient`/DP client 的 engine identity 选择、跨 producer 顺序和线程安全边界。
- hybrid KV groups 下 per-group prefix blocks 如何收敛为统一 `num_computed_tokens`，以及 partial-tail/CoW 的正确性边界。
- watermark、`scheduler_reserve_full_isl` 与 async KV load `reserved_blocks` 的容量策略和配置取舍。
- 多 prompt × `n>1` 的基本 parent/child 与两级 fan-in 已解释；仍缺 sampling streaming 的组合 E2E、单源异常/断连、admission 中途失败回滚、pending task/iterator close 与 KV 归零证明。
- multimodal cache 在线程池下的并发契约，以及 PR #50896 的最终结果。
- Rust EngineCore client 与 Python `EngineCoreRequest`/aux frame 的双向协议兼容测试。
- `torch_shm` 中 payload 发送失败后的 orphan tensor 清理与可观测性。
- 慢/断连 output consumer 下，`AsyncMPClient.outputs_queue` 对象数、collector payload bytes、ZMQ/SSE buffer 与 P99 的联合曲线尚未测量；需要 per-request byte/age 指标和明确的超限 abort/背压策略。
- 大量 waiting 请求的 Host metadata、prompt embedding 和 multimodal feature 内存曲线，以及服务层 admission 上限。
- `connector.on_new_request()` 抛异常时 queue/registry 的回滚原子性。
- EngineCore 硬失败的基本 fan-out、new-admission 拒绝、health 503 与 GPU cleanup 已解释；仍缺 DP/Ray/external launcher 同时失败、keep-alive 下 `request_states` retention、真实 Uvicorn watchdog/503/exit 时序。
- `/health` 不检查 forward progress；alive-but-stalled GPU/NCCL 只能等待其他 timeout/monitor。closed-unmerged PR #45453 未进入当前 `main`。
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
- 缺少真实 decoder layer partition topology golden；operator rename、新 custom op 或条件分支可能改变 split 拓扑而 CI 不报错。
- 缺少 attention output alias/poison 测试，以及 KV update→attention 在 AOT cache load、TP collective 与 CUDA Graph replay组合下的顺序 guard。
- FX early partition 与 Inductor late partition 的 cold start、artifact 复用、graph pool 显存和线上 P99 尚无同模型对照。
- 缺少从真实 ASGI socket disconnect 到 `AsyncLLM.generate` abort、Scheduler terminal、KV/connector ref 归零的跨层测试；当前 mid-stream 测试通过提前退出 async iterator 模拟取消。
- 多 prompt × `n>1` 断连是否完整关闭所有 iterator/child、ADD 尚未被 Core 消费时的快速取消、API 进程硬杀后的孤儿请求回收尚未系统覆盖。
- collector exception 覆盖 pending chunk、SSE error event 与 `[DONE]` 的完整序列缺少端到端断言；logprobs × 长输出 × `n>1` 的 payload 复制成本也未基准化。
- connector 延迟释放与 GPU in-flight deferred fence 同时存在时，缺少断言所有引用和 blocks 最终归零的故障注入。
- 缺少真实 ASGI late-error 联合测试：`HTTP 200 → partial data → error(code=500) → [DONE]`，并同时断言 HTTP 2xx bucket、engine `finished_reason=error` 与 `final_usage_info is None`。
- 缺少 `P>1 × n>1` 单 child error 后所有 merge task、collector、Core child、KV blocks 与 connector refs 归零的跨层故障注入。
- completion streaming 没有 resume cursor/exactly-once contract；partial output 后的 retry、deduplication 与 usage/accounting 语义尚未形成稳定协议。
- `vllm:request_success{finished_reason="error"}` 的命名可能被 dashboard 误聚合；HTTP 2xx 与 engine error 的跨指标告警规则尚未建立。
- 缺少 alive Worker withheld-response 的跨层测试：`execute_model` deadline 到期后应断言 `TimeoutError → ENGINE_CORE_DEAD → all collectors`，并验证 Scheduler/KV/connector/device context 随进程退出归零。
- TP=1 UniProc alive GPU/NCCL hang 目前缺少独立 supervisor/progress watchdog；`VLLM_ENGINE_ITERATION_TIMEOUT_S` 的 V1 语义需要删除、接线或明确弃用。
- Multiproc、Ray V2、legacy Ray 和 external launcher 对 execution deadline、process death、callback 可抢占性与 termination ownership 尚无统一 contract。
- 300 秒静态阈值在长 prefill、首次 compile/capture、大 TP collective 与严格 SLO 之间缺少分布式 trace 和 workload-aware 配置准则。

- fatal dump 缺少 Tensor/anon_repr 隐私矩阵、原异常保留、matched batch-queue plan 与 make_stats drain/reset 的直接测试。
- 缺少 deterministic、versioned、bounded 的 structured crash envelope；当前 dict/set 字段顺序不稳定，日志不适合作为机器重放格式。
- 缺少独立 supervisor 持久化、rank/stream progress、KV block refcount/free-list 摘要，以及 SIGKILL/native crash 下的证据保全。
- `scheduled_spec_decode_tokens` 与 connector/mm metadata 的字段级敏感性尚未形成统一审计和日志大小上限。

- ModelRunner exception 的现有 E2E 已覆盖 fatal fan-out、new-admission 拒绝和 GPU memory threshold，但缺 owner-specific resource census；进程销毁不能替代 F4 精确断言。
- Worker death 的 Ray V2/FT 测试覆盖 callback、状态转换和部分关闭语义，尚缺默认 MP 全链 collector/KV/connector/device cleanup parity。
- multiprocess timeout 只有 fake-clock/deadline 单元测试，尚缺真实 alive Worker withheld-response 的 fatal convergence 与资源基线 E2E。

## 下一批候选章节

1. 下一主线：shutdown ownership——`EngineCore.shutdown → Executor/Worker termination → Scheduler/KV/connector cleanup` 中谁拥有最后一次释放权，哪些 owner 能在 fatal 前返回 acknowledgement。
2. Hang 回访：TP=1 supervisor/progress heartbeat、alive Worker withheld-response E2E、`TimeoutError → ENGINE_CORE_DEAD → collectors` 与跨 Executor deadline parity。
3. 输出性能回访：真实慢读 socket、collector bytes/age、logprobs 长输出和 per-request budget。
4. fan-in 回访：P>1×n>1 sampling streaming、单源异常/断连、admission rollback 与 task/KV 归零 E2E。
5. 错误协议回访：真实 ASGI late-error 的 status/body/metrics/usage 联合测试与客户端 retry contract。
6. Executor 回访：`TP>1 × PP>1` 故障注入、connector sideband 与 external launcher contract。
7. 协议专项回访：定义有界 backpressure，并建立 Python↔Rust 双向 golden fixtures。

## 第七篇知识图谱回顾

- 已打通：`LLM.generate → Renderer/InputProcessor → EngineCoreRequest wire → EngineCore queue/failure → Scheduler logical admission → KV physical admission/preemption → Worker device-state materialization`。
- 已闭合边界：用户请求已经能够被追踪为一次具体的 `input_ids/positions/block_tables/slot_mapping` 设备执行计划。
- 当前最大盲区：ModelRunner 输出何时成为 Scheduler 可提交 token；async/speculative progress 如何回滚；finished 状态如何同时释放 Scheduler registry、KV ownership 与 Worker slot。
- 后续路线调整：先完成返回事务，再进入 Attention metadata 和真实 KV cache tensor；暂不提前跳到 Sampling 或 CUDA Graph 优化。


## 第二次七章知识图谱回顾（第 08–14 章）

- 已打通：`ModelRunnerOutput commit/rollback → Attention KV write/read → Sampling → PP state convergence → TP×PP Executor DAG → distributed graph key → torch.compile partition/replay`。
- 已闭合：Scheduler 与设备执行的返回事务、跨 rank 控制/数据面、CUDA Graph key 和 FX mutation ordering 已能串成同一正确性协议。
- 当前最大盲区：内部 fail-stop、取消和 finished 状态如何穿过异步 Serving 协程，形成唯一且可观察的 HTTP/streaming 终态。
- 后续路线调整：进入 Serving；随后以跨层测试、性能与故障注入回收 graph、connector、backpressure 和 finish transaction 知识债。



## 第三次七章知识图谱回顾（第 15–21 章）

- 已打通：`disconnect/abort → KV deferred free`、`output coalescing → P×n fan-in → SSE terminal`、`timeout/exception → fatal evidence → EngineDeadError/health`。
- 已闭合：正常取消与 fatal failure 的资源语义已经分离；HTTP 终态、Engine admission health 和 forward progress 也不再混为同一不变量。
- 当前最大盲区：fatal-path 测试还不能同时证明所有 request collector、Scheduler registry、KV/connector refs、Worker/device context 随 Engine 退出归零；dump 也不是 durable snapshot。
- 后续路线调整：以 fault injection、structured diagnostics 和资源归零测试收束故障诊断阶段，再回访 connector、backpressure 与跨 backend parity。
