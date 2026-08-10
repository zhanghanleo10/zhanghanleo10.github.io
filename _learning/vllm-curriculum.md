# vLLM 源码课程账本

## 总体路线

`入口 API 与 Renderer/InputProcessor → 请求对象与跨进程边界 → EngineCore → Scheduler → KV Cache Manager → Executor/Worker → ModelRunner → Attention → Sampling → 分布式执行 → CUDA Graph/torch.compile → Serving → 测试、性能与故障诊断`

## 当前阶段

- 阶段 1：前端请求进入 EngineCore 前的语义冻结与协议边界。
- 当前主线：一次 offline `LLM.generate()` 请求从 prompt 到 Scheduler 的完整生命周期；已完成前端对象冻结与 ADD wire protocol，待完成协议风险收束。

## 已完成章节

| 日期/章节 | 主题 | vLLM commit | 文章 |
| --- | --- | --- | --- |
| 2026-08-10 01 | `LLM.generate → Renderer → InputProcessor → EngineCoreRequest` | [`751f2ccd`](https://github.com/vllm-project/vllm/commit/751f2ccdd3b7ed2ba65dcb04dbd93187edc25b2d) | [请求边界]({{ '/articles/vllm-input-pipeline-request-boundary/' | relative_url }}) |
| 2026-08-10 02 | `SyncMPClient → Msgpack/ZMQ/Tensor IPC → EngineCoreProc.input_queue` | [`51562de5`](https://github.com/vllm-project/vllm/commit/51562de5ab16bacd821d7130187b6bebdd293f93) | [跨进程 wire protocol]({{ '/articles/vllm-enginecore-request-wire-protocol/' | relative_url }}) |

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
- `EngineCore.preprocess_add_request`
- `Request.from_engine_core_request`
- `EngineCoreRequestType`

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

## 前置依赖与版本注意

- 本课程以每章记录的 `main` commit 为准，不把 v0.22.1 专题中的实现自动视为当前事实。
- 直接向 InputProcessor 传 raw prompt、向 LLMEngine 传 `EngineCoreRequest` 均处于 v0.18 移除迁移期。

## 尚未解释的知识债

- `AsyncMPClient`/DP client 的 engine identity 选择、跨 producer 顺序和线程安全边界。
- ADD/ABORT 竞争、`aborts_queue + input_queue` 双写和幂等性。
- output 方向的 reusable msgpack payload、ZMQ high-water mark 与 backpressure。
- EngineCore 如何把 `EngineCoreRequest` 转成 Scheduler `Request`。
- `n>1` parallel sampling 的 parent/child ID 与前端合并。
- multimodal cache 在线程池下的并发契约，以及 PR #50896 的最终结果。
- Rust EngineCore client 与 Python `EngineCoreRequest`/aux frame 的双向协议兼容测试。
- `torch_shm` 中 payload 发送失败后的 orphan tensor 清理与可观测性。

## 下一批候选章节

1. 同日第 3 章：ADD/ABORT 顺序、backpressure、输出 buffer、EngineCore death、Python/Rust compatibility 与升级风险。
2. 下一主线：`EngineCore.add_request → Request.from_engine_core_request → Scheduler.add_request`。
3. 后续：Scheduler waiting/running 状态与 token budget 形成物理 batch。
