---
layout: page
title: "PTO 全栈课程账本"
permalink: /learning/pto-curriculum/
---

# PTO 全栈课程账本

最后更新：2026-08-10。

## 总体路线

1. 真实模型请求与缓存状态：`pypto-serving → pypto-lib`
2. Python DSL 捕获与 Tensor/Tile IR
3. Block/Execution Graph、SPMD 与 task DAG
4. PTOAS verifier、Pass、EmitC/设备产物
5. pto-isa Tile/Buffer/Pipeline 与跨代实现
6. simpler Host/AICPU/AICore runtime、TensorMap/RingBuffer
7. Golden、数值正确性、性能分析与模型级集成
8. 版本兼容、CI、架构债与完整模型案例

当前阶段：阶段 1，先用 DeepSeek-V4 decode 的真实数据路径建立六仓边界，再下钻编译和运行时。

## 已完成章节

| 日期 | 章节 | 核心中间产物 | 文章 |
| --- | --- | --- | --- |
| 2026-08-10 | DeepSeek-V4 SWA page-run TLOAD | block IDs → block table → SWA indices → L1 Tile → TLOAD | [课程 01]({% post_url 2026-08-10-pto-deepseek-v4-swa-page-run-tload %}) |

## 六仓版本与覆盖矩阵

| 仓库 | 本章 commit | 已覆盖文件/符号 | 覆盖状态 |
| --- | --- | --- | --- |
| pto-isa | [`0cefc9a`](https://github.com/hw-native-sys/pto-isa/commit/0cefc9a5a1c24c62655cc345d408559595a8af32) | `TLOAD` 文档；A2/A3、A5 `TLoad.hpp` | 入门 |
| simpler | [`a8d7ce1`](https://github.com/hw-native-sys/simpler/commit/a8d7ce12c7433442f4930baf9daf6ab4e3b7edb5) | `Worker.register/submit/run`；`compute_task_fanin`；orchestrator TensorMap stages | 入门 |
| PTOAS | [`307d048`](https://github.com/hw-native-sys/PTOAS/commit/307d0484a9e7d5e36f01b253d2bebe4d2f45fe81) | `TLoadOp::verify`；`PTOTLoadToTLOAD`；dynamic subview lit tests | 初步 |
| pypto | [`2dc2bb2`](https://github.com/hw-native-sys/pypto/commit/2dc2bb25434afa7b6229e6f1b6c9b361f81f286f) | `create_l1/gather_row`；Tensor→Tile conversion；`MakeGatherRowCodegenPTO`；dynamic-valid ST test | 初步 |
| pypto-lib | [`5b8d1e9`](https://github.com/hw-native-sys/pypto-lib/commit/5b8d1e9846ff7401f0f8525bc5a5b67c8191c13e) | `build_swa_metadata`；DSpark/MTP `decode_sparse_attn_csa`；golden/spec/runner | 深挖 1 |
| pypto-serving | [`272b874`](https://github.com/hw-native-sys/pypto-serving/commit/272b87492695f78d44c2e8cfe808f372706de594) | `DeepSeekV4CacheMetadataBuilder`；`PreparedDecodeInputs`；`prepare_decode_inputs`；`_run_l3` | 初步 |

## 已确认跨仓接口与不变量

- Scheduler/CacheManager 拥有 physical block ID；Serving 将其打包为固定形状、runner-owned shared tensors。
- `build_swa_metadata` 在设备 program 内把 absolute logical position 经 block table 映射成 physical row。
- SWA 页内连续性允许 page-run；compressed top-k 没有相同保证。
- `gather_row.shapes` 是编译期物理盒子，`valid_shape` 是运行期真实传输范围；后者必须同时限制目标 subview 与 GM partition。
- PyPTO 的 `tile.gather_row` lower 为 `pto.subview + pto.partition_view + pto.tload`，不是 `MGATHER`。
- PTOAS 校验并把 `pto.tload` 发为 pto-isa `TLOAD`；simpler 负责 task 依赖与运行时序，不解释 KV 分页。
- `DeepSeekV4PreparedDecodeInputs` 字段只在下一次 decode prepare 前有效；临时 upload 在 `_run_l3` finally 中释放。

## 前置依赖

- Python、PyTorch、Paged KV Cache、Ascend AIC/AIV/AICPU 基础。
- 后续章节默认理解本章的 block table、static shape 与 runtime valid shape 区分。

## 尚未解释的知识债

- Tensor Graph → Tile Graph → Block Graph → Execution Graph 的完整 pass 顺序和中间 IR。
- `pl.spmd` grid 到 `PTO2TaskPayload`、resource shape 与物理 core 的精确映射。
- simpler TensorMap region overlap、RingBuffer backpressure、task completion/reclamation。
- PTO Bytecode/device binary 的生成、缓存、注册和版本 ABI。
- Serving 六组 cache pool 的分配、共享、eviction 和 request teardown。
- CSA/HCA/compressed cache 的联合数据流、量化误差与全模型 golden。
- A2/A3 与 A5 的 graphability、同步、DMA 和数值差异。
- 六仓发布/安装版本兼容矩阵与跨仓 CI。

## 下一批候选主线

1. `pl.spmd(24)` 如何变成 Execution Graph task，并由 simpler TensorMap 建 DAG。
2. DeepSeek-V4 decode metadata 六组 block table 与 slot mapping 的完整生命周期。
3. PTOAS `pto.tload` 周边的 subview/partition legalization 与 A2/A3/A5 lowering 对照。
4. pypto-lib Golden runner：Torch reference、resident mode、回读与误差阈值。
5. 从一个 CSA layer 扩展到完整 `decode_fwd` 的 cache/weight/residual pipeline。
