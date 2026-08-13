---
layout: page
title: "PTO 全栈课程账本"
permalink: /learning/pto-curriculum/
---

# PTO 全栈课程账本

最后更新：2026-08-13。

## 总体路线

1. PTO ISA 基础：Tile 类型系统、GlobalTensor、valid region、layout 与 memory location
2. 数据搬运与计算指令：load/store/move、elementwise、reduce、matmul
3. Event、pipeline、double buffering、TPipe/RingBuffer 与 Tile 生命周期
4. Python DSL 捕获与 Tensor/Tile IR
5. Block/Execution Graph、SPMD 与 task DAG
6. PTOAS verifier、Pass、EmitC/设备产物
7. simpler Host/AICPU/AICore runtime、TensorMap/RingBuffer
8. pypto-lib kernel/模型、Golden、性能与 serving 集成

当前阶段：ISA 基础阶段。已建立 Tile、布局、有效区、Event 与跨核 Ring ownership 基础；下一步补齐 GlobalTensor 地址映射，再进入 TMOV/TMATMUL、reduce 与真实 GEMM kernel。

## 已完成章节

| 日期 | 章节 | 核心中间产物 | 文章 |
| --- | --- | --- | --- |
| 2026-08-10 | DeepSeek-V4 SWA page-run TLOAD | block IDs → block table → SWA indices → L1 Tile → TLOAD | [课程 01]({% post_url 2026-08-10-pto-deepseek-v4-swa-page-run-tload %}) |
| 2026-08-11 | Tile 容量、有效区、布局与位置 | capacity shape + valid region + ND/NZ/ZN + TileType | [课程 02]({% post_url 2026-08-11-pto-isa-tile-capacity-valid-layout %}) |
| 2026-08-12 | Event 与 load-compute-store 流水 | MTE2 → Event → PIPE_V → Event → MTE3 | [课程 03]({% post_url 2026-08-12-pto-isa-event-pipeline %}) |
| 2026-08-13 | 双向 TPipe 的 Ring ownership | PyPTO workspace → 双 GM ring → TPUSH/TPOP → ready/free credits | [课程 04]({% post_url 2026-08-13-pto-isa-bidirectional-tpipe-ring-ownership %}) |

## ISA 知识地图

| 主题 | 已确认结论 | 状态 |
| --- | --- | --- |
| Tile static capacity | 编译期决定资源盒子、布局合法性和指令特化 | 已讲透基础 |
| Valid region | destination 通常定义语义域；是连续前缀，不是任意 mask；区外默认 unspecified | 已讲透基础 |
| TileType/location | Vec、Mat、Left、Right、Acc 等位置参与指令重载与校验 | 已讲基础 |
| Layout | BLayout 描述外层次序，SLayout/SFractalSize 描述盒化基块；逻辑 shape 相同不等于物理布局相同 | 已讲基础 |
| 对齐 | 未盒化 row-major 行宽通常需要 32 B 对齐；尾块用固定 capacity + dynamic valid 表达 | 已讲基础 |
| TLOAD/TSTORE | 实际传输范围受 valid region 控制；跨布局、dtype、location 受代际约束 | 已讲 2 次 |
| TADD | 以 destination valid region 为迭代域；输入兼容性仍是调用方 contract | 已讲基础 |
| Event | 精确表达 producer/consumer pipeline 依赖，不是全局 barrier | 已讲透基础 |
| Pipe mapping | TLOAD→MTE2、TADD→V、Vec TSTORE→MTE3、TMATMUL→M | 已讲基础 |
| Event ID 生命周期 | Record 后必须有 Wait；ID 有限并按 pipe pair 管理 | 已讲基础 |
| TPipe/TPUSH/TPOP | A2/A3 跨 Cube/Vector Tile 传递包含 GM payload ring 与 ready/free credit | 已讲基础 |
| DIR_BOTH ownership | C2V/V2C 必须使用两条不重叠 ring；总 footprint=`2×SlotNum×SlotSize` | 已讲透核心缺陷 |
| Credit batching | 初始 SlotNum 个 credit 免等待；之后按 SyncPeriod 批量 free/wait | 已建立模型，待完整演算 |
| Double buffering | 既需 producer→consumer，也需 consumer→下一 producer 的 slot 复用依赖 | 已建立问题 |
| Matrix/Reduce/通信 ISA | 尚未系统覆盖 | 待学习 |

## 六仓版本与覆盖矩阵

| 仓库 | 最近分析 commit | 已覆盖文件/符号 | 覆盖状态 |
| --- | --- | --- | --- |
| pto-isa | [67e230d](https://github.com/hw-native-sys/pto-isa/commit/67e230d5e92fe351303a8b5b7cc16809e4a0532e) | pto_tile.hpp、event.hpp、fifo.hpp、TPush.hpp、TPop.hpp、Tile/Event、TLOAD/TADD/TSTORE、TPUSH/TPOP | ISA 深挖 3 |
| simpler | [a8d7ce1](https://github.com/hw-native-sys/simpler/commit/a8d7ce12c7433442f4930baf9daf6ab4e3b7edb5) | Worker、compute_task_fanin、orchestrator TensorMap stages | 入门 |
| PTOAS | [307d048](https://github.com/hw-native-sys/PTOAS/commit/307d0484a9e7d5e36f01b253d2bebe4d2f45fe81) | TLoadOp::verify、PTOTLoadToTLOAD、dynamic subview tests | 初步 |
| pypto | [7102058](https://github.com/hw-native-sys/pypto/commit/71020585278b68f56c72c40d5570f07dbb20bc8b) | create_l1/gather_row、Tensor→Tile、gm_pipe_layout、ComputeGMPipeWorkspaceElements、PrepareGMSlotBufferLayout | 跨仓深挖 1 |
| pypto-lib | [5b8d1e9](https://github.com/hw-native-sys/pypto-lib/commit/5b8d1e9846ff7401f0f8525bc5a5b67c8191c13e) | build_swa_metadata、decode_sparse_attn_csa、golden | 深挖 1 |
| pypto-serving | [272b874](https://github.com/hw-native-sys/pypto-serving/commit/272b87492695f78d44c2e8cfe808f372706de594) | cache metadata、prepared inputs、_run_l3 | 初步 |

## 已确认接口与不变量

- Tile capacity shape 是静态资源/type contract；valid region 是本次运行的真实语义域。
- Valid region 是左上角连续前缀；区外元素除非指令明示，否则不得假设为零或保持不变。
- Tile location 与 layout 都参与指令合法性；shape/dtype 相同不足以证明两个 Tile 可互换。
- Scheduler/CacheManager 拥有 physical block ID；Serving 打包固定形状 shared tensor。
- SWA 页内连续性允许 page-run；compressed top-k 没有相同保证。
- gather_row.shapes 是静态盒子，valid_shape 是运行期传输范围；必须同时限制目标 subview 与 GM partition。
- Event 是跨 pipe 依赖边；Record/Wait 生命周期必须闭合。
- Auto 模式把放置和同步责任交给编译器，不代表硬件不需要同步。
- CPU-SIM 可验证数学语义，但不能单独证明真机跨 pipeline event 正确。
- simpler 负责 task 依赖与运行时序，不解释 KV 分页或 ISA 内部 Tile 语义。
- A2/A3 `DIR_BOTH` 的 C2V/V2C 是两条独立 GM ring；V2C base=`GM_SLOT_BUFFER+SlotNum×SlotSize`。
- 两方向独立 flag 只保证逻辑顺序，不保证 payload 物理隔离；producer 与 consumer 必须共享完全相同的方向 base。
- `pypto` workspace 分配、pipe offset 与 `pto-isa` 地址计算构成跨仓布局 ABI；扩容必须先于启用第二段地址。
- 一个仍在途的 payload 在 ready 到 free 的生命周期内必须独占其完整物理字节区间。

## 待验证推断

- `TPOP_IMPL(Pipe&, GlobalData&)` 在 A2/A3 `DIR_BOTH` 的 Cube 分支未显式应用 `cons.entryOffset`，是否构成 V2C 直接视图地址错误，尚缺直接调用与设备测试。
- `SyncPeriod` 的批量 credit 可减少同步消息，但对具体吞吐/等待时间的影响需要 device trace 和对照实验。
- TPUSH 文档的 A2/A3 local FIFO 表述可能混淆“GM transport backing”与“consumer-local destination”。

## 尚未解释的知识债

- GlobalTensor 的五维 shape/stride、partition 与二维 Tile 视图映射。
- TMOV、TRESHAPE、transpose 与 ND/NZ/ZN 布局转换的完整合法矩阵。
- TMATMUL 的 Left/Right/Acc Tile、fractal 与 Cube pipeline。
- reduce 指令的 valid region、精度提升和跨行/列语义。
- 默认 SlotNum=4 下 credit batching、ring wrap-around、析构 drain 的完整状态机。
- A2/A3 GM ring 与 A5 consumer SRAM 的 TPUSH/TPOP 差异。
- Tensor Graph → Tile Graph → Block Graph → Execution Graph 的 pass 顺序。
- pl.spmd 到 task payload、resource shape 与物理 core 的映射。
- PTOAS bytecode/device binary、版本 ABI 与跨仓 CI。
- A2/A3 与 A5 的同步、DMA、layout 和数值差异。

## 下一批候选主线

1. 深挖 GlobalTensor 五维 shape/stride/partition，补齐 GM→Tile 地址映射与 subOffset 边界。
2. 用真实 GEMM kernel 串起 GlobalTensor、TLOAD、Left/Right/Acc Tile、TMATMUL 与 TSTORE。
3. 分析 TMOV/TRESHAPE 和 ND/NZ/ZN 转换，解释 layout conversion 的成本。
4. 补齐默认深度 TPipe 的 credit batching、wrap-around 与 GlobalTensor TPOP 疑点。
5. 再向 PyPTO/PTOAS 追踪 Tile/Event/TPipe contract 如何在 IR 中表达和 lowering。
