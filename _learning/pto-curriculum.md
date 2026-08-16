---
layout: page
title: "PTO 全栈课程账本"
permalink: /learning/pto-curriculum/
---

# PTO 全栈课程账本

最后更新：2026-08-16。

## 总体路线

1. PTO ISA 基础：Tile 类型系统、GlobalTensor、valid region、layout 与 memory location
2. 数据搬运与计算指令：load/store/move、elementwise、reduce、matmul
3. Event、pipeline、double buffering、TPipe/RingBuffer 与 Tile 生命周期
4. Python DSL 捕获与 Tensor/Tile IR
5. Block/Execution Graph、SPMD 与 task DAG
6. PTOAS verifier、Pass、EmitC/设备产物
7. simpler Host/AICPU/AICore runtime、TensorMap/RingBuffer
8. pypto-lib kernel/模型、Golden、性能与 serving 集成

当前阶段：ISA 基础阶段。已打通 `GlobalTensor → TLOAD/MTE2 → L1 Mat → TMOV/MTE1 → L0A/L0B → TMATMUL/M → Acc → TSTORE/FIX`，并完成第七篇知识图谱回顾；下一步聚焦非整除 GEMM tail、compact transfer 与 padding ownership。

## 已完成章节

| 日期 | 章节 | 核心中间产物 | 文章 |
| --- | --- | --- | --- |
| 2026-08-10 | DeepSeek-V4 SWA page-run TLOAD | block IDs → block table → SWA indices → L1 Tile → TLOAD | [课程 01]({% post_url 2026-08-10-pto-deepseek-v4-swa-page-run-tload %}) |
| 2026-08-11 | Tile 容量、有效区、布局与位置 | capacity shape + valid region + ND/NZ/ZN + TileType | [课程 02]({% post_url 2026-08-11-pto-isa-tile-capacity-valid-layout %}) |
| 2026-08-12 | Event 与 load-compute-store 流水 | MTE2 → Event → PIPE_V → Event → MTE3 | [课程 03]({% post_url 2026-08-12-pto-isa-event-pipeline %}) |
| 2026-08-13 | 双向 TPipe 的 Ring ownership | PyPTO workspace → 双 GM ring → TPUSH/TPOP → ready/free credits | [课程 04]({% post_url 2026-08-13-pto-isa-bidirectional-tpipe-ring-ownership %}) |
| 2026-08-14 | GlobalTensor 五维地址映射 | pointer + 5D shape/stride/layout → ND/DN folding → DMA burst/gap → 2D Tile | [课程 05]({% post_url 2026-08-14-pto-isa-globaltensor-5d-address-mapping %}) |
| 2026-08-15 | 真实 GEMM 的 TMATMUL 与 K 分块累加 | GM ND/DN → L1 Mat → L0 Left/Right → fp32 Acc → FIX/TSTORE | [课程 06]({% post_url 2026-08-15-pto-isa-gemm-tmatmul-k-accumulation %}) |
| 2026-08-16 | TMOV：L1 Mat 到 L0A/L0B 的角色化重排 | L1 NZ/ZN → MTE1 → L0A ZZ/L0B ZN → TMATMUL | [课程 07]({% post_url 2026-08-16-pto-isa-tmov-l1-to-l0-fractal %}) |

## ISA 知识地图

| 主题 | 已确认结论 | 状态 |
| --- | --- | --- |
| Tile static capacity | 编译期决定资源盒子、布局合法性和指令特化 | 已讲透基础 |
| Valid region | destination 通常定义语义域；是连续前缀，不是任意 mask；区外默认 unspecified | 已讲透基础 |
| TileType/location | Vec、Mat、Left、Right、Acc 等位置参与指令重载与校验 | 已讲基础 |
| Layout | BLayout 描述外层次序，SLayout/SFractalSize 描述盒化基块；逻辑 shape 相同不等于物理布局相同 | 已讲基础 |
| GlobalTensor descriptor | pointer 不拥有 GM；shape 定坐标域，element-stride 定物理距离，layout 定 2D↔5D 解释/特化 | 已讲透 ND/DN 基础 |
| ND/DN address folding | ND 折叠前四维为 row；DN 以 shape3 为 row、其余相关维折叠为 col；最终地址是 5D index 与 stride 的点积 | 已讲透基础 |
| GlobalTensor subview | C++ `TASSIGN` 重绑 pointer；IR `partition_view` 是纯逻辑 descriptor；TPipe `subOffset` 是 ring slot 内字节偏移 | 已建立边界 |
| 对齐 | 未盒化 row-major 行宽通常需要 32 B 对齐；尾块用固定 capacity + dynamic valid 表达 | 已讲基础 |
| TLOAD/TSTORE | 实际传输范围受 valid region 控制；跨布局、dtype、location 受代际约束 | 已讲 3 次 |
| A2/A3 ND/DN DMA | ND/DN 把语义映射为 nBurst/lenBurst/gmGap；shape、valid shape、32 B 对齐与 burst 上限是 backend contract | 已讲基础，gap 整除待验证 |
| TADD | 以 destination valid region 为迭代域；输入兼容性仍是调用方 contract | 已讲基础 |
| Event | 精确表达 producer/consumer pipeline 依赖，不是全局 barrier | 已讲透基础 |
| Pipe mapping | TLOAD→MTE2、TADD→V、Vec TSTORE→MTE3、TMATMUL→M | 已讲基础 |
| Event ID 生命周期 | Record 后必须有 Wait；ID 有限并按 pipe pair 管理 | 已讲基础 |
| TPipe/TPUSH/TPOP | A2/A3 跨 Cube/Vector Tile 传递包含 GM payload ring 与 ready/free credit | 已讲基础 |
| DIR_BOTH ownership | C2V/V2C 必须使用两条不重叠 ring；总 footprint=`2×SlotNum×SlotSize` | 已讲透核心缺陷 |
| Credit batching | 初始 SlotNum 个 credit 免等待；之后按 SyncPeriod 批量 free/wait | 已建立模型，待完整演算 |
| Double buffering | GEMM 的 L1 与 L0A/L0B 均以 ping-pong 运行；既需正向数据依赖，也需反向 slot 归还 | 已讲透一个真实实例 |
| TMATMUL | Left×Right→Acc；A2/A3 half/bf16→fp32、int8→int32；运行时 M/K/N∈[1,4095] | 已讲基础与真实 kernel |
| K-slice accumulation | 首 slice 用 TMATMUL 初始化 Acc，后续 slice 用 TMATMUL_ACC；Acc 跨全部 K-loop 常驻 | 已讲透基础 |
| Cube pipe chain | TLOAD/MTE2 → TMOV/MTE1 → TMATMUL/M → TSTORE/FIX，含反向复用 event 与末尾 drain | 已讲透一个真实实例 |
| TMOV Mat→Left/Right（A2/A3） | 保持有效域逻辑值，完成 L1→L0 role transfer；A 的 NZ→ZZ 改外层 block order，B 的 ZN→ZN 保布局搬运 | 已讲透核心路径 |
| TMOV compact/tail | 默认非 compact 走静态 capacity extent；CompactMode::Normal 把 runtime valid/aligned extent 传给底层 | 已建立边界，待设备量化 |
| Reduce/通信 ISA | 尚未系统覆盖 | 待学习 |

## 六仓版本与覆盖矩阵

| 仓库 | 最近分析 commit | 已覆盖文件/符号 | 覆盖状态 |
| --- | --- | --- | --- |
| pto-isa | [f51c92f](https://github.com/hw-native-sys/pto-isa/commit/f51c92f610827daad0ddfb383072e03d514b4ae9) | pto_tile.hpp、tile_offsets.hpp、TLoad.hpp、TMatmul.hpp、TMATMUL.md、GlobalTensor.md、partition-view.md、add_custom.cpp、gemm_basic_custom.cpp、gemm_basic host/test、Shape/Stride/GlobalTensor、ND/DN folding、TASSIGN、TLOAD、TMOV、TMATMUL/TMATMUL_ACC、TSTORE、Tile/Event、TPUSH/TPOP、TMovToLeft/Right、TExtractToA/B、TileLeft/Right alias、CPU TMOV 与 A2/A3 device ST | ISA 深挖 6 |
| simpler | [a8d7ce1](https://github.com/hw-native-sys/simpler/commit/a8d7ce12c7433442f4930baf9daf6ab4e3b7edb5) | Worker、compute_task_fanin、orchestrator TensorMap stages | 入门 |
| PTOAS | [307d048](https://github.com/hw-native-sys/PTOAS/commit/307d0484a9e7d5e36f01b253d2bebe4d2f45fe81) | TLoadOp::verify、PTOTLoadToTLOAD、dynamic subview tests | 初步 |
| pypto | [7102058](https://github.com/hw-native-sys/pypto/commit/71020585278b68f56c72c40d5570f07dbb20bc8b) | create_l1/gather_row、Tensor→Tile、gm_pipe_layout、ComputeGMPipeWorkspaceElements、PrepareGMSlotBufferLayout | 跨仓深挖 1 |
| pypto-lib | [5b8d1e9](https://github.com/hw-native-sys/pypto-lib/commit/5b8d1e9846ff7401f0f8525bc5a5b67c8191c13e) | build_swa_metadata、decode_sparse_attn_csa、golden | 深挖 1 |
| pypto-serving | [272b874](https://github.com/hw-native-sys/pypto-serving/commit/272b87492695f78d44c2e8cfe808f372706de594) | cache metadata、prepared inputs、_run_l3 | 初步 |

## 已确认接口与不变量

- Tile capacity shape 是静态资源/type contract；valid region 是本次运行的真实语义域。
- Valid region 是左上角连续前缀；区外元素除非指令明示，否则不得假设为零或保持不变。
- Tile location 与 layout 都参与指令合法性；shape/dtype 相同不足以证明两个 Tile 可互换。
- GlobalTensor 不拥有 GM；其有效期不得超过底层 allocation，且 stride 单位固定为元素。
- 五维坐标到 GM 的元素 offset 是 `Σ(ik×stridek)`；layout 负责从 Tile `(row,col)` 恢复五维坐标，不能替代 stride。
- ND 的 `validRows=shape0×shape1×shape2×shape3`、`validCols=shape4`；DN 的 `validRows=shape3`、`validCols=shape0×shape1×shape2×shape4`。
- A2/A3 plain ND/DN `TLOAD` 额外受 32 B、`nBurst<4096` 与 backend gap 表达约束；CPU-SIM 数学可执行不等于设备路径合法。
- `TASSIGN(GlobalTensor)`、IR `partition_view` 与 TPipe `subOffset` 分别属于 pointer rebase、逻辑子视图与 transport slot 字节偏移，单位和 owner 不得混用。
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
- `TMATMUL` 只消费已经位于 L0A/L0B、role/layout 合法的 Left/Right Tile；普通 Mat Tile 必须先经 `TMOV` 转换。
- K 分块 GEMM 的首 slice 必须初始化 Acc，后续 slice 必须显式累加；fp32 Acc 在全部 K-loop 和 FIX store 完成前保持所有权。
- L1/L0 ping-pong 既需要 MTE2→MTE1→M 的正向可见性，也需要 M→MTE1→MTE2 的反向 slot 复用依赖。
- `gemm_basic` 的 B 逻辑 shape 是 `[K,N]`，物理 ABI 是 contiguous `[N,K]` DN；host transpose-copy 与 kernel layout 必须一致。
- 不同 memory space 可以使用相同数值 offset；`TASSIGN(..., 0x0)` 只在各自 L1/L0A/L0B/L0C 地址域内解释。
- A2/A3 `TMOV(Mat→Left/Right)` 要求 source/destination 静态 shape 相同；它改变 memory location，并按 operand role 选择物理 block routing，不可按同 shape memcpy 删除。
- A2/A3 half 的 Left/Right fractal 基块为 16×16；`Transpose` 分支表示 inner-fractal orientation mismatch，不等同于数学矩阵转置。
- 默认非 compact Left/Right 路径按静态 capacity 组织 MTE1；dynamic valid 不自动缩短搬运，且 destination valid 不得超过 source 已初始化区域。
- CPU-SIM 的逐逻辑元素 TMOV 与 A2/A3 设备 TileLeft alias 不同，只能证明数学语义，不能证明 MTE1 block order。

## 待验证推断

- A2/A3 plain ND/DN 中 `gmGap` 通过字节数右移 5 位得到；如果 gap 不是 32 B 整数倍，是否由调用前的公共 verifier 拒绝，当前尚未找到闭合证据。
- `pto.partition_view` 的 offsets/sizes 如何在 PTOAS/PyPTO lowering 中具体变成 GlobalTensor pointer/shape/stride，尚未跨仓追踪。
- `TPOP_IMPL(Pipe&, GlobalData&)` 在 A2/A3 `DIR_BOTH` 的 Cube 分支未显式应用 `cons.entryOffset`，是否构成 V2C 直接视图地址错误，尚缺直接调用与设备测试。
- `SyncPeriod` 的批量 credit 可减少同步消息，但对具体吞吐/等待时间的影响需要 device trace 和对照实验。
- TPUSH 文档的 A2/A3 local FIFO 表述可能混淆“GM transport backing”与“consumer-local destination”。
- `TMATMUL` 文档与 CPU simulator 声明/检查静态 shape equality，但当前 A2/A3 wrapper 可见检查只覆盖 dtype/location；下层 verifier 是否统一拒绝 mismatch 尚未确认。
- `gemm_basic` 理论 GM 请求量约 39 MiB、算术强度约 79 FLOP/byte，但 L2 reuse 与 MTE/MMAD overlap 未经本课程设备 trace 验证。
- `gemm_basic` 每个输出 core 的 32 个 K-slice 静态 TMOV payload 为 1.5 MiB、24 core 合计 36 MiB；是否成为 wall-time 瓶颈尚缺 MTE1/M stall device trace。
- dynamic tail 下 compact 对齐后的真实 transfer extent、发射开销以及 invalid capacity 的硬件读取范围仍需 isolated device test。

## 尚未解释的知识债

- NZ/5HD 的 C0 盒化五维地址映射及其与二维 Tile 的关系。
- GlobalTensor 动态 shape/stride 的乘法溢出、partition OOB 与统一 verifier contract。
- `partition_view` → lowering → GlobalTensor/TLOAD 的跨仓证据链。
- TMOV、TRESHAPE、transpose 与 ND/NZ/ZN 布局转换的完整合法矩阵。
- A2/A3 `TMATMUL` 的 `m==1→16` 特例如何约束 capacity、valid region 与最终 store。
- `gemm_basic` 在真实设备上的 MTE2/MTE1/M/FIX overlap、L2 reuse 和 buffer 容量余量。
- 非整除 M/N/K tail 下 TMATMUL valid region、padding 与 output ownership。
- reduce 指令的 valid region、精度提升和跨行/列语义。
- 默认 SlotNum=4 下 credit batching、ring wrap-around、析构 drain 的完整状态机。
- A2/A3 GM ring 与 A5 consumer SRAM 的 TPUSH/TPOP 差异。
- Tensor Graph → Tile Graph → Block Graph → Execution Graph 的 pass 顺序。
- pl.spmd 到 task payload、resource shape 与物理 core 的映射。
- PTOAS bytecode/device binary、版本 ABI 与跨仓 CI。
- A2/A3 与 A5 的同步、DMA、layout 和数值差异。

## 下一批候选主线

1. 深挖非整除 GEMM tail：`TileLeftCompact/TileRightCompact`、valid/C0 对齐、padding ownership 与额外 MTE1 流量。
2. 追踪 `partition_view` 经 PyPTO/PTOAS lowering 成为 GlobalTensor/TLOAD 的跨仓地址 contract。
3. 结合最新 TPipe pending-credit 修复，补齐默认深度下 batching、wrap-around、析构 drain 与连续 dispatch 状态机。
4. 进入 reduce 指令的 valid region、累加精度和跨行/列语义。
5. 对照 A5 的 ND→NZ/ZN 与 A2/A3 Mat→Left/Right，补全 TMOV 合法矩阵与代际漂移。
