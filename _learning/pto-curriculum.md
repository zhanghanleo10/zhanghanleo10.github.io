---
layout: page
title: "PTO 全栈课程账本"
permalink: /learning/pto-curriculum/
---

# PTO 全栈课程账本

最后更新：2026-08-30。

## 总体路线

1. PTO ISA 基础：Tile 类型系统、GlobalTensor、valid region、layout 与 memory location
2. 数据搬运与计算指令：load/store/move、elementwise、reduce、matmul
3. Event、pipeline、double buffering、TPipe/RingBuffer 与 Tile 生命周期
4. Python DSL 捕获与 Tensor/Tile IR
5. Block/Execution Graph、SPMD 与 task DAG
6. PTOAS verifier、Pass、EmitC/设备产物
7. simpler Host/AICPU/AICore runtime、TensorMap/RingBuffer
8. pypto-lib kernel/模型、Golden、性能与 serving 集成

当前阶段：ISA 语义与 compiler lowering 交替推进。课程 20 已闭合 `PTODSL reserve_buffer → ReserveBufferOp → Level owner contract → aligned hole-fit → ResolveReservedBuffers → i32 base`，并确认 placement-time `occupied` 与 post-pass semantic no-alias verifier 是两套不同证明；当前 internal-hole early-return 不登记新区间，是在真实内洞可达时可能触发的 allocator 完整性缺口。下一步把 reserve/import 地址接入 `initialize_pipe`、peer binding 与 flag/base ABI。

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
| 2026-08-17 | 非整除 GEMM tail：Compact 搬运与 padding ownership | base Mat → aligned envelope → Compact Left/Right → `mad(m,k,n)` → valid output | [课程 08]({% post_url 2026-08-17-pto-isa-compact-tail-padding-ownership %}) |
| 2026-08-18 | partition_view 到 GlobalTensor/TLOAD | logical offsets/sizes → rank-5 view → `base+Σ(offset×stride)` → TLOAD | [课程 09]({% post_url 2026-08-18-pto-partition-view-globaltensor-tload-lowering %}) |
| 2026-08-19 | TPipe pending-credit 与连续 dispatch | TMATMUL → TPUSH → GM ring wrap → TPOP → batched free → destructor drain | [课程 10]({% post_url 2026-08-19-pto-isa-tpipe-credit-drain-continuous-dispatch %}) |
| 2026-08-20 | Row/Column Reduce 的语义与 scratch | valid `R×C` → `TROWSUM R×1` / `TCOLSUM 1×C` → binary/sequential → semantic store | [课程 11]({% post_url 2026-08-20-pto-isa-trowsum-tcolsum-valid-region-precision %}) |
| 2026-08-21 | 多 Tile Online Softmax | QK Tile → running max/sum → alpha 重标定 → P@V → running O / global sum | [课程 12]({% post_url 2026-08-21-pto-online-softmax-running-max-sum %}) |
| 2026-08-22 | 四阶段 FlashAttention Pipeline | Cube QK/PV ↔ 三条 GM FIFO ↔ Vector P/GU → preload/exp-ring reuse | [课程 13]({% post_url 2026-08-22-pto-flashattention-four-stage-pipeline-preload-exp-ring %}) |
| 2026-08-23 | `PIPE_V` Barrier 删除证明边界 | `InsertSync → generated C++ → gu/softmax matcher → fail-closed audit` | [课程 14]({% post_url 2026-08-23-pto-pipe-v-barrier-removal-proof-boundaries %}) |
| 2026-08-24 | InsertSync 同步对象生命周期 | `BaseMemInfo → RAW/WAR/WAW → SyncOperation → event ID → SyncCodegen` | [课程 15]({% post_url 2026-08-24-ptoas-insertsync-dependency-event-lifecycle %}) |
| 2026-08-25 | PlanMemory 物理复用与 async WAR | `BufferLife → physical addr reuse → cross-root alias → MTE3→V event` | [课程 16]({% post_url 2026-08-25-ptoas-planmemory-physical-reuse-async-war %}) |
| 2026-08-26 | Modern PlanMemory touching 与 A3 target hazard | `writer-defined birth → touching → load/split-TPOP facts → no-reuse gate → InsertSync boundary` | [课程 17]({% post_url 2026-08-26-ptoas-modern-memplan-touching-target-hazard %}) |
| 2026-08-27 | Modern PlanMemory phi family 与 loop back-edge | `valueToRoots closure → cross-branch pairs → loop-carried deny-set → pairwise reuse group → addr` | [课程 18]({% post_url 2026-08-27-ptoas-memplan-phi-family-loop-backedge %}) |
| 2026-08-28 | Modern PlanMemory largest-first 与 reuse cost | `RootInfo size/alignment → address-space order → legal groups/fresh → cost/capacity → packed offsets` | [课程 19]({% post_url 2026-08-28-ptoas-memplan-largest-first-reuse-cost-fragmentation %}) |
| 2026-08-30 | ReserveBuffer aligned hole-fit 与 Level 地址所有权 | `ReserveBufferOp → level validation → occupied merge/first-fit → base → i32 constant` | [课程 20]({% post_url 2026-08-30-ptoas-reserve-buffer-hole-fit-level-contract %}) |

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
| partition_view lowering | PyPTO 以 valid_shape 生成 subview；PTOAS 将 rank N right-align 到 5D，以 signed 64-bit `Σ(offset×element_stride)` rebase pointer，并继承 source stride/layout | 已讲透跨仓基础 |
| 对齐 | 未盒化 row-major 行宽通常需要 32 B 对齐；尾块用固定 capacity + dynamic valid 表达 | 已讲基础 |
| TLOAD/TSTORE | 实际传输范围受 valid region 控制；跨布局、dtype、location 受代际约束 | 已讲 3 次 |
| A2/A3 ND/DN DMA | ND/DN 把语义映射为 nBurst/lenBurst/gmGap；shape、valid shape、32 B 对齐与 burst 上限是 backend contract | 已讲基础，gap 整除待验证 |
| TADD | 以 destination valid region 为迭代域；输入兼容性仍是调用方 contract | 已讲基础 |
| Event | 精确表达 producer/consumer pipeline 依赖，不是全局 barrier | 已讲透基础 |
| Pipe mapping | TLOAD→MTE2、TADD→V、Vec TSTORE→MTE3、TMATMUL→M | 已讲基础 |
| Event ID 生命周期 | Record 后必须有 Wait；ID 有限并按 pipe pair 管理 | 已讲基础 |
| TPipe/TPUSH/TPOP | A2/A3 跨 Cube/Vector Tile 传递包含 GM payload ring 与 ready/free credit | 已讲基础 |
| DIR_BOTH ownership | C2V/V2C 必须使用两条不重叠 ring；总 footprint=`2×SlotNum×SlotSize` | 已讲透核心缺陷 |
| Credit batching | 初始 SlotNum 个 entries 免等待；之后 consumer 每 SyncPeriod 个 pop 发一个 credit，producer 每 SyncPeriod 个 wrap 前消费一个 | 已讲透 depth=8/40 entries |
| TPipe drain | pending=`consumer notified - producer steady waited`；析构精确消费余额，使同一 FlagID 在下一 dispatch 前回到零 credit | 已讲透正常完成路径 |
| Double buffering | GEMM 的 L1 与 L0A/L0B 均以 ping-pong 运行；既需正向数据依赖，也需反向 slot 归还 | 已讲透一个真实实例 |
| TMATMUL | Left×Right→Acc；A2/A3 half/bf16→fp32、int8→int32；运行时 M/K/N∈[1,4095] | 已讲基础与真实 kernel |
| K-slice accumulation | 首 slice 用 TMATMUL 初始化 Acc，后续 slice 用 TMATMUL_ACC；Acc 跨全部 K-loop 常驻 | 已讲透基础 |
| Cube pipe chain | TLOAD/MTE2 → TMOV/MTE1 → TMATMUL/M → TSTORE/FIX，含反向复用 event 与末尾 drain | 已讲透一个真实实例 |
| TMOV Mat→Left/Right（A2/A3） | 保持有效域逻辑值，完成 L1→L0 role transfer；A 的 NZ→ZZ 改外层 block order，B 的 ZN→ZN 保布局搬运 | 已讲透核心路径 |
| TMOV compact/tail | `CompactMode::Normal` 从 valid 推导 fractal/C0 对齐的 MTE1 envelope；不缩小 allocation，也不保证 padding 为零 | 已讲透 A2/A3 核心路径 |
| Tail GEMM ownership | source capacity 覆盖 envelope；TEXTRACT 写 envelope；`mad(m,k,n)` 与 TSTORE 把语义重新收紧到 valid | 已讲透一个真实 int8 case |
| Row/Column Reduce | `TROWSUM` 只定义 `R×1`、`TCOLSUM` 只定义 `1×C`；valid prefix 控制数学域，A2/A3 scratch 与 A5 register path 资源需求不同，binary/sequential 改变依赖深度与浮点顺序 | 已讲透 sum 基础 |
| Online Row Softmax | 每行维护 `global_max/global_sum`；新 Tile 以 `alpha=exp((old_max-new_max)×scale)` 同时重标定旧 denominator 与 running output numerator；P/PV/alpha 必须保持同一 tile identity | 已讲透真实 FA recurrence |
| Four-stage FA pipeline | Cube 内 QK/PV 串行交错、Vector 内 P/GU drain-first；三条 8-slot GM FIFO 传 payload，`EXP_RING==QK_PRELOAD` 保存 sideband alpha，三段计数保证每 Tile 四阶段各一次 | 已讲透 A3 non-causal DSL 主路径 |
| `PIPE_V` barrier 证明 | PTOAS 依据 alias/dependency 插入同步；pto-isa 再按生成文本删减。当前 `gu` matcher 不强制存在 MTE2→V wait，softmax parser 失败会被当作无依赖 | 已讲透 patch 边界，硬件时序待 trace |
| BaseMemInfo/alias | root、scope、候选地址、字节包络与 unknown-range 共同描述可能访问范围；已知本地物理区间可跨 SSA root 比较 | 已讲透 InsertSync 基础 |
| RAW/WAR/WAW 生成 | `now.use×front.def`、`now.def×front.use`、`now.def×front.def`；同 pipe 变 barrier，跨 pipe 变 set/wait | 已讲透线性与控制流基础 |
| Event ID 分配 | 每个有向 pipe pair 独立维护 8-ID 生命周期池；slot-keyed event 逐 lane 绑定，分配失败降级原位置 `PIPE_ALL` | 已讲透 compiler contract |
| PlanMemory physical reuse | SSA `BufferLife` 只决定规划复用资格；异步 reader 的物理 ownership 必须由 materialized address 上的跨 root WAR event 延长到设备完成 | 已讲透 legacy level2 主路径 |
| Modern PlanMemory touching gate | `freeIndex==allocIndex` 仅通过 strict-lifetime gate；A3 的 load-derived + split-TPOP + same-writer 组合由 target gate 禁止共址，semantic gate 继续约束通用 inplace | 已讲透 hard-gate 边界，target 设备证据待补 |
| Modern PlanMemory phi/loop gate | 同一 `scf.if` result 位的对向、本地 root 可越过静态 lifetime overlap；select/view/result 必须保留 root-set closure；任一 loop-carried root 取消互斥豁免 | 已讲透 branch/loop 证明边界 |
| Modern PlanMemory placement | `slotBytes=alignUp(rawBytes, alignment)`，group footprint 取 member 最大值；非 Cube space 才按 totalBytes 降序，全部 legal group 与 fresh 以 fits/cost/projectedBytes/stable order 比较，容量压力只可推翻性能偏好 | 已讲透 greedy placement 主路径，权重待真机标定 |
| ReserveBuffer placement | `size` 是 byte count、result 是 `i32 base`；Level 1/2 由 PlanMemory 做 aligned first-fit，Level 3 由作者显式给 base；resolve 后 marker 退化为常量 | 已讲透地址生命周期；internal-hole 记账待回归 |
| 通信 ISA | 尚未系统覆盖 | 待学习 |

## 六仓版本与覆盖矩阵

| 仓库 | 最近分析 commit | 已覆盖文件/符号 | 覆盖状态 |
| --- | --- | --- | --- |
| pto-isa | [3186c38](https://github.com/hw-native-sys/pto-isa/commit/3186c381bd49e1164092e67ff1b3564302754e76) | 既有 Tile/DMA/GEMM/TPipe/Reduce/Online Softmax；新增 `compile.sh`、`patch_vec_barriers.py`、barrier pattern reference、`run.py` case1..case8 | ISA 深挖 12 |
| simpler | [a8d7ce1](https://github.com/hw-native-sys/simpler/commit/a8d7ce12c7433442f4930baf9daf6ab4e3b7edb5) | Worker、compute_task_fanin、orchestrator TensorMap stages | 入门 |
| PTOAS | [cc519bc](https://github.com/hw-native-sys/PTOAS/commit/cc519bc92db73b2a2cdfd7c409fe7dfdf72d85e4) | 既有 InsertSync、legacy/modern reuse、hard gates、phi/loop 与 cost placement；新增 `ReserveBufferOp`、PTODSL `reserve_buffer`、`planReserveBufferBase`、Level rules、`PTOResolveReservedBuffersPass`、`verifySemanticNoAliasRanges` 及 reserve level/resolve lit | 跨仓深挖 8 |
| pypto | [ba15fd6](https://github.com/hw-native-sys/pypto/commit/ba15fd66f929de7c03d04f4a4cae7f5751d56bc2) | create_l1/gather_row、Tensor→Tile、gm_pipe_layout、TileLoadOp::DeduceTileLoadType、MakeTileLoadCodegenPTO、EmitPartitionViewPTO、FlattenTileNDTo2D、dynamic shape tests | 跨仓深挖 2 |
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
- PlanMemory 的 SSA last use 只释放地址规划资格，不证明异步 pipe 已完成；重叠字节的旧 readers 完成前，新 writer 不得获得 physical ownership。
- level1/level2 的跨 root reuse contract 要求 PlanMemory 先物化地址，再由已启用的 InsertSync 或等价方案覆盖物理 RAW/WAR/WAW；PlanMemory 单独成功不是完整正确性证明。
- 已知本地物理范围以同 address space 的半开区间比较：端点相接不 alias，不同 SSA root 仍可 alias。
- modern PlanMemory 的 `freeIndex == allocIndex` 是 touching，只免除 strict lifetime overlap；是否共址还必须通过 target-specific 与 semantic no-alias hard gates。
- A3 target gate 只拒绝 `loadDerivedRoot + split-TPOP-derived operand + same DPS writer index` 的组合；它不是“所有 TPOP 永不复用”的全局规则。
- reuse cost 只在 hard gate 通过后选择 group/fresh address；容量压力可改变性能偏好，但不能放松正确性 gate。
- allocator 负责 op 内 storage alias 合法性；InsertSync 负责可由同步边表达的跨 op physical ownership handoff，二者不可互相替代。
- branch-exclusive 豁免只覆盖同一 `scf.if` result 位的对向、本地 root；同一分支 root、外层捕获 root、不同 if root 不因“属于控制流”自动互斥。
- `valueToRoots` 必须对 select、view、region yield/result 做集合闭包；merged alias 的每个可能 root 都必须参与 lifetime、semantic conflict 与 group pairwise gate。
- 任一 root 进入 `scf.for/scf.while` 的 loop-carried closure 后，单次迭代的 branch exclusivity 不足以证明跨 back-edge 不共存，必须取消 phi 豁免并把 lifetime 扩展到整个 loop。
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
- Compact tail 必须满足 `capacity ≥ aligned envelope ≥ semantic valid`；MTE1 可多搬对齐 padding，但不得据此缩小 L0 allocation。
- `PadValue::Null` 不承诺 invalid envelope 为零；GEMM 的数学边界必须由 `mad(m,k,n)` 与有效区 TSTORE 重新收紧。
- Compact 只改变 L1→L0 MTE1 extent；若 GM→L1 仍加载完整 base matrix，不能由此推出 HBM/MTE2 或端到端收益。
- `KAligned` 属于 caller-owned Tile 状态；进入读取该字段的 float Compact/TMATMUL 路径前必须有确定初始化。
- PyPTO `tile.load` 的 `shapes` 决定 Tile capacity，`valid_shape` 同时决定 partition sizes 与 destination Tile valid；两端必须共享同一语义有效域。
- `partition_view` 不分配、不复制、不拥有内存；其生命周期受 source Tensor allocation 支配。
- partition offsets 与 strides 都以元素计；PTOAS 通过 signed 64-bit `Σ(offset×stride)` rebase pointer，result shape 取 partition sizes，stride/layout 继承 source。
- 低 rank TensorView/PartitionTensorView right-align 到 rank 5 只是 descriptor 规范化；左补 shape=1、offset=0、size=1，不构成 transpose 或数据搬运。
- final `TLOAD(dst, src)` 信任已经物化的 GlobalTensor descriptor，不重新计算 bounds/stride；动态 OOB、整数宽度和 layout 合法性必须在更早阶段守住。
- A2/A3 TPipe 的 data-ready 是逐 entry 的 record/wait；free-space 才按 `SyncPeriod=(SlotNum<=2?SlotNum:SlotNum/2)` 批量通知与消费，两条同步通道不能合并记账。
- free credit 代表一批按序释放的 `SyncPeriod` 个 slots；producer 只有在 `tileIndex>=SlotNum` 且位于周期边界时才能消费，保证首次 wrap 前不覆盖在途 payload。
- 正常完成时必须满足 `notifiedFree = steadyWaited + destructorDrained`；析构后同一 FlagID 的 credit 余额必须为零，否则多 drain 会死锁、少 drain 会污染下一 dispatch。
- `countPendingFreeCredits(prod.tileIndex)` 隐含所有 produced entries 已被 consumer pop/free；它是正常退出 drain，不是 producer/consumer 数量不匹配时的 cancellation protocol。
- `TROWSUM` 只读取 source 的 `validRow×validCol` 前缀并只定义目标第 0 列的 `validRow` 个值；目标 capacity 的其他元素 unspecified。
- `TCOLSUM` 只读取 source valid prefix 并只定义目标第 0 行的 `validCol` 个值；binary 与 sequential 路径数学等价但依赖链、scratch traffic 和浮点累加顺序不同。
- Reduce scratch 在指令完成前由该调用独占且不得与活跃 src/dst 别名；A2/A3 `TROWSUM` 使用 tmp，A5 当前仅为公共 ABI 保留同一 operand。
- 完整输出 allocation 的初始化属于调用方或测试 harness 责任；已合入 `TMULS→TROWSUM` 测试修复通过 memset 非语义区，而不是改变 reduce 的数学定义。
- sum 的 accumulator dtype 与顺序属于数值 contract；当前设备主路径通常同 dtype，CPU widening 能力不能自动外推到 A2/A3/A5。
- 多 Tile softmax 必须维护每行 running max 与 running denominator；各 Tile 的局部 softmax 不能直接拼接。
- `alpha=exp((old_max-new_max)×scale)` 必须同时重标定旧 global sum 和旧 running output numerator；两侧使用不同 alpha 会静默破坏 attention 结果。
- `m2_global_max/l2_global_sum` 跨 S1 Tile 持有，`l1_exp_max_ififo[slot]` 活到匹配 GU 消费，`runningOTile` 活到最终除 global sum 并 store；这些 buffer 不得提前复用。
- P、PV 与 alpha 必须共享同一 `tile_id`/ring-slot identity；TPipe ready/free 只解决可见性与回收，不替代 sideband identity。
- 当前 FA 主循环以 `S1/Tile_S1` 取 Tile 数；单指令支持 valid tail 不等于整个 kernel 支持非整除 S1。
- 四阶段 FA 只有两个可并行资源域：Cube 上 QK/PV 同域串行，Vector 上 P/GU 同域串行；跨域由 QK C2V、P V2C、PV C2V 三条 FIFO 解耦。
- `EXP_RING==QK_PRELOAD` 是当前调度的 hard invariant；steady state 必须先 `GU(t)` 消费 `alpha[t%ring]`，再由 `P(t+preload)` 复用同一槽。
- 对 N 个 KV Tile，prologue 产生 preload 个 QK/P，steady 处理 `N-preload` 个旧 GU/新 P 对，epilogue drain 最后 preload 个 PV/GU；每个 Tile 四阶段恰好一次。
- 默认 `S1_TILE=256` 时 QK/P/PV 三类 8-slot GM ring 每 block 合计 2 MiB；这属于 transport backing，不能与 192 KiB Vector UB 预算混算。
- PTOAS `InsertSyncAnalysis` 以 root buffer/view alias 和地址区间识别 RAW/WAR/WAW；同 pipe 依赖生成 barrier，跨 pipe 依赖生成 set/wait，控制流与零次循环必须保守处理。
- `PIPE_V` 删除只能在依赖证明成功后进行；解析未知、调用签名变化或变量身份无法恢复时必须 fail-closed 保留 barrier。
- 当前 `gu` matcher 实际允许 `TROWEXPANDMUL → barrier → TADD` 中不存在任何 `wait_flag(PIPE_MTE2, PIPE_V)`；这比文档声称的目标 pattern 更宽。
- 当前 softmax matcher 只识别 `vN` Tile 名并假设第一个 Tile operand 是 destination；解析失败返回空访问集，进而可能把真实依赖误判为无依赖。
- 最终输出数值测试与端到端 latency 不能证明删 barrier 后不存在低概率调度 race，也不能把多项调优的总收益归因给同步删除。
- `BaseMemInfo` 是 compiler 侧访问证据，不拥有运行时内存；依赖判断必须同时考虑 address space、root/view、候选地址区间、字节包络与 unknown-range。
- `InsertSyncAnalysis` 的 RAW/WAR/WAW 分别来自 `now.use×front.def`、`now.def×front.use`、`now.def×front.def`；同 pipe 生成 barrier，跨 pipe 生成共享 `kSyncIndex` 的 set/wait。
- loop body 可能 zero-trip，body 内 `alreadySync` 不能提升到外层；if/else 只有两条路径都覆盖时才能提升同步事实。
- slot-keyed dynamic event 只排序 `slotSSAExpr % slotCount` 对应的 lane，不能作为另一个 region 或整个 pipe pair 的覆盖证据。
- 普通 event ID 池按有向 `(srcPipe,dstPipe)` 隔离且每池 8 个 ID；event 生命周期冲突无法消解时必须在原程序点降级 `PIPE_ALL`。
- `SyncOperation` 从 analysis 创建，经 motion/remove 更新挂载位置与 tombstone，再由 allocator 填 `eventIds`，最终由 `SyncCodegen` 物化为静态或动态 flag/barrier；pass 结束后分析对象统一释放。
- modern PlanMemory 的 `rawBytes` 先按 address-space alignment 得到 `slotBytes`，multi-buffer 的 `totalBytes=slotBytes×slotCount`；每个 address space 独立受 `MemSpec.capacityBytes` 约束。
- `orderBySize` 只对 VEC/BIAS/SCALING 等非 Cube local space 生效；MAT/LEFT/RIGHT/ACC 保留 `allocIndex/stableOrder`，不能把 largest-first 当作全局排序。
- 当前 placement 不是 literal first-fit：每个 root 遍历全部 legal `ReuseGroup`，并与 fresh group 以 `fits → reuseCost → projectedPackedBytes → stable group order` 比较。
- fresh group 的基础 cost 为 1；capacity pressure 可在 fresh slack 小于 `max(totalBytes, alignmentBytes)` 时强制选择已 fit 的 legal reuse，但不能绕过 lifetime/phi、target 或 semantic gate。
- `ReuseGroup.sizeBytes` 取 member `totalBytes` 最大值；顺序 pack 的同一 space 通常不会产生任意中间 hole，主要浪费来自 slot rounding、group max slack 与早期 group 扩张造成的 tail growth。
- group 构建完成后才 materialize offsets；任何 end 超过 capacity 都 fail closed，不回滚、不放松 correctness gate。真正合并 occupied interval 并做 aligned first-fit 的是后续 `reserve_buffer(auto=true)`。
- `ReserveBufferOp` 以 byte count 请求 VEC/MAT local range，结果是 `i32 base`；它没有 Tile shape/dtype/layout，也没有运行时 allocator 生命周期。
- Level 1/2 的 reserve 地址 owner 是 PlanMemory，只允许 `auto=true` 且输入不得预填 base；Level 3 的 owner 是作者，只允许 `auto=false` 与显式 base。
- auto placement 必须用 `alignUp(size, alignment)` 形成半开占用区间；每个成功 placement 都必须把新区间提交给 allocator 账本，不能只返回地址。
- `verifySemanticNoAliasRanges` 只检查 op 声明的 Tile/view no-alias operand pairs，不枚举所有 ReserveBuffer/Tile ranges，不能替代全局布局 verifier。

## 待验证推断

- A2/A3 plain ND/DN 中 `gmGap` 通过字节数右移 5 位得到；如果 gap 不是 32 B 整数倍，是否由调用前的公共 verifier 拒绝，当前尚未找到闭合证据。
- PyPTO 对动态负 offset 使用 `max(offset,0)`；它避免负 pointer 但会静默改变请求语义，是否需要 strict-error 模式尚未确定。
- 动态 `offset+size<=source extent` 的统一失败路径尚未闭合；当前证据只覆盖类型推导、下界 clamp 与后端地址生成。
- PTOAS 以 signed 64-bit 构造 partition linear offset，issue157 防止 unsigned/32-bit 回归，但乘加溢出没有显式 guard。
- PyPTO 固定 PTOAS v0.57，而本章 PTOAS 源码基线是主干 fe5594a；两者的 dialect/type/template ABI 是否完全一致必须由 pin 组合 CI 证明。
- `TPOP_IMPL(Pipe&, GlobalData&)` 在 A2/A3 `DIR_BOTH` 的 Cube 分支未显式应用 `cons.entryOffset`，是否构成 V2C 直接视图地址错误，尚缺直接调用与设备测试。
- `SyncPeriod` 的批量 credit 可减少同步消息，但对具体吞吐/等待时间的影响需要 device trace 和对照实验。
- TPUSH 文档的 A2/A3 local FIFO 表述可能混淆“GM transport backing”与“consumer-local destination”。
- `TMATMUL` 文档与 CPU simulator 声明/检查静态 shape equality，但当前 A2/A3 wrapper 可见检查只覆盖 dtype/location；下层 verifier 是否统一拒绝 mismatch 尚未确认。
- `gemm_basic` 理论 GM 请求量约 39 MiB、算术强度约 79 FLOP/byte，但 L2 reuse 与 MTE/MMAD overlap 未经本课程设备 trace 验证。
- `gemm_basic` 每个输出 core 的 32 个 K-slice 静态 TMOV payload 为 1.5 MiB、24 core 合计 36 MiB；是否成为 wall-time 瓶颈尚缺 MTE1/M stall device trace。
- Compact 的 aligned request envelope 已由源码闭合，但真实 MTE1 transaction、周期、stall 与端到端收益仍需 isolated device trace。
- `Tile::isKAligned_` 当前未见显式默认初始化，而 A2/A3 float Compact/TMATMUL 会读取它；这是否已在所有生成路径被 caller 初始化尚未复现确认。
- CPU `TROWSUM` 允许 `half/bfloat16→float`，而当前设备文档/检查通常要求同 dtype；这是有意 parity 还是 simulator 超集尚未确认。
- A2/A3/A5 对同一浮点输入的 tree/sequential/repeat 分组误差 envelope 与 bitwise 稳定性尚未量化。
- `TCOLSUM` binary 相比 sequential 的真实性能 crossover 取决于 validRow、validCol、tmp traffic 与 barrier，缺少 device trace。
- `pto_macro_fa_softmax` 公共形参把会被更新的 `new_global_max` 标为 `__in__`；该方向标注目前是否参与 IR alias/dependency 推导尚未确认。
- fp32 `p/global_sum` 与 cast 后 fp16 P→PV 路径的端到端误差 envelope，以及不同 Tile_S1/preload 对误差的影响尚未量化。
- TFA 数据生成器注释了 S1=8192 case，而 C++ 仍声明对应测试；默认 CI 是否提供其他 golden 来源尚未闭合。
- `QK_PRELOAD=3/4 × S1_TILE=256/512 × KV_SPLIT` 的组合正确性、FIFO stall 与 exp-ring wrap 尚缺系统矩阵；现有 case1..case8 不能替代故障注入。
- 文档把 `gu` barrier 删除与 MTE2→V wait 绑定，但当前 matcher 不要求 wait；无 wait 的 pattern 在真实设备上是否安全需要指令级依赖审计与设备 trace。
- PTOAS emitter、变量命名或 wrapper 签名变化时，当前 `vN`/destination-first 文本解析 contract 是否仍成立，尚无版本化 golden 证明。
- PR #136 的性能改善同时包含 pipeline 参数、共享缓冲与 barrier patch；各项收益尚无隔离实验，不能据总 latency 推断 barrier 删除收益。
- 开放 PR #948 提议补强 local address provenance、overflow 与 inexact subview envelope；截至 2026-08-26 仍未作为当前 `e19aff7d` 基线的既有保证。
- modern memplan 设计文档计划了 `plan_memory_five_gates_lifetime_touching.pto` 与 `plan_memory_five_gates_target_hazard.pto`，但当前主干树没有这两个 direct lit；target gate 的 split/alias-view 正负例与 A3 真机 race 仍待补。
- A3 target hazard 的软件判定已闭合，但公开材料没有给出队列、端口或 writeback 级微架构时序；具体硬件成因只能标记为推断。
- `syncFinder` 从 may-zero loop 传播而 `alreadySync` 不传播的状态机仍缺 must/may 形式化证明；当前主要依赖回归测试守边界。
- 8-ID event pool 在深层 loop/branch、多 region multi-buffer 压力下的 `widen/reallocate→PIPE_ALL` 触发频率与设备 stall 尚未量化。
- reuse penalty `10/20/6/12/4`、lookahead=1 与 pressure reserve 是候选排序 heuristic，不是周期模型；尚缺 workload sweep 与真机 stall/latency 标定。
- Cube local space 关闭 size-first 的 bank/operand-pattern 收益只有代码和设计动机，没有 L0/L1 bank trace；具体微架构成因仍是推断。
- 当前 bank-risk 只用 whole-root exact co-location 代理，尚未纳入 subview 精确 interval 与 `offset % bankModulo`；fragmentation samples 的文件名也不能证明 modern 的真实 hole 行为。
- 当前 `planReserveBufferBase` 的 tail success 会登记新区间，但 internal-hole early-return 不会；若 `occupied` 真有内洞，连续两个 auto reserve 可由代码推导得到相同 base。当前 root pack 常形成连续前缀，因此该路径是否在主 pipeline 可达仍需 exact lit 证明。
- 现有 reserve direct tests 覆盖 Level contract、单个 base=0 物化与 nested resolve，但未覆盖 internal hole、多 auto reserve、alignment boundary 或 generic post-plan overlap audit。

## 尚未解释的知识债

- NZ/5HD 的 C0 盒化五维地址映射及其与二维 Tile 的关系。
- GlobalTensor 动态 shape/stride 的乘法溢出、partition OOB 与统一 verifier contract；下界 clamp 已定位，上界/overflow 未闭合。
- static/dynamic partition lowering 重复实现的等价性与长期漂移保护。
- TMOV、TRESHAPE、transpose 与 ND/NZ/ZN 布局转换的完整合法矩阵。
- A2/A3 `TMATMUL` 的 `m==1→16` 特例如何约束 capacity、valid region、padding 读取与最终 store；缺少 poison-padding 边界测试。
- `gemm_basic` 在真实设备上的 MTE2/MTE1/M/FIX overlap、L2 reuse 和 buffer 容量余量。
- Online Softmax recurrence 与四阶段正常完成路径已闭合；尚欠 S1 tail、全 mask 行、P fp16/sum fp32 误差上界、exp-ring poison/wrap、stage delay 和 early-exit cancellation 注入。
- Reduce 的 accumulator widening、整数 overflow、poison-padding 与 CPU/A2A3/A5 parity contract。
- producer/consumer 数量不匹配或 early-exit 时的 TPipe cancellation、flag 清理与超时协议。
- V2C、DIR_BOTH、同一 kernel 内顺序复用同 FlagID、ACL Graph replay 下的 pending-credit 回归。
- A2/A3 与 A5 在 ready/free credit、local SRAM/GM ring 与析构语义上的逐点差异。
- A2/A3 GM ring 与 A5 consumer SRAM 的 TPUSH/TPOP 差异。
- Tensor Graph → Tile Graph → Block Graph → Execution Graph 的 pass 顺序。
- pl.spmd 到 task payload、resource shape 与物理 core 的映射。
- PTOAS bytecode/device binary、版本 ABI 与跨仓 CI。
- `patch_vec_barriers.py` 缺少 matcher 级 negative tests：必须覆盖无 wait 的 GU、RAW/WAR/WAW、非 `vN` 变量、换行调用、alias/view 和 parser error；parser 应改为 tri-state 并在 unknown 时保留同步。
- 生成 C++ 需要固定的 barrier-count/topology golden，并以设备 poison/delay 压测验证删减后的低概率 race；当前 `run.py` 只验证终值与总 latency。
- A2/A3 与 A5 的同步、DMA、layout 和数值差异。
- PlanMemory 跨 root physical WAR、modern hard gates、branch/loop alias closure、greedy placement 与 ReserveBuffer Level/resolve 主链已闭合；仍缺 raw/slot/cursor/base checked arithmetic、internal-hole 多 reserve exact golden、Tile/ReserveBuffer 全局 interval verifier、动态 subview/reinterpret provenance、branch×target×semantic 交叉矩阵，以及 legacy/modern 的 `UB peak + event topology + device latency` 三联对照。
- Event ID exhaustion、dynamic lane 异常退出/early-exit、zero-trip 嵌套控制流需要真机 race/stall 与死锁故障注入，FileCheck topology 不能替代。

## 下一批候选主线

1. 主线：研究 `ReserveBuffer/import_reserved_buffer → initialize_pipe → backend helper`：peer name binding、local base、flag/base 分配与多 pipe ABI。
2. 为 Online Softmax/四阶段 pipeline 增加 max 上升/下降、全 mask、non-divisible S1、exp-ring poison/wrap、stage delay 与 CPU/A2A3 parity CI contract。
3. 对照 A5 的 ND→NZ/ZN 与 A2/A3 Mat→Left/Right，补全 TMOV 合法矩阵与代际漂移。
4. 为 TPipe early-exit、V2C/DIR_BOTH 与 graph replay 建立 cross-dispatch CI contract。
5. 为 partition dynamic OOB、signed 64-bit overflow 与 static/dynamic lowering 等价性建立跨仓 CI contract。
6. 用 device trace 量化 QK/PV 与 P/GU overlap、reduce tree、credit batching和 Compact 的同步/搬运 stall。

## 第二次七章知识图谱回顾（课程 08–14）

- **资源盒子到有效语义**：课程 08 用 Compact tail 固化 `capacity ≥ aligned envelope ≥ valid`，明确 padding ownership 与语义边界。
- **上层视图到物理地址**：课程 09 把 `partition_view` 变成 `base + Σ(offset×stride)`，让 Tile valid 与 GlobalTensor descriptor 共用一个有效域。
- **跨核传输到生命周期闭合**：课程 10 证明 ready/free credit、ring wrap 与 destructor drain 是同一 ownership 账本。
- **局部算子到跨 Tile 数学状态**：课程 11–12 从 Reduce 轴语义推进到 Online Softmax 的 running max/sum/output recurrence。
- **数学状态到真实 pipeline**：课程 13 把 QK、P、PV、GU 放进 Cube/Vector 两域与三条 GM FIFO，闭合 preload/steady/epilogue。
- **pipeline 到同步证明**：课程 14 追到 PTOAS InsertSync 与生成后 barrier patch，确认“匹配文本”不能替代“证明依赖不存在”。
- **当前最大缺口**：课程 16–18 已补齐 legacy physical WAR、modern touching/target hard gate 与 branch/loop alias closure；剩余最大缺口转为 fragmentation/capacity 策略、地址 provenance/overflow、branch×target×semantic 组合验证，以及设备级 race/性能归因证据。


