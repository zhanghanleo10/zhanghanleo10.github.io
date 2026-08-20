---
layout: page
title: "PTO 全栈课程账本"
permalink: /learning/pto-curriculum/
---

# PTO 全栈课程账本

最后更新：2026-08-20。

## 总体路线

1. PTO ISA 基础：Tile 类型系统、GlobalTensor、valid region、layout 与 memory location
2. 数据搬运与计算指令：load/store/move、elementwise、reduce、matmul
3. Event、pipeline、double buffering、TPipe/RingBuffer 与 Tile 生命周期
4. Python DSL 捕获与 Tensor/Tile IR
5. Block/Execution Graph、SPMD 与 task DAG
6. PTOAS verifier、Pass、EmitC/设备产物
7. simpler Host/AICPU/AICore runtime、TensorMap/RingBuffer
8. pypto-lib kernel/模型、Golden、性能与 serving 集成

当前阶段：ISA 基础与上层生成链交替推进。已打通 `PyPTO TileLoadOp → pto.partition_view → GlobalTensor → TLOAD → TMOV → TMATMUL`，闭合 A2/A3 `TPUSH/TPOP` credit 状态机，并进入 Vector Reduce：已确认 `TROWSUM [R,C]→[R,1]`、`TCOLSUM [R,C]→[1,C]` 的 valid-prefix、scratch ownership 与 binary/sequential 累加契约；下一步进入多 Tile Row Softmax 的 partial max/sum 合并。

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
| 通信 ISA | 尚未系统覆盖 | 待学习 |

## 六仓版本与覆盖矩阵

| 仓库 | 最近分析 commit | 已覆盖文件/符号 | 覆盖状态 |
| --- | --- | --- | --- |
| pto-isa | [f71e7dd](https://github.com/hw-native-sys/pto-isa/commit/f71e7dde08e27719e35dd2648bb1b4ec0cdc928e) | pto_tile.hpp、tile_offsets.hpp、TLoad.hpp、TMatmul.hpp、TRowSum.hpp、TRowReduceOps.hpp、TColSum.hpp、TPush.hpp、TPop.hpp、TROWSUM/TCOLSUM/TPUSH/TPOP/TFREE docs、row-softmax tutorial、gemm_basic、trowsum/tcolsum/tmuls_trowsum tests、Shape/Stride/GlobalTensor、TASSIGN、TLOAD、TMOV、TMATMUL/TMATMUL_ACC、TROWSUM、TCOLSUM、TSTORE、Tile/Event、TPUSH/TPOP、binary/sequential reduce、A2/A3/A5 scratch 差异 | ISA 深挖 9 |
| simpler | [a8d7ce1](https://github.com/hw-native-sys/simpler/commit/a8d7ce12c7433442f4930baf9daf6ab4e3b7edb5) | Worker、compute_task_fanin、orchestrator TensorMap stages | 入门 |
| PTOAS | [fe5594a](https://github.com/hw-native-sys/PTOAS/commit/fe5594af84793c48487d4309d8092c3b6b44a0e9) | TLoadOp::verify、PTOCanonicalizeIR、PTOMakeTensorViewToEmitC、PTOPartitionViewToEmitC/static、PTOTLoadToTLOAD、issue157/issue995/DN layout tests | 跨仓深挖 1 |
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

## 尚未解释的知识债

- NZ/5HD 的 C0 盒化五维地址映射及其与二维 Tile 的关系。
- GlobalTensor 动态 shape/stride 的乘法溢出、partition OOB 与统一 verifier contract；下界 clamp 已定位，上界/overflow 未闭合。
- static/dynamic partition lowering 重复实现的等价性与长期漂移保护。
- TMOV、TRESHAPE、transpose 与 ND/NZ/ZN 布局转换的完整合法矩阵。
- A2/A3 `TMATMUL` 的 `m==1→16` 特例如何约束 capacity、valid region、padding 读取与最终 store；缺少 poison-padding 边界测试。
- `gemm_basic` 在真实设备上的 MTE2/MTE1/M/FIX overlap、L2 reuse 和 buffer 容量余量。
- 多 Tile Row Softmax 的 partial max/sum 合并、全局归一化和数值稳定性。
- Reduce 的 accumulator widening、整数 overflow、poison-padding 与 CPU/A2A3/A5 parity contract。
- producer/consumer 数量不匹配或 early-exit 时的 TPipe cancellation、flag 清理与超时协议。
- V2C、DIR_BOTH、同一 kernel 内顺序复用同 FlagID、ACL Graph replay 下的 pending-credit 回归。
- A2/A3 与 A5 在 ready/free credit、local SRAM/GM ring 与析构语义上的逐点差异。
- A2/A3 GM ring 与 A5 consumer SRAM 的 TPUSH/TPOP 差异。
- Tensor Graph → Tile Graph → Block Graph → Execution Graph 的 pass 顺序。
- pl.spmd 到 task payload、resource shape 与物理 core 的映射。
- PTOAS bytecode/device binary、版本 ABI 与跨仓 CI。
- A2/A3 与 A5 的同步、DMA、layout 和数值差异。

## 下一批候选主线

1. 沿 `TROWMAX → TROWEXPAND → TEXP → TROWSUM → TDIV` 进入多 Tile Row Softmax，解释 partial max/sum 的全局合并。
2. 对照 A5 的 ND→NZ/ZN 与 A2/A3 Mat→Left/Right，补全 TMOV 合法矩阵与代际漂移。
3. 为 Reduce 增加 poison-padding、overflow、binary/sequential 与 CPU/A2A3/A5 parity CI contract。
4. 为 TPipe early-exit、V2C/DIR_BOTH 与 graph replay 建立 cross-dispatch CI contract。
5. 为 partition dynamic OOB、signed 64-bit overflow 与 static/dynamic lowering 等价性建立跨仓 CI contract。
6. 用 device trace 量化 reduce tree、credit batching、Compact 与普通 TMOV 的同步/搬运 stall 和端到端收益。
