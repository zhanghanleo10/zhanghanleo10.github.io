---
layout: post
title: "PTO 全栈课程 10：TPipe Credit 账本——Ring Wrap-around、析构 Drain 与连续 Dispatch"
description: "沿 A2/A3 TPipe 的真实修复追踪 TPUSH/TPOP：为什么 8-slot FIFO 的 40 次传输只应 drain 2 个 pending credits，以及错误账本如何跨 dispatch 造成 scheduler stall。"
date: 2026-08-19 09:15:00 +0800
category: "PTO · 全栈精讲"
series: "PTO 全栈课程"
tags: [PTO, PTO-ISA, TPipe, TPUSH, TPOP, RingBuffer, A2A3]
reading_time: "约 25 分钟"
mermaid: true
math: true
---

> 本章基于 [`pto-isa@60081f36`](https://github.com/hw-native-sys/pto-isa/commit/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9)，重点分析已合入的修复 [`0a15e15c`](https://github.com/hw-native-sys/pto-isa/commit/0a15e15c627deef86b4281fb08e84d39dad793de)。代码事实来自当前 `main`；设备故障与历史验证来自 [Issue #248](https://github.com/hw-native-sys/pto-isa/issues/248)、[PR #240](https://github.com/hw-native-sys/pto-isa/pull/240) 和合入 commit 的记录。本次没有自行运行 A2/A3 真机实验。

## 本篇在 PTO 课程路线中的位置

第 04 章建立了 TPipe 的空间模型：A2/A3 用 GM ring 搬运 Cube/Vector 间 payload，`TPUSH/TPOP` 以 ready/free 信号交接所有权；`DIR_BOTH` 还必须为 C2V/V2C 放置两段互不重叠的 ring。

当时尚未闭合的是时间模型：为什么 free-space 不逐 slot 同步？ring wrap-around 时生产者何时等待？C++ `TPipe` 对象析构后，为何 hardware flag 仍会污染下一次 dispatch？

今天只讲一个知识点：**A2/A3 TPipe 的批量 credit 账本及其 drain contract**。这也是一次真实 scheduler stall 的根因，而不是抽象的并发练习。

## 前置知识

- 一个 GM FIFO entry 的物理地址是 `base + (tileIndex % SlotNum) × SlotSize`。
- `TPUSH` 写 payload 后发 data-ready；`TPOP` 等待 ready 后读取 payload。
- payload slot 从 producer 开始写到 consumer 完成读取期间必须独占。
- C++ 对象的生命周期不等于 hardware flag 生命周期：同一 `FlagID` 会被后续 kernel dispatch 复用。

## 今日核心维护问题

1. `SlotNum`、`SyncPeriod`、producer/consumer `tileIndex` 如何共同保证 wrap-around 不覆盖在途 payload？
2. 析构函数为何必须精确消费 pending free credits，既不能少，也不能多？
3. 当前修复证明了什么，仍依赖哪些未编码的前置条件？

## PTO 全栈中的位置

~~~mermaid
flowchart LR
    C["Cube producer<br/>TMATMUL → AccTile"]
    P["TPUSH<br/>wait free → GM store → ready"]
    R["GM Ring<br/>slot = index % SlotNum"]
    Q["TPOP<br/>wait ready → GM load → batched free"]
    V["Vector consumer<br/>TADD → TSTORE"]
    D["~TPipe<br/>drain pending free credits"]
    F["Hardware flag state<br/>survives C++ object/dispatch"]
    C --> P --> R --> Q --> V
    Q --> F
    F --> P
    P --> D
    D --> F
~~~

本章只涉及 `hw-native-sys/pto-isa`。PyPTO 提供了暴露问题的 Qwen/mixed Cube+Vector 工作负载，但 credit 算法、修复和设备回归都位于 pto-isa；为了凑齐跨仓而加入 PyPTO 反而会稀释本章边界。

## 精确语义：两条同步通道，不是一条

当前 [`TPipe`](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/include/pto/npu/a2a3/TPush.hpp#L24-L87) 定义：

~~~cpp
SyncPeriod = SlotNum <= 2 ? SlotNum : SlotNum / 2;

shouldWaitFree(i):
    i >= SlotNum && i % SyncPeriod == 0

shouldNotifyFree(i):
    (i + 1) % SyncPeriod == 0
~~~

这里必须区分：

- **data-ready**：每个 `TPUSH` 都 `record`，每个 `TPOP` 都 `wait`，保护“本 entry 已写完”；
- **free credit**：consumer 每处理 `SyncPeriod` 个 entries 才通知一次，producer 在启动窗口填满后每隔 `SyncPeriod` 次 push 等待一次，保护“一批 slots 已可复用”。

一个 free credit 不是“一个空 slot”，而是 **`SyncPeriod` 个按序释放 slot 的批量凭证**。这种 batching 减少 cross-core sync 次数，但把正确性变成严格的算术账本。

## 真实函数链：从 Matmul 到 slot 归还

回归 kernel [`runTPushPopMatmulAddNoSplit`](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/tests/npu/a2a3/src/st/testcase/tpushpop_cv_nosplit/tpushpop_cv_nosplit_kernel.cpp#L42-L211) 的实际路径是：

~~~text
Cube:
  TLOAD A/B → TMOV Left/Right → TMATMUL Acc
  → TPUSH(mPipe, accTile)

Vector AIV0:
  TPOP(mPipe, vecFifoTile)
  → TLOAD bias → TADD → TSTORE output
~~~

[`TPUSH_IMPL`](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/include/pto/npu/a2a3/TPush.hpp#L517-L556) 的顺序是：

1. 若 `shouldWaitFree(prod.tileIndex)`，先 `prod.allocate()` 消费 free credit；
2. 写入 `(prod.tileIndex % SlotNum)` 对应 GM entry；
3. `prod.tileIndex++`；
4. 发 data-ready。

[`TPOP_IMPL`](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/include/pto/npu/a2a3/TPop.hpp#L20-L60) 则是：

1. 每个 entry 都等待 data-ready；
2. 从 `(cons.tileIndex % SlotNum)` 读取；
3. 若 `shouldNotifyFree(cons.tileIndex)`，发一个 batched free credit；
4. `cons.tileIndex++`。

TileData 路径的 free 已包含在 `TPOP` 中；文档中的 [`TFREE(Pipe&)`](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/docs/isa/TFREE.md) 在 A2/A3 是 API 对称性的 no-op，不能把代码里紧随 `TPOP` 的 `TFREE` 再记成一次通知。

## 对象、Ring 与 Credit 生命周期

以 `SlotNum=8` 为例：

- push 0–7 使用启动窗口，不等待 free；
- 第 8 次 push 将重新写 slot 0，必须先等待 consumer 已释放第一批 entries；
- consumer 在 pop index 3、7、11……后分别发出 credit；
- producer 在 push index 8、12、16……前分别消费 credit；
- kernel 尾部 consumer 可能已经发出、但 producer 因不再 push 而没有消费的 credit，必须由析构 drain。

~~~mermaid
sequenceDiagram
    participant C as Cube / Producer
    participant G as GM ring (8 slots)
    participant V as Vector / Consumer
    participant F as free-credit flag
    C->>G: push 0..7，无 free wait
    G->>V: pop 0..3
    V->>F: notify credit #1 (4 slots)
    C->>F: before push 8, consume #1
    C->>G: push 8 → wrap to slot 0
    G->>V: pop 4..7
    V->>F: notify credit #2
    C->>F: before push 12, consume #2
    Note over C,V: 同样规律持续到 tile 39
    V->>F: 一共产生 10 credits
    C->>F: steady-state 共消费 8
    C->>F: ~TPipe 精确 drain 剩余 2
~~~

payload 的所有权在 ready→pop 完成间属于该 entry；free credit 的生命周期可以滞后多个 entries。最后一个 payload 已消费，并不等于 flag 已回到零余额。

## 40 次传输的完整演算

回归配置是：

~~~cpp
TPipe<0, DIR_C2V, 4096, 8, 8, true>
TOTAL_M=640, CASE_TILE_M=16, K=32, N=64
~~~

每个 AccTile 为 `16×64 fp32 = 4096 B`；8-slot GM ring 占 `8×4096 = 32 KiB`。输出沿 M 方向分成 `640/16=40` 个 entries，ring 完整 wrap 五轮。

因为 `SlotNum=8`，所以 `SyncPeriod=4`：

| 账目 | index / 公式 | 数量 |
| --- | --- | ---: |
| consumer 已通知 credit | pop 3,7,…,39，即 `floor(40/4)` | 10 |
| producer steady wait | push 8,12,…,36 | 8 |
| 析构前 pending | `10-8` | 2 |
| 旧实现固定 drain | `SyncPeriod` | 4 |

旧实现前两次 `allocate()` 能消费真实 pending credit；第三次开始等待一个永远不会到达的通知，core 永久阻塞。Issue #248 记录的外显结果是 `507018` 与 `S1:running-stalled`，并且卡住的是 mixed Cube+Vector kernel。

反方向同样危险：若析构少 drain，一个旧 credit 会留在 hardware flag 中。下一 dispatch 的 producer 到达首次 wrap 时可能消费这个 stale credit，在**本次** consumer 尚未释放 slot 前提前覆盖 payload。于是“多 drain”表现为 deadlock，“少 drain”可能表现为无异常的数据污染。

## 修复：按真实账目 drain，而不是按深度猜

当前 [`countPendingFreeCredits(tileCount)`](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/include/pto/npu/a2a3/TPush.hpp#L69-L87) 计算：

~~~text
notified = floor(tileCount / SyncPeriod)
waited   = shouldWaitFree 在 [0, tileCount-1] 中为 true 的次数
pending  = max(notified - waited, 0)
~~~

`SlotNum==1` 单独处理：第 0 次 push 使用初始空位，之后每次都等待，所以 `waited=max(tileCount-1,0)`。其他深度从第一个不小于 `SlotNum` 的 `SyncPeriod` 对齐点开始计数。

[`~TPipe`](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/include/pto/npu/a2a3/TPush.hpp#L503-L512) 只执行 `pending` 次 `prod.allocate()`。这条公式的本质是 conservation law：

~~~text
consumer emitted = producer consumed during work + destructor consumed at drain
~~~

析构后的目标不是“ring 里没有数据”——payload 早已消费——而是同一 FlagID 的 credit 余额为零。

## 为什么采用批量 credit，而不是更简单的方案

### 方案 A：每 pop 一个 slot 就通知

令 `SyncPeriod=1`，账本最直观，也缩短单 slot 可复用的等待粒度；但 40 entries 会产生 40 次 free signal/40 次对应 wait，而当前配置只产生 10 对。对 cross-core FFTS sync 而言，这增加控制面开销。实际收益必须由设备 trace 证明，不能只按调用数推导周期。

### 方案 B：每 dispatch 更换 FlagID 或强制 reset

它可以隔离 stale state，但 FlagID 是有限硬件资源，且 reset 本身需要明确的跨 core 完成协议；若前一 dispatch 仍在途，reset 反而可能抹掉合法通知。它没有消除所有权问题，只把问题转移到 FlagID allocator。

### 方案 C：保留 batching，显式维护 credit counter

在 GM/共享设备状态保存计数更容易调试，却引入额外访存、原子/一致性与地址生命周期。当前 flag + 静态规则是更小的机制；前提是公式与实际 `shouldWait/shouldNotify` 永远同源，并有边界测试防漂移。

因此当前设计合理的理由不是“批量一定更快”，而是它以更少同步表达了顺序 FIFO 的容量释放；接受的代价是 drain 算术和生命周期更难维护。

## 性能、并行与硬件约束

- **静态传输量（推导，不是真机测量）**：每 dispatch 的 ring payload store+load 至少为 `40×4096×2 = 320 KiB`；80 次为 25 MiB。credit 修复不改变 payload 流量。
- **同步量**：每 entry 仍有一对 ready record/wait；free path 每 4 entries 一对 notify/consume，其中 2 个 consume 延迟到析构。
- **并行度**：批量 credit 允许 producer 在 8-entry 启动窗口内推进，但最迟在 wrap 前阻断，安全优先于占满 ring 后继续投机。
- **确定性**：这是控制流正确性而非数值稳定性。错误可以表现为稳定 hang，也可能因 dispatch 时序变成概率性覆盖。
- **图模式（推断）**：GM ring 地址和 FlagID 静态有利于复用；但 hardware flag 跨 replay 的零余额仍是必要前置条件。当前测试验证连续 dispatch，不等于已经证明 ACL Graph replay contract。

## 测试证据与未覆盖风险

合入 commit 增加的 A2/A3 回归位于 [host test](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/tests/npu/a2a3/src/st/testcase/tpushpop_cv_nosplit/main.cpp#L39-L140) 与 [kernel test](https://github.com/hw-native-sys/pto-isa/blob/60081f369280edf3f5eb2cdd5b06cb769ab2d8c9/tests/npu/a2a3/src/st/testcase/tpushpop_cv_nosplit/tpushpop_cv_nosplit_kernel.cpp#L42-L244)：

- compile-time `static_assert(countPendingFreeCredits(40)==2)`；
- 真实 shape `640×32 @ 32×64`，40 次 `TPUSH/TPOP`；
- 同一 stream 连续 dispatch 80 次，复用 FIFO 与 flag，专门暴露 stale-credit 累积；
- golden 是 NumPy fp32 matmul + bias。

**合入记录中的验证事实**：定向 CCE kernel target 编译通过；`SlotNum=1..64, tileCount=0..1024` 的账本与逐步状态机穷举一致；golden 生成通过。记录同时明确说明，本地完整 host executable 因环境中的 non-PIC `libgtest.a` 链接失败，未进入运行，因此不能把这部分写成已执行通过。

**历史实验材料**：PR #240 报告同一类 Qwen3 mixed A2/A3 上板单次及连续三次通过。该 PR 是定位与意图证据；最终事实仍以当前 `main` 的 `0a15e15c` post-image 为准。

仍缺的最小 CI guard：

1. 参数边界矩阵：`SlotNum={1,2,3,4,8}`，`tileCount={0,N-1,N,N+1,kP-1,kP,kP+1}`；
2. 两个 TPipe 对象在同一 kernel 内顺序复用相同 FlagID；
3. C2V、V2C、DIR_BOTH 分别做连续 dispatch；
4. producer/consumer 数量不等或 early-exit 时，应有显式取消/超时协议，而不是进入析构盲等；
5. 在 graph replay 下重复同一 case，并检查 payload 与完成性。

第 4 点是当前最重要的 contract 缺口：`countPendingFreeCredits(prod.tileIndex)` 用 producer 数量推导 consumer 已通知数量，隐含“所有 produced entries 都已被 consumer pop/free”。正常结构化 kernel 满足它；异常退出不满足。当前析构是正常完成路径的 drain，不是通用 cancellation。

## 跨版本与维护风险

修复历史本身暴露了流程债：正确逻辑曾从 GitCode 同步到 GitHub，后来逻辑上更早的变更以更晚 GitHub commit 落地，把析构覆盖回固定 drain。此次 `0a15e15c` 必须在新的同步 cut point 后重新表达修复。

因此修改这一区域时要检查：

- `shouldWaitFree`、`shouldNotifyFree` 与 `countPendingFreeCredits` 是否同一规则源；
- TPush/TPop 文档是否仍与 A2/A3 TileData/GlobalData overload 一致；
- FlagID 分配、DIR_BOTH 的独立 flag/ring 与析构 drain；
- CPU-SIM 只验证数学/布局，不能替代 FFTS 真机同步；
- GitCode↔GitHub 同步是否保留目标文件 post-image，而不是只依赖提交先后；
- pypto 的 pto-isa pin 是否已经包含该修复及真机门禁。

## 本篇结论、知识债与理解检查

核心结论：TPipe ring 的容量安全不由 `tileIndex % SlotNum` 单独保证，而由“逐 entry ready + 批量 free credit + 精确尾部 drain”共同保证。析构必须把 hardware credit 恢复到零余额；它不是可有可无的清理代码，而是下一 dispatch 的正确性前置条件。

本章补齐了默认/真实深度下的 batching、wrap-around、连续 dispatch 与析构 drain。仍欠缺：异常取消协议、V2C/DIR_BOTH 的同等回归、A5 与 A2/A3 的 credit 差异、graph replay 及设备 trace 下 batching 收益。

理解检查：

1. 为什么 `SlotNum=8` 时 producer 在 push 8 前等待，而 consumer 在 pop 3 后就可以发第一个 credit？
2. 为什么析构少消费 credit 不一定 hang，反而可能造成下一 dispatch 的静默覆盖？
3. 若 producer push 40 个 entries、consumer 只 pop 36 个，当前 `countPendingFreeCredits(40)` 为什么不再是安全的取消算法？

下一章：reduce 指令的行/列归约、valid region、累加精度与 padding 读取边界。

## 课程账本增量

- 选定仓库：`pto-isa`。
- 新增主链：`TMATMUL → TPUSH → GM ring wrap → TPOP → TADD/TSTORE → ~TPipe drain → next dispatch`。
- 新增不变量：free credit 表示 `SyncPeriod` 个顺序释放 slots；正常退出时 emitted、steady-consumed 与 drained 必须守恒；hardware flag 生命周期跨越 C++ 对象和 dispatch。
- 最高风险：early-exit/数量不匹配没有 cancellation contract，以及 GitCode/GitHub 同步顺序再次覆盖正确 post-image。

