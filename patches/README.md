# patch 说明

```
cpu-afterburner.patch     20 files changed, 1428 insertions(+), 194 deletions(-)   112 KB
```

> 2026-11-18 起，**源码已随仓库分发**（`llama.cpp/`，3,453 文件）。这份 patch 现在是
> 两件事：改动记录（review 用）+ 换上游版本时的重放工具。想确认那棵树没被私自改过，
> 跑 `scripts/verify-tree.sh`，它会要求「与基线不同的文件恰好是这 20 个」。
> 下面"怎么自己验基线"那节仍然有用——它是那个脚本的原理，也是换新上游版本时的做法。

## 基线

| | |
|---|---|
| 上游 commit | `e351231c4f4cdd89c88e696c46d0eb718c9e0ab5` |
| 日期 | 2026-09-25（官方 master 上真实存在） |
| patch 相对谁生成 | 本地快照 `c649f60`（flatten 后的工作树） |

`c649f60` 是把 fork 从独立仓库 flatten 进上层项目时产生的提交，**只存在于我们本地的
fork 历史里**（`D:\llama.cpp-fork-history.git`），官方仓库没有。我们的快照就是 `e351231`
的工作树：3639 个同路径文件**逐字节相同**，只少 3 个与源码无关的文件
（`build-xcframework.sh`、一个 Xcode plist、一个 bench log）。所以 patch 可以直接打在
`e351231` 上，不需要先复刻 flatten。实测过：

```bash
git --work-tree=/tmp/base checkout e351231 -- $(patch 里的 20 个路径)
cd /tmp/base && git init -q . && git apply --check /path/to/cpu-afterburner.patch   # 通过
```

> ⚠️ **这里早先写的是 `5096f2a`，那是错的。** `5096f2a` 是我们**自建**的快照 commit
> （`author baseline@local`、无 parent、工作区在 `llama.cpp-master/` 下），官方仓库里
> 根本没有这个对象：`git fetch https://github.com/ggml-org/llama.cpp 5096f2a` 会报
> `not our ref`。拿它去 `git checkout` 必然失败。
>
> 这个错误在 setup 脚本里躺了很久没被发现，因为当初的"验证编译"用的是 `git archive`
> 导出的树，**从没跑过脚本本身**。现在的基线是拿 patch 里 20 个文件的前像 blob 在官方
> 历史里反查出来的：只有一个 commit（`e351231`，2026-09-25）能让这 20 个文件同时对上。

**别打在别的 commit 上**。上游 `ggml/src/ggml-cpu/` 每周都在重构，打不上时 `git apply`
报的位置会跟你说"某个不相干的函数对不上"，很容易误判成 patch 坏了。

## 怎么自己验基线（别看日期）

判据不是"哪个 commit 离得近"，而是 patch 自己写下的 `index <前像>..<后像>` 行 ——
前像就是这份 patch 认为你手上应有的文件内容。一个 commit 的这 20 个 blob 全对上，
它就是基线；差一天也不叫"接近"。

```bash
PIN=e351231c4f4cdd89c88e696c46d0eb718c9e0ab5
awk '/^diff --git/{f=substr($3,3)} /^index /{h=$2; i=index(h,".."); print f, substr(h,1,i-1)}' \
    patches/cpu-afterburner.patch |
while read p h; do
    [ "$(git rev-parse "$PIN:$p" | cut -c1-7)" = "$h" ] || { echo "对不上: $p"; exit 1; }
done && echo "20/20 前像对上，这就是基线"
```

> ⚠️ 切 hash 必须用 `index(h,"..")`，**不能写 `split($2,a,"..")`**：awk 里 `".."` 是
> 正则"任意两个字符"，会把 hash 切成碎片，结果 0/20 全不匹配，让你误判基线错了。

**为什么不能拿"日期最接近"的 commit 顶替**：实测取 `23f7aab`（2026-10-16，比基线晚
21 天），20 个文件里已有 **17 个**被上游改过。但 `git apply` 只报 **2 个**文件失败
（`include/llama.h`、`src/llama-context.cpp`，都落在 KV cache 的 `type_k`/`type_v` 那段，
也正是本 patch 改动最密集的地方）；其余 15 个是**带 offset 静默通过**的。

这是最坏的失败形状：看着"只差一个 hunk，手改一下就行"，真实含义是"你在把 9 月的改动
往 10 月的代码上贴，其中 15 个文件没有任何人验证过"。手补那个 hunk，等于自己写一段
零测试覆盖的移植代码。

## 20 个文件，按主题分组

### A. MUL_MAT_ID 负载切分（decode +5.7% / prefill +18.3%）

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cpu/repack.cpp` | `+129` 抢块调度：`MMID_DYNAMIC` 编译期开关、块网格对齐 `NB_COLS=8`、前缀和、跨专家领活 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | generic / repack 两条路径的分派计数器（`g_opprof_mmid_*`） |

spec：`specs/mul-mat-id-load-balance.md`

> **MoE 模型才会走到这里**。dense 模型的 `MUL_MAT_ID` 调用数是 0，不会有任何变化。

### B. per-op profiler

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` | `+60` `GGML_OPPROF=1` 打表 / `=2` 单图逐节点 trace，含每线程耗时与启动偏斜 |

spec：`specs/op-profiler.md`

零开销（实测 19.29 vs 19.26 tok/s）。移植自 smolmaxxing，唯一平台改动是
`clock_gettime` → `ggml_time_us()`（MSVC 没有前者）。

### C. 自适应 flash attention（Spec 01）

| 文件 | 改动 |
|---|---|
| `include/llama.h` | `fa_kv_threshold` / `fa_max_tokens` 参数 |
| `common/common.h`、`common/arg.cpp`、`common/common.cpp` | `-fakv` / `--fa-kv-threshold` 命令行 |
| `src/llama-context.cpp`、`src/llama-cparams.h`、`src/llama-graph.cpp`、`src/llama-model.cpp` | 每个 decode step 按 KV 长度选 FA / legacy |

spec：`specs/fa-auto-adaptive.md`

一句话：CPU 上 FA 的退化速度比 legacy 快得多，所以 `n_kv <= 阈值` 用 FA，超了切 legacy。
运行时开关：`-fakv N`（0 = 一直 FA）。

### D. GQA 循环重排（Spec 02）

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cpu/ops.cpp` | f16 attention kernel 里交换循环顺序：每个 K/V 行只读一次，再分给该 kv 组的所有 query head |

spec：`specs/fa-gqa-loop-reorder.md`

本机实测 KV 流量倍数：Qwen3-0.6B（GQA ratio 2）≈ 2.2，Qwen3-4B / glm-edge-4b（ratio 4）≈ 3.4。
重排后倍数降到 1。只对单行 decode 生效（`neq1 == 1 && neq3 == 1`），其余回落原 kernel。
运行时开关：`LLAMA_FA_GQA=0`。

### E. 量化 V 的 legacy attention（Spec 03）

| 文件 | 改动 |
|---|---|
| `ggml/include/ggml.h`、`ggml/src/ggml.c` | 新算子 `GGML_OP_MUL_MAT_TQ`（**追加在枚举尾部**，插中间会平移所有值） |
| `ggml/src/ggml-cpu/ops.h`、`ops.cpp` | kernel：decode 走 partials、prefill 走直接写 |
| `ggml/src/ggml-cpu/common.h` | `GGML_MUL_MAT_TQ_MAX_N 16`：两种模式的分界，wsize planner 与 kernel 共用同一个宏 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | wsize 规划 |

spec：`specs/legacy-quant-v-op.md`

### F. TTS 图窗口优化（GEN_WAV）

| 文件 | 改动 |
|---|---|
| `tools/mtmd/clip.cpp` | 图按**真实帧数**构建；尾部窗口不再付整个 attention 窗口的代价，也不需要 code-0 padding 和后续裁剪 |
| `tools/mtmd/models/models.h` | `clip_graph_qwen3tts_gen` 增加 `n_frames` 成员 |
| `tools/mtmd/models/qwen3tts-gen.cpp` | 用真实 `n_frames` 建张量，加 `GGML_ASSERT(n_frames > 0 && n_frames <= wav_tfm_swa)` |

spec：`specs/tts-gen-wav-window.md`

这是纯算力优化：短尾窗口原来要算满 `wav_tfm_swa` 帧，现在只算需要的。

### G. TTS 控制行与线程透传

| 文件 | 改动 |
|---|---|
| `tools/tts/tts.cpp` | `+215` `@spk=<path>` 切换 speaker 参考、`@seed=<n>` 重播种 sampler 与音频码 RNG；sampler 生命周期归本处管理；`n_threads` 透传给 mtmd |

spec：`specs/tts-resident-loop-and-control.md`

`@seed` 的用途是**让失败可复现**：固定 seed 会永远以同样方式失败，换 seed 重试才有意义。
`n_threads` 透传是性能修复——不透传的话 TTS 的 mtmd 侧拿不到你指定的线程数。

### H. 模型转换脚本

| 文件 | 改动 |
|---|---|
| `conversion/base.py` | `+5` 补一个 deepseek-coder-1.3b-base 的 tokenizer 哈希（与 6.7b 同构但 vocab 32000，哈希不同，上游列表没登记） |

## 活跃开关一览

| 开关 | 类型 | 作用 |
|---|---|---|
| `MMID_DYNAMIC` | **编译期**（`repack.cpp` 顶部） | 0 = stock 静态切分，1 = 抢块（默认 1） |
| `MMID_DYNAMIC_CHUNKS` | 编译期 | 每线程切几块（默认 4；实测 2/4/8 在噪声内，收益来自跨专家领活而非切得细） |
| `LLAMA_FA_GQA=0` | 运行期 | 关掉 Spec 02 的 GQA 重排 |
| `-fakv N` | 运行期 | Spec 01 的 FA / legacy 分界（0 = 一直 FA） |
| `GGML_OPPROF=1` / `=2` | 运行期 | 打 per-op 表 / 单图逐节点 trace |

⚠️ `MMID_DYNAMIC` **必须是编译期宏**：把同一段 stock 循环原样包进运行期 `if`（else 为空）
就掉 26%（19.26 → 14.32 tok/s）。`gemv<>` 是巨型模板，MSVC 一旦见它落在分支内就不再生成
同样的代码。这条踩过，见 `docs/pitfalls.md`。

## 这个 patch 不包含什么

- **`specs/quant-dot-efficiency.md` 描述的东西没有实现**。那份文档自述"草案，待审阅，
  未写任何生产代码"，只做了测量。放在 `specs/` 里是为了保留"量化点积是每元素指令成本
  受限"这个结论的来历，不代表仓库里有对应代码。
- **不含权重，不含题池，不含构建产物**。
- **不含上游代码**。这个仓库只有 patch + 脚本，源码由 `scripts/setup.sh` 重建。
