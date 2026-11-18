# TTS 常驻进程、控制行与线程透传

> 状态：**已实现并定版**
> 涉及：`tools/tts/tts.cpp`（+215 / −35）
> 目标：把 TTS 从"一次一个进程、每次重付加载开销"改成能长期驻留的服务

---

## 0. 一句话

四件事，都指向同一个目的——**让 llama-tts 能当服务用，而不是当命令行工具用**：
常驻循环、`@spk=`/`@seed=` 控制行、vocoder 线程数透传、pipeline 实例复用。

---

## 1. 常驻循环（`LLAMA_TTS_LOOP=1`）

原版：给一个 `-p "text"`，跑完退出。每次调用都要重付：

- 模型加载（backbone + mmproj 两套权重）
- 首次图构建 / warmup

**这就是那个"每调用 ~3 s 的税"**。它不是推理时间，是启动成本，但调用方感受不到区别——
对一条要实时播出来的语音来说，"慢 3 秒"和"推理慢 3 秒"是一回事。

打开后：从 stdin 逐行读 prompt，**在同一个进程里连续生成**，每行输出一条 WAV。

### 1.1 一个必须避开的死锁

```cpp
// Interactive request/response protocol: read one line, generate it,
// print `[i] wrote <path>`, then read the next line. Draining stdin to
// EOF up front would deadlock a caller that keeps the pipe open
```

不能"先把 stdin 读干净再开始生成"。常驻服务会一直持有管道写端，
`getline` 到 EOF 永远等不到，整个进程就卡死。必须**逐行读、立即回**。

### 1.2 每轮迭代必须重置状态

```cpp
llama_memory_clear(llama_get_memory(lctx), true);
common_sampler_reset(smpl);
```

backbone 的 KV cache 和 sampler 都带着上一轮的状态。不清会在第 2 条起听到上一句的残留影响。

### 1.3 计时字段

每轮会打：

```
[i] timings: prompt 0.12s + generation 0.86s + vocoder 0.31s = total 1.29s (wall 1.31s)
[i]          audio = 3.20s (audio time = 2.48x process time)
```

`wall` 和 `total` 的差就是框架开销。这个字段是给"到底哪一段该优化"用的——
prompt / generation / vocoder 三段里，vocoder 往往是意外的大头。

---

## 2. `@spk=` / `@seed=` 控制行

控制行在 loop mode 下生效，不进推理。

### 2.1 `@spk=<path>`

```
@spk=/path/to/reference.wav     ← 之后所有 prompt 用这个音色/情感参考
@spk=                           ← 清空（回到无参考）
```

用途是**在同一个常驻进程里切换情感参考**，不用重启、不用重载模型。
helper 每次迭代都会重编码参考，所以切换的成本不比一条普通 prompt 高。

### 2.2 `@seed=<n>`

```
@seed=12345
```

重新播种 backbone sampler 和音频码 RNG。

存在的理由很具体：**固定 seed 会让失败以同样的方式永远失败**。
一个落进退化轨迹的 prompt（几秒静音然后撞上 token 上限），换个 seed 才是真正独立的采样轨迹；
在同一个 seed 上重试多少次都是同一条路。

⚠️ 实现上有个所有权问题：`@seed=` 创建的 sampler 归本循环释放，
初始那个归 `llama_init` 管。混了就是 double free 或者泄漏，所以有 `smpl_owned` 在管。

---

## 3. vocoder 线程数透传（性能修复）

```cpp
// code2wav (the vocoder) runs inside mmproj and defaults to 4 threads only.
// -t N must reach it, otherwise the vocoder stays single-handedly slow no
// matter how many threads the backbone gets.
mtmd_params.n_threads = params.cpuparams.n_threads;
```

**这条是最容易被忽略的性能 bug**：你给 `-t 18`，backbone 用了 18 线程，
但 code2wav 拿的还是它自己的默认值（4）。于是总时长里那一段永远不随线程数下降，
看起来像"加线程没效果"。

改动只有一行，但它解释了"为什么我加到 18 线程还是慢"这个现象的相当一部分。

---

## 4. pipeline 实例复用

```cpp
// one pipeline instance for the whole session: anything cached on the instance
// (specials lookup, code2wav graph) is kept alive across iterations
mtmd_helper::gen_audio gen(lctx, mctx.get());
```

`gen` 从"每次调用构造一次"提到循环外。这样 specials 查找、code2wav 图这些
缓存在实例上的东西跨迭代存活。

配合 loop mode，这才形成完整的"常驻"：**进程常驻 + 实例常驻 + 图常驻**。

---

## 5. 输出文件

单条模式仍写 `params.out_file`。多条/循环模式拆成 `stem_0.wav`、`stem_1.wav`……
否则每一轮都会覆盖同一个文件，而且从外部看像是"只生成了最后一条"。

---

## 6. 复现

```bash
# 单条（原行为不变）
./build/bin/llama-tts -m backbone.gguf --mmproj mmproj.gguf -p "text" -o out.wav -t 18

# 常驻循环
LLAMA_TTS_LOOP=1 ./build/bin/llama-tts -m backbone.gguf --mmproj mmproj.gguf \
    -o out.wav -t 18 <<'EOF'
@spk=/path/ref_a.wav
第一条
第二条
@spk=/path/ref_b.wav
@seed=777
第三条
EOF
```

对照 `[i] timings:` 里的 `wall` 与 `total`，以及 audio/process 的倍数，
就能看出启动成本是否真的被摊掉了。

> 注：本仓库没有附带基准数据。上面这些是机制与设计目的，
> 具体快多少取决于你的调用频率——一次性的命令行调用得不到任何收益。
