# TTS GEN_WAV：按真实帧数构图，不再为 padding 付算力

> 状态：**已实现并定版**
> 涉及：`tools/mtmd/clip.cpp`、`tools/mtmd/models/models.h`、`tools/mtmd/models/qwen3tts-gen.cpp`
> 性质：**纯算力优化**（不改语义、不改输出）

---

## 0. 一句话

GEN_WAV 阶段每批固定算 `wav_tfm_swa = 72` 帧。最后一批往往远不足 72 帧，
但原来照样算满整窗、用 code-0 补齐、再从输出里裁掉尾部。
改成按**真实帧数**构图之后，尾窗只算需要的那些帧。

---

## 1. 原实现的三步浪费

GEN_WAV 是"16 路 RVQ 码 → 原始 PCM"，跑在 flash-attn 布局的自注意力窗口里，窗口宽度
`wav_tfm_swa = 72` 帧（硬编码在 `clip.cpp`，`mtmd-helper-gen.cpp` 有一处注释要求两处保持一致）。

一段语音切成 `n_frames` 帧，产生 `ceil(n_frames / 72)` 个窗口。前 `k-1` 个窗口是满的，
**最后一个窗口通常不满**。

原实现对这个不满的窗口做了三件事：

1. **按 72 帧建图** —— 算力按满窗付
2. **用 code 0 从后方 padding** —— 补齐到 72
3. **生成后再裁掉尾部** —— 把 padding 产生的音频切掉

```
const int n_frames_w = hparams.wav_tfm_swa;        // 72
std::vector<int32_t> codes(n_frames_w * n_codes, 0);  // 全零 padding
...
// drop the tail audio that comes from the code-0 rear padding
if (n_frames < n_frames_w) { out_audio.resize(n_frames * hop); }
```

第 3 步是必要的——不裁就会在语音末尾听到 padding 生成的杂音。但它同时暴露了第 1 步的浪费：
**这段音频从头到尾都不会被使用，却按满窗算了 attention。**

---

## 2. 改法

把真实帧数一路传进图构建：

`clip.cpp` 算帧数：
```cpp
int n_frames = ctx->model.hparams.wav_tfm_swa;
if (params && params->gen_process == CLIP_GEN_PROCESS_GEN_WAV && params->codes != nullptr) {
    const int64_t n_codes = ctx->model.gen_code_head_w->ne[2] + 1;
    n_frames = (int) (params->codes->size() / n_codes);
    n_frames = std::max(1, std::min(n_frames, (int) ctx->model.hparams.wav_tfm_swa));
}
```

`models.h` 加成员：
```cpp
// real frame count of this call (GEN_WAV only): the tail window may be shorter
// than the attention window, so only the frames that are actually needed get computed
int n_frames;
```

`qwen3tts-gen.cpp` 用它建张量，并加断言：
```cpp
GGML_ASSERT(n_frames > 0 && n_frames <= hparams.wav_tfm_swa);
ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_codes);
```

padding 与裁尾两段随之**整段删掉**——不再产生多余音频，自然不需要裁。

---

## 3. 为什么不能无条件压到"真实帧数"

只有**尾窗**可以短。满窗必须保持 72。

原因在 `qwen3tts-gen.cpp` 的 prefix 契约：

```cpp
const int64_t prefix = hparams.wav_tfm_swa - 1;   // 71
```

自回归窗口靠前 71 帧做 prefix 条件。如果把满窗也压短，prefix 长度会跟着变，
条件信息就断了——**输出会变，而且不是变得更差那种变，是另一种东西**。

所以 `clip.cpp` 那行 `std::min(n_frames, (int) hparams.wav_tfm_swa)` 不是防御性代码，是语义边界。

---

## 4. 收益取决于尾巴有多长

| 情况 | 尾窗利用率 | 尾窗浪费 |
|---|---|---|
| `n_frames` 是 72 的整数倍 | 100% | 0（本来就没浪费） |
| 尾窗 60 / 72 帧 | 83% | 17% |
| 尾窗 36 / 72 帧 | 50% | 50% |
| 尾窗 8 / 72 帧 | 11% | **89%** |

一段 2 秒左右的语音（短句、单句配音）落在最后一档的概率不低——
总帧数刚过 72，第二个窗口只用到十几帧，却按满窗算。

> 本仓库不附基准数字：收益跟你的句子长度分布强相关，
> 给一个平均值反而误导。要测就用 `llama-tts` 打的 `vocoder X.XXs` 字段，
> 对比改动前后同一批文本。

---

## 5. 顺带修掉一个隐性依赖

原实现里 `out_audio` 的长度由 padding 决定，裁尾那段的 `hop` 是
`out_audio.size() / n_frames_w`。帧数一变，这个除法的前提就变了。

新实现不产生 padding 音频，所以 `out_audio` 从一开始就是对的长度，
`hop` 不再需要从"含 padding 的长度"反推。**少了一处隐式耦合**，
这也是为什么这个改动能安全地只动三个文件。
