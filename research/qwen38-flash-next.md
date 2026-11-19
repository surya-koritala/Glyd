# Qwen3.8-Flash-Next in the Glyd engine (RTX 4080 SUPER 16 GB, 60 GiB RAM, Ryzen 9 7950X3D)

Tags: **[file]** read from the checkpoint or the modeling code at the pins below, **[calc]** computed from [file] numbers, **[assume]** to be replaced by a measurement.

## 0. Pins

| What | Pin |
|---|---|
| Qwen/Qwen3.8-Flash-Next | revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, 131 safetensors shards, 336 GiB, BF16; `Qwen4ExpForConditionalGeneration`, `model_type qwen4_exp`, text `qwen4_exp_text` [file] |
| Tensor inventory | all 131 shard headers read on the PC (`st_headers.py`): 1,658 tensors, 179,999,981,459 parameters [file] |
| Modeling code | transformers 5.19.0, `models/qwen4_exp/{modular,modeling,configuration}_qwen4_exp.py` (the PC's venv and the Mac's alike). **It drops `mtp.*`** (`_keys_to_ignore_on_load_unexpected`) [file] |
| Strata (the target) | measured on the same PC: Q2_0 161-165 tok/s decode with MTP, 96-100 without, 32K prompt 9.6 s, 32K decode 129.5, GSM8K 148/150; IQ3_XXS 125-140 with MTP, GSM8K 147/150, MMLU-Pro 126/150 |

Model card [file]: "125B with 6B activated, plus 51B n-gram embedding and 4B MTP"; "MTP: 1 layer, trained with multi-steps"; QSA budget "512 blocks or 2048 tokens"; context 262,144 native; thinking mode by default (`<think>\n...</think>\n\n`).

## 1. Architecture

Text model: hidden 2560, 48 layers, vocab 248,320, untied embedding and LM head, RMSNorm eps 1e-6 [file]. Every RMSNorm of the model is zero-centred, `x * rsqrt(mean(x^2) + eps) * (1 + w)` in fp32 then rounded to bf16, except the GDN's gated norm (plain `w`) [file].

**Residual streams (Gated Residual, `hc_count 4`, `hc_lowrank 320`).** The embedding is repeated into 4 streams, `x [T, 4, 2560]`. Each layer has two `GatedResidual` blocks (attn, mlp), each [file]:
- `xn = hc_norm(x)`: RMSNorm per stream (group 2560) with a weight of 10240;
- `w = sigmoid(up(silu(down(xn) / 4)))`, down [320, 10240], up [10240, 320];
- block input `mixed = mean_s(w_s * xn_s)` [T, 2560];
- `inj = 2 sigmoid(inject(xn) / 4)`, inject [4, 10240];
- after the block: `x_s = x_s + out * inj_s` (the un-normed streams plus the output, a weight per stream; no stream-to-stream mixing, unlike GLM's mHC).
The final `hyper_connection_mixer` is the same without `inject`: its `mixed` is the LM head's input. There is **no final norm** tensor; the mixer's `hc_norm` is it.

**Layer pattern**: `12 x (3 GDN + 1 QSA)`, every layer an MoE [file]. PLE at layer index 1 (config `ple_layer_ids [2]`, one-indexed).

**Gated DeltaNet** (36 layers) = Qwen3.5's: in_proj_qkv [10240, 2560] (q 16x128, k 16x128, v 48x128), in_proj_z [6144, 2560], in_proj_b/a [48, 2560], causal conv (kernel 4, SiLU) on qkv, l2-normed q and k, `beta = sigmoid(b)`, `g = -exp(A_log) softplus(a + dt_bias)`, value head h reads key head h/3, state fp32 [48, 128, 128], then `rmsnorm(o) * w * sigmoid(z)` (**`output_gate_type sigmoid`**, Qwen3.5 has silu), out_proj [2560, 6144] [file].

**Qwen Sparse Attention** (12 layers, `layer_types` "full_attention" = `indexed_attention`): Qwen3.5's gated attention: q_proj [12288, 2560] = 24 heads x (256 query, 256 gate) interleaved per head, k/v_proj [512, 2560] (2 KV heads), q_norm/k_norm per head (1+w), RoPE on the first 64 of 256 dims (`rotate_half` halves, theta 1e7; text-only mrope = plain RoPE), softmax scale 1/16, output times `sigmoid(gate)`, o_proj [2560, 6144] [file]. Plus the **indexer** [file]:
- `index_qk_proj` [640, 2560] -> q 4 x 128 (q_layernorm, RoPE on 64 of 128) and one raw key of 128 per token, cached raw;
- blocks of 4 consecutive tokens from 0: a block key = `k_layernorm(mean of the 4 raw keys in fp32, rounded)` with RoPE at the block's first position;
- query at p scores its `P = (p+1) // 4` complete blocks: `sum_h relu(q_h . k_b) / sqrt(128)`; if `P > 512` the 512 best blocks, else all; plus the `(p+1) % 4` tail tokens. Attention runs over those tokens only.
- So a context of at most 2,051 tokens attends to everything (dense attention); beyond that, at most 2,048 + 3 tokens a query.

**MoE** (48 layers + MTP): 512 experts, top-10, softmax in fp32 over bf16 logits, top-k renormalised, then rounded to bf16 [file]. Expert = SwiGLU, `gate_up_proj [512, 1280, 2560]` (gate rows then up rows) and `down_proj [512, 2560, 640]`, stored as **3D tensors per layer** (not per expert). Shared expert (intermediate 640) times `sigmoid(shared_expert_gate . x)` [1, 2560], added [file].

**PLE / n-gram embedding** (layer 1 only) [file]:
- 16 hash heads (8 bigram + 8 trigram), each with its own prime vocabulary (20,000,003 .. 20,000,171), rows of 160 bf16; the 16 rows concatenated are 2560.
- Table: 320,001,446 rows padded to 320,001,536, stored as 128 shards `ngram_embedding.shard_K.weight [2,500,012, 160]`.
- Hash of position i: `id_h = ((t_i m0) xor (t_ctx1 m1) [xor (t_ctx2 m2)]) mod V_h + offset_h`, the multipliers and offsets in the checkpoint (`layer_multipliers` [23703573157769, 20109073645365, 8052911324071]); `ctx1 = t_{i-1}` (eos 248044 before the start), `ctx2 = eos` if `t_{i-1}` is eos, else `t_{i-2}` (the `_shift_right_ignore_eos` rule, simplified) [calc].
- Then: `key = norm_key(key_proj(e))` [10240], `value = value_proj(e)` [2560], `q = norm_query(x)` (the streams), `gate_s = <key_s, q_s> / sqrt(2560)`, `gate = sign(gate) sqrt(max(|gate|, 1e-6))`, `gv_s = sigmoid(gate_s) value`, `out = gv + silu(dwconv(norm_conv(gv)))`: a depthwise conv of 10240 channels, kernel 4, **dilation 3** (state of 9 positions).
- `x += out` before the layer's attn hyper-connection.

**MTP** (1 layer, not in transformers): `mtp.pre_fc_norm_embedding [2560]`, `mtp.pre_fc_norm_hidden [10240]` (a norm of the **4 streams**), `mtp.fc_embedding [2560, 2560]`, `mtp.fc_hidden [2560, 2560]`, one QSA layer with its two GatedResiduals, indexer and MoE (512 experts), and `mtp.hyper_connection_mixer`; no dedicated embedding, no final norm [file]. The way the 10240 normed stream values meet `fc_hidden [2560, 2560]` is not in any code we have [assume: fc_hidden on each stream, plus fc_embedding of the normed embedding, giving the layer's 4 input streams; checked by acceptance].

## 2. Tensor inventory (all 1,658 tensors, the text model's)

| Group | Tensors | Shape (BF16 unless noted) | Params |
|---|---|---|---|
| embed_tokens, lm_head | 2 | [248320, 2560] | 1.272 B |
| hyper-connections (96 in layers + final mixer) | 4 x 96 + 3 | hc_norm [10240], down [320, 10240], up [10240, 320], inject [4, 10240] | 0.641 B |
| GDN (36 layers) | 9 each | in_proj_qkv [10240, 2560], in_proj_z [6144, 2560], in_proj_a/b [48, 2560], conv1d [10240, 1, 4], A_log [48], dt_bias [48], norm [128], out_proj [2560, 6144] | 2.087 B |
| QSA (12 layers) | 9 each | q_proj [12288, 2560], k/v_proj [512, 2560], o_proj [2560, 6144], q/k_norm [256], index_qk_proj [640, 2560], q/k_layernorm [128] | 0.617 B |
| routed experts (48 layers) | 2 each | gate_up_proj [512, 1280, 2560], down_proj [512, 2560, 640] | **120.796 B** |
| shared expert + router (48) | 5 each | gate/up [640, 2560], down [2560, 640], shared_expert_gate [1, 2560], gate [512, 2560] | 0.299 B |
| PLE (layer 1) | 136 | 128 table shards [2500012, 160]; key_proj [10240, 2560], value_proj [2560, 2560], norm_key/query/conv [10240], conv1d [10240, 1, 4]; I64 multipliers [3], offsets [16], vocab sizes [16] | **51.2 B** + 0.033 B |
| MTP | 31 | as section 1; experts [512, 1280, 2560] + [512, 2560, 640] | 2.517 B experts + 0.091 B |
| vision tower (not loaded) | 340 | `model.visual.*` | 0.449 B |

## 3. Memory [calc]

| Part | BF16 | 2 bit | 2.25 bit | 2.5 bit | 3 bit | 4 bit |
|---|---|---|---|---|---|---|
| routed experts (48 layers) | 225.0 GiB | 28.1 GiB | 31.6 | 35.2 | 42.2 | 56.3 |
| MTP experts | 4.69 | 0.59 | 0.66 | 0.73 | 0.88 | 1.17 |
| n-gram table | 95.4 GiB (int8: 47.7) | 11.9 | 13.4 | 14.9 | 17.9 | 23.8 |

Always-on (everything a token reads on the GPU: hyper-connections, GDN, QSA, shared experts, routers, PLE projections, LM head; not the embedding, the table or the routed experts): **4.31 B params** = 2.26 GiB at 4.5 bits (Int4), 2.76 at 5.5 (Int5), 3.26 at 6.5 (Int6), 8.0 BF16. Per layer: GDN 77.3 M, QSA 70.8 M.

Per decoded token: 10 experts x 48 layers x 4.92 M = 2.36 B routed params (580 MB at 2 bits) + the always-on 4.31 B (2.96 GB at Int5). At about 600 GB/s that is ~5 ms of always-on reads a token without MTP; MTP passes share them.

What fits:
- **GPU (16 GB)**: always-on at Int5/Int6 (2.8-3.3 GiB), KV for 32K (12 layers x 2 heads x 256 x 2 x bf16 = 24 KiB a token: 0.75 GiB at 32K; indexer keys 3 KiB a token), GDN states 36 x 3 MiB = 108 MiB, scratch ~1 GiB; **~9-10 GiB for an expert cache** (~30% of the 24,576 experts at 2-2.5 bits).
- **RAM (60 GiB)**: the expert pack at 2-2.5 bits (28-35 GiB) mapped as page cache, the embedding as bf16 on the host (1.2 GiB, rows copied up per token), the rest of the page cache for hot n-gram rows.
- **The n-gram table stays on the NVMe in BF16**, read in place from the checkpoint's own shards (no conversion, no loss): 16 rows of 320 bytes a token (5 KiB), page cache for the hot rows. A 32K prompt reads 524K rows; decode reads 16 a token. Int8 (47.7 GiB) or a smaller table is the fallback if the page faults show in the decode time.

## 4. What the engine has for each part

| Part | Existing code | New |
|---|---|---|
| 4 streams, two blocks a layer, a block trait for the token mixer and for the experts | `model_glm.rs` (`EngineGlm`: `mixer_half`, `mlp_half`, segmented decode, verify/commit, MTP drafting) | `model_qwen4x.rs` in its pattern; the Gated Residual kernels (norm, mix, inject, final mixer) in `kernels/qwen4x.cu` |
| GDN | `qwen35.cu` `conv_silu`, `gdn_recurrent`, `gdn_chunk` (any nk/nv; the 16/48 split is Qwen3.5-27B's) | the gated norm with **sigmoid** |
| QSA attention <= 2,051 tokens | `qwen35.cu` `q35_prep` (q/k norm, 64-dim RoPE, KV write), `attn256`, `attn256_parts/merge` | - |
| QSA indexer | `glm.cu` `glm_idx_score` (sum of relu dot products: weights 1, scale 128^-0.5) and `glm_select` (top-k blocks with the tail, P <= k takes all) | block keys (mean, norm, RoPE at the block start), the indexer's query prep, attention over a token list (GQA 24/2, head 256, gate) |
| MoE | `moe.rs` core: softmax router with bf16 logits and renorm, SwiGLU without clamp, shared expert with sigmoid gate (Qwen3.5/3.6-MoE's) | experts read from the 3D tensors |
| Expert cache, host misses, skip, async copies, graph segments | `offload.rs` `ExpertStore`, `cpu_cb.rs`, `moe_glm_cb.rs`, `glm_cb.cu` | the codebook kernels are fixed at GLM's 4096 x 2048 (`GC_H`, `GC_I`, `GC_ROW`): parametrise to 2560 x 640, softmax routing, the shared gate, a rotation of 2560 |
| MTP drafting | `mtp_glm.rs`, `draft_loop.rs`, `EngineGlm::verify/commit` (KDA rewind) | the Qwen head (section 1), the GDN and PLE conv rewinds |
| PLE | - | host hashing and row gather (mmap of the shards), key/value products, the gate and the dilated conv kernel |
| Serving | `glm_serve.rs` (`glmserve` JSON lines) | `qwenserve` with the same protocol |
| Checks | `kcheck_glm.rs` + `tools/glm_ref.py` | `kcheck_qwen4x.rs` + `tools/qwen4x_ref.py` on transformers' own modules |

## 5. Open points

- MTP wiring (section 1): decide by acceptance on the real model.
- `torch.topk` among equal block scores (relu makes exact zeros) picks an order the engine cannot match; it only matters where fewer than 512 blocks score above zero.
- A rotation of 2560 (not a power of two) for the codebook experts: five Walsh-Hadamard blocks of 512 with random signs is the simple choice.
