# glyd-engine runs Qwen3.5 natively, then Gemma 4

Private design (glyd-gpu-private). Owner's direction, 2026-10-06: Glyd's own engine replaces vLLM, vLLM is the fallback only; 2026 models only. This is objective 2 of `glyd-priorities`: the engine runs the 2026 models, first Qwen3.5 (and the models built on its architecture), then Gemma 4, then the MoE models.

## Goal

`glyd run` / `glyd serve` hand a Qwen3.5 dense model to `glyd-engine` on a GPU where the engine's table is all green (the speed bar below), and to vLLM everywhere else, as today.

- **Models (2026 only):** Qwen/Qwen3.5-0.8B, -2B, -4B, -9B, -27B, Qwen/Qwen3.6-27B and Qwen/Qwen3.8-27B (the same `qwen3_5` architecture), text only. The MoE models (`qwen3_5_moe`: Qwen3.5-35B-A3B, -122B-A10B, -397B-A17B, Qwen3.6-35B-A3B; `qwen3_5_moe_text`, `Qwen3_5MoeForCausalLM`: Qwen3.8-2.4T-A95B) come after the dense ones, in the MoE phase; both model types are refused until then.
- **Weights:** every matrix the engine holds packed (the products of both layer kinds, the MLP, the embedding, the LM head), bit for bit the checkpoint's. Small tensors (norms, the causal convolution, `A_log`, `dt_bias`) stay as the checkpoint has them.
- **Correctness:** greedy tokens and teacher-forced scores against vLLM 0.30.0 on the same GPU (the engine's `check` and `score` with `tools/ref_vllm.py`), within vLLM-against-itself (another chunk size) as the noise floor.
- **The speed bar** (owner, 2026-10-05, `glyd-speed-bar`): per model and GPU, the engine at least matches vLLM 0.30.0 on tokens a second, time to first token and time per output token at 1, 8, 32, 64, 128 and 256 users, and uses less memory, or the model stays on vLLM there. `machine.rs: GPUS` gets a line only where the whole table is green.
- **Our own kernels only** (`glyd-own-kernels`): every kernel in `glyd-engine/kernels/` and the library is written here. FLA, vLLM's Triton kernels, FlashInfer and causal-conv1d are read for understanding and timed in scratch directories as yardsticks; nothing is copied, linked or fetched.

Not in scope here: the vision tower (Qwen3.5's 27-block ViT; a request with an image is refused with a 400 that names vLLM as the way), the MTP head as a draft model (its own phase, after the bar is met without it), the lossless KV cache at head 256 (the `kv-newer` work, C API 16; this engine takes it when it lands), FP8 checkpoints.

## What Qwen3.5 is (from transformers' `modeling_qwen3_5.py` and vLLM 0.30.0's `qwen3_5.py`, `qwen3_next.py`, `qwen_gdn_linear_attn.py`)

### Config

Two forms of one text model:

| | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForCausalLM` |
| :--- | :--- | :--- |
| top-level `model_type` | `qwen3_5` | `qwen3_5_text` |
| text fields | under `text_config` (`model_type: qwen3_5_text`) | at the top |
| tensor prefix | `model.language_model.` | `model.` |
| also has | `vision_config`, `model.visual.*` | nothing else |

Every 2026 dense checkpoint on the Hub is the first form. `tie_word_embeddings` is at the top level (0.8B, 2B, 4B: true; 9B, 27B, 3.6-27B, 3.8-27B: false). `eos_token_id` 248044 is in `text_config`; `generation_config.json` may list more.

| field | Qwen3.5-9B | what the engine does with it |
| :--- | :--- | :--- |
| `hidden_size`, `num_hidden_layers`, `intermediate_size`, `vocab_size` | 4096, 32, 12288, 248320 | |
| `layer_types` | 3 `linear_attention` then 1 `full_attention`, repeated | the layer schedule; `full_attention_interval` (default 4) where `layer_types` is absent |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | 16, 4, 256 | full attention, GQA 4:1 (0.8B/2B: 8/2, 4:1; the 27Bs: 24/4, 6:1), so the group size is a kernel parameter |
| `attn_output_gate` | true | `q_proj` gives a gate beside the query (below) |
| `output_gate_type` | absent (Qwen3.6-27B, Qwen3.8-27B: `swish`) | the gated norm's gate; absent, `silu` or `swish` is SiLU; anything else (vLLM also takes `sigmoid`) refused, named |
| `rope_parameters` | `rope_type: default`, `rope_theta: 1e7`, `partial_rotary_factor: 0.25`, `mrope_section: [11, 11, 10]`, `mrope_interleaved: true` | rotary on the first 64 of 256 dims; any other `rope_type` refused |
| `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim` | 16, 32, 128, 128 | the DeltaNet heads (0.8B/2B: 16 and 16; 27B: 16 and 48) |
| `linear_conv_kernel_dim` | 4 | causal depthwise convolution |
| `mamba_ssm_dtype` | float32 | the recurrent state is fp32; anything else refused |
| `rms_norm_eps` | 1e-6 | every norm, the gated one and the l2 norms' 1e-6 |
| `mtp_num_hidden_layers` | 1 | the MTP head's layers (`mtp.*`), loaded in the MTP phase |
| `hidden_act`, `attention_bias` | silu, false | anything else refused |

M-RoPE: with text only the three position streams (t, h, w) are equal, so the interleaved recomposition gives the plain rotary of 64 dims with `inv_freq = theta^(-2i/64)`. The engine checks `2 * sum(mrope_section) == rotary_dim` and uses the plain table (`weights::rope_tables(theta, 64, max_pos)`); image positions come with the vision phase.

### A layer

Both kinds: `x += mixer(norm1(x)); x += mlp(norm2(x))`, the MLP as Qwen3's (`down(silu(gate(x)) * up(x))`).

**Zero-centered RMSNorm** (vLLM's `GemmaRMSNorm`): `y = bf16(normalize_fp32(x) * (1 + w))`. The checkpoint's `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` and the final `norm` are stored centered on zero. The Qwen3 kernels multiply by `w`; Qwen3.5's add one first.

**Full attention** (`self_attn`, 1 layer in 4):

1. `q_proj` gives `heads * 2 * head_dim` rows: per head, 256 query values then 256 gate values (`view(..., heads, 512)`, `chunk(2)`).
2. `q = q_norm(q)`, `k = k_norm(k_proj(x))` per head over 256 (zero-centered); `v = v_proj(x)`.
3. Rotary on dims 0..63 of q and k (`rotate_half` within the 64: pairs `(i, i + 32)`), dims 64..255 pass through.
4. Causal attention, scale `256^-0.5`, `heads / kv_heads` query heads to a KV head (4 on 0.8B to 9B, 6 on the 27Bs).
5. `o = o_proj(attn * sigmoid(gate))`.

**Gated DeltaNet** (`linear_attn`, 3 layers in 4), per token, with `kd = nk * dk`, `vd = nv * dv`, `r = nv / nk`:

1. `qkv = in_proj_qkv(x)` (`2 kd + vd` rows: q, then k, then v), `z = in_proj_z(x)` (`vd`), `b = in_proj_b(x)`, `a = in_proj_a(x)` (`nv` each).
2. `qkv = silu(causal_conv1d(qkv))` over the `2 kd + vd` channels, kernel 4, weight `conv1d.weight [C, 1, 4]`, no bias. The state is the last 3 inputs of each channel (bf16, as vLLM keeps it).
3. Split q, k (`nk` heads of `dk`) and v (`nv` heads of `dv`); value head `h` reads key head `h / r` (`repeat_interleave`, not interleaved: vLLM's `gqa_interleaved_layout=False`).
4. `q = l2norm(q) / sqrt(dk)`, `k = l2norm(k)`, `l2norm(x) = x * rsqrt(sum(x^2) + 1e-6)`, in fp32 (vLLM's decode step keeps them fp32 in its kernel; its prompt path and transformers round them to bf16).
5. `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)` (`A_log` fp32, `dt_bias` bf16, in fp32).
6. Per value head, state `S` fp32 `[dk, dv]`:
   `S = exp(g) * S; S += k (beta * (v - S^T k))^T; o = S^T q`.
7. `o = bf16(normalize_fp32(o) * w * silu(z_fp32))` per value head over `dv`, rounded once (vLLM's `RMSNormGated`, norm before gate). `w` is `linear_attn.norm.weight`, fp32 `[dv]` in the checkpoint, **not** zero-centered, and used rounded to bf16, as vLLM loads it into a parameter of the model's dtype. (Transformers rounds differently: `bf16(n) * w`, then `* silu(z)`, one cast; the engine follows vLLM, its correctness reference.)
8. `out_proj(o)`.

A prompt runs step 6 in its chunked form (chunk 64: the WY form of Yang, Kautz and Hatamizadeh, "Gated Delta Networks", ICLR 2025, and Yang et al., "Parallelizing Linear Transformers with the Delta Rule", NeurIPS 2024): within a chunk, the decays' cumulative sums, the `I + tril(beta K K^T)` triangular solve, the intra-chunk products by MMA; across chunks, the state carried. A decode step runs it as above. Both are the same function of the inputs up to rounding; the CPU reference is the recurrent form.

### Per-sequence memory

| model | linear / full layers | DeltaNet state + conv state a sequence | KV a token (bf16) | 256 sequences' states |
| :--- | :--- | ---: | ---: | ---: |
| Qwen3.5-0.8B, -2B | 18 / 6 | 18.0 + 0.63 MiB | 12 KiB | 4.7 GiB |
| Qwen3.5-4B, -9B | 24 / 8 | 48.0 + 1.13 MiB | 32 KiB | 12.3 GiB |
| Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B | 48 / 16 | 144.0 + 2.81 MiB | 64 KiB | 36.7 GiB |

The state does not grow with the context and is rewritten every step: at 256 users of the 9B a decode step reads and writes 12.3 GiB of state, about 25 ms on an RTX 4090's 1 TB/s. vLLM pays the same. The KV cache is a quarter of a dense model's (1 layer in 4).

### Weights

Qwen3.5-9B: 775 tensors: 427 of the text model, 333 of the vision tower, 15 of the MTP head (the Hub's index; names, dtypes and shapes of the 0.8B and the 9B are the fixture `glyd-engine/tests/fixtures/qwen35_tensors.json`). The engine packs the text model's matrices, merged as vLLM merges them:

| pack (key: its first part) | parts | 9B rows x cols | packable (each part's rows % 64, cols % 16) |
| :--- | :--- | :--- | :--- |
| `self_attn.q_proj` | q_proj (q and gate), k_proj, v_proj | 10240 x 4096 | yes |
| `self_attn.o_proj` | o_proj | 4096 x 4096 | yes |
| `linear_attn.in_proj_qkv` | in_proj_qkv, in_proj_z | 12288 x 4096 | yes |
| `linear_attn.in_proj_b` | in_proj_b, in_proj_a | 2 x 32 x 4096 (0.8B/2B: 2 x 16; 27B: 2 x 48) | no: the packer merges a group only when each part passes on its own, and no part reaches 64 rows; kept bf16 on every dense model (0.5 MiB at most a layer) |
| `linear_attn.out_proj` | out_proj | 4096 x 4096 | yes |
| `mlp.gate_proj` | gate_proj, up_proj | 24576 x 4096 | yes |
| `mlp.down_proj` | down_proj | 4096 x 12288 | yes |
| `embed`, `lm_head` | | 248320 x 4096 | yes (tied on 0.8B, 2B, 4B) |

Kept as the checkpoint has them: the five zero-centered norms, `linear_attn.norm.weight` (fp32), `linear_attn.conv1d.weight` (bf16 `[C, 1, 4]`), `A_log` (fp32), `dt_bias` (bf16). Not loaded in this phase: `model.visual.*`, `mtp.*` (they are packed into the directory all the same, so the same directory serves the vision and MTP phases; that is objective 1's work in the packer).

The packer (`glyd-gpu/src/save.rs: FAMILIES`) knows only Qwen3 and the 2024 families; it gains `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM`, with the `model.language_model.` prefix and the merges above, from the same weight map the engine checks a checkpoint against (`glyd-engine/src/qwen35.rs`).

## The engine's design for Qwen3.5

### What stays

The engine's frame: the CUDA driver and the library opened at run time (C API 15), the Hub client and caches, `pack`, the tokenizer and chat template (minijinja), the sampler (top-64 on the GPU), the OpenAI server, the CLI, the status file and the hand-off to vLLM, the release build's gate (`check_wheel.py --engine-source`, `--engine`).

The library's packed product (`Lib::linear`, `Lib::unpack`) takes any packed matrix (rows % 64, cols % 16), so every Qwen3.5 product runs through the routes Qwen3's do.

### What is new

1. **`src/qwen35.rs`** (release build): `Config35` (both config forms, the refusals), the layer schedule (`Kind::Linear | Kind::Full` and the index within its kind, which is the layer's state index or KV layer), the weight map (every tensor the engine reads: name, dtype, shape, the pack it goes into; the classes it skips), the per-sequence sizes.
2. **`src/qwen35_ref.rs`** (development build only): the CPU reference in fp32 of the conv step, the l2 norm, the gated delta rule (recurrent, then chunked), the gated norm and a linear-attention layer's mixer for one token. The kernel checks compare the GPU against it.
3. **A model trait at the forward pass.** `model.rs` is Qwen3's forward, its graphs and its scheduler hooks. Qwen3.5 gets `model35.rs` beside it with the same interface the scheduler and the server call (`prefill`, `decode_rows`, `sample`, `free`), and `Engine` becomes an enum of the two. No generic layer abstraction: two forward passes, written out.
4. **Kernels** (`kernels/qwen35.cu`, the same `extern "C"` and `gen_names.py` rules as `engine.cu`):
   - the norms with `1 + w` (a flag on `rmsnorm` / `rmsnorm_add`, Qwen3's bits unchanged);
   - `q` and gate apart, q/k norm and partial rotary at head 256 (`qk_norm_rope` with the head size and rotary dims as template parameters);
   - attention at head 256: the prompt's (flash form, our own, as `attn_self`) and the decode step's over a bf16 paged cache, split along the context for long ones; the `sigmoid(gate)` product in the epilogue;
   - the DeltaNet front: conv1d (prompt: a chunk with the state in and out; decode: one step) fused with SiLU, the split, the l2 norms, `g` and `beta`;
   - the gated delta rule: decode (one CTA a sequence and value head, the 128 x 128 fp32 state read and written once, the gated norm and `silu(z)` in the epilogue) and prompt (chunked, MMA for the intra-chunk products, the triangular solve in shared memory);
   - nothing new for the MLP, the products, sampling.
5. **The KV cache.** Phase 1: a bf16 paged cache of the full-attention layers, 16-token pages, the engine's own. The lossless cache takes over at head 256 through C API 16 when `kv-newer` lands, with its own exactness checks. With 1 layer in 4 holding a cache, the engine still uses less memory than vLLM bf16 by the weights alone (Qwen3.5-9B on an RTX 4090 under vLLM 0.30.0: 12.68 GiB packed against 17.66 GiB bf16, vLLM's "Model loading took" lines; bf16 then ran out of memory at its defaults. Record: `benchmarks/gpu/v029-qwen35-9b-2026-10-05/rtx4090/glydlogs/run-20261006-013815.log` and `rtx4090/vllm_q35_9b.serve.out`, moved into the repo by PR #20).
6. **The state store.** One fp32 state and one bf16 conv state per running sequence and DeltaNet layer, in a store sized by the users asked for (`fit.rs` learns states: weights + states x users + KV x tokens + work). `n > 1` choices copy the state after the prompt. A finished prompt keeps a snapshot of its states at its last whole chunk with its KV pages, so a chat's next turn starts from there (the Qwen3 engine's prefix reuse, at chunk granularity: a recurrent state cannot be cut at any token).
7. **Many users.** The Qwen3 engine decodes up to 8 sequences together (`MAX_ROWS`). The bar asks for 256: the scheduler admits up to 256 decode rows, CUDA graphs per bucket (1, 2, 4, 8, 16, 32, 64, 128, 256), prompts chunked beside decode rows (vLLM's chunked prefill). The products at 64 to 256 rows are the risk (below).
8. **Chat, tools, thinking.** Qwen3.5's own template (`chat_template.jinja`) rendered by minijinja, checked against transformers' rendering (fixtures made where transformers runs: CI or a pod, never this Mac); thinking returned apart as Qwen3's is; tool calls in the `qwen3_coder` format (`<tool_call><function=NAME><parameter=P>...</parameter></function></tool_call>`, vLLM's parser of that name), a parser of our own beside the hermes one in `tools.rs`.
9. **The hand-off.** `cli.rs: supported()` and `machine.rs: config_refusal()` take `qwen3_5` with head 256 once the forward pass is in; `GPUS` stays empty until a table is green; `GLYD_ENGINE=1` runs it before.

### The speed bar and where it is hard

- **1 to 8 users.** Decode reads the weights; Qwen3.5-9B's packed weights measured 28% (tiered, RTX 4090) and 21% (12-bit, RTX 5090) less than bf16 (`benchmarks/gpu/v029-qwen35-9b-2026-10-05/README.md`), and the engine's step is one CUDA graph. The DeltaNet step adds 48 MiB of state read and 48 MiB written a sequence a step (9B), the same as vLLM's.
- **32 to 256 users.** The products become compute-bound and the packed weights must be decoded at that rate. Decoding a matrix to bf16 and multiplying (the prompt path) moves more bytes than vLLM's bf16 product. The library's fused products cover 1 to 64 tokens (`mma_gemm`, `mma_gemm_mid`, `mma_gemm_wg`); the bar needs them, or a decode-in-registers product of our own, at 128 and 256 rows within vLLM's time per product, shape by shape (`tools/gemm_bench.cu` with the Qwen3.5 shapes). This is the largest risk; the plan measures it first, before the forward pass is built, so the owner sees early whether the bar is reachable on each GPU.
- **Time to first token.** The chunked delta rule and the head-256 prompt attention must match FLA's and FlashAttention's times on the same GPU, kernel by kernel, against vLLM's (the yardsticks, timed in scratch).
- **Memory.** Always less: packed weights, the same states and KV as vLLM. The table records each side's peak and the most users each fits.
- **Loads that do not fit.** States grow with users: at 256 the 9B's are 12.3 GiB, which with its 12.68 GiB of packed weights is over an RTX 4090's 24 GB before any KV, and vLLM bf16 fits fewer still (on the 4090 it did not start at its defaults); the 27B's are 36.7 GiB. So the bar is: tokens a second, time to first token and time per output token at every user count of 1, 8, 32, 64, 128, 256 that both sides fit, and one more cell, the most users each side fits, green when the engine fits at least as many as vLLM. A count neither side fits is not a cell. Run 4's user counts per GPU and model come from `fit.rs` (and vLLM's own log of its maximum concurrency), written into the job before launch.

If a cell stays red on a GPU, the model stays on vLLM there (memory-only message), and the engine's work continues on that cell.

### Checks

- **Host (CI's `engine` job, both builds):** config parsing of every 2026 Qwen3.5 config (dense accepted, MoE refused with its reason), the schedule, the weight map against the real tensor indexes of Qwen3.5-0.8B and -9B (fixtures: names, dtypes, shapes), per-sequence sizes, the CPU reference's identities (the delta rule's write-then-read, the decay, the conv step against the whole convolution, the GQA mapping, the gated norm), and later the chunked form against the recurrent one, the template, the tool-call parser.
- **GPU, kernel by kernel** (a pod): each new kernel against the CPU reference on random and real inputs (Qwen3.5-0.8B's first layers): conv and gated delta rule within fp32 rounding, chunked against recurrent, attention head 256 against a plain fp32 product, Qwen3's kernels' bits unchanged.
- **GPU, the model** (a pod): Qwen3.5-0.8B and -9B against vLLM 0.30.0 bf16 and vLLM+Glyd: the Qwen3 engine's set (8 short prompts x 64 greedy tokens, 3 long prompts, teacher-forced top-1 and mean absolute logprob difference over 1,500 positions), with vLLM against itself (another chunk size) as the floor; weights unpacked equal to the checkpoint bit for bit.
- **GPU, the bar** (pods): the table above on an RTX 4090 (Ada, tiered), then an H100 and an RTX 5090 (12-bit), Qwen3.5-9B and the 27Bs (Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B) where they fit, at the user counts of the bar above.

## Gemma 4 (the next model family, its own spec once Qwen3.5's model check passes)

From transformers' `modeling_gemma4.py` and the 2026 configs (`google/gemma-4-E2B-it`, `-E4B-it`, `-12B-it`, `-31B-it`; `-26B-A4B-it` is MoE):

- **Sliding and full attention.** `sliding_attention` layers (window 512 on E2B/E4B, 1024 on 12B/31B; head 256, `rope_theta` 1e4) and `full_attention` layers (`global_head_dim` 512, `num_global_key_value_heads`, `rope_type: proportional` with `partial_rotary_factor` 0.25, theta 1e6). With `attention_k_eq_v` (12B, 31B, 26B-A4B) the full layers have no `v_proj`: the values are the keys.
- **Norms:** q, k and v norms (v without a weight), attention scaling 1.0, four norms a layer (before and after the attention and the MLP), a `layer_scalar`, the embedding scaled by `sqrt(hidden)`.
- **MLP:** `gelu_pytorch_tanh`; `use_double_wide_mlp` on some.
- **E2B / E4B:** per-layer input embeddings (`vocab_size_per_layer_input` x `hidden_size_per_layer_input` a layer, a gate and a projection a layer) and KV shared between the last `num_kv_shared_layers` layers (20 on E2B, 18 on E4B).
- **Logits:** `final_logit_softcapping` 30.
- **12B:** `Gemma4UnifiedForConditionalGeneration` (`gemma4_unified`), its own variant to read.
- **Vision and audio towers**: skipped as Qwen3.5's.

What carries over from Qwen3.5: head 256 attention, partial rotary, the paged bf16 cache, the many-users scheduler and the products at 128 to 256 rows. What is new: the sliding window in the cache and the prompt kernel, head 512, `k_eq_v`, the per-layer embeddings, the softcap. Its spec and plan are written when Qwen3.5's model check (plan Task 12) passes.

## Costs

- Host work and CI: no cost.
- GPU runs (Runpod, `rp_run.sh`, one batched job per pod, no pod outlives its job, the owner approves each before launch): the products' measurement (an RTX 4090 and an H100, about 1.5 h, about $5); the kernels (an RTX 4090, about 1.5 h, about $1); the model check (an RTX 4090, about 1.5 h, about $1); the bar (an RTX 4090, an H100, an RTX 5090, about 6 h in all, about $15). About $22 in all, in four runs.
