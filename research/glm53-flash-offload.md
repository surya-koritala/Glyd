# GLM-5.3-Flash on one 24 GB GPU + 128 GB RAM (GLY-127 research spike)

Read-only research. Nothing committed, no engine code touched, no pod started, no token used. Nothing here was **run on the model**: every number is tagged

- **[file]** read from a file at the pinned sha (config, tensor headers, modeling code, docs),
- **[calc]** computed by me from [file] numbers (script: scratchpad `model.py`, formulas in section 4),
- **[assume]** an assumption or analogy, to be replaced by a measurement,
- **[doc]** a number stated in someone else's docs (Strata, vLLM, KTransformers), not re-measured.

## 0. Sources and pins

| What | Pin |
|---|---|
| zai-org/GLM-5.3-Flash (ungated, MIT, `lastModified` 2026-09-07 12:13 UTC, i.e. 08:13 Eastern) | sha `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`; 62 safetensors shards; `Glm5NextForConditionalGeneration`, `model_type glm5_next`, config written by transformers 5.16.0 |
| Tensor names/shapes | `/api/models/.../safetensors` is a 404 for this repo, so I read all 62 shard headers with HTTP range requests (8 bytes + JSON header each, no weights): 76,108 tensors. The repo's API `safetensors` field gives totals: 321,323,031,390 params (F8_E4M3 314,396,639,232; BF16 6,926,096,640) [file] |
| Modeling code | transformers 5.19.0 wheel, `models/glm5_next/{modeling,configuration}_glm5_next.py` [file]. The HF code **does not implement the MTP layer** (loads 45 layers, ignores layer 45) |
| vLLM | `vllm/models/glm5next/{common/{model,kda,mtp,sparse_indexer,attention}.py, nvidia/ops/..., amd/...}` on main (GitHub tree listing; read `mtp.py` header only) [file]; recipe page recipes.vllm.ai/zai-org/GLM-5.3-Flash |
| KTransformers | `doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md` (prior art, see 7) |
| Strata | github.com/Niko1221/Strata, main `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (2026-10-08), **MIT** (LICENSE; ggml parts MIT, one data file under Qwen Community License) |
| Glyd engine | `/Users/suryakoritala/glyd-v030-check/glyd-engine/src` (`model35.rs`, `qwen35.rs`, `weights.rs`, `fit.rs`, `pack.rs`) |

Model card facts [file]: "320B total, 18B active", first natively multimodal GLM, hybrid sparse+linear attention, mHC. Serving: SGLang, vLLM, TokenSpeed, Transformers, KTransformers, Unsloth. Thinking is always on; `reasoning_effort` low/high/max (default max); `clear_thinking` defaults false. generation_config: temperature 1.0, top_p 0.95; eos ids 154820, 154827, 154829. vLLM recipe: FP8 checkpoint "about 306 GiB" (= 328.3 GB, matches my tensor sum 328.33 GB), Hopper or newer, FlashInfer >= 0.6.17, **Hopper has no FP8 KV cache for this model**, MTP with `num_speculative_tokens 5`. A RedHatAI NVFP4 variant exists (4-bit experts, Blackwell only) [doc].

## 1. Architecture

All from `config.json` / tensor headers [file] unless stated.

**Text model**: 45 backbone layers (0..44) + 1 MTP layer (index 45 in the checkpoint), hidden 4096, vocab 154,880, untied embedding and head (both bf16, 634.4M params each), RMSNorm eps 1e-5, **no RoPE anywhere** (NoPE: `qk_rope_head_dim 0`, `mla_use_nope`; HF passes `position_embeddings=None`), max positions 1,048,576.

**Layer pattern**: layers 3,7,...,43 (11 layers) are sparse MLA; the other 34 are KDA. Layers 0-2 have a dense MLP (intermediate 12288); layers 3-44 (42) and the MTP layer are MoE.

**MoE** (42 backbone layers + MTP):
- 288 routed experts + 1 shared expert (shared intermediate = 2048 x 1), top-8, `moe_intermediate_size` 2048.
- Router: `mlp.gate.weight` [288, 4096] **bf16**, computed in **fp32** (`moe_router_dtype float32`), `scoring_func sigmoid`, `topk_method noaux_tc` with `e_score_correction_bias` [288] fp32 used for selection only, `n_group 1`/`topk_group 1` (so no group limiting), `norm_topk_prob true` (weights = sigmoid scores of the chosen 8, divided by their sum), `routed_scaling_factor 2.5`.
- Expert: `gate_proj` [2048,4096], `up_proj` [2048,4096], `down_proj` [4096,2048], no bias: **25,165,824 params per expert** (3 x 2048 x 4096). SwiGLU with clamps: gate <= 10, up in [-10, 10] (`swiglu_limit 10`) for routed, shared and dense MLPs.
- Output = sum of 8 weighted experts + shared expert (unweighted).

**mHC (manifold-constrained hyper-connections)**, new vs. Qwen3.5 and every earlier model the engine has: `hc_mult 4`. The residual is 4 streams [4, 4096] per token. Each layer has two mixers (`hc_attn_*`, `hc_ffn_*`), each with `fn` [24, 16384] bf16, `base` [24] f32, `scale` [3] f32. It RMS-norms the flattened 16384 vector (no weight), projects to 24 = 4 (pre) + 4 (post) + 16 (comb), applies sigmoid / 2*sigmoid / softmax + 20 Sinkhorn-Knopp row/column normalisations to get a doubly-stochastic 4x4 `comb`; block input = sum_i pre_i * stream_i; new streams = post (x) block_out + comb^T @ streams. Final: mean of the 4 streams, then `model.norm`. Per-layer cost is small (24 x 16384 matvec) but it is a new op sequence on fp32 and has to be bit-compatible.

**KDA (Kimi Delta Attention) layers** (34): 64 heads, head_dim 128, q/k/v/o all bf16 in the checkpoint (not FP8): q,k,v [8192,4096], o [4096,8192]. Differences from Gated DeltaNet (Qwen3.5, which the engine has):
- Gate `g` is **per key-channel** ([heads, 128]) rather than one scalar per head: `g = -5 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`, with a low-rank f_a [128,4096], f_b [8192,128], `A_log` [64], `dt_bias` [8192]; `gate_lower_bound -5` ("safe gate"). GDN uses a scalar-per-head `exp(-exp(A_log) * softplus(a + dt_bias))`.
- Update (recurrent, per head): `S <- diag(exp(g)) S`; `delta = (v - S^T k) * beta`; `S <- S + k delta^T`; `o = S^T q` with q scaled by 1/sqrt(128), q and k l2-normalised (in-kernel). `beta = sigmoid(b_proj x)` [64,4096]. Same delta rule as GDN, but the decay multiplies the key dimension of S per channel.
- Short conv, kernel 4, depthwise, SiLU, applied separately to q, k, v (checkpoint has `q_conv1d`, `k_conv1d`, `v_conv1d` [8192,1,4]; HF concatenates them into one conv of 24576 channels).
- Output: per-head `o_norm` (RMS, weight [128]) **gated by sigmoid** of a low-rank gate `g_b(g_a(x))` (g_a [128,4096], g_b [8192,128]), then `o_proj`. GDN uses SiLU gating.
- Recurrent state per layer: 64 x 128 x 128 fp32 = **4 MiB**; x34 = **136 MiB per sequence** [calc]; conv state 3 x 8192 x 3 per layer (tiny). Compared with Qwen3.5-27B-class GDN this is a larger state per layer, and decode reads+writes it every token.
- Engine already has chunked + recurrent delta kernels (`Kern35::delta`, `delta_case`) for the scalar-gate form; KDA needs the per-channel-gate variant (vLLM ships Triton `fused_recurrent`/`kernels` for it, third_party/kda, from FLA).

**MLA / DSA sparse layers** (11 + MTP): 64 heads; `q_lora_rank 1536`, `kv_lora_rank 512`, `qk_nope_head_dim 256`, **`qk_rope_head_dim 0`**, `v_head_dim 256`. Tensors: `q_a_proj` [1536,4096] FP8, `q_a_layernorm`, `q_b_proj` [16384,1536] FP8, `kv_a_proj_with_mqa` [512,4096] FP8 (512 = latent only, no rope part), `kv_a_layernorm`, `kv_b_proj` [32768,512] **bf16** (32768 = 64 x (256+256)), `o_proj` [4096,16384] FP8. Scale 1/sqrt(256). Cached state per token per layer is only the 512-wide latent (HF expands to K/V; an engine would use the absorbed form for decode: scores in latent space, since there is no rope to block absorption). No RoPE removes a whole kernel class relative to Qwen3.5 (which rotates part of each head).
- **Sparse indexer** (every one of the 11 layers has its own; `indexer_types` are all "full", no sharing in the shipped config; code supports "shared"): `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048`. `wq_b` [4096,1536] (from the q latent), `wk` [128,4096] + LayerNorm (weight+bias), `weights_proj` [32,4096], all bf16. Score = sum_h w_h * relu(q_h . k / sqrt(128)) with w = weights_proj(x)/sqrt(32).
- **kpool compress**: `index_kpool 4`: keys are grouped 4 tokens at a time into pools; pool key = softmax-weighted average of the 4 keys, where logits = `index_kpool_compress_gate` (x) [128,4096] + `index_kpool_compress_ape` [4,128]. Top (2048/4 = 512) pools are selected, expanded back to 2048 raw tokens, and `index_kpool_always_select_tail` appends the incomplete tail pool (up to 3 tokens): width 2051. Pooling starts at the first valid token. Attention is then dense over those <= 2051 positions per query. `index_share_for_mtp_iteration` = MTP drafts reuse the indexer selection.
- Cache per token (all 11 layers): latent 512 x bf16 = 1 KiB + indexer key/gate 128+128(+1 valid flag in HF) x bf16 ~ 0.5 KiB, so **~16.9 KB/token** => 32K ctx 0.55 GB, 128K 2.2 GB, 1M 17.7 GB [calc]. Tiny next to Qwen-style full KV.

**MTP layer** (checkpoint layer 45): `enorm`, `hnorm` [4096], `eh_proj` [4096,8192] (concat of embedding and hidden -> hidden), `shared_head.norm`, then the same block as a sparse-MLA + MoE layer (own 288 experts + shared + router + indexer; ~~hc mixers~~ **correction, 2026-10-11: the MTP layer has no `hc_*` tensors (the inventory fixture `glyd-engine/tests/fixtures/glm5_next_tensors.json` has none for layer 45) and vLLM runs it on a plain residual stream, with `shared_head.norm` fused with the last residual add; measured on the real weights, the head's draft is the model's own next pick for 64% of 126 positions**). It reuses the main `lm_head`/embedding. 7.25B routed-expert params + 0.18B other [file]. `num_nextn_predict_layers 1`; vLLM recipe uses 5 speculative tokens by looping the single layer.

**Vision**: `model.visual.*`, 347 tensors, 563.6M params (24-block ViT, hidden 1024, patch 14, merger to 4096). Skip, as for Qwen (the engine already does not load vision or MTP).

**FP8 layout** [file]: `quant_method fp8`, `fmt e4m3`, `activation_scheme dynamic`, **`weight_block_size [128,128]`**. Each FP8 weight `X.weight` (F8_E4M3 [N,K]) has `X.weight_scale_inv` (F32 [ceil(N/128), ceil(K/128)]); dequant is `w = fp8 * scale_inv` per 128x128 tile (DeepSeek convention; to be confirmed against the HF finegrained-fp8 kernel before the packer is written). Examples verified from shapes: expert gate/up [2048,4096] -> scale [16,32]; down [4096,2048] -> [32,16]; dense MLP gate [12288,4096] -> [96,32]; kv_a [512,4096] -> [4,32]; o_proj [4096,16384] -> [32,128]. Checkpoint is **mixed**: FP8 = routed experts, shared experts, dense MLPs, MLA q_a/q_b/kv_a/o_proj. **bf16** = all KDA tensors, MLA `kv_b_proj`, indexer, routers, hc `fn`, embedding, lm_head, norms, MTP `eh_proj`. f32 = hc `base`/`scale`, gate bias, `A_log`, `dt_bias`, scales. Dynamic activation scheme: activations are quantised at runtime, so FP8 weights alone define the "reference"; our bf16-dequantised reference is the baseline for quality.

**Tensor-name prefixes** [file]: `model.language_model.layers.N.*`, `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight`, `lm_head.weight`, `model.visual.*`. Per layer: `input_layernorm`, `post_attention_layernorm`, `hc_attn_{fn,base,scale}`, `hc_ffn_{fn,base,scale}`, `self_attn.*`, `mlp.{gate.weight, gate.e_score_correction_bias, experts.E.{gate,up,down}_proj.weight(+_scale_inv), shared_experts.*}` (**per-expert tensors, not fused**: 12,384 x 3 expert matrices in the index; the HF runtime stacks them). MTP: layer 45 with `eh_proj`, `enorm`, `hnorm`, `shared_head.norm`. Note per-expert tensors are an advantage for an offload packer (each expert is its own contiguous blob).

## 2. Parameter accounting (from the 76,108 tensor shapes) [calc from file]

| Group | Params (B) | Notes |
|---|---:|---|
| routed experts, 42 backbone MoE layers | 304.406 | 12,096 experts x 25.17M |
| routed experts, MTP layer | 7.248 | 288 experts |
| shared experts (42 + MTP) | 1.057 + 0.025 | 25.17M each |
| dense MLP (layers 0-2) | 0.453 | 151M each |
| KDA (34 layers) | 4.683 | 137.7M/layer (q,k,v 100.7M, o 33.6M, gates/conv ~3.4M) |
| MLA (11 layers) | 1.292 | 117.4M/layer (o_proj alone 67.1M) |
| indexer (11 layers) | 0.082 | |
| MTP non-expert (MLA, indexer, eh_proj, router) | 0.16 | |
| routers (42) | 0.050 | 1.18M each |
| mHC mixers (90) | 0.035 | |
| embedding | 0.634 | row lookup only |
| lm_head | 0.634 | read every token |
| vision | 0.564 | skipped |
| norms, biases, A_log etc. | <0.01 | |
| **Total** | **321.34** (320.78 w/o vision) | FP8 scales excluded (HF count 321.323B) |

**Active per token** (no MTP): KDA 4.683 + MLA 1.292 + indexer 0.082 + dense 0.453 + shared 1.057 + routers/hc 0.085 + 8/288 of routed (8.456) + lm_head 0.634 = **16.75B**, **17.4B** counting the embedding row. With the MTP layer active (+0.42B) = **~17.8B**, consistent with the card's "18B" (the card most likely counts embedding + MTP). The model is therefore ~96% routed experts by parameter count, and only 2.6% of those are touched per token (3.1% in a sparse-expert sense: 8 of 288). **The "always-on" part is just 8.1B matrices (+0.17B small) = 2.5% of the model; the 12,096 experts are the whole offload problem.**

## 3. Memory at 3.0 / 3.5 / 4.0 / 5.5 bits per weight [calc]

Assumptions [assume]: "bpw" is the all-in cost per matrix weight (codes + scales; the engine's Int5 is 5.53 all-in, so 5.5 ~ "Int5 today"). Indexer, routers, mHC, norms stay 16-bit (0.35 GB); embedding and lm_head are quantised like everything else (the head could be held at higher precision: +0.3 GB at 8 bit). Vision excluded. 1 GB = 1e9 bytes.

| bpw | (a) everything incl. MTP + embed | (b) routed experts: backbone / MTP | (c) always-on matrices (attn, shared, dense, head) | embedding (can live in host RAM) | one expert | routed bytes per token (336 experts) |
|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 120.6 GB | 114.2 / 2.8 | 3.0 + 0.35 small | 0.24 | 9.44 MB | 3.17 GB |
| 3.5 | **140.6 GB** | 133.2 / 3.2 | 3.6 + 0.35 | 0.28 | 11.01 MB | 3.70 GB |
| 4.0 | 160.6 GB | 152.2 / 3.7 | 4.1 + 0.35 | 0.32 | 12.58 MB | 4.23 GB |
| 5.5 | 220.8 GB | 209.3 / 5.1 | 5.6 + 0.35 | 0.44 | 17.30 MB | 5.81 GB |
| (FP8 source, for reference) | 328.3 GB | | | | | |

Non-weight VRAM [calc/assume]: KV 0.55 GB @32K (2.2 @128K); KDA state 0.14 GB per sequence; CUDA context + workspaces + prefill activations + expert staging **2.0 GB** [assume; the engine's `fit.rs` uses 0.6 GB runtime + 0.4 GB headroom for dense models, I added room for MoE staging].

**VRAM left for a hot-expert cache** (32K context, MTP head not counted; each MTP draft layer needs its own ~0.2B non-expert params, ~0.1 GB):

| Card | 3.0 bpw | 3.5 bpw | 4.0 bpw | 5.5 bpw |
|---|---|---|---|---|
| 16 GB | 9.9 GB = 1,049 experts (8.7% of 12,096) | 9.4 GB = 853 (7.1%) | 8.9 GB = 706 (5.8%) | 7.4 GB = 425 (3.5%) |
| 24 GB | 17.9 GB = 1,897 (15.7%) | **17.4 GB = 1,580 (13.1%)** | 16.9 GB = 1,342 (11.1%) | 15.4 GB = 888 (7.3%) |
| 32 GB | 25.9 GB = 2,745 (22.7%) | 25.4 GB = 2,306 (19.1%) | 24.9 GB = 1,978 (16.4%) | 23.4 GB = 1,350 (11.2%) |

Experts here are **huge** (11 MB at 3.5 bpw, versus ~1.4-2 MB for Strata's Qwen3.8-Flash-Next at 2-3 bpw), so a 24 GB card holds only ~13% of them.

**RAM check, the hard constraint** [calc]: with the cache in VRAM and the rest in RAM (no duplicate copy, Strata's "resident low-RAM mode"):

| bpw | experts in RAM pool (24 GB card) | Fits 128 GB RAM with ~12 GB left for OS + KV spill + prompt buffers? |
|---|---:|---|
| 3.0 | 99 GB | yes, 29 GB spare |
| 3.5 | **119 GB** | marginal: ~9 GB left in total, no room for a second copy, long-context KV or a browser |
| 4.0 | 139 GB | **no** (needs NVMe spill) |
| 5.5 | 199 GB | no |

If the pool is a full pinned copy of all experts (Strata's default: all experts in RAM, plus the GPU's copy), 3.5 bpw needs 136.7 GB: **does not fit 128 GB**. So "calibrated 3.5-4 bits uniform on 128 GB RAM" is not feasible as stated; ~3.0-3.3 bpw average (or mixed bits, section 7) is.

## 4. Decode bandwidth model [calc, with stated assumptions]

Per decoded token (no speculation), bytes read:
- always-on (weights ~3.6 GB @3.5 bpw + 0.35 small) + KDA state read+write fp32 (2 x 136 MiB = 0.285 GB [assume fp32 state]) + KV/indexer reads (~0.03 GB at 32K) ~ **4.3 GB from VRAM**,
- routed: 42 layers x 8 experts = **336 experts = 3.70 GB @3.5 bpw** (3.17 @3.0, 4.23 @4.0), a fraction `h` (hit rate, by bytes = by count, all experts equal size) from VRAM, `1-h` from the host pool.

Time per token: `t_gpu = 5 ms + (A + h R) / (BW_vram * 0.70)` (5 ms = per-layer launch/sync overhead over 45 layers, 0.7 = achieved fraction of peak bandwidth [assume]). Miss paths:
- **P (PCIe stream)**: miss experts are copied to VRAM, then used. The router for layer L is known only after layer L-1 and L's attention, so copy cannot overlap with the compute that needs it: serial, `t = t_gpu + (1-h) R / BW_pcie`; BW_pcie 25 (Gen4 x16 effective) or 50 GB/s (Gen5) per the brief.
- **C (CPU in place; what Strata does)**: the CPU computes missed experts directly from RAM in parallel with the GPU; only activations cross PCIe. `t = max(t_gpu, (1-h) R / BW_cpu)`, BW_cpu = **45 GB/s effective** [assume: 70 GB/s DDR5 dual-channel peak x ~0.65 for a well-tuned AVX-512/AVX2 quantised GEMV]. This is an upper bound until a CPU kernel is measured; at 30 GB/s the results below drop by a third where the CPU term dominates.
- Both paths read the same DDR: PCIe-streamed experts also consume DDR bandwidth, so a hybrid (Strata's `--pcie-frac`) is capped at the DDR total (~55 GB/s effective [assume]); **PCIe 5.0 adds nothing once DDR is the limit**.
- Zero-cache floor: all experts from RAM at 45 GB/s = 3.70/45 = 82 ms => **12.2 tok/s with no skew at all** (C path).

**tok/s, 3.5 bpw, no speculation** (P4 = PCIe 4.0 x16 25 GB/s, P5 = PCIe 5.0 50 GB/s, C = CPU path 45 GB/s):

| hit rate | 4090 P4 | 4090 P5 | 4090 C | 5090 P4 | 5090 P5 | 5090 C | 3090 P4 | 3090 P5 | 3090 C |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | 11.4 | 19.8 | 24.3 | 11.9 | 21.4 | 24.3 | 11.3 | 19.5 | 24.3 |
| 70% | 16.9 | 27.1 | 40.5 | 18.2 | 30.7 | 40.5 | 16.7 | 26.6 | 40.5 |
| 80% | 22.3 | 33.4 | 60.8 | 24.8 | 39.2 | 60.8 | 22.0 | 32.5 | 60.8 |
| 90% | 32.8 | 43.3 | 63.7 | 38.7 | 54.3 | 90.8 | 31.9 | 41.8 | 60.5 |
| 95% | 42.8 | 50.9 | 62.7 | 53.9 | 67.3 | 89.6 | 41.3 | 48.9 | 59.5 |

(The 4090 / 3090 rows go flat above ~85% under C because the 5 ms overhead plus VRAM time become the limit; the 5090's extra bandwidth shows there.)

**Hit rate needed** (3.5 bpw):

| target | 4090 P4 | 4090 P5 | 4090 C @45 | 4090 C @30 | 5090 P4 | 3090 P4 |
|---|---:|---:|---:|---:|---:|---:|
| 10 tok/s | 41% | ~0% | ~0% | 19% | 39% | 42% |
| 20 tok/s | **76%** | 51% | **39%** | 60% | 73% | 77% |

At 4.0 bpw (4090): P4 needs 50% / 80%, C@45 needs 0% / 47%. At 3.0 bpw: P4 30% / 71%, C@45 0% / 29%.

**What hit rate can a 24 GB card plausibly reach?** Unknown until measured (section 5). The only analogue is Strata's Qwen3.8-Flash-Next (512 experts/layer, top-10, 48 layers) [doc]: 16 GB card with ~11-13% of experts resident: 60-63% hit; 32 GB card with ~45% resident: 90-95% hit; a layer-split measurement of 23.6% held: 69.2% (DETAILS.md lines ~613, 629, generate.cpp 3194ff). Applying that curve to our 13.1% (24 GB @3.5) gives **~60-65% [assume]**, to 19% (32 GB) **~67-70%**, to 7% (16 GB) **~45-50%**. GLM's experts are larger and fewer per layer (288), and its router uses auxiliary-loss-free balancing, which pushes toward *flatter* usage over a batch; per-domain skew can still be strong. So the skew curve is the key unknown and may be better or worse than Qwen's.

Plausible outcome at ~62% hit, 4090, 3.5 bpw: **P4 ~14 tok/s, P5 ~23, C ~33** (interpolating the table). So 10 tok/s looks reachable even over PCIe 4.0; 20 tok/s needs either the CPU path or PCIe 5.0 with a better-than-analogue hit rate.

**MTP / speculative decoding effect [assume, wide range]**: verifying k drafts reads the union of the experts of k+1 tokens, so the routed-bytes term grows (U) while tokens per pass (A) grows too. Strata measured 2.4-3.2 accepted tokens per pass with 3 drafts on its model [doc]; GLM's single MTP layer looped 5x will have lower acceptance in later positions. I model (A, U) = (1.5, 1.5) / (2.0, 1.8) / (2.5, 2.2): speedup on the 4090 at 3.5 bpw is **1.04-1.15x at 50% hit up to 1.3-1.8x at 90% hit** (e.g. P4 @70%: 16.9 -> 18.1 / 20.5 / 21.4; C @90%: 63.7 -> 83 / 103 / 117). Low-hit regimes (the likely ones) gain little, because every extra verified token multiplies the missed-expert bytes. MTP also costs another lm_head read (0.28 GB) and a 288-expert MoE layer per draft; its experts (3.2 GB) must be resident somewhere.

**Prefill** [calc]: layerwise streaming of all 133 GB of experts per chunk caps prefill at chunk / (133 / BW): with 8192-token chunks **~1.5k tok/s on PCIe 4.0, ~3k tok/s on 5.0** (4K chunks: 0.77k / 1.5k). A 32K prompt = 21 s at best on Gen4. KTransformers' "Layerwise Prefill" is the same idea [doc]. The thinking-always-on behaviour means decode speed dominates user-visible latency.

## 5. Measuring expert-use skew (the experiment that decides this)

**Goal**: for each calibration domain, per-layer and global expert counts over N tokens, plus the token-ordered routing trace, so offline code can compute (i) static top-K hit rate versus K (the cache-size curve), (ii) cross-domain transfer (profile from domain A evaluated on B; this is what a shipped profile would give), (iii) adaptive/LRU hit rate on sequential traces, (iv) the routing-weight mass hit (matters for mixed bits), (v) co-activation. Teacher-forced forward over text gives the same router inputs as decode of that text; it misses the effect of the model's own sampled thinking text, so include model-generated text for ~10% of tokens (generate with the same model, a few thousand tokens per domain).

**Pod choice** [prices from `list-gpu-types` secure cloud today, approximate]. vLLM glm5next needs Hopper or newer (FlashInfer NoPE sparse MLA), so A100 is out for vLLM, and A100 has no FP8 so transformers would dequantise to bf16 = 656 GB, which does not fit 8 x 80 GB = 640 GB. Weights are 328 GB, so 4 x 80 GB (320 GB) does **not** fit either. Options with the transformers path (device_map auto, FP8 finegrained kernels on SM89+):

| Pod | Memory | ~$/h (secure) | Notes |
|---|---:|---:|---|
| **4 x RTX PRO 6000 Blackwell 96 GB** | 384 GB | 4 x 2.49 = **~10** | cheapest that fits; SM120, FP8 ok in transformers (not verified); also the architecture of the target consumer 5090 |
| 4 x H100 NVL 94 GB | 376 GB | 4 x 3.19 = ~12.8 | |
| 4 x H200 141 GB | 564 GB | 4 x 5.29 = ~21 | room for bigger batches / vLLM |
| 8 x H100 SXM 80 GB | 640 GB | 8 x 3.99 = ~32 | vLLM-capable, overkill |

**Time and cost** [assume]: container disk >= 400 GB; download 328 GB from HF at 0.5-1.5 GB/s = 4-12 min (+ unpack, load to GPU ~5-10 min); smoke test + debugging 30-60 min; trace run: teacher-forced 2 M tokens at an assumed 2-4k tok/s in a naive pipeline (HF per-expert loop is slow; use `experts_implementation` grouped if available) = 10-17 min; extra 100k generated tokens, a few more minutes. Plan **2.5-3 h** => 4 x PRO 6000 **~$25-30**; 4 x H100 NVL ~$32-38; add margin for a failed first load: budget **$40**. Delete the pod when done; network volume not needed.

Alternative to hand-written hooks: SGLang has an expert-distribution recorder and vLLM has EPLB load stats; I did not verify either supports glm5_next, so the hook script below is the primary plan.

**Script sketch (unrun; ~60 lines; transformers >= 5.19)**

```python
# routing_trace.py: count router decisions per layer/domain, save token-ordered traces
import sys, json, glob, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
M, REV = "zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
DOM_DIR, OUT, SEQ, MAXTOK = sys.argv[1], sys.argv[2], 4096, int(sys.argv[3])  # DOM_DIR/<domain>.jsonl {"text":...}
tok = AutoTokenizer.from_pretrained(M, revision=REV)
model = AutoModelForImageTextToText.from_pretrained(M, revision=REV, device_map="auto",
        dtype=torch.bfloat16).eval()   # FP8 finegrained weights per config; 4 GPUs, sequential pipeline
NL, NE, K = 45, 288, 8
cur = {}                                # layer -> [T, 8] int16 for the sequence in flight
def hook(l):
    def f(mod, inp, out):               # out = (router_logits, topk_weights, topk_indices)
        cur[l] = (out[2].to(torch.int16).cpu(), out[1].float().cpu())
    return f
for name, m in model.named_modules():   # layers 3..44 are MoE; layers 0-2 are dense (no router)
    if type(m).__name__ == "Glm5NextTextTopkRouter":
        m.register_forward_hook(hook(int(name.split("layers.")[1].split(".")[0])))
def docs(path):
    for line in open(path):
        yield json.loads(line)["text"]
for path in sorted(glob.glob(f"{DOM_DIR}/*.jsonl")):
    dom = path.split("/")[-1][:-6]
    cnt = np.zeros((NL, NE), np.int64); mass = np.zeros((NL, NE), np.float64)
    trace, ntok = [], 0
    buf = []
    for t in docs(path):
        buf += tok(t)["input_ids"] + [tok.eos_token_id or 154820]
        while len(buf) >= SEQ and ntok < MAXTOK:
            ids = torch.tensor([buf[:SEQ]], device="cuda:0"); buf = buf[SEQ:]
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)       # text only, no vision
            seq = np.zeros((SEQ, NL, K), np.int16)
            for l, (idx, w) in cur.items():
                for j in range(K):
                    np.add.at(cnt[l], idx[:, j].numpy(), 1)
                    np.add.at(mass[l], idx[:, j].numpy(), w[:, j].numpy())
                seq[:, l] = idx.numpy()
            trace.append(seq); ntok += SEQ; cur.clear()
        if ntok >= MAXTOK: break
    np.savez_compressed(f"{OUT}/{dom}.npz", counts=cnt, mass=mass, tokens=ntok)
    np.save(f"{OUT}/{dom}.trace.npy", np.concatenate(trace))  # [tokens, 45, 8] int16 (layers 0-2 stay 0)
    print(dom, ntok, "tokens", flush=True)
# offline (numpy, no GPU): flatten counts to 42*288 pairs, sort desc, hit(K) = cumsum/total for the
# static profile; evaluate domain A's top-K on domain B's counts; replay *.trace.npy through an LRU or
# swap-every-N cache to get the adaptive hit rate.
```

**Calibration corpora**: 6 domains x ~300k tokens each: code (multi-language), agentic/tool-use transcripts, math/reasoning (thinking-style), general English chat/web, Chinese, long-context documents; plus ~100k tokens of the model's own generated thinking. A sanity run first: 50k tokens, check that counts sum to tokens x 8 per layer and that the 42 MoE layers all fired. Output size: 2 M tokens x 45 x 8 x 2 B = 1.4 GB traces.

## 6. Engine has / engine needs

Engine state read from source: CUDA-only (sm_80..120 builds), dense Qwen3.5 only (`qwen35.rs` explicitly refuses `Qwen3_5Moe*` "which the engine runs after the dense Qwen3.5 models"), MTP and vision not loaded, lowest weight layout **Int5 = 5.53 bpw all-in** (`fit.rs`); `sparrow`, `wren`, `hummingbird` (<=5/4/3 bpw) are **refused as not built** (CHANGELOG); packer reads bf16 safetensors shards (`pack.rs`, `weights.rs`).

| Piece | Engine today | Needs | Size |
|---|---|---|---|
| Gated DeltaNet recurrent + chunked kernels, conv state, delta ref (`Kern35::delta`, `qwen35_ref.rs`, `kcheck35.rs`) | yes (scalar gate per head) | **KDA**: per-key-channel gate with low-rank f/g, separate q/k/v convs, sigmoid-gated o_norm, 128x128 fp32 state x64 heads; reference + kernel check | **M** (adapt GDN kernels; port from FLA/vLLM Triton as spec) |
| Full attention (gated, rope, GQA, paged 12-bit KV) | yes | **MLA decode** with absorbed `kv_b`, NoPE, latent KV cache (512/token) | **M** |
| Sparse indexer, kpool compress, top-2048+tail gather, shared-top-k option | no | indexer scoring kernel (32 heads, relu, weighted sum), pool compress, top-k over pools, sparse MLA attend over <=2051 positions, cache for indexer keys; prefill version | **L** (the most novel kernel work) |
| mHC (4-stream residual, Sinkhorn x20, fp32) | no | fused mixer kernel before/after each sublayer, final stream mean | **S-M** |
| MoE router (sigmoid, bias-select, norm, x2.5) and grouped expert products on packed weights | no (dense only) | router kernel; top-8 gather; grouped GEMV (decode) and grouped GEMM (prefill) over per-expert packed blobs; clamped SwiGLU; shared expert; weighted combine | **L** |
| Sub-5-bit calibrated layout (3-4 bpw) and its GPU decode | Int5/Int6 only; lower levels refused | new block-code layout + packer + kernels + calibration (Hessian/importance-aware, per-expert bits) | **L** (the quality-critical piece) |
| Expert offload cache + async prefetch over PCIe (pinned host memory, copy stream, slot table, adaptive swaps, profile file) | no | pinned arena of per-expert blobs, GPU slot pool, residency table, miss list from router, CUDA stream copies, event ordering with the compute stream, profile loader/saver | **L** |
| CPU expert compute for misses (AVX-512/AVX2 GEMV on Glyd's packed format, thread pool, GPU/CPU hand-off per layer) | no | needed to beat PCIe-stream speeds (section 4); Strata does this with ggml kernels | **L** (can start with PCIe-only path = M) |
| FP8-source packing: read F8_E4M3 + 128x128 `weight_scale_inv`, dequantise to bf16 (or directly to f32) then pack; per-expert tensors; stream 328 GB shard by shard, mixed bf16/fp8 checkpoint | packer reads bf16 | add FP8 block-dequant reader (exact: fp8 value x f32 scale), keep a bf16 reference path for quality comparison; ~0.3 GB working set per expert | **S** |
| MTP drafting (eh_proj, enorm/hnorm, 1 sparse-MLA + MoE layer looped, shared indexer, verify of k tokens with per-token KDA state rollback) | no (head not loaded) | whole path plus multi-token verify through KDA/indexer | **L** |
| Tokenizer/chat template (thinking always on, reasoning_effort, tool format glm47), vision skip | partial for Qwen | template + parser | **S** |
| Quality harness (reference = bf16-dequantised FP8, KL/PPL, agentic evals) | exists for 27B | extend to a 328 GB reference on a multi-GPU pod | **M** |

Rough total: roughly 3 L + 2 M + several S before a first token, and a further L for CPU compute and L for MTP. This is a multi-month engine effort, not a variant of the Gemma/Qwen work.

## 7. Strata's approach and what Glyd would do differently

**Strata** (MIT; C++/CUDA/HIP on top of ggml pieces) [doc/files]: runs Qwen3.8-Flash-Next (48 layers x 512 experts, top-10, 24,576 experts; 34-50 GB of experts at 2-3 bpw) on 12-24 GB cards with 32-64 GB RAM.
- **Placement**: GPU holds attention/DeltaNet, routers, shared experts, head, MTP layer, KV, and an *expert cache* filling the rest of VRAM; RAM holds **all** experts, pinned; SSD holds a 28.8 GB n-gram table (and, in low-RAM mode, the experts as a mmap'd `experts.bin`).
- **Hot-expert choice**: a shipped static profile (`data/expert-profile.bin`, "STRP" format: a ranked list of all (layer, expert) pairs, built by `tools/make_profile.py` from routing traces dumped with `--dump-routing`) fills the cache at start; an adaptive tier (`--adapt-every/--adapt-swaps/--adapt-decay`) counts routing with decay and swaps hot experts in (a swap copies the evicted expert back through VRAM, per window); the learned ranking can be saved across restarts (`expert_profile_save`). Ranking is global across layers, not per-layer quotas.
- **Misses**: computed **by the CPU in place** (AVX-512/AVX2, ggml i-quant kernels) in parallel with the GPU; `--pcie-frac` additionally sends a share of misses over PCIe to be computed on the GPU (tuned per machine by `--calibrate`). Experts of the next layer stream during prefill (layerwise, chunks up to 8192) while the current layer's attention runs. Pinned arena, GPU copy kernels instead of DMA after a driver stall (issue #31). Elastic VRAM cache segments (`/v1/vram`).
- **Speed** [doc]: 53-94 tok/s on an RTX 5070 12 GB/Ryzen 5 7600/64 GB (IQ3_S...Q2_0), estimated 100-140 on a 3090; a 32 GB card with 45% of experts resident: 102-122 tok/s at 90-95% hit. Decode uses the model's MTP (2.4-3.2 tokens/pass) plus prompt-lookup drafting.
- **Quality**: no calibration of its own; it consumes community GGUFs (ISTA-DASLab GSQ-RCO, Unsloth UD-IQ4_XS/Q4_K_XL, UkisAI fine-tune), "Coder" prunes half the experts. 2-3 bpw with i-quants.
- **Prior art for GLM itself**: KTransformers supports GLM-5.3-Flash natively on RTX 40/50 with AVX-512 FP8 CPU experts but needs the **full 306 GiB FP8 in RAM (350 GB)** and zero resident GPU experts during layerwise prefill [doc]. That is the high-RAM alternative; our niche would be 128 GB.

**What Glyd would do differently**:
1. **Bits by usage, not uniform.** Cold experts (the ~85% that sit in RAM and are read rarely) at ~2.5-3 bpw; hot experts in VRAM at 4-5.5 bpw; the always-on 8 B at 5.5-6.5 (they are 2.5% of bytes). That is the only way to get "3.5-4 effective" under a 128 GB pool: e.g. 13% of experts at 5.5 + 87% at 3.0 = average 3.3 bpw, expert pool ~113 GB total. The routing trace in section 5 gives the weights for this directly (mass per expert).
2. **Calibration**: Hessian/activation-aware quantisation using the same routing traces (per-expert input statistics), with the bf16-dequantised FP8 as the reference and KL-divergence gates per tensor group, as in `research/lossy-2026`; Strata inherits whatever GGUF makers did.
3. **One packed format both sides**: a GPU-decodable block code that the CPU kernel also reads, so an expert moving between RAM and VRAM is a memcpy (Strata converts nothing too, but its CPU/GPU formats are ggml's).
4. **Verified drafts only**: keep outputs equal to the target's greedy/sampled distribution (the engine's current guarantee), with MTP gated by a measured net win.
5. **Honest accounting for RAM**: low-RAM resident mode by default at 128 GB (no duplicate copy).

## 8. Recommendation

**NO-GO as a release target now; GO for a ~$40 measurement spike; revisit after.**

Why not go:
- *Memory is tight even before engine work*: 3.5 bpw uniform = 140.6 GB total; with a 24 GB card the RAM pool is 119 GB of 128 GB; 4.0 bpw does not fit at all. Feasible only at ~3.0-3.3 bpw average, i.e. mixed precision with sub-4-bit cold experts, a regime where Glyd has **no quality evidence and no layout built** (only Int5/Int6 exist).
- *Engine scope*: MoE, KDA, MLA+DSA indexer, mHC, offload cache, CPU expert kernels, MTP are all missing; the model is architecturally further from Qwen3.5 than Gemma 4 is.
- *Competition*: KTransformers already serves this model on 4090/5090 (with 350 GB RAM), vLLM/SGLang and an NVFP4 build serve it on servers; Strata proves the approach for a 125B model that needs 4-5x less memory.
- *Skew risk*: noaux_tc balancing may flatten usage; 288 large experts are coarser than Strata's 512 small ones. At 13% cache the hit rate may be only ~50-65%.

Why the upside is real: active params are small (16.75B), the always-on part is only ~4 GB, KV is tiny (16.9 KB/token, KDA state constant), and the bandwidth model says even a zero-skew CPU path gives ~12 tok/s and a 60-65% hit gives ~14 tok/s over plain PCIe 4.0 (~33 with CPU compute), versus 10 tok/s as a usability threshold.

**Main risks**: (1) skew weaker than the Qwen analogue; (2) CPU-path bandwidth (45 GB/s) is assumed, not measured; (3) sub-4-bit quality of a 3-bit-expert 321B model is unproven in Glyd; (4) RAM headroom at 128 GB; (5) engineering size (months); (6) mHC/KDA/indexer bit-compatibility with the reference (new ops, fp32 sensitive; HF code omits MTP so MTP has only vLLM as a reference); (7) the model is thinking-always-on, so long outputs make decode speed, not prefill, the product.

**Cheapest next experiment** (decides everything else, ~$25-40, ~3 h): run the section 5 script on 4 x RTX PRO 6000 96 GB (or 4 x H100 NVL), 6 domains x 300k tokens, then compute offline: (a) static top-K hit rate at K = 853 / 1,580 / 2,306 experts (16/24/32 GB @3.5 bpw) per domain; (b) cross-domain transfer of a shipped profile; (c) adaptive-cache hit rate on sequential traces; (d) routing-mass concentration to size a mixed-bit plan. Decision rule: if static top-1,580 transfers at >= 70% hit, proceed to a PCIe-only MoE prototype plan (M-size milestones: FP8 packer, router + grouped GEMV on Int5, KDA); if < 55%, stop. A free precursor: confirm with a single small download (one shard, ~5 GB) the FP8 dequant convention and per-expert blob sizes for the packer.

## 9. Engine status, 2026-10-11 (GLY-133; measured on the RTX 4080 SUPER of the owner's PC)

Section 6 is the plan; this is what is built. Everything below is in `glyd-engine` (`src/model_glm.rs`, `src/mla_glm.rs`, `src/moe_glm.rs`, `src/mtp_glm.rs`, `kernels/glm.cu`) and is checked against `tools/glm_ref.py` (the reference, itself checked against transformers 5.19.0's modules and model) by `tools/cards/glm_pc_check.sh`.

| Piece | State |
|---|---|
| KDA (chunked and recurrent), conv state, gated norm | done; delta rule within 2.5e-3 of the largest output at 64 heads and 2,048 rows, state 1e-6 |
| Sparse MLA + indexer (absorbed form, no RoPE, latent cache of 512 + the indexer's keys and gate scores) | done; the indexer picks the reference's tokens for 2600/2600 rows; attention within 1e-5 of the largest value; a real layer at 2,300 rows passes; every launch reads positions from the device, so a decode step is one CUDA graph |
| mHC (4 streams, 20 Sinkhorn rounds), split projection for up to 8 rows | done; 6e-6 |
| Router, clamped SwiGLU, shared expert (the core MoE layer) | done at Int6, Int5, Int4 block layouts, all experts resident |
| FP8 source (128x128 block scales) | done, bit for bit with transformers' `Fp8Dequantize` |
| Whole tiny model (the real layer pattern in 8 layers) | KL 1.4e-4 nats a position against the reference with the engine's router and indexer choices run in it; top-1 100% |
| Draft verification (k+1 rows in one pass; the Kimi delta states rewound to the kept rows) | done; greedy-identical for any draft (tested with 8 schedules incl. wrong guesses at every place) |
| MTP head (layer 45) | done: `enorm`/`hnorm` of the embedding and the previous hidden state side by side (embedding first) into `eh_proj`, then one latent-attention + MoE layer **on a plain residual stream (the layer has no `hc_*` tensors)**, `shared_head.norm` fused with the last add, the model's LM head. On the real weights, with the CPU reference: its draft is the model's own next pick for 64% of 126 positions (0% with the halves swapped). Needs the layer's own 288 experts (5.65 GiB at Int6) |

**Decode, measured** (real layer shapes, 16 experts a layer all resident, position 2,600, one CUDA graph): 0.73-0.84 ms a layer (KDA + experts 0.73; latent attention + experts 0.82); the stages: experts 0.31 ms, KDA q/k/v 0.21, o product 0.07-0.14, latent attention (indexer, select, attention, values) 0.14, query 0.075, mHC 0.06. 45 layers are about 36 ms, so about 27 tok/s with every expert in VRAM, which the card cannot hold. The products run at about half the card's 736 GB/s: they are the library's small-M kernels, not the new ones. Weights read a token (always-on 7.5 GB + 8 routed experts a layer, 21 MB each at 6 bits): 14 GB at 6 bits, 7.7 GB at 3 bits (about 95 tok/s at full bandwidth, with every byte resident).

**Left** (in the order they block the owner's target of GLM on 16 GB + 60 GiB): (1) the expert cache for GLM: `moe_nh.rs`' `NhMoeCached` and `moe_hb.rs`' `NhMoeHb` are the templates (`ExpertStore`, copies, the host's half, the SSD tier); the block layouts of `moe_glm.rs` are Int6/5/4 only, and the codebook layout the weights work settled on is `moe_hb.rs`, which is written against `ConfigNemotronH` (latent width, signs of the rotation): it needs the dimensions of an MoE layer abstracted (GLM's experts read the 4,096-wide input, no latent projections); (2) the real model on the engine end to end (`glmreal`: the per-layer chain against the reference of research/glm-data), the real MTP head on the engine (`glmmtpreal`); (3) the verify pass inside the CUDA graph and acceptance for sampled decoding; (4) a decode kernel for the library's M = 1 products (twice the card's bandwidth floor would be the gain for everything but the experts); (5) lower-bit layouts (GLY-134).

## Appendix: reproduction

- Tensor table: shard headers via `Range: bytes=0-7` then `8..8+n` on each `model-000NN-of-00062.safetensors` at the pinned sha; grouped by name pattern; totals match the API's `safetensors` field.
- Model script (formulas in section 4, constants in section 3): scratchpad `/private/tmp/claude-501/-Users-suryakoritala/d09db505-88bb-4450-a82f-95fd523dc0ee/scratchpad/model.py`.
