# Nemotron 3 Super 120B-A12B on one consumer GPU (GLY-127, fallback target)

Read-only research; nothing committed, no engine code touched, no pods, no token.

Legend: **[read]** = from a file (config, safetensors headers, README, license, modeling code). **[calc]** = computed from read numbers (scripts in the session scratchpad: `meta.py`, `model.py`). **[assumed]** = my assumption, to be measured. Sizes are decimal GB (1e9 bytes) unless GiB is written.

## Verdict (short)

Feasible on paper, and unusually friendly to offload: only 4.3% of the routed experts fire per token and each is tiny (2.75 MB at 4 bpw). At 4 bpw the whole model is 61.8 GB [calc], so plan (a)/(b) need no SSD at all. Cost is engine work: Mamba-2, latent MoE and an expert cache are all new (the engine has none of them; it is dense-only Qwen3.5 with everything resident on the GPU). Go for the cheap experiment first (routing trace for hit rate), not for the engine work. See section 7.

## Sources and pins

- HF repo `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`, sha `2dc98e2afe4face0e4ce40972a915c45368bd34a`, `gated: false` [read]. Files read: `config.json`, `generation_config.json`, `README.md`, `model.safetensors.index.json` (42,683 tensors, 50 shards) and every shard header via `HfApi().get_safetensors_metadata` in a scratch venv, `chat_template.jinja`.
- Modeling code: transformers 5.19.0 wheel (`pip download --no-deps`), `models/nemotron_h/{modeling,configuration}_nemotron_h.py`. I did **not** read vLLM's nemotron_h file, so the MTP forward path is unread (transformers ignores `mtp.*`: `_keys_to_ignore_on_load_unexpected = [r"mtp.*"]` [read]).
- License: the HF repo has **no LICENSE or NOTICE file** (`resolve/main/LICENSE` is "Entry not found") [read]; README front matter says `license: other`, `nvidia-nemotron-open-model-license` and links the NVIDIA page, which I fetched and read in full (sections 1-10, "v. December 15, 2025").
- Repo extras: `bias.md`, `explainability.md`, `privacy.md`, `safety.md`, `super_v3_reasoning_parser.py` (a custom parser is required by all three servers) [read].

## 1. Architecture

### Top level [read: config.json]

| | |
|---|---|
| class | `NemotronHForCausalLM`, `model_type` nemotron_h, dtype bfloat16 |
| layers | 88, hidden 4096, vocab 131072, untied embeddings and head |
| context | max_position_embeddings 262144 (README: up to 1M with env flag) |
| norm | RMSNorm eps 1e-5 (`layer_norm_epsilon`), pre-norm: each layer is `x + mixer(norm(x))`, one mixer per layer (`backbone.layers.N.norm.weight`) |
| MLP act | `relu2` (relu then square), no gating anywhere |
| generation | temperature 1.0, top_p 0.95, eos [2, 11], bos 1, pad 0 [read: generation_config + README] |
| chat template | ChatML (`<|im_start|>`), `enable_thinking` (default True), `low_effort`, `truncate_history_thinking`, tool-call XML renderer [read] |

### Layer pattern [read: `hybrid_override_pattern`; counts [calc]]

```
MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME
```

M = Mamba-2 mixer, E = latent MoE mixer, * = attention. 88 layers = **40 M + 40 E + 8 attention**. Attention layers at indices 7, 16, 25, 36, 47, 58, 69, 78 [calc]. There is no dense MLP layer (`intermediate_size` 2688 is unused by any tensor in the index; MoE layers carry the FFN). Layers alternate M, E except where `*` is inserted in place of an M.

### Mamba-2 layer (40x) [read: config + tensor shapes + modeling code]

- `mamba_num_heads` 128, `mamba_head_dim` 64 -> inner dim 8192 (expand 2); `ssm_state_size` 128; `n_groups` 8 (B and C are shared by 16 heads each); `conv_kernel` 4 (conv bias present, `use_conv_bias` true); `chunk_size` 128; act silu.
- `in_proj.weight` [18560, 4096] = 8192 (gate z) + 10240 (conv channels = 8192 x + 2*8*128 B,C) + 128 (dt). Split order in code: `[intermediate_size, conv_dim, num_heads]`.
- `conv1d.weight` [10240, 1, 4] + bias [10240] over x,B,C; `A_log`, `D`, `dt_bias` [128] each; gated grouped RMSNorm `norm.weight` [8192] (group size 8192/8 = 1024); `out_proj.weight` [4096, 8192].
- State per layer [calc]: SSM 128 x 64 x 128 floats; `mamba_ssm_cache_dtype` is float32 -> 4 MiB/layer, **160 MiB for 40 layers**; conv state 3 x 10240 x 2 B = 61 KB/layer (2.5 MB total). Constant in context length. Decode reads and writes the SSM state each token: ~0.34 GB VRAM traffic per token (fp32, calc), about +5% on the 4 bpw token budget in section 5 (not included there).

### Attention layer (8x) [read]

- q_proj [4096, 4096] = 32 heads x 128; k_proj, v_proj [256, 4096] = **2 KV heads** x 128 (GQA 16:1); o_proj [4096, 4096]; no bias; `sliding_window` null (full attention); scaling head_dim^-0.5.
- **No positional encoding is applied** in `NemotronHAttention.forward` (no rotary call in the file; `rope_theta` 10000 and `partial_rotary_factor` 1.0 sit in the config but the modeling code never uses them) [read]. The Mamba layers carry position. So the engine's rotary `prep` path is not needed. Verify against vLLM before relying on this.
- KV cache [calc]: 2 (K,V) x 2 heads x 128 x 2 B x 8 layers = 8 KiB per token in bf16 -> 2 GiB at 256k, 8 GiB at 1M (fp8 halves; README serves with `--kv-cache-dtype fp8`). This is tiny next to Qwen-class dense models, so long context is cheap here.

### Latent MoE layer (40x) [read: modeling code + shapes]

```
router:   logits = x(4096) @ gate.weight[512, 4096]^T  (fp32)
          s = sigmoid(logits); choose top-22 of (s + e_score_correction_bias[512] fp32)
          (n_group=1, topk_group=1: the group stage is a no-op)
          w = s[top22]; w /= sum(w) (norm_topk_prob true); w *= routed_scaling_factor 5.0
routed:   z = fc1_latent_proj(x)       # [1024, 4096]   4096 -> 1024
          for each chosen expert i:    # up [2688, 1024], down [1024, 2688]
              y_i = down_i( relu2( up_i(z) ) )          # non-gated, all in the 1024 latent
          y = sum_i w_i * y_i
          out_routed = fc2_latent_proj(y)               # [4096, 1024]   1024 -> 4096
shared:   out_shared = shared.down( relu2( shared.up(x) ) )   # up [5376, 4096], down [4096, 5376], no gate, takes the full-width x
out = out_routed + out_shared   (then the residual add)
```

- The router sees the **full** 4096-d x, so the expert choice is known before the latent down-projection; the experts then work on a 1024-d vector (2 KB in bf16).
- One expert = up + down = 2 x 2688 x 1024 = **5,505,024 params** [calc] (11.0 MB bf16). Stored as separate per-expert tensors `experts.N.up_proj.weight` / `down_proj.weight` (20,480 + 20,480 tensors), not stacked 3D tensors, in the checkpoint [read]. (Transformers 5 stacks them in memory; the checkpoint loader converts.)
- Dims for block quantisation [calc]: 1024 = 8 x 128, 2688 = 21 x 128 = 10.5 x 256. A block size of 128 divides every expert dimension; a 256 super-block does **not** divide 2688 (check what block size Glyd's Int5/Int6 packs use in `pack.rs`).
- Norm_topk plus scaling 5.0 means the weighted sum has gain 5; keep fp32 for the router and the weights.

### MTP [read: shapes, config, README]

`num_nextn_predict_layers` 1, `mtp_hybrid_override_pattern` "*E": `mtp.layers.0` is attention (+ `eh_proj` [4096, 8192], `enorm`, `hnorm`, `norm`), `mtp.layers.1` is a full latent MoE with **all 512 experts** (+ `final_layernorm`). Total 2.94B params [calc], 5.9 GB in bf16. README: shared-weight MTP head design; an updated "MTPv2" head is a separate checkpoint `nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2` (not read). Reference code for the MTP forward is not in transformers.

### Embeddings, head, tensor names [read]

`backbone.embeddings.weight` [131072, 4096], `lm_head.weight` [131072, 4096], `backbone.norm_f.weight`. Layer names: `backbone.layers.N.norm.weight`, `backbone.layers.N.mixer.{in_proj,out_proj,conv1d,A_log,D,dt_bias,norm}` (M), `...mixer.{q,k,v,o}_proj.weight` (*), `...mixer.{gate.weight, gate.e_score_correction_bias, fc1_latent_proj, fc2_latent_proj, shared_experts.{up,down}_proj, experts.E.{up,down}_proj}.weight` (E). Everything is BF16 except `e_score_correction_bias` (F32, 20,992 F32 values in total).

### Other facts [read: README]

Trained at NVFP4 (weights, activations, gradients for most linears; latent projections, MTP, QKV/attention projections, embeddings kept BF16/MXFP8). An official `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` checkpoint exists (named in README for "single B200 or DGX Spark"; not inspected). Practical meaning: the experts were trained to live at 4-bit, so 4 bpw expert quantisation starts from a favourable point. Servers: vLLM 0.18.1, SGLang nightly, TRT-LLM 1.3.0rc8, all shown with TP 8 (no single-consumer-GPU path from NVIDIA).

## 2. Parameter accounting (from real tensor shapes [calc over the 42,683 headers])

Total 123,611,033,088 params (safetensors metadata says 123,611,012,096 BF16 + 20,992 F32; the same number) -> README "120B" is the backbone without MTP.

| group | params | share |
|---|---|---|
| routed experts, backbone (40 x 512 x 5,505,024) | 112,742,891,520 | 91.2% |
| Mamba in/out proj (40 layers) | 4,383,047,680 | 3.5% |
| shared experts (40 x 2 x 5376 x 4096) | 1,761,607,680 | 1.4% |
| embeddings | 536,870,912 | 0.4% |
| lm_head | 536,870,912 | 0.4% |
| latent projections (40 x 2 x 1024 x 4096) | 335,544,320 | 0.3% |
| attention q/k/v/o (8 layers) | 285,212,672 | 0.2% |
| routers (40 x 512 x 4096) | 83,906,560 | 0.07% |
| Mamba small (conv, A, D, dt, norm) + RMSNorms | 2,755,584 | ~0 |
| MTP experts (512 x 5,505,024) | 2,818,572,288 | |
| MTP other (attn, eh_proj, router, latent, shared) | 123,752,960 | |
| **MTP total** | 2,942,325,248 | 2.4% |

Backbone without MTP = 120.67B. **Always-on** (everything but routed experts, MTP and the embedding table; the embedding is a row lookup) = **7,388,946,432 params (7.39B)**, which includes lm_head.
**Active per token** = always-on 7.389B + 22 x 40 x 5.505M = 4.844B routed = **12.23B** (plus a 4096-wide embedding row). "12B active" confirmed.

Tiering consequence: 91.2% of bytes are in 20,480 experts that are each used with probability 22/512 = 4.3% per token; the 7.39B always-on part is touched on every token and belongs in VRAM.

## 3. Memory by bits per weight (decimal GB) [calc]

16 = bf16 as shipped. 11.2 = lossless, **assumed** 30% saving (as Glyd measured on Qwen bf16 [assumed to transfer; not measured on this model]). 6.5 / 5.5 / 4.0 / 3.0 = calibrated bpw as given. Every row applies the bpw uniformly to all tensors (Glyd keeps some tensors dense in practice, e.g. router, A_log, norms, so real totals are slightly higher; the router and small tensors are 0.07% of params).

| bpw | everything (backbone + MTP) | all routed experts | always-on (no embed, VRAM) | embed table | MTP | one expert (MB) | 22-expert read per MoE layer (MB) | per token: always-on + 880 expert reads (GB) |
|---|---|---|---|---|---|---|---|---|
| 16 | 247.2 | 225.5 | 14.78 | 1.07 | 5.88 | 11.01 | 242 | 24.47 |
| 11.2 (assumed) | 173.1 | 157.8 | 10.34 | 0.75 | 4.12 | 7.71 | 170 | 17.13 |
| 6.5 | 100.4 | 91.6 | 6.00 | 0.44 | 2.39 | 4.47 | 98 | 9.94 |
| 5.5 | 85.0 | 77.5 | 5.08 | 0.37 | 2.02 | 3.78 | 83 | 8.41 |
| 4.0 | 61.8 | 56.4 | 3.69 | 0.27 | 1.47 | 2.75 | 61 | 6.12 |
| 3.0 | 46.4 | 42.3 | 2.77 | 0.20 | 1.10 | 2.06 | 45 | 4.59 |

(880 = 22 experts x 40 MoE layers.) "Everything" includes MTP and the embedding table; MTP can be dropped in v1 (5.9 GB bf16, 1.5 GB at 4 bpw).

At 4 bpw everything is 61.8 GB: it fits 64 GB RAM only barely once the OS is counted, which is why plan (a) keeps ~18 GB of it in VRAM.

## 4. Placement plans

Assumptions [assumed]: VRAM reserve 2.5 GB (CUDA context, workspaces, 160 MiB SSM state, KV 8 KiB/token so ~0.5 GB at 64k, graph buffers); RAM reserve 8 GB (OS, pinned staging, the process); embedding table (row lookup) in RAM; MTP not loaded; hot = whatever VRAM is left after always-on; warm = fills the RAM budget; cold = the rest on SSD. Hot/warm/cold is by expert popularity, chosen from a routing trace. GB.

| plan | bpw | VRAM always-on | VRAM hot experts | RAM warm experts | SSD cold experts | hot % of experts | warm % | cold % |
|---|---|---|---|---|---|---|---|---|
| (a) 24 GB + 64 GB | 11.2 | 10.3 | 11.2 | 55.2 | 91.4 | 7 | 35 | 58 |
| (a) 24 GB + 64 GB | 6.5 | 6.0 | 15.5 | 55.6 | 20.5 | 17 | 61 | 22 |
| (a) 24 GB + 64 GB | 5.5 | 5.1 | 16.4 | 55.6 | 5.5 | 21 | 72 | 7 |
| **(a) 24 GB + 64 GB** | **4.0** | 3.7 | 17.8 | 38.6 | 0.0 | **32** | 68 | 0 |
| (a) 24 GB + 64 GB | 3.0 | 2.8 | 18.7 | 23.5 | 0.0 | 44 | 56 | 0 |
| (b) 16 GB + 64 GB + SSD | 11.2 | 10.3 | 3.2 | 55.2 | 99.4 | 2 | 35 | 63 |
| (b) 16 GB + 64 GB + SSD | 6.5 | 6.0 | 7.5 | 55.6 | 28.5 | 8 | 61 | 31 |
| (b) 16 GB + 64 GB + SSD | 5.5 | 5.1 | 8.4 | 55.6 | 13.5 | 11 | 72 | 17 |
| **(b) 16 GB + 64 GB + SSD** | **4.0** | 3.7 | 9.8 | 46.6 | 0.0 | **17** | 83 | 0 |
| (b) 16 GB + 64 GB + SSD | 3.0 | 2.8 | 10.7 | 31.5 | 0.0 | 25 | 75 | 0 |
| (c) 32 GB + 96 GB | 11.2 | 10.3 | 19.2 | 87.2 | 51.4 | 12 | 55 | 33 |
| (c) 32 GB + 96 GB | 6.5 | 6.0 | 23.5 | 68.1 | 0.0 | 26 | 74 | 0 |
| **(c) 32 GB + 96 GB** | **5.5** | 5.1 | 24.4 | 53.1 | 0.0 | **32** | 68 | 0 |
| (c) 32 GB + 96 GB | 4.0 | 3.7 | 25.8 | 30.6 | 0.0 | 46 | 54 | 0 |
| (c) 32 GB + 96 GB | 3.0 | 2.8 | 26.7 | 15.5 | 0.0 | 63 | 37 | 0 |

Reading it:
- 4 bpw needs no SSD on any of the three machines; the SSD only matters for 5.5+ bpw on 16/24 GB cards and for the lossless 11.2 bpw tier (plans (a)/(b): 58-63% of experts, 91-99 GB, cold; plan (c) 51 GB cold).
- 16 GB card: at 4 bpw the always-on part (3.7 GB) leaves 9.8 GB, a 17% hot set. The lossless 11.2 bpw always-on part (10.3 GB) already eats most of a 16 GB card, so lossless on 16 GB is a poor fit (2% hot).
- For a 16 GB card the always-on part is better held at higher precision than the experts (it is 3.7 GB at 4 bpw but 6.0 GB at 6.5 bpw): a mixed plan (always-on 6.5 bpw, experts 4 bpw) costs +2.3 GB of VRAM, which is 840 fewer hot experts. Worth it only if quality needs it.
- Context: KV is 8 KiB/token (bf16), so 256k tokens is 2 GiB: it fits inside the reserve on 24/32 GB cards only up to roughly 128-256k; on 16 GB, use fp8 KV or shorter context.

## 5. Decode bandwidth model (batch 1, one token per step)

Bytes per token [calc]: always-on part A from VRAM, plus 880 expert reads of s bytes (s = 2.75 MB at 4 bpw, 3.78 MB at 5.5 bpw). With hit rate h (fraction of the 880 reads served from VRAM):
`t = (A + h*880*s)/BW_vram + (1-h)*880*s/BW_link` [serial: the conservative form; with perfect overlap of copy and compute t = max(...) of the two terms].

Inputs [given by the task, not measured]: VRAM 1000 GB/s (4090), 1800 (5090), 450 (4060 Ti 16 GB); PCIe 4 = 25 GB/s, PCIe 5 = 50 GB/s for misses served from RAM; SSD Gen5 = 12 GB/s. Note [assumed, from general knowledge]: a 4060 Ti 16 GB is a x8 Gen4 card, so ~16 GB/s real; I left 25 for comparability, so its row is optimistic for misses.

Ceiling tok/s, **no kernel or launch overhead, no SSM state traffic, no router/Mamba compute**, all misses from RAM [calc]:

4.0 bpw (A = 3.69 GB, 880 reads = 2.42 GB):

| hit | 4090 + PCIe4 | 5090 + PCIe5 | 4060 Ti 16 + PCIe4 |
|---|---|---|---|
| 0% | 9.9 | 19.8 | 9.5 |
| 50% | 18.7 | 37.1 | 16.9 |
| 60% | 22.8 | 45.0 | 19.9 |
| 70% | 29.0 | 57.1 | 24.4 |
| 80% | 40.0 | 78.0 | 31.4 |
| 90% | 64.3 | 123.3 | 44.0 |
| 95% | 92.3 | 173.8 | 55.0 |
| 100% | 163.5 | 294.3 | 73.6 |

5.5 bpw (A = 5.08 GB, 880 reads = 3.33 GB):

| hit | 4090 + PCIe4 | 5090 + PCIe5 | 4060 Ti 16 + PCIe4 |
|---|---|---|---|
| 0% | 7.2 | 14.4 | 6.9 |
| 50% | 13.6 | 27.0 | 12.3 |
| 60% | 16.6 | 32.7 | 14.5 |
| 70% | 21.1 | 41.5 | 17.7 |
| 80% | 29.1 | 56.7 | 22.8 |
| 90% | 46.7 | 89.7 | 32.0 |
| 95% | 67.1 | 126.4 | 40.0 |
| 100% | 118.9 | 214.0 | 53.5 |

If misses come from the SSD instead (12 GB/s; 4.0 bpw) [calc]: hit 50% -> 9.4 / 9.6 / 8.9 tok/s; 80% -> 21.7 / 23.0 / 18.9; 90% -> 38.4 / 42.6 / 30.1; 95% -> 62 / 75 / 43 (4090 / 5090 / 4060 Ti). The SSD alone gives ~5 tok/s at 0% hit (2.42 GB / 12 GB/s = 0.2 s).

**What hit rate gives 10 and 20 tok/s** [calc; "effective" = ceiling x 0.6, where the 0.6 efficiency for launch gaps, Mamba/attention compute, SSM state traffic and imperfect copy overlap is **assumed**], misses from RAM, serial model:

| bpw | target | 4090 + PCIe4 | 5090 + PCIe5 | 4060 Ti + PCIe4 |
|---|---|---|---|---|
| 4.0 | 10 tok/s | 43% | 0% (PCIe5 alone suffices) | 49% |
| 4.0 | 20 tok/s | 75% | 44% | 82% |
| 5.5 | 10 tok/s | 60% | 15% | 67% |
| 5.5 | 20 tok/s | 83% | 61% | 91% |

With efficiency 1.0 (pure ceiling): 4 bpw 10 tok/s needs ~0-6% hit everywhere, 20 tok/s needs 54% (4090), 1% (5090), 60% (4060 Ti). With full copy/compute overlap the 0.6-efficiency hit rates drop by ~5 points (e.g. 4 bpw, 20 tok/s, 4090: 69%).

What hit rate to expect is the open question. The hot set is 32% (a), 17% (b), 32-46% (c) of all experts. Router is sigmoid with a load-balancing bias (aux-loss-free style `e_score_correction_bias`) over 512 experts with top-22, which should flatten the expert popularity distribution: a static "most popular" set may give little more hit rate than its capacity fraction; the extra comes from token-to-token and prompt-level locality (an LRU/LFU cache per layer). **[assumed] 40-60% for plan (a) at 4 bpw -> roughly 8-14 tok/s effective on a 4090 (ceiling 18.7-22.8 x 0.6)**. This number is the thing to measure (section 7).

Prefill (not decode): a chunk of a few hundred tokens touches almost every expert in every layer, so each chunk streams most of the 56 GB of 4 bpw experts: ~2.3 s per chunk at 25 GB/s (PCIe4), ~1.1 s at 50 GB/s, ~4.7 s if from SSD at 12 GB/s [calc]. Use big prefill chunks (2k+ tokens -> ~900 tok/s PCIe4 ceiling at 4 bpw, calc) and process layer by layer so each expert is loaded once per chunk.

Alternative to moving weights: compute miss experts on the CPU [assumed option]. The expert input is the 1024-d latent (2 KB), so shipping activations to the host and back costs nothing next to 2.75 MB of weights. Reading from DDR5 at an effective 60 GB/s [assumed] would give, at 4.0 bpw, 40 / 44 / 32 tok/s at 50% hit and 73 / 89 / 49 at 80% (4090 / 5090 / 4060 Ti) as a ceiling [calc], beating PCIe 4 transfers (25 GB/s) outright. Needs a CPU kernel for packed-int block-code matvec (AVX-512 / AMX), which is real work; this is the ktransformers-style design. Latent MoE is what makes it cheap.

### How latent MoE changes cache granularity

- Unit = one expert (up + down), 5.5M params: 11 MB bf16, 2.75 MB at 4 bpw, 3.8 MB at 5.5 bpw. That is 5-10x smaller than the experts of typical 100B-class MoEs (e.g. Qwen-class experts are ~tens of MB), so there are 20,480 cache entries instead of a few thousand, per-layer working set is 22 x 2.75 = 61 MB, and the whole hot set can be tracked per (layer, expert) in a flat table.
- Small units are good for cache efficiency (fine-grained popularity, no waste on rarely used neighbours) but each copy is only ~2.75-3.8 MB: still large enough for good PCIe throughput (>=1 MB), but per-copy launch overhead matters: batch the 22 misses of a layer into one gather, or use a single pinned staging buffer plus one `cudaMemcpyAsync` per expert on a copy stream. SSD reads of 2.75 MB need queue depth (22 outstanding per layer; the next MoE layer is only known one layer later), and Gen5 NVMe peaks need QD >= 16.
- Pack each expert as one contiguous blob (up then down) so a hit or miss is one range.
- The router needs only x (4096-d); the expert choice for MoE layer L is known after the preceding Mamba layer finishes, giving only one layer of lookahead (about 1 ms at 40 MoE layers per ~25 ms token) for prefetch. Speculative prefetch from layer L-1's choice (or from cross-token reuse) is a cache-policy question for the experiment, not given.
- Hot/warm/cold by popularity can be done per layer (each layer has its own 512): 40 independent pools, or one global pool ranked by hit count. A global pool needs the trace to say whether layers differ in skew.
- Always-on shared expert (5376 wide, applied to the full 4096-d x) is 44M params/layer: it sits in the always-on part (1.76B total), not in the cache.

## 6. Engine has / needs

Engine today [read: `glyd-engine/src/model35.rs`, `qwen35.rs`, `pack.rs`, `kernels/`]: Qwen3.5 dense only. `Weights35` is all GPU-resident (`Mat::Packed` Int5/Int6 block-code products, `Mat::Dense` for small ones); layer kinds are `Full` (qkv/o + rotary `prep`, `attn`, `attn_decode` with head_dim 256) and `Linear` (qkvz, ba, causal conv1d + SiLU, **gated delta rule** `gdn_chunk` (chunks of 64) / `gdn_recurrent`, gated norm, out); decode is one CUDA graph; there is already `Pinned` host memory and `copy_h2d_async` / `copy_d2h_async` in `dev.rs`. There is no MoE, no router, no expert cache, no storage reads in the decode loop. The engine hard-rejects configs other than head 256, linear heads 128/128, conv kernel 4 (`model35.rs` line ~514); conv kernel 4 matches here, the rest does not.

| piece | status | size | notes |
|---|---|---|---|
| Causal conv1d + SiLU, kernel 4, state | has (`conv_silu`) | S | here the conv covers 10240 channels (x,B,C) with bias; check bias and channel layout |
| Mamba-2 recurrent step (decode) | needs | M | selective-scan step: per head dt = softplus(dt + dt_bias), dA = exp(A*dt), state[h,64,128] update with B,C (8 groups), D skip, then gated grouped RMSNorm with z. Simpler than GDN (no delta rule), different tensors; fp32 state |
| Mamba-2 chunked scan / SSD (prefill) | needs | L | chunk_size 128; the engine's `gdn_chunk` is delta-rule specific and does not transfer. v1 fallback: run the recurrent step token by token for prefill (correct, slow) |
| Full attention, 2 KV heads, head_dim 128, no rotary, no window | partial | S-M | `attn` / `attn_decode` exist for head 256 and rotary `prep`; need head 128 and a no-RoPE path |
| Latent MoE: fc1 / fc2 latent projections | needs | S | two dense products (1024x4096, 4096x1024) with the existing packed-product path |
| Router: fp32 sigmoid, bias, top-22 of 512, renorm, x5.0 | needs | S | tiny kernel (512-wide top-k per token); group stage can be skipped (n_group = 1) |
| Grouped / gathered expert products on packed weights (relu2, non-gated, 1024 <-> 2688) | needs | M-L | batch-1 decode: 22 small matvecs per layer, so the product kernel must take an expert-pointer list in one launch (and a gather path for prefill chunks); block size must divide 2688 (128 does, 256 does not) |
| Shared expert (5376 wide, relu2, ungated) | has-ish | S | an ordinary dense MLP without gating (the engine's MLP is gate_up + down; use up only) |
| relu2 activation | needs | S | trivial |
| Expert offload cache: VRAM slots + pinned host pool + SSD tier, LRU/LFU, eviction | needs | L | pool of fixed-size expert slots (each ~2.75 MB at 4 bpw), residency table [layer][expert] -> slot, hit/miss counters. Exactly the part that does not exist |
| Prefetch / overlap: copy stream, events, pinned staging | partial | M | `Pinned` and async copies exist; a decode CUDA graph cannot contain data-dependent copies, so the MoE layer needs either graph segments per layer (route on GPU, read indices on the host, copy misses, launch) or a persistent kernel; a host sync per MoE layer (40/token) is the main overhead risk |
| SSD tier: plain `pread`/io_uring vs GPUDirect Storage | needs | M (plain), L (GDS) | plain reads into pinned memory at QD >= 16 is the first step; GDS needs a supported setup (a consumer-card path is not documented in anything I read; treat as unverified); at 4 bpw the SSD tier is not needed on 24/32 GB |
| CPU expert execution for misses | optional | L | the section 5 alternative, needs an AVX-512/AMX kernel for the packed format |
| Packer for this architecture (BF16 -> Int5/Int6 / calibrated 4.0 bpw) | partial | M | per-expert packs: 20,480 expert pairs; needs calibration data that reaches rarely used experts (imatrix-style calibration will see few tokens for cold experts at 22/512; known weak point of low-bit MoE) |
| Tokenizer, chat template (ChatML + thinking + tool XML), `super_v3` reasoning parser | partial | S-M | template renders via Jinja; the reasoning parser is a custom Python file shipped in the repo, not read |
| MTP / speculative decoding | needs | L | attention + 512-expert MoE head, `eh_proj` [4096, 8192]; speculation also unions more experts per step, hurting cache hit rate. Defer; skip loading MTP tensors (5.9 GB bf16) |
| Hybrid state manager | partial | S | SSM state (160 MiB fp32) + conv state per sequence plus 8 attention KV caches; the engine already has `ssm`/`convst` buffers and KV for Qwen3.5 linear/full layers |

Rough total: M-L for a correct all-resident pass if VRAM were large (Mamba-2 + latent MoE + router + grouped expert products), then L more for the offload cache and overlap. The Mamba-2 chunked scan is the single largest correctness-and-speed item; the expert cache is the largest design item.

## 7. Recommendation

### License (read in full, v. December 15, 2025)

Allows (section 2): perpetual, worldwide, royalty-free, irrevocable right to reproduce, prepare Derivative Works of, sublicense and distribute the Work and Derivative Works in source or object form; intro says "commercially usable" and "free to create and distribute Derivative Works"; NVIDIA claims no ownership of outputs. A quantised/re-packed checkpoint is a Derivative Work (the Work includes "checkpoints, learnt weights, parameters"). Conditions for redistribution (section 3): (a) give recipients a copy of the License; (b) retain copyright/patent/trademark/attribution notices in source form of derivatives; (c) if the Work includes a NOTICE file, include a NOTICE with "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License." (this repo has no NOTICE file, so (c) is moot today; adding the line anyway is cheap). You may add your own terms to your modifications (so Glyd-format weights can carry the Glyd licence as long as the Nemotron licence is also passed on). Trademarks (4): no NVIDIA marks except "reasonable and customary use in describing the origin" (so "derived from NVIDIA Nemotron ..." in a model card is fine, a repo named as if NVIDIA's is not). Patent/copyright suit against anyone alleging the Work or its outputs infringe ends your licence (2). Warranty/indemnity only if you offer support (7). Governed by Delaware law, Santa Clara courts (9); export compliance (10). **No acceptable-use list or other use restriction appears in this version.** Not legal advice; reread if NVIDIA updates it ("Last Modified: December 15, 2025").

### Go / no-go vs GLM-5.3-Flash

`research/glm53-flash-offload.md` did not exist when I looked (ls of `research/`), so I cannot compare numbers. What I can say for Nemotron, on its own merits:
- For: tiny experts, so very fine-grained caching; 92% of the weights are cold-friendly experts; 12.2B active params; at 4 bpw the entire model is 61.8 GB so even a 16 GB card + 64 GB RAM works with no SSD; KV is 8 KiB/token (long context cheap); native NVFP4 training means 4-bit starting point; permissive licence with no use restrictions found; ungated.
- Against: three new engine families (Mamba-2, latent MoE, expert cache), and the engine's current hybrid work (Gated DeltaNet) does not transfer to Mamba-2; weak low-bit calibration on rarely used experts; routing popularity is probably flat (load-balanced router) so hit rates may be near capacity fraction; per-layer host sync on every MoE layer; MTP needs separate work.
- Decision rule to compare with the GLM file: pick whichever model has the larger (hit-rate-independent) ceiling at the same machine and the smaller new-kernel list. Nemotron's ceiling for plan (a) at 4 bpw is 10 tok/s at 0% hit and 19-23 at 50-60% (4090 + PCIe4). Nemotron's new-kernel list is Mamba-2 scan (L), grouped experts (M-L), cache (L). If GLM needs MLA attention, grouped experts and a cache with larger experts, Nemotron is the smaller kernel set; if GLM is a standard transformer with RoPE attention (which the engine partly has), GLM is cheaper to start.

**Recommendation: GO for a measurement phase only (no engine work yet); NO-GO on engine work until the hit rate is measured.** If the measured hit rate at 32% capacity is below ~40%, drop to the CPU-exec design (section 5) or choose GLM.

### Main risks

1. Hit rate (flat routing) is unknown: 10 tok/s needs ~43% on a 4090 + PCIe4, 20 tok/s ~75% (0.6 efficiency assumed).
2. Host sync per MoE layer (40 per token) with a CUDA-graph decode design; each sync costs about tens of microseconds plus the copy wait.
3. Low-bit quality of rarely-hit experts (calibration coverage) and of the always-on part at 4 bpw.
4. Mamba-2 correctness (fp32 state, grouped norm, 8 groups); attention without RoPE contradicts the config's rope fields: verify with vLLM's model file (not read).
5. SSD tier: GPUDirect on a consumer GPU unverified; real Gen5 SSD throughput at 2.75 MB random reads below 12 GB/s sequential.
6. Download size and bandwidth: 247 GB BF16 to pack.

### Cheapest next experiment (no pods needed for step 1; a rented 80 GB pod is the alternative for step 2)

1. Run the reference model in transformers/vLLM (needs 247 GB, so a multi-GPU or CPU+disk run via `device_map="auto"`, a few tokens/s is fine) on ~50k tokens of chat/code/reasoning text with a hook on `NemotronHTopkRouter` that logs `topk_indices` per MoE layer per token.
2. Replay offline: simulate per-layer LRU / global LFU caches at capacity 17%, 32%, 46% of the 20,480 experts and report the hit rate, plus the hit rate of a static popularity set trained on one half and tested on the other. Compare against the 43% / 75% (4090) thresholds. That one number decides GO / NO-GO and cache policy, at the cost of one routed forward pass over a small corpus.
3. Optional in the same run: log how many distinct experts a 2k-token prefill chunk touches per layer (expect ~all 512) to validate the prefill streaming estimate.
