# Gemma 4 for the engine (GLY-92 architecture notes)

Text model only. Read-only research; nothing committed, no engine code touched.

## Sources and pins

All five repos are ungated (`gated: false` from the HF API); no token used.

| Model | HF revision (sha) | config transformers_version | architectures | text model_type |
|---|---|---|---|---|
| google/gemma-4-31B-it | 842da3794eaa0b77d5f08bae87a17459d91ff475 | 5.5.0.dev0 | Gemma4ForConditionalGeneration | gemma4_text |
| google/gemma-4-12B-it | 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 | 5.10.0.dev0 | **Gemma4UnifiedForConditionalGeneration** | gemma4_unified_text |
| google/gemma-4-E4B-it | ee0ef6023621cff504d758262d4e04895a5af4a2 | 5.5.0.dev0 | Gemma4ForConditionalGeneration | gemma4_text |
| google/gemma-4-E2B-it | 3e22461f65e89153144f8adb70e3b8c2cc9845a7 | 5.5.0.dev0 | Gemma4ForConditionalGeneration | gemma4_text |
| google/gemma-4-26B-A4B-it | 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 | 5.5.0.dev0 | Gemma4ForConditionalGeneration | gemma4_text |

- Code read: transformers 5.18.0 wheel, `models/gemma4/{modeling,configuration}_gemma4.py`, `models/gemma4_unified/modeling_gemma4_unified.py` + `modular_gemma4_unified.py`, and `modeling_rope_utils.py` (`_compute_proportional_rope_parameters`).
- Tensor names/shapes: `model.safetensors.index.json` + shard headers (31B, 26B-A4B: 2 shards); 12B, E4B, E2B are single-file `model.safetensors` with no index, so I read the safetensors header with HTTP range requests (first 8 bytes + JSON header only, no weights downloaded).
- Chat template: `chat_template.jinja` at the pinned sha (tokenizer_config.json carries no template). Token ids: `tokenizer.json` of 31B (all five share the same 32,169,626-byte tokenizer.json).
- The 12B text model is the same decoder as gemma4_text minus PLE and MoE (verified by reading `Gemma4UnifiedTextAttention/DecoderLayer/TextModel`; the code is a line-for-line copy). Its difference is in the multimodal wrapper only (encoder-free vision embedder). So **one text decoder implementation covers 31B, 12B, E4B, E2B and 26B-A4B**, with feature switches.

## 1. Text-model dimensions (from config.json, per size)

| | 31B | 12B | E4B | E2B | 26B-A4B |
|---|---|---|---|---|---|
| layers | 60 | 48 | 42 | 35 | 30 |
| hidden | 5376 | 3840 | 2560 | 1536 | 2816 |
| q heads | 32 | 16 | 8 | 8 | 16 |
| KV heads, sliding layers | 16 | 8 | 2 | 1 | 8 |
| KV heads, global layers | 4 | 1 | 2 | 1 | 2 |
| head_dim, sliding | 256 | 256 | 256 | 256 | 256 |
| head_dim, global (`global_head_dim`) | 512 | 512 | 512 | 512 | 512 |
| GQA ratio sliding / global | 2 / 8 | 2 / 16 | 4 / 4 | 8 / 8 | 2 / 8 |
| dense MLP intermediate | 21504 | 15360 | 10240 | 6144 (12288 on shared layers 15-34) | 2112 |
| vocab (`vocab_size`) | 262144 | 262144 | 262144 | 262144 | 262144 |
| tie_word_embeddings | true | true | true | true | true |
| max_position_embeddings | 262144 | 262144 | 131072 | 131072 | 262144 |
| rms_norm_eps | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| `attention_k_eq_v` | true | true | **false** | **false** | true |
| `num_kv_shared_layers` | 0 | 0 | 18 | 20 | 0 |
| `hidden_size_per_layer_input` (PLE) | 0 | 0 | 256 | 256 | 0 |
| sliding_window | 1024 | 1024 | 512 | 512 | 1024 |
| layer pattern | 5 sliding : 1 global x10 | 5:1 x8 | 5:1 x7 | 4:1 x7 | 5:1 x5 |
| global layer indices | 5,11,...,59 | 5,11,...,47 | 5,11,...,41 | 4,9,...,34 | 5,11,...,29 |
| `use_double_wide_mlp` | false | false | false | true | false |
| MoE | no | no | no | no | yes (see section 5) |

Notes:
- `head_dim` 256 for sliding layers and `global_head_dim` 512 for global layers: **q/k/v widths differ by layer kind**. The config class turns this into `per_layer_config[layer_idx]` (global layers get `head_dim=global_head_dim` and, when `attention_k_eq_v`, `num_key_value_heads=num_global_key_value_heads`). Checked against tensor shapes, e.g. 31B: sliding `q_proj [8192,5376]` (32x256), `k_proj [4096,5376]` (16x256); global `q_proj [16384,5376]` (32x512), `k_proj [2048,5376]` (4x512), `o_proj [5376,16384]`.
- For E2B/E4B (`attention_k_eq_v=false`), `num_global_key_value_heads` is null and the global layers use the same KV head count as sliding layers, at head_dim 512 (E4B global `k_proj [1024,2560]` = 2x512; E2B `[512,1536]` = 1x512).
- The last layer is always global (the config class forces it).
- Embedding scale: the embedding output is multiplied by `sqrt(hidden)`, **with the scale cast to the weight dtype (bf16) first** (`embed_scale.to(self.weight.dtype)`): bf16(sqrt(h)) is 73.5 (31B, exact 73.321), 62.0 (12B, 61.968), 50.5 (E4B, 50.596), 39.25 (E2B, 39.192), 53.0 (26B, 53.066). A bit-faithful engine must use these rounded values, not the exact sqrt. The tied `lm_head` uses the raw (unscaled) embedding matrix.
- Tied embeddings: no `lm_head.weight` in any checkpoint.

## 2. Attention

Per layer (`Gemma4TextAttention`):

```
q = q_norm(q_proj(x).view(heads, hd))           # RMSNorm over head dim, WITH learned weight
q = rope(q)
if not kv_shared_layer:
    k = k_proj(x).view(kvh, hd)
    v = k if (k_eq_v and global layer) else v_proj(x).view(kvh, hd)
    k = rope(k_norm(k))                          # k_norm WITH learned weight, then RoPE
    v = v_norm(v)                                # RMSNorm WITHOUT any weight (no parameter), NO rope
else:
    k, v = shared K,V from the source layer (below)
out = softmax(q k^T * 1.0 + mask) v              # scaling = 1.0 (NOT 1/sqrt(hd)); q/k norms carry the scale
out = o_proj(out)
```

- **Attention scaling is 1.0.** There is no attention-logit soft-cap in Gemma 4 (the eager path has a `softcap` argument but the layer never passes one, and the config has no `attn_logit_softcapping`). Only the final-logit soft-cap exists (section 3).
- q_norm/k_norm are RMSNorm with plain `weight` (init ones, applied as `x * w`, not `1+w`). `v_norm` has `with_scale=False`: no tensor in the checkpoint. The checkpoint holds `q_norm.weight`/`k_norm.weight` of shape [256] (sliding) or [512] (global).
- **K = V layers** (31B, 12B, 26B-A4B global layers; `attention_k_eq_v` true and layer is global): no `v_proj` tensor (31B has `v_proj` on 50 of 60 layers, i.e. missing on the 10 global ones). The one `k_proj` output feeds both: key path is `k_norm -> RoPE`, value path is `v_norm` (no weight, no RoPE). Both K and V still have to be cached (they differ after the norms).
- **Sliding window**: layer attends to keys with `0 <= q_pos - k_pos < sliding_window` (the window counts the current token, so a 1024 window = the last 1024 positions including self; mask function `dist >= 0 & dist < window`). `use_bidirectional_attention: "vision"` only affects image tokens; text-only is plain causal. Window: 1024 for 31B/12B/26B, 512 for E2B/E4B.
- **RoPE** (rotate-half / NeoX layout: pair i with i + hd/2, cos/sin built as `cat(freqs, freqs)`; computed in fp32 and cast to bf16, same as the engine's bf16 tables):
  - sliding layers: `rope_type default`, theta 10,000, full rotation of all 256 dims (128 pairs, `inv_freq = theta^(-2i/256)`).
  - global layers: `rope_type proportional`, theta 1,000,000, `partial_rotary_factor 0.25`, head_dim 512: `rope_angles = int(0.25*512//2) = 64`, so only pairs i = 0..63 rotate with `inv_freq_i = theta^(-2i/512)` (exponent divided by the **full** 512), and the other 192 inv_freq entries are zero (cos=1, sin=0, identity). Because the pairing is (i, i + 256), the rotated channels are **0..63 and 256..319**, not a contiguous 0..127 block. This differs from Qwen3.5's partial rotary (contiguous first `rotary_dim` dims, denominator = rotary_dim). Implementable as a cos/sin table [max_pos, 256] with an identity tail on the existing NeoX pairing.
  - No scaling factor (`attention_scaling = 1.0`) for either kind.
- **KV sharing** (E2B: last 20 layers = 15..34; E4B: last 18 layers = 24..41): a shared layer has no `k_proj/v_proj/k_norm/v_norm` in the model (the checkpoints still contain them; transformers ignores them on load: `_keys_to_ignore_on_load_unexpected`) and computes only Q. It reads K,V from "the last non-shared layer of the same layer type": E2B sliding -> layer 13, global -> layer 14; E4B sliding -> layer 22, global -> layer 23. The shared layer applies its own mask (sliding layers use the sliding mask on the source layer's window), its own q_norm and its own RoPE for Q. Positions are the same. Consequences: KV cache is only allocated for layers 0..14 (E2B) / 0..23 (E4B); the source layers' sliding K/V need only the window.
- Padding: rotary positions are plain 0..n-1; no position offsets.

## 3. Block structure, norms, activation, logits

Decoder layer (dense, `Gemma4TextDecoderLayer`):

```
h = x + post_attention_layernorm( attn( input_layernorm(x) ) )
h = h + post_feedforward_layernorm( mlp( pre_feedforward_layernorm(h) ) )
if PLE: h = h + post_per_layer_input_norm( per_layer_projection( gelu_tanh(per_layer_input_gate(h)) * ple_input[layer] ) )
x = h * layer_scalar                                   # layer_scalar: bf16 tensor [1] per layer, from the checkpoint
```

- **Four norms per layer** (input, post_attention, pre_feedforward, post_feedforward), norms applied *around the sublayer output before the residual add* (sandwich norm). Qwen3/3.5 has two (pre-norm only), and the engine's fused "add then norm" (`rmsnorm_add`) no longer matches: here it is `x += norm(sublayer_out)` and then the next pre-norm of the new x.
- **RMSNorm form: `x * rsqrt(mean(x^2)+eps) * w`**, weight init ones, computed in fp32 (`Gemma4RMSNorm`; `torch.pow(mean_sq+eps, -0.5)`). This is the **same as the engine's Qwen3 `rmsnorm` (w), not Qwen3.5's `rmsnorm1p` (1+w)**. Weight is stored as-is (no +1 offset). Checkpoint norm weights are the real multipliers.
- Final `model.language_model.norm.weight` is the same plain-w RMSNorm.
- **Activation: `gelu_pytorch_tanh`** (`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))`), gated MLP `down(gelu_tanh(gate(x)) * up(x))`; engine has silu*up, needs a gelu-tanh*up variant (fused gemm_silu path too).
- **Final logit soft-cap 30.0**: `logits = 30 * tanh(logits / 30)` applied to the lm_head output (after tied-embedding matmul), before softmax/sampling. Greedy argmax is invariant to it (monotone) except bf16 ties; logprobs, logsumexp and sampling must apply it.
- No biases anywhere (`attention_bias false`).
- **Per-layer embeddings (PLE), E2B and E4B only** (`hidden_size_per_layer_input = 256`):
  - Extra tensors: `model.language_model.embed_tokens_per_layer.weight [262144, L*256]` (E4B [262144, 10752] = 42 x 256; E2B [262144, 8960] = 35 x 256), `per_layer_model_projection.weight [L*256, hidden]` (E4B [10752,2560]; E2B [8960,1536]), `per_layer_projection_norm.weight [256]`, and per layer `per_layer_input_gate.weight [256, hidden]`, `per_layer_projection.weight [hidden, 256]`, `post_per_layer_input_norm.weight [hidden]`.
  - Computation, once per token before the layers: `tok = embed_tokens_per_layer[id] * sqrt(256)` (scale 16.0, bf16-exact) reshaped to [L, 256]; `ctx = RMSNorm_256( (per_layer_model_projection(embeds) * hidden^-0.5).reshape(L,256) )` where `embeds` is the already `sqrt(hidden)`-scaled input embedding; `ple_input = (ctx + tok) * 2^-0.5`. Layer i uses `ple_input[i]` after its MLP residual (formula above, using gelu_tanh).
  - "E" = effective parameters: the table `embed_tokens_per_layer` is a per-token *lookup* (one row of L*256 values per token), not a matmul, so it does not count toward compute: E4B = 7.52B text params in the file but 2.82B of them are that table (about 4.6B matmul/embedding params left, computed); E2B = 4.65B text params, 2.35B is the table (about 2.3B left, computed). Table sizes in bf16: E4B 5.64 GB, E2B 4.70 GB. The table can live packed (gather a row, unpack 10752 values) or in host memory; per decode token only one row is touched.
- E2B `use_double_wide_mlp`: layers that share KV (>= 15) have MLP intermediate 12288 instead of 6144 (checkpoint shapes confirm: `gate_proj [12288,1536]` on layers 15..34).

## 4. Dimension-specific engine observations

- Sliding layers hd 256, global layers hd 512. Qwen3 kernels in `engine.cu` are hard-wired for head 128; Qwen3.5's `attn256`/`q35_prep` are hd 256 with scale hard-coded `1/16` (`sl2 = 1/16 * log2e`). Gemma 4 needs scale 1.0 (`sl2 = log2e`), hd 256 *and* hd 512, sliding masks, and K/V that differ per layer kind.
- Largest per-layer widths: 31B global `q` 16384 x 5376, `o_proj` 5376 x 16384.

## 5. 26B-A4B MoE (`enable_moe_block: true`, all 30 layers)

Every one of the 30 layers has **both** the dense MLP and the MoE block, summed in parallel (no dense-only layers in between; `enable_moe_block` is a global flag). The dense MLP (intermediate 2112 = 3 x 704) plays the role of the "shared expert".

Config: `num_experts 128`, `top_k_experts 8`, `moe_intermediate_size 704` (note: 31B's config spells the null placeholder `expert_intermediate_size`, 12B/26B use `moe_intermediate_size`; use the 26B key `moe_intermediate_size`), `intermediate_size 2112` (dense branch), hidden 2816.

Layer forward (`Gemma4TextDecoderLayer`, MoE branch):

```
r = h                                               # residual stream after attention (pre-norm)
d  = mlp( pre_feedforward_layernorm(r) )            # dense branch
d1 = post_feedforward_layernorm_1(d)
# router sees the UN-normed residual r:
z  = RMSNorm_noscale(r) * router.scale * hidden^-0.5      # router.scale: learned [2816]
p  = softmax_fp32( router.proj(z) )                       # over all 128 experts
w, idx = topk(p, 8);  w = w / sum(w);  w = w * router.per_expert_scale[idx]   # per_expert_scale: learned [128]
e  = experts( pre_feedforward_layernorm_2(r), idx, w )    # sum_k w_k * down_k( gelu_tanh(gate_k(x)) * up_k(x) )
e2 = post_feedforward_layernorm_2(e)
h  = r + post_feedforward_layernorm( d1 + e2 )
```

- Router: `router.proj.weight [128, 2816]`, `router.scale [2816]`, `router.per_expert_scale [128]`. Softmax over all experts *before* top-k, then renormalise the selected 8 to sum to 1 (so the unnormalised probabilities do not matter beyond selection), then multiply by the per-expert scale. No router bias, no load-balancing at inference.
- Experts are **fused 3-D tensors without a `.weight` suffix**: `layers.N.experts.gate_up_proj [128, 1408, 2816]` (first 704 rows = gate, next 704 = up: `chunk(2, dim=-1)` of the output) and `layers.N.experts.down_proj [128, 2816, 704]`. Expert size 3 x 704 x 2816 = 5.95M params; experts total 22.84B params (computed from shapes).
- Six extra norms/tensors per layer vs dense: `post_feedforward_layernorm_1`, `_2`, `pre_feedforward_layernorm_2` (all [2816], plain-w RMSNorm).
- Attention in 26B-A4B: 16 q heads, sliding 8 KV x256, global 2 KV x512 (K=V), window 1024, 5:1 pattern, 30 layers (5 global).
- Active parameters per decode token: all non-expert text params (2.40B incl. embedding table 0.74B) + 8/128 of experts (1.43B); computed 3.82B counting the embedding table (3.08B if only one embedding row is counted). Matches the "A4B" name.

## 6. Multimodal wrapper and loading text only

Tensor-name prefixes (from the real headers):

| | text | skip |
|---|---|---|
| 31B, E4B, E2B, 26B-A4B (`gemma4`) | `model.language_model.` | `model.vision_tower.*`, `model.embed_vision.*`, and for E2B/E4B `model.audio_tower.*`, `model.embed_audio.*` |
| 12B (`gemma4_unified`) | `model.language_model.` | `model.vision_embedder.*` (encoder-free patch embedder: patch_ln1/dense/ln2, pos_embedding, pos_norm), `model.embed_vision.*`, `model.embed_audio.*` |

- Same prefix string as Qwen3.5's `Config35.prefix` ("model.language_model.") and it is used for every size, including 12B. No `lm_head.*` tensors. No `mtp.*`.
- Text tensors per dense layer: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `layer_scalar`, `self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}`, `mlp.{gate_proj,up_proj,down_proj}`, all `.weight` except `layer_scalar`. Tensor count per size: 31B 832 text / 1188 total, 12B 666 / 677, E4B 719 / 2130, E2B 600 / 2011, 26B-A4B 657 / 1013.
- Everything is bf16. 31B and 26B-A4B ship in 2 shards (with an index), 12B/E4B/E2B in one file.
- Text-only usage: `architectures` is `Gemma4ForConditionalGeneration` / `Gemma4UnifiedForConditionalGeneration` (engine's Qwen3 loader rejects anything but `Qwen3ForCausalLM`), `config.text_config` holds all the dims above, and `eos_token_id` at top level is `[1, 106]`. `tie_word_embeddings` is both top-level and in text_config.

## 7. Tokens, chat template, thinking, stops

Ids (31B tokenizer.json; shared by all five):

| token | id | | token | id |
|---|---|---|---|---|
| `<pad>` | 0 | | `<\|tool_call>` / `<tool_call\|>` | 48 / 49 |
| `<eos>` | 1 | | `<\|tool_response>` / `<tool_response\|>` | 50 / 51 |
| `<bos>` | 2 | | `<\|"\|>` (string delimiter) | 52 |
| `<unk>` | 3 | | `<\|think\|>` | 98 |
| `<mask>` | 4 | | `<\|channel>` / `<channel\|>` | 100 / 101 |
| `<\|tool>` / `<tool\|>` | 46 / 47 | | `<\|turn>` / `<turn\|>` | 105 / 106 |
| `<\|image>` `<image\|>` `<\|image\|>` | 255999 / 258882 / 258880 | | `<\|audio>` `<audio\|>` `<\|audio\|>` | 256000 / 258883 / 258881; `<\|video\|>` 258884 |

- **Stop tokens**: config `eos_token_id` = [1, 106] (`<eos>`, `<turn|>`); generation_config adds **50 (`<|tool_response>`)** = [1, 106, 50] (the model emits `<|tool_response>` after a tool call to hand off to the caller). 12B's generation_config also lists `suppress_tokens` [258883, 258882] (`<audio|>`, `<image|>`). Sampling defaults: temperature 1.0, top_k 64, top_p 0.95, do_sample true.
- Tokenizer: BPE, 262144 vocab, 24 added tokens, SentencePiece-style (normalizer `Replace ' ' -> '▁'`, pre_tokenizer `Split ' ' MergedWithPrevious`), `TemplateProcessing` post-processor; loads through the same `tokenizers` crate path as `tok.rs`. The template emits `<bos>` itself (so encode with `special=false` as `Tok::encode_with` supports).
- **Turn format** (roles are `user`, `model`, `system`):

```
<bos><|turn>system
{system text}<turn|>
<|turn>user
{user text}<turn|>
<|turn>model
{assistant text}<turn|>
<|turn>user
...<turn|>
<|turn>model
```

  The system turn exists only if there is a system/developer message, tools, or thinking enabled. Messages are `trim`med. Assistant history has thinking blocks stripped (`strip_thinking` removes `<|channel>...<channel|>`).
- **Thinking**: `enable_thinking` (default false). When true, `<|think|>\n` is placed at the top of the first system turn (creating the system turn if needed) and the generation prompt is just `<|turn>model\n`; the model then writes `<|channel>thought\n...<channel|>` followed by the answer. When false, the 31B/12B/26B-A4B template appends an empty thought block to the generation prompt: `<|turn>model\n<|channel>thought\n<channel|>`. **The E2B/E4B template does not** (diff of the two templates: only that `{%- if not enable_thinking -%}` block and one blank line). The reply parser regex in tokenizer_config recovers `thinking`, `tool_calls`, `content`.
- **Tool calls**: declarations `<|tool>declaration:{name}{...}<tool|>` in the system turn (rendered by the template's `format_function_declaration`); calls `<|tool_call>call:{name}{key:value,...}<tool_call|>` with string values in `<|"|>...<|"|>` and unquoted keys; responses `<|tool_response>response:{name}{...}<tool_response|>`. Not needed for a first bring-up.
- Template features used: macros, `namespace`, `dictsort`, `map('upper')`, `is mapping`/`is sequence`, `split`, `trim`, `raise_exception`. The engine's `chat.rs` uses minijinja 2.24 + pycompat with `raise_exception` registered, which should cover these; **not run** (see unverified).

## 8. Licence

- The repos contain **no LICENSE file**: the file fetch of `LICENSE` at each pinned sha returns 404 ("Entry not found"), and the HF API file list (`siblings`) shows none (files: `.gitattributes`, `README.md`, `chat_template.jinja`, `config.json`, `generation_config.json`, `model*.safetensors`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`; 31B/26B also `.eval_results/mmmu_pro.yaml`).
- Metadata: all five have `license: apache-2.0` and `license_link: https://ai.google.dev/gemma/docs/gemma_4_license` in the README front matter and the HF tag `license:apache-2.0`; the card says "License: Apache 2.0".
- The linked licence page (fetched) is titled "Apache License 2.0" and its text begins: "Apache License / Version 2.0, January 2004 / http://www.apache.org/licenses/ / TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION / 1. Definitions. "License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document...". So the terms are the standard Apache-2.0 text, hosted on ai.google.dev rather than shipped in the repo. (Different from earlier Gemma generations, which used the Gemma Terms of Use; I did not check whether an additional prohibited-use policy is linked, only the licence page head.)

## 9. What the engine has now (skimmed, not audited)

- `model.rs`/`engine.cu` (Qwen3, lossless): head_dim 128 hard-wired (`qk_norm_rope`: 64 threads, channels i and i+64, NeoX layout, bf16 cos/sin tables [max_pos, 64], fp32 inside, one rounding); plain `rmsnorm`/`rmsnorm_add` (w, not 1+w; fused add+norm); `silu_mul`/`gemm_silu`; paged lossless 12-bit KV cache (16-token pages, block tables, status rings, raw/escape pools, KV head = 128 channels); tensor-core `prefill_attn_*`; split decode attention; `argmax_bf16`, `logsumexp_bf16`; GQA by KV-head grid; tied or untied lm_head from a packed matrix.
- `model35.rs`/`qwen35.rs`/`qwen35.cu`: `Config35` (prefix "model.language_model.", `rotary_dim`, `kinds`: Linear/Full), `rmsnorm1p` (1+w), `q35_prep` (norm + partial rotary on the first 2*half dims at hd 256, writes K/V to a flat per-layer cache `kc/vc [positions, nk, 256]`), `attn256` + `attn256_parts`/`attn256_merge` (decode split-K), output gate sigmoid, Gated DeltaNet linear layer, Int5/Int6 block-code products (`glyd-v7` layouts), `lin_packed`, embedding row unpack, attention scale 1/16 hard-coded.
- Tokenizer through `tokenizers`; Jinja chat templates via minijinja; sampler on host or device.

## 10. Engine has / engine needs

Size: S = under ~1 day of kernel/loader work, M = a few days, L = a week or more including bit-level validation vs transformers.

| Piece | Engine has | Engine needs | Size |
|---|---|---|---|
| Config parser for `Gemma4*ForConditionalGeneration` / `Unified...` (nested `text_config`, `layer_types`, `rope_parameters` per kind, `per_layer_config` rules, `eos_token_id` list + generation_config) | `Config`/`Config35` parsers | New `ConfigG4`; prefix `model.language_model.`; derive per-layer (hd, kv heads, kv-owner, window) | S |
| Tensor-name mapper + packer for gemma layout (4 norms/layer, `layer_scalar`, optional `v_proj`, fused 3-D expert tensors, absent lm_head) | Packer for Qwen3 / Qwen3.5 names | Name table, handle missing v_proj (K=V), skip vision/audio prefixes, carry `layer_scalar` | M |
| RMSNorm (w) incl. fused norm variants | `rmsnorm`, `rmsnorm_add` | "x += norm(y)" (post-norm then residual add) and double-norm fusion; norm without weight (v_norm, router) | S |
| gelu-tanh gated MLP | `silu_mul`, `gemm_silu` | `gelu_tanh_mul` + fused gemm variant | S |
| q/k norm + RoPE at hd 256 (sliding, theta 1e4, full rotary) | `qk_norm_rope` (hd 128), `q35_prep` (hd 256, partial contiguous rotary) | Generalised prep: hd 256 full rotary with theta 1e4; v_norm (weightless) on V; K=V from one projection | M |
| Proportional RoPE at hd 512 (theta 1e6, 64 rotating pairs (i, i+256), identity tail) | partial rotary (contiguous dims) | cos/sin table with identity tail + hd 512 prep kernel | S |
| Attention scale 1.0 | scale 1/16 or 1/sqrt(128) constants | `sl2 = log2e`, parameterised | S |
| hd 512 attention (decode + prefill), GQA 4/8/16:1, global layers | attn256 (hd 256), prefill hd 128 | New head-dim 512 kernels (register/smem pressure: 512 dims, K/V tiles) for decode-split and prefill | L |
| Sliding-window attention hd 256 with ring-buffer KV (window 512 or 1024) | none (flat or paged full-length cache) | Window-limited mask in decode/prefill kernels; ring-buffer KV cache per sliding layer (slot = pos % window), prefill chunk longer than window must be handled | L |
| KV cache layout: per-layer different (kvh, hd, window), 50 of 60 layers windowed | Uniform per-layer cache sized by ctx | Heterogeneous cache sizing; global layers hd 512 (K,V both kept even for K=V); 12-bit lossless cache variant would need hd 256/512 pages | L |
| KV sharing across layers (E2B/E4B) | none | Shared layers compute Q only; read K/V (and ring) of source layer 13/14 (E2B) or 22/23 (E4B); skip k/v/norm tensors | M |
| Per-layer embeddings (E2B/E4B) | embedding row unpack | PLE table gather from packed [262144, L*256] (host or device), projection matmul + RMSNorm_256 + per-layer gate kernels, ple_input buffer per token | L |
| Embedding scaling sqrt(h) in bf16 | embedding row copy | multiply by bf16-rounded scale (73.5 etc.) at embed time (can be folded into a scaled copy only if bit-identical rounding is checked) | S |
| `layer_scalar` multiply per layer | none | One scalar multiply on the residual stream per layer (fuse with next norm) | S |
| Final logit soft-cap 30 | none | tanh cap kernel on logits (sampler, logsumexp; argmax can skip) | S |
| Double-wide MLP (E2B layers 15..34, intermediate 12288) | per-model fixed `inter` | per-layer intermediate size | S |
| MoE (26B-A4B): router (norm-noscale, scale, softmax, top-8, renorm, per-expert scale), 128 experts x gate_up/down with grouped GEMM, parallel dense MLP, 3 extra norms | none (no MoE in engine) | Router kernel + top-k, token->expert permutation / grouped or gather GEMM on quantised (Int5/Int6) fused 3-D tensors, combine kernel; expert-offload/placement for decode (8 of 128 experts per token per layer) | L |
| Quantised products for these matrices (Int5/Int6 block codes) | Qwen3.5 Int5/Int6 packs | Check new shapes: K=2112 (26B dense MLP down_proj in K), 704 (expert down K), 256 (PLE gate/proj), 21504, 16384 (31B global o_proj K), 2816 etc. against block/tile divisibility of `gemm_idx.h` | M |
| Chat template + stop tokens + thinking toggle | minijinja renderer, `enable_thinking`, eos from generation_config | Verify template renders; stops [1, 106, 50]; E-model vs larger-model empty-thought difference is in the template itself | S |
| Tokenizer | `tokenizers` crate | Nothing new expected (BPE, same crate) | S |
| Validation vs transformers (bf16 reference like `qwen35_ref.rs`) | kcheck/qwen35_ref | Reference/kcheck for gemma4 text incl. softcap, PLE, MoE | M |

Suggested order: (1) 12B (no PLE, no MoE, no sharing) with sliding + hd 512 + softcap as the first target since it exercises every attention piece; (2) 31B (same code, bigger); (3) E4B/E2B (add PLE + KV sharing + double-wide MLP); (4) 26B-A4B (add MoE). 31B and 12B differ only in dims.

## 11. Memory (computed from tensor shapes, not measured)

Method: parameter counts are summed from the real tensor shapes of every `model.language_model.*` tensor (E2B/E4B: excluding the checkpoint's unused k_proj/v_proj/k_norm of KV-shared layers; vision/audio excluded). Weight bytes = params x bits-per-weight / 8 with every text parameter (embedding and, for the E models, the PLE table) at the stated bits. Norms/scalars are negligible. GB = 1e9 bytes. "PLE bf16" column keeps the PLE table in bf16 and quantises the rest.

| Model | text params | embedding params | extra | bf16 GB | Swift 5.53 GB | Kestrel 6.53 GB | Swift with PLE kept bf16 |
|---|---|---|---|---|---|---|---|
| 31B | 30.70B | 1.41B | - | 61.39 | **21.22** | **25.06** | - |
| 12B | 11.91B | 1.01B | - | 23.81 | **8.23** | **9.72** | - |
| E4B | 7.46B used (7.52B in file) | 0.67B | PLE table 2.82B | 14.93 | **5.16** | **6.09** | 8.85 |
| E2B | 4.63B used (4.65B in file) | 0.40B | PLE table 2.35B | 9.26 | **3.20** | **3.78** | 6.27 |
| 26B-A4B | 25.23B | 0.74B | experts 22.84B | 50.47 | **17.44** | **20.60** | - |

Add KV cache and activations. KV bytes at bf16 (K and V stored separately; upper bound before any KV compression; global layers with K=V could store 2x less if deduplicated, which I did not assume):

| Model | KV-owning layers (sliding / global) | per-token, global layers | sliding layers (window-capped) | at 32k ctx | at 128k ctx |
|---|---|---|---|---|---|
| 31B | 50 / 10 | 80 KiB | 800 MiB total | 3.28 GiB | 10.78 GiB |
| 12B | 40 / 8 | 16 KiB | 320 MiB total | 0.81 GiB | 2.31 GiB |
| E4B | 20 / 4 (layers 0..23 only) | 16 KiB | 20 MiB total | 0.52 GiB | 2.02 GiB (model max 128k) |
| E2B | 12 / 3 (layers 0..14 only) | 6 KiB | 6 MiB total | 0.19 GiB | 0.76 GiB (model max 128k) |
| 26B-A4B | 25 / 5 | 20 KiB | 200 MiB total | 0.82 GiB | 2.70 GiB |

(Compare: Qwen3.5 caches only the full-attention layers; Gemma 4's sliding layers are small thanks to the window, and the global layers are the cost.)

## 12. Unverified or open

- Nothing was run: no numerical comparison with transformers, no engine build. All behaviour is from reading transformers 5.18.0 code and checkpoint metadata. Code at 5.18.0 vs the checkpoints' recorded versions (5.5.0.dev0 / 5.10.0.dev0) was assumed compatible; config keys all loaded cleanly against the 5.18.0 config classes by reading, not by executing.
- `layer_scalar` values were not inspected (weights not downloaded); only that it is a [1] bf16 tensor per layer applied as the last op. Initialised to 1 in code; trained values may differ.
- Exact bf16 rounding points of transformers (e.g. whether the MoE combine, softmax and gelu run in fp32) were read from eager PyTorch code; reference implementations (vLLM/llama.cpp/MLX) may round differently, as with the Qwen3.5 "vLLM 0.30 rounding points" note. Pick the reference first.
- The chat template was read, not rendered with the engine's minijinja; the macro-heavy tool-call section is untested. The `<|channel>thought\n<channel|>` empty-thought generation prefix differs between the two template variants.
- Licence: no LICENSE file in any repo; Apache-2.0 per metadata and the linked ai.google.dev page head (read the top of the page only, not the whole text, and did not look for an attached use policy).
- E-model "effective parameter" accounting (and Google's own 2.3B/4.5B figures) not cross-checked against the card.
- Memory numbers assume the Swift/Kestrel bits-per-weight apply uniformly to all matrices including embedding/PLE; the real packers may keep some tensors at higher precision (embeddings, router, first/last layers).
- Whether the existing Int5/Int6 packers handle the new K dimensions (704, 2112, 256, 16384) was not checked.
- Vision/audio paths (including 12B's encoder-free vision embedder) not covered, as requested.
- Gemma4 assistant/MTP drafter models (`gemma4_assistant`, `gemma4_unified_assistant` in transformers 5.18.0) exist but were not part of this task.
