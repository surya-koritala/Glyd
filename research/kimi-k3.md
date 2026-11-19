# Kimi-K3 text model: math and tensor spec

Source: release `config.json`, `ref/modeling_kimi_linear.py` (text model), `ref/modeling_kimi_k3.py`
(wrapper), `ref/configuration_kimi_k3.py`, `ref/encoding_k3.py`, `ref/tokenizer_config.json`,
`ref/generation_config.json`, `ref/README.md`, and tensor names from `ref/model.safetensors.index.json`
(497,220 names, 96 shards, total_size 1,560,860,324,864 bytes). Vision tower / projector (168 tensors) out of scope.

Tags: **[file]** = read in the inputs. **[lib]** = done inside the `fla` library (fla-core) kernels that the
reference calls; that code is not in the inputs, so it's described from what the library is known to do. Check it
against reference outputs. **[derived]** = worked out from the config, not stored anywhere.

All tensor names start with `language_model.`. Linear weights are `[out, in]` (y = W x).

## 0. Constants (text_config)

| name | value |
|---|---|
| hidden H | 7168 |
| layers | 93 (0..92) |
| vocab | 163840 (README says "160K") |
| heads (MLA and KDA) | 96 |
| KDA head_dim (K = V) | 128 → projection width P = 96·128 = 12288 |
| MLA q_lora / kv_lora | 1536 / 512 |
| MLA nope / "rope" / v | 128 / 64 / 128 (q/k head dim 192) |
| dense MLP width (layer 0) | 33792 |
| experts / top-k / shared | 896 / 16 / 2 |
| expert width (moe_intermediate_size) | 3072 |
| latent width (routed_expert_hidden_size) | 3584 |
| shared expert width | 2·3072 = 6144 (one merged MLP) |
| rms_norm_eps | 1e-5 (but MLA q_a/kv_a norms use 1e-6, see §4) |
| attn_res_block_size | 12 |
| situ beta / linear_beta | 4.0 / 25.0 |
| short conv kernel | 4 |
| gate_lower_bound | −5.0 |
| routed_scaling_factor | 1.0 |
| max_position_embeddings | 1,048,576 |
| tie_word_embeddings | false |

Not in the config: `rope_theta` and `rope_scaling`. The config class would default them to 10000 / None, but
nothing reads them. **The text model has no RoPE anywhere** (§4). `head_dim` isn't set either, so the class gives
7168//96 = 74; that value is never used (KDA reads `linear_attn_config.head_dim` = 128).

Size check [derived]: with the shapes below, active params ≈ 104.2B (README: 104B) and total ≈ 2.78T
(README: 2.8T). Expert params come to 2.7227T, and non-expert text params to 56.74B.

## 1. Layer numbering

The config lists are **1-based**. The code tests whether `(layer_idx + 1)` is in `kda_layers` [file]. Any layer that
isn't KDA is MLA, because `is_mla` is true. 0-based:

- **MLA (24 layers):** 3, 7, 11, …, 87, 91 (every 4th layer, idx ≡ 3 mod 4) **plus 92**. The last two layers (91, 92)
  are both MLA.
- **KDA (69 layers):** every other index 0..90 (idx mod 4 ∈ {0,1,2}). Layer 0 is KDA.
- Verified against the index: `self_attn.q_a_proj` exists exactly at {3,7,…,91,92}, and `self_attn.A_log` exactly at
  the 69 KDA indices. Both match `config − 1`.
- **Dense MLP:** layer 0 only (`mlp.{gate,up,down}_proj`). **MoE:** layers 1..92 (`block_sparse_moe.*`), because
  first_k_dense_replace = 1 and moe_layer_freq = 1. Also verified in the index.
- `g_proj` and `o_proj` exist on all 93 layers. The KDA and MLA versions have the same shapes but do different jobs.

## 2. Norms, rounding, residual scheme (AttnRes)

### 2.1 RMSNorm (`KimiRMSNorm`) [file]
`y = w ⊙ bf16( x_f32 · rsqrt(mean(x_f32²) + eps) )`. Normalize in fp32, round to bf16, then multiply by the bf16
weight, which rounds again. Two rounding points.

### 2.2 Depth mixer `Mix(sources; w_n, w_p)` [file: `_apply_attn_res`]
Sources v_0..v_n ∈ R^H are bf16 tokens. Each token is mixed independently, and nothing mixes across tokens. All math
is fp32:
```
u     = w_n ⊙ w_p                       (fp32, H-vector; w_p is the [1,H] proj weight)
s_i   = Σ_d u_d · v_i,d / sqrt(mean_d(v_i²) + 1e-5)
α     = softmax_i(s)                     (over the n+1 sources)
out   = bf16( Σ_i α_i v_i )
```
The norm weight is not applied the KimiRMSNorm way. It is folded into one fp32 query vector u, and no bf16 rounding
happens before the dot. Only the final output is rounded.

### 2.3 Running state per token during one forward
- `R`: the list of completed block vectors (bf16, H each). It starts empty.
- `p`: the partial sum for the current block (bf16). It is either "empty" or a value.

The model input is `p = embed[token]` (bf16), with R = [] [file: `_forward_attn_residual`].

Decoder layer l (0..92):
```
1. x = (R empty) ? p : Mix(R ++ [p]; self_attention_res_norm, self_attention_res_proj)
2. if l % 12 == 0:  R.append(p);  p = empty          # boundaries l = 0,12,24,...,84
3. a = Attn_l( RMSNorm_in(x) )                        # input_layernorm, eps 1e-5
4. p = (p empty) ? a : bf16(p + a)
5. y = Mix(R ++ [p]; mlp_res_norm, mlp_res_proj)
6. m = FFN_l( RMSNorm_post(y) )                       # post_attention_layernorm; dense (l=0) or MoE
7. p = bf16(p + m)
```
End of model:
```
h      = Mix(R ++ [p]; output_attn_res_norm, output_attn_res_proj)
h      = RMSNorm_final(h)              # model.norm, eps 1e-5
logits = lm_head(h)                    # bf16 [163840]
```
What this means:
- There is no plain residual stream. Every sublayer's input is a softmax-weighted mix of block summaries.
- Contents of R: R[0] = embedding. R[1] = Σ (a+m) over layers 0–11, R[2] over 12–23, …, R[7] over 72–83. At the end,
  p = Σ over layers 84–92 (9 layers).
- Sources per Mix call = |R| + 1. That's 2 in layers 0–11, 3 in layers 12–23, …, and 9 in layers 84–92 and at the
  output (max 9). At a boundary layer l = 12k, the attention mix (step 1) runs before the append, so it has one
  fewer source.
- Layer 0's self_attention_res_{norm,proj} exist in the checkpoint but are never used (R is empty at step 1).
- Step 1 at a block boundary (l = 12k, k ≥ 1) mixes [R…, p], and then p is appended. So the mixer sees the same
  set the list holds right after the append.
- Partial sums are rounded to bf16 at every add.

Engine notes: the 1/rms(v_i) of a completed block never changes, so compute it once per token. Each Mix call then
costs n+1 dots of length H plus one weighted sum. Within one token's forward you hold at most 8·H completed-block
vectors plus p. Across tokens AttnRes keeps **nothing**.

### 2.4 Per-sequence state a decode step must keep (across tokens)
- 24 MLA layers: latent KV cache, 576 bf16 per token per layer (§4.4). That's 27,648 B per token, or about
  29 GB at 1,048,576 tokens.
- 69 KDA layers: recurrent state S of 96×128×128 fp32 (6.29 MB per layer, ≈434 MB total), plus the conv tails:
  the last 3 pre-conv inputs for each of q, k, v (3·12288·3 bf16 per layer) (§3).
- No position counter is needed for the math (there's no RoPE). Positions only matter for cache bookkeeping and the
  causal mask.

## 3. KDA layer (69 layers)

Input x = RMSNorm_in(·) ∈ R^7168, bf16. Channel layout is always (h, d) with d fastest, so index = h·128 + d.

### 3.1 Tensors (all bf16 unless noted; shapes [derived], since the index has no shapes)
| tensor | shape | role |
|---|---|---|
| q_proj, k_proj, v_proj | [12288, 7168] | projections |
| q_conv1d, k_conv1d, v_conv1d .weight | [12288, 1, 4] (depthwise; no bias tensor in index) | short convs |
| f_a_proj | [128, 7168] | decay gate, low rank, part 1 |
| f_b_proj | [12288, 128] | decay gate, low rank, part 2 |
| A_log | [96] (created fp32; stored dtype unverified) | per-head decay rate |
| dt_bias | [12288] (created fp32; stored dtype unverified) | per-channel gate bias |
| b_proj | [96, 7168] | beta |
| g_proj | [12288, 7168] | **output** gate, full rank (`use_full_rank_gate`) |
| o_norm.weight | [128] (shared by all heads) | gated RMSNorm |
| o_proj | [7168, 12288] | output |

`use_full_rank_gate=true` selects `g_proj` (the output gate) instead of a low-rank `g_a_proj/g_b_proj`. The decay
gate f_a/f_b is always low rank (7168→128→12288), with no activation between the two [file].

Note on dtypes: transformers loads the model in bf16 and has no keep-fp32 list here. So in the HF reference,
A_log and dt_bias end up as bf16 values even if they are stored as fp32. Small effect; your choice.

### 3.2 Per-token math
```
q0 = q_proj x;  k0 = k_proj x;  v0 = v_proj x                    # bf16 [12288]
q1 = silu(conv4(q0)); k1 = silu(conv4(k0)); v1 = silu(conv4(v0))  # all three convolved
```
conv4 is depthwise and causal, per channel c: `y_t[c] = Σ_{i=0..3} W[c,0,i] · x_{t-3+i}[c]`. Tap 3 multiplies the
current token, and inputs before the sequence start are 0. Accumulate in fp32, apply silu, output bf16
[file: kernel=4, silu; lib: arithmetic]. For decode, keep the last 3 inputs x_{t-3..t-1} per channel (the library keeps
a 4-slot window that includes the current input). Prefill with a cache keeps going from that state.

Reshape to [96,128] per token: q1, k1, v1, and graw = f_b_proj(f_a_proj x).

Gate, beta, norms (all inside the kernel, fp32) [lib; flags from the file]:
```
g_t[h,d] = −5 · sigmoid( exp(A_log[h]) · (graw_t[h,d] + dt_bias[h·128+d]) )   ∈ (−5, 0)   (log-decay)
α_t      = exp(g_t)                                                            ∈ (e^−5, 1) per channel
β_t[h]   = sigmoid( fp32(b_proj x)[h] )
q̂ = q1 / sqrt(Σ q1² + 1e-6),  k̂ = k1 / sqrt(Σ k1² + 1e-6)     (per head, over 128; eps is the lib default)
scale    = 128^−0.5                                              (kernel default; the file doesn't pass scale)
```
The file passes `use_gate_in_kernel`, `use_beta_sigmoid_in_kernel`, `use_qk_l2norm_in_kernel`,
`lower_bound=-5`, and `safe_gate=True` (chunk path only). The −5·sigmoid(...) gate form is the library's
lower-bounded gate. The non-lower-bounded form would be `−exp(A_log)·softplus(graw+dt_bias)`, but **it is not used
here**. The decode path (`fused_recurrent_kda`) gets lower_bound but not safe_gate. I'm assuming it uses the same gate
formula.

Recurrence, per head, with S ∈ R^{K×V} = 128×128 in fp32 and S_0 = 0 [lib; standard KDA]:
```
S'   = diag(α_t) · S_{t−1}                 # decay rows (key dim)
u_t  = β_t · ( v1_t − S'ᵀ k̂_t )            # delta-rule correction, v1_t ∈ R^128
S_t  = S' + k̂_t u_tᵀ                        # = (I − β k kᵀ) diag(α) S_{t−1} + β k vᵀ
o_t  = scale · S_tᵀ q̂_t                     # read AFTER the update; bf16 output [96,128]
```
`transpose_state_layout=True` only means the library stores the state as [H, V, K]. Pick any layout you like, but
transpose if you exchange state with fla.

Output:
```
G   = g_proj x  → [96,128]  (bf16)
o'  = bf16( w_onorm ⊙ o · rsqrt(mean_d(o²) + 1e-5) ⊙ sigmoid(G) )   # per head over 128, fused fp32
out = o_proj( flatten(o') )                                          # [7168]
```
o_norm is `FusedRMSNormGated(128, eps=1e-5, activation='sigmoid')` [file]. The math inside the fused kernel (fp32,
one rounding at the end, sigmoid gate rather than swish) is [lib].

### 3.3 Chunked prefill
The file always uses the library's chunk kernel for prefill (q_len > 1, with or without a cache) and the recurrent
kernel for 1-token decode. Here is the equivalent math for one chunk of C tokens (the library uses C = 64), per head,
with chunk start state S_0. Let γ_i = Σ_{j≤i} g_j ∈ R^128 (cumulative log-decay inside the chunk, ≤ 0), and
k_i, q_i be the normalized vectors:
```
A_ij  = β_i · Σ_d k_i,d k_j,d · e^{γ_i,d − γ_j,d}          for j < i, else 0
W_i   = β_i · ( v_i − S_0ᵀ (k_i ⊙ e^{γ_i}) )
U     = (I + A)^{−1} W                                       # forward substitution, C×128
o_i   = scale · [ S_0ᵀ (q_i ⊙ e^{γ_i}) + Σ_{j≤i} (Σ_d q_i,d k_j,d e^{γ_i,d − γ_j,d}) u_j ]
S_C   = diag(e^{γ_C}) S_0 + Σ_j (k_j ⊙ e^{γ_C − γ_j}) u_jᵀ
```
Numerics: γ can reach −5·64 = −320, so never form e^{−γ_j} on its own (it overflows fp32). Only use the
differences γ_i − γ_j ≤ 0. Those are per-channel weights, so the products above are weighted dots rather than plain
GEMMs. Split into sub-chunks, or bound them using g ≥ −5 (that's what the lower bound / safe_gate is for). Any
correct C gives the same math, with different rounding.

Padding [file]: KDA layers take a 0/1 [B,T] mask and run variable-length sequences. Left padding is assumed, and the
mask is dropped once the cache is non-empty. A one-sequence engine can ignore this.

## 4. MLA on full-attention layers (24 layers): NoPE, dense, gated

### 4.1 Tensors (bf16; self_attn is on the ignore list)
| tensor | shape |
|---|---|
| q_a_proj | [1536, 7168] |
| q_a_layernorm.weight | [1536], **eps 1e-6** (built without eps → class default) |
| q_b_proj | [18432, 1536] = 96 heads × [128 nope ‖ 64 rot] |
| kv_a_proj_with_mqa | [576, 7168] = [512 c_kv ‖ 64 k_rot] |
| kv_a_layernorm.weight | [512], **eps 1e-6** |
| kv_b_proj | [24576, 512] = 96 heads × [128 k_nope ‖ 128 v] |
| g_proj | [12288, 7168] (output gate) |
| o_proj | [7168, 12288] |

### 4.2 Math, per token t, input x = RMSNorm_in(·) [file]
```
q      = q_b_proj( RMSNorm_1e-6( q_a_proj x ) )      → [96, 192] = [q_nope 128 | q_r 64]
[c, kr]= kv_a_proj_with_mqa x                         → c ∈ R^512, kr ∈ R^64 (kr is NOT normalized)
c̄      = RMSNorm_1e-6(c)
[kn_h, v_h] = kv_b_proj(c̄) split per head           → 128 | 128
k_h    = [kn_h | kr]                                  # kr shared by all 96 heads, not rotated
score_h(t,s) = (q_t,h · k_s,h) / sqrt(192)           # 1/sqrt(192) = 0.0721688
P      = softmax_s over s ≤ t (causal), fp32
o_h    = Σ_s P · v_s,h                                → [96,128] bf16, flatten head-major → [12288]
out    = o_proj( o ⊙ sigmoid(g_proj x) )              # gate on the normed layer input, bf16 multiply
```
- **No RoPE**: `rotary_emb = None`, `mla_use_nope` is asserted true, and position_ids are accepted but ignored. The
  "rope" 64 dims are just a position-free, head-shared (MQA-style) key part. Neither theta nor YaRN applies.
- num_key_value_heads = 96, so there is no GQA repeat. The flash path pads v to 192 and then slices it back, which
  changes nothing mathematically.
- Dense attention over every past token. No sparse/top-k selection.

### 4.3 Absorbed form (the same math)
Split kv_b_proj into W_UK[h] ∈ R^{128×512} and W_UV[h] ∈ R^{128×512}. Then
`score = ( (W_UK[h]ᵀ q_nope_h) · c̄_s + q_r_h · kr_s ) / sqrt(192)` and `o_h = W_UV[h] (Σ_s P c̄_s)`.
This gives different bf16 rounding than the reference, which computes kv_b_proj on c̄ and then attends.

### 4.4 KV cache per token per MLA layer
Absorbed: c̄ (512) ‖ kr (64) = 576 bf16. The HF reference caches the expanded K [96,192] and V [96,128] instead,
which is 30,720 values.

## 5. MoE (layers 1..92)

Input x = RMSNorm_post(·), bf16 [7168].

### 5.1 Router [file]
```
logits = gate.weight (896×7168) · x            # both upcast to fp32
s      = sigmoid(logits)                       # fp32 [896]
choice = s + e_score_correction_bias           # bias only used for selection
idx    = top-16 of choice                      # n_group = 1: no group limit (the group path needs n_group > 1)
w      = s[idx] / (Σ s[idx] + 1e-20)           # renormalize (moe_renormalize, top_k > 1)
w      = w · 1.0                               # routed_scaling_factor
```
topk_method "noaux_tc" just means this bias-for-choice selection. Ties and order: unsorted top-k, and order doesn't
matter. Size check [derived]: gate.weight is **not fp32** (an fp32 version would overflow the index total_size), so
it's 2 bytes, presumably bf16. The bias dtype is unverified.

### 5.2 Latent routed path [file]
```
z   = routed_expert_down_proj x                # [3584] bf16   (7168 → 3584, once per token, shared by all experts)
e_k = Expert_k(z)                              # [3584] bf16 for each chosen expert
y   = bf16( Σ_k w_k · fp32(e_k) )              # combine in fp32
y   = RMSNorm_1e-5(y; routed_expert_norm)      # latent_moe_use_norm = true; [3584]
y   = routed_expert_up_proj y                  # 3584 → 7168
```
Naming trap: `routed_expert_down_proj` maps H→latent and `routed_expert_up_proj` maps latent→H. The router
reads the full 7168 x, not z.

### 5.3 Expert MLP (896 per layer) [file]
`Expert(z) = w2( situ(w1 z, w3 z) )`, where w1 = gate [3072×3584], w3 = up [3072×3584], w2 = down [3584×3072].

### 5.4 Shared experts [file]
One MLP at full width: `shared = down_proj( situ(gate_proj x, up_proj x) )`, with gate/up [6144×7168] and down
[7168×6144], bf16. Its input is the same x as the router (not the latent).

### 5.5 Output
`m = bf16(y + shared)`.

### 5.6 Dense MLP, layer 0
`m = down_proj( situ(gate_proj x, up_proj x) )`, with gate/up [33792×7168] and down [7168×33792], bf16.

## 6. situ activation [file: `SituAndMul`]
For gate a and up b (both bf16 outputs of the two projections), in fp32:
```
situ(a, b) = bf16( 4·tanh(a/4) · sigmoid(a) · 25·tanh(b/25) )
```
So it's SiLU with the linear factor soft-capped at ±4, times an up value soft-capped at ±25. β = 4.0 and linear β =
25.0 come from the config. If β were missing the code would use 1.0, and the up cap applies only when linear β is
set. It's used in **all three** places: the routed experts, the shared experts, and the layer-0 dense MLP (all read
`hidden_act = "situ"`). Gate first: w1 / gate_proj is a, and w3 / up_proj is b.

## 7. Weight formats

### 7.1 What is quantized
Only the routed experts: `layers.{1..92}.block_sparse_moe.experts.{0..895}.w{1,2,3}`. Each has `weight_packed` +
`weight_scale` (92·896 = 82,432 of each, per w). Everything else is a plain `.weight`, presumably bf16.

The `ignore` list covers `self_attn.*` (all KDA and MLA tensors), `shared_experts.*`, `mlp.(gate|up|gate_up|down)_proj`
(the layer-0 dense MLP), `lm_head`, and vision. Some modules are never quantized because they aren't nn.Linear:
embed_tokens, all norms, gate.weight/bias, A_log, dt_bias, and the conv weights.

**Trap:** `routed_expert_{down,up}_proj`, `self_attention_res_proj`, `mlp_res_proj`, and `output_attn_res_proj` are
nn.Linear and aren't on the ignore list, but the index stores them as plain `.weight`, not packed. Decide from
whether `weight_packed` is present, not from the ignore list.

The README says "MXFP4 weights / MXFP8 activations (QAT)". The quant config has `input_activations: null` and the
reference runs bf16 activations (W4A16). Which GEMMs get MXFP8 activations in Moonshot's serving setup isn't stated
anywhere in the inputs.

### 7.2 MXFP4 layout (compressed-tensors `mxfp4-pack-quantized`, group 32, symmetric, scale uint8)
For a logical weight W[out, in]:
- `weight_packed`: uint8 [out, in/2], row-major. Byte j of row o holds W[o, 2j] in the **low nibble** (bits 0–3) and
  W[o, 2j+1] in the **high nibble** (bits 4–7) [lib convention of compressed-tensors fp4 packing].
- Nibble code: bit 3 = sign, bits 0–2 = magnitude index into E2M1 `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`. Codes 8–15 are
  the negatives (8 = −0).
- `weight_scale`: uint8 [out, in/32], E8M0. Scale = 2^(s − 127). Groups run along the input (reduction) dimension,
  per row: group g of row o covers columns 32g..32g+31 [lib convention].
- `W[o,i] = E2M1(nibble(o,i)) · 2^(scale[o, i/32] − 127)`. This is exact in bf16 for any scale that doesn't overflow
  (2-bit mantissa × power of two). There's no global/second-level scale tensor and no zero point.

Per-expert shapes [derived from the config: README has no shapes and the index stores names only]:
| tensor | logical | weight_packed | weight_scale |
|---|---|---|---|
| w1 (gate) | [3072, 3584] | uint8 [3072, 1792] | uint8 [3072, 112] |
| w3 (up) | [3072, 3584] | uint8 [3072, 1792] | uint8 [3072, 112] |
| w2 (down) | [3584, 3072] | uint8 [3584, 1536] | uint8 [3584, 96] |

Per expert that's 16.52 MB packed + 1.03 MB scales. Over all experts: 1.447 TB. That fits the 1.561 TB total once the
~113.5 GB of bf16 non-expert text tensors and ~0.9 GB for vision are added [derived]. Check the nibble order once
on real weights: dequantize one expert, run one token, and compare against a reference matmul. Swapped nibbles give
garbage, not a small error.

### 7.3 Other tensors (bf16)
embed_tokens [163840, 7168]; lm_head [163840, 7168] (untied); norm, output_attn_res_norm [7168];
output_attn_res_proj [1, 7168]; per layer: input_layernorm, post_attention_layernorm, self_attention_res_norm,
mlp_res_norm [7168], and self_attention_res_proj, mlp_res_proj [1, 7168]. MoE: gate.weight [896, 7168],
e_score_correction_bias [896], routed_expert_down_proj [3584, 7168], routed_expert_up_proj [7168, 3584],
routed_expert_norm [3584].

## 8. Head, tokens, chat encoding

### 8.1 Head
Final RMSNorm (eps 1e-5), then lm_head, giving bf16 logits. Sampling per the README: temperature 1.0, top-p 0.95 for
single-step tasks and 1.0 for agentic ones. generation_config sets only max_length 1,048,576 and eos 163586.

### 8.2 Special token ids (tokenizer_config)
| id | token | note |
|---|---|---|
| 163584 | [BOS] | config bos |
| 163585 | [EOS] | tokenizer eos_token |
| 163586 | `<\|end_of_msg\|>` | **config/generation eos: stop on this** |
| 163587 | `<\|open\|>` | added, special:false |
| 163588 | `<\|close\|>` | added, special:false |
| 163589 | `<\|sep\|>` | added, special:false |
| 163590 / 163591 | [start_header_id] / [end_header_id] | unused by encoder |
| 163593 | [EOT] | unused by encoder |
| 163602–163605 | media_begin / media_content / media_end / media_pad | vision; 163605 = media placeholder |
| 163649 | `<osagent_mode>` | unused by encoder |
| 163838 / 163839 | [UNK] / [PAD] | pad id 163839 |

The tokenizer is tiktoken-based (`tokenization_kimi.TikTokenTokenizer`). Its vocab file and split regex are **not in
the inputs**.

### 8.3 Encoding rules (encoding_k3.py, "XTML")
The output is a list of segments. Control segments (`<|open|>`, `<|close|>`, `<|sep|>`, `<|end_of_msg|>`) map
straight to their ids. All other text, including tag names, attribute text, user/tool/assistant content, and
reasoning, is ordinary BPE text. Special-token strings that show up inside text must **not** become control ids.
No newlines are added between pieces, and no BOS is emitted by the encoder. Whether the tokenizer adds [BOS] is
unknown, because tokenization_kimi.py wasn't provided.

Building blocks (`⟨o⟩`=163587, `⟨c⟩`=163588, `⟨s⟩`=163589, `⟨e⟩`=163586):
```
open(tag, attrs) = ⟨o⟩ tag { ␠key="value" } ⟨s⟩       # value escapes: & → &amp;  " → &quot;
close(tag)       = ⟨c⟩ tag ⟨s⟩
```
Messages:
```
user:      open(message, role="user"[, name=…])  CONTENT  close(message) ⟨e⟩
system:    open(message, role="system"[, name=…]) CONTENT close(message) ⟨e⟩
assistant: open(message, role="assistant"[, name=…])
             [thinking on:  open(think) REASONING close(think)]    # always present when thinking on, even if empty
             open(response) CONTENT close(response)
             [tool calls: see below]
           close(message) ⟨e⟩
tool:      open(message, role="tool", tool=NAME, index=i) CONTENT close(message) ⟨e⟩
```
Content isn't escaped. Content lists concatenate their text parts, and images become placeholders.

Example of a plain turn with thinking on (generation prompt at the end):
```
⟨o⟩message role="user"⟨s⟩Hi⟨c⟩message⟨s⟩⟨e⟩⟨o⟩message role="assistant"⟨s⟩⟨o⟩think⟨s⟩
```
The model then writes: `REASONING⟨c⟩think⟨s⟩⟨o⟩response⟨s⟩ANSWER⟨c⟩response⟨s⟩[tools]⟨c⟩message⟨s⟩⟨e⟩`.

Thinking off: the history drops the think channel entirely, and the generation prompt ends with
`…role="assistant"⟨s⟩⟨o⟩response⟨s⟩`. The README says K3 "always has thinking enabled". With thinking on, the history
keeps the reasoning of **every** past assistant turn (preserved thinking).

Tool calls, inside the assistant message after the response:
```
open(tools)
  open(call, tool=NAME, index=i)                  # i = 1-based position in tool_calls
    open(argument, key=K, type=T) TEXT close(argument)   # one per top-level argument key
  close(call)
close(tools)
```
- T is one of string | number | boolean | null | object | array.
- TEXT is the raw decoded string for strings (no quotes, no JSON escaping). For anything else it's the original JSON
  literal text, exactly as passed in the arguments string. Only the top level is parsed, and nested values are kept
  byte for byte.
- If the arguments string isn't a well-formed JSON object, the call body is
  `open(json, type="object") RAW close(json)` instead.

Tool results: one message per result, with `index` counting the tool messages since the last assistant turn (1-based).
Results are first reordered to match the order of `tool_call_id` in the preceding `tool_calls`, and `tool` is set to
the matched call's name.

Preamble order:
1. Tool declaration, if tools are given: `open(message, role="system", type="tool-declare")` +
   `# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n{JSON}\n```` + close + ⟨e⟩.
   JSON is the tools list with dict keys sorted recursively, compact separators, and non-ASCII kept as is.
2. If thinking is on and `thinking_effort` ∈ {low, high, max}: a system message with type="thinking-effort" (the
   exact text is in the file; it names `low/medium/high/max` but only accepts low/high/max).
3. The messages themselves.
4. If tool_choice is required or none: a system message with type="tool-choice".
5. If response_format is json_object or json_schema: a system message with type="response-format".
6. The generation prompt.

A `role="system"` message that carries `tools` instead renders the "## New Tools Available … extended tools" dynamic
declaration with type="tool-declare". Internal system messages strip their body.

## 9. Ambiguities / not verified
1. KDA gate formula, L2-norm eps (1e-6), default q scale (128^−0.5), state dtype (fp32), and conv arithmetic all
   live inside fla kernels that aren't in the inputs. Validate them against a reference run.
2. Whether the decode kernel uses the same lower-bounded gate as the chunk kernel (it gets lower_bound but not
   safe_gate).
3. MXFP4 nibble order and E8M0 bias 127 follow the compressed-tensors convention. The inputs don't spell them out.
4. Tensor shapes and dtypes: the index has names only. Shapes are derived from the config, and the dtypes of
   A_log, dt_bias, and e_score_correction_bias are unknown. The byte total does rule out an fp32 gate.weight.
5. The quant ignore list doesn't match the index (routed_expert_* and *_res_proj are unquantized). Trust the index.
6. Which activations the QAT'd MXFP8 applies to isn't specified. The reference is bf16 activations.
7. Tokenizer: the vocab file and regex are missing, and so is whether BOS is prepended. It's also unclear whether
   adjacent text segments (` role`, `="`, value, `"`) are BPE'd separately or merged first. That changes token ids.
   The module only says callers "consume EncodeSegment objects directly".
8. config `eos_token_id` = 163586 (`<|end_of_msg|>`), while the tokenizer's eos_token is [EOS] 163585. Stop on 163586,
   and treat 163585 as a stop too to be safe.
