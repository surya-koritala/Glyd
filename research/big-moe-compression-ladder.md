# Big MoE compression ladder: GLM-5.3-Flash and Nemotron-3-Super-120B (Linear GLY-127)

Measured 2026-10-09 on sampled real tensors (HTTP Range reads, 5.1 GB total, no full downloads, no tokens, no pods). Not committed.

Tags: **[H]** read from safetensors headers, **[M]** measured on sampled tensors, **[C]** computed from [H] sizes and [M] rates. Every number in the tables is one of these: group sizes/params are [H]; entropies, RMSE, Glyd real bytes are [M]; totals and plans are [C].

## Method

1. Pinned shas via `https://huggingface.co/api/models/<id>`: GLM `zai-org/GLM-5.3-Flash` @ `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`; Nemotron `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` @ `2dc98e2afe4face0e4ce40972a915c45368bd34a`. Range reads of the 8-byte length + JSON header of every shard (GLM 62 shards / 76,108 tensors; Nemotron 50 shards / 42,683 tensors) give name, dtype, shape, byte offsets [H].
2. Grouping by tensor name/dtype. GLM: routed experts (FP8, layers 3-44 plus the 288 experts of the MTP layer 45), shared expert (FP8), MLA attention FP8 (q_a, q_b, kv_a, o), MLA `kv_b_proj` (BF16), indexer (BF16), KDA linear attention (BF16 q/k/v/o and the low-rank f/g/b projections), dense MLP layers 0-2 (FP8), embeddings+lm_head (BF16), MTP non-expert (layer 45: eh_proj, attn, shared expert, indexer, router), vision tower (BF16), router `mlp.gate.weight` (BF16), norms/other (norms, hc_*, A_log, dt_bias, conv, biases; never sampled, kept at original size in every plan). Nemotron: routed experts (up/down, incl. MTP's 512), shared expert, Mamba (`in_proj`/`out_proj`), attention (q/k/v/o, 8 layers), latent MoE projections (`fc1/fc2_latent_proj`), embeddings+lm_head, MTP non-expert, router, norms/other. FP8 `weight_scale_inv` tensors (F32, 128x128 blocks) are counted with their weight's group.
3. Sampling: per kind (tensor name with layer/expert ids removed) first/middle/last layer; routed experts 3 layers x 4 random expert ids x each projection (seed 7); embeddings/head two 64 MB windows; tensors above 64 MB read as one contiguous 64 MB window of whole rows (start a multiple of 128 rows so FP8 scale rows align), the matching scale rows read too. Manifest (shard, absolute byte range, row window, sha256 of every sampled byte range, scale range) in `big-moe-ladder-manifest-{glm,nemo}.json`. Scripts and raw per-sample results in `big-moe-ladder-scripts/` (a venv with numpy, ml_dtypes, huggingface_hub, requests).
4. Lossless [M]: order-0 Shannon entropy per sampled tensor of the 16-bit word (H16), the 8-bit exponent, and sign+mantissa byte; FP8: entropy of the 8-bit codes. Group value = parameter-weighted mean over kinds (header param counts), mean over a kind's samples. Plug-in estimator bias is below 0.01 bit (>=1M values per sample). This is a bound for a static per-tensor order-0 coder; context modelling could beat it, real coders pay a little more. **Glyd real:** `glyd_gpu::pack::pack_tiered` (the penguin layout, built from /Users/suryakoritala/glyd-v030-check, `tiered_bytes`) run on every sampled BF16 matrix whose shape it accepts (235 tensors / windows, 3.85 GB raw; all 235 decoded bit-exact with `unpack_tiered`). GLM router tensors (288 rows, not a multiple of 64) cannot be packed and show n/a. FP8 was not run through a Glyd packer (no FP8 lossless layout in pack.rs); only the entropy bound is given.
5. Lossy [M]: the rule of `glyd-calib-spike/research/lossy-2026/kld_study.py` (`int_tensor`/`fit_scale`, fit `mse`), ported to numpy: runs of 32 along a row, symmetric codes -q..q with q=2^(N-1)-1, one bf16 scale per run chosen from 16 clip ratios 0.70..1.0 of absmax/q for least squared error (bf16 scale as the engine). GLM FP8 weights dequantised to float32 (code x block scale) first; error is vs that FP8 original (no extra bf16 rounding). Computed on at most the first 16M values of each sample (whole rows). Relative RMSE per group is energy-weighted over kinds. Stored bits/weight = N + 16/32; "entropy-coded" = entropy of the codes + 16/32 for the scale (scale kept bf16). N = 6, 5, 4, 3, 2.
6. Sizes [C]: kind params [H] x measured bits. Kinds not sampled, and norms/other and (in all-X rows) router, stay at original header bytes. Requantised FP8 groups drop their F32 scale tensors (replaced by the bf16 run scales). Vision (0.56 B params) follows the always-on level; its stored size at orig is 1.13 GB, subtract it for a text-only figure.

Caveat: weight RMSE is not output quality. It is almost identical for every group (a rounding of bell-shaped weights), so it ranks groups only weakly; KL against the bf16/FP8 model is needed for real sensitivity.

## Sampled manifest summary

| model | samples | bytes read | by group (tensors/windows, MB) |
|---|---|---|---|
| GLM | 264 | 3.66 GB | routed_experts 36 / 302; shared_expert 9 / 76; attn_mla_fp8 12 / 302; attn_mla_bf16 3 / 101; attn_indexer 12 / 45; attn_kda 27 / 826; dense_mlp 9 / 453; embed_head 4 / 268; mtp_other 25 / 244; vision 124 / 1039; router 3 / 7 |
| Nemotron | 63 | 1.46 GB | routed_experts 24 / 132; shared_expert 6 / 264; mamba 6 / 403; attn 12 / 214; latent_proj 3 / 25; embed_head 4 / 268; mtp_other 5 / 138; router 3 / 13 |

Total 5.12 GB read, under the 6 GB budget. Routed experts: 12 sampled per projection (3 layers x 4 experts) out of 12,384 (GLM) and 20,480 (Nemotron) per projection; "% params sampled" below means the share of params in kinds with at least one sample; expert entropy varies little between samples.

## GLM-5.3-Flash
`zai-org/GLM-5.3-Flash` @ `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. Sampled 264 tensors / windows, 3.66 GB read via HTTP Range (manifest: `big-moe-ladder-manifest-glm.json`).

### Groups (read from headers; params exclude FP8 scale tensors)
| group | tensors | params (B) | stored GB | dtype | samples | % params sampled |
|---|---|---|---|---|---|---|
| routed_experts | 37152 | 311.654 | 311.73 | F8_E4M3 | 36 | 100% |
| shared_expert | 126 | 1.057 | 1.06 | F8_E4M3 | 9 | 100% |
| attn_mla_fp8 | 44 | 1.107 | 1.11 | F8_E4M3 | 12 | 100% |
| attn_mla_bf16 | 11 | 0.185 | 0.37 | BF16 | 3 | 100% |
| attn_indexer | 44 | 0.082 | 0.16 | BF16 | 12 | 100% |
| attn_kda | 306 | 4.679 | 9.36 | BF16 | 27 | 100% |
| dense_mlp | 9 | 0.453 | 0.45 | F8_E4M3 | 9 | 100% |
| embed_head | 2 | 1.269 | 2.54 | BF16 | 4 | 100% |
| mtp_other | 25 | 0.185 | 0.24 | BF16/F32/F8_E4M3 | 25 | 100% |
| vision | 347 | 0.564 | 1.13 | BF16 | 124 | 97% |
| router | 42 | 0.050 | 0.10 | BF16 | 3 | 100% |
| other | 662 | 0.039 | 0.08 | BF16/F32 | 0 | 0% |
| **total** | | **321.32** | **328.33** | | | |

### Lossless bounds (measured on samples; order-0 entropy per sampled tensor, param-weighted over kinds)
BF16 groups: H16 = entropy of the 16-bit word; H(exp) = of the 8-bit exponent; H(s+m) = of sign+7-bit mantissa; "exp-coded" = H(exp)+8 (exponent entropy-coded, sign+mantissa raw); Glyd real = bytes of `pack_tiered` (penguin) / raw, measured on the same sampled tensors.
| group | H16 | H(exp) | H(s+m) | saving vs 16b (H16) | exp-coded bits | Glyd penguin real (bits/w) | Glyd saving |
|---|---|---|---|---|---|---|---|
| attn_mla_bf16 | 10.558 | 2.620 | 7.975 | 34.0% | 10.620 | 10.802 | 32.5% |
| attn_indexer | 6.725 | 2.625 | 4.139 | 58.0% | 10.625 | 10.801 | 32.5% |
| attn_kda | 7.258 | 2.593 | 5.132 | 54.6% | 10.593 | 10.807 | 32.5% |
| embed_head | 10.495 | 2.573 | 7.972 | 34.4% | 10.573 | 10.779 | 32.6% |
| mtp_other | 10.524 | 2.604 | 7.973 | 34.2% | 10.604 | 10.790 | 32.6% |
| vision | 10.518 | 2.589 | 7.974 | 34.3% | 10.589 | 10.765 | 32.7% |
| router | 10.491 | 2.569 | 7.972 | 34.4% | 10.569 | n/a | n/a |

FP8 groups: H8 = entropy of the 8-bit E4M3 codes (bits/weight); scale overhead = 128x128 block F32 scale = 0.002 bits/weight.
| group | H8 | saving vs 8b | scale overhead bits/w |
|---|---|---|---|
| routed_experts | 6.473 | 19.1% | 0.0020 |
| shared_expert | 6.588 | 17.7% | 0.0020 |
| attn_mla_fp8 | 6.483 | 19.0% | 0.0020 |
| dense_mlp | 6.517 | 18.5% | 0.0020 |
| mtp_other | 6.721 | 16.0% | 0.0013 |

### Lossy: block-32 symmetric IntN, bf16 scale chosen to minimise MSE (measured on samples; error vs the stored original, GLM FP8 dequantised first)
Cell = relative RMSE ||W-W^||/||W|| / stored bits per weight (N+0.5) / entropy-coded bits per weight (entropy of codes + 0.5 for the scale).
| group | N=6 | N=5 | N=4 | N=3 | N=2 |
|---|---|---|---|---|---|
| routed_experts | 0.0217 / 6.5 / 6.19 | 0.0440 / 5.5 / 5.20 | 0.0918 / 4.5 / 4.17 | 0.1965 / 3.5 / 3.13 | 0.4898 / 2.5 / 1.89 |
| shared_expert | 0.0226 / 6.5 / 6.12 | 0.0464 / 5.5 / 5.14 | 0.0964 / 4.5 / 4.11 | 0.2044 / 3.5 / 3.08 | 0.4965 / 2.5 / 1.84 |
| attn_mla_fp8 | 0.0217 / 6.5 / 6.18 | 0.0444 / 5.5 / 5.20 | 0.0925 / 4.5 / 4.16 | 0.1981 / 3.5 / 3.12 | 0.4930 / 2.5 / 1.88 |
| attn_mla_bf16 | 0.0217 / 6.5 / 6.21 | 0.0440 / 5.5 / 5.20 | 0.0917 / 4.5 / 4.17 | 0.1959 / 3.5 / 3.13 | 0.4889 / 2.5 / 1.89 |
| attn_indexer | 0.0223 / 6.5 / 6.17 | 0.0454 / 5.5 / 5.17 | 0.0943 / 4.5 / 4.14 | 0.2015 / 3.5 / 3.11 | 0.4980 / 2.5 / 1.86 |
| attn_kda | 0.0224 / 6.5 / 6.18 | 0.0454 / 5.5 / 5.19 | 0.0942 / 4.5 / 4.16 | 0.2007 / 3.5 / 3.12 | 0.4941 / 2.5 / 1.88 |
| dense_mlp | 0.0217 / 6.5 / 6.19 | 0.0441 / 5.5 / 5.20 | 0.0920 / 4.5 / 4.17 | 0.1971 / 3.5 / 3.12 | 0.4920 / 2.5 / 1.89 |
| embed_head | 0.0217 / 6.5 / 6.21 | 0.0440 / 5.5 / 5.21 | 0.0917 / 4.5 / 4.17 | 0.1959 / 3.5 / 3.13 | 0.4893 / 2.5 / 1.90 |
| mtp_other | 0.0246 / 6.5 / 6.18 | 0.0461 / 5.5 / 5.18 | 0.0917 / 4.5 / 4.15 | 0.1932 / 3.5 / 3.11 | 0.4770 / 2.5 / 1.87 |
| vision | 0.0219 / 6.5 / 6.20 | 0.0444 / 5.5 / 5.19 | 0.0925 / 4.5 / 4.16 | 0.1976 / 3.5 / 3.12 | 0.4924 / 2.5 / 1.88 |
| router | 0.0217 / 6.5 / 6.21 | 0.0441 / 5.5 / 5.20 | 0.0918 / 4.5 / 4.17 | 0.1962 / 3.5 / 3.13 | 0.4894 / 2.5 / 1.89 |

Per-kind detail (vision tower omitted: 124 kinds, all near rel4 0.092; N=4 rel RMSE, N=3, H16 or H8, samples):
| group | kind | params (M) | rel4 | rel3 | H16/H8 | n |
|---|---|---|---|---|---|---|
| routed_experts | mlp.experts.N.down_proj.weight | 103884.5 | 0.0925 | 0.1978 | 6.492 | 12 |
| routed_experts | mlp.experts.N.gate_proj.weight | 103884.5 | 0.0914 | 0.1957 | 6.469 | 12 |
| routed_experts | mlp.experts.N.up_proj.weight | 103884.5 | 0.0914 | 0.1959 | 6.458 | 12 |
| shared_expert | mlp.shared_experts.down_proj.weight | 352.3 | 0.1049 | 0.2194 | 6.590 | 3 |
| shared_expert | mlp.shared_experts.gate_proj.weight | 352.3 | 0.0920 | 0.1970 | 6.591 | 3 |
| shared_expert | mlp.shared_experts.up_proj.weight | 352.3 | 0.0941 | 0.2003 | 6.583 | 3 |
| attn_mla_fp8 | self_attn.kv_a_proj_with_mqa.weight | 23.1 | 0.0912 | 0.1956 | 6.478 | 3 |
| attn_mla_fp8 | self_attn.o_proj.weight | 738.2 | 0.0913 | 0.1957 | 6.485 | 3 |
| attn_mla_fp8 | self_attn.q_a_proj.weight | 69.2 | 0.0912 | 0.1954 | 6.486 | 3 |
| attn_mla_fp8 | self_attn.q_b_proj.weight | 276.8 | 0.0944 | 0.2020 | 6.477 | 3 |
| attn_mla_bf16 | self_attn.kv_b_proj.weight | 184.5 | 0.0917 | 0.1959 | 10.558 | 3 |
| attn_indexer | self_attn.indexer.index_kpool_compress_gate | 5.8 | 0.0919 | 0.1966 | 6.543 | 3 |
| attn_indexer | self_attn.indexer.weights_proj.weight | 1.4 | 0.0922 | 0.1967 | 10.477 | 3 |
| attn_indexer | self_attn.indexer.wk.weight | 5.8 | 0.0912 | 0.1954 | 7.772 | 3 |
| attn_indexer | self_attn.indexer.wq_b.weight | 69.2 | 0.0947 | 0.2023 | 6.574 | 3 |
| attn_kda | self_attn.b_proj.weight | 8.9 | 0.0958 | 0.2037 | 10.518 | 3 |
| attn_kda | self_attn.f_a_proj.weight | 17.8 | 0.0926 | 0.1977 | 10.470 | 3 |
| attn_kda | self_attn.f_b_proj.weight | 35.7 | 0.0909 | 0.1945 | 10.657 | 3 |
| attn_kda | self_attn.g_a_proj.weight | 17.8 | 0.0928 | 0.1982 | 10.469 | 3 |
| attn_kda | self_attn.g_b_proj.weight | 35.7 | 0.0912 | 0.1948 | 10.635 | 3 |
| attn_kda | self_attn.k_proj.weight | 1140.9 | 0.0924 | 0.1973 | 7.165 | 3 |
| attn_kda | self_attn.o_proj.weight | 1140.9 | 0.0989 | 0.2087 | 7.256 | 3 |
| attn_kda | self_attn.q_proj.weight | 1140.9 | 0.0924 | 0.1975 | 7.162 | 3 |
| attn_kda | self_attn.v_proj.weight | 1140.9 | 0.0926 | 0.1980 | 7.110 | 3 |
| dense_mlp | mlp.down_proj.weight | 151.0 | 0.0928 | 0.1987 | 6.436 | 3 |
| dense_mlp | mlp.gate_proj.weight | 151.0 | 0.0917 | 0.1964 | 6.579 | 3 |
| dense_mlp | mlp.up_proj.weight | 151.0 | 0.0916 | 0.1963 | 6.537 | 3 |
| embed_head | lm_head.weight | 634.4 | 0.0919 | 0.1964 | 10.483 | 2 |
| embed_head | model.language_model.embed_tokens.weight | 634.4 | 0.0911 | 0.1947 | 10.506 | 2 |
| mtp_other | eh_proj.weight | 33.6 | 0.0811 | 0.1501 | 10.492 | 1 |
| mtp_other | mlp.gate.weight | 1.2 | 0.0936 | 0.1999 | 10.535 | 1 |
| mtp_other | mlp.shared_experts.down_proj.weight | 8.4 | 0.1153 | 0.2404 | 6.880 | 1 |
| mtp_other | mlp.shared_experts.gate_proj.weight | 8.4 | 0.0992 | 0.2060 | 6.982 | 1 |
| mtp_other | mlp.shared_experts.up_proj.weight | 8.4 | 0.1034 | 0.2109 | 6.899 | 1 |
| mtp_other | self_attn.indexer.index_kpool_compress_ape | 0.0 | 0.0655 | 0.1348 | 8.274 | 1 |
| mtp_other | self_attn.indexer.index_kpool_compress_gate | 0.5 | 0.0959 | 0.2044 | 10.532 | 1 |
| mtp_other | self_attn.indexer.weights_proj.weight | 0.1 | 0.0953 | 0.2020 | 10.601 | 1 |
| mtp_other | self_attn.indexer.wk.weight | 0.5 | 0.0919 | 0.1965 | 10.444 | 1 |
| mtp_other | self_attn.indexer.wq_b.weight | 6.3 | 0.0919 | 0.1964 | 10.490 | 1 |
| mtp_other | self_attn.kv_a_proj_with_mqa.weight | 2.1 | 0.0916 | 0.1963 | 6.671 | 1 |
| mtp_other | self_attn.kv_b_proj.weight | 16.8 | 0.0937 | 0.1995 | 10.609 | 1 |
| mtp_other | self_attn.o_proj.weight | 67.1 | 0.0931 | 0.1991 | 6.741 | 1 |
| mtp_other | self_attn.q_a_proj.weight | 6.3 | 0.0923 | 0.1974 | 6.526 | 1 |
| mtp_other | self_attn.q_b_proj.weight | 25.2 | 0.0920 | 0.1967 | 6.523 | 1 |
| router | mlp.gate.weight | 49.5 | 0.0918 | 0.1962 | 10.491 | 3 |

### Total size if every quantisable group used level X (computed: header params x measured bits; router and norms/other stay original)
| level | GB | vs original |
|---|---|---|
| original | 328.33 | 100.0% |
| lossless entropy bound (H16/H8) | 261.69 | 79.7% |
| Glyd penguin (real packer ratio on BF16 groups; FP8 groups at H8) | 263.90 | 80.4% |
| all groups 6-bit stored (6.5 bpw) | 261.14 | 79.5% |
| all groups 5-bit stored (5.5 bpw) | 220.99 | 67.3% |
| all groups 4-bit stored (4.5 bpw) | 180.83 | 55.1% |
| all groups 3-bit stored (3.5 bpw) | 140.67 | 42.8% |
| all groups 2-bit stored (2.5 bpw) | 100.51 | 30.6% |
| all groups 6-bit entropy-coded | 248.60 | 75.7% |
| all groups 5-bit entropy-coded | 208.98 | 63.7% |
| all groups 4-bit entropy-coded | 167.54 | 51.0% |
| all groups 3-bit entropy-coded | 125.69 | 38.3% |
| all groups 2-bit entropy-coded | 75.97 | 23.1% |

### Mixed plans (computed). always-on = every group except routed experts, router, norms/other (attention/Mamba, shared expert, dense, latent proj, embeddings+head, MTP non-expert, vision)
| plan | stored GB | experts entropy-coded GB (always-on stored) | everything entropy-coded GB | experts GB (stored) |
|---|---|---|---|---|
| lossless only (Glyd penguin / H8 on all) | 263.90 | | | |
| always-on 6-bit, experts 5-bit | 222.25 | 210.62 | 210.24 | 214.26 |
| always-on 6-bit, experts 4-bit | 183.29 | 170.42 | 170.03 | 175.31 |
| always-on 6-bit, experts 3-bit | 144.33 | 129.82 | 129.44 | 136.35 |
| always-on 6-bit, experts 2-bit | 105.38 | 81.58 | 81.20 | 97.39 |
| always-on 5-bit, experts 4-bit | 182.09 | 169.22 | 168.85 | 175.31 |

## Nemotron-3-Super-120B-A12B-BF16
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` @ `2dc98e2afe4face0e4ce40972a915c45368bd34a`. Sampled 63 tensors / windows, 1.46 GB read via HTTP Range (manifest: `big-moe-ladder-manifest-nemo.json`).

### Groups (read from headers; params exclude FP8 scale tensors)
| group | tensors | params (B) | stored GB | dtype | samples | % params sampled |
|---|---|---|---|---|---|---|
| routed_experts | 41984 | 115.561 | 231.12 | BF16 | 24 | 100% |
| shared_expert | 82 | 1.806 | 3.61 | BF16 | 6 | 100% |
| mamba | 80 | 4.383 | 8.77 | BF16 | 6 | 100% |
| attn | 32 | 0.285 | 0.57 | BF16 | 12 | 100% |
| latent_proj | 82 | 0.344 | 0.69 | BF16 | 3 | 100% |
| embed_head | 2 | 1.074 | 2.15 | BF16 | 4 | 100% |
| mtp_other | 5 | 0.069 | 0.14 | BF16 | 5 | 100% |
| router | 41 | 0.086 | 0.17 | BF16 | 3 | 100% |
| other | 375 | 0.003 | 0.01 | BF16/F32 | 0 | 0% |
| **total** | | **123.61** | **247.22** | | | |

### Lossless bounds (measured on samples; order-0 entropy per sampled tensor, param-weighted over kinds)
BF16 groups: H16 = entropy of the 16-bit word; H(exp) = of the 8-bit exponent; H(s+m) = of sign+7-bit mantissa; "exp-coded" = H(exp)+8 (exponent entropy-coded, sign+mantissa raw); Glyd real = bytes of `pack_tiered` (penguin) / raw, measured on the same sampled tensors.
| group | H16 | H(exp) | H(s+m) | saving vs 16b (H16) | exp-coded bits | Glyd penguin real (bits/w) | Glyd saving |
|---|---|---|---|---|---|---|---|
| routed_experts | 10.496 | 2.576 | 7.973 | 34.4% | 10.576 | 10.765 | 32.7% |
| shared_expert | 10.773 | 2.842 | 7.973 | 32.7% | 10.842 | 11.284 | 29.5% |
| mamba | 10.541 | 2.612 | 7.973 | 34.1% | 10.612 | 10.801 | 32.5% |
| attn | 10.709 | 2.774 | 7.973 | 33.1% | 10.774 | 11.211 | 29.9% |
| latent_proj | 10.669 | 2.737 | 7.973 | 33.3% | 10.737 | 10.970 | 31.4% |
| embed_head | 10.533 | 2.606 | 7.973 | 34.2% | 10.606 | 10.775 | 32.7% |
| mtp_other | 10.588 | 2.654 | 7.972 | 33.8% | 10.654 | 10.840 | 32.3% |
| router | 10.667 | 2.735 | 7.972 | 33.3% | 10.735 | 10.996 | 31.3% |

### Lossy: block-32 symmetric IntN, bf16 scale chosen to minimise MSE (measured on samples; error vs the stored original, GLM FP8 dequantised first)
Cell = relative RMSE ||W-W^||/||W|| / stored bits per weight (N+0.5) / entropy-coded bits per weight (entropy of codes + 0.5 for the scale).
| group | N=6 | N=5 | N=4 | N=3 | N=2 |
|---|---|---|---|---|---|
| routed_experts | 0.0217 / 6.5 / 6.21 | 0.0442 / 5.5 / 5.20 | 0.0920 / 4.5 / 4.17 | 0.1966 / 3.5 / 3.13 | 0.4899 / 2.5 / 1.89 |
| shared_expert | 0.0238 / 6.5 / 6.10 | 0.0483 / 5.5 / 5.10 | 0.1003 / 4.5 / 4.07 | 0.2112 / 3.5 / 3.05 | 0.5047 / 2.5 / 1.81 |
| mamba | 0.0230 / 6.5 / 6.17 | 0.0468 / 5.5 / 5.16 | 0.0973 / 4.5 / 4.13 | 0.2059 / 3.5 / 3.10 | 0.5007 / 2.5 / 1.86 |
| attn | 0.0233 / 6.5 / 5.95 | 0.0471 / 5.5 / 4.96 | 0.0972 / 4.5 / 3.96 | 0.2052 / 3.5 / 2.95 | 0.4958 / 2.5 / 1.77 |
| latent_proj | 0.0225 / 6.5 / 6.12 | 0.0455 / 5.5 / 5.11 | 0.0940 / 4.5 / 4.09 | 0.1997 / 3.5 / 3.06 | 0.4896 / 2.5 / 1.84 |
| embed_head | 0.0217 / 6.5 / 6.21 | 0.0440 / 5.5 / 5.20 | 0.0916 / 4.5 / 4.17 | 0.1958 / 3.5 / 3.13 | 0.4880 / 2.5 / 1.89 |
| mtp_other | 0.0243 / 6.5 / 6.19 | 0.0478 / 5.5 / 5.18 | 0.0938 / 4.5 / 4.15 | 0.1953 / 3.5 / 3.11 | 0.4819 / 2.5 / 1.87 |
| router | 0.0238 / 6.5 / 6.08 | 0.0480 / 5.5 / 5.08 | 0.0994 / 4.5 / 4.06 | 0.2111 / 3.5 / 3.03 | 0.5092 / 2.5 / 1.81 |

Per-kind detail (vision tower omitted: 124 kinds, all near rel4 0.092; N=4 rel RMSE, N=3, H16 or H8, samples):
| group | kind | params (M) | rel4 | rel3 | H16/H8 | n |
|---|---|---|---|---|---|---|
| routed_experts | mixer.experts.N.down_proj.weight | 57780.7 | 0.0930 | 0.1986 | 10.504 | 12 |
| routed_experts | mixer.experts.N.up_proj.weight | 57780.7 | 0.0911 | 0.1946 | 10.489 | 12 |
| shared_expert | mixer.shared_experts.down_proj.weight | 902.8 | 0.0990 | 0.2101 | 10.704 | 3 |
| shared_expert | mixer.shared_experts.up_proj.weight | 902.8 | 0.1017 | 0.2125 | 10.842 | 3 |
| mamba | mixer.in_proj.weight | 3040.9 | 0.0972 | 0.2052 | 10.542 | 3 |
| mamba | mixer.out_proj.weight | 1342.2 | 0.0976 | 0.2075 | 10.540 | 3 |
| attn | mixer.k_proj.weight | 8.4 | 0.1075 | 0.2222 | 10.734 | 3 |
| attn | mixer.o_proj.weight | 134.2 | 0.0928 | 0.1982 | 10.552 | 3 |
| attn | mixer.q_proj.weight | 134.2 | 0.1007 | 0.2095 | 10.871 | 3 |
| attn | mixer.v_proj.weight | 8.4 | 0.1030 | 0.2187 | 10.611 | 3 |
| latent_proj | mixer.fcN_latent_proj.weight | 343.9 | 0.0940 | 0.1997 | 10.669 | 3 |
| embed_head | backbone.embeddings.weight | 536.9 | 0.0923 | 0.1973 | 10.571 | 2 |
| embed_head | lm_head.weight | 536.9 | 0.0912 | 0.1949 | 10.494 | 2 |
| mtp_other | eh_proj.weight | 33.6 | 0.0950 | 0.1898 | 10.614 | 1 |
| mtp_other | mixer.k_proj.weight | 1.0 | 0.0930 | 0.1984 | 10.507 | 1 |
| mtp_other | mixer.o_proj.weight | 16.8 | 0.0929 | 0.1982 | 10.610 | 1 |
| mtp_other | mixer.q_proj.weight | 16.8 | 0.0931 | 0.1987 | 10.523 | 1 |
| mtp_other | mixer.v_proj.weight | 1.0 | 0.0941 | 0.2009 | 10.518 | 1 |
| router | mixer.gate.weight | 86.0 | 0.0994 | 0.2111 | 10.667 | 3 |

### Total size if every quantisable group used level X (computed: header params x measured bits; router and norms/other stay original)
| level | GB | vs original |
|---|---|---|
| original | 247.22 | 100.0% |
| lossless entropy bound (H16/H8) | 162.29 | 65.6% |
| Glyd penguin (real packer ratio on BF16 groups; FP8 groups at H8) | 166.50 | 67.3% |
| all groups 6-bit stored (6.5 bpw) | 100.44 | 40.6% |
| all groups 5-bit stored (5.5 bpw) | 84.99 | 34.4% |
| all groups 4-bit stored (4.5 bpw) | 69.54 | 28.1% |
| all groups 3-bit stored (3.5 bpw) | 54.08 | 21.9% |
| all groups 2-bit stored (2.5 bpw) | 38.63 | 15.6% |
| all groups 6-bit entropy-coded | 95.86 | 38.8% |
| all groups 5-bit entropy-coded | 80.27 | 32.5% |
| all groups 4-bit entropy-coded | 64.31 | 26.0% |
| all groups 3-bit entropy-coded | 48.28 | 19.5% |
| all groups 2-bit entropy-coded | 29.16 | 11.8% |

### Mixed plans (computed). always-on = every group except routed experts, router, norms/other (attention/Mamba, shared expert, dense, latent proj, embeddings+head, MTP non-expert)
| plan | stored GB | experts entropy-coded GB (always-on stored) | everything entropy-coded GB | experts GB (stored) |
|---|---|---|---|---|
| lossless only (Glyd penguin / H8 on all) | 166.50 | | | |
| always-on 6-bit, experts 5-bit | 86.09 | 81.74 | 81.39 | 79.45 |
| always-on 6-bit, experts 4-bit | 71.65 | 66.81 | 66.47 | 65.00 |
| always-on 6-bit, experts 3-bit | 57.20 | 51.82 | 51.47 | 50.56 |
| always-on 6-bit, experts 2-bit | 42.76 | 33.94 | 33.60 | 36.11 |
| always-on 5-bit, experts 4-bit | 70.65 | 65.82 | 65.47 | 65.00 |

## What this suggests

All numbers below are [C] from [H] sizes and [M] rates.

**Lossless saving available**
- Nemotron (BF16, 247.2 GB): the order-0 entropy bound is 162.3 GB (-34.4%); Glyd's real penguin packer on the sampled tensors gives 10.77 bits/weight on experts (-32.7%), 166.5 GB whole model (-32.7%), i.e. 80.7 GB saved. Qwen3.5-9B reference was -29%; the larger models land slightly higher. Exponent entropy is 2.6 bits of 8; sign+mantissa is incompressible at 7.97 of 8, so the bound is basically "exponent entropy-coded".
- GLM (328.3 GB shipped, FP8 mostly): FP8 codes have 6.47 bits entropy (-19.1% on experts, 6.59 on shared expert), so about 66.6 GB (-20.3%) saved overall at the bound: 261.7 GB. The F32 scales are 0.002 bits/weight (about 0.6 GB total) and not worth coding. 
- GLM surprise: the BF16 KDA attention (q/k/v/o, 4.7 B params) and indexer tensors have H16 = 7.26 and 6.7 bits (-55% and -58%), against 10.5 bits for the other BF16 tensors. A sampled `q_proj` window holds only 3,944 distinct 16-bit values in 33.5 M weights: consistent with FP8 values times scales stored as BF16 (inference, not checked against the FP8 recipe). Glyd's penguin packer gets only 10.8 bits on them (-32.5%), so it leaves about 3.5 bits/weight on the table for KDA (about 2 GB of 9.4 GB) and 4 bits/weight on the indexer (tiny). A FP8-aware lossless layout for those tensors is a small, cheap win for this model.

**What lossy saves per group** (stored, N + 0.5 bits; vs original)
- BF16 groups (all of Nemotron; GLM KDA, indexer, kv_b, embeddings, vision): 6-bit 59.4% saved, 5-bit 65.6%, 4-bit 71.9%, 3-bit 78.1%, 2-bit 84.4%.
- GLM FP8 groups (routed/shared experts, MLA, dense): 6-bit only 18.8% saved (8.0 to 6.5 bits), 5-bit 31.3%, 4-bit 43.8%, 3-bit 56.3%, 2-bit 68.8%. At 6 bits the lossy step buys barely more than lossless entropy coding (6.47 bits lossless vs 6.5 stored).
- Entropy coding the codes on top of lossy saves 0.3 bit at 6 bits, 0.3 at 5, 0.33 at 4, 0.37 at 3, and 0.6 at 2 (codes cluster at 0; attn/shared groups save more: Nemotron attn 6-bit codes 5.95 bits).

**Where error jumps** (relative RMSE, same for every group within about 10%): N=6 0.022, N=5 0.044, N=4 0.092, N=3 0.197, N=2 0.49. It doubles per bit removed down to 3 bits, then rises 2.5x from 3 to 2 (SNR 6 dB at 2 bits, 14 dB at 3, 21 dB at 4, 27 dB at 5, 33 dB at 6). Weight error of 4 bits (9%) is the point past which plain block rounding without calibration is likely to start hurting quality; 3 bits (20%) and 2 bits (49%) almost certainly need calibration/QAT. This is a prior, not a quality measurement. Slightly more sensitive by weight RMSE: GLM shared-expert `down_proj` (0.105 at N=4) and `o_proj` of KDA (0.099), Nemotron shared expert, attn k/q/v and the router (0.099-0.108), against 0.091-0.093 for routed experts, embeddings and most others. Keep router and norms at original (under 0.1% of the bytes).

**Smallest totals under three plans** (always-on parts = attention/Mamba, shared expert, dense, latent proj, embeddings+head, MTP non-expert, vision at 6-bit stored; router and norms/other original)

| plan | GLM-5.3-Flash GB (from 328.3) | Nemotron-3-Super GB (from 247.2) |
|---|---|---|
| lossless only (Glyd penguin real on BF16, FP8 at entropy bound) | 263.9 (-19.6%) | 166.5 (-32.7%) |
| conservative: experts 5-bit | 222.3 stored / 210.6 experts entropy-coded | 86.1 / 81.7 |
| balanced: experts 4-bit | 183.3 / 170.4 | 71.7 / 66.8 |
| aggressive: experts 3-bit | 144.3 / 129.8 | 57.2 / 51.8 |
| extreme (not recommended without calibration): experts 2-bit | 105.4 / 81.6 | 42.8 / 33.9 |

"Entropy-coded" column = experts' codes entropy-coded (RAM/SSD bandwidth-bound tiers; needs a decode step on the path), always-on parts still stored; full tables above also give everything entropy-coded (about 0.4 GB less on GLM, 0.3-0.4 GB on Nemotron less than that column).

Observations:
- Nemotron starts from BF16, so even the conservative plan is 2.9x smaller and fits a 96 GB box; balanced is 71.7 GB (fits 80 GB). GLM starts from FP8, so each plan gives much less: balanced is 183 GB, and 4-bit for everything including always-on is 181 GB; 3-bit experts give 144 GB (130 GB with experts entropy-coded), the nearest to a 128 GB class; 2-bit gives 105 GB.
- Routed experts are 93.5% (Nemotron) and 97.0% (GLM) of the parameters, so the experts' bit width decides the total; the always-on level moves it by only 1-2 GB (6-bit vs 5-bit always-on: 1.0 GB on Nemotron, 1.2 GB on GLM).
- Per active token the always-on parts matter more for quality than their bytes suggest (every token uses them; a given routed expert only some tokens; Nemotron config: 22 of 512 experts per token), which is why 6-bit always-on with 3-4-bit experts is the sensible shape; this needs KL to confirm.
- Not measured here: quality (KL), the scale of the dequantisation cost, FP8 packers in Glyd, whether the 128x128 FP8 block structure could cut the 0.5 bit/weight scale cost, and experts not in the 3 sampled layers (spot variation across layers is small: H16 10.49-10.51 across experts in the per-kind rows).
