# Qwen3.8-27B's weight representations against BF16: KL divergence (2026-10-09)

Private record. Every representation's next-token distributions compared with BF16's on the same text, with its bits a weight:
`research/lossy-2026/kld_study.py` at commit `8bea51d1` (this branch), job `kldjob.sh`, on one A100 80GB PCIe (Runpod pod
`zmw74hlhfowkw7`, driver 595.91.07, $1.79/hour), 2026-10-09 from 00:06 UTC.

- **Checkpoints:** Qwen/Qwen3.8-27B at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (BF16) and Qwen/Qwen3.8-27B-FP8 at
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, loaded by transformers 5.18.0 (`AutoModelForImageTextToText`, sdpa attention, BF16
  arithmetic; the FP8 checkpoint dequantized by `FineGrainedFP8Config(dequantize=True)`), torch from the cu128 index.
- **Text:** the pinned WikiText-2 test file of `prepare_quality.py` (checked by its SHA-256), 297,193 tokens; 256 windows of 1,024 tokens
  evenly spaced through it, the first 32 tokens of each as context: **253,952 scored positions**.
- **Measures, per position:** the KL divergence of the variant's distribution from BF16's, over BF16's 64 likeliest tokens and the rest of
  the vocabulary as one bucket (a lower bound on the full divergence); whether the variant's top token is BF16's; the target's
  log-probability (perplexity).
- **Bits a weight:** over the text model's matrices; for `frac` and `int`, the empirical entropy of each tensor's codes (a lower bound for a
  coder of codes taken one at a time) plus, for `int`, the fp16 scale's 16 bits a block. Glyd's compact layout holds `frac:3` (Candidate A,
  shipped as `--lossy` in 0.29.2) at 6.774 bits a weight; FP8 is 8 bits a weight plus its 128 x 128 block scales.
- **Variants:** `frac:F` rounds every BF16 parameter to F fraction bits (nearest, ties to even; `frac:3` is Candidate A); `frac:2@G=3`
  keeps group G at 3; `int:B:BLOCK:FIT` holds each matrix as B-bit symmetric codes times an fp16 scale for each BLOCK values of a row
  (`mse`: the best of 16 clips; `absmax`: the block's largest value), other parameters BF16. No calibration data was used by any variant.

## Results

| Variant | Bits a weight (matrices) | KL mean | KL median | KL p99 | KL p99.9 | Top-1 agreement | Perplexity |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| `bf16` | 16 | 0.00000 | 0.00000 | 0.0000 | 0.0000 | 100.00% | 7.2226 |
| `fp8` (official checkpoint) | ~8 | 0.00465 | 0.00172 | 0.0362 | 0.2140 | 97.07% | 7.2410 |
| `int:6:32:mse` | 6.202 | 0.00486 | 0.00199 | 0.0341 | 0.1881 | 96.86% | 7.2501 |
| `int:6:32:absmax` | 6.198 | 0.00486 | 0.00203 | 0.0344 | 0.1933 | 96.80% | 7.2607 |
| `frac:3` (Candidate A) | 6.498 | 0.00831 | 0.00348 | 0.0665 | 0.3304 | 95.65% | 7.2205 |
| `int:5:32:mse` | 5.193 | 0.01398 | 0.00631 | 0.1144 | 0.5609 | 94.33% | 7.3047 |
| `int:5:32:absmax` | 5.178 | 0.01413 | 0.00632 | 0.1135 | 0.5320 | 94.38% | 7.3126 |
| `int:5:64:mse` | 4.839 | 0.01710 | 0.00796 | 0.1377 | 0.6350 | 93.59% | 7.2800 |
| `frac:2` | 5.510 | 0.02580 | 0.01151 | 0.2155 | 1.0146 | 92.29% | 7.3486 |
| `int:4:32:mse` | 4.160 | 0.05335 | 0.02698 | 0.4491 | 1.7270 | 88.92% | 7.4184 |

Each group of matrices at 3 fraction bits with the rest at 2 (`frac:2` is 0.02580 and 5.510):

| Group at 3 bits | Bits a weight | KL mean | Change | Top-1 agreement | Perplexity |
| :-- | --: | --: | --: | --: | --: |
| `head` (lm_head) | 5.557 | 0.02358 | −8.6% | 93.07% | 7.3399 |
| `attn_qkv` | 5.554 | 0.02440 | −5.4% | 92.42% | 7.3439 |
| `attn_o` | 5.529 | 0.02542 | −1.5% | 92.35% | 7.3554 |
| `lin_out` | 5.566 | 0.02519 | −2.4% | 92.36% | 7.3390 |
| `lin_in` | 5.659 | 0.02447 | −5.2% | 92.43% | 7.3019 |
| `mlp_down` | 5.720 | 0.02406 | −6.7% | 92.47% | 7.3176 |
| `mlp_gate_up` | 5.929 | 0.02294 | −11.1% | 92.67% | 7.3080 |
| `embed` | 5.557 | 0.02589 | +0.3% | 92.30% | 7.3506 |

## What it shows

- Six-bit block codes (`int:6:32`) are within 4.5% of the official FP8 checkpoint's mean divergence (0.00486 against 0.00465), with a lower
  99th and 99.9th percentile and 0.2 points less top-1 agreement, at 6.20 bits a weight against about 8.
- Candidate A (`frac:3`, shipped as `--lossy`) diverges 1.79 times as much as FP8 on average (0.00831) and is larger than `int:6:32`
  (6.774 bits a weight in its layout). Its perplexity, 7.2205, is the lowest after BF16's; perplexity does not order these variants as
  the divergence does.
- At five bits, block codes (`int:5:32`, 0.01398 at 5.19) diverge about half as much as rounding to two fraction bits (`frac:2`, 0.02580 at
  5.51). Fitting the scale (`mse`) changes the divergence by 0% at six bits and 1% at five.
- Without calibration, four-bit block codes diverge 11.5 times as much as FP8 (0.05335) and change 11% of top tokens.
- The codes' entropy is 0.30 to 0.34 bits below their width (`int:6`: 5.70, `int:5`: 4.69, `int:4`: 3.66, before the scales' 0.5).

Not measured here: task accuracy, generated text, speed or memory in any runtime; the block codes have no kernel yet.

## Files

`results.jsonl` (one line a variant: every measure, bits a weight by group, values changed, seconds), `ref.json` (the reference pass),
`steps.txt` (each step's status and seconds; `selftest` failed on two bounds of the test itself, fixed in `8bea51d1` and rerun on the pod
before the scoring steps: "selftest ok"), `kldjob.sh` (the job). The pod ran about 49 minutes (00:06 to about 00:55 UTC, the engine's
speed profile included: benchmarks/gpu/engine35-small-product-2026-10-09 on its branch) at $1.79/hour: about $1.46.
