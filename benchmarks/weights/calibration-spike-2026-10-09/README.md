# Calibration spike (GLY-37), 2026-10-09

Qwen/Qwen3.5-9B (revision c202236235762e1c871ad0ccb60c8ee5ba337b9a), one A100-SXM4-80GB. KL vs bf16 over 253,952 positions (WikiText-2 test windows),
calibration on 512 sequences x 2,048 tokens (WikiText-2 train, python_code_instructions_18k_alpaca, ultrachat_200k; pinned in research/lossy-2026/calib.py,
none sharing a 32-token run with the KL windows). Every variant is the engine's block-code format: runs of 32 along a row, one bf16 scale a run.

| Bits | Method | KL vs bf16 | Top-1 as bf16 | Perplexity | A100 hours |
|--:|---|--:|--:|--:|--:|
| 6 | plain (mse scales), today's kestrel | 0.0147 | 0.959 | – | – |
| 6 | GPTQ, act-order | **0.0057** | 0.972 | 8.874 | 0.185 |
| 5 | plain (mse scales), today's linnet | 0.0373 | 0.926 | 8.998 | 0.017 |
| 5 | GPTQ, act-order | **0.0138** | 0.950 | 8.884 | 0.193 |
| 4 | plain (mse scales) | 0.1070 | 0.856 | 9.570 | 0.016 |
| 4 | GPTQ | 0.0594 | 0.891 | 8.978 | 0.196 |
| 4 | GPTQ, act-order | **0.0471** | 0.898 | 9.134 | 0.193 |
| 4 | AWQ | 0.0750 | 0.877 | 9.172 | 0.162 |
| 3 | plain (mse scales) | 0.4439 | 0.685 | 12.979 | 0.016 |
| 3 | GPTQ, act-order | 0.2122 | 0.769 | 10.268 | 0.192 |


Decision: GPTQ with act-order (whole 32-runs ordered by the Hessian's diagonal, so the runs stay contiguous). It has the lowest KL at every width:
2.3x below plain rounding at 4 bits and 2.7x at 5, and 1.6x below AWQ at 4. The cost is about 0.2 A100-hours for a 9B model, and the output needs no
kernel change. Calibrated 5 bits is below plain 6 bits (0.0138 against 0.0147). 3 bits stays far off even calibrated: Hummingbird needs more than rounding.

Files: 4bit/, 3and5bit/ and 6bit/ (steps, GPU-hours, the scorer's rows, the calibration record).

## Qwen3.8-27B (revision 1d4bf0f2), same method, one A100

| Bits | Method | KL vs bf16 | Top-1 as bf16 | Perplexity | A100 hours |
|--:|---|--:|--:|--:|--:|
| 6 | plain (mse), today's kestrel | 0.00477 | – | – | – |
| 6 | GPTQ, act-order | **0.00293** | 0.975 | 7.228 | 0.596 |
| 5 | plain (mse), today's linnet | 0.01393 | – | – | – |
| 5 | GPTQ, act-order | **0.00823** | 0.955 | 7.246 | 0.601 |
| 4 | plain (mse) | 0.05351 | 0.888 | 7.419 | 0.042 |
| 4 | GPTQ, act-order | **0.03124** | 0.910 | 7.344 | 0.606 |

(Plain 6- and 5-bit rows: the cards' measured KL for this checkpoint. Qwen's FP8 checkpoint: KL 0.00465.)

On the 27B the gain is 1.6-1.7x at every width (2.6-2.7x on the 9B). Calibrated 6 bits (6.5 bits per weight) is below Qwen's own FP8
(0.00293 against 0.00465) with a fifth fewer bits. About 0.6 A100-hours per level for a 27B model.

## Below 3 bits on Qwen3.5-9B (GLY-128): rotation (h) and per-row Lloyd-Max codebooks (lm) on top of GPTQ act-order

| Bits | Method | KL vs bf16 | Top-1 as bf16 | Perplexity | bits/weight (entropy + overheads) |
|--:|---|--:|--:|--:|--:|
| 3 | GPTQ act-order (from above) | 0.2122 | 0.769 | 10.268 | 3.12 |
| 3 | hgptq | 0.1865 | 0.780 | 10.080 | 3.13 |
| 3 | lmgptq | 0.1733 | 0.785 | 10.019 | 3.32 |
| 3 | hlmgptq | **0.1569** | 0.793 | 9.829 | 3.32 |
| 2 | GPTQ act-order (ternary grid) | 1.6621 | 0.433 | 42.016 | 1.88 |
| 2 | hgptq | 1.4115 | 0.464 | 32.078 | 1.90 |
| 2 | lmgptq | 1.2056 | 0.507 | 26.762 | 2.30 |
| 2 | hlmgptq | 0.9123 | 0.547 | 19.739 | 2.30 |

Rotation plus codebook is best at both widths: 26% lower KL than GPTQ at 3 bits, 45% lower at 2. On a dense 9B model 2 bits stays
unusable (top-1 agreement 0.55) and 3 bits stays far from 4 bits (0.157 against 0.047). Each step is about 0.19 A100-hours.
