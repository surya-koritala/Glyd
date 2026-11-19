# Seven-bit Candidate A product on one RTX 4090

The seven-bit product (#97) ran on one RTX 4090 beside the FP8 lower bound, timed in the same process on the same 14 matrices. It is correct and deterministic. Its text-decoder step takes 0.89–0.99x the FP8 lower bound's time at 1–64 rows, 1.10–1.18x at 65–128 and 1.36–1.65x from 129. No model was loaded; this is not whole-model VRAM, native FP8 serving or end-to-end speed.

## What the variant is

The weights are the quality screen's [candidate-a7 arm](../candidate-a7-quality-2026-10-08/README.md): Candidate A rounded to e3m3 with one power-of-two scale a row and 128 columns. Each is stored as the e4m3 byte of its value times 2^-4, whose bit 6 is always zero, packed seven bits a weight in the B fragments' order, with one int8 exponent a row and 128 columns (7.0625 bits a weight). Producer warps restore the e4m3 bytes with masks and shifts; consumers multiply with `mma.m16n8k32`, keep each 128 columns' fp32 sums apart and add them times their block's power of two. X is converted to e4m3 with one scale a token, as in the [FP8-arithmetic variant](../candidate-a-fp8-2026-10-07/README.md). Units of 64 tokens by two row blocks (variant 1) and 128 tokens by one (variant 2) were timed.

## Correctness

| Check | Result |
| :-- | --: |
| Device-unpacked e4m3 bytes of all 487,948,288 weights against the host encoding | identical |
| Weights changed by the e3m3 rounding | 9,244,286; equal to the [fixed-width census](../candidate-a-fixed-width-census-2026-10-07/README.md) count for each of the 13 whole tensors |
| Device activation bytes and scales against a reference dividing tensor by tensor | identical (0 bytes, 0 scales) |
| Repetition and graph replay, 896 products | bit identical |
| Error against the fp32 product of the same e4m3 values and scales | normalized RMS at most 1.76e-3 (the BF16 output's rounding) |
| Error against BF16 activations | normalized RMS at most 2.94e-2 |

The earlier record's unexplained activation mismatch is resolved. That harness divided by a Python number, which PyTorch evaluates by multiplying with the reciprocal; reproduced here, it differs from the device in 495,581 bytes and 61,927 scales, exactly the earlier counts, while the once-rounded division matches the device everywhere.

## Timing

Text-decoder step of Qwen3.8-27B (each language-model matrix shape times its count in the pinned checkpoint, 64 layers), matrix products only, in milliseconds. The FP8 lower bound is per-tensor e4m3 through cuBLASLt with fast accumulation, activation conversion excluded.

| Rows | FP8 lower bound | Seven-bit, variant 1 | Seven-bit, variant 2 | Best / FP8 |
| --: | --: | --: | --: | --: |
| 1 | 27.85 | 27.63 | 36.24 | 0.99 |
| 8 | 28.04 | 27.49 | 36.35 | 0.98 |
| 16 | 28.01 | 27.65 | 36.56 | 0.99 |
| 32 | 28.30 | 27.97 | 36.85 | 0.99 |
| 64 | 30.17 | 28.61 | 37.55 | 0.95 |
| 128 | 34.66 | 47.57 | 38.29 | 1.10 |
| 256 | 50.18 | 86.39 | 68.23 | 1.36 |
| 1024 | 151.91 | 321.95 | 249.72 | 1.64 |
| 2048 | 292.96 | 628.48 | 484.22 | 1.65 |

Converting a step's activations adds 0.82 ms at one row and 1.18 ms at 256. Every row count, matrix and variant is in `summary.json` and `a7/`.

Decoding no longer limits the product. At one row the large matrices read 730–830 GB/s of seven-bit weights against 860–890 GB/s for FP8, so the 12% fewer bytes are mostly spent on lower bandwidth; matrices of 1,152 columns or 1,024 rows reach only 95–240 GB/s. From 129 rows the products are bound by arithmetic: variant 2's 128-token by 64-row units, each consumer holding a second set of sums for the block scales, reach about 60% of cuBLAS's FP8 throughput at 2,048 rows. Against the earlier paths at 256 rows, the step takes 68.2 ms, against 89.0 ms for the compact BF16-arithmetic path and 98.6 ms for the compact FP8-arithmetic variant.

## Run

Source: the product merged as #97 ([7d02ede7](https://github.com/Sigbound/glyd-gpu-private/tree/7d02ede7b6478be882b20d3167ade2814ebcbf7e)), the device check relaxed to any one SM89 GPU in #98 and #99. [CI run 37790199387](https://github.com/Sigbound/glyd-gpu-private/actions/runs/37790199387) built the bundle at merge commit `12250f7dc79bc10061771dfc8bc9c856e0aa3966`, whose `research/` and `gpu/` trees equal #99's head and merged main `187c477c`. The artifact zip SHA256 `0d00fb01a90027de70c3e7c023c501537c128de818b7a5db5c2d94cd6fe2c29f` matched CI; the bundle has SHA256 `a8c5c62f0b191aeb82ad2b66609f29fc5a426a738ff07a8cece3e901170cc899`, and all 136 members passed `SHA256SUMS` before upload and on the pod. `bundle-proof.json` records the approval; `job.sh` is the bounded job.

Three approved launches before this one found no instance at creation (RTX 4090 twice, L40 once) and created nothing. This batch tried each SM89 type in turn; pod `xxy9hgju7w2vs9`, an RTX 4090 (driver 580.159.04), was created at 14:32 UTC on 2026-10-08, the job ran 14:35:52–14:38:06 (`steps.txt`) and the launcher deleted the pod at 14:39. An authenticated listing at 14:39:53 UTC returned no pods (`deletion-proof.json`). The guard accounts 464 seconds at $0.74/hour, a conservative $0.095378 GPU estimate excluding storage, within the $0.55 cap; one pod, no retry (`cost.json`). `RECORD-SHA256SUMS` covers this record.
