# Candidate A FP8-arithmetic product on one RTX 4090

The diagnostic FP8-arithmetic variant (#92) ran on one RTX 4090 after the exact compact matrices, which passed again. The variant is correct and deterministic, but it is about twice the time of the FP8 lower bound at every measured row count and slower than the BF16-arithmetic compact path. No model was loaded; this is not whole-model VRAM, native FP8 serving or end-to-end speed.

## What the variant is

Producer warps decode the compact weights and convert each pair of steps in shared memory to e4m3 with one power-of-two scale a row (the smallest covering the row's largest magnitude); values below e4m3's range round to nearest even, so it is a separate numerical variant. X is converted to e4m3 with one scale a token; consumers multiply with `mma.m16n8k32` and fp32 sums, and both scales apply to the sums. Units of 64 tokens by two row blocks (variant 1), 128 by two (2) and 256 by one (3) were timed.

## Correctness

| Check | Result |
| :-- | :-- |
| Device e4m3 bytes of all 487,948,288 weights | identical to PyTorch's conversion of the same scaled weights |
| Weights changed by e4m3 rounding | 133,132 (0.0273% of these 14 matrices) |
| Row scales | the smallest covering scale on every row |
| Repetition and graph replay | bit identical for every product |
| Error against the exact Candidate A product | normalized RMS at most 2.94e-2 (per-tensor FP8 lower bound in the earlier batch: 4.73e-2) |
| Exact compact matrices (frozen protocol) | 20/20 cases, 528 products, 48 refusals, all passing |

The device's activation conversion differed from the harness's PyTorch reference in 0.02–1.2% of activation bytes and in 61,927 of 124,936 per-token scales, and products differed from the fp32 product of the reference's e4m3 values by normalized RMS up to 1.31e-2. The cause is not established; it affects that comparison, not the timings or the weight conversion.

## Timing

Text-decoder step of Qwen3.8-27B (each language-model matrix shape times its count in the pinned checkpoint, 64 layers), matrix products only, in milliseconds. The compact BF16-arithmetic and FP8 lower-bound columns come from this run's exact matrices.

| Rows | Compact, BF16 arithmetic | Compact, FP8 arithmetic (best unit) | FP8 lower bound |
| --: | --: | --: | --: |
| 1 | 24.29 | 50.98 | 27.10 |
| 16 | 25.40 | 51.00 | 27.26 |
| 32 | 33.65 | 51.34 | 27.52 |
| 64 | 50.00 | 51.94 | 29.75 |
| 128 | 50.78 | 54.53 | 33.87 |
| 256 | 89.04 | 98.57 | 51.01 |
| 1024 | 326.35 | 358.02 | 159.02 |
| 2048 | 650.58 | 699.64 | 293.70 |

Converting a step's activations adds 1.0 ms at one row and 10.4 ms at 2,048. Every row count and unit is in `summary.json`.

Decoding, not multiplication, limits the variant. With units of 64 tokens the producer warps — half of each block — decode all the weights, and the step takes about twice the BF16-arithmetic path, whose eight warps all decode. With larger units each weight is still decoded once for every 128 or 256 tokens, and the faster e4m3 products wait on that decode; the 256-token unit with two producers is slowest. Faster tensor-core arithmetic does not help until decoding costs much less.

## Run

Source [39b5aaf9](https://github.com/Sigbound/glyd-gpu-private/tree/39b5aaf92c6823ada524fa9459beb177a8cedb73), merged as #92 (the repository moved to Sigbound before launch). CI run 37712546656 built the bundle at merge commit `b53466e226ad0bd75d06e25eedf5379fd025030d`, whose `gpu/` and `research/` trees equal the reviewed head. The artifact zip SHA256 `b6c4ded40d8d9abcee563f40ee30df8d29f5b3dbb1f9a67beec230c418945d81` matched CI; the bundle has SHA256 `bbe31e8bcc2d77caffe51a349f4284040919442cb48faca420653544bf455736`, and all 134 members passed `SHA256SUMS` before upload and on the pod. `bundle-proof.json` records the approval.

Pod `njxd9fryk7h2lk` was created at 02:04 UTC on 2026-10-08; the job ran 02:07:45–02:18:13 (`steps.txt`; exact matrices to 02:15:01, the variant to 02:18:13), and the launcher deleted the pod at 02:18. An authenticated listing at 02:18:35 UTC returned no pods (`deletion-proof.json`). The guard accounts 901 seconds at $0.74/hour, a conservative $0.185206 GPU estimate excluding storage, within the $0.75 cap; one attempt, no retry (`cost.json`).

`fp8/report.json` and `fp8/*.json` hold every variant check and timing; `matrices/report.json` is the exact report. `RECORD-SHA256SUMS` covers this record.
