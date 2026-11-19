# Seven-bit product, first tuning round, on one RTX 4090

Seven variants of the seven-bit product (#101) ran beside the FP8 lower bound in one process on the same 14 matrices. The best variant at each row count takes 0.85x the FP8 lower bound's text-decoder step at one row, 0.87x at 8, 0.94x at 32 and 0.95x at 64, and 1.19x at 128 and 1.44x at 256. No model was loaded; this is not whole-model VRAM, native FP8 serving or end-to-end speed.

## Variants

The weights are the [candidate-a7 arm](../candidate-a7-quality-2026-10-08/README.md)'s, stored as in the [first seven-bit record](../candidate-a7-kernel-2026-10-08/README.md) except that each 16-column step's eight fragment registers are one seven-word group and the block exponents are ordered by row block, so that one step can be read alone.

| Variant | Arithmetic | Unit |
| --: | :-- | :-- |
| 1 | FP8 | 64 tokens by two 64-row blocks, four 32-row consumers |
| 2 | FP8 | 128 tokens by one row block, four 32-row consumers |
| 3 | BF16, X as given (no conversion) | the engine's small-M product, up to 64 tokens |
| 4 | FP8 | 64 tokens by two row blocks, eight 16-row consumers |
| 5 | FP8 | 128 tokens by two row blocks (spills: 168 registers) |
| 6 | FP8 | 256 tokens by one row block (spills: 168 registers) |
| 7 | FP8 | 128 tokens by one row block, eight 16-row consumers |

## Correctness

The device-unpacked bytes equal the host encoding for all weights; activation bytes and scales equal the once-rounded reference; every product of every variant repeats and replays bit for bit. Normalized RMS against each variant's own reference (the fp32 product of the same e4m3 values, or of the BF16 activations for variant 3) is at most 1.82e-3, the BF16 output's rounding.

## Timing

Text-decoder step of Qwen3.8-27B (each language-model matrix shape times its count in the pinned checkpoint, 64 layers), matrix products only, in milliseconds. FP8 lower bound: per-tensor e4m3 through cuBLASLt with fast accumulation, activation conversion excluded (the FP8 variants also exclude theirs; variant 3 has none).

| Rows | FP8 lower bound | 1 | 2 | 3 | 4 | 7 | Best / FP8 |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 27.91 | 29.05 | 40.03 | **23.73** | 27.98 | 41.62 | 0.85 |
| 8 | 28.08 | 28.77 | 39.02 | **24.33** | 27.80 | 40.54 | 0.87 |
| 16 | 28.02 | 28.82 | 39.03 | **24.96** | 27.89 | 41.16 | 0.89 |
| 32 | 28.36 | 29.32 | 39.70 | **26.59** | 28.27 | 41.48 | 0.94 |
| 64 | 30.34 | 29.95 | 40.83 | 34.03 | **28.93** | 42.61 | 0.95 |
| 128 | 34.39 | 50.43 | **40.87** | | 48.29 | 42.21 | 1.19 |
| 256 | 49.61 | 91.00 | **71.52** | | 85.96 | 77.25 | 1.44 |
| 2048 | 291.00 | 684.20 | **487.95** | | 639.27 | 566.35 | 1.68 |

Variants 5 and 6 take 96–235 ms at every row count (their spills). Every row count and matrix is in `summary.json` and `a7/`.

The engine's small-M product (variant 3) is the fastest up to 32 rows: it reads seven-bit weights directly with BF16 arithmetic and no activation conversion. From 33 rows its BF16 arithmetic costs more than the FP8 variants'. Variants 1 and 2 are 5–7% slower than with the first record's layout (fourteen 4-byte loads a k32 pair in place of seven 8-byte ones). From 65 rows no variant reaches the FP8 lower bound: with 64 output rows a unit, X is read from L2 once for every 64 rows, and the larger units (5, 6) exceed their registers because each consumer keeps a second set of sums for the block scales.

## Run

Source: #101 at [a08ebad7](https://github.com/Sigbound/glyd-gpu-private/tree/a08ebad71ca36bbdabdd0fbf34c2db9c87c767c8), merged as `e623a9e9`. [CI run 37797610038](https://github.com/Sigbound/glyd-gpu-private/actions/runs/37797610038) built the bundle at merge commit `bdf01fa6353df5f6b4da5f74795a0405f128f56e`, whose `research/` and `gpu/` trees equal the head and merged main. The artifact zip SHA256 `13c05f9369085238137d7d5cafb06c16dce38a96ef96604e2c653f28a4d2999e` matched CI; the bundle has SHA256 `3653289547ca2744a5da33fe63466d8b117239746da7b14c3077fc1f0a0870c6`, and all 136 members passed `SHA256SUMS` before upload and on the pod. `bundle-proof.json` records the approval; `job.sh` is the bounded job.

Pod `jufwqhmwhptnko`, an RTX 4090, was created at 15:27 UTC on 2026-10-08; the job ran 15:30:35–15:33:50 (`steps.txt`) and the launcher deleted the pod at 15:34. An authenticated listing at 15:35:05 UTC returned no pods (`deletion-proof.json`). The guard accounts 540 seconds at $0.74/hour, a conservative $0.111 GPU estimate excluding storage, within the $0.45 cap; one pod, no retry (`cost.json`). `RECORD-SHA256SUMS` covers this record.
