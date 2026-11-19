# Candidate A compact matrices from device memory, with an FP8 baseline

The second isolated-matrix batch passed every check on one RTX 4090 on 2026-10-07, using the inputs of the [first batch](../candidate-a-compact-2026-10-07/README.md) unchanged. Each timed operation now reads its own weight copy, and an FP8 product is timed beside the compact, expanded and BF16 products. No model was loaded; this is not whole-model VRAM, native FP8 serving or end-to-end speed.

## Correctness

All 20 cases decoded bit for bit; 528 products passed and 48 large products with an inner dimension not divisible by 64 returned the documented refusal. Above 64 rows both large-product block shapes were also forced: each passed the reference bounds and repeated bit for bit (worst normalized RMS 1.68e-3), and integer products were exact. Default compact products: worst normalized RMS 1.78e-3 against the 4e-3 bound.

## Device-memory timing

For model shapes every path's copies spanned at least 201 MB, against the 72 MiB L2. BF16 products of the large matrices now report 0.94–0.98 TB/s and smaller matrices no longer exceed device bandwidth, unlike the first batch.

The FP8 product uses per-tensor e4m3 weights and activations through cuBLASLt with fast accumulation and excludes activation quantization, so it is a lower bound on FP8 product time; it is not the official checkpoint's block-scaled serving path. Its worst normalized RMS error was 4.73e-2 (recorded, not gated).

Text-decoder step of Qwen3.8-27B: each language-model matrix shape times its count in the pinned checkpoint (64 layers), in milliseconds. Matrix products only; attention, normalization, the LM head and other operations are excluded.

| Rows | Compact | FP8 (lower bound) | BF16 | Compact / FP8 |
| --: | --: | --: | --: | --: |
| 1 | 24.64 | 27.93 | 50.52 | 0.88 |
| 8 | 25.11 | 28.13 | 53.06 | 0.89 |
| 16 | 25.71 | 28.24 | 53.57 | 0.91 |
| 17 | 33.28 | 28.17 | 58.78 | 1.18 |
| 32 | 34.04 | 28.42 | 60.26 | 1.20 |
| 64 | 50.55 | 30.76 | 60.23 | 1.64 |
| 128 | 50.63 | 34.64 | 69.06 | 1.46 |
| 256 | 88.91 | 49.67 | 104.08 | 1.79 |
| 1024 | 330.33 | 150.98 | 356.20 | 2.19 |
| 2048 | 639.69 | 290.54 | 668.79 | 2.20 |

At 1–16 rows the compact products take 0.88–0.91x the FP8 lower bound; the largest matrices read their weights at 0.87 TB/s. From 17 rows the compact time steps up while FP8 stays memory-bound, reaching 1.64x at 64 rows: the BF16-arithmetic compact products are no longer limited by weight bytes there. From 129 rows FP8 arithmetic is about twice as fast as the BF16-arithmetic paths (compact, expanded and BF16 alike). The small 1,152-column vision matrices reach only 0.11–0.32 TB/s in the compact path at one row.

Above 128 rows the compact default now selects the expanded representation's block shape and matches its time (1.00–1.03x across the 15 shapes, against 1.06–1.10x in the first batch). The forced 256-token shape is 1.6x slower at 65–128 rows and the 128-token shape 6–8% slower from 129 rows, so the default choice was the faster at every measured row count. All 16 row counts, the summed per-shape table, the scaled LM-head estimate and both forced shapes are in `summary.json`.

## Run

Source [409834f6](https://github.com/surya-koritala/glyd-gpu-private/tree/409834f62d55b7c094e68db4ee603643590628e7), merged as #89. [CI run 37700935893](https://github.com/surya-koritala/glyd-gpu-private/actions/runs/37700935893) built the bundle at merge commit `80adaefbb83ffb00c28e7349bd1e7953f2a0fd10`, whose `gpu/` and `research/` trees equal the reviewed head and merged main. The artifact zip SHA256 `6d2bf1fb1f3e3ace367c16d31aba08399eaa74959c6b65948aa9a3af5feaa95e` matched CI; the 2,028,800,000-byte bundle has SHA256 `59fdc283389a34c6e1795a3a96048c913ef430bcee02466fb1044a15a74cb6bb`, and all 133 members passed `SHA256SUMS` before upload and on the pod. Fixture files are identical to the first batch; only the timing policy changed. `bundle-proof.json` records the approval.

Pod `c4ivo0fmxyaapi` was created at 23:36 UTC; the bundle upload took until 23:48, the job ran 23:48:36–23:55:32 (`steps.txt`), and the launcher deleted the pod at 23:56. An authenticated listing at 23:56:19 UTC returned no pods (`deletion-proof.json`). The guard accounts 1,265 seconds at $0.74/hour, a conservative $0.260028 GPU estimate excluding storage, within the $0.75 cap; one attempt, no retry (`cost.json`).

`matrices/report.json` is the complete device report. `RECORD-SHA256SUMS` covers this record.
