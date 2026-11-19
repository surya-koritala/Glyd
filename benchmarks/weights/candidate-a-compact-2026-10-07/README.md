# Candidate A compact matrices on one RTX 4090

The prepared isolated-matrix batch passed every correctness check on one RTX 4090 on 2026-10-07. All 20 cases decoded their rounded Candidate A words bit for bit, 528 products passed and the 48 unsupported large products returned the documented refusal. No model was loaded; this is not whole-model VRAM, native FP8 performance or serving speed.

## Correctness

| Check | Result |
| :-- | :-- |
| Decoded weights | 20/20 cases bit for bit, including every compactable BF16 word and every BF16 word through the unchanged fallback |
| Products | 528 pass; worst normalized RMS error 1.78e-3 against the 4e-3 bound (the expanded representation's worst is the same); worst maximum error 3.68e-3 of the reference maximum against 1e-2 |
| Integer products | Bit exact for both compact and expanded paths |
| Repetition and graph replay | Same output bits in every product |
| Refusals | 48/48 large products with an inner dimension not divisible by 64, plus all invalid-dimension and misaligned-input checks |

## Weight-buffer size

Across the 15 model-shape matrices (492,906,496 weights), the compact device buffers hold 6.7736 bits per weight, against 10.7736 for the existing expanded representation. That is 57.67% smaller than BF16, 37.13% smaller than the expanded representation and 15.33% smaller than one byte per weight. Official FP8 also stores scales, so its matrix bytes are at least one byte per weight. Per-matrix rates range from 6.71 to 7.46 bits; the small 1,152-column vision matrices are highest. The device allocator delta exceeded the exact buffer bytes by at most 1,049,571 bytes per matrix (block rounding). Non-matrix tensors, incompatible shapes and whole-model residency are not covered.

## Timing (diagnostic, L2-warm)

Summed medians over the 15 model matrices, without bias, in milliseconds:

| Rows | Compact | Expanded | BF16 `mm` | Compact / BF16 | Compact / expanded |
| --: | --: | --: | --: | --: | --: |
| 1 | 0.445 | 0.555 | 0.858 | 0.52 | 0.80 |
| 16 | 0.480 | 0.681 | 0.960 | 0.50 | 0.71 |
| 64 | 1.011 | 1.078 | 1.054 | 0.96 | 0.94 |
| 128 | 1.110 | 1.120 | 1.390 | 0.80 | 0.99 |
| 129 | 1.951 | 1.818 | 1.411 | 1.38 | 1.07 |
| 256 | 1.971 | 1.850 | 2.133 | 0.92 | 1.07 |
| 1024 | 7.115 | 6.476 | 7.530 | 0.94 | 1.10 |
| 2048 | 13.794 | 13.053 | 13.578 | 1.02 | 1.06 |

All 16 row counts are in `summary.json`. These timings replay the same weights ten times in a graph, so matrices that fit in the RTX 4090's 72 MiB L2 cache are partly served from it. BF16 products of 42–63 MB matrices report 2.8–3.3 TB/s effective, well above device memory bandwidth, while the 105–178 MB matrices report about 0.94 TB/s. The compact buffers are smaller and therefore benefit more; the small-row speedups above are not device-memory results. The protocol has no official FP8 product.

Above 128 rows the compact path uses one fixed large-product configuration, while the expanded representation switches to 256-token blocks; that matches the 6–10% gap from 129 rows. The small-row compact path uses the generic product rather than the expanded type's specialized step implementation.

## Run

Source [24092fe6](https://github.com/surya-koritala/glyd-gpu-private/tree/24092fe6a5f89252fd31428ea33fc3637b61ebd1), merged as #85. [CI run 37666406376](https://github.com/surya-koritala/glyd-gpu-private/actions/runs/37666406376) built the bundle at merge commit `814fa2a1363f5e91f3669a5577fd8269427e865c`, whose tree equals the reviewed head. The artifact zip SHA256 `f14aa729a4fa590790cbc3db3a21123e01a3f35389454019d4bb2ce3ad120823` matched the CI upload; the 2,028,707,840-byte bundle has SHA256 `3b7e9f691ff6883002a63e2a405f081353d14de4a2675d219241ce82456a67ad`, and all 133 members passed `SHA256SUMS` before upload and on the pod (`input-checks.txt`). `bundle-proof.json` records the approval.

Pod `v365t4bx14kj5m` was created at 22:41 UTC; the job ran 22:42:53–22:47:12 (`steps.txt`), and the launcher deleted the pod at 22:47. An authenticated listing at 23:00:30 UTC returned no pods (`deletion-proof.json`). The guard accounts 491 seconds at $0.74/hour, a conservative $0.100928 GPU estimate excluding storage, within the $0.75 cap; one attempt, no retry (`cost.json`).

`matrices/report.json` is the complete device report; per-case files, `smi.csv`, `machine.txt` and logs are retained. `RECORD-SHA256SUMS` covers this record.
