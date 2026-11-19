# Candidate A values in 7- and 6-bit fixed-width formats

The compact Candidate A representation is 6.7736 bits per weight but costs a decode on every use ([timing](../candidate-a-timing-2026-10-07/README.md), [FP8 arithmetic](../candidate-a-fp8-2026-10-07/README.md)). This census asks whether a fixed-width store that becomes e4m3 by an exponent offset — a few bit operations instead of a decode — could hold the fixed Candidate A values in fewer than eight bits. It counts, for every matrix of the pinned checkpoint, the values that are exactly values of each format times the smallest power-of-two scale covering their block. It changes no weights and is not a quality, memory or speed result.

Formats: e4m3fn (eight bits, for reference); e3m3 (seven bits: sign, three exponent bits with bias 3, three fraction bits, subnormals, no NaN; largest 30, subnormal step 2^-5); e2m3 (six bits: bias 1, largest 7.5, step 2^-3).

Checkpoint `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, 617 matrices, 27,776,577,536 values. The matrices multiplied during text generation (497 tensors, 25,621,954,560 values):

| Format | Scale block | Inexact values | Underflow | Blocks with an inexact value | Worst block |
| :-- | :-- | --: | --: | --: | --: |
| e4m3fn | 1 x 32 | 0.0060% | 0.0011% | 0.19% | 2 |
| e4m3fn | 1 x 128 | 0.0076% | 0.0014% | 0.97% | 3 |
| e4m3fn | whole row | 0.0108% | 0.0020% | 45.04% | 107 |
| e3m3 | 1 x 32 | 1.4345% | 0.2679% | 36.44% | 28 of 32 |
| e3m3 | 1 x 128 | 1.8240% | 0.3407% | 88.49% | 117 of 128 |
| e3m3 | whole row | 2.6038% | 0.4866% | 100.00% | 9,098 |
| e2m3 | 1 x 32 | 22.1654% | 4.2835% | 99.85% | 31 of 32 |
| e2m3 | 1 x 128 | 27.7020% | 5.4445% | 100.00% | 127 of 128 |

The 128 x 128 blocks are worse for every format (e3m3 3.2149%, e2m3 43.1768%); all matrices, every group and the per-tensor counts are in `summary.json` and `tensors/`. The e4m3fn counts reproduce the [earlier census](../candidate-a-fp8-census-2026-10-07/README.md) exactly.

Six bits leave a fifth to two fifths of the values outside the format. Seven bits leave 1.4–1.8% with per-row blocks, nearly all of them the smallest values of their block. Holding those exactly in a side list costs, for example, a 5-bit position and a 16-bit value each plus a scale byte a block: about 7 + 8/32 + 0.0143 x 21 ≈ 7.55 bits with 1 x 32 blocks, or 7 + 8/128 + 0.0182 x 23 ≈ 7.48 bits with 1 x 128 blocks — about 6% below eight bits, and with exceptions in most blocks. Rounding them instead would make a seven-bit store about 12% below eight bits but change 1.4–1.8% of the weights, a separate lossy variant that needs its own quality evaluation. These sizes are arithmetic from the counts, not encoded files.

The source is [50de26da](https://github.com/Sigbound/glyd-gpu-private/tree/50de26da38469ae4ec2a29cf715f8ba4e9edba0a). Unit tests enumerate every value of each format and check every rounded BF16 word at 160 scales. [CI run 37718734087](https://github.com/Sigbound/glyd-gpu-private/actions/runs/37718734087) ran the tests and then the census from 02:40:24 to 03:06:47 UTC on 2026-10-08; `ci.json` records the jobs and artifact digest. No GPU was used. `RECORD-SHA256SUMS` covers this record.
