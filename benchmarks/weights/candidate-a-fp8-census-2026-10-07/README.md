# Candidate A values as block-scaled e4m3

Candidate A keeps three fraction bits, as e4m3 does, with BF16's eight-bit exponent. This census reads every two-dimensional tensor of the pinned checkpoint, applies the Candidate A rounding, and counts the values that are exactly e4m3fn values times the smallest power-of-two scale covering their block (128x128, 1x128 and 1x32 blocks, and whole rows). It decides whether an FP8-arithmetic variant could keep the fixed weights bit for bit. It changes no weights, involves no activations and is not a quality, memory or speed result.

Checkpoint `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: 617 matrices with 27,776,577,536 values, plus 582 non-matrix tensors (4,850,416 values) counted separately. All 1,199 tensors in 18 hash-verified shards were covered. There are no non-finite values and 7,528 zeros.

| Block (rows x inner) | Inexact values | Underflow | Blocks with an inexact value | Worst block |
| :-- | --: | --: | --: | --: |
| 128 x 128 | 3,782,164 (0.0136%) | 726,027 | 85.09% | 267 of 16,384 |
| 1 x 128 | 2,136,346 (0.0077%) | 407,885 | 0.98% | 4 of 128 |
| 1 x 32 | 1,685,308 (0.0061%) | 323,338 | 0.19% | 3 of 32 |
| Whole row | 3,073,020 (0.0111%) | 582,499 | 43.02% of rows | 107 in a row |

The matrices multiplied during text generation (full attention, linear attention, MLP and LM head: 497 tensors, 25,621,954,560 values) behave the same: 0.0134%, 0.0076%, 0.0060% and 0.0108% inexact for the four scales, with at most 3 inexact values in any 1 x 128 block. Vision matrices are the highest group for blocks (0.0253%, 0.0114%, 0.0096%); MTP is highest for whole rows (0.0306%). No matrix is exact under any block. Scale exponents span -15 to -6 for 128 x 128 blocks, -43 to -6 for 1 x 32 blocks and -41 to -6 for whole rows, within an eight-bit power-of-two scale.

Every inexact value is below 2^-6 after scaling while its block's largest value is above 224, so it is smaller than 1/14,336 of the block's largest magnitude. "Underflow" values are below the smallest e4m3 subnormal after scaling; the rest lose low bits in the subnormal range.

Consequently, an FP8-arithmetic variant cannot reproduce the fixed Candidate A weights bit for bit with block scales alone. It needs either a correction for the inexact values (about 2.0 million in the text products with 1 x 128 blocks), or a separately validated variant that rounds them. Per-row 1 x 128 or 1 x 32 blocks leave about half the 128 x 128 count (56% and 45%) and confine the worst case to a few values per block. One scale a whole row, which lets a product apply the weight scale once after its sums, rounds 0.0108% of the text-product values, between the two. Activation quantization, which FP8 arithmetic also requires, is outside this census.

`summary.json` aggregates all matrices, text products and each group; `tensors/` holds every matrix's counts. The source is [62569a68](https://github.com/surya-koritala/glyd-gpu-private/tree/62569a68e781b5e195cdca8ede5835e3e24bf682) (`research/lossy-2026/fp8_census.py` and `rust/src/bin/fp8_census.rs`). Its unit tests check every rounded BF16 word at 160 scales against an enumeration of all e4m3fn values. [CI run 37710987725](https://github.com/surya-koritala/glyd-gpu-private/actions/runs/37710987725) ran the tests and then the census from 01:07:04 to 01:29:57 UTC on 2026-10-08; `ci.json` records the jobs and artifact digest. It reproduced the three block counts of the earlier run 37707411784 exactly and added whole rows. No GPU was used. `RECORD-SHA256SUMS` covers this record.
