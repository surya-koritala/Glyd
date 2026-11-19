# Gemma 4 QAT checkpoints: int4 inside bf16? (2026-10-14)

`qat_check.py REPO OUT.json` reads one 256-row slice of every text tensor kind (first, middle and last layer) by HTTP range and asks
`lattice_int4.run` (branch lossless-ceiling) whether every group of 32 is exactly int4 x scale, with a bit-exact round trip.

| | result | bits a value if stored as int4 + scales |
|---|---|---|
| google/gemma-4-31B-it-qat-q4_0-unquantized | **every kind exact** (embedding, q/k/v/o, gate/up/down) | 4.24-4.35 → about 16.5 GB for the text model |
| google/gemma-4-26B-A4B-it-qat-q4_0-unquantized | attention and embedding exact; experts, dense MLP and router not (38-51% of groups at int4/G32; other group sizes and widths fit less: qat_groups.py, qat_widths.py) | 4.2-4.35 on attention, 9-11.5 on the rest |

So Penguin (lossless) of Gemma 4 31B QAT is the QAT model bit for bit at about 16.5 GB: a 24 GB card holds it.

## The scale's type (2026-10-14, later)

The engine's four-bit layout (gpu/glyd_gpu.cu `Int4`) decodes round_bf16((u - 8) s) with a **bf16** scale. On 31B QAT that reproduces only
~40% of groups (`qat_scale.py`): Google's Q4_0 scales are fp16. Across every group checked (1,466,368): an fp16 scale reproduces
**97.1%**, fp32 the other **2.9%**, none fail. So Penguin on QAT needs a sibling layout: the same 4-bit codes, an fp16 scale a run of 32
(round_bf16(k * s) in fp32: exact, a 4-bit k times an 11-bit significand fits 24 bits), and an exception list of fp32 scales for the
2.9%: about 4.5 bits a value, 31B ≈ 17 GB.

## The scale's form, settled (2026-10-14, evening)

`qat_scale16.py` (Hub slices by HTTP range, 256 rows of the embedding and of every matrix of layers 0, 5, 30 and 59; exact bf16 rounding
cells, a power of two's half as wide toward zero): **every one of 1,898,496 groups** is k x s with a scale of at most a **12-bit
significand** (fp16's 11 fit 97.1%); the exponents span at most 9 binades a tensor (2^-14 to 2^-4 over all); 12 groups of the embedding
need a code of +8, so they take a negative scale. So the layout `i4x` (gpu/glyd_gpu.cu `Int4H`, glyd-gpu/src/pack.rs `pack_int4h`): Int4's
codes and places, a 16-bit scale word of sign, 4-bit exponent over the pack's base and 11 fraction bits, the product in fp32 and one
rounding. 4.5 bits a weight; all 27 sampled matrices pack and decode bit for bit (31B's text matrices ≈ 17 GB).
