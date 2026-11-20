# Lossless limits of every weight format, on 2026 checkpoints

Research from the former private repo glyd-gpu-private (now part of this repository), 2026-10-06. How far below its stored size each weight format can go losslessly, and how much of
that Glyd's layouts already take: real tensors of 30 checkpoints released in 2026 (10 bf16, 18 quantized, 2 GGUF), 300 samples of
about 12 MB each read by HTTP range from the Hub (attention, MLP, experts, embeddings, LM heads, vision towers, MTP heads), measured
on a Runpod pod's CPU. Bit for bit throughout: nothing is requantized. The results JSON and every table of this page are in
`results/` (`results/report/tables.md`, `results/report/summary.json`, a JSON a checkpoint).

Three kinds of number, kept apart on every row:

- **Estimates**: the order-0 entropy of the stored value (a table a tensor, plug-in, Miller-Madow corrected) and the best single
  static context (sign / exponent / mantissa or code / scale split, row, column, 64x128 tile, the block's scale, the group's zero
  point, the left and upper neighbours, the matching tensor of another layer or expert), plug-in with its tables free and the best of
  the contexts chosen after the fact. A lower bound for a coder that uses only that context; slightly optimistic.
- **The CM code length**: a context-mixing model (every context above but the zero point, mixed, learned as it goes, nothing sent
  ahead) on the first rows of each part, up to 48 Mbit (about half of a bf16 sample's values), its rate applied to the whole part.
  It is what an arithmetic coder driven by that model would spend: the achieved side of the line, not a lower bound.
- **Achieved**: Glyd's layouts as measured (the tiered layout 10.72-10.81 bits a bf16 weight on GPUs, 10.75 pooled, `gpu/README.md`;
  as built with every byte counted, 10.756 pooled, 32.8%, `research/format-33`, PR #47; the 12-bit layout 12.04; FP8's `fp8ref`,
  6.70-6.80, packed and unpacked bit for bit on every FP8 sample here); format-33's T4b 10.712 (33.05%) and its rANS of the exponent
  10.650 (33.4%), sizes with every header counted; the quant study's fixed-rate layouts (`research/quant-2026/README.md`, computed
  from the samples' histograms, not built); and the general compressors (xz, zstd, zpaq) on the samples' stored bytes. "Layout
  estimate" in the tables is the per-tensor code length of Glyd's layout computed from each sample (the tiered code over bf16's
  exponent without its tile heads, capped at the stored 16 bits; fp8ref for FP8 codes); every other part as stored.

**The true limit of a format, for the contexts tried, lies between its best estimate and its best achieved code.** On these tensors
the static estimate and the CM are within 0.02 bits of each other on every format but INT4 AWQ (0.05: its best static context is
the group's zero point, which the CM does not see), so the limit is pinned to about 0.02 bits. The room is what lies between that
limit and Glyd's layouts.

## The key table

Bits a value pooled over the format's own samples (its codes, scales and zero points; no 16-bit parts), each sample with equal
weight (10 a checkpoint whatever its size). "Whole checkpoints" is byte-weighted over the format's checkpoints, 16-bit parts and
scales included, so a format's biggest checkpoint dominates (GLM-5.3 is 73% of the trained-bf16 bytes, Kimi-K3 67% of MXFP4's); the
per-checkpoint range is in brackets. For a quantized checkpoint "with Glyd's layouts" saves on its 16-bit and FP8 parts only (marked
†): its FP4, INT4, INT8 and GGUF bytes are as stored. "Room" is the saving at the floor less the saving with Glyd's layouts, in points
of the stored bytes, against the static estimate and against the CM.

| Format | Checkpoints / samples | Stored | Estimate: order 0 | Estimate: best static context | CM code length | Achieved: Glyd's layout | Fixed-rate layout (computed) | zpaq -m5 (static on its samples) | Own bytes: room at the floor (static / CM) | Whole checkpoints: off at the static floor | off with Glyd's layouts | Room left, points (static / CM) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 9 / 90 | 16.000 | 10.542 | **10.458** | 10.466 | 10.810 estimate; as built 10.72-10.81 (tiered), 12.04 (12-bit); T4b 10.712, rANS 10.650 | 10.810 | 10.518 (10.481) | 34.6% / 34.6% | 34.6% (34.4-34.8%) | 32.1% (32.7-33.5% without Nemotron; Nemotron 22.8%, see below) | **2.5 / 2.4** (1.1-1.9 a checkpoint without Nemotron) |
| bf16 holding FP8 values (MiniMax-M3) | 1 / 10 | 16.000 | 7.686 | **7.670** | 7.676 | 10.682 estimate (tiered) | 10.682 | 6.508 (6.483) | 52.1% / 52.0% | 59.3% | 33.4% | **25.9 / 25.8** |
| FP8, 128x128 block scales | 2 / 18 | 8.001 | 6.562 | **6.514** | 6.510 | 6.726 (fp8ref, round trip exact) | 6.726 | 6.555 (6.533) | 18.6% / 18.6% | 20.5% (19.4-21.8%) | 18.0% (16.8-19.4%) | **2.5 / 2.6** |
| FP8, a scale a tensor | 1 / 9 | 8.000 | 6.480 | **6.475** | 6.484 | 6.700 (fp8ref) | 6.700 | 6.498 (6.477) | 19.1% / 18.9% | 19.5% | 16.7% | **2.8 / 2.7** |
| FP8, a scale a row | 1 / 9 | 8.007 | 6.541 | **6.510** | 6.522 | 6.804 (fp8ref) | 6.801 | 6.566 (6.531) | 18.7% / 18.5% | 21.0% | 17.6% | **3.3 / 3.2** |
| MXFP8 | 1 / 9 | 8.250 | 6.527 | **6.496** | 6.500 | 6.957 (fp8ref, E8M0 scales as stored) | 6.760 | 6.551 (6.503) | 21.3% / 21.2% | 21.4% | 15.9% | **5.5 / 5.5** |
| NVFP4 | 4 / 30 | 4.500 | 4.192 | **4.154** | 4.144 | 4.500 (as stored) | 4.260 | 4.193 (4.157) | 7.7% / 7.9% | 17.5% (12.1-24.7%) | 12.3% † (6.0-21.2%) | **5.2 / 5.4** |
| MXFP4 | 3 / 22 | 4.250 | 3.774 | **3.720** | 3.719 | 4.250 (as stored) | 4.055 | 3.770 (3.726) | 12.5% / 12.5% | 14.6% (14.2-17.1%) | 3.4% † (2.4-6.5%) | **11.2 / 11.2** |
| INT4 GPTQ | 1 / 9 | 4.156 | 3.522 | **3.511** | 3.517 | 4.156 (as stored) | 3.712 | 3.553 (3.505) | 15.5% / 15.4% | 21.1% | 10.1% † | **11.0 / 10.9** |
| INT4 AWQ | 2 / 18 | 4.606 | 4.261 | **4.189** | 4.239 (no zero-point context) | 4.606 (as stored) | 4.431 | 4.220 (4.158) | 9.1% / 8.0% | 18.0% (16.7-19.0%) | 8.4% † (4.4-11.6%) | **9.5 / 8.8** |
| INT4 compressed-tensors | 2 / 18 | 4.309 | 3.747 | **3.726** | 3.723 | 4.309 (as stored) | 3.959 | 3.736 (3.707) | 13.5% / 13.6% | 22.8% (22.1-24.2%) | 12.6% † (11.8-14.1%) | **10.2 / 10.7** |
| INT8 W8A8 | 1 / 9 | 8.003 | 6.998 | **6.966** | 6.972 | 8.003 (as stored) | 7.220 | 6.960 (6.943) | 13.0% / 12.9% | 26.3% | 20.6% † | **5.7 / 5.6** |
| GGUF Q4_K_M (Qwen3.5-9B) | 1 / 10 | 5.047 | 4.793 | **4.746** | 4.745 | 5.047 (as stored) | 5.021 | 5.287 (5.094; its samples stored at 5.394) | 6.0% / 6.0% | 5.9% | 0.0% | **5.9 / 5.9** |
| GGUF Q8_0 (Qwen3.5-9B) | 1 / 10 | 8.500 | 7.904 | **7.887** | 7.885 | 8.500 (as stored) | 8.435 | 8.082 (7.875) | 7.2% / 7.2% | 7.2% | 0.0% | **7.2 / 7.2** |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 7 / 11 | 8.051 | 6.552 | **6.508** | 6.498 | 6.795 (fp8ref) | 6.757 | 6.565 (6.520) | 19.2% / 19.3% | - | - | - |
| 16-bit parts of quantized checkpoints | 18 / 18 | 16.000 | 10.307 | **10.284** | 10.265 | 10.855 estimate (tiered) | 10.855 | - | 35.7% / 35.8% | - | - | - |

Order 0 over samples (min / median / max), the spread behind each row: bf16 9.356 / 10.495 / 13.098; MiniMax-M3 6.483 / 6.497 /
10.655; FP8 block 6.486 / 6.544 / 6.771, a tensor 6.473 / 6.477 / 6.506, a row 6.486 / 6.503 / 6.617; MXFP8 6.499 / 6.504 / 6.555;
NVFP4 3.933 / 3.968 / 3.974; MXFP4 3.671 / 3.752 / 3.895; GPTQ 3.448 / 3.473 / 3.480; AWQ 3.610 / 3.864 / 3.896; compressed-tensors
3.428 / 3.600 / 3.798; INT8 6.921 / 7.021 / 7.068; Q4_K_M 3.859 / 4.863 / 7.529; Q8_0 7.585 / 7.671 / 7.690 (the codes alone, scales
not counted). Every sample of every row has a CM record; phase 2 had 0 errors.

Read with:

- **bf16, trained (9 checkpoints, 90 samples).** The order-0 floor on a typical tensor is 10.47-10.50 bits (10th-90th percentile;
  34.4-34.6%). The best static context takes 0.005-0.04 bits more off (median 0.018: row or column wins 88 of 90 samples, by about
  what the plug-in bias leaves), and the CM lands on the static estimate. The pooled 10.458 against 10.542 order-0 is the outliers
  below, not typical tensors. The mantissa is where the bits are: the CM spends 6.91 of its 7 bits (order-0 6.96), the sign 0.99 of
  1, the exponent 2.56 of 8. No context tried predicts mantissa bits, so for these contexts the limit is the order-0 entropy, sign 1
  + exponent ~2.56 + mantissa ~6.95, about 10.45-10.50 bits; it is lower only if something predicts the mantissa, which nothing here
  does. Whole trained-bf16 checkpoints: 34.4-34.8% off at the static floor (34.3-34.7% at the CM) against 32.7-33.5% at the layout
  estimate and 32.8% as built: **1.1-1.9 points left on every trained bf16 checkpoint but Nemotron**; format-33's T4b (10.712,
  33.05%) takes 0.3 of them at +31% decode instructions.
- **Nemotron-3-Super's early experts** (`backbone.layers.3.mixer.experts.262.down_proj`, `layers.5.mixer.experts.305.up_proj`):
  13.10 and 12.49 bits order-0, magnitudes that differ by column (layer 3) and by row (layer 5); the column and the row context bring
  them to 10.48 and 10.47, the CM to 10.67 and 10.52. The per-tensor tiered estimate on them is 16 (capped; 16.59 and 14.59
  uncapped), which is why that checkpoint's whole "with Glyd's layouts" shows 22.8% and its room 11.8 points: the as-built packer's
  step offsets were not measured on those tensors, and they are two of the five expert draws its projection rests on (the experts
  carry most of its 247 GB). Its room is an artifact of the estimate until those tensors are packed. Other outliers: gemma-4-31B-it's
  vision `position_embedding_table` (12.78 order-0, 9.84 by column, CM 9.40), granite-4.2-8b's embedding (9.36 order-0, which the
  tiered code takes only to 10.63), gemma-4-12B-it-qat's bf16 LM head (9.20, CM 8.70).
- **MiniMax-M3 is bf16 holding FP8 values**: its attention, MLP and experts are 6.48-6.51 bits order-0 (an FP8 upcast; only its
  embedding, LM head and vision tower are real bf16). It is its own row: pooled into bf16 it would drag that row's floor to about
  10.15 and its room to about 9 points. On it the floor is 59.3% and the tiered layout 33.4%: 26 points, all of it an FP8 code's.
- **FP8 (block, per-tensor, per-row scales, MXFP8): 6.48-6.51 at the floor against 6.70-6.80 with fp8ref** (round-tripped exact on
  every sample): 2.4-3.3 points on whole checkpoints, 5.5 on MXFP8 (its E8M0 scales, 8 bits stored a 32-value block, are 0.5-1.0 bit:
  fp8ref leaves them as stored). The row or block-scale context is the best static one; the CM with position and scale reaches it.
- **FP4: NVFP4 codes are 3.93-3.97 bits of 4 (1-2%), MXFP4's 3.67-3.90 (3-8%)**; the room is in the scales (NVFP4's E4M3 scales
  3.6-4.0 bits of 8, MXFP4's E8M0 0.5-1.0 bit of 8) and in the 16-bit and FP8 parts. On NVFP4's own bytes 7.7% at the floor; on
  MXFP4's 12.5%; nothing of it built. Whole checkpoints: 5.2-5.4 points (NVFP4) and 11.2 (MXFP4) left. The FP8 attention and
  embeddings inside the FP4 checkpoints (11 samples) are a row of their own: 6.51 at the floor, 6.80 with fp8ref.
- **INT4 depends on the quantizer**: GPTQ's symmetric codes 3.45-3.48 bits of 4 (15.5% on its own bytes), compressed-tensors
  3.43-3.80 (13.5%), AWQ 3.61-3.90 (9.1%; its best static context is the group's zero point, which the CM lacks, so the CM is 0.05
  above the static estimate there). INT8 W8A8 6.92-7.07 of 8 (13.0%). Their 16-bit group scales and 16-bit parts are the rest.
- **GGUF (Qwen3.5-9B Q4_K_M and Q8_0)**: 5.9% and 7.2% at the floor over the whole file, the block scales and minimums included;
  Glyd has no layout for them (0.0%).
- **The 16-bit parts of quantized checkpoints** (18 samples, one a checkpoint: embeddings, LM heads, attention, MTP heads) code as
  trained bf16 does, 10.28 at the floor against 10.86 at the tiered estimate; one of them (amd/MiniMax-M3-MXFP4's attention) is
  FP8-valued (6.55).
- **General compressors** on the samples' stored bytes (3 a checkpoint): zpaq -m5 is the strongest, 0.02-0.06 bits above the static
  estimate on every format but GGUF (0.19-0.21); xz -9e and zstd --ultra -22 --long are
  0.03-2.0 bits above it (they do not see the fields: 11.35 and 12.48 on bf16). cmix was not run (not in apt; hours a 12 MB sample).
- **Weighting.** The bits columns are sample-weighted averages (10 samples a checkpoint, and every kind with 0.1% of a checkpoint's
  bytes gets one: Qwen3.5-9B's MTP head is 2.5% of its bytes and 10% of its samples); the whole-checkpoint columns are byte-weighted
  and rest on the samples of each (format, kind) group, whose counts are in the checkpoint table below (a big MoE's experts on 5-9
  draws).

## What each context buys

The CM ladder on the samples it ran on (3 a checkpoint): order 0; + position (row, column, tile) and the block's scale; + the left,
upper and corner neighbours; + the matching tensor of another layer or expert. The static columns are the best of each kind, plug-in.
"CM's share of the values": the CM codes whole rows from the top up to 48 Mbit a part, so on bf16 about half of a sample's values
(the static estimates use all of them).

| Format | Samples | Stored | Order 0 | Fields apart | Best position context | Best neighbour context | Matching tensor | Best single context | CM: order 0 | + position, scale | + neighbours | + matching tensor | CM's share of the values |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 27 | 16.000 | 10.505 | 10.552 | 10.481 | 10.503 | 10.512 | 10.481 | 10.493 | 10.490 | 10.493 | 10.493 | 0.540 |
| bf16 holding FP8 values | 3 | 16.000 | 6.496 | 6.543 | 6.483 | 6.495 | 6.496 | 6.483 | 6.499 | 6.497 | 6.499 | 6.500 | 0.500 |
| FP8, 128x128 block scales | 6 | 8.000 | 6.582 | 6.621 | 6.533 | 6.572 | 6.577 | 6.533 | 6.546 | 6.519 | 6.521 | 6.521 | 0.600 |
| FP8, a scale a tensor | 3 | 8.000 | 6.484 | 6.532 | 6.477 | 6.483 | 6.484 | 6.477 | 6.483 | 6.484 | 6.487 | 6.487 | 0.562 |
| FP8, a scale a row | 3 | 8.000 | 6.572 | 6.613 | 6.529 | 6.569 | 6.572 | 6.529 | 6.548 | 6.535 | 6.538 | 6.538 | 1.000 |
| MXFP8 | 3 | 8.000 | 6.524 | 6.568 | 6.484 | 6.521 | 6.524 | 6.484 | 6.527 | 6.483 | 6.485 | 6.485 | 0.500 |
| NVFP4 | 9 | 4.000 | 3.959 | 3.982 | 3.928 | 3.957 | 3.960 | 3.928 | 3.959 | 3.911 | 3.911 | 3.911 | 0.554 |
| MXFP4 | 8 | 4.000 | 3.759 | 3.867 | 3.706 | 3.753 | 3.759 | 3.706 | 3.759 | 3.704 | 3.703 | 3.703 | 0.833 |
| INT4 GPTQ | 3 | 4.000 | 3.461 | - | 3.452 | 3.460 | 3.461 | 3.452 | 3.462 | 3.454 | 3.455 | 3.455 | 1.000 |
| INT4 AWQ | 6 | 4.000 | 3.871 | - | 3.801 | 3.864 | 3.874 | 3.801 | 3.871 | 3.866 | 3.856 | 3.856 | 0.528 |
| INT4 compressed-tensors | 6 | 4.000 | 3.586 | - | 3.564 | 3.584 | 3.586 | 3.564 | 3.583 | 3.558 | 3.558 | 3.558 | 0.506 |
| INT8 W8A8 | 3 | 8.000 | 6.976 | - | 6.942 | 6.975 | 6.976 | 6.942 | 6.946 | 6.942 | 6.944 | 6.945 | 0.500 |
| GGUF Q4_K_M | 3 | 4.877 | 4.722 | - | 4.682 | 4.716 | 4.315 | 4.682 | 4.722 | 4.688 | 4.686 | 4.686 | 0.561 |
| GGUF Q8_0 | 3 | 8.000 | 7.660 | - | 7.645 | 7.659 | 7.648 | 7.645 | 7.661 | 7.639 | 7.640 | 7.640 | 0.604 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 4 | 8.000 | 6.570 | 6.610 | 6.512 | 6.564 | 6.575 | 6.512 | 6.542 | 6.499 | 6.501 | 6.501 | 0.564 |

On trained bf16 and FP8 the position contexts buy 0.01-0.05 bits and the neighbours and the matching tensor nothing more (the
mixer learns to ignore them: the CM with every context is within 0.01 bits of the CM with position alone). On FP4 and INT4 the
scale context is the one that matters (NVFP4 3.959 to 3.911, MXFP4 3.759 to 3.704 bits). The matching tensor is worth nothing
anywhere: the weights of neighbouring layers and experts are independent at this level. (The "Matching tensor" column averages only the samples that have one: GGUF Q4_K_M's 4.315 is two of its three samples, each at its own order 0, not a gain.)

Where the bits go (the CM's spend by field, every context on):

| Format | Samples | Field | Stored bits | CM bits (all contexts) | Order-0 entropy of the field alone |
| :--- | ---: | :--- | ---: | ---: | ---: |
| bf16 | 90 | exponent | 8 | 2.560 | 2.631 |
| bf16 | 90 | sign | 1 | 0.991 | 1.000 |
| bf16 | 90 | mantissa | 7 | 6.914 | 6.959 |
| bf16 holding FP8 values | 10 | exponent | 8 | 2.561 | 2.570 |
| bf16 holding FP8 values | 10 | sign | 1 | 0.996 | 1.000 |
| bf16 holding FP8 values | 10 | mantissa | 7 | 4.119 | 4.163 |
| FP8, 128x128 block scales | 18 | exponent | 4 | 2.572 | 2.622 |
| FP8, 128x128 block scales | 18 | sign | 1 | 1.000 | 1.000 |
| FP8, 128x128 block scales | 18 | mantissa | 3 | 2.938 | 2.981 |
| FP8, a scale a tensor | 9 | exponent | 4 | 2.554 | 2.551 |
| FP8, a scale a tensor | 9 | sign | 1 | 1.000 | 1.000 |
| FP8, a scale a tensor | 9 | mantissa | 3 | 2.931 | 2.978 |
| FP8, a scale a row | 9 | exponent | 4 | 2.584 | 2.603 |
| FP8, a scale a row | 9 | sign | 1 | 1.000 | 1.000 |
| FP8, a scale a row | 9 | mantissa | 3 | 2.936 | 2.979 |
| MXFP8 | 9 | exponent | 4 | 2.550 | 2.573 |
| MXFP8 | 9 | sign | 1 | 1.000 | 1.000 |
| MXFP8 | 9 | mantissa | 3 | 2.934 | 2.984 |
| NVFP4 | 30 | magnitude | 3 | 2.920 | - |
| NVFP4 | 30 | sign | 1 | 0.993 | 1.000 |
| MXFP4 | 22 | magnitude | 3 | 2.756 | - |
| MXFP4 | 22 | sign | 1 | 0.943 | 0.996 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 11 | exponent | 4 | 2.563 | 2.609 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 11 | sign | 1 | 0.996 | 1.000 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 11 | mantissa | 3 | 2.934 | 2.980 |
| 16-bit parts of quantized checkpoints | 18 | exponent | 8 | 2.720 | 2.563 |
| 16-bit parts of quantized checkpoints | 18 | sign | 1 | 0.989 | 0.999 |
| 16-bit parts of quantized checkpoints | 18 | mantissa | 7 | 6.556 | 6.829 |

## Every checkpoint

"Samples by group": the (format, kind) draws each projection rests on; a group with no sample is projected at the checkpoint's
other samples of that format (bf16 by kind where the kind was sampled), a part with no sample as stored.

| Checkpoint | Format | Stored GB | Samples by group | Layout estimate | Fixed-rate layouts | Order 0 | Static floor | CM | Room left: static / CM (points) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| zai-org/GLM-5.3-BF16 | bf16 | 1,506.7 | bf16 expert 6, bf16 attn 1, bf16 mlp 1, bf16 lm_head 1, bf16 embed 1 | 33.5% | 33.5% | 34.5% | 34.6% | 34.5% | 1.1 / 1.0 |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | bf16 | 247.2 | bf16 expert 5, bf16 attn 1, bf16 mtp 1, bf16 mlp 1, bf16 embed 1, bf16 lm_head 1 | 22.8% | 22.8% | 29.2% | 34.5% | 34.3% | 11.8 / 11.5 |
| google/gemma-4-31B-it | bf16 | 62.5 | bf16 mlp 6, bf16 attn 2, bf16 embed 1, bf16 vision 1 | 32.7% | 32.7% | 34.1% | 34.6% | 34.5% | 1.9 / 1.8 |
| meta-models/Muse-Glimmer-30B | bf16 | 59.6 | bf16 mlp 6, bf16 attn 1, bf16 vision 1, bf16 lm_head 1, bf16 embed 1 | 33.2% | 33.2% | 34.4% | 34.6% | 34.4% | 1.3 / 1.2 |
| Qwen/Qwen3.5-27B | bf16 | 55.6 | bf16 mlp 5, bf16 attn 1, bf16 lm_head 1, bf16 embed 1, bf16 vision 1, bf16 mtp 1 | 33.1% | 33.1% | 34.4% | 34.5% | 34.4% | 1.4 / 1.3 |
| Qwen/Qwen3.8-27B | bf16 | 55.6 | bf16 mlp 5, bf16 attn 1, bf16 lm_head 1, bf16 embed 1, bf16 vision 1, bf16 mtp 1 | 33.1% | 33.1% | 34.4% | 34.5% | 34.4% | 1.4 / 1.3 |
| google/gemma-4-12B-it | bf16 | 23.9 | bf16 mlp 7, bf16 attn 1, bf16 embed 1, bf16 vision 1 | 33.2% | 33.2% | 34.3% | 34.4% | 34.3% | 1.2 / 1.1 |
| Qwen/Qwen3.5-9B | bf16 | 19.3 | bf16 mlp 4, bf16 attn 2, bf16 lm_head 1, bf16 embed 1, bf16 vision 1, bf16 mtp 1 | 33.0% | 33.0% | 34.3% | 34.4% | 34.4% | 1.4 / 1.4 |
| ibm-granite/granite-4.2-8b | bf16 | 17.6 | bf16 mlp 7, bf16 attn 1, bf16 lm_head 1, bf16 embed 1 | 33.1% | 33.1% | 34.7% | 34.8% | 34.7% | 1.7 / 1.6 |
| MiniMaxAI/MiniMax-M3 | bf16 holding FP8 values | 854.2 | bf16 expert 5, bf16 attn 1, bf16 mlp 1, bf16 lm_head 1, bf16 embed 1, bf16 vision 1 | 33.4% | 33.4% | 59.3% | 59.3% | 59.2% | 25.9 / 25.8 |
| Qwen/Qwen3.6-35B-A3B-FP8 | FP8, 128x128 block scales | 37.5 | fp8 expert 6, fp8 attn 1, fp8 mtp 1, fp8 mlp 1, bf16 lm_head 1 | 16.8% | 16.8% | 18.5% | 19.4% | 19.5% | 2.6 / 2.7 |
| Qwen/Qwen3.8-27B-FP8 | FP8, 128x128 block scales | 30.9 | fp8 mlp 6, fp8 attn 2, fp8 mtp 1, bf16 lm_head 1 | 19.4% | 19.4% | 21.4% | 21.8% | 21.9% | 2.4 / 2.4 |
| mistralai/Mistral-Small-4-119B-2603 | FP8, a scale a tensor | 120.9 | fp8 expert 7, fp8 attn 1, fp8 mlp 1, bf16 lm_head 1 | 16.7% | 16.7% | 19.4% | 19.5% | 19.3% | 2.8 / 2.7 |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | FP8, a scale a row | 28.6 | fp8 expert 7, fp8 attn 1, fp8 mlp 1, bf16 lm_head 1 | 17.6% | 17.7% | 20.6% | 21.0% | 20.9% | 3.3 / 3.2 |
| MiniMaxAI/MiniMax-M3-MXFP8 | MXFP8 | 443.7 | fp8 expert 7, fp8 attn 1, fp8 mlp 1, bf16 lm_head 1 | 15.9% | 18.3% | 21.1% | 21.4% | 21.4% | 5.5 / 5.5 |
| nvidia/Gemma-4-31B-IT-NVFP4 | NVFP4 | 32.6 | nvfp4 mlp 9, bf16 attn 1 | 21.2% | 23.2% | 24.3% | 24.7% | 24.7% | 3.5 / 3.5 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | NVFP4 | 23.4 | nvfp4 expert 6, fp8 attn 1, nvfp4 lm_head 1, nvfp4 mlp 1, bf16 mtp 1 | 6.0% | 10.3% | 11.7% | 12.1% | 12.7% | 6.1 / 6.7 |
| nvidia/Qwen3.8-27B-NVFP4 | NVFP4 | 21.9 | nvfp4 mlp 4, fp8 attn 4, nvfp4 lm_head 1, bf16 embed 1 | 11.9% | 14.3% | 16.3% | 16.9% | 16.9% | 5.0 / 5.0 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | NVFP4 | 21.6 | nvfp4 expert 6, fp8 attn 1, nvfp4 mlp 1, nvfp4 lm_head 1, bf16 mtp 1 | 6.2% | 10.3% | 12.2% | 13.0% | 13.5% | 6.8 / 7.3 |
| moonshotai/Kimi-K3 | MXFP4 | 1,560.9 | mxfp4 expert 9, bf16 attn 1 | 2.4% | 6.5% | 12.9% | 14.2% | 14.1% | 11.7 / 11.6 |
| deepseek-ai/DeepSeek-V4.1-Flash | MXFP4 | 510.3 | mxfp4 expert 3, fp8 embed 2, mxfp4 mtp 1, fp8 attn 1, fp8 mlp 1, fp8 mtp 1, bf16 other 1 | 6.5% | 10.0% | 13.1% | 14.8% | 15.0% | 8.3 / 8.5 |
| amd/MiniMax-M3-MXFP4 | MXFP4 | 242.7 | mxfp4 expert 8, mxfp4 mlp 1, bf16 attn 1 | 2.9% | 7.2% | 16.2% | 17.1% | 17.1% | 14.2 / 14.2 |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | INT4 GPTQ | 24.4 | int4 expert 9, bf16 attn 1 | 10.1% | 17.4% | 20.9% | 21.1% | 21.0% | 11.0 / 10.9 |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | INT4 AWQ | 25.4 | int4 expert 9, bf16 attn 1 | 11.6% | 13.2% | 18.1% | 19.0% | 18.3% | 7.3 / 6.7 |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | INT4 AWQ | 20.4 | int4 mlp 6, int4 attn 2, int4 mtp 1, bf16 lm_head 1 | 4.4% | 7.1% | 15.6% | 16.7% | 15.9% | 12.3 / 11.5 |
| RedHatAI/Qwen3.8-27B-INT4 | INT4 compressed-tensors | 19.5 | int4 mlp 6, int4 attn 3, bf16 lm_head 1 | 11.8% | 19.0% | 21.9% | 22.1% | 22.1% | 10.2 / 10.2 |
| google/gemma-4-12B-it-qat-w4a16-ct | INT4 compressed-tensors | 10.3 | int4 mlp 7, int4 attn 2, bf16 lm_head 1 | 14.1% | 17.2% | 23.8% | 24.2% | 25.5% | 10.2 / 11.5 |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | INT8 W8A8 | 14.0 | int8 mlp 8, int8 attn 1, bf16 attn 1 | 20.6% | 24.3% | 26.0% | 26.3% | 26.2% | 5.7 / 5.6 |
| unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf | GGUF Q4_K_M | 5.7 | Q4_K mlp 3, Q6_K lm_head 1, Q5_K attn 1, Q6_K mlp 1, Q4_K embed 1, Q4_K attn 1, Q6_K attn 1, Q8_0 attn 1 | 0.0% | 0.5% | 5.0% | 5.9% | 5.9% | 5.9 / 5.9 |
| unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q8_0.gguf | GGUF Q8_0 | 9.5 | Q8_0 mlp 6, Q8_0 attn 2, Q8_0 lm_head 1, Q8_0 embed 1 | 0.0% | 0.8% | 7.0% | 7.2% | 7.2% | 7.2 / 7.2 |

## General compressors on the same samples

Bits a weight of the sample's stored bytes (codes, scales and zero points as stored), the first 3 samples of each checkpoint,
single-threaded: `xz -9e`, `zstd --ultra -22 --long=27`, `zpaq -m5`.

| Format | Samples | Stored | xz -9e | zstd -22 --long | zpaq -m5 | Static floor | CM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 27 | 16.000 | 11.354 | 12.484 | 10.518 | 10.481 | 10.493 |
| bf16 holding FP8 values | 3 | 16.000 | 7.096 | 7.975 | 6.508 | 6.483 | 6.500 |
| FP8, 128x128 block scales | 6 | 8.001 | 6.646 | 6.640 | 6.555 | 6.533 | 6.521 |
| FP8, a scale a tensor | 3 | 8.000 | 6.589 | 6.557 | 6.498 | 6.477 | 6.487 |
| FP8, a scale a row | 3 | 8.006 | 6.648 | 6.651 | 6.566 | 6.531 | 6.541 |
| MXFP8 | 3 | 8.250 | 6.656 | 6.639 | 6.551 | 6.503 | 6.505 |
| NVFP4 | 9 | 4.500 | 4.246 | 4.251 | 4.193 | 4.157 | 4.141 |
| MXFP4 | 8 | 4.250 | 3.832 | 3.801 | 3.770 | 3.726 | 3.723 |
| INT4 GPTQ | 3 | 4.156 | 3.589 | 3.560 | 3.553 | 3.505 | 3.510 |
| INT4 AWQ | 6 | 4.606 | 4.300 | 4.309 | 4.220 | 4.158 | 4.210 |
| INT4 compressed-tensors | 6 | 4.314 | 3.795 | 3.785 | 3.736 | 3.707 | 3.701 |
| INT8 W8A8 | 3 | 8.002 | 7.053 | 7.049 | 6.960 | 6.943 | 6.946 |
| GGUF Q4_K_M | 3 | 5.394 | 5.359 | 5.347 | 5.287 | 5.094 | 5.099 |
| GGUF Q8_0 | 3 | 8.500 | 8.235 | 8.162 | 8.082 | 7.875 | 7.870 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 4 | 8.074 | 6.659 | 6.641 | 6.565 | 6.520 | 6.510 |

## One table for many tensors

The key's order-0 entropy (bf16: the exponent; FP8: the exponent; FP4: the magnitude; INT: the code) with a table a tensor, a table a
kind over all layers, and one table a checkpoint, the rest of the value given the key unchanged. What a layout pays for sharing
its tables.

| Format | A table a tensor | A table a kind | One table a checkpoint |
| :--- | ---: | ---: | ---: |
| bf16 | 10.542 | 10.589 | 10.776 |
| bf16 holding FP8 values | 7.686 | 7.686 | 7.845 |
| FP8, 128x128 block scales | 6.562 | 6.564 | 6.577 |
| FP8, a scale a tensor | 6.480 | 6.546 | 6.722 |
| FP8, a scale a row | 6.538 | 6.540 | 6.548 |
| MXFP8 | 6.510 | 6.511 | 6.511 |
| NVFP4 | 3.957 | 3.957 | 3.958 |
| MXFP4 | 3.754 | 3.755 | 3.755 |
| INT4 GPTQ | 3.468 | 3.468 | 3.468 |
| INT4 AWQ | 3.879 | 3.879 | 3.879 |
| INT4 compressed-tensors | 3.602 | 3.605 | 3.605 |
| INT8 W8A8 | 6.997 | 6.998 | 6.998 |
| GGUF Q8_0 | 7.669 | 7.669 | 7.669 |

## Method

- **Samples** (`plan.py`, seed 0, from the headers alone): 10 a checkpoint (a quantized one: 9 of its quantized tensors and 1 of
  its 16-bit parts), spread over the (format, kind) groups by bytes (every group with 0.1% of the bytes gets one; the rest by bytes),
  a tensor drawn by its bytes within the group, whole stored rows of about 12 MB at a random row offset. The matching tensor of a
  sample: the neighbouring expert of the same layer, else the same tensor of the nearest earlier layer with the same shape; the
  same rows of it. The headers (`hdr.py`, `gguf.py`) came by HTTP range too, 2-3 s between calls. Equal weight a sample in the
  pooled bits; bytes in the whole-checkpoint projections.
- **Estimates** (`measure.py`, phase 1): the order-0 entropy of the stored value with a table a tensor (plug-in, Miller-Madow); the
  conditional entropies H(key | context) + H(value | key) with the key the exponent (floats), the magnitude (FP4), the bit length
  (INT8) or the whole value, a context counting only with 10 values a nonzero cell; the float fields apart. Side streams (scales,
  zero points, GGUF block scales and minimums) measured the same way when they have 64 values or more, else as stored. 16-bit side
  streams with a few hundred values (FP8's 128x128 block scales) get a plug-in order-0 biased low; their weight in the pooled
  numbers is negligible.
- **The CM** (`cm.c`, phase 2): bit by bit, most informative field first; a model a context, each a hashed 64-byte line of 15
  counters over the nibble's bits; a logistic mixer a bit position trained online; the sum of -log2 p over every bit coded. Checked
  against sources of known entropy and a copy of the left neighbour (`measure.py --selftest`). It does not see the zero-point context.
- **Layouts**: the tiered code over the exponent and the quant study's fixed-rate layouts are computed from each sample's histogram
  (`research/quant-2026/quant.py`); FP8's fp8ref packs and unpacks every sample bit for bit. The as-built numbers are the records cited above.
  The phase-1 records were written before the zero points' rename (a second "zeros" part); `report.py` folds it into "zero points"
  so they count once.
- **Pod**: one Runpod pod, `limits3`, an RTX 2000 Ada as a CPU box (48 CPUs visible, a cgroup quota of 5.1 cores), $0.24/h; the plan,
  the self-test, phase 1 (319 s, 0 errors), phase 2 (480 tasks) and `report.py` ran there (`job.sh`); the records were pulled every
  minute and the pod deleted at the end. Cost: 1,352 s at $0.24/h, $0.09 (two earlier attempts on the same pod type, cut short, $0.10).

## Files

`plan.py` (the census and the samples from the headers), `hdr.py` and `gguf.py` (headers and GGUF blocks), `measure.py` (phase 1
estimates, phase 2 CM and compressors), `cm.c` (the context-mixing estimator), `report.py` (the tables), `job.sh` (the pod's
job); `results/`: `plan/` (a plan a checkpoint), `phase1/` (a record a checkpoint), `phase2.jsonl` (a line a task), `report/`
(`tables.md`, `summary.json`, a JSON a checkpoint with the projection), `env.txt`, `selftest.txt`, `plan.txt`, `run.log`, `report.log`.
