# 8-bit and 4-bit weights, losslessly: how many bits, and what a decode costs

Real checkpoints, counted on a CPU: nothing here is wired into an engine and no kernel was run. Two questions: how
many bits a weight does a lossless code need (the entropy ceilings), and what can a GPU decode in a few integer
instructions (the practical layouts: their sizes counted exactly, their decode costs estimated).

| Checkpoint | Format | What was read |
| :--- | :--- | :--- |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | MXFP4 experts (blocks of 32, an E8M0 scale each), bf16 the rest | 4 of its 92 MoE layers, whole: 1, 12, 47, 90 (10,752 expert matrices, 118.4 G weights) |
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | MXFP4 experts, bf16 the rest | 4 of 14 shards, holding layers 0-2, 4-7, 18-23, 31-33 (2,560 matrices, 30.8 G weights) |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | FP8 e4m3, a scale a 128x128 block | 4 of 163 shards, layers 0-3, 9-10, 31, 55-56 (978 matrices, 16.2 G weights) |
| [Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) | FP8 e4m3, a scale a 128x128 block | the whole model (252 matrices, 6.9 G weights) |

The size of every tensor of every checkpoint comes from the safetensors headers alone (`census.py`: HTTP range reads
of the headers). Weights came by `hf_hub_download` of single shards with no token, and each shard was deleted once it
was counted (112 GB moved in all, on a Mac; the dev box only compiled the decode kernels, on its CPU). This directory
holds histograms and sums, no weights.

## Result

Bits a weight of the payload (codes and scales), lossless, a table of its own for each tensor. "Off" is against the
stored size: MXFP4 4.25 bits with its scales, FP8 8 bits (its scales add 0.002).

| | Stored | Order 0, a table a tensor | Ceiling, best contexts tried | Fixed-rate layout | Entropy-coded layout |
| :--- | ---: | ---: | ---: | :--- | :--- |
| Kimi K3 experts, MXFP4 | 4.25 | 3.776 (11.1% off) | **3.722 (12.4% off)** | scales in 2 bits, nibbles as stored: 4.065 (4.4% off) | the scales so, nibbles by a prefix code with a table a scale value: 3.805 (10.5% off) |
| gpt-oss-120b experts, MXFP4 | 4.25 | 3.906 (8.1% off) | **3.854 (9.3% off)** | the same: 4.065 (4.4% off) | the same: 3.940 (7.3% off) |
| DeepSeek-V3 routed experts, FP8 | 8 | 6.534 (18.3% off) | **6.520 (18.5% off)** | exponent in 3 bits, an escape for the rest: 7.137 (10.8% off; 3.0-3.2% escapes in layers 9-56, 8.5% in layer 3) | exponent Huffman: 6.629 (17.1% off); in tiers of 2-bit digits: 6.695 (16.3% off) |
| Qwen3-8B-FP8 matrices, FP8 | 8 | 6.646 (16.9% off) | **6.590 (17.6% off)** | exponent in 3 bits + escape: 7.172 (10.4% off; 4.3% escapes) | exponent Huffman: 6.739 (15.8% off); tiers: 6.845 (14.4% off) |

K3, gpt-oss and Qwen3 are the means over the samples; DeepSeek-V3 weights its four samples by the layers they stand
for (below, "interp"). None of the bits include the offsets an entropy-coded layout needs to start lanes in the middle
of a tensor (0.05 bit a weight is added for them in the fit tables below, an assumption). The exponent "3 bits +
escape" and "tiers" are Glyd's `fast` and `mma` codes of the bf16 case, applied to FP8's 4-bit exponent.

By sample, bits a weight (a table a tensor):

| MXFP4 sample | Codes H0 | Scales H0 | Ceiling | Entropy-coded layout |
| :--- | ---: | ---: | ---: | ---: |
| K3 layer 1 (delta attention) | 3.763 | 0.900 | 3.727 | 3.803 |
| K3 layer 12 (delta attention) | 3.750 | 0.553 | 3.726 | 3.820 |
| K3 layer 47 (MLA) | 3.754 | 0.637 | 3.717 | 3.798 |
| K3 layer 90 (delta attention) | 3.757 | 0.669 | 3.719 | 3.799 |
| gpt-oss shards 3, 5, 9, 13 | 3.858-3.868 | 2.43-2.73 | 3.846-3.861 | 3.933-3.955 |

| DeepSeek-V3 routed experts | Code H0 | Exponent H0 | Ceiling | Exponent Huffman | Exponents outside the top 8 values |
| :--- | ---: | ---: | ---: | ---: | ---: |
| layer 3 (94 matrices) | 7.014 | 3.051 | 6.860 | 7.096 | 13.4% |
| layers 9-10 | 6.509 | 2.575 | 6.505 | 6.605 | 1.6% |
| layer 31 | 6.491 | 2.558 | 6.487 | 6.587 | 1.5% |
| layers 55-56 | 6.508 | 2.574 | 6.503 | 6.604 | 1.7% |

Decode cost, **estimates**: the SASS of straightforward, unrolled, untuned decoders (`decode_sass.cu`; sm_89, sm_90a
and sm_100a; ALU instructions only, loads, stores and branches left out; 64 weights a thread, the table in shared
memory):

| Decode | ALU instructions a weight | Table lookups a weight |
| :--- | ---: | ---: |
| MXFP4 scale layout (2-bit offsets from a row base, an escape to a side list) | 0.25-0.26 (134 for 16 scales) | 0 |
| MXFP4 nibbles by a prefix code (16 symbols, the sign in the code) | 9.6 (sm_89, sm_90a), 10.5 (sm_100a) | 1 |
| FP8 exponent by a prefix code, sign and mantissa from a raw plane of 4 bits | 14.2 (sm_89, sm_90a), 14.8 (sm_100a) | 1 |

The fixed-rate MXFP4 layout leaves the nibbles as stored, so a path that reads packed FP4 today reads them unchanged
and pays only the scale decode. The prefix-coded layouts put 10 to 15 integer instructions between memory and the
tensor cores for every weight: they are for capacity, and for layers decoded once to a scratch buffer and used for
many tokens. (Not counted: the "3 bits + escape" and "tiers" FP8 decodes, which index their side streams by a count
over a warp's escapes.)

## What the numbers say

- **FP8 e4m3 has 15-18% to give; the redundancy is the exponent.** The sign is 1.000 bit, the 3 mantissa bits 2.98,
  the exponent field 2.56-2.58 of its 4 bits (DeepSeek-V3 layers 9-56; 3.05 in layer 3; 2.70 for Qwen3-8B; a table a
  matrix). Given the exponent, sign and mantissa are 3.93-3.97 bits. The block scales put each block's largest value at
  448, so every block's largest exponent field is 15 in both models (100% of blocks): a per-block base has nothing to
  add. The scales (32 or 16 bits a 16,384 weights) are 0.02% of the file.
- **Conditioning adds little.** The exponent given its left, upper, or both neighbours saves 0.01-0.05 bit, given
  the block's scale 0.01-0.04, given the position in the block 0.000; a table for each column or row, whose cost is not
  counted, 0.04-0.08. A table for each *matrix* matters: over DeepSeek-V3's four samples the experts need 6.668 bits
  with one table for all of them and 6.555 with one each (the ceilings above are the latter).
- **Early layers give less.** DeepSeek-V3's layer 3 experts: 12.3% off at order 0 (14.2% with contexts), against
  18.6-18.9% in layers 9, 10, 31, 55, 56. Fixed-width windows suffer more: 13.4% of layer 3's exponents fall outside the
  top 8 values (k = 3) against 1.5-1.7% later and 3.8% in Qwen3-8B; a window of 4 (k = 2) misses 24-75%. Kimi K3's
  layer 1, the earliest MoE layer, is like its later ones.
- **MXFP4's nibbles hold 3-6%, its scales most of the rest.** Kimi K3's 4-bit codes are 3.756 bits (6.1% off), gpt-oss's
  3.864 (3.4%): the magnitude index (8 values) has 2.75 bits (K3) and no value over 22% of the codes, flat enough that
  a fixed-width code cannot use it. The E8M0 scales (a byte a 32 weights, 5.9% of the file) are 0.55-0.90 bits in K3 and
  2.4-2.7 in gpt-oss. All of K3's scales in a row sit in a window of 3 values (0.3-1.1 per million outside it);
  gpt-oss's 0.01-0.17% do not.
- **Contexts for the nibbles: only the block's scale gives more than 0.03 bit.** Given the previous 1 or 2 codes 0.005-0.03
  bit, the row above 0.000-0.006, the position in the block 0.000, a block summary sent as side information (its
  largest code) net 0.02-0.03; given the block's scale, already decoded, 0.054 for K3 and 0.027 for gpt-oss. In K3's
  layer 12 the codes are 3.18 bits at scale 122 (1.2% of blocks) and 3.92 at scale 120 (8.4%): a block whose scale
  is high has probably an outlier, and its other codes sit low (an interpretation). K3's block maxima are 3, 4 or 6 (21%, 63%, 16%: the scale
  rounded up), gpt-oss's 4 or 6 (50%, 50%).
- **A generic compressor already gets close.** zstd -19 on the stored bytes, a tensor of codes and one of scales
  apart: K3's packed nibbles come to 94.2% (3.77 bits) and its scales to 9.1-10.0%, 3.79 bits in all, 10.8% off;
  gpt-oss's to 96.9% and 13.9%, 3.91 bits, 8.0% off; DeepSeek-V3 layer 31's experts 82.1-82.2% (6.57 bits); Qwen3-8B-FP8
  82.8-83.1%. The ceilings above are 12.4%, 9.3%, 6.49 bits (layer 31) and 17.6%: a purpose-built codec has 0.5 to 1.6
  points beyond zstd -19, and the entropy-coded layouts below are at or just under zstd's level, not above it (xz -6 is
  within a point of zstd).

## The layouts

The FP8 layout of an exponent as a k-bit offset from a per-block base plus exceptions finds no base
to use (every block's largest exponent is 15), so its window is the top 2^k values; k = 3 is 7 bits and 1.5-1.7%
exceptions in DeepSeek-V3's later layers (3.8% in Qwen3-8B, 13.4% in layer 3), each costing a code and a position
(22 bits assumed: 8 and 14 for a place in a 128x128 block), 7.35 bits in all, 8% off, against 7.14 for the escape
stream below. k = 2 is out (24-75% exceptions).

- **FP8, 3 bits + escape** (Glyd's `fast`): sign and mantissa raw, the exponent a 3-bit code for the tensor's 7
  commonest values, the eighth code an escape to the exponent raw in a side stream (3.0-3.2% of weights in
  DeepSeek-V3 layers 9-56). 7.14 bits in DeepSeek-V3's experts, 7.17 in Qwen3-8B.
- **FP8, tiers of 2-bit digits** (Glyd's `mma`): the 3 commonest exponents one digit, a 3 going on to the next 3, then
  the next 3, then the exponent raw (0.8% of the weights take the raw tail in layers 9-56, 3.2% in layer 3, 1.6% in Qwen3-8B): 6.70 bits and 6.85.
- **FP8, exponent Huffman**, a table a tensor: 6.63 and 6.74. Lengths limited to 8 bits cost up to 0.03 bit, to 10 bits
  0.004 (on the pooled histograms of the DeepSeek-V3 and Qwen3 samples).
- **MXFP4 scales in 2 bits**: a base byte a row (the middle of its window of 3: 0.003 bit a weight) and a 2-bit code
  a block, 3 an escape to a side list. 2.000 bits a scale in K3 (about 1 per million escapes), 2.007 in gpt-oss.
  By itself 4.4% off, at 0.26 instruction a weight.
- **MXFP4 nibbles by a prefix code**: Huffman over the 16 symbols with a table for each scale value: 3.740 bits a
  nibble in K3 against 3.793 with one table; lengths limited to 8 bits cost 0.002 or less.

Not counted in the entropy-coded layouts: the offsets that let lanes start in the middle of a tensor. The fit tables
add **0.05 bit a weight** (an assumption; Glyd's prefix-coded bf16 layout is 0.28 bit over its tensor floor,
10.88 against about 10.6, its Huffman rounding included, so the allowance may be low: every 0.1 bit is 34 GB on Kimi K3).

## What it means

Sizes are of the whole checkpoints. The bf16 tensors (attention, shared experts, latent projections, router,
embeddings) are coded at their measured floor plus 0.2 bit (the gap between Glyd's tiered bf16 layout, 10.80, and the
10.6 floor in `gpu/README.md`); K3's attention floor is 10.88 bits (11.66 in layer 12, 10.5-10.6 in the others).
Capacity: a 192 GB B200 (1,536 GB a node), and 180 GB, which NVIDIA's DGX B200 lists (1,440 GB a node). **A reserve of 8 GB a
GPU** (assumed) is kept for the CUDA context, communication buffers and activations; the last number is how many
contexts of 128k tokens the rest holds, at the KV a context needs by the model's config.

**Kimi K3**, 1,560.9 GB stored. A 128k context needs 4.06 GB: 24 MLA layers x 576 latent values x 2 bytes x 131,072
tokens is 3.62 GB, and the 69 delta-attention layers' fp32 state (96 x 128 x 128 x 4 bytes a layer) 0.43 GB, with no
copy of the latent cache across ranks, as with data-parallel attention (copied to 8 tensor-parallel ranks it is
about 29 GB a context).

| Layout | Weights, GB | Off | 8xB200, 192 GB: headroom / after the reserve (128k contexts) | 8xB200, 180 GB | 8xH200, 141 GB: headroom |
| :--- | ---: | ---: | ---: | ---: | ---: |
| as stored | 1,560.9 | | -24.9 / -88.9 (0) | -120.9 / -184.9 | -432.9 |
| fixed rate: scales in 2 bits, nibbles as stored | 1,462.1 | 6.3% | 73.9 / 9.9 (2.4) | -22.1 / -86.1 | -334.1 |
| entropy coded: scales in 2 bits, nibbles by a prefix code (+0.05 bit) | 1,390.5 | 10.9% | 145.5 / 81.5 (20) | 49.5 / -14.5 | -262.5 |
| ceiling, order 0 a tensor | 1,362.4 | 12.7% | 173.6 / 109.6 (27) | 77.6 / 13.6 (3.3) | -234.4 |
| ceiling, best contexts | 1,344.0 | 13.9% | 192.0 / 128.0 (31) | 96.0 / 32.0 (7.9) | -216.0 |

As stored, Kimi K3 is 1.6% over one 8xB200 node of 192 GB. The fixed-rate layout makes it fit with 74 GB over,
a few contexts after the reserve; the entropy-coded one leaves 145 GB, about 20 contexts of 128k after it. On 180 GB
GPUs only the ceilings leave room. On 8xH200 it fits under no lossless layout (the lowest ceiling is 1,344 GB, the
node 1,128 GB).

**DeepSeek-V3**, 688.6 GB stored (the weights alone do not fit 8xH100's 640 GB). A 128k context needs 9.21 GB: 61 layers
x 576 latent values x 2 bytes x 131,072 tokens. "interp": the early layers' lower savings fade across the sampled
depths (3, 9.5, 31, 55.5); "equal": the four samples count equally, which gives layer 3 a quarter of the experts.

| Layout | Weights, GB | Off | 8xH100, 640 GB: headroom / after the reserve (128k contexts) | 8xH200, 1,128 GB |
| :--- | ---: | ---: | ---: | ---: |
| as stored | 688.6 | | -48.6 / -112.6 (0) | 439.4 / 375.4 (41) |
| 3 bits + escape, interp / equal | 612.7 / 616.0 | 11.0 / 10.5% | 27.3 / -36.7 | 515.3 / 451.3 (49) |
| tiers of 2-bit digits, interp / equal | 579.5 / 589.1 | 15.8 / 14.4% | 60.5 / -3.5 | 548.5 / 484.5 (53) |
| exponent Huffman (+0.05 bit), interp / equal | 573.9 / 581.7 | 16.7 / 15.5% | 66.1 / 2.1 (0.2) | 554.1 / 490.1 (53) |
| ceiling, order 0 a tensor, interp / equal | 561.5 / 569.5 | 18.5 / 17.3% | 78.5 / 14.5 (1.6) | 566.5 / 502.5 (55) |

On 8xH100 the weights fit with an entropy-coded layout (66 GB over); one 128k context then fits only with a reserve of
about 7 GB a GPU or less (2 GB spare at 8 GB, 9.21 GB wanted, a quarter of that with an fp8 latent cache, not counted).
On 8xH200 it already fits as stored with 40 contexts of 128k; the lossless layouts add 12 more.

**gpt-oss-120b**, 65.2 GB stored. A 128k context needs 4.84 GB (18 full-attention layers x 8 KV heads x 64 x 2 x 2
bytes x 131,072 tokens; the 18 sliding layers' 128 tokens add 5 MB).

| Layout | Weights, GB | Off | One 80 GB GPU: headroom / after the reserve (128k contexts) |
| :--- | ---: | ---: | ---: |
| as stored | 65.2 | | 14.8 / 6.8 (1.4) |
| fixed rate: scales in 2 bits, nibbles as stored | 61.2 | 6.2% | 18.8 / 10.8 (2.2) |
| entropy coded (+0.05 bit) | 60.1 | 7.9% | 19.9 / 11.9 (2.5) |
| ceiling, best contexts | 58.1 | 11.0% | 21.9 / 13.9 (2.9) |

gpt-oss-120b already fits one 80 GB GPU. The layouts give 4 to 5 GB more room: 1.4 contexts of 128k become 2.2 (fixed
rate) or 2.5 (entropy coded).

**Qwen3-8B-FP8**, 9.4 GB stored: exponent Huffman 7.6 GB (19.8% off: its embedding and lm_head, bf16, are 26% of the
file and take 33% off), tiers 7.7 GB (18.8%), 3 bits + escape 7.9 GB (16.3%).

## Prior art

- [ECF8](https://arxiv.org/abs/2510.02676) (arXiv 2510.02676, "To Compress or Not? Pushing the Frontier of Lossless
  GenAI Model Weights Compression with Exponent Concentration"): Huffman-coded FP8 exponents, GPU decode by cascaded
  256-entry tables. It reports 14.8% off DeepSeek-R1-0528, 14.4% off Qwen3-235B-A22B-Instruct-2507-FP8 and 9.8% off
  Qwen3-8B-FP8, and an exponent entropy of 2 to 3 bits: measured here, 2.56-3.05. Here DeepSeek-V3 is 16.7% off as a
  whole model with an entropy-coded layout (18.7% the ceiling) and Qwen3-8B-FP8's matrices 15.8%; its 9.8% on
  Qwen3-8B is close to the 10.4% of the fixed-rate escape layout.
- [Lossless Compression of Neural Network Components: Weights, Checkpoints, and K/V Caches in Low-Precision
  Formats](https://arxiv.org/abs/2508.19263) (arXiv 2508.19263): Huffman on FP8 streams; for FP4 (MXFP4 and NVFP4) it finds
  the 4-bit values close to uniform and compresses only the scales (5% off DeepSeek-R1's NVFP4, where the scales are 10% of
  the bytes). Measured here: gpt-oss's post-trained nibbles are nearly uniform (3.864 bits) and K3's, trained with
  quantisation in the loop, are not (3.756): the nibbles give 3 to 6%, the scales 3.5 to 5.5% of the stored size.
- [DFloat11](https://arxiv.org/abs/2504.11651) (bf16 to about 70%, exponent Huffman-coded, decoded by tables on the GPU)
  and [ZipNN](https://arxiv.org/abs/2411.05239) are the bf16 case `gpu/` already covers; both rest on the exponent's
  entropy, as FP8's does.
- Not measured: NVFP4 (blocks of 16, E4M3 scales), E5M2, any other FP8 or FP4 release; K3's layers are 4 of 92.

## Licence

The Kimi K3 License (`LICENSE` in its repository, 3,065 bytes) is MIT-like: use, copy, modification, distribution and
sale of the weights without restriction, the notice kept. Two conditions bind a model-as-a-service business with more
than 20 million dollars of revenue in 12 months (a separate agreement with Moonshot AI before commercial use) and a
product with more than 100 million monthly users or 20 million dollars of monthly revenue ("Kimi K3" shown in it);
neither applies to internal use or to use through Moonshot's own or certified partners' products. Downloading and
counting weights is within it and the repository is not gated (no token was used). A losslessly re-encoded copy of
the weights is a copy of the Software and carries the notice. gpt-oss and Qwen3 are Apache 2.0; DeepSeek-V3's model
licence is its own (permissive, with use restrictions) and was not needed to count.

## Method, and what is assumed

Measured: every histogram (`fp4.py`, `fp8.py`, `bf16.py`: of the stored bytes, a matrix at a time, nothing lossy),
every entropy (`ent.py`: plug-in, with the Miller-Madow correction for the conditional ones), every Huffman length,
every escape and exception count, the generic compressors (`generic.py`), the SASS counts, the census.
Assumed or estimated, labelled where they enter: the decode instruction counts (untuned code); 0.05 bit for the
offsets; the bf16 code at floor + 0.2 bit, and 10.55 bits where a role was not read (K3's dense layer, embeddings,
lm_head and vision tower; 7.3% of K3's bytes are bf16); 8 GB of reserve a GPU; the KV a context (the latent cache of
512 + 64 values, fp32 delta-attention state, no replication); the depth interpolation for DeepSeek-V3; the tables'
size (16 entries a tensor, 1-2 MB for all of DeepSeek-V3's 45,000 tensors, not counted).
The "ceiling" is the lowest of the models tried (order 0 a tensor; the neighbours; the scale; the position; a block
summary sent as side information, its cost counted; for scales the left and upper neighbours and a row median),
not a bound on what any model could do.

```
python census.py REPO results/census_REPO.json            # headers only
python fetch.py REPO FILE                                   # one shard, no token (HF_HOME a throwaway directory)
python fp4.py SHARD out.json  |  fp8.py SHARD out.json  |  bf16.py SHARD out.json
python report_fp4.py out.json --json results/fp4_X.json  |  report_fp8.py out.json --json results/fp8_X.json
python fit.py && python tables.py                           # the sizes and the tables above, from results/
python generic.py ...                                       # zstd -19 and xz on raw tensors
nvcc -arch=sm_100a -cubin decode_sass.cu -o d.cubin && cuobjdump -sass d.cubin | python sass_count.py scales=512 decodeILi6=64 decodeILi8=64
```

`results/` holds the summaries (`fp4_*`, `fp8_*`, `bf16_*`, `census_*`, `generic.json`, `sass_count.txt`) and `fit.json`.
