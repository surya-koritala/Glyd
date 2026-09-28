# Glyd weights on the GPU

A bf16 model's weights held compressed in VRAM, decoded on the GPU bit
for bit. A bf16 weight is a sign, 8 exponent bits and 7 mantissa bits;
in a trained model the sign and mantissa are noise, and the exponent
carries about 2.6 bits of information. Both formats keep the sign and
mantissa byte as it is and code the exponent:

| Format | Exponent | Bits a weight (Qwen2.5) | Decode |
| :--- | :--- | ---: | :--- |
| `huffman` (`pack`) | per-tensor prefix code read by counting leading zeros (as short as Huffman's on every tensor measured), 32 streams a tile | 10.88 | a lane decodes its stream in turn |
| `fast` (`pack_fast`) | 3-bit code into the tensor's 7 most common exponents, an escape to the exponent itself | 11.25 | bit operations, every weight in parallel |
| `mma` (`pack_mma`) | 2-bit digits in tiers: the tensor's 3 commonest exponents, digit 3 going on to the next 3, then the next 3, then the exponent itself; laid out in the order the tensor cores take their operand | 10.80 | in registers, straight into the tensor cores' operands |
| `mma12` (`pack_mma12`) | a 4-bit code into the tensor's 15 commonest exponents, the rest in a step's exception list; the same layout | 12.04 | three byte permutes per four weights: for GPUs whose memory outruns the tiered decode |

The floor for any code that sees each tensor's exponents on their own
is about 10.6 bits a weight.

For one-token steps (generation) the product is fused: `fast_gemv` and
`gemv` read the packed weights, decode them in registers and multiply,
never writing bf16 out. Otherwise a matrix is decoded into one scratch
buffer and multiplied by PyTorch (`unpack`, `fast_unpack`).

## Measured

RTX 4080 SUPER (16 GB), PyTorch 2.14, CUDA 13.0; Qwen2.5-7B-Instruct,
greedy (`e2e.py MODEL --format mma --fused --baseline`): generating for
1 to 48 sequences at once, a prompt's forward pass, and perplexity on
Wikipedia text (enwik8 from its 10th MB, 200 windows of 64 tokens):

| | bf16 | Glyd `mma` |
| :--- | ---: | ---: |
| Peak VRAM | 15.25 GB | **10.61 GB** |
| 1 sequence | 43.4 tokens/s | **55.7** (1.28x) |
| 4 sequences | 167.9 | **217.5** (1.30x) |
| 16 sequences | 652.0 | **814.2** (1.25x) |
| 32 sequences | 1153.7 | **1518.9** (1.32x) |
| 48 sequences | 1664.8 | **1882.5** (1.13x) |
| 64 sequences | 2160.0 | **2244.7** (1.04x) |
| Prompt of 16 tokens | 24 ms | **19 ms** |
| Prompt of 64 tokens | 27 ms | **26 ms** |
| Prompt of 128 tokens | 29 ms | 29 ms |
| Prompt of 256 tokens | 43 ms | 45 ms |
| Prompt of 512 tokens | 79 ms | 85 ms |
| Prompt of 1024 tokens | 154 ms | 163 ms |
| Prompt of 2048 tokens | 301 ms | 330 ms |
| Prompt of 4096 tokens | 645 ms | 702 ms |
| Perplexity, 64-token windows | 17.0015 | 17.0052 |
| Perplexity, 512-token windows | 7.5677 | 7.5660 |

The weights are the model's to the bit; the product sums in another
order than cuBLAS, which moves the logits by a rounding: the next token
chosen is bf16's 98.13% of the time (98.49% on the 512-token windows).
bf16 against itself, two windows a pass instead of one: perplexity
17.0153, the same next token 98.33% of the time.

`mma_gemm` (1 to 64 tokens): a warp step is 1024 weights, 64 rows by
16 columns, in the order `mma.sync.m16n8k16` takes its B operand: 1280
bytes (each lane's 32 tier-1 digits, then its 32 sign-and-mantissa
bytes) and a block of the step's escapes (its tier-2 digits in words of
16 back from the block's end, its tier-3 digits and exponent bytes from
its start). A lane's weights arrive in four loads and two words of the
block's ends; one warp scan places the tier-2 digits, each lane decodes
16 of them (and their tier-3 digits, placed by a second scan) into
shared memory, and each lane's tier-1 digits take their symbols or
those bytes in one table lookup and one byte permute for 4 weights. A
pair of bf16s is then one permute and one rotate (each weight's byte
carries its pair's other sign). No lane waits on another's escapes:
a step decodes in the same instructions however its escapes fall. The
steps are split evenly over the blocks (stream-K), a row block's parts
added in a fixed order by its last block: the same result every run. On
7B's matrices it reads the packed weights at 94% of the bandwidth bf16's
product reaches, 1.30-1.34x faster than bf16 at one token on the MLP's,
1.06-1.19x at 64.

`mma_gemm_big` (more tokens: a prompt) is a tiled GEMM with its warps
split: four produce, copying X's tile of each stage (64 columns) into
shared memory by `cp.async` and decoding W's steps of it into B
fragments there; four consume, each 64 tokens by 64 rows on the tensor
cores with their fragments double-buffered, never waiting on a decode
(on GeForce Ada eight, each 64 tokens by 32 rows: below, prompts on
GeForce Ada).
Three stages are in flight, passed between the two by named barriers. A
tile is 128 tokens by 128 rows of W, or past 128 tokens 256 by 64 (a
weight decoded once for twice the tokens; on GeForce Ada only where the
last tile of 256 would be more than half full, or past 1024 tokens tiered
and 4224 12-bit, as measured). On GeForce Ada as many
blocks as the GPU holds at once each take an equal share of the tiles'
stages in turn, their stages in flight from one tile to the next
(stream-K: no wave part empty, no pipeline filled again); a tile that
several blocks share is summed by the last of them to finish, in their
order (the same result every run). Elsewhere, until measured there, a
block a tile, K split where the tiles would leave the last wave part
empty, the parts summed by a second kernel in a fixed order. Past 128 tokens the product is bound by the tensor cores,
not by memory, so the most it can be is bf16's time; it is within 5-10%
of it. Measured on Qwen2.5-7B's matrices: the consumers alone come
within 1-4% of cuBLAS (one warp an SM quarter keeps the tensor cores
full: 106 TFLOPS, as cuBLAS's kernel); the rest is the producers'
decoding sharing the SM. Longer prompts on GeForce Ada decode each
matrix once, ahead of its product (below: long prompts).

The other formats, 128 new tokens, one sequence:

| | Peak VRAM | Tokens/s | Tokens as bf16's |
| :--- | ---: | ---: | ---: |
| bf16 | 15.25 GB | 43.3 | |
| `fast`, fused | 11.05 GB | 55.1 | 128 of 128 |
| `huffman`, fused | 10.60 GB | 52.0 | 54 of 128 |
| `fast`, decoded then PyTorch's matmul | 11.05 GB | 18.3 | 128 of 128 |

The fused `fast` product on 7B's matrices (`shapes.py`): 662–695 GB/s of
packed weights, 1.29–1.48x bf16's matrix-vector time (the small key and
value projections, which sit in L2, 0.66x). The fused `huffman` product:
1.17–1.46x (575–690 GB/s of packed weights). Profiled (Nsight Compute),
its integer pipe had been the limit (83% busy, 44 instructions a
weight): the reader is two words and a funnel shift, a code's rank is its
value plus a stored base (taken mod 32 by the shuffle), a weight's float
is assembled by byte permutes around an exponent kept in a float's place,
and the next stream word is asked for a word ahead. Rows longer than a
tile are split across tiles, their parts added in fp32 by the last tile
of the row. Tried and dropped: two interleaved streams a lane, a table
giving up to three codes a lookup. The fused product sums in
another order than cuBLAS, as any two GEMM kernels do; the decoded path
multiplies with PyTorch's own kernel and gives bf16's logits bit for bit
where a matrix is decoded whole (Qwen2.5-0.5B: logits and 128 tokens
identical; the 7B output layer is decoded in blocks to cap the scratch).

Several tokens at once in the fast format: `fast_gemm` reads it,
decodes each 64-weight step of a 64-row tile into shared memory once for
all the tile's tokens and multiplies on the tensor cores; K is split
across blocks where W has few rows, the parts added in a fixed order (the
same result every run). `e2e.py` uses it for prompts of up to 64 tokens;
longer ones decode each matrix and use PyTorch's matmul. A prompt's
forward pass, Qwen2.5-7B (`--prefill`):

| Prompt | bf16 | Glyd `fast` |
| ---: | ---: | ---: |
| 16 tokens | 24 ms | 35 ms (was 61) |
| 64 tokens | 27 ms | 40 ms (was 61) |
| 128 tokens | 29 ms | 61 ms |
| 512 tokens | 79 ms | 113 ms |
| 2048 tokens | 301 ms | 337 ms |

Past 64 tokens the fused kernel is not yet faster than decoding the matrix
and multiplying (profiled: its loads queue up and stall, at 29%
occupancy); a GEMM at cuBLAS's efficiency is what closes that.

## Two layouts: the most memory, or the lightest decode

The tiered code (`mma`) takes the most off (10.80 bits a weight) and
costs the most arithmetic to decode; where memory is the limit, as on an
RTX 4080 SUPER at a few tokens a step, that is the faster one too. Where
the GPU's memory outruns the decode (an H100's HBM3, or many tokens a
step), the 12-bit layout (`mma12`, 12.04 bits) decodes four weights with
three byte permutes and nothing across lanes. Qwen2.5-7B-Instruct, RTX
4080 SUPER, the same harness:

| | bf16 | `mma` | `mma12` |
| :--- | ---: | ---: | ---: |
| Weights | 15.23 GB | **10.32 GB** | 11.42 GB |
| Tokens/s at 1 / 8 / 32 / 64 sequences | 43.4 / 332.4 / 1153.7 / 2160.0 | **55.7 / 424.2 / 1518.9** / 2244.7 | 51.9 / 398.6 / 1452.4 / **2455.9** |
| Prompt of 64 / 2048 tokens | 27 / 301 ms | 26 / 330 ms | **23** / 318 ms |
| Perplexity; MMLU (1,000) | 17.0015; 73.50% | 17.0052; 73.50% | 17.0052; 73.50% |

Per matrix (down_proj, 3584 x 18944): one token 141 us (`mma`), 152
(`mma12`), 198 (bf16); 64 tokens 193, 170, 236.

On an H100 SXM (HBM3, 3.35 TB/s) the order turns: the tiered decode is
bound by arithmetic, the 12-bit one keeps up with memory. Qwen3-32B's
layer 0, one token: down_proj 75-79 us (`mma12`), 126 (`mma`), 90
(cuBLAS); gate and up 75-76, 130, 86-88; 16 tokens 80-82, 132-137, 89-91
(at 64 tokens cuBLAS leads: 95 against 141-143). End to end, GPU time a
generated token (Hugging Face's loop, bound by the CPU at about 68 ms a
token for all three, over 16 tokens):

| H100 SXM | bf16 | `mma` | `mma12` |
| :--- | ---: | ---: | ---: |
| Qwen3-32B: weights | 65.52 GB | **44.45 GB** | 49.23 GB |
| Qwen3-32B: GPU time a token | 28.22 ms | 40.58 ms | **26.39 ms** |
| Qwen3-32B: MMLU (1,000) | 78.3% | 78.2% | 78.1% |
| Qwen2.5-7B: weights | 15.23 GB | **10.32 GB** | 11.42 GB |
| Qwen2.5-7B: GPU time a token | 7.47 ms | 10.73 ms | **7.35 ms** |
| Qwen2.5-7B: MMLU (1,000) | 73.2% | 73.2% | 73.3% |

(benchmarks/gpu/lambda-gpu_1x_h100_sxm5-20260926-114529 for bf16 and
`mma`, and -120552 for `mma12`; the first run's `mma12` kept its
exponents in local memory, since fixed: same bytes, same answers.)

### Many tokens a step on an H100: the copy engine and wgmma

`mma_gemm_wg` (the 12-bit layout, Hopper) takes steps of 17 to 128
tokens, and prompts of up to 512 (below). One lane of a warp of its own hands the copy engine (TMA) a
stage at a time: 64 columns of 128 of W's rows, their compressed steps
as bulk copies of 6 KB, their exceptions, and X's tile through a tensor
map in wgmma's 128-byte swizzle, into a ring of 4 to 8 stages in shared
memory, each landing on an mbarrier. Each consumer warpgroup decodes its
64 rows straight into wgmma's A registers, as Machete does with 4-bit
weights, and multiplies with X read from shared memory. The work is
split evenly over the SMs (stream-K); rows shared by blocks are summed
by the last to finish, in a fixed order: the same result every run.
As first built (2026-09-26, before the changes further below), on an
H100 SXM, Qwen3-32B's matrices took, GPU time in us (bf16 through cuBLAS
/ `mma_gemm` / `mma_gemm_wg`; at 128 tokens the middle column is the
matrix decoded for cuBLAS):

| Tokens | 1 | 16 | 32 | 64 | 128 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| gate, up (25600 x 5120) | 88 / 75 / 78 | 89 / 81 / 79 | 90 / 113 / **83** | 94 / 140 / **92** | 99 / 335 / 124 |
| down (5120 x 25600) | 90 / 74 / 78 | 91 / 78 / 78 | 92 / 105 / **83** | 96 / 127 / **92** | 98 / 328 / 121 |
| q (8192 x 5120) | 29 / 29 / 31 | 30 / 33 / 31 | 30 / 45 / 34 | 31 / 59 / 40 | 32 / 113 / 48 |

A small matrix had few stages a block, so a copy's latency at the start
and the sum of shared rows at the end were not covered: Qwen2.5-7B's
k_proj (512 x 3584) took 15-22 us against cuBLAS's 6, q_proj and o_proj
(3584 x 3584) 15-23 against 7-8. End to end they still cost more than
the large matrices saved past 16 sequences (GPU time a step over 16
tokens, the prompt's share included: 3.45 ms of Qwen3-32B's at 32 and
64 sequences, decoded for cuBLAS):

| H100 SXM, GPU time a step | 1 | 16 | 32 | 64 sequences |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-32B, bf16 (65.52 GB) | 28.23 | 31.22 | 32.44 | 35.50 ms |
| Qwen3-32B, `mma12` (49.23 GB) | **26.14** | 31.42 | 37.71 | 43.04 ms |
| Qwen2.5-7B, bf16 (15.23 GB) | 7.47 | 8.28 | 8.54 | 9.06 ms |
| Qwen2.5-7B, `mma12` (11.42 GB) | **7.22** | 8.85 | 10.52 | 11.97 ms |

Perplexity through it (64-token windows): 17.0065 against bf16's
17.0178 (Qwen2.5-7B). Next then: the small matrices (the launch
overlapped with the kernel before it, a cheaper sum of shared rows: the
second done since, below) and prompts. `python glyd_gpu.py` checks
`mma_gemm_wg` on a Hopper GPU (benchmarks/gpu/h100-tma-2026-09-26).

On an H100 PCIe (2026-09-28) four things held it back, found by
profiling a model's step kernel by kernel and the kernel stage by stage:

- A stage's exceptions were set by every lane for every entry, and a few
  layers' matrices have dozens a stage (Qwen3-8B's gate and up in layers
  1-3: up to 85, 0.4% of their weights): those products took 421-577 us
  against the other layers' 102. The lanes now take a stage's entries 32
  at a time, each setting one byte in its warp's scratch (`mma_gemm_mid`
  too); outputs bit for bit as before.
- ptxas serialized every wgmma (its C7513): a pair of stages' second one
  was skipped inside the loop, so a path ran a set of A registers into
  its products while earlier ones read it. The stages now run in pairs
  and an odd one after them. With four consumer warpgroups and the TMA
  warp a thread had 96 registers (spills and C7512): two warpgroups now,
  a unit of 128 rows, at every token count, and a unit's sums added in
  place (a copy of them had held 64 more registers at 128 tokens).
- Tiles of 96 and 112 tokens as well as 128 (X's tile a stage, beside
  W's 12 KB, is the most read past 64 tokens).
- A matrix of few units and short blocks (Qwen3-8B's o, 4096 x 4096)
  takes a whole number of blocks a unit, summed over as few parts.

GPU time a step (`e2e.py --format auto --fused --merge --profile`,
bf16's from the same GPU; Qwen3-32B's bf16 row from bf16prof.py, since
e2e.py's own bf16 profile runs out of memory at 32B:
e2e-qwen3-32b-bf16.txt):

| H100 PCIe, GPU time a step | 1 | 8 | 32 | 64 sequences |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-8B, bf16 | 12.30 | 13.34 | 14.57 | 15.66 ms |
| Qwen3-8B, `mma12`, before | 11.24 | 12.45 | 15.55 | 17.57 ms |
| Qwen3-8B, `mma12` | **11.26** | **12.46** | **13.55** | **14.86 ms** |
| Qwen3-32B, bf16 | 42.44 | 44.40 | 47.10 | 49.56 ms |
| Qwen3-32B, `mma12`, before | 34.75 | 37.48 | 44.97 | 49.84 ms |
| Qwen3-32B, `mma12` | **34.76** | **37.53** | **40.65** | **43.90 ms** |

A layer's products against cuBLAS's (weights read from memory, as a
model's step reads them), 32 / 64 / 96 / 128 tokens: Qwen3-8B 0.84x /
0.86x / 0.95x / 1.01x (were 0.92x / 1.00x / 1.20x / 1.20x), Qwen3-32B
0.77x / 0.84x / 0.91x / 1.04x; the output layer (151936 x 4096) 0.78x
at 32 and 0.93x at 64. Still longer than cuBLAS's: the small matrices
past 32 tokens (Qwen3-8B's q, k, v 1.04x and o 1.14x at 64), and 113-128
tokens (benchmarks/gpu/h100-hopper-2026-09-28).

`mma_gemm` (steps of 1-16 tokens on Hopper) now takes a step's
exceptions the same way where a block's steps have more than one each
(Qwen3-8B's gate, up and down in layers 1-3: 4-5 a step), in a copy of
its loop of its own: those layers take 210 us at 8 tokens, were 242-269,
the others as before. With a unit's parts counted by one thread's
acq_rel atomic (not every thread's fence) and no division a stage in the
TMA kernel's producer, a step's GPU time at 1 / 8 / 32 / 64 sequences is
11.13 / 12.32 / 13.48 / 14.64 ms on Qwen3-8B (was 11.26 / 12.46 / 13.55
/ 14.86), 34.65 / 37.34 / 41.24 / 44.41 on Qwen3-32B (was 34.76 / 37.53
/ 40.65 / 43.90: at 17-128 tokens its layer takes 2-6% more than at
153fc96, 1.08x cuBLAS's time at 128 tokens, as measured; the cause not
found) (benchmarks/gpu/h100-prompts-2026-09-28). On an RTX 4080 SUPER
the same changes leave `mma_gemm_mid` (17-64 tokens, two warpgroups
there) at main's time, within 0.2% at 17-32 tokens and 0.8% faster at
48-64 on Qwen2.5-7B's matrices, and the steps at main's but on
Qwen3-8B's layer 2, 2-3% faster; main's library against this build bit
for bit (benchmarks/gpu/rtx4080s-hopper-branch-2026-09-28).

### Prompts on an H100

GLinear sends prompts of up to 512 tokens to `mma_gemm_wg` too
(`GLYD_WG_MAX`); past that it decodes the matrix for cuBLAS, as before.
Past 128 tokens the kernel's tile is 256 tokens, or 192 where that takes
no more chunks (129-192 tokens, 257-384), so a weight is decoded once
for up to 256 tokens. A launch takes up to two chunks (512 tokens, where
O / 64 is even; else one: its units stay within the O / 64 done
counters) in one stream-K split, chunk by chunk, so the blocks at work
at once read the same weights. The TMA warp is the first of a warpgroup
that hands its registers to the consumers (`setmaxnreg`: 40 a thread
there, 232 a consumer's), which at 256 tokens hold 128 sums a thread and
still two sets of A registers.

One decoder layer's products, weights read from memory (us; cuBLAS on
bf16 / decoded for cuBLAS / `mma_gemm_wg`; GLinear's pick in bold):

| H100 PCIe, tokens | 129 | 192 | 256 | 384 | 512 | 640 | 1024 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 251 / 666 / **280** | 247 / 677 / **302** | 254 / 697 / **364** | 322 / 781 / **634** | 465 / 921 / **736** | 549 / **983** / 1087 | 812 / **1300** / 1436 | 3620 / **4124** / 5761 |
| Qwen3-32B | 598 / 1733 / **695** | 612 / 1753 / **733** | 629 / 1864 / **907** | 857 / 1975 / **1512** | 1103 / 2294 / **1665** | 1484 / **2633** / 2440 | 2436 / **3520** / 3418 | 9461 / **10474** / 13431 |

One forward pass over a prompt (`e2e.py --format auto --fused --merge
--prefill`; before: `GLYD_WG_MAX=128`, the matrices decoded for cuBLAS
past 128 tokens), ms, bf16 / Glyd before / Glyd:

| H100 PCIe, tokens | 129 | 192 | 256 | 384 | 512 | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 26.8 / 33.9 / 22.3 | 22.7 / 35.1 / 23.2 | 23.4 / 36.7 / 24.5 | 25.9 / 41.6 / 33.2 | 32.6 / 47.7 / 39.2 | 55.8 / 71.1 / 69.9 | 110.4 / 126.9 / 125.0 | 224.5 / 242.6 / 242.3 |
| Qwen3-32B | 58.6 / 131.1 / 63.1 | 64.1 / 140.1 / 69.1 | 70.3 / 142.1 / 79.7 | 89.3 / 165.3 / 118.3 | 113.1 / 189.8 / 141.0 | 217.2 / 295.4 / 296.7 | 426.6 / 517.2 / 519.6 | 866.2 / 981.0 / 974.5 |

Qwen3-8B's pass at 129-256 tokens is mostly the host's launches, and its
time varies from run to run: in four runs of these builds bf16's took
22.7-51.5 ms there and Glyd's 22.3-28.7, while a layer's products take
1.12-1.43x cuBLAS's time. Every other length is still longer than
bf16's. A chunk costs Qwen3-8B's layer about 250 us in tiles of 128
tokens, 300 in 192 and 365 in 256, and no one part of it is what bounds
it: builds that skip one (timing only) are 5-10% faster without X's
copies, 7-14% without W's, 8-16% without the decode and 10-14% without
the sum of a unit's parts, and without the decode still 1.07-1.34x
cuBLAS's time at 192-512 tokens; wgmma reading A from shared memory in
place of registers is no faster. At 256 tokens ncu has 0.31 instructions
issued a cycle a scheduler, whose two consumer warps wait on the
decode's dependent instructions and on the stage's barrier. Next here:
the parts' sums (a unit shared by fewer blocks), then the decode off the
consumers' path. The whole blocks a unit above now apply only where they
idle at most a sixth of the blocks and give a unit 3 or more, or past
128 tokens (parts of 96 and 128 KB in tiles of 192 and 256; timed at
256): on other models' q, k, v and o the rule had cost up to 47%
(Gemma-2-9B's q, k, v: 64 blocks of 114). Logs:
benchmarks/gpu/h100-prompts-2026-09-28; the builds that skip a part, and
the A/B builds, are a commit and a patch (builds/*.patch): no-x, half-x,
no-w, no-decode and a-registers-one-set on 11f2a42, no-parts-sum and
a-from-shared-memory-no-decode on 2370b55, units-row-by-row and
whole-blocks-switch on 84348d6 (the synthetic whole-blocks runs on its
kernel as it was before RU was 32-bit), every-thread-fence on 709bb28;
those that skip a part timed by kbench.py with its check against fp32
taken out (`sed '/assert err < 1e-2/d'`).

### Which layout on which GPU

Measured the way a server runs a model: q, k and v as one product and
gate and up as another, for bf16 and Glyd alike (`e2e.py --merge`, as
vLLM runs them), and the GPU time of a generated token apart from the
prompt's (`--profile`: 17 steps less 1, over 16). Qwen2.5-7B-Instruct,
GPU time a token at 1 / 8 / 32 / 64 sequences (logs:
benchmarks/gpu/lambda-gpu_1x_*-2026092617*, -18*, and
rtx4080s-layouts-2026-09-26):

| GPU | bf16 | tiered (`mma`, 10.80 bits) | 12-bit (`mma12`, 12.04 bits) |
| :--- | ---: | ---: | ---: |
| RTX 4080 SUPER 16 GB | 21.97 / 22.85 / 26.19 / 27.67 ms | **16.52 / 17.37 / 18.80** / 25.14 | 17.48 / 18.25 / 19.67 / **21.66** |
| A10 24 GB | 33.71 / 34.77 / 35.50 / 37.98 ms | 24.33 / 25.61 / 35.99 / 49.66 | **24.40 / 25.76 / 29.03 / 33.05** |
| A100 40 GB | 14.18 / 14.87 / 16.18 / 17.80 ms | 15.34 / 16.00 / 23.99 / 28.81 | **11.84 / 13.77** / 17.16 / 19.95 |
| H100 SXM 80 GB | 6.90 / 7.50 / 8.02 / 8.55 ms | | **6.55 / 7.24** / 8.69 / 9.99 |
| H100, Qwen3-32B | 27.90 / 29.78 / 31.44 / 33.52 ms | | **24.50 / 26.75** / 33.46 / 37.35 |

So: on Ada the tiered layout, the smallest, is also the fastest to 32
sequences (24-28% under bf16's time); on an A10 the 12-bit one is 13-28%
under at every count; on an A100 and an H100 the 12-bit one is 3-16%
under to 8 sequences and 6-17% over at 32 and 64 (their small matrices).
Perplexity as bf16's everywhere (17.00 against 17.01 on the A10, for
one). `glyd_gpu.best_layout()` (and `e2e.py --format auto`) takes the
tiered layout on Ada and wherever only it fits, the 12-bit one
elsewhere.

On GDDR Ampere and Ada, steps of 17 to 64 tokens run `mma_gemm_mid`, the
TMA kernel's plan with this generation's instructions: a producer warp's
cp.async onto each stage's mbarrier, X's tile read by ldmatrix, the rows
decoded into mma.sync's registers. It takes an A10's 64 sequences from
34.64 to 33.05 ms and an RTX 4080's from 21.99 to 21.66.

On an A100 that plan was the slower (its waits on the mbarriers are polls
on this generation), and `mma_gemm_mid` is a kernel of its own there:
`mma_gemm_big`'s split of the warps, the producers' reads staged. Each
producer warp takes one step of each stage (64 columns of two row
blocks): it copies the step and the step's exceptions by cp.async three or
four stages ahead into its share of a ring, and decodes them from there
into B fragments in shared memory; four consumer warps multiply, 16 or 32
tokens by a row block each; named barriers pass the stages between the
two, and the work is split evenly over the SMs (stream-K). Profiled
(Nsight Compute), `mma_gemm` at 64 tokens had kept 2 warps a scheduler,
each waiting on its next step's loads, on its exceptions (a load that
waits on another) and on X's rows, which every row block reads again from
L2 (48 MB for q, k and v at 64 tokens, against the matrix's 38). GPU time
of one layer's matrices (q, k, v and gate, up merged), CUDA graphs, in us:

| Qwen3-8B, layer 12 | 17 tokens | 32 | 48 | 64 |
| :--- | ---: | ---: | ---: | ---: |
| cuBLAS on bf16 | 310 | 315 | 320 | 318 |
| `mma_gemm` | 298 | 321 | 388 | 418 |
| `mma_gemm_mid` | **273** | **276** | **310** | **311** |

Qwen3-32B's layer 20 takes 0.79-0.87x cuBLAS's time at 17 to 64 tokens,
every matrix under it; Qwen3-8B's gate, up and down 0.81-0.94x, its q,
k, v (6144 x 4096) 1.01-1.13x and o (4096 x 4096) 1.24-1.36x (few stages
a block, so the pipeline's start and the sum of shared rows weigh). End
to end (`e2e.py --format auto --fused --merge --profile 32`), GPU time a
step of Qwen3-8B at 1 / 8 / 32 / 64 sequences: 15.83 / 19.69 / 19.79 /
21.80 ms against bf16's 17.45 / 20.12 / 20.82 / 21.67 (with `mma_gemm`:
15.83 / 19.47 / 21.27 / 25.96). Past 64 tokens `mma_gemm_big` runs,
1.18-1.20x cuBLAS's time at 96 and 128.

### Steps of 65-128 tokens and prompts on an A100

On an A100 (SXM4 40 GB, 2026-09-28) `mma_gemm_mid` now takes steps of up
to 128 tokens, in one launch: units of two row blocks by 96 tokens (four
consumer warps of 48) or by 128 (eight of 32: four of 64 spill their
sums). Its consumers take the next stage's fragments after a unit's sum
out rather than holding them through it, which also ends the 64-token
kernel's spill; its producers decode through `Nib`'s own code (bit for
bit as before). A step of up to 128 tokens is one C call. A layer's
products (q, k, v and gate, up merged; CUDA graphs, weights read from
memory), cuBLAS on bf16 = 1.0:

| Tokens | 17 | 32 | 48 | 64 | 65 | 96 | 128 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B, layer 12, before | 0.87 | 0.86 | 0.96 | 0.97 | 1.20 | 1.19 | 1.20 |
| Qwen3-8B, layer 12 | 0.86 | 0.85 | **0.91** | **0.92** | **1.07** | **1.05** | 1.16 |
| Qwen3-14B, layer 20, before | 0.78 | 0.78 | 0.82 | 0.82 | 1.11 | 1.12 | 1.20 |
| Qwen3-14B, layer 20 | 0.78 | 0.78 | **0.78** | **0.78** | **0.92** | **0.90** | **1.01** |

GPU time a step (`e2e.py --format auto --fused --merge --profile 16`;
Qwen3-14B's bf16 from `bf16prof.py`, as bf16's and Glyd's copies do not
fit 40 GB at once):

| A100, GPU time a step | 1 | 8 | 32 | 64 | 128 sequences |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B, bf16 | 18.88 | 19.59 | 21.12 | 21.13 | 25.71 ms |
| Qwen3-8B, `mma12`, before | 15.27 | 18.88 | 19.14 | 21.69 | 29.45 |
| Qwen3-8B, `mma12` | **15.27** | **18.81** | **18.54** | **21.05** | 28.17 |
| Qwen3-14B, bf16 | 26.76 | 28.92 | 32.89 | 36.01 | 41.15 ms |
| Qwen3-14B, `mma12`, before | 23.10 | 25.24 | 29.24 | 32.95 | 49.23 |
| Qwen3-14B, `mma12` | **23.11** | **25.05** | **28.70** | **31.81** | 42.41 |

At 128 sequences Qwen3-8B generates 2993.6 tokens/s against bf16's
2997.4 (was 2747.0 against 3018.1), Qwen3-14B 2594.0 against 2657.4.

A prompt's products are bound by the traffic between L2 and the SMs:
`mma_gemm_big`'s blocks of 256 tokens by 64 rows read 38 KB of it for a
million products (X's tile for every 64 of W's rows), blocks of 256 by
128 in bf16 (cuBLAS's shape here) 24 KB, and it took 1.57-1.60x cuBLAS's
time at every length past 256 tokens, the ratio of the two. On an A100
a 12-bit prompt now runs blocks of 256 tokens by 128 rows: eight
consumer warps that load a step's fragments as they multiply it (a
thread's 128 sums in 168 registers), four producer warps; blocks of 128
tokens where the last block of 256 would be half empty or less, to 640
tokens (257-384 and 513-640: 13-14% and 5-15% faster there; GeForce
Ada's rule, below, with the A100's bound); and past 768 tokens each
matrix decoded for cuBLAS (2.9 ns a weight, bound by DRAM, while
cuBLAS's time grows with the tokens). A layer's products, cuBLAS = 1.0
(Qwen3-8B / Qwen3-14B):

| Tokens | 129 | 256 | 384 | 512 | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 1.62 / 1.54 | 1.60 / 1.65 | 2.07 / 1.95 | 1.60 / 1.46 | 1.44 / 1.45 | 1.49 / 1.54 | 1.58 / 1.58 |
| now | 1.44 / 1.42 | 1.42 / 1.43 | 1.77 / 1.70 | 1.49 / 1.43 | 1.33 / 1.35 | 1.17 / 1.18 | 1.09 / 1.09 |

One forward pass (`e2e.py --prefill`), ms (the time to first token
through `generate()` is 3-5 ms more, alike):

| Prompt | 128 | 256 | 512 | 1024 | 2048 | 4096 tokens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B, bf16 | 41.0 | 41.1 | 48.4 | 90.1 | 174.3 | 347.9 |
| Qwen3-8B, `mma12`, before | 44.9 | | 68.3 | 120.7 | 237.1 | 489.2 |
| Qwen3-8B, `mma12` | **40.9** | 45.5 | 64.5 | 112.0 | 195.7 | 368.2 |
| Qwen3-14B, bf16 | 45.1 | 48.9 | 82.7 | 151.2 | 291.1 | 584.6 |
| Qwen3-14B, `mma12`, before | 50.8 | | 113.1 | 207.3 | 416.6 | 855.4 |
| Qwen3-14B, `mma12` | 45.3 | 62.9 | 111.7 | 198.6 | 347.5 | 653.7 |

Still slower than bf16: prompts past 128 tokens (Qwen3-8B 6-33%,
Qwen3-14B 12-35%), and Qwen3-8B's steps of 97-128 tokens (1.16x a
layer). A prompt's matrices decoded ahead beside the products before
them (as on GeForce Ada, `GLYD_AHEAD_MIN=129`) were slower on an A100
than each matrix decoded before its product, at every length measured and
2, 3 or 4 warps an SM (Qwen3-8B, 2048 tokens: 214.8-274.3 ms against
195.7; the fused kernel took 237.1 there before): not used here. Logs:
benchmarks/gpu/a100-ampere-2026-09-28.

### Short prompts: one C call a product, and stream-K

To 512 tokens (tiered) or 2047 (12-bit; to 1023 when this was written) a
prompt's products on GeForce Ada are `mma_gemm_big`'s, and on other GPUs
but Hopper every prompt's (an A100's 12-bit to 768 tokens).
Two things set a short prompt's time against bf16's there. The host:
Qwen3-1.7B's pass of 128 tokens is issued in about the time the GPU
takes to run it, and a prompt's product cost 12 us of host time a call
(a Python path, and a second launch to sum a split K) against
F.linear's 6. And the kernel's grid: where its blocks would not fill
the GPU, K was split and the parts summed by a second kernel (0.6-0.8
ms of a Qwen3-1.7B pass at 128-512 tokens), and waves ran part empty.
A prompt's product is now one C call, as a generation step's
(`_lib.step`), and on GeForce Ada the kernel runs by stream-K (above;
other GPUs keep the grid until it is measured there). The 12-bit
layout's fused kernel is then the faster one to 1024 tokens (the decode
ahead starts there, was past 640; for exact and unfused products still
past 640). One forward pass
(`e2e.py --prefill --merge`, bf16 and Glyd merged alike), RTX 4080
SUPER, ms, main / now:

| Prompt | 128 | 256 | 384 | 512 | 640 | 768 | 896 | 1024 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B, bf16 | 10.8 | 14.0 | 18.7 | 24.3 | 25.9 | 32.1 | 38.2 | 42.3 |
| tiered | 11.4 / 10.3 | 15.8 / 14.6 | 23.9 / 20.1 | 25.4 / 23.8 | 34.7 / 28.4 | 36.1 / 33.8 | 44.3 / 39.4 | 46.2 / 43.3 |
| 12-bit | 11.4 / 10.2 | 15.4 / 14.5 | 23.2 / 19.2 | 24.7 / 23.7 | 33.8 / 27.9 | 35.4 / 33.1 | 43.3 / 39.0 | 45.2 / 43.5 |
| Qwen3-4B-Instruct-2507, bf16 | 20.7 | 28.0 | 39.5 | 49.1 | 62.7 | 73.0 | 88.4 | 96.1 |
| tiered | 19.7 / 19.4 | 29.9 / 29.3 | 50.2 / 44.4 | 53.3 / 52.6 | 75.4 / 67.4 | 79.0 / 76.2 | 100.2 / 90.6 | 104.0 / 97.5 |
| 12-bit | 18.6 / 19.2 | 29.3 / 28.7 | 49.3 / 41.7 | 52.0 / 51.5 | 73.6 / 65.4 | 77.1 / 76.2 | 96.7 / 91.0 | 100.5 / 98.9 |

(main: 0.21.0; the columns past 512 are the long prompts' change, below,
as well.) The time to the first token through `generate()` moves as the
pass (Qwen3-1.7B's at 128 tokens 11.6 ms tiered, 11.8 12-bit, against
bf16's 13.0 and 12.8); generation is as before. At 128 tokens every one
is under bf16's time, and Qwen3-1.7B's at 512. Past that the fused
kernel's products stay 7-18% over cuBLAS's in a pass (both models at 256
and 384 tokens, profiled): switched off one at a time (Qwen3-4B's layer
at 512 tokens), its consumers alone come within 2% of cuBLAS, its
producers' copies of X's tiles cost 4% (a tile of 64 rows reads X again
for every 64 rows of W), W's loads 2.5%, the decode 1.6% (12-bit) or 6%
(tiered); at 257-384 and 513-640 tokens (blocks of 128) each weight is
decoded M / 128 times, which the tiered decode cannot keep up with
(Qwen3-4B at 384: 1056 us a layer against the 12-bit layout's 903 and
cuBLAS's 847). The SM clock is not it: 2640-2655 MHz against cuBLAS's
2670 (290-304 W of 320). Logs:
benchmarks/gpu/rtx4080s-short-prompts-2026-09-28.

### Prompts on GeForce Ada: two consumers a scheduler

`mma_gemm_big`'s products on GeForce Ada keep their four producer warps
(X's tile by `cp.async`, W's steps decoded into B fragments in shared
memory) and now have eight consumers, each 64 tokens by 32 rows (half a
row block: 64 sums a thread), in place of four of 64 by 64. With the
producers they fit the 168 registers a thread a block of 12 warps gets,
and each of the SM's four schedulers has two consumers to keep its
tensor cores busy instead of one, each as lean as cuBLAS's (2.2
instructions a product against 2.3). Blocks of 256 tokens take them in
both layouts, blocks of 128 by two row blocks in the 12-bit layout; the
tiered layout's blocks of 128 keep four (its producers, a weight decoded
for 128 tokens, fall behind eight consumers: 11-14% slower). The order of
the sums is a unit's stages' as before, so the bits are main's (1176
products compared with main's kernel of the same tiling).

A layer's products (q,k,v and gate,up merged; each timed alone after an
L2 flush, median of 9; layer 10's real weights), cuBLAS = 1.00 in the same
run, main / now (each through its own library, two runs of each in turn;
lengths 300, 640 and 896 take blocks of 128):

| Tokens | 256 | 300 | 512 | 640 | 896 | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B, 12-bit | 0.997 / 0.973 | 1.145 / 1.116 | 0.973 / 0.951 | 1.124 / 1.081 | 1.049 / 1.011 | 1.055 / 1.026 | 1.014 / 0.981 | 1.014 / 0.977 |
| Qwen3-4B-Instruct-2507, 12-bit | 0.979 / 0.962 | 1.133 / 1.106 | 1.056 / 1.028 | 1.075 / 1.039 | 1.052 / 1.016 | 1.066 / 1.034 | 1.071 / 1.035 | 1.082 / 1.043 |
| Qwen3-8B, 12-bit | 0.938 / 0.914 | 1.064 / 1.030 | 1.008 / 0.979 | 1.114 / 1.067 | 1.032 / 0.988 | 1.012 / 0.982 | 1.045 / 1.012 | 1.049 / 1.019 |
| Qwen3-1.7B, tiered | 1.030 / 1.014 | 1.221 / 1.213 | 1.001 / 0.987 | 1.232 / 1.223 | 1.129 / 1.119 | 1.080 / 1.060 | 1.036 / 1.014 | 1.033 / 1.011 |
| Qwen3-4B-Instruct-2507, tiered | 1.026 / 1.019 | 1.158 / 1.152 | 1.087 / 1.070 | 1.132 / 1.124 | 1.121 / 1.115 | 1.093 / 1.076 | 1.091 / 1.072 | 1.101 / 1.079 |
| Qwen3-8B, tiered | 0.965 / 0.951 | 1.140 / 1.132 | 1.035 / 1.013 | 1.214 / 1.207 | 1.121 / 1.113 | 1.036 / 1.016 | 1.066 / 1.039 | 1.070 / 1.068 |

The tiered layout's blocks of 128 (300, 640, 896) keep four consumers of
a row block (the same bits; the compiler schedules the kernel a little
differently since its source took CR): 0.5-0.9% in these runs. Under 1.00 a layer has a
product cuBLAS is slow on (Qwen3-1.7B's down at 2048-4096 tokens: 602 /
1199 us against 527 / 1039; Qwen3-8B's q,k,v and o at 1024). With it the
12-bit layout's fused kernel is as fast as the decode ahead or faster to
2047 tokens, so it takes the prompt to there (was to 1023; one pass,
1024-1792 tokens: Qwen3-4B 0.5-2.4% faster at four lengths of seven,
within 0.5% at three, Qwen3-1.7B 2.5-7.4% faster at six, 3% slower at
1280; at 2048 Qwen3-4B 3.3% slower).

One forward pass (`e2e.py --prefill --merge`, bf16 and Glyd merged
alike; bf16 in the same runs), ms, main / now, each the mean of two runs
in fresh processes (main, now, now, main):

| Tokens | 128 | 256 | 384 | 512 | 640 | 768 | 1024 | 1536 | 2048 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B, bf16 | 10.8 | 14.1 | 18.7 | 24.4 | 25.9 | 32.1 | 42.3 | 63.0 | 86.5 | 184.6 |
| tiered | 10.4 / 10.6 | 14.6 / 14.4 | 20.1 / 20.0 | 23.8 / 23.4 | 28.4 / 28.4 | 33.8 / 33.8 | 43.4 / 43.6 | 65.5 / 65.8 | 87.2 / 87.4 | 185.8 / 186.4 |
| 12-bit | 10.2 / 10.1 | 14.5 / 14.2 | 19.2 / 18.6 | 23.7 / 23.2 | 27.8 / 26.8 | 33.1 / 32.4 | 43.5 / 42.5 | 67.8 / 63.6 | 87.2 / 88.2 | 185.1 / 185.3 |
| Qwen3-4B-Instruct-2507, bf16 | 20.7 | 28.1 | 39.5 | 49.1 | 62.7 | 73.0 | 96.3 | 147.5 | 199.0 | 446.1 |
| tiered | 19.5 / 19.4 | 29.3 / 28.9 | 44.7 / 44.5 | 52.9 / 52.3 | 67.8 / 68.1 | 76.7 / 77.1 | 98.2 / 98.3 | 150.1 / 150.1 | 200.2 / 199.9 | 450.1 / 449.6 |
| 12-bit | 19.3 / 19.0 | 28.9 / 28.3 | 41.9 / 40.8 | 51.8 / 50.9 | 65.8 / 64.2 | 76.6 / 75.2 | 99.3 / 99.2 | 150.2 / 150.0 | 200.3 / 201.0 | 448.5 / 449.9 |

At 2048 and 4096 tokens both decode ahead (code this change does not
touch): alone in fresh processes they take the same time (Qwen3-4B 12-bit
199.0 / 445.7 ms in all four runs, Qwen3-1.7B within 0.4 ms); after the
shorter prompts of the table's runs they came 0.3-1.2% apart, either way
by model. The time to the first token moves with the pass. Generation
(`--batch 1,8,32,64 --tokens 64`) is as before: every kernel a step runs
has the same SASS as main's build (498 kernel builds compared; only the
prompt kernel differs), and the runs agree within 1% (Qwen3-4B 12-bit
73.2 / 551.7 / 1954.5 / 3381.2 tokens/s against main's 73.2 / 551.5 /
1953.2 / 3381.2).

Measured and not taken (Qwen3-4B's layer, 12-bit):

- All warps multiplying and each decoding its own B fragments in
  registers a step ahead, the way CUTLASS's and Marlin's main loops run
  (tiles of 128 x 256 and 256 x 128, 2-4 `cp.async` stages, mbarriers,
  codes decoded by value): at best 1.10x cuBLAS at 4096 tokens, 1.5-2.4x
  at 256-512. Nsight Compute: the decode, its patch and addressing make
  7-8 instructions a product (cuBLAS's kernel: 2.3), a warp is away from
  its tensor-core instructions some 40% of the time, and with two warps a
  scheduler the pipe idles whenever both are: 42.6% busy against cuBLAS's
  48.7%.
- W decoded a stage ahead into shared memory by all eight warps (each
  weight once a tile, a block barrier a stage): 1.22x, the barrier
  aligning every warp.
- CUTLASS's sm80 mixed-input GEMM (v4.7.1, `OpMultiplyAddMixedInputUpcast`)
  with its cheapest converter, u8 to bf16: 3.1-6.4% over its own bf16 GEMM
  (which is cuBLAS's time) on Qwen3-4B's gate,up and down at 1024 and 4096
  tokens. The 12-bit code's converter (the exponent table, the paired
  sign-and-mantissa bytes, the exceptions) would only be heavier.
- Codes by value: the 15 commonest exponents of all 700 Linears of
  Qwen3-1.7B, 4B and 8B are contiguous, so a pack whose codes are the
  exponent less the first decodes 8 weights in 7 instructions, not 18.
  1.5% in the all-warps kernel above; nothing here, where the 12-bit
  producers idle 60% of the time. Not packed that way.

Logs, and the scripts that took them (mb.py per layer, bits.py, the
CUTLASS benchmark, e2etab.py and layertab.py for the tables):
benchmarks/gpu/rtx4080s-prefill-2026-09-28 (layer-*: the per-layer table
above; e2e-*: the pass; route*: the fused kernel against the decode ahead;
phase1*, groups2, tm128: the candidates as prototyped, v0 main's kernel,
v18 all warps decoding in registers, v23 / v24 consumers of half a row
block in blocks of 128 / 256, v26 v24 with two consumer barriers).

### Long prompts: each matrix decoded once, beside the products before it

A prompt's products are bound by the tensor cores. `mma_gemm_big`
decodes each weight again for every 256 tokens and its producers'
decoding costs its consumers 5-10% of cuBLAS's time; decoded once into
the scratch buffer, a matrix costs its decode, 0.44 ms a layer of
Qwen3-4B on an RTX 4080 SUPER (6% of the layer's products at 4096
tokens, 22% at 1024, 43% at 512), unless it runs beside something. On
GeForce Ada a prompt past 512 tokens (from 2048 in the 12-bit layout,
whose fused kernel is the faster to there; from 1024 when this was
written) now decodes each matrix ahead of its product, on a second
stream, beside the products before it (`model.Ahead`):

- the order: the GLinears a prompt calls whole, recorded from the first
  such prompt (merged groups once; that one runs the fused kernel), each
  matrix given a place in the scratch buffer used as a ring and decoded
  there as soon as the products that read its place are done (the buffer
  holds the largest twice at least: Qwen3-8B's 0.40 GB, was 0.27);
- beside what: cuBLAS's kernels for the large products (CUTLASS's blocks
  of 256 x 128 and 128 x 256, 224 registers a thread, 72 KB of shared
  memory) leave an SM room for a few more warps, and slow down 0-4% beside
  them; its kernels for the small ones (a single stage, or 96 KB of
  shared memory) leave none or slow down 13-19%. A decode runs beside the
  order's first product and those of 0.4 times the largest matrix's
  weights (an output layer aside) and 30 GFLOP at least, as many rows as
  the product's time allows;
- how: 3 warps an SM in the tiered layout, 2 in the 12-bit one, a block
  an SM, on a stream of high priority, launched 5 us after the product
  (`hold`). A block placed first on an idle SM sets its shared-memory
  carveout, which cannot change until the SM is empty: a decode placed
  first left the product's blocks no room (a down projection took 1.47x
  its time);
- a product waits for its matrix's decode on a CUDA event, never the
  host, and decodes on the current stream what the products before it
  had no time for (all of the order's first, 0.07 ms).

`exact=True` runs the same path, F.linear as before: the logits are
bf16's bit for bit (`check_api.py`: a prompt of 2100 tokens). One forward
pass (`e2e.py --prefill`; q, k, v and gate, up merged, bf16 and Glyd
alike), RTX 4080 SUPER, ms, before and after:

| Prompt | 128 | 512 | 1024 | 2048 | 4096 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B, bf16 | 10.6 | 24.3 | 42.1 | 86.6 | 185.3 |
| tiered, was | 11.5 | 25.2 | 45.9 | 91.5 | 190.1 |
| tiered | 11.2 | 25.2 | 43.3 | 86.8 | 186.0 |
| 12-bit, was | 11.3 | 24.7 | 45.0 | 89.6 | 186.5 |
| 12-bit | 11.3 | 24.7 | 43.4 | 87.2 | 185.6 |
| Qwen3-4B-Instruct-2507, bf16 | 20.7 | 49.1 | 96.1 | 198.3 | 445.4 |
| tiered, was | 19.7 | 53.1 | 102.8 | 211.5 | 476.5 |
| tiered | 19.7 | 53.1 | 97.5 | 199.5 | 447.5 |
| 12-bit, was | 18.6 | 52.0 | 100.3 | 205.4 | 463.0 |
| 12-bit | 18.6 | 52.0 | 98.5 | 198.9 | 445.8 |
| Qwen3-8B, tiered, was | 33.2 | 94.4 | 185.6 | 370.8 | 811.3 |
| Qwen3-8B, tiered | 33.1 | 94.8 | 175.2 | 345.1 | 767.6 |
| Qwen3-8B, 12-bit, was | 30.3 | 91.1 | 179.8 | 363.8 | 786.6 |
| Qwen3-8B, 12-bit | 30.2 | 91.0 | 175.7 | 345.7 | 762.7 |

(bf16's Qwen3-8B does not fit 16 GB.) Between those lengths the fused
kernel's blocks now fit the prompt too, on GeForce Ada and in an A100's
12-bit layout to 640 tokens (elsewhere as before, until measured):
Qwen3-4B's 300 tokens take 44.7 ms tiered and 39.9 12-bit against
bf16's 38.2 (were 48.7 and 47.8), its 640 tokens 67.1 and 66.5 against
62.9 (were 75.2 and 73.8). The time to the first token through
`generate()` moves as the pass; generation is as before.
What is left over bf16's time past 1024 tokens is the decode's traffic
(3.35-3.5 bytes a weight) beside cuBLAS's products, and what the
products before the first could not hide. At 1024 tokens Qwen3-1.7B's
down projection (2048 x 6144) gets a single-stage kernel from cuBLAS,
too slow beside a decode to host one. To 512 tokens the fused kernel
stays: beside products that short, a decode costs more than it hides. On
an A100 it was slower than each matrix decoded first, as measured
(above). On an H100,
whose cuBLAS kernels differ, and the L4, L40S and RTX 6000 Ada, which sum
in fp32 at twice the GeForce rate (a product's time decodes half as much
beside it), the path is off until measured:
`GLYD_AHEAD_MIN=513` takes it; `GLYD_AHEAD_WARPS`, `GLYD_AHEAD_RATE` and
`GLYD_AHEAD_FLOPS` tune it (logs:
benchmarks/gpu/rtx4080s-prompts-2026-09-27).

## Popular models

`sizes.py MODEL_DIR ...` packs every Linear layer's matrix of a model in
both layouts and unpacks it, compared bit for bit: every projection's
matrix, and a layer's experts kept as one tensor (Gemma 4, Llama 4) a
matrix an expert. Nineteen popular open models (H100 SXM, 2026-09-26,
`benchmarks/gpu/popular-h100-2026-09-26`; A10, 2026-09-27,
`benchmarks/gpu/open-models-a10-2026-09-27`):

| Model | Matrices | bf16 | `mma` | `mma12` |
| :--- | ---: | ---: | ---: | ---: |
| GLM-4.5-Air | 107.96 B | 215.92 GB | 144.58 GB (10.71 bits, −33.0%) | 162.43 GB (12.04, −24.8%) |
| Llama 4 Scout 17B-16E | 105.97 B | 211.93 GB | 142.18 GB (10.73, −32.9%) | 159.42 GB (12.04, −24.8%) |
| Qwen3-Next 80B-A3B | 80.64 B | 161.28 GB | 109.40 GB (10.85, −32.2%) | 123.34 GB (12.24, −23.5%) |
| Llama 3.3 70B Instruct | 68.45 B | 136.90 GB | 91.93 GB (10.74 bits, −32.9%) | 103.00 GB (12.04, −24.8%) |
| Qwen3 30B-A3B (MoE) | 29.90 B | 59.79 GB | 40.22 GB (10.76, −32.7%) | 44.98 GB (12.04, −24.8%) |
| Gemma 3 27B | 25.74 B | 51.48 GB | 34.58 GB (10.75, −32.8%) | 38.74 GB (12.04, −24.8%) |
| Muse Glimmer 30B | 25.66 B | 51.33 GB | 34.48 GB (10.75, −32.8%) | 38.61 GB (12.04, −24.8%) |
| Qwen3.8 27B | 24.76 B | 49.52 GB | 33.28 GB (10.75, −32.8%) | 37.25 GB (12.04, −24.8%) |
| Gemma 4 26B-A4B | 24.50 B | 49.00 GB | 32.92 GB (10.75, −32.8%) | 36.87 GB (12.04, −24.8%) |
| Mistral Small 3.2 24B | 22.63 B | 45.26 GB | 30.34 GB (10.72, −33.0%) | 34.05 GB (12.04, −24.8%) |
| Phi-4 | 13.63 B | 27.26 GB | 18.28 GB (10.73, −32.9%) | 20.51 GB (12.04, −24.8%) |
| DeepSeek-R1-Distill-Qwen 14B | 13.21 B | 26.42 GB | 17.99 GB (10.89, −31.9%) | 19.89 GB (12.05, −24.7%) |
| Gemma 3 12B | 10.90 B | 21.80 GB | 14.63 GB (10.74, −32.9%) | 16.41 GB (12.04, −24.8%) |
| Llama 3.1 8B Instruct | 6.98 B | 13.96 GB | 9.39 GB (10.76, −32.8%) | 10.50 GB (12.04, −24.8%) |
| Mistral 7B Instruct v0.3 | 6.98 B | 13.96 GB | 9.40 GB (10.77, −32.7%) | 10.50 GB (12.04, −24.7%) |
| Qwen3 8B | 6.95 B | 13.89 GB | 9.44 GB (10.87, −32.1%) | 10.46 GB (12.05, −24.7%) |
| Qwen3 4B 2507 | 3.63 B | 7.27 GB | 4.93 GB (10.85, −32.2%) | 5.47 GB (12.04, −24.8%) |
| Llama 3.2 3B Instruct | 2.82 B | 5.64 GB | 3.79 GB (10.75, −32.8%) | 4.24 GB (12.04, −24.8%) |
| SmolLM3 3B | 2.81 B | 5.62 GB | 3.77 GB (10.73, −32.9%) | 4.23 GB (12.04, −24.8%) |

Llama and Gemma from `unsloth/` (the same weights, ungated). bf16 against
Glyd (`mma`) end to end, the same run (`e2e.py --baseline --ppl --mmlu
300`), where it loads the model as a causal LM on one GPU:

| Model | Perplexity bf16 / Glyd | Next token as bf16's | MMLU (300) bf16 / Glyd | Answers as bf16's |
| :--- | ---: | ---: | ---: | ---: |
| Phi-4 | 14.7888 / 14.7855 | 98.91% | 76.67% / 76.33% | 99.33% |
| DeepSeek-R1-Distill-Qwen 14B | 27.1955 / 27.1975 | 98.52% | 78.00% / 78.00% | 100% |
| Llama 3.1 8B Instruct | 19.5915 / 19.5909 | 98.68% | 71.67% / 72.00% | 99.67% |
| Mistral 7B Instruct v0.3 | 12.1422 / 12.1473 | 99.05% | 60.67% / 60.67% | 100% |
| Qwen3 8B | 20.7490 / 20.7428 | 98.47% | 74.00% / 74.33% | 99.67% |
| SmolLM3 3B | 29.1467 / 29.1422 | 97.90% | 63.33% / 63.33% | 99.67% |
| Qwen3.8 27B (H100 PCIe) | 15.1946 / 15.1941 | 98.82% | 79.67% / 80.00% | 99.67% |
| Gemma 3 12B (H100 PCIe) | | 95.56% | 74.00% / 74.00% | 99.33% |
| Qwen3 4B 2507 (A10) | 22.4636 / 22.4672 | 98.53% | 71.00% / 71.00% | 99.33% |
| Llama 3.2 3B Instruct (A10) | 25.1298 / 25.1437 | 98.58% | 63.67% / 64.33% | 98.00% |

The last four with the 12-bit layout the GPU picks (`--format auto`) and
q, k, v and gate, up merged (`--merge`). Qwen3.8 27B with Glyd uses
41,071 MiB of GPU memory against bf16's 51,771 (nvidia-smi), under a
48 GB card's 49,140; GPU time a token 34.73 / 51.42 / 84.52 ms at 1 / 8 /
32 sequences against 40.21 / 50.60 / 85.19. Gemma 3's perplexity is left
out: the windows start without the BOS token Gemma needs.

## Larger models

On rented GPUs (`scripts/gpu_lambda.sh`: one Lambda Cloud instance a
run, terminated at the end; raw logs in `benchmarks/gpu/lambda-*`), bf16
and Glyd in the same run, 64 new tokens a sequence, MMLU on the same
1,000 questions (0-shot):

| Model, GPUs | Weights | Tokens/s at 1 / 8 / 32 / 64 sequences | Prompt of 2048 | Perplexity | MMLU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-32B, bf16, 2x A6000 48 GB | 65.52 GB | 9.5 / 74.4 / 275.9 / 488.4 | 1332 ms | 17.0796 | 78.5% |
| Qwen3-32B, Glyd, **1x** A6000 | **44.45 GB** | **11.8 / 94.9 / 289.9** / 400.5 | 2082 ms | 17.0788 | 78.0% |
| Qwen2.5-72B, bf16, 4x A6000 | 145.41 GB | 4.5 / 35.0 / 134.6 / 254.3 | 2689 ms | 10.6035 | 81.9% |
| Qwen2.5-72B, Glyd, **3x** A6000 | **97.80 GB** | **6.4 / 49.0 / 153.8** / 218.1 | 4195 ms | 10.5996 | 81.8% |
| Qwen3-32B, bf16, H100 SXM 80 GB | 65.52 GB | 13.4 / 123.2 / 502.0 / 988.4 | 262 ms | 17.0814 | 78.2% |
| Qwen3-32B, Glyd, H100 SXM | **44.45 GB** | 19.9 / 157.1 / 465.1 / 789.5 | 305 ms | 17.0849 | 78.2% |
| Qwen2.5-7B, bf16, H100 SXM | 15.23 GB | 25.9 / 207.1 / 823.0 / 1670.2 | 55 ms | 17.0178 | 73.3% |
| Qwen2.5-7B, Glyd, H100 SXM | **10.32 GB** | 57.4 / 455.5 / 1655.5 / 2807.1 | 64 ms | 17.0162 | 73.4% |

The MMLU answers are bf16's on 99.2% (Qwen3-32B, A6000), 99.4%
(Qwen2.5-72B), 100% (Qwen3-32B, H100) and 99.9% (Qwen2.5-7B, H100) of
the questions. Across GPUs the A6000s run Glyd's model on fewer of them,
so their pipeline has fewer stages. On the H100 Hugging Face's generation
loop is bound by the CPU at few sequences (Qwen3-32B, profiled over 16
tokens: 54.8 ms a token for bf16, 55.8 for Glyd), and the GPU's work is
not Glyd's gain there: 40.6 ms of GPU time a token against bf16's 28.2,
Qwen3-32B's MLP matrices 128 us against cuBLAS's 90 at one token (the
decode is bound by arithmetic when memory moves 3.35 TB/s).

## The KV cache

The keys and values a model keeps for the tokens it has seen are bf16
like its weights, and as compressible: on Qwen2.5-7B over 2,048 tokens of
enwik8 the exponent carries 2.80 bits in the keys (2.59 given the
channel) and 2.67 in the values, a floor of 10.5-10.6 bits a value, 34%
under bf16. `kv.py` holds a Hugging Face model's cache that way:
`GlydKVCache(config)` keeps each layer's newest tokens as they are and
packs every full page of 64 in the mma layout's tiered code, keys by
token and values transposed (the operands of attention's two products),
the layer's tiers taken from its first page. With `fused=True` and
`use_fused_attention(model)`, a step of one new token a sequence runs
`attn_decode` on the packed pages: a block per KV head and run of pages,
a page a warp, keys and values decoded in registers, the head's queries
as one tensor-core tile, an online softmax, the blocks merged in a fixed
order. Other steps (a prompt) get the keys and values decoded exactly.

Qwen2.5-7B-Instruct, weights in the mma layout, RTX 4080 SUPER, an
enwik8 prompt then 128 new tokens (`e2e.py --kv 1024,4096,16384`):

| Prompt | KV cache, plain | Packed | A step, plain | Fused | Peak memory, plain | Packed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 tokens | 66 MB | **46 MB** (70.4%) | 18.3 ms | 19.2 ms | 10.80 GB | 10.78 GB |
| 4,096 tokens | 242 MB | **167 MB** (69.1%) | 19.1 ms | 19.6 ms | 11.41 GB | 11.34 GB |
| 16,384 tokens | 947 MB | **651 MB** (68.7%) | 21.6 ms | 21.7 ms | 13.87 GB | 13.58 GB |

Decoded back, the cache is the plain cache's bit for bit: the same 128
tokens. Through `attn_decode` the sums run in another order than
FlashAttention's, as with the weights: 256 tokens of the text fed one at
a time after the prompt, perplexity 2.9895 / 4.3106 / 2.4778 against the
plain cache's 2.9950 / 4.3130 / 2.4809, the next token the plain cache's
97.3% / 99.6% / 99.6% of the time.

## Running

Needs PyTorch with CUDA, and nvcc (the extension builds on first import)
or the prebuilt library: `bash build_lib.sh` builds
libglyd_gpu_cuda13.so (the kernels behind a C API, the CUDA runtime
linked in; code for sm_80, sm_86, sm_89, sm_90a, and sm_100 and sm_120
where nvcc has them (CUDA 12.8 on), PTX for the GPUs after them; the
Hopper kernel, wgmma's, in the sm_90a code alone) next to glyd_gpu.py,
which then uses it through ctypes instead of building; GLYD_GPU_LIB
names another. The Python side is the glyd package's
(bindings/python/glyd/gpu: kernels.py, _lib.py over the library, and
model.py, the modules e2e.py runs); glyd_gpu.py is it for the scripts
here, taken from this checkout. The API a user types,
glyd.from_pretrained and the rest, is in bindings/python/README.md.

    python check_capi.py [LIBRARY]            # every entry point through the library and through the JIT build, bit for bit
    python check_api.py [MODEL ...]           # glyd.from_pretrained, compress, save_pretrained, verify: against bf16, bit for bit where exact
    python check.py model.safetensors         # every tensor packed, unpacked, compared; speeds
    python shapes.py MODEL_DIR                # fused product against bf16, one layer's matrices
    python gemm.py MODEL_DIR 1,16,64          # several tokens: every product against bf16, one layer's matrices
    python e2e.py MODEL_DIR --format mma --fused --baseline [--batch 1,8,32] [--compile] [--prefill 16,64] [--ppl TEXT] [--mmlu 1000] [--kv 1024,4096]
    python kv.py                              # the KV cache packed and decoded bit for bit; attn_decode against SDPA
    python sizes.py MODEL_DIR ...             # every Linear's matrix in both layouts, bit for bit: bits a weight, GB

## The library

`build_lib.sh` builds glyd_gpu.cu's kernels alone, behind their C API:
libglyd_gpu_cuda12.so or libglyd_gpu_cuda13.so by nvcc's CUDA major version,
with no PyTorch in it. It carries its own CUDA runtime, linked in statically,
so it needs only the driver. A program that calls it links its own runtime
beside that one, so take the library whose CUDA major version is the
program's toolkit and runtime: libglyd_gpu_cuda12.so for CUDA 12,
libglyd_gpu_cuda13.so for CUDA 13.
[glyd_gpu.h](glyd_gpu.h) declares every function and says what it takes: the
arrays of a packed matrix (the tiered and 12-bit layouts, the fast and dense
formats), a product's workspace query before its call, the stream, the return
codes. glyd_gpu.cu includes it, so nvcc holds each definition to its
declaration, in the library's build and the JIT's alike. The glyd package
calls the library through ctypes (`_lib.py`, whose argument lists
`bindings/python/test_gpu.py` checks against the header); an engine in C,
C++, Rust or any language with a C FFI calls the same functions (Rust through
bindgen over the header, or its declarations written out).

Every release carries it on its own for Linux x86_64 and aarch64 (glibc 2.28
or later), CUDA 12 (built with 12.8) and 13:
`glyd-gpu-TAG-linux-ARCH-cudaN.tar.gz`, holding the library, glyd_gpu.h,
examples/unpack.c, this directory's LICENSE and a README
([README-lib.md](README-lib.md), with the example's build line for that
download's library), each with its `.sha256`.

[examples/unpack.c](examples/unpack.c), C and the C API alone: a matrix of a
model saved by `glyd.save_pretrained` (glyd-v1: the packs' buffers in
safetensors, their shapes and tiers in glyd.json) decoded on the GPU by
`glyd_gpu_mma_unpack` and checked against the bf16 checkpoint it was packed
from, bit for bit; a merged pack tensor by tensor (the runtime's lib64, or
lib in a toolkit from pip, as setup_env.sh's):

    bash build_lib.sh .
    gcc -O2 -I . -I $CUDA_HOME/include examples/unpack.c -o unpack \
        -L . -lglyd_gpu_cuda13 -L $CUDA_HOME/lib64 -lcudart -Wl,-rpath,$PWD:$CUDA_HOME/lib64:$CUDA_HOME/lib
    python -m glyd.gpu pack Qwen/Qwen3-0.6B qwen3-0.6b-glyd

On an RTX 4080 SUPER (CUDA 13.0), the first pack, then a merged one (every
one of Qwen3-0.6B's 112 packs, its 196 Linears, decodes to the checkpoint's
bits so; a bit flipped in the checkpoint is found):

    $ ./unpack qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/
    libglyd_gpu: C API 2, CUDA runtime 13000
    model.layers.0.self_attn.o_proj: [1024, 2048], 10.86 bits a weight packed, decoded on the GPU
      model.layers.0.self_attn.o_proj.weight [1024, 2048]: the checkpoint's, bit for bit
    $ ./unpack qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ model.layers.0.self_attn.q_proj
    libglyd_gpu: C API 2, CUDA runtime 13000
    model.layers.0.self_attn.q_proj: [4096, 1024], 10.79 bits a weight packed, decoded on the GPU
      model.layers.0.self_attn.q_proj.weight [2048, 1024]: the checkpoint's, bit for bit
      model.layers.0.self_attn.k_proj.weight [1024, 1024]: the checkpoint's, bit for bit
      model.layers.0.self_attn.v_proj.weight [1024, 1024]: the checkpoint's, bit for bit

## License

The files under gpu/ are under the [Business Source License 1.1](LICENSE):
source available, free for personal, educational, research and other
non-commercial use; any commercial production use needs a license
(suryakoritala1324@gmail.com); each version converts to Apache-2.0 four
years after its release. The Glyd codec is BSD-3-Clause OR GPL-2.0.
