# v0.28 tiered-fast: the smallest layout fast on data-center GPUs. Phase 1 design

Branch `tiered-fast` (worktree /Users/suryakoritala/glyd-tiered-fast, from private/main 4f87e465 = v0.27.0). Phase 1 changed
nothing in the library. It produced a model, a format prototype (T2) and a kernel prototype, in `gpu/experimental/tiered_fast/`
(tools and result texts, section 10). No cloud GPU: all measurements are on the dev L4 (sm_89), under the box lock; the
A100/H100 numbers in this file are projections from them and say so.

Names: `mma` is the smallest (tiered) layout, T1 its v0.27.0 packed format, T2 the format proposed here; `mma12` the 12-bit
layout. A "step" is 1024 weights (64 rows x 16 columns; one warp; 32 lanes x 32 weights in mma.m16n8k16's B-fragment order).
"clk/step" is SM clocks one SM spends on one step (measured as ns a step an SM x the SM clock sampled around the timing). All L4
numbers are Qwen3-8B tensors, o_proj 4096x4096 of layer 10 L2-resident unless said ("hot"), 58 SMs, interleaved rounds.

## 1. Result in short

1. **The gap is compute, not bandwidth.** The step kernel for the smallest layout spends about 295 SM-clocks a step on the A100,
   the H100 and the L4 (hot) alike; the 12-bit layout about 200 (A100) and ~165 (H100). The DRAM floors at 80% of peak are
   169 (A100) and 119 (H100) clk/step: the kernel runs at 1.7-2.4x its floor, which is "about half the bandwidth".
2. **Why (measured, ncu):** 604 warp instructions a step against the 12-bit layout's 184, on a 16-lane integer pipe (2 issue
   clocks an instruction). The ALU pipe is 56% busy, the issue slots 55%, the shared-memory wavefront pipe 53% (half of those
   wavefronts are bank conflicts); a SMSP has 3.7 warps and 1.0 eligible. Three resources at 55-60% and long dependent chains:
   a stall-bound kernel, so the speed comes from fewer instructions *and* better latency hiding, not either alone.
3. **What phase 1 found works (L4, M = 1 token, clk/step; T1 lib = 298.2, 12-bit lib = 131.8):**

   | step | clk/step | change |
   | :-- | --: | --: |
   | T2 format, the library's kernel (8-warp blocks) | 262.4 | -12% |
   | + lean decode, 4-warp blocks | 224.5 | -25% |
   | + swapped MMA operands (up to 8 tokens), a flag in the format marks the rare steps the hot decode cannot take and the general decode runs out of line, two steps decoded together (**the shipped form: exact for every pack**) | **205.5** | **-31%** |
   | the same with one step at a time | 215.4 | -28% |
   | one step at a time with 2-warp blocks (only for matrices of 250+ steps an SM) | 203.3 | -32% |

   The cold path costs 1.2 hot steps for an ordinary step forced through it (260 vs 215 clk) and about 2.2 for a heavy one (g1: 470 clk a step,
   T1 485); a matrix with more than 25% cold steps goes to the 12-bit layout (section 6.2).
4. **Plan (section 7):** T2b (same fixed part, a dense tier 3 and a literal list; +0.013 to +0.034 bits a weight, and a per-matrix
   choice of the 12-bit layout where it is smaller makes the whole-model size within -0.015..+0.020 of today's) and a rebuilt
   step kernel for 1-16 tokens, then a producer/consumer kernel for 17-64, with a gate after each step.
5. **Expected (A100/H100 are projections: the L4's clk/step carries over because the kernel is compute-bound; G0 checks it):**

   | tokens | T1 today | plan | 12-bit | plan vs 12-bit, A100 / H100 |
   | --: | --: | --: | --: | :-- |
   | 1 | 298 | ~205 | A100 ~200, H100 ~165 | +3% / +24% |
   | 8 | 307 | ~225 | same | +13% / +36% |
   | 16 | 321 | ~262 | same | +31% / +59% |
   | 17-64 | 452-672 | ~360-535 in the same structure (measured, L4-hot); ~260-330 with option 11 (estimate) | A100 ~200-240, H100 ~130-160 (12-bit mid/wg routes) | +10..65% / +60..155% with option 11; +50..170% / +120..310% without |

   Qwen3-8B, one sequence, engine: A100 15.4 -> about 11.0 ms (12-bit 10.8, bf16 13.3); H100 9.9 -> about 7.6 ms (12-bit 6.15). So the
   plan puts the A100 level with the 12-bit layout at one token and the H100 about 24% behind; at 8-64 tokens the smallest layout stays
   behind the 12-bit layout's kernels on both. "At least as fast at 1-64 tokens" is not reached; section 7.5 says what would.
6. **Weeks:** 6 to the first deliverable (T2b in the library, the 1-16 token kernel, measured on an A100 and an H100); +3 for
   17-64 tokens; +1 for prompt routes; the Hopper prompt kernel +3-5, optional. The first real-GPU evidence is one week after the
   go: the harness job of section 9.2 (30 min on each of an A100 and an H100).
7. **Fallbacks (section 7.4):** if the real GPUs disagree with the L4 the plan stops after one week and one job; if the 1-16 token
   kernel misses its gate, T2 + 4-warp blocks (-25%) ship and `auto` keeps the 12-bit layout on A100/H100; a fixed 3-bit
   exponent code (11.25 bits, -29.7%) is the only format that reaches H100 parity on paper and breaks the 0.1-bit rule and the
   headline: the owner's call.

## 2. Where it stands (v0.27.0, measured)

### 2.1 The layouts in the engine and the roofline

Rust engine, Qwen3-8B, one sequence, CUDA graphs, 4096-token cache (`$S/ceiling/results-eng{a,h}100`):

| GPU | mma (10.37 GiB) | mma12 (11.48 GiB) | ratio | GEMM clk/step* mma / mma12 |
| :-- | --: | --: | --: | --: |
| A100 SXM4 40 GB | 15.40 ms (65.0 tok/s) | 10.78 ms (92.8 tok/s) | 1.43 | ~295 / ~200 |
| H100 SXM5 80 GB | 9.92 ms (100.8 tok/s) | 6.15 ms (162.6 tok/s) | 1.61 | ~280 / ~165 |

*step time minus 1.1 ms for the non-GEMM kernels (an estimate), times SMs x 1024 / 7.57 B weights, times the SM clock (A100 1.41,
H100 1.755 GHz). The A100 and H100 jobs measure these directly. (vLLM bf16 on the A100: 13.3 ms.)

| GPU | B/clk/SM at 80% | DRAM floor clk/step mma, mma12 | measured mma, mma12 | mma pack read rate |
| :-- | --: | --: | --: | --: |
| A100 40 GB (1555 GB/s, 108 SMs) | 8.2 | 169, 188 | ~295, ~200 | 0.73 TB/s = 47% of peak |
| H100 SXM (3350 GB/s, 132 SMs) | 11.6 | 119, 133 | ~280, ~165 | 1.1 TB/s = 33% of peak |
| A100 80 GB (2039 GB/s) | 10.7 | 129, 144 | not measured | as the H100 |

The A100 40 GB is the lenient case: its 12-bit kernel sits at its DRAM floor, so the smallest layout has to reach about 200.
An A100 80 GB or a GH200 behaves like the H100.

### 2.2 Instruction budget a step (ncu, L4, hot, M = 1; warp instructions a step)

| | T1 lib | T2 lib | flagged single step, swapped, 2-warp blocks | 12-bit lib |
| :-- | --: | --: | --: | --: |
| all | 603.8 | 512.7 | 421.9 | 184.2 |
| ALU pipe (SHF, LOP3, ISETP, PRMT, IADD3, SEL, LEA) | 307.1 | 264.6 | 221.0 | 97.8 |
| FMA-heavy pipe (IMAD) | 138.4 | 122.1 | 102.3 | 44.9 |
| LSU (LDS, STS, LDG, SHFL) | 75.2 | 66.1 | 58.2 | 15.5 |
| branch | 53.5 | 34.4 | 21.0 | 12.5 |
| tensor (HMMA) | 8 | 8 | 4 | 8 |
| registers a thread | 126 | 127 | 124 | 128 |
| warps a SMSP / eligible | 3.72 / 1.02 | 3.68 / 1.02 | 3.55 / 1.06 | 3.85 / 0.68 |
| ALU pipe / issue / LSU wavefronts busy | 56 / 56 / 50% | 56 / 55 / 53% | 59 / 58 / 59% | 43 / 42 / - |

The same kernels compile to 125-128 registers on sm_80 and sm_90a: the shape carries over. The decode's mass in T2 (513): tier-1 consumption of 8
groups of 4 weights and the 16 bf16 pairs about 105; the tier-2/3 stage (lane a word, table decode into scratch) about 88; the positions (popc,
redux, 5-level shuffle scan, window reads) about 75; step loads, x loads, loop, MMA about 160 (the 12-bit layout's whole decode is 58). The
flagged kernel drops loop, load and x overhead (swapped operands: 4 HMMA and 2 x loads; 2-warp blocks) and the checks (the flag).

### 2.3 What limits it (ncu, T1 and T2 library kernels, 8-warp blocks, 127 registers, two blocks an SM)

- Per-warp CPI 6.7-6.8 (issue 1.0): wait 1.39 (fixed-latency dependences), short scoreboard 1.05 (LDS, SHFL, POPC), not selected 0.86,
  math-pipe throttle 0.73, barrier 0.61, long scoreboard 0.26, MIO throttle 0.24, branch resolving 0.19.
- ALU pipe 55.8% of peak, FMA-heavy 23%, issue 54.7%, LSU wavefronts 53%: 120 shared wavefronts a step against 39 ideal, 54 of the
  excess from the 16 table lookups `LDS [R + table]` (3.2 wavefronts each), the rest from the escape-byte windows.
  3.68 warps and 1.02 eligible a SMSP.
- clk/step = I x CPI / (4 W x 0.91) (I instructions, W warps a SMSP, 0.91 = the share of the launch an SM is busy): T2 lib
  513 x 6.8 / (4 x 3.68 x 0.91) = 260 (measured 262). The floor is the ALU pipe: 265 x 2 / 4 = 132 clk/step (T1: 153).
- What parity needs: the 12-bit layout's clk/step (A100 ~200, H100 ~165) with the ALU pipe at 60-65% needs about 225-230 ALU
  instructions a step; T2 has 265, the flagged kernel 221 (its pair 210).

## 3. Phase 1 measurements (L4; every variant also checked against a reference product and, at 1 token, bit for bit against T1's)

The harness (`harness.cu`, `v2.cuh`) includes glyd_gpu.cu, adds the T2 decode policy `Tiered2` and a copy of `mma_gemm_kernel`
(`step_kernel<Fmt,MT,UNR,NW,MB,SW>`) whose options are bits of a template parameter; packs come from `pack_v2.py`. Results:
`res/all_j3.txt`, `res/j4.txt` to `res/j13.txt`, `res/models/`.

### 3.1 T2 and lean decode, one step a warp, 8-warp blocks (clk/step)

| tokens | T1 lib | T2 lib | T2 best option | 12-bit lib |
| --: | --: | --: | --: | --: |
| 1 | 298.5 | 262.4 (-12.1%) | 250.1 (-16%: options 8, 40, 104) | 132.7 |
| 16 | 320.8 | 287.8 (-10.3%) | 276.8 | 156.4 |
| 64 | 665 | 572 (-14%) | 547 | 390 |

Options (`v2.cuh`): predicated-shuffle scan -0.8%; literals by one shuffle-window pass -2.9%; both -3.5%; hoisted tier-1 lookups
-2%; radix-4 scan -1% (worse alone); tier-1 consumption through a register window +2%; sixteen copies of the group table +2%
(the address arithmetic costs more than the conflicts it removes: 3.6); two steps a pass in two register sets -0.3%.

### 3.2 Block size and occupancy (M = 1, 4096x4096; the first three rows: checks and slow paths inline, the others: the shipped form)

| structure | clk/step |
| :-- | --: |
| 8-warp blocks, 2 an SM (16 warps) | 255.7 |
| 4-warp blocks, 4 an SM | 224.5 (-12%) |
| one 8-warp block an SM (a timing experiment: padded shared memory) | 332 |
| flagged hot/cold, swapped, 4-warp blocks, 4 an SM, one step a pass | 215.0 |
| the same, two steps a pass (a pair) | **205.4** |
| flagged hot/cold, swapped, 3-warp blocks, 5 an SM | 228.6 |
| flagged hot/cold, swapped, 2-warp blocks, 8 an SM | 203.3 (only for large matrices: 3.2b) |
| flagged hot/cold, swapped, 1-warp blocks, 16 an SM | 259.3 |

Smaller blocks win because the cross-warp barriers and their imbalance go (the barrier stall falls from 0.61 to 0.16 of a 6.3-6.8 CPI;
the 12-bit kernel's is 1.29 of 9.4); one warp a block loses it again to per-block costs (the table fill, partial sums and the last-arrival
protocol a block). More than 4 warps a SMSP does not pay (blocks of 4 at 5 and 6 an SM: equal or worse, spills at 80 registers).

### 3.2b Matrix size: a call has a fixed part (L4, M = 1, hot; corners of one o_proj; clk/step)

| steps an SM (L4: 58 SMs) | 70 (1024x4096) | 141 (2048x4096) | 282 (4096x4096) |
| :-- | --: | --: | --: |
| T1 lib | 517 | 367 | 298 |
| 12-bit lib | 320 | 194 | 133 |
| T2 lib | 472 | 332 | 263 |
| flagged single, swapped, 4-warp blocks | 378 | 258 | 215 |
| flagged pair, swapped, 4-warp blocks | 369 | 255 | 206 |
| flagged single, swapped, 2-warp blocks | 477 | 289 | 204 |
| 4-warp pair / T1 | 0.71 | 0.69 | 0.69 |
| 2-warp single / T1 | 0.92 | 0.79 | 0.68 |
| 12-bit / T1 | 0.62 | 0.53 | 0.45 |

(The L4 ran at 2.0, 1.74 and 1.57 GHz in the three runs: compare within a column.) Launch gap, ramp and tail are about the same size for every
layout, so small matrices cost more a step for all of them and the gap to the 12-bit layout shrinks (plan / 12-bit: 1.15, 1.31, 1.55). Two-warp
blocks pay only with long work a block (below about 200 steps an SM they lose to 4-warp blocks by up to 25%); 4-warp blocks keep -30% at every size.
Steps an SM of the Qwen3-8B matrices, one product: A100 (108 SMs) o_proj 152, qkv merged 228, down 455, gate+up merged 910; H100 (132 SMs)
124, 186, 372, 745: most of a layer's steps are in matrices of 370+ steps an SM, the o_proj and qkv are in the 4-warp regime.

### 3.3 Swapped operands (up to 8 tokens)

W's registers as the A operand (a0..a3 = R[4i], R[4i+2], R[4i+1], R[4i+3]), X's two registers a lane as B: 4 MMAs a step, 16
accumulators instead of 32, 2 x loads instead of 4, no change to the decode (the registers are the same, reordered).
8-warp blocks 255.4 -> 228.4 (-10.6%); 4-warp blocks 230.7 -> 222.6 (-3.5%). Registers of the single-step kernel 128 -> 104-115.

### 3.4 Pair decode, runtime checks, and the flag (M = 1)

| variant | regs | clk/step |
| :-- | --: | --: |
| single step, checks and slow paths inline, 4-warp blocks | 128 | 224.5 |
| pair, checks and slow paths inline (3 blocks of 4 warps) | 168 | 259.6 |
| pair, forced to 128 registers (76-92 bytes of spill) | 128 | 311 |
| pair, fast-only (the slow paths removed, nothing checked), 4-warp blocks | 127 | 206.4 |
| single, fast-only, swapped, 2-warp blocks | 122 | 201.0 |
| runtime checks, the general decode out of line: single, swapped, 2-warp / pair, 4-warp | 124 / 128 | 209.2 / 225.9 |
| **flag in `block_base`, the general decode out of line: single, swapped, 2-warp** | 124 | **203.3** |
| the same, a pair, 4-warp blocks / 2-warp blocks | 128 / 128 | 205.4 / 202.6 |

The pair decodes two steps' phases interleaved (instructions -19% alone: ALU 265 -> 224, FMA-heavy 122 -> 79) and only pays when it
fits 128 registers without spill: the slow paths (T1 > 512 escapes, windows, more than 32 literals) are what pushed it to 168. Checking
at run time costs 4-9% (209 / 226 vs 201 / 206); the flag, computed by the packer, costs 1%. With 2-warp blocks the single step
catches up with the pair; with 4-warp blocks at 8 tokens the pair is 3.5% better (224.7 vs 234.2). ncu of the two flagged 2-warp
kernels: single 422 instructions a step (ALU 221, FMA-heavy 102, LSU 58, branch 21), 124 registers, 3.55 warps a SMSP, ALU pipe 59%, issue
57.6%, LSU wavefronts 58.6%; pair 384 instructions (ALU 210), long-scoreboard stall 0.85, same time.

Forced cold steps (every N-th step of the 4096x4096 tensor flagged, so the general decode takes it; products unchanged, bit for bit): clk/step
at 0 / 1 / 5 / 10 / 20 / 50 / 100% cold: 215.5 / 216.5 / 218.5 / 220.4 / 224.5 / 237.0 / 259.6 (4-warp single); pair 205.7 / 206.7 / 209.7 / 213.4 / 218.0 /
232.9 / 255.1. An ordinary step through the general decode costs the T2 library's 262 clk: 1.2 hot steps. Real cold steps are the heavy ones (many
escapes): `m2` (3.6% of its steps cold, 4096x2560, 176 steps an SM) 329 against T1's 369 and T2 lib's 358; `m1` (18.4%, 2560x9728) 331 against 327 and 312.

The general decode out of line is bit-exact on a heavy tensor where every step takes it (g1: every flagged and checked variant equals T1's
product bit for bit; 470-485 clk/step, T1 485, T2 lib 475; 2-warp variants take 640-660 there: irrelevant, heavy tensors go to the 12-bit layout).

### 3.5 x loads, M > 8

X's fragment loads (4 `LDG.32` a step at 9-16 tokens, 16 at 64) hit 8 different cache lines each: a time experiment with all rows
aliased (coalesced loads, wrong products) gives -6.6% at 16 tokens and -9% at 64 (`xfake`). A pre-permuted X in fragment order
(one `LDG.128` a step) or a shared-memory tile with ldmatrix would take it; not in the first deliverable.

### 3.6 Shared-memory bank conflicts

ncu per-instruction counters (swapped single-step, 96 registers): 35 LDS a step, 86 wavefronts (32 ideal): 54 excess, 44 of them from
the table lookups. Tables replicated 16 times (two lanes a bank) cut conflicts by 28% and wavefronts by 12%, but the address arithmetic added 41
instructions a step and the 16 KB table costs occupancy: slower (249 vs 224 with a costly fill; 225.9 vs 215.5 with the fill made cheap, the flagged
4-warp kernel and the index written `(f << 5) | lane`, 5% slower; 32 copies slower still). Replication cannot be tested at the A100's and H100's
shared-memory sizes (164-228 KB) on the L4 (100 KB); the A100/H100 job carries the variants.

### 3.6b 8 and 16 tokens (the shipped form)

| tokens | T1 lib | T2 lib | flagged single (best structure) | flagged pair 4-warp | 12-bit lib |
| --: | --: | --: | --: | --: | --: |
| 8 (swapped) | 306.8 | 272.3 | 232.8 (2-warp, no hoisting) / 234.2 (4-warp) | 224.7 | 142.1 |
| 16 (standard) | 320.7 | 287.7 | 261.5 (4-warp) | 302 | 156.3 |

(2-warp blocks are bad at 16 tokens: 338, 32 accumulators and 128 registers; the plan uses 4-warp blocks from 9 tokens.)

### 3.7 Higher token counts (L4 hot; the library's shipped kernels, and the flagged single-step kernel with `MT` = 2-4)

| tokens | T1 lib | T2 lib | flagged single (best of 4- and 8-warp blocks) | 12-bit step kernel | 12-bit shipped route (mid) |
| --: | --: | --: | --: | --: | --: |
| 17 | 452 | 390 | 358 | 208 | 295 |
| 24 | 462 | 401 | 380 | 217 | 294 |
| 32 | 473 | 414 | 388 | 232 | 296 |
| 48 | 642 | 541 | 512 | 361 | 369 |
| 64 | 672 | 574 | 535 | 393 | 368 |

(`MT` = 2-4 means 64-128 accumulator registers: 156 registers, 2-3 blocks of 4 warps or one of 8; the 2-warp and 3-block variants at 48-64 tokens
spill: 1200-2800.) At 17-64 tokens the new decode in the same structure gains only 6-8% over the T2 library kernel: the accumulators and X fragments
decide there, not the decode; the kernel is 21% (17 tokens) to 45% (64) behind the 12-bit layout's shipped cp.async/mma.sync "mid" route on the L4.
Option 11 (producers decode, consumers multiply) is the structure that separates the two.

### 3.8 DRAM-bound regime (L4 cold: 10 copies of a 4096x4096 corner / 3 copies of up_proj 12288x4096; the L4's 300 GB/s)

T1 489-494, T2 lib 485-490, T2 kernels 466-487, 12-bit 503-511 (u10: 419-427 against 449-452): the smallest layout is 4-7% faster
than the 12-bit layout where memory is the limit (the L4, A10, RTX 40 cards), and every T2 variant ties: there the compute is
hidden. That is why the L4 shows nothing and the A100/H100 do.

### 3.9 Bits a weight (CPU, every Linear of each model's decoder layers; block_base's 4 B a step included)

| model | T1 | T2 (3-byte entries) | T2b (18-bit entries) | best(T1, 12-bit) | best(T2b, 12-bit) |
| :-- | --: | --: | --: | --: | --: |
| Qwen3-8B | 10.868 | 10.945 | 10.902 (+0.034) | 10.839 | 10.853 (-0.015) |
| Llama-3.1-8B | 10.758 | 10.798 | 10.774 (+0.016) | 10.757 | 10.772 (+0.014) |
| Qwen2.5-7B | 10.796 | 10.844 | 10.816 (+0.021) | 10.796 | 10.816 (+0.020) |
| Mistral-7B-v0.3 | 10.772 | 10.820 | 10.792 (+0.020) | 10.767 | 10.780 (+0.008) |
| Qwen3-4B | 10.844 | 10.892 | 10.862 (+0.018) | 10.843 | 10.859 (+0.015) |
| Yi-1.5-6B | 10.717 | 10.750 | 10.730 (+0.013) | 10.717 | 10.730 (+0.013) |
| Qwen3-1.7B | 10.766 | 10.816 | 10.790 (+0.023) | | |

(best(x, 12-bit): per tensor the smaller of x and the 12-bit layout's measured size, exceptions included.) The heavy-tailed tensors
(Qwen3 layers 1-3 MLPs, Mistral layers 0-1 q/k: 12-14 bits and more in T1, up to 15.8 in T2) are smaller in the 12-bit layout
(12.05-12.5): 6.5% of Qwen3-8B's weights. T2b's decode costs the same as T2's (the entry is a bit field read from the same shuffle
window; only an index multiply changes).

How often a step leaves the hot path (tier-1 escapes > 512, tier-3 digits or the literal list outside the 128-byte window, more than 32
literals; `v2limits.py`), in the tensors that are not heavy: Qwen3-8B 0.006% of the steps, Llama-3.1-8B 0.03%, Mistral-7B 0.04%, Qwen3-4B 1.2%,
Qwen2.5-7B 1.3% (nearly all with more than 512 tier-1 escapes); in the heavy tensors 67-99%. So every step needs a way out, and the way out
must be exact; a per-matrix rejection would reject most matrices of Qwen3-4B and Qwen2.5-7B. At 1% cold steps a matrix runs about 0.5% slower
(3.4: a forced cold step costs 1.2 hot steps).

### 3.10 Prompts (the fused prompt kernel, producers decode and consumers mma.sync; L4 hot; clk/step)

| tokens | T1 big | T2 big | 12-bit big |
| --: | --: | --: | --: |
| 128 | 840.5 | 746.8 (-11%) | 530.9 |
| 256 | 1656.5 | 1461.5 (-12%) | 920.6 |
| 1024 | 5805.9 | 4978.2 (-14%) | 2908.4 |

T2 helps the fused prompt kernel by 11-14% and leaves it 1.4-1.7x the 12-bit kernel on the L4; prompts go to decode+cuBLAS from the 12-bit
layout's thresholds (A100 769 tokens, L4 896 tiered), where the unpack, memory-bound in T2 (2 B written, 1.35 B read a weight against the 12-bit
layout's 3.5 B), takes over.

## 4. Earlier work, what it refuted, and what is different

| Idea (where) | Verdict then | This design |
| :-- | :-- | :-- |
| A fixed 3-bit exponent code, "mma11" (explore-report-1 area 4; format-report-1 section 1) | 11.25-11.38 bits (-29%); escapes cost 3/4 of today's decode | Not taken: breaks the 0.1-bit rule and the headline; the fallback if H100 parity is required |
| A cheaper 12-bit layout (split byte: 8.5 vs 22 instructions a k-block) | Adopted (today's mma12): "no packed format is both near 33% and a third as costly to decode" | Same conclusion for the decode: tiered stays ~2.5x. The target is parity by throughput (blocks, ILP, hot/cold), not by decode cost |
| Prompts: fused kernels L2-bound at cuBLAS's bytes a MAC; decode ahead beside cuBLAS; no-producer 8-warp kernels (kernel-report-a100) | 1.09-2.07x cuBLAS; ahead slower on A100 | Not repeated. Prompts: decode+cuBLAS from the 12-bit layout's thresholds, T2 unpack memory-bound |
| Per-lane escapes at a stored offset (no warp scans, ~11.0 bits; explore-report-1) | marginal at 4-5 ops a weight | Not taken (+0.2 bits); the scan is 5% of the instructions |
| Register capping with spills (this work, 3.4) | pair at (256,2): 68 B spill, slow | New: the spill came from the slow paths, not the pair; fast path and cold path apart fit 128 registers |
| Bigger/replicated tables, register windows (3.1, 3.6) | +2% | Refuted again in the simple forms |
| Hopper TMA/wgmma for prompts (hopper2-report-1) | 12-bit: 3-4.5% layer gain at 128-512 tokens | Tiered needs the same kernel with T2 producers (option 12) |
| SPLIT (decode on reserved SMs) | 12-bit only; 3-12% at 2048+ | Not for 1-64 tokens; prompts later |

Different this time: the earlier reports treated the tiered decode as a fixed cost per k-block; here the instruction count
falls by a third (T2, swapped operands, pairs) and the model (2.3) says what the cut is worth at each occupancy: the first
attempt (pair kernel at 2 warps a SMSP, 178 registers) made the kernel slower while cutting instructions 19%.

## 5. Options

Gain = change in clk/step at 1-16 tokens unless said; M = measured on the L4 in phase 1, E = estimate. Cost in engineer-weeks
including L4 tests (one engineer; one A100 and one H100 job a milestone).

| # | Option | Gain | Cost | Risk | Status |
| :-: | :-- | :-- | :-: | :-- | :-- |
| 1 | T2b format: dense tier 3, literals as 18-bit entries, fixed part and `data` byte for byte T1's | -12% (M) | 2.0 (format, both packers, loader, unpack, C API, tests) | low; +0.013..0.034 bits | prototype |
| 2 | Lean decode options 8, 32, 1 | -3..-4% (M) | in 1 | low | measured |
| 3 | 4-warp blocks (barrier and imbalance gone); 2-warp blocks for matrices of 250+ steps an SM | -12% (M); then -5% (M, large matrices only; worse below 200 steps an SM) | 0.5 | low; determinism by item order | measured |
| 4 | Swapped MMA operands, up to 8 tokens | -4..-10% (M) | 0.5 | low | measured |
| 5 | Hot decode without runtime checks; the general decode out of line; a flag in `block_base` bit 31 marks the steps the hot path cannot take | 4-9% over run-time checks (M); the flag costs 1% (203.3 vs 201.0) | 1.0 (packer flag, loader, kernel) | low-medium: the flag must be exact | measured (harness) |
| 6 | Pair decode (two steps interleaved) in 128 registers | -4.6% at 1 token (205.5 vs 215.4), -4% at 8 (M); equal at 2-warp blocks; worse at 16 tokens | 1.0 | medium: ptxas variance (the single step is the fallback) | in the plan, with the single-step fallback |
| 7 | Conflict-free group table (replicated, index `(f << 5) | lane`) on A100/H100 | -3..-6% (E); on the L4 16 copies lost 5% (3.6) | 1.0 | medium | A100/H100 job |
| 8 | X pre-permuted to fragment order (or shared tile + ldmatrix) | -7% at 16 tokens, -9% at 64 (M, `xfake`) | 1.0-1.5 | medium: an extra launch or a producer change | experiment |
| 9 | ALU/FMA balance by source form (shifts as IMAD) | -3..-5% (E) | 1.0 | low | estimate |
| 10 | cp.async ring (A100/Ada) / TMA bulk (Hopper) of a step's bytes into shared memory: St (16 registers) leaves the registers, prefetch depth 2-3 | -5..-8% (E) | 2.0 | medium (L4 smem 100 KB) | estimate |
| 11 | Producer/consumer kernel at 17-64 tokens: producers decode (<= 128 registers, no accumulators), consumers mma.sync with x via ldmatrix, handoff through shared memory (A100: `mma12_ws`/mid skeleton; Hopper TMA producers) | 64 tokens: 572 -> ~330 (E) | 3.0 | medium-high | estimate |
| 12 | Hopper wgmma tiered kernel for 129-1023 tokens (T2 producers, wgmma consumers) | prompts 129-511: 2x -> 1.1-1.3x 12-bit (E) | 3-5 | high | not planned |
| 13 | Prompts: decode+cuBLAS from the 12-bit layout's thresholds with the T2 unpack | parity >= 769 (A100) / 1024 (H100) (E) | 1.0 | low | partly measured (3.10: BIG T2 -11..-14%) |
| 14 | Per-matrix best-of(T2b, 12-bit) by size | whole-model bits -0.015..+0.020; heavy tensors fast | 0.5 | low | measured sizes |
| 15 | Fixed 3-bit code (mma11, 11.25 bits) | decode ~1.1-1.3x 12-bit | 4-6 | breaks 0.1-bit rule and the headline | rejected; fallback |
| 16 | Tensor-core prefix scans, DPX, clusters, selector arithmetic instead of the table | unknown / worse (ALU-bound) | - | high | not pursued |

## 6. The kernel and the format, as designed

### 6.1 Step kernel for 1-16 tokens (options 3, 4, 5, 6)

- Up to 8 tokens: blocks of 4 warps (128 threads), 4 an SM at <= 128 registers, swapped MMA operands (16 accumulators); two consecutive steps are
  decoded together when both are hot (their phases interleaved, each step in its own scratch), one step otherwise. 9-16 tokens: 4-warp blocks, the
  standard MMA form (32 accumulators), one step at a time (a pair would spill). For matrices of 250+ steps an SM (on an H100 the merged gate/up and the
  down projection), 2-warp blocks with one step at a time are 5% faster (204 vs 215): chosen on the host by size; optional. The register file and
  the 4 SMSPs are the same on the A100 and H100; the L4's 24-block limit is not binding. Warps take interleaved k-ranges of a row block (as today); a
  block's partial sums go through a per-warp shared slab; blocks that share a row block combine through the workspace and the last arrival, in a fixed
  order (as today).
- A warp's loop: load the next step(s) (fixed part 1280 B: 2 + 8 words a lane; the block's window and tail words; x fragments), decode, MMAs, loop.
  The decode: popcounts and a scan, window reads, the group decodes into the scratch, the literal stores, tier-1 consumption, the bf16 pairs.
- The hot decode assumes: at most 512 tier-1 escapes, the block's tier-3 digits and literal entries inside its first 128 bytes, at most 32
  literals. The packer computes a step's flag exactly (bit 31 of its `block_base` entry); the kernel tests it (a shift) and runs the general
  decode, out of line and by value, when set (it is the T2 decode with every path; its result comes back through the scratch). A cold step costs
  1.2 hot steps (an ordinary step forced through it, 3.4) to 2.2 (a heavy one): a matrix with more than 25% cold steps goes to the 12-bit layout.
- Registers 124-128 a thread, no spill; the cold path adds none to the hot loop (a `__noinline__` function: its registers are saved only on
  the cold path). If a toolchain spills the pair, the build selects the single-step kernel (215 instead of 205): a per-architecture build check.
- x fragments: per step 2 `LDG.32` (swapped) or 4; at 9-16 tokens they cost 7% (3.5): the x layout is option 8.

### 6.2 Format T2b (the only format change)

- **Same:** the fixed part of a step (1280 B: tier-1 2-bit digits, then the 8-bit sign/mantissa bytes), `data` byte for byte,
  `block_base` (int32 a step: the block's byte offset), the three `tiers` words (as 4 + 4 + 4 symbols: tier 3 holds four), the
  ranking of exponents (commonest first).
- **Changed (a step's block):** tier 3 is a plain 2-bit digit of 4 symbols (ranks 6-9), no escape digit. A weight of rank >= 10
  (T1: a tier-3 escape followed by a raw byte) is a tier-1 escape whose tier-2 digit is 0 (a placeholder), with an entry in the
  step's literal list: its place among the step's tier-1 escapes (10 bits) and its exponent byte (8 bits), 18 bits packed from a byte
  boundary (nl = floor(8 x list bytes / 18) is exact). Block = tier-3 digits of the tier-2 escapes (from the block's start), the
  literal list, the tier-2 digits of the tier-1 escapes (words of 16, the c-th word back from the end), as T1.
- **Flag:** bit 31 of `block_base[step]`: the step needs the general decode (more than 512 tier-1 escapes, more than 32 literals, tier-3
  digits or literals outside the 128-byte window). The entries' other 31 bits are the offsets as now (blocks < 2 GiB a matrix: the
  packer splits or falls back to the 12-bit layout otherwise).
- **Per matrix:** the packer sizes T2b and the 12-bit layout and keeps the smaller; a matrix with more than 25% cold steps goes to the 12-bit layout
  whatever its size (Qwen3-4B layer 1 down_proj has 18.4% cold steps and stays T2b: 11.5 bits against 12.04, at about T1's speed).
- **Why faster:** one scan level and one set of window shuffles fewer (the third tier's byte list was a second scan whose result the
  tier-2 stage needed); the literals are single-byte stores after a barrier.
- **Cost:** +0.013..+0.034 bits a weight whole-model (3.9); per matrix the packer also sizes the 12-bit layout and keeps the smaller
  (exceptions included): whole-model -0.015..+0.020 against today. Re-packing: a T1 pack is decoded and re-packed (bf16 round
  trip on the GPU or the CPU packer: 3-8 s for an 8B model on 16 cores, the cost A100/H100 users already pay for the 12-bit
  pack); saved checkpoints get a new format name for T2 packs; glyd-v1/v2 packs (T1) keep loading through the conversion.
- **C API:** new functions (`glyd_gpu_mma2_*`, or a layout argument), API version 3; the T1 functions unchanged, so version 2 callers
  keep working. The library keeps the T1 decode only for unpack (conversion at load).

## 7. The plan

### 7.1 Steps and gates

0. **Week 0-1: harness on real GPUs.** The phase-1 harness on one A100 and one H100, 30 min each (9.2): library T1/T2/12-bit and the
   prototype structures, hot and streaming, 1/8/16/64 tokens, with ncu pipe/stall tables. **G0:** the A100/H100 T1 and 12-bit
   clk/step within 15% of 2.1, the best prototype within 10% of its L4 number, ALU-pipe utilization 50-65% (the model's premise).
   Fail: stop, re-model, no library work.
1. **Weeks 1-3: T2b in the library (options 1, 2, 14).** Spec, Rust packer and `pack_mma` T2, the flag, unpack kernel, `Tiered2`
   policy and the existing step kernel on it (8-warp blocks, any M: -12%), conversion at load, C API 3. **G1 (L4):** every tensor of 5 models
   and adversarial tensors bit for bit; the step kernel -12% or better; whole-model bits within +0.02 of T1; every flag checked against the general decode.
2. **Weeks 3-5: the 1-16 token kernel (options 3-5).** **G2 (L4, hot, 4096x4096, the shipped kernel, 3 runs):** <= 212 clk/step at 1 token, <= 235 at 8,
   <= 270 at 16, and at 70 and 141 steps an SM within -25% of T1. Then the A100/H100 library job (9.3). **G3:** A100 within +5% of the 12-bit layout at 1 token and +20% at 8; H100 within +30%
   at 1 token.
3. **Weeks 5-6, only if G3 passes for the A100 and the H100 is within 20-30%:** options 7-10 (conflict-free table, x layout, ALU/FMA
   balance, staging). Stop at 8 weeks of kernel work in all.
4. **Weeks 6-9: 17-64 tokens (option 11).** **G4:** within 20% of the 12-bit layout's mid/wg kernels on the A100, 50% on the H100.
5. **Week 9-10: prompts (option 13);** thresholds from the 12-bit layout's, measured per GPU. The Hopper tiered wgmma kernel (12) only
   on the owner's request.
6. `auto`: the smallest layout wherever the library's route table says within 3% of the 12-bit layout at 1-64 tokens (per GPU
   class, from the last A100/H100 jobs), else the 12-bit layout; no wording change either way.

### 7.2 Why this and not another

- Instructions alone did not pay (the first pair kernel at 2 warps a SMSP: -19% instructions, +1% time); occupancy alone does not (more than
  4 warps a SMSP is flat). What paid: smaller blocks (-15% from the library kernel to 4-warp blocks), swapped operands, and getting the hot loop
  into 128 registers by moving the slow paths out of it.
- Runtime checks cost 4-9%; the flag moves them to the packer where they are exact, for 1%.
- The pair decode, which looked like the lever until the slow paths were taken out, then looked redundant at 2-warp blocks, is worth 4-5% at
  4-warp blocks and keeps 4-warp blocks (the only structure that holds -30% on small matrices) at the best; the single-step kernel is the fallback.
- T2 first because it is the only change that touches the format and everything else builds on one decoder.
- Prompts are not first: a fused kernel decodes each weight again for every 256 tokens; the unpack-then-cuBLAS route costs 3.35 B a weight
  against the 12-bit layout's 3.5.

### 7.3 Expected numbers (projection; replace with the A100/H100 jobs')

clk/step = I x CPI / (4 W x 0.91), I instructions a step, W warps a SMSP, CPI the per-warp cycles an instruction, 0.91 the share of the launch
an SM is busy. Measured points (ncu, L4): T1 604 x 6.7 / (4 x 3.72 x 0.91) = 299; T2 lib 513 x 6.8 / (4 x 3.68 x 0.91) = 260; a flagged
single-step swapped kernel with 2-warp blocks 422 x 6.3 / (4 x 3.55 x 0.90) = 208 (measured 203), its pair 384 x 6.9 / (4 x 3.52 x 0.90) = 209 (measured 203-205). The A100 and H100 differ from the L4 in memory
(the DRAM floors above) and in nothing the decode touches (the same 64K registers, 4 SMSPs, 16-lane integer pipe, 128-byte shared-memory
banks), so the compute-bound clk/step carries over: T1 is 298 (L4), ~295 (A100), ~280 (H100) from the engine.

| tokens | clk/step (L4) | A100 12-bit ~200: plan vs | H100 12-bit ~165: plan vs | e2e, Qwen3-8B one sequence (A100 / H100) |
| --: | --: | --: | --: | --: |
| 1 | ~205 | +3% | +24% | 11.0 / 7.6 ms (now 15.4 / 9.9; 12-bit 10.8 / 6.15) |
| 8 | ~225 | +13% | +36% | |
| 16 | ~262 | +31% | +59% | |

The 12-bit clk/step are derived from the engine's step times (2.1): taking 0.7-1.5 ms for the non-GEMM kernels gives A100 191-208 and H100 146-171, so
"plan vs 12-bit" at one token is -1..+8% (A100) and +20..+40% (H100). The A100's 12-bit number is its DRAM floor (+6%), the H100's about 1.35 x its
floor: the H100's 12-bit kernel is not at the memory limit, so the H100 comparison is the harder one. The conflict-free table, x layout, staging and ALU/FMA options (7-10) are each 3-8%; stacked they might
take 15%: A100 -10% (ahead of the 12-bit layout), H100 +5..+10% at one token. That is a second gate, not a promise.

### 7.4 Kill criteria and fallbacks

- G0 fails: re-model; no library work.
- G2 fails (shipped kernel > 225 clk/step at 1 token): ship T2 with 4-warp blocks (-25% together, 224 clk/step): A100 +12%, H100
  +35%; `auto` keeps the 12-bit layout on A100/H100; spend nothing on options 7-13.
- G3: A100 within the gate and H100 beyond +30% at 1 token: the headline holds on the A100; the owner decides between a Hopper-only
  `auto` rule (12-bit at 1-64 tokens, smallest layout elsewhere) and the 3-bit code (option 15, -29.7%).
- Binary size or compile time of T2 x MT x variants x arch over the release budget: T1 kernels are dropped (T1 only through unpack
  and the conversion), leaving one tiered decoder; the swapped and standard kernels share the decode policy.

### 7.5 What this plan does not reach

At 8-64 tokens the smallest layout stays 13% (A100, 8 tokens) to more than 100% (H100, 64 tokens) slower than the 12-bit layout's kernels: the decode is 7 integer ALU
instructions a weight (221 a step; the 12-bit layout's is 3: 98 a step) and no restructuring found in phase 1 lowers it much further. The
remaining levers (conflict-free tables, x layout, staging, ALU/FMA balance: options 7-10) are each 3-8%, together perhaps 15% if
they stack. Parity on H100 needs a decode about 1.7x cheaper per weight than the best here, which is what a fixed-width
exponent code buys and what the 33% layout cannot have.

## 8. Risks

1. **L4-to-A100/H100 transfer.** Same SM ISA family, registers, SMSPs, integer pipe; T1's clk/step is 298 (L4), ~295 (A100), ~280 (H100).
   Mitigation: G0 first.
2. **ptxas variance** (CUDA 12.x vs 13.x: 125-128 registers for one kernel; the plan's kernel sits at 124-128 under a 128 bound). Mitigation: launch
   bounds, a build-time SASS check (registers, spill bytes) per architecture in the release script.
3. **Flag exactness.** A step wrongly marked hot decodes wrongly (lossless broken). Mitigation: the packer computes the flag from the
   same quantities the kernel uses; a test decodes every step of 7 models with the general path and compares; the harness' fast-only and
   flagged kernels compare bit for bit against T1's product.
4. **Bits and the headline.** Whole-model -0.015..+0.020: the 33% wording measured 32.3% on Qwen3-8B, unchanged to 0.2 points.
5. **Results change in the last bits.** A new work split (2-warp blocks, pairs, swapped operands) changes the fp32 summation order, so
   products differ from v0.27's in rounding (they already differ across GPUs: the split depends on the SM count). `exact` mode
   (decode + cuBLAS) is untouched. Tests compare to an fp32 reference (< 1e-2 of the largest output) and to the kernel's own earlier build.
6. **Release size and compile time:** T2 x MT (1, 2, 4) x {swapped, standard} x {single, pair} x sm_80/86/89/90a; counted in the harden map;
   mitigation in 7.4.
7. **Other branches:** the MoE and KV helpers edit glyd_gpu.cu; T2 adds a policy struct and instantiation lines, and `block_base`
   gets a flag bit that every reader must mask (the existing readers: `load_step`, MoE paths): one grep, one test.
8. **Compatibility:** vLLM plugin, Rust engine (`smallest_is_fastest`), `glyd.gpu` (`best_layout`, `e2e.py --format auto`), checkpoints: the C API
   addition is additive; the layout choice stays in one function.

## 9. Test plan

### 9.1 Correctness on the L4 (every milestone)

1. Format: `pack_v2.py` vs the Rust packer vs `pack_mma` byte for byte (data, blocks, block_base with flags, tiers) on every tensor of Qwen3-8B,
   Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, a MoE (granite-3.1-3b-a800m: experts, routers), and synthetic tensors: 256 exponents uniform,
   one exponent, NaN payloads, +-0, subnormals, +-inf, 50% rank >= 10, steps with 513+ tier-1 escapes, 33+ literals, literals and tier-3
   digits at the window's edge (positions 988-992), a block start with every alignment (blk & 3), K = 16, O = 64.
2. Unpack: the GPU unpack kernel returns every weight's bits (`torch.equal` on int16 views) for T2 on all of the above; the T1 -> T2
   conversion at load returns the same.
3. Products (`check_capi` style): 1, 2, 7, 8, 9, 16, 17, 63, 64 tokens, every tensor of Qwen3-8B layer 10 and the heavy tensors,
   against an fp32 reference (< 1e-2 of the largest output), against the kernel's own single-step build (< 1 bf16 ulp), and twice
   (deterministic). The self-test (1-600 tokens, 8 shapes, exceptions) and check_capi (1500 calls) as for any release.
4. Static: registers and spill bytes of every instantiation on sm_80/86/89/90a (`ptxas -v`; `regs.py`), the hot loop's instruction counts
   (`sassstat.py`) against budgets.
5. Speed on the L4 (the harness, hot and cold, interleaved, clock-normalized) against the gates of 7.1.

### 9.2 A100 and H100 harness job (phase 2 week 1; at most 30 min each; no downloads, no setting changes)

A tarball `tf-job.tgz` (built by me; rehearsed end to end on the L4 first): `tf_job.sh`, `summarize.py`, prebuilt `bin/harness_80` and
`bin/harness_90` (static cudart; built from the phase-1 sources for sm_80 and sm_90a; `harness_89` is the L4's, for a rehearsal) and the packs
`o10` (o_proj 4096x4096 of Qwen3-8B layer 10: 22.6 MB tiered, 25 MB 12-bit, L2-resident on both GPUs), `u10` (up_proj 12288x4096: 68 MB
tiered, 75 MB 12-bit, streams from HBM; weights rotated over 3 copies), `c2` (the first 2048 rows of o10), `g1` (a heavy-tailed tensor) and `m2` (3.6% cold
steps). The machine needs the NVIDIA driver (ncu optional, with sudo); no CUDA
toolkit, no PyTorch; `summarize.py` needs python3. Run: `tar xzf tf-job.tgz && cd tf-job && ./tf_job.sh`, then send back `results-*`.

`./tf_job.sh` (each step with a timeout; results in `results-<host>-<time>/`, summary printed and saved):
1. `nvidia-smi -q` and a query line (clocks, power cap, driver): recorded, not set.
2. Streaming, `u10`: 1, 8, 16 tokens (and 32, 64): the library's T1, T2 and 12-bit kernels, the shipped-route kernels at 17-128 tokens, and the
   prototype structures (flagged single step and pair, 2- and 4-warp blocks, swapped operands, group table 16/32 times over); products against the
   reference and bit for bit.
3. L2-resident, `o10`: the same. (A 12-17 us call on an H100 carries 1-2 us of launch gap: compare ratios, and take the streaming runs as the
   reference.)
4. Exactness of the cold path on this GPU: `g1` (a heavy tensor, every step takes the general decode), `m2` (3.6% of the steps cold among hot ones: the
   switch inside one product), and `o10` with every 20th and every 5th step forced cold (the product must not change; the time is the cost of a cold
   step on this GPU); flagged kernels compared bit for bit with T1's product.
5. A small matrix (`c2`, 2048x4096: 62 steps an SM on an H100) at 1 and 8 tokens; prompt kernels on `o10`: 128, 256, 1024 tokens (T1 big, T2 big,
   12-bit big).
6. `ncu` (if present), 1 token: the T1, T2 and 12-bit library kernels and the best prototype (the 4-warp pair): pipe utilization, issue stalls,
   instruction mix, shared-memory wavefronts and conflicts.
Time: about 8 min of runs, ncu 8-10 min on an H100, setup 5 min. `summarize.py` prints clk/step per variant, the ratios to the 12-bit and T1
library kernels, and the ncu tables that G0 needs.

### 9.3 A100 and H100 library job (phase 2 week 5; at most 30 min each)

The library built with T2 and the new step kernel (compiled `libglyd_gpu` for sm_80 / sm_90a as gpu_lambda.sh does) and `e2e.py --format
auto/mma/mma12 --merge --profile 1 8 16 64` on Qwen3-8B (and 14B on the A100), the per-layer step table (`gemm.py` style: the 7 matrices,
1/8/16/32/64 tokens, mma vs mma12 vs cuBLAS bf16), prompts 129-4096 tokens one pass, the engine's batch-1 decode benchmark
(`bench-mma-lib`, `bench-mma12-lib`), `check_capi` and the self-test on each GPU, and the harness job's ncu step again.

## 10. Tools (in `gpu/experimental/tiered_fast/`; phase-1 result texts in `res/`)

- `pack_v2.py`: CPU packers T1/T2 (`pack`), the 12-bit layout (`pack_nib`), the reference decoder, `cold_flags`/`flag_bb`, `dump` (pack files for
  the harness), `python pack_v2.py flag PREFIX`; `test_pack_v2.py`, `dbg_pack.py`; `v2bits.py`, `v2bits_all.py`, `v2bits_best.py`, `v2limits.py`,
  `expstats.py`: bits a weight and fast-path limits per model and tensor.
- `harness.cu`, `v2.cuh`: the T2 decode policy and its option bits, the step kernel copy (`NW` warps a block, `MB` blocks an SM, swapped operands,
  pairs, flagged steps), the benchmark (interleaved rounds, rotated order, clk/step from the clock probe), unpack and product checks, prompt-kernel
  and shipped-route comparisons. `build_harness.sh [OUT] [ARCH]`, `ncu_h.sh`, `regs.py` (registers and spills from `ptxas -v`), `stepbench.py`
  (torch microbench of the library kernels on real tensors).
- `sassstat.py`, `ncu_show.py`, `ncu_cmp.py`, `mix_show.py`, `src_show.py`, `src_blocks.py`, `src_all.py`, `sass_stalls.py`: SASS opcode/pipe
  statistics, ncu raw/mix/source views, per-line instruction, wavefront and stall tables.
- `tf_job.sh`, `summarize.py`: the A100/H100 job (9.2).

The plan's kernel is `o552u1w2b8s` (OPT 552 = 8 + 32 + 512: literals through the window, hoisted tier-1 lookups, flagged steps; one step a warp; 2-warp
blocks, 8 an SM; swapped) up to 8 tokens and `o552u1w4b4` (4-warp blocks, 4 an SM) from 9 to 16; 568 = the same decoding two steps together.
Option bits of `Tiered2o<OPT>`: 1 predicated scan, 2 register window, 4 16-copy table, 8 literals via window, 16 pair scratch, 32 hoisted
tier-1 lookups, 64 radix-4 scan, 128 fast-only, 256 bail-out to the general decode, 512 flagged steps (hot/cold), 1024/2048/4096 single
assumptions, 8192 coalesced-x timing experiment, 16384/32768 32/16-copy tables. Variant names: `oOPTuUNRwNWbMB[s]` (UNR 1 single, 3 pair; NW warps a
block; MB blocks an SM; s swapped operands).
