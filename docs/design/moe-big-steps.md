# The MoE grouped products at 768 to 8,192 tokens a step (2026-10-03, second version)

Design note. The first version (the same day) was written from the H100's sweep alone and listed its hypotheses as "none profiled". This one adds what has been measured since: Nsight Compute on the H100's grouped gate and up product (`benchmarks/gpu/h100-moe-v028-2026-10-03/prof`), the A100's two runs (`benchmarks/gpu/a100-moe-v028-2026-10-03`: `README-first-run.md`, then `README.md`, the final numbers), and three captures on an L4, which runs the A100's tiled kernel as it is (`.../l4-tiled`). Numbers are measured where they say so; a gain is an estimate where it says so. The A100's second run is the last section but one.

## What the sweeps say

Qwen3-30B-A3B's layer 0 (128 experts, 8 a token, hidden 2048, intermediate 768), GPU time in a CUDA graph, against vLLM's own bf16 layer with its default (untuned) fused-MoE configuration.

H100 SXM at 1,980 MHz (the H100's record):

| Tokens a step | bf16 us | Glyd us | Glyd / bf16 | gate+up us (TFLOP/s) | down us (TFLOP/s) |
| ---: | ---: | ---: | ---: | :--- | :--- |
| 232 | 432 | 438 | 1.01x | 280 (42) | 150 (39) |
| 1,152 | 502 | 693 | 1.38x | 356 (163) | 302 (96) |
| 2,048 | 607 | 1,052 | 1.73x | 535 (193) | 495 (104) |
| 4,096 | 940 | 1,856 | 1.97x | 867 (238) | 909 (113) |
| 8,192 | 1,713 | 3,453 | 2.01x | 1,551 (266) | 1,736 (119) |

A100-SXM4-40GB at 1,410 MHz (the A100's first run, `README-first-run.md`, with the settings of that run: the tiled kernel from 48 pairs an expert for both products, the rows kernel off; the second run's are below):

| Tokens a step | bf16 us | Glyd us | Glyd / bf16 | gate+up us | down us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 232 | 946 | 1,341 | 1.42x | 707 | 620 |
| 768 | 1,157 | 2,135 | 1.85x | 1,423 | 677 |
| 1,152 | 1,214 | 2,440 | 2.01x | 1,647 | 753 |
| 2,048 | 1,540 | 2,815 | 1.83x | 1,693 | 1,059 |
| 4,096 | 2,482 | 4,865 | 1.96x | 2,897 | 1,855 |
| 8,192 | 4,450 | 8,983 | 2.02x | 5,327 | 3,437 |

The gate and up run at 25% of the A100's peak at 8,192 tokens, the down at 19%, vLLM's bf16 layer at 45%. On both GPUs it is the same story from about 768 tokens: the compute-bound range, where Glyd's products take twice bf16's time.

## What the serves say

The A100's server cuts a step at 2,048 tokens (the H100's: 8,192): a prompt of 1,024 tokens is one step of about 1,030, and a saturated pass is prompt steps of 2,048 tokens and decode steps. On the A100 (Qwen3-16B-A3B, 64 experts, lossless KV cache) the prompt steps are 24.2 of the pass's 47.6 s (147 ms a step, 80 us a prompt token against bf16's 52); at one request a second the first token is 119 to 125 ms against bf16's 102, 17 to 23 ms later, which is one prompt step's MoE layers. The kernel both run is the tiled one: the A100's two open numbers (first token at low load 1.17 to 1.23x; saturated each-token 2.12 to 2.26x) ride on it. On the H100 (Qwen3-30B-A3B) 40% of the saturated pass (5.7 of 14.3 s) is 32 prompt steps at 22 us a token against bf16's 15.

## What the H100's profile says (gate and up, 232 and 8,192 tokens)

`mma12_moe_wg_kernel`, at ncu's locked clock (1.44 GHz; the sweep's 1.98 GHz times are shorter: 342 against 280 us at 232 tokens, 1.75 ms against 1.55 at 8,192). The capture is the gate and up product alone: the job's two captured launches were both gate+up (the check calls it once for each of 12 routings before it calls the down). The job now makes the check's calls once, so that its two launches are gate+up and down.

| | 232 tokens (tile of 32) | 8,192 tokens (tile of 128) |
| :--- | ---: | ---: |
| Duration | 341.8 us | 1.75 ms |
| DRAM throughput; bytes | 53.8%; 614 MB (the packed gate+up weights, 607 MB, once) | 16.6%; 876 MB |
| Tensor pipe | 10.3% | 34.4% |
| Issue slots active | 31.5% | 29.2% |
| Warp instructions | 76.7 M | 379.8 M (4.95x for 4x the tiles) |
| Warps an SM; eligible a scheduler | 9 (14%); 0.39 of 2.25 | 9; 0.36 of 2.25 |
| Registers a thread; shared memory | 116; 165 KB | 159; 226 KB |
| Stalls (of those sampled) | wait 25%, long scoreboard 24%, selected 13%, short scoreboard 13%, branch 7%, warpgroup arrive 6% | wait 27%, long scoreboard 18%, selected 13%, short scoreboard 13%, branch 7%, warpgroup arrive 6% |

1. **The kernel is latency-bound.** DRAM is at 17% and the tensor pipe at 34% at 8,192 tokens. One CTA of 9 warps an SM leaves a scheduler 2.25 warps; each issues an instruction every 7.2 to 7.7 cycles, and 0.36 of them are eligible on a cycle. At 232 tokens it reads the weights once at 1.8 TB/s of the 3.35 the part has (the copy rate is 2.96).
2. **A stage takes 3.3 times the tensor work.** At 8,192 tokens 2.53 M cycles for 512 tiles of 384 (unit, stage) pairs, 1,489 an SM, are 1,697 cycles a stage against the 512 cycles of tensor work of its eight wgmma m64n128k16 (both warpgroups). A consumer warp issues about 240 instructions a stage, nearly all of them the decode of its 16 rows.
3. **The decode is repeated for every tile.** Warp instructions grow with the tiles (4.95x for 4x), not with the tokens: the instruction stream is the decode's. The profile supports the first version's hypotheses 1 (the decode is on the critical path) and 2 (each tile decodes the expert again). Hypothesis 3 (the epilogue not overlapped) the capture cannot show (gate+up's is one 32nd of a unit); 4 and 5 (the down's fp32 pair rows, the route) are not in it.
4. By instruction (the stall samples of the same captures by SASS instruction; the executed counts at 8,192 tokens): 393 M warp instructions, of which the wgmma (HGMMA) are 7 M, 1.8%: 56 others to each. The ALU work of the decode (LOP3 9.9%, IMAD 11.2%, PRMT 7.9%, SHF 5.9%, ISETP 5.8%, VIADD 5.1%, IADD3 4.3%, LDS 7.0%) is the rest of the stream. The BRA instructions hold 22% of the samples (29% at 232 tokens) and 92 to 93% of the long-scoreboard stalls: the mbarrier waits, which are the producer warp's most of the time (one warp of nine) and the consumers' wait for a stage. The capture does not say how much of the BRA is the consumers waiting for data and how much the producer idling; per-instruction samples of the consumers alone would.
5. ncu's shared-memory report (2.4 to 2.8 wavefronts a request, "bank conflicts" 50 to 59% of the wavefronts) is for 16-byte accesses, whose ideal is 4 wavefronts a request: not read as conflicts. The global stores are 16 of every 32 bytes of a sector (the epilogue's 4-byte lanes): ncu's estimate is 19 to 25%, a minor share. Instruction fetch is not among the stalls (under 6%).

## What an L4 says about the A100's kernel

The A100 and the L4 run `mma_gemm_big_kernel` (producer warps decode the 12-bit layout into shared memory, four consumer warps multiply, one block an SM). Per-instruction stall samples (Nsight Compute, `l4-tiled/sass`) of its gate and up product on the real layer at 2,048 tokens and on a compute-bound layer (Qwen3-30B-A3B's shapes, 6 experts so that the weights stay in the L2 cache), the instructions told apart by where they sit:

- the consumers' main loop: 34 to 40% of the warp instructions and 35 to 50% of the stall samples, stalled on the tensor pipe's issue interval (wait and math pipe throttle: 51 to 64% of its samples) and on the barriers for the producers (13 to 27%);
- the producers' loop: 54 to 59% of the instructions (IMAD 27% of them, the decode's byte permutes 7%), 31 to 40% of the samples, mostly waiting for the memory (long scoreboard) and for the consumers (barrier);
- **the epilogue and the paths a step rarely takes: 6 to 8% of the instructions and 29 to 31% of the samples** (55% of those instruction fetch stalls). 17,000 instructions of code, run only by the consumers once the producers are done: about two fifths of a consumer warp's life, so of a tile's.

The epilogue chose the gate's activation at each of its 128 products a thread (SiLU's code and GELU's one among the other, the IEEE division and its slow path in each) and the down's read each pair's row and weight one after the other, behind its own stores. It is fixed (commit 7820a5ea): the activation is chosen once, SiLU divides by the approximate reciprocal (2 ulp, before the bf16 the product is rounded to), the down asks for its 8 rows and weights together. On the L4: the compute-bound layer 21% (128 pairs an expert), 12%, 13% and 14% (1,024 pairs) off the layer, the gate and up 16 to 25%, the down 9 to 16%; the real layer (from memory, at the L4's DRAM floor now: 88% and 92%) 11% at 1,152 tokens, 6% at 2,048, none at 384 where the decode kernels run. Instruction-fetch stalls: 18 to 19% of all the stalls before, 5 to 6% after.

The other two changes of the round are measured there too: **skipping the m-tiles with no pairs** takes 17% (128 pairs an expert) and 7% (256) off the compute-bound layer and nothing where the tiles are full; **a tile of 256 tokens** (decode each weight once for 256 pairs, eight consumer warps) was 9 to 14% faster at 512 and more pairs an expert before the epilogue fix, and is 1 to 3.5% faster at 512 and 1,024 now, 9.5% slower at 256 and 60% slower at 128: its eight warps have not the registers to ask for their fragments a step ahead, and what a larger tile shared was the epilogue. It stays a variant.

What the L4 could not say was how much of this is the A100's (its tensor rate is twice the L4's per SM and its memory five times); the second run's capture of the A100 itself is below.

## Options

Each alone, an estimate unless it says measured; effort is days of work with the checks.

| # | Change | GPU | Gain | Effort | Gain a week |
| :--- | :--- | :--- | :--- | ---: | :--- |
| T1 | The tiled kernel's epilogues shrunk (built) | A100, L4, any `mma.sync` GPU | measured on the A100 (the first run's settings with the new code, against that run's own numbers): the layer 9% at 2,048, 11% at 4,096, 12% at 8,192 tokens and 23-29% at 768-1,152 (gate+up 12-15% and 31-39%, down 5-7% and 8-11%); on an L4 12-25% of the products with the weights in cache, 6-11% from memory | 1 day | highest |
| T2 | m-tiles with no pairs skipped in the tiled kernel (built) | the same | measured on the A100 (`noskip`): 1-5% from 768 to 8,192 tokens; on an L4 up to 17% at 128 pairs an expert, 4-8% on the real layer | 1 day | high |
| T3 | The A100's thresholds per product (the tiled GEMM from 84 pairs an expert for the gate and up, 48 for the down), the rows kernel from 17 tokens, the decode kernels' m-tile skipping (built, from the first run's sweep) | A100 | measured on the A100 (built-in against `old`): 15-28% at 32-512 tokens, 14% at 768, 1-5% from 1,536; but 16% worse at 1,152, where the thresholds are now wrong (below) | 2 days | high, the thresholds to redo |
| T4 | A tile of 256 tokens (built, off) | A100, L4 | none where it was measured (see above) | 3 days | none |
| T5 | The producers' instructions slimmed (the exceptions' loads asked for with the step's, the two steps' exception loops merged; the 32-bit row indices that took IMAD from 28% to 8% of their instructions are in): they are 55% of the instructions | A100, L4 | none: the A100's capture has them finishing at about 60% of a tile's life | 2-3 days | dropped |
| T6 | The producers' reads staged by `cp.async` into a ring, as the dense mid-size kernel does (`mma12_ws_kernel`) | A100 | unknown | 1-2 weeks | unknown |
| T7 | Two blocks an SM (smaller tiles), or one block that starts the next tile's stages while it finishes its epilogue (24% and 27% of a consumer warp's life on the A100, gate and up and down) | A100 | unknown; two blocks need under 128 registers a thread | 1-2 weeks | the capture's case |
| H1 | A tile of 256 pairs for the wgmma kernel: it halves the decodes from about 4,096 tokens, and the profile says time follows the decodes | H100 | 35-45% of gate+up at 8,192 tokens (the first version had 14%, made with the tensor pipe as the bound); the sums need 128 registers, spills possible | 3-5 days, needs an H100 run | high |
| H2 | The down's pair rows in bf16, weighted in fp32 before the rounding as vLLM's kernel does | all | 5% of the layer at 8,192 tokens | days | low |
| H3 | The route's tile table in parallel | H100 | 3% | days | low |
| H4 | Decode and products in different warps (SS wgmma from shared memory) | H100 | 1.3-1.7 ms of 3.4 at 8,192 tokens | 3-4 weeks | the dearest |

## The pick

T1, T2 and T3 are built, on the A100-class code, which an L4 runs: the best gain per week and what both the prompt-heavy steps and the A100's first token ride on (the 1,030-token prompt step of the E=64 model is the tiled kernel's, and so is the A100's 2,048-token saturated step). All three leave the results as they were (134 checks pass on the L4: the real layer at 17-4,096 tokens in seven settings, skewed routing, exceptions by the dozen, GELU), apart from the last bits of the gate's activation. The A100's second run measured them (the table and the last section): more than the L4's percentages at 768-1,152 tokens, a little less above.

Next, after the release: (1) the A100's epilogue and consumers (below); (2) H1 on the H100, parked until after the release (the profile is its case; the down kernel's capture, which the job's prof part now makes, is its other half).

## What the A100's second run says (sweep, serve and Nsight on the tree of this note; `benchmarks/gpu/a100-moe-v028-2026-10-03/moea100c`)

- Serve (Qwen3-16B-A3B, COMPARABLE at 1,410 MHz): saturated 6.18 requests a second, 1.50x bf16 (the first run's like-for-like mode, the attention bf16: 5.44, 1.35x), first token 0.53x, each token 2.04x (2.12x); at one request a second the first token is 1.08x (110 against 102 ms; 1.17x), each token 0.88x. Prompt steps 133 ms (72 us a token; 140 and 76), decode steps 30.5 ms (37.5).
- Sweep, Qwen3-30B-A3B's layer: 0.96-1.12x bf16 from 32 to 512 tokens (232: 1.01x, was 1.42x), 1.22x at 768, 1.67x at 1,152, 1.58x at 2,048, 1.70-1.75x at 4,096-8,192 (was 1.83-2.02x from 768 on).
- Isolated on the A100 (`old`, the first run's settings with the new code, against that run's own `new`): the layer -23% at 768, -29% at 1,152, -9% at 2,048, -11% at 4,096, -12% at 8,192 tokens, nothing at 512 and below; the built-in dispatch against `old`: -15% to -28% from 32 to 512, -14% at 768, +16% at 1,152, -1% to -5% above.
- The thresholds built in from the first run (the tiled gate and up from 84 pairs an expert, the down from 48) are now off, the tiled kernel being faster: `gu64` takes 16% off at 1,152 tokens, the rows kernel for the down (`dn96`) 11% off at 768 and costs 4% at 1,152; 64 for both would give each (1.09x and 1.40x bf16, from the variants' parts, not run together). Not changed.
- A tile of 256 (`tm256`, 2 stages) takes 22%, 13% and 19% off at 3,072, 4,096 and 8,192 tokens (1.34x, 1.48x, 1.43x bf16, ahead of the decoded route's 1.92x, 1.75x, 1.44x) and costs 1-32% from 768 to 2,048; 3 stages (`nb3`) are worse than 2. The A100's server cuts a step at 2,048 tokens, so neither is reached; the decoded route's start (3,072) was set against the first run's kernel and is now about 4,500.
- Nsight on the A100 (the real layer, 2,048 tokens, gate and up 1.77 ms at ncu's clock): tensor pipe 25.5% of peak, issue slots 28%, DRAM 25%, 204.6 M warp instructions (the L4's 205.7 M: the same code), 255 registers, no spills.
- By instruction: the consumers' loop is 39% of the instructions and 47% of the samples (wait 49%: the tensor pipe's issue gaps), the epilogue and rare code 5% and 15% (instruction fetch still 29% of those), the producers 56% and 38%: they wait for the consumers (the barrier, 23%) and are done at about 60% of a consumer warp's life (the down: 50%). The producers are not the bound; the epilogue is 24% of a consumer warp's life for the gate and up, 27% for the down.
- So T5 is dropped, and what is left is the epilogue's overlap, 4 stages (128 KB fits an A100's shared memory, an L4's does not) and consumers of half a row block (8 warps, a tile of 128: the dense kernel's Ada shape, which an L4 can try).

## What would change this

- A tuned bf16 baseline makes every gap larger; the comparison to quote stays the untuned one, named.
- The capture put the A100's stage on the consumers and the epilogue; if the consumers of half a row block do not take it, T6 (the ring) is the next, and T5 stays dropped.
- A workload of short prompts (decode steps only) is at bf16's time or ahead of it on both GPUs; none of this touches it.
