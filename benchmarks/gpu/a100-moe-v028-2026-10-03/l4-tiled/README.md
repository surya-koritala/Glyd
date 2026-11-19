# The A100's tiled MoE kernel on an L4: three captures and an epilogue (2026-10-03)

One NVIDIA L4 (Ada, sm_89, 58 SMs, 48 MB L2, 24 GB GDDR6 at 300 GB/s; the development machine, an AWS g6.4xlarge), the development library built for it with the job's flags. The L4 runs `mma_gemm_big_kernel` as the A100 does: the same source (producer warps decode the 12-bit layout into shared memory, four consumer warps multiply on `mma.sync`, one block an SM). It does not share the A100's memory (1.55 TB/s against 300 GB/s) or its tensor rate per SM (about twice the L4's), so what an L4 measures is what the kernel's own code costs, not the A100's time; the A100's own numbers on this tree are the second run's (`../README.md`: the serve, the sweep and an Nsight capture of the same kernel).

Why it was done: the A100's saturated serve spends 24.2 of 47.6 s in prompt steps and its first token at low load is one prompt step, and both run the tiled kernel (`../README-first-run.md`: 25% of the A100's peak at 8,192 tokens for the gate and up, 19% for the down, against bf16's 45%). The H100's capture (`../../h100-moe-v028-2026-10-03/prof`) said its kernel is latency-bound; this asks the same of the A100's.

## What was found

**The kernel's epilogue was about 40% of a tile's time.** Nsight Compute's per-instruction stall samples (`sass/`, `sass-regions.txt`) of the gate and up product, split into the consumers' main loop, the producers' loop, and the rest (the epilogue and the paths a step rarely takes):

| Gate and up, an L4 | Warp instructions | Static instructions | Samples in the epilogue and rare code | Their stalls |
| :--- | ---: | ---: | ---: | :--- |
| Real layer, 2,048 tokens, as it was | 210 M (7% in the epilogue) | 17,432 | 29.1% | instruction fetch 55%, wait 22% |
| Compute-bound layer, 1,536 tokens, as it was | 32.2 M (8%) | 17,566 | 31.2% | instruction fetch 54%, wait 22% |
| the same, the epilogues shrunk | 30.4 M (6%) | 15,851 | 12.6% | wait 30%, instruction fetch 30% |

Only the consumers run it, after the producers are done, so a quarter of the warps' samples are about two fifths of a consumer warp's life. The epilogue chose the gate's activation at each of its 128 products (SiLU's code and GELU's one among the other, the IEEE division and its slow path in each), and the down's read each pair's row and weight in turn, behind its own stores. Now the activation is chosen once, SiLU divides by the approximate reciprocal (2 ulp, before the bf16 the product is rounded to), and the down asks for its 8 rows and weights together. Instruction-fetch stalls fall from 18-19% of all the stalls to 5-6%; 2,717 static instructions run in the epilogue now.

On the real layer (128 experts, from memory) the gate and up product's DRAM throughput went from 78% to 88% and the down's is 92%: the kernel is at the L4's memory floor there; on the compute-bound layer the gate and up's tensor pipe went from 32% to 46%.

| Layer, us (median of two rounds, the trees in turn) | Gate and up before | after | Down before | after | All before | after |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compute-bound (6 experts, 128 pairs an expert) | 167 | 125 (-25%) | 80 | 68 (-16%) | 295 | 232 (-21%) |
| 256 pairs | 306 | 239 (-22%) | 157 | 143 (-9%) | 529 | 463 (-12%) |
| 512 pairs | 594 | 462 (-22%) | 325 | 286 (-12%) | 1,066 | 924 (-13%) |
| 1,024 pairs | 1,145 | 965 (-16%) | 804 | 719 (-11%) | 2,023 | 1,739 (-14%) |
| Real layer, 1,152 tokens | 3,232 | 2,887 (-11%) | 2,146 | 1,904 (-11%) | 5,368 | 4,797 (-11%) |
| Real layer, 2,048 tokens | 4,234 | 3,934 (-7%) | 3,087 | 2,831 (-8%) | 7,353 | 6,882 (-6%) |
| Real layer, 384 tokens (the decode kernels) | 2,543 | 2,542 | 1,436 | 1,437 | 4,005 | 4,005 |

Against vLLM's bf16 layer on the compute-bound layer (its 620 and 1,542 us at 512 and 1,024 pairs an expert, `hold2.txt`): 1.72x to 1.49x at 512 and 1.31x to 1.13x at 1,024.

**Skipping the m-tiles with no pairs** (the consumers multiply the tokens of an expert's last tile alone, `GLYD_GPU_BIG_SKIP`): on the compute-bound layer with the epilogues shrunk 232 against 280 us at 128 pairs an expert (-17%), 463 against 500 at 256 (-7%), 924 against 915 at 512 and 1,739 against 1,778 at 1,024 (the tiles are full: no difference to 2%); with the old epilogues -3% and -9% at 128 and 256. On the real layer (the old epilogues) -8%, -4% and -6% at 384, 1,152 and 2,048 tokens. On.

**A tile of 256 tokens** (eight consumer warps, `GLYD_GPU_BIG_TM=256`, so that a weight is decoded once for 256 pairs): with the epilogues as they were, 9-14% faster at 512-1,024 pairs an expert, 6% slower at 256, 28-36% slower at 64-128; with them shrunk, 1% faster at 512, 3.5% at 1,024, 9.5% slower at 256 and 60% slower at 128. The eight warps do not hide their fragment loads as the four-warp loop does (it asks for a step's fragments a step ahead; the eight-warp loop has not the registers), and what a larger tile shared was the epilogue's cost. Off; a variant for the A100's sweep.

**The decode kernels** (the gate and up product `mma_moe_kernel` and the rows kernel for the down, Qwen3-30B-A3B's real layer at 232 tokens, an L4): 94% and 95% of the L4's DRAM throughput, long-scoreboard stalls 74% and 79%: at the L4's memory floor, so there is nothing of their own to find on an L4. The A100's gate and up at 232 tokens runs at about 63% of its copy rate (707 us against 446 at the floor); that gap is not visible here.

**The job on an L4** (hold 4, `hold4-smoke-of-the-job.txt`): `job-next/moe_job.sh` with the class forced to an A100 and tiny models for the serves, 210 s end to end: the check (162 checks), the sweep with the A100's variants, the prof part with its per-instruction export, the dense part; the serves did not start (the tiny model the smoke used was no longer on the machine), so the serve flow itself was not exercised there; it is the A100 run's, but for its mode list (`kv again`). The GPU test of the tiled kernel's shapes (`test_moe_tiled_kernel_shapes`) passes on the tree's library, and so do the tiered layout's checks (the L4's own layout) through the tiled kernel in four settings and with GELU (`hold4-decode-and-tests.txt`).

## Files

- `hold1.txt`, `hold2.txt`, `hold3.txt`, `hold4-*.txt`: the four holds of the box's lock, as printed (`scripts/hold_base.sh`, `hold_tm.sh`, `hold_epi.sh`; `hold_dec.sh` the decode kernel's capture). Hold 1 the tree as the A100's run had it (a real-layer and a compute-bound sweep, ncu of the tiled kernel at 2,048 tokens); hold 2 the tile of 256 and the skipping on the compute-bound layer and the real one, ncu of both tiles, 128 checks; hold 3 the epilogues: the two trees in turn twice, ncu, 134 checks (every path: the real layer at 17-4,096 tokens in 7 settings, skewed routing, exceptions by the dozen, a layer with a K of four stages, GELU), all pass.
- `data/`: the JSON of every sweep, the checks' lines and the ncu summaries and raw metrics (`raw-*.csv.gz`; the reports themselves, 15-72 MB each, are not kept).
- `sass/`: each capture's stall samples by instruction (`ncu --page source --print-source sass`), gzipped; `sass-regions.txt` is `scripts/sassreg.py` over them. The regions are told apart by where the HMMA and LDGSTS instructions are, not by source lines.
- `scripts/`: the holds' scripts, `devbuild.sh` (a tree's development library for this GPU, the job's flags), `devrun.sh`, `ptxv.sh` (ptxas's resource lines for sm_80), `ncu_summary.py` (the job's).

The kernel with its epilogues shrunk uses 255 registers on sm_80 (before 239) with no spills; the eight-warp tile spills 524 bytes at 168.
