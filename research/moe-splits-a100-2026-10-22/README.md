# The FP4 experts' splits and blocks an SM, one A100 40 GB (2026-10-22)

Lambda gpu_1x_a100_sxm4 (us-east-1; no H100 had capacity), 01:23-01:43 EDT, about $0.70. `moe_splits` gives DeepSeek-V4-Flash's 384 units (6 experts x 64
row blocks) one split; on an H100 (264 slots) that is 1.45 waves, on this A100 (216) 1.78. `GLYD_GPU_MOE_SLOTS` forces the slots, so the splits;
`GLYD_GPU_STAGES=8 GLYD_GPU_STAGE_BLOCKS=3` the staged loop at 3 blocks an SM. `launch.sh`, `job.sh`, `inner.sh`; `out/`.

DeepSeek at the release's widths (random, 3 layers), a step with a graph:

| | ms |
|---|---|
| default (1 split), twice | 2.176, 2.251 |
| slots 528 (1 split) | 2.268 |
| slots 792, 1,056 (2 splits) | 2.254, 2.352 |
| staged, 3 blocks an SM | 2.257 |
| staged, 3 blocks an SM, 2 splits | 2.396 |

Kimi-K3's first two real layers, its experts on the GPU, bf16 in the 12-bit layout (`--layout mma12`), a step with a graph:

| | ms |
|---|---|
| default, twice | 4.177, 4.116 |
| slots 1,848 (its down's 896 units in 2 splits) | 4.128 |
| staged, 3 blocks an SM | 4.131 |

Nothing beats the default past the run-to-run spread (about 3.5%): more splits cost their parts' sums, the staged loop its shared memory. The
heuristic stays. On the 4080 the same: 2.350 / 2.353 / 2.373 ms for 1, 2, 3 splits. Whether an H100's 1.45 waves change this is for a run there.
(dsv4check on the small random model came out the same with the forced slots, but its products are too small to split: K / 256 is 1 there. A split changes the
fp32 order of a product's sums, so its digits by a little.)
