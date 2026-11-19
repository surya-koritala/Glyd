# Staged and prefetched one-token expert products on an A100 (2026-10-19)

Lambda gpu_1x_a100_sxm4 (A100-SXM4-40GB), about 40 minutes, about $1.35. No H100 launch went through (two 400s on listed capacity).
Branch hbm-prefetch: `GLYD_GPU_STAGES` (weight steps through a shared-memory ring, cp.async), `GLYD_GPU_STAGE_BLOCKS`, `GLYD_GPU_PF`.
Both moechecks (STAGES=4 with 3 blocks, STAGES=8) all within bounds.

The whole MoE layer at one token (`moecheck dsv4 --speed`, us):

| Layout | as built | PF 2 | PF 4 | STAGES 4 | 4, 3 blocks | 8 | 8, 3 blocks |
|---|---|---|---|---|---|---|---|
| int5 | 210.5 | 181.4 | 185.8 | **176.8** | 179.2 | 203.8 | 218.4 |
| int6 | 192.3 | 193.3 | 196.6 | **188.9** | 190.5 | 212.2 | 225.4 |
| int4 | 161.3 | 160.8 | 162.6 | **153.5** | 156.7 | 178.2 | 192.6 |
| fp4 | **186.4** | 198.5 | 201.5 | 191.0 | 196.2 | 220.0 | 231.2 |
| fp4, packed scales | **205.2** | 220.0 | 222.1 | 210.7 | 216.1 | 242.5 | 253.5 |

DeepSeek's 3-layer step with a graph (its experts are fp4 with packed scales): 3.216 ms as built, 3.259 with STAGES 4, 3.360 with 8.

- **Staging helps the integer layouts (4-16%), not FP4.**
  - Nsight: the staged kernel takes 80 registers (113 before), but its 66 KB of shared memory a block leaves 2 blocks an SM on the A100 (3 by registers).
  - A slot's 512 bytes of activations are most of that; most lanes' are zero at one token.
- **FP4 is bound by integer issue, not memory.**
  - A step's decode is about 120 integer instructions (prmt, lop3, shifts) for 592 bytes.
  - The integer pipe issues at half the rate of the others.
  - That caps the FP4 experts near 75% of an H100's bandwidth and below the A100's before any latency.
- **Next:**
  - FP4: the scales' preparation once a run (2 steps), not each step.
  - Staging: smaller slots (activations in registers), then an H100 run.
