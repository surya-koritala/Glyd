# A100: the FP4 decode on the floating-point pipe, and the attention and compressor kernels (2026-10-19)

Lambda gpu_1x_a100_sxm4, about 40 minutes, about $1.40.
- **old:** the engine with the hyper-connection fixes, and the first staging library.
- **new:** branch hbm-prefetch at a65dbbcd: the FP4 decode, smaller staging slots, attention scores/output, indexer scores, the one-row compressor product.

**Outputs.** dsv4check on the random model gives the same digits as on the 4080 (11 of 24, 2.4437), with the FP4 layout and with five-bit experts (`GLYD_V4_FP4=0`). moecheck with STAGES=4: all within bounds.

**The experts' layer at one token** (`moecheck dsv4 --speed`, us):

| Layout | old | new | new, STAGES 4 | new, 4, 3 blocks |
|---|---|---|---|---|
| fp4, packed scales | 205.4 | **174.3** | 176.7 | 179.5 |
| fp4 | 186.8 | **166.1** | 166.5 | 180.8 |
| int4 | 161.7 | 161.7 | 163.5 | 189.5 |
| int6 | 192.2 | 192.4 | 195.1 | 233.0 |

int5 is left out: it is the first layout each run times, and it reads 171-210 us from the GPU's clocks rising.

Nsight, the FP4 gate and up product: 86.9 us (123.4 in the first staging session), 102 registers, issue slots 52% busy, DRAM at 40% of peak.

**DeepSeek's 3-layer step** (`dsv4prof`, ms):

| | old | new | new, STAGES 4 | new, 4, 3 blocks |
|---|---|---|---|---|
| with a graph | 3.201 | 2.925 | 2.707 | 2.968 |
| mixture of experts (no graph) | 1.239 | 1.120 | 1.168 | 1.115 |
| attention | 0.405 | 0.337 | 0.339 | 0.327 |
| compressors | 0.306 | 0.264 | 0.269 | 0.259 |

STAGES 4 is 7.5% faster with a graph, but slower alone and without a graph. Repeat on an H100 before making it a default.
