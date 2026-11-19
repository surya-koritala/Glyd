# A100 80GB: the FP8 product's three slots and the experts' staging, each twice (2026-10-19)

Lambda gpu_1x_a100_sxm4 (an A100-SXM4-80GB this time, about 2 TB/s), about 40 minutes, about $1.35.
Branch hbm-prefetch at 628d6190. Outputs: dsv4check with both options gives 11 of 24, 2.4437; f8moecheck gives 0.475 bf16 steps either way.

| | default | GLYD_F8_SLOTS=3 | GLYD_GPU_STAGES=4 | both |
|---|---|---|---|---|
| FP8 experts at GLM's shape, one token | 175.1 / 175.2 us (770 GB/s) | 173.7 / 173.6 (776) | | |
| DeepSeek, 3 layers, with a graph | 2.815 / 2.816 ms | 2.837 / 2.838 | 2.843 / 2.844 | 2.870 / 2.868 |
| GLM (FP8), 4 layers, with a graph | 3.927 / 4.153 ms | 3.790 / 3.848 | | |

- **Neither option is worth making a default.**
  - Two steps of copies in flight barely move the FP8 experts (+0.8%), so that kernel is not bound by copies in flight.
  - At 38% of the A100's bandwidth it is bound by its decode (core, variable parts, heads), as FP4's was by its integer issue.
  - The staging's 7.5% on the A100 40GB did not repeat here (DeepSeek's step is 1% slower with it).
- **GLM's step varies 3-6% between runs** on this GPU; its numbers need more repeats.
- **Next:** the FP8 decode's instruction count. Or, on datacenter GPUs, the FP8 weights as released: 8 bits, no decode, the release's own W8A8 product on FP8 tensor cores.
