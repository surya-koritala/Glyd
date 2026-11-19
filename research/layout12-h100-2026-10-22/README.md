# Penguin's two lossless layouts on one H100: Kimi-K3 22% faster in the 12-bit one (2026-10-22)

Lambda gpu_1x_h100_sxm5 (us-south-3), 00:55-01:15 EDT, about $1.50. The build of #223 (depth mix and norm). `dsv4prof` with `--layout mma` (the
smallest lossless layout, tiered, today's default for the dynamic engine) and `--layout mma12` (the 12-bit one), twice each. `launch.sh`, `job.sh`,
`inner.sh`; `out/`.

| Step with a graph | mma (tiered) | mma12 (12-bit) |
|---|---|---|
| Kimi-K3, its first 2 real layers + head, experts on the GPU | 2.685 / 2.674 ms (19.33 GiB) | **2.102 / 2.102 ms** (19.83 GiB) |
| DeepSeek-V4-Flash widths, 3 layers (random) | 1.239 / 1.244 ms | 1.375 / 1.375 ms |

Kimi: the tiered layout's one-token product reads 0.9-1.5 TB/s on the H100 (`../fp4-one-h100-2026-10-22`), the 12-bit layout's 2.7 TB/s. Its
bf16 matrices take 8% more memory in it: about 9 GiB of Kimi-K3's 116 GiB on 8x 80 GB, some 600 of the expert cache's ~29,000 places.

DeepSeek is no comparison: its loader holds the FP8 matrices exact only with the tiered layout asked (deepseek_v4.rs `exact8`), so mma12 made them
bf16. Its head alone, bf16 either way: 0.348 ms tiered, 0.300 ms 12-bit.

So: Kimi-K3 on HBM GPUs (A100, H100, B200) holds its bf16 matrices in the 12-bit layout (split::fit, cli.rs kimi_cache); the 4080's tiered product is at
80-88% of its rate and keeps the smallest layout.
