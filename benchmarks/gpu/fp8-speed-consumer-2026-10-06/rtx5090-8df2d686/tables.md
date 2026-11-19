# NVIDIA GeForce RTX 5090 (sm_120, L2 96 MiB)

vLLM's kernels: Mistral-Small-4-119B-2603: FlashInferFP8ScaledMMLinearKernel; gemma-4-26B-A4B-it-FP8-dynamic: CutlassFP8ScaledMMLinearKernel

Shapes x tokens left out of the sums: 92 of 140 (not stock's kernel: 76, L2: 16)

## Per-shape sums: a runner's times added over vLLM's added (each distinct shape once; dense: the expert shapes left out)

| model | tp | shapes | runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | ours | 1.19 | 1.31 | 1.99 | 2.35 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | decode + vLLM's kernel | 1.58 | 1.52 | 1.54 | 1.52 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | ours | 1.21 | 1.31 | 1.96 | 2.32 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | decode + vLLM's kernel | 1.60 | 1.53 | 1.55 | 1.55 |

## Per shape: geometric mean of a runner's time over vLLM's, all models (in brackets: shapes at most vLLM's; then the range)

| runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| ours | 1.19 (1/7; 0.97-1.35) | 1.33 (0/7; 1.17-1.55) | 2.05 (0/7; 1.76-2.34) | 2.42 (0/7; 1.76-2.69) |
| decode + vLLM's kernel | 1.57 (0/7; 1.41-1.68) | 1.51 (0/7; 1.41-1.64) | 1.53 (0/7; 1.41-1.65) | 1.51 (0/7; 1.31-1.64) |

## Per shape: ours over the decode route (in brackets: shapes where the route is faster; then the range)

| kind | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| channel | 0.76 (0/7; 0.66-0.86) | 0.88 (1/7; 0.72-1.01) | 1.34 (7/7; 1.07-1.64) | 1.60 (7/7; 1.14-2.05) |
| tensor | - | - | - | - |

## Every shape (microseconds)

| model | tensor | tp | N x K | M | vLLM | ours | decode + vLLM's kernel | decode alone | copies (MB) | left out |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 1 | 14.9 | 20.9 | 27.2 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 8 | 17.7 | 22.5 | 26.7 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 32 | 18.6 | 31.7 | 32.1 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 64 | 18.5 | 43.1 | 28.9 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 1 | 9.8 | 13.7 | 16.6 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 8 | 11.9 | 17.3 | 17.4 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 32 | 12.9 | 25.3 | 18.4 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 64 | 14.1 | 38.4 | 19.4 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 1 | 19.7 | 24.1 | 31.0 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 8 | 21.7 | 25.4 | 30.7 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 32 | 22.5 | 39.3 | 36.3 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 64 | 22.9 | 52.9 | 35.2 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 1 | 14.7 | 19.1 | 24.8 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 8 | 16.5 | 20.5 | 24.6 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 32 | 17.3 | 29.2 | 29.4 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 64 | 17.2 | 40.0 | 26.6 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 1 | 9.7 | 13.3 | 16.6 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 8 | 11.9 | 15.7 | 17.7 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 32 | 12.6 | 22.4 | 20.1 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 64 | 13.3 | 33.1 | 19.5 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 1 | 6.8 | 6.8 | 9.4 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 8 | 8.3 | 9.0 | 10.7 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 32 | 10.1 | 18.0 | 12.6 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 64 | 10.1 | 30.3 | 12.6 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 1 | 4.3 | 5.0 | 7.2 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 8 | 5.5 | 5.7 | 8.5 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 32 | 6.5 | 9.0 | 9.4 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 64 | 6.5 | 13.6 | 9.5 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 1 | 10.5 | 13.5 | 17.9 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 8 | 12.7 | 15.7 | 18.9 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 32 | 13.4 | 24.5 | 21.7 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 64 | 14.3 | 35.7 | 21.0 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 1 | 7.5 | 8.9 | 12.0 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 8 | 9.3 | 11.1 | 13.4 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 32 | 10.4 | 21.4 | 14.9 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 64 | 10.8 | 33.9 | 15.2 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 1 | 12.4 | 16.9 | 20.7 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 8 | 14.1 | 18.6 | 21.3 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 32 | 14.9 | 25.4 | 24.8 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 64 | 14.7 | 35.2 | 24.2 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 1 | 5.7 | 5.4 | 8.6 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 8 | 6.9 | 7.6 | 9.2 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 32 | 8.5 | 16.6 | 11.4 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 64 | 8.7 | 28.6 | 11.6 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 1 | 3.8 | 5.0 | 6.5 | - | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 8 | 4.9 | 6.1 | 7.6 | - | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 32 | 5.5 | 10.9 | 8.2 | - | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 64 | 5.6 | 17.6 | 8.3 | - | 34 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 1 | 51.2 | 52.5 | 87.5 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 8 | 52.6 | 53.4 | 85.7 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 32 | 53.6 | 87.3 | 92.3 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 64 | 54.5 | 114.2 | 92.9 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 1 | 24.1 | 26.4 | 36.3 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 8 | 26.3 | 28.3 | 37.1 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 32 | 27.8 | 44.2 | 40.8 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 64 | 30.2 | 59.5 | 42.2 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 1 | 57.4 | 58.8 | 107.7 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 8 | 58.8 | 60.0 | 106.4 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 32 | 60.1 | 97.9 | 114.4 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 64 | 60.8 | 127.8 | 114.0 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 1 | 36.9 | 39.9 | 55.4 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 8 | 39.0 | 41.6 | 59.2 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 32 | 40.4 | 64.7 | 62.1 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 64 | 42.5 | 85.3 | 62.9 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 1 | 24.1 | 26.8 | 36.7 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 8 | 26.0 | 28.5 | 39.1 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 32 | 27.3 | 44.2 | 42.6 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 64 | 29.2 | 59.5 | 44.2 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 1 | 115.1 | 111.8 | 297.3 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 8 | 117.7 | 113.7 | 301.0 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 32 | 120.2 | 192.6 | 305.9 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 64 | 121.7 | 252.7 | 307.4 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 1 | 59.5 | 59.7 | 118.8 | - | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 8 | 61.6 | 61.6 | 129.5 | - | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 32 | 63.5 | 101.8 | 136.3 | - | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 64 | 66.0 | 132.7 | 135.3 | - | 267 | not stock's kernel |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 1 | 19.5 | 23.8 | 30.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 8 | 19.8 | 26.6 | 29.2 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 32 | 19.8 | 36.5 | 29.1 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 64 | 20.9 | 49.4 | 32.1 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 1 | 21.0 | 24.4 | 33.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 8 | 21.8 | 25.9 | 31.8 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 32 | 21.5 | 39.4 | 31.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 64 | 22.6 | 53.1 | 34.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 1 | 13.1 | 16.5 | 21.3 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 8 | 13.8 | 19.1 | 22.1 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 32 | 13.5 | 31.6 | 22.1 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 64 | 18.2 | 48.6 | 27.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 1 | 21.1 | 23.5 | 35.4 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 8 | 22.5 | 26.4 | 36.8 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 32 | 22.3 | 39.4 | 36.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 64 | 31.2 | 55.0 | 48.3 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 1 | 12.8 | 16.8 | 19.7 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 8 | 13.3 | 19.0 | 19.8 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 32 | 13.3 | 27.5 | 19.9 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 64 | 14.7 | 39.4 | 24.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 1 | 9.4 | 12.7 | 14.8 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 8 | 9.8 | 15.2 | 15.1 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 32 | 9.5 | 21.4 | 15.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 64 | 12.3 | 32.2 | 18.5 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 1 | 9.4 | 9.1 | 13.4 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 8 | 9.9 | 12.6 | 13.9 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 32 | 9.9 | 22.8 | 13.9 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 64 | 13.8 | 37.2 | 18.1 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 1 | 6.3 | 6.7 | 8.8 | - | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 8 | 6.5 | 8.3 | 8.9 | - | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 32 | 6.4 | 14.9 | 9.0 | - | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 64 | 7.6 | 24.2 | 10.2 | - | 127 | L2 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 1 | - | 12.9 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 8 | - | 17.4 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 32 | - | 23.4 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 64 | - | 36.9 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 1 | - | 8.4 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 8 | - | 9.6 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 32 | - | 18.5 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 64 | - | 28.3 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 1 | - | 5.2 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 8 | - | 6.3 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 32 | - | 11.1 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 64 | - | 17.9 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 1 | - | 18.5 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 8 | - | 20.7 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 32 | - | 31.7 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 64 | - | 45.1 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 1 | - | 13.0 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 8 | - | 15.3 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 32 | - | 21.8 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 64 | - | 32.5 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 1 | - | 6.3 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 8 | - | 7.6 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 32 | - | 12.8 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 64 | - | 19.9 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 1 | - | 5.0 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 8 | - | 6.1 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 32 | - | 10.7 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 64 | - | 17.5 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 1 | - | 13.4 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 8 | - | 17.0 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 32 | - | 24.5 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 64 | - | 37.6 | - | - | 201 |  |

## Numerics: error against an exact float64 product of the dequantized weights (max: of the sum of |x w|; rms: of the outputs' rms)

| model | tensor | M | vLLM kernel | vLLM max | vLLM rms | ours max | ours rms |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 4.3e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 5.1e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 32 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 5.1e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 64 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 4.9e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 1 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 2.1e-04 | 1.6e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 8 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.0e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 32 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.7e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 64 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.3e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.9e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 8 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 4.9e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 32 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 5.0e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 64 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 6.4e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 1 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 2.5e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 8 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.5e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 32 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 4.5e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 64 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 1 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 2.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 8 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 2.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 32 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.4e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 64 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.1e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 2.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 8 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.5e-04 | 1.6e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 32 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.1e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 64 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 3.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 2.7e-03 | 2.6e-02 | 2.8e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | 5.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 3.9e-03 | 2.6e-02 | 3.8e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.6e-02 | 4.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | CutlassFP8ScaledMMLinearKernel | 3.0e-03 | 2.7e-02 | 2.7e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 8 | CutlassFP8ScaledMMLinearKernel | 3.8e-03 | 2.6e-02 | 3.9e-04 | 1.6e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 32 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | 4.0e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 64 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | 4.9e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.6e-02 | 2.4e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.7e-02 | 2.9e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.7e-02 | 3.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.2e-03 | 2.6e-02 | 3.2e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | FlashInferFP8ScaledMMLinearKernel | - | - | 2.5e-04 | 1.6e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 8 | FlashInferFP8ScaledMMLinearKernel | - | - | 3.4e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 32 | FlashInferFP8ScaledMMLinearKernel | - | - | 3.8e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 64 | FlashInferFP8ScaledMMLinearKernel | - | - | 3.7e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 1 | FlashInferFP8ScaledMMLinearKernel | - | - | 6.4e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 8 | FlashInferFP8ScaledMMLinearKernel | - | - | 7.0e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 32 | FlashInferFP8ScaledMMLinearKernel | - | - | 8.6e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 64 | FlashInferFP8ScaledMMLinearKernel | - | - | 9.2e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 32 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 64 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.6e-03 | 1.7e-03 |
