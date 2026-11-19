# NVIDIA GeForce RTX 5090 (sm_120, L2 96 MiB)

vLLM's kernels: Mistral-Small-4-119B-2603: FlashInferFP8ScaledMMLinearKernel; gemma-4-26B-A4B-it-FP8-dynamic: CutlassFP8ScaledMMLinearKernel

Shapes x tokens left out of the sums: 92 of 140 (not stock's kernel: 76, L2: 16)

## Per-shape sums: a runner's times added over vLLM's added (each distinct shape once; dense: the expert shapes left out)

| model | tp | shapes | runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | ours | 1.19 | 1.29 | 2.42 | 4.33 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | decode + vLLM's kernel | 1.61 | 1.54 | 1.55 | 1.53 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | ours | 1.22 | 1.31 | 2.42 | 4.43 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | decode + vLLM's kernel | 1.63 | 1.55 | 1.57 | 1.56 |

## Per shape: geometric mean of a runner's time over vLLM's, all models (in brackets: shapes at most vLLM's; then the range)

| runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| ours | 1.17 (1/7; 0.85-1.33) | 1.29 (0/7; 1.10-1.47) | 2.45 (0/7; 2.05-2.88) | 4.31 (0/7; 3.50-5.29) |
| decode + vLLM's kernel | 1.60 (0/7; 1.47-1.68) | 1.53 (0/7; 1.43-1.64) | 1.55 (0/7; 1.44-1.67) | 1.52 (0/7; 1.31-1.66) |

## Per shape: ours over the decode route (in brackets: shapes where the route is faster; then the range)

| kind | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| channel | 0.73 (0/7; 0.58-0.84) | 0.84 (0/7; 0.77-0.97) | 1.58 (7/7; 1.38-1.88) | 2.83 (7/7; 2.38-3.38) |
| tensor | - | - | - | - |

## Every shape (microseconds)

| model | tensor | tp | N x K | M | vLLM | ours | decode + vLLM's kernel | decode alone | copies (MB) | left out |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 1 | 14.9 | 19.3 | 26.3 | 15.6 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 8 | 16.6 | 20.8 | 25.7 | 15.6 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 32 | 18.1 | 35.1 | 32.1 | 15.6 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 64 | 17.5 | 85.6 | 27.9 | 15.6 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 1 | 8.9 | 12.0 | 15.6 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 8 | 10.8 | 13.5 | 17.3 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 32 | 11.9 | 23.7 | 18.8 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 64 | 12.8 | 51.9 | 18.9 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 1 | 18.7 | 28.3 | 30.3 | 21.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 8 | 20.5 | 30.2 | 31.5 | 21.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 32 | 21.7 | 54.5 | 35.8 | 21.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 64 | 21.8 | 117.8 | 34.7 | 21.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 1 | 13.7 | 17.5 | 24.1 | 14.2 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 8 | 15.4 | 20.1 | 23.8 | 14.2 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 32 | 16.6 | 36.7 | 29.4 | 14.2 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 64 | 16.1 | 84.8 | 25.9 | 14.2 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 1 | 9.1 | 11.9 | 15.7 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 8 | 10.8 | 13.3 | 16.6 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 32 | 11.7 | 22.9 | 18.9 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 64 | 12.3 | 50.7 | 18.6 | 8.8 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 1 | 5.7 | 5.6 | 8.9 | 3.8 | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 8 | 7.2 | 7.0 | 10.3 | 3.8 | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 32 | 8.7 | 14.3 | 12.0 | 3.8 | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 64 | 9.0 | 23.9 | 12.3 | 3.8 | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 1 | 3.8 | 4.6 | 6.5 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 8 | 5.3 | 5.7 | 8.0 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 32 | 6.2 | 12.5 | 9.0 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 64 | 6.2 | 21.7 | 8.9 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 1 | 9.5 | 12.5 | 17.0 | 9.5 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 8 | 11.7 | 13.8 | 17.9 | 9.5 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 32 | 12.5 | 23.5 | 20.0 | 9.5 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 64 | 13.4 | 51.7 | 20.0 | 9.5 | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 1 | 6.5 | 7.5 | 11.5 | 5.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 8 | 8.2 | 9.0 | 13.0 | 5.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 32 | 9.5 | 16.3 | 14.5 | 5.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 64 | 10.0 | 34.0 | 14.9 | 5.5 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 1 | 11.3 | 17.1 | 19.6 | 11.4 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 8 | 13.0 | 18.8 | 20.4 | 11.4 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 32 | 14.1 | 33.4 | 24.4 | 11.4 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 64 | 13.6 | 68.4 | 22.9 | 11.4 | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 1 | 5.1 | 4.7 | 8.1 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 8 | 6.6 | 6.0 | 9.4 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 32 | 8.1 | 13.3 | 11.0 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 64 | 8.4 | 23.0 | 11.2 | 2.7 | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 1 | 3.3 | 4.7 | 5.9 | 2.6 | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 8 | 4.6 | 6.0 | 7.3 | 2.7 | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 32 | 5.3 | 14.0 | 7.9 | 2.6 | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 64 | 5.3 | 24.7 | 7.9 | 2.6 | 34 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 1 | 49.9 | 60.5 | 90.5 | 86.2 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 8 | 51.5 | 64.2 | 84.0 | 86.3 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 32 | 52.5 | 120.2 | 91.7 | 86.4 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 64 | 52.6 | 300.5 | 92.4 | 86.3 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 1 | 23.2 | 26.3 | 35.3 | 26.9 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 8 | 25.4 | 28.3 | 36.0 | 26.8 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 32 | 26.8 | 47.3 | 38.8 | 26.7 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 64 | 29.0 | 121.5 | 41.1 | 26.8 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 1 | 56.7 | 79.8 | 107.5 | 102.2 | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 8 | 58.1 | 84.4 | 105.4 | 102.4 | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 32 | 59.3 | 147.8 | 112.5 | 102.4 | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 64 | 59.4 | 404.6 | 113.3 | 102.4 | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 1 | 35.9 | 49.4 | 55.4 | 63.7 | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 8 | 37.9 | 52.7 | 58.9 | 63.8 | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 32 | 39.1 | 97.8 | 61.3 | 63.7 | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 64 | 41.8 | 222.0 | 63.8 | 63.6 | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 1 | 23.2 | 32.4 | 36.3 | 27.0 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 8 | 25.0 | 34.8 | 39.1 | 26.7 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 32 | 26.1 | 63.9 | 42.1 | 26.6 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 64 | 28.0 | 142.0 | 43.3 | 26.6 | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 1 | 114.0 | 150.5 | 296.7 | 218.8 | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 8 | 116.7 | 164.4 | 299.5 | 218.8 | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 32 | 119.1 | 320.3 | 305.6 | 218.5 | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 64 | 120.8 | 862.1 | 306.9 | 218.7 | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 1 | 58.8 | 60.1 | 117.9 | 108.5 | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 8 | 60.5 | 64.2 | 128.5 | 109.0 | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 32 | 61.7 | 135.0 | 135.4 | 109.1 | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 64 | 63.3 | 411.7 | 131.5 | 108.9 | 267 | not stock's kernel |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 1 | 18.9 | 22.5 | 30.3 | 19.3 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 8 | 19.4 | 25.3 | 28.8 | 19.4 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 32 | 19.4 | 47.2 | 28.7 | 19.4 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 64 | 20.3 | 107.5 | 31.8 | 19.4 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 1 | 20.7 | 23.6 | 33.2 | 21.8 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 8 | 21.3 | 25.5 | 31.5 | 21.7 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 32 | 21.3 | 43.7 | 31.6 | 21.8 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 64 | 22.3 | 106.6 | 34.4 | 21.7 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 1 | 12.6 | 15.9 | 21.1 | 10.7 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 8 | 13.4 | 17.9 | 21.7 | 10.7 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 32 | 13.1 | 32.3 | 21.7 | 10.7 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 64 | 18.0 | 67.2 | 27.8 | 10.7 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 1 | 21.0 | 26.2 | 35.3 | 19.4 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 8 | 22.4 | 28.6 | 36.7 | 19.4 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 32 | 22.1 | 52.4 | 36.8 | 19.4 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 64 | 31.5 | 115.7 | 48.6 | 19.4 | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 1 | 12.3 | 16.4 | 19.4 | 10.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 8 | 12.9 | 18.9 | 19.5 | 10.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 32 | 12.8 | 36.8 | 19.6 | 10.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 64 | 14.4 | 75.6 | 23.9 | 10.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 1 | 8.9 | 10.9 | 14.4 | 7.1 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 8 | 9.5 | 12.8 | 14.8 | 7.0 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 32 | 9.3 | 24.2 | 14.9 | 7.1 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 64 | 12.0 | 52.1 | 18.2 | 7.0 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 1 | 9.0 | 7.7 | 13.2 | 4.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 8 | 9.5 | 10.5 | 13.6 | 4.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 32 | 9.5 | 23.2 | 13.7 | 4.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 64 | 13.6 | 47.8 | 17.9 | 4.9 | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 1 | 5.9 | 5.8 | 8.6 | 3.8 | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 8 | 6.0 | 7.8 | 8.7 | 3.8 | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 32 | 5.9 | 18.3 | 8.7 | 3.8 | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 64 | 7.1 | 32.5 | 10.1 | 3.8 | 127 | L2 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 1 | - | 10.4 | - | 6.0 | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 8 | - | 12.1 | - | 6.0 | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 32 | - | 22.3 | - | 6.0 | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 64 | - | 48.0 | - | 6.0 | 204 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 1 | - | 7.3 | - | 5.3 | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 8 | - | 8.5 | - | 5.3 | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 32 | - | 15.3 | - | 5.3 | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 64 | - | 32.8 | - | 5.3 | 201 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 1 | - | 4.6 | - | 3.1 | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 8 | - | 5.9 | - | 3.1 | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 32 | - | 13.9 | - | 3.1 | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 64 | - | 24.7 | - | 3.1 | 101 | L2 |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 1 | - | 18.9 | - | 13.7 | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 8 | - | 21.6 | - | 13.7 | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 32 | - | 38.2 | - | 13.7 | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 64 | - | 89.6 | - | 13.7 | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 1 | - | 11.5 | - | 8.4 | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 8 | - | 12.9 | - | 8.4 | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 32 | - | 22.4 | - | 8.4 | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 64 | - | 50.1 | - | 8.4 | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 1 | - | 5.3 | - | 3.7 | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 8 | - | 6.5 | - | 3.7 | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 32 | - | 13.3 | - | 3.7 | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 64 | - | 22.8 | - | 3.7 | 134 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 1 | - | 4.5 | - | 2.6 | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 8 | - | 5.9 | - | 2.6 | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 32 | - | 13.8 | - | 2.6 | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 64 | - | 24.7 | - | 2.6 | 50 | L2 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 1 | - | 11.6 | - | 8.5 | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 8 | - | 13.1 | - | 8.5 | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 32 | - | 22.9 | - | 8.5 | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 64 | - | 51.0 | - | 8.5 | 201 |  |

## Numerics: error against an exact float64 product of the dequantized weights (max: of the sum of |x w|; rms: of the outputs' rms)

| model | tensor | M | vLLM kernel | vLLM max | vLLM rms | ours max | ours rms |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | - | - |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | 5.1e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 32 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | - | - |
| Qwen3.6-35B-A3B-FP8 | qkv | 64 | DeepGemmFp8BlockScaledMMKernel | (not stock's) | (not stock's) | - | - |
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
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 2.7e-03 | 2.6e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 3.9e-03 | 2.6e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.6e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | CutlassFP8ScaledMMLinearKernel | 3.0e-03 | 2.7e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 8 | CutlassFP8ScaledMMLinearKernel | 3.8e-03 | 2.6e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 32 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 64 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.6e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.7e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.7e-02 | - | - |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.2e-03 | 2.6e-02 | - | - |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | FlashInferFP8ScaledMMLinearKernel | - | - | - | - |
| Mistral-Small-4-119B-2603 | qkv_a | 8 | FlashInferFP8ScaledMMLinearKernel | - | - | 3.4e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 32 | FlashInferFP8ScaledMMLinearKernel | - | - | - | - |
| Mistral-Small-4-119B-2603 | qkv_a | 64 | FlashInferFP8ScaledMMLinearKernel | - | - | - | - |
| Mistral-Small-4-119B-2603 | q_b | 1 | FlashInferFP8ScaledMMLinearKernel | - | - | 6.4e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 8 | FlashInferFP8ScaledMMLinearKernel | - | - | 7.0e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 32 | FlashInferFP8ScaledMMLinearKernel | - | - | 8.6e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 64 | FlashInferFP8ScaledMMLinearKernel | - | - | 9.2e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 32 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 64 | FlashInferFP8ScaledMMLinearKernel | - | - | 1.6e-03 | 1.7e-03 |
