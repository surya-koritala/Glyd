# NVIDIA GeForce RTX 5090 (sm_120, L2 96 MiB)

vLLM's kernels: Mistral-Small-4-119B-2603: FlashInferFP8ScaledMMLinearKernel; gemma-4-26B-A4B-it-FP8-dynamic: CutlassFP8ScaledMMLinearKernel

Shapes x tokens left out of the sums: 92 of 140 (not stock's kernel: 76, L2: 16)

## Per-shape sums: a runner's times added over vLLM's added (each distinct shape once; dense: the expert shapes left out)

| model | tp | shapes | runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | ours | 1.22 | 1.33 | 2.15 | 2.62 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | decode + vLLM's kernel | 1.60 | 1.52 | 1.54 | 1.54 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | ours | 1.24 | 1.33 | 2.11 | 2.59 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | decode + vLLM's kernel | 1.61 | 1.53 | 1.55 | 1.56 |

## Per shape: geometric mean of a runner's time over vLLM's, all models (in brackets: shapes at most vLLM's; then the range)

| runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| ours | 1.22 (0/7; 1.00-1.41) | 1.36 (0/7; 1.17-1.61) | 2.21 (0/7; 1.89-2.51) | 2.70 (0/7; 1.95-3.01) |
| decode + vLLM's kernel | 1.59 (0/7; 1.48-1.67) | 1.52 (0/7; 1.43-1.62) | 1.54 (0/7; 1.45-1.65) | 1.53 (0/7; 1.32-1.67) |

## Per shape: ours over the decode route (in brackets: shapes where the route is faster; then the range)

| kind | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| channel | 0.77 (0/7; 0.67-0.87) | 0.89 (1/7; 0.72-1.03) | 1.43 (7/7; 1.15-1.73) | 1.77 (7/7; 1.25-2.19) |
| tensor | - | - | - | - |

## Every shape (microseconds)

| model | tensor | tp | N x K | M | vLLM | ours | decode + vLLM's kernel | decode alone | copies (MB) | left out |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 1 | 14.7 | 20.4 | 25.3 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 8 | 16.3 | 22.1 | 24.8 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 32 | 17.1 | 33.0 | 30.3 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 64 | 17.1 | 46.3 | 26.8 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 1 | 8.8 | 13.3 | 15.2 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 8 | 10.5 | 16.9 | 16.7 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 32 | 11.6 | 26.2 | 17.7 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 64 | 12.8 | 40.6 | 18.8 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 1 | 18.5 | 23.8 | 29.5 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 8 | 20.2 | 24.9 | 30.5 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 32 | 21.0 | 41.5 | 34.5 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 64 | 21.3 | 57.7 | 33.1 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 1 | 13.5 | 18.6 | 23.3 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 8 | 15.1 | 19.8 | 22.9 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 32 | 15.7 | 30.7 | 27.9 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 64 | 15.9 | 43.7 | 24.9 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 1 | 8.7 | 12.9 | 15.2 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 8 | 10.4 | 15.2 | 16.0 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 32 | 11.2 | 22.6 | 17.9 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 64 | 12.2 | 34.1 | 18.1 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 1 | 5.7 | 6.4 | 8.9 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 8 | 6.8 | 8.5 | 10.0 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 32 | 8.8 | 17.8 | 11.9 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 64 | 8.7 | 30.5 | 11.9 | - | 134 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 1 | 3.6 | 4.8 | 6.2 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 8 | 5.0 | 5.6 | 7.7 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 32 | 5.9 | 9.1 | 8.6 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 64 | 5.8 | 14.0 | 8.5 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 1 | 9.4 | 13.1 | 16.3 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 8 | 11.3 | 15.4 | 17.1 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 32 | 12.0 | 25.1 | 19.9 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 64 | 13.1 | 37.6 | 19.3 | - | 208 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 1 | 6.3 | 8.5 | 11.5 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 8 | 7.8 | 10.7 | 12.7 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 32 | 9.1 | 21.4 | 14.1 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 64 | 9.5 | 34.5 | 14.4 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 1 | 11.1 | 16.5 | 19.1 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 8 | 12.6 | 18.2 | 19.7 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 32 | 13.4 | 26.4 | 23.3 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 64 | 13.3 | 37.8 | 22.0 | - | 201 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 1 | 5.2 | 5.3 | 7.9 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 8 | 6.3 | 7.5 | 8.9 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 32 | 8.1 | 16.7 | 10.8 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 64 | 8.2 | 29.1 | 10.9 | - | 67 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 1 | 3.2 | 4.9 | 5.8 | - | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 8 | 4.3 | 6.0 | 6.9 | - | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 32 | 4.9 | 11.0 | 7.4 | - | 34 | not stock's kernel |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 64 | 4.9 | 17.9 | 7.5 | - | 34 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 1 | 49.7 | 51.9 | 83.6 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 8 | 51.2 | 52.7 | 81.5 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 32 | 52.0 | 93.1 | 89.7 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 64 | 52.6 | 127.8 | 89.6 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 1 | 23.0 | 25.8 | 34.2 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 8 | 25.1 | 27.7 | 35.1 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 32 | 26.3 | 46.9 | 38.8 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 64 | 28.7 | 65.9 | 40.0 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 1 | 56.1 | 57.5 | 106.6 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 8 | 57.4 | 59.0 | 104.4 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 32 | 58.6 | 105.1 | 112.9 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 64 | 59.3 | 145.1 | 112.8 | - | 252 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 1 | 35.8 | 39.2 | 53.8 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 8 | 37.4 | 40.6 | 57.4 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 32 | 38.8 | 69.0 | 60.9 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 64 | 41.0 | 95.3 | 61.3 | - | 210 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 1 | 22.8 | 26.2 | 34.8 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 8 | 24.6 | 28.0 | 37.1 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 32 | 25.8 | 46.6 | 40.7 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 64 | 27.8 | 65.2 | 42.3 | - | 220 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 1 | 113.6 | 110.6 | 296.4 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 8 | 115.8 | 111.8 | 298.8 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 32 | 118.1 | 207.8 | 304.3 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 64 | 119.8 | 286.2 | 305.7 | - | 357 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 1 | 58.0 | 58.9 | 117.3 | - | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 8 | 60.1 | 60.8 | 128.2 | - | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 32 | 61.5 | 109.6 | 135.7 | - | 267 | not stock's kernel |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 64 | 64.5 | 149.5 | 132.6 | - | 267 | not stock's kernel |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 1 | 18.6 | 23.3 | 29.2 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 8 | 19.0 | 25.9 | 28.0 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 32 | 18.9 | 38.8 | 28.0 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 64 | 20.0 | 55.3 | 30.7 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 1 | 20.2 | 23.9 | 32.2 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 8 | 20.9 | 25.4 | 30.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 32 | 20.9 | 41.5 | 30.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 64 | 21.8 | 58.6 | 33.3 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 1 | 12.4 | 15.8 | 20.4 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 8 | 13.1 | 18.5 | 21.0 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 32 | 12.8 | 31.1 | 21.1 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 64 | 17.3 | 48.7 | 26.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 1 | 20.5 | 22.9 | 34.2 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 8 | 21.9 | 25.6 | 35.6 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 32 | 21.7 | 41.0 | 35.7 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 64 | 30.3 | 59.2 | 47.2 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 1 | 12.0 | 16.3 | 18.9 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 8 | 12.6 | 18.6 | 18.9 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 32 | 12.5 | 28.3 | 19.0 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 64 | 13.8 | 41.7 | 23.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 1 | 8.8 | 12.3 | 14.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 8 | 9.2 | 14.8 | 14.4 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 32 | 8.9 | 21.9 | 14.5 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 64 | 11.5 | 33.6 | 17.6 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 1 | 8.7 | 8.7 | 12.8 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 8 | 9.2 | 12.2 | 13.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 32 | 9.2 | 23.0 | 13.3 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 64 | 13.1 | 37.9 | 17.3 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 1 | 5.7 | 6.3 | 8.4 | - | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 8 | 5.8 | 7.8 | 8.5 | - | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 32 | 5.7 | 14.8 | 8.6 | - | 127 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 64 | 6.9 | 24.3 | 9.8 | - | 127 | L2 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 1 | - | 12.4 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 8 | - | 17.0 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 32 | - | 23.9 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 64 | - | 38.5 | - | - | 204 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 1 | - | 8.0 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 8 | - | 9.4 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 32 | - | 18.6 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 64 | - | 29.0 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 1 | - | 4.9 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 8 | - | 6.0 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 32 | - | 11.0 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 64 | - | 18.1 | - | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 1 | - | 18.1 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 8 | - | 20.2 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 32 | - | 33.4 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 64 | - | 49.0 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 1 | - | 12.6 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 8 | - | 14.8 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 32 | - | 22.2 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 64 | - | 33.6 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 1 | - | 5.9 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 8 | - | 7.1 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 32 | - | 12.5 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 64 | - | 20.1 | - | - | 134 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 1 | - | 4.8 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 8 | - | 5.9 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 32 | - | 10.9 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 64 | - | 17.9 | - | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 1 | - | 12.9 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 8 | - | 16.5 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 32 | - | 25.6 | - | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 64 | - | 39.9 | - | - | 201 |  |

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
