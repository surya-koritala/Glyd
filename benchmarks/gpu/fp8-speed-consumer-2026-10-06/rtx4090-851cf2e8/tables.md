# NVIDIA GeForce RTX 4090 (sm_89, L2 72 MiB)

vLLM's kernels: DeepSeek-V4.1-Flash: MarlinMxfp8LinearKernel; GLM-5.3-Flash: MarlinFP8ScaledMMLinearKernel; MiniMax-M3-MXFP8: MarlinMxfp8LinearKernel; Mistral-Small-4-119B-2603: CutlassFP8ScaledMMLinearKernel; Qwen3.6-35B-A3B-FP8: MarlinFP8ScaledMMLinearKernel; Qwen3.8-27B-FP8: MarlinFP8ScaledMMLinearKernel; gemma-4-26B-A4B-it-FP8-dynamic: CutlassFP8ScaledMMLinearKernel

Shapes x tokens left out of the sums: 72 of 472 (L2: 72)

## Per-shape sums: a runner's times added over vLLM's added (each distinct shape once; dense: the expert shapes left out)

| model | tp | shapes | runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | 1 | all | ours | 1.01 | 1.05 | 1.24 | 1.72 |
| GLM-5.3-Flash | 1 | all | decode + vLLM's kernel | 2.02 | 2.04 | 2.02 | 2.06 |
| GLM-5.3-Flash | 1 | dense | ours | 1.00 | 1.03 | 1.20 | 1.70 |
| GLM-5.3-Flash | 1 | dense | decode + vLLM's kernel | 2.10 | 2.12 | 2.08 | 2.11 |
| GLM-5.3-Flash | 2 | all | ours | 1.03 | 1.08 | 1.30 | 1.77 |
| GLM-5.3-Flash | 2 | all | decode + vLLM's kernel | 1.36 | 1.40 | 1.55 | 1.69 |
| GLM-5.3-Flash | 2 | dense | ours | 1.01 | 1.05 | 1.26 | 1.77 |
| GLM-5.3-Flash | 2 | dense | decode + vLLM's kernel | 1.36 | 1.39 | 1.55 | 1.70 |
| GLM-5.3-Flash | 4 | all | ours | 1.12 | 1.28 | 1.76 | 1.85 |
| GLM-5.3-Flash | 4 | all | decode + vLLM's kernel | 1.37 | 1.39 | 1.49 | 1.61 |
| GLM-5.3-Flash | 4 | dense | ours | 1.12 | 1.22 | 1.58 | 1.84 |
| GLM-5.3-Flash | 4 | dense | decode + vLLM's kernel | 1.30 | 1.32 | 1.47 | 1.66 |
| GLM-5.3-Flash | 8 | all | ours | 1.00 | 1.25 | 1.88 | 1.91 |
| GLM-5.3-Flash | 8 | all | decode + vLLM's kernel | 1.36 | 1.38 | 1.38 | 1.37 |
| GLM-5.3-Flash | 8 | dense | ours | 1.20 | 1.40 | 1.91 | 2.00 |
| GLM-5.3-Flash | 8 | dense | decode + vLLM's kernel | 1.44 | 1.47 | 1.48 | 1.54 |
| Qwen3.8-27B-FP8 | 1 | all | ours | 0.94 | 0.96 | 1.11 | 1.68 |
| Qwen3.8-27B-FP8 | 1 | all | decode + vLLM's kernel | 2.33 | 2.32 | 2.23 | 2.17 |
| MiniMax-M3-MXFP8 | 1 | all | ours | 0.95 | 0.99 | 1.47 | 1.80 |
| MiniMax-M3-MXFP8 | 1 | all | decode + vLLM's kernel | 2.06 | 2.07 | 2.05 | 2.12 |
| MiniMax-M3-MXFP8 | 1 | dense | ours | 0.93 | 0.97 | 1.46 | 1.80 |
| MiniMax-M3-MXFP8 | 1 | dense | decode + vLLM's kernel | 2.19 | 2.19 | 2.14 | 2.21 |
| MiniMax-M3-MXFP8 | 2 | all | ours | 1.05 | 1.10 | 1.54 | 1.85 |
| MiniMax-M3-MXFP8 | 2 | all | decode + vLLM's kernel | 1.70 | 1.71 | 1.76 | 1.85 |
| MiniMax-M3-MXFP8 | 2 | dense | ours | 1.00 | 1.03 | 1.50 | 1.82 |
| MiniMax-M3-MXFP8 | 2 | dense | decode + vLLM's kernel | 1.78 | 1.79 | 1.82 | 1.92 |
| MiniMax-M3-MXFP8 | 4 | all | ours | 1.29 | 1.47 | 1.78 | 1.99 |
| MiniMax-M3-MXFP8 | 4 | all | decode + vLLM's kernel | 1.36 | 1.37 | 1.50 | 1.54 |
| MiniMax-M3-MXFP8 | 4 | dense | ours | 1.24 | 1.38 | 1.68 | 1.89 |
| MiniMax-M3-MXFP8 | 4 | dense | decode + vLLM's kernel | 1.30 | 1.32 | 1.45 | 1.53 |
| MiniMax-M3-MXFP8 | 8 | all | ours | 1.22 | 1.57 | 2.10 | 2.20 |
| MiniMax-M3-MXFP8 | 8 | all | decode + vLLM's kernel | 1.51 | 1.52 | 1.58 | 1.47 |
| MiniMax-M3-MXFP8 | 8 | dense | ours | 1.24 | 1.52 | 1.95 | 2.08 |
| MiniMax-M3-MXFP8 | 8 | dense | decode + vLLM's kernel | 1.41 | 1.42 | 1.52 | 1.47 |
| Mistral-Small-4-119B-2603 | 1 | all | ours | 1.16 | 1.33 | 1.65 | 1.74 |
| Mistral-Small-4-119B-2603 | 1 | all | decode + vLLM's kernel | 1.48 | 1.48 | 1.44 | 1.50 |
| Mistral-Small-4-119B-2603 | 1 | dense | ours | 1.16 | 1.34 | 1.71 | 1.77 |
| Mistral-Small-4-119B-2603 | 1 | dense | decode + vLLM's kernel | 1.49 | 1.49 | 1.45 | 1.50 |
| Mistral-Small-4-119B-2603 | 2 | all | ours | 1.20 | 1.42 | 1.83 | 1.84 |
| Mistral-Small-4-119B-2603 | 2 | all | decode + vLLM's kernel | 1.60 | 1.61 | 1.52 | 1.49 |
| Mistral-Small-4-119B-2603 | 2 | dense | ours | 1.23 | 1.44 | 1.70 | 2.09 |
| Mistral-Small-4-119B-2603 | 2 | dense | decode + vLLM's kernel | 1.54 | 1.54 | 1.44 | 1.38 |
| Mistral-Small-4-119B-2603 | 4 | all | ours | 1.10 | 1.44 | 2.02 | 1.71 |
| Mistral-Small-4-119B-2603 | 4 | all | decode + vLLM's kernel | 1.57 | 1.57 | 1.46 | 1.35 |
| Mistral-Small-4-119B-2603 | 8 | all | ours | 0.83 | 1.25 | 2.41 | 2.63 |
| Mistral-Small-4-119B-2603 | 8 | all | decode + vLLM's kernel | 1.40 | 1.40 | 1.33 | 1.20 |
| DeepSeek-V4.1-Flash | 1 | all | ours | 1.04 | 1.12 | 1.65 | 1.96 |
| DeepSeek-V4.1-Flash | 1 | all | decode + vLLM's kernel | 1.49 | 1.51 | 1.65 | 1.78 |
| DeepSeek-V4.1-Flash | 2 | all | ours | 1.14 | 1.26 | 1.59 | 1.89 |
| DeepSeek-V4.1-Flash | 2 | all | decode + vLLM's kernel | 1.43 | 1.46 | 1.55 | 1.66 |
| DeepSeek-V4.1-Flash | 4 | all | ours | 1.25 | 1.43 | 1.68 | 1.98 |
| DeepSeek-V4.1-Flash | 4 | all | decode + vLLM's kernel | 1.50 | 1.52 | 1.56 | 1.57 |
| DeepSeek-V4.1-Flash | 8 | all | ours | 1.02 | 1.39 | 1.90 | 1.96 |
| DeepSeek-V4.1-Flash | 8 | all | decode + vLLM's kernel | 1.38 | 1.39 | 1.41 | 1.32 |
| Qwen3.6-35B-A3B-FP8 | 1 | all | ours | 1.13 | 1.22 | 1.51 | 1.76 |
| Qwen3.6-35B-A3B-FP8 | 1 | all | decode + vLLM's kernel | 1.30 | 1.32 | 1.37 | 1.48 |
| Qwen3.6-35B-A3B-FP8 | 1 | dense | ours | 1.15 | 1.23 | 1.47 | 1.75 |
| Qwen3.6-35B-A3B-FP8 | 1 | dense | decode + vLLM's kernel | 1.29 | 1.32 | 1.38 | 1.51 |
| Qwen3.6-35B-A3B-FP8 | 2 | all | ours | 1.27 | 1.43 | 1.97 | 1.95 |
| Qwen3.6-35B-A3B-FP8 | 2 | all | decode + vLLM's kernel | 1.40 | 1.41 | 1.52 | 1.59 |
| Qwen3.6-35B-A3B-FP8 | 2 | dense | ours | 1.27 | 1.43 | 1.97 | 1.95 |
| Qwen3.6-35B-A3B-FP8 | 2 | dense | decode + vLLM's kernel | 1.40 | 1.41 | 1.52 | 1.59 |
| Qwen3.6-35B-A3B-FP8 | 4 | all | ours | 1.47 | 1.75 | 2.45 | 2.12 |
| Qwen3.6-35B-A3B-FP8 | 4 | all | decode + vLLM's kernel | 1.53 | 1.55 | 1.56 | 1.49 |
| Qwen3.6-35B-A3B-FP8 | 4 | dense | ours | 1.47 | 1.75 | 2.45 | 2.12 |
| Qwen3.6-35B-A3B-FP8 | 4 | dense | decode + vLLM's kernel | 1.53 | 1.55 | 1.56 | 1.49 |
| Qwen3.6-35B-A3B-FP8 | 8 | all | ours | 1.15 | 1.51 | 2.26 | 2.39 |
| Qwen3.6-35B-A3B-FP8 | 8 | all | decode + vLLM's kernel | 1.45 | 1.47 | 1.42 | 1.35 |
| Qwen3.6-35B-A3B-FP8 | 8 | dense | ours | 1.15 | 1.51 | 2.26 | 2.39 |
| Qwen3.6-35B-A3B-FP8 | 8 | dense | decode + vLLM's kernel | 1.45 | 1.47 | 1.42 | 1.35 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | ours | 1.08 | 1.16 | 1.48 | 1.91 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | decode + vLLM's kernel | 1.38 | 1.37 | 1.43 | 1.54 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | ours | 1.07 | 1.13 | 1.38 | 1.87 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | decode + vLLM's kernel | 1.36 | 1.35 | 1.42 | 1.57 |

## Per shape: geometric mean of a runner's time over vLLM's, all models (in brackets: shapes at most vLLM's; then the range)

| runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| ours | 1.11 (22/100; 0.63-1.68) | 1.27 (12/100; 0.91-2.06) | 1.67 (0/100; 1.02-3.01) | 1.90 (0/100; 1.49-2.63) |
| decode + vLLM's kernel | 1.50 (0/100; 1.19-2.88) | 1.51 (0/100; 1.21-2.86) | 1.57 (0/100; 1.20-2.65) | 1.61 (0/100; 1.16-2.52) |

## Per shape: ours over the decode route (in brackets: shapes where the route is faster; then the range)

| kind | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| block | 0.73 (1/41; 0.31-1.04) | 0.80 (9/41; 0.32-1.27) | 1.01 (23/41; 0.39-1.90) | 1.12 (31/41; 0.67-1.97) |
| channel | 0.77 (0/8; 0.66-0.82) | 0.86 (0/8; 0.81-0.96) | 1.09 (5/8; 0.71-1.61) | 1.28 (8/8; 1.03-1.58) |
| ds32 | 0.77 (0/22; 0.49-0.98) | 0.89 (7/22; 0.48-1.14) | 1.10 (15/22; 0.69-1.69) | 1.23 (20/22; 0.76-1.87) |
| mx | 0.73 (0/20; 0.32-0.98) | 0.82 (7/20; 0.34-1.26) | 1.05 (14/20; 0.54-1.56) | 1.17 (15/20; 0.68-1.74) |
| tensor | 0.74 (0/9; 0.59-0.81) | 0.90 (1/9; 0.82-1.00) | 1.23 (7/9; 0.87-1.81) | 1.29 (9/9; 1.07-2.19) |

## Every shape (microseconds)

| model | tensor | tp | N x K | M | vLLM | ours | decode + vLLM's kernel | decode alone | copies (MB) | left out |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | qkv_a | 1 | 2048 x 4096 | 1 | 12.9 | 15.4 | 17.4 | - | 201 |  |
| GLM-5.3-Flash | qkv_a | 1 | 2048 x 4096 | 8 | 12.9 | 18.1 | 17.7 | - | 201 |  |
| GLM-5.3-Flash | qkv_a | 1 | 2048 x 4096 | 32 | 14.6 | 29.8 | 21.3 | - | 201 |  |
| GLM-5.3-Flash | qkv_a | 1 | 2048 x 4096 | 64 | 18.7 | 35.1 | 28.7 | - | 201 |  |
| GLM-5.3-Flash | q_b | 1 | 16384 x 1536 | 1 | 29.5 | 30.1 | 42.0 | - | 201 |  |
| GLM-5.3-Flash | q_b | 1 | 16384 x 1536 | 8 | 29.4 | 31.3 | 42.8 | - | 201 |  |
| GLM-5.3-Flash | q_b | 1 | 16384 x 1536 | 32 | 33.1 | 40.7 | 49.1 | - | 201 |  |
| GLM-5.3-Flash | q_b | 1 | 16384 x 1536 | 64 | 37.8 | 56.8 | 63.4 | - | 201 |  |
| GLM-5.3-Flash | o | 1 | 4096 x 16384 | 1 | 74.6 | 86.1 | 162.9 | - | 201 |  |
| GLM-5.3-Flash | o | 1 | 4096 x 16384 | 8 | 74.7 | 87.9 | 164.7 | - | 201 |  |
| GLM-5.3-Flash | o | 1 | 4096 x 16384 | 32 | 76.6 | 97.8 | 172.0 | - | 201 |  |
| GLM-5.3-Flash | o | 1 | 4096 x 16384 | 64 | 81.4 | 149.9 | 191.3 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 1 | 24576 x 4096 | 1 | 115.6 | 103.3 | 302.0 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 1 | 24576 x 4096 | 8 | 116.8 | 106.7 | 302.4 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 1 | 24576 x 4096 | 32 | 121.8 | 125.6 | 291.8 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 1 | 24576 x 4096 | 64 | 140.9 | 222.6 | 310.1 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 1 | 4096 x 12288 | 1 | 56.7 | 53.6 | 84.1 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 1 | 4096 x 12288 | 8 | 56.7 | 55.2 | 87.9 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 1 | 4096 x 12288 | 32 | 58.5 | 70.5 | 100.0 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 1 | 4096 x 12288 | 64 | 62.2 | 115.5 | 125.5 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 1 | 4096 x 4096 | 1 | 20.8 | 22.7 | 25.7 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 1 | 4096 x 4096 | 8 | 20.8 | 24.3 | 26.7 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 1 | 4096 x 4096 | 32 | 22.3 | 33.9 | 32.1 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 1 | 4096 x 4096 | 64 | 25.3 | 47.3 | 41.0 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 1 | 4096 x 2048 | 1 | 11.9 | 15.1 | 16.5 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 1 | 4096 x 2048 | 8 | 11.9 | 16.7 | 17.0 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 1 | 4096 x 2048 | 32 | 13.2 | 24.0 | 20.0 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 1 | 4096 x 2048 | 64 | 16.0 | 30.1 | 25.8 | - | 201 |  |
| GLM-5.3-Flash | q_b | 2 | 8192 x 1536 | 1 | 16.9 | 18.6 | 25.1 | - | 201 |  |
| GLM-5.3-Flash | q_b | 2 | 8192 x 1536 | 8 | 16.9 | 19.6 | 25.4 | - | 201 |  |
| GLM-5.3-Flash | q_b | 2 | 8192 x 1536 | 32 | 17.0 | 26.1 | 28.6 | - | 201 |  |
| GLM-5.3-Flash | q_b | 2 | 8192 x 1536 | 64 | 19.0 | 41.1 | 35.6 | - | 201 |  |
| GLM-5.3-Flash | o | 2 | 4096 x 8192 | 1 | 38.7 | 39.3 | 46.4 | - | 201 |  |
| GLM-5.3-Flash | o | 2 | 4096 x 8192 | 8 | 38.7 | 41.0 | 48.5 | - | 201 |  |
| GLM-5.3-Flash | o | 2 | 4096 x 8192 | 32 | 40.3 | 52.7 | 57.9 | - | 201 |  |
| GLM-5.3-Flash | o | 2 | 4096 x 8192 | 64 | 43.8 | 82.2 | 76.8 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 2 | 12288 x 4096 | 1 | 57.1 | 56.2 | 86.5 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 2 | 12288 x 4096 | 8 | 57.1 | 57.5 | 87.4 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 2 | 12288 x 4096 | 32 | 62.9 | 69.0 | 104.3 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 2 | 12288 x 4096 | 64 | 76.0 | 118.7 | 124.7 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 2 | 4096 x 6144 | 1 | 29.8 | 30.2 | 35.4 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 2 | 4096 x 6144 | 8 | 29.8 | 31.7 | 36.6 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 2 | 4096 x 6144 | 32 | 31.3 | 42.8 | 43.4 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 2 | 4096 x 6144 | 64 | 34.6 | 64.3 | 56.9 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 2 | 4096 x 1024 | 1 | 7.5 | 9.9 | 11.5 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 2 | 4096 x 1024 | 8 | 7.5 | 11.5 | 11.7 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 2 | 4096 x 1024 | 32 | 8.7 | 18.2 | 13.8 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 2 | 4096 x 1024 | 64 | 11.6 | 21.4 | 18.0 | - | 201 |  |
| GLM-5.3-Flash | q_b | 4 | 4096 x 1536 | 1 | 9.9 | 13.9 | 16.2 | - | 201 |  |
| GLM-5.3-Flash | q_b | 4 | 4096 x 1536 | 8 | 9.9 | 15.7 | 16.5 | - | 201 |  |
| GLM-5.3-Flash | q_b | 4 | 4096 x 1536 | 32 | 11.2 | 23.1 | 19.0 | - | 201 |  |
| GLM-5.3-Flash | q_b | 4 | 4096 x 1536 | 64 | 13.9 | 25.6 | 24.0 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 4 | 6144 x 4096 | 1 | 30.5 | 31.3 | 36.5 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 4 | 6144 x 4096 | 8 | 30.6 | 33.2 | 36.9 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 4 | 6144 x 4096 | 32 | 31.6 | 43.6 | 44.5 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 4 | 6144 x 4096 | 64 | 36.4 | 67.0 | 59.9 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 4 | 4096 x 3072 | 1 | 16.5 | 18.7 | 21.2 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 4 | 4096 x 3072 | 8 | 16.5 | 20.3 | 21.8 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 4 | 4096 x 3072 | 32 | 17.9 | 28.9 | 25.8 | - | 201 |  |
| GLM-5.3-Flash | mlp_down | 4 | 4096 x 3072 | 64 | 20.8 | 38.6 | 33.9 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 4 | 1024 x 4096 | 1 | 10.7 | 10.8 | 14.6 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 4 | 1024 x 4096 | 8 | 10.7 | 16.1 | 14.8 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 4 | 1024 x 4096 | 32 | 13.2 | 34.3 | 18.1 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 4 | 1024 x 4096 | 64 | 19.9 | 36.5 | 26.8 | - | 201 |  |
| GLM-5.3-Flash | expert_down | 4 | 4096 x 512 | 1 | 5.3 | 6.9 | 11.0 | - | 134 |  |
| GLM-5.3-Flash | expert_down | 4 | 4096 x 512 | 8 | 5.3 | 7.7 | 11.2 | - | 134 |  |
| GLM-5.3-Flash | expert_down | 4 | 4096 x 512 | 32 | 6.4 | 11.5 | 12.7 | - | 134 |  |
| GLM-5.3-Flash | expert_down | 4 | 4096 x 512 | 64 | 8.9 | 17.0 | 15.8 | - | 134 |  |
| GLM-5.3-Flash | q_b | 8 | 2048 x 1536 | 1 | 7.5 | 9.0 | 12.7 | - | 201 |  |
| GLM-5.3-Flash | q_b | 8 | 2048 x 1536 | 8 | 7.5 | 11.3 | 12.8 | - | 201 |  |
| GLM-5.3-Flash | q_b | 8 | 2048 x 1536 | 32 | 9.2 | 20.0 | 15.0 | - | 201 |  |
| GLM-5.3-Flash | q_b | 8 | 2048 x 1536 | 64 | 13.3 | 26.0 | 20.3 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 8 | 3072 x 4096 | 1 | 16.4 | 19.5 | 21.7 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 8 | 3072 x 4096 | 8 | 16.4 | 22.1 | 22.2 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 8 | 3072 x 4096 | 32 | 18.8 | 33.4 | 26.6 | - | 201 |  |
| GLM-5.3-Flash | mlp_gate_up | 8 | 3072 x 4096 | 64 | 23.0 | 46.4 | 35.5 | - | 201 |  |
| GLM-5.3-Flash | expert_gate_up | 8 | 512 x 4096 | 1 | 12.6 | 7.9 | 15.2 | - | 134 |  |
| GLM-5.3-Flash | expert_gate_up | 8 | 512 x 4096 | 8 | 12.6 | 12.1 | 15.3 | - | 134 |  |
| GLM-5.3-Flash | expert_gate_up | 8 | 512 x 4096 | 32 | 16.6 | 30.4 | 19.8 | - | 134 |  |
| GLM-5.3-Flash | expert_gate_up | 8 | 512 x 4096 | 64 | 28.6 | 51.6 | 33.1 | - | 134 |  |
| GLM-5.3-Flash | expert_down | 8 | 4096 x 256 | 1 | 3.1 | 5.1 | 10.1 | - | 67 | L2 |
| GLM-5.3-Flash | expert_down | 8 | 4096 x 256 | 8 | 3.2 | 6.2 | 10.2 | - | 67 | L2 |
| GLM-5.3-Flash | expert_down | 8 | 4096 x 256 | 32 | 4.3 | 11.1 | 11.3 | - | 67 | L2 |
| GLM-5.3-Flash | expert_down | 8 | 4096 x 256 | 64 | 6.7 | 17.7 | 13.9 | - | 67 | L2 |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 1 | 81.8 | 77.5 | 183.9 | - | 220 |  |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 8 | 82.2 | 79.1 | 184.4 | - | 220 |  |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 32 | 88.8 | 95.9 | 187.4 | - | 220 |  |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 64 | 101.2 | 165.3 | 209.5 | - | 220 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 1 | 37.2 | 38.8 | 45.9 | - | 220 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 8 | 37.2 | 40.7 | 46.8 | - | 220 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 32 | 38.5 | 53.2 | 55.7 | - | 220 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 64 | 43.8 | 77.6 | 73.9 | - | 220 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 1 | 92.3 | 84.3 | 235.5 | - | 252 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 8 | 93.0 | 86.4 | 235.9 | - | 252 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 32 | 100.4 | 105.7 | 227.2 | - | 252 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 64 | 111.5 | 176.5 | 235.3 | - | 252 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 1 | 59.2 | 58.3 | 89.7 | - | 210 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 8 | 59.2 | 59.5 | 90.4 | - | 210 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 32 | 64.2 | 72.9 | 109.0 | - | 210 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 64 | 78.6 | 122.8 | 133.9 | - | 210 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 1 | 37.2 | 40.2 | 44.9 | - | 220 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 8 | 37.2 | 42.0 | 45.7 | - | 220 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 32 | 38.4 | 54.0 | 55.7 | - | 220 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 64 | 43.7 | 80.1 | 75.6 | - | 220 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 1 | 194.5 | 173.2 | 559.6 | - | 357 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 8 | 196.0 | 179.4 | 560.4 | - | 357 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 32 | 210.4 | 215.0 | 557.5 | - | 357 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 64 | 228.1 | 384.5 | 575.6 | - | 357 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 1 | 98.9 | 92.7 | 238.2 | - | 267 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 8 | 99.0 | 95.4 | 239.1 | - | 267 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 32 | 102.1 | 117.8 | 243.0 | - | 267 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 64 | 111.5 | 197.3 | 257.9 | - | 267 |  |
| MiniMax-M3-MXFP8 | qkv | 1 | 9216 x 6144 | 1 | 64.7 | 64.3 | 104.8 | - | 226 |  |
| MiniMax-M3-MXFP8 | qkv | 1 | 9216 x 6144 | 8 | 64.6 | 65.8 | 105.9 | - | 226 |  |
| MiniMax-M3-MXFP8 | qkv | 1 | 9216 x 6144 | 32 | 71.7 | 101.7 | 119.6 | - | 226 |  |
| MiniMax-M3-MXFP8 | qkv | 1 | 9216 x 6144 | 64 | 71.8 | 135.4 | 151.6 | - | 226 |  |
| MiniMax-M3-MXFP8 | o | 1 | 6144 x 8192 | 1 | 57.9 | 58.5 | 82.5 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 1 | 6144 x 8192 | 8 | 57.9 | 59.4 | 83.0 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 1 | 6144 x 8192 | 32 | 60.1 | 91.8 | 98.5 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 1 | 6144 x 8192 | 64 | 66.1 | 121.9 | 122.2 | - | 201 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | 24576 x 6144 | 1 | 174.2 | 154.9 | 476.9 | - | 302 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | 24576 x 6144 | 8 | 175.5 | 161.3 | 477.3 | - | 302 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | 24576 x 6144 | 32 | 182.1 | 253.9 | 466.7 | - | 302 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | 24576 x 6144 | 64 | 196.0 | 337.8 | 494.6 | - | 302 |  |
| MiniMax-M3-MXFP8 | mlp_down | 1 | 6144 x 12288 | 1 | 85.8 | 80.9 | 188.6 | - | 226 |  |
| MiniMax-M3-MXFP8 | mlp_down | 1 | 6144 x 12288 | 8 | 85.7 | 84.0 | 189.0 | - | 226 |  |
| MiniMax-M3-MXFP8 | mlp_down | 1 | 6144 x 12288 | 32 | 88.9 | 132.0 | 193.8 | - | 226 |  |
| MiniMax-M3-MXFP8 | mlp_down | 1 | 6144 x 12288 | 64 | 96.6 | 175.1 | 211.4 | - | 226 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 1 | 6144 x 6144 | 1 | 44.4 | 44.6 | 60.5 | - | 226 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 1 | 6144 x 6144 | 8 | 44.4 | 46.4 | 61.4 | - | 226 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 1 | 6144 x 6144 | 32 | 46.1 | 70.9 | 73.6 | - | 226 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 1 | 6144 x 6144 | 64 | 52.5 | 95.3 | 91.7 | - | 226 |  |
| MiniMax-M3-MXFP8 | expert_down | 1 | 6144 x 3072 | 1 | 24.0 | 27.8 | 29.6 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 1 | 6144 x 3072 | 8 | 24.0 | 29.8 | 30.0 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 1 | 6144 x 3072 | 32 | 25.3 | 40.4 | 35.9 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 1 | 6144 x 3072 | 64 | 30.5 | 55.6 | 46.1 | - | 208 |  |
| MiniMax-M3-MXFP8 | index_q | 1 | 512 x 6144 | 1 | 15.2 | 10.5 | 18.9 | - | 201 |  |
| MiniMax-M3-MXFP8 | index_q | 1 | 512 x 6144 | 8 | 15.2 | 17.7 | 18.9 | - | 201 |  |
| MiniMax-M3-MXFP8 | index_q | 1 | 512 x 6144 | 32 | 17.9 | 33.7 | 22.5 | - | 201 |  |
| MiniMax-M3-MXFP8 | index_q | 1 | 512 x 6144 | 64 | 29.5 | 56.7 | 36.1 | - | 201 |  |
| MiniMax-M3-MXFP8 | qkv | 2 | 4608 x 6144 | 1 | 34.0 | 36.8 | 40.6 | - | 226 |  |
| MiniMax-M3-MXFP8 | qkv | 2 | 4608 x 6144 | 8 | 34.0 | 38.2 | 41.1 | - | 226 |  |
| MiniMax-M3-MXFP8 | qkv | 2 | 4608 x 6144 | 32 | 35.5 | 57.5 | 49.2 | - | 226 |  |
| MiniMax-M3-MXFP8 | qkv | 2 | 4608 x 6144 | 64 | 41.9 | 78.5 | 65.6 | - | 226 |  |
| MiniMax-M3-MXFP8 | o | 2 | 6144 x 4096 | 1 | 30.9 | 32.7 | 36.7 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 2 | 6144 x 4096 | 8 | 30.9 | 34.2 | 37.4 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 2 | 6144 x 4096 | 32 | 32.1 | 51.3 | 45.1 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 2 | 6144 x 4096 | 64 | 37.8 | 69.2 | 58.3 | - | 201 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 2 | 12288 x 6144 | 1 | 85.9 | 81.2 | 191.0 | - | 226 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 2 | 12288 x 6144 | 8 | 86.3 | 83.5 | 191.5 | - | 226 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 2 | 12288 x 6144 | 32 | 93.1 | 131.8 | 198.1 | - | 226 |  |
| MiniMax-M3-MXFP8 | mlp_gate_up | 2 | 12288 x 6144 | 64 | 97.3 | 175.0 | 216.2 | - | 226 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 2 | 3072 x 6144 | 1 | 23.6 | 27.7 | 29.3 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 2 | 3072 x 6144 | 8 | 23.6 | 30.0 | 29.9 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 2 | 3072 x 6144 | 32 | 26.0 | 42.9 | 35.7 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 2 | 3072 x 6144 | 64 | 31.4 | 60.5 | 47.2 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 2 | 6144 x 1536 | 1 | 13.5 | 18.6 | 21.7 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 2 | 6144 x 1536 | 8 | 13.5 | 20.4 | 21.9 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 2 | 6144 x 1536 | 32 | 14.6 | 27.1 | 25.5 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 2 | 6144 x 1536 | 64 | 18.7 | 37.9 | 32.1 | - | 208 |  |
| MiniMax-M3-MXFP8 | qkv | 4 | 2304 x 6144 | 1 | 19.1 | 24.5 | 24.8 | - | 212 |  |
| MiniMax-M3-MXFP8 | qkv | 4 | 2304 x 6144 | 8 | 19.1 | 27.8 | 25.1 | - | 212 |  |
| MiniMax-M3-MXFP8 | qkv | 4 | 2304 x 6144 | 32 | 21.0 | 35.6 | 30.0 | - | 212 |  |
| MiniMax-M3-MXFP8 | qkv | 4 | 2304 x 6144 | 64 | 26.1 | 50.4 | 40.3 | - | 212 |  |
| MiniMax-M3-MXFP8 | o | 4 | 6144 x 2048 | 1 | 17.1 | 20.3 | 22.2 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 4 | 6144 x 2048 | 8 | 17.1 | 22.2 | 22.5 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 4 | 6144 x 2048 | 32 | 18.4 | 30.5 | 27.0 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 4 | 6144 x 2048 | 64 | 23.0 | 42.4 | 34.9 | - | 201 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 4 | 1536 x 6144 | 1 | 15.4 | 19.2 | 20.2 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 4 | 1536 x 6144 | 8 | 15.4 | 23.2 | 20.4 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 4 | 1536 x 6144 | 32 | 17.0 | 31.9 | 24.1 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 4 | 1536 x 6144 | 64 | 21.9 | 47.6 | 33.6 | - | 208 |  |
| MiniMax-M3-MXFP8 | expert_down | 4 | 6144 x 768 | 1 | 8.6 | 13.7 | 14.5 | - | 203 |  |
| MiniMax-M3-MXFP8 | expert_down | 4 | 6144 x 768 | 8 | 8.6 | 15.1 | 14.7 | - | 203 |  |
| MiniMax-M3-MXFP8 | expert_down | 4 | 6144 x 768 | 32 | 9.6 | 19.8 | 18.2 | - | 203 |  |
| MiniMax-M3-MXFP8 | expert_down | 4 | 6144 x 768 | 64 | 13.8 | 28.5 | 22.1 | - | 203 |  |
| MiniMax-M3-MXFP8 | qkv | 8 | 1152 x 6144 | 1 | 13.9 | 15.5 | 18.9 | - | 205 |  |
| MiniMax-M3-MXFP8 | qkv | 8 | 1152 x 6144 | 8 | 13.9 | 20.2 | 19.0 | - | 205 |  |
| MiniMax-M3-MXFP8 | qkv | 8 | 1152 x 6144 | 32 | 15.6 | 31.6 | 22.3 | - | 205 |  |
| MiniMax-M3-MXFP8 | qkv | 8 | 1152 x 6144 | 64 | 21.9 | 48.5 | 31.3 | - | 205 |  |
| MiniMax-M3-MXFP8 | o | 8 | 6144 x 1024 | 1 | 10.2 | 14.5 | 15.1 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 8 | 6144 x 1024 | 8 | 10.2 | 16.3 | 15.2 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 8 | 6144 x 1024 | 32 | 11.1 | 20.5 | 18.2 | - | 201 |  |
| MiniMax-M3-MXFP8 | o | 8 | 6144 x 1024 | 64 | 15.4 | 29.2 | 23.7 | - | 201 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 8 | 768 x 6144 | 1 | 13.3 | 15.2 | 17.9 | - | 203 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 8 | 768 x 6144 | 8 | 13.3 | 22.6 | 18.0 | - | 203 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 8 | 768 x 6144 | 32 | 15.3 | 32.9 | 21.1 | - | 203 |  |
| MiniMax-M3-MXFP8 | expert_gate_up | 8 | 768 x 6144 | 64 | 23.7 | 53.5 | 30.8 | - | 203 |  |
| MiniMax-M3-MXFP8 | expert_down | 8 | 6144 x 384 | 1 | 6.4 | 8.2 | 14.1 | - | 151 |  |
| MiniMax-M3-MXFP8 | expert_down | 8 | 6144 x 384 | 8 | 6.4 | 9.4 | 14.2 | - | 151 |  |
| MiniMax-M3-MXFP8 | expert_down | 8 | 6144 x 384 | 32 | 6.1 | 16.0 | 14.4 | - | 151 |  |
| MiniMax-M3-MXFP8 | expert_down | 8 | 6144 x 384 | 64 | 9.4 | 23.7 | 17.6 | - | 151 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 1 | 10.3 | 14.1 | 18.1 | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 8 | 10.4 | 18.2 | 18.2 | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 32 | 13.2 | 34.9 | 21.5 | - | 204 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 64 | 20.2 | 39.4 | 28.9 | - | 204 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 1 | 8.2 | 9.8 | 12.6 | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 8 | 8.2 | 11.4 | 12.7 | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 32 | 11.9 | 18.3 | 17.4 | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 64 | 13.5 | 21.6 | 20.3 | - | 201 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 1 | 5.3 | 6.5 | 8.2 | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 8 | 5.4 | 7.5 | 8.2 | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 32 | 10.8 | 12.2 | 14.1 | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 64 | 11.8 | 18.9 | 15.4 | - | 101 | L2 |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 1 | 21.5 | 22.5 | 29.0 | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 8 | 21.6 | 24.1 | 29.2 | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 32 | 25.5 | 33.5 | 34.6 | - | 201 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 64 | 27.5 | 47.2 | 42.5 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 1 | 12.6 | 14.9 | 18.3 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 8 | 12.7 | 16.5 | 18.4 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 32 | 16.5 | 24.0 | 23.3 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 64 | 18.2 | 30.3 | 27.7 | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 1 | 5.8 | 7.1 | 8.9 | - | 134 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 8 | 5.8 | 8.3 | 8.9 | - | 134 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 32 | 8.4 | 14.2 | 12.0 | - | 134 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 64 | 10.7 | 22.3 | 14.7 | - | 134 |  |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 1 | 3.5 | 5.1 | 6.3 | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 8 | 3.5 | 6.1 | 6.4 | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 32 | 7.0 | 11.0 | 10.0 | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 64 | 8.3 | 17.5 | 11.3 | - | 50 | L2 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 1 | 12.7 | 15.1 | 20.7 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 8 | 12.7 | 17.9 | 20.9 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 32 | 15.4 | 29.3 | 24.2 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 64 | 20.6 | 35.2 | 31.8 | - | 201 |  |
| Mistral-Small-4-119B-2603 | q_b | 4 | 1024 x 1024 | 1 | 4.6 | 5.2 | 7.4 | - | 67 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 4 | 1024 x 1024 | 8 | 4.7 | 6.5 | 7.5 | - | 67 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 4 | 1024 x 1024 | 32 | 7.5 | 12.4 | 10.5 | - | 67 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 4 | 1024 x 1024 | 64 | 10.1 | 20.2 | 13.1 | - | 67 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 4 | 1536 x 256 | 1 | 3.3 | 5.0 | 6.1 | - | 25 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 4 | 1536 x 256 | 8 | 3.3 | 6.0 | 6.0 | - | 25 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 4 | 1536 x 256 | 32 | 5.9 | 10.8 | 8.7 | - | 25 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 4 | 1536 x 256 | 64 | 7.2 | 17.4 | 9.9 | - | 25 | L2 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 4 | 1024 x 4096 | 1 | 9.9 | 10.7 | 16.6 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 4 | 1024 x 4096 | 8 | 10.0 | 15.6 | 16.7 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 4 | 1024 x 4096 | 32 | 12.8 | 34.3 | 19.9 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 4 | 1024 x 4096 | 64 | 20.0 | 36.7 | 27.3 | - | 201 |  |
| Mistral-Small-4-119B-2603 | expert_down | 4 | 4096 x 512 | 1 | 6.2 | 6.9 | 8.6 | - | 134 |  |
| Mistral-Small-4-119B-2603 | expert_down | 4 | 4096 x 512 | 8 | 6.2 | 7.7 | 8.6 | - | 134 |  |
| Mistral-Small-4-119B-2603 | expert_down | 4 | 4096 x 512 | 32 | 9.8 | 11.5 | 13.2 | - | 134 |  |
| Mistral-Small-4-119B-2603 | expert_down | 4 | 4096 x 512 | 64 | 11.5 | 17.2 | 15.3 | - | 134 |  |
| Mistral-Small-4-119B-2603 | q_b | 8 | 512 x 1024 | 1 | 4.4 | 5.1 | 7.2 | - | 34 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 8 | 512 x 1024 | 8 | 4.5 | 6.4 | 7.3 | - | 34 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 8 | 512 x 1024 | 32 | 7.4 | 12.3 | 10.2 | - | 34 | L2 |
| Mistral-Small-4-119B-2603 | q_b | 8 | 512 x 1024 | 64 | 9.8 | 19.9 | 12.6 | - | 34 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | 768 x 256 | 1 | 3.3 | 5.0 | 6.0 | - | 13 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | 768 x 256 | 8 | 3.3 | 6.0 | 5.9 | - | 13 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | 768 x 256 | 32 | 5.9 | 10.8 | 8.6 | - | 13 | L2 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | 768 x 256 | 64 | 7.1 | 17.4 | 9.7 | - | 13 | L2 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 8 | 512 x 4096 | 1 | 9.6 | 7.9 | 13.4 | - | 134 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 8 | 512 x 4096 | 8 | 9.7 | 12.1 | 13.5 | - | 134 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 8 | 512 x 4096 | 32 | 12.6 | 30.2 | 16.7 | - | 134 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 8 | 512 x 4096 | 64 | 19.9 | 52.3 | 23.9 | - | 134 |  |
| Mistral-Small-4-119B-2603 | expert_down | 8 | 4096 x 256 | 1 | 3.8 | 5.1 | 6.6 | - | 67 | L2 |
| Mistral-Small-4-119B-2603 | expert_down | 8 | 4096 x 256 | 8 | 3.8 | 6.2 | 6.7 | - | 67 | L2 |
| Mistral-Small-4-119B-2603 | expert_down | 8 | 4096 x 256 | 32 | 8.2 | 11.0 | 11.2 | - | 67 | L2 |
| Mistral-Small-4-119B-2603 | expert_down | 8 | 4096 x 256 | 64 | 9.4 | 17.7 | 12.5 | - | 67 | L2 |
| DeepSeek-V4.1-Flash | wq_a | 1 | 1280 x 5120 | 1 | 12.7 | 15.2 | 17.7 | - | 203 |  |
| DeepSeek-V4.1-Flash | wq_a | 1 | 1280 x 5120 | 8 | 12.7 | 19.9 | 17.8 | - | 203 |  |
| DeepSeek-V4.1-Flash | wq_a | 1 | 1280 x 5120 | 32 | 14.2 | 30.9 | 20.9 | - | 203 |  |
| DeepSeek-V4.1-Flash | wq_a | 1 | 1280 x 5120 | 64 | 20.2 | 48.3 | 29.1 | - | 203 |  |
| DeepSeek-V4.1-Flash | wkv | 1 | 512 x 5120 | 1 | 13.5 | 9.8 | 16.7 | - | 168 |  |
| DeepSeek-V4.1-Flash | wkv | 1 | 512 x 5120 | 8 | 13.5 | 15.6 | 16.8 | - | 168 |  |
| DeepSeek-V4.1-Flash | wkv | 1 | 512 x 5120 | 32 | 15.9 | 33.5 | 19.8 | - | 168 |  |
| DeepSeek-V4.1-Flash | wkv | 1 | 512 x 5120 | 64 | 25.8 | 57.3 | 30.6 | - | 168 |  |
| DeepSeek-V4.1-Flash | wq_b | 1 | 32768 x 1280 | 1 | 47.9 | 46.3 | 95.1 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 1 | 32768 x 1280 | 8 | 48.0 | 46.5 | 96.8 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 1 | 32768 x 1280 | 32 | 50.5 | 72.7 | 105.7 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 1 | 32768 x 1280 | 64 | 54.8 | 97.0 | 128.0 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_a | 1 | 8192 x 4096 | 1 | 39.8 | 40.7 | 48.1 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 1 | 8192 x 4096 | 8 | 39.8 | 41.9 | 48.9 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 1 | 8192 x 4096 | 32 | 40.1 | 65.3 | 58.6 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 1 | 8192 x 4096 | 64 | 43.8 | 88.5 | 78.0 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_b | 1 | 5120 x 8192 | 1 | 48.9 | 48.3 | 69.6 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 1 | 5120 x 8192 | 8 | 49.0 | 50.2 | 70.5 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 1 | 5120 x 8192 | 32 | 50.7 | 76.8 | 82.7 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 1 | 5120 x 8192 | 64 | 56.8 | 103.8 | 101.9 | - | 210 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 1 | 4096 x 1280 | 1 | 8.9 | 14.0 | 16.1 | - | 204 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 1 | 4096 x 1280 | 8 | 8.9 | 15.7 | 16.3 | - | 204 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 1 | 4096 x 1280 | 32 | 10.2 | 18.8 | 18.5 | - | 204 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 1 | 4096 x 1280 | 64 | 12.9 | 26.3 | 23.3 | - | 204 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 1 | 4608 x 5120 | 1 | 29.1 | 31.7 | 35.7 | - | 212 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 1 | 4608 x 5120 | 8 | 29.1 | 33.7 | 36.2 | - | 212 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 1 | 4608 x 5120 | 32 | 30.4 | 51.0 | 43.1 | - | 212 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 1 | 4608 x 5120 | 64 | 35.9 | 70.4 | 56.3 | - | 212 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 1 | 5120 x 2304 | 1 | 16.9 | 19.7 | 24.7 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 1 | 5120 x 2304 | 8 | 16.9 | 21.5 | 24.9 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 1 | 5120 x 2304 | 32 | 17.5 | 29.5 | 29.6 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 1 | 5120 x 2304 | 64 | 21.3 | 40.6 | 36.0 | - | 201 |  |
| DeepSeek-V4.1-Flash | wq_b | 2 | 16384 x 1280 | 1 | 25.5 | 27.4 | 40.6 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 2 | 16384 x 1280 | 8 | 25.5 | 28.5 | 41.3 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 2 | 16384 x 1280 | 32 | 29.3 | 38.3 | 47.3 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 2 | 16384 x 1280 | 64 | 28.3 | 50.4 | 55.7 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_a | 2 | 4096 x 4096 | 1 | 21.3 | 23.6 | 26.6 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 2 | 4096 x 4096 | 8 | 21.3 | 25.1 | 27.5 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 2 | 4096 x 4096 | 32 | 22.7 | 36.3 | 32.4 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 2 | 4096 x 4096 | 64 | 25.6 | 49.4 | 41.7 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_b | 2 | 5120 x 4096 | 1 | 26.2 | 28.3 | 32.7 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 2 | 5120 x 4096 | 8 | 26.2 | 30.2 | 33.1 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 2 | 5120 x 4096 | 32 | 27.6 | 43.2 | 39.2 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 2 | 5120 x 4096 | 64 | 33.6 | 58.5 | 50.8 | - | 210 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 2 | 2048 x 1280 | 1 | 7.1 | 8.9 | 12.8 | - | 168 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 2 | 2048 x 1280 | 8 | 7.1 | 11.1 | 12.9 | - | 168 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 2 | 2048 x 1280 | 32 | 8.5 | 17.5 | 14.7 | - | 168 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 2 | 2048 x 1280 | 64 | 12.6 | 26.6 | 19.3 | - | 168 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 2 | 2304 x 5120 | 1 | 16.5 | 19.8 | 22.1 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 2 | 2304 x 5120 | 8 | 16.5 | 22.9 | 22.3 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 2 | 2304 x 5120 | 32 | 18.5 | 32.1 | 26.8 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 2 | 2304 x 5120 | 64 | 22.7 | 46.3 | 35.4 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 2 | 5120 x 1152 | 1 | 10.4 | 14.5 | 18.5 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 2 | 5120 x 1152 | 8 | 10.4 | 16.3 | 18.6 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 2 | 5120 x 1152 | 32 | 10.9 | 19.8 | 21.8 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 2 | 5120 x 1152 | 64 | 14.7 | 28.4 | 25.6 | - | 201 |  |
| DeepSeek-V4.1-Flash | wq_b | 4 | 8192 x 1280 | 1 | 14.8 | 18.9 | 24.5 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 4 | 8192 x 1280 | 8 | 14.8 | 19.9 | 24.8 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 4 | 8192 x 1280 | 32 | 15.1 | 26.9 | 27.9 | - | 210 |  |
| DeepSeek-V4.1-Flash | wq_b | 4 | 8192 x 1280 | 64 | 18.6 | 38.0 | 33.7 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_a | 4 | 2048 x 4096 | 1 | 13.5 | 15.9 | 18.3 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 4 | 2048 x 4096 | 8 | 13.5 | 18.6 | 18.6 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 4 | 2048 x 4096 | 32 | 14.9 | 25.0 | 21.8 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 4 | 2048 x 4096 | 64 | 18.9 | 36.3 | 29.2 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_b | 4 | 5120 x 2048 | 1 | 14.9 | 19.4 | 20.6 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 4 | 5120 x 2048 | 8 | 14.9 | 21.1 | 20.8 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 4 | 5120 x 2048 | 32 | 16.2 | 26.2 | 24.7 | - | 210 |  |
| DeepSeek-V4.1-Flash | wo_b | 4 | 5120 x 2048 | 64 | 20.9 | 36.1 | 32.0 | - | 210 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 4 | 1024 x 1280 | 1 | 6.5 | 5.7 | 9.4 | - | 84 | L2 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 4 | 1024 x 1280 | 8 | 6.4 | 7.7 | 9.4 | - | 84 | L2 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 4 | 1024 x 1280 | 32 | 7.8 | 16.7 | 11.1 | - | 84 | L2 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 4 | 1024 x 1280 | 64 | 12.0 | 28.2 | 15.5 | - | 84 | L2 |
| DeepSeek-V4.1-Flash | shared_w13 | 4 | 1152 x 5120 | 1 | 12.5 | 15.1 | 17.4 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 4 | 1152 x 5120 | 8 | 12.5 | 19.9 | 17.5 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 4 | 1152 x 5120 | 32 | 14.4 | 28.1 | 20.6 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 4 | 1152 x 5120 | 64 | 20.6 | 44.6 | 28.9 | - | 201 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 4 | 5120 x 576 | 1 | 7.1 | 9.2 | 13.9 | - | 189 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 4 | 5120 x 576 | 8 | 7.1 | 10.6 | 13.9 | - | 189 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 4 | 5120 x 576 | 32 | 12.5 | 16.4 | 19.3 | - | 189 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 4 | 5120 x 576 | 64 | 11.6 | 24.1 | 18.9 | - | 189 |  |
| DeepSeek-V4.1-Flash | wo_a | 8 | 1024 x 4096 | 1 | 11.5 | 11.0 | 15.7 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 8 | 1024 x 4096 | 8 | 11.5 | 16.0 | 15.8 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 8 | 1024 x 4096 | 32 | 13.4 | 23.6 | 18.5 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_a | 8 | 1024 x 4096 | 64 | 19.9 | 37.6 | 27.2 | - | 201 |  |
| DeepSeek-V4.1-Flash | wo_b | 8 | 5120 x 1024 | 1 | 9.2 | 14.3 | 14.5 | - | 204 |  |
| DeepSeek-V4.1-Flash | wo_b | 8 | 5120 x 1024 | 8 | 9.2 | 16.0 | 14.7 | - | 204 |  |
| DeepSeek-V4.1-Flash | wo_b | 8 | 5120 x 1024 | 32 | 10.0 | 19.7 | 17.2 | - | 204 |  |
| DeepSeek-V4.1-Flash | wo_b | 8 | 5120 x 1024 | 64 | 14.7 | 28.3 | 22.3 | - | 204 |  |
| DeepSeek-V4.1-Flash | indexer_wq_b | 8 | 512 x 1280 | 1 | 8.2 | 5.5 | 11.1 | - | 42 | L2 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 8 | 512 x 1280 | 8 | 8.2 | 7.5 | 11.1 | - | 42 | L2 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 8 | 512 x 1280 | 32 | 10.0 | 16.4 | 13.0 | - | 42 | L2 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 8 | 512 x 1280 | 64 | 17.0 | 27.7 | 20.1 | - | 42 | L2 |
| DeepSeek-V4.1-Flash | shared_w13 | 8 | 576 x 5120 | 1 | 13.8 | 10.0 | 17.4 | - | 189 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 8 | 576 x 5120 | 8 | 13.8 | 15.8 | 17.5 | - | 189 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 8 | 576 x 5120 | 32 | 16.3 | 32.0 | 20.4 | - | 189 |  |
| DeepSeek-V4.1-Flash | shared_w13 | 8 | 576 x 5120 | 64 | 26.4 | 53.7 | 31.2 | - | 189 |  |
| DeepSeek-V4.1-Flash | shared_w2 | 8 | 5120 x 288 | 1 | 7.0 | 7.0 | 14.4 | - | 94 | L2 |
| DeepSeek-V4.1-Flash | shared_w2 | 8 | 5120 x 288 | 8 | 7.6 | 8.0 | 15.1 | - | 94 | L2 |
| DeepSeek-V4.1-Flash | shared_w2 | 8 | 5120 x 288 | 32 | 12.7 | 12.8 | 20.1 | - | 94 | L2 |
| DeepSeek-V4.1-Flash | shared_w2 | 8 | 5120 x 288 | 64 | 11.5 | 19.7 | 19.5 | - | 94 | L2 |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 1 | 23.3 | 27.6 | 29.8 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 8 | 23.3 | 29.0 | 30.5 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 32 | 28.4 | 38.1 | 37.6 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 64 | 36.0 | 55.4 | 51.3 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 1 | 13.0 | 15.8 | 18.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 8 | 12.9 | 18.5 | 18.2 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 32 | 14.7 | 30.1 | 21.8 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 64 | 18.7 | 35.5 | 29.2 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 1 | 30.2 | 32.1 | 37.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 8 | 30.1 | 33.2 | 37.8 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 32 | 35.1 | 42.1 | 44.5 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 64 | 42.6 | 68.7 | 60.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 1 | 21.3 | 23.3 | 26.9 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 8 | 21.2 | 24.4 | 27.6 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 32 | 21.4 | 31.7 | 31.4 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 64 | 23.8 | 50.7 | 40.2 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 1 | 11.9 | 15.5 | 17.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 8 | 11.9 | 17.1 | 17.5 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 32 | 13.2 | 24.5 | 20.3 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 64 | 16.2 | 30.6 | 26.5 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 1 | 8.5 | 7.5 | 11.1 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 8 | 8.5 | 9.7 | 11.2 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 32 | 10.8 | 19.9 | 14.0 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 64 | 17.5 | 32.3 | 21.6 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 1 | 4.2 | 5.2 | 8.4 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 8 | 4.2 | 6.0 | 8.4 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 32 | 5.8 | 10.0 | 10.0 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 64 | 9.4 | 15.1 | 13.9 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 1 | 13.6 | 19.1 | 19.3 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 8 | 13.6 | 20.8 | 19.5 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 32 | 14.8 | 29.9 | 22.9 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 64 | 18.9 | 38.5 | 30.4 | - | 208 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 1 | 8.6 | 10.4 | 12.8 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 8 | 8.6 | 13.3 | 12.9 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 32 | 10.2 | 24.1 | 15.4 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 64 | 14.4 | 26.7 | 21.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 1 | 16.9 | 20.0 | 22.6 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 8 | 16.9 | 21.9 | 22.8 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 32 | 18.0 | 31.1 | 27.2 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 64 | 21.9 | 42.2 | 36.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 1 | 10.7 | 5.6 | 13.7 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 8 | 10.7 | 7.9 | 13.7 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 32 | 15.0 | 18.1 | 18.0 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 64 | 27.0 | 30.2 | 30.5 | - | 67 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 1 | 3.0 | 5.1 | 6.9 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 8 | 3.0 | 6.2 | 6.9 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 32 | 3.9 | 11.0 | 7.9 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 64 | 6.1 | 17.5 | 10.2 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | qkv | 4 | 2304 x 2048 | 1 | 8.6 | 14.4 | 13.8 | - | 203 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 4 | 2304 x 2048 | 8 | 8.5 | 17.6 | 13.9 | - | 203 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 4 | 2304 x 2048 | 32 | 10.0 | 30.2 | 16.6 | - | 203 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 4 | 2304 x 2048 | 64 | 14.5 | 32.8 | 22.3 | - | 203 |  |
| Qwen3.6-35B-A3B-FP8 | o | 4 | 2048 x 1024 | 1 | 6.3 | 7.2 | 8.9 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | o | 4 | 2048 x 1024 | 8 | 6.3 | 8.5 | 9.0 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | o | 4 | 2048 x 1024 | 32 | 7.9 | 14.4 | 10.9 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | o | 4 | 2048 x 1024 | 64 | 11.9 | 22.2 | 15.6 | - | 134 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 4 | 3072 x 2048 | 1 | 9.7 | 14.7 | 15.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 4 | 3072 x 2048 | 8 | 9.7 | 17.1 | 15.3 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 4 | 3072 x 2048 | 32 | 11.4 | 27.3 | 18.3 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 4 | 3072 x 2048 | 64 | 15.6 | 33.9 | 24.5 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 4 | 256 x 2048 | 1 | 10.4 | 5.5 | 13.2 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 4 | 256 x 2048 | 8 | 10.4 | 7.7 | 13.2 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 4 | 256 x 2048 | 32 | 14.2 | 17.9 | 17.0 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 4 | 256 x 2048 | 64 | 25.4 | 29.7 | 28.5 | - | 34 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 4 | 2048 x 128 | 1 | 2.2 | 4.2 | 6.0 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 4 | 2048 x 128 | 8 | 2.2 | 4.2 | 6.1 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 4 | 2048 x 128 | 32 | 2.9 | 5.3 | 6.8 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 4 | 2048 x 128 | 64 | 4.2 | 6.7 | 8.2 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | 1152 x 2048 | 1 | 7.5 | 8.7 | 10.7 | - | 151 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | 1152 x 2048 | 8 | 7.5 | 11.4 | 10.8 | - | 151 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | 1152 x 2048 | 32 | 9.5 | 21.6 | 13.3 | - | 151 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | 1152 x 2048 | 64 | 14.8 | 38.7 | 19.6 | - | 151 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 8 | 1536 x 2048 | 1 | 8.2 | 9.4 | 12.1 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 8 | 1536 x 2048 | 8 | 8.2 | 12.3 | 12.2 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 8 | 1536 x 2048 | 32 | 10.3 | 23.0 | 14.8 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 8 | 1536 x 2048 | 64 | 15.6 | 33.9 | 21.5 | - | 201 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 8 | 128 x 2048 | 1 | 10.2 | 5.5 | 12.9 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 8 | 128 x 2048 | 8 | 10.3 | 7.7 | 13.0 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 8 | 128 x 2048 | 32 | 14.1 | 17.8 | 16.8 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 8 | 128 x 2048 | 64 | 25.3 | 29.3 | 28.2 | - | 17 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 8 | 2048 x 64 | 1 | 3.7 | 3.9 | 7.5 | - | 8 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 8 | 2048 x 64 | 8 | 4.3 | 3.9 | 8.1 | - | 8 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 8 | 2048 x 64 | 32 | 5.0 | 5.0 | 8.8 | - | 8 | L2 |
| Qwen3.6-35B-A3B-FP8 | expert_down | 8 | 2048 x 64 | 64 | 6.4 | 6.6 | 10.5 | - | 8 | L2 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 1 | 30.3 | 29.5 | 36.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 8 | 31.1 | 30.6 | 37.8 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 32 | 37.5 | 39.7 | 55.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 64 | 39.9 | 62.9 | 60.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 1 | 33.3 | 34.8 | 42.6 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 8 | 34.1 | 36.0 | 42.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 32 | 34.6 | 46.0 | 45.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 64 | 36.5 | 72.8 | 54.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 1 | 17.3 | 19.3 | 25.4 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 8 | 17.6 | 21.8 | 25.8 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 32 | 20.4 | 33.8 | 29.4 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 64 | 22.7 | 47.3 | 36.9 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 1 | 30.4 | 33.6 | 43.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 8 | 31.3 | 36.2 | 44.2 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 32 | 33.8 | 48.3 | 48.4 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 64 | 36.8 | 69.0 | 62.5 | - | 208 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 1 | 17.6 | 18.8 | 24.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 8 | 17.8 | 20.6 | 24.6 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 32 | 21.9 | 30.6 | 30.8 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 64 | 24.2 | 43.5 | 36.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 1 | 11.3 | 14.1 | 17.4 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 8 | 11.5 | 16.4 | 17.7 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 32 | 14.2 | 26.2 | 21.3 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 64 | 16.6 | 34.3 | 25.7 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 1 | 9.4 | 10.5 | 15.9 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 8 | 9.6 | 14.5 | 16.2 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 32 | 12.2 | 30.5 | 19.0 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 64 | 17.1 | 36.6 | 24.3 | - | 202 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 1 | 6.6 | 8.1 | 10.0 | - | 127 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 8 | 6.7 | 9.7 | 10.1 | - | 127 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 32 | 9.5 | 16.7 | 13.7 | - | 127 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 64 | 11.6 | 25.0 | 15.9 | - | 127 |  |

## Numerics: error against an exact float64 product of the dequantized weights (max: of the sum of |x w|; rms: of the outputs' rms)

| model | tensor | M | vLLM kernel | vLLM max | vLLM rms | ours max | ours rms |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | qkv_a | 1 | MarlinFP8ScaledMMLinearKernel | 2.4e-04 | 2.2e-03 | 2.3e-04 | 1.6e-03 |
| GLM-5.3-Flash | qkv_a | 8 | MarlinFP8ScaledMMLinearKernel | 3.4e-04 | 2.3e-03 | 3.4e-04 | 1.7e-03 |
| GLM-5.3-Flash | qkv_a | 32 | MarlinFP8ScaledMMLinearKernel | 3.9e-04 | 2.3e-03 | 3.4e-04 | 1.7e-03 |
| GLM-5.3-Flash | qkv_a | 64 | MarlinFP8ScaledMMLinearKernel | 3.8e-04 | 2.3e-03 | 3.6e-04 | 1.7e-03 |
| GLM-5.3-Flash | q_b | 1 | MarlinFP8ScaledMMLinearKernel | 5.1e-04 | 2.3e-03 | 3.9e-04 | 1.6e-03 |
| GLM-5.3-Flash | q_b | 8 | MarlinFP8ScaledMMLinearKernel | 6.3e-04 | 2.3e-03 | 6.3e-04 | 1.6e-03 |
| GLM-5.3-Flash | q_b | 32 | MarlinFP8ScaledMMLinearKernel | 6.2e-04 | 2.3e-03 | 5.9e-04 | 1.7e-03 |
| GLM-5.3-Flash | q_b | 64 | MarlinFP8ScaledMMLinearKernel | 7.1e-04 | 2.3e-03 | 6.2e-04 | 1.7e-03 |
| GLM-5.3-Flash | o | 1 | MarlinFP8ScaledMMLinearKernel | 1.3e-04 | 2.3e-03 | 1.1e-04 | 1.7e-03 |
| GLM-5.3-Flash | o | 8 | MarlinFP8ScaledMMLinearKernel | 1.8e-04 | 2.3e-03 | 1.4e-04 | 1.7e-03 |
| GLM-5.3-Flash | o | 32 | MarlinFP8ScaledMMLinearKernel | 1.9e-04 | 2.3e-03 | 1.8e-04 | 1.7e-03 |
| GLM-5.3-Flash | o | 64 | MarlinFP8ScaledMMLinearKernel | 1.8e-04 | 2.3e-03 | 1.7e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 1 | MarlinFP8ScaledMMLinearKernel | 2.9e-04 | 2.3e-03 | 2.0e-04 | 1.6e-03 |
| Qwen3.8-27B-FP8 | qkv | 8 | MarlinFP8ScaledMMLinearKernel | 3.6e-04 | 2.3e-03 | 3.1e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 32 | MarlinFP8ScaledMMLinearKernel | 3.7e-04 | 2.3e-03 | 3.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 64 | MarlinFP8ScaledMMLinearKernel | 4.3e-04 | 2.3e-03 | 3.4e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 1 | MarlinFP8ScaledMMLinearKernel | 2.7e-04 | 2.3e-03 | 2.1e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 8 | MarlinFP8ScaledMMLinearKernel | 3.0e-04 | 2.3e-03 | 2.4e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 32 | MarlinFP8ScaledMMLinearKernel | 3.0e-04 | 2.3e-03 | 2.9e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 64 | MarlinFP8ScaledMMLinearKernel | 3.0e-04 | 2.3e-03 | 2.9e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | MarlinFP8ScaledMMLinearKernel | 2.6e-04 | 2.3e-03 | 2.1e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 8 | MarlinFP8ScaledMMLinearKernel | 3.4e-04 | 2.3e-03 | 3.4e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 32 | MarlinFP8ScaledMMLinearKernel | 3.6e-04 | 2.3e-03 | 3.3e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 64 | MarlinFP8ScaledMMLinearKernel | 3.6e-04 | 2.3e-03 | 3.4e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | qkv | 1 | MarlinMxfp8LinearKernel | 3.9e-04 | 1.7e-03 | 3.9e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | qkv | 8 | MarlinMxfp8LinearKernel | 4.3e-04 | 1.7e-03 | 4.3e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | qkv | 32 | MarlinMxfp8LinearKernel | 4.6e-04 | 1.7e-03 | 4.6e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | qkv | 64 | MarlinMxfp8LinearKernel | 5.1e-04 | 1.7e-03 | 5.1e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | o | 1 | MarlinMxfp8LinearKernel | 2.5e-04 | 1.6e-03 | 2.5e-04 | 1.6e-03 |
| MiniMax-M3-MXFP8 | o | 8 | MarlinMxfp8LinearKernel | 2.7e-04 | 1.7e-03 | 2.7e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | o | 32 | MarlinMxfp8LinearKernel | 3.5e-04 | 1.7e-03 | 3.5e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | o | 64 | MarlinMxfp8LinearKernel | 4.0e-04 | 1.7e-03 | 4.0e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | MarlinMxfp8LinearKernel | 4.5e-04 | 1.7e-03 | 4.5e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 8 | MarlinMxfp8LinearKernel | 4.6e-04 | 1.7e-03 | 4.6e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 32 | MarlinMxfp8LinearKernel | 4.7e-04 | 1.7e-03 | 4.7e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 64 | MarlinMxfp8LinearKernel | 4.9e-04 | 1.7e-03 | 4.9e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.6e-02 | 3.6e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 8 | CutlassFP8ScaledMMLinearKernel | 3.1e-03 | 2.6e-02 | 3.3e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 32 | CutlassFP8ScaledMMLinearKernel | 3.6e-03 | 2.7e-02 | 4.1e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 64 | CutlassFP8ScaledMMLinearKernel | 4.4e-03 | 2.7e-02 | 4.4e-04 | 1.6e-03 |
| Mistral-Small-4-119B-2603 | q_b | 1 | CutlassFP8ScaledMMLinearKernel | 7.3e-03 | 2.7e-02 | 6.3e-04 | 1.6e-03 |
| Mistral-Small-4-119B-2603 | q_b | 8 | CutlassFP8ScaledMMLinearKernel | 8.3e-03 | 2.7e-02 | 7.5e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 32 | CutlassFP8ScaledMMLinearKernel | 8.4e-03 | 2.7e-02 | 7.1e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 64 | CutlassFP8ScaledMMLinearKernel | 7.7e-03 | 2.6e-02 | 8.3e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | CutlassFP8ScaledMMLinearKernel | 1.2e-02 | 3.0e-02 | 1.2e-03 | 1.6e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | CutlassFP8ScaledMMLinearKernel | 1.4e-02 | 2.7e-02 | 1.4e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 32 | CutlassFP8ScaledMMLinearKernel | 1.4e-02 | 2.7e-02 | 1.4e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 64 | CutlassFP8ScaledMMLinearKernel | 1.4e-02 | 2.6e-02 | 1.5e-03 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wq_a | 1 | MarlinMxfp8LinearKernel | 3.1e-04 | 1.6e-03 | 3.1e-04 | 1.6e-03 |
| DeepSeek-V4.1-Flash | wq_a | 8 | MarlinMxfp8LinearKernel | 3.8e-04 | 1.6e-03 | 3.8e-04 | 1.6e-03 |
| DeepSeek-V4.1-Flash | wq_a | 32 | MarlinMxfp8LinearKernel | 4.3e-04 | 1.7e-03 | 4.3e-04 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wq_a | 64 | MarlinMxfp8LinearKernel | 4.1e-04 | 1.7e-03 | 4.1e-04 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wkv | 1 | MarlinMxfp8LinearKernel | 2.9e-04 | 1.6e-03 | 2.9e-04 | 1.6e-03 |
| DeepSeek-V4.1-Flash | wkv | 8 | MarlinMxfp8LinearKernel | 3.7e-04 | 1.7e-03 | 3.7e-04 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wkv | 32 | MarlinMxfp8LinearKernel | 3.7e-04 | 1.7e-03 | 3.7e-04 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wkv | 64 | MarlinMxfp8LinearKernel | 3.8e-04 | 1.6e-03 | 3.8e-04 | 1.6e-03 |
| DeepSeek-V4.1-Flash | wq_b | 1 | MarlinMxfp8LinearKernel | 8.2e-04 | 1.7e-03 | 8.2e-04 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wq_b | 8 | MarlinMxfp8LinearKernel | 1.1e-03 | 1.7e-03 | 1.1e-03 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wq_b | 32 | MarlinMxfp8LinearKernel | 1.3e-03 | 1.7e-03 | 1.3e-03 | 1.7e-03 |
| DeepSeek-V4.1-Flash | wq_b | 64 | MarlinMxfp8LinearKernel | 1.1e-03 | 1.7e-03 | 1.1e-03 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | MarlinFP8ScaledMMLinearKernel | 6.3e-04 | 2.3e-03 | 6.3e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | MarlinFP8ScaledMMLinearKernel | 6.0e-04 | 2.3e-03 | 4.7e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 32 | MarlinFP8ScaledMMLinearKernel | 6.2e-04 | 2.3e-03 | 5.7e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 64 | MarlinFP8ScaledMMLinearKernel | 5.9e-04 | 2.3e-03 | 5.9e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 1 | MarlinFP8ScaledMMLinearKernel | 2.9e-04 | 2.3e-03 | 2.2e-04 | 1.6e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 8 | MarlinFP8ScaledMMLinearKernel | 4.1e-04 | 2.3e-03 | 3.6e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 32 | MarlinFP8ScaledMMLinearKernel | 3.5e-04 | 2.3e-03 | 3.2e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 64 | MarlinFP8ScaledMMLinearKernel | 5.0e-04 | 2.3e-03 | 4.6e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | MarlinFP8ScaledMMLinearKernel | 5.0e-04 | 2.3e-03 | 4.5e-04 | 1.6e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 8 | MarlinFP8ScaledMMLinearKernel | 5.6e-04 | 2.3e-03 | 5.0e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 32 | MarlinFP8ScaledMMLinearKernel | 5.7e-04 | 2.3e-03 | 5.3e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 64 | MarlinFP8ScaledMMLinearKernel | 6.6e-04 | 2.3e-03 | 5.5e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 3.4e-03 | 2.7e-02 | 3.0e-04 | 1.6e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 3.4e-03 | 2.6e-02 | 3.9e-04 | 1.6e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 3.8e-03 | 2.6e-02 | 3.7e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.9e-03 | 2.6e-02 | 4.4e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | CutlassFP8ScaledMMLinearKernel | 3.4e-03 | 2.6e-02 | 3.4e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 8 | CutlassFP8ScaledMMLinearKernel | 3.8e-03 | 2.7e-02 | 3.9e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 32 | CutlassFP8ScaledMMLinearKernel | 3.6e-03 | 2.7e-02 | 4.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 64 | CutlassFP8ScaledMMLinearKernel | 4.3e-03 | 2.7e-02 | 4.6e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 2.5e-03 | 2.7e-02 | 2.4e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.7e-02 | 2.7e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 2.9e-03 | 2.7e-02 | 3.0e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.0e-03 | 2.7e-02 | 3.5e-04 | 1.7e-03 |
