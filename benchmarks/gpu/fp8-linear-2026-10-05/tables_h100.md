
### H100 (run 3, the kernel as committed): NVIDIA H100 80GB HBM3 (sm_90)

Gate: 15 of 356 at most vLLM's time; by tokens: M=1 9/89, M=8 6/89, M=32 0/89, M=64 0/89; worst ratio 7.30, geometric mean 2.52

vLLM's kernels: GLM-5.3-Flash (block): FlashInferFp8DeepGEMMDynamicBlockScaledKernel[FlashInferFp8BlockScaledMMKernel<32<=DeepGemmFp8BlockScaledMMKernel]; Qwen3.8-27B-FP8 (block): FlashInferFp8DeepGEMMDynamicBlockScaledKernel[FlashInferFp8BlockScaledMMKernel<32<=DeepGemmFp8BlockScaledMMKernel]; MiniMax-M3-MXFP8 (mx): MarlinMxfp8LinearKernel; Mistral-Small-4-119B-2603 (tensor): CutlassFP8ScaledMMLinearKernel; DeepSeek-V4.1-Flash (ds32): MarlinMxfp8LinearKernel

| model | tensor | tp | N x K | M=1 vLLM / ours | M=8 | M=32 | M=64 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | qkv_a | 1 | 2048 x 4096 | 10.0 / 14.6 | 9.9 / 16.2 | 11.4 / 23.9 | 11.1 / 34.3 |
| GLM-5.3-Flash | q_b | 1 | 16384 x 1536 | 13.1 / 31.0 | 13.3 / 32.9 | 15.5 / 54.5 | 15.6 / 65.5 |
| GLM-5.3-Flash | o | 1 | 4096 x 16384 | 30.6 / 79.1 | 31.0 / 83.8 | 33.9 / 131.8 | 36.7 / 210.0 |
| GLM-5.3-Flash | mlp_gate_up | 1 | 24576 x 4096 | 39.1 / 105.1 | 39.9 / 110.2 | 43.3 / 164.3 | 43.8 / 243.2 |
| GLM-5.3-Flash | mlp_down | 1 | 4096 x 12288 | 24.7 / 64.7 | 24.8 / 66.6 | 26.9 / 105.4 | 29.0 / 145.9 |
| GLM-5.3-Flash | expert_gate_up | 1 | 4096 x 4096 | 11.3 / 24.5 | 11.3 / 26.0 | 13.3 / 34.9 | 13.6 / 71.2 |
| GLM-5.3-Flash | expert_down | 1 | 4096 x 2048 | 7.9 / 14.6 | 8.0 / 16.0 | 10.1 / 23.2 | 10.0 / 33.2 |
| GLM-5.3-Flash | q_b | 2 | 8192 x 1536 | 9.3 / 19.5 | 9.3 / 20.9 | 11.3 / 28.0 | 11.2 / 38.6 |
| GLM-5.3-Flash | o | 2 | 4096 x 8192 | 18.0 / 42.9 | 17.8 / 45.0 | 20.0 / 87.2 | 21.1 / 119.7 |
| GLM-5.3-Flash | mlp_gate_up | 2 | 12288 x 4096 | 22.4 / 71.1 | 22.9 / 74.2 | 25.1 / 91.2 | 25.4 / 139.0 |
| GLM-5.3-Flash | mlp_down | 2 | 4096 x 6144 | 14.5 / 33.7 | 14.5 / 35.4 | 16.7 / 45.8 | 17.3 / 71.6 |
| GLM-5.3-Flash | expert_down | 2 | 4096 x 1024 | 6.4 / 10.0 | 6.4 / 11.3 | 8.2 / 17.8 | 8.3 / 26.9 |
| GLM-5.3-Flash | q_b | 4 | 4096 x 1536 | 7.4 / 14.7 | 7.4 / 17.2 | 9.4 / 29.5 | 9.2 / 44.5 |
| GLM-5.3-Flash | mlp_gate_up | 4 | 6144 x 4096 | 13.5 / 35.0 | 13.5 / 37.7 | 16.1 / 52.2 | 16.2 / 83.9 |
| GLM-5.3-Flash | mlp_down | 4 | 4096 x 3072 | 9.7 / 19.4 | 9.7 / 21.2 | 11.7 / 29.1 | 11.8 / 39.6 |
| GLM-5.3-Flash | expert_gate_up | 4 | 1024 x 4096 | 9.2 / 10.3 | 9.1 / 12.7 | 11.1 / 23.6 | 11.1 / 38.5 |
| GLM-5.3-Flash | expert_down | 4 | 4096 x 512 | 5.6 / 8.2 | 5.6 / 9.5 | 7.3 / 15.3 | 7.2 / 22.5 |
| GLM-5.3-Flash | q_b | 8 | 2048 x 1536 | 6.9 / 9.9 | 7.0 / 12.0 | 8.5 / 21.4 | 8.2 / 34.2 |
| GLM-5.3-Flash | mlp_gate_up | 8 | 3072 x 4096 | 10.6 / 20.1 | 10.5 / 22.9 | 12.6 / 35.9 | 12.7 / 54.6 |
| GLM-5.3-Flash | expert_gate_up | 8 | 512 x 4096 | 9.1 / 8.5 **ok** | 9.1 / 10.8 | 10.9 / 21.6 | 10.8 / 36.0 |
| GLM-5.3-Flash | expert_down | 8 | 4096 x 256 | 5.1 / 8.2 | 5.1 / 9.8 | 6.5 / 17.2 | 6.6 / 27.4 |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 29.8 / 86.7 | 30.3 / 90.5 | 33.6 / 133.4 | 34.4 / 197.9 |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 16.1 / 45.5 | 16.4 / 50.2 | 19.8 / 80.5 | 19.8 / 91.8 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 33.0 / 85.4 | 33.4 / 90.2 | 36.6 / 136.0 | 37.6 / 216.8 |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 23.6 / 69.0 | 23.9 / 71.5 | 26.5 / 100.4 | 27.0 / 156.5 |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 15.8 / 43.3 | 16.1 / 45.7 | 18.8 / 71.3 | 19.1 / 114.5 |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 68.1 / 244.8 | 68.9 / 258.2 | 70.7 / 288.4 | 72.1 / 452.2 |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 37.9 / 116.6 | 38.5 / 121.5 | 43.8 / 149.1 | 46.2 / 273.2 |
| MiniMax-M3-MXFP8 | qkv | 1 | 9216 x 6144 | 26.4 / 84.1 | 26.6 / 86.2 | 31.7 / 103.5 | 40.2 / 215.4 |
| MiniMax-M3-MXFP8 | o | 1 | 6144 x 8192 | 24.5 / 68.1 | 24.6 / 72.4 | 28.9 / 102.2 | 39.1 / 219.3 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | 24576 x 6144 | 59.0 / 156.0 | 59.3 / 162.2 | 64.5 / 245.3 | 76.9 / 525.8 |
| MiniMax-M3-MXFP8 | mlp_down | 1 | 6144 x 12288 | 33.8 / 108.3 | 33.8 / 113.9 | 39.9 / 142.8 | 53.1 / 295.0 |
| MiniMax-M3-MXFP8 | expert_gate_up | 1 | 6144 x 6144 | 19.9 / 56.3 | 19.9 / 60.4 | 22.9 / 86.9 | 31.6 / 152.2 |
| MiniMax-M3-MXFP8 | expert_down | 1 | 6144 x 3072 | 13.0 / 31.2 | 12.9 / 34.0 | 14.6 / 57.1 | 20.5 / 76.4 |
| MiniMax-M3-MXFP8 | index_q | 1 | 512 x 6144 | 20.7 / 10.2 **ok** | 20.8 / 13.7 **ok** | 23.9 / 28.2 | 34.5 / 53.3 |
| MiniMax-M3-MXFP8 | qkv | 2 | 4608 x 6144 | 17.5 / 44.1 | 17.5 / 45.9 | 19.0 / 70.1 | 26.7 / 125.5 |
| MiniMax-M3-MXFP8 | o | 2 | 6144 x 4096 | 15.5 / 34.6 | 15.4 / 37.5 | 17.3 / 52.7 | 24.2 / 119.5 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 2 | 12288 x 6144 | 32.9 / 106.1 | 33.2 / 110.9 | 37.7 / 140.2 | 46.4 / 269.3 |
| MiniMax-M3-MXFP8 | expert_gate_up | 2 | 3072 x 6144 | 13.8 / 29.1 | 13.7 / 32.2 | 15.0 / 47.1 | 21.2 / 79.0 |
| MiniMax-M3-MXFP8 | expert_down | 2 | 6144 x 1536 | 10.0 / 18.9 | 10.1 / 21.7 | 11.0 / 38.3 | 15.8 / 66.8 |
| MiniMax-M3-MXFP8 | qkv | 4 | 2304 x 6144 | 13.0 / 24.2 | 13.0 / 26.6 | 14.5 / 38.2 | 20.5 / 71.8 |
| MiniMax-M3-MXFP8 | o | 4 | 6144 x 2048 | 11.1 / 20.4 | 11.0 / 23.1 | 12.2 / 35.4 | 17.2 / 66.1 |
| MiniMax-M3-MXFP8 | expert_gate_up | 4 | 1536 x 6144 | 13.0 / 19.1 | 12.9 / 20.9 | 14.3 / 29.3 | 20.3 / 54.3 |
| MiniMax-M3-MXFP8 | expert_down | 4 | 6144 x 768 | 7.4 / 11.9 | 7.3 / 14.4 | 9.6 / 28.9 | 13.6 / 51.7 |
| MiniMax-M3-MXFP8 | qkv | 8 | 1152 x 6144 | 13.8 / 14.7 | 13.9 / 17.0 | 15.7 / 28.0 | 22.1 / 52.6 |
| MiniMax-M3-MXFP8 | o | 8 | 6144 x 1024 | 9.3 / 14.2 | 9.3 / 15.5 | 10.2 / 22.4 | 14.1 / 41.2 |
| MiniMax-M3-MXFP8 | expert_gate_up | 8 | 768 x 6144 | 16.0 / 14.2 **ok** | 15.9 / 16.6 | 18.1 / 27.7 | 25.8 / 51.4 |
| MiniMax-M3-MXFP8 | expert_down | 8 | 6144 x 384 | 7.5 / 9.5 | 7.5 / 11.3 | 5.7 / 19.6 | 7.2 / 35.2 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 7.5 / 13.9 | 7.5 / 15.5 | 8.9 / 22.9 | 8.8 / 33.3 |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 5.9 / 9.9 | 6.0 / 11.3 | 6.3 / 17.6 | 6.3 / 26.7 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 5.1 / 8.2 | 5.1 / 9.9 | 5.3 / 17.1 | 5.5 / 27.1 |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 10.5 / 23.7 | 10.7 / 25.5 | 11.5 / 33.8 | 10.8 / 69.3 |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 7.4 / 14.4 | 7.5 / 15.8 | 7.8 / 22.7 | 7.7 / 32.7 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 5.4 / 8.3 | 5.4 / 9.7 | 5.8 / 15.6 | 5.8 / 24.2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 4.8 / 7.8 | 4.8 / 9.6 | 5.0 / 16.8 | 5.1 / 27.1 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 8.0 / 14.4 | 8.0 / 16.0 | 9.6 / 23.6 | 9.3 / 33.9 |
| Mistral-Small-4-119B-2603 | q_b | 4 | 1024 x 1024 | 5.2 / 8.0 | 5.3 / 9.4 | 5.5 / 15.5 | 5.5 / 24.1 |
| Mistral-Small-4-119B-2603 | kv_b | 4 | 1536 x 256 | 3.8 / 6.9 | 3.9 / 8.5 | 4.1 / 15.7 | 4.2 / 25.6 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 4 | 1024 x 4096 | 7.1 / 10.2 | 7.3 / 12.7 | 7.2 / 23.5 | 7.3 / 38.2 |
| Mistral-Small-4-119B-2603 | expert_down | 4 | 4096 x 512 | 5.3 / 8.2 | 5.3 / 9.6 | 5.6 / 15.3 | 5.7 / 22.3 |
| Mistral-Small-4-119B-2603 | q_b | 8 | 512 x 1024 | 4.5 / 6.9 | 4.6 / 8.2 | 4.6 / 14.5 | 4.8 / 23.3 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | 768 x 256 | 3.8 / 6.8 | 3.9 / 8.4 | 4.0 / 15.3 | 4.0 / 25.3 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 8 | 512 x 4096 | 6.8 / 8.5 | 7.0 / 11.0 | 6.9 / 21.5 | 7.1 / 35.9 |
| Mistral-Small-4-119B-2603 | expert_down | 8 | 4096 x 256 | 4.9 / 8.1 | 5.0 / 9.7 | 5.1 / 17.0 | 5.3 / 27.1 |
| DeepSeek-V4.1-Flash | wq_a | 1 | 1280 x 5120 | 12.6 / 14.8 | 12.5 / 17.1 | 14.1 / 27.7 | 19.9 / 51.3 |
| DeepSeek-V4.1-Flash | wkv | 1 | 512 x 5120 | 18.6 / 10.2 **ok** | 18.6 / 13.5 **ok** | 21.1 / 27.4 | 30.8 / 51.5 |
| DeepSeek-V4.1-Flash | wq_b | 1 | 32768 x 1280 | 20.5 / 51.6 | 20.9 / 54.9 | 20.0 / 77.9 | 24.2 / 176.4 |
| DeepSeek-V4.1-Flash | wo_a | 1 | 8192 x 4096 | 17.1 / 41.5 | 17.2 / 44.1 | 17.8 / 72.5 | 24.2 / 147.5 |
| DeepSeek-V4.1-Flash | wo_b | 1 | 5120 x 8192 | 21.6 / 55.1 | 22.0 / 59.3 | 24.8 / 92.4 | 34.0 / 190.1 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 1 | 4096 x 1280 | 6.9 / 14.2 | 7.0 / 16.4 | 8.2 / 25.9 | 10.9 / 46.2 |
| DeepSeek-V4.1-Flash | shared_w13 | 1 | 4608 x 5120 | 15.8 / 38.4 | 15.8 / 40.1 | 16.6 / 60.6 | 23.3 / 123.6 |
| DeepSeek-V4.1-Flash | shared_w2 | 1 | 5120 x 2304 | 10.5 / 20.0 | 10.4 / 22.7 | 12.1 / 35.6 | 17.0 / 66.4 |
| DeepSeek-V4.1-Flash | wq_b | 2 | 16384 x 1280 | 11.1 / 26.5 | 11.1 / 28.5 | 13.8 / 45.7 | 18.3 / 83.5 |
| DeepSeek-V4.1-Flash | wo_a | 2 | 4096 x 4096 | 11.5 / 24.6 | 11.5 / 26.3 | 12.5 / 35.3 | 16.9 / 87.7 |
| DeepSeek-V4.1-Flash | wo_b | 2 | 5120 x 4096 | 14.5 / 30.6 | 14.6 / 33.7 | 15.6 / 47.3 | 21.8 / 103.6 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 2 | 2048 x 1280 | 8.3 / 9.9 | 8.3 / 12.6 | 9.6 / 23.9 | 13.5 / 44.4 |
| DeepSeek-V4.1-Flash | shared_w13 | 2 | 2304 x 5120 | 12.1 / 20.5 | 12.0 / 24.5 | 13.5 / 40.2 | 18.9 / 75.4 |
| DeepSeek-V4.1-Flash | shared_w2 | 2 | 5120 x 1152 | 8.7 / 14.6 | 8.7 / 16.6 | 9.7 / 26.4 | 13.8 / 48.1 |
| DeepSeek-V4.1-Flash | wq_b | 4 | 8192 x 1280 | 9.1 / 19.2 | 9.1 / 21.3 | 8.4 / 35.4 | 11.2 / 61.0 |
| DeepSeek-V4.1-Flash | wo_a | 4 | 2048 x 4096 | 10.2 / 14.7 | 10.2 / 16.3 | 11.6 / 24.3 | 15.8 / 44.7 |
| DeepSeek-V4.1-Flash | wo_b | 4 | 5120 x 2048 | 11.1 / 20.0 | 11.3 / 22.7 | 11.6 / 35.3 | 16.3 / 65.8 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 4 | 1024 x 1280 | 8.5 / 8.3 **ok** | 8.6 / 11.1 | 9.9 / 22.8 | 14.2 / 38.5 |
| DeepSeek-V4.1-Flash | shared_w13 | 4 | 1152 x 5120 | 13.4 / 14.7 | 13.3 / 17.0 | 15.1 / 27.6 | 21.5 / 51.2 |
| DeepSeek-V4.1-Flash | shared_w2 | 4 | 5120 x 576 | 7.2 / 9.9 | 7.2 / 11.8 | 13.1 / 20.4 | 18.8 / 37.9 |
| DeepSeek-V4.1-Flash | wo_a | 8 | 1024 x 4096 | 12.9 / 10.4 **ok** | 12.9 / 12.9 **ok** | 15.1 / 23.8 | 21.0 / 44.2 |
| DeepSeek-V4.1-Flash | wo_b | 8 | 5120 x 1024 | 9.1 / 14.4 | 9.0 / 15.6 | 9.8 / 22.5 | 13.6 / 40.9 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 8 | 512 x 1280 | 13.2 / 7.8 **ok** | 13.3 / 10.5 **ok** | 15.5 / 21.9 | 22.3 / 37.7 |
| DeepSeek-V4.1-Flash | shared_w13 | 8 | 576 x 5120 | 19.0 / 10.2 **ok** | 18.9 / 13.5 **ok** | 21.3 / 27.5 | 31.1 / 51.7 |
| DeepSeek-V4.1-Flash | shared_w2 | 8 | 5120 x 288 | 9.3 / 8.3 **ok** | 10.2 / 10.1 **ok** | 15.5 / 17.7 | 21.1 / 28.7 |

Numerics (error against the exact product: largest max, mean rms, over the shapes and M measured):

| model | scales | vLLM kernel | vLLM max | vLLM rms | ours max | ours rms |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | block | FlashInferFp8DeepGEMMDynamicBlockScaledKernel | 8.1e-03 | 3.7e-02 | 6.3e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | block | FlashInferFp8DeepGEMMDynamicBlockScaledKernel | 4.8e-03 | 3.7e-02 | 4.5e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | mx | MarlinMxfp8LinearKernel | 4.8e-04 | 1.7e-03 | 4.8e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | tensor | CutlassFP8ScaledMMLinearKernel | 1.5e-02 | 2.7e-02 | 1.5e-03 | 1.7e-03 |
| DeepSeek-V4.1-Flash | ds32 | MarlinMxfp8LinearKernel | 1.2e-03 | 1.7e-03 | 1.2e-03 | 1.7e-03 |
