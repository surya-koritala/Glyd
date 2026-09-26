Loading weights:   0%|          | 0/339 [00:00<?, ?it/s]Loading weights: 100%|██████████| 339/339 [00:00<00:00, 17147.69it/s]
[transformers] The attention mask is not set with a batched input, and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[W926 08:07:52.240289510 CUDACachingAllocator.cpp:528] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 12845056, total: 16717119488).
bf16 (weights 15.23 GB): batch 1: 43.4 tokens/s (43.4 a sequence), peak VRAM 15.25 GB
bf16 (weights 15.23 GB): batch 4: 167.9 tokens/s (42.0 a sequence), peak VRAM 15.27 GB
bf16 (weights 15.23 GB): batch 16: 652.0 tokens/s (40.7 a sequence), peak VRAM 15.36 GB
bf16 (weights 15.23 GB): batch 32: 1153.7 tokens/s (36.1 a sequence), peak VRAM 15.48 GB
bf16 (weights 15.23 GB): batch 48: 1664.8 tokens/s (34.7 a sequence), peak VRAM 15.59 GB
bf16 (weights 15.23 GB): batch 64: 2160.0 tokens/s (33.8 a sequence), peak VRAM 15.71 GB
bf16 prefill: 16 tokens 24 ms (672 tokens/s), 64 tokens 27 ms (2353 tokens/s), 128 tokens 29 ms (4422 tokens/s), 256 tokens 43 ms (5899 tokens/s), 512 tokens 79 ms (6448 tokens/s), 1024 tokens 154 ms (6630 tokens/s), 2048 tokens 301 ms (6802 tokens/s), 4096 tokens 645 ms (6350 tokens/s)
bf16 perplexity: 17.0015 (12600 tokens)
mma fused: packed in 8 s; weights 10.32 GB against 15.23 GB bf16 (67.8%), scratch 0.27 GB, VRAM in use 10.60 GB on 1 GPU
glyd mma fused: batch 1: 55.7 tokens/s (55.7 a sequence), peak VRAM 10.61 GB
glyd mma fused: batch 4: 217.5 tokens/s (54.4 a sequence), peak VRAM 10.63 GB
glyd mma fused: batch 16: 814.2 tokens/s (50.9 a sequence), peak VRAM 10.72 GB
glyd mma fused: batch 32: 1518.9 tokens/s (47.5 a sequence), peak VRAM 10.84 GB
glyd mma fused: batch 48: 1882.5 tokens/s (39.2 a sequence), peak VRAM 10.95 GB
glyd mma fused: batch 64: 2244.7 tokens/s (35.1 a sequence), peak VRAM 11.07 GB
glyd mma prefill: 16 tokens 19 ms (849 tokens/s), 64 tokens 26 ms (2500 tokens/s), 128 tokens 29 ms (4472 tokens/s), 256 tokens 45 ms (5656 tokens/s), 512 tokens 85 ms (5991 tokens/s), 1024 tokens 163 ms (6301 tokens/s), 2048 tokens 330 ms (6204 tokens/s), 4096 tokens 702 ms (5832 tokens/s)
glyd mma perplexity: 17.0052 (12600 tokens)
next-token choice as bf16's: 98.13%
logits bit-identical: False
generated tokens identical to bf16: 0 of 64
text:  long before the computer era in the late 1940s. In the early days, nearly all media were analog: sound was recorded on wax cylinders and magnetic tape, images were stored as
