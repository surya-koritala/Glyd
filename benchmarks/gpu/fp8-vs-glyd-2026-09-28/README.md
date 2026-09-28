# What bf16, FP8 and Glyd change in a model's answers, 2026-09-28

Qwen3-4B-Instruct-2507 on an RTX 4080 SUPER, one mode a process (`gpu/fp8_compare.py`, run by `fp8_job.sh` and
`fp8_job2.sh`): bf16; bf16 through eager attention in place of SDPA (bf16's own variation between two valid kernels);
Glyd with `exact=True` and by default (the tiered layout, fused products); FP8 as Qwen releases it
(Qwen/Qwen3-4B-Instruct-2507-FP8: e4m3 weights in 128 x 128 blocks, activations quantized per token). Glyd 0.22.0's
code (main at 4fd327d's wheel), transformers 5.17, torch 2.14 + CUDA 13.

| | weights GB | perplexity | top token = bf16's | greedy answers = bf16's | MMLU (1000, 0-shot) | MMLU answers changed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 8.04 | 11.2435 | 100.00% | 50 of 50 | 68.30% | 0 of 1000 |
| bf16, eager attention | 8.04 | 11.2469 | 98.63% | 35 of 50 | 68.30% | 16 of 1000 |
| Glyd, exact=True | 6.78 | 11.2435 | 100.00% | 50 of 50 | 68.30% | 0 of 1000 |
| Glyd | 6.27 | 11.2436 | 99.99% | 34 of 50 | 68.10% | 13 of 1000 |
| FP8 | 4.41 | 11.2597 | 95.98% | 14 of 50 | 68.10% | 46 of 1000 |

Perplexity: WikiText-2's test text, 64 windows of 1024 tokens; the top token compared at each of their positions.
Greedy answers: 50 fixed prompts through the chat template, 64 new tokens each, compared whole. MMLU: cais/mmlu's test
split in a fixed shuffle, 0-shot, the answer letter's logit (as `gpu/e2e.py --mmlu`). Weights GB: memory allocated
after loading (Glyd's includes its scratch; exact=True decodes each matrix whole into a larger one).
