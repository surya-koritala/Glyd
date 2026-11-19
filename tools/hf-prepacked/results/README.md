# Measured 2026-10-06: Runpod, NVIDIA A100-SXM4-80GB, glyd 0.29.0 (private main 63625822, its release libraries)

Each save: `python -m glyd_gpu pack` (glyd-v4, tiered); check.py (loaded as saved with verify, one greedy chat answer); `glyd verify`. Pods: hfpp2 (Qwen3.5-4B, gemma-4-E4B-it, Qwen3.5-9B), hfpp5 (gemma-4-12B-it, Qwen3.5-27B, Qwen3.8-27B), hfpp6 (gemma-4-31B-it); raw logs in the old scratchpad's ceiling/results-hfpp{2,5,6}.

| Model | bf16, the same tensors | Save on disk | Less | GPU memory loaded | Not packed (bf16) | Tensors checked | Whole bf16 checkpoint | Answer |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| Qwen/Qwen3.5-4B | 8.41 GB | 6.08 GB | 27.7% | 6.83 GB | 1.28 GB | 202 + 226 | 9.32 GB | A lossless compressor guarantees that the original data can ... |
| google/gemma-4-E4B-it | 15.88 GB | 12.98 GB | 18.3% | 12.08 GB | 7.01 GB | 595 + 1,484 | 15.99 GB | A lossless compressor guarantees that the original data can ... |
| Qwen/Qwen3.5-9B | 17.91 GB | 12.72 GB | 29.0% | 12.39 GB | 2.05 GB | 202 + 226 | 19.31 GB | A lossless compressor guarantees that the original data can ... |
| google/gemma-4-12B-it | 23.92 GB | 16.78 GB | 29.8% | 17.80 GB | 2.03 GB | 333 + 346 | 23.92 GB | A lossless compressor guarantees that the original data can ... |
| Qwen/Qwen3.8-27B | 53.79 GB | 37.02 GB | 31.2% | 36.54 GB | 2.60 GB | 402 + 450 | 55.56 GB | A lossless compressor guarantees that the original data can ... |
| Qwen/Qwen3.5-27B | 53.79 GB | 37.04 GB | 31.1% | 36.56 GB | 2.60 GB | 402 + 450 | 55.56 GB | A lossless compressor guarantees that the original data can ... |
| google/gemma-4-31B-it | 62.55 GB | 43.14 GB | 31.0% | 44.47 GB | 3.40 GB | 549 + 641 | 62.55 GB | A lossless compressor guarantees that the original data can ... |

- GPU memory: `torch.cuda.memory_allocated()` after `glyd.from_pretrained(save, layout="mma", verify=True)`, everything the load allocates.
- What the saves leave out of the original (tensor names compared with the Hub's headers): Qwen3.5/3.8, the vision tower and the MTP head (transformers' Qwen3_5ForCausalLM; the language model's tensors renamed `model.language_model.*` to `model.*`); gemma-4-E4B-it, the k_proj, v_proj and k_norm of its 18 KV-shared layers (54 tensors, 0.11 GB, unused by the model); gemma-4-12B-it, nothing (10 vision-embedder tensors renamed); gemma-4-31B-it, nothing.
- "Not packed": tensors saved in bf16 as they are. gemma-4-E4B-it's 7.0 GB of them (its per-layer embedding table, 5.6 GB, and its tied embedding, 1.3 GB; its audio and vision towers are packed) is why it saves 18%; Qwen3.5-4B's tied embedding (1.3 GB) why it saves 28%.

