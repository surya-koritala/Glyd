# Quantized 2026 checkpoints under vLLM: stock against Glyd, measured (2026-10-06)

Private report (glyd-gpu-private). Every row of the first table was **measured** on one Runpod pod: stock vLLM 0.30.0 and vLLM with
`--quantization glyd` on the same checkpoint, the same GPU, one after the other. No bf16 baseline was run (quantized checkpoints
have none). Rows for checkpoints that fit none of the GPUs allowed tonight (RTX 4090, RTX 5090, L40S) are **projected** in the
second table, from the sampled bytes of `research/quant-2026`, and say so.

Two Glyd builds, each on an open PR, neither merged nor a default:

- **16-bit parts** (PR #42, branch `quant-16bit-parts`; four commits made these records: `5c7ce578` the L40S run 1, `fd1c2b45` the
  RTX 4090, `8da693ae` the RTX 5090 runs 4 and 5 and the L40S run 2a, `8d498a63` the L40S run 2, the head; each row names its own in the
  Commit column): the checkpoint's quantized
  layers run with vLLM's own method, exactly as stock runs them; Glyd packs the 16-bit parts the checkpoint keeps (the Linears and
  experts left bf16, LM heads, vision-tower Linears where vLLM hands the tower the quantization config) as it packs a bf16 model
  (the tiered layout on Ada, the 12-bit one on the RTX 5090). It runs in two configurations, and the table shows both: the **fused**
  run (vLLM's default: compiled, CUDA graphs, Glyd's fused products) gives the memory and tok/s cells, and its outputs are **not**
  stock's bits (PR #42's `quant_parts.py`: "fused products are not stock's bits: agreement"); the **exact** run (eager, every product
  from the decoded matrix) is what the bit-for-bit verdict was established on, it holds more memory (its own column), and its speed
  was not measured.
- **FP8 codes** (PR #37, branch `qf8-codes`, `5e1446c6` on the L40S, `a2907d7d` on the RTX 5090, `39400c40` on the RTX 4090): the FP8
  weights themselves held in the lossless FP8 format (6.7 bits a value) and decoded bit for bit into a scratch before vLLM's own FP8
  kernel runs; a mixture of experts' experts packed too, only the routed ones decoded. Opt-in (`GLYD_FP8=1`, development builds): slower
  than stock at every load measured.

## Measured

Columns: weights as vLLM logs them ("Model loading took"); users at 8K = vLLM's own "Maximum concurrency" for 8,192-token requests
(its KV cache in tokens beside it; where a run was started at another context, that context is named); tokens a second = output tokens of
all requests together, 512 random prompt tokens in and 256 out for the 16-bit parts build (the median of 2), 1,024 in and 256 out for the
FP8 codes build (the median of 3); "-" = not measured at that load; "†" = KV-bound: that run's KV cache holds fewer tokens than the
load's requests need (users x (prompt + output)), so the number is queueing, not throughput at that load. Exact-mode weights = the
weights of the run the bit-for-bit verdict comes from (the 16-bit parts build's exact eager run; the FP8 codes build's one run is exact
mode, so its cell repeats the Glyd weights). Bit for bit says what was compared and how. The fused run against stock = for the 16-bit
parts build, how far the benchmarked (fused) run's greedy tokens follow stock's on the 8 prompts (quant_parts.py's agreement: the share
of each prompt's positions equal before its first difference, averaged; 1.00 = no difference in 64 tokens), and how many packs that run
verified against their weights as it made them (0 before `8d498a63`: those commits verified only in the exact run). The Glyd build
cell names the build's benchmark and exact-check settings; they differ between the builds, so stock's own numbers differ between a
16-bit parts row and an FP8 codes row of the same checkpoint and GPU (Qwen3.6-35B-A3B-FP8 on the L40S: stock 966 tok/s at 32 users
in one, 735 in the other): compare stock and Glyd within a row, never across builds. Each tok/s median has its runs' range in brackets
(shown where the runs differ by 1 tok/s or more); "within noise" = stock's range and Glyd's overlap, so the row says neither above nor
below stock at that load. Commit = the commit of the run's pod and, where the exact check ran on another pod, of that one. What Glyd packs =
the fused run's inventory (the exact run's is one layer fewer where it leaves the LM head: see Memory below). Ceiling = what
`research/quant-2026` measured as left in the whole stored checkpoint (fixed-rate layouts / the best contexts), of which these builds
take the 16-bit parts and, on FP8 checkpoints, the FP8 codes.

| Checkpoint | GPU | Glyd build | Format | Weights GiB, stock / Glyd | Saved GiB (%) | Exact-mode weights GiB (saved %) | Users at 8K (KV tokens), stock / Glyd | tok/s 1 user, stock / Glyd | tok/s 8 users, stock / Glyd | tok/s 32 users, stock / Glyd | tok/s 64 users, stock / Glyd | Bit for bit | The fused run against stock | What Glyd packs | Ceiling (quant-2026): fixed rate / best contexts | Commit (run / exact check) | Record |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | RTX 4090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT8 W8A8 | 12.73 / 10.96 | 1.77 (13.9%) | 11.58 (9.0%) | 15.3 (125,114 tokens) / 20.3 (166,074 tokens) | 74 / 84 (1.13x) | - / 569 | 1,418 [1,417-1,418] / 1,533 (1.08x) | - / 2,127 | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 131 of 131 packs verified against their weights) | greedy agreement 0.26; 0 of 132 packs verified as packed | 132 layers, 5.49 GiB bf16 -> 3.68 GiB | 24.2% / 26.0% | fd1c2b45 | `records/rtx4090-qbm/RedHatAI_Qwen3.5-9B-quantized.w8a8` |
| google/gemma-4-12B-it-qat-w4a16-ct | RTX 4090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT4 QAT (compressed-tensors) | 8.28 / 7.63 | 0.65 (7.9%) | 8.26 (0.2%) | 4.1 (33,498 tokens) / 4.4 (35,682 tokens) | 92 / 98 (1.06x) | - / 561 | 1,117 [1,117-1,117] / 1,146 (1.03x) | - / 1,001 [1,000-1,001]† | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 1 of 1 packs verified against their weights) | greedy agreement 1.00; 0 of 3 packs verified as packed | 3 layers, 1.92 GiB bf16 -> 1.26 GiB | 17.0% / 23.6% | fd1c2b45 | `records/rtx4090-qbm/google_gemma-4-12B-it-qat-w4a16-ct` |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | RTX 4090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT4 AWQ (compressed-tensors), fp16 parts | 19.20 / 18.25; stock did not start; Glyd did not start | - | - | - / - | - | - | - | - | not run | - | - | 7.8% / 16.3% | fd1c2b45 | `records/rtx4090-qbm/cyankiwi_Qwen3.6-27B-AWQ-INT4` |
| nvidia/Qwen3.8-27B-NVFP4 | RTX 5090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | NVFP4 + FP8 attention | 19.92 / 19.78 | 0.14 (0.7%) | 19.81 (0.6%) | 6.7 (54,452 tokens) / 6.8 (55,898 tokens) | 68 / 66 (0.98x) | - / 457 | 796 / 785 (0.99x) | - / 996 | with FlashInfer's NVFP4 GEMM (the configuration benchmarked), 1 of 2 Glyd exact runs differs: the run's first one (its stock eager failed to start) equals the re-check's stock eager in every field; the re-check's own: FAIL: Glyd exact eager differs from stock eager (first at long_logprobs [520][0][1]); with vLLM's CUTLASS NVFP4 GEMM (FlashInfer's NVFP4 kernels off by VLLM_DISABLED_KERNELS, stock and Glyd; speed not measured in it): bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 83 of 83 packs verified against their weights) (`records/rtx5090-qbm-run5/nvidia_Qwen3.8-27B-NVFP4`) | greedy agreement 0.71; 0 of 83 packs verified as packed | 83 layers, 0.60 GiB bf16 -> 0.45 GiB | 14.4% / 16.7% | 8da693ae | `records/rtx5090-qbm-run4/nvidia_Qwen3.8-27B-NVFP4` (exact check: `records/rtx5090-qbm-run4/nvidia_Qwen3.8-27B-NVFP4-recheck`) |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | RTX 5090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | NVFP4 + FP8 mixer | 17.82 / 17.74 | 0.08 (0.4%) | 17.74 (0.4%) | 62.5 (512,000 tokens) / 63.2 (517,734 tokens) | 313 / 315 (1.01x) | - / 1,273 [1,236-1,309] | 2,810 [2,659-2,960] / 2,809 [2,746-2,873] (1.00x, within noise) | - / 3,752 [3,659-3,844] | FAIL: Glyd exact eager differs from stock eager (first at tokens [0][24]); stock eager also differs from itself here (first at tokens [0][24]), so not passed and not attributed (layer-level check pending) | greedy agreement 0.64; 0 of 12 packs verified as packed | 12 layers, 0.26 GiB bf16 -> 0.20 GiB | 10.4% / 13.1% | 8da693ae | `records/rtx5090-qbm-run4/nvidia_NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | RTX 5090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | NVFP4 + FP8 attention | 20.37 / 20.24 | 0.13 (0.6%) | 20.26 (0.5%) | 18.4 (150,966 tokens) / 18.8 (153,892 tokens) | 260 [259-260] / 254 [252-255] (0.98x) | - / 1,242 [1,236-1,248] | 2,777 [2,774-2,780] / 2,780 [2,772-2,787] (1.00x, within noise) | - / 3,443 [3,429-3,456] | FAIL: Glyd exact eager differs from stock eager (first at tokens [0][28]); stock eager also differs from itself here (first at tokens [0][28]), so not passed and not attributed (layer-level check pending) | greedy agreement 0.70; 0 of 83 packs verified as packed | 83 layers, 0.57 GiB bf16 -> 0.43 GiB | 10.2% / 12.1% | 8da693ae | `records/rtx5090-qbm-run5/nvidia_Qwen3.6-35B-A3B-NVFP4` |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | RTX 5090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT4 AWQ | 22.23 / 21.29 | 0.94 (4.2%) | 21.84 (1.8%) | 12.2 (100,059 tokens) / 15.1 (123,465 tokens) | 198 / 215 (1.09x) | - / 1,040 [1,037-1,043] | 2,399 [2,382-2,416] / 2,357 [2,346-2,368] (0.98x) | - / 2,335 [2,316-2,355] | FAIL: Glyd exact eager differs from stock eager (first at tokens [0][11]); stock eager also differs from itself here (first at tokens [0][13]), so not passed and not attributed (layer-level check pending) | greedy agreement 0.29; 0 of 244 packs verified as packed | 244 layers, 4.14 GiB bf16 -> 3.12 GiB | 13.4% / 20.1% | 8da693ae | `records/rtx5090-qbm-run4/QuantTrio_Qwen3.6-35B-A3B-AWQ` |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | RTX 5090 | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT8 W8A8 | 12.73 / 11.41; stock did not start; Glyd did not start | - | - | - / - | - | - | - | - | not run | - | - | 24.2% / 26.0% | 8da693ae | `records/rtx5090-qbm-run4/RedHatAI_Qwen3.5-9B-quantized.w8a8` |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | RTX 5090 | FP8 codes (1,024 in, 256 out, median of 3; exact: compiled, top-20) | FP8, a scale a row | 25.80 / 22.94 | 2.86 (11.1%) | 22.94 (same run: exact mode) | 1.9 (7,867 tokens) at 4,096 / 5.3 (21,567 tokens) at 4,096 | 203 [96-204] / 125 [76-125] (0.61x, within noise) | - / 529 [519-532] | 736 [719-744]† / 626 [619-642]† (0.85x) | - | bit for bit: logits (tokens, their log probabilities, top-20 prompt log probabilities of 4 prompts) in the compiled run; the first batch's greedy tokens at 1 and 32 users | - | 150 layers packed (30 of them experts' layers): 19.15 GiB held for 22.82 GiB | 15.2% / 21.0% | a2907d7d | `records/rtx5090-qf8-run2/RedHatAI__gemma-4-26B-A4B-it-FP8-dynamic` |
| Qwen/Qwen3.8-27B-FP8 | RTX 5090 | FP8 codes (1,024 in, 256 out, median of 3; exact: compiled, top-20) | FP8, block scales | - / -; stock did not start; Glyd did not start | - | - | - / - | - | - | - | - | not run | - | - | 19.4% / 21.7% | a2907d7d | `records/rtx5090-qf8-run2/Qwen__Qwen3.8-27B-FP8` |
| Qwen/Qwen3.8-27B-FP8 | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | FP8, block scales | 28.90 / 27.99 | 0.91 (3.1%) | 28.74 (0.6%) | 10.5 (86,256 tokens) / 11.6 (94,930 tokens) | 25 / 26 (1.03x) | - | 388 / 395 (1.02x) | - | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 83 of 83 packs verified against their weights) | greedy agreement 0.87; 84 of 84 packs verified as packed | 84 layers, 2.97 GiB bf16 -> 1.99 GiB | 19.4% / 21.7% | 8d498a63 / 5c7ce578 | `records/l40s-qbp-run2/Qwen_Qwen3.8-27B-FP8` (exact check: `records/l40s-qbp-run1/Qwen_Qwen3.8-27B-FP8`) |
| Qwen/Qwen3.8-27B-FP8 | L40S | FP8 codes (1,024 in, 256 out, median of 3; exact: compiled, top-20) | FP8, block scales | 28.90 / 25.63 | 3.27 (11.3%) | 25.63 (same run: exact mode) | 12.8 (104,740 tokens) / 17.5 (143,360 tokens) | 25 / 11 (0.45x) | - / 75 | 292 [292-293] / 197 (0.67x) | - | bit for bit: logits (tokens, their log probabilities, top-20 prompt log probabilities of 4 prompts) in the compiled run; the first batch's greedy tokens at 1 and 32 users | - | 256 layers packed (0 of them experts' layers): 19.10 GiB held for 22.66 GiB | 19.4% / 21.7% | 5e1446c6 | `records/l40s-qf8-run5/Qwen__Qwen3.8-27B-FP8` |
| Qwen/Qwen3.6-35B-A3B-FP8 | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | FP8, block scales | 34.25 / 33.76 | 0.49 (1.4%) | 34.11 (0.4%) | 10.9 (89,526 tokens) / 12.6 (103,570 tokens) | 134 [133-134] / 140 [140-141] (1.05x) | - | 966 [959-973] / 971 [963-979] (1.01x, within noise) | - | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 83 of 83 packs verified against their weights) | greedy agreement 0.81; 84 of 84 packs verified as packed | 84 layers, 1.52 GiB bf16 -> 1.02 GiB | 17.5% / 19.9% | 8d498a63 / 5c7ce578 | `records/l40s-qbp-run2/Qwen_Qwen3.6-35B-A3B-FP8` (exact check: `records/l40s-qbp-run1/Qwen_Qwen3.6-35B-A3B-FP8`) |
| Qwen/Qwen3.6-35B-A3B-FP8 | L40S | FP8 codes (1,024 in, 256 out, median of 3; exact: compiled, top-20) | FP8, block scales | 34.25 / 30.15 | 4.10 (12.0%) | 30.15 (same run: exact mode) | 13.9 (113,943 tokens) / 32.7 (268,101 tokens) | 133 [133-134] / 85 (0.64x) | - / 238 [233-249] | 735 [723-747] / 431 [419-443] (0.59x) | - | FAIL at 32 users (exact-bench: the first batch's greedy tokens differ; the 4 prompts' logits bit for bit; the same-blocks rerun that tells vLLM's block admission from Glyd is not in this record) | - | 200 layers packed (40 of them experts' layers): 26.32 GiB held for 31.31 GiB | 17.5% / 19.9% | 5e1446c6 | `records/l40s-qf8-run5/Qwen__Qwen3.6-35B-A3B-FP8` |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | FP8, a scale a row | 25.80 / 25.19 | 0.61 (2.4%) | 25.63 (0.7%) | 6.8 (55,378 tokens) / 7.2 (58,592 tokens) | 111 / 120 (1.08x) | - | 1,045 [1,028-1,062] / 1,059 [1,055-1,064] (1.01x, within noise) | - | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 136 of 136 packs verified against their weights) | greedy agreement 0.51; 0 of 138 packs verified as packed | 138 layers, 1.90 GiB bf16 -> 1.27 GiB | 15.2% / 21.0% | 5c7ce578 | `records/l40s-qbp-run1/RedHatAI_gemma-4-26B-A4B-it-FP8-dynamic` |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | L40S | FP8 codes (1,024 in, 256 out, median of 3; exact: compiled, top-20) | FP8, a scale a row | 25.80 / 22.94 | 2.86 (11.1%) | 22.94 (same run: exact mode) | 7.1 (58,052 tokens) / 8.7 (71,285 tokens) | 107 [78-107] / 80 [62-80] (0.75x, within noise) | - / 241 [233-256] | 869 [840-877] / 501 [473-508] (0.58x) | - | bit for bit: logits (tokens, their log probabilities, top-20 prompt log probabilities of 4 prompts) in the compiled run; the first batch's greedy tokens at 1 and 32 users | - | 150 layers packed (30 of them experts' layers): 19.15 GiB held for 22.82 GiB | 15.2% / 21.0% | 5e1446c6 | `records/l40s-qf8-run5/RedHatAI__gemma-4-26B-A4B-it-FP8-dynamic` |
| RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8 | L40S | FP8 codes (1,024 in, 256 out, median of 3; exact: compiled, top-20) | FP8, a scale a tensor | 30.20 / 26.84 | 3.36 (11.1%) | 26.84 (same run: exact mode) | 77.1 (631,808 tokens) / 111.8 (915,456 tokens) | 142 / 92 (0.65x) | - / 208 [207-213] | 922 [894-950] / 429 [411-446] (0.47x) | - | bit for bit: logits (tokens, their log probabilities, top-20 prompt log probabilities of 4 prompts) in the compiled run; the first batch's greedy tokens at 1 and 32 users | - | 127 layers packed (23 of them experts' layers): 24.14 GiB held for 28.74 GiB | not sampled; FP8 codes 16.6% / 20.1% of their bytes | 5e1446c6 | `records/l40s-qf8-run5/RedHatAI__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8` |
| nvidia/Gemma-4-31B-IT-NVFP4 | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | NVFP4, bf16 attention | 31.18 / 25.25 | 5.93 (19.0%) | 26.11 (16.3%) | 1.7 (13,941 tokens) / 3.4 (28,121 tokens) | 22 / 26 (1.20x) | - | 231 [230-231]† / 422 [421-423] (1.83x) | - | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 120 of 120 packs verified against their weights) | greedy agreement 0.64; 0 of 122 packs verified as packed | 122 layers, 18.62 GiB bf16 -> 12.54 GiB | 23.2% / 24.5% | 8da693ae / 5c7ce578 | `records/l40s-qbp-run2a/nvidia_Gemma-4-31B-IT-NVFP4` (exact check: `records/l40s-qbp-run1/nvidia_Gemma-4-31B-IT-NVFP4`) |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | NVFP4 + FP8 attention | 20.37 / 20.22 | 0.15 (0.7%) | 20.20 (0.8%) | 59.3 (485,668 tokens) / 59.6 (488,009 tokens) | 165 [164-165] / 164 [163-165] (1.00x, within noise) | - | 1,345 [1,323-1,367] / 1,353 [1,350-1,356] (1.01x, within noise) | - | FAIL: Glyd exact eager differs from stock eager (first at tokens [0][28]); stock eager also differs from itself here (first at tokens [0][28]), so not passed and not attributed (layer-level check pending) | greedy agreement 0.67; 83 of 83 packs verified as packed | 83 layers, 0.57 GiB bf16 -> 0.39 GiB | 10.2% / 12.1% | 8d498a63 | `records/l40s-qbp-run2/nvidia_Qwen3.6-35B-A3B-NVFP4` |
| google/gemma-4-12B-it-qat-w4a16-ct | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT4 QAT (compressed-tensors) | 8.28 / 7.63 | 0.65 (7.9%) | 8.26 (0.2%) | 11.2 (91,998 tokens) / 11.5 (94,109 tokens) | 75 / 81 (1.08x) | - | 1,028 / 1,072 [1,072-1,073] (1.04x) | - | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 1 of 1 packs verified against their weights) | greedy agreement 1.00; 0 of 3 packs verified as packed | 3 layers, 1.92 GiB bf16 -> 1.26 GiB | 17.0% / 23.6% | 5c7ce578 | `records/l40s-qbp-run1/google_gemma-4-12B-it-qat-w4a16-ct` |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT4 AWQ (compressed-tensors), fp16 parts | 19.20 / 18.27 | 0.93 (4.8%) | 19.04 (0.8%) | 21.5 (176,368 tokens) / 22.6 (185,524 tokens) | 39 / 40 (1.04x) | - | 521 [519-522] / 531 [530-532] (1.02x) | - | bit for bit: 8 prompts' greedy tokens and top-5 logprobs, a 700-token prompt's logprobs, an image chat (stock eager vs Glyd exact eager; 83 of 83 packs verified against their weights) | greedy agreement 1.00; 0 of 84 packs verified as packed | 84 layers, 2.97 GiB bf16 -> 1.99 GiB | 7.8% / 16.3% | 5c7ce578 | `records/l40s-qbp-run1/cyankiwi_Qwen3.6-27B-AWQ-INT4` |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | L40S | 16-bit parts (512 in, 256 out, median of 2; exact: eager, top-5) | INT4 GPTQ | 21.12 / 19.83 | 1.29 (6.1%) | 20.63 (2.3%) | 55.4 (453,485 tokens) / 59.4 (486,253 tokens) | 117 / 145 (1.24x) | - | 1,321 [1,316-1,325] / 1,394 [1,393-1,394] (1.06x) | - | FAIL: Glyd exact eager differs from stock eager (first at tokens [0][5]); stock eager also differs from itself here (first at tokens [0][5]), so not passed and not attributed (layer-level check pending) | greedy agreement 0.60; 0 of 244 packs verified as packed | 244 layers, 4.14 GiB bf16 -> 2.80 GiB | 17.6% / 20.7% | 5c7ce578 | `records/l40s-qbp-run1/Qwen_Qwen3.5-35B-A3B-GPTQ-Int4` |

(The RTX 4090 rows of the qrp session's pod `qrp-4090` — RedHatAI/Qwen3.5-9B-FP8-dynamic with both builds, QuantTrio/Qwen3.5-9B-AWQ, stock at 8 and 64 users for the W8A8 and Gemma-4-12B QAT rows — are not here: the pod was deleted by its launcher's exit trap while still installing vLLM, before any row; `job/` holds the job to run again, and `records/rtx4090-qrp/` and the table take its rows when they land.)

### Reading the table

- **Memory.** The saved column is the fused run's. The 16-bit parts build saves what the checkpoint keeps 16-bit: 19.0% of the loaded
  weights on Gemma-4-31B-IT-NVFP4 (its attention is bf16), 13.9% on Qwen3.5-9B W8A8, 3-8% on the INT4 and block-FP8 checkpoints, under
  1% on the NVFP4 MoEs. In the exact configuration, the one verified bit for bit, it holds more: 16.3% on Gemma-4-31B-IT-NVFP4,
  9.0% on the W8A8 (11.58 GiB against 12.73), 0.2-2.3% on every other row (the exact-mode column). Most of that gap is the LM head:
  the fused runs pack it (`language_model.lm_head`, 1.89 GiB bf16 on the W8A8 checkpoint, 2.37 GiB on Qwen3.8-27B-FP8, 2.62 GiB on
  Gemma-4-31B), the exact runs that were verified bit for bit do not (131 against 132 layers on the W8A8 row, 120 against 122 on
  Gemma-4-31B, 83 against 84 on Qwen3.8-27B-FP8), so the LM head's saving is not yet of a verified configuration. The embedding
  (`VocabParallelEmbedding`, built without the quantization config, so vLLM's unquantized method; the same bytes again) is left
  unpacked on every row in both runs, and 27 vision-tower Linears under GlydLinearMethod keep 0.25 GiB of bf16 weight on the Qwen
  rows. That is the gap to the ceiling: the W8A8 row's exact run packs 3.59 GiB of its 8.7 GB of 16-bit parts and sits at 9.0%
  against 24.2%. The FP8 codes build saves 11-12% of the loaded weights on every FP8 checkpoint (15.7-16.1% of the FP8 bytes
  themselves; the scratches cost the largest Linear and one layer's experts). The two builds are not yet one (the composite config is
  the qcf session's work), so their savings are not added in any row. Only the exact-mode column is a saving of a configuration
  verified bit for bit (0.2-16.3% of the loaded weights on the rows that passed); the fused column's 0.4-19.0% is of a configuration
  whose outputs are not stock's bits. Gemma-4-31B-IT-NVFP4's 19.0% (25.25 GiB) is from `8da693ae` (run 2a); that commit's exact mode
  was not run on this checkpoint: the bit-for-bit verdict and the 26.11 GiB (16.3%) are run 1's at `5c7ce578`.
- **Users.** The KV cache grows by what the weights lose, more than proportionally where the model is large against the GPU
  (Gemma-4-31B-IT-NVFP4 on the L40S: 1.7 to 3.4 users of 8K; Qwen3.6-35B-A3B-FP8 with the FP8 codes: 13.9 to 32.7).
- **Speed.** All tok/s cells are the fused run's; the exact configuration's speed was not measured. Stock was run at 1 and 32 users
  only. The 16-bit parts build, against the runs' ranges: above stock at every load measured on 7 rows (both RTX 4090 rows;
  Qwen3.8-27B-FP8, Gemma-4-31B-IT-NVFP4, gemma-4-12B QAT, cyankiwi AWQ and the GPTQ MoE on the L40S); within noise at 32 users on
  Nemotron-3.5-Lightning NVFP4 and Qwen3.6-35B-A3B-NVFP4 on the RTX 5090, Qwen3.6-35B-A3B-FP8 and gemma-4-26B FP8 on the L40S, and at
  both loads on Qwen3.6-35B-A3B-NVFP4 on the L40S (its 1.01x at 32 users is inside stock's 1,323-1,367); below stock, outside the runs'
  ranges, on 4 cells: nvidia/Qwen3.8-27B-NVFP4 on the RTX 5090 at 1 user (0.98x) and 32 (0.99x), nvidia/Qwen3.6-35B-A3B-NVFP4 on the
  RTX 5090 at 1 user (254 against 260, 0.98x; run 5, vLLM's CUTLASS NVFP4 GEMM) and QuantTrio/Qwen3.6-35B-A3B-AWQ on the RTX 5090 at 32
  (0.98x). Each median is of 2 runs, a thin base for any of these verdicts. The 1.83x at 32 users on Gemma-4-31B-IT-NVFP4 is bounded by
  stock's KV cache (13,941 tokens for 32 requests of 768: stock queues, Glyd's 28,121 do not), not kernel speed; the gemma-4-12B QAT
  64-user cell on the RTX 4090 (1,001, below its 32-user 1,146) is the same effect on Glyd's side (35,682 tokens for 49,152 needed), and
  both gemma-4-26B FP8 cells at 32 users on the RTX 5090 are KV-bound (†). The FP8 codes build is below stock at 32 users on every row
  (0.47-0.85x); at 1 user its ranges include a slow first run of 3 on both sides of the gemma-4-26B rows (stock 96 against 203-204 on
  the RTX 5090), so those cells read "within noise" though the two later runs of each side are 0.61x and 0.75x apart. Each product
  reads the pack, writes the weight out and reads it again; `f8_linear`, the product straight from the pack, is slower than vLLM's own
  FP8 kernels on every shape measured at 32-64 tokens, so not faster at every load (at 1-8 tokens on the RTX 4090 at or under them on some
  shapes; `benchmarks/gpu/fp8-speed-consumer-2026-10-06`). It stays opt-in. Nothing in this table qualifies
  as a default: the configuration benchmarked is not the one verified bit for bit, the verified one's speed is not measured, and stock
  was not run at 8 and 64 users.
- **Bit for bit.** Of the 22 rows: 12 bit for bit, 7 FAIL, 3 not run (neither stock nor Glyd started). For the 16-bit parts build
  the verdict is the exact eager run's, not the benchmarked run's: stock vLLM eager against Glyd exact eager (8 prompts' greedy tokens
  and top-5 logprobs, a 700-token prompt's logprobs at every position, an image chat), with every pack decoded as it is made and
  compared with its weights (131 of 131, 83 of 83, ...). The fused run the memory and tok/s cells come from gives outputs that are not
  stock's bits: its greedy tokens follow stock's for 0.26-1.00 of the 64 tokens (the column), and its four checked products differ from
  F.linear on the decoded matrix by up to 3.6e-3 relative; in the records before `8d498a63` (`fd1c2b45`, `8da693ae`, `5c7ce578`) that
  run verified no pack as it made them (0 of N), only the exact run did. The FP8 codes build's one run is exact mode: the verdict
  compares the compiled runs' logits (tokens, their log probabilities, the top-20 prompt log probabilities of 4 prompts) with
  inductor's deterministic mode in both, then the first batch's greedy tokens at 1 and 32 users (exact-bench); Qwen/Qwen3.6-35B-A3B-FP8
  on the L40S FAILs the 32-user one (its record is at `5e1446c6`, before `39400c40`'s same-blocks rerun that tells vLLM's block
  admission from Glyd). The 7 FAILs:
  - 5 where stock vLLM eager also differs from itself on the same checkpoint (Qwen3.5-35B-A3B-GPTQ-Int4, QuantTrio's
    Qwen3.6-35B-A3B-AWQ, Nemotron-3.5-Lightning NVFP4, Qwen3.6-35B-A3B-NVFP4 on the L40S and on the RTX 5090: Triton's autotuned
    choices and non-deterministic MoE kernels). The records print EXACT FAIL and they count as not passed; the cell gives where Glyd
    first differs and where stock's second run does. On QuantTrio's AWQ Glyd differs first (tokens [0][11], stock against itself at
    [0][13]), and Glyd was compared with the first stock run only. The layer-level bit-exact check for these is the `moe-exact`
    session's; its verdicts go into this table when they land.
  - nvidia/Qwen3.8-27B-NVFP4 on the RTX 5090 with FlashInfer's NVFP4 GEMM, vLLM's default there and the configuration the row's memory
    and tok/s come from: 1 of 2 Glyd exact runs differs from stock (the 700-token prompt's logprobs at position 520) while stock eager
    equals itself. With vLLM's CUTLASS NVFP4 GEMM (`VLLM_DISABLED_KERNELS` naming FlashInfer's NVFP4 kernels, stock and Glyd; run 5,
    pod `qbm-5090b`) the same checkpoint is bit for bit, 83 of 83 packs verified; that configuration's speed was not measured.
  - Qwen/Qwen3.6-35B-A3B-FP8 with the FP8 codes on the L40S at 32 users, above.
- **Fits.** Qwen3.8-27B-FP8 on a 32 GB RTX 5090 at utilization 0.95 (`records/rtx5090-qf8-run1`, PR #37's `rtx5090-a/` at `1959f272`):
  stock runs out of memory while vLLM profiles it; with the FP8 codes build it loads (25.15 GiB) and profiles with KV room for 39
  sequences, then stops because the run asked for 64 (vLLM: "max_num_seqs (64) exceeds available Mamba cache blocks (39)"). So it fits
  only with Glyd there, but no run served it: one at 39 sequences or fewer was not made. At the default 0.9 neither starts (the row's
  record `records/rtx5090-qf8-run2`, PR #37's `rtx5090-b/`: stock "No available memory for the cache blocks"). cyankiwi's Qwen3.6-27B-AWQ-INT4 does
  not start on a 24 GB RTX 4090 either way (19.2 GiB of weights, then 12.25 GiB for its vision tower's profile). Qwen3.5-9B W8A8 does not
  run on the RTX 5090 at all: vLLM 0.30's CUTLASS INT8 kernel has no SM120 build (stock and Glyd alike).

## Projected (not measured: too big for the GPUs allowed tonight)

From `research/quant-2026` (2026-10-06): the stored bytes sampled by HTTP range, projected group by group; "today's builds" = the 16-bit
parts plus the FP8 codes where the checkpoint has FP8, in both of Glyd's layouts, since `best_layout` picks the tiered one (32.6% of the
16-bit parts) on Ada only and the 12-bit one (25%) on every other GPU, and none of these but RedHatAI/Qwen3.8-27B-INT4 can run on an Ada
card; "all fixed-rate layouts" adds the MXFP4 and NVFP4 scale layouts, which are computed from the samples, not built. The FP8 codes are
taken at the format's 16.6%; the measured build pays its scratches and gets 15.7-16.1% of the FP8 bytes (Mistral-Small-4: about 0.6-1.1
GB less than the 19.3 GB below). The FP8 codes are opt-in (`GLYD_FP8=1`) and slower than stock wherever measured, so the "of which"
column is not a default saving: without them DeepSeek-V4.1-Flash keeps 1.3 GB (0.3%, tiered) / 1.0 GB (12-bit) and Mistral-Small-4
1.0 GB (0.8%, tiered) / 0.8 GB (0.7%, 12-bit). Downloads: Hugging Face's last-30-days and all-time counts read 2026-10-06 (`downloads.json`; the all-time column of
`research/quant-2026` was read earlier the same day). Nothing in this table ran on a GPU.

| Checkpoint (projected) | Format | Downloads, last 30 days / all time | Stored GB | Saved, today's builds: tiered layout (Ada) | Saved, today's builds: 12-bit layout (A100, H100, Blackwell) | Of which the FP8 codes (opt-in) | Saved, all fixed-rate layouts | Best contexts |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moonshotai/Kimi-K3 | MXFP4 | 1,265,400 / 5,271,523 | 1,560.9 | 36.6 GB (2.3%) | 28.6 GB (1.8%) | - | 100.8 GB (6.5%) | 14.1% |
| deepseek-ai/DeepSeek-V4.1-Flash | MXFP4 + FP8 | 869,321 / 869,321 | 510.3 | 33.3 GB (6.5%) | 33.0 GB (6.5%) | 32.0 GB (6.3%) | 51.0 GB (10.0%) | 15.2% |
| amd/MiniMax-M3-MXFP4 | MXFP4 | 83,735 / 609,609 | 242.7 | 7.0 GB (2.9%) | 5.3 GB (2.2%) | - | 17.1 GB (7.0%) | 16.3% |
| mistralai/Mistral-Small-4-119B-2603 | FP8, a scale a tensor | 78,463 / 670,091 | 120.9 | 20.3 GB (16.8%) | 20.1 GB (16.6%) | 19.3 GB (16.0%) | 20.3 GB (16.8%) | 19.4% |
| RedHatAI/Qwen3.8-27B-INT4 | INT4 (compressed-tensors) | 573,721 / 740,121 | 19.5 | 2.3 GB (11.8%) | 1.7 GB (8.9%) | - | 3.8 GB (19.3%) | 22.4% |

(RedHatAI/Qwen3.8-27B-INT4 fits an L40S (tiered) and an RTX 5090 (12-bit) but was not run tonight; its 16-bit parts are 35.5% of the
file. The three MXFP4 models need 4 to 20 data-center GPUs, where the 12-bit column applies.)

## What was not measured, and why

- RTX 4090 rows for the FP8 checkpoints of 26B and up: none fits 24 GB, stock or Glyd (the smallest Glyd weights are 22.9 GiB before
  vLLM's CUDA context). The one FP8 checkpoint on the RTX 4090 is Qwen3.5-9B's: Qwen publishes no FP8 twin of Qwen3.5-9B, so the most
  downloaded 2026 one, RedHatAI/Qwen3.5-9B-FP8-dynamic (2026-05-11, 51,752 downloads, a scale a row), stands for it.
- Stock vLLM at 8 and 64 users on the L40S and RTX 5090 rows: stock was run briefly (1 and 32 users) in the sibling sessions' pods, as
  agreed; the qrp session's RTX 4090 pod was to run stock at 1, 8, 32 and 64 and was deleted before it did.
- The exact configuration's speed (the 16-bit parts build in exact mode): not measured by any session; the tok/s cells are the fused
  run's, whose outputs are not stock's bits.
- Data-center GPUs (A100, H100): refused tonight.
- The composite (16-bit parts and FP8 codes in one run): not built yet (the qcf session); no row adds the two.

## Records and reproduction

`records/<gpu>-<session>-<run>/<checkpoint>/` holds each run's JSON (`stock.json`, `glyd.json`, `glyd-exact.json`, `summary.json`;
`check_fp8_vllm.json` and `check.txt` for the FP8 codes build), the pod's `steps.txt`, `env.txt`, `gpu.txt` and the commit (`COMMIT`),
copied from the pods' results (`results-qbm-4090`, `results-qbm-5090-run4`, `results-qbm-5090b` as `rtx5090-qbm-run5` with its vLLM
logs, which name the NVFP4 kernel each run used, `results-qbp-l40s2`, `results-qf8-l40s5`,
`results-qf8-5090b` and the qrp session's `results-qrp-4090`) and from PR #42's `benchmarks/gpu/quant-16bit-parts-2026-10-06/l40s-run1`
and its run 2a; vLLM's own logs stay with those. `python3 table.py measured.json` prints the first table and writes every number it
is made of; where a row's exact check was run on another pod (the Record column's second directory), `COMMIT` in each record directory
names the commit of that pod. The pod jobs: PR #42's `benchmarks/gpu/quant-16bit-parts-2026-10-06/job.sh` (`gpu/vllm/quant_parts.py`), PR #37's
`benchmarks/gpu/fp8-in-vllm-2026-10-06/job/q8_job.sh` (`gpu/fp8/check_fp8_vllm.py`), the qrp session's `job/job.sh` (the two in one
pod: the 16-bit parts build at commit `8d498a63`, then the FP8 codes build at `39400c40`, `job/qpenv`), and the qbm session's
`job/qbm/job.sh` (PR #42's job with stock at 1 and 32 users beside the exact check, Glyd alone at 1, 8, 32 and 64; `qpenv-4090`,
`qpenv-5090` and `qpenv-5090b` the three pods' settings, `recheck.sh` the RTX 5090 run 4 re-check). Its plugin-test gate let two
failures through on the RTX 5090 pods (`test_vllm_fp8.py`'s FP8 GPU rule, `test_vllm.py`'s `test_kv_refusals`; `records/*/tests.txt`).
PR #42 and PR #37 are not merged: their files named here are on those branches at the commits in the Commit column.

Sessions beside this one on 2026-10-06 whose pods made these records: quant-16bit-parts (qbp), fp8-in-vllm (qf8), the quantized
benchmarks (qbm), fp8-speed-consumer (fp8s). Spend: the qrp session's RTX 4090 pod about $0.10 (pod created 12:21 UTC, RTX 4090 at $0.74/h, cap $4); the qbm session's six pods
$2.95 by the guard's ledger (`rp_budget/ledger.tsv`), one more RTX 5090 pod that never reached ssh up to $0.25; the other siblings'
pods are in their own records.
