# Glyd against bf16 on 2026 models: memory, users, tokens/s (2026-10-06)

bf16 is ESTIMATED from Glyd's own measurements (tools/report/report.py), not run for this report; where an earlier run on the same pod measured bf16, the measured number is shown instead and marked so. Internal: never published.

## Measured: Glyd (measured) against bf16 (estimated, or measured where marked)

Weights: as vLLM loads them per GPU ("Model loading took"); bf16 estimated from the checkpoint. Users at 8K: requests of 8,192 tokens the KV cache holds at once by vLLM's own accounting (a hybrid model's recurrent state counted as vLLM reserves it): **meas.** = vLLM's logged "Maximum concurrency" of a server started at max-len 8,192; **≈** = converted from a server started at another max-len (by one request's KV cost at each length; not a measurement); bf16 **est.** = Glyd's users scaled by the KV memory bf16's larger weights leave, at Glyd's settings (glyd serve's util 0.92: about 0.5 GiB more than vLLM's default 0.90 on a 24 GB GPU, 1.6 GiB on an H100). Tokens/s: output tokens a second of `vllm bench serve` (about 1,000 tokens in, 256 out) at N users at once; (speed1.py) = one streamed request instead, O / (TTFT + (O-1) / decode rate). The estimate's error where bf16 was not run is the held-out error below.

| Model | GPU | Format / Glyd layout | Weights GiB: Glyd meas. / bf16 est. | Saved GiB (%) | Users at 8K: Glyd / bf16 | Tok/s 1 user: Glyd meas. / bf16 (Glyd / bf16) | Tok/s 8 users: Glyd meas. / bf16 | Tok/s 32 users: Glyd meas. / bf16 | Tok/s 64 users: Glyd meas. / bf16 | Settings | Record |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| Qwen3.5-0.8B | RTX 4090 | bf16 / tiered | 1.38 / 1.72 | 0.34 (19.7%) | ≈37.3 / 34.8 est. | - | - | - | - | eager, max-len 262,144, util 0.35 | `benchmarks/gpu/v029-qwen35-9b-2026-10-05` |
| Qwen3.5-0.8B | RTX 5090 | bf16 / 12-bit | 1.48 / 1.72 | 0.24 (13.9%) | ≈37.5 / 35.7 est. | - | - | - | - | eager, max-len 262,144, util 0.27 | `benchmarks/gpu/v029-qwen35-9b-2026-10-05` |
| Qwen3.5-35B-A3B | H100 SXM | bf16 / 12-bit | 53.21 / 65.52 | 12.31 (18.8%) | ≈55.2 / 14.9 ≈ from bf16 meas. at 4,096 | - | - | - | - | compiled, max-len 262,144, 760 seqs, util 0.92 | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | bf16 / tiered | 12.69 / 17.66 | 4.97 (28.1%) | 19.3 meas. / 5.4 meas. | 74 / 54 meas. (1.36x) | 443 / 350 meas. (1.27x) | 988 / 550 meas. (1.80x) | 1,068 / 548 meas. (1.95x) | compiled, max-len 189,440, util 0.92 | pod results `results-m29a` (not a record yet); pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 5090 | bf16 / 12-bit | 13.89 / 17.66 | 3.77 (21.3%) | ≈34.1 / 23.6 ≈ from bf16 meas. at 4,096 | 110 / 92 meas. (1.20x) | - | - | - | compiled (speed1.py) | `benchmarks/gpu/v029-qwen35-9b-2026-10-05` |
| Qwen3.6-35B-A3B | H100 SXM | bf16 / 12-bit | 53.20 / 65.52 | 12.32 (18.8%) | 61.6 meas. / 18.0 meas. | - | - | - | - | eager, max-len 8,192, 32 seqs, util 0.90 | pod results `results-m29b` (not a record yet) |
| Qwen3.8-27B | H100 SXM | bf16 / 12-bit | 39.52 / 51.09 | 11.57 (22.6%) | 38.1 meas. / 23.6 meas. | 55 / 49 meas. (1.14x) | 355 / 330 meas. (1.07x) | 920 / 903 meas. (1.02x) | 1,256 / 1,270 meas. (0.99x) | compiled, max-len 262,144, 595 seqs, util 0.92 | pod results `results-m29b` (not a record yet); pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | L40S | bf16 / tiered | 35.60 / 51.09 | 15.49 (30.3%) | ≈3.4 / does not fit | 14 / does not fit | 93 / does not fit | 89 / does not fit | 88 / does not fit | compiled, max-len 16,384, 48 seqs, util 0.92 | pod results `results-sbl40s` (not a record yet) |
| gemma-4-12B-it | RTX 4090 | bf16 / tiered | 15.67 / 22.41 | 6.74 (30.1%) | ≈12.4 / does not fit | - | - | - | - | compiled, max-len 146,432, util 0.92, glyd 0.29+runfam | pod results `results-runfam26c` (not a record yet) |
| granite-4.2-8b | RTX 4090 | bf16 / tiered | 11.18 / 16.51 | 5.33 (32.3%) | 8.0 meas. / 3.9 meas. | - | - | - | - | compiled, max-len 50,176, util 0.92 | pod results `results-m29a` (not a record yet); pod results `results-runfam26` (not a record yet) |
| NVIDIA-Nemotron-3-Nano-4B-BF16 | RTX 4090 | bf16 / tiered | 6.06 / 7.53 | 1.47 (19.5%) | - | - | - | - | - | compiled, max-len 262,144, util 0.68, glyd 0.29+runfam | pod results `results-runfam26c` (not a record yet) |
| MiniCPM5-2B | RTX 4090 | bf16 / tiered | 3.31 / 4.82 | 1.51 (31.3%) | ≈53.3 / 48.7 est. | - | - | - | - | compiled, max-len 131,072, util 0.92 | pod results `results-runfam26` (not a record yet) |

## Reading the table

- **Memory**: Glyd saves 13.9% to 32.3% of the weights on the 12 measured rows: 12-bit layout 13.9% to 22.6% (Qwen3.5-0.8B on RTX 5090, Qwen3.5-35B-A3B on H100 SXM, Qwen3.5-9B on RTX 5090, Qwen3.6-35B-A3B on H100 SXM, Qwen3.8-27B on H100 SXM); tiered layout 19.5% to 32.3% (MiniCPM5-2B on RTX 4090, NVIDIA-Nemotron-3-Nano-4B-BF16 on RTX 4090, Qwen3.5-0.8B on RTX 4090, Qwen3.5-9B on RTX 4090, Qwen3.8-27B on L40S, gemma-4-12B-it on RTX 4090, granite-4.2-8b on RTX 4090).
- **Users at 8K**: Glyd holds 37.3 against 34.8 est. (Qwen3.5-0.8B, RTX 4090); 37.5 against 35.7 est. (Qwen3.5-0.8B, RTX 5090); 55.2 against 14.9 ≈ from bf16 meas. at 4,096 (Qwen3.5-35B-A3B, H100 SXM); 19.3 against 5.4 meas. (Qwen3.5-9B, RTX 4090); 34.1 against 23.6 ≈ from bf16 meas. at 4,096 (Qwen3.5-9B, RTX 5090); 61.6 against 18.0 meas. (Qwen3.6-35B-A3B, H100 SXM); 38.1 against 23.6 meas. (Qwen3.8-27B, H100 SXM); 8.0 against 3.9 meas. (granite-4.2-8b, RTX 4090); 53.3 against 48.7 est. (MiniCPM5-2B, RTX 4090). Rows marked ≈ are converted from another max-len, not measured.
- **bf16 users estimate against bf16 measured**, Glyd and bf16 at the same settings: -5.4% to +1.2% (4 pairs).
- **bf16 users estimate against bf16 measured**, Glyd and bf16 at other settings (max-len, seqs, eager): -28.0% to -0.9% (5 pairs). A negative error means Glyd's run kept more memory outside the KV cache than bf16's did, so bf16 est. users at such settings are low and Glyd's lead in users high by as much.
- **Tokens/s, bf16 measured on the same pod** (Qwen3.5-9B, RTX 4090, pod results `results-sb4090` (not a record yet)): Glyd / bf16 1.22x to 1.95x over 9 loads (1, 2, 4, 8, 16, 32, 64, 128, 256 users; lowest at 16); Glyd's time to first token is slower at 1, 2, 4, 8, 16 users.
- **Tokens/s, bf16 measured on the same pod** (Qwen3.5-9B, RTX 5090, `benchmarks/gpu/v029-qwen35-9b-2026-10-05`): Glyd / bf16 1.20x over 1 loads (1 users; lowest at 1); Glyd's time to first token is slower at 1 users.
- **Tokens/s, bf16 measured on the same pod** (Qwen3.8-27B, H100 SXM, pod results `results-sbh100` (not a record yet)): Glyd / bf16 0.99x to 1.14x over 9 loads (1, 2, 4, 8, 16, 32, 64, 128, 256 users; lowest at 64); Glyd's time to first token is slower at 1, 2, 4, 8, 16, 32, 64 users.
- **Tokens/s against the estimate**: the estimate's held-out error is mean 11%, worst 41% (by GPU and kind below), so a Glyd / bf16 est. ratio within that band of 1.00x is not a difference. No speed claim rests on an estimate.

## Projected: models not measured yet (memory only, from the inventory)

PROJECTED, not measured: loaded GiB = what vLLM loads in bf16; Glyd = the inventory's projected saving of the checkpoint with today's plugin (tiered on Ada; 12-bit on the other GPUs; on an A100/H100 a MoE's attention stays bf16) plus vLLM's buffers. Caveat: Qwen3.5-35B-A3B, the MoE measured on an H100, saved 18.8% there against 23.4% projected, so the A100/H100 column may be high for MoEs.

| Model | Loaded GiB (bf16) | Glyd GiB, tiered (Ada) | saved % | Glyd GiB, 12-bit (5090) | saved % | 12-bit on A100/H100 % |
| :-- | --: | --: | --: | --: | --: | --: |
| ATH-MaaS/Marco-Mini-Instruct | 32.26 | 21.82 | 32.5% | 24.31 | 24.7% | 24.5% |
| ATH-MaaS/Marco-Nano-Instruct | 15.03 | 10.19 | 32.5% | 11.35 | 24.7% | 24.2% |
| CohereLabs/North-Micro-Vision-Instruct | 4.76 | 3.66 | 23.7% | 3.92 | 18.0% | 18.0% |
| CohereLabs/North-Mini-Code-1.0 | 56.91 | 38.79 | 31.9% | 43.11 | 24.3% | 23.6% |
| CohereLabs/command-a-plus-05-2026-bf16 | 407.58 | 276.02 | 32.3% | 307.39 | 24.6% | 24.1% |
| IFM/K2-Horizon-0.9B | 2.14 | 1.49 | 32.5% | 1.64 | 24.8% | 24.8% |
| IFM/K2-Horizon-3.7B | 9.55 | 6.49 | 32.5% | 7.22 | 24.8% | 24.8% |
| IFM/K2-Horizon-32B | 64.91 | 43.86 | 32.5% | 48.88 | 24.8% | 24.8% |
| IFM/K2-Horizon-375B-A23B | 706.38 | 476.92 | 32.5% | 531.59 | 24.8% | 24.4% |
| IFM/K2-Horizon-7B | 16.89 | 11.44 | 32.5% | 12.74 | 24.8% | 24.8% |
| IFM/K2-Horizon-MoVA-36B-A4B | 69.88 | 47.22 | 32.5% | 52.62 | 24.7% | 18.7% |
| IQuestLab/IQuest-Coder-V1-14B-Instruct | 27.02 | 18.28 | 32.5% | 20.37 | 24.8% | 24.8% |
| JetBrains/Mellum2-12B-A2.5B-Instruct | 22.76 | 15.41 | 32.5% | 17.16 | 24.7% | 23.5% |
| Kwaipilot/KAT-Coder-V2.5-Dev | 64.69 | 44.03 | 32.0% | 48.96 | 24.4% | 23.4% |
| LGAI-EXAONE/EXAONE-4.5-33B | 63.11 | 42.64 | 32.5% | 47.53 | 24.8% | 24.8% |
| LGAI-EXAONE/K-EXAONE-2.0-750B-A37B | 1394.93 | 941.76 | 32.5% | 1049.72 | 24.8% | 24.4% |
| LiquidAI/LFM2-24B-A2B | 44.54 | 30.19 | 32.3% | 33.61 | 24.6% | 24.5% |
| LiquidAI/LFM2.5-1.2B-Instruct | 2.31 | 1.68 | 28.8% | 1.83 | 21.9% | 21.9% |
| LiquidAI/LFM2.5-2.6B | 5.15 | 3.68 | 29.3% | 4.03 | 22.3% | 22.3% |
| LiquidAI/LFM2.5-230M | 0.56 | 0.46 | 23.0% | 0.48 | 17.5% | 17.5% |
| LiquidAI/LFM2.5-350M | 0.79 | 0.62 | 26.3% | 0.66 | 20.1% | 20.1% |
| LiquidAI/LFM2.5-8B-A1B | 15.90 | 10.94 | 31.5% | 12.12 | 24.0% | 23.8% |
| LiquidAI/LFM2.5-VL-1.6B | 3.10 | 2.31 | 26.7% | 2.50 | 20.4% | 20.4% |
| LiquidAI/LFM2.5-VL-3B | 5.95 | 4.31 | 28.2% | 4.70 | 21.5% | 21.5% |
| LiquidAI/LFM2.5-VL-450M | 0.97 | 0.74 | 26.4% | 0.80 | 20.1% | 20.1% |
| MiniMaxAI/MiniMax-M3 | 795.64 | 787.29 | 1.1% | 789.28 | 0.8% | 0.8% |
| Nanbeige/Nanbeige4.1-3B | 7.46 | 5.08 | 32.5% | 5.64 | 24.8% | 24.8% |
| Nanbeige/Nanbeige4.2-3B | 7.90 | 5.37 | 32.5% | 5.97 | 24.8% | 24.8% |
| Qwen/Qwen3-Coder-Next | 148.54 | 100.72 | 32.2% | 112.12 | 24.5% | 24.1% |
| Qwen/Qwen3.5-122B-A10B | 228.42 | 154.80 | 32.2% | 172.35 | 24.6% | 23.7% |
| Qwen/Qwen3.5-27B | 51.09 | 35.40 | 30.8% | 39.14 | 23.4% | 23.4% |
| Qwen/Qwen3.5-2B | 4.25 | 3.22 | 24.9% | 3.47 | 19.0% | 19.0% |
| Qwen/Qwen3.5-397B-A17B | 739.23 | 499.84 | 32.4% | 556.90 | 24.7% | 24.2% |
| Qwen/Qwen3.5-4B | 8.59 | 6.23 | 27.9% | 6.79 | 21.2% | 21.2% |
| Qwen/Qwen3.6-27B | 51.09 | 35.40 | 30.8% | 39.14 | 23.4% | 23.4% |
| Qwen/Qwen3.8-2.4T-A95B | 4507.37 | 3043.87 | 32.5% | 3392.73 | 24.7% | 24.3% |
| Qwen/Qwen3.8-Flash-Next | 330.55 | 255.45 | 22.7% | 273.35 | 17.3% | 16.9% |
| Tele-AI/TeleChat3-Coder-36B-Thinking | 67.63 | 45.69 | 32.5% | 50.92 | 24.8% | 24.8% |
| XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | 17.66 | 12.67 | 28.5% | 13.86 | 21.7% | 21.7% |
| ai-sage/GigaChat3.1-10B-A1.8B-bf16 | 20.01 | 13.56 | 32.4% | 15.10 | 24.7% | 23.2% |
| ai-sage/GigaChat3.1-702B-A36B-bf16 | 1307.79 | 882.93 | 32.5% | 984.14 | 24.8% | 24.4% |
| ai21labs/AI21-Jamba2-3B | 6.08 | 5.00 | 18.2% | 5.26 | 13.8% | 13.8% |
| ai21labs/AI21-Jamba2-Mini | 96.19 | 67.08 | 30.3% | 74.02 | 23.1% | 23.0% |
| allenai/Olmo-Hybrid-7B | 13.97 | 9.71 | 30.8% | 10.73 | 23.4% | 23.4% |
| allenai/Olmo-Hybrid-Instruct-DPO-7B | 13.97 | 9.71 | 30.8% | 10.73 | 23.4% | 23.4% |
| arcee-ai/Trinity-Large-Thinking | 742.65 | 501.70 | 32.5% | 559.17 | 24.7% | 24.5% |
| bharatgenai/Param2-17B-A2.4B-Thinking | 32.08 | 21.70 | 32.5% | 24.17 | 24.8% | 24.4% |
| dots-studio/dots3-note-prev | 535.28 | 361.41 | 32.5% | 402.89 | 24.7% | 24.3% |
| google/diffusiongemma-26B-A4B-it | 48.23 | 46.78 | 3.0% | 47.12 | 2.3% | 2.3% |
| google/gemma-4-26B-A4B-it | 48.20 | 46.76 | 3.0% | 47.10 | 2.3% | 2.3% |
| google/gemma-4-31B-it | 58.38 | 39.80 | 31.9% | 44.23 | 24.3% | 24.3% |
| google/gemma-4-E2B-it | 9.64 | 6.60 | 32.0% | 7.32 | 24.4% | 24.4% |
| google/gemma-4-E4B-it | 14.92 | 10.18 | 32.0% | 11.31 | 24.4% | 24.4% |
| ibm-granite/granite-4.1-30b | 53.90 | 36.42 | 32.5% | 40.59 | 24.8% | 24.8% |
| ibm-granite/granite-4.1-3b | 6.47 | 4.41 | 32.5% | 4.90 | 24.8% | 24.8% |
| ibm-granite/granite-4.1-8b | 16.51 | 11.18 | 32.5% | 12.45 | 24.8% | 24.8% |
| ibm-granite/granite-4.2-30b | 54.66 | 36.94 | 32.5% | 41.17 | 24.8% | 24.8% |
| ibm-granite/granite-4.2-3b | 6.95 | 4.73 | 32.5% | 5.26 | 24.8% | 24.8% |
| ibm-granite/granite-swash-2b | 4.12 | 2.83 | 32.5% | 3.14 | 24.8% | 24.8% |
| ibm-granite/granite-swash-3b-a600m | 5.76 | 3.93 | 32.5% | 4.36 | 24.7% | 23.8% |
| ibm-granite/granite-vision-4.1-4b | 7.57 | 5.34 | 30.1% | 5.87 | 22.9% | 22.9% |
| inclusionAI/Ling-2.5-1T | 1885.56 | 1273.74 | 32.5% | 1419.67 | 24.7% | 24.1% |
| inclusionAI/Ling-2.6-flash | 194.19 | 131.53 | 32.3% | 146.47 | 24.6% | 24.0% |
| inclusionAI/Ling-3.0-flash | 231.87 | 156.83 | 32.4% | 174.72 | 24.7% | 24.2% |
| inclusionAI/Ling-3.0-flash-VL | 232.68 | 157.47 | 32.3% | 175.40 | 24.6% | 24.1% |
| inclusionAI/Ling-3.0-tiny | 14.83 | 10.22 | 31.4% | 11.32 | 23.9% | 22.7% |
| jdopensource/JoyAI-LLM-Flash | 91.29 | 61.67 | 32.5% | 68.74 | 24.7% | 24.2% |
| kakaocorp/kanana-2-30b-a3b-instruct-2601 | 57.26 | 38.70 | 32.5% | 43.13 | 24.7% | 23.7% |
| kakaocorp/kanana-2-3b-instruct | 6.67 | 4.54 | 32.5% | 5.05 | 24.8% | 24.8% |
| llm-jp/llm-jp-4-32b-a3b-thinking | 59.99 | 40.54 | 32.5% | 45.18 | 24.7% | 24.0% |
| llm-jp/llm-jp-4-33b-thinking | 62.01 | 41.90 | 32.5% | 46.69 | 24.8% | 24.8% |
| llm-jp/llm-jp-4-8b-instruct | 16.13 | 10.93 | 32.5% | 12.17 | 24.8% | 24.8% |
| llm-jp/llm-jp-4.1-32b-a3b-thinking | 59.99 | 40.54 | 32.5% | 45.18 | 24.7% | 24.0% |
| llm-jp/llm-jp-4.1-33b-thinking | 62.01 | 41.90 | 32.5% | 46.69 | 24.8% | 24.8% |
| llm-jp/llm-jp-4.1-8b-thinking | 16.13 | 10.93 | 32.5% | 12.17 | 24.8% | 24.8% |
| meituan-longcat/LongCat-Flash-Lite | 127.87 | 105.64 | 17.4% | 110.94 | 13.2% | 12.9% |
| meta-models/Muse-Glimmer-30B | 55.59 | 39.55 | 28.9% | 43.37 | 22.0% | 22.0% |
| microsoft/Phi-4-reasoning-vision-15B | 28.29 | 19.54 | 31.1% | 21.62 | 23.7% | 23.7% |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 224.89 | 220.46 | 2.0% | 221.52 | 1.5% | 1.5% |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | 1023.30 | 1003.55 | 1.9% | 1008.26 | 1.5% | 1.5% |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 58.95 | 57.83 | 1.9% | 58.10 | 1.4% | 1.4% |
| nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 | 140.49 | 135.80 | 3.3% | 136.91 | 2.5% | 2.5% |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 | 61.63 | 60.51 | 1.8% | 60.78 | 1.4% | 1.4% |
| openbmb/MiniCPM-V-4.6 | 2.47 | 1.94 | 22.6% | 2.06 | 17.2% | 17.2% |
| openbmb/MiniCPM-o-4_5 | 17.59 | 12.41 | 29.7% | 13.64 | 22.6% | 22.6% |
| openbmb/MiniCPM5-1B | 2.14 | 1.49 | 32.5% | 1.64 | 24.8% | 24.8% |
| ornith-ai/Ornith-1.5-35B-A3B | 65.52 | 44.67 | 31.9% | 49.64 | 24.3% | 23.4% |
| ornith-ai/Ornith-1.5-9B | 17.66 | 12.67 | 28.5% | 13.86 | 21.7% | 21.7% |
| poolside/Laguna-M.1 | 420.71 | 284.06 | 32.5% | 316.66 | 24.7% | 23.9% |
| poolside/Laguna-S-2.1 | 219.11 | 147.96 | 32.5% | 164.93 | 24.7% | 24.1% |
| poolside/Laguna-XS-2.1 | 62.42 | 42.19 | 32.5% | 47.02 | 24.7% | 23.7% |
| poolside/Laguna-XS.2 | 62.42 | 42.19 | 32.5% | 47.02 | 24.7% | 23.7% |
| sapientinc/HRM-Text-1B | 2.33 | 1.62 | 32.5% | 1.79 | 24.8% | 24.8% |
| stepfun-ai/Step-3.5-Flash | 366.99 | 253.59 | 30.9% | 280.63 | 23.5% | 23.0% |
| stepfun-ai/Step-3.7-Flash | 370.68 | 256.22 | 30.9% | 283.53 | 23.5% | 23.0% |
| stepfun-ai/Step3-VL-10B | 19.08 | 13.05 | 31.8% | 14.49 | 24.2% | 24.2% |
| swiss-ai/Apertus-v1.1-0.5B-Instruct | 1.20 | 0.85 | 32.5% | 0.93 | 24.8% | 24.8% |
| swiss-ai/Apertus-v1.1-1.5B-Instruct | 2.94 | 2.03 | 32.5% | 2.25 | 24.8% | 24.8% |
| swiss-ai/Apertus-v1.1-4B-Instruct | 7.26 | 4.94 | 32.5% | 5.49 | 24.8% | 24.8% |
| tencent/Hy3 | 549.67 | 371.40 | 32.4% | 413.94 | 24.7% | 24.2% |
| tencent/Hy4-preview | 1434.25 | 969.17 | 32.4% | 1080.02 | 24.7% | 24.0% |
| thinkingmachines/Inkling | 1764.27 | 1196.22 | 32.2% | 1331.71 | 24.5% | 24.3% |
| thinkingmachines/Inkling-Small | 491.36 | 333.48 | 32.1% | 371.15 | 24.5% | 24.3% |
| tiiuae/Falcon-H1-Tiny-R-0.6B | 1.29 | 1.08 | 18.1% | 1.13 | 13.8% | 13.8% |
| zai-org/GLM-4.7-Flash | 55.90 | 37.78 | 32.5% | 42.11 | 24.7% | 23.9% |
| zai-org/GLM-5 | 1385.77 | 935.58 | 32.5% | 1042.96 | 24.7% | 24.3% |
| zai-org/GLM-5.3-BF16 | 1384.78 | 934.91 | 32.5% | 1042.08 | 24.8% | 24.3% |
| zai-org/GLM-5.3-Flash-BF16 | 584.80 | 396.07 | 32.3% | 441.09 | 24.6% | 24.2% |

## How good is the bf16 estimate

Constants fitted on 30 bf16 points measured on 2026 models (6 runs): bandwidth efficiency 0.98, decode saturation 0.11 (an empirical term, not a compute crossover), prompt compute 0.72, a step's fixed time 3.39 ms. Recurrent state is taken as float32, read and written each step (not confirmed from vLLM 0.30).

- In-sample (fitted and checked on the same points): mean 6%, worst 15% (30 points).
- **Held-out** (each run predicted by constants fitted without it; the error band to use): mean 11%, worst 41% (30 points).
  - by GPU: H100 SXM mean 11%, worst 41% (20 points) from 4 run(s); RTX 4090 mean 11%, worst 13% (9 points) from 1 run(s); RTX 5090 mean 16%, worst 16% (1 points) from 1 run(s).
  - by kind: MoE mean 15%, worst 41% (9 points) from 1 run(s); hybrid mean 10%, worst 16% (21 points) from 5 run(s).
  - Extrapolated: no bf16 speed was measured on L40S; MoE estimates rest on 1 MoE run(s).

Every bf16 point, with Glyd where the same pod measured it (TTFT p50 ms, Glyd / bf16):

| Model | GPU | Users | bf16 meas. tok/s | bf16 est. in-sample | held-out | Glyd meas. | Glyd / bf16 meas. | TTFT ms Glyd / bf16 | Source |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| Qwen3.5-9B | RTX 5090 | 1 | 92 | 77 (-15%) | 77 (-16%) | 110 | 1.20x | 131 / 113 | `benchmarks/gpu/v029-qwen35-9b-2026-10-05` |
| Qwen3.5-9B | RTX 4090 | 1 | 54 | 50 (-9%) | 48 (-11%) | 74 | 1.36x | 165 / 145 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 2 | 101 | 96 (-4%) | 93 (-7%) | 136 | 1.36x | 253 / 218 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 4 | 189 | 181 (-4%) | 173 (-9%) | 250 | 1.32x | 458 / 383 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 8 | 350 | 324 (-7%) | 304 (-13%) | 443 | 1.27x | 734 / 561 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 16 | 564 | 534 (-5%) | 489 (-13%) | 690 | 1.22x | 904 / 733 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 32 | 550 | 534 (-3%) | 489 (-11%) | 988 | 1.80x | 925 / 7,936 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 64 | 548 | 534 (-2%) | 489 (-11%) | 1,068 | 1.95x | 1,359 / 22,772 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 128 | 549 | 534 (-3%) | 489 (-11%) | 1,003 | 1.83x | 17,290 / 50,028 | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 256 | 553 | 534 (-3%) | 489 (-11%) | 1,007 | 1.82x | 49,262 / 107,738 | pod results `results-sb4090` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 1 | 49 | 52 (+6%) | 53 (+8%) | 55 | 1.14x | 194 / 164 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 2 | 93 | 101 (+9%) | 103 (+11%) | 105 | 1.13x | 287 / 252 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 4 | 179 | 195 (+9%) | 198 (+11%) | 198 | 1.11x | 466 / 409 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 8 | 330 | 362 (+10%) | 369 (+12%) | 355 | 1.07x | 817 / 690 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 16 | 570 | 633 (+11%) | 648 (+14%) | 591 | 1.04x | 1,497 / 1,346 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 32 | 903 | 1,013 (+12%) | 1,040 (+15%) | 920 | 1.02x | 2,457 / 2,153 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 64 | 1,270 | 1,237 (-3%) | 1,205 (-5%) | 1,256 | 0.99x | 3,078 / 2,873 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 128 | 1,253 | 1,249 (-0%) | 1,216 (-3%) | 1,349 | 1.08x | 3,358 / 14,081 | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 256 | 1,278 | 1,249 (-2%) | 1,216 (-5%) | 1,418 | 1.11x | 25,002 / 33,536 | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 1 | 196 | 188 (-4%) | 252 (+28%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 2 | 298 | 332 (+11%) | 422 (+41%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 4 | 517 | 540 (+4%) | 641 (+24%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 8 | 774 | 802 (+4%) | 886 (+15%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 16 | 1,206 | 1,112 (-8%) | 1,158 (-4%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 32 | 1,752 | 1,517 (-13%) | 1,516 (-13%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 64 | 1,668 | 1,709 (+2%) | 1,690 (+1%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 128 | 1,752 | 1,709 (-2%) | 1,690 (-4%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 256 | 1,724 | 1,709 (-1%) | 1,690 (-2%) | - | - | - | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 256 | 1,203 | 1,240 (+3%) | 1,250 (+4%) | - | - | - | pod results `results-rpq38` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 256 | 1,208 | 1,235 (+2%) | 1,242 (+3%) | - | - | - | pod results `results-rpq38` (not a record yet) |

bf16 users at the bf16 run's own max-len (vLLM's logged "Maximum concurrency"), estimated from the same pod's Glyd run; settings that differ between the two are listed (other max-len: converted by one request's KV cost; eager or not, seqs: other memory beside the KV cache):

| Model | GPU | Glyd run (version, settings) | bf16 settings | bf16 meas. users | bf16 est. | error | Source |
| :-- | :-- | :-- | :-- | --: | --: | --: | :-- |
| Qwen3.5-9B | RTX 5090 | 0.29, compiled, max-len 262,144, util 0.92 | compiled, max-len 4,096, util 0.92 | 37.14 at 4,096 | 36.80 | -0.9% | `benchmarks/gpu/v029-qwen35-9b-2026-10-05` |
| Qwen3.8-27B | H100 SXM | 0.29, eager, max-len 8,192, 32 seqs, util 0.90 | eager, max-len 8,192, 32 seqs, util 0.90 | 23.59 at 8,192 | 23.86 | +1.2% | pod results `results-m29b` (not a record yet) |
| Qwen3.6-35B-A3B | H100 SXM | 0.29, eager, max-len 8,192, 32 seqs, util 0.90 | eager, max-len 8,192, 32 seqs, util 0.90 | 18.00 at 8,192 | 17.96 | -0.2% | pod results `results-m29b` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 0.29, compiled, max-len 189,440, util 0.92 | compiled, max-len 4,096, 95 seqs, util 0.92 | 7.14 at 4,096 | 5.14 | -28.0% | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 0.29, compiled, max-len 189,440, util 0.92 | compiled, max-len 4,096, 128 seqs, util 0.92 | 6.79 at 4,096 | 5.14 | -24.3% | pod results `results-sb4090` (not a record yet) |
| Qwen3.5-9B | RTX 4090 | 0.29, eager, max-len 8,192, 32 seqs, util 0.90 | eager, max-len 8,192, 32 seqs, util 0.90 | 5.36 at 8,192 | 5.23 | -2.5% | pod results `results-m29a` (not a record yet) |
| granite-4.2-8b | RTX 4090 | 0.29, eager, max-len 8,192, 32 seqs, util 0.90 | eager, max-len 8,192, 32 seqs, util 0.90 | 3.90 at 8,192 | 3.69 | -5.4% | pod results `results-m29a` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 0.29, compiled, max-len 262,144, 595 seqs, util 0.92 | compiled, 256 seqs, util 0.92 | 1.10 at 262,144 | 1.06 | -3.4% | pod results `results-sbh100` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 0.29, compiled, max-len 262,144, 760 seqs, util 0.92 | compiled, max-len 4,096, 128 seqs, util 0.92 | 20.90 at 4,096 | 16.21 | -22.5% | pod results `results-sbh100` (not a record yet) |

Noise floor: 2 bf16 servers of Qwen3.5-9B on RTX 4090 at identical settings (compiled, max-len 4,096, 16 seqs, util 0.94, pod results `results-e35r1` (not a record yet)) had 25.37, 25.37 GiB of KV memory: 0% apart from run to run.

Noise floor: 2 bf16 servers of Qwen3.5-9B on RTX 4090 at identical settings (compiled, max-len 4,096, 16 seqs, util 0.94, pod results `results-e35r2` (not a record yet)) had 2.94, 2.94 GiB of KV memory: 0% apart from run to run.

Noise floor: 2 bf16 servers of Qwen3.8-27B on H100 SXM at identical settings (compiled, max-len 4,096, 160 seqs, util 0.90, pod results `results-rpq38` (not a record yet)) had 18.64, 17.21 GiB of KV memory: 8% apart from run to run.

bf16 weights, estimated against measured (vLLM's "Model loading took"):

| Model | GPU | measured GiB | estimated GiB | error | Source |
| :-- | :-- | --: | --: | --: | :-- |
| Qwen3.5-9B | RTX 4090 | 17.66 | 17.66 | -0.0% | `benchmarks/gpu/v029-qwen35-9b-2026-10-05` |
| Qwen3.5-9B | RTX 5090 | 17.66 | 17.66 | -0.0% | `benchmarks/gpu/v029-qwen35-9b-2026-10-05` |
| Qwen3.5-9B | RTX 4090 | 16.80 | 16.81 | +0.0% | pod results `results-e35r1` (not a record yet) |
| Qwen3.5-0.8B | RTX 4090 | 1.72 | 1.72 | -0.1% | pod results `results-e35r1` (not a record yet) |
| gemma-4-26B-A4B-it | H100 SXM | 48.54 | 48.20 | -0.7% | pod results `results-m29b` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 51.10 | 51.09 | -0.0% | pod results `results-m29b` (not a record yet) |
| gemma-4-31B-it | H100 SXM | 58.99 | 58.38 | -1.0% | pod results `results-m29b` (not a record yet) |
| Qwen3.6-35B-A3B | H100 SXM | 65.53 | 65.52 | -0.0% | pod results `results-m29b` (not a record yet) |
| Qwen3.5-9B | RTX 5090 | 16.80 | 16.81 | +0.0% | pod results `results-ts5090f` (not a record yet) |
| gemma-4-E4B-it | RTX 4090 | 15.08 | 14.92 | -1.1% | pod results `results-m29a` (not a record yet) |
| NVIDIA-Nemotron-3-Nano-4B-BF16 | RTX 4090 | 7.47 | 7.53 | +0.8% | pod results `results-m29a` (not a record yet) |
| granite-4.2-8b | RTX 4090 | 16.42 | 16.51 | +0.5% | pod results `results-m29a` (not a record yet) |
| Qwen3.5-35B-A3B | H100 SXM | 65.53 | 65.52 | -0.0% | pod results `results-sbh100` (not a record yet) |
| Qwen3.8-27B | H100 SXM | 50.22 | 50.23 | +0.0% | pod results `results-rpq38` (not a record yet) |

## Notes (by hand; everything above is generated from the runs)

- **35B-A3B MoEs on the H100** save 18.8%, under the inventory's projection for that GPU (23.4%); the gap is not explained yet.
- **Qwen3.5-35B-A3B on the H100 (results-sbh100)**: Glyd's server started (its memory is in the table) but the pod stopped before
  Glyd's bench ran, and gemma-4-26B-A4B was not reached; the tok/s dashes on that row are not a failed bench.
- **Not in the table, from tonight's v0.29 sweep (results-m29a, results-m29b)**: Glyd 0.29.0 did not start on gemma-4-12B,
  gemma-4-E4B, gemma-4-26B-A4B and gemma-4-31B (transformers 5.18's AmbiguousGlobalPerLayerAttributeError on a per-layer
  attribute), and refused NVIDIA-Nemotron-3-Nano-4B on an RTX 4090 with an estimate of 33.6 GiB packed for a model bf16 loads in
  7.47 GiB. The gemma-4-12B and Nemotron-3-Nano rows above come from the branch build of PR #21 (glyd 0.29+runfam).
- **Qwen3.5-122B-A10B on 4 x H100 (results-bigmoe-4xh100)**: the download finished and bf16's check server started; no weights,
  KV or Glyd numbers were written, so the model stays in the projected table.
- Rows marked "pod results ... (not a record yet)" come from tonight's pods; their records are not in benchmarks/gpu yet.
