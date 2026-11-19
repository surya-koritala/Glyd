# Open-weight language models released in 2026

Research of 2026-10-06 (Hugging Face API; release date = the repo's `createdAt`). 185 first-party model family sizes from 63 makers: **164 with bf16 weights** (released in bf16, or with a first-party bf16 copy), **12 FP8-only originals**, **9 FP4 or mixed-precision originals**. Data: `models.json` (one record per row, every field below plus dtype shares, quantization config, layer types, head size and the other repos of the family).

Scope: one row per original first-party family size (base and instruct, format twins and small variants of the same weights listed under `also`); first-party quantized copies are format twins, not rows. Not rows: finetunes, depth-upscaled or continued-training models built on another maker's 2026 model (on Qwen3.5: Ornith, Hemmingway-1, Nex-N2/N2.5, Intern-S2, KAT-Coder-V2.5, Agnes-2.5-Pro, Agnes-3.0-Flash, matilda-jev, Index-*, MiniCPM-V-4.6, ZDTaichu5.0, Clef, MiMo-V2.6-Distill; on DeepSeek-V4: Nex-N2.5-Max; on GLM-5: Atria-Dawn) or on a model released before 2026 (Keye-VL-2.0, MiniCPM-o-4.5, Mage-VL, Step3-VL-10B, gpt-oss-puzzle, the Nemotron-Labs Elastic/Cascade/Teacher checkpoints, Holotron4, gnani-evon, Antares, VibeThinker); new checkpoints of families first released before 2026 (Trinity-Mini/Nano, K-EXAONE-236B, Falcon-H1R-7B, EuroLLM, ALIA, Bielik, Molmo2-based models); research ablations (allenai EMO/StdMoE/BAR, marin delphi, IFM LoopedLM, Dayhoff, SparseLM); drafters (DSpark, EAGLE, assistant and MTP drafter repos); narrow-task models (OCR, ASR, TTS, translation, embedding, reranking, guard, robotics, image or video generation).
Makers with no new open-weight language model in 2026 as of this date: OpenAI, xAI, Apple, Snowflake, Nous Research, Prime Intellect, Hugging Face, Baidu (image and OCR models only), ByteDance Seed (only Stable-DiffCoder). Meta's 2026 open release is Muse Glimmer (Muse Spark 1.2 weights announced, not published). StepFun Step-5 Preview and Fireworks Ember-1 are API-only.

Columns: params from the safetensors header totals (or the name, `-A..B` for active; MoE active otherwise estimated from config); `GiB on Hub` = sum of the release's `.safetensors` files; Features from `config.json` (text_config included): MoE, hybrid/linear attention (layer types such as linear_attention, mamba, KDA), a vision tower, an audio tower, MTP layers; `bf16 GiB` = the bf16 release (or its first-party bf16 copy, named in brackets; half the file size for the few released in fp32); `vLLM 0.30` = the architecture is in `vllm/model_executor/models/registry.py` at tag v0.30.0.

## 1. bf16 originals (164)

| Repo | Maker | Released | Total B | Active B | Format | bf16 GiB | Arch | Features | vLLM 0.30 | License | Downloads |
| :-- | :-- | :-- | --: | --: | :-- | --: | :-- | :-- | :-- | :-- | --: |
| allenai/Olmo-Hybrid-Instruct-DPO-7B (+3) | AI2 | 2026-02-20 | 7.43 | 7.43 | bf16 | 13.8 | OlmoHybridForCausalLM | hybrid/linear attn (linear_attention) | yes | apache-2.0 | 775 |
| ai21labs/AI21-Jamba2-3B | AI21 | 2026-01-06 | 3.2 | 3.2 | bf16 | 6.0 | JambaForCausalLM | hybrid/linear attn (mamba) | yes | apache-2.0 | 21,925 |
| ai21labs/AI21-Jamba2-Mini (+1) | AI21 | 2026-01-06 | 51.57 | 12.11 | bf16 | 96.1 | JambaForCausalLM | MoE, hybrid/linear attn (mamba) | yes | apache-2.0 | 676 |
| amd/Instella-MoE-16B-A3B-Think (+3) | AMD | 2026-07-23 | 16 | 3 | bf16 | 29.5 | InstellaMoEForCausalLM | MoE | no | other | 2,850 |
| Aleph-Alpha/Kolibri-1 (+1) | Aleph Alpha | 2026-10-02 | 78.1 | 3.79 | fp8 e4m3 block-scaled 128x128 | 145.5 (Kolibri-1-BF16) | Kolibri1ForCausalLM | MoE | no | apache-2.0 | 2,453 |
| ATH-MaaS/Marco-Nano-Instruct (+1) | Alibaba AIDC | 2026-04-03 | 8 | 0.6 | bf16 | 14.9 | Qwen3MoeForCausalLM | MoE, Qwen3-MoE architecture | yes | apache-2.0 | 372 |
| ATH-MaaS/Marco-Mini-Instruct (+2) | Alibaba AIDC | 2026-04-08 | 17.25 | 0.87 | bf16 | 32.1 | Qwen3MoeForCausalLM | MoE, Qwen3-MoE architecture | yes | apache-2.0 | 206 |
| Qwen/Qwen3-Coder-Next (+2) | Alibaba Qwen | 2026-01-30 | 79.67 | 3.87 | bf16 | 148.4 | Qwen3NextForCausalLM | MoE, hybrid/linear attn (linear attention) | yes | apache-2.0 | 596,359 |
| Qwen/Qwen3.5-397B-A17B (+1) | Alibaba Qwen | 2026-02-16 | 397 | 17 | bf16 | 751.4 | Qwen3_5MoeForConditionalGeneration | MoE, hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 448,069 |
| Qwen/Qwen3.5-27B (+1) | Alibaba Qwen | 2026-02-24 | 27.78 | 27.78 | bf16 | 51.7 | Qwen3_5ForConditionalGeneration | hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 1,888,406 |
| Qwen/Qwen3.5-35B-A3B (+3) | Alibaba Qwen | 2026-02-24 | 35 | 3 | bf16 | 67.0 | Qwen3_5MoeForConditionalGeneration | MoE, hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 1,537,762 |
| Qwen/Qwen3.5-122B-A10B (+1) | Alibaba Qwen | 2026-02-24 | 122 | 10 | bf16 | 233.0 | Qwen3_5MoeForConditionalGeneration | MoE, hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 555,162 |
| Qwen/Qwen3.5-4B (+1) | Alibaba Qwen | 2026-02-27 | 4.66 | 4.66 | bf16 | 8.7 | Qwen3_5ForConditionalGeneration | hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 7,968,478 |
| Qwen/Qwen3.5-9B (+1) | Alibaba Qwen | 2026-02-27 | 9.65 | 9.65 | bf16 | 18.0 | Qwen3_5ForConditionalGeneration | hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 8,827,840 |
| Qwen/Qwen3.5-0.8B (+1) | Alibaba Qwen | 2026-02-28 | 0.87 | 0.87 | bf16 | 1.6 | Qwen3_5ForConditionalGeneration | hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 2,539,302 |
| Qwen/Qwen3.5-2B (+1) | Alibaba Qwen | 2026-02-28 | 2.27 | 2.27 | bf16 | 4.2 | Qwen3_5ForConditionalGeneration | hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 4,784,610 |
| Qwen/Qwen3.6-35B-A3B (+1) | Alibaba Qwen | 2026-04-15 | 35 | 3 | bf16 | 67.0 | Qwen3_5MoeForConditionalGeneration | MoE, hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 3,411,526 |
| Qwen/Qwen3.6-27B (+1) | Alibaba Qwen | 2026-04-21 | 27.78 | 27.78 | bf16 | 51.7 | Qwen3_5ForConditionalGeneration | hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 2,384,695 |
| Qwen/Qwen3.8-27B (+1) | Alibaba Qwen | 2026-08-05 | 27.78 | 27.78 | bf16 | 51.7 | Qwen3_5ForConditionalGeneration | hybrid/linear attn (linear_attention), vision, MTP x1 | yes | apache-2.0 | 6,758,884 |
| Qwen/Qwen3.8-2.4T-A95B (+1) | Alibaba Qwen | 2026-08-08 | 2400 | 95 | bf16 | 4,556.4 | Qwen3_5MoeForCausalLM | MoE, hybrid/linear attn (linear_attention), MTP x1 | yes | other | 32,817 |
| Qwen/Qwen3.8-Flash-Next (+1) | Alibaba Qwen | 2026-08-24 | 180 |  | bf16 | 335.3 | Qwen4ExpForConditionalGeneration | MoE, hybrid/linear attn (linear_attention), vision, MTP x1 | yes | other | 1,530,359 |
| inclusionAI/LLaDA2.1-mini | Ant inclusionAI | 2026-02-09 | 16.26 | 1.43 | bf16 | 30.3 | LLaDA2MoeModelLM | MoE, text diffusion LM | no | apache-2.0 | 115,647 |
| inclusionAI/LLaDA2.1-flash | Ant inclusionAI | 2026-02-09 | 102.89 | 6.15 | bf16 | 191.6 | LLaDA2MoeModelLM | MoE, text diffusion LM | no | apache-2.0 | 1,139 |
| inclusionAI/Ming-flash-omni-2.0 | Ant inclusionAI | 2026-02-10 | 104.25 | 7.51 | bf16 | 214.4 | BailingMM2NativeForConditionalGeneration | MoE, vision, audio | no | mit | 4,476 |
| inclusionAI/Ling-2.5-1T | Ant inclusionAI | 2026-02-14 | 1012.23 | 63.58 | bf16 | 1,885.4 | BailingMoeV2_5ForCausalLM | MoE | yes | mit | 810 |
| inclusionAI/Ling-2.6-flash (+2) | Ant inclusionAI | 2026-04-28 | 107.49 | 10.76 | bf16 | 200.2 | BailingMoeV2_5ForCausalLM | MoE, MTP x1 | yes | mit | 4,444 |
| inclusionAI/LLaDA2.2-flash | Ant inclusionAI | 2026-07-16 | 102.89 | 6.15 | bf16 | 191.6 | LLaDA2MoeModelLM | MoE, text diffusion LM | no | apache-2.0 | 707 |
| inclusionAI/Ling-3.0-flash (+3) | Ant inclusionAI | 2026-08-02 | 127.49 | 8.58 | bf16 | 237.5 | BailingMoeV3ForCausalLM | MoE, hybrid/linear attn (kda (linear attention)), MTP x1 | yes | mit | 13,183 |
| inclusionAI/Ling-3.0-tiny (+2) | Ant inclusionAI | 2026-08-10 | 7.89 | 1.38 | bf16 | 14.7 | BailingMoeV3ForCausalLM | MoE, hybrid/linear attn (kda (linear attention)) | yes | mit | 16,312 |
| inclusionAI/Ling-3.0-flash-VL (+2) | Ant inclusionAI | 2026-09-04 | 124.85 | 5.94 | bf16 | 232.6 | BailingMoeV3VLForConditionalGeneration | MoE, hybrid/linear attn (kda (linear attention)), vision | yes | mit | 15,775 |
| inclusionAI/LLaDA2.2-mini | Ant inclusionAI | 2026-09-05 | 16.26 | 1.43 | bf16 | 30.3 | LLaDA2MoeModelLM | MoE, text diffusion LM | no | apache-2.0 | 1,424 |
| arcee-ai/Trinity-Large-Thinking (+4) | Arcee AI | 2026-04-01 | 398.64 | 13.37 | bf16 | 742.5 | AfmoeForCausalLM | MoE | yes | other | 1,725 |
| bharatgenai/Param2-17B-A2.4B-Thinking | BharatGen | 2026-02-16 | 17 | 2.4 | bf16 | 31.9 | Param2MoEForCausalLM | MoE | yes |  | 5,411 |
| ByteDance-Seed/Stable-DiffCoder-8B-Instruct (+1) | ByteDance Seed | 2026-01-15 | 8.25 | 8.25 | bf16 | 15.4 | StableDiffcoderForCausalLM | text diffusion code LM | no | mit | 557 |
| Tele-AI/TeleChat3-Coder-36B-Thinking | China Telecom | 2026-01-29 | 36.24 | 36.24 | bf16 | 67.5 | TeleChat3ForCausalLM |  | yes |  | 66 |
| XingChen-AGI/Xing4.0-29B-A4B (+1) | China Telecom | 2026-09-16 | 29 | 4 | bf16 | 58.1 | Xing4_0ForCausalLM | MoE, MTP x1 | no | apache-2.0 | 50,497 |
| CohereLabs/tiny-aya-global (+7) | Cohere | 2026-02-13 | 3.35 | 3.35 | bf16 | 6.2 | Cohere2ForCausalLM | config.json gated | yes | cc-by-nc-4.0 (gated) | 9,046 |
| CohereLabs/command-a-plus-05-2026-bf16 (+2) | Cohere | 2026-05-11 | 218.75 | 25.48 | bf16 | 407.5 | Cohere2VisionForConditionalGeneration | MoE, vision | yes | apache-2.0 | 35,838 |
| CohereLabs/North-Mini-Code-1.0 (+1) | Cohere | 2026-06-05 | 30.48 | 3.31 | bf16 | 56.8 | Cohere2MoeForCausalLM | MoE | yes | apache-2.0 | 9,268 |
| CohereLabs/North-Micro-Vision-Instruct | Cohere | 2026-08-10 | 2.48 | 2.48 | bf16 | 4.6 | CohereCompassForConditionalGeneration | vision | yes | apache-2.0 | 167,351 |
| EssentialAI/rnj-1.5-instruct | Essential AI | 2026-04-22 | 8.31 | 8.31 | fp32 | 15.5 | Rnj1ForCausalLM | post-trained on its 2025 base | yes | apache-2.0 | 891 |
| Gensyn/open-1b-sft (+1) | Gensyn | 2026-09-15 | 1.61 | 1.61 | fp32 | 3.0 | Open1BForCausalLM |  | no | apache-2.0 | 1,300 |
| google/gemma-4-E2B-it (+1) | Google | 2026-03-02 | 5.12 | 5.12 | bf16 | 9.5 | Gemma4ForConditionalGeneration | vision, audio | yes | apache-2.0 | 3,025,976 |
| google/gemma-4-E4B-it (+1) | Google | 2026-03-02 | 8 | 8 | bf16 | 14.9 | Gemma4ForConditionalGeneration | vision, audio | yes | apache-2.0 | 4,438,133 |
| google/gemma-4-26B-A4B-it (+1) | Google | 2026-03-11 | 26 | 4 | bf16 | 48.1 | Gemma4ForConditionalGeneration | MoE, vision, audio | yes | apache-2.0 | 12,724,590 |
| google/gemma-4-31B-it (+1) | Google | 2026-03-11 | 31.27 | 31.27 | bf16 | 58.3 | Gemma4ForConditionalGeneration | vision, audio | yes | apache-2.0 | 9,872,605 |
| google/gemma-4-12B-it (+1) | Google | 2026-05-23 | 11.96 | 11.96 | bf16 | 22.3 | Gemma4UnifiedForConditionalGeneration | vision, audio | yes | apache-2.0 | 1,840,993 |
| google/diffusiongemma-26B-A4B-it | Google | 2026-06-09 | 26 | 4 | bf16 | 48.1 | DiffusionGemmaForBlockDiffusion | MoE, vision, text diffusion LM | yes | apache-2.0 | 576,872 |
| openpangu/openPangu-2.0-Flash (+1) | Huawei openPangu | 2026-06-30 | 100.14 | 14.32 | bf16 | 186.5 | OpenPanguV2ForCausalLM | MoE, MTP x3, HF org openpangu | no |  | 202 |
| openpangu/openPangu-2.0-Pro (+1) | Huawei openPangu | 2026-07-30 | 541.49 | 55.07 | bf16 | 1,008.6 | OpenPanguV2ForCausalLM | MoE, MTP x3, HF org openpangu | no |  | 180 |
| ibm-granite/granite-4.1-3b (+2) | IBM | 2026-04-06 | 3.4 | 3.4 | bf16 | 6.3 | GraniteForCausalLM |  | yes | apache-2.0 | 540,575 |
| ibm-granite/granite-4.1-8b (+2) | IBM | 2026-04-06 | 8.79 | 8.79 | bf16 | 16.4 | GraniteForCausalLM |  | yes | apache-2.0 | 171,134 |
| ibm-granite/granite-4.1-30b (+2) | IBM | 2026-04-06 | 28.87 | 28.87 | bf16 | 53.8 | GraniteForCausalLM |  | yes | apache-2.0 | 253,761 |
| ibm-granite/granite-swash-2b | IBM | 2026-07-01 | 2.14 | 2.14 | bf16 | 4.0 | GraniteSWAForCausalLM |  | yes | apache-2.0 | 21,678 |
| ibm-granite/granite-swash-3b-a600m | IBM | 2026-07-01 | 3.02 | 0.6 | bf16 | 5.6 | GraniteMoeSWAForCausalLM | MoE | yes | apache-2.0 | 20,408 |
| ibm-granite/granite-4.2-3b (+2) | IBM | 2026-08-07 | 3.66 | 3.66 | bf16 | 6.8 | GraniteForCausalLM | post-trained on the granite-4.1 base | yes | apache-2.0 | 50,088 |
| ibm-granite/granite-4.2-8b (+2) | IBM | 2026-08-07 | 8.79 | 8.79 | bf16 | 16.4 | GraniteForCausalLM | post-trained on the granite-4.1 base | yes | apache-2.0 | 128,760 |
| ibm-granite/granite-4.2-30b (+2) | IBM | 2026-08-07 | 29.28 | 29.28 | bf16 | 54.5 | GraniteForCausalLM | post-trained on the granite-4.1 base | yes | apache-2.0 | 36,966 |
| IQuestLab/IQuest-Coder-V1-14B-Instruct (+2) | IQuest | 2026-02-14 | 14.44 | 14.44 | bf16 | 26.9 | IQuestCoderForCausalLM |  | yes | other | 228 |
| IQuestLab/IQuest-Q1 | IQuest | 2026-09-28 | 320.32 | 14.89 | bf16 | 605.3 | IQuestQ1ForCausalLM | MoE | no | other | 1,047 |
| jdopensource/JoyAI-LLM-Flash (+2) | JD | 2026-02-14 | 49.29 | 3.65 | bf16 | 91.8 | DeepseekV3ForCausalLM | MoE, MTP x1 | yes |  | 275 |
| JetBrains/Mellum2-12B-A2.5B-Instruct (+2) | JetBrains | 2026-05-26 | 12 | 2.5 | bf16 | 22.6 | MellumForCausalLM | MoE | yes | apache-2.0 | 7,353 |
| kakaocorp/kanana-2-30b-a3b-instruct-2601 (+2) | Kakao | 2026-01-14 | 30.67 | 3.61 | bf16 | 57.1 | DeepseekV3ForCausalLM | MoE | yes | other | 1,398 |
| kakaocorp/kanana-2-1.3b-instruct (+1) | Kakao | 2026-07-24 | 1.29 | 1.29 | bf16 | 2.4 | Kanana2TinyForCausalLM |  | no | other | 4,076 |
| kakaocorp/kanana-2-3b-instruct (+1) | Kakao | 2026-07-24 | 3.51 | 3.51 | bf16 | 6.5 | Qwen3ForCausalLM |  | yes | other | 4,305 |
| LGAI-EXAONE/EXAONE-4.5-33B (+1) | LG AI Research | 2026-04-04 | 34.35 | 34.35 | bf16 | 64.0 | Exaone4_5_ForConditionalGeneration | vision, MTP x1 | yes | other | 52,165 |
| LGAI-EXAONE/K-EXAONE-2.0-750B-A37B (+2) | LG AI Research | 2026-07-29 | 750 | 37 | bf16 | 1,395.8 | ExaoneMoeForCausalLM | MoE, MTP x1 | yes | apache-2.0 | 3,469 |
| LiquidAI/LFM2.5-VL-1.6B | Liquid AI | 2026-01-05 | 1.6 | 1.6 | bf16 | 3.0 | Lfm2VlForConditionalGeneration | hybrid/linear attn (conv), vision | yes | other | 29,469 |
| LiquidAI/LFM2.5-1.2B-Instruct (+3) | Liquid AI | 2026-01-06 | 1.17 | 1.17 | bf16 | 2.2 | Lfm2ForCausalLM | hybrid/linear attn (conv) | yes | other | 117,183 |
| LiquidAI/LFM2-24B-A2B | Liquid AI | 2026-02-24 | 24 | 2 | bf16 | 44.4 | Lfm2MoeForCausalLM | MoE, hybrid/linear attn (conv) | yes | other | 3,355 |
| LiquidAI/LFM2.5-350M (+1) | Liquid AI | 2026-03-31 | 0.35 | 0.35 | bf16 | 0.7 | Lfm2ForCausalLM | hybrid/linear attn (conv) | yes | other | 64,902 |
| LiquidAI/LFM2.5-VL-450M | Liquid AI | 2026-04-08 | 0.45 | 0.45 | bf16 | 0.8 | Lfm2VlForConditionalGeneration | hybrid/linear attn (conv), vision | yes | other | 33,499 |
| LiquidAI/LFM2.5-8B-A1B (+1) | Liquid AI | 2026-05-28 | 8 | 1 | bf16 | 15.8 | Lfm2MoeForCausalLM | MoE, hybrid/linear attn (conv) | yes | other | 31,655 |
| LiquidAI/LFM2.5-230M (+1) | Liquid AI | 2026-06-24 | 0.23 | 0.23 | bf16 | 0.4 | Lfm2ForCausalLM | hybrid/linear attn (conv) | yes | other | 88,215 |
| LiquidAI/LFM2.5-2.6B (+1) | Liquid AI | 2026-07-28 | 2.7 | 2.7 | bf16 | 5.0 | Lfm2ForCausalLM | hybrid/linear attn (conv) | yes | other | 103,387 |
| LiquidAI/LFM2.5-VL-3B | Liquid AI | 2026-08-11 | 3.12 | 3.12 | bf16 | 5.8 | Lfm2VlForConditionalGeneration | hybrid/linear attn (conv), vision | yes | other | 25,985 |
| IFM/K2-Horizon-0.9B | MBZUAI IFM | 2026-09-01 | 1.08 | 1.08 | bf16 | 2.0 | K2HorizonForCausalLM |  | yes | apache-2.0 | 37,094 |
| IFM/K2-Horizon-3.7B | MBZUAI IFM | 2026-09-01 | 5.06 | 5.06 | bf16 | 9.4 | K2HorizonForCausalLM |  | yes | apache-2.0 | 22,360 |
| IFM/K2-Horizon-7B (+1) | MBZUAI IFM | 2026-09-01 | 9 | 9 | bf16 | 16.8 | K2HorizonForCausalLM |  | yes | apache-2.0 | 28,295 |
| IFM/K2-Horizon-32B (+2) | MBZUAI IFM | 2026-09-01 | 34.78 | 34.78 | bf16 | 64.8 | K2HorizonForCausalLM |  | yes | apache-2.0 | 6,404 |
| IFM/K2-Horizon-MoVA-36B-A4B (+1) | MBZUAI IFM | 2026-09-01 | 36 | 4 | bf16 | 69.7 | K2HorizonForCausalLM | MoE | yes | apache-2.0 | 19,286 |
| IFM/K2-Horizon-375B-A23B (+2) | MBZUAI IFM | 2026-09-01 | 375 | 23 | bf16 | 706.3 | K2HorizonForCausalLM | MoE | yes | apache-2.0 | 16,281 |
| meituan-longcat/LongCat-Flash-Thinking-2601 (+2) | Meituan LongCat | 2026-01-14 | 561.86 | 33.38 | bf16 | 1,046.8 | LongcatCausalLM | MoE, update of a 2025 LongCat model | no | mit | 2,494 |
| meituan-longcat/LongCat-Flash-Lite (+2) | Meituan LongCat | 2026-01-27 | 69.07 | 3 | bf16 | 128.7 | LongcatFlashNgramForCausalLM | MoE | yes | mit | 5,223 |
| meituan-longcat/LongCat-Next | Meituan LongCat | 2026-03-25 | 74.26 |  | bf16 | 150.0 | LongcatNextForCausalLM | MoE, audio | no | mit | 380 |
| meituan-longcat/LongCat-2.0 (+2) | Meituan LongCat | 2026-07-05 | 1775.56 | 48 | bf16 | 3,307.8 | LongcatCausalLM | MoE | no | mit | 2,060 |
| facebook/MobileMoE-S-SFT (+1) | Meta | 2026-06-11 | 1.26 | 1.26 | bf16 | 2.4 | MobileMoEForCausalLM | config.json gated, research | no | fair-noncommercial-research-license (gated) | 78 |
| facebook/MobileMoE-M-SFT (+1) | Meta | 2026-06-11 | 2.82 | 2.82 | bf16 | 5.2 | MobileMoEForCausalLM | config.json gated, research | no | fair-noncommercial-research-license (gated) | 87 |
| facebook/MobileMoE-L-SFT (+1) | Meta | 2026-06-11 | 5.33 | 5.33 | bf16 | 9.9 | MobileMoEForCausalLM | config.json gated, research | no | fair-noncommercial-research-license (gated) | 11 |
| meta-models/Muse-Glimmer-30B | Meta | 2026-08-09 | 29.78 | 29.78 | bf16 | 55.5 | MuseGlimmerForConditionalGeneration | vision | yes | apache-2.0 | 276,520 |
| microsoft/Phi-4-reasoning-vision-15B | Microsoft | 2026-01-23 | 15.12 | 15.12 | bf16 | 28.2 | Phi4ForCausalLMV | vision, built on a 2025 Phi model plus a vision encoder | yes | mit | 2,559 |
| MiniMaxAI/MiniMax-M3 (+1) | MiniMax | 2026-06-02 | 427.04 |  | bf16 | 795.5 | MiniMaxM3SparseForConditionalGeneration | MoE, vision, MTP x7 | yes | other | 185,785 |
| Motif-Technologies/Motif-3 (+3) | Motif Technologies | 2026-08-07 | 314.84 | 13.23 | bf16 | 586.4 | MotifForCausalLM | MoE, MTP x1 | no | mit | 2,721 |
| nvidia/Nemotron-Labs-Diffusion-3B (+1) | NVIDIA | 2026-03-02 | 3.83 | 3.83 | bf16 | 7.2 | NemotronLabsDiffusionModel | text diffusion LM | no | other | 63,708 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (+1) | NVIDIA | 2026-03-07 | 3.97 | 3.97 | bf16 | 7.4 | NemotronHForCausalLM | hybrid/linear attn (mamba) | yes | other | 2,753,879 |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (+3) | NVIDIA | 2026-03-10 | 120 | 12 | bf16 | 230.2 | NemotronHForCausalLM | MoE, hybrid/linear attn (mamba), MTP x1 | yes | other | 1,152,632 |
| nvidia/Nemotron-Labs-Diffusion-8B (+1) | NVIDIA | 2026-03-18 | 8.49 | 8.49 | bf16 | 15.9 | NemotronLabsDiffusionModel | text diffusion LM | no | other | 10,184 |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (+2) | NVIDIA | 2026-04-20 | 30 | 3 | bf16 | 61.5 | NemotronH_Nano_Omni_Reasoning_V3 | MoE, hybrid/linear attn (mamba), vision, audio | yes | other | 353,994 |
| nvidia/Nemotron-Labs-Diffusion-14B (+1) | NVIDIA | 2026-04-22 | 13.51 | 13.51 | bf16 | 25.3 | NemotronLabsDiffusionModel | text diffusion LM | no | other | 15,706 |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 (+2) | NVIDIA | 2026-06-03 | 550 | 55 | bf16 | 1,044.1 | NemotronHForCausalLM | MoE, hybrid/linear attn (mamba), MTP x1 | yes | other | 476,003 |
| nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 (+1) | NVIDIA | 2026-06-24 | 75 | 9 | bf16 | 145.8 | NemotronHPuzzleForCausalLM | MoE, hybrid/linear attn (mamba), MTP x1, pruned from Nemotron 3 Super | yes | other | 872 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (+2) | NVIDIA | 2026-08-01 | 30 | 3 | bf16 | 61.3 | NemotronHForCausalLM | MoE, hybrid/linear attn (mamba), MTP x1 | yes | other | 571,678 |
| NaiveAI/Naive-N0.5-Flash (+1) | NaiveAI | 2026-09-27 | 308.89 | 9.32 | bf16 | 575.4 | NaiveN05FlashForCausalLM | MoE | no | mit | 2,036 |
| Nanbeige/Nanbeige4.1-3B | Nanbeige | 2026-02-10 | 3.93 | 3.93 | bf16 | 7.3 | LlamaForCausalLM | post-trained on its 2025 base | yes | apache-2.0 | 5,032 |
| Nanbeige/Nanbeige4.2-3B (+2) | Nanbeige | 2026-07-21 | 4.17 | 4.17 | bf16 | 7.8 | NanbeigeForCausalLM |  | yes | apache-2.0 | 85,972 |
| openbmb/MiniCPM-SALA | OpenBMB | 2026-02-11 | 9.48 | 9.48 | bf16 | 17.7 | MiniCPMSALAForCausalLM |  | no | apache-2.0 | 3,721 |
| openbmb/MiniCPM5-1B (+1) | OpenBMB | 2026-05-21 | 1.08 | 1.08 | bf16 | 2.0 | LlamaForCausalLM |  | yes | apache-2.0 | 408,311 |
| openbmb/MiniCPM5-2B (+1) | OpenBMB | 2026-09-06 | 2.52 | 2.52 | bf16 | 4.7 | LlamaForCausalLM |  | yes | apache-2.0 | 1,145,692 |
| PleIAs/Baguettotron-MoE | PleIAs | 2026-04-30 | 13.29 | 1.16 | bf16 | 24.8 | BaguettotronMoeForCausalLM | MoE | no | apache-2.0 | 28 |
| poolside/Laguna-XS.2 (+1) | Poolside | 2026-04-23 | 33.44 | 2.24 | bf16 | 62.3 | LagunaForCausalLM | MoE | yes | apache-2.0 | 271,000 |
| poolside/Laguna-M.1 (+2) | Poolside | 2026-06-15 | 225.8 | 23.46 | bf16 | 420.6 | LagunaForCausalLM | MoE | yes | apache-2.0 | 247,594 |
| poolside/Laguna-XS-2.1 (+1) | Poolside | 2026-06-20 | 33.44 | 3.02 | bf16 | 62.3 | LagunaForCausalLM | MoE | yes | openmdw-1.1 | 223,497 |
| poolside/Laguna-S-2.1 (+1) | Poolside | 2026-07-13 | 117.56 | 8.45 | bf16 | 219.0 | LagunaForCausalLM | MoE | yes | openmdw-1.1 | 189,780 |
| pfnet/plamo-3-nict-2604-31b-base | Preferred Networks | 2026-06-12 | 31.76 | 31.76 | bf16 | 59.2 | Plamo3ForCausalLM | config.json gated | yes | other (gated) | 151 |
| RWKV/RWKV7-G1k-1.5B-20260930 (+2) | RWKV | 2026-09-30 | 1.53 | 1.53 | bf16 | 2.8 | Rwkv7ForCausalLM |  | no | apache-2.0 | 322 |
| RWKV/RWKV7-G1k-2.9B-20260930 (+2) | RWKV | 2026-09-30 | 2.95 | 2.95 | bf16 | 5.5 | Rwkv7ForCausalLM |  | no | apache-2.0 | 279 |
| RWKV/RWKV7-G1k-7.2B-20260930 (+2) | RWKV | 2026-09-30 | 7.2 | 7.2 | bf16 | 13.4 | Rwkv7ForCausalLM |  | no | apache-2.0 | 212 |
| RWKV/RWKV7-G1k-13.3B-20260930 (+2) | RWKV | 2026-09-30 | 13.27 | 13.27 | bf16 | 24.7 | Rwkv7ForCausalLM |  | no | apache-2.0 | 181 |
| dots-studio/dots3-note-prev (+1) | Rednote dots | 2026-08-09 | 288.44 | 25.15 | bf16 | 537.3 | Dots3NoteForCausalLM | MoE, vision, audio | yes | apache-2.0 | 941 |
| RekaAI/reka-edge-2603 | Reka | 2026-03-11 | 7.13 | 7.13 | bf16 | 13.3 | Yasa2ForConditionalGeneration | vision | no | other | 618 |
| sapientinc/HRM-Text-1B | Sapient | 2026-05-17 | 1.18 | 1.18 | bf16 | 2.2 | HrmTextForCausalLM |  | yes | apache-2.0 | 19,142 |
| sarvamai/sarvam-30b (+1) | Sarvam | 2026-03-03 | 32.15 | 4.52 | fp32 | 59.9 | SarvamMoEForCausalLM | MoE | yes | apache-2.0 | 168,030 |
| sarvamai/sarvam-105b (+1) | Sarvam | 2026-03-03 | 106.03 | 12.41 | fp32 | 197.5 | SarvamMLAForCausalLM | MoE | yes | apache-2.0 | 24,238 |
| ai-sage/GigaChat3.1-10B-A1.8B-bf16 (+1) | Sber | 2026-03-21 | 10 | 1.8 | bf16 | 21.4 | DeepseekV3ForCausalLM | MoE, MTP x1 | yes | mit | 5,722 |
| ai-sage/GigaChat3.1-702B-A36B-bf16 (+1) | Sber | 2026-03-21 | 702 | 36 | bf16 | 1,332.6 | DeepseekV3ForCausalLM | MoE, MTP x1 | yes | mit | 104 |
| ai-sage/GFusion-10B-A1.8B-bf16 (+2) | Sber | 2026-07-01 | 10 | 1.8 | bf16 | 19.9 | GFusionForDiffusionLM | MoE, MTP x1, text diffusion LM | no | mit | 112 |
| ai-sage/GigaChat3.5-432B-A28B (+4) | Sber | 2026-07-05 | 432 | 28 | fp8 e4m3 block-scaled 128x128 | 807.9 (GigaChat3.5-432B-A28B-bf16) | GigaChat35ForCausalLM | MoE, hybrid/linear attn (linear attention), MTP x2 | no | mit | 609 |
| ServiceNow-AI/SuperApriel-15B-Instruct (+1) | ServiceNow | 2026-04-07 | 25.07 | 25.07 | fp32 | 46.7 | Apriel2ForConditionalGeneration | supernet checkpoint (several mixer variants per layer) | no | mit | 342 |
| internlm/Intern-S1-Pro (+1) | Shanghai AI Lab | 2026-02-02 | 917.04 | 22.85 | fp8 e4m3 block-scaled 128x128 | 1,708.2 (Intern-S1-Pro-BF16) | InternS1ProForConditionalGeneration | MoE, vision | yes | apache-2.0 | 1,567 |
| Soofi-Project/Soofi-S-Instruct-Preview (+3) | Soofi | 2026-06-16 | 31.58 | 31.58 | bf16 | 58.8 | NemotronHForCausalLM | config.json gated, NemotronH architecture, same size as Nemotron 3 Nano 30B-A3B; the Hub metadata does not say whether it was trained from scratch | yes | other (gated) | 289 |
| stepfun-ai/Step-3.5-Flash (+3) | StepFun | 2026-02-01 | 199.38 | 11 | bf16 | 371.4 | Step3p5ForCausalLM | MoE, MTP x3 | yes | apache-2.0 | 207,907 |
| stepfun-ai/Step-3.7-Flash (+2) | StepFun | 2026-05-23 | 201.37 |  | bf16 | 375.1 | Step3p7ForConditionalGeneration | MoE, vision, MTP x3 | yes | apache-2.0 | 24,744 |
| swiss-ai/Apertus-v1.1-0.5B-Instruct (+1) | Swiss AI | 2026-05-01 | 0.57 | 0.57 | bf16 | 1.1 | ApertusForCausalLM |  | yes | apache-2.0 | 759 |
| swiss-ai/Apertus-v1.1-4B-Instruct (+1) | Swiss AI | 2026-05-01 | 3.83 | 3.83 | bf16 | 7.1 | ApertusForCausalLM |  | yes | apache-2.0 | 849 |
| swiss-ai/Apertus-v1.1-1.5B-Instruct (+1) | Swiss AI | 2026-05-03 | 1.51 | 1.51 | bf16 | 2.8 | ApertusForCausalLM |  | yes | apache-2.0 | 743 |
| swiss-ai/Apertus-v1.5-8B (+1) | Swiss AI | 2026-07-24 | 8.9 | 8.9 | bf16 | 17.1 | Apertus1p5ForConditionalGeneration | config.json gated | no | apache-2.0 (gated) | 681,716 |
| swiss-ai/Apertus-v1.5-70B (+1) | Swiss AI | 2026-07-24 | 72.01 | 72.01 | bf16 | 134.7 | Apertus1p5ForConditionalGeneration | config.json gated | no | apache-2.0 (gated) | 12,730 |
| tiiuae/Falcon-H1-Tiny-R-0.6B | TII Falcon | 2026-01-12 | 0.62 | 0.62 | bf16 | 1.2 | FalconH1ForCausalLM | hybrid/linear attn (mamba) | yes | other | 1,125 |
| tencent/Hy3 (+3) | Tencent Hunyuan | 2026-07-02 | 298.79 | 24.43 | bf16 | 556.5 | HYV3ForCausalLM | MoE, MTP x1 | yes | apache-2.0 | 372,392 |
| tencent/Hy4-preview (+1) | Tencent Hunyuan | 2026-08-27 | 779.96 | 49.75 | bf16 | 1,452.8 | HYV4ForCausalLM | MoE, MTP x1 | yes | apache-2.0 | 25,811 |
| thinkingmachines/Inkling (+1) | Thinking Machines | 2026-07-14 | 952.38 | 46.41 | bf16 | 1,773.9 | InklingForConditionalGeneration | MoE, vision, audio, MTP x8 | yes | apache-2.0 | 871,023 |
| thinkingmachines/Inkling-Small (+1) | Thinking Machines | 2026-07-27 | 265.96 | 14.3 | bf16 | 495.4 | InklingForConditionalGeneration | MoE, vision, audio, MTP x8 | yes | apache-2.0 | 684,270 |
| trillionlabs/Trida-7B | Trillion Labs | 2026-01-19 | 7.53 | 7.53 | bf16 | 14.0 | TridaForDLM | text diffusion LM | no | apache-2.0 | 173 |
| trillionlabs/Tri-21B-Think (+1) | Trillion Labs | 2026-02-18 | 20.73 | 20.73 | fp32 | 38.6 | TrillionForCausalLM | post-trained on its 2025 base | no | apache-2.0 | 362 |
| trillionlabs/Gravity-16B-A3B-Preview (+1) | Trillion Labs | 2026-04-15 | 16 | 3 | bf16 | 30.3 | GravityMoEForCausalLM | MoE | no | apache-2.0 | 343 |
| trillionlabs/Gravity-30B-A5B-Preview (+1) | Trillion Labs | 2026-09-09 | 30 | 5 | bf16 | 55.1 | GravityMoEForCausalLM | MoE | no | apache-2.0 | 542 |
| upstage/Solar-Open2-250B | Upstage | 2026-07-22 | 250.29 | 14.74 | bf16 | 466.2 | SolarOpen2ForCausalLM | MoE, hybrid/linear attn (kda (linear attention)) | no | other | 4,455 |
| yandex/AliceAI-T5-35B-A0.6B | Yandex | 2026-09-10 | 35 | 0.6 | bf16 | 64.4 | AliceAIT5MoEForConditionalGeneration | encoder-decoder | no |  | 3,359 |
| yandex/AliceAI-Foundation-80B-A3B-Base | Yandex | 2026-09-12 | 80 | 3 | bf16 | 151.4 | AliceAIForCausalLM | MoE, hybrid/linear attn (linear_attention), MTP x1 | no | apache-2.0 | 4,640 |
| zai-org/GLM-4.7-Flash | Zhipu Z.ai | 2026-01-19 | 31.22 | 5.17 | bf16 | 58.2 | Glm4MoeLiteForCausalLM | MoE, MTP x1 | yes | mit | 1,677,746 |
| zai-org/GLM-5 (+1) | Zhipu Z.ai | 2026-02-11 | 753.86 | 51.74 | bf16 | 1,404.2 | GlmMoeDsaForCausalLM | MoE, MTP x1 | yes | mit | 447,689 |
| zai-org/GLM-5.1 (+1) | Zhipu Z.ai | 2026-04-03 | 753.86 | 51.74 | bf16 | 1,404.2 | GlmMoeDsaForCausalLM | MoE, MTP x1 | yes | mit | 316,149 |
| zai-org/GLM-5.2 (+1) | Zhipu Z.ai | 2026-06-16 | 753.33 | 51.2 | bf16 | 1,403.2 | GlmMoeDsaForCausalLM | MoE, MTP x1 | yes | mit | 681,301 |
| zai-org/GLM-5.3 (+1) | Zhipu Z.ai | 2026-08-25 | 753.33 | 51.2 | fp8 e4m3 block-scaled 128x128 | 1,403.2 (GLM-5.3-BF16) | GlmMoeDsaForCausalLM | MoE, MTP x1 | yes | other | 1,436,511 |
| zai-org/GLM-5.3-Flash (+1) | Zhipu Z.ai | 2026-08-25 | 321.32 | 25.37 | fp8 e4m3 block-scaled 128x128; other parts: BF16 2% | 598.5 (GLM-5.3-Flash-BF16) | Glm5NextForConditionalGeneration | MoE, hybrid/linear attn (linear_attention), vision, MTP x1 | yes | mit | 5,740,734 |
| Zyphra/ZAYA1-VL-8B | Zyphra | 2026-04-20 | 9.72 |  | bf16 | 18.1 | Zaya1VLForConditionalGeneration | MoE, vision | no | apache-2.0 | 77,899 |
| Zyphra/ZAYA1-8B (+2) | Zyphra | 2026-05-04 | 8.84 | 1.29 | bf16 | 16.5 | ZayaForCausalLM | MoE | no | apache-2.0 | 12,056 |
| Zyphra/ZAYA1-74B-preview | Zyphra | 2026-05-05 | 74.79 | 5.34 | bf16 | 139.3 | ZayaForCausalLM | MoE | no | apache-2.0 | 616 |
| llm-jp/llm-jp-4-32b-a3b-thinking (+1) | llm-jp | 2026-03-16 | 32.14 | 3.83 | bf16 | 59.9 | Qwen3MoeForCausalLM | MoE | yes | apache-2.0 | 2,583 |
| llm-jp/llm-jp-4-8b-instruct (+2) | llm-jp | 2026-03-30 | 8.59 | 8.59 | bf16 | 16.0 | LlamaForCausalLM |  | yes | apache-2.0 | 6,419 |
| llm-jp/llm-jp-4-33b-thinking (+1) | llm-jp | 2026-08-14 | 33.22 | 33.22 | bf16 | 61.9 | LlamaForCausalLM |  | yes | apache-2.0 | 3,443 |
| llm-jp/llm-jp-4.1-8b-thinking | llm-jp | 2026-09-15 | 8.59 | 8.59 | bf16 | 16.0 | LlamaForCausalLM |  | yes | apache-2.0 | 1,137 |
| llm-jp/llm-jp-4.1-32b-a3b-thinking | llm-jp | 2026-09-16 | 32.14 | 3.83 | bf16 | 59.9 | Qwen3MoeForCausalLM | MoE | yes | apache-2.0 | 964 |
| llm-jp/llm-jp-4.1-33b-thinking | llm-jp | 2026-09-16 | 33.22 | 33.22 | bf16 | 61.9 | LlamaForCausalLM |  | yes | apache-2.0 | 960 |

## 2. FP8 originals, no first-party bf16 copy (12)

| Repo | Maker | Released | Total B | Active B | Format | GiB on Hub | Arch | Features | vLLM 0.30 | License | Downloads |
| :-- | :-- | :-- | --: | --: | :-- | --: | :-- | :-- | :-- | :-- | --: |
| Agnes-AI/Agnes-2.5-Flash-Base | Agnes AI | 2026-08-29 | 202.43 | 16.4 | fp8 e4m3 block-scaled 128x128 | 189.9 | AgnesForCausalLM | MoE | no | apache-2.0 | 245 |
| inclusionAI/Ring-2.5-1T | Ant inclusionAI | 2026-02-10 | 1012.23 | 63.58 | fp8 e4m3 per-channel (compressed-tensors) | 945.7 | BailingMoeV2_5ForCausalLM | MoE | yes | mit | 4,104 |
| inclusionAI/Ling-2.6-1T (+1) | Ant inclusionAI | 2026-04-29 | 1025.41 | 76.76 | fp8 e4m3 per-channel (compressed-tensors); other parts: BF16 2% | 970.3 | BailingMoeV2_5ForCausalLM | MoE, MTP x1 | yes | mit | 1,413 |
| inclusionAI/Ring-2.6-1T | Ant inclusionAI | 2026-05-14 | 1025.41 | 76.76 | fp8 e4m3 per-channel (compressed-tensors); other parts: BF16 2% | 970.3 | BailingMoeV2_5ForCausalLM | MoE, MTP x1 | yes | mit | 1,156 |
| MiniMaxAI/MiniMax-M2.5 | MiniMax | 2026-02-12 | 228.7 | 11.04 | fp8 e4m3 block-scaled 128x128 | 214.3 | MiniMaxM2ForCausalLM | MoE, MTP x3 | yes | other | 495,509 |
| MiniMaxAI/MiniMax-M2.7 | MiniMax | 2026-04-09 | 228.69 | 11.03 | fp8 e4m3 block-scaled 128x128 | 214.3 | MiniMaxM2ForCausalLM | MoE, MTP x3 | yes | other | 1,018,598 |
| mistralai/Mistral-Small-4-119B-2603 (+3) | Mistral AI | 2026-01-23 | 119.4 | 7.06 | fp8 e4m3 per-tensor; other parts: BF16 1% | 225.2 | Mistral3ForConditionalGeneration | MoE, vision | yes | apache-2.0 | 67,146 |
| mistralai/Mistral-Medium-3.5-128B | Mistral AI | 2026-03-31 | 127.7 | 127.7 | fp8 e4m3 per-tensor; other parts: BF16 5% | 248.9 | Mistral3ForConditionalGeneration | vision | yes | other | 102,582 |
| moondream/moondream3.1-9B-A2B | Moondream | 2026-06-30 | 9 | 2 | fp8 e4m3 (no quantization_config); other parts: BF16 13% | 9.8 |  |  | no | other | 15,716 |
| skt/A.X-K2 (+1) | SK Telecom | 2026-07-28 | 691.69 | 36.37 | fp8 e4m3 block-scaled 128x128 | 646.7 | AXK2ForCausalLM | MoE | no | apache-2.0 | 36,715 |
| XiaomiMiMo/MiMo-V2.5 (+1) | Xiaomi MiMo | 2026-04-27 | 310.78 | 11.2 | fp8 e4m3 block-scaled 128x128; other parts: BF16 1% | 294.0 | MiMoV2ForCausalLM | MoE, vision, audio | yes | mit | 235,042 |
| XiaomiMiMo/MiMo-V2.5-Pro (+1) | Xiaomi MiMo | 2026-04-27 | 1023.18 | 29.64 | fp8 e4m3 block-scaled 128x128 | 962.4 | MiMoV2ForCausalLM | MoE | yes | mit | 21,704 |

## 3. FP4 and mixed-precision originals (9)

| Repo | Maker | Released | Total B | Active B | Format | GiB on Hub | Arch | Features | vLLM 0.30 | License | Downloads |
| :-- | :-- | :-- | --: | --: | :-- | --: | :-- | :-- | :-- | :-- | --: |
| deepseek-ai/DeepSeek-V4-Flash (+4) | DeepSeek | 2026-04-22 | 290.94 | 20.41 | fp4 experts + fp8 e4m3 block-scaled 128x128; other parts: F8_E4M3 2% | 148.7 | DeepseekV4ForCausalLM | MoE, MTP x1 | yes | mit | 1,014,357 |
| deepseek-ai/DeepSeek-V4-Pro (+3) | DeepSeek | 2026-04-22 | 1598.84 | 75.62 | fp4 experts + fp8 e4m3 block-scaled 128x128; other parts: F8_E4M3 1% | 805.3 | DeepseekV4ForCausalLM | MoE, MTP x1 | yes | mit | 419,917 |
| deepseek-ai/DeepSeek-V4.1-Flash | DeepSeek | 2026-09-10 | 763.21 |  | fp4 experts + fp8 e4m3 block-scaled 32x32; other parts: F8_E4M3 27% | 475.3 | DeepseekV41ForCausalLM | MoE, vision, MTP x3 | yes | mit | 869,321 |
| moonshotai/Kimi-K2.5 | Moonshot AI | 2026-01-01 | 1026.88 | 33.33 | int4 experts (compressed-tensors, group 32), rest bf16; other parts: BF16 1% | 554.3 | KimiK25ForConditionalGeneration | MoE, vision | yes | other | 326,693 |
| moonshotai/Kimi-K2.6 | Moonshot AI | 2026-04-14 | 1026.88 | 33.33 | int4 experts (compressed-tensors, group 32), rest bf16; other parts: BF16 1% | 554.3 | KimiK25ForConditionalGeneration | MoE, vision | yes | other | 463,718 |
| moonshotai/Kimi-K2.7-Code | Moonshot AI | 2026-06-11 | 1026.88 | 33.33 | int4 experts (compressed-tensors, group 32), rest bf16; other parts: BF16 1% | 554.3 | KimiK25ForConditionalGeneration | MoE, vision | yes | other | 113,013 |
| moonshotai/Kimi-K3 | Moonshot AI | 2026-06-13 | 2779.93 |  | mxfp4 experts (compressed-tensors, group 32), rest bf16; other parts: BF16 2% | 1,453.7 | KimiK3ForConditionalGeneration | MoE, hybrid/linear attn (kda (linear attention)), vision | yes | other | 1,243,137 |
| XiaomiMiMo/MiMo-V2.6-Flash-RL (+1) | Xiaomi MiMo | 2026-09-21 | 310.76 | 11.18 | mxfp4 experts + fp8 e4m3 block-scaled 128x128; other parts: BF16 1%, F8_E4M3 1% | 165.5 | MiMoV2ForCausalLM | MoE, vision, audio, MTP x3 | yes | mit | 52,380 |
| XiaomiMiMo/MiMo-V2.6-Pro-RL (+1) | Xiaomi MiMo | 2026-09-21 | 1024.22 | 30.67 | mxfp4 experts + fp8 e4m3 block-scaled 128x128; other parts: F8_E4M3 1%, BF16 1% | 534.1 | MiMoV2ForCausalLM | MoE, vision, audio | yes | mit | 86,278 |

## 4. Which bf16 models fit one GPU with Glyd

Estimate: weights x 0.675 (33% smaller) + 3 GiB (activations, CUDA graphs, a small KV cache); bf16 for comparison = weights + 3 GiB. 24 GB = 24 GiB (RTX 4090), 48 GB = 48 GiB, 80 GB = 80 GiB (A100/H100). The x0.675 assumes every weight is packed; today's vLLM path leaves vision towers, MTP heads, linear-attention parameters and norms unpacked, so models with those come out above the estimate.

### One 24 GB GPU (73)

| Repo | bf16 GiB | bf16 + 3 GiB | Glyd est. GiB | Fits only with Glyd |
| :-- | --: | --: | --: | :-- |
| LiquidAI/LFM2.5-230M | 0.4 | 3.4 | 3.3 |  |
| LiquidAI/LFM2.5-350M | 0.7 | 3.7 | 3.4 |  |
| LiquidAI/LFM2.5-VL-450M | 0.8 | 3.8 | 3.6 |  |
| swiss-ai/Apertus-v1.1-0.5B-Instruct | 1.1 | 4.1 | 3.7 |  |
| tiiuae/Falcon-H1-Tiny-R-0.6B | 1.2 | 4.2 | 3.8 |  |
| Qwen/Qwen3.5-0.8B | 1.6 | 4.6 | 4.1 |  |
| IFM/K2-Horizon-0.9B | 2.0 | 5.0 | 4.4 |  |
| openbmb/MiniCPM5-1B | 2.0 | 5.0 | 4.4 |  |
| LiquidAI/LFM2.5-1.2B-Instruct | 2.2 | 5.2 | 4.5 |  |
| sapientinc/HRM-Text-1B | 2.2 | 5.2 | 4.5 |  |
| facebook/MobileMoE-S-SFT | 2.4 | 5.4 | 4.6 |  |
| kakaocorp/kanana-2-1.3b-instruct | 2.4 | 5.4 | 4.6 |  |
| swiss-ai/Apertus-v1.1-1.5B-Instruct | 2.8 | 5.8 | 4.9 |  |
| RWKV/RWKV7-G1k-1.5B-20260930 | 2.8 | 5.8 | 4.9 |  |
| LiquidAI/LFM2.5-VL-1.6B | 3.0 | 6.0 | 5.0 |  |
| Gensyn/open-1b-sft | 3.0 | 6.0 | 5.0 |  |
| ibm-granite/granite-swash-2b | 4.0 | 7.0 | 5.7 |  |
| Qwen/Qwen3.5-2B | 4.2 | 7.2 | 5.9 |  |
| CohereLabs/North-Micro-Vision-Instruct | 4.6 | 7.6 | 6.1 |  |
| openbmb/MiniCPM5-2B | 4.7 | 7.7 | 6.2 |  |
| LiquidAI/LFM2.5-2.6B | 5.0 | 8.0 | 6.4 |  |
| facebook/MobileMoE-M-SFT | 5.2 | 8.2 | 6.5 |  |
| RWKV/RWKV7-G1k-2.9B-20260930 | 5.5 | 8.5 | 6.7 |  |
| ibm-granite/granite-swash-3b-a600m | 5.6 | 8.6 | 6.8 |  |
| LiquidAI/LFM2.5-VL-3B | 5.8 | 8.8 | 6.9 |  |
| ai21labs/AI21-Jamba2-3B | 6.0 | 9.0 | 7.0 |  |
| CohereLabs/tiny-aya-global | 6.2 | 9.2 | 7.2 |  |
| ibm-granite/granite-4.1-3b | 6.3 | 9.3 | 7.3 |  |
| kakaocorp/kanana-2-3b-instruct | 6.5 | 9.5 | 7.4 |  |
| ibm-granite/granite-4.2-3b | 6.8 | 9.8 | 7.6 |  |
| swiss-ai/Apertus-v1.1-4B-Instruct | 7.1 | 10.1 | 7.8 |  |
| nvidia/Nemotron-Labs-Diffusion-3B | 7.2 | 10.2 | 7.9 |  |
| Nanbeige/Nanbeige4.1-3B | 7.3 | 10.3 | 7.9 |  |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 7.4 | 10.4 | 8.0 |  |
| Nanbeige/Nanbeige4.2-3B | 7.8 | 10.8 | 8.2 |  |
| Qwen/Qwen3.5-4B | 8.7 | 11.7 | 8.9 |  |
| IFM/K2-Horizon-3.7B | 9.4 | 12.4 | 9.4 |  |
| google/gemma-4-E2B-it | 9.5 | 12.5 | 9.4 |  |
| facebook/MobileMoE-L-SFT | 9.9 | 12.9 | 9.7 |  |
| RekaAI/reka-edge-2603 | 13.3 | 16.3 | 12.0 |  |
| RWKV/RWKV7-G1k-7.2B-20260930 | 13.4 | 16.4 | 12.1 |  |
| allenai/Olmo-Hybrid-Instruct-DPO-7B | 13.8 | 16.8 | 12.3 |  |
| trillionlabs/Trida-7B | 14.0 | 17.0 | 12.5 |  |
| inclusionAI/Ling-3.0-tiny | 14.7 | 17.7 | 12.9 |  |
| google/gemma-4-E4B-it | 14.9 | 17.9 | 13.1 |  |
| ATH-MaaS/Marco-Nano-Instruct | 14.9 | 17.9 | 13.1 |  |
| ByteDance-Seed/Stable-DiffCoder-8B-Instruct | 15.4 | 18.4 | 13.4 |  |
| EssentialAI/rnj-1.5-instruct | 15.5 | 18.5 | 13.4 |  |
| LiquidAI/LFM2.5-8B-A1B | 15.8 | 18.8 | 13.6 |  |
| nvidia/Nemotron-Labs-Diffusion-8B | 15.9 | 18.9 | 13.8 |  |
| llm-jp/llm-jp-4-8b-instruct | 16.0 | 19.0 | 13.8 |  |
| llm-jp/llm-jp-4.1-8b-thinking | 16.0 | 19.0 | 13.8 |  |
| ibm-granite/granite-4.2-8b | 16.4 | 19.4 | 14.1 |  |
| ibm-granite/granite-4.1-8b | 16.4 | 19.4 | 14.1 |  |
| Zyphra/ZAYA1-8B | 16.5 | 19.5 | 14.1 |  |
| IFM/K2-Horizon-7B | 16.8 | 19.8 | 14.3 |  |
| swiss-ai/Apertus-v1.5-8B | 17.1 | 20.1 | 14.6 |  |
| openbmb/MiniCPM-SALA | 17.7 | 20.7 | 14.9 |  |
| Qwen/Qwen3.5-9B | 18.0 | 21.0 | 15.1 |  |
| Zyphra/ZAYA1-VL-8B | 18.1 | 21.1 | 15.2 |  |
| ai-sage/GFusion-10B-A1.8B-bf16 | 19.9 | 22.9 | 16.4 |  |
| ai-sage/GigaChat3.1-10B-A1.8B-bf16 | 21.4 | 24.4 | 17.4 | yes |
| google/gemma-4-12B-it | 22.3 | 25.3 | 18.0 | yes |
| JetBrains/Mellum2-12B-A2.5B-Instruct | 22.6 | 25.6 | 18.3 | yes |
| RWKV/RWKV7-G1k-13.3B-20260930 | 24.7 | 27.7 | 19.7 | yes |
| PleIAs/Baguettotron-MoE | 24.8 | 27.8 | 19.7 | yes |
| nvidia/Nemotron-Labs-Diffusion-14B | 25.3 | 28.3 | 20.1 | yes |
| IQuestLab/IQuest-Coder-V1-14B-Instruct | 26.9 | 29.9 | 21.2 | yes |
| microsoft/Phi-4-reasoning-vision-15B | 28.2 | 31.2 | 22.0 | yes |
| amd/Instella-MoE-16B-A3B-Think | 29.5 | 32.5 | 22.9 | yes |
| trillionlabs/Gravity-16B-A3B-Preview | 30.3 | 33.3 | 23.4 | yes |
| inclusionAI/LLaDA2.2-mini | 30.3 | 33.3 | 23.4 | yes |
| inclusionAI/LLaDA2.1-mini | 30.3 | 33.3 | 23.4 | yes |

### One 48 GB GPU (33)

| Repo | bf16 GiB | bf16 + 3 GiB | Glyd est. GiB | Fits only with Glyd |
| :-- | --: | --: | --: | :-- |
| bharatgenai/Param2-17B-A2.4B-Thinking | 31.9 | 34.9 | 24.6 |  |
| ATH-MaaS/Marco-Mini-Instruct | 32.1 | 35.1 | 24.7 |  |
| trillionlabs/Tri-21B-Think | 38.6 | 41.6 | 29.1 |  |
| LiquidAI/LFM2-24B-A2B | 44.4 | 47.4 | 33.0 |  |
| ServiceNow-AI/SuperApriel-15B-Instruct | 46.7 | 49.7 | 34.5 | yes |
| google/gemma-4-26B-A4B-it | 48.1 | 51.1 | 35.4 | yes |
| google/diffusiongemma-26B-A4B-it | 48.1 | 51.1 | 35.5 | yes |
| Qwen/Qwen3.6-27B | 51.7 | 54.7 | 37.9 | yes |
| Qwen/Qwen3.8-27B | 51.7 | 54.7 | 37.9 | yes |
| Qwen/Qwen3.5-27B | 51.7 | 54.7 | 37.9 | yes |
| ibm-granite/granite-4.1-30b | 53.8 | 56.8 | 39.3 | yes |
| ibm-granite/granite-4.2-30b | 54.5 | 57.5 | 39.8 | yes |
| trillionlabs/Gravity-30B-A5B-Preview | 55.1 | 58.1 | 40.2 | yes |
| meta-models/Muse-Glimmer-30B | 55.5 | 58.5 | 40.4 | yes |
| CohereLabs/North-Mini-Code-1.0 | 56.8 | 59.8 | 41.3 | yes |
| kakaocorp/kanana-2-30b-a3b-instruct-2601 | 57.1 | 60.1 | 41.6 | yes |
| XingChen-AGI/Xing4.0-29B-A4B | 58.1 | 61.1 | 42.2 | yes |
| zai-org/GLM-4.7-Flash | 58.2 | 61.2 | 42.3 | yes |
| google/gemma-4-31B-it | 58.3 | 61.3 | 42.3 | yes |
| Soofi-Project/Soofi-S-Instruct-Preview | 58.8 | 61.8 | 42.7 | yes |
| pfnet/plamo-3-nict-2604-31b-base | 59.2 | 62.2 | 42.9 | yes |
| llm-jp/llm-jp-4-32b-a3b-thinking | 59.9 | 62.9 | 43.4 | yes |
| llm-jp/llm-jp-4.1-32b-a3b-thinking | 59.9 | 62.9 | 43.4 | yes |
| sarvamai/sarvam-30b | 59.9 | 62.9 | 43.4 | yes |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 61.3 | 64.3 | 44.4 | yes |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 | 61.5 | 64.5 | 44.5 | yes |
| llm-jp/llm-jp-4-33b-thinking | 61.9 | 64.9 | 44.8 | yes |
| llm-jp/llm-jp-4.1-33b-thinking | 61.9 | 64.9 | 44.8 | yes |
| poolside/Laguna-XS.2 | 62.3 | 65.3 | 45.0 | yes |
| poolside/Laguna-XS-2.1 | 62.3 | 65.3 | 45.0 | yes |
| LGAI-EXAONE/EXAONE-4.5-33B | 64.0 | 67.0 | 46.2 | yes |
| yandex/AliceAI-T5-35B-A0.6B | 64.4 | 67.4 | 46.5 | yes |
| IFM/K2-Horizon-32B | 64.8 | 67.8 | 46.7 | yes |

### One 80 GB GPU (6)

| Repo | bf16 GiB | bf16 + 3 GiB | Glyd est. GiB | Fits only with Glyd |
| :-- | --: | --: | --: | :-- |
| Qwen/Qwen3.6-35B-A3B | 67.0 | 70.0 | 48.2 |  |
| Qwen/Qwen3.5-35B-A3B | 67.0 | 70.0 | 48.2 |  |
| Tele-AI/TeleChat3-Coder-36B-Thinking | 67.5 | 70.5 | 48.6 |  |
| IFM/K2-Horizon-MoVA-36B-A4B | 69.7 | 72.7 | 50.1 |  |
| jdopensource/JoyAI-LLM-Flash | 91.8 | 94.8 | 65.0 | yes |
| ai21labs/AI21-Jamba2-Mini | 96.1 | 99.1 | 67.8 | yes |

### Multi-GPU (52)

| Repo | bf16 GiB | Glyd est. GiB | 80 GB GPUs |
| :-- | --: | --: | --: |
| meituan-longcat/LongCat-Flash-Lite | 128.7 | 89.9 | 2 |
| swiss-ai/Apertus-v1.5-70B | 134.7 | 93.9 | 2 |
| Zyphra/ZAYA1-74B-preview | 139.3 | 97.0 | 2 |
| Aleph-Alpha/Kolibri-1-BF16 | 145.5 | 101.2 | 2 |
| nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 | 145.8 | 101.4 | 2 |
| Qwen/Qwen3-Coder-Next | 148.4 | 103.2 | 2 |
| meituan-longcat/LongCat-Next | 150.0 | 104.3 | 2 |
| yandex/AliceAI-Foundation-80B-A3B-Base | 151.4 | 105.2 | 2 |
| openpangu/openPangu-2.0-Flash | 186.5 | 128.9 | 2 |
| inclusionAI/LLaDA2.1-flash | 191.6 | 132.4 | 2 |
| inclusionAI/LLaDA2.2-flash | 191.6 | 132.4 | 2 |
| sarvamai/sarvam-105b | 197.5 | 136.3 | 2 |
| inclusionAI/Ling-2.6-flash | 200.2 | 138.2 | 2 |
| inclusionAI/Ming-flash-omni-2.0 | 214.4 | 147.7 | 2 |
| poolside/Laguna-S-2.1 | 219.0 | 150.8 | 2 |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | 230.2 | 158.4 | 2 |
| inclusionAI/Ling-3.0-flash-VL | 232.6 | 160.0 | 2 |
| Qwen/Qwen3.5-122B-A10B | 233.0 | 160.3 | 3 |
| inclusionAI/Ling-3.0-flash | 237.5 | 163.3 | 3 |
| Qwen/Qwen3.8-Flash-Next | 335.3 | 229.3 | 3 |
| stepfun-ai/Step-3.5-Flash | 371.4 | 253.7 | 4 |
| stepfun-ai/Step-3.7-Flash | 375.1 | 256.2 | 4 |
| CohereLabs/command-a-plus-05-2026-bf16 | 407.5 | 278.0 | 4 |
| poolside/Laguna-M.1 | 420.6 | 286.9 | 4 |
| upstage/Solar-Open2-250B | 466.2 | 317.7 | 4 |
| thinkingmachines/Inkling-Small | 495.4 | 337.4 | 5 |
| dots-studio/dots3-note-prev | 537.3 | 365.7 | 5 |
| tencent/Hy3 | 556.5 | 378.7 | 5 |
| NaiveAI/Naive-N0.5-Flash | 575.4 | 391.4 | 5 |
| Motif-Technologies/Motif-3 | 586.4 | 398.8 | 5 |
| zai-org/GLM-5.3-Flash-BF16 | 598.5 | 407.0 | 6 |
| IQuestLab/IQuest-Q1 | 605.3 | 411.6 | 6 |
| IFM/K2-Horizon-375B-A23B | 706.3 | 479.7 | 6 |
| arcee-ai/Trinity-Large-Thinking | 742.5 | 504.2 | 7 |
| Qwen/Qwen3.5-397B-A17B | 751.4 | 510.2 | 7 |
| MiniMaxAI/MiniMax-M3 | 795.5 | 540.0 | 7 |
| ai-sage/GigaChat3.5-432B-A28B-bf16 | 807.9 | 548.3 | 7 |
| openpangu/openPangu-2.0-Pro | 1,008.6 | 683.8 | 9 |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | 1,044.1 | 707.7 | 9 |
| meituan-longcat/LongCat-Flash-Thinking-2601 | 1,046.8 | 709.6 | 9 |
| ai-sage/GigaChat3.1-702B-A36B-bf16 | 1,332.6 | 902.5 | 12 |
| LGAI-EXAONE/K-EXAONE-2.0-750B-A37B | 1,395.8 | 945.2 | 12 |
| zai-org/GLM-5.2 | 1,403.2 | 950.2 | 12 |
| zai-org/GLM-5.3-BF16 | 1,403.2 | 950.2 | 12 |
| zai-org/GLM-5 | 1,404.2 | 950.8 | 12 |
| zai-org/GLM-5.1 | 1,404.2 | 950.8 | 12 |
| tencent/Hy4-preview | 1,452.8 | 983.7 | 13 |
| internlm/Intern-S1-Pro-BF16 | 1,708.2 | 1156.0 | 15 |
| thinkingmachines/Inkling | 1,773.9 | 1200.4 | 16 |
| inclusionAI/Ling-2.5-1T | 1,885.4 | 1275.7 | 16 |
| meituan-longcat/LongCat-2.0 | 3,307.8 | 2235.8 | 28 |
| Qwen/Qwen3.8-2.4T-A95B | 4,556.4 | 3078.6 | 39 |
