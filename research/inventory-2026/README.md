# 2026 bf16 models under the glyd vLLM plugin: what is packed, what is not

Desk inventory, 2026-10-06. For every 2026 open bf16 model read, each checkpoint tensor is classified as the plugin at private/main (`glyd-gpu-py/glyd_gpu/vllm_plugin.py`) packs it under vLLM 0.30.0: packed, left unpacked (why), or not loaded by vLLM. Nothing was run: safetensors headers were read by HTTP range and vLLM 0.30.0's model files (the sdist) were read for what each architecture builds with the quant config. The tensor-level data is `inventory.json`; the scripts that made it are beside it.

**Method.** GlydConfig.get_quant_method: a LinearBase built with the quant config of >= 1<<20 weights (vLLM's merged layer: q/k/v, gate/up, in_proj_qkv/z, b/a, MLA q_a/kv_a summed), rows % 64 == 0, columns % 16 == 0, bf16; RoutedExperts: SiLU, no bias, no expert parallel, H % 64, 2I % 128, I % 16; VocabParallelEmbedding / ParallelLMHead (exact types) built with the quant config: dim % 64, >= 4096 rows, head_dtype bf16. Layout rule on an RTX 4090/5090 (on an A100/H100 a MoE model's attention Linears stay bf16: dense_stays).

**Projections, not measurements.** PROJECTED, not measured: saving = packed bytes x (1 - bits/16) / bytes vLLM loads; tiered 10.80 bits a weight, 12-bit 12.04. Ceilings with everything packed: 32.5% / 24.75%.

**Check against a measurement.** Qwen3.5-9B: 17.53 GiB loaded (18.83 GB), 15.36 GiB of it packed -> 12.54 GiB tiered + 0.13 GiB vLLM's own buffers (bf16's measured 17.66 GiB less 17.53 checkpoint GiB) = 12.67 GiB against 12.68 GiB measured (RTX 4090, v0.29, 2026-10-05); 12-bit: 13.86 against 13.89 GiB measured (RTX 5090). The alternatives (embedding packed: 12.05; vision tower unpacked: 12.86) do not match.

## Findings

1. **The rules reproduce the measurement.** Qwen3.5-9B: projected 12.67 GiB on an RTX 4090 (tiered) and 13.86 GiB on an RTX 5090 (12-bit) against 12.68 and 13.89 GiB measured. That fixes three facts the handoff had otherwise: Qwen3.5's **vision tower is packed** (vLLM gives `visual` the quant config; only its MLP `fc1`, 4,304 rows, is not), its **embedding is not** (vLLM builds it without the quant config; the log line "the embedding and the LM head packed too" comes from the plan, not the layers), and the MTP head, DeltaNet `conv1d`, `A_log`, `dt_bias` and norms cost nothing that matters (MTP is not loaded at all without speculative decoding; the rest is under 0.1%). Qwen3.5-9B's gap from the tiered ceiling (28.5% against 32.5%) is the embedding (3.5 points) and `fc1` (0.5).
2. **117 models read (113 not counting fine-tunes), 29,188 GB loaded by vLLM; 26,435 GB packed today: 29.4% projected tiered, byte-weighted.** 55 models are within 0.1 point of the tiered ceiling today (Granite 4.x, EXAONE 4.5, K-EXAONE, GLM-5/5.3, GLM-4.7-Flash, Hy3, Hy4, Mellum2, MiniCPM5, Apertus, Qwen3.8-2.4T, Laguna, K2-Horizon, ...), 30 within 2.5 points, 15 at 20-30%, 12 under 20%.
3. **Correctness: SwiGLU experts with a clamp are packed and run without it.** Hy4-preview and GLM-5.3-Flash set `swiglu_limit` 10; vLLM 0.30 keeps their activation SILU and passes the limit in `FusedMoEConfig.swiglu_limit`; `GlydMoEMethod.unsupported()` does not look at it, so their experts (1,488 GB and 609 GB) are packed and the grouped products skip the clamp below the decoded-route threshold. Not bit for bit with bf16 there. Fix item 0.
4. **The largest unpacked bytes are mixture-of-experts layers with an activation the grouped products lack**: relu² without a gate (all five Nemotron-H MoEs: 1,509 GB), swigluoai (MiniMax-M3: 826 GB), GELU-tanh (Gemma 4 26B-A4B, diffusiongemma: 91 GB), swiglustep (Step 3.5/3.7's two clamped layers: 36 GB). Those eight models save 1-3% today (Step 31%).
5. **The most common gap is the embedding vLLM builds without the quant config: 52 models, 58.4 GB** (Qwen3.5/3.6/3.8, Qwen3-Next, Qwen4Exp, LFM2/2.5, Nemotron-H, Jamba, Olmo-Hybrid, Muse-Glimmer, Cohere North and Command A+, Ling 3.0, Step, Hy3/Hy4, GLM-5.3-Flash and others). It costs the small models most: Qwen3.5-0.8B 9.7 points, LFM2.5-230M 9.5, Qwen3.5-2B 7.5, North-Micro-Vision 7.0.
6. **Qwen3.8-Flash-Next does not load with `--quantization glyd`**: vLLM's Qwen4Exp PLE n-gram embedding refuses any quant config but FP8/ModelOpt. Its table is 102 GB of 355. LongCat-Flash-Lite's n-gram table (62.8 GB, 45%) loads but is built without the quant config.
7. **Vision and audio towers**: packed where vLLM passes the quant config (Qwen3.5/3.6/3.8, EXAONE 4.5, Step, GLM-5.3-Flash, Ling VL, Gemma 4 E2B/E4B, Cohere North-Micro); not where vLLM withholds it because 4,304 is not a multiple of 64 (Gemma 4 26B/31B, SigLIP MLPs in Granite vision and Command A+) or builds the tower without it (Muse-Glimmer, Nemotron Omni, Inkling): 9 models, 11.2 GB, plus 4.0 GB of 4,304-row MLPs in 16 models.
8. **The ceiling is the layout, not the coverage.** With every weight packed the tiered layout gives 32.5% and the 12-bit one 24.75%; 33% needs at most 10.72 bits a weight on all of them. On an A100/H100 a mixture of experts' attention Linears also stay bf16 (`dense_stays`): 0.1-1 point less there.

## Per model

GB = 1e9 bytes, as in the checkpoint. Loaded = what vLLM 0.30.0 loads (MTP and nextn layers are not). Tiered / 12-bit: projected whole-model saving over the loaded bytes. A row marked * uses vLLM rules read automatically from the model file (embedding and head only); † a fine-tune or same-architecture derivative.

| Model | Arch | Loaded GB | Packed GB | Unpacked GB | Unpacked, by class (GB) | Tiered | 12-bit |
| :-- | :-- | --: | --: | --: | :-- | --: | --: |
| ai-sage/GigaChat3.1-10B-A1.8B-bf16* | DeepseekV3ForCausalLM | 21.3 | 21.3 | 0.05 | small Linears (< 1M weights) 0.05 | 32.4% | 24.7% |
| ai-sage/GigaChat3.1-702B-A36B-bf16* | DeepseekV3ForCausalLM | 1,404.1 | 1,403.9 | 0.23 | MoE routers (quant_config=None, fp32 gates) 0.22 | 32.5% | 24.8% |
| ai21labs/AI21-Jamba2-3B | JambaForCausalLM | 6.4 | 3.6 | 2.82 | Mamba (v1) mixer Linears built without the quant config (Jamba) 2.14; LM head built without the quant config 0.34; embedding built without the quant config 0.34 | 18.2% | 13.8% |
| ai21labs/AI21-Jamba2-Mini | JambaForCausalLM | 103.1 | 96.2 | 6.97 | Mamba (v1) mixer Linears built without the quant config (Jamba) 5.89; embedding built without the quant config 0.54; LM head built without the quant config 0.54 | 30.3% | 23.1% |
| allenai/Olmo-Hybrid-7B | OlmoHybridForCausalLM | 14.9 | 14.1 | 0.79 | embedding built without the quant config 0.77; small Linears (< 1M weights) 0.01 | 30.8% | 23.4% |
| allenai/Olmo-Hybrid-Instruct-DPO-7B | OlmoHybridForCausalLM | 14.9 | 14.1 | 0.79 | embedding built without the quant config 0.77; small Linears (< 1M weights) 0.01 | 30.8% | 23.4% |
| arcee-ai/Trinity-Large-Thinking* | AfmoeForCausalLM | 797.3 | 796.0 | 1.32 | embedding built without the quant config 1.23; MoE routers (quant_config=None, fp32 gates) 0.08 | 32.5% | 24.7% |
| ATH-MaaS/Marco-Mini-Instruct* | Qwen3MoeForCausalLM | 34.5 | 34.5 | 0.01 | MoE routers (quant_config=None, fp32 gates) 0.01 | 32.5% | 24.7% |
| ATH-MaaS/Marco-Nano-Instruct* | Qwen3MoeForCausalLM | 16.0 | 16.0 | 0.01 | MoE routers (quant_config=None, fp32 gates) 0.01 | 32.5% | 24.7% |
| bharatgenai/Param2-17B-A2.4B-Thinking* | Param2MoEForCausalLM | 34.3 | 34.3 | 0.01 | - | 32.5% | 24.8% |
| CohereLabs/command-a-plus-05-2026-bf16 | Cohere2VisionForConditionalGeneration | 437.5 | 434.6 | 2.88 | embedding built without the quant config 2.15; vision/audio tower not given the quant config 0.54; vision/audio projector / patch Linears built without the quant config 0.16; MoE routers (quant_config=None, fp32 gates) 0.03 | 32.3% | 24.6% |
| CohereLabs/North-Micro-Vision-Instruct | CohereCompassForConditionalGeneration | 5.0 | 3.6 | 1.35 | embedding built without the quant config 1.07; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27 | 23.7% | 18.0% |
| CohereLabs/North-Mini-Code-1.0 | Cohere2MoeForCausalLM | 61.0 | 59.9 | 1.10 | embedding built without the quant config 1.07; MoE routers (quant_config=None, fp32 gates) 0.03 | 31.9% | 24.3% |
| dots-studio/dots3-note-prev* | Dots3NoteForCausalLM | 574.6 | 574.4 | 0.20 | MoE routers (quant_config=None, fp32 gates) 0.12; small Linears (< 1M weights) 0.07; conv weights / patch embeddings 0.01 | 32.5% | 24.7% |
| google/diffusiongemma-26B-A4B-it | DiffusionGemmaForBlockDiffusion | 51.6 | 4.8 | 46.84 | MoE experts: GELU-tanh (Gemma 4) 45.68; vision/audio tower not given the quant config 1.10; conv weights / patch embeddings 0.05; MoE routers (quant_config=None, fp32 gates) 0.02 | 3.0% | 2.3% |
| google/gemma-4-12B-it | Gemma4UnifiedForConditionalGeneration | 23.9 | 23.9 | 0.02 | conv weights / patch embeddings 0.02 | 32.5% | 24.7% |
| google/gemma-4-26B-A4B-it | Gemma4ForConditionalGeneration | 51.6 | 4.8 | 46.84 | MoE experts: GELU-tanh (Gemma 4) 45.68; vision/audio tower not given the quant config 1.10; conv weights / patch embeddings 0.05; MoE routers (quant_config=None, fp32 gates) 0.02 | 3.0% | 2.3% |
| google/gemma-4-31B-it | Gemma4ForConditionalGeneration | 62.5 | 61.4 | 1.15 | vision/audio tower not given the quant config 1.10; conv weights / patch embeddings 0.05 | 31.9% | 24.3% |
| google/gemma-4-E2B-it | Gemma4ForConditionalGeneration | 10.2 | 10.0 | 0.16 | small Linears (< 1M weights) 0.13; conv weights / patch embeddings 0.03 | 32.0% | 24.4% |
| google/gemma-4-E4B-it | Gemma4ForConditionalGeneration | 15.9 | 15.7 | 0.22 | small Linears (< 1M weights) 0.19; conv weights / patch embeddings 0.03 | 32.0% | 24.4% |
| ibm-granite/granite-4.1-30b | GraniteForCausalLM | 57.7 | 57.7 | 0.00 | - | 32.5% | 24.8% |
| ibm-granite/granite-4.1-3b | GraniteForCausalLM | 6.8 | 6.8 | 0.00 | - | 32.5% | 24.8% |
| ibm-granite/granite-4.1-8b | GraniteForCausalLM | 17.6 | 17.6 | 0.00 | - | 32.5% | 24.8% |
| ibm-granite/granite-4.2-30b | GraniteForCausalLM | 58.6 | 58.6 | 0.00 | - | 32.5% | 24.8% |
| ibm-granite/granite-4.2-3b | GraniteForCausalLM | 7.3 | 7.3 | 0.00 | - | 32.5% | 24.8% |
| ibm-granite/granite-4.2-8b | GraniteForCausalLM | 17.6 | 17.6 | 0.00 | - | 32.5% | 24.8% |
| ibm-granite/granite-swash-2b | GraniteSWAForCausalLM | 4.3 | 4.3 | 0.00 | - | 32.5% | 24.8% |
| ibm-granite/granite-swash-3b-a600m | GraniteMoeSWAForCausalLM | 6.0 | 6.0 | 0.00 | - | 32.5% | 24.7% |
| ibm-granite/granite-vision-4.1-4b | Granite4VisionForConditionalGeneration | 8.0 | 7.4 | 0.60 | vision/audio tower not given the quant config 0.54; vision/audio projector / patch Linears built without the quant config 0.05 | 30.1% | 22.9% |
| IFM/K2-Horizon-0.9B* | K2HorizonForCausalLM | 2.2 | 2.2 | 0.00 | - | 32.5% | 24.8% |
| IFM/K2-Horizon-3.7B* | K2HorizonForCausalLM | 10.1 | 10.1 | 0.00 | - | 32.5% | 24.8% |
| IFM/K2-Horizon-32B* | K2HorizonForCausalLM | 69.6 | 69.6 | 0.00 | - | 32.5% | 24.8% |
| IFM/K2-Horizon-375B-A23B* | K2HorizonForCausalLM | 758.3 | 758.2 | 0.14 | MoE routers (quant_config=None, fp32 gates) 0.14 | 32.5% | 24.8% |
| IFM/K2-Horizon-7B* | K2HorizonForCausalLM | 18.0 | 18.0 | 0.00 | - | 32.5% | 24.8% |
| IFM/K2-Horizon-MoVA-36B-A4B* | K2HorizonForCausalLM | 74.9 | 74.9 | 0.04 | MoE routers (quant_config=None, fp32 gates) 0.02; small Linears (< 1M weights) 0.01 | 32.5% | 24.7% |
| inclusionAI/Ling-2.5-1T* | BailingMoeV2_5ForCausalLM | 2,024.5 | 2,021.6 | 2.90 | embedding built without the quant config 2.58; MoE routers (quant_config=None, fp32 gates) 0.32 | 32.5% | 24.7% |
| inclusionAI/Ling-2.6-flash* | BailingMoeV2_5ForCausalLM | 208.4 | 207.0 | 1.35 | embedding built without the quant config 1.29; MoE routers (quant_config=None, fp32 gates) 0.07 | 32.3% | 24.6% |
| inclusionAI/Ling-3.0-flash | BailingMoeV3ForCausalLM | 248.8 | 247.9 | 0.92 | embedding built without the quant config 0.80; MoE routers (quant_config=None, fp32 gates) 0.10 | 32.4% | 24.7% |
| inclusionAI/Ling-3.0-flash-VL | BailingMoeV3VLForConditionalGeneration | 249.7 | 248.5 | 1.20 | embedding built without the quant config 0.80; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; MoE routers (quant_config=None, fp32 gates) 0.10 | 32.3% | 24.6% |
| inclusionAI/Ling-3.0-tiny | BailingMoeV3ForCausalLM | 15.8 | 15.2 | 0.54 | embedding built without the quant config 0.48; small Linears (< 1M weights) 0.05 | 31.4% | 23.9% |
| IQuestLab/IQuest-Coder-V1-14B-Instruct* | IQuestCoderForCausalLM | 28.9 | 28.9 | 0.00 | - | 32.5% | 24.8% |
| jdopensource/JoyAI-LLM-Flash* | DeepseekV3ForCausalLM | 97.9 | 97.8 | 0.04 | MoE routers (quant_config=None, fp32 gates) 0.04 | 32.5% | 24.7% |
| JetBrains/Mellum2-12B-A2.5B-Instruct | MellumForCausalLM | 24.3 | 24.3 | 0.01 | - | 32.5% | 24.7% |
| kakaocorp/kanana-2-30b-a3b-instruct-2601* | DeepseekV3ForCausalLM | 61.3 | 61.3 | 0.03 | MoE routers (quant_config=None, fp32 gates) 0.02 | 32.5% | 24.7% |
| kakaocorp/kanana-2-3b-instruct* | Qwen3ForCausalLM | 7.0 | 7.0 | 0.00 | - | 32.5% | 24.8% |
| Kwaipilot/KAT-Coder-V2.5-Dev† | Qwen3_5MoeForConditionalGeneration | 69.3 | 68.3 | 1.07 | embedding built without the quant config 1.02; MoE routers (quant_config=None, fp32 gates) 0.04 | 32.0% | 24.4% |
| LGAI-EXAONE/EXAONE-4.5-33B | Exaone4_5_ForConditionalGeneration | 67.6 | 67.6 | 0.01 | - | 32.5% | 24.8% |
| LGAI-EXAONE/K-EXAONE-2.0-750B-A37B | ExaoneMoeForCausalLM | 1,497.7 | 1,497.4 | 0.24 | MoE routers (quant_config=None, fp32 gates) 0.24 | 32.5% | 24.8% |
| LiquidAI/LFM2-24B-A2B | Lfm2MoeForCausalLM | 47.7 | 47.4 | 0.28 | embedding built without the quant config 0.27; MoE routers (quant_config=None, fp32 gates) 0.01 | 32.3% | 24.6% |
| LiquidAI/LFM2.5-1.2B-Instruct | Lfm2ForCausalLM | 2.3 | 2.1 | 0.27 | embedding built without the quant config 0.27 | 28.8% | 21.9% |
| LiquidAI/LFM2.5-2.6B | Lfm2ForCausalLM | 5.4 | 4.9 | 0.53 | embedding built without the quant config 0.52 | 29.3% | 22.3% |
| LiquidAI/LFM2.5-230M | Lfm2ForCausalLM | 0.5 | 0.3 | 0.13 | embedding built without the quant config 0.13 | 23.0% | 17.5% |
| LiquidAI/LFM2.5-350M | Lfm2ForCausalLM | 0.7 | 0.6 | 0.13 | embedding built without the quant config 0.13 | 26.3% | 20.1% |
| LiquidAI/LFM2.5-8B-A1B | Lfm2MoeForCausalLM | 16.9 | 16.4 | 0.53 | embedding built without the quant config 0.52 | 31.5% | 24.0% |
| LiquidAI/LFM2.5-VL-1.6B | Lfm2VlForConditionalGeneration | 3.2 | 2.6 | 0.57 | embedding built without the quant config 0.27; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; vision/audio projector / patch Linears built without the quant config 0.03 | 26.7% | 20.4% |
| LiquidAI/LFM2.5-VL-3B | Lfm2VlForConditionalGeneration | 6.2 | 5.4 | 0.82 | embedding built without the quant config 0.52; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; vision/audio projector / patch Linears built without the quant config 0.03 | 28.2% | 21.5% |
| LiquidAI/LFM2.5-VL-450M | Lfm2VlForConditionalGeneration | 0.9 | 0.7 | 0.17 | embedding built without the quant config 0.13; vision/audio projector / patch Linears built without the quant config 0.02; small Linears (< 1M weights) 0.01 | 26.4% | 20.1% |
| llm-jp/llm-jp-4-32b-a3b-thinking* | Qwen3MoeForCausalLM | 64.3 | 64.3 | 0.02 | MoE routers (quant_config=None, fp32 gates) 0.02 | 32.5% | 24.7% |
| llm-jp/llm-jp-4-33b-thinking | LlamaForCausalLM | 66.4 | 66.4 | 0.00 | - | 32.5% | 24.8% |
| llm-jp/llm-jp-4-8b-instruct | LlamaForCausalLM | 17.2 | 17.2 | 0.00 | - | 32.5% | 24.8% |
| llm-jp/llm-jp-4.1-32b-a3b-thinking* | Qwen3MoeForCausalLM | 64.3 | 64.3 | 0.02 | MoE routers (quant_config=None, fp32 gates) 0.02 | 32.5% | 24.7% |
| llm-jp/llm-jp-4.1-33b-thinking | LlamaForCausalLM | 66.4 | 66.4 | 0.00 | - | 32.5% | 24.8% |
| llm-jp/llm-jp-4.1-8b-thinking | LlamaForCausalLM | 17.2 | 17.2 | 0.00 | - | 32.5% | 24.8% |
| meituan-longcat/LongCat-Flash-Lite | LongcatFlashNgramForCausalLM | 137.2 | 73.5 | 63.70 | n-gram embedding table built without the quant config (LongCat-Flash-Lite) 62.81; embedding built without the quant config 0.81; not bf16 (F32 tensors) 0.07; n-gram projections (plain parameter) 0.02 | 17.4% | 13.2% |
| meta-models/Muse-Glimmer-30B | MuseGlimmerForConditionalGeneration | 59.6 | 53.0 | 6.54 | vision/audio tower not given the quant config 3.83; embedding built without the quant config 2.69 | 28.9% | 22.0% |
| microsoft/Phi-4-reasoning-vision-15B* | Phi4ForCausalLMV | 30.2 | 28.9 | 1.31 | embedding built without the quant config 1.03; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.28 | 31.1% | 23.7% |
| MiniMaxAI/MiniMax-M3 | MiniMaxM3SparseForConditionalGeneration | 854.2 | 27.7 | 826.52 | MoE experts: swigluoai (MiniMax-M3) 826.24; MoE routers (quant_config=None, fp32 gates) 0.18; small Linears (< 1M weights) 0.09 | 1.1% | 0.8% |
| Nanbeige/Nanbeige4.1-3B | LlamaForCausalLM | 7.9 | 7.9 | 0.00 | - | 32.5% | 24.8% |
| Nanbeige/Nanbeige4.2-3B* | NanbeigeForCausalLM | 8.3 | 8.3 | 0.00 | - | 32.5% | 24.8% |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 | NemotronH_Nano_Omni_Reasoning_V3 | 66.0 | 3.7 | 62.35 | MoE experts: relu2 without a gate (Nemotron-H) 58.75; vision/audio tower not given the quant config 2.68; embedding built without the quant config 0.70; conv weights / patch embeddings 0.19; MoE routers (quant_config=None, fp32 gates) 0.02 | 1.8% | 1.4% |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | NemotronHForCausalLM | 7.9 | 4.8 | 3.13 | Linear shape: rows not a multiple of 64 (decoder) 2.31; embedding built without the quant config 0.82 | 19.7% | 15.0% |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | NemotronHForCausalLM | 241.3 | 14.6 | 226.73 | MoE experts: relu2 without a gate (Nemotron-H) 225.49; embedding built without the quant config 1.07; MoE routers (quant_config=None, fp32 gates) 0.17 | 2.0% | 1.5% |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | NemotronHForCausalLM | 1,098.6 | 65.3 | 1,033.36 | MoE experts: relu2 without a gate (Nemotron-H) 1030.79; embedding built without the quant config 2.15; MoE routers (quant_config=None, fp32 gates) 0.40 | 1.9% | 1.5% |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | NemotronHForCausalLM | 63.2 | 3.7 | 59.47 | MoE experts: relu2 without a gate (Nemotron-H) 58.75; embedding built without the quant config 0.70; MoE routers (quant_config=None, fp32 gates) 0.02 | 1.9% | 1.4% |
| nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 | NemotronHPuzzleForCausalLM | 150.7 | 15.5 | 135.20 | MoE experts: relu2 without a gate (Nemotron-H) 135.02; MoE routers (quant_config=None, fp32 gates) 0.17 | 3.3% | 2.5% |
| openbmb/MiniCPM-o-4_5* | MiniCPMO | 18.7 | 17.1 | 1.63 | embedding built without the quant config 1.29; Linear shape: rows not a multiple of 64 (decoder) 0.30; small Linears (< 1M weights) 0.03 | 29.7% | 22.6% |
| openbmb/MiniCPM-V-4.6* | MiniCPMV4_6ForConditionalGeneration | 2.5 | 1.7 | 0.76 | embedding built without the quant config 0.51; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.24; plain nn.Linear / nn.Embedding in towers 0.01 | 22.6% | 17.2% |
| openbmb/MiniCPM5-1B | LlamaForCausalLM | 2.2 | 2.2 | 0.00 | - | 32.5% | 24.8% |
| openbmb/MiniCPM5-2B | LlamaForCausalLM | 5.0 | 5.0 | 0.00 | - | 32.5% | 24.8% |
| ornith-ai/Ornith-1.5-35B-A3B† | Qwen3_5MoeForConditionalGeneration | 70.2 | 68.9 | 1.35 | embedding built without the quant config 1.02; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; MoE routers (quant_config=None, fp32 gates) 0.04 | 31.9% | 24.3% |
| ornith-ai/Ornith-1.5-9B† | Qwen3_5ForConditionalGeneration | 18.8 | 16.5 | 2.33 | embedding built without the quant config 2.03; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; small Linears (< 1M weights) 0.01 | 28.5% | 21.7% |
| poolside/Laguna-M.1* | LagunaForCausalLM | 451.6 | 451.4 | 0.14 | MoE routers (quant_config=None, fp32 gates) 0.14 | 32.5% | 24.7% |
| poolside/Laguna-S-2.1* | LagunaForCausalLM | 235.1 | 235.0 | 0.09 | MoE routers (quant_config=None, fp32 gates) 0.07; small Linears (< 1M weights) 0.02 | 32.5% | 24.7% |
| poolside/Laguna-XS-2.1* | LagunaForCausalLM | 66.9 | 66.8 | 0.05 | MoE routers (quant_config=None, fp32 gates) 0.04 | 32.5% | 24.7% |
| poolside/Laguna-XS.2* | LagunaForCausalLM | 66.9 | 66.8 | 0.05 | MoE routers (quant_config=None, fp32 gates) 0.04 | 32.5% | 24.7% |
| Qwen/Qwen3-Coder-Next | Qwen3NextForCausalLM | 159.3 | 158.0 | 1.36 | embedding built without the quant config 0.62; LM head built without the quant config 0.62; MoE routers (quant_config=None, fp32 gates) 0.10 | 32.2% | 24.5% |
| Qwen/Qwen3.5-0.8B | Qwen3_5ForConditionalGeneration | 1.7 | 1.2 | 0.53 | embedding built without the quant config 0.51; small Linears (< 1M weights) 0.02 | 22.4% | 17.1% |
| Qwen/Qwen3.5-122B-A10B | Qwen3_5MoeForConditionalGeneration | 245.1 | 243.2 | 1.91 | embedding built without the quant config 1.53; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; MoE routers (quant_config=None, fp32 gates) 0.08; small Linears (< 1M weights) 0.03 | 32.2% | 24.6% |
| Qwen/Qwen3.5-27B | Qwen3_5ForConditionalGeneration | 54.7 | 51.8 | 2.87 | embedding built without the quant config 2.54; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; small Linears (< 1M weights) 0.05 | 30.8% | 23.4% |
| Qwen/Qwen3.5-2B | Qwen3_5ForConditionalGeneration | 4.4 | 3.4 | 1.03 | embedding built without the quant config 1.02 | 24.9% | 19.0% |
| Qwen/Qwen3.5-35B-A3B | Qwen3_5MoeForConditionalGeneration | 70.2 | 68.9 | 1.35 | embedding built without the quant config 1.02; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; MoE routers (quant_config=None, fp32 gates) 0.04 | 31.9% | 24.3% |
| Qwen/Qwen3.5-397B-A17B | Qwen3_5MoeForConditionalGeneration | 793.6 | 791.0 | 2.62 | embedding built without the quant config 2.03; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; MoE routers (quant_config=None, fp32 gates) 0.25; small Linears (< 1M weights) 0.05 | 32.4% | 24.7% |
| Qwen/Qwen3.5-4B | Qwen3_5ForConditionalGeneration | 9.1 | 7.8 | 1.29 | embedding built without the quant config 1.27 | 27.9% | 21.2% |
| Qwen/Qwen3.5-9B | Qwen3_5ForConditionalGeneration | 18.8 | 16.5 | 2.33 | embedding built without the quant config 2.03; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; small Linears (< 1M weights) 0.01 | 28.5% | 21.7% |
| Qwen/Qwen3.6-27B | Qwen3_5ForConditionalGeneration | 54.7 | 51.8 | 2.87 | embedding built without the quant config 2.54; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; small Linears (< 1M weights) 0.05 | 30.8% | 23.4% |
| Qwen/Qwen3.6-35B-A3B | Qwen3_5MoeForConditionalGeneration | 70.2 | 68.9 | 1.35 | embedding built without the quant config 1.02; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; MoE routers (quant_config=None, fp32 gates) 0.04 | 31.9% | 24.3% |
| Qwen/Qwen3.8-2.4T-A95B | Qwen3_5MoeForCausalLM | 4,839.6 | 4,834.8 | 4.86 | embedding built without the quant config 4.07; MoE routers (quant_config=None, fp32 gates) 0.77; conv weights / patch embeddings 0.01 | 32.5% | 24.7% |
| Qwen/Qwen3.8-27B | Qwen3_5ForConditionalGeneration | 54.7 | 51.8 | 2.87 | embedding built without the quant config 2.54; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; small Linears (< 1M weights) 0.05 | 30.8% | 23.4% |
| Qwen/Qwen3.8-Flash-Next | Qwen4ExpForConditionalGeneration | 354.8 | 248.1 | 106.65 | n-gram (PLE) embedding table (Qwen4Exp: refuses the glyd config) 102.40; decoder Linears built without the quant config (hyper-connections, misc) 1.28; embedding built without the quant config 1.27; LM head built without the quant config 1.27; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; MoE routers (quant_config=None, fp32 gates) 0.13; small Linears (< 1M weights) 0.02 | 0 (does not load; 22.7% once it does) | 0 (17.3%) |
| sapientinc/HRM-Text-1B* | HrmTextForCausalLM | 2.4 | 2.4 | 0.00 | - | 32.5% | 24.8% |
| stepfun-ai/Step-3.5-Flash | Step3p5ForCausalLM | 393.9 | 374.6 | 19.31 | MoE experts: swiglustep (Step 3.5/3.7 clamped layers) 18.12; embedding built without the quant config 1.06; MoE routers (quant_config=None, fp32 gates) 0.10; small Linears (< 1M weights) 0.03 | 30.9% | 23.5% |
| stepfun-ai/Step-3.7-Flash | Step3p7ForConditionalGeneration | 397.9 | 378.1 | 19.75 | MoE experts: swiglustep (Step 3.5/3.7 clamped layers) 18.12; embedding built without the quant config 1.06; conv weights / patch embeddings 0.43; MoE routers (quant_config=None, fp32 gates) 0.10; small Linears (< 1M weights) 0.03 | 30.9% | 23.5% |
| stepfun-ai/Step3-VL-10B | StepVLForConditionalGeneration | 20.3 | 19.9 | 0.44 | conv weights / patch embeddings 0.43 | 31.8% | 24.2% |
| swiss-ai/Apertus-v1.1-0.5B-Instruct | ApertusForCausalLM | 1.1 | 1.1 | 0.00 | - | 32.5% | 24.8% |
| swiss-ai/Apertus-v1.1-1.5B-Instruct | ApertusForCausalLM | 3.0 | 3.0 | 0.00 | - | 32.5% | 24.8% |
| swiss-ai/Apertus-v1.1-4B-Instruct | ApertusForCausalLM | 7.7 | 7.7 | 0.00 | - | 32.5% | 24.8% |
| Tele-AI/TeleChat3-Coder-36B-Thinking* | TeleChat3ForCausalLM | 72.5 | 72.5 | 0.00 | - | 32.5% | 24.8% |
| tencent/Hy3 | HYV3ForCausalLM | 590.1 | 589.0 | 1.12 | embedding built without the quant config 0.99; MoE routers (quant_config=None, fp32 gates) 0.12 | 32.4% | 24.7% |
| tencent/Hy4-preview | HYV4ForCausalLM | 1,539.9 | 1,536.5 | 3.38 | embedding built without the quant config 1.48; LM head kept in float32 (head_dtype) 1.48; MoE routers (quant_config=None, fp32 gates) 0.24; not bf16 (F32 tensors) 0.12; small Linears (< 1M weights) 0.04 | 32.4% | 24.7% |
| thinkingmachines/Inkling | InklingForConditionalGeneration | 1,894.2 | 1,876.9 | 17.36 | shared sink experts as plain parameters (Inkling) 14.50; embedding as a plain parameter (Inkling) 2.47; MoE routers (quant_config=None, fp32 gates) 0.20; vision/audio tower not given the quant config 0.18 | 32.2% | 24.5% |
| thinkingmachines/Inkling-Small | InklingForConditionalGeneration | 527.4 | 521.5 | 5.90 | shared sink experts as plain parameters (Inkling) 4.03; embedding as a plain parameter (Inkling) 1.65; vision/audio tower not given the quant config 0.14; MoE routers (quant_config=None, fp32 gates) 0.08 | 32.1% | 24.5% |
| tiiuae/Falcon-H1-Tiny-R-0.6B* | FalconH1ForCausalLM | 1.2 | 0.7 | 0.55 | Linear shape: rows not a multiple of 64 (decoder) 0.30; small Linears (< 1M weights) 0.12; LM head built without the quant config 0.07; embedding built without the quant config 0.07 | 18.1% | 13.8% |
| XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B† | Qwen3_5ForConditionalGeneration | 18.8 | 16.5 | 2.33 | embedding built without the quant config 2.03; Linear shape: rows not a multiple of 64 (vision MLP 4304) 0.27; small Linears (< 1M weights) 0.01 | 28.5% | 21.7% |
| zai-org/GLM-4.7-Flash | Glm4MoeLiteForCausalLM | 59.9 | 59.9 | 0.01 | MoE routers (quant_config=None, fp32 gates) 0.01 | 32.5% | 24.7% |
| zai-org/GLM-5 | GlmMoeDsaForCausalLM | 1,487.8 | 1,487.4 | 0.39 | MoE routers (quant_config=None, fp32 gates) 0.24; decoder Linears built without the quant config (hyper-connections, misc) 0.15 | 32.5% | 24.7% |
| zai-org/GLM-5.3-BF16 | GlmMoeDsaForCausalLM | 1,486.8 | 1,486.5 | 0.28 | MoE routers (quant_config=None, fp32 gates) 0.24; decoder Linears built without the quant config (hyper-connections, misc) 0.04 | 32.5% | 24.8% |
| zai-org/GLM-5.3-Flash-BF16 | Glm5NextForConditionalGeneration | 627.8 | 623.5 | 4.23 | MLA attention built with quant_config=None 2.74; embedding built without the quant config 1.27; MoE routers (quant_config=None, fp32 gates) 0.10; hyper-connection parameters (plain) 0.07; conv weights / patch embeddings 0.04; small Linears (< 1M weights) 0.01 | 32.3% | 24.6% |

## Unpacked classes, ranked

unpacked classes over the models inventoried (fine-tunes of a listed base counted once), by GB and by number of models with >= 0.5 MB of the class.

| Class | GB | Models |
| :-- | --: | --: |
| MoE experts: relu2 without a gate (Nemotron-H) | 1,508.80 | 5 |
| MoE experts: swigluoai (MiniMax-M3) | 826.24 | 1 |
| n-gram (PLE) embedding table (Qwen4Exp: refuses the glyd config) | 102.40 | 1 |
| MoE experts: GELU-tanh (Gemma 4) | 91.35 | 2 |
| n-gram embedding table built without the quant config (LongCat-Flash-Lite) | 62.81 | 1 |
| embedding built without the quant config | 58.39 | 52 |
| MoE experts: swiglustep (Step 3.5/3.7 clamped layers) | 36.24 | 2 |
| shared sink experts as plain parameters (Inkling) | 18.52 | 2 |
| vision/audio tower not given the quant config | 11.21 | 9 |
| Mamba (v1) mixer Linears built without the quant config (Jamba) | 8.03 | 2 |
| MoE routers (quant_config=None, fp32 gates) | 5.79 | 55 |
| embedding as a plain parameter (Inkling) | 4.12 | 2 |
| Linear shape: rows not a multiple of 64 (vision MLP 4304) | 4.01 | 16 |
| Linear shape: rows not a multiple of 64 (decoder) | 2.91 | 3 |
| LM head built without the quant config | 2.83 | 5 |
| MLA attention built with quant_config=None | 2.74 | 1 |
| conv weights / patch embeddings | 1.49 | 57 |
| LM head kept in float32 (head_dtype) | 1.49 | 1 |
| decoder Linears built without the quant config (hyper-connections, misc) | 1.48 | 5 |
| small Linears (< 1M weights) | 1.24 | 38 |
| vision/audio projector / patch Linears built without the quant config | 0.29 | 5 |
| not bf16 (F32 tensors) | 0.20 | 4 |
| plain nn.Linear / nn.Embedding in towers | 0.10 | 23 |
| norms | 0.10 | 112 |
| hyper-connection parameters (plain) | 0.07 | 1 |
| biases | 0.03 | 45 |
| n-gram projections (plain parameter) | 0.02 | 1 |
| small params (A_log, dt_bias, D, scales) | 0.01 | 73 |

The same classes by how many models have them (the classes of 5 or more models):

| Class | Models | GB |
| :-- | --: | --: |
| norms | 112 | 0.10 |
| small params (A_log, dt_bias, D, scales) | 73 | 0.01 |
| conv weights / patch embeddings | 57 | 1.49 |
| MoE routers (quant_config=None, fp32 gates) | 55 | 5.79 |
| embedding built without the quant config | 52 | 58.39 |
| biases | 45 | 0.03 |
| small Linears (< 1M weights) | 38 | 1.24 |
| plain nn.Linear / nn.Embedding in towers | 23 | 0.10 |
| Linear shape: rows not a multiple of 64 (vision MLP 4304) | 16 | 4.01 |
| vision/audio tower not given the quant config | 9 | 11.21 |
| MoE experts: relu2 without a gate (Nemotron-H) | 5 | 1,508.80 |
| LM head built without the quant config | 5 | 2.83 |
| decoder Linears built without the quant config (hyper-connections, misc) | 5 | 1.48 |
| vision/audio projector / patch Linears built without the quant config | 5 | 0.29 |

## Fix plan (in order)

Files: `glyd-gpu-py/glyd_gpu/vllm_plugin.py` (the plugin), `glyd-gpu-py/glyd_gpu/preflight.py` (the size estimate), `gpu/glyd_gpu.cu` (kernels). Every item keeps bit-for-bit output; each needs its own pod check (`gpu/vllm/acceptance.sh` plus the plugin's exact checks) on a 2026 model that has the class.

**0. Correctness first: clamped SwiGLU experts are packed and run without the clamp.** vLLM 0.30 carries a SwiGLU clamp as `FusedMoEConfig.swiglu_limit` (and `swiglu_alpha`, `swiglu_beta`) with `activation` still `SILU`. `GlydMoEMethod.unsupported()` checks only the activation, so Hy4-preview (limit 10) and GLM-5.3-Flash (limit 10) have their experts packed, and the grouped products (`mma_moe`, fused mode, below `decode_min` tokens a step) apply plain SiLU: whenever a gate or up value passes the limit the output differs from vLLM's bf16. The decoded route (vLLM's Triton kernel on the decoded experts) applies the clamp. Change: in `GlydMoEMethod.__init__`, where `moe.swiglu_limit` (or alpha/beta) is set, force the decoded route (`self.decode_min = 0`, as exact mode does; refuse where vLLM's bf16 backend is not Triton), or return a reason from `unsupported()` until `mma_moe`'s epilogue takes the clamp. Add the clamp to `mma_moe` (gate `min(g, limit)`, up `clamp(u, -limit, limit)`) to get the fused speed back. Test on GLM-5.3-Flash (643 GB: an H100x8 or B200 pod) or a reduced-layer copy.

**1. Embeddings (and LM heads) that vLLM builds without the quant config: 52 models, 58.4 GB (heads: 5 models, 2.8 GB).** Qwen3.5/3.6/3.8, Qwen3-Next, Qwen4Exp, LFM2/2.5, Nemotron-H, Jamba, Olmo-Hybrid, Muse-Glimmer, Cohere2Moe/Compass, Bailing (Ling 3.0), Step 3.5/3.7, Hy3/Hy4, GLM-5.3-Flash construct `VocabParallelEmbedding(vocab, hidden)` with no `quant_config`, so `get_quant_method` is never called and the table stays bf16 (Qwen3-Next, Qwen4Exp and Jamba build `ParallelLMHead` the same way; a tied head then stays too, `GlydEmbeddingMethod.tie_weights` falling back to vLLM's). The plugin's log line "the embedding and the LM head packed too" is printed from the plan, not from the layers, so on Qwen3.5-9B it claims a pack that does not happen (the measured 12.68 GiB matches the embedding unpacked: see the check above). Change, in `vllm_plugin.register()`: wrap `VocabParallelEmbedding.__init__` so that, when `quant_config is None and quant_method is None` and `get_current_vllm_config_or_none().quant_config` is a `GlydConfig`, it passes that config (ParallelLMHead goes through the same `__init__`); `table_method`/`table_why` then decide as for any table (rows < 4096 and dim % 64 stay; speculative decoding and fraction < 1 keep them; a tied head shares the pack). Fix the log line to say what each layer did. Risks: a model whose forward reads `embed_tokens.weight` directly (none of the 2026 files above do; Hy4 ties by assignment only when `tie_word_embeddings`, which its checkpoint does not set); fused embed+norm kernels that require an unquantized table (`has_full_vocab_on_rank` already checks); a table built with `prefix=""` is not found in a glyd save's manifest (it packs from bf16 at load instead: give `saved()` the model's own name for it). `preflight.table_bytes()` must count what is packed after this (today it assumes both tables packed for every model). Gain (projected, tiered): Qwen3.5-0.8B +9.7 points (22.4% to 32.1%), Qwen3.5-2B +7.5, Qwen3.5-9B +3.5 (28.5% to 32.0%), Qwen3.5-27B, 3.6-27B, 3.8-27B +1.5, LFM2.5-230M +9.5, North-Micro-Vision +7.0.

**2. Mixture-of-experts layers with an activation the grouped products do not have: 1,509 GB on the five Nemotron-H MoEs (Nano-Omni, 3.5-Lightning, 3-Super, 3-Ultra, Labs-3-Puzzle), 826 GB on MiniMax-M3, 91.4 GB on Gemma 4 26B-A4B and diffusiongemma-26B-A4B, 36 GB on Step 3.5/3.7 (their two clamped layers).** `GlydMoEMethod.unsupported()` keeps them bf16: Nemotron-H's routed experts are relu² without a gate (`activation_without_mul`, w13 is [E, I, H], not [E, 2I, H]); Gemma 4's are GELU-tanh; Step's clamped layers are `swiglustep`; MiniMax-M3's `swigluoai_uninterleave` (alpha 1.702, limit 7). These models save 1.0 to 3% today (projected) and 32% with the experts packed. Two steps: (a) plugin only, the memory now: for an activation vLLM's Triton kernel runs, take the decoded route for every token count (`decode_min = 0`): `mma_moe_unpack` decodes the routed experts and `moe_kernel.apply(..., activation=layer.activation)` runs them, so the result is vLLM's bit for bit; this needs the no-gate shape (w13 [E, I, H]: `(2 * I) % 128` becomes `I % 64`, and the w13 pack and unpack sized I) and costs a decode a step (slower at small batches; memory-only wording until measured). (b) kernels: `mma_moe` epilogues for relu² without a gate, GELU-tanh times up, swiglustep and swigluoai, so the fused route serves 1 to `decode_min` tokens. Nemotron first (most bytes over most models), MiniMax-M3 (one model, 826 GB), then Gemma 4 and Step.

**3. n-gram embedding tables: Qwen3.8-Flash-Next does not load with `--quantization glyd`; LongCat-Flash-Lite's 62.8 GB table (45% of the model) is built without the quant config.** `Qwen4ExpPLEEmbeddingMethod.from_quant_config` (vllm/models/qwen4_exp/nvidia/ngram_embedding.py) raises `NotImplementedError` for any quant config but FP8/ModelOpt, and the model has a PLE n-gram table on layer 2 (128 shards of [2,500,012, 160], 102.4 GB of its 354.8 GB). Change: in `register()`, wrap `from_quant_config` to return the unquantized PLE method for a `GlydConfig` (the model then loads: projected 22.7% tiered with the table bf16, its LM head and embedding also built without the quant config: item 1). Then a Glyd PLE method (row lookups from a pack, as `vllm_embed`): its rows are 160 wide, not a multiple of 64 (`mma_rows` needs dim % 64; columns padded to 192 would cost 20% of the table's saving): kernel support for 160 (or % 32) widths first. With `--engram-config` CPU offload the table sits in pinned host memory and is not GPU memory at all: say so in the model's notes. LongCat-Flash-Lite (`longcat_flash_ngram.py`): its `oe_embedder` is a `VocabParallelEmbedding` of 12 stacked tables, 256 wide, built without the quant config: item 1's change gives it the config, and `table_why` takes it (dim 256 % 64 = 0, 10.2M rows): projected 17.4% to 32.3% (32.5% with its embedding).

**4. Linear rows not a multiple of 64: SigLIP-style vision MLPs (fc1 4,304 rows, 16 models, 4.0 GB) and Mamba2 `in_proj` rows in Nemotron-3-Nano-4B (17,504 rows, 2.3 GB, 29% of that model), Falcon-H1-Tiny and MiniCPM-o (3 models, 2.9 GB).** `GlydLinearMethod.create_weights` hands any `O % 64` matrix to vLLM. Change: pad as `GlydEmbeddingMethod` already does: meta weight `[ceil64(O), K]`, the loaders write the first O rows (QKV/merged shard offsets are unchanged), `_check_loaded` as now, the pad rows zero; `glyd::vllm_linear` takes `out_features` = O and the kernels write O columns (or the op slices, as `vllm_logits` does with `rows != out_features`). Risk: `_decoded`'s cuBLAS route reads the decoded `[O_pad, K]` buffer: pass `w[:O]`.

**5. Vision/audio towers that vLLM gives no quant config: 9 models, 11.2 GB** (Gemma 4 26B/31B and diffusiongemma: `gemma4_mm.py` withholds it when the vision intermediate 4,304 is not a multiple of 64; SigLIP `fc1/fc2` in Granite vision and Command A+ for the same reason (`siglip.py` `quantizable`); Muse-Glimmer's encoder (QKV/o built without it, MLP and adapter plain `nn.Linear`); Nemotron Omni's RADIO and Parakeet; Inkling's towers (plain `nn.Linear`)). Towers that do get it (Qwen3.5's `visual`, EXAONE 4.5, Step, GLM-5.3-Flash, Ling VL, Gemma 4 E2B/E4B) are packed today. Change: a post-load pass in the plugin, wrapping `vllm.model_executor.model_loader.utils.process_weights_after_loading` (after vLLM's own processing): for modules under a tower prefix (`visual`, `vision_tower`, `vision_model`, `vision_encoder`, `audio_tower`, `audio`, `sound_encoder`, projectors) whose 2-D bf16 weight is >= 1M weights and still on `UnquantizedLinearMethod`, or is a plain `nn.Linear`, pack it and route its forward through `glyd::vllm_linear` (bias kept), dropping the bf16. The peak is today's (the tower is resident in bf16 until then). With item 4, the 4,304-row MLPs pack too. Risks: a tower module whose forward reads `.weight` itself (check each: HF `AutoModel` towers in Gemma 4 call `nn.Linear` modules; MMEncoderAttention takes q/k/v outputs); encoder CUDA graphs (Kimi-style `prepare_encoder_cudagraph_metadata`) capture the op as for decoders; prefill of many image patches takes the decoded route (cuBLAS on a decoded buffer: a decode per tower Linear per image batch).

**6. Mamba v1 mixer Linears (Jamba2: 8.0 GB; 31% of Jamba2-3B, 5.3% of Jamba2-Mini)** are built with no quant config (`layers/mamba/mamba_mixer.py`: `in_proj`, `x_proj`, `dt_proj`, `out_proj`). The forward calls them as modules, so the post-load pass of item 5 takes them with an allowlist entry for `.mamba.` (not `conv1d`, whose weight the scan reads directly); `x_proj` has 288 rows (item 4). Also Jamba's embedding and LM head (item 1).

**7. Inkling (1,894 GB) and Inkling-Small (527 GB): the embedding is an `InklingReplicatedEmbedding` (a plain parameter read by the fused `embed_rmsnorm` kernel) and the shared "sink" experts are plain bf16 parameters (`InklingSinkExperts`: 13.5 GB and 3.75 GB).** Model-specific: replace the embedding's forward with a packed row lookup followed by the RMSNorm (losing the fused gather+norm kernel), and pack each sink expert's w13/w2 as Linears for the plain GEMMs it runs. Gain 0.3 and 0.35 points (both models are at 32.2% already): after items 0 to 6.

**8. Small, model-specific:** Hy4-preview's LM head (head_dtype float32 from `enable_lm_head_fp32`: the plan keeps it; packing needs `vllm_logits` to accumulate into fp32 output, 1.5 GB of 1,540); GLM-5.3-Flash's MLA layers (`quant_config=None`, 2.7 GB of 643: the post-load pass with a `self_attn.` allowlist for that architecture, after checking the MLA kernels take the products from the modules); Qwen4Exp's hyper-connection Linears (`quant_config=None`, 1.3 GB of 355).

**Leave as they are:** MoE routers (`GateLinear`, `quant_config=None`, fp32 logits: 5.8 GB over 55 models), Linears under 1M weights (1.2 GB over 38 models: a product takes 3x bf16's time for nothing), conv weights read directly by the scans (`conv1d`, short conv), norms, biases, A_log/dt_bias/D: together under 0.3% of any model.

**Then the layout.** With everything above packed the tiered layout gives at most 32.5% (10.80 bits a weight); 33% needs at most 10.72 bits on every weight. The 12-bit layout, which `best_layout` picks on every GPU but Ada, caps at 24.75%: the projections show both. On an A100/H100 a mixture of experts' attention Linears also stay bf16 (`dense_stays`): `projected_12bit_saving_on_a100_h100_pct` per MoE model in inventory.json.

## Not inventoried

- moonshotai/Kimi-K3: not a bf16 checkpoint: experts MXFP4 (U8, 2.72T values), 57B bf16 params (attention, embeddings)
- moonshotai/Kimi-K2.5: not a bf16 checkpoint: experts INT4 (I32-packed, 1.01T), 12B bf16 params
- moonshotai/Kimi-K2.6: not a bf16 checkpoint: experts INT4 (I32-packed), as K2.5
- moonshotai/Kimi-K2.7-Code: not a bf16 checkpoint: experts INT4 (I32-packed), as K2.5
- swiss-ai/Apertus-v1.5-8B: gated on Hugging Face: headers need an accepted license and a token (8.6B bf16 params + 294M F32; ApertusForCausalLM, as Apertus-v1.1-4B: everything packable)
- swiss-ai/Apertus-v1.5-70B: gated on Hugging Face (71.7B bf16 params + 294M F32; ApertusForCausalLM)
- zai-org/GLM-5.1: same architecture and parameter count as zai-org/GLM-5 (753,864,119,552 bf16): GLM-5's row applies (282 shards not re-read)
- zai-org/GLM-5.2: same architecture and parameter count as zai-org/GLM-5.3-BF16 (753,329,921,024 bf16): GLM-5.3-BF16's row applies
- Motif-Technologies/Motif-3: MotifForCausalLM: listed in vLLM 0.30.0's registry as removed (since 0.10.2): vLLM does not serve it
- ornith-ai/Ornith-1.5-397B: a Qwen3.5-397B-A17B fine-tune (third party): Qwen3.5-397B-A17B's row applies
- ibm-granite/granite-4.0-3b-vision: a LoRA layout (base_model.model.*, lora_A/lora_B over GraniteMoeHybrid): not a plain bf16 checkpoint
- google/gemma-4-E2B, -E4B, -12B, -26B-A4B, -31B (base): the same tensors as the -it rows
- google/gemma-4-*-it-assistant: speculative drafts (vLLM keeps a draft bf16 unless --speculative-config asks for glyd)
- facebook/MobileMoE-S-SFT: gated on Hugging Face (1.3B params, MobileMoEForCausalLM): headers need an accepted license and a token
- facebook/MobileMoE-M-SFT: gated on Hugging Face (2.8B params, MobileMoEForCausalLM): headers need an accepted license and a token
- facebook/MobileMoE-L-SFT: gated on Hugging Face (5.3B params, MobileMoEForCausalLM): headers need an accepted license and a token
- mistralai/Mistral-Small-4-119B-2603: not a bf16 checkpoint (fp8)
- mistralai/Mistral-Medium-3.5-128B: not a bf16 checkpoint (fp8)
- microsoft/Mage-VL: MageVLForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- nvidia/Nemotron-Labs-Diffusion-3B: NemotronLabsDiffusionModel: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- nvidia/Nemotron-Labs-Diffusion-8B: NemotronLabsDiffusionModel: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- nvidia/Nemotron-Labs-Diffusion-14B: NemotronLabsDiffusionModel: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- deepseek-ai/DeepSeek-V4-Flash: not a bf16 checkpoint (fp8)
- deepseek-ai/DeepSeek-V4-Pro: not a bf16 checkpoint (fp8)
- deepseek-ai/DeepSeek-V4.1-Flash: not a bf16 checkpoint (fp8)
- zai-org/GLM-5.3: not a bf16 checkpoint (fp8)
- zai-org/GLM-5.3-Flash: not a bf16 checkpoint (fp8)
- MiniMaxAI/MiniMax-M2.5: not a bf16 checkpoint (fp8)
- MiniMaxAI/MiniMax-M2.7: not a bf16 checkpoint (fp8)
- XiaomiMiMo/MiMo-V2.5: not a bf16 checkpoint (fp8)
- XiaomiMiMo/MiMo-V2.5-Pro: not a bf16 checkpoint (fp8)
- XiaomiMiMo/MiMo-V2.6-Flash-RL: not a bf16 checkpoint (fp8)
- XiaomiMiMo/MiMo-V2.6-Pro-RL: not a bf16 checkpoint (fp8)
- ByteDance-Seed/Stable-DiffCoder-8B-Instruct: StableDiffcoderForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- meituan-longcat/LongCat-Flash-Thinking-2601: LongcatCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- meituan-longcat/LongCat-Next: LongcatNextForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- meituan-longcat/LongCat-2.0: LongcatCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- inclusionAI/Ring-2.5-1T: not a bf16 checkpoint (compressed-tensors)
- inclusionAI/Ling-2.6-1T: not a bf16 checkpoint (compressed-tensors)
- inclusionAI/Ring-2.6-1T: not a bf16 checkpoint (compressed-tensors)
- inclusionAI/Ming-flash-omni-2.0: BailingMM2NativeForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- inclusionAI/LLaDA2.1-mini: LLaDA2MoeModelLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- inclusionAI/LLaDA2.1-flash: LLaDA2MoeModelLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- inclusionAI/LLaDA2.2-mini: LLaDA2MoeModelLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- inclusionAI/LLaDA2.2-flash: LLaDA2MoeModelLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- Kwai-Keye/Keye-VL-2.0-30B-A3B: KeyeVL2MoeForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- CohereLabs/tiny-aya-global: gated on Hugging Face (3.3B params, Cohere2ForCausalLM): headers need an accepted license and a token
- upstage/Solar-Open2-250B: SolarOpen2ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- sarvamai/sarvam-30b: not a bf16 checkpoint (F32 32.2B)
- sarvamai/sarvam-105b: not a bf16 checkpoint (F32 106.0B)
- ServiceNow-AI/SuperApriel-15B-Instruct: not a bf16 checkpoint ()
- RekaAI/reka-edge-2603: Yasa2ForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- openbmb/MiniCPM-SALA: MiniCPMSALAForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- internlm/Intern-S1-Pro: not a bf16 checkpoint (fp8)
- amd/Instella-MoE-16B-A3B-Think: InstellaMoEForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- Zyphra/ZAYA1-8B: ZayaForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- Zyphra/ZAYA1-VL-8B: Zaya1VLForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- Zyphra/ZAYA1-74B-preview: ZayaForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- kakaocorp/kanana-2-1.3b-instruct: Kanana2TinyForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- trillionlabs/Gravity-16B-A3B-Preview: GravityMoEForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- trillionlabs/Gravity-30B-A5B-Preview: GravityMoEForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- trillionlabs/Tri-21B-Think: not a bf16 checkpoint (F32 20.7B)
- trillionlabs/Trida-7B: TridaForDLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- skt/A.X-K2: not a bf16 checkpoint (fp8)
- Aleph-Alpha/Kolibri-1: not a bf16 checkpoint (fp8)
- PleIAs/Baguettotron-MoE: BaguettotronMoeForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- pfnet/plamo-3-nict-2604-31b-base: gated on Hugging Face (31.8B params, Plamo3ForCausalLM): headers need an accepted license and a token
- yandex/AliceAI-Foundation-80B-A3B-Base: AliceAIForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- yandex/AliceAI-T5-35B-A0.6B: AliceAIT5MoEForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- XingChen-AGI/Xing4.0-29B-A4B: Xing4_0ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- Agnes-AI/Agnes-3.0-Flash: AgnesForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- Agnes-AI/Agnes-2.5-Flash-Base: not a bf16 checkpoint (fp8)
- RWKV/RWKV7-G1k-1.5B-20260930: Rwkv7ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- RWKV/RWKV7-G1k-2.9B-20260930: Rwkv7ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- RWKV/RWKV7-G1k-7.2B-20260930: Rwkv7ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- RWKV/RWKV7-G1k-13.3B-20260930: Rwkv7ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- ai-sage/GigaChat3.5-432B-A28B: not a bf16 checkpoint (fp8)
- ai-sage/GFusion-10B-A1.8B-bf16: GFusionForDiffusionLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- moondream/moondream3.1-9B-A2B: not a bf16 checkpoint (F32 0.0B, BF16 1.2B, F8_E4M3 8.1B)
- Gensyn/open-1b-sft: not a bf16 checkpoint (F32 1.6B)
- EssentialAI/rnj-1.5-instruct: not a bf16 checkpoint (F32 8.3B)
- TaichuAI/ZDTaichu5.0-9B: ZDTaichu5_0_ForConditionalGeneration: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- IQuestLab/IQuest-Q1: IQuestQ1ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- NaiveAI/Naive-N0.5-Flash: not a bf16 checkpoint ()
- Soofi-Project/Soofi-S-Instruct-Preview: gated on Hugging Face (31.6B params, NemotronHForCausalLM): headers need an accepted license and a token
- Maincode/matilda-jev-v1: MatildaJevModel: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- openpangu/openPangu-2.0-Flash: OpenPanguV2ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)
- openpangu/openPangu-2.0-Pro: OpenPanguV2ForCausalLM: not in vLLM 0.30.0's registry (vLLM does not serve it; the plugin does not apply)

## Files

- `inventory.json`: per model the bytes (total, loaded, packed, unpacked by class and in detail, not loaded), the towers' packed bytes, the projections, the gain per class, the vLLM rules used; the ranking; the exclusions.
- `fetch.py` (headers by HTTP range), `classify.py` (the plugin's rules and the per-architecture vLLM facts, each with the file it was read from), `autofam.py` (the automatic rules for architectures not read by hand), `report.py` (inventory.json and this file), `fixplan.md`, `findings.md`.
- Re-run: `pip download vllm==0.30.0 --no-deps --no-binary :all:` and unpack it (`VLLM_SRC` names the folder), `python3 fetch.py repos_all.txt` (headers into headers/, about 2 s a shard; Hugging Face rate-limits bursts), `python3 screen.py LIST screen.json && python3 more.py` for the wider list, then `python3 report.py && README=1 python3 report.py`.
