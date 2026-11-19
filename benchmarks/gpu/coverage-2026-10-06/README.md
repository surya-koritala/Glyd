# coverage-2026 on a Runpod RTX 4090 (2026-10-06)

What was checked: the vLLM plugin with coverage-2026 (the tables and Linears vLLM builds without the quant config packed after the load, rows not a multiple of 64, Jamba's Mamba mixers) on 2026 models, against bf16 and against glyd-gpu 0.29.0.

glyd_gpu: release.yml's compiled x86_64 wheel of branch r291-validate (private main + run-2026-families + this branch; commits in each pod's `COMMIT`), versioned `0.29.0+r291` and installed by v0.29.0's `install.sh` (glyd 0.29.0, vLLM 0.30.0, torch 2.13.0, transformers 5.18.0). Machines: Runpod secure cloud, one NVIDIA GeForce RTX 4090 (24,564 MiB), driver 580.159.04. Scripts: `job_pod4.sh`, `job_pod5.sh`, `job_pod7.sh` (pods 4 and 5 also ran run-2026-families' check: `benchmarks/gpu/run-families-2026-10-06`; pod 6 ran that check alone). Records: `pod4/`, `pod5/`, `pod7/`.

## Results

`check_vllm.py --brief` (Glyd as it ships, against bf16 from the same run; `pod5/check_brief_report.txt`, `pod7/check_9b_report.txt`):

| Model (2026) | Every pack decoded bit for bit | Tables packed (expected) | Greedy top-1 agreement with bf16 (bf16 eager's) | Exact mode: bf16 eager's tokens and logprobs bit for bit | Weights, Glyd / bf16 | Record |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Qwen/Qwen3.5-0.8B | PASS (134) | 1 (1) | 0.9968 (0.9968) | PASS (8 of 8) | 1.23 / 1.72 GiB | pod5 |
| LiquidAI/LFM2.5-350M | PASS (64) | 1 (1) | 0.9916 (0.9916) | PASS | 0.48 / 0.69 GiB | pod5 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | PASS (84) | 2 (2) | 0.9948 (0.9922) | PASS | 5.40 / 7.47 GiB | pod5 |
| ai21labs/AI21-Jamba2-3B | PASS (112) | 1 (1) | 0.9942 (0.9935) | PASS | 4.71 / 5.68 GiB | pod5 |
| Qwen/Qwen3.5-9B (language model) | PASS (128) | 2 (2) | 0.9968 (0.9968) | PASS | 11.38 / 16.80 GiB | pod7 |

- Every other check passed, except one: Qwen3.5-0.8B's layer products against F.linear, 1.02e-02 against a bound of 1e-2. glyd-gpu 0.29.0 gives the same 1.02e-02 on the same layer (an MLP down projection, 1,024 x 3,584, at 513 tokens; `pod7/l08_r291_worst.txt`, `pod7/l08_v029_worst.txt`), so it is not a change of this branch.
- Qwen3.5-9B: bf16 does not fit a 24 GB card with vLLM's defaults (pod 5: out of memory capturing CUDA graphs). Pod 7 ran every run, bf16 and Glyd alike, with `CHECK_VLLM_LLM_KW={"language_model_only": true, "max_num_seqs": 16}`. The vision tower is covered by Qwen3.5-0.8B's run.
- The plugin's tests (`test_vllm.py`, `GLYD_REQUIRE_VLLM=1`): all passed, the installed wheel and the sources, with the GPU parts of test_moe_activations, test_padded_rows, test_after_load, test_embedding_ops and test_scratch_rule run (pods 4 and 5).

Qwen3.5-9B under `vllm serve --quantization glyd --max-model-len 4096`, `vllm bench serve` random 1,024 in / 256 out (`pod5/speed-*/u*.json`):

| glyd-gpu | Users | Output tok/s | TTFT median ms | TPOT median ms | Weights (vLLM) |
| :-- | --: | --: | --: | --: | --: |
| this branch | 1 | 74.1 | 167.8 | 12.90 | 12.01 GiB |
| 0.29.0 | 1 | 73.9 | 171.7 | 12.90 | 12.69 GiB |
| this branch, again | 1 | 73.9 | 167.6 | 12.91 | 12.01 GiB |
| this branch | 32 | 954.9 | 940.5 | 29.76 | 12.01 GiB |
| 0.29.0 | 32 | 954.9 | 941.4 | 29.72 | 12.69 GiB |

Not run: the 8 to 256 user loads (pod 5 reached its time cap) and the towers check. `check_vllm.py --towers ibm-granite/granite-vision-4.1-4b` could not start bf16 either: vLLM 0.30.0's llava/pixtral import fails with transformers 5.18 (`pod7/towers/*.log`). The towers stay off by default.

Pod 4's `check_vllm` stopped on FlashInfer's sampler compiling without ninja; pod 5 ran with `VLLM_USE_FLASHINFER_SAMPLER=0`, as glyd run and glyd serve start vLLM.

## Spend

Pods 4, 5 and 7 (r291-val, r291-val2, r291-val4) at $0.74/h: 43.6, 58.7 and 43.2 min, shared with run-2026-families' check (pod 6, 9.2 min, ran that alone); all four about $1.91.
