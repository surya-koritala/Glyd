# glyd run and glyd serve on 2026 model families: Runpod RTX 4090 (2026-10-06)

What was checked: for each model, `glyd run MODEL --prompt` and `glyd serve MODEL`, then four API requests to the served model:
- a tool call (`get_weather`, `tool_choice` auto), and the turn after the tool's answer;
- thinking shown apart: `chat_template_kwargs.enable_thinking` set, the thinking in `reasoning_content`, the answer in `content` with no thinking or tool markup;
- a long prompt of about 90% of the context that glyd chose (at most 200,000 tokens), to show that context fits.

The installed glyd_gpu is branch `run-2026-families`, built as a pure-Python wheel with the v0.29.0 release run's CUDA libraries. It was installed by v0.29.0's `install.sh` (glyd 0.29.0, vLLM 0.30.0, torch 2.13.0, transformers 5.18.0). Every option was left at its default.

Machines: Runpod secure cloud, one NVIDIA GeForce RTX 4090 (24,564 MiB) each, driver 595.91.07 (pod 1) and 580.159.04 (pods 2 to 6). Scripts: `job_pod1.sh`, `job_pod2_pod3.sh`, `job_pod4.sh`, `job_pod5.sh`, `job_pod6.sh`, `api_check.py`. Records: `pod1/`, `pod2-released-wheel/`, `pod3/`, `pod4/`, `pod5/`, `pod6/` (`job.log`, `summary.txt` where the job finished, `numbers.txt`, each step's output, vLLM's logs as `*.vllm.log`, `api_*.json`).

## Results

| Model (2026) | glyd run --prompt | glyd serve: tool call, turn after, thinking apart, long prompt | Context chosen | vLLM's KV cache | Weights (vLLM) | Record |
| :-- | :-- | :-- | --: | --: | --: | :-- |
| Qwen/Qwen3.5-9B | ok (vLLM refused the first context and glyd restarted it shorter: fixed since, see below) | ok, ok, ok, ok (183,800 prompt tokens) | 196,608 after the restart | 197,126 tokens | 12.69 GiB | pod1 |
| ibm-granite/granite-4.2-8b | ok | ok, ok, ok, ok (41,586) | 50,176 | 61,312 tokens | 11.18 GiB | pod1 |
| google/gemma-4-12B-it | ok, first start | ok, ok, ok, ok (136,479) | 146,432 | 159,380 tokens | 15.67 GiB | pod3 |
| openbmb/MiniCPM5-2B | answer ok; the thinking reached the answer (no reasoning parser) | ok, FAIL (`<think>` in the content), FAIL (same), ok (105,329) | 131,072 | 436,016 tokens | 3.31 GiB | pod1 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | did not start: the plugin's old count said 33.6 GiB | did not start | | | | pod1 |

The branch at its end (glyd_gpu built by release.yml as the compiled wheel, from private main with this branch and coverage-2026 merged, versioned `0.29.0+r291`). Pod 4 ran it before the last two commits, which change the served settings of models with Mamba2 layers alone; pod 6 ran the final branch (r291-validate 5a4be1f0):

| Model (2026) | glyd run --prompt | glyd serve: tool call, turn after, thinking apart, long prompt | Context chosen | vLLM's KV cache (serve) | Weights (vLLM) | Record |
| :-- | :-- | :-- | --: | --: | --: | :-- |
| Qwen/Qwen3.5-9B | ok, first start | ok, ok, ok, ok (171,238 prompt tokens) | 183,296 | 217,469 tokens | 12.01 GiB | pod4 |
| google/gemma-4-12B-it | ok, first start | ok, ok, ok, ok (116,217) | 124,928 | 147,881 tokens | 15.67 GiB | pod4 |
| openbmb/MiniCPM5-2B | ok, first start, no thinking in the answer | ok, ok, ok, ok (105,329) | 131,072 | 435,968 tokens | 3.31 GiB | pod4 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | ok, first start | ok, ok, ok, ok (198,899) | 262,144 | 653,027 tokens | 5.24 GiB | pod6 |

Pod 4 ran all four models: 27 of 28 checks passed. The one failure was glyd serve on Nemotron 3 Nano 4B, out of memory while vLLM captured CUDA graphs at 92% of the GPU. vLLM 0.30's Mamba2 scan asks for memory outside its profiled budget while capturing. Pod 5 tried 88% and failed the same way. Since then glyd serve gives a model with Mamba2 layers glyd run's memory share (71% here), which started every time; no other model's settings change (`test_onboard.test_estimates_2026`). Pod 6 ran Nemotron 3 Nano 4B on that final branch: 11 of 11 checks passed. Parsers as started: Gemma 4 gemma4 and gemma4, MiniCPM5 minicpm5 and qwen3, Nemotron 3 qwen3_coder and nemotron_v3, Qwen3.5 qwen3_coder and qwen3.

Parsers as started (from vLLM's command line in the logs):
- Qwen3.5: qwen3_coder and qwen3.
- Granite 4.2: qwen3_coder and nemotron_v3.
- Gemma 4: gemma4 and gemma4.
- MiniCPM5: minicpm5 only.

Pod 2 is not a run of the branch. glyd-gpu 0.29.0 had reached PyPI by then, and the installer took it instead of the branch's wheel (`pod2-released-wheel/which_glyd_gpu.txt`). That makes it, unplanned, a record of the **released v0.29.0**, which fails on:
- Gemma 4 12B: `AmbiguousGlobalPerLayerAttributeError` on `num_key_value_heads`, in glyd run and glyd serve alike;
- Nemotron 3 Nano 4B: "weights take about 33.6 GiB";
- MiniCPM5: no tool parser; the tool call returns HTTP 400.

Pod 3 used the branch's wheel, versioned `0.29.0+runfam` so that the installer prefers it. Gemma 4 12B passed every check, and the chosen context fit vLLM on the first try. The pod was deleted about 18 minutes in, during Nemotron's run, when bf16 GPU work was paused for the quantized sessions (`rp_budget/ALLOW`). Nemotron 3 Nano 4B, MiniCPM5-2B with its new parser, and Qwen3.5-9B's first-try start were therefore **not** measured with the final branch.

## What the runs changed in the branch

- **The plugin's count** (`vllm_plugin._linear_bytes`) is now glyd run's layer-by-layer count. It reads the config past transformers 5.18's per-layer guard, which v0.29.0 did not. This was the cause of both start failures.
  - Gemma 4: the heterogeneous config raised on `num_key_value_heads`.
  - Nemotron 3 Nano 4B: every layer was counted as attention plus a gated MLP, 33.6 GiB, and the model was refused.
- **MiniCPM5's reasoning parser** is now qwen3: its chat template opens the thinking in the prompt.
- **The context estimate**:
  - Hybrid models (linear attention or Mamba) add 0.65 GiB to the memory outside the KV cache. Qwen3.5-9B's peak activation was 2.34 GiB, against 0.83 for Granite 4.2 8B and 1.03 for Gemma 4 12B. vLLM also needed 6.32 GiB of KV cache for 203,776 tokens, about 1.6% over 32 KiB a token.
  - A sliding-window layer is sized at its window plus 5,120 tokens in flight. vLLM kept about 4,990 tokens beyond Gemma 4 12B's 1,024-token window.
  - With both, the estimate for these GPUs is 183,296 tokens for Qwen3.5-9B (vLLM held 197,126) and 124,928 for Gemma 4 12B (vLLM held 159,380). The tests pin both.
  - Not measured: a Mamba model's own figures (Nemotron's run was cut short).

## The estimates against the checkpoints

`estimate_vs_headers.txt` compares, for every 2026 checkpoint whose safetensors headers were read, the Linears and tables counted from its config with the bytes in its safetensors headers (by tensor name, read by HTTP range with no download; `estimate_vs_headers.py`).
- Nothing is counted over the checkpoint by more than 0.4%. Gemma 4 31B's 0.4% is the V matrix that vLLM makes from K.
- Configs the count does not read are counted under the checkpoint, which keeps the memory estimate on the safe side:
  - Falcon-H1-Tiny, K2-Horizon, HRM-Text, MiniCPM-o, granite-4.0-3b-vision;
  - LongCat-Flash-Lite and Nemotron-Labs Puzzle, which fall back to the measured mean.

## Spend

Three RTX 4090 pods at $0.74/h: 39 min, 9 min and 18.5 min, about **$0.82** in all (`rp_budget/ledger.tsv`: runfam26, runfam26b, runfam26c). Pods 4 to 6 (r291-val, r291-val2, r291-val3; they also ran coverage-2026's checks, in `job.log` after this check's summary): 43.6, 58.7 and 9.2 min, about $1.38.
