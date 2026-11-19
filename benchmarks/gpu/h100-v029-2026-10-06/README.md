# v0.29.0 on an H100: the 2026 bf16 models of 20-75 GB (2026-10-06, stopped early)

One Runpod H100 80GB HBM3 pod (driver 580.126.09, Ubuntu 24.04, 224 CPUs, 400 GB disk), started 05:02 UTC with `jobB.sh`: Glyd as
released (`glyd 0.29.0`, `glyd-gpu 0.29.0` from PyPI, vLLM 0.30.0, torch 2.13.0, transformers 5.18.0, by tag v0.29.0's `install.sh`),
then `../rtx4090-v029-2026-10-06/sweep.sh` on `listB.txt`.

**The pod was deleted from outside this run at about 05:17 UTC**, 15 minutes in, while Muse-Glimmer-30B was loading (the session-wide
Runpod budget files were changed then and refuse this work's pods: "bf16 GPU work is paused by the owner"). 4 of the 38 rows were
measured. Spend on this pod: about $0.94 (3.49 $/h).

Method (as `../rtx4090-v029-2026-10-06/README.md` section 2): `GLYD_KV=off vllm serve MODEL --quantization glyd` and plain
`vllm serve MODEL`, both `--enforce-eager --max-model-len 8192 --max-num-seqs 32 --dtype bfloat16`, one after the other, one greedy chat
answer each. On this GPU the plugin chooses the 12-bit layout (`glyd: mma12 layout`). Rows: `logs/sweep.tsv`; logs: `logs/sweep/<model>/`.

| Model | Glyd weights GiB | bf16 weights GiB | Saved | Glyd KV tokens | bf16 KV tokens | Glyd answer | bf16 answer | Note |
| :-- | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| Qwen/Qwen3.8-27B | 39.48 | 51.10 | 22.7% | 312,259 | 193,234 | Paris | Paris | 12-bit layout |
| Qwen/Qwen3.6-35B-A3B | 53.20 | 65.53 | 18.8% | 504,978 | 147,456 | Paris | Paris | 12-bit layout; the inventory projected 24.3% for this model in this layout: 5.5 points not yet explained |
| google/gemma-4-31B-it | | 58.99 | | | 14,339 | | Paris | Glyd: the plugin stops on Gemma 4's per-layer config (transformers' `AmbiguousGlobalPerLayerAttributeError`: `num_key_value_heads`, in `vllm_plugin._linear_bytes`) |
| google/gemma-4-26B-A4B-it | | 48.54 | | | 107,660 | | Paris | Glyd: the same Gemma 4 error |
| meta-models/Muse-Glimmer-30B | | | | | | | | interrupted (the pod was deleted while Glyd's server was loading) |

Not run: the other 33 rows of `listB.txt` (Nemotron-3.5-Lightning-30B-A3B, EXAONE-4.5-33B, KAT-Coder-V2.5-Dev, granite-4.2-30b,
GLM-4.7-Flash and the rest; 5 configs-only rows; the same-shape twins Qwen3.5-27B, Qwen3.6-27B, Qwen3.5-35B-A3B, Laguna-XS.2,
llm-jp-4-32b-a3b and llm-jp-4-33b last). Two gated rows are not in the list: pfnet/plamo-3-nict-2604-31b-base, Soofi-S-Instruct-Preview.

Launched with the session's `rp_run.sh`: `GPU="NVIDIA H100 80GB HBM3" MAX_MIN=150 RP_DISK=400 NAME=m29b UPLOAD="sweep.sh chat1.py listB.txt" bash rp_run.sh jobB.sh`.
