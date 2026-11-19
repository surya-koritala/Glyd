# v0.29.0 on an RTX 5090: the layout question (not run)

Prepared on 2026-10-06 and not run: the session-wide Runpod budget files refuse this work's pods ("bf16 GPU work is paused by the
owner"), after the RTX 4090 and H100 pods of the same work were deleted (`../rtx4090-v029-2026-10-06/`, `../h100-v029-2026-10-06/`).

What it measures, for whether `best_layout` should pick the tiered layout on Blackwell (the bar: faster than vLLM bf16 at every load
and less memory): Qwen3.5-9B and gemma-4-12B-it, each served three ways, one after the other, by Glyd as released (v0.29.0's install.sh):
vLLM bf16 (`vllm serve MODEL`), Glyd's default (`glyd serve MODEL`, the 12-bit layout on this GPU) and Glyd tiered
(`GLYD_LAYOUT=mma glyd serve MODEL`); for each, the weights ("Model loading took"), the KV cache, and `vllm bench serve` with random
prompts of 1,000 tokens in and 256 out at 1, 8, 32 and 64 users (requests/s, output tokens/s, first-token and per-token p50/p99).

    GPU="NVIDIA GeForce RTX 5090" MAX_MIN=120 NAME=m29c UPLOAD="bench5090.sh summary3.py" bash rp_run.sh jobC.sh

`bench5090.sh` is the speed bar's job (`speed_bar_job.sh` of 2026-10-06) with the third server and these four loads; `summary3.py` its
summary with three columns. Expect gemma-4-12B-it's Glyd servers to fail on v0.29.0 as they did on the 4090 and the H100 (the plugin
stops on Gemma 4's per-layer config) until that is fixed.
