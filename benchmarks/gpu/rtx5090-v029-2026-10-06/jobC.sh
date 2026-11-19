#!/usr/bin/env bash
# Pod C (one RTX 5090): bench5090.sh on Qwen3.5-9B and gemma-4-12B-it: vLLM bf16, Glyd's default (12-bit here), Glyd tiered (GLYD_LAYOUT=mma);
# vllm bench serve 1,000 tokens in / 256 out at 1, 8, 32 and 64 users. Uploaded beside it: bench5090.sh, summary3.py.
MODELS="Qwen/Qwen3.5-9B google/gemma-4-12B-it" MAX_MIN=105 bash "$HOME/bench5090.sh"
