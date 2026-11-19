#!/usr/bin/env bash
# Qwen3.5-122B-A10B on 4 H100 80GB (tensor parallel 4): check_vllm.py --brief --tp 4 first, then bf16 and Glyd (auto).
MODEL=Qwen/Qwen3.5-122B-A10B TP=4 MODES="bf16 glyd" UTIL=0.9 CHECK=brief bash "$HOME/big_moe.sh"
