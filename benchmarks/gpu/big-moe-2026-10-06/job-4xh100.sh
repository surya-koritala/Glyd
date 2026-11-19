#!/usr/bin/env bash
# One 4 x H100 80GB pod (tensor parallel 4), one model at a time on the disk: Qwen3.5-122B-A10B (check_vllm --brief first, then bf16
# and Glyd), then NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (bf16 and Glyd, then check_vllm --brief if the pod's time is left).
MODEL=Qwen/Qwen3.5-122B-A10B TP=4 MODES="bf16 glyd" CHECK=brief CHECK_TIMEOUT=2400 R=$HOME/results/qwen35-122b W=$HOME/w1 bash "$HOME/big_moe.sh"
rm -rf "$HOME/hf/hub/models--Qwen--Qwen3.5-122B-A10B" "$HOME/w1/cache-"*
MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 TP=4 MODES="bf16 glyd" CHECK=last LMO= R=$HOME/results/nemotron3-super W=$HOME/w2 bash "$HOME/big_moe.sh"
touch "$HOME/results/DONE"
