#!/usr/bin/env bash
# The L4 run: Qwen3-8B and Qwen3-4B-Instruct-2507 from ~/hf, bf16, Glyd and exact, the full measurement set.
touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT
mkdir -p $HOME/respond/l4files && cp $HOME/respond/resp_src_l4.tar $HOME/respond/l4files/resp_src.tar && cp $HOME/respond/resp_job.sh $HOME/respond/l4files/
export FILES=$HOME/respond/l4files R=$HOME/respond/results-l4 W=$HOME/respond/w HF_HUB_OFFLINE=1 RESP_BUDGET=10800 RESP_END=11400
export RESP_RUNS="Qwen/Qwen3-8B:bf16 Qwen/Qwen3-8B:glyd Qwen/Qwen3-8B:exact Qwen/Qwen3-4B-Instruct-2507:bf16 Qwen/Qwen3-4B-Instruct-2507:glyd Qwen/Qwen3-4B-Instruct-2507:exact"
flock $HOME/.glyd-box.lock bash $HOME/respond/l4files/resp_job.sh
