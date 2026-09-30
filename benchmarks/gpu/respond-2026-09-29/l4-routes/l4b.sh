#!/usr/bin/env bash
# The L4 table again: bf16 eager, bf16 compiled (bf16c) and Glyd default on the l4-routes library, Qwen3-8B and
# Qwen3-4B-Instruct-2507, full repeats; run after run3 (the lock taken in turn).
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
export FILES=$HOME/respond2/files R=$HOME/respond2/results-l4b W=$HOME/respond2/w HF_HUB_OFFLINE=1 RESP_BUDGET=10800 RESP_END=11400
export RESP_RUNS="Qwen/Qwen3-8B:bf16 Qwen/Qwen3-8B:bf16c Qwen/Qwen3-8B:glyd Qwen/Qwen3-4B-Instruct-2507:bf16 Qwen/Qwen3-4B-Instruct-2507:bf16c Qwen/Qwen3-4B-Instruct-2507:glyd"
until [ -f ~/l4routes/results-new/DONE ]; do sleep 20; done
flock $HOME/.glyd-box.lock bash $HOME/respond2/files/resp_job.sh
