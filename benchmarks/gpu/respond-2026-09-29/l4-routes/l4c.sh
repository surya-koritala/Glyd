#!/usr/bin/env bash
# Glyd's 12-bit layout on the L4 (glyd12), after the L4 table (l4b): Qwen3-8B and Qwen3-4B-Instruct-2507, full repeats
[ -e ~/.glyd-busy ] || { touch ~/.glyd-busy; trap "rm -f ~/.glyd-busy" EXIT; }
export FILES=$HOME/respond2/files12 R=$HOME/respond2/results-l4c W=$HOME/respond2/w12 HF_HUB_OFFLINE=1 RESP_BUDGET=10800 RESP_END=11400
export RESP_RUNS="Qwen/Qwen3-8B:glyd12 Qwen/Qwen3-4B-Instruct-2507:glyd12"
until [ -f ~/respond2/results-l4b/DONE ]; do sleep 20; done
flock $HOME/.glyd-box.lock bash $HOME/respond2/files12/resp_job.sh
