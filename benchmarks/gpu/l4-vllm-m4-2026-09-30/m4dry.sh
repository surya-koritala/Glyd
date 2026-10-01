#!/bin/bash
# M4's job on the L4 as a dry run, under the box's lock: VJ_TP=2 on one GPU (it must stop at once), then moebench on
# granite (2 rates, short) and profile on Qwen3-1.7B (a few M), each a job of its own, the uv cache and models the box's.
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
cd ~/m4dry
common="FILES=$HOME/m4dry W=$HOME/m4dry/w HF_HOME=$HOME/hf UV_CACHE_DIR=$HOME/.cache/uv"
echo "== $(date -u +%T) VJ_TP=2 on one GPU"
env $common R=$HOME/m4dry/results-tp VJ_TP=2 VJ_STEPS=check bash vllm_job.sh > job-tp.out 2>&1; echo "exit $?"; cat results-tp/steps.txt
echo "== $(date -u +%T) moebench and profile"
env $common R=$HOME/m4dry/results VJ_STEPS=moebench,profile VJ_MOE=ibm-granite/granite-3.1-3b-a800m-instruct VJ_MODEL=Qwen/Qwen3-1.7B VJ_RATES="1 inf" VJ_PROMPTS="16 32" BATCHES="1 8 32" PROMPTS="512 2048" bash vllm_job.sh > job.out 2>&1; echo "exit $?"
cat results/summary.txt
