#!/usr/bin/env bash
# e2e12.py on the release candidate (r1_src.tar, 85ae892) against main (sb_main.tar, db8e7b0) on an L4 (an AWS
# g6.4xlarge): r1_job.sh's step (g) alone, Qwen3-1.7B, Qwen3-4B-Instruct-2507 and granite-3.1-3b-a800m-instruct from
# the machine's cache (~/hf, offline). ~/.glyd-busy holds the machine's idle stop off; the lock, one GPU job at a time.
touch ~/.glyd-busy
trap "rm -f ~/.glyd-busy" EXIT
export FILES=$HOME/sb-rc R=$HOME/sb-rc/results-e2e W=$HOME/sb-rc/w ONLY=e2e ATTR=0 HF_HUB_OFFLINE=1
export MODELS="Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507 ibm-granite/granite-3.1-3b-a800m-instruct"
export E2E_MODELS="$MODELS"
flock $HOME/.glyd-box.lock bash $HOME/sb-rc/r1_ada.sh
