#!/bin/bash
# Once m3_prep.sh's Yi check is done: stop m3_prep.sh (session $1; its sanitizer runs come back after), then the M3 job's
# dry run on this L4 (Qwen3-0.6B: check and a short bench), detcost.sh, and m3_prep.sh again (the sanitizer runs).
until [ -f ~/vllm-work/m3prep/check-Yi-1.5-6B-Chat/report.txt ] || ! kill -0 $1 2> /dev/null; do sleep 10; done
sleep 2; pkill -TERM -s $1; sleep 5; pkill -KILL -s $1
echo "== $(date -u +%T) m3_prep.sh stopped after the Yi check"
mkdir -p ~/m3dry && cd ~/m3dry && rm -rf results w
echo "== $(date -u +%T) the M3 job's dry run"
flock ~/.glyd-box.lock env FILES=~/m3dry R=~/m3dry/results W=~/m3dry/w HF_HOME=~/hf UV_CACHE_DIR=~/.cache/uv VJ_MODEL=Qwen/Qwen3-0.6B VJ_STEPS=check,bench VJ_RATES="1 inf" VJ_PROMPTS="16 32" bash ~/m3dry/vllm_job.sh > ~/m3dry/job.out 2>&1
echo "exit $?"; cat ~/m3dry/results/steps.txt
echo "== $(date -u +%T) detcost"
bash ~/vllm-work/detcost.sh
echo "== $(date -u +%T) m3_prep.sh again"
bash ~/vllm-work/m3_prep.sh
