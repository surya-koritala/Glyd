#!/usr/bin/env bash
# v0.26.0 validation on Hopper: option 2 on Qwen3-14B (o3_job.sh), the vLLM plugin's checks, the MoE route per layer
# and served (Qwen3-30B-A3B), then the bf16 repeatability diagnostic (diag_job.sh). Each job's ~/results goes to
# ~/out/NAME; ~/results gets them all at the end, then ALLDONE.
mkdir -p ~/out
run() { local name=$1 lim=$2; shift 2; rm -rf ~/results; mkdir -p ~/results; echo "$(date -u +%T) start $name" >> ~/out/steps.txt
  env "$@" R=$HOME/results timeout $lim bash ~/$name.sh > ~/out/$name.job.log 2>&1; echo "$(date -u +%T) $name exit $?" >> ~/out/steps.txt
  rm -rf ~/out/$name; mv ~/results ~/out/$name; }
nvidia-smi --query-gpu=name --format=csv,noheader > ~/out/gpu.txt 2>&1
for n in vllm_chk vllm_moer vllm_moeb; do cp ~/vllm_job.sh ~/$n.sh; done
run o3_job 1560 MODELS=Qwen/Qwen3-14B
run vllm_chk 1800 VJ_STEPS=check
run vllm_moer 1800 VJ_STEPS=moeroutes VJ_MOE=Qwen/Qwen3-30B-A3B
run vllm_moeb 1800 VJ_STEPS=moebench VJ_MOE=Qwen/Qwen3-30B-A3B
run diag_job 960
run kv_job 1260
run budget_job 1800
rm -rf ~/results; mkdir -p ~/results; cp -r ~/out/* ~/results/; touch ~/results/DONE ~/results/ALLDONE
