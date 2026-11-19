#!/usr/bin/env bash
# The KL of one level against bf16 (rp_run.sh contract; one A100 pod). Uploaded beside it: kl.env (MODEL, REV, VARIANT) and research/lossy-2026's kld_study.py,
# prepare_quality.py, matched_quality.py, quality_env.sh, quality-requirements.txt. Writes kl.jsonl (kld_study.py's results.jsonl).
set -u
R=~/results; mkdir -p $R ~/v; cd ~
. ~/kl.env
P=$R/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
trap 'touch $R/DONE' EXIT
export HF_XET_HIGH_PERFORMANCE=1
step pyenv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && ~/v/venv/bin/pip install -q "transformers==5.18.0" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub>=1.5,<2" hf_xet pyarrow flash-linear-attention'  # (as benchmarks/gpu/int5-a100-2026-10-09/kl.sh)
step selftest ~/v/venv/bin/python ~/kld_study.py selftest
step download ~/v/venv/bin/hf download "$MODEL" --revision "$REV" --local-dir ~/v/m
step ref ~/v/venv/bin/python ~/kld_study.py ref ~/v/m ~/v/kl
step score ~/v/venv/bin/python ~/kld_study.py score ~/v/m ~/v/kl "$VARIANT"
cp ~/v/kl/results.jsonl $R/kl.jsonl
