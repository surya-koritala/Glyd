#!/usr/bin/env bash
# Calibrated 4-bit rounding against plain rounding (GLY-37): the KL, top-1 and perplexity of Qwen/Qwen3.5-9B at int:4:32:mse:bf16 (baseline), gptq:4:32:bf16, gptq:4:32:bf16:act
# and awq:4:32:bf16, with the wall-clock of each (rp_run.sh contract; one A100 pod; no tokens: model and datasets are ungated). Uploaded beside it: research/lossy-2026's
# kld_study.py, prepare_quality.py, matched_quality.py, calib.py. Writes calib.jsonl (kld_study.py's results.jsonl), calib.json (the calibration data's ids, revisions, counts),
# ref.json, steps.txt (seconds each step took, one score step a method: GPU-seconds of a pod running one job at a time) and DONE.
#   NAME=calib-9b MAX_MIN=180 UPLOAD="research/lossy-2026/kld_study.py research/lossy-2026/prepare_quality.py research/lossy-2026/matched_quality.py research/lossy-2026/calib.py" bash tools/runpod/rp_class.sh a100 tools/cards/calib_job.sh
# CALIB_SEQS=64 (set in the pod's environment, or a calib.env uploaded) is a quick run on fewer sequences; the default is 512 of 2,048 tokens.
set -u
R=~/results; mkdir -p $R ~/v; cd ~
[ -f ~/calib.env ] && . ~/calib.env
MODEL=${MODEL:-Qwen/Qwen3.5-9B}; REV=${REV:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}; N=${CALIB_SEQS:-512}
P=$R/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
trap 'touch $R/DONE' EXIT
export HF_XET_HIGH_PERFORMANCE=1
PY=~/v/venv/bin/python
step pyenv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && ~/v/venv/bin/pip install -q "transformers==5.18.0" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub>=1.5,<2" hf_xet pyarrow flash-linear-attention'  # (as kl_job.sh)
step selftest $PY ~/kld_study.py selftest
step calib_selftest $PY ~/calib.py selftest   # (a CPU check: informational here, the fla kernels may refuse CPU tensors)
step download ~/v/venv/bin/hf download "$MODEL" --revision "$REV" --local-dir ~/v/m
step ref $PY ~/kld_study.py ref ~/v/m ~/v/kl
step calib_build $PY ~/calib.py build ~/v/m ~/v/kl --seqs "$N"
for v in ${VARIANTS:-int:4:32:mse:bf16 gptq:4:32:bf16 gptq:4:32:bf16:act awq:4:32:bf16}; do   # (VARIANTS: any kld_study variants, e.g. the winner at 3 and 5 bits)
  case $v in int:*) c="";; *) c="--calib-seqs $N";; esac
  step "score_$(echo $v | tr ':' '_')" $PY ~/kld_study.py score ~/v/m ~/v/kl $v $c
done
cp ~/v/kl/results.jsonl $R/calib.jsonl; cp ~/v/kl/calib.json ~/v/kl/ref.json $R/
awk -F'[ :]+' '/^score_/ {s+=$(NF-1); printf "%s %.3f GPU-hours\n", $1, $(NF-1)/3600} END {printf "score steps total %.3f GPU-hours\n", s/3600}' $P > $R/gpu_hours.txt
