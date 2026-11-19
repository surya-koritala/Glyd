#!/usr/bin/env bash
# Expert-routing skew of one MoE model on a 4-GPU pod (rp_run.sh contract; launched by tools/cards/glm_skew.sh, Linear GLY-127).
# Uploaded beside it: skew.env (MODEL=glm|nemotron, optionally CAL_SEQS HELD_SEQS BATCH GEN_NEW JOB_MAX_MIN FLA) and research/lossy-2026/glm_skew.py.
# ~/results: steps.txt (wall clock per step), counts/ (per layer and domain), decode.npz + decode.json, timing.json, skew.json + skew.md, routing/ (raw ids, moved in at the end), versions.txt, logs.
# transformers 5.19.0, not vLLM (see glm_skew.py and the launcher): GLM stays FP8 (128x128 block scales, Triton kernels), Nemotron is bf16; layers are split over the GPUs (device_map auto).
set -u
R=~/results; mkdir -p $R ~/v; cd ~
set -a; . ~/skew.env; set +a   # (exported: the pyenv step's bash -c reads FLA)
T0=$(date +%s); P=$R/steps.txt; : > $P
DEADLINE=$((T0 + ${JOB_MAX_MIN:-120} * 60))   # the launcher's cap (150 min) also covers the pod's start: the trace stops new work 8-10 min before this
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
collect() { [ -d ~/raw ] && mv ~/raw $R/routing; touch $R/DONE; }
trap collect EXIT
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=$HOME/v/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/v/venv/bin/python
eval "$(python3 ~/glm_skew.py plan "$MODEL")"   # REPO REV K NE N_MOE ...

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > $R/gpus.txt 2>&1; df -h ~ > $R/disk-start.txt 2>&1
# the CUDA build of torch: wheels for 12.8 run on any driver >= 570 (Hopper, Ampere and the Blackwell SM120 cards); PEP 668: a venv
# kernels: transformers 5.19.0 accepts only 0.17.x (GLM's FP8 Triton kernel comes from the hub through it); flash-linear-attention = GLM's KDA kernels (FLA=0: torch reference, slower)
step pyenv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && ~/v/venv/bin/pip install -q "transformers==5.19.0" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub>=1.5,<2" hf_xet pyarrow numpy "kernels>=0.17,<0.18" $([ "${FLA:-1}" = 1 ] && echo flash-linear-attention)'
$PY -m pip freeze 2> /dev/null | grep -i -E '^(torch|transformers|triton|kernels|accelerate|safetensors|huggingface|hf.xet|flash.linear|numpy|pyarrow)' > $R/versions.txt
step selftest $PY ~/glm_skew.py selftest
# the weights (the pinned revision) and the token sequences (pinned datasets) at the same time
dl() { for i in 1 2 3; do ~/v/venv/bin/hf download "$REPO" --revision "$REV" --local-dir ~/v/m && return 0; done; return 1; }
step download dl &
step prep $PY ~/glm_skew.py prep "$MODEL" ~/v/data --cal-seqs "${CAL_SEQS:-24}" --held-seqs "${HELD_SEQS:-4}" &
wait
cp ~/v/data/prep.json $R/ 2> /dev/null
df -h ~ > $R/disk-after-download.txt 2>&1
step trace $PY ~/glm_skew.py trace "$MODEL" --weights ~/v/m --data ~/v/data --out $R --raw ~/raw --batch "${BATCH:-2}" --gen-new "${GEN_NEW:-128}" --deadline $DEADLINE
step analyze $PY ~/glm_skew.py analyze "$MODEL" $R
nvidia-smi --query-gpu=index,memory.used --format=csv >> $R/gpus.txt 2>&1
