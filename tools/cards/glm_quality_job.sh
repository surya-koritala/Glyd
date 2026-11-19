#!/usr/bin/env bash
# Output quality of GLM-5.3-Flash with its routed experts rounded to 2-4 bits, on a 4-GPU pod (rp_run.sh contract; launched by tools/cards/glm_quality.sh, Linear GLY-127).
# Uploaded beside it: quality.env (optionally VARIANTS CAL_SEQS HELD_SEQS COUNT_SEQS GPTQ_SEQS GPTQ_MIN_TOK CHUNK MEM_FRAC JOB_MAX_MIN FLA SELFTEST_GPU), research/lossy-2026/glm_quality.py and glm_skew.py.
# ~/results: steps.txt (wall clock per step), quality.md + quality.json + results.jsonl (rewritten after every variant), counts/ (routing counts, glm_skew.py's format), perpos/ (KL of every position),
# meta.json, versions.txt, logs. transformers 5.19.0 as in the skew job: GLM stays FP8 on the GPUs (128x128 block scales, Triton kernels) for the whole job; the experts are rounded layer by layer from it.
set -u
R=~/results; mkdir -p $R ~/v; cd ~
set -a; . ~/quality.env; set +a   # (exported: the pyenv step's bash -c reads FLA, the run reads MEM_FRAC)
T0=$(date +%s); P=$R/steps.txt; : > $P
DEADLINE=$((T0 + ${JOB_MAX_MIN:-128} * 60))   # the launcher's cap (150 min) also covers the pod's start: no variant is started that would not end 2 min before this
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
collect() { touch $R/DONE; }
trap collect EXIT
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=$HOME/v/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/v/venv/bin/python
eval "$(python3 ~/glm_quality.py plan)"   # REPO REV POD_CLASS DISK K NE N_MOE ... ORDER

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > $R/gpus.txt 2>&1; df -h ~ > $R/disk-start.txt 2>&1
# the hub client first, so the weights (the pinned revision, 328 GB) download while torch and the rest install, the checks run and the token sequences are made
step venv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet' || exit 1
dl() { for i in 1 2 3; do timeout 1800 ~/v/venv/bin/hf download "$REPO" --revision "$REV" --local-dir ~/v/m && return 0; done; return 1; }   # (a stalled attempt is cut after 30 min and resumes)
step download dl &
DL=$!
# the CUDA build of torch: wheels for 12.8 run on any driver >= 570 (Hopper, Ampere and the Blackwell SM120 cards); PEP 668: a venv
# the versions glm_skew.py's run on four RTX PRO 6000 had (torch 2.11.0+cu128, triton 3.6.0, accelerate 1.15.0, fla 0.5.2): the hub's FP8 kernel is built per torch and CUDA
# kernels: transformers 5.19.0 accepts only 0.17.x (GLM's FP8 Triton kernel comes from the hub through it)
step pyenv bash -c '~/v/venv/bin/pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 && ~/v/venv/bin/pip install -q "transformers==5.19.0" "accelerate==1.15.0" "safetensors>=0.8" pyarrow numpy "kernels>=0.17,<0.18"' || exit 1
# a tiny random Glm5Next on the CPU through every variant, the block codes against calib.py's: a failure here means the code is wrong, so stop before the weights are loaded (the pod is deleted).
# Before flash-linear-attention: once it is installed the model binds its Triton KDA kernels, which cannot take the CPU tensors of this test
step selftest $PY ~/glm_quality.py selftest || exit 1
# a slow host stops here, not at the cap (2026-10-10: one pod fetched 9 of 73 files in 30 min): the weights must be on disk 45 min after the start at the rate so far
t=$(( $(date +%s) - T0 )); b=$(du -sbc ~/v/m ~/v/hf 2> /dev/null | tail -1 | cut -f1)
[ $(( ${b:-0} / (t + 1) * 2700 )) -ge $(( ${WEIGHTS_GB%.*} * 1000000000 )) ] || { echo "download: $(( ${b:-0} / 1000000 / (t + 1) )) MB/s after $(( t / 60 )) min, too slow for the weights in 45 min: stopped" >> $P; exit 1; }
# flash-linear-attention = GLM's KDA kernels (FLA=0: the torch reference, much slower); the pins keep pip from touching transformers
[ "${FLA:-1}" = 1 ] && { step fla ~/v/venv/bin/pip install -q "torch==2.11.0" "transformers==5.19.0" "kernels>=0.17,<0.18" "flash-linear-attention==0.5.2" || exit 1; }
$PY -m pip freeze 2> /dev/null | grep -i -E '^(torch|transformers|triton|kernels|accelerate|safetensors|huggingface|hf.xet|flash.linear|fla.core|numpy|pyarrow)' > $R/versions.txt
# the same held-out text and calibration sequences as the skew job (pinned datasets)
step prep $PY ~/glm_skew.py prep glm ~/v/data --cal-seqs "${CAL_SEQS:-24}" --held-seqs "${HELD_SEQS:-4}" || exit 1
wait $DL || { echo "download: failed" >> $P; exit 1; }
cp ~/v/data/prep.json $R/ 2> /dev/null
df -h ~ > $R/disk-after-download.txt 2>&1
step run $PY ~/glm_quality.py run --weights ~/v/m --data ~/v/data --out $R --deadline $DEADLINE --held-seqs "${HELD_SEQS:-4}" --count-seqs "${COUNT_SEQS:-12}" --gptq-seqs "${GPTQ_SEQS:-24}" \
  --gptq-min-tok "${GPTQ_MIN_TOK:-2048}" --chunk "${CHUNK:-0}" ${VARIANTS:+--variants "$VARIANTS"}
step report $PY ~/glm_quality.py report $R
nvidia-smi --query-gpu=index,memory.used --format=csv >> $R/gpus.txt 2>&1
