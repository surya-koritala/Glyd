#!/usr/bin/env bash
# Output quality of Nemotron 3 Super with its routed experts rounded to 2-4 bits, on a 4-GPU pod (rp_run.sh contract; launched by tools/cards/nemotron_quality.sh, Linear GLY-131).
# Uploaded beside it: quality.env (optionally VARIANTS CAL_SEQS HELD_SEQS COUNT_SEQS GPTQ_SEQS GPTQ_MIN_TOK CHUNK MEM_FRAC JOB_MAX_MIN SELFTEST_GPU OMP_NUM_THREADS EXPORT), research/lossy-2026/nemotron_quality.py,
# nemotron_hbr.py, glm_quality.py and glm_skew.py.
# EXPORT=1 (hbr40 in VARIANTS): hbr40's codes are written to ~/results/export, one file a MoE layer as its pass goes (~42 GB, nemotron_hbr.py's format). rp_run.sh does not put that directory in its minute's tar: it copies
# each finished file once as it appears (tools/runpod/pull_export.sh), so by the end of the pass most of the export is already on the launching machine, under $GLYD_RUNS/results-NAME/export.
# ~/results: steps.txt (wall clock per step), quality.md + quality.json + results.jsonl (rewritten after every variant), counts/ (routing counts, glm_skew.py's format), perpos/ (KL of every position),
# meta.json, versions.txt, logs. transformers 5.19.0 as in the skew job: the model stays bf16 on the GPUs for the whole job (no mamba-ssm or causal-conv1d: the torch Mamba-2, as the skew job ran it), the experts are rounded layer by layer from it.
set -u
R=~/results; mkdir -p $R ~/v; cd ~
set -a; . ~/quality.env; set +a   # (exported: the run reads MEM_FRAC and SELFTEST_GPU)
T0=$(date +%s); P=$R/steps.txt; : > $P
DEADLINE=$((T0 + ${JOB_MAX_MIN:-128} * 60))   # the launcher's cap (150 min) also covers the pod's start: no variant is started that would not end 2 min before this
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
collect() { touch $R/DONE; }
trap collect EXIT
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=$HOME/v/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the CPUs this container really has: torch sizes its CPU thread pools by the host's cores, but a container is often given a quota of a few (cgroup cpu.max), and that many threads on that quota
# are throttled to a crawl. Seen 2026-10-10: the GLM quality job's selftest on a pod was still running after 8 min (1 min on a laptop; a forward of its tiny model took 11 s); the cause is a
# guess, the threads or the download taking every CPU, and both are dealt with here: the thread count, and the selftest after the download
cpus() { local q p c=${CGROOT:-/sys/fs/cgroup}; { read -r q p < $c/cpu.max; } 2> /dev/null && [ "$q" != max ] && { echo $(( (q + p - 1) / p )); return; }
         { read -r q < $c/cpu/cpu.cfs_quota_us; read -r p < $c/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:--1}" -gt 0 ] && { echo $(( (q + p - 1) / p )); return; }; nproc; }
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(cpus)}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
PY=~/v/venv/bin/python
eval "$(python3 ~/nemotron_quality.py plan)"   # REPO REV POD_CLASS DISK K NE N_MOE ... ORDER SETUP_MIN PASSES_MIN

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > $R/gpus.txt 2>&1; df -h ~ > $R/disk-start.txt 2>&1; echo "CPUs for torch: $OMP_NUM_THREADS (nproc $(nproc))" > $R/cpus.txt
# the hub client first, so the weights (the pinned revision, 247 GB) download while torch and the rest install and the token sequences are made
step venv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet' || exit 1
dl() { for i in 1 2 3; do timeout 1800 ~/v/venv/bin/hf download "$REPO" --revision "$REV" --local-dir ~/v/m && return 0; done; return 1; }   # (a stalled attempt is cut after 30 min and resumes)
step download dl &
DL=$!
# the CUDA build of torch: wheels for 12.8 run on any driver >= 570 (Hopper, Ampere and the Blackwell SM120 cards); PEP 668: a venv
# the versions glm_skew.py's run on four A100 had (torch 2.11.0+cu128, accelerate 1.15.0); it ran with kernels and flash-linear-attention installed, which this model does not use
step pyenv bash -c '~/v/venv/bin/pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 && ~/v/venv/bin/pip install -q "transformers==5.19.0" "accelerate==1.15.0" "safetensors>=0.8" pyarrow numpy' || exit 1
$PY -m pip freeze 2> /dev/null | grep -i -E '^(torch|transformers|triton|kernels|accelerate|safetensors|huggingface|hf.xet|numpy|pyarrow)' > $R/versions.txt
# the same held-out text and calibration sequences as the skew job (pinned datasets)
step prep $PY ~/glm_skew.py prep nemotron ~/v/data --cal-seqs "${CAL_SEQS:-24}" --held-seqs "${HELD_SEQS:-4}" &
PREP=$!
# a slow host stops here, not at the cap (2026-10-10: a GLM pod fetched 9 of 73 files in 30 min): 8 min in, the weights must project to be on disk 45 min after the start
while kill -0 $DL 2> /dev/null && [ $(( $(date +%s) - T0 )) -lt 480 ]; do sleep 15; done
if kill -0 $DL 2> /dev/null; then
  t=$(( $(date +%s) - T0 )); b=$(du -sbc ~/v/m ~/v/hf 2> /dev/null | tail -1 | cut -f1)
  [ $(( ${b:-0} / (t + 1) * 2700 )) -ge $(( ${WEIGHTS_GB%.*} * 1000000000 )) ] || { echo "download: $(( ${b:-0} / 1000000 / (t + 1) )) MB/s after $(( t / 60 )) min, too slow for the weights in 45 min: stopped" >> $P; exit 1; }
fi
wait $DL || { echo "download: failed" >> $P; exit 1; }
df -h ~ > $R/disk-after-download.txt 2>&1
# a tiny random NemotronH on the CPU through every variant, the block codes against calib.py's, and on the GPUs the codes on each card and the tiny model split over them: a failure here means the
# code is wrong, so stop before the weights are loaded (the pod is deleted). After the download, so that it has the CPUs (hf_xet takes them all), and in 15 min at most
step selftest timeout 900 $PY ~/nemotron_quality.py selftest || exit 1
wait $PREP || { echo "prep: failed" >> $P; exit 1; }
cp ~/v/data/prep.json $R/ 2> /dev/null
step run $PY ~/nemotron_quality.py run --weights ~/v/m --data ~/v/data --out $R --deadline $DEADLINE --held-seqs "${HELD_SEQS:-4}" --count-seqs "${COUNT_SEQS:-12}" --gptq-seqs "${GPTQ_SEQS:-24}" \
  --gptq-min-tok "${GPTQ_MIN_TOK:-2048}" --chunk "${CHUNK:-0}" ${VARIANTS:+--variants "$VARIANTS"} ${EXPORT:+--export $R/export}
[ -n "${EXPORT:-}" ] && step export-check $PY ~/nemotron_hbr.py check $R/export --quick   # (the files' tensors: names, dtypes, shapes; the full read is `check DIR` on the laptop)
# the export to the private bucket (s3.env: temporary keys for that bucket alone, sourced in this subshell only, deleted after), 64 parallel parts of 64 MB; consumer pods download it from there
if [ -n "${EXPORT:-}" ] && [ -f ~/s3.env ]; then
  DEST="s3://glyd-models-712416033631/nemotron-3-super/hbr40/$(TZ=America/New_York date +%Y%m%d-%H%M)-EST/export/"
  step s3-cli bash -c 'python3 -m venv ~/v/awsenv && ~/v/awsenv/bin/pip install -q awscli'
  s3up() { ( set -a; . ~/s3.env; set +a; export AWS_CONFIG_FILE=$HOME/v/aws-config; A=~/v/awsenv/bin/aws
             $A configure set default.s3.max_concurrent_requests 64 && $A configure set default.s3.multipart_chunksize 64MB || exit 1
             t=$(date +%s); $A s3 cp --recursive --only-show-errors $R/export "$DEST" || exit 1
             b=$(du -sb $R/export | cut -f1); s=$(( $(date +%s) - t + 1 )); echo "s3: $b bytes to $DEST in $s s ($(( b / 1000000 / s )) MB/s)" ); }
  step s3-upload s3up
  grep -h "^s3:" $R/s3-upload.log >> $P
  rm -f ~/s3.env
elif [ -n "${EXPORT:-}" ]; then
  echo "s3: no s3.env: the export stays on the pod, copied by rp_run.sh as it goes" >> $P
fi
step report $PY ~/nemotron_quality.py report $R
nvidia-smi --query-gpu=index,memory.used --format=csv >> $R/gpus.txt 2>&1
