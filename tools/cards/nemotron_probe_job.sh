#!/usr/bin/env bash
# What the routed experts of Nemotron 3 Super leave to an expert-offload design, on a 4-GPU pod (rp_run.sh contract; launched by tools/cards/nemotron_probe.sh, Linear GLY-130): relu2 sparsity inside
# the experts, look-ahead routing and per-expert importance, over the held-out text of glm_skew.py and greedy decode steps, on the bf16 model with nothing rounded.
#   Launch (4 x A100 80 GB, class nemo, $7.16/h):  bash tools/cards/nemotron_probe.sh        (--dry-run first: it prints the plan and the worst case, and starts nothing)
#   Cost: the job is ~22 min (setup ~8, pass ~9, decodes ~4, report ~1; guesses until a first run) and the pod's start and the last pull add 5-15, so ~$3.2-4.4; MAX_MIN (default 60, at most 150) caps it at $7.16.
# Uploaded beside it: probe.env (optionally CAL_SEQS HELD_SEQS DECODE_PROMPTS DECODE_NEW CHUNK MEM_FRAC JOB_MAX_MIN SELFTEST_GPU OMP_NUM_THREADS), research/lossy-2026/nemotron_probe.py, nemotron_quality.py,
# glm_quality.py and glm_skew.py.
# ~/results: steps.txt (wall clock per step), probe.md + probe.json (after the pass and after the decodes), meta.json, prefill.npz and decode.npz (rewritten after every MoE layer), versions.txt, logs.
# transformers 5.19.0 as in the skew and quality jobs: the model stays bf16 on the GPUs for the whole job (no mamba-ssm or causal-conv1d: the torch Mamba-2, as those jobs ran it).
set -u
R=~/results; mkdir -p $R ~/v; cd ~
set -a; . ~/probe.env; set +a   # (exported: the run reads MEM_FRAC and SELFTEST_GPU)
T0=$(date +%s); P=$R/steps.txt; : > $P
DEADLINE=$((T0 + ${JOB_MAX_MIN:-45} * 60))   # the launcher's cap (MAX_MIN) also covers the pod's start: no phase is started that would not end 2 min before this
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
collect() { touch $R/DONE; }
trap collect EXIT
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=$HOME/v/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the CPUs this container really has: torch sizes its CPU thread pools by the host's cores, but a container is often given a quota of a few (cgroup cpu.max), and that many threads on that quota
# are throttled to a crawl. Seen 2026-10-10: the GLM quality job's selftest on a pod was still running after 8 min (1 min on a laptop; a forward of its tiny model took 11 s); the cause is a
# guess, the threads or the download taking every CPU, and both are dealt with here: the thread count, the selftest after the download, and a stop on a host with too few
cpus() { local q p c=${CGROOT:-/sys/fs/cgroup}; { read -r q p < $c/cpu.max; } 2> /dev/null && [ "$q" != max ] && { echo $(( (q + p - 1) / p )); return; }
         { read -r q < $c/cpu/cpu.cfs_quota_us; read -r p < $c/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:--1}" -gt 0 ] && { echo $(( (q + p - 1) / p )); return; }; nproc; }
CPUS=$(cpus); export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPUS}; export MKL_NUM_THREADS=$OMP_NUM_THREADS
# a host with a quota of a few CPUs stops here, before the download and the installs that share them (this job's selftest takes 12 s on one thread of a laptop; the quota seen on the GLM pod made its
# selftest twenty times slower, and 8 is a floor chosen, not measured)
[ "$CPUS" -ge 8 ] || { echo "cpus: $CPUS, too few for the download, the installs and the selftest to share in time: stopped (launch again for another host)" >> $P; exit 1; }
PY=~/v/venv/bin/python
eval "$(python3 ~/nemotron_probe.py plan)"   # REPO REV POD_CLASS DISK K NE N_MOE ... WEIGHTS_GB SETUP_MIN PASSES_MIN

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > $R/gpus.txt 2>&1; df -h ~ > $R/disk-start.txt 2>&1; echo "CPUs for torch: $OMP_NUM_THREADS (the container has $CPUS, nproc $(nproc))" > $R/cpus.txt
# the hub client first, so the weights (the pinned revision, 247 GB) download while torch and the rest install and the token sequences are made
step venv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet' || exit 1
dl() { for i in 1 2 3; do timeout 1800 ~/v/venv/bin/hf download "$REPO" --revision "$REV" --local-dir ~/v/m && return 0; done; return 1; }   # (a stalled attempt is cut after 30 min and resumes)
step download dl &
DL=$!
# the CUDA build of torch: wheels for 12.8 run on any driver >= 570 (Hopper, Ampere and the Blackwell SM120 cards); PEP 668: a venv
# the versions glm_skew.py's run on four A100 had (torch 2.11.0+cu128, accelerate 1.15.0); it ran with kernels and flash-linear-attention installed, which this model does not use
step pyenv bash -c '~/v/venv/bin/pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 && ~/v/venv/bin/pip install -q "transformers==5.19.0" "accelerate==1.15.0" "safetensors>=0.8" pyarrow numpy' || exit 1
$PY -m pip freeze 2> /dev/null | grep -i -E '^(torch|transformers|triton|kernels|accelerate|safetensors|huggingface|hf.xet|numpy|pyarrow)' > $R/versions.txt
# the same held-out text and calibration sequences as the skew and quality jobs (pinned datasets): the first 12 calibration sequences of each domain are the quality job's routing counts' (CAL_SEQS 12)
step prep $PY ~/glm_skew.py prep nemotron ~/v/data --cal-seqs "${CAL_SEQS:-12}" --held-seqs "${HELD_SEQS:-4}" &
PREP=$!
# a slow host stops here, not at the cap (2026-10-10: a GLM pod fetched 9 of 73 files in 30 min): 8 min in, the weights must project to be on disk 30 min after the start (the quality job projects
# 45 min for its JOB_MAX_MIN of 128; this one's is 45, and 30 leaves 15 for the selftest and the pass)
while kill -0 $DL 2> /dev/null && [ $(( $(date +%s) - T0 )) -lt 480 ]; do sleep 15; done
if kill -0 $DL 2> /dev/null; then
  t=$(( $(date +%s) - T0 )); b=$(du -sbc ~/v/m ~/v/hf 2> /dev/null | tail -1 | cut -f1)
  [ $(( ${b:-0} / (t + 1) * 1800 )) -ge $(( ${WEIGHTS_GB%.*} * 1000000000 )) ] || { echo "download: $(( ${b:-0} / 1000000 / (t + 1) )) MB/s after $(( t / 60 )) min, too slow for the weights in 30 min: stopped" >> $P; exit 1; }
fi
wait $DL || { echo "download: failed" >> $P; exit 1; }
df -h ~ > $R/disk-after-download.txt 2>&1
# a tiny random NemotronH on the CPU against a naive reference in every statistic, hand-made cases with known answers, and on the GPUs the same cases on each card and the tiny model split over them: a failure
# here means the code is wrong, so stop before the weights are loaded (the pod is deleted). After the download, so that it has the CPUs (hf_xet takes them all), and in 15 min at most
step selftest timeout 900 $PY ~/nemotron_probe.py selftest || exit 1
wait $PREP || { echo "prep: failed" >> $P; exit 1; }
cp ~/v/data/prep.json $R/ 2> /dev/null
step run $PY ~/nemotron_probe.py run --weights ~/v/m --data ~/v/data --out $R --deadline $DEADLINE --held-seqs "${HELD_SEQS:-4}" --cal-seqs "${CAL_SEQS:-12}" \
  --decode-prompts "${DECODE_PROMPTS:-20}" --decode-new "${DECODE_NEW:-64}" --chunk "${CHUNK:-0}"
step report $PY ~/nemotron_probe.py report $R
nvidia-smi --query-gpu=index,memory.used --format=csv >> $R/gpus.txt 2>&1
