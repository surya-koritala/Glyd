#!/usr/bin/env bash
# A calibrated checkpoint on one pod (GLY-123; rp_run.sh contract: results in ~/results, ~/results/DONE at the end): the model at its revision downloaded, GPTQ with act-order at the level's width (research/lossy-2026/calib.py:
# 512 sequences of 2,048 tokens of pinned data, as the study scored it), its codes and scales packed as they are by glyd-gpu pack --plan --codes into a glyd-v7 directory, which glyd-gpu verify decodes on the CPU and calib.py
# check finds decoded to the codes times the scales, bit for bit. The first step to fail ends the job. Uploaded beside it: src.tar (git archive, prefix src/) and pack.env (MODEL, REV the full commit, LEVEL linnet or kestrel).
# Writes steps.txt, a log a step, plan.json, calibration.json, glyd.json, files.json (each file of the directory: bytes, sha256), gpu.txt (rp_run.sh) and, last, packed/: the directory itself, moved into the results only once
# every step passed, as rp_run.sh pulls the results every minute (about 19 GB for a 27B linnet; ponytail: it comes back through the laptop with the results, rsync it from the pod if that is too slow). No token on the pod:
# the model and the datasets are public; the upload to the Hub is huggingface.yml's, in GitHub Actions (input prebuilt: the directory's files as the assets of a release).
#   W=$(mktemp -d); git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar; printf 'MODEL=Qwen/Qwen3.8-27B\nREV=<the full commit>\nLEVEL=linnet\n' > $W/pack.env
#   NAME=calpack-27b-linnet MAX_MIN=150 RP_DISK=250 UPLOAD="$W/src.tar $W/pack.env" bash tools/runpod/rp_class.sh a100 tools/cards/calibrated_pack_job.sh     (about 1 hour of an A100 for a 27B model)
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
. ~/pack.env
P=$R/steps.txt; : > $P
trap 'touch $R/DONE' EXIT
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s" >> $P; [ $rc = 0 ] || exit $rc; }
case $LEVEL in linnet) BITS=5; LAYOUT=i5;; kestrel) BITS=6; LAYOUT=i6;; *) echo "LEVEL is linnet or kestrel" > $R/error.txt; exit 2;; esac
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_CACHE=~/v/hub  # (the cache: hf and glyd-gpu both read it from here)
S=~/v/src; PY=~/v/venv/bin/python; B=$S/target/release/glyd-gpu; K=~/v/kl; C=$K/codes; M=$HF_HUB_CACHE/models--${MODEL/\//--}
step pyenv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && ~/v/venv/bin/pip install -q "transformers==5.18.0" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub>=1.5,<2" hf_xet pyarrow flash-linear-attention'  # (as calib_job.sh)
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1'  # (the toolchain huggingface.yml packs with)
. ~/.cargo/env
step build bash -c "cd $S && cargo build --release --manifest-path glyd-gpu/Cargo.toml --bin glyd-gpu"
# the model in the Hub's cache as cpu-pack has it (glyd-gpu reads a repo id's pinned commit from refs/main), so that glyd.json names the repo and the commit
step download bash -c "~/v/venv/bin/hf download '$MODEL' --revision '$REV' --include '*.safetensors' --include '*.json' --include '*.txt' --include '*.model' --include '*.jinja' && mkdir -p $M/refs && printf '%s' '$REV' > $M/refs/main"
D=$M/snapshots/$REV
step plan env PYTHONPATH=$S/glyd-gpu-py $PY $S/huggingface/plan.py "$MODEL" $R/plan.json --revision "$REV" --lossy
cd $S/research/lossy-2026
step ref $PY kld_study.py ref $D $K
step calib_build $PY calib.py build $D $K --seqs 512
step quantize $PY calib.py quantize $D $K --bits $BITS --seqs 512
step pack $B pack "$MODEL" ~/v/packed --plan $R/plan.json --layout $LAYOUT --shard-bytes 1000000000 --codes $C
step verify $B verify ~/v/packed --device cpu
step check $PY calib.py check ~/v/packed $R/plan.json $C
cp ~/v/packed/glyd.json $C/calibration.json $K/calib.json $R/
$PY -c 'import hashlib, json, sys; from pathlib import Path; print(json.dumps([{"file": f.name, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(Path(sys.argv[1]).iterdir())], indent=0))' ~/v/packed > $R/files.json
nvidia-smi > $R/gpu-final.txt
mv ~/v/packed $R/packed
