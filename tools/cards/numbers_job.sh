#!/usr/bin/env bash
# One model card's numbers on one pod (rp_run.sh contract: results in ~/results, ~/results/DONE at the end). Uploaded beside it: src.tar (git archive, prefix src/)
# and numbers.env (MODEL, REV, LEVEL). Writes bench.json (glyd-engine bench's line), fits.txt (the exit code of preflight at a 32,768 context), size.txt (bytes of
# a packed directory, if the model is one), gpu.txt (rp_run.sh).
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
. ~/numbers.env
P=$R/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
trap 'touch $R/DONE' EXIT
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
. ~/.cargo/env
step hfvenv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet'
( ~/v/venv/bin/hf download "$MODEL" --revision "$REV" --local-dir ~/v/m > $R/download.log 2>&1; echo "download: rc=$?" >> $P ) &
step lib bash -c 'cd ~/v/src && bash gpu/build_lib.sh ~/v/lib'
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so | head -1)
step build bash -c 'cd ~/v/src && cargo build --release -p glyd-engine'
wait
E=~/v/src/target/release/glyd-engine
case $LEVEL in swift) LAY="--layout i5";; kestrel) LAY="--layout i6";; *) LAY="";; esac
# (the bench's own context is 4,096: its GPU memory is the figure for a 4k conversation)
$E microbench ~/v/m $LAY --prompt 1024 --new 256 --ctx 4096 > $R/bench.out 2> $R/bench.err; echo "bench: rc=$?" >> $P
grep '^{' $R/bench.out | tail -1 > $R/bench.json
$E preflight ~/v/m:$LEVEL --context 32768 > $R/fits.out 2> $R/fits.err; echo $? > $R/fits.txt
[ -f ~/v/m/glyd.json ] && du -sb ~/v/m | cut -f1 > $R/size.txt
nvidia-smi > $R/gpu-final.txt
