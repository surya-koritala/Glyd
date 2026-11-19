#!/usr/bin/env bash
# GLY-39 / GLY-134: the four- and three-bit block layouts (Int4 = :wren, Int3) on a GPU (rp_run.sh contract: results in ~/results, ~/results/DONE at the end). Uploaded beside it: src.tar (git archive, prefix src/).
# The library and the engine built; examples/int4 and int3 (the GPU's decode of four matrices is unpack_int4 / unpack_int3 bit for bit, and its products from 1 to 512 tokens are the tiered layout's of the same
# weights), int5 and int6 beside them (the dispatch they share); Qwen3-0.6B packed at i4 and i3, saved as glyd-v7 and verified by the GPU's decode against glyd.json's sha256; Qwen3.5-0.8B run from the Hub as :wren
# and from its directory, and a raw completion at i4 and i3; Qwen3.5-9B benched at i4 (and i5, i3: what a step down costs or gains); the crates' tests.
# Writes steps.txt, a log a step, bench-*.json (glyd-engine bench's line), gpu.txt (rp_run.sh). tag-hummingbird's status 1 is the refusal asked for (the three-bit layout is no level).
#   W=$(mktemp -d); git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar
#   NAME=int4-a100 MAX_MIN=75 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh a100 tools/cards/int4_check_job.sh        (Ampere; the class l40 for Ada)
set -u
START=$(date +%s)
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; echo "$n: rc=$? $(( $(date +%s) - t )) s" >> $P; }
late() { [ $(( $(date +%s) - START )) -gt $(( $1 * 60 )) ]; }   # (the optional steps are left out past $1 minutes of the job, so that the results are back before the launcher's MAX_MIN of 75)
trap 'touch $R/DONE' EXIT
export PATH=/usr/local/cuda/bin:$PATH HF_XET_HIGH_PERFORMANCE=1 GLYD_CACHE=$HOME/v/cache
ls -d /usr/local/cuda* >> $P 2>&1; nvcc --version | tail -2 >> $P 2>&1
step rustup bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
. ~/.cargo/env
step hfvenv bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q "huggingface_hub>=1.5,<2" hf_xet'
# the three models, while the library builds (Qwen3.5 at the revisions of benchmarks/gpu/engine-qwen35-2026-10-06)
dl() { ~/v/venv/bin/hf download "$1" ${2:+--revision "$2"} --local-dir "$3" > $R/download-$4.log 2>&1; echo "download-$4: rc=$?" >> $P; touch ~/v/$4.dl; }
( dl Qwen/Qwen3-0.6B "" ~/v/q06 q06; dl Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17 ~/v/m08 m08; dl Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a ~/v/m9 m9 ) &
step lib bash -c 'cd ~/v/src && bash gpu/build_lib.sh ~/v/lib'
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so | head -1)
step build bash -c 'cd ~/v/src && cargo build --release -p glyd-engine -p glyd-gpu && cargo build --release -p glyd-gpu --example int4 --example int3 --example int5 --example int6'
X=~/v/src/target/release/examples; E=~/v/src/target/release/glyd-engine; G=~/v/src/target/release/glyd-gpu
for n in int4 int3 int5 int6; do step $n-check $X/$n; done
# glyd-v7 saves: packed on the CPU (each pack checked against its weights), decoded by the GPU's library and compared with glyd.json's sha256
until [ -f ~/v/q06.dl ]; do sleep 5; done
for l in i4 i3; do
  step pack-q06-$l $G pack ~/v/q06 ~/v/p06-$l --layout $l
  step verify-q06-$l $G verify ~/v/p06-$l --device cuda:0
done
# the level tag: Qwen3.5-0.8B from the Hub (the engine downloads it, finds no glyd/Qwen3.5-0.8B-wren, packs it into its cache), the same from the directory; a raw completion at i4 and i3 (packed as it loads)
until [ -f ~/v/m08.dl ]; do sleep 5; done
step run-wren-hub $E run Qwen/Qwen3.5-0.8B:wren --no-think -p "Name three primes above 100."
step run-wren-dir $E run ~/v/m08:wren --no-think -p "What is 17 times 23?"
step tag-hummingbird $E run ~/v/m08:hummingbird -p "hi"
for l in i4 i3; do step generate-$l $E generate ~/v/m08 "The three primary colours of light are" --new 32 --layout $l; done
until [ -f ~/v/m9.dl ]; do sleep 5; done
step preflight-wren $E preflight ~/v/m9:wren --context 32768
# the bench's own context is 4,096: its GPU memory is the figure for a 4k conversation
bench() { local n=$1; shift; $E bench ~/v/m9 "$@" --prompt 1024 --new 256 --ctx 4096 > $R/$n.out 2> $R/$n.err; echo "$n: rc=$?" >> $P; grep '^{' $R/$n.out | tail -1 > $R/$n.json; }
bench bench-i4 --layout i4
late 40 || bench bench-i5 --layout i5
late 45 || bench bench-i3 --layout i3
step test-gpu bash -c 'cd ~/v/src && cargo test --release -p glyd-gpu'
late 50 || step test-engine bash -c 'cd ~/v/src && cargo test -p glyd-engine'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > $R/gpu-final.txt
