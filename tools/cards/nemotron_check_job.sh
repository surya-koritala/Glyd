#!/usr/bin/env bash
# Nemotron 3 Super's Mamba-2 and attention layers on a GPU, against PyTorch (Linear GLY-132; rp_run.sh contract: results in ~/results, ~/results/DONE at the end; one A100 80 GB pod).
# Builds the library and the engine from the tarball, reads a few layers of nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (pinned revision 2dc98e2a) from the Hub by HTTP range (about 0.9 GB, no whole
# shard, no token), and runs, each against tools/nemotron_ref.py's reference:
#   1. selftest   the reference against transformers 5.19.0's own functions (CPU), and on the real layers' weights against its Mamba-2 and attention modules (real-vs-transformers)
#   2. kernels    glyd-engine nhkernels: kernels/nemotron.cu on random inputs, at a small shape and the model's (128 heads, 8 groups), and on the real layers' activations
#   3. layers     glyd-engine nhlayers: layers 0, 2 and 40 (Mamba-2) and 7 and 36 (attention) with their real weights through the engine's own code: the prompt in one call and in odd pieces, the scan
#                 token by token, single steps one by one and replayed from a CUDA graph, a prompt then steps; every stage of every case
#   4. toy        glyd-engine nhtoy: those five layers as a model (a part of the real embedding and head): prefill and decode steps as a graph and not, logits of every step
# Uploaded beside it: src.tar (git archive of the tree, prefix src/). Results: summary.txt (made last), steps.txt, one .log a step, kernels-*/, layers/, toy/, gpu.txt (rp_run.sh).
#
#   cd <the tree at the commit to test> && W=$(mktemp -d) && git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar && \
#   NAME=nemotron-check MAX_MIN=60 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh a100 tools/cards/nemotron_check_job.sh        # $1.79/h; about 40 minutes
# (rp_guard.sh deletes glyd-* pods older than its MAX_AGE: MAX_AGE=160 bash tools/runpod/rp_guard.sh & first.) Results come to $GLYD_RUNS/results-nemotron-check.
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
T0=$(date +%s)
MAX_MIN=${JOB_MAX_MIN:-60}   # (the launcher's cap, MAX_MIN=60 in the launch line below, counts the pod's start too: the job stops starting steps 8 minutes before its own)
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
late() { [ $(( ($(date +%s) - T0) / 60 )) -ge $((MAX_MIN - 8)) ]; }
trap 'touch $R/DONE' EXIT
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # (the repo is ungated: nothing here reads or sends a token)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1
SRC=~/v/src; E=$SRC/glyd-engine; REF=$E/tools/nemotron_ref.py
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
{ TZ=EST5EDT date; nvidia-smi; nvcc --version | tail -2; echo "arch sm_$ARCH"; cat $SRC/COMMIT 2> /dev/null; } > $R/machine.txt 2>&1

# the Python environment (torch is the template's where it has one), the Rust toolchain, the library and the engine, at the same time (every step has a time limit: a kernel that hangs ends its step, not the job)
( step pyenv timeout -k 10 1500 bash -c 'python3 -m venv --system-site-packages ~/v/venv && ( ~/v/venv/bin/python -c "import torch" 2> /dev/null || ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 ) && ~/v/venv/bin/pip install -q numpy safetensors "transformers==5.19.0"' ) &
PYENV=$!
step rustup timeout -k 10 600 bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1'
. ~/.cargo/env
( step lib timeout -k 10 2400 bash -c "cd $SRC && bash gpu/build_lib.sh ~/v/lib" ) &
LIB=$!
[ -n "$ARCH" ] && export GLYD_ENGINE_ARCHS=$ARCH   # (this GPU alone)
step engine timeout -k 10 2400 bash -c "cd $SRC && cargo build --release -p glyd-engine"
wait $PYENV
PY=~/v/venv/bin/python
B=$SRC/target/release/glyd-engine
step versions bash -c "$PY -c 'import torch, transformers; print(\"torch\", torch.__version__, torch.version.cuda, \"transformers\", transformers.__version__, torch.cuda.get_device_name())'; $B --version"
step selftest timeout -k 10 600 $PY -I $REF selftest
# the layers, by range from the Hub (while the library finishes): layers 0, 2, 40 (Mamba-2) and 7, 36 (attention), 1,024 embeddings of random tokens as their input
step fetch timeout -k 10 1200 $PY -I $REF fetch ~/v/mini --layers 0,2,7,36,40 --tokens 1024 --seed 1
step real-vs-transformers timeout -k 10 900 $PY -I $REF real ~/v/mini   # (the reference itself, on these layers' real weights: transformers' own modules, fp32)
wait $LIB
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so 2> /dev/null | head -1)
[ -x $B ] || { echo "no engine: see engine.log" >> $P; exit 1; }
[ -n "$GLYD_GPU_LIB" ] || echo "no library (see lib.log): the kernel checks need none, the layers and the toy model do" >> $P

# 2. the kernels on random inputs: a small shape, and the model's
for cfg in "small:--nh 16 --ng 2 --n 300 --split 150 --na 200" "real:--nh 128 --ng 8 --n 700 --split 333 --na 1000 --seed 2" "act:--n 700 --split 333 --na 1000 --from $HOME/v/mini"; do
    n=${cfg%%:*}; a=${cfg#*:}
    late && { echo "kernels-$n: skipped, late" >> $P; continue; }
    step kernels-$n-gen timeout -k 10 600 $PY -I $REF kernels gen ~/v/k-$n $a
    step kernels-$n-run timeout -k 10 900 $B nhkernels ~/v/k-$n/in ~/v/k-$n/out
    step kernels-$n-check timeout -k 10 900 $PY -I $REF kernels check ~/v/k-$n
    mkdir -p $R/kernels-$n; cp ~/v/k-$n/in/*.txt $R/kernels-$n/ 2> /dev/null
done
# 3. the layers: every stage of every case against the reference
late || [ -z "$GLYD_GPU_LIB" ] || {
    step layers-gen timeout -k 10 300 $PY -I $REF layers gen ~/v/mini ~/v/io
    step layers-run timeout -k 10 1500 $B nhlayers ~/v/mini ~/v/io
    step layers-check timeout -k 10 1500 $PY -I $REF layers check ~/v/mini ~/v/io
    # what dt < 0.001 does (informational: it is expected to FAIL where dt is small): transformers' chunk scan clamps dt to time_step_min (0.001); vLLM's and the engine's do not
    step layers-check-dtmin timeout -k 10 1500 $PY -I $REF layers check ~/v/mini ~/v/io --dt-min 0.001
}
# 4. the five layers as a model
late || [ -z "$GLYD_GPU_LIB" ] || {
    step toy-run timeout -k 10 900 $B nhtoy ~/v/mini/toy ~/v/mini/toy
    step toy-check timeout -k 10 600 $PY -I $REF toy ~/v/mini
}
mkdir -p $R/layers $R/toy
cp ~/v/io/spec.json $R/layers/ 2> /dev/null; cp ~/v/mini/toy/toy*.txt $R/toy/ 2> /dev/null; cp ~/v/mini/config.json $R/ 2> /dev/null

# the summary: every check's verdict, every case's time
{
    echo "Nemotron 3 Super's Mamba-2 and attention layers on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), tree $(cat $SRC/COMMIT 2> /dev/null)"
    echo "--- steps"; cat $P
    echo "--- selftest"; grep -E "^(ok|FAIL|selftest)" $R/selftest.log
    echo "--- the reference against transformers' modules on the real layers"; grep -E "^(ok|FAIL|real)" $R/real-vs-transformers.log 2> /dev/null
    for n in small real act; do echo "--- kernels, $n"; grep -E "^(ok|FAIL|kernels|mamba case|attention)" $R/kernels-$n-run.log $R/kernels-$n-check.log 2> /dev/null; done
    echo "--- layers: the verdicts (FAIL lines first), then every line"; grep -E "FAIL|^layers" $R/layers-check.log
    echo "--- layers: time in the layer by case (the engine's own, with the products)"; grep -E "^case " $R/layers-run.log
    echo "--- layers with transformers' dt limit (--dt-min 0.001): the layer from its input"; grep -E "reference alone|^layers" $R/layers-check-dtmin.log
    echo "--- layers: all"; cat $R/layers-check.log
    echo "--- toy model"; grep -E "^(graph|eager):|^(ok|FAIL|toy)|top-1" $R/toy-run.log $R/toy-check.log 2> /dev/null
} > $R/summary.txt 2>&1
cat $R/summary.txt | head -150
nvidia-smi > $R/gpu-final.txt 2>&1
