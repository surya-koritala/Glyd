#!/usr/bin/env bash
# Nemotron 3 Super's latent mixture-of-experts layers on a GPU, against PyTorch (Linear GLY-132; rp_run.sh contract: results in ~/results, ~/results/DONE at the end; one A100 80 GB pod).
# Builds the library and the engine from the tarball and runs, each a step in steps.txt with its log:
#   1. selftest   tools/nemotron_ref.py selftest (CPU): the reference against transformers 5.19.0's own functions and modules, NemotronHMoE among them; the tiny model's config against the engine's fixture
#   2. emu        kernels/moe.cu on the host (tools/moe_emu_test.cpp), as moe_check_job.sh runs it
#   3. moecheck   glyd-engine moecheck --dump: the core (router, grouped products over Int5, Int6 and Int4 experts, the sparse second product over their row layouts, shared expert) against the CPU reference, then
#                 tools/moe_ref.py on its dump (the script is tried on the CPU reference's own dump first)
#   4. tiny       for each level (6: six-bit codes, 5: five-bit, 4: wren, the experts four-bit and the latent projections and shared expert six-bit) a random tiny NemotronH of the pattern M E * E M E
#                 (nemotron_ref.py moetoy: no Hub read; the experts, latent projections and shared expert are on the grid of the layouts the engine holds them in, so that its packs hold them exactly: an expert's down
#                 matrix on the grid of both its block layout and its row layout), then
#                 layers  glyd-engine nhlayers on its three mixture-of-experts layers (src/moe_nh.rs through the engine's own code): the prompt in one call and in odd pieces, single steps one by one and replayed from
#                         a CUDA graph, a prompt then steps; every stage (fc1_latent_proj, the router's choices and weights against the reference's, the experts' sum, fc2_latent_proj with the shared expert);
#                         the experts' down matrices are packed in the row layout as the engine loads them, so these cases run the sparse product; then again with GLYD_ENGINE_DENSE_DOWN=1 (the block layout: the
#                         library's dense product) against the same reference
#                 toy     glyd-engine nhtoy on the whole tiny model: prefill in two chunks and decode steps, as a graph and not, the logits of every step (the median position within 5e-2 of the reference's
#                         largest logit, the top-1 token the reference's at 98% of the positions)
#   5. regression a few real layers (0, 2, 7, 36, 40: about 0.9 GB read from the Hub by HTTP range, no token) through nhlayers and nhtoy, the checks nemotron_check_job.sh runs: the Mamba-2 and attention
#                 layers and the toy model's top-1 agreement (98.2% there) through the code this change touched
# Uploaded beside it: src.tar (git archive of the tree, prefix src/). Results: summary.txt (made last), steps.txt, one .log a step, tiny6/, tiny5/, tiny4/ (spec.json, toy.txt), gpu.txt (rp_run.sh).
#
#   cd <the tree at the commit to test> && W=$(mktemp -d) && git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar && \
#   NAME=nh-moe-check MAX_MIN=60 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh a100 tools/cards/nh_moe_check_job.sh        # $1.79/h; about 40 minutes, $1.2 (the cap, 60 minutes: $1.79)
# (It took 28 minutes with Int5 and Int6 alone, benchmarks/gpu/nh-moe-check-2026-10-10: a layout more is half again the host's packing and PyTorch's reference, so the steps' limits and the cap are raised.)
# (rp_guard.sh deletes glyd-* pods older than its MAX_AGE: MAX_AGE=120 bash tools/runpod/rp_guard.sh & first.) Results come to $GLYD_RUNS/results-nh-moe-check.
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
T0=$(date +%s)
MAX_MIN=${JOB_MAX_MIN:-60}   # (the launcher's cap, MAX_MIN=60 in the launch line above, counts the pod's start too: the job stops starting steps 8 minutes before its own)
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
late() { [ $(( ($(date +%s) - T0) / 60 )) -ge $((MAX_MIN - 8)) ]; }
trap 'touch $R/DONE' EXIT
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # (the repo is ungated: nothing here reads or sends a token)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 RUST_BACKTRACE=1
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
step emu timeout -k 10 600 bash -c "cd $E && c++ -std=c++17 -O1 -fno-strict-aliasing -pthread tools/moe_emu_test.cpp -o ~/v/moe_emu_test && ~/v/moe_emu_test"
wait $PYENV
PY=~/v/venv/bin/python
B=$SRC/target/release/glyd-engine
step versions bash -c "$PY -c 'import torch, transformers; print(\"torch\", torch.__version__, torch.version.cuda, \"transformers\", transformers.__version__, torch.cuda.get_device_name())'; $B --version"
step selftest timeout -k 10 600 $PY -I $REF selftest
# the real layers, by range from the Hub (while the library finishes)
step fetch timeout -k 10 1200 $PY -I $REF fetch ~/v/mini --layers 0,2,7,36,40 --tokens 1024 --seed 1
wait $LIB
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so 2> /dev/null | head -1)
[ -x $B ] || { echo "no engine: see engine.log" >> $P; exit 1; }
[ -n "$GLYD_GPU_LIB" ] || { echo "no library: see lib.log" >> $P; exit 1; }

# 3. the core: the CPU reference's own dump through the PyTorch script first (if it fails there it is the script's, not the GPU's), then the GPU's
step moecheck-cpu timeout -k 10 1500 bash -c "$B moecheck swiglu-8of128 relu2-22of512 --cpu-dump ~/v/cpudump && $PY -I $E/tools/moe_ref.py ~/v/cpudump"
step moecheck timeout -k 10 1500 $B moecheck --dump ~/v/dump
[ -d ~/v/dump ] && step moecheck-torch timeout -k 10 1500 $PY -I $E/tools/moe_ref.py ~/v/dump

# 4. the random tiny model at each level: its mixture-of-experts layers by stage, then the whole model
for lv in 6 5 4; do
    late && { echo "tiny$lv: skipped, late" >> $P; continue; }
    T=~/v/tiny$lv
    step tiny$lv-make timeout -k 10 300 $PY -I $REF moetoy $T --level $lv
    step tiny$lv-layers-gen timeout -k 10 300 $PY -I $REF layers gen $T $T/io --level $lv --kinds E
    cp -r $T/io $T/io-dense 2> /dev/null   # (the inputs and the reference's outputs, for the run with the experts' down matrices dense)
    step tiny$lv-layers-run timeout -k 10 900 $B nhlayers $T $T/io
    step tiny$lv-layers-check timeout -k 10 900 $PY -I $REF layers check $T $T/io
    step tiny$lv-dense-layers-run env GLYD_ENGINE_DENSE_DOWN=1 timeout -k 10 900 $B nhlayers $T $T/io-dense
    step tiny$lv-dense-layers-check timeout -k 10 900 $PY -I $REF layers check $T $T/io-dense
    step tiny$lv-toy-run timeout -k 10 900 $B nhtoy $T/toy $T/toy
    step tiny$lv-toy-check timeout -k 10 600 $PY -I $REF toy $T
    mkdir -p $R/tiny$lv; cp $T/io/spec.json $R/tiny$lv/ 2> /dev/null; cp $T/toy/toy.txt $R/tiny$lv/ 2> /dev/null
done

# 5. the real Mamba-2 and attention layers through the code this change touched (what nemotron_check_job.sh steps 3 and 4 run)
late || {
    step real-layers-gen timeout -k 10 300 $PY -I $REF layers gen ~/v/mini ~/v/io
    step real-layers-run timeout -k 10 1500 $B nhlayers ~/v/mini ~/v/io
    step real-layers-check timeout -k 10 1500 $PY -I $REF layers check ~/v/mini ~/v/io
    step real-toy-run timeout -k 10 900 $B nhtoy ~/v/mini/toy ~/v/mini/toy
    step real-toy-check timeout -k 10 600 $PY -I $REF toy ~/v/mini
}

# the summary: every check's verdict
{
    echo "Nemotron 3 Super's mixture-of-experts layers on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), tree $(cat $SRC/COMMIT 2> /dev/null)"
    echo "--- steps"; cat $P
    echo "--- selftest (the reference against transformers)"; grep -E "^(ok|FAIL|selftest)" $R/selftest.log 2> /dev/null | grep -E "mixture|moe|FAIL|selftest|config"
    echo "--- kernels/moe.cu on the host"; tail -1 $R/emu.log 2> /dev/null
    echo "--- the core on the GPU against the CPU (moecheck)"; grep -E "^(ok|FAIL|all|FAILED|----|[A-Za-z].*sm_)" $R/moecheck.log 2> /dev/null
    echo "--- PyTorch on the CPU reference's dump, and on the GPU's"; grep -E "^(ok|FAIL|all|FAILED|no case)" $R/moecheck-cpu.log $R/moecheck-torch.log 2> /dev/null
    for lv in 6 5 4; do
        echo "--- tiny model, level $lv: mixture-of-experts layers by stage (verdicts; the experts' down matrices sparse, then dense)"; grep -E "FAIL|^layers" $R/tiny$lv-layers-check.log $R/tiny$lv-dense-layers-check.log 2> /dev/null
        echo "--- tiny model, level $lv: time in the layer by case"; grep -E "^case " $R/tiny$lv-layers-run.log 2> /dev/null
        echo "--- tiny model, level $lv: the whole model"; grep -E "^(graph|eager):|^(ok|FAIL|toy)|top-1" $R/tiny$lv-toy-run.log $R/tiny$lv-toy-check.log 2> /dev/null
    done
    echo "--- real layers: the verdicts (FAIL lines first)"; grep -E "FAIL|^layers" $R/real-layers-check.log 2> /dev/null
    echo "--- real layers: the toy model"; grep -E "^(graph|eager):|^(ok|FAIL|toy)|top-1" $R/real-toy-run.log $R/real-toy-check.log 2> /dev/null
} > $R/summary.txt 2>&1
cat $R/summary.txt | head -200
nvidia-smi > $R/gpu-final.txt 2>&1
