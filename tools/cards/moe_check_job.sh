#!/usr/bin/env bash
# The mixture-of-experts layer's checks on one GPU pod (rp_run.sh contract: results in ~/results, ~/results/DONE at the end; Linear GLY-101). Uploaded beside it: src.tar (git archive, prefix src/).
# Builds the GPU library (gpu/glyd_gpu.cu: its grouped products take Int5, Int6 and Int4 packs) and the engine, then runs, each a step in steps.txt with its log:
#   emu        kernels/moe.cu and kernels/moe_rows.cu on the host (glyd-engine/tools/moe_emu_test.cpp: the router, the activations, the shared expert's sum against plain formulas; the sparse second product against the dense sum)
#   moecheck   `glyd-engine moecheck --dump`: the layer on the GPU against the CPU reference, Int5, Int6 and Int4, four routers and activations, 1 token to a prompt chunk, the decode step as a CUDA graph
#   speed      `glyd-engine moecheck --no-check --speed`: the layer at the real shapes of Qwen3.6-35B-A3B, Gemma-4-26B-A4B, GLM-5.3-Flash and Nemotron 3 Super (microseconds a layer, tokens a second, GB/s of packs read;
#              Nemotron's again with its second matrices dense and sparse, on activations that are 92% zeros)
#   selftest   tools/moe_ref.py on the CPU reference's own dump (no GPU): the PyTorch script is right before it judges the GPU
#   torch      tools/moe_ref.py on moecheck's dump: transformers-style bf16 PyTorch against the GPU's outputs
# ~/results: steps.txt, summary.txt (made last), gpu.txt, one .log a step (the dumps stay in ~/v: rp_run.sh pulls ~/results every minute). Optional environment: JOB_MAX_MIN (default 50; a step that would not end in time is skipped).
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
T0=$(date +%s); DEADLINE=$((T0 + ${JOB_MAX_MIN:-50} * 60))
# a step: its name, the seconds it may take at most, the command (its output in NAME.log, its time and exit code in steps.txt); skipped where too little time is left
step() {
    local n=$1 secs=$2 left=$((DEADLINE - $(date +%s))); shift 2
    [ "$left" -gt 60 ] || { echo "$n: skipped, $left s left" >> $P; return 1; }
    [ "$secs" -le "$left" ] || secs=$left
    local t=$(date +%s); timeout "$secs" "$@" > $R/$n.log 2>&1; local rc=$?
    echo "$n: rc=$rc $(( $(date +%s) - t )) s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P
    return $rc
}
trap 'touch $R/DONE' EXIT
export PATH=/usr/local/cuda/bin:$PATH RUST_BACKTRACE=1
nvidia-smi > $R/gpu.txt 2>&1; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader >> $R/gpu.txt; nproc >> $R/gpu.txt; command -v nvcc >> $R/gpu.txt; nvcc --version >> $R/gpu.txt 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")   # (the engine's kernels for this GPU alone; the library's build is gpu/build_lib.sh's, all of them)
step rustup 300 bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1'
. ~/.cargo/env
# the library and the engine at once
( cd ~/v/src && timeout 1800 bash gpu/build_lib.sh ~/v/lib > $R/lib.log 2>&1; echo "lib: rc=$? $(( $(date +%s) - T0 )) s from the start" >> $P ) &
step engine 1500 bash -c "cd ~/v/src && GLYD_ENGINE_ARCHS=$ARCH cargo build --release -p glyd-engine"
wait
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so 2> /dev/null | head -1)
E=~/v/src/target/release/glyd-engine
G=~/v/src/glyd-engine
step emu 600 bash -c "cd $G && c++ -std=c++17 -O1 -fno-strict-aliasing -pthread tools/moe_emu_test.cpp -o ~/v/moe_emu_test && ~/v/moe_emu_test"
# PyTorch (CPU) for the reference: the image's, else the CPU wheel in a venv
PY=python3
if ! $PY -c 'import torch, numpy' 2> /dev/null; then
    step pyenv 900 bash -c 'python3 -m venv ~/v/venv && ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu && ~/v/venv/bin/pip install -q numpy' && PY=~/v/venv/bin/python
fi
$PY -c 'import torch; print(torch.__version__)' > $R/torch-version.txt 2>&1
# the PyTorch script on the CPU reference's dump first: if it fails there, it is the script's, not the GPU's
step selftest 900 bash -c "$E moecheck swiglu-8of128 relu2-22of512 --cpu-dump ~/v/cpudump && $PY $G/tools/moe_ref.py ~/v/cpudump"
step moecheck 1500 $E moecheck --dump ~/v/dump
step speed 1500 $E moecheck --no-check --speed   # (Nemotron 3 Super's second product dense and sparse too: packs of four experts a layout, two layers a layout)
[ -d ~/v/dump ] && step torch 900 $PY $G/tools/moe_ref.py ~/v/dump
{
    echo "mixture-of-experts layer (GLY-101) on $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1); tree $(gzip -dc ~/src.tar 2> /dev/null | git get-tar-commit-id 2> /dev/null); torch $(tail -1 $R/torch-version.txt)"
    echo "--- steps"; cat $P
    echo "--- kernels/moe.cu on the host"; tail -1 $R/emu.log 2> /dev/null
    echo "--- the layer on the GPU against the CPU (moecheck)"; grep -E '^(ok|FAIL|all|FAILED|----|[A-Za-z].*sm_)' $R/moecheck.log 2> /dev/null
    echo "--- PyTorch on the CPU reference's dump (the script's own check)"; grep -E '^(ok|FAIL|all|FAILED)' $R/selftest.log 2> /dev/null
    echo "--- PyTorch on the GPU's outputs"; grep -E '^(ok|FAIL|all|FAILED|no case)' $R/torch.log 2> /dev/null
    echo "--- speed"; grep -E '^(qwen36|gemma4|glm53|nemotron3|  T=)' $R/speed.log 2> /dev/null
} > $R/summary.txt 2>&1
nvidia-smi > $R/gpu-final.txt 2>&1
