#!/usr/bin/env bash
# glyd-engine on one GPU, unattended: nvcc and Rust, the engine built, Qwen3-8B downloaded and packed in both layouts, its KV tables, the tokens and scores against vLLM+Glyd's reference (made on an L4:
# the teacher-forced top-1 agreement is the figure that carries over between GPUs), prefill (1,024 and 3,500 tokens) and decode with cuBLAS, the engine's own products and the library's routes, a first run
# of Qwen3-0.6B through the engine's own downloader.
#
# MINUTES: 38 at most (it stops itself at MAX_MIN, default 38; about 33 in use: setup 8, model and tables 9, tests 14, the rest)
#
# For an instance of Lambda's gpu-base-24-04 (its driver runs CUDA 13), this checkout at $W/glyd (W: $HOME):
#   GPU=a100 GLYD_GPU_LIB=/path/libglyd_gpu_cuda13.so bash glyd-engine/jobs/engine_gpu_job.sh       (or engine_a100_job.sh, engine_h100_job.sh)
# GLYD_GPU_LIB: the prebuilt library (28 MB, from the glyd-gpu wheel); without it the job builds one with gpu/build_lib.sh (about 5 minutes more, in step with the engine's build).
# Results in $W/results/engine-$GPU/: summary.txt (made last), job.log, one .out a step, clocks.csv; DONE when it ends, FAILED (with the reason) where it could not go on.
# SKIP_SETUP=1 (a machine that has nvcc, Rust, the engine built at ENGINE_BIN and the model in HF_HOME: the dry run on the L4): the setup, the build and the download are not made.
set -uo pipefail
GPU="${GPU:-gpu}"
MAX_MIN="${MAX_MIN:-38}"
W="${W:-$HOME}"
REPO="${REPO:-$W/glyd}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
R="$W/results/engine-$GPU"
mkdir -p "$R"
T0=$(date +%s)
log() { echo "$(date -u +%T) ($(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
left() { echo $(( T0 + MAX_MIN * 60 - $(date +%s) )); }
fail() { log "FAILED: $*"; echo "$*" > "$R/FAILED"; touch "$R/DONE"; exit 1; }
# a step: its name, the seconds it may take, the command; one the job has no time for is skipped
step() {
    local name=$1 secs=$2
    shift 2
    local l
    l=$(left)
    if [ "$l" -lt 90 ]; then log "skipped $name: $l s left"; return 0; fi
    [ "$secs" -gt $((l - 30)) ] && secs=$((l - 30))
    log "$name (at most $secs s)"
    timeout "$secs" "$@" > "$R/$name.out" 2>&1
    log "$name: exit $?"
}
echo "MINUTES: $MAX_MIN at most" > "$R/summary.txt"
{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv; nproc; free -g | head -2; df -h "$W" | tail -1; git -C "$REPO" rev-parse --short HEAD 2> /dev/null; } > "$R/machine.txt" 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -l 5 > "$R/clocks.csv" 2>&1 &
CLK=$!
trap 'kill $CLK 2> /dev/null' EXIT

cv=$(nvidia-smi | grep -o "CUDA Version: [0-9]*" | grep -o "[0-9]*$")
[ "${cv:-0}" -ge 13 ] || fail "the driver runs CUDA ${cv:-?}; the engine's kernels and library are built for CUDA 13 (driver 580 or newer)"

if [ -z "${SKIP_SETUP:-}" ]; then
    log "setup: nvcc, cuBLAS and the Hub client from pip, Rust"
    mkdir -p "$HOME/tools/uv"
    [ -x "$HOME/tools/uv/uv" ] || curl -sL "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$HOME/tools/uv"
    UV="$HOME/tools/uv/uv"
    "$UV" venv -q --python 3.12 "$W/enveng" && "$UV" pip install -q --python "$W/enveng/bin/python" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" \
        "nvidia-cuda-runtime==13.0.*" "nvidia-cublas==13.0.*" huggingface_hub hf_xet > "$R/pip.out" 2>&1 || fail "the pip packages (see pip.out)"
    NV=$(ls -d "$W"/enveng/lib/python3.12/site-packages/nvidia/cu13)
    export CUDA_HOME="$NV" PATH="$NV/bin:$W/enveng/bin:$PATH"
    export GLYD_CUBLAS="$(ls "$NV"/lib/libcublas.so.13 2> /dev/null | head -1)"
    command -v cargo > /dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > "$R/rustup.out" 2>&1
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    case "$GPU" in a100) ARCHS="80" ;; h100) ARCHS="90a" ;; *) ARCHS="80 86 89 90a" ;; esac
    export GLYD_ENGINE_ARCHS="$ARCHS"
    log "building the engine (architectures: $ARCHS), and the model downloading meanwhile"
    (cd "$REPO" && CARGO_TARGET_DIR="$W/target" cargo build --release -p glyd-engine > "$R/build.out" 2>&1; echo "build exit $?" >> "$R/build.out") &
    BUILD=$!
    if [ -z "${GLYD_GPU_LIB:-}" ] || [ ! -f "${GLYD_GPU_LIB:-/none}" ]; then
        log "no GLYD_GPU_LIB: building the library (gpu/build_lib.sh)"
        (bash "$REPO/gpu/build_lib.sh" "$W/lib" > "$R/libbuild.out" 2>&1) &
        LIB=$!
    fi
    export HF_HOME="$W/hf" HF_HUB_ENABLE_HF_TRANSFER=0
    ("$W/enveng/bin/hf" download "$MODEL" > "$R/download.out" 2>&1; echo "download exit $?" >> "$R/download.out") &
    DL=$!
    wait $BUILD
    tail -1 "$R/build.out" | grep -q "exit 0" || fail "the engine's build failed (see build.out)"
    [ -n "${LIB:-}" ] && { wait $LIB; export GLYD_GPU_LIB="$(ls "$W"/lib/libglyd_gpu_cuda13.so 2> /dev/null)"; }
    wait $DL
    B="$W/target/release/glyd-engine"
else
    B="${ENGINE_BIN:?SKIP_SETUP=1 needs ENGINE_BIN}"
    export HF_HOME="${HF_HOME:-$HOME/hf}"
fi
[ -f "${GLYD_GPU_LIB:-/none}" ] || fail "no GLYD_GPU_LIB (the library the engine loads)"
[ -x "$B" ] || fail "no engine at $B"
export GLYD_GPU_LIB GLYD_CUBLAS
export GLYD_CACHE="$W/glyd-cache"
log "ready: engine $B, library $GLYD_GPU_LIB, cuBLAS ${GLYD_CUBLAS:-none}"

step doctor 120 "$B" doctor
# the packed model in the layout this GPU takes, then in the other one
step pack-default 900 "$B" prepare "$MODEL"
PD=$(tail -1 "$R/pack-default.out")
[ -f "$PD/glyd.json" ] || fail "no packed model ($PD): see pack-default.out"
case "$PD" in *mma12) L1=mma12; L2=mma ;; *) L1=mma; L2=mma12 ;; esac
P2="$W/packed-$L2"
step "pack-$L2" 900 "$B" pack "$MODEL" "$P2" --layout "$L2"
step kvcal 900 "$B" kvcal "$PD" "$R/kvtab-8192.safetensors" --positions 8192
cp "$R/kvtab-8192.safetensors" "$PD/kv-tables.safetensors" && cp "$R/kvtab-8192.safetensors" "$P2/kv-tables.safetensors"
REF="$REPO/glyd-engine/tests/fixtures/ref-8b-glyd1.json"

# the layouts in turn: tokens and scores with cuBLAS (and, in the GPU's own layout, with the engine's own products), then prefill and decode by the three ways of multiplying a long prompt
modes="blas own lib"
[ -n "${GLYD_CUBLAS:-}" ] || modes="own lib"
for pair in "$L1:$PD" "$L2:$P2"; do
    L=${pair%%:*}
    D=${pair#*:}
    first=${modes%% *}
    if [ -f "$REF" ]; then
        step "check-$L-$first" 600 env GLYD_ENGINE_GEMM="$first" "$B" check "$D" "$REF" "$R/tokens-$L-$first.json" --new 64 --ctx 4096
        step "score-$L-$first" 300 env GLYD_ENGINE_GEMM="$first" "$B" score "$D" "$REF" "$R/score-$L-$first.json" --ctx 4096
        if [ "$L" = "$L1" ] && [ "$first" != own ]; then
            step "check-$L-own" 600 env GLYD_ENGINE_GEMM=own "$B" check "$D" "$REF" "$R/tokens-$L-own.json" --new 64 --ctx 4096
            step "score-$L-own" 300 env GLYD_ENGINE_GEMM=own "$B" score "$D" "$REF" "$R/score-$L-own.json" --ctx 4096
        fi
    fi
    for m in $modes; do
        step "bench-$L-$m" 300 env GLYD_ENGINE_GEMM="$m" "$B" bench "$D" --ctx 4096 --prompt 1024,3500 --new 64 --reps 3
    done
done
# the engine's own downloader, packer and tables on a small model, from nothing
step first-use 600 env HF_HOME="$W/hf-fresh" GLYD_CACHE="$W/glyd-cache-fresh" "$B" run Qwen/Qwen3-0.6B --prompt "What is the capital of France? Answer in one word." --no-think --context 4096
step first-use-again 120 env HF_HOME="$W/hf-fresh" GLYD_CACHE="$W/glyd-cache-fresh" "$B" run Qwen/Qwen3-0.6B --prompt "What is the capital of France? Answer in one word." --no-think --context 4096

log "summary"
python3 "$REPO/glyd-engine/tools/engine_job_summary.py" "$R" "$REF" >> "$R/summary.txt" 2>&1
log "done"
touch "$R/DONE"
