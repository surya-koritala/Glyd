#!/usr/bin/env bash
# glyd-engine runs Qwen3.5 (2026) on one GPU, unattended (glyd-engine/docs/design/plans/2026-10-06-engine-qwen35.md, tasks 10 to 12, its GPU runs batched in one):
#   1. Qwen3.5's kernels against the CPU reference (glyd-engine kcheck35);
#   2. Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-9B: greedy tokens of 16 short prompts and 3 long ones (the tree's README.md's first 1,030, 2,100 and 3,500 tokens; 64 tokens each) against
#      vLLM 0.30.0 bf16 (tools/ref_vllm.py: vLLM as one batch and one prompt at a time, the floor), and the 0.8B's
#      prompts in chunks of 100 tokens (the long ones in several calls, the state carried between them);
#   3. Qwen/Qwen3.5-9B, one user: a prompt of 1,024 random tokens and 256 more, the engine against vLLM 0.30.0 bf16 and vLLM with Glyd (tools/ref_speed.py), and GPU memory.
# The library and the engine are built here from the tarball for this GPU alone; vLLM 0.30.0 from PyPI, with Glyd's plugin from the tarball (glyd-gpu 0.28.0's entry point, the
# tarball's package by PYTHONPATH, its library by GLYD_GPU_LIB). nvidia-smi's memory and clocks each second (smi.csv).
#
# Since R3b (#137) the model runs on the description executor; the hand-written engine's layer dump and its token-by-token prompt mode went with it.
#
# MINUTES: 100 at most (setup and downloads about 15; the checks about 20; the speed runs about 20)
#
# In ~: this script and engine35_src.tar (git archive of the tree, with COMMIT). Results in ~/results: summary.txt (made last), one .out a step, steps.txt; DONE when it ends.
set -uo pipefail
R=${R:-$HOME/results}
W=${W:-$HOME/e35}
MAX_MIN=${MAX_MIN:-100}
mkdir -p "$R" "$W/tmp"
T0=$(date +%s)
log() { echo "$(date -u +%T) (+$(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( T0 + MAX_MIN * 60 - $(date +%s) )); }
fin() { kill "${SMI:-0}" 2> /dev/null; touch "$R/DONE"; }
trap fin EXIT
fail() { log "FAILED: $*"; echo "$*" > "$R/FAILED"; exit 1; }
# a step: its name, the seconds it may take, the command (output in NAME.out, its start and end in steps.txt)
step() {
    local name=$1 secs=$2 l
    shift 2
    l=$(left)
    if [ "$l" -lt 60 ]; then log "skipped $name: $l s left"; return 0; fi
    [ "$secs" -gt $((l - 30)) ] && secs=$((l - 30))
    log "start $name (at most $secs s)"
    timeout "$secs" "$@" > "$R/$name.out" 2>&1
    local rc=$?
    log "end $name: exit $rc"
    return $rc
}
{ date -u; uname -a; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm,power.limit --format=csv; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -l 1 > "$R/smi.csv" 2>&1 &
SMI=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/engine35_src.tar" || fail "no engine35_src.tar"
cp "$W/src/COMMIT" "$R/COMMIT" 2> /dev/null
SRC=$W/src
E=$SRC/glyd-engine
log "GPU sm_$ARCH, tree $(cat "$SRC/COMMIT" 2> /dev/null)"

log "setup: uv 0.12.21, the venv (vLLM 0.30.0, glyd-gpu 0.28.0's plugin entry, accelerate, nvcc), Rust 1.98.1, the downloads"
mkdir -p "$W/uv"
curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/download/0.12.21/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" || fail "no uv"
UV=$W/uv/uv
( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet &&
  for m in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B; do
      "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model']))" "$m" || exit 1
  done ) > "$R/download.out" 2>&1 &
DL=$!
( "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" "glyd-gpu[vllm]==0.28.0" accelerate &&
  "$UV" pip install --python "$W/venv/bin/python" "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-runtime==13.0.*" "nvidia-cublas==13.0.*" ) > "$R/venv.out" 2>&1 &
VENV=$!
( curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1 ) > "$R/rustup.out" 2>&1 &
RUST=$!
wait $VENV || fail "the venv did not install (venv.out)"
PY=$W/venv/bin/python
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$CU/bin:$W/venv/bin:$HOME/.cargo/bin:$PATH
export GLYD_CUBLAS=$(ls "$CU"/lib/libcublas.so.13 2> /dev/null | head -1)
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', m.version('transformers'), '| glyd-gpu', m.version('glyd-gpu'), '|', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
echo "nvcc: $(nvcc --version | tail -1)" >> "$R/env.txt"
log "env: $(head -1 "$R/env.txt")"

# the library, for this GPU alone (the flags of gpu/build_lib.sh's development build)
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
LIB=$W/lib/libglyd_gpu_cuda13.so
mkdir -p "$W/lib"
( nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$SRC/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o" ) > "$R/libbuild.out" 2>&1 &
LB=$!
wait $RUST
# shellcheck disable=SC1091
source "$HOME/.cargo/env" 2> /dev/null
( cd "$SRC" && GLYD_ENGINE_ARCHS="$ARCH" CARGO_TARGET_DIR="$W/target" cargo build --release -p glyd-engine ) > "$R/build.out" 2>&1 &
EB=$!
wait $LB || fail "the library did not build (libbuild.out)"
wait $EB || fail "the engine did not build (build.out)"
B=$W/target/release/glyd-engine
export GLYD_GPU_LIB=$LIB
log "built: the library and the engine"
wait $DL || fail "the downloads (download.out)"
D08=$(sed -n 1p <(grep "^/" "$R/download.out"))
D9=$(sed -n 2p <(grep "^/" "$R/download.out"))
[ -f "$D08/config.json" ] && [ -f "$D9/config.json" ] || fail "the models are not where download.out says"
log "models: $D08, $D9"

# 1. the kernels
step kcheck35 300 "$B" kcheck35

# 2 and 3. the models against vLLM bf16 (the long prompts from the tree's README.md), and the 0.8B's hidden states against transformers'
KW9='{"language_model_only": true, "max_num_seqs": 16, "gpu_memory_utilization": 0.94}'
TXT=$SRC/README.md
step ref-0.8b 900 env VLLM_ENABLE_V1_MULTIPROCESSING=0 "$PY" "$E/tools/ref_vllm.py" "$D08" "$R/ref-0.8b.json" bf16 "$TXT"
step ref-9b 1200 env VLLM_ENABLE_V1_MULTIPROCESSING=0 REF_KW="$KW9" "$PY" "$E/tools/ref_vllm.py" "$D9" "$R/ref-9b.json" bf16 "$TXT" || {
    log "vLLM bf16 did not run the 9B with CUDA graphs on this GPU: eager"
    step ref-9b-eager 1200 env VLLM_ENABLE_V1_MULTIPROCESSING=0 REF_KW='{"language_model_only": true, "max_num_seqs": 16, "gpu_memory_utilization": 0.94, "enforce_eager": true}' "$PY" "$E/tools/ref_vllm.py" "$D9" "$R/ref-9b.json" bf16 "$TXT"
}
for m in 0.8b 9b; do
    D=$D08; [ "$m" = 9b ] && D=$D9
    [ -f "$R/ref-$m.json" ] || { log "no vLLM reference for $m"; continue; }
    step "check-$m" 900 "$B" check "$D" "$R/ref-$m.json" "$R/tokens-$m.json" --new 64 --ctx 4096
    step "compare-$m" 60 "$PY" "$E/tools/compare.py" "$R/tokens-$m.json" "$R/ref-$m.json"
done
if [ -f "$R/ref-0.8b.json" ]; then
    step check-0.8b-chunk100 900 "$B" check "$D08" "$R/ref-0.8b.json" "$R/tokens-0.8b-chunk100.json" --new 64 --ctx 4096 --chunk 100
    step compare-0.8b-chunk100 60 "$PY" "$E/tools/compare.py" "$R/tokens-0.8b-chunk100.json" "$R/ref-0.8b.json"
fi

step generate-9b 300 "$B" generate "$D9" "The capital of France is" --new 32 --ctx 4096

# 4. one user on the 9B: the engine (prompt's linear attention in chunks, and token by token), vLLM bf16, vLLM with Glyd
step bench-9b 900 "$B" bench "$D9" --prompt 1024 --new 256 --reps 3 --ctx 4096 --profile
step speed-9b-vllm 1200 env VLLM_ENABLE_V1_MULTIPROCESSING=0 REF_KW="$KW9" "$PY" "$E/tools/ref_speed.py" "$D9" "$R/speed-9b-vllm.json" bf16 1024 256 3 || {
    step speed-9b-vllm-eager 1200 env VLLM_ENABLE_V1_MULTIPROCESSING=0 REF_KW='{"language_model_only": true, "max_num_seqs": 16, "gpu_memory_utilization": 0.94, "enforce_eager": true}' "$PY" "$E/tools/ref_speed.py" "$D9" "$R/speed-9b-vllm.json" bf16 1024 256 3
}
step speed-9b-glyd 1200 env VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH="$SRC/glyd-gpu-py" REF_KW="$KW9" "$PY" "$E/tools/ref_speed.py" "$D9" "$R/speed-9b-glyd.json" glyd1 1024 256 3

log "summary"
{
    echo "glyd-engine, Qwen3.5 (2026), on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); tree $(cat "$SRC/COMMIT" 2> /dev/null); $(head -1 "$R/env.txt")"
    echo "--- kernels against the CPU (kcheck35)"; grep -E "^(ok|FAIL|all|FAILED)" "$R/kcheck35.out"
    for f in compare-0.8b compare-0.8b-chunk100 compare-9b; do echo "--- $f"; cat "$R/$f.out" 2> /dev/null; done
    echo "--- a completion (9B)"; tail -2 "$R/generate-9b.out" 2> /dev/null
    echo "--- one user, Qwen3.5-9B, 1,024 prompt tokens, 256 new"
    for f in bench-9b; do echo "$f: $(grep '^{' "$R/$f.out" | tail -1)"; grep "loaded in" "$R/$f.out" | head -1; done
    echo "--- the engine's decode step by kind of work (9B)"; sed -n '/decode step.s GPU time/,/total/p' "$R/bench-9b.out"
    for f in speed-9b-vllm speed-9b-glyd; do echo "$f: $(cat "$R/$f.json" 2> /dev/null)"; grep -h "Model loading took\|GPU KV cache size" "$R/$f.out" "$R/$f-eager.out" 2> /dev/null | head -2; done
    echo "--- GPU memory used at most during each step (nvidia-smi, MiB)"
    python3 - "$R" << 'EOF'
import csv, datetime, re, sys
R = sys.argv[1]
smi = []
for row in csv.reader(open(f"{R}/smi.csv")):
    try:
        smi.append((datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), int(row[1])))
    except Exception:
        pass
start = {}
day = smi[0][0].date() if smi else datetime.date.today()
for line in open(f"{R}/steps.txt"):
    m = re.match(r"(\d\d:\d\d:\d\d) \(\+\d+ min\) (start|end) (\S+)", line)
    if not m:
        continue
    t = datetime.datetime.combine(day, datetime.time.fromisoformat(m.group(1)))
    if m.group(2) == "start":
        start[m.group(3)] = t
    elif m.group(3).rstrip(":") in start:
        name = m.group(3).rstrip(":")
        peak = max((mib for ts, mib in smi if start[name] <= ts <= t), default=None)
        print(f"{name}: {peak}")
EOF
} > "$R/summary.txt" 2>&1
log "done"
