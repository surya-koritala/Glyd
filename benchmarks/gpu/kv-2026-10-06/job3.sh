#!/usr/bin/env bash
# kv-2026, third run (one RTX 4090): the KV cache's tokens at the same memory, and vLLM's own spread for the greedy agreement.
#   1. the KV cache vLLM makes in the same bytes (--kv-cache-memory-bytes 6 GiB: the first runs' starts found pools of 5.4 to 7.2 GB at --gpu-memory-utilization 0.9, by the start), vLLM's cache (its
#      default block) and kv lossless, at --max-model-len 8,192 and 32,768 (kv lossless's tables for 32,768 made on its first start, timed);
#   2. vLLM's own cache at kv lossless's block (704), its requests in a batch and each alone (its own spread when only the batch changes), and kv lossless in a batch: the greedy tokens' agreement.
# In ~: this, kv_src.tar (as job.sh's) and the first run's tables for 8,192 (kvtab-*.npz, *.log). Results in ~/results; DONE at the end.
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R" "$W/tmp"
T0=$(date +%s); END=${END:-2100}
exec > >(tee -a "$R/job.log") 2>&1
trap 'touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
note() { echo "$(date -u +%T) (+$(el) s) $*" | tee -a "$R/steps.txt"; }
left() { echo $(( END - $(el) )); }
step() { local name=$1 secs=$2; shift 2; local l; l=$(left)
    [ "$l" -lt 90 ] && { note "skipped $name: $l s left"; return 99; }
    [ "$secs" -gt $((l - 60)) ] && secs=$((l - 60))
    note "$name (at most $secs s)"; local t=$SECONDS
    timeout -k 30 "$secs" "$@" > "$R/$name.out" 2> "$R/$name.err"; local rc=$?
    note "$name: exit $rc after $((SECONDS - t)) s"; return $rc; }

{ uname -srvmo; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv; nproc; date -u; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/kv_src.tar" || { note "no kv_src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT" 2> /dev/null
mkdir -p "$R/glydcache/kv" && cp "$HOME"/kvtab-* "$R/glydcache/kv/" 2> /dev/null
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet &&
  timeout 1500 "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))" "$MODEL" > "$W/model.dir" 2> "$R/dl.err" ) > /dev/null 2>&1 &
DL=$!
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" "glyd-gpu[vllm]==0.28.0"; } > "$R/env-install.txt" 2>&1 || { note "vLLM did not install"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/env-nvcc.txt" 2>&1 || { note "nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
note "environment"
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
build() { mkdir -p "$W/lib/$1"; nvcc "${F[@]}" -c -o "$W/lib/$1/glyd_gpu.o" "$2" > "$R/build-$1.txt" 2>&1 &&
          nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/lib/$1/libglyd_gpu_cuda$MAJOR.so" "$W/lib/$1/glyd_gpu.o" >> "$R/build-$1.txt" 2>&1; }
build branch "$W/src/gpu/glyd_gpu.cu"; note "library: exit $?"
LIB=$W/lib/branch/libglyd_gpu_cuda$MAJOR.so
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB GLYD_CACHE=$R/glydcache VLLM_CACHE_ROOT=$W/vllmcache
cd "$W/src"
wait $DL; MD=$(tail -1 "$W/model.dir" 2> /dev/null); [ -d "$MD" ] || { note "the model did not download"; MD=$MODEL; }
H="$PY gpu/vllm/kv_hybrid.py"
MEM=$((6 * 1024 * 1024 * 1024))
for mml in 8192 32768; do
    step "cap-stock-$mml" 900 $H run "$R/cap-stock-$mml.json" --kv off --mml $mml --kv-memory $MEM --capacity-only --model "$MD"
    step "cap-kv-$mml" 1500 $H run "$R/cap-kv-$mml.json" --kv lossless --mml $mml --kv-memory $MEM --capacity-only --model "$MD"
done
step agree-stock704 1200 $H run "$R/agree-stock704.json" --kv off --block-size 704 --model "$MD"
step agree-stock704-alone 1200 $H run "$R/agree-stock704-alone.json" --kv off --block-size 704 --alone --model "$MD"
step agree-kv 1200 $H run "$R/agree-kv.json" --kv lossless --model "$MD"
for pair in "stock704 kv" "stock704 stock704-alone"; do set -- $pair
    "$PY" gpu/vllm/kv_hybrid.py compare "$R/agree-$1.json" "$R/agree-$2.json" > "$R/compare-$1-$2.txt" 2>&1; note "compare $1 $2: exit $?"; cat "$R/compare-$1-$2.txt"
done
"$PY" - "$R" > "$R/capacity.txt" 2>&1 <<'EOF2'
import json, os, sys
R = sys.argv[1]
for mml in (8192, 32768):
    try:
        s, k = (json.load(open(os.path.join(R, f"cap-{n}-{mml}.json")))["cache"] for n in ("stock", "kv"))
    except Exception as e:
        print(mml, "missing:", e); continue
    print(f"--max-model-len {mml:,}, 6 GiB of KV cache: vLLM's {s['kv_cache_size_tokens']:,} tokens ({s['num_gpu_blocks']} blocks of {s['block_size']}, {s['max_concurrency']:.2f} requests of {mml:,} at once); "
          f"kv lossless {k['kv_cache_size_tokens']:,} ({k['num_gpu_blocks']} blocks of {k['block_size']}, {k['max_concurrency']:.2f}; its reserves {k.get('reserves_bytes', 0) / 2**20:.1f} MiB apart): "
          f"{k['kv_cache_size_tokens'] / s['kv_cache_size_tokens']:.3f}x the tokens")
EOF2
cat "$R/capacity.txt"; ls "$R/glydcache/kv/" > "$R/tables.txt"; grep -h "made in\|kvcal:" "$R"/glydcache/kv/*.log >> "$R/tables.txt" 2> /dev/null
note "end"
