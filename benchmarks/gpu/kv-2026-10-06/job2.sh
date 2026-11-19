#!/usr/bin/env bash
# kv-2026, second run (one RTX 4090): what the first left open.
#   1. kv lossless against vLLM's own cache at the same block size (704 tokens: kv lossless's for Qwen3.5-9B; a hybrid model's prefill steps and prefix hits follow the block size, and the first run
#      compared vLLM's default, 528): prompt logprobs bit for bit, prefix hits, greedy agreement; vLLM's own cache twice, for its own run-to-run spread.
#   2. the kernel checks with main's library and the branch's, twice each: whether the escape pool's high-water check that failed once in the first run fails with main's library too.
# In ~: this, kv_src.tar (as job.sh's) and the first run's tables (kvtab-*.npz, *.log). Results in ~/results; DONE at the end.
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
build branch "$W/src/gpu/glyd_gpu.cu" & B1=$!
build main "$W/src/main/glyd_gpu.cu" & B2=$!
wait $B1; r1=$?; wait $B2; r2=$?
note "libraries: branch exit $r1, main exit $r2"
LIB=$W/lib/branch/libglyd_gpu_cuda$MAJOR.so; LIBMAIN=$W/lib/main/libglyd_gpu_cuda$MAJOR.so
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB GLYD_CACHE=$R/glydcache VLLM_CACHE_ROOT=$W/vllmcache
cd "$W/src"
for i in 1 2; do
    GLYD_GPU_LIB=$LIBMAIN step "kernels-main-$i" 600 "$PY" gpu/vllm/check_vllm.py --kv --kernels-only --out "$R/kernels-main-$i"
    note "main's library, run $i: $(grep -c '^PASS' "$R/kernels-main-$i.out") pass; $(grep '^FAIL' "$R/kernels-main-$i.out" | cut -c1-160 | tr '\n' '|')"
    step "kernels-branch-$i" 600 "$PY" gpu/vllm/check_vllm.py --kv --kernels-only --out "$R/kernels-branch-$i"
    note "the branch's library, run $i: $(grep -c '^PASS' "$R/kernels-branch-$i.out") pass; $(grep '^FAIL' "$R/kernels-branch-$i.out" | cut -c1-160 | tr '\n' '|')"
done
wait $DL; MD=$(tail -1 "$W/model.dir" 2> /dev/null); [ -d "$MD" ] || { note "the model did not download"; MD=$MODEL; }
H="$PY gpu/vllm/kv_hybrid.py"
step hybrid-stock704 1200 $H run "$R/hybrid-stock704.json" --kv off --block-size 704 --model "$MD"
step hybrid-kv 1500 $H run "$R/hybrid-kv.json" --kv lossless --model "$MD"
step hybrid-stock704b 1200 $H run "$R/hybrid-stock704b.json" --kv off --block-size 704 --model "$MD"
for pair in "stock704 kv" "stock704 stock704b"; do set -- $pair
    "$PY" gpu/vllm/kv_hybrid.py compare "$R/hybrid-$1.json" "$R/hybrid-$2.json" > "$R/compare-$1-$2.txt" 2>&1; note "compare $1 $2: exit $?"; cat "$R/compare-$1-$2.txt"
done
ls "$R/glydcache/kv/" > "$R/tables.txt"
note "end"
