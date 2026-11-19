#!/usr/bin/env bash
# kv-2026: the lossless KV cache at head size 256 and on hybrid models, on one RTX 4090 (Ada), Qwen/Qwen3.5-9B (2026: Gated DeltaNet beside full attention, head 256), vLLM 0.30.0.
# In ~: this and kv_src.tar (the branch's tree, `git archive`, with main/glyd_gpu.cu and main/glyd_gpu.h: main's library, for head size 128's bits). Results in ~/results; DONE at the end.
#   1. the library of the branch and of main, for this GPU; the plugin's tests (no GPU needed, and the library's page sizes);
#   2. the KV kernels' checks (head size 128's as before, and head size 256's), head size 128's outputs of both libraries hashed and compared, the decode kernel at head size 256 against FlashAttention 2;
#   3. Qwen3.5-9B with Glyd's weights (fraction 1, the text model only) in vLLM: vLLM's own KV cache, then kv lossless (its tables made on the first start, timed), prompt logprobs bit for bit, prefix hits,
#      greedy agreement, the KV cache's tokens; kv lossless with verify on a small cache under churn (every write checked; blocks passing between the cache groups);
#   4. vllm serve, vLLM's cache and kv lossless: vllm bench serve at 1, 8, 32 and 64 requests at once and all at once (random 1,024 in, 256 out), and the KV cache vLLM reports.
set -u
R=$HOME/results; W=$HOME/w; mkdir -p "$R" "$W/tmp"
T0=$(date +%s); END=${END:-3300}
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
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

{ uname -srvmo; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv; lscpu | grep -E "Model name"; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt"
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/kv_src.tar" || { note "no kv_src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT" 2> /dev/null
MODEL=${MODEL:-Qwen/Qwen3.5-9B}

# ---- the environment, the model's download beside it --------------------------------------------------------------------------------------------------------------------------
UV=$(command -v uv); [ -z "$UV" ] && { mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"; UV=$W/uv/uv; }
( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet && d=$(date +%s) &&
  timeout 1500 "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))" "$MODEL" > "$W/model.dir" 2> "$R/dl.err" &&
  echo "the model: $(du -sbL "$(tail -1 "$W/model.dir")" | cut -f1) bytes in $(( $(date +%s) - d )) s" >> "$R/steps.txt" ) > /dev/null 2>&1 &
DL=$!
t=$(date +%s)
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" "glyd-gpu[vllm]==0.28.0"; } > "$R/env-install.txt" 2>&1 || { note "vLLM did not install (env-install.txt)"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/env-nvcc.txt" 2>&1 || { note "nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
"$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', m.version('transformers'), '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1
echo "nvcc: $(nvcc --version | tail -1)" >> "$R/env.txt"; cat "$R/env.txt"
note "environment: $(( $(date +%s) - t )) s"

# ---- the libraries: the branch's and main's, for this GPU -----------------------------------------------------------------------------------------------------------------------
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
   -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
build() { mkdir -p "$W/lib/$1"; nvcc "${F[@]}" -Xptxas -v -c -o "$W/lib/$1/glyd_gpu.o" "$2" > "$R/build-$1.txt" 2>&1 &&
          nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/lib/$1/libglyd_gpu_cuda$MAJOR.so" "$W/lib/$1/glyd_gpu.o" >> "$R/build-$1.txt" 2>&1; }
t=$(date +%s)
build branch "$W/src/gpu/glyd_gpu.cu" & B1=$!
build main "$W/src/main/glyd_gpu.cu" & B2=$!
wait $B1; r1=$?; wait $B2; r2=$?
note "libraries: branch exit $r1, main exit $r2 ($(( $(date +%s) - t )) s)"
grep -E "kvc.*(registers|spill)|Function properties for .*kvc" "$R/build-branch.txt" | head -80 > "$R/regs-branch.txt"
LIB=$W/lib/branch/libglyd_gpu_cuda$MAJOR.so; LIBMAIN=$W/lib/main/libglyd_gpu_cuda$MAJOR.so
[ -f "$LIB" ] || { note "the branch's library did not build (build-branch.txt)"; exit 1; }
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB GLYD_CACHE=$R/glydcache VLLM_CACHE_ROOT=$W/vllmcache

# ---- the plugin's tests, the kernels' checks, head size 128's bits, the decode kernel's time --------------------------------------------------------------------------------------
cd "$W/src/glyd-gpu-py/tests" && step tests 600 env GLYD_REQUIRE_VLLM=1 "$PY" test_vllm.py; cd "$W/src"
step kernels 900 "$PY" gpu/vllm/check_vllm.py --kv --kernels-only --out "$R/kernels"
grep -c "^PASS" "$R/kernels.out" > "$R/kernels-pass.txt"; grep "^FAIL" "$R/kernels.out" > "$R/kernels-fail.txt"
note "kernel checks: $(cat "$R/kernels-pass.txt") pass, $(grep -c . "$R/kernels-fail.txt") fail"
if [ -f "$LIBMAIN" ]; then
    GLYD_GPU_LIB=$LIBMAIN step bits-main 300 "$PY" gpu/vllm/kv_bits.py "$R/bits-main.json"
    step bits-branch 300 "$PY" gpu/vllm/kv_bits.py "$R/bits-branch.json"
    "$PY" gpu/vllm/kv_bits.py compare "$R/bits-main.json" "$R/bits-branch.json" > "$R/bits.txt" 2>&1; note "$(cat "$R/bits.txt")"
fi
step bench256 900 "$PY" gpu/vllm/kv_decode_bench.py --head 256 --kv-heads 4 --group 4 --layers 8 --calls 40 --repeat 3 --shapes 1x1024,1x8192,8x1024,8x4096,32x1024,64x1024,32x4096 --out "$R/bench256.json"

# ---- Qwen3.5-9B in vLLM: vLLM's cache, kv lossless, kv lossless with verify under churn ------------------------------------------------------------------------------------------
wait $DL; MD=$(tail -1 "$W/model.dir" 2> /dev/null); [ -d "$MD" ] || { note "the model did not download (dl.err)"; MD=$MODEL; }
H="$PY gpu/vllm/kv_hybrid.py"
step hybrid-stock 1200 $H run "$R/hybrid-stock.json" --kv off --model "$MD"
step hybrid-kv 1800 $H run "$R/hybrid-kv.json" --kv lossless --model "$MD"
ls "$R/glydcache/kv/" > "$R/tables.txt" 2>&1; grep -h "made in\|kvcal" "$R/hybrid-kv.err" "$R"/glydcache/kv/*.log 2> /dev/null | tail -12 >> "$R/tables.txt"
"$PY" gpu/vllm/kv_hybrid.py compare "$R/hybrid-stock.json" "$R/hybrid-kv.json" > "$R/hybrid-compare.txt" 2>&1; note "compare: exit $?"; cat "$R/hybrid-compare.txt"
step hybrid-verify 1200 $H run "$R/hybrid-verify.json" --kv lossless --verify --model "$MD" --blocks 48 --seqs 16 --churn 48

# ---- vllm serve: vLLM's cache against kv lossless --------------------------------------------------------------------------------------------------------------------------------
serve_bench() {  # NAME KV
    local name=$1 kv=$2 port=8100 d="$R/serve-$1"; mkdir -p "$d"
    [ "$(left)" -lt 400 ] && { note "skipped serving $name: $(left) s left"; return; }
    note "serve $name (kv $kv)"
    GLYD_KV=$kv vllm serve "$MD" --served-model-name "$MODEL" --quantization glyd --language-model-only --max-model-len 8192 --max-num-seqs 128 --gpu-memory-utilization 0.9 --port $port > "$d/serve.txt" 2>&1 &
    local sv=$! t=$SECONDS
    for _ in $(seq 1 180); do curl -sf -m 3 "localhost:$port/v1/models" > /dev/null && break; kill -0 $sv 2> /dev/null || break; sleep 5; done
    if ! curl -sf -m 3 "localhost:$port/v1/models" > /dev/null; then note "$name did not start (serve-$name/serve.txt)"; kill $sv 2> /dev/null; wait $sv 2> /dev/null; return; fi
    note "$name up in $((SECONDS - t)) s, GPU memory $(gpu_used) MiB"
    grep -h -o -E "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|Model loading took [0-9.]+ GiB|Setting attention block size to [0-9]+ tokens|glyd[ :].*KV cache.*" "$d/serve.txt" | sort -u > "$d/kv.txt"
    local i=0
    for c in warm 1 8 32 64 all; do
        local n=$([ "$c" = warm ] && echo 16 || ([ "$c" = all ] && echo 256 || echo $(( c < 8 ? 8 : 2 * c ))))
        local conc=$([ "$c" = all ] || [ "$c" = warm ] && echo "" || echo "--max-concurrency $c")
        [ "$(left)" -lt 150 ] && { note "skipped $name at $c: $(left) s left"; break; }
        nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits -l 1 > "$d/smi-$c.csv" &
        local smi=$!
        timeout 600 vllm bench serve --backend vllm --model "$MODEL" --tokenizer "$MD" --port $port --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos --num-prompts "$n" --request-rate inf $conc \
            --seed $((100 + i)) --save-result --result-dir "$d" --result-filename "c$c.json" --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 > "$d/bench-$c.txt" 2>&1
        note "$name at $c: exit $?"; kill $smi 2> /dev/null; i=$((i + 1))
    done
    kill $sv; wait $sv 2> /dev/null; sleep 5
}
serve_bench stock off
serve_bench kv lossless

"$PY" - "$R" > "$R/summary-serve.txt" 2>&1 <<'EOF'
import json, os, sys
R = sys.argv[1]
print("| server | load | requests/s | output tok/s | TTFT median ms | TPOT median ms |")
print("| :-- | :-- | --: | --: | --: | --: |")
for name in ("stock", "kv"):
    for c in ("1", "8", "32", "64", "all"):
        p = os.path.join(R, f"serve-{name}", f"c{c}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"| {name} | {c} | {d['request_throughput']:.3f} | {d['output_throughput']:.1f} | {d['median_ttft_ms']:.1f} | {d['median_tpot_ms']:.2f} |")
for name in ("stock", "kv"):
    p = os.path.join(R, f"serve-{name}", "kv.txt")
    print(name, ":", open(p).read().replace("\n", "; ") if os.path.exists(p) else "no server")
EOF
cat "$R/summary-serve.txt"
note "end"
