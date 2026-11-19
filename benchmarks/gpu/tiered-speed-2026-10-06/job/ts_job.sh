#!/usr/bin/env bash
# ts_job.sh: the tiered layout (mma) against the 12-bit one (mma12) and bf16 on one Runpod GPU, per matrix of 2026 models, unattended.
# In ~: this and ts_src.tar (the tree's gpu/, glyd-gpu-py/, this job/ and COMMIT). Steps:
#   env      uv venv with vLLM 0.30.0 (its PyTorch), nvcc and cuobjdump of PyTorch's CUDA
#   build    the tree's library for this GPU alone (sm_XX), then its SASS: registers and instruction mix of the dense kernels (sass.txt)
#   checks   the library's self-test (glyd_gpu_dev.py) and check_step.py on this GPU; capi: check_capi.py (against PyTorch's JIT build)
#   fetch    one full-attention and one linear-attention layer of Qwen3.5-9B and -27B, a sliding and a global layer of Gemma-4-12B and -31B,
#            Qwen3.5-9B's output layer, Qwen3.5-35B-A3B's experts of layer 3: by byte range (fetch_tensors.py)
#   bench    layouts.py on each model's matrices (1-64 tokens, prompts 128-2048), moe_layer.py on the experts (both layouts, vLLM's bf16)
#   vllm     check_vllm.py --quick on Qwen3.5-9B (both layouts against bf16, exact mode bit for bit), the tree's package and library
#   serve    vllm serve Qwen3.5-9B (text only) a server a mode of TS_MODES ("bf16 mma12 mma mmacc mma12b": bf16, Glyd forced to each layout,
#            mmacc the tiered one with the option cc12, the 12-bit layout again last for the drift), vllm bench serve at 1, 8, 16, 32 and 64 users (1,024 tokens in, 256 out, a seed a pass,
#            the same passes for every mode), after a warm-up of 64 prompts
#   summary  summarize.py -> summary.txt
# results/DONE from the exit trap; results/ALLDONE when it ran to its end. TS_STEPS (default "build checks bench") picks steps
# (build is always made); TS_END (seconds, 3000) the hard stop; TS_VARIANTS: layouts.py's --variants (the tiered layout's products under them).
set -u
R=$HOME/results; W=$HOME/tsw
mkdir -p "$R/log" "$W/tmp"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
END=${TS_END:-3000}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
note() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; echo "   $*"; }
trap 'touch "$R/DONE"' EXIT
( sleep "$END"; echo "hard stop at +$(el) s" >> "$R/steps.txt"; pkill -TERM -P $$; kill -TERM $$ ) > /dev/null 2>&1 & WD=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache
want() { case " ${TS_STEPS:-build checks bench} " in *" $1 "*) return 0 ;; esac; return 1; }

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
MAXMHZ=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
echo "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute $CC (sm_$ARCH), top SM clock $MAXMHZ MHz" | tee "$R/gpu-short.txt"
cd "$W" && rm -rf src && mkdir src && tar -C src -xf "$HOME/ts_src.tar" || { echo "no ts_src.tar"; exit 1; }
cp src/COMMIT "$R/COMMIT"; cat src/COMMIT

step "env: uv, vLLM 0.30.0, nvcc, cuobjdump"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { echo "no uv"; exit 1; }
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0"; } > "$R/log/env-vllm.txt" 2>&1 || { note "FAIL: vLLM did not install"; tail -5 "$R/log/env-vllm.txt"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "FAIL: nvcc did not install"; exit 1; }
"$UV" pip install --python "$PY" "nvidia-cuda-cuobjdump==${MM%%.*}.*" > "$R/log/env-cuobjdump.txt" 2>&1 || note "no cuobjdump from pip (log/env-cuobjdump.txt)"
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
OBJDUMP=$(command -v cuobjdump || find "$W/venv" /usr/local/cuda*/bin -name cuobjdump -type f 2> /dev/null | head -1)
"$PY" -c "import importlib.metadata as md, torch; print('vllm', md.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability())" | tee "$R/env.txt"
echo "nvcc: $(nvcc --version | tail -1); cuobjdump: ${OBJDUMP:-none}" | tee -a "$R/env.txt"
note "env"

DLM=
if want vllm || want serve; then  # Qwen3.5-9B, beside the rest
  export HF_HOME=$W/hf HF_XET_HIGH_PERFORMANCE=1
  ( timeout 1500 "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model']))" > "$R/log/dl-model.txt" 2>&1; echo "exit $?" >> "$R/log/dl-model.txt" ) &
  DLM=$!
  "$UV" pip install --python "$PY" "glyd-gpu==0.28.0" > "$R/log/env-entry.txt" 2>&1 || note "FAIL: glyd-gpu's entry point did not install"  # (the plugin's entry point; the package itself from the tree, PYTHONPATH)
fi

FETCH=
if want bench; then  # the matrices, beside the build
  mkdir -p "$W/m"
  ( cd "$W/src/gpu"
    P=model.language_model.layers
    q35() { "$PY" fetch_tensors.py "$1" "$2" gate_up=$P.3.mlp.gate_proj.weight+$P.3.mlp.up_proj.weight down=$P.3.mlp.down_proj.weight \
      qkv=$P.3.self_attn.q_proj.weight+$P.3.self_attn.k_proj.weight+$P.3.self_attn.v_proj.weight o=$P.3.self_attn.o_proj.weight \
      gdn_in=$P.0.linear_attn.in_proj_qkv.weight+$P.0.linear_attn.in_proj_z.weight gdn_out=$P.0.linear_attn.out_proj.weight; }
    g4() { "$PY" fetch_tensors.py "$1" "$2" gate_up=$P.0.mlp.gate_proj.weight+$P.0.mlp.up_proj.weight down=$P.0.mlp.down_proj.weight \
      qkv=$P.0.self_attn.q_proj.weight+$P.0.self_attn.k_proj.weight+$P.0.self_attn.v_proj.weight o=$P.0.self_attn.o_proj.weight \
      qk_global=$P.5.self_attn.q_proj.weight+$P.5.self_attn.k_proj.weight o_global=$P.5.self_attn.o_proj.weight; }
    q35 Qwen/Qwen3.5-9B "$W/m/Qwen3.5-9B.safetensors" &
    q35 Qwen/Qwen3.5-27B "$W/m/Qwen3.5-27B.safetensors" &
    g4 google/gemma-4-12B-it "$W/m/gemma-4-12B-it.safetensors" &
    g4 google/gemma-4-31B-it "$W/m/gemma-4-31B-it.safetensors" &
    "$PY" fetch_tensors.py Qwen/Qwen3.5-9B "$W/m/head-Qwen3.5-9B.safetensors" lm_head=lm_head.weight &
    "$PY" fetch_tensors.py Qwen/Qwen3.5-35B-A3B "$W/m/experts-Qwen3.5-35B-A3B.safetensors" w13=$P.3.mlp.experts.gate_up_proj w2=$P.3.mlp.experts.down_proj &
    wait ) > "$R/log/fetch.txt" 2>&1 &
  FETCH=$!
fi

step "build: the library for sm_$ARCH"
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden -Xptxas -v)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so
mkdir -p "$W/lib"
{ time { nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; }; } > "$R/log/build.txt" 2>&1
[ -f "$LIB" ] || { note "FAIL: the library did not build (log/build.txt)"; tail -20 "$R/log/build.txt"; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/glyd-gpu-py
"$PY" -c "from glyd_gpu import kernels as g, _lib; g.lib(); print('C API', _lib._lib.glyd_gpu_api_version(), '| GPU code', _lib.gpu())" 2>&1 | tee "$R/log/load.txt"
note "build: $(grep real "$R/log/build.txt"); $(tail -1 "$R/log/load.txt")"

if [ -n "$OBJDUMP" ]; then  # the dense kernels' SASS: registers (ptxas, in the build log) and instruction mix (whole and by loop)
  mkdir -p "$W/sass" && "$OBJDUMP" -sass "$LIB" > "$W/sass/all.sass" 2> "$R/log/cuobjdump.txt"
  "$OBJDUMP" -res-usage "$LIB" > "$R/res-usage.txt" 2>&1
  awk -v d="$W/sass" '/Function :/ {n++; f=sprintf("%s/%03d.sass", d, n); print $3 > (d "/names.txt")} f {print > f}' "$W/sass/all.sass"
  { for pat in step_kernel mma_gemm_kernelI6Tiered mma_gemm_kernelI3Nib mma12_mid_kernel mma12_tma_kernel mma_gemm_big_kernelI6Tiered mma_gemm_big_kernelI3Nib mma_unpack_kernelI6Tiered mma_unpack_kernelI3Nib; do
      grep -n "$pat" "$W/sass/names.txt" | while IFS=: read -r k name; do
        echo "######## $name"; python3 "$W/src/gpu/experimental/tiered_fast/sassstat.py" "$(printf '%s/%03d.sass' "$W/sass" "$k")" | head -40; echo
      done
    done; } > "$R/sass.txt" 2>&1
  note "sass: $(grep -c '^########' "$R/sass.txt") kernels"
fi

cd "$W/src/gpu"
if want checks; then
  step "checks: the self-test, check_step.py"
  timeout 1500 "$PY" glyd_gpu_dev.py > "$R/check-selftest.txt" 2>&1; note "self-test exit $? ($(tail -1 "$R/check-selftest.txt" | cut -c1-160))"
  timeout 1500 "$PY" check_step.py > "$R/check-step.txt" 2>&1; note "check_step exit $? ($(tail -1 "$R/check-step.txt" | cut -c1-160))"
fi
if want capi; then  # (the library against PyTorch's JIT build of the same source, every output bit for bit, and against fp32)
  step "check_capi.py"
  "$UV" pip install --python "$PY" ninja > "$R/log/env-ninja.txt" 2>&1
  timeout 2400 "$PY" check_capi.py "$LIB" > "$R/check-capi.txt" 2>&1; note "check_capi exit $? ($(tail -1 "$R/check-capi.txt" | cut -c1-160))"
fi

if want bench; then
  step "bench: wait for the matrices"
  wait "$FETCH"; tail -12 "$R/log/fetch.txt"; ls -la "$W/m" | tee -a "$R/log/fetch.txt"
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -l 1 > "$R/smi-bench.csv" 2> /dev/null & SMI=$!
  for m in Qwen3.5-9B Qwen3.5-27B gemma-4-12B-it gemma-4-31B-it; do
    [ -f "$W/m/$m.safetensors" ] || { note "no matrices of $m"; continue; }
    step "layouts.py $m"
    timeout 1500 "$PY" layouts.py "$W/m/$m.safetensors" --hot --mhz "$MAXMHZ" --tokens "1 2 4 8 16 24 32 48 64 128 512 1024 2048" ${TS_VARIANTS:+--variants "$TS_VARIANTS"} --json "$R/layouts-$m.json" > "$R/layouts-$m.txt" 2>&1
    note "layouts $m exit $?"
  done
  [ -f "$W/m/head-Qwen3.5-9B.safetensors" ] && { timeout 900 "$PY" layouts.py "$W/m/head-Qwen3.5-9B.safetensors" --mhz "$MAXMHZ" --tokens "1 8 16 32 64" --routes "" --json "$R/layouts-head-Qwen3.5-9B.json" > "$R/layouts-head-Qwen3.5-9B.txt" 2>&1; note "layouts lm_head exit $?"; }
  if [ -f "$W/m/experts-Qwen3.5-35B-A3B.safetensors" ]; then
    for l in mma mma12; do
      timeout 900 "$PY" moe_layer.py --file "$W/m/experts-Qwen3.5-35B-A3B.safetensors" --k 8 --layout $l --tokens "1 8 16 32 64 256 1024" --check --bf16 --json "$R/moe-$l.json" > "$R/moe-$l.txt" 2>&1
      note "moe_layer $l exit $?"
    done
  fi
  kill "$SMI" 2> /dev/null
fi

if want vllm || want serve; then
  step "the model"
  wait "$DLM"; tail -2 "$R/log/dl-model.txt"; export HF_HUB_OFFLINE=1
fi

if want vllm; then
  step "check_vllm.py --quick Qwen/Qwen3.5-9B"
  ( cd "$W/src/gpu/vllm" && timeout 3000 "$PY" check_vllm.py --quick --out "$R/check_vllm" Qwen/Qwen3.5-9B ) > "$R/check-vllm.txt" 2>&1
  note "check_vllm exit $? ($(grep -E "PASS|FAIL|passed|failed" "$R/check-vllm.txt" | tail -1 | cut -c1-200))"
fi

if want serve; then
  cat > "$W/vllm-logging.json" << 'EOJ'
{"version": 1, "disable_existing_loggers": false,
 "formatters": {"vllm": {"class": "vllm.logging_utils.NewLineFormatter", "datefmt": "%m-%d %H:%M:%S", "format": "%(levelname)s %(asctime)s.%(msecs)03d [%(fileinfo)s:%(lineno)d] %(message)s"}},
 "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm", "level": "INFO", "stream": "ext://sys.stdout"}},
 "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false}}}
EOJ
  export VLLM_LOGGING_CONFIG_PATH=$W/vllm-logging.json
  mkdir -p "$R/serve"
  bench() {  # MODE NAME USERS PROMPTS SEED
    timeout 900 "$W/venv/bin/vllm" bench serve --backend vllm --model Qwen/Qwen3.5-9B --port 8012 --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos \
      --num-prompts "$4" --max-concurrency "$3" --request-rate inf --seed "$5" --save-result --result-dir "$R/serve" --result-filename "$1-$2.json" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 > "$R/serve/bench-$1-$2.txt" 2>&1
  }
  for mode in ${TS_MODES:-bf16 mma12 mma mmacc mma12b}; do
    q=(--quantization glyd); lay=${mode%b}
    [ "$mode" = mmacc ] && { lay=mma; q+=(--additional-config '{"glyd": {"cc12": true}}'); }
    [ "$mode" = bf16 ] && q=()
    step "serve $mode"
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -l 1 > "$R/serve/smi-$mode.csv" 2> /dev/null & SMI=$!
    GLYD_LAYOUT=$lay VLLM_CACHE_ROOT=$W/cache/$lay "$W/venv/bin/vllm" serve Qwen/Qwen3.5-9B "${q[@]}" --language-model-only --max-model-len 4096 --gpu-memory-utilization 0.9 \
      --port 8012 --enable-logging-iteration-details > "$R/serve/serve-$mode.txt" 2>&1 &
    SV=$!
    up=0; for i in $(seq 1 120); do curl -sf localhost:8012/v1/models > /dev/null && { up=1; break; }; kill -0 $SV 2> /dev/null || break; sleep 5; done
    if [ "$up" = 1 ]; then
      bench "$mode" warm 64 64 100
      for u in 1 8 16 32 64; do n=$(( u == 1 ? 8 : 4 * u )); bench "$mode" "u$u" "$u" "$n" "$u"; done
      note "serve $mode: $(grep -h "Model loading took\|GPU KV cache size\|glyd: m" "$R/serve/serve-$mode.txt" | sed 's/.*\] //' | tr '\n' ';' | cut -c1-300)"
    else
      note "serve $mode: the server did not start ($(tail -2 "$R/serve/serve-$mode.txt" | tr '\n' ' ' | cut -c1-200))"
    fi
    kill $SV 2> /dev/null; wait $SV 2> /dev/null; kill $SMI 2> /dev/null
    pkill -9 -f "EngineCore|vllm serve" 2> /dev/null; sleep 5
  done
fi

step "summary"
python3 "$W/src/job/summarize.py" "$R" > "$R/summary.txt" 2> "$R/log/summary-err.txt"; note "summary exit $?"
step "done in $(el) s"
{ pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null
touch "$R/ALLDONE"
