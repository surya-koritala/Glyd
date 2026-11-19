#!/usr/bin/env bash
# ps_job.sh: a prompt's products and first-token latency, Glyd against bf16, on one Runpod GPU, unattended (2026 models only).
# In ~: this and ps_src.tar (the tree's gpu/, glyd-gpu-py/, this job/ and COMMIT). Steps:
#   env      uv venv with vLLM 0.30.0 (its PyTorch) and glyd-gpu 0.28.0 (vLLM's entry point for the plugin; the package itself from the
#            tree, PYTHONPATH), nvcc of PyTorch's CUDA
#   build    the tree's library for this GPU alone (sm_XX), a development build (GLYD_DEC_MIN and the plugin's GLYD_AHEAD_PROMPT read)
#   check    the library's self-test (glyd_gpu_dev.py: every bf16 bit pattern, both layouts, the products against fp32)
#   pass     prompt_pass.py on Qwen3.5-9B, Qwen3.5-27B and gemma-4-31B-it shapes at 128-8192 tokens: bf16, the plugin's products (glyd),
#            decoded first, decoded ahead (warps 0, 2, 4 an SM), GLinear; both layouts
#   serve    vllm serve Qwen/Qwen3.5-9B a mode of PS_MODES (bf16; glyd: the default layout and routes; ahead: the same with
#            GLYD_AHEAD_PROMPT=1, plus PS_AHEAD_ENV), the same flags for each; one user: 3 warm-up prompts, then 8 prompts of 1,000 tokens
#            and 5 of 8,000 (random tokens, 8 out), vllm bench serve's TTFT; a greedy completion of one fixed prompt a mode
#   summary  ps_summary.py -> summary.txt
# results/DONE from the exit trap; results/ALLDONE when it ran to its end. PS_STEPS (default "build check pass serve"); PS_END (s, 4200).
set -u
R=$HOME/results; W=$HOME/psw
mkdir -p "$R/log" "$W/tmp"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
END=${PS_END:-4200}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
note() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; echo "   $*"; }
trap 'touch "$R/DONE"' EXIT
( sleep "$END"; echo "hard stop at +$(el) s" >> "$R/steps.txt"; pkill -TERM -P $$; kill -TERM $$ ) > /dev/null 2>&1 & WD=$!
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache
want() { case " ${PS_STEPS:-build check pass serve} " in *" $1 "*) return 0 ;; esac; return 1; }

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
echo "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute $CC (sm_$ARCH)" | tee "$R/gpu-short.txt"
cd "$W" && rm -rf src && mkdir src && tar -C src -xf "$HOME/ps_src.tar" 2> /dev/null || { echo "no ps_src.tar"; exit 1; }
cp src/COMMIT "$R/COMMIT"; cat src/COMMIT

step "env: uv, vLLM 0.30.0, nvcc"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { echo "no uv"; exit 1; }
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" "glyd-gpu[vllm]==0.28.0"; } > "$R/log/env-vllm.txt" 2>&1 || { note "FAIL: vLLM did not install"; tail -5 "$R/log/env-vllm.txt"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "FAIL: nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
"$PY" -c "import importlib.metadata as md, torch; print('vllm', md.version('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability())" | tee "$R/env.txt"
echo "nvcc: $(nvcc --version | tail -1)" | tee -a "$R/env.txt"
note "env"

DLM=
if want serve; then  # Qwen3.5-9B, beside the build and the passes
  export HF_HOME=$W/hf HF_XET_HIGH_PERFORMANCE=1
  ( timeout 1500 "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model']))" > "$R/log/dl-model.txt" 2>&1; echo "exit $?" >> "$R/log/dl-model.txt" ) &
  DLM=$!
fi

step "build: the library for sm_$ARCH"
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so
mkdir -p "$W/lib"
{ time { nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" && nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$LIB" "$W/lib/glyd_gpu.o"; }; } > "$R/log/build.txt" 2>&1
[ -f "$LIB" ] || { note "FAIL: the library did not build (log/build.txt)"; tail -20 "$R/log/build.txt"; exit 1; }
export GLYD_GPU_LIB=$LIB PYTHONPATH=$W/src/glyd-gpu-py
"$PY" -c "from glyd_gpu import kernels as g, _lib; g.lib(); print('C API', _lib._lib.glyd_gpu_api_version(), '| GPU code', _lib.gpu())" 2>&1 | tee "$R/log/load.txt"
note "build: $(grep real "$R/log/build.txt"); $(tail -1 "$R/log/load.txt")"

cd "$W/src/gpu"
if want check; then
  step "check: the library's self-test"
  timeout 1500 "$PY" glyd_gpu_dev.py > "$R/check-selftest.txt" 2>&1; note "self-test exit $? ($(tail -1 "$R/check-selftest.txt" | cut -c1-160))"
fi

if want pass; then
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -l 1 > "$R/smi-pass.csv" 2> /dev/null & SMI=$!
  for m in ${PS_MODELS:-Qwen3.5-9B Qwen3.5-27B gemma-4-31B-it}; do
    step "prompt_pass.py $m"
    timeout 2400 "$PY" prompt_pass.py "$m" --tokens "${PS_TOKENS:-128 512 1024 2048 4096 8192}" --json "$R/pass-$m.json" > "$R/pass-$m.txt" 2>&1
    note "pass $m exit $? ($(tail -1 "$R/pass-$m.txt" | cut -c1-200))"
  done
  kill "$SMI" 2> /dev/null
fi

if want serve; then
  step "the model"
  wait "$DLM"; tail -2 "$R/log/dl-model.txt"; export HF_HUB_OFFLINE=1
  cat > "$W/vllm-logging.json" << 'EOJ'
{"version": 1, "disable_existing_loggers": false,
 "formatters": {"vllm": {"class": "vllm.logging_utils.NewLineFormatter", "datefmt": "%m-%d %H:%M:%S", "format": "%(levelname)s %(asctime)s.%(msecs)03d [%(fileinfo)s:%(lineno)d] %(message)s"}},
 "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm", "level": "INFO", "stream": "ext://sys.stdout"}},
 "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false}}}
EOJ
  export VLLM_LOGGING_CONFIG_PATH=$W/vllm-logging.json VLLM_USE_FLASHINFER_SAMPLER=0
  mkdir -p "$R/serve"
  bench() {  # MODE NAME INPUT PROMPTS SEED
    timeout 900 "$W/venv/bin/vllm" bench serve --backend vllm --model Qwen/Qwen3.5-9B --port 8012 --dataset-name random --random-input-len "$3" --random-output-len 8 --ignore-eos \
      --num-prompts "$4" --max-concurrency 1 --request-rate inf --seed "$5" --save-result --result-dir "$R/serve" --result-filename "$1-$2.json" \
      --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 > "$R/serve/bench-$1-$2.txt" 2>&1
  }
  for mode in ${PS_MODES:-bf16 glyd ahead}; do
    q=(--quantization glyd); envs=()
    [ "$mode" = bf16 ] && q=()
    [ "$mode" = ahead ] && envs=(GLYD_AHEAD_PROMPT=1 ${PS_AHEAD_ENV:-})
    step "serve $mode ${envs[*]}"
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$R/serve/smi-$mode.csv" 2> /dev/null & SMI=$!
    env "${envs[@]}" VLLM_CACHE_ROOT="$W/cache/$mode" "$W/venv/bin/vllm" serve Qwen/Qwen3.5-9B "${q[@]}" --language-model-only --max-model-len 9216 --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 --port 8012 > "$R/serve/serve-$mode.txt" 2>&1 &
    SV=$!
    up=0; for i in $(seq 1 120); do curl -sf localhost:8012/v1/models > /dev/null && { up=1; break; }; kill -0 $SV 2> /dev/null || break; sleep 5; done
    if [ "$up" = 1 ]; then
      curl -s localhost:8012/v1/completions -H 'Content-Type: application/json' \
        -d '{"model": "Qwen/Qwen3.5-9B", "prompt": "The three primary colors of light are", "max_tokens": 32, "temperature": 0}' > "$R/serve/greedy-$mode.json"
      bench "$mode" warm1k 1000 3 7; bench "$mode" warm8k 8000 2 8
      bench "$mode" p1k 1000 8 1; bench "$mode" p8k 8000 5 2
      note "serve $mode: $(grep -h "Model loading took\|GPU KV cache size\|glyd: m" "$R/serve/serve-$mode.txt" | sed 's/.*\] //' | tr '\n' ';' | cut -c1-300)"
    else
      note "serve $mode: the server did not start ($(grep -h -m2 -E 'Error|error|out of memory' "$R/serve/serve-$mode.txt" | tr '\n' ' ' | cut -c1-300))"
    fi
    kill $SV 2> /dev/null; wait $SV 2> /dev/null; kill $SMI 2> /dev/null
    pkill -9 -f "EngineCore|vllm serve" 2> /dev/null; sleep 5
  done
fi

step "summary"
python3 "$W/src/job/ps_summary.py" "$R" > "$R/summary.txt" 2> "$R/log/summary-err.txt"; note "summary exit $?"
step "done in $(el) s"
{ pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null
touch "$R/ALLDONE"
