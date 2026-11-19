#!/usr/bin/env bash
# fp8_vllm_job.sh: gpu/fp8/check_fp8_vllm.py on one Runpod GPU pod, unattended (scratchpad/rp_run.sh uploads this and f8src.tar, runs it, pulls ~/results, deletes the pod):
# vLLM 0.30.0 (uv, a venv), glyd-gpu 0.28.0's wheel for its library, this tree's glyd_gpu (f8src.tar: glyd-gpu-py/ and gpu/fp8/) on PYTHONPATH.
#   1. the plugin's unit tests (tests/test_vllm.py, tests/test_vllm_fp8.py: no GPU needed, run here against the real vLLM)
#   2. the slices (gpu/fp8/make_slice.py: the real FP8 weights of the first layers, read by HTTP range) while the tests run
#   3. check_fp8_vllm.py on each (exact against stock vLLM FP8: eager, graph, graphonly; the weights as vLLM logs them; the speed)
#   4. a bf16 smoke (--quantization glyd on a small bf16 model: the bf16 path as it was)
# F8_PLAN items also: minimax3:<modes> (MiniMaxAI/MiniMax-M3-MXFP8's 3 dense layers), mistral2:<modes> (mistralai/Mistral-Small-4-119B-2603's first 2 layers, in Mistral's own format as vLLM serves it: per-tensor static FP8, MLA, a mixture of experts), seeds (qwen4, unpinned,
# one hash seed a pair of runs, autotuners pinned: F8_SEEDS, ~/xbits.json: the outputs the mismatch of the first run gave).
# F8_PLAN: "qwen4:eager,graph,graphonly glm3:eager,compile qwen4pt:eager whole:eager,graph qwen4diag qwen4diagpin" (default: qwen4diag qwen4diagpin qwen4 glm3 qwen4pt, each run while 10 minutes are left)
# (qwen4diag: stock 3 times and Glyd twice compiled, each compile fresh, as Triton's autotuners and Python's hash seed come; qwen4diagpin: stock twice and Glyd once with the autotuners' first
# configs and one hash seed. The ordinary checks pin them too unless F8_PIN=0.): <name>:<modes>; qwen4diag: check_fp8_vllm.py --diag on the Qwen slice.
# qwen4pt: the Qwen slice requantized to per-tensor FP8 (pertensor_slice.py). whole: Qwen/Qwen3.8-27B-FP8's language model, all 64 layers (make_slice.py --layers 64), fetched at the start beside the install.
# F8_END: seconds at the most (2160: a 45-minute pod less its creation and its teardown). Results: ~/results/{job.log,steps.txt,tests.txt,<name>/...,DONE}.
set -u
[ -f "$HOME/f8env" ] && . "$HOME/f8env"  # (F8_END, F8_PLAN, F8_PIN for this pod: rp_run.sh uploads it with UPLOAD=)
R=${R:-$HOME/results}
W=${W:-$HOME/f8w}
END=${F8_END:-2160}
PLAN=${F8_PLAN:-"qwen4diag qwen4diagpin qwen4:eager,graph,graphonly glm3:eager,compile qwen4pt:eager,graph,graphq"}
mkdir -p "$R/log" "$W/tmp"
T0=$(date +%s)
exec > >(tee -a "$R/job.log") 2>&1
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
note() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; echo "   $*"; }
left() { echo $(( END - $(el) )); }
( sleep $(( END + 60 )); echo "hard stop at +$(el) s" >> "$R/steps.txt"; kill -TERM $$; pkill -TERM -P $$ ) > /dev/null 2>&1 &
WD=$!
finish() { { pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null; touch "$R/DONE"; }
trap finish EXIT

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm --format=csv; lscpu | grep -E "Model name"; nproc; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN  # (the models are public: no token is read, none sent)
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAMEGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$NAMEGPU (compute $CC)" | tee "$R/machine-short.txt"

step "the tree, the slices' downloads (background)"
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/f8src.tar" || { note "FAIL: no f8src.tar"; exit 1; }
[ -f "$W/src/COMMIT" ] && cp "$W/src/COMMIT" "$R/COMMIT"
for item in $PLAN; do  # (the slices are built with the system's python3: standard library only)
  name=${item%%:*}; name=${name%diagpin}; name=${name%diag}; [ "$name" = qwen4pt ] && name=qwen4; [ "$name" = seeds ] && name=qwen4
  case $name in
    minimax3) ( python3 "$W/src/gpu/fp8/make_slice.py" MiniMaxAI/MiniMax-M3-MXFP8 "$W/slice-minimax3" --layers 3 > "$R/log/slice-minimax3.txt" 2>&1; echo "exit $?" >> "$R/log/slice-minimax3.txt" ) & ;;
    mistral2) ( python3 "$W/src/gpu/fp8/make_slice.py" mistralai/Mistral-Small-4-119B-2603 "$W/slice-mistral2" --layers 2 --max-gb 14 --weights consolidated > "$R/log/slice-mistral2.txt" 2>&1; echo "exit $?" >> "$R/log/slice-mistral2.txt" ) & ;;
    qwen4) [ -d "$W/slice-qwen4" ] || ( python3 "$W/src/gpu/fp8/make_slice.py" Qwen/Qwen3.8-27B-FP8 "$W/slice-qwen4" --layers 4 > "$R/log/slice-qwen4.txt" 2>&1; echo "exit $?" >> "$R/log/slice-qwen4.txt" ) & ;;
    whole) ( python3 "$W/src/gpu/fp8/make_slice.py" Qwen/Qwen3.8-27B-FP8 "$W/whole" --layers 64 --max-gb 40 > "$R/log/slice-whole.txt" 2>&1; echo "exit $?" >> "$R/log/slice-whole.txt" ) & ;;
    glm3) [ -d "$W/slice-glm3" ] || ( python3 "$W/src/gpu/fp8/make_slice.py" zai-org/GLM-5.3-Flash "$W/slice-glm3" --layers 3 > "$R/log/slice-glm3.txt" 2>&1; echo "exit $?" >> "$R/log/slice-glm3.txt" ) & ;;
  esac
done

step "uv, vLLM 0.30.0"
UV=$(command -v uv || ls "$HOME/tools/uv/uv" "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then
  mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv
fi
[ -x "${UV:-}" ] || { note "FAIL: no uv"; exit 1; }
t=$(date +%s)
{ rm -rf "$W/venv" && "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" && "$UV" pip install --python "$W/venv/bin/python" --no-deps "glyd-gpu==0.28.0"; } > "$R/log/env.txt" 2>&1 || { note "FAIL: vLLM / glyd-gpu did not install (log/env.txt)"; tail -5 "$R/log/env.txt"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || note "nvcc did not install (log/env-nvcc.txt): FlashInfer's JIT may fail"
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
# (the toolkit's libraries, linkable as -lcudart, -lnvrtc ...: FlashInfer's JIT of the FP8 block-scaled GEMM on Hopper links -lnvrtc, which the wheels hold as libnvrtc.so.13 alone)
mkdir -p "$CU/lib64"
for f in "$CU"/lib/lib*.so.*; do b=${f##*/}; [ -e "$CU/lib64/${b%%.so*}.so" ] || ln -sf "../lib/$b" "$CU/lib64/${b%%.so*}.so"; done 2> /dev/null
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
SP=$("$PY" -c "import glyd_gpu, os; print(os.path.dirname(glyd_gpu.__file__))")
LIB=$(ls "$SP"/libglyd_gpu_cuda*.so 2> /dev/null | head -1)
export PYTHONPATH=$W/src/glyd-gpu-py GLYD_GPU_LIB=$LIB  # (this tree's package; the wheel's library: the FP8 path needs none, the bf16 smoke does)
"$PY" -c "
import importlib.metadata as m, torch
v = lambda k: m.version(k) if k else '-'
import glyd_gpu
print('vllm', v('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', v('transformers'), '| flashinfer', v('flashinfer-python'), '| glyd-gpu wheel', v('glyd-gpu'), '| glyd_gpu', glyd_gpu.__file__, '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1 || { note "FAIL: $(tail -2 "$R/env.txt" | tr '\n' ' ')"; exit 1; }
cat "$R/env.txt"; echo "nvcc: $(nvcc --version | tail -1)" | tee -a "$R/env.txt"
note "env: $(( $(date +%s) - t )) s"

if [ "$CC" = "9.0" ]; then  # (Hopper: vLLM's FP8 block-scaled GEMM there is FlashInfer's, built with nvcc on first use: built here once, with the link fixed if it asks for a library the wheels name differently)
  step "FlashInfer's FP8 block-scaled GEMM, built once"
  fi_build() { "$PY" -c "from flashinfer.gemm.gemm_base import get_fp8_blockscale_gemm_runner_sm90 as g; g(); print('built')" > "$R/log/flashinfer-fp8.txt" 2>&1; }
  fi_build || {
    for lib in $(grep -o -- "-l[a-z0-9_]*" "$R/log/flashinfer-fp8.txt" | sort -u | grep -E "lnvrtc|lnvJitLink|lcublasLt|lcublas|lcudart|lnvToolsExt"); do
      n=${lib#-l}; f=$(find "$W/venv" -name "lib$n.so*" 2> /dev/null | head -1)
      [ -n "$f" ] && [ ! -e "$CU/lib64/lib$n.so" ] && ln -sf "$f" "$CU/lib64/lib$n.so" && note "linked lib$n.so -> $f"
    done
    fi_build; }
  note "flashinfer fp8 gemm: $(tail -1 "$R/log/flashinfer-fp8.txt" | cut -c1-120)"
fi

step "the plugin's tests (no GPU needed; the real vLLM)"
( cd "$W/src/glyd-gpu-py/tests" && GLYD_REQUIRE_VLLM=1 "$PY" test_vllm_fp8.py && GLYD_REQUIRE_VLLM=1 "$PY" test_vllm.py ) > "$R/tests.txt" 2>&1
note "tests: exit $? ($(grep -c ' ok$' "$R/tests.txt") ok; $(grep -c Traceback "$R/tests.txt") tracebacks)"

waitslice() {  # NAME: its download done and ok
  until grep -q "^exit" "$R/log/slice-$1.txt" 2> /dev/null; do [ "$(left)" -lt 0 ] && return 1; sleep 3; done
  grep -q "^exit 0" "$R/log/slice-$1.txt"
}
for item in $PLAN; do
  name=${item%%:*}; modes=${item#*:}; run=check
  dflags="--stocks 3 --ops 2"
  case $name in *diagpin) run=diagpin; name=${name%diagpin}; modes=graph; dflags="--pin --stocks 2 --ops 1" ;; *diag) run=diag; name=${name%diag}; modes=graph ;; esac
  case $name in seeds) run=seeds; name=qwen4; modes=graph ;; esac
  name_=$name; [ "$name" = qwen4pt ] && name=qwen4
  need=600; [ "$name" = whole ] && need=1200; [ "$run" = seeds ] && need=480
  [ "$(left)" -lt "$need" ] && { note "$name ($run): not run, $(left) s left"; continue; }
  case $name in
    qwen4) dir=$W/slice-qwen4; flags="--language-model-only"; waitslice qwen4 || { note "$name: no slice ($(tail -2 "$R/log/slice-qwen4.txt" | tr '\n' ' '))"; continue; }
      if [ "$name_" = qwen4pt ]; then  # (the same weights, per-tensor FP8: vLLM's per-tensor path, CUTLASS, a transposed [K, N] weight)
        [ -d "$W/slice-qwen4pt" ] || "$PY" "$W/src/gpu/fp8/pertensor_slice.py" "$dir" "$W/slice-qwen4pt" > "$R/log/pertensor.txt" 2>&1 || { note "qwen4pt: no slice ($(tail -2 "$R/log/pertensor.txt" | tr '\n' ' '))"; continue; }
        dir=$W/slice-qwen4pt
      fi ;;
    minimax3) dir=$W/slice-minimax3; flags="--language-model-only --no-bench --no-again"; waitslice minimax3 || { note "$name: no slice ($(tail -2 "$R/log/slice-minimax3.txt" | tr '\n' ' '))"; continue; } ;;
    mistral2) dir=$W/slice-mistral2; flags="--no-bench --no-again"; waitslice mistral2 || { note "$name: no slice ($(tail -2 "$R/log/slice-mistral2.txt" | tr '\n' ' '))"; continue; } ;;
    glm3) dir=$W/slice-glm3; flags="--language-model-only --no-bench --no-again"; waitslice glm3 || { note "$name: no slice ($(tail -2 "$R/log/slice-glm3.txt" | tr '\n' ' '))"; continue; } ;;
    whole) dir=$W/whole; flags="--language-model-only --no-again"; need=1200; waitslice whole || { note "$name: no download ($(tail -2 "$R/log/slice-whole.txt" | tr '\n' ' '))"; continue; } ;;
    *) note "no plan item $name"; continue ;;
  esac
  step "$name: check_fp8_vllm.py ($run, $modes)"
  out=$R/$name_; case $run in diag*|seeds) out=$R/$name-$run ;; esac
  mkdir -p "$out"
  case $run in
    diag*) extra="--diag $dflags" ;;
    seeds) extra="--seeds ${F8_SEEDS:-0,1,2,3,4}"; [ -f "$HOME/xbits.json" ] && extra="$extra --x-bits $HOME/xbits.json" ;;
    *) extra="--modes $modes"; [ "${F8_PIN:-1}" = 1 ] && extra="$extra --pin" ;;
  esac
  timeout "$(( $(left) > 60 ? $(left) : 60 ))" "$PY" "$W/src/gpu/fp8/check_fp8_vllm.py" "$dir" --out "$out" $extra $flags > "$out/check.txt" 2>&1
  note "$name ($run): exit $?: $(grep -E '^(PASS|FAIL|INCONCLUSIVE|INFO: (diag|outcomes|equal))' "$out/check.txt" | cut -c1-110 | tr '\n' '|')"
done

if [ "$(left)" -gt 420 ]; then
  step "bf16 smoke: --quantization glyd on a small bf16 model, as before"
  mkdir -p "$R/bf16"
  timeout "$(( $(left) < 400 ? $(left) : 400 ))" "$PY" - > "$R/bf16/smoke.txt" 2>&1 << 'EOF'
import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-0.6B", quantization="glyd", enforce_eager=True, max_model_len=1024, gpu_memory_utilization=0.4, additional_config={"glyd": {"kv": "off"}})
cfg = llm.llm_engine.vllm_config
print("quant_config:", type(cfg.quant_config).__name__, "| quantization:", cfg.model_config.quantization, "| glyd:", cfg.additional_config.get("glyd"))
out = llm.generate(["The capital of France is", "def add(a, b):"], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
for o in out:
    print(repr(o.outputs[0].text))
assert type(cfg.quant_config).__name__ == "GlydConfig" and all(o.outputs[0].token_ids for o in out)
print("bf16 smoke: ok")
EOF
  note "bf16 smoke: exit $? ($(tail -1 "$R/bf16/smoke.txt" | cut -c1-100))"
fi
step "done in $(el) s"
note "done"
touch "$R/ALLDONE"
