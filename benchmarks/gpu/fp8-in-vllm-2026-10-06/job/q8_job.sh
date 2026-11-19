#!/usr/bin/env bash
# q8_job.sh: the lossless FP8 codes under vLLM on whole 2026 FP8 checkpoints, on one Runpod pod, unattended (rp_run.sh uploads this, q8src.tar and
# q8env, runs it, pulls ~/results, deletes the pod): vLLM 0.30.0 (uv, a venv), this tree's library (gpu/build_lib.sh) and glyd_gpu (q8src.tar), then
#   1. gpu/fp8/check_f8.py: the format's pack, unpack and experts' unpack, and f8_linear, bit for bit / within bounds on this GPU (the checkpoints run where
#      the pack and the decoders pass)
#   2. the plugin's FP8 tests against the real vLLM (tests/test_vllm_fp8.py)
#   3. Q8_BENCH=1: gpu/fp8/bench_f8.py at Qwen3.8-27B-FP8's shapes: f8_linear, the decode alone and vLLM's own FP8 GEMM at 1, 8, 32, 64 tokens
#   4. each checkpoint of Q8_MODELS ("repo|modes|flags[|name]" separated by spaces, flags with commas for spaces; name: the results' folder's suffix, where a
#      checkpoint runs more than once), downloaded in the background in that order (once each):
#      gpu/fp8/check_fp8_vllm.py (stock vLLM and Glyd exact on the same checkpoint: logits bit for bit, the weights vLLM logs, KV tokens, concurrency at
#      8,192 tokens, output tokens a second at 1, 8, 32 users), then with Q8_SERVE=1 stock alone at vLLM's own compile defaults (--modes serve --only stock:
#      the speed reference, 1 and 32 users; the same flags), the checkpoint deleted after.
# Q8_END: seconds at the most. Results: ~/results/{job.log,steps.txt,check_f8.log,tests.txt,bench_f8.*,<model>/...,DONE}.
set -u
[ -f "$HOME/q8env" ] && . "$HOME/q8env"
R=$HOME/results; W=$HOME/q8w; END=${Q8_END:-5400}
mkdir -p "$R/log" "$W/tmp" "$W/m"
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
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN GLYD_F8_CODEC GLYD_F8_FUSED_MAX  # (public models: no token read or sent)
export GLYD_F8_MEMLOG=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=$W/uvcache PYTHONDONTWRITEBYTECODE=1 HF_XET_HIGH_PERFORMANCE=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (compute $CC)" | tee "$R/machine-short.txt"
mkdir -p "$W/src" && tar -C "$W/src" -xf "$HOME/q8src.tar" || { note "FAIL: no q8src.tar"; exit 1; }
cp "$W/src/COMMIT" "$R/COMMIT" 2> /dev/null; cat "$R/COMMIT" 2> /dev/null

step "uv"
UV=$(command -v uv || ls "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv; fi
[ -x "${UV:-}" ] || { note "FAIL: no uv"; exit 1; }

step "the checkpoints' downloads (background, in order)"
( for item in ${Q8_MODELS:-}; do
    repo=${item%%|*}; d=$W/m/${repo//\//__}; t=$(date +%s)
    [ -f "$R/log/dl-${repo//\//__}.txt" ] && continue
    "$UV" tool run --from "huggingface_hub[hf_xet]" hf download "$repo" --local-dir "$d" --exclude "every_eval_ever/*" > "$R/log/dl-${repo//\//__}.txt" 2>&1
    echo "exit $? in $(( $(date +%s) - t )) s, $(du -sh "$d" 2> /dev/null | cut -f1)" >> "$R/log/dl-${repo//\//__}.txt"
  done ) &

step "vLLM 0.30.0, nvcc"
t=$(date +%s)
{ "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0"; } > "$R/log/env-vllm.txt" 2>&1 || { note "FAIL: vLLM did not install (log/env-vllm.txt)"; tail -5 "$R/log/env-vllm.txt"; exit 1; }
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install -q --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || { note "FAIL: nvcc did not install"; exit 1; }
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64"
for f in "$CU"/lib/lib*.so.*; do b=${f##*/}; [ -e "$CU/lib64/${b%%.so*}.so" ] || ln -sf "../lib/$b" "$CU/lib64/${b%%.so*}.so"; done 2> /dev/null
NVRTC=$(find "$W/venv" -name 'libnvrtc.so.*' ! -name '*builtins*' | head -1)  # (FlashInfer's JIT links -lnvrtc)
[ -n "$NVRTC" ] && for d in "$CU/lib" "$CU/lib64"; do ln -sf "$NVRTC" "$d/libnvrtc.so"; done
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH LIBRARY_PATH=$CU/lib64:$CU/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
note "env: $(( $(date +%s) - t )) s"

step "the library (gpu/build_lib.sh)"
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
( cd "$W/src" && time bash gpu/build_lib.sh "$W/lib" ) > "$R/log/build.txt" 2>&1 || { note "FAIL: the library did not build (log/build.txt)"; tail -20 "$R/log/build.txt"; exit 1; }
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so PYTHONPATH=$W/src/glyd-gpu-py
# (vLLM finds the plugin by its entry point, as a glyd-gpu wheel declares it (pyproject.toml): declared here for this tree's package, on PYTHONPATH)
EP=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/glyd_src-0.0.dist-info
mkdir -p "$EP" && printf 'Metadata-Version: 2.1\nName: glyd-src\nVersion: 0.0\n' > "$EP/METADATA" && printf '[vllm.general_plugins]\nglyd = glyd_gpu.vllm_entry:register\n' > "$EP/entry_points.txt"
"$PY" -c "
import importlib.metadata as md, torch
from glyd_gpu import kernels as g, _lib
g.lib()
v = lambda p: md.version(p) if p else '-'
from importlib.metadata import entry_points
assert 'glyd_gpu.vllm_entry:register' in [e.value for e in entry_points(group='vllm.general_plugins')], 'the plugin entry point'
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
assert 'glyd' in QUANTIZATION_METHODS, 'glyd not registered'
print('vllm', v('vllm'), '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| flashinfer', v('flashinfer-python'), '|', torch.cuda.get_device_name(), torch.cuda.get_device_capability(), '| C API', _lib._lib.glyd_gpu_api_version())" > "$R/env.txt" 2>&1 || { note "FAIL: $(tail -2 "$R/env.txt" | tr '\n' ' ')"; exit 1; }
echo "nvcc: $(nvcc --version | tail -1)" >> "$R/env.txt"; cat "$R/env.txt"
note "library: $(grep real "$R/log/build.txt")"

step "check_f8.py: the format's kernels on this GPU"
( cd "$W/src" && timeout 900 "$PY" gpu/fp8/check_f8.py ) > "$R/check_f8.log" 2>&1
note "check_f8: exit $? ($(grep -E '^(test_|case_)\S+ (ok|FAILED)' "$R/check_f8.log" | tr '\n' ' ' | cut -c1-400))"
# (the checkpoints run where the format's pack, its decoders and their CUDA graphs pass: f8_linear (case_lin*) is off in serving, a check's GLYD_F8_FUSED_MAX only)
grep -q "^case_experts ok" "$R/check_f8.log" && grep -q "^case_dense ok" "$R/check_f8.log" && grep -q "^case_graph ok" "$R/check_f8.log" && ! grep -E "^(test_\S+|case_strides|case_experts|case_dense|case_graph) FAILED" "$R/check_f8.log" > /dev/null; F8OK=$?

step "the plugin's FP8 tests against vLLM 0.30.0"
( cd "$W/src/glyd-gpu-py/tests" && GLYD_REQUIRE_VLLM=1 timeout 600 "$PY" test_vllm_fp8.py ) > "$R/tests.txt" 2>&1
note "tests: exit $? ($(grep -c ' ok$' "$R/tests.txt") ok; $(grep -c Traceback "$R/tests.txt") tracebacks)"

if [ "${Q8_BENCH:-0}" = 1 ] && [ "$F8OK" = 0 ]; then
  step "bench_f8.py: Qwen3.8-27B-FP8's shapes"
  ( cd "$W/src" && timeout 900 "$PY" gpu/fp8/bench_f8.py --models Qwen3.8-27B-FP8 --tp 1 --ms 1,8,32,64 --out "$R/bench_f8.json" ) > "$R/bench_f8.log" 2>&1
  note "bench_f8: exit $? ($(tail -1 "$R/bench_f8.log" | cut -c1-150))"
fi

for item in ${Q8_MODELS:-}; do
  repo=${item%%|*}; rest=${item#*|}; modes=${rest%%|*}; flags=${rest#*|}; tag=; case $flags in *"|"*) tag=-${flags#*|}; flags=${flags%%|*};; esac
  flags=${flags//,/ }; dkey=${repo//\//__}; key=$dkey$tag; d=$W/m/$dkey
  [ "$(left)" -lt 900 ] && { note "$repo: not run, $(left) s left"; continue; }
  if [ "$F8OK" != 0 ]; then note "$repo: not run: the format's kernels did not pass check_f8 here"; continue; fi
  until grep -q "^exit" "$R/log/dl-$dkey.txt" 2> /dev/null; do [ "$(left)" -lt 900 ] && break; sleep 5; done
  grep -q "^exit 0" "$R/log/dl-$dkey.txt" 2> /dev/null || { note "$repo: no download ($(tail -2 "$R/log/dl-$dkey.txt" 2> /dev/null | tr '\n' ' '))"; continue; }
  note "$repo$tag: downloaded ($(tail -1 "$R/log/dl-$dkey.txt"))"
  step "$repo: check_fp8_vllm.py ($modes $flags)"
  mkdir -p "$R/$key"
  F8_RUN_TIMEOUT=${Q8_RUN_TIMEOUT:-1500} timeout "$(( $(left) > 120 ? $(left) - 60 : 60 ))" "$PY" "$W/src/gpu/fp8/check_fp8_vllm.py" "$d" --out "$R/$key" --modes "$modes" --pin $flags > "$R/$key/check.txt" 2>&1
  note "$repo$tag: exit $?: $(grep -E '^(PASS|FAIL|INCONCLUSIVE|INFO)' "$R/$key/check.txt" | cut -c1-160 | tr '\n' '|')"
  if [ "${Q8_SERVE:-0}" = 1 ] && [ "$(left)" -gt 600 ]; then  # (stock at vLLM's defaults: no --pin, no deterministic inductor, the fusions on)
    mkdir -p "$R/$key/serve"
    F8_RUN_TIMEOUT=${Q8_RUN_TIMEOUT:-1500} timeout "$(( $(left) - 60 ))" "$PY" "$W/src/gpu/fp8/check_fp8_vllm.py" "$d" --out "$R/$key/serve" --modes serve --only stock $flags > "$R/$key/serve/check.txt" 2>&1
    note "$repo$tag: stock at vLLM's defaults: exit $?: $("$PY" -c 'import json, sys; r = json.load(open(sys.argv[1]))["runs"]["stock-serve"]; print(r.get("error") or "weights %s GiB, KV %s tokens, %s" % (r.get("weights_gib"), r.get("kv_tokens"), ", ".join("%s %.1f tok/s" % (k, v["tokens_per_s"]) for k, v in r["bench"].items())))' "$R/$key/serve/check_fp8_vllm.json" 2>&1 | tail -1)"
  fi
  rm -rf "$R/$key"/cache-*
  case " ${Q8_MODELS#*"$item"} " in *" $repo|"*) ;; *) rm -rf "$d";; esac  # (the checkpoint kept while a later item runs it again)
done
step "done in $(el) s"
note "done"
touch "$R/ALLDONE"
