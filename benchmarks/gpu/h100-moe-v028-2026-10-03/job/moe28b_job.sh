#!/usr/bin/env bash
# Glyd's mixture-of-experts serving on one H100 (the moe-v028 branch with an H100's settings built into its code: the wgmma products from 17
# tokens a step with the tile by the pairs an expert has, m-tiles with no pairs left out, the tiled GEMM from 96 pairs an expert, the decoded route from
# 8,192 tokens, the router and the attention products of a mixture of experts' model left bf16), unattended, about 24 minutes (30 at most), the model downloaded included. In ~: this, and
# moe28b_src.tar (the branch's gpu/ and glyd-gpu-py/, job/ the tables' scripts, COMMIT). Private: its results are not for a public file. Its steps:
#   env     machine, uv, a venv with vLLM and pip's nvcc for torch's CUDA; the model downloading meanwhile (the layer's shards first, then all of it)
#   build   the branch's library for this GPU (about 2 minutes), beside bf16's serve (no library in it), which starts once the model is here
#   check   moe_layer.py --check: the experts' products of the model's own layer against an fp32 product with the settings built in, the wgmma products'
#           tile forced to each size, a refused product made an error, uneven loads, and a layer with exceptions by the dozen a step, three processes of their
#           own (120, 90 and 90 s at most); the Glyd serves run only if it passes (if the built-in settings fail it, the products are checked once more with
#           GLYD_GPU_MOE_WG=768, where the last job started the wgmma ones, and the serves run with that, noted in steps.txt)
#   serve   vllm bench serve (gpu/vllm/bench_serve.sh, a seed a pass, distinct prompts), 64 prompts at 1 a second and 256 at once, the servers logging each
#           iteration (--enable-logging-iteration-details: the tokens, the sequences and the milliseconds of each step, the KV cache's use): bf16; Glyd
#           (glyd-picked: the built-in settings, vLLM's own KV cache); Glyd with the lossless KV cache (glyd-picked+kv: its tables for the model made at its first
#           start, the engine's child process running the model on a bundled text: a few minutes, in that server's start; run first, a server of `glyd serve` on an
#           H100 has it by default); bf16 again last, so that a machine that drifts during the run shows as the two bf16 passes apart. The summary opens with whether the
#           modes can be compared: NOT COMPARABLE where a saturated pass's median SM clock (over its own seconds) is under 1,700 MHz, the passes' medians differ by
#           more than 5%, a pass was over 85 C, or bf16's two passes differ by more than 5% in requests a second; then each mode's requests a second and token times
#           against bf16's with the clocks beside them, then where the saturated pass's time went by kind of step.
#   sweep   moe_layer.py (CUDA graphs, the layer's weights from memory as in a step through the layers) by tokens a step, 16 to 8,192: vLLM's own bf16
#           layer, the built-in settings, the other kernels (the wgmma products off), and the wgmma tile forced to each size: where the built-in tile rule
#           and starts are right (a table of the rule's tile against each forced one), and what the decoded route costs beside them
#   dense   dense_layer.py: Qwen3-30B-A3B's qkv, o and router products at 1-512 tokens, each route of the library (the small-M kernel, the wgmma kernel,
#           the matrix unpacked then cuBLAS) beside cuBLAS: the mid-size products' route, which an H100 takes for 17-1,024 tokens
#   prof    (not in the default: M28_PARTS="prof", about 7 minutes with the env and the build) Nsight Compute (installed from NVIDIA's apt repository where the
#           machine has none) on the grouped wgmma products' kernels at 232 (a step of the saturated pass's decode) and 8,192 tokens (its prompt steps): the reports and their raw metrics, the compute and memory throughputs
# One directory a part under ~/results, a summary.txt; results/DONE from the exit trap however the job ends, results/ALLDONE when it ran to its end.
#   bash ~/moe28b_job.sh
#   M28_PARTS="check sweep dense" bash ~/moe28b_job.sh        the layer's numbers alone, about 9 minutes
# Env: M28_PARTS (default: check serve sweep dense, run in that order), M28_SERVE_MODES ("kv picked again": the Glyd modes, the one run first the one this job is for, then bf16 once more; "dense" too: the attention products packed, as an H100 does not by itself), M28_MODEL (Qwen/Qwen3-30B-A3B: its layer
# M28_LAYER, 0, and the serve), M28_SERVE_MODEL (the serve's model where the GPU is under 70 GB: allenai/OLMoE-1B-7B-0924), M28_END (1800 s: the hard stop),
# M28_SWEEP_T (the sweep's tokens a step), M28_RATES ("1 inf") and M28_PROMPTS ("64 256"), M28_SERVE_MAX (a mode's seconds at most: 700; the kv mode 1000), M28_NO_DL (1: the models are in HF_HOME),
# M28_VENV (a venv with vLLM already: nothing installed; M28_CU its CUDA directory with bin/nvcc), VJ_VLLM (vllm==0.30.0), M28_REBUILD (1: the library built again where this tree's is in W already), M28_GLYD_ENV ("VAR=V ...": the Glyd servers' environment, e.g. GLYD_GPU_MOE_WG=768 for the earlier start of the wgmma products), M28_LOCK_CLOCK (MHz: the SM clock locked for the job, reset at its end: for a GPU that throttles), M28_THERMAL (abort, the default: the job ends if 40 s of matmuls on the idle GPU, M28_PREFLIGHT_S, give it a thermal slowdown or take it to M28_HOT_C, 84 C; warn: goes on; off: not run), M28_PROF_T and M28_PROF_K (the prof part), M28_CHECK_EXTRA (a variant more in the check, for testing the job); FILES (~), R (~/results), W (~/m28w),
# HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/m28w}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W/tmp"
T0=$(date +%s)
END=${M28_END:-1800}
PARTS=${M28_PARTS:-"check serve sweep dense"}
MODEL=${M28_MODEL:-Qwen/Qwen3-30B-A3B}
LAYER=${M28_LAYER:-0}
RATES=${M28_RATES:-"1 inf"}
PROMPTS=${M28_PROMPTS:-"64 256"}
exec > >(tee -a "$R/job.log") 2>&1
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
want() { case " $PARTS " in *" $1 "*) return 0 ;; esac; return 1; }
note() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; echo "   $*"; }
fail() { note "FAIL: $*"; exit 1; }
( sleep $(( END + 60 )); grep -qF -- "$0" /proc/$$/cmdline 2> /dev/null || exit
  echo "hard stop at +$(el) s" >> "$R/steps.txt"; kill -TERM $$; pkill -TERM -P $$ ) > /dev/null 2>&1 &
WD=$!
DLPID=
CLOCKLOCKED=0
export M28_JOB=$$
finish() {
  [ "${CLOCKLOCKED:-0}" = 1 ] && sudo -n nvidia-smi -rgc > /dev/null 2>&1
  { pkill -TERM -P "$WD"; kill "$WD"; } 2> /dev/null; [ -n "$DLPID" ] && { pkill -TERM -P "$DLPID"; kill "$DLPID"; } 2> /dev/null
  touch "$R/DONE"
  for f in /proc/[0-9]*/environ; do p=${f#/proc/}; p=${p%/environ}; [ "$p" = "$$" ] || { { tr '\0' '\n' < "$f"; } 2> /dev/null | grep -qx "M28_JOB=$$" && kill -9 "$p" 2> /dev/null; }; done
}
trap finish EXIT

echo "== expected minutes (the whole job: $(( END / 60 )) at most): env and downloads 2, build 2 beside bf16's serve (5), check 1, Glyd's serve with the KV cache 7 and without 5, bf16 again 4, sweep 1, dense 0"
step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; lscpu | grep -E "Model name|Architecture"; nproc
  free -g | head -2; df -h "$HOME" | tail -1; date -u; echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}"; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAMEGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
SERVE_MODEL=$MODEL
[ "${MIB:-0}" -lt 70000 ] && SERVE_MODEL=${M28_SERVE_MODEL:-allenai/OLMoE-1B-7B-0924}
GB=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
[ "${GB:-0}" -ge 200 ] || echo "WARNING: $GB GB free in ~, 200 wanted (the model, the venv and the caches)"
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/moe28b_src.tar" || fail "no moe28b_src.tar in $FILES"
[ -f "$W/src/gpu/moe_layer.py" ] && [ -f "$W/src/gpu/dense_layer.py" ] && [ -f "$W/src/job/sweep_summary.py" ] || fail "moe28b_src.tar lacks gpu/moe_layer.py, gpu/dense_layer.py or job/sweep_summary.py"
cp "$W/src/COMMIT" "$R/COMMIT"
echo "$NAMEGPU (compute $CC, sm_$ARCH, $MIB MiB), $(uname -m) host, $(nproc) CPUs; tree $(cat "$W/src/COMMIT"); parts $PARTS; model $MODEL, serve $SERVE_MODEL; rates $RATES" | tee "$R/machine-short.txt"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -gt 0 ] && echo "WARNING: other processes on the GPU" | tee -a "$R/machine-short.txt"
if [ -n "${M28_LOCK_CLOCK:-}" ]; then  # (a GPU that throttles with its temperature runs the modes at different clocks: one clock for all of them, reset at the end)
  if sudo -n nvidia-smi -lgc "$M28_LOCK_CLOCK,$M28_LOCK_CLOCK" > "$R/log/lock-clock.txt" 2>&1; then CLOCKLOCKED=1; echo "the SM clock locked at $M28_LOCK_CLOCK MHz" >> "$R/machine-short.txt"; else echo "the clock lock failed (log/lock-clock.txt)" >> "$R/machine-short.txt"; fi
fi
python3 "$W/src/job/sweep_summary.py" selftest > "$R/log/selftest-summary.txt" 2>&1 || fail "the tables' script fails its self-test (log/selftest-summary.txt)"
note "machine"

step "uv, the venv, the downloads"
t=$(date +%s)
UV=$(command -v uv || ls "$HOME/tools/uv/uv" "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ] && [ -z "${M28_VENV:-}" ]; then
  mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv
fi
[ -x "${UV:-}" ] || [ -n "${M28_VENV:-}" ] || fail "no uv"
export HF_HOME=${HF_HOME:-$HOME/hf} TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=${UV_CACHE_DIR:-$W/uvcache}
if [ -n "${M28_VENV:-}" ]; then
  W_VENV=$M28_VENV; PY=$W_VENV/bin/python; CU=${M28_CU:?M28_VENV needs M28_CU, its CUDA directory with bin/nvcc}
  echo "the venv $M28_VENV, as it is" > "$R/log/env-vllm.txt"
else
  W_VENV=$W/venv
  if [ -x "$W/venv/bin/python" ] && "$W/venv/bin/python" -c "import vllm" 2> /dev/null; then
    echo "an earlier job's venv, $W/venv" > "$R/log/env-vllm.txt"
  else
    { rm -rf "$W/venv" && "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "${VJ_VLLM:-vllm==0.30.0}"; } > "$R/log/env-vllm.txt" 2>&1 || fail "vLLM did not install (log/env-vllm.txt)"
  fi
  PY=$W/venv/bin/python
  MM=$("$PY" -c "import torch; print(torch.version.cuda)")
  "$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || fail "nvcc did not install (log/env-nvcc.txt)"
  CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
  mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
fi
export CUDA_HOME=$CU PATH=$W_VENV/bin:$CU/bin:$PATH
# The plugin's entry point from the source tree: a dist-info naming it (vLLM finds plugins by entry point), the package itself from PYTHONPATH
if [ "$("$PY" -c "from importlib.metadata import entry_points; print(any(e.name == 'glyd' for e in entry_points(group='vllm.general_plugins')))" 2> /dev/null)" != True ]; then
  SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")
  mkdir -p "$SP/glyd_gpu_src-0.0.0.dist-info"
  printf 'Metadata-Version: 2.1\nName: glyd-gpu-src\nVersion: 0.0.0\n' > "$SP/glyd_gpu_src-0.0.0.dist-info/METADATA"
  printf '[vllm.general_plugins]\nglyd = glyd_gpu.vllm_entry:register\n' > "$SP/glyd_gpu_src-0.0.0.dist-info/entry_points.txt"
fi
export PYTHONPATH=$W/src/glyd-gpu-py PYTHONDONTWRITEBYTECODE=1
"$PY" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__, '| GPU', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1 || fail "no vLLM on the GPU ($(tail -2 "$R/env.txt" | tr '\n' ' '))"
echo "nvcc: $(nvcc --version | tail -1)" >> "$R/env.txt"
cat "$R/env.txt"
HFPY=$PY
thermal_preflight() {  # the idle GPU under bf16 matmuls for M28_PREFLIGHT_S seconds (40), beside the build: a machine that cannot cool it throttles within seconds, and the modes of a job on it run at different clocks
  local d="$R/log" smi bad
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.sw_power_cap --format=csv,noheader,nounits -l 1 > "$d/preflight-smi.csv" 2> /dev/null &
  smi=$!
  timeout 120 "$PY" -c "
import time, torch
a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
b = torch.randn_like(a)
t = time.time()
while time.time() - t < ${M28_PREFLIGHT_S:-40}:
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
" > "$d/preflight.txt" 2>&1
  kill "$smi" 2> /dev/null
  python3 "$W/src/job/sweep_summary.py" preflight "$d/preflight-smi.csv" "${M28_HOT_C:-84}" > "$d/preflight-verdict.txt" 2>&1 && bad=0 || bad=1
  [ "$bad" = 1 ] && touch "$d/preflight.bad"
  note "thermal preflight: $(cat "$d/preflight-verdict.txt")"
  echo done >> "$d/preflight-verdict.txt"
}
rm -f "$R/log/preflight.bad" "$R/log/preflight-verdict.txt"
[ "${M28_THERMAL:-abort}" != off ] && { thermal_preflight & }
dl() {  # REPO LAYER WHOLE: the repo's configs and index and the shards holding the layer's experts (WHOLE 0), or all of it (1); log/dl-NAME-layer.txt or -whole.txt its time and size
  local n t tag; n=$(basename "$1"); t=$(date +%s); tag=$([ "$3" = 1 ] && echo whole || echo layer)
  timeout 1500 "$HFPY" - "$@" > "$W/$n.$tag.dir" 2> "$R/log/dl-$n-$tag.txt" << 'EOF'
import json, sys
from huggingface_hub import hf_hub_download, snapshot_download
repo, layer, whole = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
try:
    wm = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    files = sorted({f for k, f in wm.items() if f"layers.{layer}." in k and ("experts" in k or "block_sparse_moe" in k)})
except Exception:
    files = []
for f in files:
    hf_hub_download(repo, f)
if whole:
    print(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))
else:
    print(snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"]))
EOF
  local e=$?; echo "exit $e: $(du -sbL "$(tail -1 "$W/$n.$tag.dir")" 2> /dev/null | cut -f1) bytes in $(( $(date +%s) - t )) s (at +$(el) s)" >> "$R/log/dl-$n-$tag.txt"
}
if [ "${M28_NO_DL:-0}" != 1 ]; then
  ( dl "$MODEL" "$LAYER" 0
    if want serve; then dl "$SERVE_MODEL" "$LAYER" 1; fi ) &
  DLPID=$!
fi
note "env: $(( $(date +%s) - t )) s ($(head -1 "$R/env.txt" | cut -c1-120))"

step "build: the library for sm_$ARCH (bf16's serve beside it)"
t=$(date +%s)
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
mkdir -p "$W/lib/new"
NEW=$W/lib/new/libglyd_gpu_cuda$MAJOR.so
BPID=
if [ -f "$NEW" ] && cmp -s "$W/src/COMMIT" "$W/lib/new/COMMIT" && [ -z "${M28_REBUILD:-}" ]; then  # (an earlier run of this job on this machine built this tree's library)
  echo "the library of this tree is built already (M28_REBUILD=1: again)" > "$R/log/build-new.txt"
else
  rm -f "$W/lib/new/COMMIT"
  { timeout "$(tmo 900)" nvcc "${F[@]}" -c -o "$W/lib/new/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
    timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$NEW" "$W/lib/new/glyd_gpu.o" && cp "$W/src/COMMIT" "$W/lib/new/COMMIT"; } > "$R/log/build-new.txt" 2>&1 &
  BPID=$!
fi

# FlashInfer builds its sampling kernels with nvcc the first time a server samples a token (a minute and more on a fresh instance, in that server's start): done now, beside the build
FIPID=
if want serve && [ "${M28_NO_FIJIT:-0}" != 1 ]; then
  ( timeout 300 "$PY" -c "
import torch, flashinfer
x = torch.randn(8, 4096, device='cuda')
flashinfer.sampling.top_k_top_p_sampling_from_logits(x, torch.full((8,), 20, dtype=torch.int32, device='cuda'), torch.full((8,), 0.95, device='cuda'), deterministic=True)
torch.cuda.synchronize()
print('flashinfer: the sampling kernels are built')
" ) > "$R/log/flashinfer-jit.txt" 2>&1 &
  FIPID=$!
fi

# ---- serve ----
SERVE_LOGGING="--enable-logging-iteration-details"
# vLLM's log lines with milliseconds (its own format has seconds): a step's time is the time between its iteration line and the next one's (the iteration line's own 'elapsed time' is the engine's wait for the result, which async scheduling shortens)
cat > "$W/vllm-logging.json" << 'EOF'
{"version": 1, "disable_existing_loggers": false,
 "formatters": {"vllm": {"class": "vllm.logging_utils.NewLineFormatter", "datefmt": "%m-%d %H:%M:%S", "format": "%(levelname)s %(asctime)s.%(msecs)03d [%(fileinfo)s:%(lineno)d] %(message)s"}},
 "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm", "level": "INFO", "stream": "ext://sys.stdout"}},
 "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false}}}
EOF
serve_mode() {  # NAME BENCH_MODE LIB "VAR=V ..." [SECONDS AT MOST]: vllm bench serve of one mode (its server started once: the compile cache shared by the modes of a kind)
  local name=$1 bm=$2 lib=$3 envs=$4 max=${5:-${M28_SERVE_MAX:-700}} d="$R/serve/$1" t e smi
  mkdir -p "$d"
  t=$(date +%s)
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader,nounits -l 1 > "$d/smi-reasons.csv" 2> /dev/null &
  smi=$!
  ( cd "$R" && env $envs ${lib:+GLYD_GPU_LIB=$lib} VLLM="$W_VENV/bin/vllm" R="$d" VLLM_CACHE_ROOT="$W/cache/$bm" VLLM_LOGGING_CONFIG_PATH="$W/vllm-logging.json" MODES="$bm" WARM=0 RATES="$RATES" PROMPTS="$PROMPTS" IN=1024 OUT=256 UTIL=0.9 \
      SERVE_ARGS="$SERVE_LOGGING" BUSYWAIT=60 COOL=60 COOLWAIT=60 timeout "$(tmo "$max")" bash "$W/src/gpu/vllm/bench_serve.sh" "$SERVE_MODEL" ) >> "$d/bench.txt" 2>&1
  e=$?
  kill "$smi" 2> /dev/null
  note "serve $name: exit $e in $(( $(date +%s) - t )) s"
  grep -q "did not start\|not run" "$d/bench.txt" && { note "serve $name: no results ($(grep -h "did not start\|not run" "$d/bench.txt" | head -1 | cut -c1-120))"; e=1; }
  return $e
}
serve_summ() {
  { echo "$SERVE_MODEL on $NAMEGPU (tree $(cat "$W/src/COMMIT")); vllm bench serve, 1,024 tokens in and 256 out, rates $RATES, prompts $PROMPTS (a seed a pass)"; echo
    for m in bf16 glyd-picked glyd-picked+kv glyd-picked+kv+packed-dense bf16-2; do
      [ -d "$R/serve/$m" ] || continue
      echo "######## $m"
      python3 "$W/src/gpu/vllm/bench_summary.py" "$R/serve/$m" 2> /dev/null | grep -v "^$" | head -40
      echo
    done; } > "$R/serve/summary.txt" 2> /dev/null
  local args=() m f
  for m in "bf16 serve-bf16.txt" "glyd-picked serve-glyd.txt" "glyd-picked+kv serve-kv+glyd.txt" "glyd-picked+kv+packed-dense serve-kv+glyd.txt" "bf16-2 serve-bf16.txt"; do
    f="$R/serve/${m% *}/${m#* }"; [ -f "$f" ] && args+=("${m% *}=$f")
  done
  [ ${#args[@]} -gt 0 ] && python3 "$W/src/job/sweep_summary.py" steps "${args[@]}" > "$R/serve/steps.txt" 2> /dev/null
  python3 "$W/src/job/sweep_summary.py" clocks "$R/serve" "$R/serve/summary.txt" > "$R/serve/clocks.txt" 2> /dev/null  # (the saturated passes' clocks and temperatures, and whether the modes can be compared)
  grep -h "glyd" "$R/serve/glyd-picked+kv/serve-kv+glyd.txt" 2> /dev/null | grep -i "kv\|tables\|calibrat" | head -30 > "$R/serve/kv-notes.txt"
}
BFPID=
thermal_verdict() {  # 0 where the preflight found the GPU holding its clock (or was not asked), after it has ended
  [ "${M28_THERMAL:-abort}" = off ] && return 0
  until grep -q "^done" "$R/log/preflight-verdict.txt" 2> /dev/null; do [ "$(el)" -ge "$END" ] && return 0; sleep 2; done
  [ ! -f "$R/log/preflight.bad" ] || [ "${M28_THERMAL:-abort}" = warn ]
}
bf16_early() {  # bf16's server needs no library: it starts when the model and FlashInfer's kernels are here, beside the build
  thermal_verdict || return 1
  [ -n "$FIPID" ] && while kill -0 "$FIPID" 2> /dev/null; do sleep 2; done
  if [ "${M28_NO_DL:-0}" != 1 ]; then
    local n; n=$(basename "$SERVE_MODEL")
    until grep -q "^exit" "$R/log/dl-$n-whole.txt" 2> /dev/null; do [ "$(el)" -ge "$END" ] && return 1; sleep 3; done
    grep -q "^exit 0" "$R/log/dl-$n-whole.txt" || { note "serve: no $SERVE_MODEL ($(tail -1 "$R/log/dl-$n-whole.txt" 2> /dev/null))"; return 1; }
  fi
  serve_mode bf16 bf16 "" ""
  serve_summ
}
if want serve; then
  bf16_early &
  BFPID=$!
fi
[ -n "$BPID" ] && wait "$BPID"
thermal_verdict || fail "this GPU throttles under load (log/preflight-verdict.txt): the modes would run at different clocks. Another machine, or M28_LOCK_CLOCK=MHz (a clock it holds), or M28_THERMAL=warn to go on"
[ -f "$NEW" ] || fail "the library did not build (log/build-new.txt)"
export GLYD_GPU_LIB=$NEW
"$PY" -c "from glyd_gpu import _lib, kernels as g; g.lib(); print('library: C API', _lib._lib.glyd_gpu_api_version(), '| GPU code', _lib.gpu())" >> "$R/env.txt" 2>&1 || fail "the library does not load ($(tail -1 "$R/env.txt"))"
note "build: sm_$ARCH in $(( $(date +%s) - t )) s ($(tail -1 "$R/env.txt"))"
[ -n "$BFPID" ] && { wait "$BFPID"; BFPID=; }  # (the GPU is bf16's server's until it has run)

# The layer's shards (and the config) before the layer's runs
layer_ready() {  # 0 once the model's index and the layer's shards are downloaded
  [ "${M28_NO_DL:-0}" = 1 ] && return 0
  local n; n=$(basename "$MODEL")
  until grep -q "^exit" "$R/log/dl-$n-layer.txt" 2> /dev/null; do [ "$(el)" -ge "$END" ] && return 1; sleep 3; done
  grep -q "^exit 0" "$R/log/dl-$n-layer.txt"
}
LAYER_ARGS=(--hf "$MODEL" --layer "$LAYER" --layout mma12)
[ "$CC" = "8.9" ] && LAYER_ARGS=(--hf "$MODEL" --layer "$LAYER")  # (Ada: its own layout, as the plugin takes it)
export HF_HUB_OFFLINE=1

# ---- check ----
CHECK_VARIANTS="new:;off:GLYD_GPU_MOE_WG=0;nt16:GLYD_GPU_MOE_WG_NT=16;nt32:GLYD_GPU_MOE_WG_NT=32;nt64:GLYD_GPU_MOE_WG_NT=64;nt128:GLYD_GPU_MOE_WG_NT=128;strict:GLYD_GPU_MOE_WG_STRICT=1${M28_CHECK_EXTRA:+;$M28_CHECK_EXTRA}"
CHECKED=0
part_check() {
  local d="$R/check"; mkdir -p "$d"
  step "check: the experts' products of $MODEL's layer $LAYER with the built-in settings against an fp32 product, three processes of 120, 90 and 90 s at most"
  layer_ready || { note "check: no $MODEL layer"; touch "$d/DONE"; return; }
  local t=$(date +%s) e
  ( cd "$W/src/gpu" && timeout 120 "$PY" -u moe_layer.py "${LAYER_ARGS[@]}" --tokens "1 8 17 33 64 232 384 512 1152 4096" --check --no-time --variants "$CHECK_VARIANTS" ) > "$d/check.txt" 2>&1
  e=$?
  if [ "$e" = 0 ]; then  # (uneven loads: an expert with many chunks of pairs)
    ( cd "$W/src/gpu" && timeout 90 "$PY" -u moe_layer.py "${LAYER_ARGS[@]}" --skew 1.5 --tokens "64 232 1152 4096" --check --no-time --variants "new:;nt32:GLYD_GPU_MOE_WG_NT=32,GLYD_GPU_MOE_WG_STRICT=1" ) >> "$d/check.txt" 2>&1
    e=$?
  fi
  if [ "$e" = 0 ]; then  # (exceptions by the dozen a step: the copies of a stage's exceptions past what its slot holds are read from memory)
    ( cd "$W/src/gpu" && timeout 90 "$PY" -u moe_layer.py --synthetic 64,8,2048,768 --wild 0.02 --layout mma12 --tokens "17 232 1152" --check --no-time --variants "new:;strict:GLYD_GPU_MOE_WG_STRICT=1" ) >> "$d/check.txt" 2>&1
    e=$?
  fi
  note "check: exit $e in $(( $(date +%s) - t )) s, $(grep -c checked "$d/check.txt") checks ($(grep -m1 "FAILED\|Error\|error" "$d/check.txt" | cut -c1-160))"
  if [ "$e" != 0 ] && [ -z "${M28_GLYD_ENV:-}" ]; then  # (the built-in settings failed: the wgmma products from 768 tokens, where the last job started them, checked once more, so that the serves are not lost)
    ( cd "$W/src/gpu" && GLYD_GPU_MOE_WG=768 timeout 90 "$PY" -u moe_layer.py "${LAYER_ARGS[@]}" --tokens "1 8 17 33 64 232 512 1152 4096" --check --no-time --variants "new:" ) > "$d/check-wg768.txt" 2>&1
    if [ $? = 0 ]; then M28_GLYD_ENV="GLYD_GPU_MOE_WG=768"; e=0; note "check: the built-in settings failed (check/check.txt); with GLYD_GPU_MOE_WG=768 the products pass (check/check-wg768.txt): the Glyd serves run with it"; fi
  fi
  if [ "$e" = 0 ]; then touch "$d/PASSED"; CHECKED=1; else touch "$d/BAD"; [ "$e" = 124 ] && note "check: TIMED OUT (a kernel that does not end); last line: $(tail -1 "$d/check.txt" | cut -c1-160)"; fi
  ( cd "$W/src/glyd-gpu-py/tests" && timeout 60 "$PY" -c "import test_gpu; test_gpu.test_moe_wg_workspace()" ) > "$d/wgws.txt" 2>&1  # (the sizes the library gives the wgmma route, against what it needs: informational)
  note "check: the wgmma workspace's test $([ $? = 0 ] && echo passed || echo FAILED) ($(tail -1 "$d/wgws.txt" | cut -c1-140))"
  touch "$d/DONE"
}

# ---- sweep ----
SWEEP_T=${M28_SWEEP_T:-"16 24 32 48 64 128 232 384 512 768 1152 2048 4096 8192"}
SWEEP_VARIANTS="new:;off:GLYD_GPU_MOE_WG=0;nt16:GLYD_GPU_MOE_WG_NT=16;nt32:GLYD_GPU_MOE_WG_NT=32;nt64:GLYD_GPU_MOE_WG_NT=64;nt128:GLYD_GPU_MOE_WG_NT=128"
part_sweep() {
  local d="$R/sweep" t e
  mkdir -p "$d"
  step "sweep: the layer by tokens a step, vLLM's bf16 layer beside it: the built-in settings, the other kernels, each wgmma tile"
  layer_ready || { note "sweep: no $MODEL layer"; touch "$d/DONE"; return; }
  t=$(date +%s)
  ( cd "$W/src/gpu" && timeout "$(tmo 400)" "$PY" -u moe_layer.py "${LAYER_ARGS[@]}" --tokens "$SWEEP_T" --bf16 --variants "$SWEEP_VARIANTS" --json "$d/layer.json" ) > "$d/layer.txt" 2>&1
  e=$?
  { echo "$MODEL's layer $LAYER on $NAMEGPU (tree $(cat "$W/src/COMMIT")): GPU time a layer under CUDA graphs, us (new: the built-in settings; off: the wgmma products off; ntN: their tile forced to N pairs)"; head -1 "$d/layer.txt" | cut -c1-260; echo
    python3 "$W/src/job/sweep_summary.py" table "$d/layer.json"
    echo; echo "The wgmma tile the built-in rule takes against each forced tile (us a layer):"; python3 "$W/src/job/tile_verdict.py" "$d/layer.json"; } > "$d/summary.txt" 2> "$d/summary.err"
  note "sweep: exit $e in $(( $(date +%s) - t )) s"
  touch "$d/DONE"
}

# ---- dense ----
part_dense() {
  local d="$R/dense" t e
  mkdir -p "$d"
  step "dense: the dense products of a step, each route of the library against cuBLAS (Qwen3-30B-A3B's qkv, o and router shapes)"
  t=$(date +%s)
  ( cd "$W/src/gpu" && timeout "$(tmo 240)" "$PY" -u dense_layer.py --shapes "5120,2048;2048,4096;128,2048" --tokens "1 8 16 32 64 128 232 512" --routes "default gemm wg decode" --layout mma12 --json "$d/dense.json" ) > "$d/dense.txt" 2>&1
  e=$?
  python3 "$W/src/job/sweep_summary.py" dense table "$d/dense.json" > "$d/summary.txt" 2> "$d/summary.err"
  note "dense: exit $e in $(( $(date +%s) - t )) s"
  touch "$d/DONE"
}

# ---- prof (opt-in: M28_PARTS="prof", about 7 minutes with the env and the build) ----
part_prof() {  # Nsight Compute on the grouped products' kernels of the model's layer at 232 and 8,192 tokens (M28_PROF_T; M28_PROF_K: the kernels' name, mma12_moe_wg): the report files and their raw metrics
  local d="$R/prof" ncu t=$(date +%s) T pkg
  mkdir -p "$d"
  step "prof: Nsight Compute on $MODEL's layer $LAYER, kernels ${M28_PROF_K:-mma12_moe_wg}"
  layer_ready || { note "prof: no $MODEL layer"; touch "$d/DONE"; return; }
  ncu=$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2> /dev/null | tail -1)
  if [ -z "$ncu" ] && sudo -n true 2> /dev/null; then  # (the CUDA apt repository: no login)
    { . /etc/os-release
      wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/$(uname -m | sed 's/aarch64/sbsa/')/cuda-keyring_1.1-1_all.deb" -O "$W/keyring.deb" && sudo -n dpkg -i "$W/keyring.deb" && sudo -n apt-get update -qq &&
        pkg=$(apt-cache search '^cuda-nsight-compute-[0-9]' | awk '{print $1}' | sort -V | tail -1) && echo "installing $pkg" && sudo -n apt-get install -y -qq "$pkg"; } > "$d/install.txt" 2>&1
    ncu=$(ls /usr/local/cuda*/bin/ncu 2> /dev/null | tail -1)
  fi
  [ -n "$ncu" ] || { note "prof: no ncu on this machine and none installed (prof/install.txt)"; touch "$d/DONE"; return; }
  echo "$ncu: $("$ncu" --version | tail -1)" > "$d/ncu-version.txt"
  for T in ${M28_PROF_T:-232 8192}; do  # (as root: the counters are the administrators')
    ( cd "$W/src/gpu" && sudo -n env "PATH=$PATH" "HOME=$HOME" "PYTHONPATH=$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 "GLYD_GPU_LIB=${GLYD_GPU_LIB:-}" "HF_HOME=$HF_HOME" HF_HUB_OFFLINE=1 "CUDA_HOME=$CUDA_HOME" \
        timeout "$(tmo 300)" "$ncu" --set full -k "regex:${M28_PROF_K:-mma12_moe_wg}" -c 2 -f -o "$d/wg-$T" "$PY" -u moe_layer.py "${LAYER_ARGS[@]}" --tokens "$T" --check --no-time --variants "new:" ) > "$d/ncu-$T.txt" 2>&1
    sudo -n chown "$(id -u):$(id -g)" "$d"/wg-"$T".ncu-rep 2> /dev/null
    [ -f "$d/wg-$T.ncu-rep" ] && { "$ncu" -i "$d/wg-$T.ncu-rep" --page details > "$d/details-$T.txt" 2>&1; "$ncu" -i "$d/wg-$T.ncu-rep" --page raw --csv > "$d/raw-$T.csv" 2>&1; }
  done
  python3 "$W/src/job/ncu_summary.py" "$d"/raw-*.csv > "$d/summary.txt" 2> "$d/summary.err"
  note "prof: ${ncu##*/} in $(( $(date +%s) - t )) s, $(ls "$d"/*.ncu-rep 2> /dev/null | wc -l) reports"
  touch "$d/DONE"
}

part_serve() {  # (bf16's serve ran beside the build)
  local d="$R/serve" modes=${M28_SERVE_MODES:-"kv picked again"}
  mkdir -p "$d"
  step "serve: $SERVE_MODEL, Glyd with the built-in settings, and with the lossless KV cache (bf16's ran beside the build)"
  has() { case " $modes " in *" $1 "*) return 0 ;; esac; return 1; }
  late() { [ $(( END - $(el) )) -lt "$2" ] && { note "serve $1: not started, $(( END - $(el) )) s left"; return 0; }; return 1; }  # (a mode takes 4-6 minutes, the KV cache's 7-9)
  if [ "$CHECKED" != 1 ] && want check; then note "serve: the Glyd modes not run, the check did not pass (check/check.txt)"; touch "$d/DONE"; return; fi
  has kv && { late glyd-picked+kv 600 || { serve_mode glyd-picked+kv kv+glyd "$NEW" "${M28_GLYD_ENV:-}" "${M28_SERVE_KV_MAX:-1000}"; serve_summ; }; }
  has picked && { late glyd-picked 330 || { serve_mode glyd-picked glyd "$NEW" "${M28_GLYD_ENV:-}" "${M28_SERVE_MAX:-700}"; serve_summ; }; }
  has dense && { late glyd-picked+kv+packed-dense 330 || { serve_mode glyd-picked+kv+packed-dense kv+glyd "$NEW" "GLYD_MOE_DENSE=packed ${M28_GLYD_ENV:-}" "${M28_SERVE_MAX:-700}"; serve_summ; }; }  # (the attention products packed as well: the A/B of the built-in setting)
  has again && { late bf16-2 330 || { serve_mode bf16-2 bf16 "" "" "${M28_SERVE_MAX:-700}"; serve_summ; }; }  # (bf16 again, last: the machine drifting during the run shows as the two bf16 passes apart)
  [ -s "$d/clocks.txt" ] && note "serve: $(head -1 "$d/clocks.txt" | cut -c1-400)"
  touch "$d/DONE"
}

summ() {  # results/summary.txt: the machine, the steps, each part's summary
  { cat "$R/machine-short.txt" 2> /dev/null; [ -s "$R/serve/clocks.txt" ] && { echo; echo "################ $(head -1 "$R/serve/clocks.txt")"; }; echo; cat "$R/steps.txt" 2> /dev/null
    [ -s "$R/serve/clocks.txt" ] && { echo; echo "################ can the modes be compared? the SM clock and temperature of each saturated pass, over the seconds its server's iteration lines cover"; cat "$R/serve/clocks.txt"; }
    [ -f "$R/serve/summary.txt" ] && { echo; echo "################ serve, each mode against bf16 (the tables below)"; python3 "$W/src/job/sweep_summary.py" serve "$R/serve/summary.txt" "$R/serve" 2> /dev/null; }
    [ -s "$R/serve/steps.txt" ] && { echo; echo "################ where the saturated pass's time went (the servers' iteration lines)"; cat "$R/serve/steps.txt"; }
    [ -s "$R/serve/kv-notes.txt" ] && { echo; echo "################ the KV cache's lines in its server's log"; cat "$R/serve/kv-notes.txt"; }
    for f in "$R"/check/check.txt; do [ -f "$f" ] && { echo; echo "################ check"; grep -c checked "$f" | sed 's/$/ checks/'; tail -4 "$f"; }; done
    for f in "$R"/sweep/summary.txt "$R"/dense/summary.txt "$R"/prof/summary.txt "$R"/serve/summary.txt; do [ -f "$f" ] && { echo; echo "################ ${f#"$R"/}"; cat "$f"; }; done
  } > "$R/summary.tmp" 2>&1; mv "$R/summary.tmp" "$R/summary.txt"
}
want check && { part_check; summ; }
want serve && { if [ $(( END - $(el) )) -lt 330 ]; then note "serve: not started, $(( END - $(el) )) s left"; else part_serve; summ; fi; }  # (the serves before the layer's tables: the time that is short is theirs to lose last)
want sweep && { if [ $(( END - $(el) )) -lt 120 ]; then note "sweep: not started, $(( END - $(el) )) s left"; else part_sweep; summ; fi; }
want dense && { if [ $(( END - $(el) )) -lt 120 ]; then note "dense: not started, $(( END - $(el) )) s left"; else part_dense; summ; fi; }
want prof && { if [ $(( END - $(el) )) -lt 200 ]; then note "prof: not started, $(( END - $(el) )) s left"; else part_prof; summ; fi; }
for f in "$R"/log/dl-*.txt; do [ -f "$f" ] && echo "download $(basename "$f" .txt): $(tail -1 "$f")"; done >> "$R/steps.txt"
step "done in $(el) s"
note "done"
summ
touch "$R/ALLDONE"
