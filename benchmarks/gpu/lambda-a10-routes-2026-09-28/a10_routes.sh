#!/usr/bin/env bash
# A prompt's routes on an A10 (sm_86, 150 W, full-rate tensor cores), Qwen3-8B in the 12-bit layout: per layer
# (route_layer.py: each product alone after an L2 flush, the fused kernel's variants, the matrix decoded then cuBLAS;
# a pass of 12 layers by route: fused, decoded, decoded ahead, cuBLAS, with the SM clock and power of each) and end
# to end (e2e.py --prefill 128,512,1024,2048,4096, bf16 beside it in the same process, each route forced: today's
# fused kernel, GLYD_DEC_MIN=65 decoded for cuBLAS, GLYD_AHEAD_MIN=65 decoded ahead), nvidia-smi every 100 ms beside
# each run; then generate()'s step compiled (the static cache, glyd.from_pretrained's default) against eager, by the
# cache's length (loop.py). Unattended: every step has its own timeout, a failed step is logged and the rest go on,
# past BUDGET seconds the runs left are skipped; results in ~/results, DONE last.
#   bash ~/a10_routes.sh     (in ~: this script, glyd-route.tar, route_layer.py, loop.py, summary.py)
# Env, for a smoke test elsewhere: R (~/results), W (~/routes), ENVSH (~/gpuenv/cuda.sh), HF_HOME (~/hf), MS
# (route_layer.py's M), LENGTHS (e2e's prompts), MODEL (Qwen/Qwen3-8B), LAYOUTS ("mma12 mma": the tiered layout per
# layer too, a mixture of experts' attention on an A10), E2E (1), BUDGET (1680).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/routes}
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
MODEL=${MODEL:-Qwen/Qwen3-8B}
MS=${MS:-128,256,384,512,640,768,1024,1536,2048,3072,4096}
LENGTHS=${LENGTHS:-128,512,1024,2048,4096}
LAYOUTS=${LAYOUTS:-"mma12 mma"}
BUDGET=${BUDGET:-1680}  # s: past it the runs left are skipped, the summary and DONE still written
mkdir -p "$R" "$W/log"
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
step() { echo "== $(date +%T) (+$(( $(date +%s) - T0 )) s) $*"; }
left() { [ $(( $(date +%s) - T0 )) -lt "$BUDGET" ] || { echo "over the budget: skipped"; false; }; }
smi() { nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,pstate,clocks_event_reasons.active --format=csv -lms 100 > "$1" 2>&1 & echo $!; }

step "the branch's gpu/ and bindings/python"
rm -rf "$W/glyd" && mkdir -p "$W/glyd" && tar -C "$W/glyd" -xf "$SRC/glyd-route.tar" && cat "$W/glyd/COMMIT" 2>/dev/null
ENVSH=${ENVSH:-$HOME/gpuenv/cuda.sh}
[ -f "$ENVSH" ] || { step "no $ENVSH: setup_env.sh"; timeout 900 bash "$W/glyd/gpu/setup_env.sh" ~/gpuenv > "$W/log/setup.txt" 2>&1; echo "setup exit $?"; }
source "$ENVSH"
python -c "import accelerate" 2>/dev/null || { [ -x ~/tools/uv/uv ] && timeout 120 ~/tools/uv/uv pip install -q --python "$(command -v python)" accelerate; }  # (from_pretrained's device map)
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import hf_transfer" 2>/dev/null || { [ -x ~/tools/uv/uv ] && timeout 120 ~/tools/uv/uv pip install -q --python "$(command -v python)" hf_transfer; }
python -c "import hf_transfer" 2>/dev/null || { echo "no hf_transfer: plain downloads"; unset HF_HUB_ENABLE_HF_TRANSFER; }

step "machine"
{ nvidia-smi; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,power.default_limit --format=csv
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; nvcc --version | tail -2
  python -c "import torch, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'transformers', transformers.__version__)"
  lscpu | grep "Model name"; nproc; free -g | head -2; date -u; } > "$R/machine.txt" 2>&1

step "$MODEL, in the background"
( timeout 900 python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*']))" > "$W/model.dir" 2> "$W/log/download.txt"; echo "download exit $?" >> "$W/log/download.txt" ) &
DL=$!

step "the library (build_lib.sh)"
( cd "$W/glyd/gpu" && timeout 900 bash build_lib.sh "$W/lib" ) > "$W/log/build_lib.txt" 2>&1; echo "build_lib exit $?"
LIB=$(ls "$W"/lib/libglyd_gpu_cuda*.so 2>/dev/null | head -1)
export GLYD_GPU_LIB=$LIB
echo "library: $LIB"

step "waiting for the download"
wait $DL
tail -2 "$W/log/download.txt"
D=$(tail -1 "$W/model.dir" 2>/dev/null)
[ -f "$D/config.json" ] || { echo "NO MODEL: $D"; }

step "per layer"
for layout in $LAYOUTS; do
  left && [ -f "$D/config.json" ] || continue
  ms=$MS; [ "$layout" = mma ] && ms=512,1024,2048,4096
  ( cd "$W/glyd/gpu" && PYTHONPATH="$W/glyd/gpu" timeout 900 python -u "$SRC/route_layer.py" "$D" --layout $layout --M "$ms" --tag "$layout" ) >> "$R/layer.txt" 2>> "$W/log/layer-$layout.txt"
  echo "route_layer $layout exit $?"
done

step "end to end, each route forced (bf16 beside it)"
for route in fused:  decoded:GLYD_DEC_MIN=65 ahead:GLYD_AHEAD_MIN=65; do
  name=${route%%:*}; env_=${route#*:}
  left && [ "${E2E:-1}" = 1 ] || continue
  pid=$(smi "$R/smi-e2e-$name.csv")
  ( cd "$W/glyd/gpu" && echo "route $name: ${env_:-(no knob)}" && env $env_ timeout 900 python -u e2e.py "$MODEL" --format mma12 --fused --merge --baseline --tokens 16 --batch 1 --prefill "$LENGTHS" ) > "$R/e2e-$name.txt" 2>&1
  echo "e2e $name exit $?" | tee -a "$R/e2e-$name.txt"
  kill $pid
done

step "the fast loop: generate()'s step compiled (static cache) against eager, by the cache's length"
for b in 1 8; do
  for mode in static eager; do
    left || continue
    ( cd "$W/glyd/gpu" && GLYD_GPU_LIB=$LIB PYTHONPATH="$W/glyd/bindings/python" timeout 600 python -u "$SRC/loop.py" "$MODEL" $mode $b ) >> "$R/loop.txt" 2>> "$W/log/loop.txt"
    echo "loop $mode $b exit $?"
  done
done

step "summary"
python "$SRC/summary.py" "$R" > "$R/summary.txt" 2> "$W/log/summary.txt"; echo "summary exit $?"
cat "$R/summary.txt"
mkdir -p "$R/log" && cp "$W"/log/*.txt "$R/log/"
step "done in $(( $(date +%s) - T0 )) s"
touch "$R/DONE"
