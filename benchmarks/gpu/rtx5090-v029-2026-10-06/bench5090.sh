#!/usr/bin/env bash
# The RTX 5090 layout question (from the speed bar's job, benchmarks/gpu speed-bar): Glyd as released (install.sh of v0.29.0, `glyd serve MODEL`,
# its defaults: the 12-bit layout on this GPU), the same with the tiered layout forced (GLYD_LAYOUT=mma glyd serve MODEL), and vLLM's own bf16
# (`vllm serve MODEL`, the same vLLM 0.30.0 from Glyd's tool environment), each model on this GPU, one server after the other.
# Each server: `vllm bench serve`, random prompts of 1,000 tokens in and 256 out (--ignore-eos), at 1, 2, 4, ... 256 users
# (--max-concurrency N, every request sent at once), max(8, 3N) prompts a level, a seed of its own a level (the same prompts for both
# servers), after a warm-up of its own seed; the prefix cache's queries and hits from /metrics around each level; nvidia-smi each second.
# Each server starts at its defaults (bf16: vLLM's, with the PyTorch sampler that Glyd's default sets, VLLM_USE_FLASHINFER_SAMPLER=0, so that the two
# sample alike and FlashInfer's sampler is not compiled at the first request). Where one does not start: --max-num-seqs at the most vLLM's own
# message says fits, else the ladder below (fewer sequences, then --max-model-len 4096, then eager); after `--` for Glyd. Each try kept and named.
#   MODELS="Qwen/Qwen3.5-9B" bash job.sh        results in ~/results (summary.txt rewritten after every server), DONE at the end
set -uo pipefail
R=$HOME/results; mkdir -p "$R"; T0=$(date +%s); END=$(( ${MAX_MIN:-140} * 60 ))
exec > >(tee -a "$R/job.log") 2>&1
trap 'pkill -f "glyd serve|vllm.entrypoints|bin/vllm serve|EngineCore" 2> /dev/null; touch "$R/DONE"' EXIT
el() { echo $(( $(date +%s) - T0 )); }
log() { echo "$(date -u +%T) (+$(el) s) $*"; }
LEVELS=(${LEVELS:-1 8 32 64})
export HF_HOME=$HOME/hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN LD_LIBRARY_PATH PYTHONPATH

{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv,noheader; uname -m; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt"

# Where glyd-gpu 0.29.0 is not on PyPI yet: its wheel built here from tag v0.29.0's glyd-gpu-py (glydgpu_src.tar) with the GPU libraries
# the release run built (libglyd_gpu_cuda*.so uploaded beside it), no engine program (inert by default; glyd serve uses vLLM), found by the installer
if [ -f "$HOME/glydgpu_src.tar" ]; then
  log "glyd-gpu's wheel from tag v0.29.0"
  mkdir -p "$HOME/gsrc" "$HOME/glibs" "$HOME/wheels"; tar -C "$HOME/gsrc" -xf "$HOME/glydgpu_src.tar"; mv "$HOME"/libglyd_gpu_cuda*.so "$HOME/glibs/"
  mkdir -p "$HOME/bbin"; ln -sf "$(command -v python3)" "$HOME/bbin/python"
  { { python3 -m pip install -q build || python3 -m pip install -q --break-system-packages build; } && PATH="$HOME/bbin:$PATH" GLYD_GPU_LIBS="$HOME/glibs" GLYD_GPU_NO_ENGINE=1 bash "$HOME/gsrc/glyd-gpu-py/build.sh" \
    && cp "$HOME"/gsrc/glyd-gpu-py/dist/*.whl "$HOME/wheels/"; } > "$R/wheel-build.txt" 2>&1 || { log "the wheel did not build (wheel-build.txt)"; exit 1; }
  export UV_FIND_LINKS=$HOME/wheels; ls "$HOME/wheels" | tee -a "$R/wheel-build.txt"
  echo "glyd-gpu 0.29.0: built on this pod from tag v0.29.0 with the release run's GPU libraries (sha256 $(sha256sum "$HOME"/glibs/*.so | cut -c1-16 | tr '\n' ' '))" > "$R/glyd-gpu-source.txt"
fi
# Glyd as a user installs it (the release's own installer, from its tag)
log "install"
curl -LsSf https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.0/scripts/install.sh | sh > "$R/install.txt" 2>&1 || { log "install failed (install.txt)"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
TOOL="$(uv tool dir)/glyd"; PY=$TOOL/bin/python; VLLM=$TOOL/bin/vllm
export PATH="$PATH:$TOOL/bin"  # (the tool environment's programs, ninja among them, for FlashInfer's compiles, as in an activated vLLM environment)
"$PY" -c "import importlib.metadata as m; print(*(f'{k} {m.version(k)}' for k in ('glyd', 'glyd-gpu', 'vllm', 'torch', 'transformers')), sep=' | ')" | tee "$R/env.txt"
"$VLLM" bench serve --help > /dev/null 2>&1 || log "WARNING: vllm bench serve does not start"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
download() { "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))" "$1"; }
metric() {  # PORT: "queries hits" of the prefix cache, from the server's /metrics
  curl -s -m 5 "http://127.0.0.1:$1/metrics" | awk '/^vllm:prefix_cache_queries(_total)?[{ ]/ {q += $NF} /^vllm:prefix_cache_hits(_total)?[{ ]/ {h += $NF} END {printf "%d %d\n", q, h}'
}
SRV=
stop() {
  [ -n "$SRV" ] && { kill -TERM "$SRV" 2> /dev/null; for _ in $(seq 1 60); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done; kill -KILL "$SRV" 2> /dev/null; }
  pkill -f "vllm.entrypoints|bin/vllm serve|EngineCore" 2> /dev/null; SRV=
  for _ in $(seq 1 60); do [ "$(gpu_used)" -lt 1000 ] && break; sleep 2; done
}
up() {  # DIR PORT LOG CMD...: the server in the background; 0 once /v1/models answers (at most 1500 s), else stopped, 1
  local d=$1 port=$2 out=$3; shift 3
  "$@" > "$out" 2>&1 < /dev/null & SRV=$!
  for _ in $(seq 1 750); do
    curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > "$d/models.json" 2> /dev/null && return 0
    kill -0 "$SRV" 2> /dev/null || break; sleep 2
  done
  stop; return 1
}
bench() {  # NAME DIR PORT MODEL: the warm-up and the levels
  local name=$1 d=$2 port=$3 model=$4 i n smi a b rc
  "$VLLM" bench serve --backend vllm --model "$model" --port "$port" --dataset-name random --random-input-len 1000 --random-output-len 256 --ignore-eos \
    --num-prompts 64 --max-concurrency 64 --request-rate inf --seed 999 > "$d/warmup.txt" 2>&1
  echo "idle after warm-up: $(gpu_used) MiB" >> "$d/memory.txt"
  for i in "${!LEVELS[@]}"; do
    n=${LEVELS[$i]}
    [ "$(el)" -gt $(( END - 240 )) ] && { log "$name: level $n not run, the time is short"; break; }
    nvidia-smi --query-gpu=timestamp,memory.used,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits -l 1 > "$d/smi-c$n.csv" 2> /dev/null & smi=$!
    a=$(metric "$port")
    "$VLLM" bench serve --backend vllm --model "$model" --port "$port" --dataset-name random --random-input-len 1000 --random-output-len 256 --ignore-eos \
      --num-prompts $(( 3 * n > 8 ? 3 * n : 8 )) --max-concurrency "$n" --request-rate inf --seed $(( 100 + i )) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --save-result --result-dir "$d" --result-filename "c$n.json" > "$d/bench-c$n.txt" 2>&1
    rc=$?; b=$(metric "$port"); kill "$smi" 2> /dev/null
    echo "$n $a $b" >> "$d/prefix.txt"
    log "$name: $n users, exit $rc, $(grep -h 'Request throughput' "$d/bench-c$n.txt" | tr -s ' ')"
  done
}
summary() { python3 "$HOME/summary3.py" "$R" > "$R/summary.txt" 2>&1; }
# The ladder: the defaults first, then fewer sequences at once, a shorter context, eager. Qwen3.5's architecture keeps a state a sequence and vLLM 0.30
# sizes its profiling for --max-num-seqs (1,024 from 70 GiB, else 256): on an RTX 4090, an L40S and an H100 the defaults ran out of memory, and vLLM
# then names the most sequences that fit ("lower max_num_seqs to at most N"), which is tried next where it says so.
LADDER=("" "--max-num-seqs 256" "--max-num-seqs 128" "--max-model-len 4096 --max-num-seqs 128" "--max-model-len 4096 --max-num-seqs 64"
        "--max-model-len 4096 --max-num-seqs 32 --enforce-eager")

DL=
for model in ${MODELS:?}; do
  tag=$(basename "$model")
  if [ -z "$DL" ]; then log "download $model"; download "$model" > "$R/dl-$tag.txt" 2>&1 || log "download failed: $model"; fi
  next=$(echo ${MODELS} | tr ' ' '\n' | grep -A1 -x "$model" | tail -n +2 | head -1)  # (the next model downloads while this one runs)
  DL=; [ -n "$next" ] && { download "$next" > "$R/dl-$(basename "$next").txt" 2>&1 & DL=$!; }
  for kind in ${KINDS:-bf16 glyd glydmma}; do
    [ "$(el)" -gt $(( END - 600 )) ] && { log "$tag $kind: not run, the time is short"; continue; }
    d=$R/$tag/$kind; mkdir -p "$d"; port=$([ $kind = bf16 ] && echo 8100 || echo 8200)
    if [ $kind = glydmma ]; then export GLYD_LAYOUT=mma; else unset GLYD_LAYOUT; fi
    started=; TRIES=("${LADDER[@]}"); k=-1
    while [ $((++k)) -lt ${#TRIES[@]} ] && [ $k -lt 9 ]; do
      extra=${TRIES[$k]}
      log "$tag $kind: start (try $k: ${extra:-defaults})"
      if [ $kind = bf16 ]; then  # (glyd and glydmma: glyd serve; glydmma with GLYD_LAYOUT=mma in its environment)
        VLLM_USE_FLASHINFER_SAMPLER=0 up "$d" $port "$d/serve-try$k.txt" "$VLLM" serve "$model" --port $port $extra
      else
        export XDG_STATE_HOME=$d/state-try$k
        up "$d" $port "$d/serve-try$k.txt" glyd serve "$model" --port $port ${extra:+-- $extra}
      fi && { started=$k; break; }
      log "$tag $kind: try $k did not start: $(grep -h -m1 -E 'Error|error|Refus|refus|out of memory' "$d/serve-try$k.txt" "$d"/state-try$k/glyd/logs/*.log 2> /dev/null | cut -c1-300)"
      # weights that do not fit the GPU: a shorter context or fewer sequences will not help
      if grep -q -E "out of memory|OutOfMemory" "$d/serve-try$k.txt" "$d"/state-try$k/glyd/logs/*.log 2> /dev/null && ! grep -q "Model loading took" "$d/serve-try$k.txt" "$d"/state-try$k/glyd/logs/*.log 2> /dev/null; then
        log "$tag $kind: the weights do not fit this GPU"; break
      fi
      most=$(grep -h -o -E "lower max_num_seqs to at most [0-9]+" "$d/serve-try$k.txt" "$d"/state-try$k/glyd/logs/*.log 2> /dev/null | grep -o -E "[0-9]+$" | head -1)
      [ -n "$most" ] && TRIES=("${TRIES[@]:0:k+1}" "$(echo "$extra" | sed -E 's/--max-num-seqs [0-9]+//') --max-num-seqs $most" "${TRIES[@]:k+1}")
    done
    [ -n "$started" ] || { echo "did not start" > "$d/started.txt"; summary; continue; }
    echo "try $started: ${TRIES[$started]:-defaults}" > "$d/started.txt"
    echo "after start: $(gpu_used) MiB" > "$d/memory.txt"
    bench "$tag $kind" "$d" $port "$model"
    stop
    summary
  done
  [ -n "$DL" ] && wait "$DL"
  unset GLYD_LAYOUT
  rm -rf "$HF_HOME/hub/models--${model/\//--}"  # (the disk for the next one)
done
log "done"
summary; cat "$R/summary.txt"
