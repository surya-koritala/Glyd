#!/usr/bin/env bash
# The 2026 bf16 sweep on one GPU: for each model of LIST ("REPO MODE BYTES" a line; MODE full or config), in order until DEADLINE (epoch s):
#   full:   the checkpoint downloaded (top-level *.safetensors, configs, tokenizer; the next model's download beside the current runs), then
#           Glyd:  GLYD_KV=off vllm serve REPO --quantization glyd $FLAGS   and   bf16:  vllm serve REPO $FLAGS
#           (vLLM 0.30.0 of Glyd's tool environment, one after the other, port 8100/8200), each: up or not, seconds to ready, "Model loading took",
#           "GPU KV cache size", nvidia-smi's memory used once ready, the plugin's "glyd:" lines, one greedy answer (chat1.py), the error where it did not start;
#           bf16 not tried where its checkpoint is larger than the GPU. Then the model's files deleted.
#   config: configs only (no weights), Glyd's command, the error recorded (vLLM 0.30.0 has no class for these architectures; no remote code is run).
# FLAGS are the same for both: eager (the weights and the KV cache measured without compiling; the compiled path is the acceptance's and the
# speed runs'), an 8,192-token window and 32 sequences (the same room asked of both), bf16. One row a server in $R/sweep.tsv; logs in $R/sweep/<model>/.
#   LIST=listA.txt DEADLINE=<epoch> TOOL=<glyd tool env> bash sweep.sh
set -uo pipefail
R=${R:-$HOME/results}; S=$R/sweep; mkdir -p "$S"
VLLM=$TOOL/bin/vllm; PY=$TOOL/bin/python
FLAGS=(--enforce-eager --max-model-len 8192 --max-num-seqs 32 --dtype bfloat16)
export HF_HOME=${HF_HOME:-$HOME/hf} HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
slog() { echo "$(date -u +%T) sweep: $*" | tee -a "$R/job.log"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
[ -f "$R/sweep.tsv" ] || printf 'model\tserver\tstarted\tsecs\tweights_gib\tkv_tokens\tsmi_mib\tanswer\tglyd_lines\terror\n' > "$R/sweep.tsv"
clean() { tr '\t\n\r' '   ' | tr -s ' ' | cut -c1-"${1:-300}"; }

download() {  # REPO MODE: into the HF cache; prints the snapshot dir
  "$PY" - "$1" "$2" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, mode = sys.argv[1], sys.argv[2]
pats = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "*.vocab"] + (["*.safetensors"] if mode == "full" else [])
print(snapshot_download(repo, allow_patterns=pats, ignore_patterns=["*/*"]))
EOF
}

stop_srv() {
  [ -n "${SRV:-}" ] && { kill -TERM "$SRV" 2> /dev/null; for _ in $(seq 1 60); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done; kill -KILL "$SRV" 2> /dev/null; }
  pkill -f "vllm serve|EngineCore|vllm.entrypoints" 2> /dev/null; SRV=
  for _ in $(seq 1 60); do [ "$(gpu_used)" -lt 1000 ] && break; sleep 2; done
}

measure() {  # MODEL KIND PORT LOG CMD...: one server, its row
  local model=$1 kind=$2 port=$3 log=$4; shift 4
  "$@" > "$log" 2>&1 < /dev/null & SRV=$!
  local t=$SECONDS up= ans= mem=
  while [ $((SECONDS - t)) -lt "${UP_WAIT:-900}" ]; do
    curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1 && { up=up; break; }
    kill -0 "$SRV" 2> /dev/null || break; sleep 2
  done
  local secs=$((SECONDS - t))
  if [ -n "$up" ]; then mem=$(gpu_used); ans=$(python3 "$HOME/chat1.py" "http://127.0.0.1:$port" "$model" 2>&1 | clean 260); else up=no; fi
  stop_srv
  local w kv gl err=
  w=$(grep -o -E "Model loading took [0-9.]+ GiB" "$log" | tail -1 | grep -o -E "[0-9.]+")
  kv=$(grep -o -E "GPU KV cache size: [0-9,]+ tokens" "$log" | tail -1 | grep -o -E "[0-9][0-9,]*" | tr -d ,)
  gl=$(grep -o -E "glyd: .*" "$log" | sort -u | clean 400)
  [ "$up" = up ] || err=$(grep -E "Error|error:|raise |not supported|out of memory|OutOfMemory|No module|Cannot|cannot|ValueError|RuntimeError" "$log" | grep -v -E "^\s*$|Traceback" | tail -3 | clean 400)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$model" "$kind" "$up" "$secs" "${w:-}" "${kv:-}" "${mem:-}" "${ans:-}" "${gl:-}" "${err:-}" >> "$R/sweep.tsv"
  slog "$model $kind: $up after $secs s, weights ${w:-?} GiB, KV ${kv:-?} tokens${err:+, error: ${err:0:160}}"
  RESULT=$up
}

ROWS=(); while IFS= read -r l; do ROWS+=("$l"); done < <(grep -v -E '^[[:space:]]*(#|$)' "${LIST:?}")
DL=; DLDIR=
for i in "${!ROWS[@]}"; do
  read -r model mode bytes <<< "${ROWS[$i]}"
  tag=${model//\//__}; d=$S/$tag; mkdir -p "$d"
  if [ $(( DEADLINE - $(date +%s) )) -lt "${NEED_S:-420}" ]; then
    slog "stop: under ${NEED_S:-420} s left; not run: ${ROWS[*]:$i}" ; printf '%s\n' "${ROWS[@]:$i}" > "$R/sweep-left.txt"; break
  fi
  # this model's files (downloaded beside the previous one, or now)
  if [ -n "$DL" ]; then wait "$DL"; DL=; else download "$model" "$mode" > "$d/download.txt" 2>&1; fi
  snap=$(tail -1 "$d/download.txt")
  if [ ! -d "$snap" ]; then
    printf '%s\t%s\tno\t0\t\t\t\t\t\t%s\n' "$model" "download" "$(grep -E "Error|error" "$d/download.txt" | tail -2 | clean 300)" >> "$R/sweep.tsv"
    slog "$model: download failed"; continue
  fi
  echo "$(du -sb "$snap" -L 2> /dev/null | cut -f1) bytes on disk" >> "$d/download.txt"
  # the next one's files while this one runs
  if [ $((i + 1)) -lt ${#ROWS[@]} ]; then
    read -r nm nmode _ <<< "${ROWS[$((i + 1))]}"; nd=$S/${nm//\//__}; mkdir -p "$nd"
    download "$nm" "$nmode" > "$nd/download.txt" 2>&1 & DL=$!
  fi
  slog "== $model ($mode, $((bytes / 1000000000)) GB bf16)"
  measure "$model" glyd 8100 "$d/glyd.log" env GLYD_KV=off "$VLLM" serve "$model" --port 8100 --quantization glyd "${FLAGS[@]}"
  if [ "$RESULT" != up ] && grep -q -E "greater than the derived max_model_len|max_model_len.*(greater|larger)" "$d/glyd.log"; then
    measure "$model" glyd-2k 8100 "$d/glyd-2k.log" env GLYD_KV=off "$VLLM" serve "$model" --port 8100 --quantization glyd --enforce-eager --max-model-len 2048 --max-num-seqs 32 --dtype bfloat16
  fi
  if [ "$mode" = full ]; then
    if [ "$(( bytes / 1048576 ))" -gt "$GPU_MIB" ]; then
      printf '%s\tbf16\tnot tried\t0\t\t\t\t\t\tthe bf16 checkpoint (%s bytes) is larger than the GPU (%s MiB)\n' "$model" "$bytes" "$GPU_MIB" >> "$R/sweep.tsv"
    else
      measure "$model" bf16 8200 "$d/bf16.log" "$VLLM" serve "$model" --port 8200 "${FLAGS[@]}"
      if [ "$RESULT" != up ] && grep -q -E "greater than the derived max_model_len|max_model_len.*(greater|larger)" "$d/bf16.log"; then
        measure "$model" bf16-2k 8200 "$d/bf16-2k.log" "$VLLM" serve "$model" --port 8200 --enforce-eager --max-model-len 2048 --max-num-seqs 32 --dtype bfloat16
      fi
    fi
  fi
  echo "$model" >> "$R/sweep-done.txt"
  [ -n "$DL" ] && kill -0 "$DL" 2> /dev/null || true
  rm -rf "$HF_HOME/hub/models--${model/\//--}"
  df -h "$HOME" | tail -1 > "$d/disk-after.txt"
done
[ -n "$DL" ] && { kill "$DL" 2> /dev/null; wait "$DL" 2> /dev/null; }
slog "done"
