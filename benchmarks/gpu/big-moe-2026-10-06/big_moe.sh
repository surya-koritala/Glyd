#!/usr/bin/env bash
# One large 2026 mixture-of-experts model over all the GPUs of one machine (tensor parallel), unattended: vLLM 0.30.0 and
# Glyd 0.29.0 as pip installs them (glyd[vllm]==0.29.0 from PyPI, which brings glyd-gpu 0.29.0). For each server in MODES:
# vllm serve MODEL --tensor-parallel-size TP (text only), the time until it answers, each GPU's "Model loading took", the KV
# cache's tokens and maximum concurrency as vLLM logs them, nvidia-smi's memory, one chat answer (greedy), then vllm bench serve
# saturated (128 prompts of IN=1000 tokens, OUT=256 out, all at once, the same seed for every server) after 8 warm-up prompts.
#   MODEL=Qwen/Qwen3.5-122B-A10B TP=4 MODES="bf16 glyd" CHECK=brief bash big_moe.sh
# Modes: bf16 (vLLM's own), glyd (--quantization glyd, layout and kv auto), glyd-mma (GLYD_LAYOUT=mma, the tiered layout).
# VERIFY=1: every Glyd pack decoded and compared with its weights bit for bit as it is made (GLYD_VERIFY=1); the startup time then
# includes that check and is not the default startup.
# CHECK=brief (CHECK=last: after the servers): gpu/vllm/check_vllm.py --brief --tp TP (bf16 with CUDA graphs and eager, Glyd auto with every layer checked,
# exact mode eager bit for bit against bf16 eager), first, before the servers, so a pod cap cannot cut it; needs check_vllm.py in ~ and bf16 fitting.
# R, W: results and work folders (default ~/results, ~/w); LMO="" leaves out --language-model-only (text-only models).
# results/: machine.txt, env.txt, log/, MODE/{serve.log,startup.txt,lines.txt,smi.csv,chat.json,bench.json}, summary.txt; results/DONE at the end.
set -u
MODEL=${MODEL:?}; TP=${TP:?}; MODES=${MODES:-bf16 glyd}; UTIL=${UTIL:-0.9}; MML=${MML:-32768}; IN=${IN:-1000}; OUT=${OUT:-256}; N=${N:-128}
R=${R:-$HOME/results}; W=${W:-$HOME/w}; PORT=8012; mkdir -p "$R/log" "$W/tmp"
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
note() { echo "[+$(el) s] $*" | tee -a "$R/summary.txt"; }
SV=; DL=
# the download runs in the background: an early exit (no uv, a failed install) stops it too, so the next model's job never shares the disk with it
trap '[ -n "$SV" ] && kill $SV 2> /dev/null; [ -n "$DL" ] && kill $DL 2> /dev/null; pkill -f snapshot_download 2> /dev/null; pkill -f "vllm serve" 2> /dev/null; touch "$R/DONE"' EXIT
{ nvidia-smi; nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv; nvidia-smi topo -m
  lscpu | grep -E "Model name|Architecture|^CPU\(s\)"; free -g | head -2; df -h "$HOME" | tail -1; date -u; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH PYTHONPATH GLYD_GPU_LIB HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$HOME/hf HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 TMPDIR=$W/tmp TOKENIZERS_PARALLELISM=false
note "$(nvidia-smi -L | grep -c '^GPU') x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); model $MODEL, TP $TP, modes $MODES, util $UTIL, max-model-len $MML; $(df -BG --output=avail "$HOME" | tail -1 | tr -d ' ') free"

# uv; the model downloading (hf_xet / hf_transfer) in a venv of its own beside vLLM's and Glyd's install
mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv"
UV=$W/uv/uv; [ -x "$UV" ] || { note "FAIL: no uv"; exit 1; }
( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" "huggingface_hub[hf_xet]" hf_transfer > "$R/log/dl-env.txt" 2>&1
  d=$(date +%s)
  "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', '*.tiktoken'], max_workers=32))" "$MODEL" > "$W/model.dir" 2> "$R/log/dl.txt"
  echo "exit $?: $(du -sbL "$(tail -1 "$W/model.dir")" 2> /dev/null | cut -f1) bytes in $(( $(date +%s) - d )) s" >> "$R/log/dl.txt" ) &
DL=$!
"$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "vllm==0.30.0" "glyd[vllm]==0.29.0" > "$R/log/env.txt" 2>&1 || { note "FAIL: vLLM and glyd[vllm] did not install (log/env.txt)"; exit 1; }
export PATH=$W/venv/bin:$PATH
python -c "import vllm, torch, glyd_gpu, importlib.metadata as m; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'glyd', m.version('glyd'), 'glyd-gpu', m.version('glyd-gpu'))" > "$R/env.txt" 2>&1
note "environment: $(cat "$R/env.txt" | tail -1)"
wait $DL; note "download: $(tail -1 "$R/log/dl.txt")"
grep -q "^exit 0" "$R/log/dl.txt" || { note "FAIL: the download"; exit 1; }
export HF_HUB_OFFLINE=1

gone() { pkill -f "vllm serve"; for i in $(seq 1 60); do [ $i = 24 ] && pkill -9 -f "VLLM|vllm";  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ] && return; sleep 5; done; }
check() {
  ( cd "$W" && VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 timeout "${CHECK_TIMEOUT:-3600}" python "$HOME/check_vllm.py" --brief --tp "$TP" --out "$R/check" "$MODEL" ) > "$R/check.txt" 2>&1
  rc=$?; [ $rc = 124 ] && rc="TIMEOUT after ${CHECK_TIMEOUT:-3600} s, no result"
  note "check_vllm --brief --tp $TP: exit $rc; $(grep -c '^PASS' "$R/check.txt") PASS, $(grep -c '^FAIL' "$R/check.txt") FAIL"
  grep -E "^(PASS|FAIL)" "$R/check.txt" | cut -c1-400 >> "$R/summary.txt"
}
[ "${CHECK:-}" = brief ] && check
for mode in $MODES; do
  D=$R/$mode; mkdir -p "$D"
  q=(); envs=(VLLM_CACHE_ROOT=$W/cache-$mode)
  case $mode in
    bf16) ;;
    glyd) q=(--quantization glyd); envs+=(VLLM_DISABLE_SHARED_EXPERTS_STREAM=1) ;;
    glyd-mma) q=(--quantization glyd); envs+=(VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 GLYD_LAYOUT=mma) ;;
    *) note "unknown mode $mode"; continue ;;
  esac
  [ "${VERIFY:-0}" = 1 ] && [ "$mode" != bf16 ] && envs+=(GLYD_VERIFY=1)
  cmd=(vllm serve "$MODEL" --tensor-parallel-size "$TP" --gpu-memory-utilization "$UTIL" --max-model-len "$MML" ${LMO---language-model-only} --port "$PORT" "${q[@]}")
  echo "env ${envs[*]} ${cmd[*]}" > "$D/command.txt"
  gone; s=$(date +%s)
  env "${envs[@]}" "${cmd[@]}" > "$D/serve.log" 2>&1 &
  SV=$!
  ( while kill -0 $SV 2> /dev/null; do nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader; sleep 5; done ) > "$D/smi.csv" 2>&1 &
  ok=0
  for _ in $(seq 1 720); do curl -sf "localhost:$PORT/v1/models" > /dev/null && { ok=1; break; }; kill -0 $SV 2> /dev/null || break; sleep 5; done
  echo "$(( $(date +%s) - s )) s to answer (ok=$ok)" > "$D/startup.txt"
  grep -E "Model loading took|GPU KV cache size|Maximum concurrency|init engine|glyd:|Error|error:|OutOfMemory|out of memory" "$D/serve.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-600 > "$D/lines.txt"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$D/memory.csv"
  if [ $ok != 1 ]; then
    note "$mode: did not start ($(cat "$D/startup.txt")); last lines: $(grep -v '^\s*$' "$D/serve.log" | tail -3 | cut -c1-300 | tr '\n' ' ')"
    kill $SV 2> /dev/null; wait $SV 2> /dev/null; SV=; pkill -f "vllm serve"; continue
  fi
  note "$mode: $(cat "$D/startup.txt"); $(grep -o 'Model loading took [0-9.]* GiB' "$D/lines.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';') $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$D/lines.txt" | head -1); $(grep -o 'Maximum concurrency for [0-9,]* tokens per request: [0-9.]*x' "$D/lines.txt" | head -1)"
  curl -s "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"In two sentences: what does lossless compression of a neural network's weights mean?\"}], \"max_tokens\": 200, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}" > "$D/chat.json"
  note "$mode chat: $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"][:300].replace(chr(10), " "))' "$D/chat.json" 2>&1)"
  b=(vllm bench serve --backend vllm --model "$MODEL" --port "$PORT" --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" --ignore-eos --request-rate inf)
  "${b[@]}" --num-prompts 8 --seed 1000 > "$D/warmup.txt" 2>&1
  echo "${b[*]} --num-prompts $N --seed 0 --save-result --result-dir $D --result-filename bench.json" > "$D/bench-command.txt"
  "${b[@]}" --num-prompts "$N" --seed 0 --save-result --result-dir "$D" --result-filename bench.json > "$D/bench.txt" 2>&1
  note "$mode bench: $(python -c 'import json,sys; print("{completed} of {num_prompts} done, {request_throughput:.2f} req/s, {output_throughput:.1f} output tok/s, TTFT mean {mean_ttft_ms:.0f} ms, TPOT mean {mean_tpot_ms:.1f} ms".format(**json.load(open(sys.argv[1]))))' "$D/bench.json" 2>&1 | tail -1)"
  kill $SV 2> /dev/null; wait $SV 2> /dev/null; SV=; pkill -f "vllm serve"
done

[ "${CHECK:-}" = last ] && { gone; check; }
note "done"
