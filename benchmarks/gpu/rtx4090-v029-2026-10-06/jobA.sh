#!/usr/bin/env bash
# Pod A (one RTX 4090): (1) the v0.29.0 release acceptance, gpu/vllm/acceptance.sh of private tag v0.29.0 (accsrc.tar), the glyd flow on the host
# (no Docker on a pod; the CUDA toolkit moved out of /usr/local first; run as a user that is not root), v0.29.0's install.sh from its tag,
# Qwen/Qwen3.5-9B, Open WebUI by uvx; then the same with --card 4080s. (2) sweep.sh on listA.txt (the 2026 bf16 models that fit 24 GB with Glyd)
# with Glyd installed by the same install.sh, until the pod's time is near its end. Uploaded beside it: accsrc.tar, sweep.sh, chat1.py, listA.txt.
# Results in ~/results; DONE at the end.
set -uo pipefail
R=$HOME/results; mkdir -p "$R"; T0=$(date +%s); MAX_MIN=${MAX_MIN:-150}
log() { echo "$(date -u +%T) (+$(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
trap 'log "ending"; pkill -f "vllm serve|EngineCore|open-webui" 2> /dev/null; touch "$R/DONE"' EXIT
URL=https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.0/scripts/install.sh
{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -m; grep PRETTY /etc/os-release; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt" | tee -a "$R/job.log"
SKIP_ACC=${SKIP_ACC:-}

if [ -z "$SKIP_ACC" ]; then
  # ---- (1) the acceptance ---------------------------------------------------------------------------------------------------------------
  (apt-get update -qq && apt-get install -y -qq iproute2 procps curl) > "$R/apt.txt" 2>&1
  mkdir -p /opt/hidden; for c in /usr/local/cuda*; do [ -e "$c" ] && mv "$c" /opt/hidden/; done
  ls /opt/hidden > "$R/hidden-cuda.txt"; command -v nvcc >> "$R/hidden-cuda.txt" 2>&1; ss -ltn > "$R/ports-before.txt" 2>&1
  id acc > /dev/null 2>&1 || useradd -m -s /bin/bash acc
  mkdir -p /home/acc/src && tar -C /home/acc/src -xf "$HOME/accsrc.tar" && chown -R acc:acc /home/acc
  acc() {  # NAME ARGS...: one acceptance run as acc, its summary and logs into results/acc-NAME
    local name=$1; shift
    log "acceptance $name: $*"
    runuser -u acc -- env -i HOME=/home/acc PATH=/usr/local/bin:/usr/bin:/bin HF_HOME=/home/acc/hf LANG=C.UTF-8 \
      timeout -k 30 3300 bash -c 'cd ~/src/gpu/vllm && bash acceptance.sh "$@"' _ --work "/home/acc/w-$name" "$@" > "$R/acc-$name.out" 2>&1
    local rc=$?
    mkdir -p "$R/acc-$name"; cp "/home/acc/w-$name/summary.txt" "$R/acc-$name/" 2> /dev/null
    cp -r "/home/acc/w-$name/logs" "$R/acc-$name/" 2> /dev/null; rm -f "$R/acc-$name/logs/page.html" "$R/acc-$name/logs"/*.pid
    log "acceptance $name: exit $rc, $(grep -c '^ok' "$R/acc-$name/summary.txt" 2> /dev/null) ok, $(grep -c '^FAIL' "$R/acc-$name/summary.txt" 2> /dev/null) FAIL, last: $(tail -1 "$R/acc-$name/summary.txt" 2> /dev/null | cut -c1-100)"
    echo "$name $rc" >> "$R/acc-exits.txt"
  }
  acc 24gb --flow glyd --host --install-url "$URL" --model Qwen/Qwen3.5-9B --webui uvx
  sleep 20
  acc 4080s --flow glyd --host --install-url "$URL" --model Qwen/Qwen3.5-9B --webui uvx --card 4080s
  pkill -u acc 2> /dev/null; sleep 5
fi

# ---- (2) the sweep ------------------------------------------------------------------------------------------------------------------------
log "install for the sweep (root, v0.29.0's install.sh)"
curl -LsSf "$URL" | sh > "$R/install-sweep.txt" 2>&1 || log "install failed (install-sweep.txt)"
export PATH="$HOME/.local/bin:$PATH"; TOOL="$(uv tool dir)/glyd"
"$TOOL/bin/python" -c "import importlib.metadata as m; print(*(f'{k} {m.version(k)}' for k in ('glyd', 'glyd-gpu', 'vllm', 'torch', 'transformers')), sep=' | ')" | tee "$R/env.txt"
export HF_HOME=/home/acc/hf; [ -d "$HF_HOME" ] || HF_HOME=$HOME/hf
TOOL=$TOOL LIST=$HOME/listA.txt DEADLINE=$(( T0 + (MAX_MIN - 12) * 60 )) R=$R bash "$HOME/sweep.sh"
log "job done"
