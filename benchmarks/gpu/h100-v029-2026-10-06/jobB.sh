#!/usr/bin/env bash
# Pod B (one H100 80GB HBM3): sweep.sh on listB.txt (the 2026 bf16 models of 20-75 GB with Glyd), Glyd installed as a user does
# (v0.29.0's install.sh from its tag, PyPI's glyd 0.29.0 and glyd-gpu 0.29.0), until the pod's time is near its end.
# Uploaded beside it: sweep.sh, chat1.py, listB.txt. Results in ~/results; DONE at the end.
set -uo pipefail
R=$HOME/results; mkdir -p "$R"; T0=$(date +%s); MAX_MIN=${MAX_MIN:-150}
log() { echo "$(date -u +%T) (+$(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
trap 'log "ending"; pkill -f "vllm serve|EngineCore" 2> /dev/null; touch "$R/DONE"' EXIT
{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -m; grep PRETTY /etc/os-release; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt" | tee -a "$R/job.log"
log "install (v0.29.0's install.sh)"
curl -LsSf https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.0/scripts/install.sh | sh > "$R/install.txt" 2>&1 || log "install failed (install.txt)"
export PATH="$HOME/.local/bin:$PATH"; TOOL="$(uv tool dir)/glyd"
"$TOOL/bin/python" -c "import importlib.metadata as m; print(*(f'{k} {m.version(k)}' for k in ('glyd', 'glyd-gpu', 'vllm', 'torch', 'transformers')), sep=' | ')" | tee "$R/env.txt"
TOOL=$TOOL LIST=$HOME/listB.txt NEED_S=480 DEADLINE=$(( T0 + (MAX_MIN - 12) * 60 )) R=$R bash "$HOME/sweep.sh"
log "job done"
