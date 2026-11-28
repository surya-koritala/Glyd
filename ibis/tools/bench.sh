#!/usr/bin/env bash
# How fast it is on this computer: a short chat, then prompts of about 4K, 16K and 32K tokens, against the running server (tools/serve.sh).
# About 5 minutes; writes bench-report.md, then prints one block to paste into an issue.   tools/bench.sh [URL]   (default http://127.0.0.1:8080)
set -euo pipefail
. "${GLYD_QWEN:-$HOME/.glyd-qwen-3.8-flash}/env.sh"
URL=${1:-http://127.0.0.1:8080}
"$GLYD_ENGINE" qwenbench --url "$URL" --out bench-report.md
q() { nvidia-smi --query-gpu="$1" --format=csv,noheader 2> /dev/null | head -1 || echo "?"; }
MODE=$(curl -fsS "$URL/v1/glyd/status" 2> /dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mode", "?"))' 2> /dev/null || echo "?")
echo
echo "---- paste from here ----"
echo '```'
echo "GPU:     $(q name,memory.total)"
echo "Driver:  $(q driver_version)"
echo "PCIe:    now gen $(q pcie.link.gen.current) x$(q pcie.link.width.current), max gen $(q pcie.link.gen.max) x$(q pcie.link.width.max)"
echo "CPU:     $(lscpu 2> /dev/null | sed -n 's/^Model name: *//p' | head -1)"
echo "RAM:     $(free -g | awk '/^Mem:/ {print $2}') GiB"
echo "OS:      $(. /etc/os-release 2> /dev/null && echo "$PRETTY_NAME"), kernel $(uname -r)"
echo "Mode:    $MODE"
echo "Engine:  $("$GLYD_ENGINE" --version 2> /dev/null || echo "?"), sha256 $(sha256sum "$GLYD_ENGINE" 2> /dev/null | cut -c1-12)"
echo "Repo:    $(git -C "${GLYD_REPO:-$(dirname "$0")}" rev-parse --short HEAD 2> /dev/null || echo "?")"
grep '^|' bench-report.md
echo '```'
echo "---- to here ----"
