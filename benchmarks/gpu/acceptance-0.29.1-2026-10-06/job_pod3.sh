#!/usr/bin/env bash
# Published v0.29.1 acceptance from getglyd.com on one RTX 4090.
# Source: dc56ad719140f1dac1ea1b971638b4d49e2ebe5f; the harness includes the measured-context refusal check.
# MAX_MIN=75, JOBMAX=65; no compiler or CUDA toolkit, a non-root user, no model flags.
set -uo pipefail
R=$HOME/results; mkdir -p "$R"; T0=$(date +%s); JOBMAX=65
log() { echo "$(date -u +%T) (+$(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
trap 'log "ending"; pkill -u acc 2>/dev/null; touch "$R/DONE"' EXIT
URL=https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.1/scripts/install.sh
{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -m; grep PRETTY /etc/os-release; } > "$R/machine.txt" 2>&1
(apt-get update -qq && apt-get install -y -qq iproute2 procps curl) > "$R/apt.txt" 2>&1 || exit 1
mkdir -p /opt/hidden /opt/hidden-cc
for c in /usr/local/bin/uv /usr/local/bin/uvx /usr/bin/uv /usr/bin/uvx /usr/bin/gcc* /usr/bin/cc /usr/bin/x86_64-linux-gnu-gcc* /usr/bin/c89* /usr/bin/c99*; do [ -e "$c" ] && mv "$c" /opt/hidden-cc/; done
for c in /usr/local/cuda*; do [ -e "$c" ] && mv "$c" /opt/hidden/; done
{ ls /opt/hidden /opt/hidden-cc; command -v nvcc uv gcc cc; } > "$R/hidden.txt" 2>&1
ss -ltn > "$R/ports-before.txt" 2>&1
id acc >/dev/null 2>&1 || useradd -m -s /bin/bash acc
mkdir -p /home/acc/src /home/acc/hf
tar -C /home/acc/src -xf "$HOME/accsrc.tar" || exit 1
mv /home/acc/src/gpu/vllm /home/acc/src/gpu/gate
chown -R acc:acc /home/acc
sha256sum "$HOME/accsrc.tar" /home/acc/src/gpu/gate/acceptance.sh > "$R/accsrc.sha256"
curl -fsSL https://getglyd.com/install.sh -o "$R/served-install.sh" || exit 1
curl -fsSL "$URL" -o "$R/tag-install.sh" || exit 1
cmp -s "$R/served-install.sh" "$R/tag-install.sh" || { log "served installer differs from v0.29.1: no model run"; exit 1; }
sha256sum "$R/served-install.sh" "$R/tag-install.sh" > "$R/served.txt"
log "served installer matches v0.29.1 byte for byte"
W=/home/acc/w-24gb
limit=$(( T0 + JOBMAX * 60 - $(date +%s) - 60 ))
[ "$limit" -gt 1800 ] || { log "not enough time for the acceptance"; exit 1; }
log "acceptance: --flow glyd --host --served --install-url $URL --model Qwen/Qwen3.5-9B --webui uvx"
runuser -u acc -- env -i HOME=/home/acc PATH=/usr/local/bin:/usr/bin:/bin HF_HOME=/home/acc/hf LANG=C.UTF-8 \
  timeout -k 30 "$limit" bash -c 'cd ~/src/gpu/gate && bash acceptance.sh "$@"' _ --work "$W" --flow glyd --host --served --install-url "$URL" \
  --model Qwen/Qwen3.5-9B --webui uvx > "$R/acc-24gb.out" 2>&1
RC=$?
mkdir -p "$R/acc-24gb"
cp "$W/summary.txt" "$R/acc-24gb/" 2>/dev/null
cp -r "$W/logs" "$R/acc-24gb/" 2>/dev/null
rm -f "$R/acc-24gb/logs/page.html" "$R/acc-24gb/logs"/*.pid
log "acceptance: exit $RC, $(grep -c '^ok' "$R/acc-24gb/summary.txt" 2>/dev/null) ok, $(grep -c '^FAIL' "$R/acc-24gb/summary.txt" 2>/dev/null) failed"
pkill -u acc 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$R/gpu-end.txt" 2>&1
exit "$RC"
