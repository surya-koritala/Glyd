#!/usr/bin/env bash
# v0.29.1 release acceptance on one RTX 4090, the published release as a user installs it.
# (1) gpu/vllm/acceptance.sh of private tag v0.29.1 (c6ad9ac7; accsrc.tar = git archive of acceptance.sh, its README and scripts/install*),
#     the glyd flow on the host (no Docker on a pod): /usr/local/cuda*, uv and gcc moved out first, run as a user that is not root,
#     v0.29.1's install.sh from its tag, Qwen/Qwen3.5-9B, Open WebUI by uvx; --served (the line the README gives, from getglyd.com) when
#     getglyd.com serves the tag's install.sh byte for byte, otherwise the tag's file (and that is said in served.txt).
#     The tree is unpacked as gpu/gate: the acceptance's Ctrl-C check kills every process whose command line says vllm.
# (2) the same user, a fresh home, the same install line, then glyd run google/gemma-4-12B-it --prompt (Gemma 4 starts in 0.29.1).
# Results in ~/results; DONE at the end.
set -uo pipefail
R=$HOME/results; mkdir -p "$R"; T0=$(date +%s); JOBMAX=${JOBMAX:-95}
log() { echo "$(date -u +%T) (+$(( ($(date +%s) - T0) / 60 )) min) $*" | tee -a "$R/job.log"; }
left() { echo $(( T0 + JOBMAX * 60 - $(date +%s) )); }
trap 'log "ending"; pkill -u acc 2> /dev/null; pkill -f "EngineCore|open-webui" 2> /dev/null; touch "$R/DONE"' EXIT
URL=https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.1/scripts/install.sh
{ date -u; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; uname -m; grep PRETTY /etc/os-release; nproc; free -g | head -2; df -h "$HOME" | tail -1; } > "$R/machine.txt" 2>&1
cat "$R/machine.txt" | tee -a "$R/job.log"

(apt-get update -qq && apt-get install -y -qq iproute2 procps curl) > "$R/apt.txt" 2>&1
mkdir -p /opt/hidden /opt/hidden-cc
for c in /usr/local/bin/uv /usr/local/bin/uvx /usr/bin/gcc* /usr/bin/cc /usr/bin/x86_64-linux-gnu-gcc* /usr/bin/c89* /usr/bin/c99*; do [ -e "$c" ] && mv "$c" /opt/hidden-cc/; done
for c in /usr/local/cuda*; do [ -e "$c" ] && mv "$c" /opt/hidden/; done
{ ls /opt/hidden /opt/hidden-cc; command -v nvcc uv gcc cc; } > "$R/hidden.txt" 2>&1; ss -ltn > "$R/ports-before.txt" 2>&1
id acc > /dev/null 2>&1 || useradd -m -s /bin/bash acc
mkdir -p /home/acc/src /home/acc/hf && tar -C /home/acc/src -xf "$HOME/accsrc.tar" && mv /home/acc/src/gpu/vllm /home/acc/src/gpu/gate && chown -R acc:acc /home/acc
sha256sum "$HOME/accsrc.tar" /home/acc/src/gpu/gate/acceptance.sh > "$R/accsrc.sha256"
asacc() { runuser -u acc -- env -i HOME="$1" PATH="$1/.local/bin:/usr/local/bin:/usr/bin:/bin" HF_HOME=/home/acc/hf LANG=C.UTF-8 bash -c "$2"; }

# ---- getglyd.com's install.sh against the tag's ----------------------------------------------------------------------------------------------
curl -fsSL https://getglyd.com/install.sh -o "$R/served-install.sh"; curl -fsSL "$URL" -o "$R/tag-install.sh"
SERVED=
if cmp -s "$R/served-install.sh" "$R/tag-install.sh"; then SERVED=--served; echo "getglyd.com serves the v0.29.1 tag's install.sh byte for byte: --served" > "$R/served.txt"
else echo "getglyd.com does not serve the v0.29.1 tag's install.sh ($(grep -m1 '^GLYD_VERSION=' "$R/served-install.sh")): the run installs from $URL, no --served" > "$R/served.txt"; fi
sha256sum "$R/served-install.sh" "$R/tag-install.sh" >> "$R/served.txt"; log "$(head -1 "$R/served.txt")"

# ---- (1) the acceptance ----------------------------------------------------------------------------------------------------------------------
W=/home/acc/w-24gb
log "acceptance: --flow glyd --host $SERVED --install-url $URL --model Qwen/Qwen3.5-9B --webui uvx"
runuser -u acc -- env -i HOME=/home/acc PATH=/usr/local/bin:/usr/bin:/bin HF_HOME=/home/acc/hf LANG=C.UTF-8 \
  timeout -k 30 3600 bash -c 'cd ~/src/gpu/gate && bash acceptance.sh "$@"' _ --work "$W" --flow glyd --host $SERVED --install-url "$URL" \
  --model Qwen/Qwen3.5-9B --webui uvx > "$R/acc-24gb.out" 2>&1
RC=$?
mkdir -p "$R/acc-24gb"; cp "$W/summary.txt" "$R/acc-24gb/" 2> /dev/null; cp -r "$W/logs" "$R/acc-24gb/" 2> /dev/null; rm -f "$R/acc-24gb/logs/page.html" "$R/acc-24gb/logs"/*.pid
log "acceptance: exit $RC, $(grep -c '^ok' "$R/acc-24gb/summary.txt" 2> /dev/null) ok, $(grep -c '^FAIL' "$R/acc-24gb/summary.txt" 2> /dev/null) FAIL, last: $(tail -1 "$R/acc-24gb/summary.txt" 2> /dev/null | cut -c1-200)"
pkill -u acc 2> /dev/null; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$R/gpu-after-acc.txt" 2>&1

# ---- (2) Gemma 4: glyd run --prompt, the same install line in a fresh home -------------------------------------------------------------------
G=/home/acc/g; G_R=$R/gemma; mkdir -p "$G_R"; runuser -u acc -- mkdir -p "$G"
log "gemma: install ($([ -n "$SERVED" ] && echo getglyd.com || echo the tag)'s install.sh, a fresh home)"
if [ -n "$SERVED" ]; then LINE='curl -LsSf https://getglyd.com/install.sh | sh'; else LINE="curl -LsSf $URL | sh"; fi
asacc "$G" "cd ~ && $LINE" > "$G_R/install.log" 2>&1; IRC=$?
log "gemma: install exit $IRC: $(tail -2 "$G_R/install.log" | tr '\n' '|' | cut -c1-200)"
asacc "$G" 'py=$(uv tool dir)/glyd/bin/python; $py -c "import importlib.metadata as m; print(*(f\"{k} {m.version(k)}\" for k in (\"glyd\", \"glyd-gpu\", \"vllm\", \"torch\", \"transformers\")), sep=\" | \")"' > "$G_R/versions.txt" 2>&1
cat "$G_R/versions.txt" | tee -a "$R/job.log"
grep -q 'glyd 0.29.1 | glyd-gpu 0.29.1 |' "$G_R/versions.txt" && echo "ok   installed: $(cat "$G_R/versions.txt")" > "$G_R/summary.txt" || echo "FAIL installed: $(cat "$G_R/versions.txt")" > "$G_R/summary.txt"
T=$(( $(left) - 120 )); [ "$T" -gt 1800 ] && T=1800
if [ "$IRC" = 0 ] && [ "$T" -gt 300 ]; then
  log "gemma: glyd run google/gemma-4-12B-it --prompt (at most $T s)"; t=$SECONDS
  asacc "$G" "cd ~ && timeout -k 30 $T glyd run google/gemma-4-12B-it --prompt 'What is the capital of France? Answer in one word.'" > "$G_R/run.out" 2> "$G_R/run.err"; GRC=$?
  log "gemma: exit $GRC after $((SECONDS - t)) s; answer: $(tr -d '\n' < "$G_R/run.out" | cut -c1-80)"
  cp "$G"/.local/state/glyd/logs/*.log "$G_R/" 2> /dev/null
  # (as the acceptance does: a traceback after vLLM's own "[shutdown]" lines is its output handler finding the engine stopped on purpose)
  bad=$( { cat "$G_R/run.out" "$G_R/run.err"; for f in "$G_R"/*.log; do [ -f "$f" ] && sed '/\[shutdown\]/q' "$f"; done; } 2> /dev/null | grep -c -E 'AmbiguousGlobalPerLayerAttributeError|Traceback' || true)
  if [ "$GRC" = 0 ] && grep -qi paris "$G_R/run.out" && [ "$bad" = 0 ] && ! grep -q -E '<think>|</think>|<\|channel' "$G_R/run.out"; then
    echo "ok   glyd run google/gemma-4-12B-it --prompt: exit 0, $(tr -d '\n' < "$G_R/run.out" | cut -c1-40), no traceback; $(grep -E '^Settings:' "$G_R/run.err" | cut -c1-200)" >> "$G_R/summary.txt"
  else
    echo "FAIL glyd run google/gemma-4-12B-it --prompt: exit $GRC, answer $(tr -d '\n' < "$G_R/run.out" | cut -c1-40), $bad traceback lines: $(tail -3 "$G_R/run.err" | tr '\n' '|' | cut -c1-300)" >> "$G_R/summary.txt"
  fi
  grep -q 'starting again' "$G_R/run.err" && echo "FAIL glyd run google/gemma-4-12B-it: started vLLM twice (starting again)" >> "$G_R/summary.txt" || echo "ok   glyd run google/gemma-4-12B-it: one start" >> "$G_R/summary.txt"
  grep -h -o -E "Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|--max-model-len [0-9]+|--gpu-memory-utilization [0-9.]+|--tool-call-parser [a-z0-9_]+|--reasoning-parser [a-z0-9_]+" "$G_R"/*.log 2> /dev/null | sort -u > "$G_R/numbers.txt"
else
  echo "FAIL gemma not run: install exit $IRC, $T s left" >> "$G_R/summary.txt"
fi
cat "$G_R/summary.txt" | tee -a "$R/job.log"
pkill -u acc 2> /dev/null; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$R/gpu-end.txt" 2>&1
mv /opt/hidden-cc/* /usr/bin/ 2> /dev/null; mv /usr/bin/uv /usr/bin/uvx /usr/local/bin/ 2> /dev/null; mv /opt/hidden/* /usr/local/ 2> /dev/null
log "job done"
