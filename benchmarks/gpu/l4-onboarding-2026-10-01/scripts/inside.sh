#!/bin/bash
# In the container (ubuntu:24.04, no toolkit): install Glyd as a user does, then the stages named in $STAGES.
set -u
export DEBIAN_FRONTEND=noninteractive
if ! command -v curl >/dev/null; then apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq curl ca-certificates ${GCC:+gcc} >/dev/null 2>&1; fi
if [ -n "${GCC:-}" ] && ! command -v gcc >/dev/null; then apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq gcc >/dev/null 2>&1; fi
echo "== tools: nvcc=$(command -v nvcc || echo none) gcc=$(command -v gcc || echo none) cuda_dir=$(ls -d /usr/local/cuda* 2>/dev/null | head -1 || echo none) user=$(id -un)"
export PATH="$HOME/.local/bin:$PATH"
PORT=${PORT:-8100}; MODEL=${MODEL:-Qwen/Qwen3-8B}
T=$HOME/.local/share/uv/tools/glyd/bin/python
for stage in ${STAGES:-install doctor run}; do
  echo; echo "== $(date -u +%T) stage $stage"
  case $stage in
    install)
      if [ -n "${LOCAL_INSTALL:-}" ]; then sh /onb/src/scripts/install.sh; else curl -LsSf "${INSTALL_URL:-https://raw.githubusercontent.com/surya-koritala/Glyd/onboarding/scripts/install.sh}" | sh; fi
      uv tool list --show-with 2>&1 | head -5 ;;
    freeze) uv pip freeze --python "$T" 2>&1 | grep -iE "^(vllm|torch|transformers|safetensors|tokenizers|pydantic|huggingface|hf-xet|triton|flashinfer|numpy|fastapi|uvicorn|glyd)" ;;
    doctor) glyd doctor ;;
    run) glyd run $MODEL --port $PORT ${RUN_FLAGS:-} --prompt "${PROMPT:-In one sentence, what is lossless compression?}" ${RUN_EXTRA:-} ; echo "run rc=$?" ;;
    runfail) glyd run $MODEL --port $PORT ${RUN_FLAGS:-} --prompt "hi" ${RUN_EXTRA:-} ; echo "run rc=$?" ;;
    bigprompt) $T -c 'print("word " * 60000)' | glyd run $MODEL --port $PORT ${RUN_FLAGS:-} ; echo "run rc=$?" ;;  # (a prompt on stdin, longer than any window here)
    serve)  # a server in the background, waited for
      glyd serve $MODEL --port $PORT ${SERVE_FLAGS:-} > /tmp/serve.out 2>&1 &
      for i in $(seq 1 900); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; pgrep -f "glyd serve" >/dev/null || break; sleep 1; done
      cat /tmp/serve.out ;;
    http) PY=$T bash /onb/http_checks.sh $PORT ;;
    bench) $T /onb/bench_conc.py $PORT - ${BENCH_N:-1 4 8 16} ;;
    webui) $T /onb/check_openwebui.py webui http://127.0.0.1:3000 "$MODEL" ;;  # (Open WebUI, started on the host, on this server)
    api) $T /onb/check_openwebui.py api http://127.0.0.1:$PORT/v1 ;;
    stop) pkill -TERM -f "glyd serve" ; for i in $(seq 1 60); do pgrep -f "glyd serve" >/dev/null || break; sleep 1; done; tail -3 /tmp/serve.out; sleep 2 ;;
    zig)  # no gcc here: ziglang from PyPI as $CC, through a wrapper (zig's linker driver does not take Triton's -l:libcuda.so.1)
      uv pip install --python $T ziglang 2>&1 | tail -1
      cat > /root/zigcc <<PYEOF
#!$T
import os, sys
args = sys.argv[1:]
dirs = [a[2:] for a in args if a.startswith("-L")]
out = [next((os.path.join(d, a[3:]) for d in dirs if os.path.exists(os.path.join(d, a[3:]))), a) if a.startswith("-l:") else a for a in args]
os.execv(sys.executable, [sys.executable, "-m", "ziglang", "cc", *out])
PYEOF
      chmod +x /root/zigcc
      echo "gcc: $(command -v gcc || echo none)"
      CC=/root/zigcc glyd run $MODEL --port $PORT ${RUN_FLAGS:-} --prompt "${PROMPT:-In one sentence, what is lossless compression?}" ; echo "run rc=$?" ;;
    shell) bash -c "${CMD:-true}" ;;
  esac
done
mkdir -p /hf/glyd-logs; for f in $HOME/.local/state/glyd/logs/*.log; do [ -f "$f" ] && cp --update=none "$f" /hf/glyd-logs/; done  # (the host reads them there)
echo; echo "== $(date -u +%T) logs: $(ls $HOME/.local/state/glyd/logs 2>/dev/null | tr '\n' ' ')"
echo "== tracebacks in the logs: $(cat $HOME/.local/state/glyd/logs/*.log 2>/dev/null | grep -c Traceback)  allocator warnings: $(cat $HOME/.local/state/glyd/logs/*.log 2>/dev/null | grep -c 'CUDACachingAllocator')"
