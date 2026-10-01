#!/usr/bin/env bash
# The release gate for "glyd[vllm]": the quickstart of gpu/vllm/README.md, run from nothing on a machine with no CUDA
# toolkit. It makes a clean environment, installs glyd[vllm] (a version from PyPI, or a wheel), hides nvcc, leaves the
# server only the GPU memory of a card, runs the quickstart's serve command as the README gives it, chats through the
# OpenAI API (streaming, two turns, a tool call), and runs Open WebUI on it both ways the README gives (uvx, Docker),
# chatting through its chat endpoint as the browser does. It fails on a traceback in the server's log, an allocator
# out-of-memory warning (either kind), a missing answer, or a model Open WebUI does not list.
#
#   bash acceptance.sh [--version V | --wheel FILE] [--budget-mib N] [--card-mib N] [--geforce] [--command CMD]
#                      [--webui uvx|docker|both|none] [--hold MINUTES] [--image IMG] [--work DIR] [--host]
#
#   --version V      glyd[vllm]==V from PyPI in place of the README's install line; --wheel FILE: this wheel, with its
#                    vllm extra; with neither, the README's install line as it is
#   --budget-mib N   the GPU memory the server finds free when it starts (vLLM's "Free memory on device"), in MiB: a
#                    process holds the rest of the GPU. --card-mib N: the card's total as CUDA reports it (an RTX 4080
#                    SUPER's is 15942): the command's --gpu-memory-utilization is scaled so that its budget in GiB is the
#                    card's. --geforce: the plugin reads the GPU as GeForce Ada (for an L4 standing in for one). On an
#                    RTX 4080 SUPER none of the three is needed. --card 4080s is --budget-mib 14828 --card-mib 15942 --geforce
#   --command CMD    the serve command to run in place of the README's
#   --webui ROUTES   the Open WebUI routes to run: uvx, docker, both (the default) or none
#   --hold MINUTES   after the checks leave the server and Open WebUI (the uvx route) up for a person to try, until the
#                    time is up or DIR/stop exists
#   --image IMG      the container's image. The default is Ubuntu 26.04 with gcc (Triton builds its launchers with a C
#                    compiler) and nothing else: no nvcc, no /usr/local/cuda
#   --work DIR       its environment, caches and logs (default ~/glyd-acceptance); HF_HOME is the model's cache
#   --host           no container: the commands run in a stripped environment on this machine (nvcc must not be found,
#                    so there must be no /usr/local/cuda)
#
# Needs: Linux, an NVIDIA GPU and its driver, Docker with the NVIDIA container toolkit (or --host, and Docker for the
# Docker route), and the network. Ports 8000 and 3000 must be free. It runs the README's blocks marked
# "<!-- acceptance: setup | serve | webui-uvx | webui-docker -->". Exit status 0 if nothing failed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
README=$HERE/README.md
VERSION= WHEEL= BUDGET= CARD= GEFORCE= COMMAND= WEBUI=both HOLD= IMAGE= WORK=$HOME/glyd-acceptance HOST=
while [ $# -gt 0 ]; do
  case $1 in
    --version) VERSION=$2; shift;; --wheel) WHEEL=$2; shift;;
    --budget-mib) BUDGET=$2; shift;; --card-mib) CARD=$2; shift;; --geforce) GEFORCE=1;;
    --card) [ "$2" = 4080s ] || { echo "unknown card $2 (4080s)"; exit 2; }; BUDGET=14828 CARD=15942 GEFORCE=1; shift;;
    --command) COMMAND=$2; shift;; --webui) WEBUI=$2; shift;; --hold) HOLD=$2; shift;; --image) IMAGE=$2; shift;; --work) WORK=$2; shift;; --host) HOST=1;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d;s/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option $1 (--help)"; exit 2;;
  esac
  shift
done
mkdir -p "$WORK" && WORK=$(cd "$WORK" && pwd)
HF=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p "$WORK/home" "$WORK/logs" "$WORK/bin" "$WORK/uv-cache" "$WORK/wheel" "$HF"
LOGS=$WORK/logs
FAILS=()
: > "$WORK/summary.txt"
say() { printf '%s\n' "$*" | tee -a "$WORK/summary.txt"; }
fail() { FAILS+=("$*"); say "FAIL $*"; }
ok() { say "ok   $*"; }
count() { grep -c "$1" "$2" 2> /dev/null || true; }
block() {  # the fenced block after the README's marker "<!-- acceptance: $1 -->"
  awk -v n="$1" '$0 == "<!-- acceptance: " n " -->" {f = 1; next} f == 1 && /^```/ {f = 2; next} f == 2 && /^```/ {exit} f == 2 {print}' "$README"
}
for p in 8000 3000; do
  ss -ltn 2> /dev/null | grep -q ":$p " && { echo "port $p is in use: the server and Open WebUI need it"; exit 2; }
done
case $WEBUI in *docker*|both) docker ps -a --format '{{.Names}}' | grep -qx open-webui && { echo "a container named open-webui exists: remove it, or run with --webui uvx"; exit 2; };; esac

# --- uv, and where the commands run: a container with a C compiler and no CUDA toolkit, or a stripped environment
UV=$(command -v uv || true)
if [ -z "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$WORK/bin UV_NO_MODIFY_PATH=1 sh > "$LOGS/uv-install.log" 2>&1 || { echo "uv is not installed, and its installer failed"; exit 2; }
  UV=$WORK/bin/uv
fi
UV=$(readlink -f "$UV"); UVX=$(dirname "$UV")/uvx
[ -x "$UVX" ] || UVX=$(command -v uvx || echo "$UV")
NAME=glyd-accept-$$
if [ -z "$HOST" ]; then
  IN=/work; INHF=/hf
  if [ -z "$IMAGE" ]; then
    IMAGE=glyd-accept:ubuntu26.04
    printf 'FROM ubuntu:26.04\nRUN apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev ca-certificates && rm -rf /var/lib/apt/lists/*\n' | docker build -q -t $IMAGE - > /dev/null || exit 2
  fi
  E=(-e HOME=$IN/home -e UV_CACHE_DIR=$IN/uv-cache -e HF_HOME=$INHF -e UV_LINK_MODE=copy)
  docker run -d --rm --name $NAME --gpus all --network host --ipc host --user "$(id -u):$(id -g)" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility "${E[@]}" \
    -v "$WORK:$IN" -v "$HF:$INHF" -v "$UV:/usr/local/bin/uv:ro" -v "$UVX:/usr/local/bin/uvx:ro" $IMAGE sleep infinity > /dev/null || exit 2
  run() { docker exec -i -w $IN/home "${E[@]}" $NAME bash -c "$1"; }
  bg() { docker exec -d -w $IN/home "${E[@]}" $NAME bash -c "$1"; }
else
  IN=$WORK; INHF=$HF
  mkdir -p "$WORK/tools"; ln -sf "$UV" "$WORK/tools/uv"; ln -sf "$UVX" "$WORK/tools/uvx"
  E=(env -i HOME=$IN/home PATH=$WORK/tools:/usr/local/bin:/usr/bin:/bin UV_CACHE_DIR=$IN/uv-cache HF_HOME=$INHF UV_LINK_MODE=copy)
  run() { (cd "$IN/home" && "${E[@]}" bash -c "$1"); }
  bg() { (cd "$IN/home" && "${E[@]}" bash -c "$1" > /dev/null 2>&1 &); }
fi
cleanup() {
  if [ -z "$HOST" ]; then docker rm -f $NAME > /dev/null 2>&1; else pkill -u "$(id -u)" -f "$WORK/home/" 2> /dev/null; fi
  [ -n "${OWUI:-}" ] && docker rm -f open-webui > /dev/null 2>&1
}
trap cleanup EXIT

say "== $(date -u +%FT%TZ): glyd[vllm] ${VERSION:+==$VERSION}${WHEEL:+wheel $WHEEL}, ${IMAGE:-the host}, GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), budget ${BUDGET:-all} MiB free, card ${CARD:-as is} MiB"

# --- 1. no nvcc
if run 'command -v nvcc > /dev/null || test -e /usr/local/cuda || test -n "${CUDA_HOME:-}${CUDA_PATH:-}"'; then
  fail "nvcc is reachable here (on PATH, in /usr/local/cuda or CUDA_HOME): not a machine without a CUDA toolkit"; exit 1
fi
ok "no nvcc: not on PATH, no /usr/local/cuda, no CUDA_HOME"

# --- 2. the README's setup block (the venv and the install), its install line replaced by the version or wheel asked for
SETUP=$(block setup)
[ -n "$SETUP" ] || { echo "no <!-- acceptance: setup --> block in $README"; exit 2; }
SPEC=
[ -n "$VERSION" ] && SPEC="glyd[vllm]==$VERSION"
[ -n "$WHEEL" ] && { cp "$WHEEL" "$WORK/wheel/" && SPEC="$IN/wheel/$(basename "$WHEEL")[vllm]"; }
[ -z "$SPEC" ] || SETUP=$(printf '%s\n' "$SETUP" | sed -E "s|^(uv pip install).*|\\1 \"$SPEC\"|")
VENV=$(printf '%s\n' "$SETUP" | sed -nE 's|.*source ([^ ]+)/bin/activate.*|\1|p' | head -1)
VENV=${VENV/#\~/$IN/home}
printf '%s\n' "$SETUP" > "$WORK/setup.sh"
say "-- setup: $(printf '%s' "$SETUP" | tr '\n' ';')"
t0=$(date +%s)
run "cd $IN/home && rm -rf $VENV && bash -e $IN/setup.sh" > "$LOGS/install.log" 2>&1 || { fail "the setup block failed (logs/install.log)"; tail -5 "$LOGS/install.log"; exit 1; }
run "uv pip freeze --python $VENV/bin/python" > "$LOGS/freeze.txt" 2>&1
ok "installed in $(( $(date +%s) - t0 )) s: $(grep -iE '^(glyd|vllm|torch|flashinfer-python|transformers|triton)==' "$LOGS/freeze.txt" | tr '\n' ' ')"

# --- 3. the GPU's memory: a process holds all but the budget (and the plugin reads the GPU as GeForce Ada, if asked)
cat > "$WORK/hog.py" <<'EOF'
import time, torch, sys
want = int(sys.argv[1]); held = []
torch.cuda.init()
while True:
    free = torch.cuda.mem_get_info()[0] >> 20
    if free <= want:
        break
    held.append(torch.empty(min(free - want, 256) << 20, dtype=torch.uint8, device="cuda"))
free, total = (v >> 20 for v in torch.cuda.mem_get_info())
print(f"hog: {sum(t.numel() for t in held) >> 20} MiB held, {free} of {total} MiB free", flush=True)
while True:
    time.sleep(3600)
EOF
mkdir -p "$WORK/geforce" && cat > "$WORK/geforce/sitecustomize.py" <<'EOF'
import importlib.abc, importlib.machinery, sys


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "glyd.gpu.vllm_plugin":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            module._gpu = lambda d: 1089  # glyd_gpu.h: GLYD_GPU_GEFORCE + 89
            print("acceptance: the plugin reads this GPU as GeForce Ada (1089)", file=sys.stderr, flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
EOF
TOTAL=; [ -n "$CARD" ] && TOTAL=$(run "$VENV/bin/python -c 'import torch; print(torch.cuda.mem_get_info()[1] >> 20)'" 2> /dev/null | tail -1)
if [ -n "$BUDGET" ]; then
  # the server's own CUDA context takes some of what the hog leaves (190 MiB on an L4)
  bg "$VENV/bin/python $IN/hog.py $(( BUDGET + 190 )) > $IN/logs/hog.log 2>&1"
  for _ in $(seq 1 60); do grep -q '^hog:' "$LOGS/hog.log" 2> /dev/null && break; sleep 1; done
  say "-- $(cat "$LOGS/hog.log" 2> /dev/null)"
fi

# --- 4. the README's serve command as it is (its memory share scaled to the card, where a card is given)
CMD=${COMMAND:-$(block serve)}
[ -n "$CMD" ] || { echo "no <!-- acceptance: serve --> block in $README"; exit 2; }
if [ -n "$CARD" ] && [ -n "$TOTAL" ] && [ "$TOTAL" -gt "$CARD" ] && [[ $CMD =~ --gpu-memory-utilization[\ =]([0-9.]+) ]]; then
  U=$(awk -v u="${BASH_REMATCH[1]}" -v c="$CARD" -v t="$TOTAL" 'BEGIN {printf "%.3f", u * c / t}')
  say "-- --gpu-memory-utilization ${BASH_REMATCH[1]} is $U here: the same budget as on a card of $CARD MiB (this one has $TOTAL)"
  CMD=${CMD/${BASH_REMATCH[0]}/--gpu-memory-utilization $U}
fi
say "-- serve: $(printf '%s' "$CMD" | tr '\n' ' ' | tr -s ' ')"
{ echo "source $VENV/bin/activate"; [ -n "$GEFORCE" ] && echo "export PYTHONPATH=$IN/geforce"; echo "echo \$\$ > $IN/logs/server.pid"; echo "exec env $CMD"; } > "$WORK/serve.sh"
: > "$LOGS/server.log"; rm -f "$LOGS/server.pid"
bg "bash $IN/serve.sh > $IN/logs/server.log 2>&1"
up=; t0=$(date +%s)
for _ in $(seq 1 600); do
  grep -q 'Application startup complete' "$LOGS/server.log" && { up=1; break; }
  [ -s "$LOGS/server.pid" ] && ! run "kill -0 \$(cat $IN/logs/server.pid)" > /dev/null 2>&1 && break
  sleep 3
done
OOM=$(count 'with OOM' "$LOGS/server.log"); TB=$(count 'Traceback' "$LOGS/server.log")
say "-- server: $([ -n "$up" ] && echo "up after $(( $(date +%s) - t0 )) s" || echo 'did not come up'); $(grep -E 'Model loading took|GPU KV cache size|Free memory on device' "$LOGS/server.log" | sed -E 's/^.*\] //' | cut -c1-140 | tr '\n' ';')"
grep -m1 'glyd: no nvcc' "$LOGS/server.log" | sed -E 's/^.*glyd: /note: glyd: /' | cut -c1-160 | while read -r l; do say "$l"; done
[ -n "$up" ] || fail "the server did not come up (logs/server.log)"
[ "$TB" = 0 ] || fail "$TB tracebacks in the server's log: $(grep -E '^([A-Za-z_]+\.)*[A-Za-z]+(Error|Exception): |^\(EngineCore[^)]*\) ([A-Za-z_.]+)?(Error|Exception): ' "$LOGS/server.log" | sed -E 's/^\([^)]*\) //' | sort | uniq -c | sort -rn | head -2 | sed -E 's/^ +//' | cut -c1-160 | tr '\n' ';')"
[ "$OOM" = 0 ] || fail "$OOM allocator out-of-memory warnings in the server's log ($(count 'memory allocation failed with OOM' "$LOGS/server.log") allocation, $(count 'memory mapping failed with OOM' "$LOGS/server.log") mapping)"
[ "$TB$OOM" != 00 ] || ok "no traceback and no allocator warning in the server's log"

# --- 5. the OpenAI API, and Open WebUI as the browser uses it
cat > "$WORK/check.py" <<'EOF'
import json, re, sys, time, urllib.request


def call(url, body=None, token=None, timeout=300):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})}
    return urllib.request.urlopen(urllib.request.Request(url, None if body is None else json.dumps(body).encode(), h), timeout=timeout)


def stream(r):
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])


def turn(base, model, messages, **p):
    t0, text, usage = time.perf_counter(), [], None
    with call(base + "/chat/completions", {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 400, **p}) as r:
        for d in stream(r):
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                t = c["delta"].get("content")
                if t:
                    text.append(t)
    return "".join(text), f"{(usage or {}).get('completion_tokens', 0)} tokens in {time.perf_counter() - t0:.1f} s"


def check(name, good, detail):
    print(("ok   " if good else "FAIL ") + f"{name}: {detail}", flush=True)
    return good


def api(base):
    model = json.load(call(base + "/models"))["data"][0]["id"]
    q1 = [{"role": "user", "content": "What is the capital of France? Answer in one word. /no_think"}]
    a1, s1 = turn(base, model, q1, temperature=0)
    ok = check("OpenAI API, turn 1 (streamed, greedy)", "paris" in a1.lower(), f"{a1.strip()[:60]!r}, {s1}")
    q2 = q1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And of Germany? Answer in one word. /no_think"}]
    a2, s2 = turn(base, model, q2, temperature=0.7, top_p=0.8, top_k=20)
    ok &= check("OpenAI API, turn 2 (streamed, top-p, turn 1 as history)", "berlin" in a2.lower(), f"{a2.strip()[:60]!r}, {s2}")
    tool = {"type": "function", "function": {"name": "get_weather", "description": "The weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    r = json.load(call(base + "/chat/completions", {"model": model, "tool_choice": "auto", "tools": [tool], "max_tokens": 300, "temperature": 0,
                                                    "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool. /no_think"}]}))
    tc = (r["choices"][0]["message"].get("tool_calls") or [{}])[0].get("function", {})
    return ok & check("OpenAI API, a tool call (tool_choice auto)", tc.get("name") == "get_weather" and "paris" in tc.get("arguments", "").lower(), f"{tc}")


def webui(url, model):
    for _ in range(150):
        try:
            if json.load(call(url + "/health", timeout=5)).get("status"):
                break
        except Exception:
            time.sleep(2)
    token = json.load(call(url + "/api/v1/auths/signin", {"email": "", "password": ""}))["token"]
    ids = [m["id"] for m in json.load(call(url + "/api/models", token=token))["data"]]
    ok = check("Open WebUI lists the model", model in ids, f"/api/models: {ids}")

    def chat(msg):  # the browser's request (Open WebUI 0.11): features, params, a session (with which Open WebUI offers the model its built-in tools)
        body = {"stream": True, "model": model, "messages": [{"role": "user", "content": msg}], "params": {}, "tool_servers": [],
                "features": {"voice": False, "image_generation": False, "code_interpreter": False, "web_search": False, "memory": True},
                "variables": {"{{USER_NAME}}": "User", "{{CURRENT_DATE}}": time.strftime("%Y-%m-%d")}, "session_id": "acceptance"}
        text, tools = [], []
        with call(url + "/api/chat/completions", body, token) as r:
            for d in stream(r):
                for c in d.get("choices", []):
                    delta = c.get("delta", {})
                    text.append(delta.get("content") or "")
                    tools += [t["function"]["name"] for t in delta.get("tool_calls") or [] if t.get("function", {}).get("name")]
        return "".join(text), tools

    a, _ = chat("What is the capital of France? Answer in one word. /no_think")
    ok &= check("Open WebUI chat (the browser's request, its tools on)", "paris" in a.lower(), f"{a.strip()[:80]!r}")
    a, tools = chat("What is the current Unix timestamp? Use your tools. /no_think")
    return ok & check("Open WebUI chat, a tool call in the stream", "get_current_timestamp" in tools, f"tool calls {tools}, text {a.strip()[:60]!r}")


sys.exit(0 if (api(sys.argv[2]) if sys.argv[1] == "api" else webui(sys.argv[2], sys.argv[3])) else 1)
EOF
if [ -n "$up" ]; then
  run "$VENV/bin/python $IN/check.py api http://127.0.0.1:8000/v1" > "$LOGS/chat.log" 2>&1 || fail "the OpenAI API chat (logs/chat.log)"
  tee -a "$WORK/summary.txt" < "$LOGS/chat.log"
  MODEL=$(run "$VENV/bin/python -c \"import json,urllib.request as u; print(json.load(u.urlopen('http://127.0.0.1:8000/v1/models'))['data'][0]['id'])\"" 2> /dev/null | tail -1)
  for route in uvx docker; do
    case $WEBUI in both|*$route*) ;; *) continue;; esac
    say "-- Open WebUI, $route: $(block webui-$route | tr '\n' ' ' | tr -s ' ' | cut -c1-200)"
    block webui-$route > "$WORK/webui-$route.sh"
    if [ $route = uvx ]; then
      bg "setsid bash -c 'echo \$\$ > $IN/logs/webui.pid; exec bash $IN/webui-uvx.sh' > $IN/logs/webui-uvx.log 2>&1"
    else
      OWUI=1
      bash "$WORK/webui-docker.sh" > "$LOGS/webui-docker.log" 2>&1 || { fail "the README's docker command failed (logs/webui-docker.log)"; continue; }
    fi
    run "$VENV/bin/python $IN/check.py webui http://127.0.0.1:3000 '$MODEL'" > "$LOGS/webui-$route.check.log" 2>&1 || fail "Open WebUI, $route (logs/webui-$route.check.log)"
    tee -a "$WORK/summary.txt" < "$LOGS/webui-$route.check.log"
    if [ $route = uvx ]; then run "kill -TERM -- -\$(cat $IN/logs/webui.pid)" > /dev/null 2>&1; else docker rm -f open-webui > /dev/null 2>&1; OWUI=; fi
    for _ in $(seq 1 30); do ss -ltn 2> /dev/null | grep -q ':3000 ' || break; sleep 1; done
    ! ss -ltn 2> /dev/null | grep -q ':3000 ' || { fail "port 3000 is still in use after stopping Open WebUI ($route)"; break; }
  done
  [ "$(count Traceback "$LOGS/server.log")" = "$TB" ] || fail "tracebacks in the server's log during the chats"
  [ "$(count 'with OOM' "$LOGS/server.log")" = "$OOM" ] || fail "allocator warnings in the server's log during the chats"
fi

if [ -n "$HOLD" ] && [ -n "$up" ]; then
  block webui-uvx > "$WORK/webui-uvx.sh"
  bg "setsid bash -c 'echo \$\$ > $IN/logs/webui.pid; exec bash $IN/webui-uvx.sh' > $IN/logs/webui-uvx.log 2>&1"
  say "-- held for $HOLD minutes (touch $WORK/stop to end): the server on http://127.0.0.1:8000, Open WebUI on http://127.0.0.1:3000"
  rm -f "$WORK/stop"
  for _ in $(seq 1 $(( HOLD * 20 ))); do [ -e "$WORK/stop" ] && break; sleep 3; done
fi

if [ ${#FAILS[@]} = 0 ]; then say "PASSED"; else say "FAILED (${#FAILS[@]}): $(printf '%s; ' "${FAILS[@]}")"; fi
[ ${#FAILS[@]} = 0 ]
