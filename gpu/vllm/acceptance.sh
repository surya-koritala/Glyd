#!/usr/bin/env bash
# The release gate for Glyd's local chat: gpu/vllm/README.md's quickstart, run from nothing on a machine with no CUDA
# toolkit, as a user runs it. It fails on a traceback in a server's log, an allocator out-of-memory warning (either
# kind), a missing answer, or a model Open WebUI does not list. Two flows:
#
#   --flow glyd   (the default) "Local chat, like Ollama": the README's two commands. In a clean container (Ubuntu
#                 26.04, a user that is not root, no CUDA toolkit and, by default, no C compiler) it runs install.sh
#                 (this checkout's, in place of the copy getglyd.com serves) with the README's install line, then
#                 `glyd doctor`, `glyd run MODEL --prompt`, then `glyd serve MODEL` and over HTTP: the chat page, the
#                 OpenAI API (two streamed turns, the thinking apart, a tool call), a conversation longer than the
#                 window (the API's refusal, and the terminal chat's own message), and Open WebUI by the README's
#                 routes. It lists what the install resolved (uv's tool list and pip freeze) and fails on a pre-release
#                 among the dependencies. Where the budget is too small for MODEL (--card 8gb) it expects the refusal
#                 with a model to try, and runs that model.
#   --flow vllm   the README's "Advanced" section: it makes a clean environment, installs glyd[vllm] (a version from
#                 PyPI, or a wheel), hides nvcc, leaves the server only the GPU memory of a card, runs the section's
#                 serve command as the README gives it, chats through the OpenAI API (streaming, two turns, a tool
#                 call), and runs Open WebUI on it by the routes the README gives (uvx, Docker), chatting through its
#                 chat endpoint as the browser does.
#
#   bash acceptance.sh [--flow glyd|vllm] [--version V | --wheel FILE] [--budget-mib N] [--card-mib N] [--geforce]
#                      [--card 4080s|8gb] [--model M] [--compiler none|gcc] [--command CMD] [--webui ROUTES]
#                      [--hold MINUTES] [--image IMG] [--work DIR] [--host] [--pip-refusal]
#
#   --version V      glyd[vllm]==V from PyPI (glyd flow: install.sh with GLYD_VERSION=V; vllm flow: in place of the
#                    README's install line); --wheel FILE: this wheel, with its vllm extra; with neither, the glyd flow
#                    installs the release install.sh names, and the vllm flow the README's install line as it is
#   --budget-mib N   the GPU memory the server finds free when it starts (vLLM's "Free memory on device"), in MiB: a
#                    process holds the rest of the GPU. --card-mib N: the card's total as CUDA reports it (an RTX 4080
#                    SUPER's is 15942): the vllm flow's --gpu-memory-utilization is scaled so that its budget in GiB is
#                    the card's (glyd works that out itself). --geforce: the plugin reads the GPU as GeForce Ada (for an
#                    L4 standing in for one). On an RTX 4080 SUPER none of the three is needed.
#                    --card 4080s is --budget-mib 14828 --card-mib 15942 --geforce (the owner's card with a desktop);
#                    --card 8gb is --budget-mib 7500, expects the refusal of the model (Qwen/Qwen3-4B, unless --model
#                    says another) with a smaller one to try, and runs that
#   --model M        the model glyd run and glyd serve are given (default Qwen/Qwen3-8B)
#   --compiler C     the glyd flow's container: none (the default: no gcc, so install.sh adds ziglang) or gcc
#   --command CMD    the vllm flow's serve command in place of the README's
#   --webui ROUTES   the Open WebUI routes to run: uvx, docker (host network), bridge (the container's own network, which
#                    needs the server on 0.0.0.0 with a key), both (uvx and docker; the default), all, or none
#   --hold MINUTES   after the checks leave the server and Open WebUI (the uvx route) up for a person to try, until the
#                    time is up or DIR/stop exists
#   --image IMG      the container's image. The default is Ubuntu 26.04 with curl (and gcc for the vllm flow or
#                    --compiler gcc) and nothing else: no nvcc, no /usr/local/cuda
#   --work DIR       its environment, caches and logs (default ~/glyd-acceptance); HF_HOME is the model's cache (an
#                    empty one makes the download part of the run)
#   --host           the vllm flow only: no container, the commands run in a stripped environment on this machine (nvcc
#                    must not be found, so there must be no /usr/local/cuda)
#   --pip-refusal    the glyd flow, after the checks: pip-install the same wheel into a virtual environment (no
#                    installer, so no ziglang) and expect `glyd run` to stop with the compiler's install command
#                    (only with --compiler none)
#
# Needs: Linux, an NVIDIA GPU and its driver, Docker with the NVIDIA container toolkit (or --host, and Docker for the
# Docker route), and the network. Ports 8000 and 3000 must be free. It runs the README's blocks marked
# "<!-- acceptance: install | run | webui-uvx | webui-docker | webui-bridge | setup | serve -->". Exit status 0 if
# nothing failed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
README=$HERE/README.md
FLOW=glyd VERSION= WHEEL= BUDGET= CARD= GEFORCE= COMMAND= WEBUI=both HOLD= IMAGE= WORK=$HOME/glyd-acceptance HOST= MODEL=Qwen/Qwen3-8B COMPILER=none EXPECT=fit PIPREF= MODEL_SET=
while [ $# -gt 0 ]; do
  case $1 in
    --flow) FLOW=$2; shift;;
    --version) VERSION=$2; shift;; --wheel) WHEEL=$2; shift;;
    --budget-mib) BUDGET=$2; shift;; --card-mib) CARD=$2; shift;; --geforce) GEFORCE=1;;
    --card) case $2 in 4080s) BUDGET=14828 CARD=15942 GEFORCE=1;; 8gb) BUDGET=7500 EXPECT=refusal;; *) echo "unknown card $2 (4080s, 8gb)"; exit 2;; esac; shift;;
    --model) MODEL=$2 MODEL_SET=1; shift;; --compiler) COMPILER=$2; shift;;
    --command) COMMAND=$2; shift;; --webui) WEBUI=$2; shift;; --hold) HOLD=$2; shift;; --image) IMAGE=$2; shift;; --work) WORK=$2; shift;; --host) HOST=1;;
    --pip-refusal) PIPREF=1;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d;s/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option $1 (--help)"; exit 2;;
  esac
  shift
done
case $FLOW in glyd|vllm) ;; *) echo "unknown flow $FLOW (glyd, vllm)"; exit 2;; esac
[ $EXPECT = fit ] || [ -n "$MODEL_SET" ] || MODEL=Qwen/Qwen3-4B  # (the 8 GB card: the model that does not quite fit)
case $COMPILER in none|gcc) ;; *) echo "unknown compiler $COMPILER (none, gcc)"; exit 2;; esac
[ -z "$HOST" ] || [ "$FLOW" = vllm ] || { echo "--host is for the vllm flow: the glyd flow installs into a home of its own, in a container"; exit 2; }
[ "$WEBUI" != all ] || WEBUI=uvx,docker,bridge
[ "$WEBUI" != both ] || WEBUI=uvx,docker
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
route() { case ",$WEBUI," in *,"$1",*) return 0;; *) return 1;; esac; }
for p in 8000 3000; do
  ss -ltn 2> /dev/null | grep -q ":$p " && { echo "port $p is in use: the server and Open WebUI need it"; exit 2; }
done
route docker || route bridge && { docker ps -a --format '{{.Names}}' | grep -qx open-webui && { echo "a container named open-webui exists: remove it, or run with --webui uvx"; exit 2; }; }

# --- the helpers both flows use: the process that holds the GPU's memory, the GeForce emulation, and the chats' checks
write_hog() {
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
}
start_hog() {  # start_hog PYTHON: a process of this Python holds all of the GPU's memory but the budget
  [ -n "$BUDGET" ] || return 0
  write_hog
  # the server's own CUDA context takes some of what the hog leaves (190 MiB on an L4)
  bg "$1 $IN/hog.py $(( BUDGET + 190 )) > $IN/logs/hog.log 2>&1"
  for _ in $(seq 1 60); do grep -q '^hog:' "$LOGS/hog.log" 2> /dev/null && break; sleep 1; done
  say "-- $(cat "$LOGS/hog.log" 2> /dev/null)"
}
write_check() {
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


def thinking(base):
    """The reasoning parser: the thinking in a field of its own, the answer's content with no <think> in it."""
    model = json.load(call(base + "/models"))["data"][0]["id"]
    reasoning, content = [], []
    with call(base + "/chat/completions", {"model": model, "stream": True, "max_tokens": 2000, "temperature": 0, "messages": [{"role": "user", "content": "What is 17 times 3? Work it out, then give the number."}]}) as r:
        for d in stream(r):
            for c in d.get("choices", []):
                reasoning.append(c["delta"].get("reasoning") or c["delta"].get("reasoning_content") or "")
                content.append(c["delta"].get("content") or "")
    reasoning, content = "".join(reasoning), "".join(content)
    return check("the thinking apart from the answer", bool(reasoning.strip()) and "<think>" not in content and "</think>" not in content and "51" in content,
                 f"{len(reasoning)} characters of reasoning, answer {content.strip()[:50]!r}")


def too_long(base):
    """A conversation longer than the window: the server's plain refusal (HTTP 400, and the number it names)."""
    d = json.load(call(base + "/models"))["data"][0]
    try:
        call(base + "/chat/completions", {"model": d["id"], "max_tokens": 10, "messages": [{"role": "user", "content": "word " * (int(d["max_model_len"]) + 1000)}]})
    except urllib.error.HTTPError as e:
        msg = json.loads(e.read())["error"]["message"]
        return check("a conversation longer than the window", e.code == 400 and f"maximum context length is {d['max_model_len']}" in msg, f"HTTP {e.code}, {msg[:90]!r}")
    return check("a conversation longer than the window", False, "the server took it")


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


what = sys.argv[1]
sys.exit(0 if (api(sys.argv[2]) & thinking(sys.argv[2]) & too_long(sys.argv[2]) if what == "api" else webui(sys.argv[2], sys.argv[3])) else 1)
EOF
}

# --- uv, and where the commands run: a container with no CUDA toolkit, or (the vllm flow) a stripped environment
UV= UVX=
if [ "$FLOW" = vllm ]; then
  UV=$(command -v uv || true)
  if [ -z "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$WORK/bin UV_NO_MODIFY_PATH=1 sh > "$LOGS/uv-install.log" 2>&1 || { echo "uv is not installed, and its installer failed"; exit 2; }
    UV=$WORK/bin/uv
  fi
  UV=$(readlink -f "$UV"); UVX=$(dirname "$UV")/uvx
  [ -x "$UVX" ] || UVX=$(command -v uvx || echo "$UV")
fi
NAME=glyd-accept-$$
VOLUME=$NAME-open-webui
if [ -z "$HOST" ]; then
  IN=/work; INHF=/hf
  if [ -z "$IMAGE" ]; then
    PKGS="ca-certificates curl procps"; TAG=curl
    if [ "$FLOW" = vllm ] || [ "$COMPILER" = gcc ]; then PKGS="$PKGS gcc libc6-dev"; TAG=gcc; fi
    IMAGE=glyd-accept:ubuntu26.04-$TAG
    printf 'FROM ubuntu:26.04\nRUN apt-get update -qq && apt-get install -y -qq --no-install-recommends %s && rm -rf /var/lib/apt/lists/*\n' "$PKGS" | docker build -q -t $IMAGE - > /dev/null || exit 2
  fi
  E=(-e HOME=$IN/home -e UV_CACHE_DIR=$IN/uv-cache -e HF_HOME=$INHF -e UV_LINK_MODE=copy)
  MOUNTS=()
  if [ "$FLOW" = vllm ]; then MOUNTS=(-v "$UV:/usr/local/bin/uv:ro" -v "$UVX:/usr/local/bin/uvx:ro"); else E+=(-e PATH=$IN/home/.local/bin:/usr/local/bin:/usr/bin:/bin); fi
  docker run -d --rm --name $NAME --gpus all --network host --ipc host --user "$(id -u):$(id -g)" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility "${E[@]}" \
    -v "$WORK:$IN" -v "$HF:$INHF" ${MOUNTS[@]+"${MOUNTS[@]}"} $IMAGE sleep infinity > /dev/null || exit 2
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
  [ -n "${OWUI:-}" ] && docker rm -f open-webui > /dev/null 2>&1 && docker volume rm "$VOLUME" > /dev/null 2>&1
  true
}
trap cleanup EXIT

say "== $(date -u +%FT%TZ): $FLOW flow, glyd[vllm] ${VERSION:+==$VERSION}${WHEEL:+wheel $WHEEL}, ${IMAGE:-the host}, GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), budget ${BUDGET:-all} MiB free, card ${CARD:-as is} MiB"

# --- 1. no nvcc
if run 'command -v nvcc > /dev/null || test -e /usr/local/cuda || test -n "${CUDA_HOME:-}${CUDA_PATH:-}"'; then
  fail "nvcc is reachable here (on PATH, in /usr/local/cuda or CUDA_HOME): not a machine without a CUDA toolkit"; exit 1
fi
ok "no nvcc: not on PATH, no /usr/local/cuda, no CUDA_HOME"

# --- Open WebUI, one route at a time against the server on port 8000 (the README's block, a volume of its own where Docker keeps one)
webui_route() {  # webui_route ROUTE PYTHON MODEL [KEY]
  local route=$1 py=$2 model=$3 key=${4:-none}
  say "-- Open WebUI, $route: $(block webui-$route | tr '\n' ' ' | tr -s ' ' | cut -c1-200)"
  block webui-$route | sed "s|-v open-webui:|-v $VOLUME:|; s|YOUR_KEY|$key|g" > "$WORK/webui-$route.sh"
  case $route in
    uvx) bg "setsid bash -c 'echo \$\$ > $IN/logs/webui.pid; exec bash $IN/webui-uvx.sh' > $IN/logs/webui-uvx.log 2>&1";;
    *) OWUI=1; bash "$WORK/webui-$route.sh" > "$LOGS/webui-$route.log" 2>&1 || { fail "the README's $route command failed (logs/webui-$route.log)"; return; };;
  esac
  run "$py $IN/check.py webui http://127.0.0.1:3000 '$model'" > "$LOGS/webui-$route.check.log" 2>&1 || fail "Open WebUI, $route (logs/webui-$route.check.log)"
  tee -a "$WORK/summary.txt" < "$LOGS/webui-$route.check.log"
  if [ $route = uvx ]; then run "kill -TERM -- -\$(cat $IN/logs/webui.pid)" > /dev/null 2>&1; else docker rm -f open-webui > /dev/null 2>&1; docker volume rm "$VOLUME" > /dev/null 2>&1; OWUI=; fi
  for _ in $(seq 1 30); do ss -ltn 2> /dev/null | grep -q ':3000 ' || break; sleep 1; done
  ! ss -ltn 2> /dev/null | grep -q ':3000 ' || { fail "port 3000 is still in use after stopping Open WebUI ($route)"; return 1; }
}

# --- the glyd flow: install.sh, glyd doctor, glyd run, glyd serve, the page, the API, Open WebUI
glyd_logs() {  # the traceback and allocator checks over every server log this home has
  local f tb oom n=0 body
  for f in "$WORK"/home/.local/state/glyd/logs/*.log; do
    [ -f "$f" ] || continue
    # (a traceback after vLLM's own "[shutdown]" lines is its output handler finding the engine stopped, which glyd's stop does on purpose: vLLM 0.30 logs it at ERROR on some stops)
    body=$(sed '/\[shutdown\]/q' "$f")
    n=$((n + 1)); tb=$(printf '%s\n' "$body" | grep -c Traceback || true); oom=$(count 'with OOM' "$f")
    [ "$tb" = 0 ] || fail "$tb tracebacks in $(basename "$f") before its shutdown: $(printf '%s\n' "$body" | grep -E '^([A-Za-z_]+\.)*[A-Za-z]+(Error|Exception): |^\(EngineCore[^)]*\) ([A-Za-z_.]+)?(Error|Exception): |Traceback' | sed -E 's/^\([^)]*\) //' | sort | uniq -c | sort -rn | head -3 | sed -E 's/^ +//' | cut -c1-160 | tr '\n' ';')"
    [ "$oom" = 0 ] || fail "$oom allocator out-of-memory warnings in $(basename "$f") ($(count 'memory allocation failed with OOM' "$f") allocation, $(count 'memory mapping failed with OOM' "$f") mapping)"
    cp "$f" "$LOGS/" 2> /dev/null
  done
  [ "$n" != 0 ] || fail "glyd wrote no server log (~/.local/state/glyd/logs)"
  say "-- $n server logs checked: no traceback, no allocator out-of-memory warning unless failed above"
}

glyd_flow() {
  local line env rc t0 py gf ans new up key
  # 2. the README's install line, with this checkout's install.sh where the line fetches getglyd.com's
  line=$(block install)
  [ "$line" = 'curl -LsSf https://getglyd.com/install.sh | sh' ] || { fail "the README's <!-- acceptance: install --> block is not the one line this script replaces: $line"; return 1; }
  cp "$HERE/../../scripts/install.sh" "$WORK/install.sh" || { fail "no scripts/install.sh beside gpu/vllm"; return 1; }
  env=
  [ -n "$VERSION" ] && env="GLYD_VERSION=$VERSION "
  [ -n "$WHEEL" ] && { cp "$WHEEL" "$WORK/wheel/" && env="GLYD_SPEC='$IN/wheel/$(basename "$WHEEL")[vllm]' "; }
  say "-- install: $line   (this checkout's scripts/install.sh instead of getglyd.com's${env:+; $env})"
  rm -rf "$WORK/home/.local" "$WORK/home/.cache"
  t0=$(date +%s)
  run "${env}sh $IN/install.sh" > "$LOGS/install.log" 2>&1; rc=$?
  [ $rc = 0 ] || { fail "install.sh exited $rc (logs/install.log)"; tail -8 "$LOGS/install.log"; return 1; }
  ok "install.sh took $(( $(date +%s) - t0 )) s and exited 0; its last lines: $(tail -n 3 "$LOGS/install.log" | tr '\n' '|' | cut -c1-200)"
  grep -q 'glyd doctor' "$LOGS/install.log" && grep -q 'Ready: glyd run' "$LOGS/install.log" || fail "install.sh did not end with glyd doctor saying Ready: glyd run ... (logs/install.log)"
  run 'command -v glyd' > /dev/null 2>&1 || fail "glyd is not on the PATH after the install"
  grep -qE 'local/bin' "$WORK/home/.bashrc" "$WORK/home/.profile" 2> /dev/null && ok "uv's PATH line is in the shell's profile (a new terminal finds glyd)" || say "note: no PATH line in .bashrc or .profile (uv tool update-shell found the path already set, or could not write)"
  py=$(run 'echo $(uv tool dir)/glyd/bin/python' | tail -1)
  run "uv tool list --show-with; uv pip freeze --python $py" > "$LOGS/freeze.txt" 2>&1
  say "-- resolved: $(grep -iE '^(glyd|vllm|torch|flashinfer-python|transformers|safetensors|tokenizers|pydantic|triton|ziglang)[ =@]' "$LOGS/freeze.txt" | sed -E 's/ \(.*//; s/ @ .*//' | tr '\n' ' ')"
  # (opentelemetry's instrumentation packages publish only betas, 0.66b0: no stable release to take instead; pydantic, safetensors and tokenizers have both)
  pre=$(grep -E '^[A-Za-z0-9_.-]+==[0-9][0-9.]*(a|b|rc|dev)[0-9]+' "$LOGS/freeze.txt" | grep -v '^glyd==' | grep -vE '^opentelemetry-[a-z-]+==[0-9.]+b[0-9]+$' | tr '\n' ' ')
  [ -z "$pre" ] && ok "no pre-release among the dependencies (install.sh names no --prerelease; opentelemetry's betas are all there is of those)" || fail "pre-releases among the dependencies: $pre"
  if [ $COMPILER = none ]; then
    grep -q '^ziglang==' "$LOGS/freeze.txt" && ok "no gcc: install.sh added $(grep -o '^ziglang==[0-9.]*' "$LOGS/freeze.txt")" || fail "no gcc here, and install.sh added no ziglang"
  else
    grep -q '^ziglang==' "$LOGS/freeze.txt" && fail "ziglang was installed on a machine with gcc"
  fi
  run 'glyd doctor' > "$LOGS/doctor.txt" 2>&1; rc=$?
  [ $rc = 0 ] && grep -q 'Ready: glyd run' "$LOGS/doctor.txt" && ok "glyd doctor: $(grep 'Ready:' "$LOGS/doctor.txt")" || { fail "glyd doctor (logs/doctor.txt)"; cat "$LOGS/doctor.txt"; }
  # 3. the GPU's memory: a process holds all but the budget
  start_hog "$py"
  [ -z "$BUDGET" ] || { run 'glyd doctor' > "$LOGS/doctor-budget.txt" 2>&1; say "-- glyd doctor under the budget:"; sed -n '/^Models that fit/,$p' "$LOGS/doctor-budget.txt" | sed 's/^/     /' | tee -a "$WORK/summary.txt"; }
  gf=; [ -n "$GEFORCE" ] && gf="PYTHONPATH=$IN/geforce "
  write_check
  # 4. the README's run line, with --prompt (the chat itself is a terminal's): one answer
  line=$(block run)
  [ "$line" = "glyd run $MODEL" ] || say "note: the README's run block ($line) is not 'glyd run $MODEL': this run uses $MODEL"
  ans=$MODEL
  run "${gf}glyd run $MODEL --prompt 'What is the capital of France? Answer in one word. /no_think'" > "$LOGS/run.out" 2> "$LOGS/run.err"; rc=$?
  if [ $EXPECT = refusal ]; then
    if [ $rc != 0 ] && grep -q 'needs about' "$LOGS/run.err" && grep -qE 'Or try [A-Za-z0-9_./-]+, which needs about' "$LOGS/run.err"; then
      ok "$MODEL refused, with a model to try: $(grep -E 'needs about|Or try' "$LOGS/run.err" | sed 's/^ *//' | cut -c1-200 | tr '\n' '|')"
      grep -qF "needs about" "$README" && grep -qF "Or try" "$README" || fail "the README does not quote the refusal's wording (needs about ... Or try ...)"
      new=$(sed -nE 's/.*Or try ([A-Za-z0-9_./-]+),.*/\1/p' "$LOGS/run.err" | head -1)
      ans=$new
      run "${gf}glyd run $new --prompt 'What is the capital of France? Answer in one word. /no_think'" > "$LOGS/run2.out" 2> "$LOGS/run2.err"; rc=$?
      cat "$LOGS/run2.err" >> "$LOGS/run.err"; cp "$LOGS/run2.out" "$LOGS/run.out"
    else
      fail "$MODEL was not refused with a model to try (exit $rc): $(tail -3 "$LOGS/run.err" | tr '\n' '|')"
    fi
  fi
  if [ $rc = 0 ] && grep -qi paris "$LOGS/run.out"; then
    ok "glyd run $ans --prompt: $(tr -d '\n' < "$LOGS/run.out" | cut -c1-60), $(grep -E '^Settings:' "$LOGS/run.err" | cut -c1-260)"
    say "     $(grep -E '^Ready in' "$LOGS/run.err" | cut -c1-80)"
  else
    fail "glyd run $ans --prompt (exit $rc): $(tail -3 "$LOGS/run.err" | tr '\n' '|' | cut -c1-300) (logs/run.err)"
  fi
  grep -qE 'Traceback|unexpected error' "$LOGS/run.out" "$LOGS/run.err" && fail "a traceback or an unexpected error in glyd run's own output"
  glyd_logs
  [ $EXPECT = refusal ] && { glyd_pip; return; }
  # 5. glyd serve, then the page, the API, the longer-than-the-window conversation, Open WebUI
  : > "$LOGS/serve.out"
  bg "${gf}glyd serve $MODEL --port 8000 > $IN/logs/serve.out 2>&1"
  up=; t0=$(date +%s)
  for _ in $(seq 1 600); do
    run 'curl -sf http://127.0.0.1:8000/v1/models' > /dev/null 2>&1 && { up=1; break; }
    run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break
    sleep 3
  done
  [ -n "$up" ] || { fail "glyd serve did not come up (logs/serve.out)"; tail -5 "$LOGS/serve.out"; glyd_logs; return 1; }
  ok "glyd serve up after $(( $(date +%s) - t0 )) s: $(grep -E '^(Settings|Ready in)' "$LOGS/serve.out" | cut -c1-150 | tr '\n' '|')"
  run "curl -s -D $IN/logs/page.head -o $IN/logs/page.html http://127.0.0.1:8000/" > /dev/null 2>&1
  head -1 "$LOGS/page.head" | grep -q ' 200' && grep -qi '^content-type: text/html' "$LOGS/page.head" && grep -q '<title>Glyd' "$LOGS/page.html" && ok "GET / is the chat page ($(wc -c < "$LOGS/page.html" | tr -d ' ') bytes, $(grep -io '^content-security-policy: [^;]*' "$LOGS/page.head" | tr -d '\r'))" || fail "GET / is not the chat page (logs/page.head)"
  grep -qE "(src|href)=[\"']https?:|url\(https?:|@import" "$LOGS/page.html" && fail "the chat page names an external resource"
  [ "$(run 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/nope')" = 404 ] && ok "other paths stay vLLM's (/nope is 404; the API answers below)" || fail "/nope is not 404"
  run "$py $IN/check.py api http://127.0.0.1:8000/v1" > "$LOGS/chat.log" 2>&1 || fail "the OpenAI API checks (logs/chat.log)"
  tee -a "$WORK/summary.txt" < "$LOGS/chat.log"
  big=$(run "yes word | head -n 60000 | tr '\n' ' ' | ${gf}glyd run $MODEL 2>&1 >/dev/null; echo rc=\$?" | tail -3 | tr '\n' ' ')
  case $big in *"This prompt is longer than the model's window"*"rc=1"*) ok "glyd run --prompt (a prompt on stdin) says so when the prompt outgrows the window: $(printf '%s' "$big" | cut -c1-150)";; *) fail "no plain message from glyd run for a prompt longer than the window: $big";; esac
  win=$(run 'curl -s http://127.0.0.1:8000/v1/models' | sed -nE 's/.*"max_model_len":([0-9]+).*/\1/p' | head -1)
  chatout=$(run "( sleep 6; echo '\"\"\"'; yes \"\$(yes word | head -n 600 | tr '\n' ' ')\" | head -n \$(( ${win:-10240} / 600 + 3 )); echo '\"\"\"'; sleep 10; echo /bye ) | ${gf}script -qec 'glyd run $MODEL' /dev/null" 2>&1 | tr -d '\r' | sed 's/\x1b\[[0-9;?]*[A-Za-z]//g')
  grep -qF "This conversation is longer than the model's window" "$README" && grep -qF "Start a new chat with /clear" "$README" || fail "the README does not quote the message the terminal chat gives for a conversation past the window"
  case $chatout in *"This conversation is longer than the model's window"*"Start a new chat with /clear"*) ok "the terminal chat says so when the conversation outgrows the window (typed as one message of several lines)";; *) fail "no plain message in the terminal chat for a conversation longer than the window: $(printf '%s' "$chatout" | tail -5 | tr '\n' '|' | cut -c1-300)";; esac
  for r in uvx docker; do route $r && { webui_route $r "$py" "$MODEL" || break; }; done
  if route bridge; then  # the container on its own network reaches the server on the host's address, which needs the server on every interface and a key
    run 'pkill -TERM -f "[g]lyd serve"' > /dev/null 2>&1
    for _ in $(seq 1 60); do run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break; sleep 1; done
    key=acceptance-$RANDOM$RANDOM
    bg "${gf}glyd serve $MODEL --port 8000 --host 0.0.0.0 -- --api-key $key > $IN/logs/serve-bridge.out 2>&1"
    up=; for _ in $(seq 1 600); do run "curl -sf -H 'Authorization: Bearer $key' http://127.0.0.1:8000/v1/models" > /dev/null 2>&1 && { up=1; break; }; sleep 3; done
    [ -n "$up" ] || fail "glyd serve --host 0.0.0.0 -- --api-key did not come up (logs/serve-bridge.out)"
    [ "$(run 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models')" = 401 ] && ok "the key is asked for (/v1/models without it is 401)" || fail "the server on 0.0.0.0 took a request without its key"
    [ -z "$up" ] || webui_route bridge "$py" "$MODEL" "$key"
  fi
  if [ -n "$HOLD" ]; then
    block webui-uvx > "$WORK/webui-uvx.sh"
    bg "setsid bash -c 'echo \$\$ > $IN/logs/webui.pid; exec bash $IN/webui-uvx.sh' > $IN/logs/webui-uvx.log 2>&1"
    say "-- held for $HOLD minutes (touch $WORK/stop to end): the server on http://127.0.0.1:8000, Open WebUI on http://127.0.0.1:3000"
    rm -f "$WORK/stop"
    for _ in $(seq 1 $(( HOLD * 20 ))); do [ -e "$WORK/stop" ] && break; sleep 3; done
  fi
  # 6. stopped: the port and the GPU's memory let go
  run 'pkill -TERM -f "[g]lyd serve"' > /dev/null 2>&1
  for _ in $(seq 1 60); do run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break; sleep 1; done
  run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 && fail "glyd serve did not stop on SIGTERM" || ok "glyd serve stopped on SIGTERM"
  ss -ltn 2> /dev/null | grep -q ':8000 ' && fail "port 8000 is still in use after glyd serve stopped" || true
  glyd_logs
  glyd_pip
}

glyd_pip() {  # --pip-refusal: the same wheel by pip into a virtual environment: no installer, no ziglang: the compiler's refusal
  [ -n "$PIPREF" ] && [ $COMPILER = none ] || return 0
  local spec=${WHEEL:+$IN/wheel/$(basename "$WHEEL")[vllm]} out
  spec=${spec:-glyd[vllm]${VERSION:+==$VERSION}}
  run "uv venv --python 3.12 $IN/pipenv > /dev/null 2>&1 && uv pip install --python $IN/pipenv/bin/python '$spec' > $IN/logs/pip.log 2>&1" || { fail "the pip install did not work (logs/pip.log)"; return; }
  out=$(run "$IN/pipenv/bin/glyd run $MODEL --prompt hi 2>&1; echo rc=\$?" | tr '\n' ' ')
  case $out in
    *"needs a C compiler"*"build-essential"*"installer again"*"rc=1"*) ok "pip install, no compiler: glyd run stops with the install command ($(printf '%s' "$out" | cut -c1-200))";;
    *) fail "pip install, no compiler: not the refusal expected: $out";;
  esac
}

if [ "$FLOW" = glyd ]; then
  glyd_flow
  if [ ${#FAILS[@]} = 0 ]; then say "PASSED"; else say "FAILED (${#FAILS[@]}): $(printf '%s; ' "${FAILS[@]}")"; fi
  [ ${#FAILS[@]} = 0 ]
  exit
fi

# --- the vllm flow. 2. the README's setup block (the venv and the install), its install line replaced by the version or wheel asked for
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
write_hog
TOTAL=; [ -n "$CARD" ] && TOTAL=$(run "$VENV/bin/python -c 'import torch; print(torch.cuda.mem_get_info()[1] >> 20)'" 2> /dev/null | tail -1)
start_hog "$VENV/bin/python"

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
write_check
if [ -n "$up" ]; then
  run "$VENV/bin/python $IN/check.py api http://127.0.0.1:8000/v1" > "$LOGS/chat.log" 2>&1 || fail "the OpenAI API chat (logs/chat.log)"
  tee -a "$WORK/summary.txt" < "$LOGS/chat.log"
  MODEL=$(run "$VENV/bin/python -c \"import json,urllib.request as u; print(json.load(u.urlopen('http://127.0.0.1:8000/v1/models'))['data'][0]['id'])\"" 2> /dev/null | tail -1)
  for r in uvx docker; do route $r && { webui_route $r "$VENV/bin/python" "$MODEL" || break; }; done
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
