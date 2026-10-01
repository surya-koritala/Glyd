#!/usr/bin/env bash
# The release gate for Glyd's local chat: gpu/vllm/README.md's quickstart, run from nothing on a machine with no CUDA
# toolkit, as a user runs it. It fails on a traceback in a server's log, an allocator out-of-memory warning (either
# kind), a missing answer, or a model Open WebUI does not list. Three flows:
#
#   --flow glyd   (the default) "Local chat, like Ollama": the README's two commands. In a clean container (Ubuntu
#                 26.04, a user that is not root, no CUDA toolkit and, by default, no C compiler) it runs install.sh
#                 (this checkout's, in place of the copy getglyd.com serves) with the README's install line, then
#                 `glyd doctor`, `glyd run MODEL --prompt`, then `glyd serve MODEL` and over HTTP: the chat page, the
#                 OpenAI API (two streamed turns, the thinking apart, a tool call), a conversation longer than the
#                 window (the API's refusal, and the terminal chat's own message), and Open WebUI by the README's
#                 routes (its login is on: nothing is answered without one, its first account is signed up as a person
#                 does on first visit and is the administrator, and its CORS is limited to its own addresses). It lists
#                 what the install resolved (uv's tool list and pip freeze) and fails on a pre-release among the
#                 dependencies, and on a package that is not at the version install.sh's list gives. Where the
#                 budget is too small for MODEL (--card 8gb) it expects the refusal with a model to try, and runs that model.
#                 Around those, what a user meets: install.sh where a program of the user's own is where uv puts its link (it
#                 stops, and leaves it), run again (an update: the shell-profile edit said where ~/.local/bin is off PATH, a
#                 glyd ahead of it on PATH said with the line to add, glyd doctor's own row for it); glyd serve under nohup
#                 (SIGHUP does not stop it), through the Rust glyd with --rust-glyd; the server's refusal of another site's
#                 script (Origin) and a rebinding name (Host); a server with an API key given as -- --api-key and as the
#                 README's VLLM_API_KEY (ready, the key on no command line or in a log, the note on what stays open); an
#                 answer to a prompt full of escape sequences; Ctrl-C while the model loads (once, twice); the engine killed
#                 under a terminal chat; uv tool uninstall glyd.
#   --flow cli    install.sh on a machine with no NVIDIA GPU (a container started without --gpus): the compression program
#                 from the release's tarball (its sha256 checked), no uv or tool environment, glyd run said to need Linux
#                 with an NVIDIA GPU, a file through glyd and back, an update, and a program of the user's own left alone.
#   --flow vllm   the README's "Advanced" section: it makes a clean environment, installs glyd[vllm] (a version from
#                 PyPI, or a wheel), hides nvcc, leaves the server only the GPU memory of a card, runs the section's
#                 serve command as the README gives it, chats through the OpenAI API (streaming, two turns, a tool
#                 call), and runs Open WebUI on it by the routes the README gives (uvx, Docker), chatting through its
#                 chat endpoint as the browser does.
#
#   bash acceptance.sh [--flow glyd|vllm|cli] [--version V | --wheel FILE] [--budget-mib N] [--card-mib N] [--geforce]
#                      [--card 4080s|8gb] [--model M] [--compiler none|gcc] [--command CMD] [--webui ROUTES]
#                      [--hold MINUTES] [--image IMG] [--work DIR] [--host] [--pip-refusal] [--rust-glyd FILE] [--reuse]
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
#   --rust-glyd FILE the Rust glyd built from this tree (cargo build --release --bin glyd): the glyd flow puts it first on PATH
#                    for glyd doctor and glyd serve (it passes run, serve, doctor and login to the Python tool), the cli flow
#                    for glyd run (there is no tool: it says so)
#   --reuse          the glyd flow, in the --work of a run that installed: not installed (or checked) again, the rest run
#                    (the way to run only what comes after the install again)
#   --up-wait SECONDS  how long the glyd flow waits for a server it started with a key, or to kill (default 1800, glyd's own limit):
#                    shorter where the code under test is expected not to get ready
#   --install-url URL  the glyd flow's install.sh from this URL (a release's raw GitHub file) in place of this checkout's, which
#                    is what a release's acceptance runs; with neither --version nor --wheel it installs what that script pins,
#                    from PyPI
#   --served         the glyd flow installs by the README's line as it is, `curl -LsSf https://getglyd.com/install.sh | sh`,
#                    from the network: the script the site serves. A copy of it is kept for the cases that run the script
#                    again, and with --install-url it is compared with that file (no --version or --wheel: the line is run as it is)
#
# Needs: Linux, an NVIDIA GPU and its driver, Docker with the NVIDIA container toolkit (or --host, and Docker for the
# Docker route), and the network. Ports 8000 and 3000 must be free. It runs the README's blocks marked
# "<!-- acceptance: install | run | serve-key | webui-uvx | webui-docker | webui-bridge | setup | serve -->". Exit
# status 0 if nothing failed. The logs of the last run only: DIR/logs is emptied when it starts.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
README=$HERE/README.md
FLOW=glyd VERSION= WHEEL= BUDGET= CARD= GEFORCE= COMMAND= WEBUI=both HOLD= IMAGE= WORK=$HOME/glyd-acceptance HOST= MODEL=Qwen/Qwen3-8B COMPILER=none EXPECT=fit PIPREF= MODEL_SET= RUSTGLYD= REUSE= UPWAIT=1800 INSTALL_URL= SERVED=
while [ $# -gt 0 ]; do
  case $1 in
    --flow) FLOW=$2; shift;;
    --version) VERSION=$2; shift;; --wheel) WHEEL=$2; shift;;
    --budget-mib) BUDGET=$2; shift;; --card-mib) CARD=$2; shift;; --geforce) GEFORCE=1;;
    --card) case $2 in 4080s) BUDGET=14828 CARD=15942 GEFORCE=1;; 8gb) BUDGET=7500 EXPECT=refusal;; *) echo "unknown card $2 (4080s, 8gb)"; exit 2;; esac; shift;;
    --model) MODEL=$2 MODEL_SET=1; shift;; --compiler) COMPILER=$2; shift;;
    --command) COMMAND=$2; shift;; --webui) WEBUI=$2; shift;; --hold) HOLD=$2; shift;; --image) IMAGE=$2; shift;; --work) WORK=$2; shift;; --host) HOST=1;;
    --pip-refusal) PIPREF=1;;
    --rust-glyd) RUSTGLYD=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift;; --reuse) REUSE=1;;
    --up-wait) UPWAIT=$2; shift;;
    --install-url) INSTALL_URL=$2; shift;; --served) SERVED=1;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d;s/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option $1 (--help)"; exit 2;;
  esac
  shift
done
case $FLOW in glyd|vllm|cli) ;; *) echo "unknown flow $FLOW (glyd, vllm, cli)"; exit 2;; esac
[ -z "$RUSTGLYD" ] || [ -x "$RUSTGLYD" ] || { echo "--rust-glyd: $RUSTGLYD is not an executable"; exit 2; }
[ -z "$SERVED" ] || [ -z "$VERSION$WHEEL" ] || { echo "--served runs the README's install line as it is: no --version or --wheel"; exit 2; }
[ $EXPECT = fit ] || [ -n "$MODEL_SET" ] || MODEL=Qwen/Qwen3-4B  # (the 8 GB card: the model that does not quite fit)
case $COMPILER in none|gcc) ;; *) echo "unknown compiler $COMPILER (none, gcc)"; exit 2;; esac
[ -z "$HOST" ] || [ "$FLOW" = vllm ] || { echo "--host is for the vllm flow: the glyd and cli flows install into a home of their own, in a container"; exit 2; }
[ "$WEBUI" != all ] || WEBUI=uvx,docker,bridge
[ "$WEBUI" != both ] || WEBUI=uvx,docker
mkdir -p "$WORK" && WORK=$(cd "$WORK" && pwd)
HF=${HF_HOME:-$HOME/.cache/huggingface}
rm -rf "$WORK/logs"  # (the logs of this run only: a file of an earlier run in them reads as this run's)
mkdir -p "$WORK/home" "$WORK/logs" "$WORK/bin" "$WORK/uv-cache" "$WORK/wheel" "$HF"
LOGS=$WORK/logs
[ -z "$RUSTGLYD" ] || { mkdir -p "$WORK/rust" && cp "$RUSTGLYD" "$WORK/rust/glyd" && chmod +x "$WORK/rust/glyd"; }
FAILS=()
: > "$WORK/summary.txt"
say() { printf '%s\n' "$*" | tee -a "$WORK/summary.txt"; }
fail() { FAILS+=("$*"); say "FAIL $*"; }
ok() { say "ok   $*"; }
count() { grep -c "$1" "$2" 2> /dev/null || true; }
sha256() { { sha256sum "$1" 2> /dev/null || shasum -a 256 "$1"; } | cut -d' ' -f1; }
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
import json, os, re, sys, time, urllib.request


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


def webui_cors(url):
    """The README's CORS_ALLOW_ORIGIN: a script of another site gets no grant and its preflight is refused; the page's own origin is granted."""
    def req(origin, method="POST", pre=False):
        h = {"Origin": origin, "Content-Type": "application/json"}
        if pre:
            h.update({"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,authorization"})
        r = urllib.request.Request(url + "/api/v1/auths/signin", None if method == "OPTIONS" else json.dumps({"email": "", "password": ""}).encode(), h, method=method)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, resp.headers.get("access-control-allow-origin")
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("access-control-allow-origin")
    evil, pre, own = req("http://evil.example"), req("http://evil.example", "OPTIONS", True), req("http://localhost:3000")
    return check("Open WebUI's CORS is limited to its own addresses", evil[1] is None and pre[0] == 400 and pre[1] is None and own[1] == "http://localhost:3000",
                 f"another site's request: {evil}, its preflight: {pre}, the page's own origin: {own}")


def code_of(url, body=None, token=None):
    """The HTTP status of a request, whether the server answered it or refused it."""
    try:
        with call(url, body, token, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def webui(url, model):
    for _ in range(150):
        try:
            if json.load(call(url + "/health", timeout=5)).get("status"):
                break
        except Exception:
            time.sleep(2)
    # the login is on (the README's commands do not set WEBUI_AUTH=False): nothing is answered without one, and the sign-in that no-login mode took is refused
    anon, empty = code_of(url + "/api/models"), code_of(url + "/api/v1/auths/signin", {"email": "", "password": ""})
    ok = check("Open WebUI asks for a login", anon == 401 and empty == 400, f"/api/models with no token: {anon}; a sign-in with empty credentials: {empty}")
    # the first visit: the first account made is the administrator
    account = json.load(call(url + "/api/v1/auths/signup", {"name": "Acceptance", "email": "acceptance@example.com", "password": "Pw-" + os.urandom(6).hex(), "profile_image_url": "/user.png"}))
    token = account["token"]
    ok &= check("the first account made is the administrator", account.get("role") == "admin", f"role {account.get('role')!r}")
    ids = [m["id"] for m in json.load(call(url + "/api/models", token=token))["data"]]
    ok &= check("Open WebUI lists the model", model in ids, f"/api/models: {ids}")
    ok &= webui_cors(url)

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


CONTROL = r"[\x00-\x08\x0b-\x1f\x7f]"


def raw(base, prompt_file):
    """The control characters in the model's own answer to a prompt that carries escape sequences: what glyd run's output is compared with."""
    model = json.load(call(base + "/models"))["data"][0]["id"]
    r = json.load(call(base + "/chat/completions", {"model": model, "max_tokens": 100, "temperature": 0, "messages": [{"role": "user", "content": open(prompt_file).read()}]}))
    text = r["choices"][0]["message"].get("content") or ""
    print(f"the model's own answer has {len(re.findall(CONTROL, text))} control characters ({text.strip()[:30]!r})", flush=True)
    return True


what = sys.argv[1]
if what == "api":
    good = api(sys.argv[2]) & thinking(sys.argv[2]) & too_long(sys.argv[2])
elif what == "raw":
    good = raw(sys.argv[2], sys.argv[3])
else:
    good = webui(sys.argv[2], sys.argv[3])
sys.exit(0 if good else 1)
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
  E=(-e HOME=$IN/home -e UV_CACHE_DIR=$IN/uv-cache -e HF_HOME=$INHF)
  MOUNTS=()
  # (the glyd flow keeps uv's own default, hardlinks from its cache, as on a user's disk: the cache and the home are one mount here; the vllm flow copies)
  if [ "$FLOW" = vllm ]; then MOUNTS=(-v "$UV:/usr/local/bin/uv:ro" -v "$UVX:/usr/local/bin/uvx:ro"); E+=(-e UV_LINK_MODE=copy); else E+=(-e PATH=$IN/home/.local/bin:/usr/local/bin:/usr/bin:/bin); fi
  GPUS=(--gpus all); [ "$FLOW" != cli ] || GPUS=()  # (the cli flow: a machine with no NVIDIA GPU)
  docker run -d --rm --name $NAME ${GPUS[@]+"${GPUS[@]}"} --network host --ipc host --user "$(id -u):$(id -g)" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility "${E[@]}" \
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

PKG="glyd[vllm] as install.sh pins it, from PyPI"; [ -z "$VERSION" ] || PKG="glyd[vllm]==$VERSION from PyPI"; [ -z "$WHEEL" ] || PKG="glyd[vllm] wheel $WHEEL"
say "== $(date -u +%FT%TZ): $FLOW flow, $PKG, ${IMAGE:-the host}, GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2> /dev/null | head -1), budget ${BUDGET:-all} MiB free, card ${CARD:-as is} MiB"

# --- 1. no nvcc
if run 'command -v nvcc > /dev/null || test -e /usr/local/cuda || test -n "${CUDA_HOME:-}${CUDA_PATH:-}"'; then
  fail "nvcc is reachable here (on PATH, in /usr/local/cuda or CUDA_HOME): not a machine without a CUDA toolkit"; exit 1
fi
ok "no nvcc: not on PATH, no /usr/local/cuda, no CUDA_HOME"

# --- Open WebUI, one route at a time against the server on port 8000 (the README's block, a volume of its own where Docker keeps one)
webui_route() {  # webui_route ROUTE PYTHON MODEL [KEY]
  local route=$1 py=$2 model=$3 key=${4:-none}
  say "-- Open WebUI, $route: $(block webui-$route | tr '\n' ' ' | tr -s ' ' | cut -c1-200)"
  local before; before=$(du -sm "$WORK/uv-cache" 2> /dev/null | cut -f1)
  block webui-$route | sed "s|-v open-webui:|-v $VOLUME:|; s|YOUR_KEY|$key|g" > "$WORK/webui-$route.sh"
  [ $route != uvx ] || rm -rf "$WORK/home/.open-webui"  # (the first account is made in every route: an earlier run's data folder would hold it already)
  case $route in
    uvx) bg "setsid bash -c 'echo \$\$ > $IN/logs/webui.pid; exec bash $IN/webui-uvx.sh' > $IN/logs/webui-uvx.log 2>&1";;
    *) OWUI=1; bash "$WORK/webui-$route.sh" > "$LOGS/webui-$route.log" 2>&1 || { fail "the README's $route command failed (logs/webui-$route.log)"; return; };;
  esac
  run "$py $IN/check.py webui http://127.0.0.1:3000 '$model'" > "$LOGS/webui-$route.check.log" 2>&1 || fail "Open WebUI, $route (logs/webui-$route.check.log)"
  if [ $route = uvx ]; then say "     (the uvx route brought $(( ($(du -sm "$WORK/uv-cache" 2> /dev/null | cut -f1) - ${before:-0}) / 1024 )) GB into uv's cache: Open WebUI and its own PyTorch)"
  else say "     (the image: $(docker image inspect --format '{{.Size}}' ghcr.io/open-webui/open-webui:v0.11.4 2> /dev/null | awk '{printf "%.1f GB", $1 / 1e9}'))"; fi
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

# --- helper scripts that run inside the container (written here: one place for the quoting)
write_guard() {
  cat > "$WORK/guard.sh" <<'EOF'
#!/bin/bash
# What a server started by glyd on 127.0.0.1 answers: its own page and this computer's programs, and not a page of another site
# (CORS lets a script from any origin reach 127.0.0.1) nor a name that DNS points here (DNS rebinding sends no Origin at all).
b=http://127.0.0.1:8000
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
bad=0
chk() { if [ "$2" = "$3" ]; then echo "ok   $1: $3"; else echo "FAIL $1: got $3, wanted $2"; bad=1; fi; }
chk "a program, no Origin (curl)" 200 "$(code $b/v1/models)"
chk "the page's own origin, localhost" 200 "$(code -H 'Origin: http://localhost:8000' -H 'Host: localhost:8000' $b/v1/models)"
chk "the page's own origin, 127.0.0.1" 200 "$(code -H 'Origin: http://127.0.0.1:8000' $b/v1/models)"
chk "a script of another site (Origin)" 403 "$(code -H 'Origin: http://evil.example' $b/v1/models)"
chk "its preflight" 403 "$(code -X OPTIONS -H 'Origin: http://evil.example' -H 'Access-Control-Request-Method: POST' -H 'Access-Control-Request-Headers: content-type' $b/v1/chat/completions)"
chk "its POST" 403 "$(code -X POST -H 'Origin: http://evil.example' -H 'Content-Type: application/json' -d '{}' $b/v1/chat/completions)"
chk "a rebinding name (Host)" 421 "$(code -H 'Host: evil.example:8000' $b/v1/models)"
chk "the page under a rebinding name" 421 "$(code -H 'Host: evil.example:8000' $b/)"
chk "the page itself" 200 "$(code $b/)"
chk "no CORS grant to another site" 0 "$(curl -s -D - -o /dev/null -H 'Origin: http://evil.example' $b/v1/models | tr -d '\r' | grep -ic '^access-control-allow-origin')"
chk "CORS names the page's own origin" "http://localhost:8000" "$(curl -s -D - -o /dev/null -H 'Origin: http://localhost:8000' -H 'Host: localhost:8000' $b/v1/models | tr -d '\r' | sed -n 's/^[Aa]ccess-[Cc]ontrol-[Aa]llow-[Oo]rigin: //p')"
exit $bad
EOF
}

# --- the glyd flow's parts
glyd_foreign() {  # a program of the user's own where uv puts its entry point: install.sh stops, and leaves it as it was
  local env=$1 rc h=$IN/home-foreign
  rm -rf "$WORK/home-foreign"; mkdir -p "$WORK/home-foreign/.local/bin"
  printf '#!/bin/sh\necho mine\n' > "$WORK/home-foreign/.local/bin/glyd"; chmod +x "$WORK/home-foreign/.local/bin/glyd"
  run "HOME=$h ${env}sh $IN/install.sh" > "$LOGS/install-foreign.log" 2>&1; rc=$?
  if [ $rc = 1 ] && grep -q "is not Glyd's Python tool" "$LOGS/install-foreign.log" && grep -q 'UV_TOOL_BIN_DIR' "$LOGS/install-foreign.log" \
     && [ "$(cat "$WORK/home-foreign/.local/bin/glyd")" = "$(printf '#!/bin/sh\necho mine')" ] && ! grep -q 'and Python 3.12' "$LOGS/install-foreign.log"; then
    ok "a glyd of the user's own in ~/.local/bin: install.sh stopped before it installed Glyd and left that program as it was ($(grep -m1 "is not Glyd's" "$LOGS/install-foreign.log" | cut -c1-110)...)"
  else
    fail "install.sh with a glyd of the user's own in ~/.local/bin (exit $rc, logs/install-foreign.log): $(tail -3 "$LOGS/install-foreign.log" | tr '\n' '|' | cut -c1-250)"
  fi
  rm -rf "$WORK/home-foreign"
}

glyd_install() {  # the README's install line with this checkout's script, and what it left
  local line=$1 env=$2 rc t0 py pre touched f
  if [ -n "$SERVED" ]; then say "-- install: $line   (as the README gives it: the script getglyd.com serves, from the network)"
  else say "-- install: $line   ($INSTALL_NOTE instead of getglyd.com's${env:+; $env})"; fi
  rm -rf "$WORK/home/.local" "$WORK/home/.cache" "$WORK/home/.config" "$WORK"/home/.bashrc "$WORK"/home/.bash_profile "$WORK"/home/.profile "$WORK"/home/.zshenv "$WORK"/home/.zshrc  # (a clean home: an earlier run's edits of a shell startup file are not this run's)
  t0=$(date +%s)
  if [ -n "$SERVED" ]; then run "$line" > "$LOGS/install.log" 2>&1; rc=$?; else run "${env}sh $IN/install.sh" > "$LOGS/install.log" 2>&1; rc=$?; fi
  [ $rc = 0 ] || { fail "install.sh exited $rc (logs/install.log)"; tail -8 "$LOGS/install.log"; return 1; }
  ok "install.sh took $(( $(date +%s) - t0 )) s and exited 0; its last lines: $(tail -n 3 "$LOGS/install.log" | tr '\n' '|' | cut -c1-200)"
  grep -q 'glyd doctor' "$LOGS/install.log" && grep -q 'Ready: glyd run' "$LOGS/install.log" && grep -q 'Next: glyd run' "$LOGS/install.log" || fail "install.sh did not end with glyd doctor saying Ready: glyd run ... and the next command (logs/install.log)"
  grep -q "edits your shell's startup file" "$LOGS/install.log" && fail "install.sh announced a shell-profile edit where ~/.local/bin is on PATH already"
  grep -q 'Installing uv' "$LOGS/install.log" && ok "uv was installed by its own installer, at a version: $(grep -m1 'Installing uv' "$LOGS/install.log" | cut -c1-70); $(grep -m1 'downloading uv' "$LOGS/install.log" | cut -c1-60)" || fail "install.sh did not install uv (the clean machine has none): logs/install.log"
  run 'command -v glyd' > /dev/null 2>&1 || fail "glyd is not on the PATH after the install"
  touched=; for f in .bashrc .profile .zshenv .zshrc .bash_profile; do [ -e "$WORK/home/$f" ] && touched="$touched $f"; done
  if [ -n "$touched" ]; then
    fail "the install edited a shell startup file where ~/.local/bin was on PATH already ($touched)"
  else
    ok "no shell startup file touched (~/.local/bin was on PATH; uv's installer was told to edit none)"
  fi
  py=$(run 'echo $(uv tool dir)/glyd/bin/python' | tail -1)
  run "uv tool list --show-with; uv pip freeze --python $py" > "$LOGS/freeze.txt" 2>&1
  say "-- resolved: $(grep -iE '^(glyd|vllm|torch|flashinfer-python|transformers|safetensors|tokenizers|pydantic|triton|ziglang)[ =@]' "$LOGS/freeze.txt" | sed -E 's/ \(.*//; s/ @ .*//' | tr '\n' ' ')"
  # (opentelemetry's instrumentation packages publish only betas, 0.66b0: no stable release to take instead; pydantic, safetensors and tokenizers have both)
  pre=$(grep -E '^[A-Za-z0-9_.-]+==[0-9][0-9.]*(a|b|rc|dev)[0-9]+' "$LOGS/freeze.txt" | grep -v '^glyd==' | grep -vE '^opentelemetry-[a-z-]+==[0-9.]+b[0-9]+$' | tr '\n' ' ')
  [ -z "$pre" ] && ok "no pre-release among the dependencies (install.sh names no --prerelease; opentelemetry's betas are all there is of those)" || fail "pre-releases among the dependencies: $pre"
  # the packages are the versions install.sh lists (the last acceptance run's): none other, and none of another version
  INSTALL_SH="$WORK/install.sh" python3 "$HERE/../../scripts/install_constraints.py" --list | sort > "$LOGS/constraints.list"  # (the list in the script that was run)
  grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' "$LOGS/freeze.txt" | grep -v '^glyd==' | sort > "$LOGS/freeze.pins"
  if [ -z "$(comm -23 "$LOGS/freeze.pins" "$LOGS/constraints.list")" ]; then
    ok "every one of the $(wc -l < "$LOGS/freeze.pins" | tr -d ' ') packages installed is at the version install.sh's list gives ($(wc -l < "$LOGS/constraints.list" | tr -d ' ') listed; ziglang is only there where there is no compiler)"
  else
    fail "packages installed at versions that install.sh's list does not give: $(comm -23 "$LOGS/freeze.pins" "$LOGS/constraints.list" | head -8 | tr '\n' ' ') (python3 scripts/install_constraints.py logs/freeze.txt)"
  fi
  if [ $COMPILER = none ]; then
    grep -q '^ziglang==' "$LOGS/freeze.txt" && ok "no gcc: install.sh added $(grep -o '^ziglang==[0-9.]*' "$LOGS/freeze.txt")" || fail "no gcc here, and install.sh added no ziglang"
  else
    grep -q '^ziglang==' "$LOGS/freeze.txt" && fail "ziglang was installed on a machine with gcc"
  fi
}

glyd_again() {  # install.sh again, which is the README's way to update: nothing to do, the PATH edit said, a glyd that comes first said; then glyd doctor's own row for it
  local env=$1 rc out
  run "SHELL=/bin/bash PATH=/usr/local/bin:/usr/bin:/bin ${env}sh $IN/install.sh" > "$LOGS/install-again.log" 2>&1; rc=$?   # (uv's folder is not on this PATH: the edit is said, then made)
  if [ $rc = 0 ] && grep -q "is not on your PATH: adding it, which edits your shell's startup file" "$LOGS/install-again.log" && grep -q 'Open a new terminal' "$LOGS/install-again.log" \
     && grep -qE 'Created configuration file|Updated configuration file|already up-to-date' "$LOGS/install-again.log"; then
    ok "install.sh run again (an update; ~/.local/bin not on PATH): exit 0, the shell-profile edit said first, uv's own line naming the file: $(grep -E 'configuration file|up-to-date' "$LOGS/install-again.log" | head -1 | cut -c1-120)"
  else
    fail "install.sh run again with ~/.local/bin off PATH (exit $rc, logs/install-again.log): $(tail -4 "$LOGS/install-again.log" | tr '\n' '|' | cut -c1-300)"
  fi
  grep -qs 'local/bin' "$WORK/home/.bashrc" "$WORK/home/.profile" "$WORK/home/.zshenv" && ok "the PATH line is in the shell's startup file ($(grep -ls 'local/bin' "$WORK/home/.bashrc" "$WORK/home/.profile" "$WORK/home/.zshenv" | xargs -n1 basename | tr '\n' ' '))" || fail "the PATH edit install.sh announced is in none of .bashrc, .profile, .zshenv"
  say "     (uv's answer to the update: $(grep -m1 -E 'already installed|Installed|Uninstalled|Audited' "$LOGS/install-again.log" | cut -c1-100))"
  mkdir -p "$WORK/shadow"; printf '#!/bin/sh\necho "the compression program"\n' > "$WORK/shadow/glyd"; chmod +x "$WORK/shadow/glyd"
  run "PATH=$IN/shadow:$IN/home/.local/bin:/usr/local/bin:/usr/bin:/bin ${env}sh $IN/install.sh" > "$LOGS/install-shadow.log" 2>&1; rc=$?
  if [ $rc = 0 ] && grep -q "another glyd comes first on your PATH: $IN/shadow/glyd" "$LOGS/install-shadow.log" && grep -qF "export PATH=\"$IN/home/.local/bin:\$PATH\"" "$LOGS/install-shadow.log" && grep -qF "$IN/home/.local/bin/glyd run MODEL" "$LOGS/install-shadow.log"; then
    ok "a glyd ahead of it on PATH: install.sh says which, and the line to add and the path to run it by"
  else
    fail "install.sh with another glyd first on PATH did not say so with the fix (exit $rc, logs/install-shadow.log): $(tail -5 "$LOGS/install-shadow.log" | tr '\n' '|' | cut -c1-300)"
  fi
  run "PATH=$IN/shadow:\$PATH $IN/home/.local/bin/glyd doctor" > "$LOGS/doctor-shadow.txt" 2>&1
  grep -q 'another program named glyd comes first on your PATH' "$LOGS/doctor-shadow.txt" && grep -qF "export PATH=\"$IN/home/.local/bin:\$PATH\"" "$LOGS/doctor-shadow.txt" && ok "glyd doctor reports it too, with the same line" || fail "glyd doctor did not report the glyd ahead of it (logs/doctor-shadow.txt)"
  if [ -n "$RUSTGLYD" ]; then  # the Rust glyd built from this tree, first on PATH: it passes run, serve, doctor and login to the Python tool
    run "PATH=$IN/rust:\$PATH glyd --version | head -1; PATH=$IN/rust:\$PATH glyd doctor" > "$LOGS/doctor-rust.txt" 2>&1; rc=$?
    if [ $rc = 0 ] && grep -q 'Ready: glyd run' "$LOGS/doctor-rust.txt" && ! grep -q 'Unknown option' "$LOGS/doctor-rust.txt"; then
      ok "the Rust glyd first on PATH hands glyd doctor to the Python tool ($(head -1 "$LOGS/doctor-rust.txt"))"
    else fail "the Rust glyd first on PATH did not run glyd doctor through to the Python tool (logs/doctor-rust.txt): $(tail -3 "$LOGS/doctor-rust.txt" | tr '\n' '|' | cut -c1-200)"; fi
  fi
}

glyd_key() {  # glyd serve with an API key: ready (the probe needs no key), the key on no command line or log, the note says what stays open  [glyd_key OUTFILE KEY FORM]
  local out=$1 key=$2 form=$3 n
  grep -q '^Ready in' "$out" && grep -q '^Serving' "$out" && grep -q 'asks for the API key' "$out" && ok "glyd serve with the key ($form): ready, serving, and its banner says it asks for the key" || fail "glyd serve with the key ($form) did not get ready and say so (logs/$(basename "$out")): $(tail -3 "$out" | tr '\n' '|' | cut -c1-200)"
  grep -q 'plain HTTP' "$out" && grep -q '/health' "$out" && grep -q 'guards /v1 only' "$out" && ok "its note on an open address names plain HTTP and the endpoints a key does not guard" || fail "glyd serve's note on a server open to the network is missing what stays open (logs/$(basename "$out"))"
  n=$(run "ps -eo args | grep -v 'glyd serve' | grep -v grep | grep -c -- '$key' || true" | tail -1)
  [ "${n:-1}" = 0 ] && ok "the key is on no other command line (ps)" || fail "the API key is on $n command lines (ps), besides glyd's own"
  n=$(cat "$WORK"/home/.local/state/glyd/logs/*.log "$out" 2> /dev/null | grep -c -- "$key" || true)
  [ "${n:-1}" = 0 ] && ok "the key is in no log and not in glyd's output" || fail "the API key is in $n lines of the logs or glyd's output"
  [ "$form" = flag ] && { grep -q 'taken off the server' "$out" && ok "given as -- --api-key, glyd says it was taken off the server's command line" || fail "no note that the key given as --api-key was moved to the environment"; }
  [ "$(run "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models")" = 401 ] && ok "/v1 without the key is 401" || fail "the server took a request to /v1 without its key"
  [ "$(run "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health")" = 200 ] && ok "/health stays open (the note says so)" || fail "/health is not open"
  [ "$(run "curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer $key' http://127.0.0.1:8000/v1/models")" = 200 ] && ok "/v1 with the key is 200" || fail "the key is not accepted"
}

glyd_hup() {  # nohup ignores SIGHUP, and glyd keeps it ignored: the server under it lives through the terminal closing
  local pid
  pid=$(run 'pgrep -f "[/]bin/glyd serve" | head -1' | tail -1)  # (the tool's own process: its command line has the script's path, the wrapper's has not)
  [ -n "$pid" ] || { fail "no glyd serve process to send SIGHUP to"; return; }
  run "kill -HUP $pid" > /dev/null 2>&1; sleep 8
  if run "kill -0 $pid" > /dev/null 2>&1 && [ "$(run "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health")" = 200 ] && ! grep -q 'Stopping the server' "$LOGS/serve.out"; then
    ok "glyd serve under nohup kept running when it was sent SIGHUP (nohup's ignore is kept)"
  else
    fail "glyd serve under nohup did not survive SIGHUP (logs/serve.out): $(tail -3 "$LOGS/serve.out" | tr '\n' '|' | cut -c1-200)"
  fi
}

glyd_ctrlc() {  # Ctrl-C while the model loads, once and twice: glyd exits promptly, and nothing it started is left running or holding the GPU
  local n pid t gone left rc
  for n in 1 2; do
    : > "$LOGS/ctrlc$n.out"
    bg "setsid bash -c 'echo \$\$ > $IN/logs/ctrlc.pid; ${gf}glyd run $MODEL --prompt hi; echo rc=\$?' > $IN/logs/ctrlc$n.out 2>&1"
    for _ in $(seq 1 60); do grep -q '^Settings:' "$LOGS/ctrlc$n.out" 2> /dev/null && break; sleep 1; done
    sleep 12   # (the weights are loading)
    pid=$(cat "$LOGS/ctrlc.pid" 2> /dev/null)
    run "kill -INT -- -$pid" > /dev/null 2>&1
    [ $n = 2 ] && { sleep 1.5; run "kill -INT -- -$pid" > /dev/null 2>&1; }
    t=$(date +%s); gone=
    for _ in $(seq 1 90); do grep -q '^rc=' "$LOGS/ctrlc$n.out" && { gone=1; break; }; sleep 1; done
    t=$(( $(date +%s) - t ))
    rc=$(sed -n 's/^rc=//p' "$LOGS/ctrlc$n.out" | head -1)
    for _ in $(seq 1 20); do left=$(run 'pgrep -fa "EngineCore|vllm" | grep -v pgrep' | tail -3 | tr '\n' ' '); [ -z "$left" ] && break; sleep 1; done
    if [ -n "$gone" ] && [ "$rc" = 130 ] && [ -z "$left" ] && grep -q 'Stopping the server' "$LOGS/ctrlc$n.out" && ! grep -qE 'Traceback|unexpected error' "$LOGS/ctrlc$n.out"; then
      ok "Ctrl-C ${n}x while the model loads: glyd exited 130 in $t s, no vLLM process left ($(grep -E 'Stopping|stopp' "$LOGS/ctrlc$n.out" | head -1 | cut -c1-60))"
    else
      fail "Ctrl-C ${n}x while the model loads: exit ${rc:-none} after $t s, left running: ${left:-nothing}, said it was stopping: $(grep -c 'Stopping the server' "$LOGS/ctrlc$n.out"); $(tail -3 "$LOGS/ctrlc$n.out" | tr '\n' '|' | cut -c1-200) (logs/ctrlc$n.out)"
      run 'pkill -KILL -f "EngineCore|vllm|glyd run"' > /dev/null 2>&1; sleep 3
    fi
    ss -ltn 2> /dev/null | grep -q ':8000 ' && { fail "port 8000 is in use after the Ctrl-C"; return; }
  done
}

glyd_chat_dies() {  # a terminal chat with an answer under way, and the engine killed under it: a plain message, no hang, no traceback
  local up out
  : > "$LOGS/serve-die.out"
  bg "${gf}glyd serve $MODEL --port 8000 > $IN/logs/serve-die.out 2>&1"
  up=; for _ in $(seq 1 $((UPWAIT / 3))); do run 'curl -sf http://127.0.0.1:8000/v1/models' > /dev/null 2>&1 && { up=1; break; }; run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break; sleep 3; done
  [ -n "$up" ] || { fail "glyd serve did not come up for the engine-killed check (logs/serve-die.out)"; return; }
  cat > "$WORK/chatdies.sh" <<EOF
#!/bin/bash
# the question is typed 8 s in; the engine is killed 3 s after the answer has started to come (not before: a short answer is over by then)
out=$IN/logs/chatdies.out
( sleep 8; echo 'Write a long story, at least 2000 words, about a lighthouse keeper, and do not stop before the end. /no_think'; sleep 60; echo /bye ) | ${gf}timeout 300 script -qec 'glyd run $MODEL' /dev/null > \$out 2>&1 &
chat=\$!
sleep 9
base=\$(stat -c %s \$out)
for i in \$(seq 1 40); do [ \$(( \$(stat -c %s \$out) - base )) -gt 600 ] && break; sleep 1; done
sleep 3
pkill -KILL -f 'EngineCore'
wait \$chat
EOF
  run "bash $IN/chatdies.sh" > /dev/null 2>&1
  out=$(tr -d '\r' < "$LOGS/chatdies.out" | sed 's/\x1b\[[0-9;?]*[A-Za-z]//g')
  if printf '%s' "$out" | grep -qE 'The server (stopped answering|could not finish the answer)' && ! printf '%s' "$out" | grep -qE 'Traceback|unexpected error'; then
    ok "the engine killed under a terminal chat's answer: $(printf '%s' "$out" | grep -E 'The server (stopped answering|could not finish)' | head -1 | cut -c1-140)"
  else
    fail "the engine killed under a terminal chat's answer: no plain message, or a traceback (logs/chatdies.out): $(printf '%s' "$out" | tail -4 | tr '\n' '|' | cut -c1-250)"
  fi
  for _ in $(seq 1 60); do run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break; sleep 1; done
  run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 && { fail "glyd serve did not exit after its engine was killed (logs/serve-die.out)"; run 'pkill -TERM -f "[g]lyd serve"' > /dev/null 2>&1; sleep 5; } || ok "glyd serve exited when its engine died: $(grep -iE 'stopped|exited|died' "$LOGS/serve-die.out" | head -1 | cut -c1-120)"
  run 'pkill -KILL -f "EngineCore|vllm"' > /dev/null 2>&1; sleep 3
}

glyd_remove() {  # the README's removal: uv tool uninstall glyd takes the tool and its link, and leaves the logs, the models and uv
  local logs=$WORK/home/.local/state/glyd
  run 'uv tool uninstall glyd' > "$LOGS/uninstall.log" 2>&1
  if ! run 'command -v glyd' > /dev/null 2>&1 && [ ! -e "$WORK/home/.local/bin/glyd" ] && [ -d "$logs/logs" ] && [ -x "$WORK/home/.local/bin/uv" ] && ! run 'uv tool list' 2>&1 | grep -q '^glyd '; then
    ok "uv tool uninstall glyd: the tool and its link are gone; ~/.local/state/glyd ($(ls "$logs" | tr '\n' ' ')), the Hugging Face cache and uv stay (as the README says)"
  else
    fail "uv tool uninstall glyd did not leave what the README says (logs/uninstall.log): $(cat "$LOGS/uninstall.log" | tail -3 | tr '\n' '|')"
  fi
}

glyd_flow() {
  local line env rc t0 py gf ans new up key rp big win chatout pid
  # 2. the README's install line, with this checkout's install.sh where the line fetches getglyd.com's
  line=$(block install)
  [ "$line" = 'curl -LsSf https://getglyd.com/install.sh | sh' ] || { fail "the README's <!-- acceptance: install --> block is not the one line this script replaces: $line"; return 1; }
  if [ -n "$SERVED" ]; then
    curl -fsSL https://getglyd.com/install.sh -o "$WORK/install.sh" || { fail "https://getglyd.com/install.sh is not served"; return 1; }
    INSTALL_NOTE="the script getglyd.com serves"
  elif [ -n "$INSTALL_URL" ]; then
    curl -fsSL "$INSTALL_URL" -o "$WORK/install.sh" || { fail "could not fetch $INSTALL_URL"; return 1; }
    INSTALL_NOTE="$INSTALL_URL"
  else
    cp "$HERE/../../scripts/install.sh" "$WORK/install.sh" || { fail "no scripts/install.sh beside gpu/vllm"; return 1; }
    INSTALL_NOTE="this checkout's scripts/install.sh"
  fi
  say "-- install.sh: $INSTALL_NOTE, sha256 $(sha256 "$WORK/install.sh")"
  if [ -n "$SERVED" ] && [ -n "$INSTALL_URL" ]; then  # (the site serves the release's own file)
    [ "$(curl -fsSL "$INSTALL_URL" | sha256 /dev/stdin)" = "$(sha256 "$WORK/install.sh")" ] && ok "the script getglyd.com serves is $INSTALL_URL, byte for byte" || fail "the script getglyd.com serves is not $INSTALL_URL"
  fi
  env=
  [ -n "$VERSION" ] && env="GLYD_VERSION=$VERSION "
  [ -n "$WHEEL" ] && { cp "$WHEEL" "$WORK/wheel/" && env="GLYD_SPEC='$IN/wheel/$(basename "$WHEEL")[vllm]' "; }
  rp=; [ -z "$RUSTGLYD" ] || rp="PATH=$IN/rust:\$PATH "
  write_guard
  if [ -n "$REUSE" ]; then
    say "-- --reuse: the install in $WORK/home as the last run left it (not run again, and not checked again)"
  else
    glyd_foreign "$env"
    glyd_install "$line" "$env" || return 1
    glyd_again "$env"
  fi
  py=$(run 'echo $(uv tool dir)/glyd/bin/python' | tail -1)
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
  [ $EXPECT = refusal ] && { glyd_remove; glyd_pip; return; }
  # 5. glyd serve (under nohup, and through the Rust glyd where there is one), then the page, the API, the longer-than-the-window conversation, Open WebUI
  : > "$LOGS/serve.out"
  bg "${gf}${rp}nohup glyd serve $MODEL --port 8000 > $IN/logs/serve.out 2>&1"
  up=; t0=$(date +%s)
  for _ in $(seq 1 600); do
    run 'curl -sf http://127.0.0.1:8000/v1/models' > /dev/null 2>&1 && { up=1; break; }
    run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break
    sleep 3
  done
  [ -n "$up" ] || { fail "glyd serve did not come up (logs/serve.out)"; tail -5 "$LOGS/serve.out"; glyd_logs; return 1; }
  ok "glyd serve up after $(( $(date +%s) - t0 )) s${RUSTGLYD:+ (started through the Rust glyd first on PATH)}: $(grep -E '^(Settings|Ready in)' "$LOGS/serve.out" | cut -c1-150 | tr '\n' '|')"
  run "curl -s -D $IN/logs/page.head -o $IN/logs/page.html http://127.0.0.1:8000/" > /dev/null 2>&1
  head -1 "$LOGS/page.head" | grep -q ' 200' && grep -qi '^content-type: text/html' "$LOGS/page.head" && grep -q '<title>Glyd' "$LOGS/page.html" && ok "GET / is the chat page ($(wc -c < "$LOGS/page.html" | tr -d ' ') bytes, $(grep -io '^content-security-policy: [^;]*' "$LOGS/page.head" | tr -d '\r'))" || fail "GET / is not the chat page (logs/page.head)"
  grep -qE "(src|href)=[\"']https?:|url\(https?:|@import" "$LOGS/page.html" && fail "the chat page names an external resource"
  [ "$(run 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/nope')" = 404 ] && ok "other paths stay vLLM's (/nope is 404; the API answers below)" || fail "/nope is not 404"
  run "bash $IN/guard.sh" > "$LOGS/guard.log" 2>&1 && ok "the server on 127.0.0.1 answers its own page and local programs, and refuses another site's script and a rebinding name ($(grep -c '^ok' "$LOGS/guard.log") checks, logs/guard.log)" || { fail "the server on 127.0.0.1 let in a page of another origin or a rebinding Host (logs/guard.log)"; grep '^FAIL' "$LOGS/guard.log"; }
  glyd_hup
  run "$py $IN/check.py api http://127.0.0.1:8000/v1" > "$LOGS/chat.log" 2>&1 || fail "the OpenAI API checks (logs/chat.log)"
  tee -a "$WORK/summary.txt" < "$LOGS/chat.log"
  # (a model that repeats escape sequences does not move the user's terminal: glyd run attaches to this server)
  printf 'Repeat this text exactly, character for character, and say nothing else: \033[31mRED\033[0m \033]0;pwned\007 end /no_think' > "$WORK/esc-prompt.txt"
  run "${gf}glyd run $MODEL --prompt \"\$(cat $IN/esc-prompt.txt)\"" > "$LOGS/escapes.out" 2> "$LOGS/escapes.err"; rc=$?
  run "$py $IN/check.py raw http://127.0.0.1:8000/v1 $IN/esc-prompt.txt" > "$LOGS/escapes.raw" 2>&1
  if LC_ALL=C grep -q $'[\x01-\x08\x0b-\x1f\x7f]' "$LOGS/escapes.out" "$LOGS/escapes.err"; then
    fail "glyd run's output carries control characters (logs/escapes.out, escapes.err): $(cat "$LOGS/escapes.out" "$LOGS/escapes.err" | LC_ALL=C tr -d '[:print:]\n\t' | od -c | head -2 | tr '\n' ' ' | cut -c1-100)"
  elif [ $rc != 0 ]; then
    fail "glyd run with a prompt full of escape sequences exited $rc (logs/escapes.err): $(tail -2 "$LOGS/escapes.err" | tr '\n' '|' | cut -c1-200)"
  else
    ok "glyd run's answer to a prompt full of escape sequences has no control character in it ($(wc -c < "$LOGS/escapes.out" | tr -d ' ') bytes: $(tr -d '\n' < "$LOGS/escapes.out" | cut -c1-50)); $(cat "$LOGS/escapes.raw" | cut -c1-120)"
  fi
  big=$(run "yes word | head -n 60000 | tr '\n' ' ' | ${gf}glyd run $MODEL 2>&1 >/dev/null; echo rc=\$?" | tail -3 | tr '\n' ' ')
  case $big in *"This prompt is longer than the model's window"*"rc=1"*) ok "glyd run --prompt (a prompt on stdin) says so when the prompt outgrows the window: $(printf '%s' "$big" | cut -c1-150)";; *) fail "no plain message from glyd run for a prompt longer than the window: $big";; esac
  win=$(run 'curl -s http://127.0.0.1:8000/v1/models' | sed -nE 's/.*"max_model_len":([0-9]+).*/\1/p' | head -1)
  chatout=$(run "( sleep 6; echo '\"\"\"'; yes \"\$(yes word | head -n 600 | tr '\n' ' ')\" | head -n \$(( ${win:-10240} / 600 + 3 )); echo '\"\"\"'; sleep 10; echo /bye ) | ${gf}timeout 600 script -qec 'glyd run $MODEL' /dev/null" 2>&1 | tr -d '\r' | sed 's/\x1b\[[0-9;?]*[A-Za-z]//g')  # (a limit: a glyd that starts a server of its own is not at the prompt when the input has been typed)
  grep -qF "This conversation is longer than the model's window" "$README" && grep -qF "Start a new chat with /clear" "$README" || fail "the README does not quote the message the terminal chat gives for a conversation past the window"
  case $chatout in *"This conversation is longer than the model's window"*"Start a new chat with /clear"*) ok "the terminal chat says so when the conversation outgrows the window (typed as one message of several lines)";; *) fail "no plain message in the terminal chat for a conversation longer than the window: $(printf '%s' "$chatout" | tail -5 | tr '\n' '|' | cut -c1-300)";; esac
  for r in uvx docker; do route $r && { webui_route $r "$py" "$MODEL" || break; }; done
  if true; then  # a server on every interface needs a key; Open WebUI's bridge route (the container on its own network, reaching the server by the host's address) is the user of one
    glyd_stop_serve
    key=acceptance-$RANDOM$RANDOM
    # the key as -- --api-key: glyd's readiness probe needs none of it (it asked /v1 and was told 401, so a server with a key never came up)
    : > "$LOGS/serve-flag.out"
    bg "${gf}glyd serve $MODEL --port 8000 --host 0.0.0.0 -- --api-key $key > $IN/logs/serve-flag.out 2>&1"
    up=; for _ in $(seq 1 $((UPWAIT / 3))); do grep -q '^Serving' "$LOGS/serve-flag.out" && { up=1; break; }; run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break; sleep 3; done
    if [ -n "$up" ]; then glyd_key "$LOGS/serve-flag.out" "$key" flag; else fail "glyd serve --host 0.0.0.0 -- --api-key never printed Serving (logs/serve-flag.out): $(tail -3 "$LOGS/serve-flag.out" | tr '\n' '|' | cut -c1-200)"; fi
    glyd_stop_serve
    # the key as the README says (VLLM_API_KEY in the environment), and Open WebUI against it
    key=acceptance-$RANDOM$RANDOM
    line=$(block serve-key)
    [ -n "$line" ] || fail "the README has no <!-- acceptance: serve-key --> block"
    line=$(printf '%s' "$line" | sed "s|YOUR_KEY|$key|; s|Qwen/Qwen3-8B|$MODEL|")
    say "-- serve with a key: $(printf '%s' "$line" | sed "s|$key|YOUR_KEY|")"
    : > "$LOGS/serve-bridge.out"
    bg "${gf}$line > $IN/logs/serve-bridge.out 2>&1"
    up=; for _ in $(seq 1 $((UPWAIT / 3))); do grep -q '^Serving' "$LOGS/serve-bridge.out" && { up=1; break; }; run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break; sleep 3; done
    if [ -n "$up" ]; then glyd_key "$LOGS/serve-bridge.out" "$key" env; route bridge && webui_route bridge "$py" "$MODEL" "$key"; else fail "the README's serve-key command never printed Serving (logs/serve-bridge.out): $(tail -3 "$LOGS/serve-bridge.out" | tr '\n' '|' | cut -c1-200)"; fi
  fi
  if [ -n "$HOLD" ]; then
    block webui-uvx > "$WORK/webui-uvx.sh"
    bg "setsid bash -c 'echo \$\$ > $IN/logs/webui.pid; exec bash $IN/webui-uvx.sh' > $IN/logs/webui-uvx.log 2>&1"
    say "-- held for $HOLD minutes (touch $WORK/stop to end): the server on http://127.0.0.1:8000, Open WebUI on http://127.0.0.1:3000"
    rm -f "$WORK/stop"
    for _ in $(seq 1 $(( HOLD * 20 ))); do [ -e "$WORK/stop" ] && break; sleep 3; done
  fi
  # 6. stopped: the port and the GPU's memory let go
  glyd_stop_serve
  run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 && fail "glyd serve did not stop on SIGTERM" || ok "glyd serve stopped on SIGTERM"
  ss -ltn 2> /dev/null | grep -q ':8000 ' && fail "port 8000 is in use after glyd serve stopped" || true
  glyd_logs
  # 7. what a user does to it: Ctrl-C while the model loads, the engine dying under a chat (their logs are not checked for tracebacks: those are the point)
  [ -n "$BUDGET" ] || { glyd_ctrlc; glyd_chat_dies; }
  glyd_remove
  glyd_pip
}

glyd_stop_serve() {  # SIGTERM to glyd serve, and its port and processes let go
  run 'pkill -TERM -f "[g]lyd serve"' > /dev/null 2>&1
  for _ in $(seq 1 60); do run 'pgrep -f "[g]lyd serve"' > /dev/null 2>&1 || break; sleep 1; done
}

# --- the cli flow: install.sh on a machine with no NVIDIA GPU (a container started without --gpus): the compression program from the release, and what glyd run says
cli_flow() {
  local env rc out
  cp "$HERE/../../scripts/install.sh" "$WORK/install.sh" || { fail "no scripts/install.sh beside gpu/vllm"; return 1; }
  env=; [ -z "$VERSION" ] || env="GLYD_VERSION=$VERSION "
  rm -rf "$WORK/home/.local" "$WORK/home/.cache" "$WORK/home-foreign"
  say "-- install.sh with no NVIDIA GPU: $(run 'command -v nvidia-smi || echo no nvidia-smi' | tail -1)"
  run "${env}sh $IN/install.sh" > "$LOGS/install.log" 2>&1; rc=$?
  [ $rc = 0 ] || { fail "install.sh exited $rc (logs/install.log)"; tail -8 "$LOGS/install.log"; return 1; }
  ok "install.sh exited 0; its output: $(tr '\n' '|' < "$LOGS/install.log" | cut -c1-420)"
  grep -q 'No NVIDIA GPU answered' "$LOGS/install.log" && grep -q 'needs Linux with an NVIDIA GPU' "$LOGS/install.log" && ok "it says plainly that glyd run needs Linux with an NVIDIA GPU" || fail "install.sh did not say that glyd run needs Linux with an NVIDIA GPU (logs/install.log)"
  grep -q 'sha256' "$LOGS/install.log" && ok "the tarball was checked against the release's sha256" || fail "no sha256 check in install.sh's output"
  [ ! -e "$WORK/home/.local/bin/uv" ] && [ ! -d "$WORK/home/.local/share/uv" ] && ok "no uv, and no tool environment, for a machine that cannot run glyd run" || fail "install.sh put uv or a tool environment where there is no GPU"
  for p in glyd glyd-store glyd-gpu; do [ -L "$WORK/home/.local/bin/$p" ] || fail "no link ~/.local/bin/$p"; done
  run 'glyd --version' > "$LOGS/version.txt" 2>&1; rc=$?
  [ $rc = 0 ] && ok "glyd --version: $(head -2 "$LOGS/version.txt" | tr '\n' ' ')" || fail "glyd --version (logs/version.txt)"
  run "base64 /dev/urandom | head -c 3000000 > $IN/home/a.txt; glyd --max $IN/home/a.txt -o $IN/home/a.glyd && glyd -d $IN/home/a.glyd -o $IN/home/b.txt && cmp $IN/home/a.txt $IN/home/b.txt" > "$LOGS/roundtrip.txt" 2>&1 \
    && ok "a file through glyd and back: $(run "stat -c %s $IN/home/a.txt $IN/home/a.glyd" | tr '\n' ' ') bytes, identical" || fail "the compression round trip (logs/roundtrip.txt)"
  if [ -n "$RUSTGLYD" ]; then  # the Rust glyd of this tree: run, serve, doctor and login say what they are, in place of the usage text of a file named run
    run "PATH=$IN/rust:\$PATH glyd run Qwen/Qwen3-8B; echo rc=\$?" > "$LOGS/run-rust.txt" 2>&1
    grep -q 'not installed here' "$LOGS/run-rust.txt" && grep -q 'rc=127' "$LOGS/run-rust.txt" && ! grep -q 'Usage:' "$LOGS/run-rust.txt" && ok "glyd run, where there is no Python tool: $(head -1 "$LOGS/run-rust.txt" | cut -c1-200)" || fail "glyd run with no Python tool did not say so (logs/run-rust.txt): $(head -3 "$LOGS/run-rust.txt" | tr '\n' '|' | cut -c1-200)"
  fi
  run "${env}sh $IN/install.sh" > "$LOGS/install-again.log" 2>&1; rc=$?
  [ $rc = 0 ] && ok "install.sh run again (an update) replaced its own links and exited 0" || fail "install.sh run again exited $rc (logs/install-again.log): $(tail -3 "$LOGS/install-again.log" | tr '\n' '|')"
  mkdir -p "$WORK/home-foreign/.local/bin"; printf '#!/bin/sh\necho mine\n' > "$WORK/home-foreign/.local/bin/glyd"; chmod +x "$WORK/home-foreign/.local/bin/glyd"
  run "HOME=$IN/home-foreign ${env}sh $IN/install.sh" > "$LOGS/install-foreign.log" 2>&1; rc=$?
  [ $rc = 1 ] && grep -q 'this installer did not make it' "$LOGS/install-foreign.log" && [ "$(cat "$WORK/home-foreign/.local/bin/glyd")" = "$(printf '#!/bin/sh\necho mine')" ] && ok "a glyd of the user's own in ~/.local/bin is not replaced" || fail "install.sh with a glyd of the user's own (exit $rc, logs/install-foreign.log)"
  rm -rf "$WORK/home-foreign"
}

glyd_pip() {  # --pip-refusal: the same wheel by pip into a virtual environment: no installer, no ziglang: the compiler's refusal
  [ -n "$PIPREF" ] && [ $COMPILER = none ] || return 0
  local spec=${WHEEL:+$IN/wheel/$(basename "$WHEEL")[vllm]} out
  spec=${spec:-glyd[vllm]${VERSION:+==$VERSION}}
  rm -rf "$WORK/pipenv"  # (uv venv will not replace one: a work directory that was used before has it)
  run "uv venv --python 3.12 $IN/pipenv > $IN/logs/pip.log 2>&1 && uv pip install --python $IN/pipenv/bin/python '$spec' >> $IN/logs/pip.log 2>&1" || { fail "the pip install did not work (logs/pip.log)"; return; }
  out=$(run "$IN/pipenv/bin/glyd run $MODEL --prompt hi 2>&1; echo rc=\$?" | tr '\n' ' ')
  case $out in
    *"needs a C compiler"*"build-essential"*"installer again"*"rc=1"*) ok "pip install, no compiler: glyd run stops with the install command ($(printf '%s' "$out" | cut -c1-200))";;
    *) fail "pip install, no compiler: not the refusal expected: $out";;
  esac
}

if [ "$FLOW" = cli ]; then
  cli_flow
  if [ ${#FAILS[@]} = 0 ]; then say "PASSED"; else say "FAILED (${#FAILS[@]}): $(printf '%s; ' "${FAILS[@]}")"; fi
  [ ${#FAILS[@]} = 0 ]
  exit
fi
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
