#!/usr/bin/env bash
# Qwen2.5-72B-Instruct served by vLLM over 2 GPUs (tensor parallel), vLLM's own bf16 against Glyd (--quantization glyd), on one
# 2x H100 80 GB session: unattended, on an x86_64 or aarch64 host whose NVIDIA driver runs CUDA 13 (580 or newer), 75 minutes at
# most. In ~: this and big72_src.tar (v0.26.0's tree, 10e8cae: gpu/ for the library, bindings/python for the glyd package and its
# vLLM plugin, gpu/vllm/ for the bench). Its steps, in order:
#   machine, uv, then the download of VJ_MODEL (~145 GB into ~/hf, in the background, timed; no Hugging Face token)
#   env    a venv with vLLM (VJ_VLLM, from PyPI) and pip's nvcc for torch's CUDA; the glyd package from the tree (editable, no
#          dependencies)
#   build  the library for this GPU alone (build_lib.sh's flags)
#   serve  a mode at a time, VJ_MODES in order (Glyd first: its numbers are the point, and the clock cannot take them), each
#          by gpu/vllm/bench_serve.sh on an empty compile cache (a cold start): one vllm serve (--max-model-len 4096,
#          --gpu-memory-utilization VJ_UTIL, --tensor-parallel-size VJ_TP, VJ_SERVE_ARGS), the server's weights, KV cache and
#          maximum concurrency at 4,096 tokens noted, then vllm bench serve at VJ_RATES requests a second (inf: every request
#          at once), VJ_PROMPTS prompts each, VJ_IN tokens in, VJ_OUT out, --ignore-eos, a seed for each rate (so no rate's prompts
#          are an earlier one's on the server, whose prefix cache is on, and each mode gets the same prompts). A server that does not start (bf16
#          with no room for the KV cache of one 4,096-token request: vLLM refuses) is logged with its message, and the job goes
#          on: there is nothing of that mode to bench. A mode that cannot start in what is left of the clock is not run
#   retry  where bf16 had no room for the KV cache at VJ_UTIL (vLLM's refusal, not another failure), at VJ_RETRY_UTIL (0.95):
#          Glyd's cold start with its KV cache noted and no benches, then bf16 again, benched as before where it starts. They
#          go in a results directory of their own (bench-MODEL-utilRETRY)
# results/summary.txt rewritten after every step, with one table, a row for each run and its gpu_memory_utilization (Glyd and
# bf16 at VJ_UTIL, and at the retry's) and each server's highest prefix cache hit rate (above 1%: the run is flagged NOT
# COMPARABLE); results/DONE from the exit trap however the job ends, and a hard stop at VJ_END (75
# minutes): every step's timeout ends by it, and a watchdog stops what is left.
#   bash ~/big72_job.sh
#   VJ_RETRY_UTIL=0 bash ~/big72_job.sh                 (without the retry)
# Env: VJ_MODEL (Qwen/Qwen2.5-72B-Instruct), VJ_TP (2), VJ_MODES ("glyd bf16"), VJ_UTIL (0.9), VJ_RATES ("1 inf"), VJ_PROMPTS
# ("64 192": a count for each rate), VJ_IN (1024), VJ_OUT (256), VJ_SERVE_ARGS (--max-num-seqs 128 --compilation-config
# {"max_cudagraph_capture_size":128}: the budget job's, which cut each start; empty: vLLM's own), VJ_MODE_MAX (1800 s: the most a
# mode takes), VJ_EST_GLYD (600 s) and VJ_EST_BF16 (420 s): what a mode's start takes, below which what is left of the clock skips
# the mode, VJ_RETRY_UTIL (0.95; 0 or empty: no retry; not above VJ_UTIL: none), VJ_BUDGET (3600 s: no mode starts past it),
# VJ_END (4500 s), VJ_DL_MAX (3300 s: the download's own limit), VJ_NEED_GB (free disk wanted; 180 for a 72B model), VJ_VLLM
# (vllm==0.30.0); FILES (~), R (~/results), W (~/big72w), HF_HOME (~/hf).
set -u
R=${R:-$HOME/results}
W=${W:-$HOME/big72w}
FILES=${FILES:-$HOME}
mkdir -p "$R/log" "$W/tmp"
trap 'touch "$R/DONE"' EXIT
B=$R/bench-$(basename "${VJ_MODEL:-Qwen/Qwen2.5-72B-Instruct}")
RETRY=${VJ_RETRY_UTIL-0.95}; [ "$RETRY" = 0 ] && RETRY=
B2=$B-util$RETRY
for d in "$B" "$B2"; do [ -d "$d" ] && mv "$d" "$d.before-$(date +%H%M%S)"; done  # (an earlier run's logs in this results directory: kept, and out of this summary)
exec > >(tee -a "$R/job.log") 2>&1
T0=$(date +%s)
BUDGET=${VJ_BUDGET:-3600}
END=${VJ_END:-4500}
MODEL=${VJ_MODEL:-Qwen/Qwen2.5-72B-Instruct}
NAME=$(basename "$MODEL")
B=$R/bench-$NAME
TP=${VJ_TP:-2}
MODES=${VJ_MODES:-glyd bf16}
RATES=${VJ_RATES:-1 inf}
PROMPTS=${VJ_PROMPTS:-64 192}
UTIL=${VJ_UTIL:-0.9}
CG='{"max_cudagraph_capture_size":128}'  # (no spaces: bench_serve.sh word-splits SERVE_ARGS)
SERVE=${VJ_SERVE_ARGS---max-num-seqs 128 --compilation-config $CG}
el() { echo $(( $(date +%s) - T0 )); }
step() { echo "== $(date -u +%T) (+$(el) s) $*"; }
tmo() { local t=$(( END - $(el) )); [ "$t" -gt "$1" ] && t=$1; echo $(( t > 10 ? t : 10 )); }  # a step's timeout: $1 s, or to END
# The hard stop: a step that has no timeout of its own (an install, a stuck curl) is stopped here, and the trap writes DONE (the
# shell first, which ends once its foreground command does; then its children)
( sleep $(( END + 60 )); grep -qF -- "$0" /proc/$$/cmdline 2> /dev/null || exit  # (still this job: not another process by the number)
  echo "hard stop at +$(el) s" >> "$R/steps.txt"; kill -TERM $$; pkill -TERM -P $$ ) > /dev/null 2>&1 &
WD=$!
DLPID=
finish() { { pkill -TERM -P $WD; kill $WD; } 2> /dev/null; [ -n "$DLPID" ] && { pkill -TERM -P "$DLPID"; kill "$DLPID"; } 2> /dev/null; summ 2> /dev/null; touch "$R/DONE"; }
trap finish EXIT
cat > "$W/summary.py" <<'PYEOF'
"""big72_job.sh's summary.txt: the machine, one table with a row for each run (bf16 and Glyd at each gpu_memory_utilization), what
did not start and why, bf16 against Glyd at the same setting, and the steps.

    python3 summary.py RESULTS BENCHDIR RETRYDIR MODEL TP UTIL RETRY "RATES" "PROMPTS" "MODES" "SERVE ARGS" """
import json
import os
import re
import sys

R, B, B2, MODEL, TP, UTIL, RETRY, RATES, PROMPTS, MODES, SERVE = sys.argv[1:12]
RATES, PROMPTS, MODES = RATES.split(), PROMPTS.split(), MODES.split()
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PFX = re.compile(r"^(?:\([A-Za-z_0-9]+(?: pid=\d+)?\) )?(?:(?:INFO|WARNING|ERROR|DEBUG|CRITICAL) +\d\d-\d\d \d\d:\d\d:\d\d \[[^\]]*\] )?")
STAMP = re.compile(r"(?:INFO|WARNING|ERROR) +\d\d-\d\d (\d\d):(\d\d):(\d\d) ")
EXC = re.compile(r"^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception): ")
NOROOM = re.compile(r"No available memory for the cache blocks|KV cache is needed|larger than the available KV cache")
NAME = {"bf16": "bf16", "glyd": "Glyd"}


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def server(mode, d):
    """What a run's server log says (None where it was not run): whether it answered, its weights, KV cache and maximum
    concurrency, how long the start took, and where it did not start, its error lines."""
    p = os.path.join(d, f"serve-{mode}.txt")
    if not os.path.exists(p):
        return None
    raw = [ANSI.sub("", l) for l in read(p).splitlines()]
    msgs = [PFX.sub("", l) for l in raw]
    text = "\n".join(msgs)
    s = {"log": os.path.relpath(p, R), "ok": "Application startup complete" in text}
    m = re.search(r"Model loading took ([\d.]+) GiB", text)
    s["weights"] = float(m.group(1)) if m else None
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    s["tokens"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
    s["conc"] = float(m.group(2)) if m else None
    s["avail"] = re.findall(r"Available KV cache memory: (-?[\d.]+) GiB", text)
    m = re.search(r"glyd: (mma12|mma) layout", text)
    s["layout"] = {"mma12": "12-bit (mma12)", "mma": "smallest (mma)"}[m.group(1)] if m else None
    t = [int(h) * 3600 + int(mi) * 60 + int(sec) for h, mi, sec in (STAMP.search(l).groups() for l in raw if STAMP.search(l))]
    up = [int(h) * 3600 + int(mi) * 60 + int(sec) for h, mi, sec in (STAMP.search(l).groups() for l in raw if "Starting vLLM server" in l and STAMP.search(l))]
    s["start"] = (up[0] - t[0]) % 86400 if t and up else None
    h = [float(x) for x in re.findall(r"Prefix cache hit rate: ([\d.]+)%", text)]
    s["hit"] = max(h) if h else None
    errs = []
    for l in msgs:
        l = l.strip()
        if EXC.match(l) and l not in errs:
            errs.append(l)
    s["errs"] = errs
    s["ctx"] = [l.strip() for l in msgs if re.search(r"Available KV cache memory|Free memory on device", l)][:4]
    s["tail"] = [l.strip() for l in msgs if l.strip()][-3:]
    s["noroom"] = not s["ok"] and any(NOROOM.search(l) for l in errs)
    return s


def bench(mode, rate, d):
    try:
        return json.load(open(os.path.join(d, f"{mode}-rate{rate}.json")))
    except (OSError, ValueError):
        return None


def progress(mode, rate, d):
    """Where a bench that wrote no result ended: the last count of its progress bar."""
    p = os.path.join(d, f"bench-{mode}-rate{rate}.txt")
    if not os.path.exists(p):
        return "not run"
    m = re.findall(r"(\d+)/(\d+) \[", read(p).replace("\r", "\n"))
    return f"no result ({m[-1][0]} of {m[-1][1]} requests when it ended)" if m else "no result"


def ratio(a, b):
    return f"{a / b:.2f}x" if a is not None and b else None


# The runs, a row each: both modes at UTIL; and, where bf16 had no room for the KV cache at UTIL (or the retry ran), at RETRY
# Glyd's start for its KV cache alone and bf16's start, benched where it started
runs = [{"mode": m, "util": UTIL, "dir": B, "kvonly": False, "S": server(m, B)} for m in ("glyd", "bf16") if m in MODES]
first = next((r for r in runs if r["mode"] == "bf16"), None)
if RETRY and first and (os.path.isdir(B2) or (first["S"] and first["S"]["noroom"])):
    if "glyd" in MODES:
        runs.append({"mode": "glyd", "util": RETRY, "dir": B2, "kvonly": True, "S": server("glyd", B2)})
    runs.append({"mode": "bf16", "util": RETRY, "dir": B2, "kvonly": False, "S": server("bf16", B2)})

columns = ["Run", "--gpu-memory-utilization", "Server", "Weights, each GPU (GiB)", "Available KV cache memory, each GPU (GiB)", "KV cache (tokens)", "Max concurrency at 4,096 tokens", "Prefix cache hit rate, at most"]
tags = []
for rate, n in zip(RATES, PROMPTS):
    tag = "saturated" if rate == "inf" else f"rate {rate}"
    tags.append((rate, n, tag))
    columns += [f"{tag}, {n} prompts: requests/s"] + ([f"{tag}: output tokens/s"] if rate == "inf" else []) + [f"{tag}: TTFT mean / p99 (ms)", f"{tag}: TPOT mean / p99 (ms)"]


def cells(r):
    s = r["S"]
    row = [NAME[r["mode"]], r["util"]]
    if s is None:
        row += ["not run"] + ["-"] * (len(columns) - 3)
        return row
    row.append(("yes, about %s s" % s["start"] if s["start"] is not None else "yes") + (", KV cache alone" if r["kvonly"] else "") if s["ok"] else "no: see below")
    row += ["-" if s["weights"] is None else f"{s['weights']:.2f}", s["avail"][0] if s["avail"] else "-", "-" if s["tokens"] is None else f"{s['tokens']:,}", "-" if s["conc"] is None else f"{s['conc']:.2f}x",
            "-" if s["hit"] is None or r["kvonly"] else f"{s['hit']:.1f}%"]
    for rate, n, tag in tags:
        width = 4 if rate == "inf" else 3
        d = bench(r["mode"], rate, r["dir"])
        if r["kvonly"]:
            row += ["not benched"] + ["-"] * (width - 1)
        elif d is None:
            row += [progress(r["mode"], rate, r["dir"]) if s["ok"] else "-"] + ["-"] * (width - 1)
        else:
            row += [f"{d['request_throughput']:.2f}"] + ([f"{d['output_throughput']:.1f}"] if rate == "inf" else []) + [f"{d['mean_ttft_ms']:,.0f} / {d['p99_ttft_ms']:,.0f}", f"{d['mean_tpot_ms']:.1f} / {d['p99_tpot_ms']:.1f}"]
    return row


print(read(os.path.join(R, "machine-short.txt")).strip())
env = read(os.path.join(R, "env.txt")).splitlines()
print(env[0] if env else "(no env yet)")
print(f"vllm serve --max-model-len 4096 --gpu-memory-utilization {UTIL} --tensor-parallel-size {TP} {SERVE}; each server started cold, on an empty compile cache")
if len(runs) > 2:
    print(f"at {RETRY}, after bf16 had no room for the KV cache at {UTIL}: Glyd's start for its KV cache alone (no benches), then bf16 again, benched where it started")
print(f"vllm bench serve: random dataset, {os.environ.get('VJ_IN', '1024')} tokens in, {os.environ.get('VJ_OUT', '256')} out, --ignore-eos, a seed for each rate (0, 1: no rate's prompts are an earlier one's on the server; each mode gets the same prompts); rate 1: one request a second, saturated: every request at once")
hits = {f"{NAME[r['mode']]} at {r['util']}": r["S"]["hit"] for r in runs if r["S"] and not r["kvonly"] and r["S"]["hit"] is not None}
print("Prefix cache hit rate, at most (the servers' logs): " + (", ".join(f"{k} {v:.1f}%" for k, v in hits.items()) or "none read"))
over = [f"{k} {v:.1f}%" for k, v in hits.items() if v > 1]
FLAG = f"NOT COMPARABLE: the prefix cache skipped the prefill of repeated prompts (hit rate above 1%: {', '.join(over)})" if over else ""
if FLAG:
    print(FLAG)
dl = read(os.path.join(R, "log", f"dl-{os.path.basename(MODEL)}.txt")).strip().splitlines()
m = re.match(r"exit (\d+): (\d*) bytes in (\d+) s \(at \+(\d+) s\)", dl[-1]) if dl else None
print("download (no token): " + (f"exit {m[1]}, {int(m[2] or 0):,} bytes in {m[3]} s = {int(m[2] or 0) / 1e9 / max(int(m[3]), 1):.2f} GB/s, finished at +{m[4]} s of the job" if m else "not finished"))
print()
print(f"{MODEL}, {TP} GPU{'s' if TP != '1' else ''} (tensor parallel): a row for each run")
print()
print("| " + " | ".join(columns) + " |")
print("| :--- | :--- | :--- | " + " | ".join(["---:"] * (len(columns) - 3)) + " |")
for r in runs:
    print("| " + " | ".join(cells(r)) + " |")
print()
for r in runs:
    s, tag = r["S"], f"{NAME[r['mode']]} at --gpu-memory-utilization {r['util']}"
    if s is None:
        print(f"{tag}: not run (see the steps below).")
    elif not s["ok"]:
        print(f"{tag} did not start" + (": there is no room for the KV cache of even one 4,096-token request." if s["noroom"] else "."))
        print(f"  the server's log ({s['log']}):")
        for l in s["ctx"] + (s["errs"][:1] + s["errs"][-1:] if len(s["errs"]) > 1 else s["errs"] or s["tail"]):
            print(f"  | {l}")
    elif r["kvonly"]:
        print(f"{tag}: started for its KV cache alone (not benched): {s['tokens']:,} tokens, {s['conc']:.2f}x a 4,096-token request, each GPU holding {s['weights']} GiB of weights.")
    elif s["conc"] is not None and s["conc"] < 1:
        print(f"{tag}: the KV cache holds under one 4,096-token request ({s['tokens']:,} tokens, {s['conc']:.2f}x).")
    elif s["tokens"] is not None:
        print(f"{tag}: KV cache {s['tokens']:,} tokens, {s['conc']:.2f}x a 4,096-token request, each GPU holding {s['weights']} GiB of weights.")
settings = list(dict.fromkeys(r["util"] for r in runs))
print()
print("Glyd against bf16, at the same setting" + (" (NOT COMPARABLE, as flagged above)" if FLAG else "") + ":")
for u in settings:
    g = next((r for r in runs if r["util"] == u and r["mode"] == "glyd"), None)
    b = next((r for r in runs if r["util"] == u and r["mode"] == "bf16"), None)
    if not (g and b and g["S"] and b["S"]):
        print(f"  {u}: not both run.")
    elif not b["S"]["ok"]:
        print(f"  {u}: bf16 did not start" + (f"; Glyd's KV cache is {g['S']['tokens']:,} tokens." if g["S"]["tokens"] else "."))
    elif not g["S"]["ok"]:
        print(f"  {u}: Glyd did not start; bf16's KV cache is {b['S']['tokens']:,} tokens.")
    else:
        line = f"  {u}: KV cache {ratio(g['S']['tokens'], b['S']['tokens'])} ({g['S']['tokens']:,} against {b['S']['tokens']:,} tokens), weights {ratio(g['S']['weights'], b['S']['weights'])} ({g['S']['weights']} against {b['S']['weights']} GiB)"
        for rate, n, tag in tags:
            gd, bd = bench("glyd", rate, g["dir"]), bench("bf16", rate, b["dir"])
            if gd and bd:
                line += f"; {tag} requests/s {ratio(gd['request_throughput'], bd['request_throughput'])}, TTFT {ratio(gd['mean_ttft_ms'], bd['mean_ttft_ms'])}, TPOT {ratio(gd['mean_tpot_ms'], bd['mean_tpot_ms'])}"
        print(line + ".")
benched = [(NAME[r["mode"]], r["util"]) for r in runs if any(bench(r["mode"], rate, r["dir"]) for rate in RATES)]
if len({u for _, u in benched}) > 1:
    print("Benched: " + ", ".join(f"{n} at {u}" for n, u in benched) + ". Those are different settings, so their requests/s and token times are not like for like; the KV cache of the two at one setting is.")
print()
print(read(os.path.join(R, "steps.txt")).strip())
if FLAG:
    print()
    print(FLAG)
PYEOF
summ() {  # results/summary.txt from what is there
  python3 "$W/summary.py" "$R" "$B" "$B2" "$MODEL" "$TP" "$UTIL" "$RETRY" "$RATES" "$PROMPTS" "$MODES" "$SERVE" > "$R/summary.tmp" 2>&1; mv "$R/summary.tmp" "$R/summary.txt"
}
done_() { echo "$(date -u +%T) (+$(el) s) $*" >> "$R/steps.txt"; summ; }
fail() { done_ "FAIL: $*"; exit 1; }
idle() {  # every GPU's memory free again, up to $1 s; else the servers' leftovers (processes with VJ_SRV in their environment) killed
  local i p
  for i in $(seq 1 $(( $1 / 5 ))); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1500' | wc -l)" -eq 0 ] && return 0
    sleep 5
  done
  for p in $(grep -la "VJ_SRV=$$" /proc/[0-9]*/environ 2> /dev/null | cut -d/ -f3); do kill -9 "$p" 2> /dev/null; done
  sleep 10
}

step "machine"
{ uname -srvmo; nvidia-smi; nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
  nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g'; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; lscpu | grep -E "Model name|Architecture"; nproc
  free -g | head -2; df -h "$HOME" | tail -1; date -u; echo "host LD_LIBRARY_PATH (unset for the job): ${LD_LIBRARY_PATH:-(none)}"; } > "$R/machine.txt" 2>&1
unset LD_LIBRARY_PATH  # the host's CUDA libraries never ahead of the venv's own
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN  # the model is ungated: no token is used
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAMEGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NG=$(nvidia-smi -L | grep -c "^GPU")
ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
[ "$NG" -ge "$TP" ] || fail "VJ_TP $TP, but $NG GPUs here"
GB=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
NEED=${VJ_NEED_GB:-$(case $MODEL in (*72B*) echo 180 ;; (*) echo 30 ;; esac)}  # the model (145 GB for the 72B), the venv and its caches
[ "${GB:-0}" -ge "$NEED" ] || echo "WARNING: $GB GB free in ~, $NEED wanted for $MODEL"
rm -rf "$W/src" && mkdir -p "$W/src" && tar -C "$W/src" -xf "$FILES/big72_src.tar" || fail "no big72_src.tar in $FILES"
{ grep -q 'SEED:-0' "$W/src/gpu/vllm/bench_serve.sh" && grep -q 'Prefix cache hit rate' "$W/src/gpu/vllm/bench_summary.py"; } ||
  fail "big72_src.tar's bench scripts use one seed for every pass and note no prefix cache hit rate: its gpu/vllm/bench_serve.sh and bench_summary.py are the old ones"
echo "$NG x $NAMEGPU (compute $CC, sm_$ARCH, $MIB MiB each), $(uname -m) host, $(nproc) CPUs; tree $(cat "$W/src/COMMIT"); model $MODEL, tensor parallel $TP; modes $MODES; rates $RATES, prompts $PROMPTS, ${VJ_IN:-1024} tokens in, ${VJ_OUT:-256} out" | tee "$R/machine-short.txt"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -gt 0 ] && echo "WARNING: other processes on the GPUs" | tee -a "$R/machine-short.txt"
done_ "machine"

step "uv, then the download (in the background, timed)"
UV=$(command -v uv || ls "$HOME/tools/uv/uv" "$W/uv/uv" 2> /dev/null | head -1)
if [ -z "$UV" ]; then
  mkdir -p "$W/uv" && curl -sSfL --retry 3 "https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz" | tar xz --strip-components=1 -C "$W/uv" && UV=$W/uv/uv
fi
[ -x "$UV" ] || fail "no uv"
export HF_HOME=${HF_HOME:-$HOME/hf} TOKENIZERS_PARALLELISM=false TMPDIR=$W/tmp UV_CACHE_DIR=${UV_CACHE_DIR:-$W/uvcache}
[ -x "$W/dl/bin/python" ] || ( "$UV" venv -q --python 3.12 "$W/dl" && "$UV" pip install -q --python "$W/dl/bin/python" huggingface_hub hf_xet ) > "$R/log/dl-env.txt" 2>&1 || fail "no download env (log/dl-env.txt)"
dl() {  # REPO: its snapshot, W/NAME.dir its directory; log/dl-NAME.txt its time and size
  local n t; n=$(basename "$1"); t=$(date +%s)
  timeout "${VJ_DL_MAX:-3300}" "$W/dl/bin/python" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))" "$1" > "$W/$n.dir" 2> "$R/log/dl-$n.txt"
  local e=$?; echo "exit $e: $(du -sbL "$(tail -1 "$W/$n.dir")" 2> /dev/null | cut -f1) bytes in $(( $(date +%s) - t )) s (at +$(el) s)" >> "$R/log/dl-$n.txt"
}
dl "$MODEL" &
DLPID=$!
got() {  # REPO: 0 once its download is done and whole (1 if it failed, or past the budget)
  local n; n=$(basename "$1")
  until grep -q "^exit" "$R/log/dl-$n.txt" 2> /dev/null; do [ "$(el)" -ge "$BUDGET" ] && return 1; sleep 3; done
  grep -q "^exit 0" "$R/log/dl-$n.txt" && [ -f "$(tail -1 "$W/$n.dir")/config.json" ]
}

step "env: vLLM from PyPI (${VJ_VLLM:-vllm==0.30.0}) and nvcc, then the glyd package"
t=$(date +%s)
if [ -x "$W/venv/bin/python" ] && "$W/venv/bin/python" -c "import vllm" 2> /dev/null; then
  echo "an earlier job's venv, $W/venv" > "$R/log/env-vllm.txt"
else
  { rm -rf "$W/venv" && "$UV" venv -q --python 3.12 "$W/venv" && "$UV" pip install --python "$W/venv/bin/python" "${VJ_VLLM:-vllm==0.30.0}"; } > "$R/log/env-vllm.txt" 2>&1 || fail "vLLM did not install (log/env-vllm.txt)"
fi
PY=$W/venv/bin/python
MM=$("$PY" -c "import torch; print(torch.version.cuda)")
"$UV" pip install --python "$PY" "nvidia-cuda-nvcc==$MM.*" "nvidia-cuda-cccl==$MM.*" "nvidia-cuda-crt==$MM.*" "nvidia-nvvm==$MM.*" "nvidia-cuda-runtime==$MM.*" > "$R/log/env-nvcc.txt" 2>&1 || fail "nvcc did not install (log/env-nvcc.txt)"
CU=$("$PY" -c "import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], 'cu' + '$MM'.split('.')[0]))")
mkdir -p "$CU/lib64" && ln -sf "../lib/$(ls "$CU/lib" | grep -m1 '^libcudart.so')" "$CU/lib64/libcudart.so"
export CUDA_HOME=$CU PATH=$W/venv/bin:$CU/bin:$PATH
cp "$W/src/LICENSE" "$W/src/COPYING" "$W/src/bindings/python/"; cp "$W/src/glyd-store/LICENSE" "$W/src/bindings/python/LICENSE-glyd-store"; cp "$W/src/gpu/LICENSE" "$W/src/bindings/python/LICENSE-glyd-gpu"
"$UV" pip install --python "$PY" --no-deps -e "$W/src/bindings/python" > "$R/log/env-glyd.txt" 2>&1 || fail "the glyd package did not install (log/env-glyd.txt)"
"$PY" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__, '| GPUs', torch.cuda.device_count(), 'x', torch.cuda.get_device_name())" > "$R/env.txt" 2>&1 || fail "no vLLM on the GPU ($(tail -2 "$R/env.txt" | tr '\n' ' '))"
echo "nvcc: $(nvcc --version | tail -1)" >> "$R/env.txt"
cat "$R/env.txt"
done_ "env: vLLM and nvcc in $(( $(date +%s) - t )) s ($(head -1 "$R/env.txt"))"

step "build: the library for sm_$ARCH"
t=$(date +%s)
F=(-O3 -std=c++20 --expt-relaxed-constexpr -isystem "$CU/include" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
   -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -gencode "arch=compute_$ARCH,code=sm_$ARCH" -Xcompiler -fPIC,-fvisibility=hidden)
MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
mkdir -p "$W/lib"
if [ -f "$W/lib/libglyd_gpu_cuda$MAJOR.so" ] && [ "$W/lib/libglyd_gpu_cuda$MAJOR.so" -nt "$FILES/big72_src.tar" ]; then
  echo "an earlier job's build of this tarball" > "$R/log/build.txt"
else
  rm -f "$W/lib/libglyd_gpu_cuda$MAJOR.so"
  { timeout "$(tmo 900)" nvcc "${F[@]}" -c -o "$W/lib/glyd_gpu.o" "$W/src/gpu/glyd_gpu.cu" &&
    timeout 300 nvcc "${F[@]}" -shared -Xlinker --exclude-libs,ALL -cudart static -L"$CU/lib" -L"$CU/lib64" -o "$W/lib/libglyd_gpu_cuda$MAJOR.so" "$W/lib/glyd_gpu.o"; } > "$R/log/build.txt" 2>&1
fi
[ -f "$W/lib/libglyd_gpu_cuda$MAJOR.so" ] || fail "the library did not build (log/build.txt)"
export GLYD_GPU_LIB=$W/lib/libglyd_gpu_cuda$MAJOR.so PYTHONSAFEPATH=1
"$PY" -c "from glyd.gpu import _lib, kernels as g; g.lib(); print('library: C API', _lib._lib.glyd_gpu_api_version(), '| GPU code', _lib.gpu())" >> "$R/env.txt" 2>&1 || fail "the library does not load ($(tail -1 "$R/env.txt"))"
done_ "build: sm_$ARCH in $(( $(date +%s) - t )) s ($(tail -1 "$R/env.txt"))"

step "serve: $MODEL over $TP GPUs, a mode at a time: $MODES (waiting for its download)"
got "$MODEL" || fail "no $MODEL: $(tail -2 "$R/log/dl-$NAME.txt" 2> /dev/null | tr '\n' ' ')"
echo "download: $(tail -1 "$R/log/dl-$NAME.txt")"
cat > "$W/vllm-kvonly" <<'EOF'
#!/usr/bin/env bash
# the venv's vllm with its `bench` doing nothing: bench_serve.sh then starts a server, notes its KV cache, and stops it
[ "$1" = bench ] && exit 0
exec "$VLLM_REAL" "$@"
EOF
chmod +x "$W/vllm-kvonly"
serve_mode() {  # MODE UTIL DIR [kvonly]: bench_serve.sh for a mode at a gpu_memory_utilization, its results in DIR (kvonly: the server started and its KV cache noted, no benches); 0 where the server answered
  local mode=$1 u=$2 d=$3 kvonly=${4:-} MT left t e v tag
  case $mode in (glyd) MT=${VJ_EST_GLYD:-600} ;; (*) MT=${VJ_EST_BF16:-420} ;; esac
  tag="$mode at $u${kvonly:+, KV cache alone}"
  left=$(( END - $(el) ))
  if [ "$(el)" -ge "$BUDGET" ] || [ "$left" -lt "$MT" ]; then done_ "serve $tag: not run, $left s of the clock left (a start takes $MT s; none starts past $BUDGET s)"; return 1; fi
  t=$(date +%s)
  idle 120
  rm -rf "$W/cache/$mode"  # (each start on an empty compile cache)
  v=$W/venv/bin/vllm; [ -n "$kvonly" ] && v=$W/vllm-kvonly
  ( cd "$R" && HF_HUB_OFFLINE=1 VJ_SRV=$$ VLLM="$v" VLLM_REAL="$W/venv/bin/vllm" R="$d" VLLM_CACHE_ROOT="$W/cache/$mode" MODES="$mode" WARM=0 \
      RATES="$RATES" PROMPTS="$PROMPTS" IN="${VJ_IN:-1024}" OUT="${VJ_OUT:-256}" UTIL="$u" TP="$TP" BUSYWAIT=120 COOL=60 COOLWAIT=60 \
      SERVE_ARGS="$SERVE" \
      timeout "$(tmo "${VJ_MODE_MAX:-1800}")" bash "$W/src/gpu/vllm/bench_serve.sh" "$MODEL" ) >> "$d.txt" 2>&1
  e=$?
  python3 "$W/src/gpu/vllm/bench_summary.py" "$d" > "$d/summary.txt" 2>> "$R/log/summary-err.txt"  # (every mode so far, where a timeout cut the mode's own)
  idle 120
  h=; [ -z "$kvonly" ] && h=$(grep -ho "Prefix cache hit rate: [0-9.]*" "$d/serve-$mode.txt" 2> /dev/null | awk 'BEGIN { m = 0 } { if ($5 + 0 > m) m = $5 + 0 } END { if (NR) printf "%.1f", m }')
  done_ "serve $tag: exit $e in $(( $(date +%s) - t )) s; $(grep -h "GPU KV cache size" "$d/kv-$mode.txt" 2> /dev/null | head -1 | grep . || echo "the server did not start (${d#$R/}/serve-$mode.txt)")${h:+; prefix cache hit rate at most $h%}$(awk -v h="${h:-0}" 'BEGIN { if (h > 1) print "; NOT COMPARABLE" }')"
  grep -q "GPU KV cache size" "$d/kv-$mode.txt" 2> /dev/null
}
for mode in $MODES; do serve_mode "$mode" "$UTIL" "$B"; done
# bf16 had no room for the KV cache at UTIL (vLLM's refusal): at the retry's utilization Glyd's cold start, its KV cache alone, then bf16
# again, benched where it starts
if [ -n "$RETRY" ] && awk -v a="$RETRY" -v b="$UTIL" 'BEGIN { exit !(a > b) }' && echo " $MODES " | grep -q " bf16 " &&
   grep -qE "No available memory for the cache blocks|KV cache is needed|larger than the available KV cache" "$B/serve-bf16.txt" 2> /dev/null; then
  step "retry: bf16 had no room for the KV cache at $UTIL; at $RETRY, Glyd's KV cache alone, then bf16 (benched where it starts)"
  echo " $MODES " | grep -q " glyd " && serve_mode glyd "$RETRY" "$B2" kvonly
  serve_mode bf16 "$RETRY" "$B2"
fi
for f in "$R"/log/dl-*.txt; do [ -f "$f" ] && echo "download $(basename "$f" .txt): $(tail -1 "$f")"; done | grep -v "dl-env" >> "$R/steps.txt"
step "done in $(el) s"
done_ "done"
