#!/usr/bin/env python3
"""Run the same tests against any OpenAI-compatible server and print one block to paste: speed on a short, a ~4K and a ~30K prompt,
measured here from the stream (time to the first token, then tokens a second), a GSM8K + MMLU-Pro subset with the settings of
results/v0.1.0 (greedy, thinking off, at most 3,072 tokens an answer), and the machine.

Run it against this project's server and against whatever else you use, on the same computer, one server at a time:
    python3 bench/compare.py --url http://127.0.0.1:8080 --model qwen3.8-flash-next --note "precise mode"
    python3 bench/compare.py --url http://127.0.0.1:8000 --model OTHER-NAME --note "what it is, its settings"
--speed-only skips the quality part (which takes about an hour and makes its own Python environment for Inspect AI, as
bench/run_bench.sh does). --questions N (a multiple of 25, at most 150) runs the first N of the 150 questions of each.
The speed part needs nothing but Python 3."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VERSION = "compare-v1"   # the prompt set and settings; change the name when either changes
HERE = Path(__file__).resolve().parent


def filler(label, lines):
    """A long prompt made the same way every time: numbered notes with made-up values, a different label at the top of each prompt
    (so no prompt is the start of another and a server's prompt cache can't reuse one for the next), a question at the end."""
    colors = ["red", "blue", "green", "amber", "violet", "gray", "white", "black"]
    things = ["box", "jar", "crate", "bag", "tin", "basket", "drawer", "shelf"]
    body = "\n".join(f"Note {i}: the {colors[i * 7 % 8]} {things[i * 5 % 8]} in room {i * 37 % 101} holds {i * 53 % 997} marbles." for i in range(1, lines + 1))
    return (f"Inventory list {label}. Read it, then answer the question at the end.\n\n{body}\n\n"
            f"Question: how many marbles does note {lines // 2} say, and in which room? Then write a 150-word summary of what kinds of notes the list has.")


TESTS = [  # name, prompt, max_tokens
    ("Chat (short prompt)", "Write a 400-word story about a lighthouse keeper who finds a message in a bottle.", 512),
    ("~4K prompt", filler("A", 178), 256),
    ("~30K prompt", filler("B", 1290), 256),   # (about 30,000 tokens with Qwen's tokenizer: fits a server started with a 32,768 context)
]


def stream(url, model, prompt, max_tokens):
    """One streamed chat request, timed here: seconds to the first token (thinking or answer), tokens after it, the server's token counts."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                                           "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")})
    t0, first, last, chunks, usage = time.time(), None, None, 0, {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
    made = usage.get("completion_tokens") or chunks
    ptoks = usage.get("prompt_tokens")
    ttft = (first or time.time()) - t0
    write = (made - 1) / (last - first) if first and last and last > first and made > 1 else None
    return {"prompt_tokens": ptoks, "ttft": ttft, "read": ptoks / ttft if ptoks else None, "write": write, "made": made, "counted": "server" if usage.get("completion_tokens") else "chunks"}


def machine():
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    q = lambda f: run("nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader").splitlines()[:1] or ["?"]
    cpu = next((l.split(":", 1)[1].strip() for l in run("lscpu").splitlines() if l.startswith("Model name")), platform.processor() or "?")
    ram = next((f"{int(l.split()[1]) / 2**20:.0f} GiB" for l in open("/proc/meminfo") if l.startswith("MemTotal")), "?") if os.path.exists("/proc/meminfo") else "?"
    osname = next((l.split("=", 1)[1].strip().strip('"') for l in open("/etc/os-release") if l.startswith("PRETTY_NAME")), "") if os.path.exists("/etc/os-release") else platform.platform()
    return [f"GPU:     {q('name,memory.total')[0]}", f"Driver:  {q('driver_version')[0]}",
            f"PCIe:    now gen {q('pcie.link.gen.current')[0]} x{q('pcie.link.width.current')[0]}, max gen {q('pcie.link.gen.max')[0]} x{q('pcie.link.width.max')[0]}",
            f"CPU:     {cpu}", f"RAM:     {ram}", f"OS:      {osname}, kernel {platform.release()}"]


def quality(url, model, n, out):
    """Inspect AI's gsm8k and mmlu_pro through bench/eval_slice.py (slices of 25) and bench/merge_slices.py, as bench/run_bench.sh runs them."""
    venv = Path(os.environ.get("COMPARE_VENV", Path.home() / ".cache" / "compare-venv-eval"))
    py = venv / "bin" / "python"
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(venv / "bin" / "pip"), "install", "-q", "inspect-ai==0.3.278", "inspect-evals==0.24.0", "openai==3.28.0"], check=True)
    lines = []
    for task in ["gsm8k", "mmlu_pro"]:
        dirs = []
        for s in range(0, n, 25):
            d = out / f"{task}-{s}"
            subprocess.run([str(py), str(HERE / "eval_slice.py"), url + "/v1", model, task, "1/2", "150", str(s), str(s + 25), str(d)], check=True,
                           env={**os.environ, "GLYD_API_KEY": os.environ.get("OPENAI_API_KEY", "none")})
            dirs.append(str(d))
        subprocess.run([str(py), str(HERE / "merge_slices.py"), str(out / f"{task}.json")] + dirs, check=True, capture_output=True)   # (the venv's Python: Inspect AI reads its own logs)
        r = json.load(open(out / f"{task}.json"))[task]
        lines.append(f"{'GSM8K' if task == 'gsm8k' else 'MMLU-Pro':9} {r['correct']} / {r['answered']}   ({r['hit_max_tokens']} hit the 3,072 cap, mean {r['mean_output_tokens']:.0f} tokens an answer)")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="the server, e.g. http://127.0.0.1:8080 (without /v1)")
    ap.add_argument("--model", required=True, help="the model name the server wants")
    ap.add_argument("--note", default="", help="what the server is and how it was started (quant, context, flags): printed in the block")
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--out", default="compare-out")
    a = ap.parse_args()
    url = a.url.rstrip("/").removesuffix("/v1")
    if a.questions % 25 or not 25 <= a.questions <= 150:
        sys.exit("--questions must be 25, 50, ... 150")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("warm-up request ...", flush=True)
    stream(url, a.model, "Say hello.", 16)
    rows = []
    for name, prompt, mx in TESTS:
        r = stream(url, a.model, prompt, mx)
        f = lambda x, p=1: "?" if x is None else f"{x:.{p}f}"
        rows.append(f"| {name} | {r['prompt_tokens'] or '?'} | {f(r['ttft'], 2)} | {f(r['read'], 0)} | {f(r['write'])} | {r['made']} |")
        print(rows[-1], flush=True)
    qual = [] if a.speed_only else quality(url, a.model, a.questions, out)
    block = ["```", f"compare.py {VERSION}, {time.strftime('%Y-%m-%d %H:%M %Z')}", f"Server:  {url}, model {a.model}", f"Note:    {a.note or '(none given)'}"]
    block += machine()
    block += ["| Test | Prompt tokens | First token (s) | Prompt read (tokens/s) | Writing (tokens/s) | Tokens written |", "|---|--:|--:|--:|--:|--:|"] + rows
    block += qual or ["Quality: not run (--speed-only)"]
    block += ["```"]
    text = "\n".join(block)
    (out / "compare.md").write_text(text + "\n")
    print("\n---- paste from here ----\n" + text + "\n---- to here ----")


if __name__ == "__main__":
    main()
