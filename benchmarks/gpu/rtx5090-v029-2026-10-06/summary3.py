"""The speed bar's tables from a job's results: per model, each number of users, vLLM bf16 against Glyd (requests/s, output tokens/s,
first-token and per-token latency p50/p99), which is faster, the prefix cache hit rate, and the memory each server used.
    python3 speed_bar_summary.py RESULTS_DIR"""
import glob
import json
import os
import re
import statistics
import sys

COLS = [("request_throughput", "req/s", 1), ("output_throughput", "out tok/s", 1),
        ("median_ttft_ms", "TTFT p50 ms", -1), ("p99_ttft_ms", "TTFT p99 ms", -1),
        ("median_tpot_ms", "TPOT p50 ms", -1), ("p99_tpot_ms", "TPOT p99 ms", -1)]


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def server_logs(d):
    return sorted(glob.glob(f"{d}/serve-try*.txt") + glob.glob(f"{d}/state-try*/glyd/logs/*.log"))


def vllm_lines(d, started):
    """What the started server's vLLM log says of its weights and KV cache."""
    if not started.startswith("try "):
        return ""
    k = started.split()[1].rstrip(":")
    text = read(f"{d}/serve-try{k}.txt") + "".join(read(p) for p in glob.glob(f"{d}/state-try{k}/glyd/logs/*.log"))
    out = []
    for pat in (r"Model loading took [0-9.]+ GiB", r"GPU KV cache size: [0-9,]+ tokens", r"Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x"):
        m = re.findall(pat, text)
        if m:
            out.append(m[-1])
    hit = [float(x) for x in re.findall(r"Prefix cache hit rate: ([0-9.]+)%", text)]
    if hit:
        out.append(f"log's highest prefix cache hit rate {max(hit):.1f}%")
    return "; ".join(out)


def levels(d):
    """{users: (the bench JSON, prefix hit rate %, peak MiB, median SM MHz)}."""
    prefix = {}
    for line in read(f"{d}/prefix.txt").splitlines():
        p = line.split()
        if len(p) == 5:
            n, q0, h0, q1, h1 = map(int, p)
            prefix[n] = 100.0 * (h1 - h0) / (q1 - q0) if q1 > q0 else None
    out = {}
    for f in glob.glob(f"{d}/c*.json"):
        n = int(re.search(r"c(\d+)\.json$", f).group(1))
        mem, clk = [], []
        for row in read(f"{d}/smi-c{n}.csv").splitlines():
            p = [x.strip() for x in row.split(",")]
            try:
                mem.append(float(p[1]))
                clk.append(float(p[2]))
            except (IndexError, ValueError):
                pass
        out[n] = (json.load(open(f)), prefix.get(n), max(mem) if mem else None, statistics.median(clk) if clk else None)
    return out


def fmt(v, nd=1):
    return "-" if v is None else f"{v:,.{nd}f}"


KINDS = (("bf16", "vLLM bf16"), ("glyd", "Glyd (default layout)"), ("glydmma", "Glyd, tiered (GLYD_LAYOUT=mma)"))


def model_table(tag, root):
    print(f"## {tag}\n")
    lv = {}
    for k, name in KINDS:
        d = f"{root}/{tag}/{k}"
        if not os.path.isdir(d):
            continue
        s = read(f"{d}/started.txt").strip() or "not run"
        lay = sorted(set(re.findall(r"glyd: (mma12|mma) layout", "".join(read(p) for p in server_logs(d)))))
        print(f"- {name}: started {s}; {vllm_lines(d, s)}{'; layout ' + '/'.join(lay) if lay else ''}; nvidia-smi: {' / '.join(read(f'{d}/memory.txt').split(chr(10))).strip(' /')}")
        lv[k] = levels(d)
    lb = lv.get("bf16", {})
    print()
    print("| Users | Server | " + " | ".join(c[1] for c in COLS) + " | Prefix hit % | Peak MiB | SM MHz | Glyd against bf16 |")
    print("| ---: | :--- | " + " | ".join("---:" for _ in COLS) + " | ---: | ---: | ---: | :--- |")
    for n in sorted(set().union(*lv.values()) if lv else []):
        for k, name in KINDS:
            if k not in lv:
                continue
            levs = lv[k]
            if n not in levs:
                print(f"| {n} | {name} | " + " | ".join("-" for _ in COLS) + " | - | - | - | |")
                continue
            j, hit, mem, clk = levs[n]
            verdict = ""
            if k != "bf16" and n in lb:
                jb = lb[n][0]
                slower = [c[1] for c in COLS if (j[c[0]] - jb[c[0]]) * c[2] < 0]
                verdict = "faster or equal on all" if not slower else "slower: " + ", ".join(slower)
            print(f"| {n} | {k} | " + " | ".join(fmt(j.get(c[0]), 2 if c[0] == "request_throughput" else 1) for c in COLS)
                  + f" | {fmt(hit)} | {fmt(mem, 0)} | {fmt(clk, 0)} | {verdict} |")
    print()


def main(root):
    print(read(f"{root}/machine.txt").strip())
    print(read(f"{root}/env.txt").strip(), "\n")
    for tag in sorted(os.listdir(root)):
        if os.path.isdir(f"{root}/{tag}") and (any(os.path.isdir(f"{root}/{tag}/{k}") for k, _ in KINDS)):
            model_table(tag, root)


if __name__ == "__main__":
    main(sys.argv[1])
