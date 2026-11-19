"""moe28_job.sh's sweep tables: moe_layer.py's JSON files (the variants of the new library, the v027 library, vLLM's bf16 layer, the
decoded route's parts) as text.

    python3 sweep_summary.py table OUT.json...          the layer by tokens a step: bf16, v027, each variant, the plugin's pick
    python3 sweep_summary.py pick A.json B.json        the variables of the best variants, as KEY=V pairs on a line (decode, then big)
    python3 sweep_summary.py wgpick A.json B.json      the Hopper grouped products' settings: GLYD_GPU_MOE_WG (the tokens a step from which they beat the
                                                       others at every larger length measured) and GLYD_MOE_DECODE_MIN, as KEY=V pairs; nothing where they never do
    python3 sweep_summary.py dense table D.json        dense_layer.py's products by tokens a step: bf16 (cuBLAS), each variant
    python3 sweep_summary.py dense pick D.json         the dense small-M product's best setting as KEY=V (nothing where no setting is 2% under the default)
    python3 sweep_summary.py serve SUMMARY.txt         the serve's modes against bf16: requests a second, first token, each token, by rate
    python3 sweep_summary.py steps MODE=LOG ...        where the saturated pass's time went, from the servers' iteration lines (--enable-logging-iteration-details):
                                                       steps with prompt tokens and decode-only steps by sequences a step, their seconds and means
    python3 sweep_summary.py selftest
"""
import collections
import os
import json
import re
import sys

# each variant's variables, as the job's sweeps name them (the settings it sets; the others stay as they are)
VARS = {
    "new": {},
    "rows0": {"GLYD_GPU_MOE_ROWS": "0"},
    "na": {"GLYD_GPU_NA": "1"},
    "pf2": {"GLYD_GPU_PF": "2"},
    "pf4": {"GLYD_GPU_PF": "4"},
    "pf8": {"GLYD_GPU_PF": "8"},
    "pf4b": {"GLYD_GPU_PF": "4", "GLYD_GPU_PATCH_B": "1"},
    "pf4na": {"GLYD_GPU_PF": "4", "GLYD_GPU_NA": "1"},
    "pf8na": {"GLYD_GPU_PF": "8", "GLYD_GPU_NA": "1"},
    "pfd2": {"GLYD_GPU_PFD": "2"},
    "pfd4": {"GLYD_GPU_PFD": "4"},
    "pfd8": {"GLYD_GPU_PFD": "8"},
    "pfd4b": {"GLYD_GPU_PFD": "4", "GLYD_GPU_PATCH_B": "1"},
    "db": {"GLYD_GPU_PATCH_B": "1"},
    "x2": {"GLYD_GPU_XMAX": "2"},
    "x8": {"GLYD_GPU_XMAX": "8"},
    "slots132": {"GLYD_GPU_MOE_SLOTS": "132"},
    "slots528": {"GLYD_GPU_MOE_SLOTS": "528"},
    "big32": {"GLYD_GPU_MOE_BIG": "32"},
    "big96": {"GLYD_GPU_MOE_BIG": "96"},
    "xa": {"GLYD_GPU_BIG_XAHEAD": "1"},
    "bpf2": {"GLYD_GPU_BIG_PF": "2"},
    "xa_bpf2": {"GLYD_GPU_BIG_XAHEAD": "1", "GLYD_GPU_BIG_PF": "2"},
}


def load(paths):
    """{T: {variant: row}} over the files (a variant named in several: the last)."""
    out = collections.defaultdict(dict)
    for p in paths:
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        for r in d["rows"]:
            out[r["T"]][r["variant"]] = r
    return out


def table(paths):
    rows = load(paths)
    lines = []
    lines.append("| Tokens a step | bf16 (vLLM's layer) | v027 | new | best variant | best / bf16 | the plugin's decoded route* / bf16 |")
    lines.append("| ---: | ---: | ---: | ---: | :--- | ---: | ---: |")
    for T in sorted(rows):
        vs = rows[T]
        bf = next((r["bf16_us"] for r in vs.values() if r.get("bf16_us")), None)
        old = vs.get("v027")
        new = vs.get("new")
        cand = {n: r for n, r in vs.items() if n != "v027" and r.get("all_us")}
        if not cand:
            continue
        bn, br = min(cand.items(), key=lambda x: x[1]["all_us"])
        dec = (new or br)["route_us"] + (new or br)["unpack_us"] + bf if bf else None
        f = lambda v: f"{v:,.0f} us" if v else "-"
        lines.append(f"| {T:,} | {f(bf)} | {f(old['all_us']) if old else '-'} | {f(new['all_us']) if new else '-'} | {bn} {br['all_us']:,.0f} us | {br['all_us'] / bf:.2f}x |"
                     f" {dec / bf:.2f}x |" if bf else f"| {T:,} | - | {f(old['all_us']) if old else '-'} | {f(new['all_us']) if new else '-'} | {bn} {br['all_us']:,.0f} us | - | - |")
    lines.append("")
    lines.append("*route + the experts decoded + vLLM's own layer on them (the decoded route's GPU time, vLLM's kernels taken as the bf16 layer's).")
    lines.append("")
    lines.append("Each variant, us a layer (all = route, gate+up, down with the sum), against new:")
    names = sorted({n for vs in rows.values() for n in vs}, key=lambda n: (n not in ("v027", "new"), n))
    lines.append("| Tokens | " + " | ".join(names) + " |")
    lines.append("| ---: | " + " | ".join("---:" for _ in names) + " |")
    for T in sorted(rows):
        new = rows[T].get("new")
        cells = []
        for n in names:
            r = rows[T].get(n)
            cells.append("-" if not r or not r.get("all_us") else f"{r['all_us']:,.0f}" + (f" ({r['all_us'] / new['all_us']:.2f})" if new and new.get("all_us") and n != "new" else ""))
        lines.append(f"| {T:,} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("gate+up and down alone (us), by variant:")
    for key in ("gate_up_us", "down_us", "route_us"):
        lines.append(f"{key}:")
        lines.append("| Tokens | " + " | ".join(names) + " |")
        lines.append("| ---: | " + " | ".join("---:" for _ in names) + " |")
        for T in sorted(rows):
            lines.append(f"| {T:,} | " + " | ".join("-" if not rows[T].get(n) or key not in rows[T][n] else f"{rows[T][n][key]:,.0f}" for n in names) + " |")
        lines.append("")
    return "\n".join(lines)


def best(rows, names, Ts, key="all_us", margin=0.02):
    """The variant of names with the least `key` summed over Ts, `new` kept unless another is `margin` better."""
    tot = {}
    for n in names:
        v = [rows[T][n][key] for T in Ts if T in rows and n in rows[T] and key in rows[T][n]]
        if len(v) == len([T for T in Ts if T in rows]) and v:
            tot[n] = sum(v)
    if not tot:
        return "new"
    b = min(tot, key=tot.get)
    return b if "new" not in tot or tot[b] < tot["new"] * (1 - margin) else "new"


def pick(a_paths, b_paths):
    """KEY=V pairs for the best variants: the decode sweep's (the step loop's PF, PATCH_B, NA, ROWS, SLOTS, XMAX, by 8-512 tokens) and the big
    kernel's (BIG, XAHEAD, BIG_PF, by 768 tokens on). Variables a variant leaves unnamed stay as they are."""
    ra, rb = load(a_paths), load(b_paths)
    dec = best(ra, [n for n in ("new", "pf2", "pf4", "pf8", "pf4b", "pf4na", "pf8na", "na") if any(n in v for v in ra.values())], [8, 32, 64, 128, 232, 512])
    rw = best(ra, ["new", "rows0"], [32, 64, 128, 232, 512], key="down_us")
    sl = best(ra, [n for n in ("new", "slots132", "slots528") if any(n in v for v in ra.values())], [1, 8, 32])
    xm = best(ra, [n for n in ("new", "x2", "x8") if any(n in v for v in ra.values())], [8, 32, 64, 128, 232, 512])
    bg = best(rb, [n for n in ("new", "big32", "big96", "xa", "bpf2", "xa_bpf2") if any(n in v for v in rb.values())], [768, 1152, 2048, 4096])
    env = {}
    for n in (dec, sl, xm, bg):
        env.update(VARS.get(n, {}))
    if rw == "rows0":
        env.update(VARS["rows0"])
    return " ".join(f"{k}={v}" for k, v in sorted(env.items())), dict(decode=dec, down=rw, slots=sl, xmax=xm, big=bg)


def serve_modes(text):
    """The serve's summary text (a block `######## mode` and bench_summary.py's table for each) as {mode: {rate: (requests a second, first token ms, each token ms)}} (means)."""
    modes, cur = collections.OrderedDict(), None
    for line in text.splitlines():
        if line.startswith("######## "):
            cur = line[9:].strip()
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if cur and len(c) >= 7 and line.startswith("|") and re.fullmatch(r"[\d.]+|inf", c[0]):
            try:
                modes.setdefault(cur, {})[c[0]] = (float(c[2]), float(c[4].split("/")[0]), float(c[5].split("/")[0]))
            except ValueError:
                pass
    return modes


def serve_ratios(text, clocks=None):
    """The serve's summary text as a table of each mode's requests a second, first token and each token (ms, means) by rate, with the ratio to bf16's (requests a second: above 1 is faster;
    the times: below 1); clocks (pass_clocks): the SM clock each pass ran at (the median) and its highest temperature, beside them."""
    modes = serve_modes(text)
    if "bf16" not in modes:
        return "no bf16 row"
    head = "| Rate (req/s) | Mode | Requests/s | vs bf16 | First token (ms) | vs bf16 | Each token (ms) | vs bf16 |" + (" SM clock, median (MHz) | Highest temperature (C) |" if clocks else "")
    lines = [head, "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |" + (" ---: | ---: |" if clocks else "")]
    for rate in sorted({r for m in modes.values() for r in m}, key=lambda r: float("inf") if r == "inf" else float(r)):
        base = modes["bf16"].get(rate)
        for name, m in modes.items():
            if rate in m:
                q, f, t = m[rate]
                row = f"| {rate} | {name} | {q:.2f} | " + (f"{q / base[0]:.2f}x | {f:.0f} | {f / base[1]:.2f}x | {t:.1f} | {t / base[2]:.2f}x |" if base else "- | - | - | - | - |")
                if clocks:
                    v = clocks.get((name, rate))
                    row += f" {v[0]:.0f} | {v[2]:.0f} |" if v else " | |"
                lines.append(row)
    return "\n".join(lines)


STEP = re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)(?:\.(\d+))? \[[^\]]*\] (?:Engine \d+: )?Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, "
                  r"(\d+) generation tokens, iteration elapsed time: ([\d.]+) ms.*?KV cache usage: ([\d.]+)%")
def _saturated_rows(text, sat=None):
    """[[context tokens, generation requests, ms, KV usage %, iteration, seconds of the day or None]] of the saturated pass: from the first iteration of `sat` context tokens or more
    (default: 0.45 of the most any iteration of the last burst of lines has, at least 1,500: the pass of every request at once fills a step's budget, 8,192 tokens on an H100, 2,048 on
    an A100 or an L4, and the rate pass before it feeds a prompt of 1,024 at a time), to the end."""
    rows, day, last = [], 0, None
    for m in STEP.finditer(text):
        t = day * 86400 + int(m[3]) * 3600 + int(m[4]) * 60 + int(m[5]) + (int(m[6]) / 10 ** len(m[6]) if m[6] else 0.0)
        if last is not None and t < last - 1:  # (midnight)
            day, t = day + 1, t + 86400
        last = t
        rows.append([int(m[9]), int(m[10]), float(m[12]), float(m[13]), int(m[7]), t if m[6] else None])
    # (the passes of a server's log are bursts of lines, the idle of the bench's start-up and cool-down between them: the saturated pass is the last, where the lines have milliseconds.
    # A rate pass of a slow mode can have a step of two prompts at once, 2,048 tokens: the first step of that many is not the saturated pass's.)
    start = 0
    for i in range(1, len(rows)):
        if rows[i][5] is not None and rows[i - 1][5] is not None and rows[i][5] - rows[i - 1][5] > 3.0:
            start = i
    rows = rows[start:]
    if sat is None:
        sat = max(1500, 0.45 * max([r[0] for r in rows] or [0]))
    for i, r in enumerate(rows):
        if r[0] >= sat:
            return rows[i:]
    return []


def saturated_steps(text, sat=None):
    """[(context tokens, generation requests, ms, KV usage %)] of the saturated pass (_saturated_rows). ms: the time from the step's line to the next one's where the log has
    milliseconds (the servers of the job log them), else the engine's own wait for the step's result ('iteration elapsed time': under async scheduling it
    leaves out the part of the step the engine spent preparing the next one)."""
    rows = _saturated_rows(text, sat)
    for i in range(1, len(rows)):
        if rows[i][5] is not None and rows[i - 1][5] is not None and rows[i][4] == rows[i - 1][4] + 1:
            rows[i][2] = (rows[i][5] - rows[i - 1][5]) * 1e3
    return [tuple(r[:4]) for r in rows]


def saturated_window(text, sat=None):
    """(first, last) second of the day of the saturated pass's iteration lines, or None where the log has no milliseconds."""
    ts = [r[5] for r in _saturated_rows(text, sat) if r[5] is not None]
    return (ts[0], ts[-1]) if ts else None


def steps_table(logs, sat=None):
    """logs: {mode: log text}. For each mode the saturated pass's steps: those with prompt tokens, and the decode-only ones by the sequences a step
    (<64, 64-127, 128-191, 192-255, 256 and more): how many, their seconds in all, their mean ms, the milliseconds a token, and the pass's highest KV cache usage."""
    bins = [(0, 64, "<64"), (64, 128, "64-127"), (128, 192, "128-191"), (192, 256, "192-255"), (256, 10**9, "256+")]
    lines = ["| Mode | Steps | Seconds | Prompt steps | s | ms a step | us a token | Decode steps | s | ms a step | us a token | Decode steps' mean sequences | Highest KV usage |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    detail = []
    for mode, text in logs.items():
        rows = saturated_steps(text, sat)
        if not rows:
            lines.append(f"| {mode} | no saturated pass in the log | | | | | | | | | | | |")
            continue
        pre, dec = [r for r in rows if r[0] > 0], [r for r in rows if r[0] == 0]
        sp, sd = sum(r[2] for r in pre) / 1e3, sum(r[2] for r in dec) / 1e3
        mean_seqs = sum(r[1] * r[2] for r in dec) / max(1e-9, sum(r[2] for r in dec))
        ptok, dtok = sum(r[0] + r[1] for r in pre), sum(r[1] for r in dec)  # (a prompt step's tokens: its prompt tokens and its sequences' one each)
        lines.append(f"| {mode} | {len(rows)} | {sp + sd:.1f} | {len(pre)} | {sp:.1f} | {1e3 * sp / max(1, len(pre)):.0f} | {1e6 * sp / max(1, ptok):.0f} | {len(dec)} | {sd:.1f} | {1e3 * sd / max(1, len(dec)):.1f} | {1e6 * sd / max(1, dtok):.0f} | {mean_seqs:.0f} | {max(r[3] for r in rows):.0f}% |")
        for lo, hi, name in bins:
            sel = [r for r in dec if lo <= r[1] < hi]
            if sel:
                detail.append(f"| {mode} | decode steps of {name} sequences | {len(sel)} | {sum(r[2] for r in sel) / 1e3:.1f} s | {sum(r[2] for r in sel) / len(sel):.1f} ms a step |")
    out = "\n".join(lines)
    if detail:
        out += "\n\n| Mode | Steps | Count | Seconds | Mean |\n| :--- | :--- | ---: | ---: | ---: |\n" + "\n".join(detail)
    return out


def smi_samples(path):
    """[(second of the day, temperature C, SM clock MHz, power W)] of an nvidia-smi csv (timestamp, temperature, clock, power, ...)."""
    rows = []
    for line in open(path, errors="replace"):
        c = [x.strip() for x in line.split(",")]
        m = re.search(r"(\d\d):(\d\d):(\d\d)(?:\.(\d+))?\s*$", c[0]) if c else None
        try:
            rows.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + (int(m[4]) / 10 ** len(m[4]) if m[4] else 0.0), float(c[1]), float(c[2]), float(c[3])))
        except (TypeError, IndexError, ValueError):
            pass
    return rows


def pass_clocks(serve_dir):
    """{(mode, rate): (median SM clock MHz, lowest, highest temperature C, median power W, seconds)} of each pass of each mode under serve_dir: the saturated pass over the seconds of
    its server's iteration lines (its own window), a rate pass over its whole nvidia-smi csv (bench_serve.sh's smi-MODE-rateR.csv)."""
    import glob
    import statistics
    out = {}
    for f in sorted(glob.glob(os.path.join(serve_dir, "*", "smi-*-rate*.csv"))):
        mode = os.path.basename(os.path.dirname(f))
        rate = os.path.basename(f)[:-4].split("-rate")[-1]
        r = smi_samples(f)
        if rate == "inf":
            logs = glob.glob(os.path.join(serve_dir, mode, "serve-*.txt"))
            win = saturated_window(open(logs[0], errors="replace").read()) if logs else None
            if win:
                r = [x for x in r if win[0] <= x[0] <= win[1]] or r
        if r:
            clk = [x[2] for x in r]
            out[(mode, rate)] = (statistics.median(clk), min(clk), max(x[1] for x in r), statistics.median(x[3] for x in r), len(r))
    return out


def clock_floor():
    """The SM clock a saturated pass must hold: 86% of the GPU's top clock (M28_CLOCK_MAX, MHz: 1,703 of an H100's 1,980; 1,213 of an A100's 1,410), 1,700 where the top is not given."""
    try:
        top = float(os.environ.get("M28_CLOCK_MAX", ""))
    except ValueError:
        top = 0.0
    return float(round(0.86 * top)) if top > 0 else 1700.0


def comparable(rows, floor=None, spread=0.05, hot=85.0, drift=0.05):
    """(ok, [reasons]) for the saturated passes of a run, rows [(mode, median SM clock, highest temperature, requests a second)]: not comparable where a pass's median clock is under `floor`,
    the passes' medians differ by more than `spread` of the highest, a pass was over `hot` C, or bf16's two passes (a mode named bf16 and one named bf16-2) differ by more than `drift` in
    requests a second. The clock a GPU runs at depends on its temperature and power, so passes at other clocks measure the clocks."""
    why = []
    floor = clock_floor() if floor is None else floor
    for mode, clk, temp, _ in rows:
        if clk < floor:
            why.append(f"{mode}: median SM clock {clk:.0f} MHz, under {floor:.0f}")
        if temp > hot:
            why.append(f"{mode}: {temp:.0f} C, over {hot:.0f}")
    clocks = [r[1] for r in rows]
    if len(clocks) > 1 and max(clocks) > 0 and (max(clocks) - min(clocks)) / max(clocks) > spread:
        why.append(f"the passes' median clocks differ by {100 * (max(clocks) - min(clocks)) / max(clocks):.0f}% ({min(clocks):.0f} to {max(clocks):.0f} MHz), over {100 * spread:.0f}%")
    rps = {m: q for m, _, _, q in rows}
    if "bf16" in rps and "bf16-2" in rps and max(rps["bf16"], rps["bf16-2"]) > 0:
        d = abs(rps["bf16"] - rps["bf16-2"]) / max(rps["bf16"], rps["bf16-2"])
        if d > drift:
            why.append(f"bf16's two passes differ by {100 * d:.0f}% in requests a second ({rps['bf16']:.2f} and {rps['bf16-2']:.2f}): the machine drifted during the run")
    return not why, why


def clock_report(serve_dir, rps):
    """The banner and the table of the clocks: rps {mode: requests a second of its saturated pass}. COMPARABLE or NOT COMPARABLE with the reasons, then each saturated pass's clocks."""
    pc = pass_clocks(serve_dir)
    rows = [(m, v[0], v[2], rps.get(m, float("nan"))) for (m, r), v in sorted(pc.items()) if r == "inf"]
    if not rows:
        return "no saturated pass with an nvidia-smi csv"
    ok, why = comparable(rows)
    lines = [("COMPARABLE: " + "; ".join(f"{m} {c:.0f} MHz, {t:.0f} C" for m, c, t, _ in rows)) if ok else "NOT COMPARABLE: " + "; ".join(why),
             "", "| Mode | Pass | Requests/s | SM clock median (MHz) | lowest | Highest temperature (C) | Power median (W) | Seconds |", "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (m, r), v in sorted(pc.items()):
        q = rps.get(m) if r == "inf" else None
        lines.append(f"| {m} | {r} | {'' if q is None else f'{q:.2f}'} | {v[0]:.0f} | {v[1]:.0f} | {v[2]:.0f} | {v[3]:.0f} | {v[4]} |")
    import glob
    for f in sorted(glob.glob(os.path.join(serve_dir, "*", "smi-reasons.csv"))):  # (a mode's whole run: the slowdown reasons the driver gave, a second each)
        n, total = {"hardware thermal slowdown": 0, "software thermal slowdown": 0, "software power cap": 0, "hardware power brake": 0}, 0
        for line in open(f, errors="replace"):
            c = [x.strip() for x in line.split(",")]
            if len(c) >= 8:
                total += 1
                for k, v in zip(n, c[4:8]):
                    n[k] += v == "Active"
        lines.append(f"{os.path.basename(os.path.dirname(f))}: the driver's slowdown reasons, seconds active of {total}: " + ", ".join(f"{k} {v}" for k, v in n.items()))
    return "\n".join(lines)


def preflight(path, hot=84.0):
    """(throttled, text) for a stress run's nvidia-smi samples (timestamp, temperature, SM clock, power, hardware and software thermal slowdown, software power cap a second): throttled where the
    driver gave a thermal slowdown in the last 10 s of it or the GPU was at `hot` C or more. A power cap is not that: a GPU at its power limit runs under its highest clock on any machine."""
    rows = []
    for line in open(path, errors="replace"):
        c = [x.strip() for x in line.split(",")]
        try:
            rows.append((float(c[1]), float(c[2]), float(c[3]), c[4] == "Active" or c[5] == "Active", c[6] == "Active"))
        except (IndexError, ValueError):
            pass
    if len(rows) < 5:
        return False, f"{len(rows)} samples: no verdict"
    tail = rows[-10:]
    peak, clk, power = max(r[0] for r in tail), sum(r[1] for r in tail) / len(tail), sum(r[2] for r in tail) / len(tail)
    thermal, cap = sum(r[3] for r in tail), sum(r[4] for r in tail)
    bad = thermal > 0 or peak >= hot
    return bad, f"{len(rows)} s of load: the last 10 s {peak:.0f} C at most, SM clock {clk:.0f} MHz, {power:.0f} W, thermal slowdown in {thermal} of them, power cap in {cap}"


def dense_rows(path):
    """{(O, K, M): {variant: row}} of dense_layer.py's JSON."""
    out = collections.defaultdict(dict)
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return out
    for r in d["rows"]:
        out[(r["O"], r["K"], r["T"])][r["variant"]] = r
    return out


def dense_table(path):
    rows = dense_rows(path)
    names = sorted({n for vs in rows.values() for n in vs}, key=lambda n: (n != "new", n))
    lines = ["| Matrix (O x K) | Tokens a step | bf16 (cuBLAS) | " + " | ".join(names) + " | best / bf16 |", "| :--- | ---: | ---: | " + " | ".join("---:" for _ in names) + " | ---: |"]
    for key in sorted(rows):
        vs = rows[key]
        bf = next(iter(vs.values()))["bf16_us"]
        cells = [f"{vs[n]['glyd_us']:,.1f} ({vs[n]['glyd_us'] / bf:.2f})" if n in vs else "-" for n in names]
        best = min(v["glyd_us"] for v in vs.values())
        lines.append(f"| {key[0]} x {key[1]} | {key[2]} | {bf:,.1f} us | " + " | ".join(cells) + f" | {best / bf:.2f}x |")
    return "\n".join(lines)


def dense_pick(path, margin=0.02, upto=16):
    """The dense small-M product's best setting (the variants of dense_layer.py by the sum of their times over the lengths up to `upto`, every
    matrix): KEY=V pairs, or "" where none is `margin` under the default (the variant named new)."""
    rows = dense_rows(path)
    tot = collections.defaultdict(float)
    count = collections.Counter()
    for (O, K, M), vs in rows.items():
        if M > upto:
            continue
        for n, r in vs.items():
            tot[n] += r["glyd_us"]
            count[n] += 1
    full = [n for n in tot if count[n] == max(count.values())]
    if "new" not in full:
        return ""
    best = min(full, key=lambda n: tot[n])
    if best != "new" and tot[best] < tot["new"] * (1 - margin):
        return " ".join(f"{k}={v}" for k, v in sorted(VARS.get(best, {}).items()))
    return ""


def wg_pick(ra, rb, margin=0.02):
    """(wg_min, decode_min, why): the grouped Hopper products (the variants named wg...) against the library's other grouped variants at each length
    measured. wg_min: the least T from which the best wg variant is `margin` under the best other at every length from it on (None: never, not even
    at the largest). decode_min: with them on, the least T (1,024 on) from which the decoded route (route + unpack + vLLM's layer) is `margin` under
    them at every length from it on, else -1 (never)."""
    rows = collections.defaultdict(dict)
    for r in (ra, rb):
        for T, vs in r.items():
            rows[T].update(vs)
    Ts = sorted(rows)
    wg, alt, dec = {}, {}, {}
    for T in Ts:
        vs = rows[T]
        w = [v["all_us"] for n, v in vs.items() if n.startswith("wg") and v.get("all_us")]
        o = [v["all_us"] for n, v in vs.items() if not n.startswith("wg") and n != "v027" and v.get("all_us")]
        bf = next((v["bf16_us"] for v in vs.values() if v.get("bf16_us")), None)
        any_row = next((v for n, v in vs.items() if not n.startswith("wg") and "route_us" in v and "unpack_us" in v), None)
        if w and o:
            wg[T], alt[T] = min(w), min(o)
            if bf and any_row and T >= 1024:
                dec[T] = any_row["route_us"] + any_row["unpack_us"] + bf
    if not wg:
        return None, None, "no wg rows"
    wg_min = None
    for T in sorted(wg, reverse=True):
        if wg[T] < alt[T] * (1 - margin):
            wg_min = T
        else:
            break
    if wg_min is None:
        return None, None, "the grouped products are not under the others at the largest length"
    d_min = -1
    for T in sorted(wg, reverse=True):
        if T in dec and dec[T] < wg[T] * (1 - margin):
            d_min = T
        else:
            break
    return wg_min, d_min, f"wg from {wg_min} tokens; decoded route {'from ' + str(d_min) if d_min > 0 else 'never'}"


def selftest():
    """The tables' own checks, whatever M28_CLOCK_MAX the job has set (the floor follows it)."""
    was = os.environ.pop("M28_CLOCK_MAX", None)
    try:
        _selftest()
    finally:
        if was is not None:
            os.environ["M28_CLOCK_MAX"] = was


def _selftest():
    def mk(vs):
        return {"rows": [dict(T=T, variant=n, all_us=a, down_us=a / 2, gate_up_us=a / 2, route_us=5, unpack_us=500, bf16_us=400) for T, n, a in vs]}
    import tempfile
    d = tempfile.mkdtemp()
    a = mk([(8, "new", 100), (8, "pf4", 80), (32, "new", 200), (32, "pf4", 150), (64, "new", 300), (64, "pf4", 250), (128, "new", 400), (128, "pf4", 300), (232, "new", 500), (232, "pf4", 400), (512, "new", 700), (512, "pf4", 600)])
    b = mk([(768, "new", 900), (768, "xa", 905), (1152, "new", 1500), (1152, "xa", 1490), (2048, "new", 2000), (2048, "xa", 2010), (4096, "new", 3000), (4096, "xa", 3010)])
    json.dump(a, open(d + "/a.json", "w"))
    json.dump(b, open(d + "/b.json", "w"))
    env, why = pick([d + "/a.json"], [d + "/b.json"])
    assert env == "GLYD_GPU_PF=4" and why["decode"] == "pf4" and why["big"] == "new", (env, why)
    assert "bf16" in table([d + "/a.json", d + "/b.json"])
    # the grouped Hopper products: slower to 64 tokens, then the faster to the end; the decoded route the faster only at 8,192
    c = mk([(32, "new", 200), (32, "wg", 260), (64, "new", 300), (64, "wg", 310), (128, "new", 400), (128, "wg", 300), (512, "new", 700), (512, "wg", 500)])
    e = mk([(768, "new", 900), (768, "wg", 800), (2048, "new", 2000), (2048, "wg", 1500), (8192, "new", 8000), (8192, "wg", 3000)])
    for r in e["rows"]:
        r["unpack_us"], r["route_us"], r["bf16_us"] = (600, 100, 1700) if r["T"] == 8192 else (5000, 100, 1700)  # the decoded route 2,400 against wg 3,000 at 8,192 and no match before
    json.dump(c, open(d + "/c.json", "w"))
    json.dump(e, open(d + "/e.json", "w"))
    wmin, dmin, why = wg_pick(load([d + "/c.json"]), load([d + "/e.json"]))
    assert (wmin, dmin) == (128, 8192), (wmin, dmin, why)
    assert wg_pick(load([d + "/a.json"]), load([d + "/b.json"])) == (None, None, "no wg rows")
    dn = {"rows": [dict(O=5120, K=2048, T=M, variant=n, glyd_us=u, bf16_us=10.0) for M, n, u in [(1, "new", 12), (1, "pfd4", 10), (8, "new", 14), (8, "pfd4", 12), (32, "new", 20), (32, "pfd4", 30)]]}
    json.dump(dn, open(d + "/dn.json", "w"))
    assert dense_pick(d + "/dn.json") == "GLYD_GPU_PFD=4", dense_pick(d + "/dn.json")
    assert "5120 x 2048" in dense_table(d + "/dn.json")
    def lines(rows, ms):  # (the rows' time of day: the third field, seconds)
        return "".join(f"(APIServer pid=1) INFO 10-03 12:00:{int(t):02d}{('.%03d' % round((t % 1) * 1000)) if ms else ''} [loggers.py:185] Engine 000: Iteration({i}): {c} context requests, {n} context tokens, {g} generation requests, {g} generation tokens, "
                       f"iteration elapsed time: {w:.2f} ms, GPU KV cache usage: {u}%\n" for i, (c, n, g, w, u, t) in enumerate(rows))
    rows = [(1, 1024, 3, 30.0, 1, 1.0), (8, 8192, 0, 0.1, 20, 10.0), (5, 4000, 100, 0.1, 40, 10.150), (0, 0, 200, 0.1, 70, 10.190), (0, 0, 20, 0.1, 60, 10.202)]
    st = steps_table({"bf16": lines(rows, True)}, 4096)
    assert "| bf16 | 4 |" in st and "Prompt steps" in st and "decode steps of <64" in st.replace("Decode", "decode"), st
    assert "40.0 ms a step" in st and "12.0 ms a step" in st and "| 75 |" in st, st  # the steps' times from the lines' own times (150, 40 and 12 ms; the pass's first step has its engine's wait, 0.1)
    assert saturated_steps(lines(rows, False), 4096)[1][2] == 0.1  # (seconds only: the engine's own wait)
    sv = "######## bf16\n| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n| inf | bf16 | 12.03 | 3000.0 | 6737 / 9000 | 28.6 / 30.0 | 256 | 1980 MHz, 60 C |\n\n######## glyd-new\n| inf | glyd | 10.07 | 2500.0 | 4248 / 6000 | 64.5 / 70.0 | 256 | 1980 MHz, 60 C |\n"
    r = serve_ratios(sv)
    assert "0.84x" in r and "2.26x" in r, r
    pf = tempfile.mkdtemp() + "/pf.csv"
    open(pf, "w").write("".join(f"2026/10/03 18:00:{i:02d}.000, {60 + i}, {1980 - 30 * i}, 600, {'Active' if i > 20 else 'Not Active'}, Not Active, Active, Not Active\n" for i in range(25)))
    bad, txt = preflight(pf)
    assert bad and "thermal slowdown in 4 of them" in txt, txt  # (a thermal slowdown in the last 10 s)
    open(pf, "w").write("".join(f"2026/10/03 18:00:{i:02d}.000, 62, 1750, 690, Not Active, Not Active, Active, Not Active\n" for i in range(25)))
    assert not preflight(pf)[0], "a power cap alone is not throttling"
    sd = tempfile.mkdtemp()
    for mode, clk in (("bf16", 1980), ("glyd-picked", 1500), ("bf16-2", 1975)):  # (iteration lines with milliseconds, a pass's clocks sampled a second around them)
        os.makedirs(f"{sd}/{mode}")
        lines = "".join(f"(APIServer pid=1) INFO 10-03 18:00:{10 + i:02d}.000 [loggers.py:185] Engine 000: Iteration({i}): {c} context requests, {n} context tokens, 5 generation requests, 5 generation tokens, iteration elapsed time: 1.00 ms, GPU KV cache usage: 5%\n"
                        for i, (c, n) in enumerate([(1, 1024), (8, 8192), (0, 0), (0, 0), (0, 0), (0, 0)]))
        open(f"{sd}/{mode}/serve-{'glyd' if mode == 'glyd-picked' else 'bf16'}.txt", "w").write(lines)
        fname = f"{sd}/{mode}/smi-{'glyd' if mode == 'glyd-picked' else 'bf16'}-rateinf.csv"
        open(fname, "w").write("".join(f"2026/10/03 18:00:{8 + i:02d}.500, 70, {clk if 11 <= 8 + i <= 15 else 345}, 400\n" for i in range(10)))
    pcs = pass_clocks(sd)
    assert pcs[("bf16", "inf")][0] == 1980 and pcs[("glyd-picked", "inf")][0] == 1500, pcs  # (the saturated pass's seconds alone: not the idle ones around it)
    ok, why = comparable([("bf16", 1980, 70, 12.0), ("glyd-picked", 1500, 70, 11.0)])
    assert not ok and any("under 1700" in w for w in why) and any("differ by 24%" in w for w in why), why
    assert comparable([("bf16", 1980, 70, 12.0), ("glyd-picked", 1960, 80, 11.0), ("bf16-2", 1975, 71, 11.9)])[0]
    assert not comparable([("bf16", 1980, 70, 12.0), ("bf16-2", 1975, 71, 10.0)])[0] and not comparable([("bf16", 1980, 86, 12.0)])[0]
    was = os.environ.get("M28_CLOCK_MAX")
    try:
        os.environ["M28_CLOCK_MAX"] = "1410"  # (an A100: a pass at 1,300 MHz holds its clock, one at 1,100 does not)
        assert clock_floor() == 1213.0 and comparable([("bf16", 1300, 60, 5.0), ("glyd-picked", 1290, 61, 6.0), ("bf16-2", 1295, 60, 5.1)])[0]
        ok, why = comparable([("bf16", 1300, 60, 5.0), ("glyd-picked", 1100, 61, 6.0)])
        assert not ok and any("under 1213" in w for w in why), why
        os.environ["M28_CLOCK_MAX"] = "x"
        assert clock_floor() == 1700.0
    finally:
        os.environ.pop("M28_CLOCK_MAX", None)
        if was is not None:
            os.environ["M28_CLOCK_MAX"] = was
    rep = clock_report(sd, {"bf16": 12.0, "glyd-picked": 11.0, "bf16-2": 11.9})
    assert rep.startswith("NOT COMPARABLE") and "| bf16-2 | inf | 11.90 |" in rep, rep
    sv = "######## bf16\n| inf | bf16 | 12.03 | 3000.0 | 6737 / 9000 | 28.6 / 30.0 | 256 | 1980 MHz, 60 C |\n\n######## glyd-picked\n| inf | glyd | 10.07 | 2500.0 | 4248 / 6000 | 64.5 / 70.0 | 256 | 1980 MHz, 60 C |\n"
    assert "| 1980 | 70 |" in serve_ratios(sv, {("bf16", "inf"): (1980, 1980, 70, 400, 5)}), serve_ratios(sv, {("bf16", "inf"): (1980, 1980, 70, 400, 5)})
    two = "".join(f"(APIServer pid=1) INFO 10-03 18:00:{i // 20:02d}.{(i % 20) * 50:03d} [loggers.py:185] Engine 000: Iteration({i}): {c} context requests, {n} context tokens, 3 generation requests, 3 generation tokens, iteration elapsed time: 0.02 ms, KV cache usage: 5.0%\n"
                  for i, (c, n) in enumerate([(1, 1024), (2, 2046), (0, 0), (0, 0)]))  # (a rate pass: one step of two prompts)
    two += "".join(f"(APIServer pid=1) INFO 10-03 18:00:{30 + i // 20:02d}.{(i % 20) * 50:03d} [loggers.py:185] Engine 000: Iteration({10 + i}): {c} context requests, {n} context tokens, 9 generation requests, 9 generation tokens, iteration elapsed time: 0.02 ms, KV cache usage: 50.0%\n"
                   for i, (c, n) in enumerate([(2, 2048), (2, 2048), (0, 0), (0, 0), (0, 0)]))  # (the saturated pass, 28 s later)
    assert len(saturated_steps(two)) == 5, saturated_steps(two)
    print("sweep_summary: self-test passed")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "table":
        print(table(sys.argv[2:]))
    elif cmd == "steps":
        print(steps_table({a.split("=", 1)[0]: open(a.split("=", 1)[1], errors="replace").read() for a in sys.argv[2:]}))
    elif cmd == "preflight":
        bad, text = preflight(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 84.0)
        print(("THROTTLED: " if bad else "ok: ") + text)
        sys.exit(1 if bad else 0)
    elif cmd == "serve":
        print(serve_ratios(open(sys.argv[2]).read(), pass_clocks(sys.argv[3]) if len(sys.argv) > 3 else None))
    elif cmd == "clocks":  # SERVE_DIR SERVE_SUMMARY: the banner (COMPARABLE or not) and the table of the clocks
        print(clock_report(sys.argv[2], {m: v["inf"][0] for m, v in serve_modes(open(sys.argv[3]).read()).items() if "inf" in v}))
    elif cmd == "dense":
        print(dense_table(sys.argv[3]) if sys.argv[2] == "table" else dense_pick(sys.argv[3]))
    elif cmd == "wgpick":
        wmin, dmin, why = wg_pick(load(sys.argv[2].split(",")), load(sys.argv[3].split(",")))
        print(f"GLYD_GPU_MOE_WG={wmin} GLYD_MOE_DECODE_MIN={dmin}" if wmin else "")
        print("wgpick:", why, file=sys.stderr)
    elif cmd == "pick":
        a, b = sys.argv[2].split(","), sys.argv[3].split(",")
        env, why = pick(a, b)
        print(env)
        print("picked:", why, file=sys.stderr)
    elif cmd == "selftest":
        selftest()
