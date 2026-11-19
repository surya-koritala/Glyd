#!/usr/bin/env python3
"""The record of the few-users runs: python record_users.py FETCHED_DIR OUT_DIR

FETCHED_DIR holds what the runs left on the box (copied by hand, as the files lie under ~/engine-out there):
  conc/RUN/   engine_conc_bench.sh's results, one directory a run: engine1, engine2, kvglyd1, kvglyd2 (the first measurement) and engineA, kvglydA, kvglydB, engineB (the alternated runs of the final build)
  p3/         the final build's checks (the lossless cache's proofs as `p3/NAME.out`; `p3/multi/`: clients at once, the API's smoke and soak, the layout default, a step's profile, the seeded draws)
OUT_DIR gets the README, the results' files (the vllm bench serve JSONs and prints, the hit rates, the engine's counts, the GPU's temperature and clock) and the checks' outputs, cleaned of terminal codes; a word of the engine's own
vocabulary in any of them stops the run (the record says what was measured, not how). SOURCES lists the checks' files: one that is missing is a warning, and STRICT=1 makes it an error."""
import json
import os
import re
import shutil
import sys

F, OUT = sys.argv[1], sys.argv[2]
STRICT = os.environ.get("STRICT") == "1"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BANNED = re.compile(r"\b(pool|arena|chunk|page|ring|slab|escape|void|raw)s?\b", re.I)
os.makedirs(OUT, exist_ok=True)
missing = []


def path(*p):
    return os.path.join(F, *p)


def copy_clean(src, dst, drop=None):
    text = ANSI.sub("", open(src, errors="replace").read()).replace("\r", "\n")
    if drop:
        text = "\n".join(l for l in text.split("\n") if not re.search(drop, l))
    bad = BANNED.search(text.replace("chat page", ""))  # (the chat's web page is not the cache's)
    if bad:
        raise SystemExit(f"{src}: the word {bad[0]!r}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(text)


def text_of(*p):
    f = path(*p)
    if not os.path.exists(f):
        missing.append("/".join(p))
        return ""
    return ANSI.sub("", open(f, errors="replace").read())


def num(s):
    return int(s.replace(",", ""))


def fmt(n):
    return f"{int(n):,}"


# ---- the bench runs ---------------------------------------------------------------------------------------------------------------
runs = sorted(d for d in os.listdir(path("conc")) if os.path.isdir(path("conc", d))) if os.path.isdir(path("conc")) else []
rows = []  # (server, users, run, json, hit)
for run in runs:
    d = path("conc", run)
    server = re.sub(r"(\d+|[A-Z])$", "", run)
    for f in sorted(os.listdir(d)):
        m = re.fullmatch(rf"{server}-c(\d+)\.json", f)
        if not m:
            continue
        j = json.load(open(os.path.join(d, f)))
        hit = float(open(os.path.join(d, f"hit-{server}-c{m[1]}.txt")).read().strip() or 0)
        rows.append((server, int(m[1]), run, j, hit))
        # the JSON without its per-request arrays (the texts, the latencies of every token)
        slim = {k: v for k, v in j.items() if not isinstance(v, list)}
        os.makedirs(os.path.join(OUT, run), exist_ok=True)
        json.dump(slim, open(os.path.join(OUT, run, f), "w"), indent=1)
        copy_clean(os.path.join(d, f"bench-{server}-c{m[1]}.txt"), os.path.join(OUT, run, f"bench-{server}-c{m[1]}.txt"))
        shutil.copy(os.path.join(d, f"hit-{server}-c{m[1]}.txt"), os.path.join(OUT, run, f"hit-{server}-c{m[1]}.txt"))
        smi = os.path.join(d, f"smi-{server}-c{m[1]}.csv")
        if os.path.exists(smi):
            shutil.copy(smi, os.path.join(OUT, run, os.path.basename(smi)))
    for extra in sorted(f for f in os.listdir(d) if re.fullmatch(r"stats-engine(-\w+)?\.json", f)):
        shutil.copy(os.path.join(d, extra), os.path.join(OUT, run, extra))
    log = os.path.join(d, f"serve-{server}.log")
    if server == "engine" and os.path.exists(log):
        copy_clean(log, os.path.join(OUT, run, "serve-engine.log"), drop=r"chat page")

# ---- the checks' outputs ----------------------------------------------------------------------------------------------------------
SOURCES = {
    "kvtest.out": ("p3", "kvtest.out"),
    "kvtest-first-tables.out": ("p3", "kvtest-first-tables.out"),
    "extra-memory-16-2.out": ("p3", "random-16-2.out"),
    "extra-memory-8-1.out": ("p3", "random-8-1.out"),
    "extra-memory-3-0.out": ("p3", "random-3-0.out"),
    "server-extra-memory.out": ("p3", "multi", "exhaust-3-0.out"),
    "churn-natural.out": ("p3", "churn-natural.out"),
    "churn-random.out": ("p3", "churn-random.out"),
    "churn-mixed-little-extra-memory.out": ("p3", "churn-mixed-little.out"),
    "churn-server-natural.out": ("p3", "serve-churn.out"),
    "rowcheck-fresh.out": ("p3", "rowcheck-fresh.out"),
    "rowcheck-churn-300.out": ("p3", "rowcheck-churn300.out"),
    "run-again.out": ("p3", "recompute.out"),
    "write-check-natural-room.out": ("p3", "verify-natural.out"),
    "write-check-random-room.out": ("p3", "verify-random.out"),
    "sampling-candidates.out": ("p3", "topkcheck.out"),
    "sampling-seeded-host-one.out": ("p3", "multi", "seeded-host-one.out"),
    "sampling-seeded-host-many.out": ("p3", "multi", "seeded-host-many.out"),
    "sampling-seeded-gpu-one.out": ("p3", "multi", "seeded-gpu-one.out"),
    "sampling-seeded-gpu-many.out": ("p3", "multi", "seeded-gpu-many.out"),
    "clients-at-once.out": ("p3", "multi", "multi4.out"),
    "clients-memory-too-small.out": ("p3", "multi", "multi3tight.out"),
    "step-profile.out": ("p3", "multi", "rowprofile.out"),
    "rows-bits.out": ("p3", "multi", "rowcheck.out"),
    "api-smoke.out": ("p3", "multi", "smoke.out"),
    "api-soak.out": ("p3", "multi", "soak.out"),
    "layout-default.out": ("p3", "multi", "layout.out"),
}
for name, src in SOURCES.items():
    f = path(*src)
    if os.path.exists(f):
        copy_clean(f, os.path.join(OUT, "checks", name), drop=r"^sched:")
    else:
        missing.append("/".join(src))
for src in (("p3", "per-step.txt"),):
    if os.path.exists(path(*src)):
        copy_clean(path(*src), os.path.join(OUT, "checks", src[-1]))


def check(name):
    f = os.path.join(OUT, "checks", name)
    return open(f).read() if os.path.exists(f) else ""


def need(t, pat, what, flags=re.S):
    m = re.search(pat, t, flags)
    if not m:
        missing.append(what)
        return None
    return m


# ---- the bench tables ---------------------------------------------------------------------------------------------------------------
order = {"engine": 0, "kvglyd": 1, "bf16": 2}
label = {"engine": "glyd-engine", "kvglyd": "vLLM with Glyd (weights and KV cache)", "bf16": "vLLM, bf16"}
first = lambda run: run in ("engine1", "engine2", "kvglyd1", "kvglyd2")
middle = lambda run: run == "engineA"
head = "| server | users | run | requests | output tok/s | per user tok/s | time to first token ms (mean, p99) | time per token ms (mean, p99) | prefix cache hit % |\n| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"
line = lambda server, users, run, j, hit: (f"| {label[server]}{' (first measurement)' if first(run) and server == 'engine' else ' (sampling on the GPU, first token not yet sent early)' if middle(run) else ''} | {users} | {run} | {j['completed']} | {j['output_throughput']:.1f} | {1000 / j['mean_tpot_ms']:.1f} | "
                                           f"{j['mean_ttft_ms']:.0f}, {j['p99_ttft_ms']:.0f} | {j['mean_tpot_ms']:.1f}, {j['p99_tpot_ms']:.1f} | {hit:.2f} |")
final_rows = [r for r in rows if not first(r[2]) and not middle(r[2])]
middle_rows = [r for r in rows if middle(r[2])]
first_rows = [r for r in rows if first(r[2])]


def table(rs):
    out = [head]
    for server, users, run, j, hit in sorted(rs, key=lambda r: (r[1], order[r[0]], r[2])):
        out.append(line(server, users, run, j, hit))
    return "\n".join(out)


def means(rs, tag):
    out = ["| server | users | runs | output tok/s (mean) | per user tok/s (mean) | time to first token ms (mean) |", "| :--- | ---: | ---: | ---: | ---: | ---: |"]
    for users in sorted({r[1] for r in rs}):
        for server in sorted({r[0] for r in rs}, key=order.get):
            sel = [r for r in rs if r[0] == server and r[1] == users]
            if sel:
                n = len(sel)
                out.append(f"| {label[server]}{tag if server == 'engine' else ''} | {users} | {n} | {sum(r[3]['output_throughput'] for r in sel) / n:.1f} | {sum(1000 / r[3]['mean_tpot_ms'] for r in sel) / n:.1f} | {sum(r[3]['mean_ttft_ms'] for r in sel) / n:.0f} |")
    return "\n".join(out)


def pass_counts(run, users, before):
    """What the engine counted in one pass (its /stats after the pass less its /stats before it: `stats-engine-cN.json`, `stats-engine-warmup.json`), per decode step and per prompt, and how much of the pass the GPU was at work."""
    def load(name):
        f = os.path.join(OUT, run, f"stats-engine-{name}.json")
        return json.load(open(f)) if os.path.exists(f) else None
    a, b = load(before), load(f"c{users}")
    j = next((r[3] for r in rows if r[2] == run and r[1] == users), None)
    if not a or not b or not j:
        return None
    d = {k: b[k] - a[k] for k in b if isinstance(b[k], (int, float)) and k in a}
    n = max(1, d["steps"])
    host = (d["decode_prep_ms"] + d["decode_post_ms"] + d["before_ms"] + d["after_ms"]) / n
    return {"rows": d["rows"] / n, "graph": d["decode_graph_ms"] / n, "host": host, "prefill": d["prefill_ms"] / max(1, d["prefill_rounds"]), "busy": 100 * (d["decode_ms"] + d["prefill_ms"]) / (1000 * j["duration"]), "steps": d["steps"]}


def versus(rs):
    """The final runs' means, the engine against vLLM with Glyd, one line for each number of users."""
    out = []
    for users in sorted({r[1] for r in rs}):
        e = [r[3] for r in rs if r[0] == "engine" and r[1] == users]
        v = [r[3] for r in rs if r[0] == "kvglyd" and r[1] == users]
        if e and v:
            mean = lambda js, k: sum(j[k] for j in js) / len(js)
            te, tv, fe, fv = mean(e, "output_throughput"), mean(v, "output_throughput"), mean(e, "mean_ttft_ms"), mean(v, "mean_ttft_ms")
            spread = lambda js: 100 * (max(j["output_throughput"] for j in js) - min(j["output_throughput"] for j in js)) / mean(js, "output_throughput")
            out.append(f"- {users} user{'s' if users > 1 else ''}: {te:.1f} tokens a second against {tv:.1f} ({100 * (te / tv - 1):+.1f}%; the runs of each within {spread(e):.1f}% and {spread(v):.1f}% of each other), the first token after {fe:.0f} ms against {fv:.0f} ({100 * (fe / fv - 1):+.0f}%)")
    return "\n".join(out)


tables = ""
if final_rows:
    tables += table(final_rows) + "\n\n" + means(final_rows, "") + "\n\nThe engine against vLLM with Glyd, the means of the final runs:\n\n" + versus(final_rows) + "\n"
if middle_rows:
    tables += "\nThe build between (the engine thread's work after each step moved to the GPU, a request's first token still sent after its round's decode step; one run, the vLLM run beside it is kvglydA):\n\n" + table(middle_rows) + "\n"
if first_rows:
    tables += "\nThe first measurement, before the engine thread's work after each step was cut (same client, same prompts):\n\n" + table(first_rows) + "\n\n" + means(first_rows, " (first measurement)") + "\n"
open(os.path.join(OUT, "table.md"), "w").write(tables)
print(tables)

# ---- sampling on the host and on the GPU, 8 users, alternated (the same build; GLYD_ENGINE_HOST_SAMPLING=1 for the host's) --------------
ab_rows = []
ab_dir = path("p2", "v6")
if os.path.isdir(ab_dir):
    for name in sorted((d for d in os.listdir(ab_dir) if d.startswith("quick-") and os.path.isdir(os.path.join(ab_dir, d))), key=lambda d: d.rsplit("-", 1)[-1]):
        s = json.load(open(os.path.join(ab_dir, name, "stats-engine.json")))
        j = json.load(open(os.path.join(ab_dir, name, "engine-c8.json")))
        n = max(1, s["steps"])
        mode = "the host" if name.startswith("quick-host") else "the GPU"
        ab_rows.append((name, mode, j, s, n))
        os.makedirs(os.path.join(OUT, "sampling", name), exist_ok=True)
        for f in ("stats-engine.json", "hit-engine-c8.txt"):
            shutil.copy(os.path.join(ab_dir, name, f), os.path.join(OUT, "sampling", name, f))
        json.dump({k: v for k, v in j.items() if not isinstance(v, list)}, open(os.path.join(OUT, "sampling", name, "engine-c8.json"), "w"), indent=1)
sampling_table = ""
if ab_rows:
    sampling_table = ("| sampling on | run | output tok/s | time per token ms | rows a step | GPU ms a step | engine thread outside the GPU's work, ms a step: before the step | after the step | the rows' work after it |\n| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
                      + "\n".join(f"| {mode} | {name[6:]} | {j['output_throughput']:.1f} | {j['mean_tpot_ms']:.1f} | {s['rows'] / n:.2f} | {s['decode_graph_ms'] / n:.1f} | {(s['decode_prep_ms'] + s['before_ms']) / n:.2f} | {s['decode_post_ms'] / n:.2f} | {s['after_ms'] / n:.2f} |"
                                for name, mode, j, s, n in ab_rows) + "\n")

# ---- the lossless cache's counts --------------------------------------------------------------------------------------------------------
kv = check("kvtest.out")
m_all = need(kv, r"in all: values checked (\d+), differing (\d+)", "kvtest: in all")
m_first = need(kv, r"first writes:\s+(\d+) values \((\d+) zeros, (\d+) negative zeros, (\d+) subnormals, (\d+) infinities, (\d+) NaNs, (\d+) outside", "kvtest: first writes")
m_stat = need(kv, r"blocks lost (\d+); extra memory in use after releasing every block: first (\d+) of (\d+), second (\d+) of (\d+)", "kvtest: status")
m_layers = need(kv, r"known-answer test of the KV cache: (\d+) layers, (\d+) blocks", "kvtest: layers")


def stress(name):
    t = check(name)
    if not t:
        return None
    d = {}
    for key, pat in {
        "head": r"stress: (\d+) requests \((\d+) tokens of prompts, (\w+), seed (\d+)\) at (\d+) users at a time on a cache of (\d+) tokens",
        "outcome": r"([\d.]+) s: (\d+) requests with every token, (\d+) ended with an error in words, (\d+) sent back to wait at least once \(most often (\d+) times\)",
        "sched": r"scheduler: prompt tokens (\d+), found in the cache (\d+), sent back for room (\d+), steps taken back for a block the cache could not store exactly (\d+) \((\d+) blocks named lost\), blocks kept in the second extra memory (\d+), ended for it (\d+)",
        "values": r"values checked (\d+), differing (\d+), values in blocks the cache had named lost \(taken back, not compared\) (\d+)",
        "mem": r"extra memory in use now that nothing is held: first (\d+) of (\d+), second (\d+) of (\d+)",
        "alone": r"against the requests one at a time: (\d+) the same tokens; (\d+) differ without having been sent back; (\d+) differ after being sent back",
        "errors": r"(\d+) requests ended with an error under load; run alone, (\d+) end so again and (\d+) are served",
        "verdict": r"stress: (all as it should be|FAILED)",
        "kvline": r"KV cache: .*?(\d+) blocks took units of the first extra memory \(used at most (\d+) of (\d+) at once\), (\d+) blocks kept in the second \(used at most (\d+) of (\d+) units\), (\d+) blocks lost",
    }.items():
        m = re.search(pat, t)
        if m:
            d[key] = m.groups()
    return d


def stress_row(title, name):
    d = stress(name)
    if not d or "head" not in d or "values" not in d:
        missing.append(name)
        return f"| {title} | (missing) |  |  |  |  |  |  |"
    h, o, s, v, mm, a = d["head"], d.get("outcome"), d["sched"], d["values"], d["mem"], d.get("alone")
    err = d.get("errors")
    ended = f"{o[2]}" + (f" (alone, {err[1]} end so again)" if err else "")
    ref = f"{a[0]} / {a[1]} / {a[2]}" if a else "(not compared)"
    return (f"| {title} | {h[0]} {h[2].lower()} | {h[4]} | {h[5]} | {fmt(v[0])} | {v[1]} | {fmt(v[2])} | {s[3]} | {s[2]} | {ended} | {ref} |")


stress_head = ("| run | requests | users | cache tokens | values compared | differing | values in lost blocks (not compared) | steps taken back | sent back to wait | ended with an error | same as alone / part from it without / after being sent back |\n"
               "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
stress_rows = [
    stress_row("natural text, 300 requests, 128 blocks", "churn-natural.out"),
    stress_row("random tokens, 300 requests, 128 blocks", "churn-random.out"),
    stress_row("natural text and random tokens, 300 requests, 128 blocks, little extra memory (16 and 2 units)", "churn-mixed-little-extra-memory.out"),
    stress_row("random tokens, extra memory 16 and 2 units", "extra-memory-16-2.out"),
    stress_row("random tokens, extra memory 8 and 1 units", "extra-memory-8-1.out"),
    stress_row("random tokens, extra memory 3 and 0 units (requests that do not fit alone end with an error)", "extra-memory-3-0.out"),
    stress_row("natural text, 100 requests, room for all of them", "write-check-natural-room.out"),
    stress_row("random tokens, 100 requests, room for all of them", "write-check-random-room.out"),
]
stress_table = stress_head + "\n" + "\n".join(stress_rows)
srv = check("churn-server-natural.out")
m_srv = need(srv, r"(\d+) requests, (\d+) wrong; prompt tokens (\d+), found in the cache (\d+), sequences sent back to wait (\d+)\s+values of the cache checked against the bf16 copy: (\d+), differing: (\d+)", "server churn")
rc_fresh, rc_churn = check("rowcheck-fresh.out"), check("rowcheck-churn-300.out")
hashes = {n: set(re.findall(r"logits hash ([0-9a-f]{16})", t)) for n, t in (("fresh", rc_fresh), ("churn", rc_churn))}
same_hash = bool(hashes["fresh"]) and len(set().union(*hashes.values())) == 1
m_rc = need(rc_churn, r"(\d+) rounds of churn first; values checked against the bf16 copy (\d+), differing (\d+), values in blocks the cache had named lost \(not compared\) (\d+)", "rowcheck churn")
sx = check("server-extra-memory.out")
sx_ok = [l for l in sx.split("\n") if l.startswith("ok")]
sx_bad = [l for l in sx.split("\n") if l.startswith("FAIL")]
re_run = check("run-again.out")

lossless = f"""## The cache stays lossless

Every check ran on the L4 with Qwen3-8B and the KV tables the engine made for it; the outputs are in `checks/`. A value the cache holds is the bits that were written, and where the cache has no room for what a request needs, the
request is never answered from what it could not hold.

- **A known answer** (`checks/kvtest.out`, `checks/kvtest-first-tables.out`). bf16 values made on the host, of every kind a cache is given (inside the range its layer's windows hold and outside it, both zeros, subnormals,
  infinities, NaNs of several payloads, the largest and the smallest numbers, blocks with few, many and more exceptions than a block can keep in its own room, and the 65,536 bit patterns once each) were written into {m_layers[2] if m_layers else '?'} blocks of 16 tokens of each of
  the {m_layers[1] if m_layers else '?'} layers, as one run of tokens, token by token and in pieces, read back and compared bit for bit; then written again from the first token and again from the middle, and compared again. {fmt(m_all[1]) if m_all else '?'} values compared
  ({fmt(m_first[2]) if m_first else '?'} zeros, {fmt(m_first[3]) if m_first else '?'} negative zeros, {fmt(m_first[4]) if m_first else '?'} subnormals, {fmt(m_first[5]) if m_first else '?'} infinities, {fmt(m_first[6]) if m_first else '?'} NaNs in each of the three passes, a fifth of the values outside their windows): **{m_all[2] if m_all else '?'} differ**. With every block released the cache's extra memory
  is empty ({m_stat[2] if m_stat else '?'} of {m_stat[3] if m_stat else '?'} and {m_stat[4] if m_stat else '?'} of {m_stat[5] if m_stat else '?'} in use), and {m_stat[1] if m_stat else '?'} blocks were lost.
- **Every write compared with a bf16 copy of what was written, at the step that makes it** (a copy of every layer's cache; each step's tokens are decoded back and compared before the next step runs): requests of natural text and of
  random tokens, 4 at a time on a cache of 128 to 144 blocks, so that finished prompts are reused, kept for later prompts and found again, and requests wait and are run again (the table below; `checks/churn-*.out`,
  `checks/extra-memory-*.out`, `checks/write-check-*.out`). In the server (`checks/churn-server-natural.out`): {m_srv[1] if m_srv else '?'} short requests of natural text from 4 clients on a cache of 128 blocks, {m_srv[2] if m_srv else '?'} wrong, {fmt(m_srv[3]) if m_srv else '?'} prompt tokens,
  {fmt(m_srv[4]) if m_srv else '?'} found in the cache, {m_srv[5] if m_srv else '?'} sent back to wait; {fmt(m_srv[6]) if m_srv else '?'} values compared, **{m_srv[7] if m_srv else '?'} differ**.
- **Where the cache has no room for what a request needs** (extra memory cut to 16 and 2, and to 8 and 1 units, so that random tokens run it out; `checks/extra-memory-*.out`): the step in which a block could not be stored exactly is named
  at once, the output of that step is taken back from the requests that hold the block (no token is made from it), and each such request waits until the ones running with it have ended, then runs again from its own text; one that is
  taken back twice runs alone; one that is lost alone with nothing kept for later prompts ends with an error in words. A request that ends so is run again alone, to see that it does not fit alone: those that made no token end so again; one that had made tokens before it ended may be served, its text run again as a prompt being computed another way than the same tokens were by steps (its keys and values then differ in their last bit) and so needing other extra memory (the table says how many end so again). The values of blocks named lost are left out of
  the comparison (counted in the table), every other value is compared. The few requests that part from their run alone do so after being run again, at a place where the model's two best choices are near-equal: a prompt run again is
  computed in another way than the same tokens were by steps (`checks/run-again.out`), so its keys and values differ in their last bit from the ones the steps made. In the server (`checks/server-extra-memory.out`): {len(sx_ok)} checks ok, {len(sx_bad)} failed: a request the cache cannot hold is answered with the reason in words, one it can hold
  is served whole, and the one after the failures gets the text it got before them.
- **After many rounds of churn** (`checks/rowcheck-fresh.out`, `checks/rowcheck-churn-300.out`): one sequence's logits with 0 to 7 others in the step, at every number of rows and in either row, on a fresh cache and after
  {m_rc[1] if m_rc else '?'} rounds of 8 sequences of random lengths on the same memory ({fmt(m_rc[2]) if m_rc else '?'} values compared, {m_rc[3] if m_rc else '?'} differ): {"the same logits bit for bit in every one" if same_hash else "NOT the same (see the files)"}.

The runs of requests against a bf16 copy (each request's tokens against the same request run alone on a cache with room, where that was made):

{stress_table}
"""
open(os.path.join(OUT, "lossless.md"), "w").write(lossless)

# ---- the rest of the README -------------------------------------------------------------------------------------------------------------
prof = {}
for m in re.finditer(r"a step of (\d+) sequences.*?\n(.*?)\n\s*([\d.]+)\s+total", check("step-profile.out"), re.S):
    prof[int(m[1])] = float(m[3])


def last(name, pat):
    t = check(name).split("\n")
    return next((l for l in reversed(t) if re.search(pat, l)), "")


step_lines = ""
passes = []
for run in runs:
    if not run.startswith("engine"):
        continue
    before = "warmup"
    for users in sorted({r[1] for r in rows if r[2] == run}):
        s = pass_counts(run, users, before)
        if s:
            passes.append((run, users, s))
        before = f"c{users}"
if passes:
    body = "\n".join(f"| {r} | {u} | {s['steps']} | {s['rows']:.2f} | {s['graph']:.1f} | {s['host']:.2f} | {s['prefill']:.0f} | {s['busy']:.1f} |" for r, u, s in passes)
    step_lines = ("\nThe engine's own counts for each pass (`stats-engine-cN.json` of each run, each less the one before it): decode steps, rows in a step, the GPU's own time for a step (milliseconds), the time the engine thread spends outside the GPU's work "
                  "for a step (milliseconds), the GPU's time for a prompt (milliseconds), and the share of the pass in which the GPU was at work on a decode step or a prompt:\n\n"
                  "| run | users | decode steps | rows a step | GPU ms a step | engine thread's other work ms a step | prompt ms | GPU at work % of the pass |\n| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n" + body + "\n")

seeded = {n: check(f"sampling-seeded-{n}.out").split("\n")[:8] for n in ("host-one", "host-many", "gpu-one", "gpu-many")}
same_seeded = "the same 8 texts in all four runs" if all(seeded.values()) and all(v == seeded["host-one"] for v in seeded.values()) else "NOT the same in all four runs (see the files)"
sampling_section = ""
if sampling_table:
    sampling_section = ("\n## Where the engine thread's time went at 8 users\n\nThe same build, 8 users, 16 requests each time, alternated (host, GPU, host, GPU: `sampling/`): the draw for each row of a step made on the host from the row's logits (about "
                        "1.7 ms a row), and made from the 64 largest logits the step's own work finds on the GPU. The engine thread's time per step outside the GPU's work:\n\n" + sampling_table)
readme = f"""# glyd-engine for a few users at once (2026-10-03)

glyd-engine serves up to 8 users at once on one GPU (`--users N`, default 4): their replies are decoded together, a prompt that begins as an earlier one reuses what was computed for it, and where the GPU's memory for the replies
runs out the newest one waits and is run again from its own text, its stream going on where it was. This record measures it on an NVIDIA L4 (AWS g6.4xlarge) with Qwen3-8B at 1, 4 and 8 users, against vLLM 0.30.0 with Glyd's weights
and Glyd's lossless KV cache (the same weights and the same cache), and checks that the cache stays lossless as the memory of finished replies is reused and as it runs short.

## What was run

The final runs and every check below are of the engine at commit f468e8c9 of the engine-mvp branch (the build between is 5c52dcbb; the first measurement is the build before the scheduler's first-token and sampling changes).

`vllm bench serve` as the same client against both servers: random prompts of 1,024 tokens, 256 tokens out, `--ignore-eos`, `--max-concurrency` 1, 4 and 8 (a user sends the next request when its reply is complete) with 8, 24 and
32 requests; a warm-up pass first, then a new set of prompts for every pass (`--seed` per pass, the same sets for both servers), so no pass finds a prompt a server has seen: the prefix cache's hit rate is printed for every pass
(the engine's own count, vLLM's log) and is 0.00 in every one of them. The client sends no sampling settings: each server answers with the sampling the model's own configuration asks for (temperature 0.6, top-k 20, top-p 0.95 for
Qwen3-8B). Both servers had the GPU to themselves, a context of 4,096 tokens, and a start from 50 C or less where it cooled in time; the L4 slows as it heats at its 72 W cap. The final runs alternated (vLLM, engine, vLLM, engine); the first
table has every one of them, the second their means; the tables after them are the runs of the earlier builds.

{tables}
Per user tok/s is the inverse of the mean time per token. In the final build a request's first token is sent as soon as its prompt is done (in the build between it was sent after the decode step of the same round, a decode step late: 46 ms at one user).{step_lines}
{sampling_section}
{lossless}
## Other checks

- Several clients at once (`checks/clients-at-once.out`): {last("clients-at-once.out", "read what they read alone")}
- Memory too small for the clients (`checks/clients-memory-too-small.out`): {last("clients-memory-too-small.out", "read what they read alone")} {last("clients-memory-too-small.out", "^stats")}
- A client that leaves midway leaves nothing behind (`checks/clients-at-once.out`): {last("clients-at-once.out", "after a client left")}
- The API's check of itself (`checks/api-smoke.out`): {sum(1 for l in check("api-smoke.out").split(chr(10)) if l.startswith("ok"))} checks ok, {sum(1 for l in check("api-smoke.out").split(chr(10)) if l.startswith("FAIL"))} failed; the soak (`checks/api-soak.out`): {last("api-soak.out", "rounds")}
- Sampling on the GPU (`checks/sampling-candidates.out`, `checks/sampling-seeded-*.out`): the 64 largest logits of every row of a step are found on the GPU and the draw is made from them (temperature, top-k, top-p and min-p as the host's own sampler applies them);
  with a seed, 8 texts came out the same with the sampling on the host and on the GPU, asked one at a time and four at a time on a cold cache each time ({same_seeded}).
- The layout default on this GPU (`checks/layout-default.out`): the smallest layout, as measured as fast as the 12-bit one on an Ada GPU; `--layout mma12` as asked.
- A decode step on the GPU takes {prof.get(1, 0):.1f} ms for one sequence of 1,100 tokens, {prof.get(4, 0):.1f} ms for 4 and {prof.get(8, 0):.1f} ms for 8 (`checks/step-profile.out`).

A reply can in general depend on how many replies are decoded with it; on this GPU the logits of a sequence were the same bit for bit at every number of rows from 1 to 8 (`checks/rows-bits.out`).
"""
bad = BANNED.search(readme)
if bad:
    raise SystemExit(f"README: the word {bad[0]!r}: {readme[max(0, bad.start() - 80):bad.end() + 40]!r}")
open(os.path.join(OUT, "README.md"), "w").write(readme)
if missing:
    msg = "missing: " + ", ".join(sorted(set(missing)))
    print(msg, file=sys.stderr)
    if STRICT:
        raise SystemExit(msg)
