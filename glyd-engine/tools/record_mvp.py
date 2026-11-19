"""The data of the MVP's record (benchmarks/gpu/engine-mvp-2026-10-03) from the files a run left on the box: python record_mvp.py FETCHED_DIR REF.json OUT_DIR [A100_DIR H100_DIR]
(A100_DIR and H100_DIR: the results directories of jobs/engine_gpu_job.sh on an A100 and an H100)

FETCHED_DIR holds batch_b/, batch_c/, batch_d/ and gemm4/ (as `bget` brings them) and the earlier plugin-tables run (earlier/eng-blas.json, eng-blas-score.json). The logs are copied with the lines about the
cache's storage shortened to "nothing lost" and the per-layer counts dropped, as the record's wording rules ask; numbers.txt is computed here, from the data, so that the README's figures are not typed twice.
"""
import glob
import json
import os
import re
import shutil
import statistics
import sys

SRC, REF, OUT = sys.argv[1:4]
JOBS = {"a100": sys.argv[4], "h100": sys.argv[5]} if len(sys.argv) > 5 else {}
BF16_ONE_USER = {"a100": (75.3, "a100-vllm-kv-step-2026-10-02, 1 x 1,024, vLLM bf16 with its own cache"), "h100": (152.8, "h100-vllm-kv-2026-10-02, 1 x 1,024, vLLM bf16 with its own cache")}
ref = json.load(open(REF))


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean_line(l):
    l = ANSI.sub("", l)
    if l.startswith("KV cache:"):
        return "KV cache: nothing lost\n" if re.search(r"(^|[^0-9])0 (pages|blocks) lost", l) else "KV cache: SEE THE RUN\n"
    l = re.sub(r", \d+ pages a layer", "", l)
    return l.replace("prompt chunk", "prompt step").replace("default chunk", "default step")


def copy_clean(src, dst, keep=None):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, errors="replace") as f, open(dst, "w") as g:
        for l in f:
            if keep is None or l.startswith(keep):
                g.write(clean_line(l))


def part(path, start, end=None):
    """The lines of a file from the one that contains `start` to the one that contains `end` (exclusive)."""
    out, on = [], False
    for l in open(path, errors="replace"):
        if not on and start in l:
            on = True
        elif on and end and end in l:
            break
        if on:
            out.append(clean_line(l))
    return "".join(out)


def put(rel, text):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text)


b, c, d, g, e, f, gg, hh = (os.path.join(SRC, x) for x in ("batch_b", "batch_c", "batch_d", "gemm4", "batch_e", "batch_f", "batch_g", "batch_h"))
# --- logs
for src, rel in ((b + "/batch_b.out", "l4/batch-b.out"), (c + "/batch_c.out", "l4/batch-c.out"), (c + "/first-use-cpu.out", "l4/first-run-8b-download-and-pack.out"), (g + "/gemm4.out", "gemm/gemm-v3.out"),
                 (b + "/serve.log", "api/serve.log"), (d + "/batch_d.out", "l4/batch-d.out"), (e + "/batch_e.out", "l4/batch-e.out"), (f + "/batch_f.out", "l4/batch-f.out"), (gg + "/batch_g.out", "l4/batch-g.out"), (hh + "/batch_h.out", "l4/batch-h.out")):
    if os.path.exists(src):
        copy_clean(src, os.path.join(OUT, rel))
if os.path.exists(b + "/compare-8b.txt"):
    # the tables' windows and held-out rates; the lines about the cache's sizing are left out
    copy_clean(b + "/compare-8b.txt", os.path.join(OUT, "tables/compare-8b-8192.txt"), keep=("windows", "meta positions", "meta refused", "meta layers", "meta heads", "meta channels", "meta held_out_rate"))
if os.path.exists(b + "/batch_b.out"):
    put("gemm/gemm-v2.out", part(b + "/batch_b.out", "== gemm", "== doctor"))
    put("api/serve-smoke.out", part(b + "/batch_b.out", "== serve"))
    put("tables/kvcal-8b-8192.out", part(b + "/batch_b.out", "== kvcal 8B", "== check, the engine's own tables"))
if os.path.exists(c + "/batch_c.out"):
    put("gemm/sweep-512-4096.out", part(c + "/batch_c.out", "== gemm sweep", "== bench, GLYD_ENGINE_GEMM=blas"))
    put("l4/prefill-by-product-route.out", part(c + "/batch_c.out", "== bench, GLYD_ENGINE_GEMM=blas", "== check, own products"))
os.makedirs(os.path.join(OUT, "clocks"), exist_ok=True)
for sub, name in ((b, "batch-b"), (c, "batch-c"), (d, "batch-d"), (e, "batch-e"), (f, "batch-f"), (gg, "batch-g"), (g, "gemm-v3")):
    if os.path.exists(sub + "/smi.csv"):
        shutil.copy(sub + "/smi.csv", os.path.join(OUT, "clocks", name + "-smi.csv"))
for sub, pat, pre in ((b, "eng-*.json", ""), (c, "eng-*.json", ""), (e, "eng-*.json", "final-"), (os.path.join(SRC, "earlier"), "eng-*.json", "earlier-")):
    for p in glob.glob(os.path.join(sub, pat)):
        os.makedirs(os.path.join(OUT, "proofs"), exist_ok=True)
        shutil.copy(p, os.path.join(OUT, "proofs", pre + os.path.basename(p)))
if os.path.exists(e + "/batch_e.out"):
    put("l4/first-run-8b-glyd.out", part(e + "/batch_e.out", "== the 8B through the glyd program", "== check and score"))
    put("api/serve-smoke-with-key.out", part(e + "/batch_e.out", "== the API, with a key", "== the terminal chat"))
    put("api/terminal-chat.out", part(e + "/batch_e.out", "== the terminal chat", "== the server under mixed traffic"))
    put("api/mixed-traffic.out", part(e + "/batch_e.out", "== the server under mixed traffic", "== every tile"))
    put("gemm/tile-bits.out", part(e + "/batch_e.out", "== every tile", " done"))

# --- the A100 and the H100: the job's text files (the cache's lines shortened), the tokens and scores, the clocks; not the pip, rustup and cargo output nor the tables' binary
for gpu, jd in JOBS.items():
    for name in sorted(os.listdir(jd)):
        src, dst = os.path.join(jd, name), os.path.join(OUT, gpu, name)
        if name in ("pip.out", "rustup.out", "build.out", "kvtab-8192.safetensors", "DONE"):
            continue
        if name.endswith((".out", ".txt", ".log", ".csv")):
            copy_clean(src, dst)
        elif name.endswith(".json"):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

# --- numbers
lines = []


def say(s=""):
    lines.append(s)


def tokens_line(name, path):
    e = json.load(open(path))
    rt = ref["single"]["tokens"]
    same = sum(1 for a, b_ in zip(rt, e["tokens"]) if a == b_)
    pos = sum(sum(1 for x, y in zip(a, b_) if x == y) for a, b_ in zip(rt, e["tokens"]))
    first = [next((i for i, (x, y) in enumerate(zip(a, b_)) if x != y), None) for a, b_ in zip(ref["long"]["tokens"], e["long_tokens"])]
    return f"{name:<46} short: {same}/8 identical, {pos}/512 positions; long: {sum(1 for f in first if f is None)}/3 identical, first differences {first}"


def score_line(name, path):
    e = json.load(open(path))
    eq = sum(1 for a, b_ in zip(ref["score_top1"], e["score_top1"]) if a == b_)
    dd = statistics.mean(abs(a - b_) for a, b_ in zip(ref["score_lp"], e["score_lp"]))
    return f"{name:<46} top-1 {eq}/{len(ref['score_top1'])} = {100 * eq / len(ref['score_top1']):.2f}%, mean |logprob difference| {dd:.4f}, position 1500: {e['score_lp'][1500]:.3f}"


say("tokens against vLLM+Glyd's (an L4's, 8,192 tokens a step; 8 short prompts of 64 tokens one at a time, 3 long ones of 1,030, 2,100 and 3,500 tokens)")
for name, p in (("plugin's tables, tiered, cuBLAS (earlier run)", SRC + "/earlier/eng-blas.json"), ("engine's own tables, tiered, cuBLAS", b + "/eng-own.json"), ("12-bit layout, cuBLAS", b + "/eng-12.json"),
                ("tiered, the engine's own products", c + "/eng-own-gemm.json"), ("12-bit, the engine's own products", c + "/eng-own-gemm-12.json")):
    if os.path.exists(p):
        say(tokens_line(name, p))
say()
say("teacher-forced top-1 over 1,535 positions against vLLM+Glyd's")
for name, p in (("plugin's tables, tiered, cuBLAS (earlier run)", SRC + "/earlier/eng-blas-score.json"), ("engine's own tables, tiered, cuBLAS", b + "/eng-own-score.json"), ("12-bit layout, cuBLAS", b + "/eng-12-score.json"),
                ("tiered, the engine's own products", c + "/eng-own-gemm-score.json")):
    if os.path.exists(p):
        say(score_line(name, p))
say()


def bench_rows(text):
    return [json.loads(l) for l in text.split("\n") if l.startswith("{")]


if os.path.exists(c + "/batch_c.out"):
    say("prefill and decode by the way a long prompt is multiplied (the last of 5 timed runs; the L4 hot, 72 W cap)")
    for title, a, z in (("tiered, cuBLAS", "bench, GLYD_ENGINE_GEMM=blas (tiered)", "bench, GLYD_ENGINE_GEMM=own (tiered)"), ("tiered, the engine's own products", "bench, GLYD_ENGINE_GEMM=own (tiered)", "bench, GLYD_ENGINE_GEMM=lib (tiered)"),
                        ("tiered, the library's routes (2,048 tokens a step)", "bench, GLYD_ENGINE_GEMM=lib (tiered)", "bench, own, not fused"), ("tiered, own, no fused SiLU", "bench, own, not fused", "bench, own, from 512 rows"),
                        ("tiered, own from 512 rows", "bench, own, from 512 rows", "bench, library routes (512, 768)"), ("tiered, library routes", "bench, library routes (512, 768)", "bench, GLYD_ENGINE_GEMM=blas (12-bit)"),
                        ("12-bit, cuBLAS", "bench, GLYD_ENGINE_GEMM=blas (12-bit)", "bench, GLYD_ENGINE_GEMM=own (12-bit)"), ("12-bit, own", "bench, GLYD_ENGINE_GEMM=own (12-bit)", "check, own products")):
        t = part(c + "/batch_c.out", a, z)
        say(f"  {title:<50} " + "; ".join(f"{r['prompt_tokens']} tokens: prefill {r['prefill_ms']:.1f} ms, decode {r['decode_ms_mean']:.2f} ms ({r['decode_tokens_per_s']:.2f}/s)" for r in bench_rows(t)))
say()
if os.path.exists(b + "/batch_b.out"):
    for title, a, z in (("12-bit (batch B)", "== bench, 12-bit layout", "== bench, tiered"), ("tiered, the default step (batch B)", "== bench, tiered, the default chunk", "== first use")):
        t = part(b + "/batch_b.out", a, z)
        say(f"  {title:<50} " + "; ".join(f"{r['prompt_tokens']} tokens: prefill {r['prefill_ms']:.1f} ms, decode {r['decode_ms_mean']:.2f} ms ({r['decode_tokens_per_s']:.2f}/s)" for r in bench_rows(t)))
say()
# the alternated timing (batch D: the build before the embedding change; batch G: the final build)
def alternated(path, title):
    t = open(path).read()
    say(title)
    vl = {}
    for m in re.finditer(r"round (\d) vllm (\{.*\})", t):
        vl[int(m.group(1))] = json.loads(m.group(2))
    rows = {}
    for m in re.finditer(r"round (\d) engine (\w+)\n((?:.*\n)*?)(?=round|\d\d:\d\d:\d\d done|\Z)", t):
        rows[(int(m.group(1)), m.group(2))] = {json.loads(x)["prompt_tokens"]: json.loads(x)["prefill_ms"] for x in m.group(3).split("\n") if x.startswith("{")}
    cols = {"vLLM+Glyd": {r: (v["prompts"]["1024"]["prefill_ms_median"], v["prompts"]["3500"]["prefill_ms_median"]) for r, v in vl.items()}}
    for mode in ("own", "blas"):
        cols["engine " + mode] = {r: (rows[(r, mode)][1024], rows[(r, mode)][3500]) for r in vl if (r, mode) in rows}
    for r in sorted(vl):
        say(f"  round {r}: " + "; ".join(f"{k} {c_[r][0]:.1f} / {c_[r][1]:.1f}" for k, c_ in cols.items() if r in c_))
    med = {k: (statistics.median(x[0] for x in c_.values()), statistics.median(x[1] for x in c_.values())) for k, c_ in cols.items() if c_}
    say("  median of rounds: " + "; ".join(f"{k} {m_[0]:.1f} / {m_[1]:.1f}" for k, m_ in med.items()))
    for k in med:
        if k != "vLLM+Glyd":
            say(f"  {k} / vLLM+Glyd: {med[k][0] / med['vLLM+Glyd'][0]:.3f} (1,024 tokens), {med[k][1] / med['vLLM+Glyd'][1]:.3f} (3,500)")


if os.path.exists(d + "/batch_d.out"):
    alternated(d + "/batch_d.out", "vLLM+Glyd (glyd@1) and the build before the embedding change, alternated, 3 rounds, prefill ms (1,024 / 3,500 tokens)")
if os.path.exists(gg + "/batch_g.out"):
    alternated(gg + "/batch_g.out", "vLLM+Glyd (glyd@1) and the final build, alternated, 3 rounds, prefill ms (1,024 / 3,500 tokens)")
# the A100 and the H100
for gpu, jd in JOBS.items():
    say(f"\n{gpu.upper()}: " + "; ".join(l.strip() for l in open(os.path.join(jd, "machine.txt")).read().split("\n")[1:2]))
    for name, path in (("tiered, cuBLAS", "tokens-mma-blas.json"), ("12-bit, cuBLAS", "tokens-mma12-blas.json"), ("12-bit, the engine's own products", "tokens-mma12-own.json")):
        if os.path.exists(os.path.join(jd, path)):
            say("  " + tokens_line(name, os.path.join(jd, path)))
    for name, path in (("tiered, cuBLAS", "score-mma-blas.json"), ("12-bit, cuBLAS", "score-mma12-blas.json"), ("12-bit, the engine's own products", "score-mma12-own.json")):
        if os.path.exists(os.path.join(jd, path)):
            say("  " + score_line(name, os.path.join(jd, path)))
    ref_tps, ref_what = BF16_ONE_USER[gpu]
    say(f"  vLLM bf16, one user, 1 x 1,024: {ref_tps} tokens a second ({ref_what})")
    for layout, label in (("mma", "tiered"), ("mma12", "12-bit")):
        cells = {}
        for mode in ("blas", "own", "lib"):
            p_ = os.path.join(jd, f"bench-{layout}-{mode}.out")
            if os.path.exists(p_):
                cells[mode] = {r["prompt_tokens"]: r for r in bench_rows(open(p_).read())}
        if not cells:
            continue
        tps = statistics.median(r[1024]["decode_tokens_per_s"] for r in cells.values())
        ms = statistics.median(r[1024]["decode_ms_mean"] for r in cells.values())
        say(f"  {label}: decode {ms:.2f} ms a token ({tps:.1f} tokens a second, {tps / ref_tps:.2f}x vLLM bf16); weights {next(iter(cells.values()))[1024]['weights_gib']:.2f} GiB")
        say("    prefill ms (1,024 / 3,500 tokens): " + "; ".join(f"{m_} {c_[1024]['prefill_ms']:.1f} / {c_[3500]['prefill_ms']:.1f}" for m_, c_ in cells.items()))
    for f_ in ("pack-default.out", "pack-mma.out", "kvcal.out", "first-use.out", "first-use-again.out"):
        p_ = os.path.join(jd, f_)
        if os.path.exists(p_):
            t_ = open(p_).read()
            say(f"  {f_[:-4]}: " + "; ".join(re.findall(r"(Packed in \S+|\([\d.]+ s\)|Tables made in \S+|Ready in \S+?:|Downloaded .*\))", ANSI.sub("", t_))[-3:]))
# the final build (batch E): cuBLAS and the engine's own products alternated, the embedding's time, the kernels of CUDA 13.0 and 12.8
if os.path.exists(e + "/batch_e.out"):
    t = open(e + "/batch_e.out").read()
    say("\nthe final build, cuBLAS and the engine's own products alternated (prefill ms, the last of 5 runs; the embedding's ms in the profile run)")
    emb = {}
    for m in re.finditer(r"round (\d) (blas|own)\n((?:.*\n)*?)(?=round|\d\d:\d\d:\d\d ==)", t):
        key = (int(m.group(1)), m.group(2))
        res = {}
        for l in m.group(3).split("\n"):
            if l.startswith("{"):
                j = json.loads(l)
                res[j["prompt_tokens"]] = j["prefill_ms"]
        em = re.findall(r"([\d.]+) ms\s+[\d.]+%\s+1 launches  embedding rows", m.group(3))
        emb[key] = (res, em)
    for (r, mode), (res, em) in sorted(emb.items()):
        say(f"  round {r} {mode:<5} 1024 tokens {res.get(1024)}, 3500 tokens {res.get(3500)}; embedding {em}")
    for mode in ("blas", "own"):
        v = [emb[(r, mode)][0] for r in (1, 2, 3) if (r, mode) in emb]
        if v:
            say(f"  {mode:<5} median of rounds: 1024 tokens {statistics.median(x[1024] for x in v)}, 3500 tokens {statistics.median(x[3500] for x in v)}")
    say("\nthe kernels built by nvcc 13.0 and by 12.8 (own products; prefill ms)")
    for m in re.finditer(r"round (\d) /home/ubuntu/engine/target-(dev7|dev12)/release/glyd-engine\n((?:.*\n)*?)(?=round|\d\d:\d\d:\d\d ==)", t):
        res = {json.loads(l)["prompt_tokens"]: json.loads(l)["prefill_ms"] for l in m.group(3).split("\n") if l.startswith("{")}
        say(f"  round {m.group(1)} CUDA {'13.0' if m.group(2) == 'dev7' else '12.8'}: 1024 tokens {res.get(1024)}, 3500 tokens {res.get(3500)}")
    for name in ("check", "score"):
        pass
    st = part(e + "/batch_e.out", "== the 8B through the glyd program", "== glyd doctor")
    real = re.findall(r"real\s+(\S+)", st)
    ready = re.findall(r"Ready in (\S+):", st)
    say(f"\nthe 8B through the glyd program from empty caches: real {real}; ready after {ready}")
    for title, a_, z_ in (("the server under mixed traffic, own products", "GLYD_ENGINE_GEMM=own: replies", "GLYD_ENGINE_GEMM=blas: replies"), ("the server under mixed traffic, cuBLAS", "GLYD_ENGINE_GEMM=blas: replies", "== every tile")):
        tx = part(e + "/batch_e.out", a_, z_)
        say(f"  {title}: " + " | ".join(l.strip() for l in tx.split("\n") if "rounds in" in l or l.startswith("all ok") or l.startswith("FAILED") or "cold replies" in l)[:600])
open(os.path.join(OUT, "numbers.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
