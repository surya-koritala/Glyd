"""Every figure of the engine-spike record, computed from the data files beside its README.

    python record.py RECORD_DIR > RECORD_DIR/numbers.txt

RECORD_DIR holds vllm/ (vLLM's references, speeds, cache sizes, chunking controls), engine/ (the engine's outputs and the proofs' log), and small/, ctx8k/.
A missing file prints as missing: the record is built from whatever the runs left.
"""
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "."


def load(*p):
    f = os.path.join(D, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def say(*a):
    print(*a)


def head(t):
    say()
    say(f"== {t}")


def cmp_tokens(ename, rname, e, r, keys=("batch", "single", "long")):
    if e is None or r is None:
        say(f"{ename} against {rname}: missing")
        return
    for k in keys:
        if k == "long":
            if "long" in r and "long_tokens" in e:
                compare.compare(f"{ename} against {rname}, long prompts {[len(p) for p in r['long_prompts']]}", e["long_tokens"], e["long_logprobs"], r["long"]["tokens"], r["long"]["logprobs"])
        else:
            compare.compare(f"{ename} against {rname}, {k}", e["tokens"], e["logprobs"], r[k]["tokens"], r[k]["logprobs"])


def diverge(name, et, el, rt, rl):
    """Where each prompt's tokens part, and the logprob each run gave its own choice there (both are the largest: a near tie where they are close)."""
    out = []
    for i, (a, b, x, y) in enumerate(zip(et, rt, el, rl)):
        n = next((j for j, (p, q) in enumerate(zip(a, b)) if p != q), None)
        out.append(f"prompt {i}: same" if n is None else f"prompt {i}: parts at token {n} ({a[n]} at {x[n]:.3f}, {b[n]} at {y[n]:.3f})")
    say(f"{name}: " + "; ".join(out))


def score_stats(name, a, b, cliff=1.0):
    """Teacher-forced: the top-1 token of two runs at each position and the logprob of the next token."""
    if a is None or b is None:
        say(f"{name}: missing")
        return None
    n = min(len(a["score_top1"]), len(b["score_top1"]))
    same = sum(1 for x, y in zip(a["score_top1"], b["score_top1"]) if x == y)
    d = [abs(x - y) for x, y in zip(a["score_lp"][:n], b["score_lp"][:n])]
    big = [i for i, x in enumerate(d) if x > cliff]
    keep = [x for i, x in enumerate(d) if i not in big]
    s = sorted(d)
    line = f"{name}: top-1 equal at {same} of {n} ({100 * same / n:.2f}%); |logprob difference| mean {sum(d) / n:.4f}, median {s[n // 2]:.4f}, 99th pct {s[int(n * 0.99)]:.4f}, largest {s[-1]:.4f}"
    if big:
        line += f"; {len(big)} position(s) over {cliff}: " + ", ".join(f"{i} ({a['score_lp'][i]:.2f} against {b['score_lp'][i]:.2f})" for i in big)
        mk = sum(1 for i, (x, y) in enumerate(zip(a["score_top1"], b["score_top1"])) if x == y and i not in big)
        line += f"; without them: mean {sum(keep) / len(keep):.4f}, top-1 equal at {mk} of {n - len(big)} ({100 * mk / (n - len(big)):.2f}%)"
    say(line)
    return d


def bench_lines(path):
    out = []
    p = os.path.join(D, *path)
    if not os.path.exists(p):
        return out
    for l in open(p):
        l = l.strip()
        if l.startswith("{") and "decode_ms_mean" in l:
            out.append(json.loads(l))
    return out


def section(text, start, end=None):
    """The part of the proofs' log from the line starting `start` to the next `== ` line."""
    m = re.search(r"^== " + re.escape(start) + r".*?$(.*?)(?=^== |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


def jlines(text):
    return [json.loads(l) for l in text.splitlines() if l.startswith("{") and "decode_ms_mean" in l]


def main():
    refs = {m: load("vllm", f"ref-8b-{m}.json") for m in ("glyd1", "kvglyd", "bf16")}
    eng = load("engine", "eng-8b.json")
    escore = load("engine", "eng-8b-score.json")
    if escore and eng is not None:
        eng.update(escore)

    head("tokens: the engine against vLLM 0.30 (Qwen3-8B, 8 prompts of up to 31 tokens and 3 long ones, 64 greedy tokens each)")
    for m, r in refs.items():
        cmp_tokens("engine", f"vLLM {m}", eng, r)
    for m in ("glyd1",):
        r = refs[m]
        if r and eng:
            for k in ("batch", "single"):
                diverge(f"engine against vLLM {m}, {k}", eng["tokens"], eng["logprobs"], r[k]["tokens"], r[k]["logprobs"])
            diverge(f"vLLM {m} batch against its one at a time", r["batch"]["tokens"], r["batch"]["logprobs"], r["single"]["tokens"], r["single"]["logprobs"])
    for m, r in refs.items():
        if r:
            compare.compare(f"vLLM {m} batch against its one at a time", r["batch"]["tokens"], r["batch"]["logprobs"], r["single"]["tokens"], r["single"]["logprobs"])

    head("long prompts: the engine, vLLM at 8,192 tokens a step and vLLM at 1,024 (the engine's step)")
    ctrl = load("vllm", "ref-8b-glyd1-b1024.json")
    if ctrl and refs["glyd1"]:
        r = refs["glyd1"]
        compare.compare("vLLM glyd1 at 1,024 against vLLM glyd1 at 8,192 (long)", ctrl["long"]["tokens"], ctrl["long"]["logprobs"], r["long"]["tokens"], r["long"]["logprobs"])
        if eng:
            compare.compare("engine against vLLM glyd1 at 1,024 (long)", eng["long_tokens"], eng["long_logprobs"], ctrl["long"]["tokens"], ctrl["long"]["logprobs"])
        for k in ("batch", "single"):
            compare.compare(f"vLLM glyd1 at 1,024 against vLLM glyd1 at 8,192 ({k})", ctrl[k]["tokens"], ctrl[k]["logprobs"], r[k]["tokens"], r[k]["logprobs"])
        if eng:
            diverge("engine against vLLM glyd1 at 8,192, long", eng["long_tokens"], eng["long_logprobs"], r["long"]["tokens"], r["long"]["logprobs"])
            diverge("engine against vLLM glyd1 at 1,024, long", eng["long_tokens"], eng["long_logprobs"], ctrl["long"]["tokens"], ctrl["long"]["logprobs"])
            diverge("vLLM glyd1 at 1,024 against at 8,192, long", ctrl["long"]["tokens"], ctrl["long"]["logprobs"], r["long"]["tokens"], r["long"]["logprobs"])
    else:
        say("missing: vllm/ref-8b-glyd1-b1024.json")

    head("teacher-forced scores over the first 1,536 tokens of a text (1,535 positions)")
    if eng and refs["glyd1"] and "score_top1" in eng:
        score_stats("engine against vLLM glyd1 (8,192 a step)", eng, refs["glyd1"])
        score_stats("engine against vLLM kvglyd (8,192 a step)", eng, refs["kvglyd"])
        score_stats("engine against vLLM bf16 (8,192 a step)", eng, refs["bf16"])
    for m in ("glyd1", "bf16"):
        base = refs[m]
        for step in (1024, 512, 256):
            s = load("vllm", f"score-{m}-{step}.json")
            if s and base:
                score_stats(f"vLLM {m} at {step} a step against vLLM {m} at 8,192", s, base)
                if eng and "score_top1" in eng:
                    score_stats(f"engine against vLLM {m} at {step} a step", eng, s)
    if refs["glyd1"] and refs["bf16"]:
        score_stats("vLLM glyd1 against vLLM bf16 (the weights' own difference)", refs["glyd1"], refs["bf16"])

    head("speed: one user, decode ms per token (1,024-token prompt, 256 new tokens unless said)")
    proofs = os.path.join(D, "engine", "proofs.out")
    text = open(proofs).read() if os.path.exists(proofs) else ""
    for title, key in (("graph x5", "bench 1024/256, graph"), ("eager", "bench 1024/256, eager"), ("128/256", "bench 128/256"), ("3500/128", "bench 3500/128")):
        js = jlines(section(text, key))
        if js:
            ms = [j["decode_ms_mean"] for j in js]
            say(f"engine {title}: decode ms/token {', '.join(f'{x:.3f}' for x in ms)}; median {statistics.median(ms):.3f} ({1000 / statistics.median(ms):.2f} tokens/s); prefill ms {', '.join('%.1f' % j['prefill_ms'] for j in js)}; p99 {', '.join('%.2f' % j['decode_ms_p99'] for j in js)}")
        else:
            say(f"engine {title}: missing")
    for m in ("glyd1", "kvglyd", "bf16"):
        s = load("vllm", f"speed-8b-{m}.json")
        if s:
            say(f"vLLM {m}: decode ms/token {s['decode_ms']} median {s['decode_ms_median']} ({s['decode_tokens_per_s']} tokens/s); prefill ms median {s['prefill_ms_median']}; LLM() construction {s['load_s']} s")
        s = load("vllm", f"speed3500-{m}.json")
        if s:
            say(f"vLLM {m} (3,500-token prompt, {s['new_tokens']} new): decode ms/token median {s['decode_ms_median']} ({s['decode_tokens_per_s']} tokens/s); prefill ms median {s['prefill_ms_median']}")
    js = jlines(section(text, "bench 1024/256, graph"))
    if js:
        e = statistics.median(j["decode_ms_mean"] for j in js)
        for m in ("glyd1", "kvglyd", "bf16"):
            s = load("vllm", f"speed-8b-{m}.json")
            if s:
                say(f"engine / vLLM {m}: {s['decode_ms_median'] / e:.3f}x the tokens a second")

    for m in ("glyd1", "kvglyd"):
        s = load("vllm", f"bracket-speed-8b-{m}.json")
        if s:
            say(f"vLLM {m} again after the engine's runs: decode ms/token {s['decode_ms']} median {s['decode_ms_median']} ({s['decode_tokens_per_s']} tokens/s); prefill ms median {s['prefill_ms_median']}")
    pr = os.path.join(D, "engine", "pair.out")
    if os.path.exists(pr):
        say("-- the engine and vLLM glyd1 alternating in one hold of the box (rounds of vLLM, then the engine, 1,024 in, 256 out)")
        eng_r, vl_r = {}, {}
        for l in open(pr):
            m = re.match(r"round (\d+) (engine|vllm) (\{.*\})", l)
            if m:
                j = json.loads(m.group(3))
                (eng_r if m.group(2) == "engine" else vl_r).setdefault(int(m.group(1)), []).append(j)
            elif l.startswith("warm-up engine"):
                eng_r.setdefault(0, []).append(json.loads(l.split(" ", 2)[2]))
        for r in sorted(eng_r):
            es = [j["decode_ms_mean"] for j in eng_r[r]]
            line = f"round {r}: engine decode ms/token {', '.join('%.3f' % x for x in es)} (median {statistics.median(es):.3f})"
            if r in vl_r:
                v = vl_r[r][0]
                line += f"; vLLM glyd1 before it {', '.join('%.3f' % x for x in v['decode_ms'])} (median {v['decode_ms_median']})"
            say(line)
        pairs = [(statistics.median(j["decode_ms_mean"] for j in eng_r[r]), vl_r[r][0]["decode_ms_median"]) for r in sorted(vl_r) if r in eng_r]
        if pairs:
            say(f"over the rounds: engine median {statistics.median(p[0] for p in pairs):.3f} ms/token, vLLM glyd1 median {statistics.median(p[1] for p in pairs):.3f}; the engine is {statistics.median(p[1] / p[0] for p in pairs):.3f}x the tokens a second (rounds: {', '.join('%.3f' % (p[1] / p[0]) for p in pairs)})")
    ab = os.path.join(D, "engine", "ab.out")
    if os.path.exists(ab):
        rows = {}
        for l in open(ab):
            if l.startswith("final"):
                k, j = l.split(" ", 1)
                rows.setdefault(k, []).append(json.loads(j)["decode_ms_mean"])
        for k, v in rows.items():
            say(f"binary of {k}, alternating with the other: decode ms/token {', '.join('%.3f' % x for x in v)} (median {statistics.median(v):.3f})")

    head("memory and start")
    for l in section(text, "info").splitlines():
        say(l)
    for l in section(text, "start").splitlines():
        say(l)
    say("start from a shell: " + " ".join(re.findall(r"real\s+\S+", section(text, "start"))))
    for k in ("bandwidth", "profile"):
        say(f"-- {k}")
        for l in section(text, k).splitlines():
            if l.strip() and not l.startswith("{"):
                say(l)
    # the products' effective bandwidth: Qwen3-8B's matrices (36 layers, hidden 4096, 32 + 8 heads of 128, intermediate 12288, vocabulary 151936) at the bits a weight `info` printed, over the profile's times
    bits = dict((k, float(v)) for k, v in re.findall(r"(\w+) (\d+\.\d+)", section(text, "info").split("bits a weight:")[-1])) if "bits a weight:" in section(text, "info") else {}
    prof = {}
    for l in section(text, "profile").splitlines():
        m = re.match(r"\s*([\d.]+) ms\s+[\d.]+%\s+\d+ launches\s+(.*)$", l)
        if m:
            prof[m.group(2).strip()] = float(m.group(1))
    names = {"qkv": "qkv product", "o": "o product", "gate_up": "gate and up product", "down": "down product", "lm_head": "LM head product"}
    dims = {"qkv": (6144 * 4096, 36), "o": (4096 * 4096, 36), "gate_up": (24576 * 4096, 36), "down": (4096 * 12288, 36), "lm_head": (151936 * 4096, 1)}
    if bits and prof:
        tb = tt = 0.0
        for k, (n, layers) in dims.items():
            gb = n * layers * bits[k] / 8 / 1e9
            ms = prof.get(names[k])
            if ms:
                say(f"{k} product: {gb:.3f} GB of weights in {ms:.3f} ms = {gb / ms * 1e3:.0f} GB/s")
                tb, tt = tb + gb, tt + ms
        if tt:
            say(f"the products together: {tb:.2f} GB in {tt:.2f} ms = {tb / tt * 1e3:.0f} GB/s of the step's {prof.get('total', 0):.2f} ms")
    co = os.path.join(D, "engine", "clock.out")
    if os.path.exists(co):
        cj = [json.loads(l) for l in open(co) if l.startswith("{") and "decode_ms_mean" in l]
        if cj:
            say(f"engine, three runs starting with the GPU at 61 C (clock.out): decode ms/token {', '.join('%.3f' % j['decode_ms_mean'] for j in cj)} (median {statistics.median(j['decode_ms_mean'] for j in cj):.3f}); prefill ms {', '.join('%.1f' % j['prefill_ms'] for j in cj)}")
        for l in open(co):
            if "same bytes" in l:
                say("repacked model run again: " + l.strip())
        rp, rs = load("engine", "eng-8b-repacked.json"), load("engine", "eng-8b-repacked-score.json")
        if rp and eng:
            say("repacked model run again: tokens, logprobs, long tokens and long logprobs equal to the final proofs': " + str(all(rp[k] == eng[k] for k in ("tokens", "logprobs", "long_tokens", "long_logprobs"))))
    ck = os.path.join(D, "engine", "clock.csv")
    if os.path.exists(ck):
        import csv

        rows = list(csv.reader(open(ck)))[1:]
        busy = [(int(r[1].split()[0]), float(r[3].split()[0]), int(r[4])) for r in rows if len(r) > 5 and float(r[3].split()[0]) >= 71]
        if busy:
            say(f"-- clock.csv, the GPU while the engine decodes ({len(busy)} one-second samples at the 72 W cap): SM clock {min(b[0] for b in busy)} to {max(b[0] for b in busy)} MHz (median {statistics.median(b[0] for b in busy):.0f}; max 2,040), power {min(b[1] for b in busy):.1f} to {max(b[1] for b in busy):.1f} W (limit 72), {min(b[2] for b in busy)} to {max(b[2] for b in busy)} C")
    sz = os.path.join(D, "engine", "sizes.txt")
    if os.path.exists(sz):
        say("-- sizes")
        say(open(sz).read().strip())

    head("vLLM's KV cache, tokens (0.9 of the GPU, max_model_len 4,096)")
    for f in sorted(glob.glob(os.path.join(D, "vllm", "cache", "*.log"))):
        for l in open(f):
            m = re.search(r'\{"model".*"kv_tokens": \d+\}', l)
            if m:
                j = json.loads(m.group(0))
                say(f"{os.path.basename(f)}: {j['mode']} at {j['max_num_batched_tokens']} a step: {j['kv_tokens']} tokens ({j['blocks']} blocks)")

    head("other sizes of Qwen3 (the L4's, 4,096-token tables)")
    for d in ("0.6b", "1.7b", "4b"):
        e = load("small", f"eng-{d}.json")
        r = load("small", f"ref-{d}-glyd1.json")
        sc = load("small", f"eng-{d}-score.json")
        if e is None or r is None:
            say(f"{d}: missing")
            continue
        cmp_tokens(f"engine {d}", f"vLLM glyd1 {d}", e, r, keys=("batch", "single", "long"))
        compare.compare(f"vLLM glyd1 {d} batch against its one at a time", r["batch"]["tokens"], r["batch"]["logprobs"], r["single"]["tokens"], r["single"]["logprobs"])
        if sc and "score_top1" in r:
            score_stats(f"engine {d} against vLLM glyd1 {d}", sc, r)
        js = bench_lines(("small", f"bench-{d}.json"))
        for j in js:
            say(f"engine {d} bench {j['prompt_tokens']}/{j['new_tokens']}: decode {j['decode_ms_mean']} ms/token ({j['decode_tokens_per_s']} tokens/s), prefill {j['prefill_ms']} ms, load {j['load_s']} s, weights {j['weights_gib']} GiB, cache {j['cache_tokens']} tokens")
        lg = os.path.join(D, "small", f"eng-{d}.log")
        if os.path.exists(lg):
            for l in open(lg):
                if "weights" in l and "bf16" in l:
                    say("  " + l.strip())
                    break

    head("an 8,192-position context (Qwen3-8B)")
    for j in bench_lines(("ctx8k", "bench-7500.json")):
        say(f"engine 7500/128 at 8,192 positions: decode {j['decode_ms_mean']} ms/token ({j['decode_tokens_per_s']} tokens/s), prefill {j['prefill_ms']} ms, cache {j['cache_tokens']} tokens")
    info = os.path.join(D, "ctx8k", "info.txt")
    if os.path.exists(info):
        for l in open(info).read().splitlines()[-4:]:
            say(l)


if __name__ == "__main__":
    main()
