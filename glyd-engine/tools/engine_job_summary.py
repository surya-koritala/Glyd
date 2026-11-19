"""The results of jobs/engine_gpu_job.sh as a short table: python3 engine_job_summary.py RESULTS_DIR REF.json
Tokens against vLLM+Glyd's reference (made on an L4), the teacher-forced top-1 figure (the one that carries over between GPUs), prefill and decode by the way a long prompt is multiplied, and the first-run times."""
import glob
import json
import os
import re
import statistics
import sys

R, REF = sys.argv[1], sys.argv[2]
ref = json.load(open(REF)) if os.path.exists(REF) else None


def name(p, prefix):
    return os.path.basename(p)[len(prefix) : -len(".json")]


def read(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


machine = open(os.path.join(R, "machine.txt")).read().split("\n") if os.path.exists(os.path.join(R, "machine.txt")) else []
print("machine:", "; ".join(l for l in machine[1:3] if l.strip()))
if ref:
    print("\ntokens against vLLM+Glyd's (an L4's), 8 short prompts of 64 tokens and 3 long ones (positions equal / first difference):")
    for p in sorted(glob.glob(os.path.join(R, "tokens-*.json"))):
        e = read(p)
        if not e:
            continue
        rt, et = ref["single"]["tokens"], e["tokens"]
        same = sum(1 for a, b in zip(rt, et) if a == b)
        pos = sum(sum(1 for x, y in zip(a, b) if x == y) for a, b in zip(rt, et))
        line = f"  {name(p, 'tokens-'):<12} short: {same}/{len(rt)} identical, {pos}/{sum(len(a) for a in rt)} positions"
        if "long_tokens" in e:
            first = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None) for a, b in zip(ref["long"]["tokens"], e["long_tokens"])]
            line += f"; long: {sum(1 for f in first if f is None)}/3 identical (first differences {first})"
        print(line)
    print("\nteacher-forced top-1 over 1,535 positions against vLLM+Glyd's (98.5-98.7% on the L4 in every configuration):")
    for p in sorted(glob.glob(os.path.join(R, "score-*.json"))):
        e = read(p)
        if not e:
            continue
        eq = sum(1 for a, b in zip(ref["score_top1"], e["score_top1"]) if a == b)
        d = statistics.mean(abs(a - b) for a, b in zip(ref["score_lp"], e["score_lp"]))
        print(f"  {name(p, 'score-'):<12} {eq}/{len(ref['score_top1'])} = {100 * eq / len(ref['score_top1']):.2f}%, mean |logprob difference| {d:.4f}")
print("\nprefill (ms, the last of 3 timed runs) and decode (tokens a second) by how a long prompt is multiplied (blas: cuBLAS, own: the engine's kernels, lib: the library's routes):")
for p in sorted(glob.glob(os.path.join(R, "bench-*.out"))):
    rows = []
    for l in open(p):
        if l.startswith("{"):
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
    if rows:
        print(f"  {os.path.basename(p)[6:-4]:<14} " + "; ".join(f"{r['prompt_tokens']} tokens: prefill {r['prefill_ms']:.1f} ms, decode {r['decode_tokens_per_s']:.2f} tok/s" for r in rows) + f"; weights {rows[0].get('weights_gib', 0):.2f} GiB")
print("\nsetup times:")
for f, pat in (("pack-default.out", r"Packed in (\S+)"), ("kvcal.out", r"\((\d+) s\)"), ("first-use.out", r"Ready in (\S+)"), ("first-use-again.out", r"Ready in (\S+)")):
    p = os.path.join(R, f)
    if os.path.exists(p):
        t = open(p).read()
        m = re.findall(pat, t)
        real = re.findall(r"real\s+(\S+)", t)
        print(f"  {f[:-4]:<18} {m[-1] if m else ''} {('real ' + real[-1]) if real else ''}")
for f in ("job.log",):
    p = os.path.join(R, f)
    if os.path.exists(p):
        print("\n" + "".join(l for l in open(p) if "skipped" in l or "FAILED" in l) or "no step skipped, none failed")
