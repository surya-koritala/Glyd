"""The engine's greedy tokens and logprobs against a reference's (tools/ref_vllm.py), per prompt.

    python compare.py ENGINE.json REF.json [BATCH_KEY ...]       BATCH_KEY: batch | single (default both)

Reported: tokens equal from the start (the length of the common prefix, of the shorter list), prompts whose every token is equal,
the first token equal, and the logprob of each token over the common prefix: mean and largest absolute difference.
"""
import json
import sys


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(name, e_tok, e_lp, r_tok, r_lp):
    n = len(e_tok)
    common = [prefix(a, b) for a, b in zip(e_tok, r_tok)]
    full = sum(1 for a, b, c in zip(e_tok, r_tok, common) if c == min(len(a), len(b)) == max(len(a), len(b)))
    first = sum(1 for a, b in zip(e_tok, r_tok) if a and b and a[0] == b[0])
    total = sum(len(a) for a in e_tok)
    from_start = sum(common)
    same_pos = sum(1 for a, b in zip(e_tok, r_tok) for x, y in zip(a, b) if x == y)
    diffs = [abs(x - y) for a, b, ea, ra, c in zip(e_tok, r_tok, e_lp, r_lp, common) for x, y in zip(ea[:c], ra[:c])]
    mean = sum(diffs) / len(diffs) if diffs else float("nan")
    print(f"{name}: {full} of {n} prompts equal in every token; first token equal {first} of {n}; {from_start} of {total} tokens equal from the start of their prompt ({same_pos} at the same position); tokens equal from the start {common}; "
          f"logprob difference over those: mean {mean:.4f}, largest {max(diffs) if diffs else float('nan'):.4f}")
    return {"full": full, "n": n, "first": first, "from_start": from_start, "total": total, "same_position": same_pos, "common": common, "mean": mean, "max": max(diffs) if diffs else None}


def main(engine, ref, *keys):
    e = json.load(open(engine))
    r = json.load(open(ref))
    out = {}
    for k in keys or ("batch", "single", "long"):
        if k == "long":
            if "long" in r and "long_tokens" in e:
                out[k] = compare(f"engine against {r['mode']} (long prompts {[len(p) for p in r['long_prompts']]})", e["long_tokens"], e["long_logprobs"], r["long"]["tokens"], r["long"]["logprobs"])
            continue
        out[k] = compare(f"engine against {r['mode']} ({k})", e["tokens"], e["logprobs"], r[k]["tokens"], r[k]["logprobs"])
    if "score_top1" in e and "score_top1" in r:  # teacher-forced over a text: the largest-logit token at each position, and the logprob of the next token
        n = min(len(e["score_top1"]), len(r["score_top1"]))
        same = sum(1 for a, b in zip(e["score_top1"], r["score_top1"]) if a == b)
        d = sorted(abs(a - b) for a, b in zip(e["score_lp"][:n], r["score_lp"][:n]))
        print(f"teacher-forced over {n} positions: top-1 token equal at {same} ({100 * same / n:.2f}%); logprob of the next token: mean difference {sum(d) / n:.4f}, median {d[n // 2]:.4f}, 99th percentile {d[int(n * 0.99)]:.4f}, largest {d[-1]:.4f}")
        out["score"] = {"n": n, "top1_equal": same, "mean": sum(d) / n, "median": d[n // 2], "p99": d[int(n * 0.99)], "max": d[-1]}
    if "batch" in r and "single" in r:  # the control: the reference against itself
        out["control"] = compare(f"{r['mode']} batch against {r['mode']} one at a time", r["batch"]["tokens"], r["batch"]["logprobs"], r["single"]["tokens"], r["single"]["logprobs"])
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
