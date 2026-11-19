"""The lossless KV cache's outputs against vLLM's own (kv: lossless against stock), on the measures of how far vLLM's own outputs move when only the batch
changes: Qwen3-8B, bf16 weights, eager, prefix caching and chunked prefill off, each run in a vLLM of its own (check_vllm.py's children).

- free: six prompts of 600 to 4,100 tokens, 256 greedy tokens each, all in one batch: the continuations the same as stock's, the token each first differs at, and the
  chosen tokens' logprob differences on the common prefix (median, p99, maximum); beside stock's own (one prompt at a time against all six at once);
- forced: 64 prompts of 64 to 2,048 tokens, each with stock's own 256-token continuation forced (a logits processor samples the teacher's token whatever the model
  says), so every decode step reads the cache as it stands: the share of positions where two runs' own top-1 tokens are the same (the first position, a prompt's own logits,
  left out) and the mean |logprob difference| of the teacher's token; kv against stock's batch, and stock's own (8 sequences at a time against 64).

    python kv_agree.py [--out DIR] [--tables FILE] [--study-dir DIR] [--quick] [MODEL]

--study-dir: the prompts of the KV study (research/kv-cache/prompts.py: chat, Wikipedia markup, Python, Dickens ...) for the six, else this repository's documents. --tables:
the model's base tables (else made as a first start makes them)."""
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_vllm import DOCS, REPO, run, same_prefix  # noqa: E402

STUDY = ["chat_qa", "doc_qa_3k", "wiki_4k", "code_4k", "list_1k", "table_2k"]  # the six of the design's divergence runs


def token_texts(tok, study_dir):
    """The texts the prompts are cut from: the study's (where given) and this repository's documents."""
    out = []
    if study_dir:
        sys.path.insert(0, study_dir)
        import prompts as PR

        out += [(pid, tok(PR.text_of(pid, tok), add_special_tokens=False).input_ids) for pid in ["chat_qa", "doc_qa_3k", "wiki_4k", "code_4k", "prose_8k", "list_1k", "table_2k", "json_3k"]]
    for name in DOCS:
        path = os.path.join(REPO, name)
        if os.path.exists(path):
            out.append((name, tok.encode(open(path, encoding="utf-8").read())))
    return out


def six(tok, study_dir):
    t = dict(token_texts(tok, study_dir))
    if study_dir:
        return [(p, t[p][:4100]) for p in STUDY]
    out = []
    for i, n in enumerate((4100, 3000, 2000, 1200, 800, 614)):  # (six windows of this repository's documents, the longest of them first)
        name = next(d for d in t if len(t[d]) >= n + 1000 * i)
        out.append((f"{name}@{1000 * i}", t[name][1000 * i : 1000 * i + n]))
    return out


def sixty_four(tok, study_dir):
    """64 windows of the texts: a start and a length (64 to 2,048 tokens) drawn from a seeded generator."""
    r = random.Random(0)
    texts = token_texts(tok, study_dir)
    out = []
    for i in range(64):
        name, ids = texts[i % len(texts)]
        n = min(len(ids) - 1, r.choice([64, 128, 256, 512, 1024, 2048]))
        o = r.randrange(0, len(ids) - n)
        out.append((f"{name}#{i}", ids[o : o + n]))
    return out


def pct(x, q):
    x = sorted(x)
    return x[min(len(x) - 1, int(q * len(x)))] if x else float("nan")


def free(a, b):
    """Continuations b against a: the prompts whose continuations are identical, each one's first differing token (n where none), and the chosen tokens' logprob differences on the common prefix."""
    same, first, d = 0, [], []
    for x, y, lx, ly in zip(a["tokens"], b["tokens"], a["logprobs"], b["logprobs"]):
        k = same_prefix(x, y)
        same += k == len(x) == len(y)
        first.append(k)
        d += [abs(u - v) for u, v in zip(lx[:k], ly[:k])]
    return {"identical": same, "of": len(a["tokens"]), "first_divergence": first, "dlogprob_median": statistics.median(d) if d else 0.0, "dlogprob_p99": pct(d, 0.99), "dlogprob_max": max(d) if d else 0.0, "tokens": len(d)}


def forced(a, b):
    """Two forced runs: the share of positions (past each prompt's first) where their top-1 tokens are the same, and the mean |logprob difference| of the teacher's token."""
    eq = n = 0
    d = []
    for x, y, lx, ly in zip(a["top1"], b["top1"], a["logprobs"], b["logprobs"]):
        eq += sum(u == v for u, v in zip(x[1:], y[1:]))
        n += len(x) - 1
        d += [abs(u - v) for u, v in zip(lx[1:], ly[1:])]
    return {"top1_agreement": eq / n, "positions": n, "positions_differing": n - eq, "mean_abs_dlogprob": sum(d) / len(d), "dlogprob_p99": pct(d, 0.99)}


def main():
    args = sys.argv[1:]
    out = args[args.index("--out") + 1] if "--out" in args else "kv_agree_results"
    study = args[args.index("--study-dir") + 1] if "--study-dir" in args else None
    tables = args[args.index("--tables") + 1] if "--tables" in args else None
    quick = "--quick" in args
    model = next((a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--out", "--tables", "--study-dir"))), "Qwen/Qwen3-8B")
    os.makedirs(out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    env = {"GLYD_KV_TABLES": tables} if tables else {}
    A, B = six(tok, study), sixty_four(tok, study)
    if quick:
        A, B = A[:3], B[:16]
    n = 64 if quick else 256
    res = {"model": model, "six": [{"name": nm, "tokens": len(i)} for nm, i in A], "sixty_four": {"prompts": len(B), "tokens": sum(len(i) for _, i in B)}}
    R = lambda name, spec: run({"kv_agree": True, "model": model, "n": n, **spec}, out, name, env)
    a = [i for _, i in A]
    sb, so, kb = R("A-stock-batch", {"kv": False, "max_num_seqs": 8, "prompts": a}), R("A-stock-one", {"kv": False, "max_num_seqs": 1, "prompts": a}), R("A-kv-batch", {"kv": True, "max_num_seqs": 8, "prompts": a})
    res["free"] = {"stock_one_vs_stock_batch": free(sb, so), "kv_vs_stock_batch": free(sb, kb), "kv_blocks": kb["blocks"], "stock_blocks": sb["blocks"], "kv_status": kb.get("kv_status")}
    b = [i for _, i in B]
    t = R("B-teacher", {"kv": False, "max_num_seqs": 64, "prompts": b})
    teacher = t["tokens"]
    fs = R("B-forced-stock-batch", {"kv": False, "max_num_seqs": 64, "prompts": b, "teacher": teacher})
    fsm = R("B-forced-stock-small", {"kv": False, "max_num_seqs": 8, "prompts": b, "teacher": teacher})
    fk = R("B-forced-kv-batch", {"kv": True, "max_num_seqs": 64, "prompts": b, "teacher": teacher})
    res["forced"] = {"stock_small_vs_stock_batch": forced(fs, fsm), "kv_vs_stock_batch": forced(fs, fk), "kv_vs_stock_small": forced(fsm, fk), "kv_status": fk.get("kv_status"),
                     "teacher_followed": {"stock": fs["tokens"] == teacher, "kv": fk["tokens"] == teacher}}
    json.dump(res, open(os.path.join(out, "agree.json"), "w"), indent=1)
    f = res["free"]
    print(f"free ({len(A)} prompts, {n} tokens): stock one at a time against stock's batch: identical {f['stock_one_vs_stock_batch']['identical']} of {len(A)}, first differences {f['stock_one_vs_stock_batch']['first_divergence']}, |dlogprob| median {f['stock_one_vs_stock_batch']['dlogprob_median']:.4f} p99 {f['stock_one_vs_stock_batch']['dlogprob_p99']:.4f} max {f['stock_one_vs_stock_batch']['dlogprob_max']:.4f}")
    print(f"      kv against stock's batch: identical {f['kv_vs_stock_batch']['identical']} of {len(A)}, first differences {f['kv_vs_stock_batch']['first_divergence']}, |dlogprob| median {f['kv_vs_stock_batch']['dlogprob_median']:.4f} p99 {f['kv_vs_stock_batch']['dlogprob_p99']:.4f} max {f['kv_vs_stock_batch']['dlogprob_max']:.4f}")
    g = res["forced"]
    for k in ("stock_small_vs_stock_batch", "kv_vs_stock_batch", "kv_vs_stock_small"):
        print(f"forced ({len(B)} prompts, {g[k]['positions']:,} positions) {k}: top-1 agreement {g[k]['top1_agreement'] * 100:.3f}% ({g[k]['positions_differing']} positions differ), mean |dlogprob| {g[k]['mean_abs_dlogprob']:.5f}, p99 {g[k]['dlogprob_p99']:.4f}")
    print(f"KV cache {kb['blocks'] * kb['block_size']:,} tokens, stock's {sb['blocks'] * sb['block_size']:,}; arena status {res['forced']['kv_status']}")


if __name__ == "__main__":
    main()
