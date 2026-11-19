"""The prefix-hit scenario of check_vllm's extension checks, with controls, and a layer by layer comparison.

run: python kv_prefix_control.py run MODEL stock|kv OUT.json [--order fwd|rev|fresh:N] [--only i,j] [--verify] [--exact] [--decode-splits 1] [--dump i:R1,i:R2 --layers 0,1,2]
  For each prompt (the check's: 8 short ones and this repository's documents), one request at a time with prefix caching on: R1 the whole prompt (one token), then R2 its first 80% (to a
  multiple of 16) and the first 300 tokens of the next prompt, 64 greedy tokens. Written: for every request its tokens, logprobs and cached tokens, and for every prefill step a digest
  (sha1 of the bytes) of the query, key, value and output of each attention layer. fwd is the check's order of the prompts; rev has the 8 short ones in the other order (another history for the
  documents); fresh:N is the prompt N alone in a new engine.
dumps: python kv_prefix_control.py dumps A-dump.npz B-dump.npz
  The tensors two runs saved with --dump, layer by layer: the elements that differ and the largest difference.
compare: python kv_prefix_control.py compare A.json B.json
  For each request the tokens' agreement, and the first layer and tensor whose digests differ (the inputs of a layer are the outputs of the one before: the first is where two runs part).
"""

import hashlib
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def digest(t):
    import torch

    return hashlib.sha1(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:20]


def run(argv):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import check_vllm as cv

    model, kind, out = argv[0], argv[1], argv[2]
    opt = lambda name, default=None: argv[argv.index(name) + 1] if name in argv else default
    order = opt("--order", "fwd")
    only = [int(x) for x in opt("--only", "").split(",") if x]
    dump = [x for x in opt("--dump", "").split(",") if x]
    dump_layers = {int(x) for x in opt("--layers", "0,1,2").split(",")}
    tok = AutoTokenizer.from_pretrained(model)
    prompts = cv.kv_prompts(tok, 4096, 6, None)
    idx = list(range(len(prompts)))
    if order == "rev":
        idx = idx[:8][::-1] + idx[8:]
    elif order.startswith("fresh:"):
        idx = [int(order.split(":")[1])]
    if only:
        idx = [i for i in idx if i in only]
    mml = 4096 + 300 + 128
    kw = dict(model=model, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.9, seed=0, max_model_len=mml, max_num_seqs=1, enable_prefix_caching=True, enable_chunked_prefill=False, max_num_batched_tokens=mml)
    if kind == "kv":
        kw.update(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless", "verify": "--verify" in argv, **({"exact": True} if "--exact" in argv else {})}})
        if "--verify" in argv:
            kw["num_gpu_blocks_override"] = int(opt("--blocks", "700"))
    if kind == "stock" and opt("--decode-splits"):  # (the control: vLLM's own FlashAttention with another split of the keys in its decode steps: another rounding there, nothing else changed)
        from vllm.v1.attention.backends import flash_attn as fa

        nsp, orig = int(opt("--decode-splits")), fa.FlashAttentionMetadataBuilder.build

        def build(self, *a, **k):
            md = orig(self, *a, **k)
            if md.max_query_len == 1:
                md.max_num_splits = nsp
            return md

        fa.FlashAttentionMetadataBuilder.build = build
    llm = LLM(**kw)
    S = {"label": None, "step": 0, "seen": set(), "dig": {}, "dump": {}}

    def install(m):
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backend import AttentionType

        n = 0
        for mod in m.modules():
            if not isinstance(mod, Attention) or mod.attn_type != AttentionType.DECODER:
                continue

            def hook(mod_, args, kwargs, output, li=n):
                lab = S["label"]
                if lab is None:
                    return
                q = args[0] if args else kwargs.get("query")
                k = args[1] if len(args) > 1 else kwargs.get("key")
                v = args[2] if len(args) > 2 else kwargs.get("value")
                if q.shape[0] < 32 or (lab, li) in S["seen"]:  # (the prefill step of the request: a decode step has one token)
                    return
                S["seen"].add((lab, li))
                S["dig"].setdefault(lab, {"ntok": int(q.shape[0]), "layers": {}})["layers"][str(li)] = {"q": digest(q), "k": digest(k), "v": digest(v), "o": digest(output)}
                if lab in dump and li in dump_layers:
                    for nm, t in (("q", q), ("k", k), ("v", v), ("o", output)):
                        S["dump"][f"{lab}|{li}|{nm}"] = t.detach().contiguous().view(torch.int16).cpu().numpy().copy()

            mod.register_forward_hook(hook, with_kwargs=True)
            n += 1
        return n

    llm.apply_model(install)
    rows = []
    greedy = dict(temperature=0, max_tokens=64, ignore_eos=True, logprobs=0)
    for i in idx:
        name, ids = prompts[i]
        share = (int(len(ids) * 0.8) // 16) * 16
        other = prompts[(i + 1) % len(prompts)][1][:300]
        second = ids[:share] + other
        S["label"] = f"{i}:{name}:R1"
        o1 = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1, logprobs=0), use_tqdm=False)[0]
        S["label"] = f"{i}:{name}:R2"
        o2 = llm.generate([TokensPrompt(prompt_token_ids=second)], SamplingParams(**greedy), use_tqdm=False)[0]
        S["label"] = None
        c1, c2 = o1.outputs[0], o2.outputs[0]
        rows.append({"i": i, "name": name, "r1_tokens": len(ids), "r1_cached": o1.num_cached_tokens, "r1_first": [c1.token_ids[0], c1.logprobs[0][c1.token_ids[0]].logprob],
                     "r2_tokens": len(second), "r2_cached": o2.num_cached_tokens, "tokens": list(c2.token_ids), "logprobs": [d[t].logprob for d, t in zip(c2.logprobs, c2.token_ids)]})
        print(f"{i} {name}: R1 {len(ids)} tokens, {o1.num_cached_tokens} cached; R2 {len(second)} tokens, {o2.num_cached_tokens} cached; first token {rows[-1]['tokens'][0]} {rows[-1]['logprobs'][0]!r}", flush=True)
    res = {"model": model, "kind": kind, "order": order, "rows": rows, "digests": S["dig"]}
    if kind == "kv" and "--verify" in argv:
        from glyd_gpu import vllm_kv

        res["verified_values"] = int(vllm_kv._VERIFIED[0])
    json.dump(res, open(out, "w"))
    if S["dump"]:
        np.savez(out.replace(".json", "") + "-dump.npz", **S["dump"])
    print("wrote", out, flush=True)


def common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    ra, rb = {r["i"]: r for r in A["rows"]}, {r["i"]: r for r in B["rows"]}
    print(f"{A['kind']}/{A['order']} ({pa}) against {B['kind']}/{B['order']} ({pb})")
    first_same = tok_same = 0
    n = 0
    for i in sorted(set(ra) & set(rb)):
        x, y = ra[i], rb[i]
        n += 1
        f = x["logprobs"][0] == y["logprobs"][0] and x["tokens"][0] == y["tokens"][0]
        first_same += f
        tok_same += x["tokens"] == y["tokens"]
        parts = []
        for r in ("R1", "R2"):
            lab = f"{i}:{x['name']}:{r}"
            da, db = A["digests"].get(lab), B["digests"].get(lab)
            if not da or not db:
                parts.append(f"{r}: no digests")
                continue
            where = None
            for li in range(len(da["layers"])):
                ea, eb = da["layers"].get(str(li)), db["layers"].get(str(li))
                if ea is None or eb is None:
                    break
                bad = [t for t in ("q", "k", "v", "o") if ea[t] != eb[t]]
                if bad:
                    where = (li, bad)
                    break
            nd = sum(1 for li in da["layers"] if any(da["layers"][li][t] != db["layers"].get(li, {}).get(t) for t in "qkvo"))
            parts.append(f"{r}: " + ("all layers equal" if where is None else f"first differs at layer {where[0]} ({','.join(where[1])}), {nd} of {len(da['layers'])} layers differ"))
        print(f"{i:2d} {x['name']:<26} cached R1 {x['r1_cached']:>5}/{y['r1_cached']:<5} R2 {x['r2_cached']:>5}/{y['r2_cached']:<5} first token {'same' if f else 'differs'} "
              f"({x['logprobs'][0]:.6f} {y['logprobs'][0]:.6f}) tokens same for {common(x['tokens'], y['tokens']):2d} of 64 | {' | '.join(parts)}")
    print(f"first token's logprob bit for bit: {first_same} of {n}; the 64 tokens identical: {tok_same} of {n}")


def dumps(pa, pb):
    """The tensors two runs saved (--dump) compared: per layer and tensor, the elements that differ and the largest difference (and the largest value)."""
    import numpy as np

    a, b = np.load(pa), np.load(pb)
    bf = lambda x: (x.reshape(-1).astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
    labs = sorted({k.split("|")[0] for k in a.files} & {k.split("|")[0] for k in b.files})
    for lab in labs:
        for li in sorted({int(k.split("|")[1]) for k in a.files if k.startswith(lab + "|")}):
            row = []
            for nm in ("q", "k", "v", "o"):
                key = f"{lab}|{li}|{nm}"
                if key not in a.files or key not in b.files or a[key].shape != b[key].shape:
                    row.append(f"{nm}: -")
                    continue
                x, y = a[key], b[key]
                d = np.abs(bf(x) - bf(y))
                row.append(f"{nm}: {int((x != y).sum()):,} of {x.size:,} differ, largest {d.max():.2e} (of {np.abs(bf(x)).max():.2f})")
            print(f"{lab} layer {li} | " + " | ".join(row))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "dumps":
        dumps(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
