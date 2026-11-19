"""How common are the pages that cost the lossless KV cache most, in real text? A page of one token repeated holds many exceptions (a page past its arena takes chunks of the escape pool, and the decode
kernel reads them: 1.09x at 92 entries a page, 1.51x at 307, where a whole cache is such pages), so before anything is done about them this counts them in what people send.

    python kv_traffic.py tokens OUT.json MODEL --data NAME=KIND:PATH[,NAME=KIND:PATH ...] [--max-tokens 4096] [--docs 20000]
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python kv_traffic.py kv OUT.json MODEL TABLES --data NAME=KIND:PATH[,...] [--max-tokens 2048] [--docs 300]

`tokens` (no GPU, no model: the tokenizer): each document's tokens (a chat through the model's chat template, text as it is), cut at --max-tokens, in pages of 16 from the document's start (as the cache
holds them): the pages in which one token is 8 of the 16 or more, 12 or more, all 16, the share of all pages and of tokens, the tokens that fill them. `kv` (vLLM in this process, eager, the model's
tables): the same documents run through the model with a hook on every layer's keys and values, each layer-page's exceptions beyond its heads' inline entries counted as the plugin counts them (with
the tables' bases), and from them the chunks the escape pool would give it (at the default arena), by the page's kind: the first page of a document (its attention sink), pages of one token repeated
(12 or more of 16), the rest; the share of layer-pages that take a chunk, their mean chunks and entries, and an estimate of what the decode kernel's pages cost beyond bf16's, from the two
measured points (92 entries a page: +9%, 307: +51%), linear between them and beyond.

KIND: sharegpt (a ShareGPT json: a list of {"conversations": [{"from", "value"}]}), jsonl (a jsonl file with "instruction", "context", "response"), code (a directory of source files, up to 1,500 by hash order),
text (one file, cut into documents of 40,000 characters), logs (a file of lines, cut every 400 lines)."""
import hashlib
import json
import os
import random
import sys

PT = 16


def documents(kind, path, limit, seed=0):
    """Up to `limit` documents (str, or a list of chat messages) of a data set."""
    rng = random.Random(seed)
    if kind == "sharegpt":
        data = json.load(open(path))
        rng.shuffle(data)
        for d in data[:limit]:
            msgs = [{"role": "user" if m["from"] == "human" else "assistant", "content": m["value"]} for m in d["conversations"] if m.get("from") in ("human", "gpt")]
            if msgs:
                yield msgs
    elif kind == "jsonl":
        rows = [json.loads(l) for l in open(path)]
        rng.shuffle(rows)
        for r in rows[:limit]:
            yield [{"role": "user", "content": (r.get("instruction", "") + "\n\n" + r.get("context", "")).strip()}, {"role": "assistant", "content": r.get("response", "")}]
    elif kind == "code":
        files = []
        for root, _, names in os.walk(path):
            files += [os.path.join(root, n) for n in names if n.endswith((".py", ".h", ".hpp", ".cu", ".cuh", ".c", ".cc", ".js", ".json", ".md", ".yaml", ".toml"))]
        files.sort(key=lambda f: hashlib.sha256(f.encode()).hexdigest())
        n = 0
        for f in files:
            try:
                t = open(f, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if len(t) > 200:
                yield t
                n += 1
                if n >= min(limit, 1500):
                    return
    elif kind == "text":
        t = open(path, encoding="utf-8", errors="replace").read()
        for i in range(0, len(t), 40000):
            if i // 40000 >= limit:
                return
            yield t[i : i + 40000]
    elif kind == "logs":
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        for i in range(0, len(lines), 400):
            if i // 400 >= limit:
                return
            yield "\n".join(lines[i : i + 400])
    else:
        raise ValueError(kind)


def token_ids(tok, doc, max_tokens):
    if isinstance(doc, list):
        if getattr(tok, "chat_template", None):
            ids = tok.apply_chat_template(doc, tokenize=True, add_generation_prompt=False)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        else:
            ids = tok.encode("\n".join(f"{m['role']}: {m['content']}" for m in doc))
    else:
        ids = tok.encode(doc)  # (as a server tokenizes a prompt: the beginning-of-text token a model's tokenizer adds is there, the first token of every sequence)
    return ids[:max_tokens]


def repeats(ids):
    """For each whole page of 16 tokens: how many times its most frequent token occurs, and which."""
    out = []
    for p in range(len(ids) // PT):
        page = ids[p * PT : (p + 1) * PT]
        best = max(set(page), key=page.count)
        out.append((page.count(best), best))
    return out


def tokens_mode(model, data, out, max_tokens, docs):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    res = {"model": model, "max_tokens": max_tokens, "data": {}}
    for name, spec in data:
        kind, path = spec
        n_docs = n_tok = n_pages = 0
        thr = {8: 0, 12: 0, 16: 0}
        fills = {}
        rep_tokens = 0
        longest = 0
        for doc in documents(kind, path, docs):
            ids = token_ids(tok, doc, max_tokens)
            if len(ids) < PT:
                continue
            n_docs += 1
            n_tok += len(ids)
            for c, t in repeats(ids):
                n_pages += 1
                for k in thr:
                    thr[k] += c >= k
                if c >= 12:
                    fills[t] = fills.get(t, 0) + 1
                    rep_tokens += PT
            run = best = 1
            for a, b in zip(ids, ids[1:]):
                run = run + 1 if a == b else 1
                best = max(best, run)
            longest = max(longest, best)
        top = sorted(fills.items(), key=lambda x: -x[1])[:8]
        res["data"][name] = {"documents": n_docs, "tokens": n_tok, "leaves": n_pages, "pages_with_8_of_16_one_token": thr[8], "pages_with_12_of_16": thr[12], "pages_of_16_of_16": thr[16],
                             "share_8": thr[8] / max(1, n_pages), "share_12": thr[12] / max(1, n_pages), "share_16": thr[16] / max(1, n_pages), "longest_run_of_one_token": longest,
                             "tokens_filling_the_12_pages": [[tok.decode([t]), t, c] for t, c in top]}
        print(name, json.dumps(res["data"][name], default=str)[:400], flush=True)
    json.dump(res, open(out, "w"), indent=1)


# ---- the keys and values: vLLM in this process ----------------------------------------------------------------------------------------
_S = {"on": False, "bases": None, "need": {}}


def _install(model):
    import torch
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    from glyd_gpu import kvcal

    names = []
    for m in model.modules():
        if not isinstance(m, Attention) or m.attn_type != AttentionType.DECODER:
            continue
        li = len(names)
        names.append(m.layer_name)

        def hook(mod, args, kwargs, li=li):
            if not _S["on"]:
                return
            k = args[1] if len(args) > 1 else kwargs.get("key")
            v = args[2] if len(args) > 2 else kwargs.get("value")
            k = k.detach().reshape(-1, mod.num_kv_heads, mod.head_size).contiguous().view(torch.int16)
            v = v.detach().reshape(-1, mod.num_kv_heads, mod.head_size_v).contiguous().view(torch.int16)
            bk, bv = _S["bases"][li]
            _, need = kvcal.leaf_needs(kvcal.outside(k, bk), kvcal.outside(v, bv))
            _S["need"][li] = need.cpu()

        m.register_forward_pre_hook(hook, with_kwargs=True)
    return names


def slowdown(entries):
    """The decode kernel's cost beyond bf16's on a page of this many exception entries beyond its arena, from the measured 92 -> +9% and 307 -> +51%, linear between, through zero, and on beyond."""
    if entries <= 0:
        return 0.0
    if entries <= 92:
        return 0.09 * entries / 92
    return 0.09 + 0.42 * (entries - 92) / 215


def kv_mode(model, tables, data, out, max_tokens, docs):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from glyd_gpu import kvcal, vllm_kv

    t = kvcal.Tables(tables)
    llm = LLM(model, dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, max_num_seqs=1, max_model_len=max_tokens + 64, max_num_batched_tokens=max_tokens + 64,
              gpu_memory_utilization=0.9, seed=0)
    names = llm.apply_model(_install)[0]
    _S["bases"] = [(torch.from_numpy(t.k[t.index[n]]).cuda(), torch.from_numpy(t.v[t.index[n]]).cuda()) for n in names]
    tok = AutoTokenizer.from_pretrained(model)
    H = int(t.k.shape[1])
    A = vllm_kv.free_bay(H)
    capacity = A + vllm_kv.ND * vllm_kv.CH  # (the most entries past a head's inline ones a page holds; a page that needs more is kept raw)
    res = {"model": model, "tables": os.path.basename(tables), "max_tokens": max_tokens, "bay": A, "capacity": capacity, "layers": len(names), "data": {}}
    for name, (kind, path) in data:
        cats = {"first page": [0, 0, 0.0, 0.0, 0.0], "repeated pages (12 of 16 one token)": [0, 0, 0.0, 0.0, 0.0], "other pages": [0, 0, 0.0, 0.0, 0.0]}  # layer-pages, with a chunk, chunks, entries, slowdown
        n_docs = n_over_docs = n_over = 0
        mx = 0
        for doc in documents(kind, path, docs):
            ids = token_ids(tok, doc, max_tokens)
            if len(ids) < 4 * PT:
                continue
            ids = ids[: len(ids) // PT * PT]
            _S["on"], _S["need"] = True, {}
            llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
            _S["on"] = False
            need = torch.stack([_S["need"][li] for li in range(len(names))])  # [layers, pages]
            rep = [c >= 12 for c, _ in repeats(ids)]
            n_docs += 1
            over = int((need > capacity).sum())
            n_over += over
            n_over_docs += over > 0
            for p in range(need.shape[1]):
                cat = cats["first page" if p == 0 else "repeated pages (12 of 16 one token)" if rep[p] else "other pages"]
                col = need[:, p]
                beyond = torch.clamp(col - A, min=0)
                chunks = (beyond + vllm_kv.CH - 1) // vllm_kv.CH
                cat[0] += col.numel()
                cat[1] += int((chunks > 0).sum())
                cat[2] += float(chunks.sum())
                cat[3] += float(beyond.sum())
                cat[4] += sum(slowdown(int(x)) for x in beyond.tolist())
                mx = max(mx, int(col.max()))
        total = sum(c[0] for c in cats.values())
        row = {"documents": n_docs, "layer_pages": total, "most_entries_beyond_inline_in_a_page": mx, "layer_pages_over_capacity": n_over, "documents_with_a_page_over_capacity": n_over_docs}
        for cname, (n, w, ch, en, sl) in cats.items():
            row[cname] = {"layer_pages": n, "share_of_all": n / max(1, total), "taking_a_chunk": w, "share_taking_a_chunk": w / max(1, n), "mean_chunks": ch / max(1, n), "mean_entries_past_inline": en / max(1, n),
                          "estimated_decode_cost_beyond_bf16": sl / max(1, n)}
        row["all_pages"] = {"share_taking_a_chunk": sum(c[1] for c in cats.values()) / max(1, total), "mean_chunks": sum(c[2] for c in cats.values()) / max(1, total),
                            "estimated_decode_cost_beyond_bf16": sum(c[4] for c in cats.values()) / max(1, total)}
        res["data"][name] = row
        print(name, json.dumps(row)[:600], flush=True)
        json.dump(res, open(out, "w"), indent=1)


def parse_data(s):
    out = []
    for item in s.split(","):
        name, _, rest = item.partition("=")
        kind, _, path = rest.partition(":")
        out.append((name, (kind, os.path.expanduser(path))))
    return out


def main():
    a = sys.argv[1:]
    mode, out = a[0], a[1]
    data = parse_data(a[a.index("--data") + 1])
    docs = int(a[a.index("--docs") + 1]) if "--docs" in a else (20000 if mode == "tokens" else 300)
    if mode == "tokens":
        tokens_mode(a[2], data, out, int(a[a.index("--max-tokens") + 1]) if "--max-tokens" in a else 4096, docs)
    else:
        kv_mode(a[2], a[3], data, out, int(a[a.index("--max-tokens") + 1]) if "--max-tokens" in a else 2048, docs)


if __name__ == "__main__":
    main()
