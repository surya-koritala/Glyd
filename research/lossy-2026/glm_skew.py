"""Expert-routing skew of a MoE model (Linear GLY-127): which experts the router picks, per layer, on six text domains and in decode.
Model-agnostic: --model glm (zai-org/GLM-5.3-Flash, FP8) or nemotron (NVIDIA-Nemotron-3-Super-120B-A12B-BF16); both are pinned below.
  plan     MODEL                 shell lines for the launcher (repo, sha, pod class, disk, cache sizes)
  prep     MODEL OUT             pinned public datasets -> token sequences of 2048, calibration + held-out (CPU; transformers tokenizer)
  trace    MODEL --weights DIR --data DIR --out DIR [--raw DIR]   forward passes with a hook on every router, then ~20 short decodes (GPUs)
  analyze  MODEL DIR             counts + decode trace -> skew.json and skew.md (numpy only)
  selftest                       synthetic routing through analyze (numpy only)
No token is used anywhere: models and datasets are ungated."""
import argparse, collections, gzip, hashlib, json, os, re, sys, tempfile, time

SEQ = 2048
DOMAINS = ("wiki", "code", "chat", "math", "multi", "reason")
COV = (5, 10, 13, 20, 30, 50)  # top-k experts per layer as a share of the experts
MODELS = {
    "glm": dict(repo="zai-org/GLM-5.3-Flash", rev="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a", cls="Glm5NextForConditionalGeneration",
                router="Glm5NextTextTopkRouter", n_moe=42, ne=288, k=8, caches=(7, 13, 19), pod="glm", disk=700, weights_gb=328.3,
                tmpl={}, suffix="</think>",  # thinking off: the template's generation prompt ends in <think>, so close it
                rule="GLM notes: static 13% profile transfers at >= 70% -> go on; < 55% -> stop"),
    "nemotron": dict(repo="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", rev="2dc98e2afe4face0e4ce40972a915c45368bd34a", cls="NemotronHForCausalLM",
                     router="NemotronHTopkRouter", n_moe=40, ne=512, k=22, caches=(17, 32, 46), pod="nemo", disk=450, weights_gb=247.2,
                     tmpl={"enable_thinking": False}, suffix="",
                     rule="Nemotron notes: 32% cache needs ~43% hit for 10 tok/s and ~75% for 20 tok/s on a 4090 + PCIe 4 (4 bpw)"),
}
# Pinned sources: (repo, dataset revision, [(file, sha256 of the file)]). All ungated; only routing ids leave the pod, no text.
DS = {
    "wiki": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
             [("wikitext-103-raw-v1/train-00000-of-00002.parquet", "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212")]),
    "code": ("codeparrot/codeparrot-clean-valid", "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",  # Python files only
             [("file-000000000054.json.gz", "7f8c20a737c9084779bcdb853325ad4774d0db52c74aa2a63fd658d6787eb35b")]),
    "chat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
             [("data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70")]),
    "math": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
             [("main/train-00000-of-00001.parquet", "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d")]),
    "multi": ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",  # zh, ru, es, taken in turn
              [("20231101.zh/train-00002-of-00006.parquet", "ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f"),
               ("20231101.ru/train-00020-of-00021.parquet", "fc1e2edc58fa27085d36c32c38922d7904077d9cdac05719efbcefa3faeea240"),
               ("20231101.es/train-00012-of-00013.parquet", "372a1b30cbf5446219dc0150257e013e69371de3cb87dc7645f5344a126e0eda")]),
    "reason": ("open-thoughts/OpenThoughts-114k", "bd093c3994fd54d2390985b66988ddf282a55eb6",
               [("data/train-00005-of-00006.parquet", "bd455b4a7638967557919b683297f8f4014efe58a7c75c90cc5da62e88bced1f")]),
}
MAXDOC = {"reason": 8192}  # tokens kept of one document (default 4096)
STRIDE = {"wiki": 40, "code": 8, "chat": 8, "math": 8, "multi": 4, "reason": 25}  # every n-th document, so the sample is not the first pages of the file
PROMPTS = [  # decode prompts (thinking off, greedy): (domain, text)
    ("chat", "Explain the difference between a mutex and a semaphore, with a short example of each."),
    ("chat", "Write a friendly email to a colleague asking to move our weekly sync from Tuesday to Thursday."),
    ("chat", "Give me three tips for preparing for a job interview at a software company."),
    ("chat", "Summarize the plot of Romeo and Juliet in five sentences."),
    ("code", "Write a Python function that merges two sorted lists into one sorted list without using sort()."),
    ("code", "Write a SQL query that returns the top 3 customers by total order value from customers(id, name) and orders(id, customer_id, amount)."),
    ("code", "Write a Rust function that reverses the words in a string."),
    ("code", "def avg(xs): return sum(xs) / len(xs) fails on an empty list. Show the corrected function."),
    ("math", "A train travels 180 km in 2.5 hours. What is its average speed in km/h? How long would 450 km take at that speed?"),
    ("math", "Solve for x: 3x + 7 = 2x - 5."),
    ("math", "What is the sum of the first 20 positive odd numbers? Show the reasoning briefly."),
    ("math", "A rectangle has perimeter 36 and its length is twice its width. Find its area."),
    ("multi", "请用中文简要介绍一下长城的历史。"),
    ("multi", "Explica brevemente cómo funciona la fotosíntesis."),
    ("multi", "Объясните простыми словами, что такое инфляция."),
    ("multi", "Erkläre kurz, warum der Himmel blau ist."),
    ("wiki", "Who was Ada Lovelace and why is she remembered?"),
    ("wiki", "Describe the main causes of the French Revolution."),
    ("reason", "Three friends share a pizza cut into 8 slices. Alice eats twice as many as Bob and Carol eats one more than Bob. All slices are eaten. How many does each eat?"),
    ("reason", "Is 221 a prime number? Explain how you decide."),
]


def now():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------------------------------------------- plan
def plan(a):
    m = MODELS[a.model]
    for k, v in dict(REPO=m["repo"], REV=m["rev"], POD_CLASS=m["pod"], DISK=m["disk"], K=m["k"], NE=m["ne"], N_MOE=m["n_moe"],
                     CACHE_PCTS=" ".join(map(str, m["caches"])), WEIGHTS_GB=m["weights_gb"]).items():
        print(f'{k}="{v}"')


# ---------------------------------------------------------------------------------------------------------------- prep
def rows(path, cols):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            for line in f:
                yield json.loads(line)
    else:
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(path).iter_batches(batch_size=256, columns=cols):
            yield from b.to_pylist()


def docs(dom, paths, tok):
    """Yield the text of one document at a time for a domain."""
    def chat(msgs):
        return tok.apply_chat_template(msgs, tokenize=False)
    if dom == "wiki":  # wikitext rows are lines: group them into ~6,000-character pieces
        buf = ""
        for r in rows(paths[0], ["text"]):
            if r["text"].strip():
                buf += r["text"]
            if len(buf) > 6000:
                yield buf
                buf = ""
    elif dom == "code":
        for r in rows(paths[0], None):
            yield r["content"]
    elif dom == "chat":
        for r in rows(paths[0], ["messages"]):
            yield chat([{"role": m["role"], "content": m["content"]} for m in r["messages"]])
    elif dom == "math":
        for r in rows(paths[0], ["question", "answer"]):
            yield chat([{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}])
    elif dom == "multi":
        its = [rows(p, ["text"]) for p in paths]
        while True:
            for it in its:
                r = next(it, None)
                if r is None:
                    return
                yield r["text"]
    elif dom == "reason":  # thought and solution -> <think>thought</think>solution, which both chat templates split on
        for r in rows(paths[0], ["conversations"]):
            c = {m["from"]: m["value"] for m in r["conversations"]}
            if "user" in c and "assistant" in c:
                t = c["assistant"]
                th = re.search(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", t, re.S)
                so = re.search(r"<\|begin_of_solution\|>(.*?)(?:<\|end_of_solution\|>|$)", t, re.S)
                yield chat([{"role": "user", "content": c["user"]},
                            {"role": "assistant", "content": f"<think>{th.group(1).strip() if th else ''}</think>{so.group(1).strip() if so else t}"}])


def prep(a):
    import numpy as np
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    m = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["rev"])
    eos = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)
    info = {"model": a.model, "repo": m["repo"], "rev": m["rev"], "seq": SEQ, "domains": {}}
    for dom in DOMAINS:
        repo, rev, files = DS[dom]
        paths = []
        for f, sha in files:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=rev)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(blk)
            assert h.hexdigest() == sha, f"{repo}/{f}: sha256 {h.hexdigest()} is not the pinned {sha}"
            paths.append(p)
        need = {"calib": a.cal_seqs * SEQ, "held": a.held_seqs * SEQ}
        got = {"calib": [], "held": []}
        n = {"calib": 0, "held": 0}
        ndoc = bad = 0
        cap = MAXDOC.get(dom, 4096)
        for i, text in enumerate(docs(dom, paths, tok)):
            if i % STRIDE[dom]:
                continue
            split = "held" if (i // STRIDE[dom]) % 8 == 7 else "calib"  # of the documents taken, the 8th, 16th, ... are held out
            if n[split] >= need[split]:
                if all(n[s] >= need[s] for s in need):
                    break
                continue
            ids = tok(text[:cap * 8], add_special_tokens=False)["input_ids"][:cap]
            if not ids:
                bad += 1
                continue
            got[split] += ids + [eos]
            n[split] += len(ids) + 1
            ndoc += 1
        for s in need:
            assert n[s] >= need[s], f"{dom}: only {n[s]} {s} tokens, need {need[s]}"
            np.save(f"{a.out}/{s}_{dom}.npy", np.array(got[s][:need[s]], dtype=np.int32).reshape(-1, SEQ))
        info["domains"][dom] = dict(repo=repo, rev=rev, files=[f for f, _ in files], stride=STRIDE[dom], docs=ndoc, skipped=bad,
                                    calib_tokens=need["calib"], held_tokens=need["held"], peek=tok.decode(got["calib"][:24]))
        print(now(), dom, info["domains"][dom]["docs"], "docs", need, flush=True)
    json.dump(info, open(f"{a.out}/prep.json", "w"), indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------------------- trace
def trace(a):
    import numpy as np, torch, transformers
    m = MODELS[a.model]
    L, NE, K = m["n_moe"], m["ne"], m["k"]
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(a.weights)
    n = torch.cuda.device_count()
    kw = dict(device_map="auto", max_memory={i: int(torch.cuda.get_device_properties(i).total_memory * 0.9) for i in range(n)}) if n else {}
    model = getattr(transformers, m["cls"]).from_pretrained(a.weights, dtype=torch.bfloat16, **kw).eval()
    dev = next(model.parameters()).device
    timing = {"load_s": round(time.time() - t0, 1), "gpus": n, "batches": [], "decode": {}}
    timing["gpu_gib"] = [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(n)]
    print(now(), f"loaded in {timing['load_s']} s on {n} GPUs, GiB per GPU {timing['gpu_gib']}", flush=True)

    cur, calls = {}, []  # one forward leaves, per router, (ids, weights) in cur; the top-level hook turns them into one call record
    layers = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == m["router"]:
            li = int(re.search(r"layers\.(\d+)\.", name).group(1))
            layers.append(li)
            mod.register_forward_hook(lambda mo, i, o, li=li: cur.__setitem__(li, (o[2].to(torch.int16), o[1].float())))
    layers.sort()
    assert len(layers) == L, f"expected {L} routers of class {m['router']}, hooked {len(layers)}"

    def top(mo, i, o):
        assert sorted(cur) == layers, f"routers that fired: {sorted(cur)}"
        ids = torch.stack([cur[l][0].cpu() for l in layers], 1).numpy().astype(np.uint16)  # [tokens, L, K]
        w = torch.stack([cur[l][1].cpu() for l in layers], 1).numpy()
        assert ids.shape[1:] == (L, K) and ids.max() < NE, f"router output {ids.shape}, max {ids.max()}"
        cur.clear()
        calls.append((ids, w, time.perf_counter()))
    model.register_forward_hook(top)

    state = {"eager": False}

    def fwd(x):  # x [b, SEQ] int32 -> list of (ids, w) per sequence; halves the batch on OOM, once tries the eager experts on another error
        try:
            with torch.no_grad():
                model(input_ids=torch.from_numpy(x).long().to(dev), use_cache=False, logits_to_keep=1)
            ids, w, _ = calls.pop()
            return [(ids[j * SEQ:(j + 1) * SEQ], w[j * SEQ:(j + 1) * SEQ]) for j in range(len(x))]
        except torch.cuda.OutOfMemoryError:
            cur.clear(); calls.clear(); torch.cuda.empty_cache()
            if len(x) == 1:
                raise
            print(now(), "OOM: batch halved", flush=True)
            h = len(x) // 2
            return fwd(x[:h]) + fwd(x[h:])
        except AssertionError:
            raise
        except Exception as e:
            if state["eager"]:
                raise
            print(now(), f"forward failed ({type(e).__name__}: {str(e)[:300]}): trying experts_implementation=eager", flush=True)
            cur.clear(); calls.clear(); state["eager"] = True
            model.set_experts_implementation("eager")
            return fwd(x)

    os.makedirs(f"{a.out}/counts", exist_ok=True)
    if a.raw:
        os.makedirs(a.raw, exist_ok=True)
    stop = lambda margin: a.deadline and time.time() > a.deadline - margin * 60
    for dom in DOMAINS:
        for split in ("calib", "held"):
            if stop(8):
                print(now(), "deadline: no new batches", flush=True)
                break
            seqs = np.load(f"{a.data}/{split}_{dom}.npy")
            cnt, mass, uniq, keep = np.zeros((L, NE), np.int64), np.zeros((L, NE)), [], []
            t1 = time.perf_counter()
            for i in range(0, len(seqs), a.batch):
                t2 = time.perf_counter()
                out = fwd(seqs[i:i + a.batch])
                torch.cuda.synchronize() if n else None
                for ids, w in out:
                    for l in range(L):
                        cnt[l] += np.bincount(ids[:, l].ravel(), minlength=NE)
                        mass[l] += np.bincount(ids[:, l].ravel(), weights=w[:, l].ravel(), minlength=NE)
                    uniq.append([len(np.unique(ids[:, l])) for l in range(L)])
                    keep.append(ids)
                timing["batches"].append(dict(domain=dom, split=split, seqs=len(out), tokens=len(out) * SEQ, s=round(time.perf_counter() - t2, 2)))
            tokens = len(seqs) * SEQ
            assert (cnt.sum(1) == tokens * K).all(), f"{dom}/{split}: counts per layer {cnt.sum(1)[:3]}... are not tokens x {K}"
            np.savez_compressed(f"{a.out}/counts/{split}_{dom}.npz", counts=cnt, mass=mass, ntok=tokens, uniq=np.mean(uniq, 0), layers=np.array(layers))
            if a.raw:
                np.savez_compressed(f"{a.raw}/ids_{split}_{dom}.npz", ids=np.concatenate(keep), layers=np.array(layers))
            print(now(), f"{split}_{dom}: {tokens} tokens in {time.perf_counter() - t1:.0f} s, {tokens / (time.perf_counter() - t1):.0f} tok/s", flush=True)
            json.dump(timing, open(f"{a.out}/timing.json", "w"), indent=1)

    # decode: ~20 short greedy generations, thinking off; the routing of every one-token step is kept
    t3, d_ids, d_gen, d_dt, outs = time.time(), [], [], [], []
    for gi, (dom, prompt) in enumerate(PROMPTS[:a.gens]):
        if stop(10) or time.time() - t3 > a.gen_min * 60:
            print(now(), "decode: time budget reached", flush=True)
            break
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **m["tmpl"]) + m["suffix"]
        enc = tok(text, add_special_tokens=False, return_tensors="pt")
        calls.clear()
        try:
            with torch.no_grad():
                g = model.generate(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev), max_new_tokens=a.gen_new,
                                   do_sample=False, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        except Exception as e:
            print(now(), f"generation {gi} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            break
        steps = [c for c in calls[1:] if len(c[0]) == 1]  # calls[0] is the prompt
        ts = [calls[0][2]] + [c[2] for c in calls[1:]]
        d_ids += [c[0][0] for c in steps]
        d_gen += [gi] * len(steps)
        d_dt += [ts[j + 1] - ts[j] for j in range(len(steps))]
        outs.append(dict(domain=dom, prompt=prompt, steps=len(steps), text=tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)[:300]))
        print(now(), f"gen {gi} ({dom}): {len(steps)} steps, {np.mean(d_dt[-len(steps):]) if steps else 0:.2f} s/step", flush=True)
        np.savez_compressed(f"{a.out}/decode.npz", ids=np.array(d_ids, np.uint16).reshape(-1, L, K), gen=np.array(d_gen, np.uint16), dt=np.array(d_dt, np.float32),
                            gen_domain=np.array([PROMPTS[i][0] for i in d_gen]), layers=np.array(layers))
        timing["decode"] = dict(gens=len(outs), steps=len(d_ids), mean_step_s=round(float(np.mean(d_dt)), 3))
        json.dump(dict(decode=timing["decode"], generations=outs), open(f"{a.out}/decode.json", "w"), indent=1, ensure_ascii=False)
        json.dump(timing, open(f"{a.out}/timing.json", "w"), indent=1)
    print(now(), "trace done", flush=True)


# ------------------------------------------------------------------------------------------------------------ analysis
def top_mask(c, k):  # per layer, the k most used experts: bool [L, NE]
    import numpy as np
    m = np.zeros(c.shape, bool)
    np.put_along_axis(m, np.argsort(-c, axis=1, kind="stable")[:, :k], True, 1)
    return m


def top_mask_global(c, n):  # the n most used (layer, expert) pairs over all layers
    import numpy as np
    m = np.zeros(c.size, bool)
    m[np.argsort(-c.ravel(), kind="stable")[:n]] = True
    return m.reshape(c.shape)


def hit(m, c):  # share of routed (token, slot) pairs whose expert is in the mask
    return float((c * m).sum() / c.sum())


def replay(D, cap, policy, seed=None):
    """D [steps, L, K] expert ids; a per-layer cache of cap experts, LRU or LFU; returns the hit share (accesses in token order)."""
    import numpy as np
    L = D.shape[1]
    h = 0
    for l in range(L):
        cache = collections.OrderedDict()
        freq = collections.Counter()
        if seed is not None:  # start from a static profile (least used first, so the most used is the most recent)
            for e in np.flatnonzero(seed[l])[:cap]:
                cache[int(e)] = 1
        for row in D[:, l, :].tolist():
            for e in row:
                if e in cache:
                    h += 1
                    cache.move_to_end(e)
                else:
                    if len(cache) >= cap:
                        if policy == "lru":
                            cache.popitem(last=False)
                        else:
                            del cache[min(cache, key=freq.__getitem__)]
                    cache[e] = 1
                freq[e] += 1
    return h / D.shape[0] / D.shape[1] / D.shape[2]


def analyze(a):
    import numpy as np
    m = MODELS[a.model]
    NE, caches = m["ne"], m["caches"]
    ld = lambda split: {d: np.load(f"{a.dir}/counts/{split}_{d}.npz") for d in DOMAINS if os.path.exists(f"{a.dir}/counts/{split}_{d}.npz")}
    cal, held = ld("calib"), ld("held")
    doms = [d for d in DOMAINS if d in cal]
    assert doms, f"no counts/calib_*.npz in {a.dir}"
    cc = {d: cal[d]["counts"].astype(np.float64) for d in doms}
    hc = {d: held[d]["counts"].astype(np.float64) for d in doms if d in held}
    pooled, hpooled = sum(cc.values()), (sum(hc.values()) if hc else None)
    L = pooled.shape[0]
    R = dict(model=a.model, repo=m["repo"], rev=m["rev"], layers=L, experts=NE, top_k=m["k"], seq=SEQ, domains=doms,
             calib_tokens={d: int(cal[d]["ntok"]) for d in doms}, held_tokens={d: int(held[d]["ntok"]) for d in held})
    cover = lambda c, p: (c * top_mask(c, max(1, round(p * NE / 100)))).sum(1) / c.sum(1)  # per layer share covered by the top p%
    sizes = sorted(set(COV) | set(caches))
    R["coverage"] = {"pooled": {str(p): dict(k=max(1, round(p * NE / 100)), mean=float(cover(pooled, p).mean()), min=float(cover(pooled, p).min()),
                                              max=float(cover(pooled, p).max()), per_layer=[round(float(x), 4) for x in cover(pooled, p)]) for p in sizes},
                     "by_domain": {d: {str(p): dict(mean=float(cover(cc[d], p).mean()), min=float(cover(cc[d], p).min()), max=float(cover(cc[d], p).max()))
                                       for p in sizes} for d in doms}}
    R["unique_experts_per_chunk"] = dict(mean_share=float(np.mean([cal[d]["uniq"].mean() for d in doms]) / NE),
                                         per_layer=[round(float(x) / NE, 3) for x in np.mean([cal[d]["uniq"] for d in doms], 0)])
    st, tr = {}, {}
    for p in caches:
        k = max(1, round(p * NE / 100))
        pm, gm = top_mask(pooled, k), top_mask_global(pooled, k * L)  # profile from all calibration domains: per-layer split / one pool over layers
        s = dict(k_per_layer=k)
        if hc:
            ms = {d: (held[d]["mass"] * pm).sum() / held[d]["mass"].sum() for d in hc}
            s.update(per_layer_alloc=dict(heldout=hit(pm, hpooled), mass_weighted=float(np.mean(list(ms.values()))), by_domain={d: hit(pm, hc[d]) for d in hc}),
                     global_alloc=dict(heldout=hit(gm, hpooled), by_domain={d: hit(gm, hc[d]) for d in hc}),
                     oracle_heldout=hit(top_mask(hpooled, k), hpooled), uniform=k / NE)
        st[str(p)] = s
        # profile from one domain evaluated on another (the diagonal on that domain's held-out text), and from all other domains on an unseen one
        tr[str(p)] = dict(matrix={a_: {b_: hit(top_mask(cc[a_], k), hc[b_] if a_ == b_ and b_ in hc else cc[b_]) for b_ in doms} for a_ in doms},
                          unseen_domain={d: hit(top_mask(pooled - cc[d], k), cc[d]) for d in doms} if len(doms) > 1 else {})
    R["static"], R["transfer"] = st, tr
    dp = f"{a.dir}/decode.npz"
    if os.path.exists(dp):
        z = np.load(dp)
        D, gd = z["ids"], z["gen_domain"]
        dc = np.stack([np.bincount(D[:, l].ravel(), minlength=NE) for l in range(L)]).astype(np.float64)
        dd = dict(steps=int(len(D)), gens=int(len(set(z["gen"].tolist()))), mean_step_s=float(z["dt"].mean()), static={}, self_oracle={}, lru={}, lru_seeded={}, lfu={}, static_by_domain={})
        for p in caches:
            k = max(1, round(p * NE / 100))
            pm = top_mask(pooled, k)
            dd["static"][str(p)], dd["self_oracle"][str(p)] = hit(pm, dc), hit(top_mask(dc, k), dc)
            dd["static_by_domain"][str(p)] = {d: hit(pm, np.stack([np.bincount(D[gd == d][:, l].ravel(), minlength=NE) for l in range(L)]).astype(np.float64))
                                              for d in sorted(set(gd.tolist()))}
            dd["lru"][str(p)], dd["lfu"][str(p)] = replay(D, k, "lru"), replay(D, k, "lfu")
            dd["lru_seeded"][str(p)] = replay(D, k, "lru", seed=pm)
        R["decode"] = dd
    tp = f"{a.dir}/timing.json"
    if os.path.exists(tp):
        T = json.load(open(tp))
        R["timing"] = dict(load_s=T.get("load_s"), prefill_tok_s=round(sum(b["tokens"] for b in T["batches"]) / max(1e-9, sum(b["s"] for b in T["batches"])), 1),
                           batches=len(T["batches"]), decode=T.get("decode"))
    json.dump(R, open(a.json or f"{a.dir}/skew.json", "w"), indent=1)
    md = markdown(R, m)
    open(a.md or f"{a.dir}/skew.md", "w").write(md)
    if not getattr(a, "quiet", False):
        print(md)
    return R


def markdown(R, m):
    P = lambda x: f"{100 * x:.1f}"
    cs = R["static"].keys()
    out = [f"## Expert-routing skew: {R['repo']} @ {R['rev'][:8]}",
           f"{R['layers']} MoE layers x {R['experts']} experts, top-{R['top_k']}; {sum(R['calib_tokens'].values())} calibration tokens in {len(R['domains'])} domains, "
           f"{sum(R['held_tokens'].values())} held-out; sequences of {R['seq']}. Hit = share of routed (token, expert) pairs served by the resident set.", "",
           "### Top-k experts per layer: share of routed tokens covered (mean over layers; % of experts in the header)", "",
           "| set | " + " | ".join(f"{p}%" for p in sorted(set(COV) | set(m["caches"]))) + " |", "|---|" + "---:|" * len(sorted(set(COV) | set(m["caches"])))]
    ps = sorted(set(COV) | set(m["caches"]))
    c = R["coverage"]
    out.append("| all domains (min-max over layers) | " + " | ".join(f"{P(c['pooled'][str(p)]['mean'])} ({P(c['pooled'][str(p)]['min'])}-{P(c['pooled'][str(p)]['max'])})" for p in ps) + " |")
    for d in R["domains"]:
        out.append(f"| {d} | " + " | ".join(P(c["by_domain"][d][str(p)]["mean"]) for p in ps) + " |")
    out.append("| uniform routing | " + " | ".join(f"{p}.0" for p in ps) + " |")
    out += ["", f"A 2048-token chunk touches {P(R['unique_experts_per_chunk']['mean_share'])}% of a layer's experts on average (prefill streaming).", ""]
    if all("per_layer_alloc" in R["static"][p] for p in cs):
        out += ["### Static profile built on the calibration domains, hit rate on held-out text", "",
                "| cache | per-layer split | one pool over layers | by mass | oracle (held-out's own top-k) | uniform | " + " | ".join(R["domains"]) + " |",
                "|---|---:|---:|---:|---:|---:|" + "---:|" * len(R["domains"])]
        for p in cs:
            s = R["static"][p]
            out.append(f"| {p}% ({s['k_per_layer']}/layer) | {P(s['per_layer_alloc']['heldout'])} | {P(s['global_alloc']['heldout'])} | {P(s['per_layer_alloc']['mass_weighted'])} | "
                       f"{P(s['oracle_heldout'])} | {P(s['uniform'])} | " + " | ".join(P(s["per_layer_alloc"]["by_domain"].get(d, float('nan'))) for d in R["domains"]) + " |")
        mid = sorted(cs, key=int)[len(cs) // 2]
        out += ["", f"Decision rule ({m['rule']}): static {mid}% profile on held-out = {P(R['static'][mid]['per_layer_alloc']['heldout'])}%.", ""]
    out += ["### A profile that never saw the domain (built on the other domains, scored on this one)", "",
            "| cache | " + " | ".join(R["domains"]) + " |", "|---|" + "---:|" * len(R["domains"])]
    for p in cs:
        out.append(f"| {p}% | " + " | ".join(P(R["transfer"][p]["unseen_domain"].get(d, float("nan"))) for d in R["domains"]) + " |")
    if "decode" in R:
        d = R["decode"]
        out += ["", f"### Decode ({d['gens']} generations, {d['steps']} steps, thinking off): static profile vs per-layer caches replayed in step order", "",
                "| cache | static profile | static, own top-k (oracle) | LRU cold | LRU seeded with profile | LFU cold |", "|---|---:|---:|---:|---:|---:|"]
        for p in cs:
            out.append(f"| {p}% | {P(d['static'][p])} | {P(d['self_oracle'][p])} | {P(d['lru'][p])} | {P(d['lru_seeded'][p])} | {P(d['lfu'][p])} |")
    if "timing" in R:
        t = R["timing"]
        out += ["", f"Wall clock: model load {t['load_s']} s, prefill {t['prefill_tok_s']} tokens/s over {t['batches']} batches" +
                (f", decode {t['decode']['mean_step_s']} s/step" if t.get("decode") else "") + "."]
    return "\n".join(out) + "\n"


# -------------------------------------------------------------------------------------------------------------- selftest
def selftest(_=None):
    import numpy as np
    rng = np.random.default_rng(0)
    L, NE, K, S = 3, 64, 4, 256
    m = dict(MODELS["glm"], ne=NE, k=K, n_moe=L, caches=(10, 25, 50))
    MODELS["tiny"] = m
    base = 1.0 / np.arange(1, NE + 1) ** 1.1  # a Zipf-like popularity; every layer and every domain gets its own order of it

    def draw(pop, n):  # n tokens x L layers: K distinct experts each, with the odds in pop [L, NE] (Gumbel top-k)
        g = rng.gumbel(size=(n, L, NE)) + np.log(pop)
        return np.argpartition(-g, K, axis=2)[:, :, :K].astype(np.uint16)

    def write(d, popfor, uniform=False):
        os.makedirs(f"{d}/counts")
        for dom in DOMAINS:
            for split, n in (("calib", 1500), ("held", 400)):
                pop = np.stack([popfor(dom, l) for l in range(L)])
                ids = draw(np.ones_like(pop) if uniform else pop, n)
                c = np.stack([np.bincount(ids[:, l].ravel(), minlength=NE) for l in range(L)])
                assert (c.sum(1) == n * K).all()
                np.savez_compressed(f"{d}/counts/{split}_{dom}.npz", counts=c, mass=c * 0.25, ntok=n, uniq=np.full(L, NE * 0.9), layers=np.arange(L))
        # decode: a cyclic scan of all experts defeats LRU/LFU smaller than the cycle; a stable hot set is learned at once
        scan = (np.arange(S * 4 * K).reshape(S * 4, 1, K) % NE).repeat(L, 1).astype(np.uint16)
        np.savez_compressed(f"{d}/decode.npz", ids=scan,
                            gen=np.zeros(S * 4, np.uint16), dt=np.full(S * 4, .1, np.float32), gen_domain=np.array(["chat"] * (S * 4)), layers=np.arange(L))

    ns = argparse.Namespace
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        perm = {(dom, l): rng.permutation(NE) for dom in DOMAINS for l in range(L)}
        shared = {l: rng.permutation(NE) for l in range(L)}
        def skewed(dom, l):  # 70% of the popularity is shared across domains, 30% is the domain's own order
            return 0.7 * base[np.argsort(shared[l])] + 0.3 * base[np.argsort(perm[dom, l])]
        write(d1, skewed)
        R = analyze(ns(model="tiny", dir=d1, json=None, md=None, quiet=True))
        c = R["coverage"]["pooled"]
        v = [c[str(p)]["mean"] for p in sorted(c, key=int)]
        assert all(x <= y + 1e-9 for x, y in zip(v, v[1:])) and v[-1] <= 1, v
        v = [c[p]["mean"] for p in ("10", "25", "50")]
        assert c["50"]["mean"] > 0.75, "a Zipf profile must cover far more than its share"
        s = R["static"]["25"]
        assert s["per_layer_alloc"]["heldout"] > 0.25 + 0.2, "static profile should beat uniform by a wide margin"
        assert s["oracle_heldout"] >= s["per_layer_alloc"]["heldout"] - 1e-9 and s["global_alloc"]["heldout"] > 0.25
        assert all(R["transfer"]["25"]["matrix"][a][a] >= R["transfer"]["25"]["matrix"][a][b] - 0.05 for a in DOMAINS for b in DOMAINS)
        dd = R["decode"]
        assert dd["lru"]["25"] == 0 and dd["lru"]["50"] == 0, "LRU smaller than a cyclic scan never hits"  # cap 16/32 < 64
        write(d2, lambda dom, l: np.ones(NE), uniform=True)
        U = analyze(ns(model="tiny", dir=d2, json=None, md=None, quiet=True))
        for p in (10, 25, 50):
            assert abs(U["coverage"]["pooled"][str(p)]["mean"] - round(p * NE / 100) / NE) < 0.03, "uniform routing: top-k% covers k%"
        assert abs(U["static"]["25"]["per_layer_alloc"]["heldout"] - 0.25) < 0.03
        # LRU/LFU/seeded replay on a stable hot set
        hot = np.tile(np.arange(K), (200, L, 1)).astype(np.uint16)
        assert replay(hot, 8, "lru") > 0.98 and replay(hot, 8, "lfu") > 0.98
        assert replay(hot, 8, "lru", seed=np.arange(NE)[None, :].repeat(L, 0) < K) == 1.0
    print(f"selftest ok: Zipf coverage at 10/25/50% = {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f} (uniform .10/.25/.50), static 25% hit {s['per_layer_alloc']['heldout']:.2f} "
          f"vs oracle {s['oracle_heldout']:.2f}, uniform data {U['static']['25']['per_layer_alloc']['heldout']:.2f}, LRU on a scan {dd['lru']['25']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", plan), ("prep", prep), ("trace", trace), ("analyze", analyze), ("selftest", selftest)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name != "selftest":
            p.add_argument("model", choices=list(MODELS))
        if name == "prep":
            p.add_argument("out")
            p.add_argument("--cal-seqs", type=int, default=24, help="sequences of 2048 per domain for the profile (24 x 6 = ~295k tokens)")
            p.add_argument("--held-seqs", type=int, default=4, help="held-out sequences per domain (4 x 6 = ~49k tokens)")
        if name == "trace":
            p.add_argument("--weights", required=True)
            p.add_argument("--data", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--raw", help="where the raw ids go (large; moved into the results at the end)")
            p.add_argument("--batch", type=int, default=2)
            p.add_argument("--gens", type=int, default=20)
            p.add_argument("--gen-new", type=int, default=128)
            p.add_argument("--gen-min", type=float, default=25, help="minutes for all generations")
            p.add_argument("--deadline", type=float, default=0, help="epoch seconds by which the job must be over")
        if name == "analyze":
            p.add_argument("dir")
            p.add_argument("--json")
            p.add_argument("--md")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
