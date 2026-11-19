"""The lossless KV cache's stress test: the pages that need the most of its exception entries (a sequence's first page, the attention sink; pages of one token repeated; random
tokens; text in fourteen languages; tables, hex, base64, code), written and read back, with none lost and none differing.

    python kv_stress.py capture TABLES [--out DIR] [--pages N] [--pct P] [--raw P] [--vocab] [--quick] [--model NAME]
    python kv_stress.py engine  TABLES [--out DIR] [--verify] [--ext] [--blocks N] [--quick] [--model NAME]

capture: vLLM (stock, eager) runs each population of prompts and the keys and values every decoder layer hands its cache (after the key norm and the rotary embedding: what
vLLM stores) go, in the hook, through Glyd's write kernel into a cache of N pages a layer (36 layers: the pages of a previous owner's random bytes, written round and round
like a block pool's) with an escape pool of P% of its bytes and a raw pool of R% (vllm_kv.make_escape's, applied to this cache), and are read back and compared bit for bit.
Populations: windows of the bundled text (natural), chat prompts through the model's template, text in fourteen languages, code, tables and the like, random token ids (the whole
embedding matrix, the vocabulary, and as `vllm bench serve` makes them: decoded to text and encoded again), one token repeated (special tokens, common, random), short cycles
and phrases repeated, 12k-token prompts; --vocab: every token id of the embedding matrix repeated 32 times (its page 0 is its first token, a sink, and page 1 pure repetition: the
worst tokens are found here); and the cache filled entirely with the worst of them (the pool is sized for that or the run says it was not). Written: per population the values
compared, differing, pages made raw and lost, the most entries a page was given, the pool's chunks taken at most; at the end the pools' books against the pages'.

engine: vLLM with kv lossless (and --verify: a bf16 shadow of every layer's cache, every write compared) on the same kinds of prompts through generate(): the engine raising
GlydKVVerifyError is the failure; written: per workload what the pool took (kv_status), and the cache entirely of repeated tokens, with the pool the plugin made. --ext: prefix
caching and chunked prefill on (1,024 tokens a step): the prompts' shared starts are cache hits and the long ones go in chunks, so rows read their contexts through the staging cache (with --verify, compared with the shadow).

Under the Business Source License 1.1 (LICENSE), as the rest of Glyd's GPU code."""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import kv_stress_text as ST  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
H, D, LAYERS, PT = 8, 128, 36, 16
# Qwen3-8B's tokens whose pages of themselves repeated needed the most of the escape pool, found by repeating every id of the embedding matrix 32 times: the ones that took the most chunks a
# layer-page (2.0 and more of the average 1.6) and the ones with the page that needed the most entries (783 of the 1,039 a page can hold). The cache is filled with each in turn.
WORST = [135553, 36292, 131124, 13454, 122261, 146234, 49097, 95457, 28286, 139713, 151213, 61656]


def log(*a):
    print(*a, flush=True)


def build_populations(tok, quick=False, vocab=False, rng=None):
    """[(name, prompts)]: every prompt of a population the same length (a multiple of 16)."""
    from glyd_gpu import kvcal

    rng = rng or random.Random(7)
    docs = kvcal.corpus()
    doc_ids = [(kind, name, kvcal.tokens(tok, kind, body)) for kind, name, body in docs]
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for f in ("README.md", "gpu/README.md", "CHANGELOG.md", "docs/engineering-notes.md", "gpu/vllm/README.md"):
        p = os.path.join(repo, f)
        if os.path.exists(p):
            doc_ids.append(("repo", f, tok.encode(open(p, encoding="utf-8").read())))
    scale = 0.1 if quick else 1.0
    cnt = lambda n: max(8, int(n * scale))

    def windows(ids, n, lp):
        return [ids[o : o + lp] for o in (rng.randrange(0, max(1, len(ids) - lp)) for _ in range(n))] if len(ids) > lp else []

    out = []
    nat = []
    for kind, name, ids in doc_ids:
        if len(ids) >= 400:
            nat += windows(ids, cnt(max(8, len(ids) // 30)), 64)
    out.append(("natural", nat))
    systems = [None, "You are a helpful assistant.", "You are a senior Python engineer. Answer briefly and show code.", "Answer in the language of the question.", "You are a careful translator.",
               "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "Today is Monday. Be concise.", "You are a tutor for high school physics.", "Respond only with JSON.", "You are a support agent for a train company."]
    paras = [p for kind, name, body in docs if kind == "prose" for p in body.split("\n\n") if len(p) > 300]
    users = [" ".join(p.split()[:70]) for p in paras] + list(ST.MULTILINGUAL.values()) + ["Explain this code:\n" + c for c in ST.CODE.values()]
    chat = []
    for u in users:
        for s_ in systems:
            ids = tok.apply_chat_template(([{"role": "system", "content": s_}] if s_ else []) + [{"role": "user", "content": u}], tokenize=True, add_generation_prompt=True)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
            if len(ids) >= 64:
                chat.append(ids[:64])
    out.append(("chat", chat[: cnt(len(chat))] if quick else chat))
    ml = []
    for lang, t in ST.MULTILINGUAL.items():
        ids = tok.encode(t)
        ml += [ids[:64]] + windows(ids, cnt(40), 64)
    out.append(("multilingual", [x for x in ml if len(x) == 64]))
    other = []
    for kind, t in list(ST.CODE.items()):
        other += windows(tok.encode(t), cnt(20), 64)
    for kind, t in ST.structured(3, 2 if quick else 12):
        other += windows(tok.encode(t), 3, 64)
    out.append(("code and structured", [x for x in other if len(x) == 64]))
    out.append(("random: the embedding matrix", [[rng.randrange(0, 151936) for _ in range(64)] for _ in range(cnt(2000))]))
    out.append(("random: the vocabulary", [[rng.randrange(0, 151669) for _ in range(64)] for _ in range(cnt(2000))]))
    txt = []
    while len(txt) < cnt(2000):
        re = tok.encode(tok.decode([rng.randrange(0, 151936) for _ in range(120)]))
        if len(re) >= 64:
            txt.append(re[:64])
    out.append(("random as vllm bench makes them (decoded, encoded)", txt))
    special = list(range(151643, 151669))
    common = [tok.encode(w)[0] for w in ["the", " the", ".", ",", "\n", " ", "0", "1", "a", "I", "-", "=", "#", "*", "(", ")", "{", "}", "def", "import", "Question", "!", "?", ";", ":", "\"", "'"]]
    rep = [[t] * 256 for t in special + common] + [[rng.randrange(0, 151936)] * 256 for _ in range(cnt(150))]
    out.append(("one token repeated (256)", rep))
    cyc = []
    for L in (2, 3, 4, 5, 8):
        for _ in range(cnt(40)):
            pat = [rng.randrange(0, 151669) for _ in range(L)]
            cyc.append((pat * (256 // L + 1))[:256])
    out.append(("short cycles of random tokens (256)", cyc))
    base = [x for x in doc_ids if x[0] in ("prose", "docs", "code")]
    loops = []
    for _ in range(cnt(150)):
        ids = rng.choice(base)[2]
        L = rng.choice((6, 10, 16, 24, 40))
        o = rng.randrange(0, len(ids) - L)
        loops.append((ids[o : o + L] * (256 // L + 1))[:256])
    out.append(("a phrase repeated (256)", loops))
    out.append(("one token repeated (4096)", [[rng.choice(common + special + [rng.randrange(0, 151669)])] * 4096 for _ in range(4 if quick else 24)]))
    if not quick:
        lng = kvcal.sequences(doc_ids[: len(docs)], 12288)
        out.append(("natural, 12k tokens", [ids for ids, _ in lng]))
    if vocab:
        out.append(("every token id repeated 32 times", [[i] * 32 for i in range(151936)]))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
class Rig:
    """The pages and the pool the hook writes into: N pages a layer, written round and round (a ring over a shuffled order), the escape pool of pct % of their bytes."""

    def __init__(self, tables, leaves, pct, raw_pct=40.0):
        import torch
        from glyd_gpu import _lib, kvcal

        self.torch, self._lib = torch, _lib
        self.t = kvcal.Tables(tables)
        self.names = None
        self.A = _lib.kv_free_bay(H)
        self.page = _lib.kv_leaf_bytes(H, D, self.A)
        self.N = leaves
        gen = torch.Generator(device="cuda").manual_seed(1)
        self.store = []
        for li in range(LAYERS):
            self.store.append(torch.randint(0, 256, (leaves * self.page,), dtype=torch.uint8, device="cuda", generator=gen))
        self.pct = pct
        self.chunks = max(1024, int(pct / 100.0 * leaves * LAYERS * self.page / (64 * 4 + 4)))
        self.lumps = max(256, int(raw_pct / 100.0 * leaves * LAYERS * self.page / (H * 2048)))
        self.esc = _lib.Extra(self.chunks, None, self.lumps, H)
        self.st = torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device="cuda")
        self.order = torch.randperm(leaves, generator=torch.Generator().manual_seed(3)).to("cuda")
        self.cursor = 0
        self.written = [torch.zeros(leaves, dtype=torch.bool, device="cuda") for _ in range(LAYERS)]
        self.bad = torch.zeros((), dtype=torch.int64, device="cuda")
        self.values = 0
        self.bases = None
        self.step_pages = None

    def bind(self, names):
        torch = self.torch
        self.names = names
        self.bases = [tuple(torch.from_numpy(a).cuda() for a in self.t.layer(n)) for n in names]

    def hook(self, li, k, v, lp):
        """One layer's step: the step's prompts (lp tokens each) written and read back."""
        torch, _lib = self.torch, self._lib
        T = k.shape[0]
        n = T // lp
        P = (lp + PT - 1) // PT
        if li == 0:  # (the step's pages: the same for every layer, from the ring)
            idx = (self.cursor + torch.arange(n * P, device="cuda")) % self.N
            self.step_pages = self.order[idx]
            self.cursor = (self.cursor + n * P) % self.N
        leaves = self.step_pages
        t = torch.arange(T, device="cuda", dtype=torch.int64)
        tok_in = t % lp
        cells = leaves.to(torch.int64)[(t // lp) * P + tok_in // PT] * PT + tok_in % PT
        kb, vb = self.bases[li]
        _lib.kv_write(k, v, cells, self.store[li], self.page, kb, vb, self.A, self.esc, li, self.st)
        ok = torch.empty(n * P, PT, H, D, dtype=torch.int16, device="cuda")
        ov = torch.empty_like(ok)
        _lib.kv_unpack(self.store[li], self.page, leaves.to(torch.int32), kb, vb, self.A, self.esc, ok, ov)
        # (the tokens of each prompt are the first lp of its P pages: the rest of the last page is a previous owner's)
        sel = (torch.arange(n, device="cuda")[:, None] * P * PT + torch.arange(lp, device="cuda")[None, :]).reshape(-1)
        self.bad += (ok.view(-1, H, D)[sel] != k).sum() + (ov.view(-1, H, D)[sel] != v).sum()
        self.written[li][leaves.to(torch.int64)] = True
        self.values += 2 * T * H * D

    def books(self):
        """(chunks taken by the bitmap, by the owners, named by the directories of the pages written, status [4], whether every directory word names a chunk owned by its page; the same of the raw
        pool's slabs: the bitmap, the owners, named by the raw pages, status [8], whether each is owned by its page)."""
        torch, e = self.torch, self.esc
        taken = sum(bin(x & 0xFFFFFFFF).count("1") for x in e.bits.tolist())
        owned = int((e.owner != 0).sum())
        rtaken = sum(bin(x & 0xFFFFFFFF).count("1") for x in e.rbits.tolist())
        rowned = int((e.rowner != 0).sum())
        named, good, rnamed, rgood = 0, True, 0, True
        for li in range(LAYERS):
            idx = self.written[li].nonzero().view(-1)
            if not idx.numel():
                continue
            rows = self.store[li].view(-1, self.page)[idx]
            used = rows[:, 16 * H : 16 * H + 4].contiguous().view(torch.int32).flatten()
            rawp = used < 0
            d = rows[:, 16 * H + 4 : 16 * H + 4 + 64].contiguous().view(torch.int32)
            if rawp.any():
                d0 = d[rawp][:, 0]
                rnamed += int(d0.numel())
                want = (((li + 1) << 24) | idx[rawp].to(torch.int32))
                rgood &= bool((e.rowner[(d0 - 1).to(torch.int64)] == want).all())
            d[rawp] = 0
            nz = d != 0
            named += int(nz.sum())
            if nz.any():
                want = (((li + 1) << 24) | idx.to(torch.int32))[:, None].expand_as(d)[nz]
                got = e.owner[(d[nz] - 1).to(torch.int64)]
                good &= bool((got == want).all())
        return taken, owned, named, int(self.st[4]), good, rtaken, rowned, rnamed, int(self.st[8]), rgood


def capture_main(a):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType
    from glyd_gpu import kernels as g

    g.lib()
    tables = a[1]
    out_dir = a[a.index("--out") + 1] if "--out" in a else "kv_stress_results"
    leaves = int(a[a.index("--pages") + 1]) if "--pages" in a else 1000
    pct = float(a[a.index("--pct") + 1]) if "--pct" in a else 1.0
    raw_pct = float(a[a.index("--raw") + 1]) if "--raw" in a else 40.0
    quick, vocab = "--quick" in a, "--vocab" in a
    model = a[a.index("--model") + 1] if "--model" in a else MODEL
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model)
    pops = build_populations(tok, quick, vocab)
    longest = max(len(p[0]) for _, p in pops if p)
    llm = LLM(model, dtype="bfloat16", enforce_eager=True, max_model_len=longest + 64, max_num_batched_tokens=max(longest + 64, 8192), enable_prefix_caching=False, enable_chunked_prefill=False,
              gpu_memory_utilization=0.9, max_num_seqs=512, num_gpu_blocks_override=max(800, longest // PT + 40), seed=0)
    rig = Rig(tables, leaves, pct, raw_pct)
    S = {"on": False, "lp": 64}

    def install(m):
        names = []
        for mod in m.modules():
            if not isinstance(mod, Attention) or mod.attn_type != AttentionType.DECODER:
                continue
            li = len(names)
            names.append(mod.layer_name)

            def hook(mod_, args_, kwargs_, li=li):
                if not S["on"]:
                    return
                k = args_[1] if len(args_) > 1 else kwargs_.get("key")
                v = args_[2] if len(args_) > 2 else kwargs_.get("value")
                k = k.detach().reshape(-1, mod_.num_kv_heads, mod_.head_size).contiguous().view(torch.int16)
                v = v.detach().reshape(-1, mod_.num_kv_heads, mod_.head_size_v).contiguous().view(torch.int16)
                rig.hook(li, k, v, S["lp"])

            mod.register_forward_pre_hook(hook, with_kwargs=True)
        return names

    rig.bind(llm.apply_model(install)[0])
    log(f"kv_stress capture: {LAYERS} layers, {leaves} pages a layer ({leaves * PT:,} tokens; a page {rig.page:,} bytes, arena {rig.A}), escape pool {rig.chunks:,} chunks ({rig.esc.buf.numel() / 2**20:.1f} MiB, {pct:g}% of the cache's bytes), raw pool {rig.lumps:,} slabs ({rig.esc.raw.numel() / 2**20:.1f} MiB)")
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    results, totals = [], {"differing": 0, "lost": 0, "raw": 0}

    def run(name, prompts):
        lp = len(prompts[0])
        assert lp % PT == 0 and all(len(p) == lp for p in prompts), name
        S["lp"] = lp
        st = rig.st
        st[1] = 0
        st[5] = st[4]
        before = st.tolist()
        bad0, vals0 = int(rig.bad), rig.values
        t0 = time.time()
        step = max(64, 8192 // lp * 64) if lp < 1024 else 2
        S["on"] = True
        for i in range(0, len(prompts), step):
            llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts[i : i + step]], sp, use_tqdm=False)
        S["on"] = False
        torch.cuda.synchronize()
        s1 = st.tolist()
        row = {"population": name, "prompts": len(prompts), "tokens_each": lp, "values": rig.values - vals0, "differing": int(rig.bad) - bad0, "raw_pages": s1[10] - before[10], "lost": s1[11] - before[11], "entries": s1[3] - before[3],
               "most_entries_a_page_was_given": s1[1], "pool_chunks_taken_at_most": s1[5], "store_chunks": rig.chunks, "pages_that_took_a_chunk": s1[6] - before[6], "seconds": round(time.time() - t0, 1)}
        results.append(row)
        totals["differing"] += row["differing"]
        totals["lost"] += row["lost"]
        totals["raw"] += row["raw_pages"]
        log(f"{name:55s} {row['prompts']:>7,} prompts of {lp:>5}: {row['values']:>14,} values, {row['differing']} differ, {row['raw_pages']:,} raw, {row['lost']} lost; a page was given {row['most_entries_a_page_was_given']:>4} entries at most; "
            f"pool {row['pool_chunks_taken_at_most']:>6,} of {rig.chunks:,} chunks taken at most ({row['pages_that_took_a_chunk']:,} pages took one) ({row['seconds']:.0f} s)")

    for name, prompts in pops:
        if prompts:
            run(name, prompts)
    fills = []
    for t in WORST:  # (the cache entirely of one token's pages: more pages than it holds, so that at the end every page of every layer is one of them)
        run(f"the cache filled with token {t} repeated (4096 each)", [[t] * 4096 for _ in range(leaves // 256 + 1)])
        fills.append({"token": t, "pool_chunks_taken_after": int(rig.st[4]), "chunks_a_layer_page": round(int(rig.st[4]) / (leaves * LAYERS), 3)})
        log(f"  every page of every layer is now token {t}'s: the pool holds {int(rig.st[4]):,} of {rig.chunks:,} chunks ({int(rig.st[4]) / (leaves * LAYERS):.3f} a layer-page; the pool has {rig.chunks / (leaves * LAYERS):.3f})")
    books = rig.books()
    ok = books[0] == books[1] == books[2] == books[3] and books[4] and books[5] == books[6] == books[7] == books[8] and books[9]
    log(f"the pools' books after all of it: chunks {books[0]} taken by the bitmap, {books[1]} by the owners, {books[2]} named by the directories of the pages written, status {books[3]}, each owned by its page: {books[4]}; "
        f"slabs {books[5]} by the bitmap, {books[6]} by the owners, {books[7]} named by the raw pages, status {books[8]}, each owned by its page: {books[9]}")
    total = {"differing": totals["differing"], "lost": totals["lost"], "raw_pages": totals["raw"], "values": rig.values, "books_agree": bool(ok), "stale_directory_words": int(rig.st[7]), "fills": fills}
    json.dump({"leaves": leaves, "pct": pct, "chunks": rig.chunks, "lumps": rig.lumps, "bay": rig.A, "leaf_bytes": rig.page, "rows": results, "total": total}, open(os.path.join(out_dir, "capture.json"), "w"), indent=1)
    log(f"kv_stress capture: {rig.values:,} values written and read back: {totals['differing']} differ, {totals['raw']:,} pages made raw, {totals['lost']} lost; books {'agree' if ok else 'DISAGREE'}")
    return 0 if totals["differing"] == 0 and totals["lost"] == 0 and ok else 1


def engine_main(a):
    """vLLM with kv lossless on the same kinds of prompts, generate() each: nothing may raise (GlydKVVerifyError, with --verify: a value that decodes to other bits than were written),
    and the cache is filled entirely with one token repeated."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from glyd_gpu import vllm_kv

    tables = a[1]
    out_dir = a[a.index("--out") + 1] if "--out" in a else "kv_stress_results"
    verify, quick, ext = "--verify" in a, "--quick" in a, "--ext" in a
    blocks = int(a[a.index("--blocks") + 1]) if "--blocks" in a else (700 if verify else None)
    model = a[a.index("--model") + 1] if "--model" in a else MODEL
    os.makedirs(out_dir, exist_ok=True)
    os.environ["GLYD_KV_TABLES"] = tables
    tok = AutoTokenizer.from_pretrained(model)
    rng = random.Random(11)
    pops = dict(build_populations(tok, True))
    kw = dict(model=model, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.9, seed=0, max_model_len=4352, max_num_batched_tokens=1024 if ext else 4352, max_num_seqs=64,
              enable_prefix_caching=ext, enable_chunked_prefill=ext, quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless", "verify": verify}})
    if blocks:
        kw["num_gpu_blocks_override"] = blocks
    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config
    nblocks = cfg.cache_config.num_gpu_blocks
    status = lambda: llm.apply_model(lambda m: vllm_kv.kv_status())[0]
    log(f"kv_stress engine: {nblocks:,} blocks ({nblocks * PT:,} tokens){' (verify: every write compared with a bf16 shadow of each layer)' if verify else ''}{' (prefix caching and chunked prefill on, 1,024 tokens a step)' if ext else ''}; escape pool {next(iter(status()['store'].values())):,} chunks")
    fill_tokens = WORST
    n_fill = max(2, int(nblocks * PT * 0.9) // 1000)
    texts = ["the " * 40, "\n" * 40, ". " * 40, "ab" * 40, "!" * 60]
    rand1k = []
    while len(rand1k) < (6 if quick else 48):
        re = tok.encode(tok.decode([rng.randrange(0, 151936) for _ in range(1300)]))
        if len(re) >= 1024:
            rand1k.append(re[:1024])
    loops = [tok.encode(t) * 12 for t in texts]
    work = [
        ("sinks: chat prompts (the first page of every sequence), 16 new tokens", pops["chat"][: 40 if quick else 400], 16, False),
        ("natural windows, 32 new tokens", pops["natural"][: 40 if quick else 300], 32, False),
        ("text in fourteen languages, 64 new tokens", pops["multilingual"], 64, False),
        ("code, tables, hex, base64, 32 new tokens", pops["code and structured"], 32, False),
        ("random token ids over the embedding matrix, 32 new tokens", pops["random: the embedding matrix"][: 40 if quick else 400], 32, True),
        ("random 1,024-token prompts as vllm bench makes them, 128 new tokens (ignore EOS)", rand1k, 128, True),
        ("a phrase or a few characters repeated (480 tokens), 256 new tokens (its continuation repeats)", loops, 256, True),
        (f"the cache filled entirely: {n_fill} prompts of one token repeated 1,000 times, 16 new tokens", [[fill_tokens[i % len(fill_tokens)]] * 1000 for i in range(n_fill)], 16, True),
        ("one token repeated 4,000 times, 64 new tokens", [[t] * 4000 for t in fill_tokens[:3]], 64, True),
    ]
    rows, failed = [], []
    for name, prompts, new, ignore in work:
        t0 = time.time()
        try:
            llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], SamplingParams(temperature=0, max_tokens=new, ignore_eos=ignore), use_tqdm=False)
            err = None
        except BaseException as e:  # (the engine in this process: what its step raised)
            err = f"{type(e).__name__}: {str(e)[:300]}"
            failed.append(name)
        st = status()
        dev = next(v for k, v in st.items() if k not in ("verified", "store", "second"))
        row = {"workload": name, "prompts": len(prompts), "raised": err, "pages_made_raw": dev[10], "pages_lost": dev[11], "most_entries_a_page_was_given": dev[1], "pool_chunks_taken_now": dev[4], "pool_chunks_taken_at_most": dev[5], "pool_pages": dev[6],
               "store_chunks": next(iter(st["store"].values())), "values_compared": st["verified"][0], "pages_checked": st["verified"][1], "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        log(f"{name}: {len(prompts)} prompts: {'RAISED ' + err if err else 'no stop'}; {dev[10]:,} pages made raw, {dev[11]} lost; a page was given {dev[1]} entries at most; pool {dev[5]:,} of {row['store_chunks']:,} chunks taken at most, {dev[4]:,} now"
            + (f"; verify: {st['verified'][0]:,} values compared over {st['verified'][1]:,} page checks, none differ" if verify else "") + f" ({row['seconds']:.0f} s)")
        if err:
            break
    json.dump({"blocks": nblocks, "verify": verify, "ext": ext, "rows": rows}, open(os.path.join(out_dir, f"engine{'-ext' if ext else ''}{'-verify' if verify else ''}.json"), "w"), indent=1)
    log(f"kv_stress engine: {'no engine stop' if not failed else 'STOPPED in ' + failed[0]}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("capture", "engine"):
        raise SystemExit(__doc__)
    sys.exit((capture_main if sys.argv[1] == "capture" else engine_main)(sys.argv[1:]))
