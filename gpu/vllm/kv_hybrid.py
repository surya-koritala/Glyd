"""The lossless KV cache on a model of head size 256 with linear-attention layers (Qwen3.5: Gated DeltaNet beside full attention), one engine a process, against vLLM's own cache:

    python kv_hybrid.py run OUT.json [--kv off|lossless] [--verify] [--model M] [--mml 8192] [--util 0.9] [--seqs 64] [--blocks N] [--churn N] [--fraction F] [--block-size N]
                                     [--kv-memory BYTES] [--capacity-only] [--alone]
    python kv_hybrid.py compare STOCK.json KV.json

run: vLLM in this process (--quantization glyd: Glyd's weights, fraction F of the layers, 1; the text model only; prefix caching and chunked prefill on, as vLLM has them, 2,048 tokens a step), the corpus's text (kvcal):
  - prompts of 64 to 7,000 tokens: every prompt token's logprob (the prompt runs in chunks: the later chunks read the earlier ones' cache), then 128 greedy tokens and their logprobs;
  - a prefix of 3,000 tokens, one request with it alone, then six that share it (prefix hits): the tokens each found cached, its first token's 5 logprobs (that token comes from the prompt's own
    tokens over the cached context), then 64 greedy tokens;
  - the KV cache vLLM made: its blocks, block size, tokens and concurrency at --max-model-len (its log's "GPU KV cache size");
  - --churn N: N requests of 200 to 3,000 tokens, 48 out, at once, on a cache of --blocks blocks: blocks pass between the cache's groups and requests are preempted (with --verify every write is checked);
  - with kv lossless: the reserves' use, the pages handed over, verify's counts.
--kv-memory: the KV cache's bytes, fixed (vLLM's kv_cache_memory_bytes: the same memory for both caches, whatever the memory profile of a start finds); kv lossless's reserves are apart, and recorded.
--capacity-only: the KV cache vLLM made, and nothing run. --alone: each request generated alone (another batch than the default's, for vLLM's own spread).
compare: the prompt logprobs bit for bit, the prefix hits' cached tokens and first tokens' logprobs bit for bit, the greedy tokens' agreement, the KV tokens' ratio. A hybrid model's block size sets
how vLLM steps its prompts (Mamba's align mode cuts them at block boundaries) and what a prefix hit can take: vLLM's own cache is compared at kv lossless's block size (--block-size), and its default
block's run is the one for the KV cache's size.
Run as check_vllm.py is (the library by GLYD_GPU_LIB, this tree's glyd_gpu on PYTHONPATH)."""
import json
import os
import sys
import time

MODEL = "Qwen/Qwen3.5-9B"


def arg(a, name, default):
    return type(default)(a[a.index(name) + 1]) if name in a else default


def texts(tok):
    """Token ids of the corpus's sections, in turn (prose, docs, code, chat, tables): one stream."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "glyd-gpu-py"))
    from glyd_gpu import kvcal

    ids = []
    for kind, _, body in kvcal.corpus():
        ids += kvcal.tokens(tok, kind, body)
    return ids


def run(out, a):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kv, verify, model = arg(a, "--kv", "lossless"), "--verify" in a, arg(a, "--model", MODEL)
    mml, util, seqs, blocks, churn, frac = arg(a, "--mml", 8192), arg(a, "--util", 0.9), arg(a, "--seqs", 64), arg(a, "--blocks", 0), arg(a, "--churn", 0), arg(a, "--fraction", 1.0)
    glyd = {"kv": kv, "fraction": frac}
    if verify:
        glyd["verify"] = True
    kw = dict(quantization="glyd", language_model_only=True, max_model_len=mml, gpu_memory_utilization=util, max_num_seqs=seqs, seed=0, max_num_batched_tokens=2048, enable_prefix_caching=True,
              enable_chunked_prefill=True, additional_config={"glyd": glyd})
    if blocks:
        kw["num_gpu_blocks_override"] = blocks
    if "--block-size" in a:  # (vLLM's own cache at kv lossless's block, 704 tokens for Qwen3.5-9B: a hybrid model's prefill steps and prefix hits follow the block size, so this is what kv lossless's outputs are compared with bit for bit)
        kw["block_size"] = arg(a, "--block-size", 16)
    if "--kv-memory" in a:
        kw["kv_cache_memory_bytes"] = arg(a, "--kv-memory", 0)
    if verify:  # (verify reads its checks on the host after every write: no CUDA graphs)
        kw["enforce_eager"] = True
    t0 = time.time()
    llm = LLM(model, **kw)
    res = {"model": model, "kv": kv, "verify": verify, "args": {k: v for k, v in kw.items() if k != "additional_config"}, "glyd": glyd, "load_s": round(time.time() - t0, 1)}
    cc = llm.llm_engine.vllm_config.cache_config
    res["cache"] = {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size, "kv_cache_size_tokens": getattr(cc, "kv_cache_size_tokens", None),
                    "max_concurrency": getattr(cc, "kv_cache_max_concurrency", None), "mamba_cache_mode": getattr(cc, "mamba_cache_mode", None), "kv_cache_layout": getattr(cc, "kv_cache_layout", None)}
    if kv == "lossless":
        from glyd_gpu import vllm_kv

        res["cache"]["reserves_bytes"] = sum(e.buf.numel() + (e.raw.numel() if e.raw is not None else 0) for e in vllm_kv._EXTRA.values())
    print("cache:", res["cache"], flush=True)
    if "--capacity-only" in a:
        with open(out, "w") as f:
            json.dump(res, f)
        return
    alone = "--alone" in a
    gen = lambda ps, sp: [o for p in ps for o in llm.generate([p], sp, use_tqdm=False)] if alone else llm.generate(ps, sp, use_tqdm=False)
    tok = llm.get_tokenizer()
    stream = texts(tok)
    piece = lambda off, n: [stream[(off + i) % len(stream)] for i in range(n)]
    lp = lambda d: None if d is None else {str(k): [v.logprob, v.rank] for k, v in d.items()}
    t1 = time.time()
    if churn:  # (first: a fresh cache, every block passing between the groups as requests come and go)
        lens = [200 + (i * 997) % 2800 for i in range(churn)]
        outs = llm.generate([TokensPrompt(prompt_token_ids=piece(i * 1013, L)) for i, L in enumerate(lens)], SamplingParams(max_tokens=48, temperature=0.0, ignore_eos=True), use_tqdm=False)
        res["churn"] = {"requests": churn, "tokens": sum(lens), "outputs": [list(o.outputs[0].token_ids) for o in outs], "s": round(time.time() - t1, 1)}
        print("churn:", churn, "requests", res["churn"]["s"], "s", flush=True)
    lens = [64, 300, 1000, 2049, 3000, 4500, 7000]
    lens = [L for L in lens if L + 128 < mml]
    prompts = [piece(37 * i + 5000, L) for i, L in enumerate(lens)]
    outs = gen([TokensPrompt(prompt_token_ids=p) for p in prompts], SamplingParams(max_tokens=128, temperature=0.0, logprobs=5, prompt_logprobs=1, ignore_eos=True))
    res["prompts"] = [{"len": L, "prompt_logprobs": [lp(d) for d in o.prompt_logprobs], "tokens": list(o.outputs[0].token_ids), "logprobs": [lp(d) for d in o.outputs[0].logprobs]} for L, o in zip(lens, outs)]
    pre = piece(20000, 3000)
    first = llm.generate([TokensPrompt(prompt_token_ids=pre + piece(30000, 40))], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    tails = [piece(31000 + 500 * i, 50 + 70 * i) for i in range(6)]
    outs = gen([TokensPrompt(prompt_token_ids=pre + t) for t in tails], SamplingParams(max_tokens=64, temperature=0.0, logprobs=5, ignore_eos=True))
    res["prefix"] = {"first_cached": first[0].num_cached_tokens, "hits": [{"tail": len(t), "cached": o.num_cached_tokens, "first_logprobs": lp(o.outputs[0].logprobs[0]), "tokens": list(o.outputs[0].token_ids),
                                                                         "logprobs": [lp(d) for d in o.outputs[0].logprobs]} for t, o in zip(tails, outs)]}
    res["run_s"] = round(time.time() - t1, 1)
    res["alone"] = alone
    if kv == "lossless":
        from glyd_gpu import vllm_kv

        st = vllm_kv.kv_status()
        res["glyd_kv"] = {"status": {str(k): v for k, v in st.items() if isinstance(k, int)}, "store": {str(k): v for k, v in st["store"].items()}, "second": {str(k): v for k, v in st["second"].items()},
                          "verified": st["verified"], "released_pages": vllm_kv.RELEASED[0], "reuse": vllm_kv._STATE.reuse, "early": vllm_kv._STATE.early, "layers": vllm_kv._STATE.layers,
                          "tables": vllm_kv._STATE.path, "tables_meta": {k: vllm_kv._STATE.tables.meta.get(k) for k in ("positions", "heads", "channels", "layers", "held_out_rate", "seconds")}}
        print("glyd kv:", {k: v for k, v in res["glyd_kv"].items() if k != "status"}, flush=True)
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"run: {out} ({res['run_s']} s)", flush=True)


def compare(a_path, b_path):
    A, B = json.load(open(a_path)), json.load(open(b_path))
    ok = True
    n = same = 0
    for pa, pb in zip(A["prompts"], B["prompts"]):
        for x, y in zip(pa["prompt_logprobs"], pb["prompt_logprobs"]):
            if x is None:
                continue
            n += 1
            same += x == y
    print(f"prompt logprobs: {same} of {n} prompt tokens bit for bit ({len(A['prompts'])} prompts of {', '.join(str(p['len']) for p in A['prompts'])} tokens, in chunks of 2,048)")
    ok &= same == n and n > 0
    agree, steps, whole = 0, 0, 0
    for pa, pb in zip(A["prompts"] + A["prefix"]["hits"], B["prompts"] + B["prefix"]["hits"]):
        ta, tb = pa["tokens"], pb["tokens"]
        k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), len(ta))
        agree += k
        steps += len(ta)
        whole += k == len(ta)
    print(f"greedy tokens: {whole} of {len(A['prompts']) + len(A['prefix']['hits'])} sequences the same to the end; {agree} of {steps} tokens before the first that differs")
    hits = list(zip(A["prefix"]["hits"], B["prefix"]["hits"]))
    cached = all(x["cached"] == y["cached"] and x["cached"] > 0 for x, y in hits)
    firsts = sum(x["first_logprobs"] == y["first_logprobs"] for x, y in hits)
    print(f"prefix hits: cached tokens {[x['cached'] for x, _ in hits]} and {[y['cached'] for _, y in hits]} ({'equal' if cached else 'NOT equal'}); first tokens' 5 logprobs bit for bit: {firsts} of {len(hits)}")
    ok &= cached and firsts == len(hits)
    ca, cb = A["cache"], B["cache"]
    if ca.get("kv_cache_size_tokens") and cb.get("kv_cache_size_tokens"):
        print(f"KV cache: {ca['kv_cache_size_tokens']:,} tokens ({ca['num_gpu_blocks']} blocks of {ca['block_size']}) and {cb['kv_cache_size_tokens']:,} ({cb['num_gpu_blocks']} blocks of {cb['block_size']}): "
              f"{cb['kv_cache_size_tokens'] / ca['kv_cache_size_tokens']:.3f}x; concurrency at {A['args']['max_model_len']:,} tokens {ca['max_concurrency']:.2f} and {cb['max_concurrency']:.2f}")
    print("ALL BIT FOR BIT WHERE THEY MUST BE" if ok else "NOT BIT FOR BIT")
    return 0 if ok else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1], a[2:])
    elif a and a[0] == "compare":
        sys.exit(compare(a[1], a[2]))
    else:
        sys.exit(__doc__)
