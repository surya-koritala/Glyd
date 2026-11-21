"""Glyd's vLLM plugin (glyd_gpu.vllm_plugin, --quantization glyd) against
vLLM's own bf16, on the models given (default Qwen/Qwen3.5-0.8B), each vLLM in
a process of its own:

- bf16 with CUDA graphs (vLLM's default), eager, and its prompts one at a
  time: the noise floor, how far vLLM's own bf16 moves with its schedule;
- glyd fused, tiered and 12-bit, CUDA graphs: 8 prompts' greedy tokens and
  their logprobs against bf16's; bf16's own continuation (1,536 tokens) fed
  back (prompt_logprobs): top-1 agreement with bf16 at least 97%, and the
  mean |logprob difference| of its tokens within twice bf16's noise floor;
- glyd exact, eager: tokens, logprobs and prompt_logprobs bf16 eager's bit
  for bit; exact under torch.compile refused, with why, but with inductor's
  deterministic mode (DETERMINISTIC), where compiled bf16's bit for bit
  (the same mode, CUDA graphs on; refused there too where the packed
  Linears have biases); fused Glyd compiled in that mode the same
  bits from one run to the next (the second loading the first's graphs);
- per layer (LLM.apply_model): every packed layer's product against
  F.linear on its decoded matrix at 1-4096 tokens, within 1e-2 and the same
  bits every run; with GLYD_VERIFY=1 every pack checked against its
  weights, bit for bit, as it was made;
- a glyd save (glyd_gpu.save_pretrained from glyd_gpu.from_pretrained, both
  layouts; --saves): loaded as saved, its tokens and logprobs those of the
  bf16 checkpoint packed at load in that layout, bit for bit; in the other
  layout (decoded and packed again) likewise (eager, which gives the same
  bits in every process: compiled, Glyd did not, even on one compile
  cache);
- the compile cache: the layouts and modes in turn on one VLLM_CACHE_ROOT,
  each in a cache of its own (vLLM's additional_config: three caches for
  bf16 and the two layouts); each layout again on it, its own graphs loaded
  and its logits against bf16's as before; bf16 compiled again on an empty
  cache, against its first compile (the compiled graph's own variation).

    python check_vllm.py [--saves] [--quick | --brief] [--fraction F] [--tp N] [--mp] [--out DIR] [MODEL ...]
    python check_vllm.py --towers [--tp N] [--out DIR] MODEL ...                (a vision model's towers packed: below)
    python check_vllm.py --kv [--kernels-only | --quick | --sanitize] [--study DIR] [--out DIR] [MODEL]        (the lossless KV cache: see kv_main below)

--quick leaves out the two runs that only describe bf16's own noise (its
prompts one at a time, and compiled again on an empty cache): every check
stays. --brief, for a model too big to load a dozen times in a job: bf16
with CUDA graphs and eager, Glyd in the layout the GPU's best (auto) with
its layers checked, and exact eager; its checks alone. --tp N: every run
over N GPUs (tensor parallel), bf16's too. --mp: vLLM's workers in
processes of their own (its multiprocessing executor), as over several
GPUs, on one. --fraction F: Glyd's runs at that fraction of the layers
(GLYD_FRACTION), every check as before on the packed ones, and one more
each: which layers are packed is the rule's (floor(L F) of the L, layer i
where floor((i + 1) F) > floor(i F), each with all its Linears, the rest
vLLM's own bf16 method), and fraction 0 in eager is bf16 eager's tokens,
logprobs and prompt_logprobs bit for bit with nothing packed.

A mixture of experts' model: its experts' products checked beside the
Linears' (against its experts decoded, in float32, with the layer's own
activation: SiLU, GELU (tanh), a clamped SwiGLU, ReLU squared without a gate
...); exact mode's experts the routed ones decoded, then bf16's own MoE
kernel, bit for bit as the rest.

Every Glyd run reports its coverage (_coverage): the bytes its packs take and
the bf16 they hold, what stays bf16 (the largest matrices named), and the
saving of the weights as held, against the projection the inventory makes
(research/inventory-2026). The tables and Linears vLLM builds without the
quant config (packed once the model has loaded) are in the same checks: every
pack verified, every product against its decoded matrix, the embedding's
lookups the checkpoint's rows bit for bit.

--towers: a vision model's towers, which Glyd leaves bf16 by default (an
image's prefill through packed tower Linears is measured first): bf16, Glyd
as it ships, and Glyd with the towers packed (GLYD_PACK_TOWERS=1, a
development build's), each on one image (a seeded random 896 x 896): the
weights' GiB, the image's first token (the median of five, ms), 64 greedy
tokens against bf16's, every pack verified and every product checked.

CHECK_VLLM_LLM_KW: a JSON object of LLM's arguments added to every run's (a
multimodal checkpoint's slice without its vision tower: {"language_model_only":
true}).

Needs vLLM (glyd[vllm]) and the library (GLYD_GPU_LIB); --saves makes its
saves with glyd[gpu]'s transformers path (GLYD_SAVE_PYTHON: a Python that
has it, where vLLM's venv lacks accelerate). Its results, a JSON file and a
log a run, in --out (default ./check_vllm_results)."""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

TOKENS = 64
LONG = 1536  # bf16's continuation fed back (prompt_logprobs)
PROMPTS = [
    "The history of data compression began",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Q: A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive?\nA:",
    "The capital of Australia is",
    "Translate to French: The weather is lovely today, so we will walk to the market.",
    "In quantum mechanics, the uncertainty principle states that",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON",
    "Once upon a time, in a village at the edge of a great forest,",
]
MS = (1, 7, 16, 17, 33, 64, 65, 128, 129, 512, 513, 1024, 4096)
AGREE, FLOOR = 0.97, 2.0  # fused: top-1 agreement at least; mean |logprob difference| at most this times bf16's floor
DETERMINISTIC = {"inductor_compile_config": {"deterministic": True, "combo_kernels": True, "benchmark_combo_kernel": False}}  # vLLM's compilation_config


def child(spec):
    """One vLLM instance (spec: model, quantization, eager, compilation_config, one_by_one, long ids, layers): its
    tokens, logprobs and prompt_logprobs, the KV cache's size, and (glyd) its layers' products against F.linear."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if spec.get("kv_real"):
        return kv_real(spec)
    if spec.get("kv_agree"):
        return kv_agree_child(spec)
    if spec.get("kv_verify"):
        return kv_verify_child(spec)
    if spec.get("kv_heavy"):
        return kv_heavy_child(spec)
    if spec.get("kv_ext"):
        return kv_ext_child(spec)
    if spec.get("image"):
        return image_child(spec)
    out = {"spec": spec}
    if spec.get("layers") or spec.get("tables"):  # (the checkpoint's own tables, which _tables reads: from the Hub's cache, or the directory)
        from huggingface_hub import snapshot_download

        os.environ["CHECK_VLLM_CKPT"] = spec["model"] if os.path.isdir(spec["model"]) else snapshot_download(spec["model"], allow_patterns=["*.safetensors", "*.json"])
    # (kv lossless: bf16 weights, the KV cache compressed, prefix caching and chunked prefill off as it needs them; "plain": those two off alone, for stock to be compared with it)
    kvkw = dict(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless", **({"exact": True} if spec.get("exact") else {})}}) if spec.get("kv") else {}
    plain = dict(enable_prefix_caching=False, enable_chunked_prefill=False) if spec.get("kv") or spec.get("plain") else {}
    # ("r1": the flags the design's capacity figure was logged at: 8,704 tokens a request and a step, 0.9, prefix caching off; only the KV cache's size is taken)
    kw = dict(model=spec["model"], dtype="bfloat16", enforce_eager=spec.get("eager", False), seed=0, tensor_parallel_size=spec.get("tp", 1), **({"quantization": spec.get("quantization")} if not kvkw else kvkw), **plain)
    kw.update(dict(max_model_len=8704, gpu_memory_utilization=0.9, max_num_batched_tokens=8704, enable_prefix_caching=False) if spec.get("r1") else dict(max_model_len=4096, gpu_memory_utilization=0.85))
    if spec.get("max_num_seqs"):
        kw["max_num_seqs"] = spec["max_num_seqs"]
    kw.update(json.loads(os.environ.get("CHECK_VLLM_LLM_KW") or "{}"))  # (more of LLM's arguments for every run: a slice's {"language_model_only": true})
    if spec.get("compilation_config"):
        kw["compilation_config"] = spec["compilation_config"]
    if spec.get("mp"):
        kw["distributed_executor_backend"] = "mp"
    try:
        llm = LLM(**kw)
    except Exception as e:  # (exact under compile: refused)
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    cfg = llm.llm_engine.vllm_config
    out["blocks"], out["block_size"] = cfg.cache_config.num_gpu_blocks, cfg.cache_config.block_size
    out["cudagraph_mode"] = str(cfg.compilation_config.cudagraph_mode)  # (as vLLM resolved it against the attention backends' support)
    try:  # (the engine in this process: its capture's count)
        from vllm.compilation.counter import compilation_counter

        out["graphs_captured"] = compilation_counter.num_cudagraph_captured
    except Exception:
        pass
    out["num_layers"] = getattr(cfg.model_config.hf_text_config, "num_hidden_layers", None)
    out["glyd"] = cfg.additional_config.get("glyd") if isinstance(cfg.additional_config, dict) else None
    if out["glyd"] is None and spec.get("quantization") and (spec.get("tp", 1) > 1 or spec.get("mp")):  # (the workers' config holds it)
        out["glyd"] = llm.collective_rpc(_glyd_key)[0]
    if spec.get("r1"):
        return out
    greedy = SamplingParams(temperature=0, max_tokens=TOKENS, logprobs=0)
    batches = [[p] for p in PROMPTS] if spec.get("one_by_one") else [PROMPTS]
    outs = [o for b in batches for o in llm.generate(b, greedy, use_tqdm=False)]
    out["tokens"] = [list(o.outputs[0].token_ids) for o in outs]
    out["logprobs"] = [[d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)] for o in outs]
    if spec.get("make_long"):  # bf16: its continuation of the first prompt, fed back by every run
        o = llm.generate([PROMPTS[0]], SamplingParams(temperature=0, max_tokens=LONG, ignore_eos=True), use_tqdm=False)[0]
        out["long_ids"] = spec["long_ids"] = list(o.prompt_token_ids) + list(o.outputs[0].token_ids)
    if spec.get("long_ids"):
        ids = spec["long_ids"]
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)[0]
        pl = o.prompt_logprobs[1:]
        out["long_logprobs"] = [d[t].logprob for d, t in zip(pl, ids[1:])]
        out["long_top1"] = [d[t].rank == 1 for d, t in zip(pl, ids[1:])]
    if spec.get("layers"):  # (a rank's each, added up)
        out["coverage"] = llm.apply_model(_coverage)[0]  # (the first rank's)
        ranks = llm.apply_model(_layers)
        out["layers"] = {k: (max if k == "worst_rel_error" else all if k == "same_bits" else sum)(r[k] for r in ranks) if k in ("packed", "verified", "moe", "moe_verified", "biased", "worst_rel_error", "same_bits") else ranks[0][k] for k in ranks[0]}
    if spec.get("layers") or spec.get("tables"):  # (the embedding and the LM head: a rank's, the first's reported, every rank's checked)
        ranks = llm.apply_model(_tables)
        out["tables"] = dict(ranks[0], ok=all(r["ok"] for r in ranks))
    return out


def image_child(spec):
    """One vLLM instance on one image (towers_main): its weights, the image's first token (the median of five, ms, after one), 64 greedy tokens
    and their logprobs, and (layers) every packed layer's product and the coverage."""
    import time

    import numpy as np
    from PIL import Image
    from vllm import LLM, SamplingParams

    out = {"spec": spec}
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (896, 896, 3), dtype=np.uint8))
    msgs = [{"role": "user", "content": [{"type": "image_pil", "image_pil": img}, {"type": "text", "text": "Describe this image in detail."}]}]
    kw = dict(model=spec["model"], dtype="bfloat16", seed=0, tensor_parallel_size=spec.get("tp", 1), max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, mm_processor_cache_gb=0, limit_mm_per_prompt={"image": 1}, **({"quantization": spec["quantization"]} if spec.get("quantization") else {}))
    try:
        llm = LLM(**kw)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    first = SamplingParams(temperature=0, max_tokens=1)
    llm.chat(msgs, first, use_tqdm=False)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        llm.chat(msgs, first, use_tqdm=False)
        times.append((time.perf_counter() - t) * 1e3)
    out["first_token_ms"] = sorted(times)[2]
    o = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=TOKENS, logprobs=0), use_tqdm=False)[0].outputs[0]
    out["tokens"], out["logprobs"] = [list(o.token_ids)], [[d[t].logprob for d, t in zip(o.logprobs, o.token_ids)]]
    if spec.get("layers"):
        out["coverage"] = llm.apply_model(_coverage)[0]
        ranks = llm.apply_model(_layers)
        out["layers"] = {k: (max if k == "worst_rel_error" else all if k == "same_bits" else sum)(r[k] for r in ranks) if k in ("packed", "verified", "moe", "moe_verified", "biased", "worst_rel_error", "same_bits") else ranks[0][k] for k in ranks[0]}
    return out


def towers_main():
    """--towers MODEL ...: bf16, Glyd as it ships and Glyd with the towers packed (GLYD_PACK_TOWERS=1), one image each (image_child); the
    checks: every pack verified and every product within 1e-2 of its decoded matrix, the same bits twice, with the towers packed; the rest
    reported (the weights, the image's first token, the greedy tokens against bf16's), for the default's choice."""
    args = sys.argv[1:]
    out_dir = args[args.index("--out") + 1] if "--out" in args else "check_vllm_results"
    tp = int(args[args.index("--tp") + 1]) if "--tp" in args else 1
    models = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--out", "--tp"))]
    if not models:
        raise SystemExit("check_vllm --towers: name a vision model (a 2026 one: ibm-granite/granite-vision-4.1-4b, google/gemma-4-31B-it, meta-models/Muse-Glimmer-30B ...)")
    os.makedirs(out_dir, exist_ok=True)
    report, failed = [], []

    def check(ok, what):
        report.append(("PASS " if ok else "FAIL ") + what)
        print(report[-1], flush=True)
        if not ok:
            failed.append(what)

    for model in models:
        tag = model.split("/")[-1] + (f"-tp{tp}" if tp > 1 else "")
        runs = {}
        for name, spec, env in (("bf16", {}, {}), ("glyd", {"quantization": "glyd", "layers": True}, {"GLYD_VERIFY": "1"}),
                                ("glyd-towers", {"quantization": "glyd", "layers": True}, {"GLYD_VERIFY": "1", "GLYD_PACK_TOWERS": "1"})):
            runs[name] = run(dict(spec, model=model, image=True, tp=tp), out_dir, f"{tag}-image-{name}", env)
        bf = runs["bf16"]
        for name in ("glyd", "glyd-towers"):
            r = runs[name]
            if "error" in r:
                check(False, f"{tag} {name}: {r['error'][:300]}")
                continue
            c = compare(r, bf) if "error" not in bf else {}
            report.append(f"TOWERS {tag} {name}: weights {r.get('weights_gib')} GiB (bf16 {bf.get('weights_gib')}); the image's first token {r['first_token_ms']:.1f} ms (bf16 {bf.get('first_token_ms', float('nan')):.1f}); "
                          f"greedy tokens against bf16's {c}; coverage {r.get('coverage')}")
            print(report[-1], flush=True)
            lay = r["layers"]
            check(lay["verified"] == lay["packed"] > 0 and lay["moe_verified"] == lay["moe"], f"{tag} {name}: every pack ({lay['packed']}) decoded to its weights bit for bit")
            check(lay["worst_rel_error"] < 1e-2 and lay["same_bits"], f"{tag} {name}: each layer's product within 1e-2 of F.linear ({lay['worst_rel_error']:.2e}) and the same bits every run")
        if all("error" not in runs[n] for n in ("glyd", "glyd-towers")):
            more = runs["glyd-towers"]["layers"]["packed"] - runs["glyd"]["layers"]["packed"]
            check(more > 0, f"{tag}: GLYD_PACK_TOWERS=1 packs the towers' Linears ({more} more packs; a release build reads no such variable)")
    with open(os.path.join(out_dir, "report-towers.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"check_vllm --towers: {'all passed' if not failed else f'{len(failed)} failed'} ({sum(line.startswith(('PASS', 'FAIL')) for line in report)} checks)")
    sys.exit(1 if failed else 0)


def _glyd_key(worker):
    """A worker's Glyd key (its vLLM config's additional_config["glyd"]): over several processes the engine's config
    does not hold it."""
    return worker.vllm_config.additional_config.get("glyd")


def _layers(model):
    """Every packed layer's product (glyd::vllm_linear, fused) against F.linear on its decoded matrix, at MS tokens,
    and every mixture of experts' layer's (its experts packed: the library's grouped products) against its experts
    decoded, in float32 (_moe): the largest relative error, and whether a second call gave the same bits; the layers
    packed, and verified."""
    import torch
    import torch.nn.functional as F
    from glyd_gpu import kernels as g
    from glyd_gpu.vllm_plugin import _layer_of
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    worst, same, n, verified, moe, moe_verified, biased = 0.0, True, 0, 0, 0, 0, 0
    torch.manual_seed(0)
    packed_layers, plain_layers = set(), set()  # the decoder layers with a packed Linear or experts, and with vLLM's own method
    for name, m in model.named_modules():
        i = _layer_of(name)
        # (only layers a quantization config was given to: a model builds some without one, a mixture of experts' router gate
        # for one, and they run vLLM's own method whatever the fraction)
        if i is not None and getattr(m, "quant_method", None) is not None and getattr(m, "quant_config", None) is not None:
            (packed_layers if getattr(m, "glyd_words", None) is not None or getattr(m, "glyd_moe", None) is not None else plain_layers).add(i)
    for m in model.modules():
        if isinstance(m, VocabParallelEmbedding):  # (the embedding and the LM head: _tables')
            continue
        if getattr(m, "glyd_moe", None) is not None:
            moe += 1
            moe_verified += bool(m.glyd_verified)
            e, b = _moe(m, (1, 7, 64, 300) if moe <= 2 else (7,))
            worst, same = max(worst, e), same and b
            continue
        if getattr(m, "glyd_words", None) is None:
            continue
        n += 1
        verified += bool(m.glyd_verified)
        biased += m.bias is not None
        O, K = m.glyd_out, m.weight.shape[1]
        rows = -(-O // 64) * 64  # (a Linear's rows not a multiple of 64: its pack's padded with zeros)
        p = g.Mma12((rows, K), m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words[0] & 0xFF) if len(m.glyd_words) == 4 else g.Mma((rows, K), m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words)
        w = g.mma_unpack(p)[:O].float()
        for M in MS if n <= 4 else (1, 64, 513):  # (every M on the first layers, three on the rest)
            x = torch.randn(M, K, dtype=torch.bfloat16, device=w.device)
            y = torch.ops.glyd.vllm_linear(x, m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words, m.bias, O, False)
            ref = F.linear(x.float(), w, None if m.bias is None else m.bias.float())
            worst = max(worst, ((y.float() - ref).abs().max() / ref.abs().max()).item())
            same &= torch.equal(y, torch.ops.glyd.vllm_linear(x, m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words, m.bias, O, False))
    return {"packed": n, "verified": verified, "moe": moe, "moe_verified": moe_verified, "biased": biased, "worst_rel_error": worst, "same_bits": same, "packed_layers": sorted(packed_layers), "plain_layers": sorted(plain_layers), **_host(model)}


def _tables(model):
    """Every embedding and LM head of the model (a rank's): its method, whether the plugin's rule packs it (eligible: a plain table of 4,096
    rows or more, its hidden size a multiple of 64), and where it is packed its pack's buffer (a tied head's is the embedding's) and bytes, its
    lookups of random ids and of the shard's ends against the checkpoint's own rows bit for bit, its product (the method's apply, for 1 to 300
    rows) against the checkpoint's matrix in float32 (a chunk of rows at a time) within 1e-2 and the same bits twice (a table whose rows the
    checkpoint holds under no name of an embedding or LM head, as LongCat's n-gram table's twelve: not compared, and said so); and how many
    packs the model should hold (one where the config ties its two tables, or the checkpoint's two are the same; else one a table)."""
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    ck, rows = os.environ["CHECK_VLLM_CKPT"], {}
    for f in sorted(glob.glob(os.path.join(ck, "*.safetensors"))):
        with safe_open(f, "pt", device="cpu") as sf:
            for k in sf.keys():
                part = k.rsplit(".", 2)
                if k.endswith(".weight") and len(part) > 1 and part[-2] in ("embed_tokens", "embeddings", "word_embeddings", "wte", "lm_head") and not re.search(r"vision|visual|audio|image|patch|position|mtp", k):
                    rows.setdefault("lm_head" if part[-2] == "lm_head" else "embed_tokens", sf.get_tensor(k))
    torch.manual_seed(0)
    out, ok = [], True
    for name, m in model.named_modules():
        if not isinstance(m, VocabParallelEmbedding):
            continue
        head = isinstance(m, ParallelLMHead)
        eligible = type(m) in (VocabParallelEmbedding, ParallelLMHead) and m.embedding_dim % 64 == 0 and m.num_embeddings >= 4096
        r = {"name": name, "method": type(m.quant_method).__name__, "head": head, "packed": getattr(m, "glyd_words", None) is not None, "eligible": eligible, "shard_rows": m.num_embeddings_per_partition}
        tab = rows.get("lm_head" if head and "lm_head" in rows else "embed_tokens")
        if r["packed"] and (tab is None or tab.shape[1] != m.embedding_dim or tab.shape[0] < m.shard_indices.org_vocab_end_index):
            r["compared"] = False
        elif r["packed"]:
            lo = m.shard_indices.org_vocab_start_index
            n = m.shard_indices.org_vocab_end_index - lo
            ids = torch.cat([torch.randint(0, n, (700,)), torch.tensor([0, n - 1])])
            y = m.quant_method.embedding(m, ids.to(m.glyd_data.device))
            same = torch.equal(y.cpu().view(torch.int16), tab[lo + ids].view(torch.int16))
            worst, twice = 0.0, True
            for M in (1, 7, 64, 65, 300):
                x = torch.randn(M, m.embedding_dim, dtype=torch.bfloat16, device=m.glyd_data.device)
                lg = m.quant_method.apply(m, x)
                ref = torch.cat([F.linear(x.float(), tab[lo + i : lo + min(i + 8192, n)].to(x.device).float()) for i in range(0, n, 8192)], dim=1)
                worst = max(worst, ((lg[:, :n].float() - ref).abs().max() / ref.abs().max()).item())
                twice &= tuple(lg.shape) == (M, m.num_embeddings_per_partition) and torch.equal(lg, m.quant_method.apply(m, x))
            r.update(compared=True, lookups_exact=same, worst_rel_error=worst, same_bits=twice)
            ok &= same and worst < 1e-2 and twice
        if r["packed"]:
            r.update(buffer=m.glyd_data.data_ptr(), bytes=sum(t.numel() * t.element_size() for t in (m.glyd_data, m.glyd_a, m.glyd_b)), pack_rows=m.glyd_out)
        out.append(r)
    mc = getattr(getattr(model, "vllm_config", None), "model_config", None)
    flagged = bool(getattr(mc, "word_embeddings_untied_by_checkpoint", False))
    tc = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    ties = bool(getattr(tc, "tie_word_embeddings", False) or getattr(model.config, "tie_word_embeddings", False))
    one = (ties and not flagged) or (flagged and "lm_head" in rows and "embed_tokens" in rows and torch.equal(rows["embed_tokens"], rows["lm_head"]))
    eligible = [t for t in out if t["eligible"]]
    return {"tables": out, "expect_packs": 1 if one else len(eligible), "config_ties": ties or flagged, "ok": ok}  # (vLLM sets tie_word_embeddings false on a model it builds untied for the checkpoint's lm_head: flagged)


def _moe(m, ts):
    """A mixture of experts' layer's product (its method's apply: the library's grouped products) against its experts
    decoded, in float32 (the gate and up sums rounded to bf16, the layer's activation on them by vLLM's own function as
    its experts apply it, then down, each token's k rows times their weights added), for T random tokens each of ts
    routed to k random experts: the largest relative error, and whether a second call gave the same bits."""
    import torch
    from glyd_gpu import kernels as g
    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, apply_moe_activation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    up, down, E = m.glyd_moe
    W1, W2 = g.mma_unpack(up).view(E, -1, up.shape[1]), g.mma_unpack(down).view(E, -1, down.shape[1])  # (bf16: an expert's to float32 as it is used)
    k = m.top_k
    worst, same = 0.0, True
    for T in ts:
        x = torch.randn(T, W1.shape[2], dtype=torch.bfloat16, device=W1.device)
        ids = torch.stack([torch.randperm(E, device=W1.device)[:k] for _ in range(T)]).to(torch.int32)
        w = torch.softmax(torch.randn(T, k, device=W1.device), dim=-1)
        y = m.quant_method.apply(m, x, w, ids, None, None)
        gu = torch.stack([W1[ids[t].long()].float() @ x[t].float() for t in range(T)])  # [T, k, 2I]
        a = gu.to(torch.bfloat16).view(T * k, -1)
        h = torch.empty(T * k, a.shape[1] // 2 if m.activation.is_gated else a.shape[1], dtype=torch.bfloat16, device=a.device)
        apply_moe_activation(m.activation, h, a, activation_config=getattr(m.quant_method, "act_config", None) or ApplyMoEActivationConfig.from_configs(m.moe_config, FusedMoEQuantConfig.make()))
        h = h.float().view(T, k, -1)
        ref = torch.stack([(w[t, :, None] * torch.stack([W2[ids[t, j].long()].float() @ h[t, j] for j in range(k)])).sum(0) for t in range(T)])
        worst = max(worst, ((y.float() - ref).abs().max() / ref.abs().max()).item())
        same &= torch.equal(y, m.quant_method.apply(m, x, w, ids, None, None))
    return worst, same


def _coverage(model):
    """The weights as a rank holds them: the bytes of Glyd's packs (each once) and of the bf16 they hold, the bytes of every other
    parameter (bf16 or not), the saving of the whole against bf16, and the largest matrices left bf16 (1M weights or more, named)."""
    packs, held, seen = 0, 0, set()
    for _, m in model.named_modules():
        bufs = [b for n, b in m.named_buffers(recurse=False) if n.startswith("glyd_")]
        key = bufs[0].data_ptr() if bufs else None
        if key is None or key in seen:
            continue
        seen.add(key)
        packs += sum(b.numel() * b.element_size() for b in bufs)
        if getattr(m, "glyd_moe", None) is not None:
            held += 2 * sum(p.shape[0] * p.shape[1] for p in m.glyd_moe[:2])
        elif hasattr(m, "embedding_dim") and getattr(m, "glyd_words", None) is not None:
            held += 2 * getattr(m, "num_embeddings_per_partition", m.glyd_out) * m.embedding_dim
        elif getattr(m, "glyd_words", None) is not None:
            held += 2 * m.glyd_out * m.weight.shape[1]
    plain, big, pseen = 0, [], set()
    for name, t in model.named_parameters(remove_duplicate=False):
        if t.data_ptr() in pseen or not t.numel():
            continue
        pseen.add(t.data_ptr())
        plain += t.numel() * t.element_size()
        if t.dim() == 2 and t.numel() >= 1 << 20:
            big.append((name, list(t.shape), str(t.dtype).replace("torch.", ""), round(t.numel() * t.element_size() / 2**20, 1)))
    big.sort(key=lambda r: -r[3])
    return {"packs_gib": packs / 2**30, "packed_bf16_gib": held / 2**30, "plain_gib": plain / 2**30, "saving": 1 - (packs + plain) / max(1, held + plain),
            "plain_matrices": len(big), "plain_matrices_gib": sum(r[3] for r in big) / 1024, "largest_plain": big[:12]}


def _host(model):
    """The host's time a product, microseconds (the calls queued, not waited for): the op (fused, a prompt's 1024 tokens
    and a step's 8) against F.linear on the same shapes, the first packed layer's."""
    import time
    import torch
    import torch.nn.functional as F

    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    m = next((m for m in model.modules() if not isinstance(m, VocabParallelEmbedding) and getattr(m, "glyd_words", None) is not None), None)
    if m is None:  # (fraction 0: nothing packed)
        return {}
    O, K = m.glyd_out, m.weight.shape[1]
    w = torch.randn(O, K, dtype=torch.bfloat16, device=m.glyd_data.device)
    out = {}
    for M in (8, 1024):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=w.device)
        for name, f in (("op", lambda: torch.ops.glyd.vllm_linear(x, m.glyd_data, m.glyd_a, m.glyd_b, m.glyd_words, m.bias, O, False)), ("linear", lambda: F.linear(x, w, m.bias))):
            for _ in range(5):
                f()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(50):
                f()
            out[f"host_us_{name}_{M}"] = round((time.perf_counter() - t) / 50 * 1e6, 1)
            torch.cuda.synchronize()
    return out


def run(spec, out_dir, name, env=None):
    """spec in a vLLM of its own (this script, --child), its JSON in out_dir/name.json."""
    path = os.path.join(out_dir, name + ".json")
    if os.environ.get("KV_REUSE") and os.path.exists(path):  # (a check that was cut short, taken up again: the runs that finished are not made again)
        with open(path) as f:
            return json.load(f)
    e = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", **(env or {}))
    e.setdefault("GLYD_KV", "off")  # (kv's default is auto: the runs that mean the KV cache say kv lossless in their config, the others are vLLM's own cache)
    if spec.get("tp", 1) > 1 or spec.get("mp"):  # (LLM.apply_model sends its function to the workers' processes)
        e["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    arg = json.dumps(spec)
    if len(arg) > 50000:  # (a long spec, prompts' token ids: in a file, not the command line)
        with open(os.path.join(out_dir, name + ".spec"), "w") as f:
            f.write(arg)
        arg = "@" + os.path.join(out_dir, name + ".spec")
    cmd = [sys.executable, os.path.abspath(__file__), "--child", arg, path]
    if os.environ.get("KV_WRAP"):  # a command put before each run (the box's GPU lock: "bash run.sh"), so that another job's turn comes between two runs and not after all of them
        cmd = os.environ["KV_WRAP"].split() + cmd
    with open(os.path.join(out_dir, name + ".log"), "w") as log:
        r = subprocess.run(cmd, env=e, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    if r.returncode or not os.path.exists(path):
        raise RuntimeError(f"{name}: exit {r.returncode}; see {name}.log")
    with open(path) as f:
        res = json.load(f)
    with open(os.path.join(out_dir, name + ".log")) as f:  # (vLLM's own figure for the weights: the memory in use once they are loaded)
        m = re.search(r"Model loading took ([\d.]+) GiB", f.read())
    if m:
        res["weights_gib"] = float(m.group(1))
    return res


def same_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def compare(a, b):
    """Tokens the same from the start (each prompt), prompts bit for bit (tokens and logprobs), the mean |logprob
    difference| over the shared prefixes; on the fed-back continuation: bit for bit, top-1 agreement, mean |diff|."""
    pre = [same_prefix(x, y) for x, y in zip(a["tokens"], b["tokens"])]
    bits = sum(x == y and lx == ly for x, y, lx, ly in zip(a["tokens"], b["tokens"], a["logprobs"], b["logprobs"]))
    d = [abs(u - v) for x, y, lx, ly in zip(a["tokens"], b["tokens"], a["logprobs"], b["logprobs"]) for u, v in list(zip(lx, ly))[: same_prefix(x, y)]]
    r = {"prefix": pre, "bit_identical": bits, "mean_abs_dlogprob": sum(d) / max(1, len(d))}
    if "long_logprobs" in a and "long_logprobs" in b:
        la, lb = a["long_logprobs"], b["long_logprobs"]
        r["long_bit_identical"] = la == lb
        r["long_mean_abs_dlogprob"] = sum(abs(u - v) for u, v in zip(la, lb)) / len(la)
        r["long_top1"] = sum(a["long_top1"]) / len(a["long_top1"])
    return r


def main():
    args = sys.argv[1:]
    saves, brief = "--saves" in args, "--brief" in args
    quick = brief or "--quick" in args
    out_dir = args[args.index("--out") + 1] if "--out" in args else "check_vllm_results"
    tp = int(args[args.index("--tp") + 1]) if "--tp" in args else 1
    mp = "--mp" in args  # (vLLM's workers in processes of their own, as over several GPUs, on one)
    fraction = args[args.index("--fraction") + 1] if "--fraction" in args else None  # (Glyd's runs at this share of the layers)
    if fraction is not None:
        os.environ["GLYD_FRACTION"] = fraction
    models = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--out", "--tp", "--fraction"))] or ["Qwen/Qwen3.5-0.8B"]
    os.makedirs(out_dir, exist_ok=True)
    report, failed = [], []

    def check(ok, what):
        report.append(("PASS " if ok else "FAIL ") + what)
        print(report[-1], flush=True)
        if not ok:
            failed.append(what)

    for model in models:
        tag = model.split("/")[-1] + (f"-tp{tp}" if tp > 1 else "")
        cache = tempfile.mkdtemp(prefix="vllm-cache-")  # one compile cache for every run of the model (its layouts and modes in turn)
        env = {"VLLM_CACHE_ROOT": cache}
        R = lambda name, spec, extra=None: run(dict(spec, model=spec.get("model", model), tp=tp, mp=mp), out_dir, f"{tag}-{name}", dict(env, **(extra or {})))
        run1 = lambda spec, name, extra: run(dict(spec, tp=tp, mp=mp), out_dir, f"{tag}-{name}", extra)
        bf16 = R("bf16", {"make_long": True})
        long = {"long_ids": bf16["long_ids"]}
        eager = R("bf16-eager", dict(long, eager=True))
        floor = compare(eager, bf16)
        noise = ""
        if not quick:
            one = R("bf16-one-by-one", {"one_by_one": True})
            again = run1(dict(long, model=model), "bf16-recompiled", {"VLLM_CACHE_ROOT": tempfile.mkdtemp(prefix="vllm-cache-")})
            noise = f"; one by one against batched {compare(one, bf16)}; compiled again on an empty cache {compare(again, bf16)}"
        print(f"{tag}: bf16 KV cache {bf16['blocks'] * bf16['block_size']} tokens; its noise: graphs against eager {floor}{noise}", flush=True)
        for layout in ("auto",) if brief else ("mma", "mma12"):
            r = R(f"glyd-{layout}", dict(long, quantization="glyd", layers=True), {"GLYD_LAYOUT": layout, "GLYD_VERIFY": "1"})
            c = compare(r, bf16)
            print(f"{tag} glyd {layout}: weights {r.get('weights_gib')} GiB (bf16 {bf16.get('weights_gib')}), KV cache {r['blocks'] * r['block_size']} tokens (bf16 {bf16['blocks'] * bf16['block_size']}); against bf16 {c}; layers {r['layers']}; tables {r.get('tables')}; additional_config {r['glyd']}", flush=True)
            cov = r.get("coverage") or {}
            report.append(f"COVERAGE {tag} {layout}: weights {r.get('weights_gib')} GiB against bf16's {bf16.get('weights_gib')}; packs {cov.get('packs_gib', 0):.2f} GiB holding {cov.get('packed_bf16_gib', 0):.2f} GiB of bf16, {cov.get('plain_gib', 0):.2f} GiB left as they are "
                          f"({cov.get('plain_matrices', 0)} matrices of 1M weights or more, {cov.get('plain_matrices_gib', 0):.2f} GiB: {cov.get('largest_plain')}); the weights {100 * cov.get('saving', 0):.2f}% smaller than bf16 (a rank's)")
            print(report[-1], flush=True)
            tb = r["tables"]
            packs = {t["buffer"] for t in tb["tables"] if t["packed"]}
            if fraction is None or Fraction(fraction) >= 1:  # (a fraction under 1 leaves the embedding and the LM head as vLLM runs them)
                worst = max([t["worst_rel_error"] for t in tb["tables"] if t.get("compared")] or [0.0])
                unseen = [t["name"] for t in tb["tables"] if t["packed"] and not t.get("compared")]
                check(tb["ok"] and all(t["packed"] for t in tb["tables"] if t["eligible"]) and len(packs) == tb["expect_packs"], f"{tag} {layout}: the embedding and the LM head packed, those built without the quant config too ({len(packs)} pack{'s' if len(packs) != 1 else ''}, {sum({t['buffer']: t['bytes'] for t in tb['tables'] if t['packed']}.values()) / 2**20:.0f} MiB; {tb['expect_packs']} expected), every lookup of random ids and of the shard's ends the checkpoint's own row bit for bit, each product within 1e-2 of its matrix ({worst:.2e}) and the same bits twice{f'; not compared (no checkpoint tensor of its name): {unseen}' if unseen else ''}")
            else:
                check(not packs and tb["ok"], f"{tag} {layout}, fraction {fraction}: the embedding and the LM head as vLLM runs them (none packed)")
            moe = r["layers"]["moe"]
            experts = f" and {moe} mixture of experts' layers' experts" if moe else ""
            check(r["layers"]["verified"] == r["layers"]["packed"] > 0 and r["layers"]["moe_verified"] == moe, f"{tag} {layout}: every pack ({r['layers']['packed']}{experts}) decoded to its weights bit for bit")
            check(r["layers"]["worst_rel_error"] < 1e-2 and r["layers"]["same_bits"], f"{tag} {layout}: each layer's product within 1e-2 of F.linear{', the experts of their decoded matrices' if moe else ''} ({r['layers']['worst_rel_error']:.2e}) and the same bits every run")
            check(c["long_top1"] >= AGREE, f"{tag} {layout}: top-1 agreement with bf16 on its continuation {c['long_top1']:.4f} (at least {AGREE}; bf16 eager's {floor['long_top1']:.4f})")
            check(c["long_mean_abs_dlogprob"] <= FLOOR * max(floor["long_mean_abs_dlogprob"], 1e-6), f"{tag} {layout}: mean |logprob difference| {c['long_mean_abs_dlogprob']:.2e}, at most {FLOOR}x bf16 eager's against its graphs ({floor['long_mean_abs_dlogprob']:.2e})")
            if fraction is not None:
                lay, L, q = r["layers"], r["num_layers"], Fraction(fraction)
                want = [i for i in range(L) if (i + 1) * q // 1 > i * q // 1]  # (the rule, as the plugin's docstring has it)
                ok = lay["packed_layers"] == want and len(want) == L * q // 1 and not set(lay["packed_layers"]) & set(lay["plain_layers"]) and set(lay["packed_layers"]) | set(lay["plain_layers"]) == set(range(L))
                check(ok, f"{tag} {layout}, fraction {fraction}: {len(want)} of {L} layers packed, spread evenly (layer i where floor((i + 1) f) > floor(i f)), each with all its Linears, the other {L - len(want)} vLLM's own method (packed {lay['packed_layers']})")
        for layout in () if brief else ("mma", "mma12"):  # again on the shared cache, the same options: its own graphs loaded, not another's
            again = R(f"glyd-{layout}-again", dict(long, quantization="glyd"), {"GLYD_LAYOUT": layout, "GLYD_VERIFY": "1"})
            loaded = "Directly load AOT compilation" in open(os.path.join(out_dir, f"{tag}-glyd-{layout}-again.log")).read()
            c = compare(again, bf16)
            check(loaded and c["long_top1"] >= AGREE and c["long_mean_abs_dlogprob"] <= FLOOR * max(floor["long_mean_abs_dlogprob"], 1e-6), f"{tag} {layout}: again on the shared compile cache, its graphs loaded ({loaded}), against bf16 as before (top-1 {c['long_top1']:.4f}, mean |diff| {c['long_mean_abs_dlogprob']:.2e})")
        aot = os.path.join(cache, "torch_compile_cache", "torch_aot_compile")
        dirs = sorted(os.listdir(aot)) if os.path.isdir(aot) else []
        if not brief:
            # (3 on one GPU; over several a rank's packs, and so its key, its own: a layout's ranks each a cache)
            check(len(dirs) == 3 if tp == 1 else len(dirs) >= 3, f"{tag}: a compile cache for each of bf16, glyd tiered and glyd 12-bit on the one VLLM_CACHE_ROOT ({len(dirs)}), none reused by another")
        x = R("glyd-exact-eager", dict(long, quantization="glyd", eager=True, tables=True), {"GLYD_EXACT": "1"})
        c = compare(x, eager) if "error" not in x else {"bit_identical": 0, "long_bit_identical": x["error"][:200]}
        check(c["bit_identical"] == len(PROMPTS) and c["long_bit_identical"] is True, f"{tag} exact, eager: bf16 eager's tokens, logprobs and prompt_logprobs bit for bit ({c['bit_identical']} of {len(PROMPTS)}; continuation {c['long_bit_identical']})")
        if "tables" in x and fraction is None:  # (exact keeps the LM head vLLM's own, so a tied table too: the embedding of an untied model alone is packed)
            tb, tied = x["tables"], x["tables"]["config_ties"]  # (a config that ties them: one table, or the checkpoint's own copy of it, which vLLM builds apart: exact keeps both as vLLM runs them)
            what = "the tied model's table stays as vLLM runs it" if tied else "the embedding packed, its rows the checkpoint's bit for bit, the LM head as vLLM runs it"
            check(tb["ok"] and all(t["packed"] != (t["head"] or tied) for t in tb["tables"] if t["eligible"]), f"{tag} exact: {what} ({[(t['name'], t['method']) for t in tb['tables']]})")
        if fraction is not None:  # (fraction 0: bf16, through the plugin, with nothing packed)
            z = R("glyd-fraction0-eager", dict(long, quantization="glyd", eager=True, layers=True), {"GLYD_FRACTION": "0"})
            c = compare(z, eager) if "error" not in z else {"bit_identical": 0, "long_bit_identical": z["error"][:200]}
            lay = z.get("layers", {})
            check(c["bit_identical"] == len(PROMPTS) and c["long_bit_identical"] is True and lay.get("packed") == 0 and lay.get("moe") == 0 and not lay.get("packed_layers"), f"{tag} fraction 0, eager: nothing packed, bf16 eager's tokens, logprobs and prompt_logprobs bit for bit ({c['bit_identical']} of {len(PROMPTS)}; continuation {c['long_bit_identical']}; KV cache {z.get('blocks', 0) * z.get('block_size', 0)} tokens, bf16 eager's {eager['blocks'] * eager['block_size']})")
        if brief:
            shutil.rmtree(cache, ignore_errors=True)
            continue
        xc = R("glyd-exact-compiled", dict(long, quantization="glyd"), {"GLYD_EXACT": "1"})
        check("error" in xc and "glyd: exact mode" in xc["error"] and "enforce-eager" in xc["error"], f"{tag} exact under torch.compile: refused, with Glyd's message ({xc.get('error', 'not refused')[:120]})")
        fresh = lambda: {"VLLM_CACHE_ROOT": tempfile.mkdtemp(prefix="vllm-cache-")}  # (each run a compile of its own)
        det = dict(long, model=model, compilation_config=DETERMINISTIC)
        bd = run1(det, "bf16-det", fresh())
        xd = run1(dict(det, quantization="glyd"), "glyd-exact-det", dict(fresh(), GLYD_EXACT="1"))
        if r["layers"]["biased"]:  # (packed Linears with biases: exact compiled refused, inductor adds bf16's apart)
            check("error" in xd and "bias" in xd["error"], f"{tag} exact under torch.compile, inductor deterministic: refused, its packed Linears having biases ({r['layers']['biased']}; {xd.get('error', 'not refused')[:120]})")
        else:
            c = compare(xd, bd) if "error" not in xd else {"bit_identical": 0, "long_bit_identical": xd["error"][:200]}
            check(c["bit_identical"] == len(PROMPTS) and c["long_bit_identical"] is True, f"{tag} exact under torch.compile, inductor deterministic: compiled bf16's tokens, logprobs and prompt_logprobs (the same mode) bit for bit ({c['bit_identical']} of {len(PROMPTS)}; continuation {c['long_bit_identical']})")
        gc = fresh()
        ga, gb = (run1(dict(det, quantization="glyd"), f"glyd-det-{n}", gc) for n in "ab")
        c = compare(ga, gb)
        check(c["bit_identical"] == len(PROMPTS) and c["long_bit_identical"], f"{tag} fused, compiled, inductor deterministic: the same bits from one run to the next, the second on the first's graphs ({c['bit_identical']} of {len(PROMPTS)}; continuation {c['long_bit_identical']}; layout {(ga.get('glyd') or {}).get('layout', '?')})")
        if saves:
            for layout in ("mma", "mma12"):
                d = tempfile.mkdtemp(prefix=f"glyd-save-{tag}-{layout}-")
                # (GLYD_SAVE_PYTHON: a Python with glyd[gpu]'s transformers path, where vLLM's venv lacks accelerate)
                subprocess.run([os.environ.get("GLYD_SAVE_PYTHON", sys.executable), "-c", "import sys, glyd_gpu; m = glyd_gpu.from_pretrained(sys.argv[1], layout=sys.argv[3], compile=False); glyd_gpu.save_pretrained(m, sys.argv[2], layout=sys.argv[3])", model, d, layout], check=True, timeout=1800)
                # eager: bit for bit from one process to the next (compiled, Glyd is not, even on one compile cache)
                ref = run(dict(long, model=model, quantization="glyd", eager=True), out_dir, f"{tag}-glyd-{layout}-ref", {"VLLM_CACHE_ROOT": cache, "GLYD_LAYOUT": layout})
                for as_layout in (layout, "mma12" if layout == "mma" else "mma"):
                    s = R(f"save-{layout}-as-{as_layout}", dict(long, model=d, quantization="glyd", eager=True), {"GLYD_LAYOUT": as_layout, "GLYD_VERIFY": "1"})
                    want = ref if as_layout == layout else run(dict(long, model=model, quantization="glyd", eager=True), out_dir, f"{tag}-glyd-{as_layout}-ref", {"VLLM_CACHE_ROOT": cache, "GLYD_LAYOUT": as_layout})
                    check(s["tokens"] == want["tokens"] and s["logprobs"] == want["logprobs"] and s["long_logprobs"] == want["long_logprobs"], f"{tag} save ({layout}) loaded as {as_layout}: the bf16 checkpoint packed at load in {as_layout}, bit for bit")
                shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"check_vllm: {'all passed' if not failed else f'{len(failed)} failed'} ({sum(line.startswith(('PASS', 'FAIL')) for line in report)} checks)")
    sys.exit(1 if failed else 0)


# ---- the lossless KV cache (--kv) ------------------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
DOCS = ("README.md", "gpu/README.md", "CHANGELOG.md", "docs/codec/engineering-notes.md", "gpu/vllm/README.md", "docs/codec/spec.md")  # this repository's own documents: the long prompts


def kv_pool(n_pages, H, A, seed=0):
    """A pool of n pages of a previous owner's bytes (random: no kernel may assume a zeroed page) and its status words."""
    import torch
    from glyd_gpu import _lib

    page = _lib.kv_leaf_bytes(H, 128, A)
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, 256, (n_pages * page,), dtype=torch.uint8, device="cuda", generator=g), page, torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device="cuda")


def kv_natural(T, H, seed, kind="k", outlier=0.0, erange=(1, 255)):
    """bf16 bits [T, H, 128] (int16) of a cache's kind: exponents round(gaussian) around a centre a (head, channel), `outlier` of the values anywhere in erange."""
    import torch

    dev = "cuda"
    gc = torch.Generator(device=dev).manual_seed(1000)
    c = (121.0 + torch.randn(H, 1, 128, device=dev, generator=gc) * (2.0 if kind == "k" else 0.7)).round()
    g = torch.Generator(device=dev).manual_seed(seed)
    e = (c + torch.randn(H, T, 128, device=dev, generator=g).round()).clamp(1, 254).to(torch.int32)
    if outlier:
        far = torch.rand(H, T, 128, device=dev, generator=g) < outlier
        e = torch.where(far, torch.randint(erange[0], erange[1], (H, T, 128), device=dev, generator=g, dtype=torch.int32), e)
    u = (torch.randint(0, 2, (H, T, 128), device=dev, generator=g, dtype=torch.int32) << 15) | (e << 7) | torch.randint(0, 128, (H, T, 128), device=dev, generator=g, dtype=torch.int32)
    return (u - (u >> 15 << 16)).to(torch.int16).permute(1, 0, 2).contiguous()


def kv_bases(x):
    """uint8 [H, D]: the window of 8 values of exponent >> 1 holding the most (glyd_gpu.kvcal's rule) of x [T, H, D] int16."""
    from glyd_gpu import kvcal

    return kvcal.best_base(kvcal.histogram(x))


def kv_kernel_checks(quick=False):
    """The write, unpack and decode kernels through the library, on the GPU (no vLLM): [(ok, what)]. quick: fewer and smaller cases (for the sanitizer)."""
    import math
    import torch
    from glyd_gpu import _lib, kernels as g

    g.lib()
    res = []
    bits = lambda t: t.contiguous().view(torch.int16)

    def check(ok, what):
        res.append((bool(ok), what))
        print(("PASS " if ok else "FAIL ") + what, flush=True)

    def write(store, page, st, A, key, value, cells, kb, vb, esc=None, layer=0, lost=None):
        _lib.kv_write(key, value, cells, store, page, kb, vb, A, esc, layer, st, lost)

    def read(store, page, leaves, H, A, kb, vb, esc=None):
        leaves = torch.as_tensor(leaves, dtype=torch.int32, device="cuda")
        ok = torch.empty(leaves.numel(), 16, H, 128, dtype=torch.int16, device="cuda")
        ov = torch.empty_like(ok)
        _lib.kv_unpack(store, page, leaves, kb, vb, A, esc, ok, ov)
        return ok, ov

    def dirs(store, page, H, leaves):
        """int32 [len(pages), 16]: the escape-pool chunks (number + 1, 0 none) the directories of these pages name."""
        pg = torch.as_tensor(list(leaves), dtype=torch.int64, device="cuda")
        raw = store.view(-1, page)[pg][:, 16 * H + 4 : 16 * H + 4 + 64].contiguous()
        return raw.view(torch.int32)

    def used_of(store, page, H, leaves):
        """int32 [len(pages)]: the pages' `used` words (raw: the sign bit, void: bit 30, the entries' count below)."""
        pg = torch.as_tensor(list(leaves), dtype=torch.int64, device="cuda")
        return store.view(-1, page)[pg][:, 16 * H : 16 * H + 4].contiguous().view(torch.int32).flatten()

    def raw_books(esc, store, page, H, leaves, st, layer, what):
        """The raw pool's books against the pages': the slabs taken (the bitmap), the owners it lists, the status word and the dir[0] of the raw pages among `pages` agree, each slab is named by one page and owned by it."""
        taken = sum(bin(x & 0xFFFFFFFF).count("1") for x in esc.rbits.tolist())
        owned = int((esc.rowner != 0).sum())
        leaves = list(leaves)
        rawp = [p for p, w in zip(leaves, used_of(store, page, H, leaves).tolist()) if w < 0]
        d0 = dirs(store, page, H, rawp)[:, 0].tolist() if rawp else []
        ok = taken == owned == len(d0) == len(set(d0)) == st.tolist()[8] and all(int(esc.rowner[c - 1]) == (((layer + 1) << 24) | p) for p, c in zip(rawp, d0))
        check(ok, f"{what}: the raw pool's books agree with the pages' ({taken} slabs taken by its bitmap, {owned} by its owners, {len(d0)} named by the raw pages, status {st.tolist()[8]}; each owned by its page)")
        return taken

    def pool_ok(esc, store, page, H, leaves, st, layer, what):
        """The pool's books against the pages': the chunks taken (the bitmap), the owners it lists, the status word and the directories of `pages` (the pages written) agree, every directory
        word names a chunk owned by its page, and no chunk is named twice (a raw page's dir[0] names a slab, not a chunk: not counted here)."""
        taken = sum(bin(x & 0xFFFFFFFF).count("1") for x in esc.bits.tolist())
        owned = int((esc.owner != 0).sum())
        d = dirs(store, page, H, leaves)
        d[used_of(store, page, H, leaves) < 0] = 0
        named = d[d != 0].tolist()
        pg = torch.as_tensor(list(leaves), dtype=torch.int64, device="cuda")
        good = all(int(esc.owner[c - 1]) == (((layer + 1) << 24) | int(pg[i])) for i in range(len(pg)) for c in d[i][d[i] != 0].tolist()) if named else True
        ok = taken == owned == len(named) == len(set(named)) == st.tolist()[4] and good
        check(ok, f"{what}: the escape pool's books agree with the pages' ({taken} chunks taken by its bitmap, {owned} by its owners, {len(named)} named by the directories, status {st.tolist()[4]}; each owned by its page)")
        return taken

    # 1. every pattern of every window of 8 values of exponent >> 1, K and V (no exception): the codes and the low bytes
    H, A, T = 1, 63, 4096
    idx = torch.arange(T, device="cuda", dtype=torch.int32)
    base = torch.clamp(torch.arange(128, device="cuda"), max=120).to(torch.uint8).view(1, 128)
    u = ((idx >> 11)[:, None] << 15) | ((base.to(torch.int32) + ((idx >> 8) & 7)[:, None]) << 8) | (idx & 255)[:, None]
    x = ((u - (u >> 15 << 16)).to(torch.int16)).view(T, H, 128)
    store, page, st = kv_pool(T // 16, H, A)
    write(store, page, st, A, x, x, torch.arange(T, device="cuda", dtype=torch.int64), base.view(H, 128), base.view(H, 128))
    k, v = read(store, page, range(T // 16), H, A, base.view(H, 128), base.view(H, 128))
    check(torch.equal(bits(k.view(T, H, 128)), bits(x)) and torch.equal(bits(v.view(T, H, 128)), bits(x)), "format: every bit pattern of every window of 8 values of exponent >> 1 (base 0 to 120: every exponent), K and V, back bit for bit")

    # 2. every 16-bit pattern as an exception (NaN, infinity, denormals, both zeros) among in-window values, ~2% of the values: in a page's own arena (A = 1023), and in a header's
    # 15 entries and the escape pool's chunks
    H, n = 8, 24 if quick else 160
    T = n * 16
    g = torch.Generator(device="cuda").manual_seed(5)
    key, value = kv_natural(T, H, 1, "k"), kv_natural(T, H, 2, "v")
    kb, vb = kv_bases(key), kv_bases(value)
    allp = torch.randperm(65536, device="cuda", generator=g).to(torch.int32)
    out = []
    for xx, b in ((key, kb), (value, vb)):
        e7 = (xx.to(torch.int32) >> 8) & 127
        bb = b.to(torch.int32)[None]
        uu = (xx.to(torch.int32) & 0xFFFF & ~(127 << 8)) | (torch.minimum(torch.maximum(e7, bb), bb + 7) << 8)
        pick = torch.rand(uu.shape, device="cuda", generator=g) < 0.02
        uu[pick] = allp[torch.arange(int(pick.sum()), device="cuda") % 65536]
        out.append((uu - (uu >> 15 << 16)).to(torch.int16))
    key, value = out
    for A, nc in ((1023, 0), (_lib.kv_free_bay(H), 6000)):
        esc = _lib.Extra(nc) if nc else None
        store, page, st = kv_pool(n, H, A)
        write(store, page, st, A, key, value, torch.arange(T, device="cuda", dtype=torch.int64), kb, vb, esc, 2)
        k, v = read(store, page, range(n), H, A, kb, vb, esc)
        s0 = st.tolist()
        where = "in the pages' own arenas of 1,023 entries" if not nc else f"in arenas of {A} entries and {nc:,} chunks of the escape pool"
        more = f", {s0[5]} chunks taken at most, {s0[6]} pages took one" if nc else ""
        check(torch.equal(bits(k.view(T, H, 128)), bits(key)) and torch.equal(bits(v.view(T, H, 128)), bits(value)) and s0[0] == 0,
              f"format: every 16-bit pattern as an exception (NaN, infinity, denormals, zeros), ~2% of {T * H * 256:,} values, K and V, back bit for bit, {where} (most entries a page was given {s0[1]}, {s0[3]:,} entries{more})")
        if nc:
            pool_ok(esc, store, page, H, range(n), st, 2, "after the first write")

    # 3. sequences written in steps of mixed sizes (whole pages, pieces that start and end inside one, single tokens, rewrites) into a pool of a previous owner's bytes;
    # for 8 KV heads (a warp a head), and for 3, 16 and 32 (fewer heads than warps; a round of 8 heads and then another)
    def recycling(H, rounds, nseq, A=63, outlier=0.0004, nc=0, nr=0):
        NP = 400
        g = torch.Generator().manual_seed(H)
        rnd = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=g))
        NT = 16 * 48 * 12
        kall, vall = kv_natural(NT, H, 11, "k", outlier), kv_natural(NT, H, 12, "v", outlier)
        kb, vb = kv_bases(kv_natural(NT, H, 21, "k")), kv_bases(kv_natural(NT, H, 22, "v"))
        store, page, st = kv_pool(NP, H, A)
        esc = _lib.Extra(nc, "cuda", nr, H) if (nc or nr) else None
        written = set()
        allok, steps, rewrites = True, 0, 0
        for r in range(rounds):
            free = torch.randperm(NP, generator=g).tolist()  # (every round on shuffled pages: a recycled page holds the last owner's bytes)
            seqs = []
            for _ in range(nseq):
                L = rnd(20, 16 * 48)
                seqs.append({"leaves": [free.pop() for _ in range((L + 15) // 16)], "target": L, "L": 0, "K": torch.zeros(L, H, 128, dtype=torch.int16, device="cuda"), "V": torch.zeros(L, H, 128, dtype=torch.int16, device="cuda")})
            active = list(seqs)
            while active and allok:
                ks, vs, sl, upd = [], [], [], []
                for sq in list(active):
                    left = sq["target"] - sq["L"]
                    n = min(left, rnd(1, 120)) if sq["L"] == 0 else (1 if rnd(0, 4) <= 1 else min(left, rnd(1, 40)))
                    a = sq["L"]
                    if sq["L"] > 4 and rnd(0, 6) == 0:  # a rewrite: back up a few tokens, write them and on
                        a = sq["L"] - rnd(1, min(sq["L"], 20) + 1)
                        n = min(sq["target"] - a, n + (sq["L"] - a))
                        rewrites += 1
                    b = a + n
                    o = rnd(0, NT - n)
                    ks.append(kall[o : o + n])
                    vs.append(vall[o : o + n])
                    t = torch.arange(a, b, device="cuda", dtype=torch.int64)
                    sl.append(torch.tensor(sq["leaves"], device="cuda", dtype=torch.int64)[t // 16] * 16 + t % 16)
                    written.update(sq["leaves"][a // 16 : (b - 1) // 16 + 1])
                    sq["K"][a:b], sq["V"][a:b] = ks[-1], vs[-1]
                    sq["L"] = max(sq["L"], b)
                    if sq["L"] >= sq["target"]:
                        active.remove(sq)
                    upd.append(sq)
                key, value, cells = torch.cat(ks), torch.cat(vs), torch.cat(sl)
                pad = rnd(0, 5)  # (padding: slots of -1, rows of nothing)
                if pad:
                    cells = torch.cat([cells, torch.full((pad,), -1, dtype=torch.int64, device="cuda")])
                    key = torch.cat([key, torch.full((pad, H, 128), 7, dtype=torch.int16, device="cuda")])
                    value = torch.cat([value, torch.full((pad, H, 128), 7, dtype=torch.int16, device="cuda")])
                write(store, page, st, A, key, value, cells, kb, vb, esc, 5)
                steps += 1
                if steps % 3 == 0 or not active:
                    for sq in upd:
                        k, v = read(store, page, sq["leaves"], H, A, kb, vb, esc)
                        if not (torch.equal(bits(k.view(-1, H, 128)[: sq["L"]]), bits(sq["K"][: sq["L"]])) and torch.equal(bits(v.view(-1, H, 128)[: sq["L"]]), bits(sq["V"][: sq["L"]]))):
                            allok = False
                            if os.environ.get("KV_DEBUG"):
                                badk = (bits(k.view(-1, H, 128)[: sq["L"]]) != bits(sq["K"][: sq["L"]])).flatten(1).any(1).nonzero().flatten().tolist()
                                badv = (bits(v.view(-1, H, 128)[: sq["L"]]) != bits(sq["V"][: sq["L"]])).flatten(1).any(1).nonzero().flatten().tolist()
                                us = used_of(store, page, H, sq["leaves"])
                                print(f"KV_DEBUG: seq of {sq['L']} tokens pages {sq['leaves'][:6]} used words {[hex(x & 0xFFFFFFFF) for x in us.tolist()][:6]} bad K tokens {badk[:20]} bad V tokens {badv[:20]} step {steps} rewrites {rewrites}", flush=True)
        if esc is not None and esc.chunks:
            pool_ok(esc, store, page, H, sorted(written), st, 5, f"recycling with the pool ({H} heads)")
        if esc is not None and esc.lumps:
            raw_books(esc, store, page, H, sorted(written), st, 5, f"recycling with the raw pool ({H} heads)")
        return allok, steps, rewrites, st.tolist(), rounds * nseq

    allok, steps, rewrites, s0, nseq = recycling(8, 1 if quick else 3, 4 if quick else 8)
    check(allok and s0[0] == 0, f"recycling and rewrites: {nseq} sequences in {steps} steps of mixed sizes ({rewrites} rewrites, padding slots) into pages of a previous owner's random bytes, every token the last value written (most entries a page was given {s0[1]}, overflow {s0[0]})")
    for H in (3, 16) if quick else (3, 16, 32):
        allok, steps, rewrites, s0, nseq = recycling(H, 1, 2 if quick else 4)
        check(allok and s0[0] == 0, f"recycling and rewrites with {H} KV heads: {nseq} sequences in {steps} steps ({rewrites} rewrites): every token the last value written (most entries a page was given {s0[1]}, overflow {s0[0]})")
    # the same into the header's own arena and the escape pool, with exceptions in 0.4% of the values (a page holds 100 to 200 of them: past its arena, into chunks: the kills of a rewrite reach them)
    for H in (8,) if quick else (8, 3, 16):
        A = _lib.kv_free_bay(H)
        allok, steps, rewrites, s0, nseq = recycling(H, 1 if quick else 2, 3 if quick else 6, A, 0.004, 5000)
        check(allok and s0[0] == 0 and s0[6] > 0, f"recycling and rewrites through the escape pool ({H} KV heads, arenas of {A} entries, exceptions in 0.4% of the values): {nseq} sequences in {steps} steps ({rewrites} rewrites): every token the last value written "
              f"(most entries a page was given {s0[1]}, {s0[6]} pages took a chunk, {s0[5]} chunks taken at most, overflow {s0[0]})")

    # the same with exceptions in 3% of the values and a small escape pool (20 chunks: two pages' worth), a raw pool for every page: most pages are raw, some take chunks, rewrites reach both
    for H in (8,) if quick else (8, 3):
        A = _lib.kv_free_bay(H)
        allok, steps, rewrites, s0, nseq = recycling(H, 1 if quick else 2, 3 if quick else 6, A, 0.03, 20, 400)
        check(allok and s0[10] > 0 and s0[11] == 0, f"recycling and rewrites through raw pages ({H} KV heads, exceptions in 3% of the values, 20 chunks and 400 slabs): {nseq} sequences in {steps} steps ({rewrites} rewrites): every token the last value written "
              f"({s0[10]} pages made raw, {s0[8]} slabs taken at the end, {s0[6]} pages took a chunk, none void)")

    # 3b. a decode step's tokens: every step writes one token to each sequence (a run of one token: the write kernel's decode-step path), among rewrites of one token at a place of its page that is
    # not the first (the kills reach its old entries; the tokens after it stay), with the path and without it (GLYD_KV_LEAN=0): every token the last value written, and the books the same
    def decode_steps(H, nseq, A, outlier, nc, nr, lean, seed=7):
        NP = 600
        g = torch.Generator().manual_seed(seed + H)
        rnd = lambda lo, hi: int(torch.randint(lo, hi, (1,), generator=g))
        NT = 16 * 48 * 6
        kall, vall = kv_natural(NT, H, 31, "k", outlier), kv_natural(NT, H, 32, "v", outlier)
        kb, vb = kv_bases(kv_natural(NT, H, 33, "k")), kv_bases(kv_natural(NT, H, 34, "v"))
        store, page, st = kv_pool(NP, H, A, seed=3)
        esc = _lib.Extra(nc, "cuda", nr, H) if (nc or nr) else None
        free = torch.randperm(NP, generator=g).tolist()
        seqs = []
        for _ in range(nseq):
            L = rnd(20, 16 * 40)
            seqs.append({"leaves": [free.pop() for _ in range((L + 15) // 16)], "target": L, "L": 0, "K": torch.zeros(L, H, 128, dtype=torch.int16, device="cuda"), "V": torch.zeros(L, H, 128, dtype=torch.int16, device="cuda")})
        os.environ["GLYD_KV_LEAN"] = "1" if lean else "0"
        allok, steps, rewrites, active = True, 0, 0, list(seqs)
        try:
            while active and allok:
                ks, vs, sl, upd = [], [], [], []
                for sq in list(active):
                    p = sq["L"]
                    if sq["L"] > 20 and rnd(0, 5) == 0:  # (a rewrite of one token that is not a page's first)
                        p = rnd(1, sq["L"])
                        while p % 16 == 0:
                            p = rnd(1, sq["L"])
                        rewrites += 1
                    o = rnd(0, NT)
                    ks.append(kall[o : o + 1])
                    vs.append(vall[o : o + 1])
                    sl.append(torch.tensor([sq["leaves"][p // 16] * 16 + p % 16], device="cuda", dtype=torch.int64))
                    sq["K"][p], sq["V"][p] = ks[-1][0], vs[-1][0]
                    if p == sq["L"]:
                        sq["L"] += 1
                        if sq["L"] >= sq["target"]:
                            active.remove(sq)
                    upd.append(sq)
                key, value, cells = torch.cat(ks), torch.cat(vs), torch.cat(sl)
                pad = rnd(0, 4)  # (padding: slots of -1, rows of nothing)
                if pad:
                    cells = torch.cat([cells, torch.full((pad,), -1, dtype=torch.int64, device="cuda")])
                    key = torch.cat([key, torch.full((pad, H, 128), 7, dtype=torch.int16, device="cuda")])
                    value = torch.cat([value, torch.full((pad, H, 128), 7, dtype=torch.int16, device="cuda")])
                write(store, page, st, A, key, value, cells, kb, vb, esc, 6)
                steps += 1
                if steps % 4 == 0 or not active:
                    for sq in upd:
                        k, v = read(store, page, sq["leaves"], H, A, kb, vb, esc)
                        if not (torch.equal(bits(k.view(-1, H, 128)[: sq["L"]]), bits(sq["K"][: sq["L"]])) and torch.equal(bits(v.view(-1, H, 128)[: sq["L"]]), bits(sq["V"][: sq["L"]]))):
                            allok = False
            for sq in seqs:  # (every sequence at the end, whatever step it ended in)
                k, v = read(store, page, sq["leaves"], H, A, kb, vb, esc)
                allok = allok and torch.equal(bits(k.view(-1, H, 128)[: sq["L"]]), bits(sq["K"][: sq["L"]])) and torch.equal(bits(v.view(-1, H, 128)[: sq["L"]]), bits(sq["V"][: sq["L"]]))
        finally:
            os.environ.pop("GLYD_KV_LEAN", None)
        if esc is not None and esc.chunks and not esc.lumps:
            pool_ok(esc, store, page, H, sorted({p for sq in seqs for p in sq["leaves"]}), st, 6, f"decode steps ({H} heads, {'lean' if lean else 'general'})")
        return allok, steps, rewrites, st.tolist()

    for H, A, outlier, nc, nr, what in ((8, 63, 0.0004, 0, 0, "the pages' own arenas of 63 entries"), (8, None, 0.004, 5000, 0, "the escape pool"), (3, None, 0.004, 5000, 0, "the escape pool, 3 heads"),
                                        (16, None, 0.004, 5000, 0, "the escape pool, 16 heads"), (8, None, 0.03, 20, 400, "raw pages")):
        if quick and H != 8:
            continue
        AA = _lib.kv_free_bay(H) if A is None else A
        nseq = 6 if quick else 12
        r = {lean: decode_steps(H, nseq, AA, outlier, nc, nr, lean) for lean in (True, False)}
        (ok1, steps, rw, s1), (ok0, _, _, s0) = r[True], r[False]
        if nr == 0:  # (no raw pool: the entries every token has are the same by either path, and none is without a place)
            good, books = s1[3] == s0[3] and s1[0] == s0[0] == 0, f"the same {s1[3]:,} entries, overflow {s1[0]}"
        else:  # (a raw pool: which pages are made raw depends on the order the pages' warps take slabs in; none is void)
            good, books = s1[10] > 0 and s0[10] > 0 and s1[11] == s0[11] == 0, f"{s1[10]} and {s0[10]} pages made raw, none void"
        check(ok1 and ok0 and good,
              f"decode steps, {what} ({H} KV heads, exceptions in {outlier * 100:g}% of the values): {nseq} sequences a token a step in {steps} steps ({rw} rewrites of one token), padding slots, pages of a previous owner's bytes: every token the last value written, "
              f"with the decode step's path and without ({books})")

    # 4. an arena that is too small, and no pools: the pages that overflow are void, counted and named in the status's ring, not hidden
    def voids_of(store, page, H, leaves):
        """The pages among `pages` that are void (not raw: its sign bit), and the blocks the status's ring names (sorted)."""
        return [p for p, w in zip(leaves, used_of(store, page, H, leaves).tolist()) if w >= 0 and w & 0x40000000]

    def ring_of(s0):
        return sorted(s0[16 + k % _lib.KV_AUX] - 1 for k in range(max(0, s0[12] - _lib.KV_AUX), s0[12]))

    H, A, n = 8, 31, 16
    T = n * 16
    key, value = kv_natural(T, H, 1, "k", 0.03), kv_natural(T, H, 2, "v", 0.03)
    store, page, st = kv_pool(n, H, A)
    write(store, page, st, A, key, value, torch.arange(T, device="cuda", dtype=torch.int64), kv_bases(kv_natural(T, H, 3, "k")), kv_bases(kv_natural(T, H, 4, "v")))
    s0 = st.tolist()
    voids = voids_of(store, page, H, list(range(n)))
    check(s0[0] > 0 and s0[1] > A and s0[11] == len(voids) > 0 and ring_of(s0) == voids and s0[10] == 0,
          f"an arena that is full and no pools: the pages that overflow are void ({s0[11]} of {n}), counted and named in the status's ring (blocks {voids[:6]}{' ...' if len(voids) > 6 else ''}), none raw, {s0[0]:,} entries found no place")

    # 4b. the escape pool's own cases: chunks taken and given back, a pool that runs out, a directory the pool does not own
    H, A, n = 8, _lib.kv_free_bay(8), 24 if quick else 120
    T = n * 16
    cells = torch.arange(T, device="cuda", dtype=torch.int64)
    kb, vb = kv_bases(kv_natural(T, H, 3, "k")), kv_bases(kv_natural(T, H, 4, "v"))
    key, value = kv_natural(T, H, 41, "k", 0.02), kv_natural(T, H, 42, "v", 0.02)  # (about 650 exceptions a page: 15 in the arena, the rest in 10 chunks)
    esc = _lib.Extra(4 * n * 10)
    store, page, st = kv_pool(n, H, A)
    write(store, page, st, A, key, value, cells, kb, vb, esc, 3)
    k, v = read(store, page, range(n), H, A, kb, vb, esc)
    s0 = st.tolist()
    taken = pool_ok(esc, store, page, H, range(n), st, 3, "heavy pages")
    check(torch.equal(bits(k.view(T, H, 128)), bits(key)) and torch.equal(bits(v.view(T, H, 128)), bits(value)) and s0[0] == 0 and s0[6] == n and taken > 5 * n,
          f"the escape pool: {n} pages of ~650 exceptions each (arenas of {A}) take {taken} chunks ({s0[6]} pages took one, {s0[3]:,} entries) and read back bit for bit; none dropped")
    key2, value2 = kv_natural(T, H, 51, "k"), kv_natural(T, H, 52, "v")
    write(store, page, st, A, key2, value2, cells, kb, vb, esc, 3)  # (every page written again from its first token: its chunks go back)
    k, v = read(store, page, range(n), H, A, kb, vb, esc)
    taken2 = pool_ok(esc, store, page, H, range(n), st, 3, "the same pages written again with light values")
    check(torch.equal(bits(k.view(T, H, 128)), bits(key2)) and torch.equal(bits(v.view(T, H, 128)), bits(value2)) and taken2 < taken // 20 and st.tolist()[0] == 0,
          f"the escape pool: the same pages written again from their first token with ordinary values give their chunks back ({taken} taken before, {taken2} after) and read back bit for bit")
    # gathered into a paged bf16 cache (the prefill rows' staging): the same values at ((block 16 + token) H + head) 2 D, keys then values, into shuffled blocks, with pages whose entries are in chunks
    pool2, page2, st2 = kv_pool(n, H, A, seed=21)
    esc2 = _lib.Extra(4 * n * 10)
    write(pool2, page2, st2, A, key, value, cells, kb, vb, esc2, 3)
    dst = torch.randperm(n + 7, generator=torch.Generator().manual_seed(4))[:n].to(torch.int32).to("cuda")
    cache = torch.full((n + 7, 16, H, 256), 0x7F7F, dtype=torch.int16, device="cuda")
    _lib.kv_gather(pool2, page2, torch.arange(n, dtype=torch.int32, device="cuda"), dst, kb, vb, A, esc2, cache.view(torch.bfloat16))
    got = cache[dst.to(torch.int64)]  # [n, 16, H, 256]
    untouched = torch.tensor(sorted(set(range(n + 7)) - set(dst.tolist())), device="cuda")
    check(torch.equal(got[..., :128].reshape(T, H, 128), key) and torch.equal(got[..., 128:].reshape(T, H, 128), value) and bool((cache[untouched] == 0x7F7F).all()),
          f"gather: {n} pages (about 650 exceptions each, in chunks of the escape pool) decoded into shuffled blocks of a paged bf16 cache, the keys and values where vLLM's FlashAttention reads them, bit for bit; the blocks that are not named untouched")
    # a pool that runs out (64 chunks for pages that need 10 each) and no raw pool: the pages without a place are void and named, the pool is whole
    esc = _lib.Extra(64)
    store, page, st = kv_pool(n, H, A)
    write(store, page, st, A, key, value, cells, kb, vb, esc, 3)
    s0 = st.tolist()
    taken = pool_ok(esc, store, page, H, range(n), st, 3, "a pool that ran out")
    voids = voids_of(store, page, H, list(range(n)))
    check(s0[0] > 0 and s0[5] == 64 and taken == 64 and s0[11] == len(voids) > 0 and ring_of(s0) == voids,
          f"the escape pool: 64 chunks for {n} pages that need ten each run out: {s0[0]:,} entries found no place, {s0[11]} pages void and named in the ring, {s0[5]} chunks taken at most, none lost")
    # a directory the pool did not write (a block of a previous owner's bytes, or of another pool): its words name no chunk that is freed
    esc = _lib.Extra(4 * n * 10)
    store, page, st = kv_pool(2 * n, H, A, seed=9)
    write(store, page, st, A, key, value, cells, kb, vb, esc, 3)  # (pages 0 .. n - 1: heavy, chunks taken; pages n .. 2 n - 1: random bytes)
    p = store.view(-1, page)
    p[n : 2 * n, 16 * H + 4 : 16 * H + 4 + 64] = p[0:n, 16 * H + 4 : 16 * H + 4 + 64].clone()  # the second half's directories name the first half's chunks (and its `used` words are what they are)
    taken = pool_ok(esc, store, page, H, range(n), st, 3, "before the stale directories are written over")
    write(store, page, st, A, key2, value2, cells + n * 16, kb, vb, esc, 3)  # the second half written from its first token: light values
    k, v = read(store, page, range(n), H, A, kb, vb, esc)
    s0 = st.tolist()
    check(torch.equal(bits(k.view(T, H, 128)), bits(key)) and torch.equal(bits(v.view(T, H, 128)), bits(value)) and s0[7] >= n and s0[4] == taken,
          f"the escape pool: directories of recycled pages that name chunks of other pages ({s0[7]} words) free none of them: the first half still reads back bit for bit and {s0[4]} of {taken} chunks are taken")

    # 4c. raw pages: a page that has no place for an entry is stored as it is: 12 bits a value in the page, the other 4 in a slab of the raw pool
    H, A = 8, _lib.kv_free_bay(8)
    n = 16 if quick else 48
    T = n * 16
    cells = torch.arange(T, device="cuda", dtype=torch.int64)
    kb, vb = kv_bases(kv_natural(T, H, 3, "k")), kv_bases(kv_natural(T, H, 4, "v"))
    gnoise = torch.Generator(device="cuda").manual_seed(77)

    def noise(t):
        u = torch.randint(0, 65536, (t, H, 128), dtype=torch.int32, device="cuda", generator=gnoise)
        return (u - (u >> 15 << 16)).to(torch.int16)

    key, value = noise(T), noise(T)  # (every 16-bit pattern: NaN, infinity, denormals, zeros ...: every value an exception)
    esc = _lib.Extra(1000, "cuda", n + 8, H)
    store, page, st = kv_pool(n, H, A)
    write(store, page, st, A, key, value, cells, kb, vb, esc, 4)
    k, v = read(store, page, range(n), H, A, kb, vb, esc)
    s0 = st.tolist()
    check(torch.equal(bits(k.view(T, H, 128)), bits(key)) and torch.equal(bits(v.view(T, H, 128)), bits(value)) and s0[10] == n and s0[11] == 0 and s0[8] == n,
          f"raw pages: {n} pages of random 16-bit patterns (every value an exception: {s0[0]:,} found no place) are made raw ({s0[10]}), take a slab each ({s0[8]}), and read back bit for bit, K and V; none void")
    raw_books(esc, store, page, H, range(n), st, 4, "raw pages")
    pool_ok(esc, store, page, H, range(n), st, 4, "raw pages")
    # gathered into a paged bf16 cache, and the same pages written again (rewrites of every token, then from the first token: the slabs go back)
    dst = torch.randperm(n + 5, generator=torch.Generator().manual_seed(6))[:n].to(torch.int32).to("cuda")
    cache = torch.full((n + 5, 16, H, 256), 0x7F7F, dtype=torch.int16, device="cuda")
    _lib.kv_gather(store, page, torch.arange(n, dtype=torch.int32, device="cuda"), dst, kb, vb, A, esc, cache.view(torch.bfloat16))
    got = cache[dst.to(torch.int64)]
    check(torch.equal(got[..., :128].reshape(T, H, 128), key) and torch.equal(got[..., 128:].reshape(T, H, 128), value), f"gather: {n} raw pages decoded into shuffled blocks of a paged bf16 cache, bit for bit")
    key2, value2 = kv_natural(T, H, 51, "k"), kv_natural(T, H, 52, "v")
    write(store, page, st, A, key2, value2, cells, kb, vb, esc, 4)
    k, v = read(store, page, range(n), H, A, kb, vb, esc)
    raw_books(esc, store, page, H, range(n), st, 4, "raw pages written again with ordinary values")
    check(torch.equal(bits(k.view(T, H, 128)), bits(key2)) and torch.equal(bits(v.view(T, H, 128)), bits(value2)) and st.tolist()[8] == 0 and st.tolist()[0] == s0[0],
          f"raw pages: the same pages written again from their first token with ordinary values give their slabs back ({s0[8]} taken before, {st.tolist()[8]} after), are no longer raw, and read back bit for bit")

    # a page made raw by a later token: tokens one at a time, step t into token t of every page, a heavy token (a whole row of random patterns) among them; after each step every token written so far read back
    nb = 12 if quick else 24
    Tb = nb * 16
    kn, vn = kv_natural(Tb, H, 61, "k"), kv_natural(Tb, H, 62, "v")
    heavy = (torch.rand(Tb, generator=torch.Generator().manual_seed(8)) < 0.1).to("cuda")
    kn[heavy], vn[heavy] = noise(int(heavy.sum())), noise(int(heavy.sum()))
    esc = _lib.Extra(3000, "cuda", nb + 4, H)
    store, page, st = kv_pool(nb, H, A)
    kb2, vb2 = kv_bases(kv_natural(Tb, H, 63, "k")), kv_bases(kv_natural(Tb, H, 64, "v"))
    stepok = True
    for t in range(16):
        sel = torch.arange(nb, device="cuda") * 16 + t
        write(store, page, st, A, kn[sel], vn[sel], sel, kb2, vb2, esc, 4)
        k, v = read(store, page, range(nb), H, A, kb2, vb2, esc)
        stepok &= torch.equal(bits(k.view(nb, 16, H, 128)[:, : t + 1]), bits(kn.view(nb, 16, H, 128)[:, : t + 1])) and torch.equal(bits(v.view(nb, 16, H, 128)[:, : t + 1]), bits(vn.view(nb, 16, H, 128)[:, : t + 1]))
    s0 = st.tolist()
    want = int(heavy.view(nb, 16).any(1).sum())
    check(stepok and s0[10] == want and s0[11] == 0, f"raw pages: {nb} pages written a token at a time, {int(heavy.sum())} heavy tokens among them: {s0[10]} of the {want} pages that hold one are made raw (by the token, with the tokens before it in the page decoded), "
          "every token written so far read back bit for bit after each step; none void")
    raw_books(esc, store, page, H, range(nb), st, 4, "pages made raw by a later token")
    pool_ok(esc, store, page, H, range(nb), st, 4, "pages made raw by a later token")
    # inside a raw page: a token written again (a slot written before: its position in the page past the first), then every page written again from its first token
    kr, vr = kv_natural(nb, H, 71, "k"), kv_natural(nb, H, 72, "v")
    sel = torch.arange(nb, device="cuda") * 16 + 5
    write(store, page, st, A, kr, vr, sel, kb2, vb2, esc, 4)
    kn.view(nb, 16, H, 128)[:, 5], vn.view(nb, 16, H, 128)[:, 5] = kr, vr
    k, v = read(store, page, range(nb), H, A, kb2, vb2, esc)
    check(torch.equal(bits(k.view(Tb, H, 128)), bits(kn)) and torch.equal(bits(v.view(Tb, H, 128)), bits(vn)) and st.tolist()[10] == want, "raw pages: token 5 of every page written again (inside the raw ones): the page reads back bit for bit, none made raw again")
    key3, value3 = kv_natural(Tb, H, 73, "k"), kv_natural(Tb, H, 74, "v")
    write(store, page, st, A, key3, value3, torch.arange(Tb, device="cuda", dtype=torch.int64), kb2, vb2, esc, 4)
    k, v = read(store, page, range(nb), H, A, kb2, vb2, esc)
    raw_books(esc, store, page, H, range(nb), st, 4, "the pages written again from their first token")
    check(torch.equal(bits(k.view(Tb, H, 128)), bits(key3)) and torch.equal(bits(v.view(Tb, H, 128)), bits(value3)) and st.tolist()[8] == 0, "raw pages: every page written again from its first token with ordinary values: no slab is taken, bit for bit")

    # a raw pool that is too small: the first slabs go to pages, the others' pages are void and named in the ring; those that got one read back bit for bit
    n2, ns = 20, 8
    key, value = noise(n2 * 16), noise(n2 * 16)
    esc = _lib.Extra(100, "cuda", ns, H)
    store, page, st = kv_pool(n2, H, A)
    write(store, page, st, A, key, value, torch.arange(n2 * 16, device="cuda", dtype=torch.int64), kb, vb, esc, 4)
    s0 = st.tolist()
    voids = voids_of(store, page, H, list(range(n2)))
    rawp = [p for p, w in enumerate(used_of(store, page, H, list(range(n2))).tolist()) if w < 0]
    k, v = read(store, page, rawp, H, A, kb, vb, esc)
    sel = torch.tensor(rawp, device="cuda")
    good = torch.equal(bits(k), bits(key.view(n2, 16, H, 128)[sel])) and torch.equal(bits(v), bits(value.view(n2, 16, H, 128)[sel]))
    check(good and len(rawp) == ns and s0[10] == ns and s0[11] == len(voids) == n2 - ns and ring_of(s0) == voids and s0[8] == ns,
          f"raw pool too small: {n2} raw pages for {ns} slabs: {len(rawp)} are raw and read back bit for bit, {s0[11]} are void and named in the ring, {s0[8]} slabs taken")
    raw_books(esc, store, page, H, range(n2), st, 4, "a raw pool that ran out")

    # the escape pool out of chunks, a raw pool with slabs: the pages that find no chunk are made raw, none dropped
    n3 = 40 if quick else 120
    T3 = n3 * 16
    key, value = kv_natural(T3, H, 41, "k", 0.02), kv_natural(T3, H, 42, "v", 0.02)  # (about 650 exceptions a page: 10 chunks of the pool)
    kb3, vb3 = kv_bases(kv_natural(T3, H, 3, "k")), kv_bases(kv_natural(T3, H, 4, "v"))
    esc = _lib.Extra(64, "cuda", n3 + 8, H)
    store, page, st = kv_pool(n3, H, A)
    write(store, page, st, A, key, value, torch.arange(T3, device="cuda", dtype=torch.int64), kb3, vb3, esc, 6)
    k, v = read(store, page, range(n3), H, A, kb3, vb3, esc)
    s0 = st.tolist()
    check(torch.equal(bits(k.view(T3, H, 128)), bits(key)) and torch.equal(bits(v.view(T3, H, 128)), bits(value)) and s0[11] == 0 and s0[10] > 0 and s0[5] <= 64,
          f"raw pages: {n3} pages of ~650 exceptions for 64 chunks: the escape pool ran dry ({s0[0]:,} entries found no place, {s0[5]} chunks taken at most), {s0[10]} pages were made raw instead and every page reads back bit for bit; none void")
    pool_ok(esc, store, page, H, range(n3), st, 6, "pages that ran the chunks out")
    raw_books(esc, store, page, H, range(n3), st, 6, "pages that ran the chunks out")
    # a block whose pages are void in several layers is named in the ring once while it stays lost (the bitmap), and again after its next life
    n5, layers = 8, 3
    key, value = noise(n5 * 16), noise(n5 * 16)
    key2, value2 = kv_natural(n5 * 16, H, 83, "k"), kv_natural(n5 * 16, H, 84, "v")
    stl, lost = torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda")
    pools = [kv_pool(n5, H, A, seed=30 + l)[0] for l in range(layers)]
    for l in range(layers):
        write(pools[l], page, stl, A, key, value, torch.arange(n5 * 16, device="cuda", dtype=torch.int64), kb, vb, None, l, lost)
    s0 = stl.tolist()
    named_once = s0[11] == layers * n5 and s0[12] == n5 and ring_of(s0) == list(range(n5)) and lost.tolist()[0] & 0x5555 == 0x5555
    for l in range(layers):  # (the next life: layer 0's first tokens clear the bits, the layers after it fail again: each block is named once more)
        write(pools[l], page, stl, A, key2 if l == 0 else key, value2 if l == 0 else value, torch.arange(n5 * 16, device="cuda", dtype=torch.int64), kb, vb, None, l, lost)
    s1 = stl.tolist()
    check(named_once and s1[12] == 2 * n5 and s1[11] == (2 * layers - 1) * n5,
          f"void pages: a block whose pages are void in {layers} layers is named in the ring once ({s0[11]} void pages, {s0[12]} blocks named), and once again after its next life ({s1[12]} named, {s1[11]} void pages)")

    # a block with a raw page is named in the raw ring once while it keeps one (the pages of later layers add none), and again after its next life
    n6, layers6 = 8, 3
    key, value = noise(n6 * 16), noise(n6 * 16)
    key2, value2 = kv_natural(n6 * 16, H, 85, "k"), kv_natural(n6 * 16, H, 86, "v")
    st6, lost6 = torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda")
    esc6 = _lib.Extra(500, "cuda", layers6 * n6, H)
    pools6 = [kv_pool(n6, H, A, seed=40 + l)[0] for l in range(layers6)]
    slots6 = torch.arange(n6 * 16, device="cuda", dtype=torch.int64)
    raw_ring = lambda w: sorted(w[_lib.KV_SECOND_AUX_AT + k % _lib.KV_AUX] - 1 for k in range(max(0, w[_lib.KV_SECOND_AUX_HEAD] - _lib.KV_AUX), w[_lib.KV_SECOND_AUX_HEAD]))
    for l in range(layers6):
        write(pools6[l], page, st6, A, key, value, slots6, kb, vb, esc6, l, lost6)
    t0 = st6.tolist()
    once = t0[10] == layers6 * n6 and t0[_lib.KV_SECOND_AUX_HEAD] == n6 and raw_ring(t0) == list(range(n6)) and t0[11] == 0 and t0[12] == 0 and lost6.tolist()[0] & 0xAAAA == 0xAAAA
    for l in range(layers6):  # (the next life: layer 0's first tokens are natural and clear the bits, the layers after it are heavy again)
        write(pools6[l], page, st6, A, key2 if l == 0 else key, value2 if l == 0 else value, slots6, kb, vb, esc6, l, lost6)
    t1 = st6.tolist()
    check(once and t1[_lib.KV_SECOND_AUX_HEAD] == 2 * n6 and t1[10] == (2 * layers6 - 1) * n6 and t1[11] == 0,
          f"raw pages: a block with a raw page in {layers6} layers is named in the raw ring once ({t0[10]} raw pages, {t0[_lib.KV_SECOND_AUX_HEAD]} blocks named, none in the void ring), and once again after its next life ({t1[_lib.KV_SECOND_AUX_HEAD]} named, {t1[10]} raw pages)")

    # the rings' places are what the status's size says: a small ring wraps (the head counts every block, the places hold some of them, the other ring's places are not touched), a large one holds them all
    n8 = 20
    key, value = noise(n8 * 16), noise(n8 * 16)
    slots8 = torch.arange(n8 * 16, device="cuda", dtype=torch.int64)
    ring_ok, ring_what = True, []
    for aux in (6, 5000):
        for kind in ("void", "raw"):  # (no raw pool: every page is void; with slabs for all: every page is raw)
            st8, lost8 = torch.zeros(_lib.kv_status_words(aux), dtype=torch.int32, device="cuda"), torch.zeros(2, dtype=torch.int32, device="cuda")
            esc8 = _lib.Extra(100, "cuda", n8, H) if kind == "raw" else None
            pool8, page8, _ = kv_pool(n8, H, A, seed=60 + aux % 7)
            write(pool8, page8, st8, A, key, value, slots8, kb, vb, esc8, 2, lost8)
            w = st8.tolist()
            at, other = (_lib.KV_AUX_AT, _lib.kv_second_aux_at(aux)) if kind == "void" else (_lib.kv_second_aux_at(aux), _lib.KV_AUX_AT)
            head = w[_lib.KV_AUX_HEAD if kind == "void" else _lib.KV_SECOND_AUX_HEAD]
            places = w[at : at + aux]
            named = sorted(x - 1 for x in places if x)
            good = head == n8 and w[11 if kind == "void" else 10] == n8 and not any(w[other : other + aux]) and len(w) == _lib.kv_status_words(aux)
            good = good and (named == list(range(n8)) if aux >= n8 else len(named) == aux and all(0 <= b < n8 for b in named) and len(set(named)) == aux)
            ring_ok &= good
            ring_what.append(f"{kind} ring of {aux}: head {head}, {len(named)} places hold a block")
    check(ring_ok, f"rings of other sizes (status words 16 + 2 x ring): {n8} blocks made void and raw in rings of 6 places (wrapping: the head counts every block, the places hold some, the other ring's are untouched) and of 5,000 (all named once): " + "; ".join(ring_what))

    # blocks cleared (kv_release): the slabs and chunks their pages hold go back to the pools, their counts and directories are zero, their bits in the lost bitmap are cleared, the others' stay; written again they are as new
    n7, layers7, rel_n = 8, 3, 6
    key, value = noise(n7 * 16), noise(n7 * 16)
    key2, value2 = kv_natural(n7 * 16, H, 87, "k"), kv_natural(n7 * 16, H, 88, "v")
    st7, lost7 = torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda")
    esc7 = _lib.Extra(500, "cuda", layers7 * n7, H)
    pools7 = [kv_pool(n7, H, A, seed=50 + l)[0] for l in range(layers7)]
    slots7 = torch.arange(n7 * 16, device="cuda", dtype=torch.int64)
    for l in range(layers7):
        write(pools7[l], page, st7, A, key, value, slots7, kb, vb, esc7, l, lost7)
    u0 = st7.tolist()
    rel = torch.arange(rel_n, dtype=torch.int32, device="cuda")
    for l in range(layers7):
        _lib.kv_release(pools7[l], page, n7, rel, H, A, esc7, l, st7, lost7)
    u1 = st7.tolist()
    taken = sum(bin(x & 0xFFFFFFFF).count("1") for x in esc7.rbits.tolist())
    words = torch.cat([used_of(pools7[l], page, H, range(n7)) for l in range(layers7)]).view(layers7, n7)
    cleared = bool((words[:, :rel_n] == 0).all()) and bool((words[:, rel_n:] < 0).all())
    ok7 = u0[8] == layers7 * n7 and u1[8] == layers7 * (n7 - rel_n) == taken and cleared and lost7.tolist()[0] & 0xFFFF == 0xA000 and u1[7] == u0[7]
    k7, v7 = read(pools7[0], page, range(n7), H, A, kb, vb, esc7)
    again = torch.equal(bits(k7.view(n7 * 16, H, 128))[rel_n * 16 :], bits(key)[rel_n * 16 :])  # (the blocks that were not cleared read back as they were written)
    for l in range(layers7):  # (the cleared blocks written again with ordinary values)
        write(pools7[l], page, st7, A, key2[: rel_n * 16], value2[: rel_n * 16], slots7[: rel_n * 16], kb, vb, esc7, l, lost7)
    k7, v7 = read(pools7[1], page, range(rel_n), H, A, kb, vb, esc7)
    u2 = st7.tolist()
    check(ok7 and again and torch.equal(bits(k7.view(rel_n * 16, H, 128)), bits(key2[: rel_n * 16])) and u2[11] == 0,
          f"released blocks: {rel_n} of {n7} blocks cleared in {layers7} layers give back {u0[8] - u1[8]} of {u0[8]} slabs (books: {taken} taken by the bitmap, {u1[8]} by the status), their counts are zero and their bits cleared, the {n7 - rel_n} others stay raw "
          f"and read back as written, and the cleared blocks written again read back bit for bit")

    # a directory the pools did not write (a raw page's `used` and dir[0], of a previous owner's bytes): none of its words frees a slab
    n4 = 12
    key, value = noise(n4 * 16), noise(n4 * 16)
    esc = _lib.Extra(100, "cuda", 2 * n4, H)
    store, page, st = kv_pool(2 * n4, H, A, seed=12)
    write(store, page, st, A, key, value, torch.arange(n4 * 16, device="cuda", dtype=torch.int64), kb, vb, esc, 4)
    p = store.view(-1, page)
    p[n4 : 2 * n4, 16 * H : 16 * H + 8] = p[0:n4, 16 * H : 16 * H + 8].clone()  # (the second half: raw pages' used words and the first half's slabs in dir[0])
    taken = raw_books(esc, store, page, H, range(n4), st, 4, "before the stale raw directories are written over")
    key2, value2 = kv_natural(n4 * 16, H, 81, "k"), kv_natural(n4 * 16, H, 82, "v")
    write(store, page, st, A, key2, value2, torch.arange(n4 * 16, 2 * n4 * 16, device="cuda", dtype=torch.int64), kb, vb, esc, 4)
    k, v = read(store, page, range(n4), H, A, kb, vb, esc)
    s0 = st.tolist()
    check(torch.equal(bits(k.view(n4 * 16, H, 128)), bits(key)) and torch.equal(bits(v.view(n4 * 16, H, 128)), bits(value)) and s0[7] >= n4 and s0[8] == taken,
          f"raw pages: directories of recycled pages that name slabs of other pages ({s0[7]} words) free none of them: the first half still reads back bit for bit and {s0[8]} of {taken} slabs are taken")

    # 5. decode attention against an fp32 reference, on pages of a previous owner's bytes, for 1 to 8 query heads a KV head (and, with 4, for 2 and 16 KV heads)
    def decode_worst(H, G, lens, A=63, outlier=0.0003, nc=0, shares=None, nr=0, heavy=0.0):
        npg = [(L + 15) // 16 for L in lens]
        Tn = sum(npg) * 16
        esc = _lib.Extra(nc, "cuda", nr, H) if (nc or nr) else None
        key, value = kv_natural(Tn, H, 31, "k", outlier, (60, 180)), kv_natural(Tn, H, 32, "v", outlier, (60, 180))
        if heavy:  # (tokens whose every value is far from its window: a page with one has more entries than a page holds, and is raw)
            hv = torch.rand(Tn, device="cuda", generator=torch.Generator(device="cuda").manual_seed(G + 5)) < heavy
            key[hv], value[hv] = kv_natural(Tn, H, 81, "k", 1.0, (80, 175))[hv], kv_natural(Tn, H, 82, "v", 1.0, (80, 175))[hv]
        kb, vb = kv_bases(kv_natural(max(Tn, 4096), H, 33, "k")), kv_bases(kv_natural(max(Tn, 4096), H, 34, "v"))  # (from enough tokens: a handful's bases leave values outside their windows past an arena of 63)
        store, page, st = kv_pool(sum(npg) + 8, H, A, seed=G)
        perm = torch.randperm(sum(npg) + 8, generator=torch.Generator().manual_seed(G)).tolist()
        bt = torch.zeros(len(lens), max(npg), dtype=torch.int32, device="cuda")
        sl, pos, truth = [], 0, []
        for b, L in enumerate(lens):
            leaves = [perm.pop() for _ in range(npg[b])]
            bt[b, : npg[b]] = torch.tensor(leaves, dtype=torch.int32, device="cuda")
            t = torch.arange(L, device="cuda", dtype=torch.int64)
            sl.append(torch.tensor(leaves, device="cuda", dtype=torch.int64)[t // 16] * 16 + t % 16)
            truth.append((key[pos : pos + L], value[pos : pos + L]))
            pos += L
        write(store, page, st, A, torch.cat([t[0] for t in truth]), torch.cat([t[1] for t in truth]), torch.cat(sl), kb, vb, esc, 7)
        gen = torch.Generator(device="cuda").manual_seed(G)
        q = (torch.randn(len(lens), H * G, 128, device="cuda", generator=gen) * 2.0).to(torch.bfloat16)
        o = torch.empty_like(q)
        scale = 1.0 / math.sqrt(128)
        sp = shares or _lib.kv_decode_shares(len(lens), H, max(npg))
        _lib.kv_decode(q, store, page, kb, vb, bt, torch.tensor(lens, dtype=torch.int32, device="cuda"), sp, H, G, A, esc, scale, o, _lib.kv_decode_workspace(len(lens), H, G, sp))
        torch.cuda.synchronize()
        if st.tolist()[11]:  # (a void page is a wrong test, not a wrong kernel: not an answer)
            return float("inf")
        if heavy and not st.tolist()[10]:  # (no page was made raw: the test did not test)
            return float("inf")
        worst = 0.0
        for b, L in enumerate(lens):
            kk, vv = truth[b][0].view(torch.bfloat16).float(), truth[b][1].view(torch.bfloat16).float()
            sc = torch.einsum("hgd,lhd->hgl", q[b].float().view(H, G, 128), kk) * scale
            ref = torch.einsum("hgl,lhd->hgd", torch.softmax(sc, -1), vv).reshape(H * G, 128)
            worst = max(worst, ((o[b].float() - ref).abs().max() / ref.abs().max()).item()) if not torch.isnan(o[b]).any() else float("inf")
        return worst

    worst = 0.0
    lens = [1, 15, 17, 100, 255, 700] if quick else [1, 15, 16, 17, 31, 33, 100, 255, 1000, 2049]
    for G in (1, 4, 8) if quick else (1, 2, 3, 4, 5, 6, 7, 8):  # (every group size the kernel takes: Qwen2.5-7B's 28 heads on 4 is 7, Qwen2.5-3B's 16 on 2 is 8, Llama-3.2's 24 on 8 is 3)
        worst = max(worst, decode_worst(8, G, lens))
    check(worst < 0.012, f"decode attention: {len(lens)} sequences of 1 to {max(lens):,} tokens (the last page partly written, the bytes past it a previous owner's), {'1, 4 and 8' if quick else '1 to 8'} query heads a KV head: within {worst:.4f} of an fp32 reference (at most 0.012 of the output's largest)")
    worst = max(decode_worst(H, 4, lens[:5]) for H in (2, 16))
    check(worst < 0.012, f"decode attention with 2 and 16 KV heads (4 query heads each): within {worst:.4f} of an fp32 reference")
    worst = max(decode_worst(H, G, lens, _lib.kv_free_bay(H), 0.004, 4000) for H, G in ((8, 4), (8, 1), (2, 8)))
    check(worst < 0.012, f"decode attention over pages whose exceptions (0.4% of the values) run past the header's arena into the escape pool's chunks, 8 and 2 KV heads: within {worst:.4f} of an fp32 reference")
    worst = max(decode_worst(H, G, lens, _lib.kv_free_bay(H), 0.0003, 300, nr=3000, heavy=0.04) for H, G in ((8, 4), (8, 1), (2, 8)))
    check(worst < 0.012, f"decode attention over pages with heavy tokens (4% of the tokens: every value far from its window), the pages that hold one raw (a raw pool of slabs), 8 and 2 KV heads: within {worst:.4f} of an fp32 reference")
    long_lens = [1, 33, 700, 2049, 4097]
    worst = max(decode_worst(8, 4, long_lens, shares=sp) for sp in (1, 2, 3, 7, 64))
    check(worst < 0.012, f"decode attention with the plan's CTA count a sequence (the most splits) fixed at 1, 2, 3, 7 and 64 (a sequence of 1 to {max(long_lens):,} tokens has at most one CTA for 16 pages): within {worst:.4f} of an fp32 reference")

    # 6. the decode kernel's plan comes from each sequence's own length: its output's bits are the same alone, in a batch, and in a batch padded with rows of no tokens (a CUDA graph's), at one
    # `splits`; a row of no tokens (its pages anything) gives zeros; and a CUDA graph (the kernel launches only: no host read, no allocation) replayed on the next step's tensors gives the bits of a call
    def decode_pool(H, G, lens, A=63, seed=3):
        npg = [(L + 15) // 16 for L in lens]
        Tn = sum(npg) * 16
        key, value = kv_natural(Tn, H, 31, "k", 0.0003, (60, 180)), kv_natural(Tn, H, 32, "v", 0.0003, (60, 180))
        kb, vb = kv_bases(kv_natural(max(Tn, 4096), H, 33, "k")), kv_bases(kv_natural(max(Tn, 4096), H, 34, "v"))
        store, page, st = kv_pool(sum(npg) + 8, H, A, seed=seed)
        perm = torch.randperm(sum(npg) + 8, generator=torch.Generator().manual_seed(seed)).tolist()
        bt = torch.zeros(len(lens), max(npg), dtype=torch.int32, device="cuda")
        sl = []
        for b, L in enumerate(lens):
            leaves = [perm.pop() for _ in range(npg[b])]
            bt[b, : npg[b]] = torch.tensor(leaves, dtype=torch.int32, device="cuda")
            t = torch.arange(L, device="cuda", dtype=torch.int64)
            sl.append(torch.tensor(leaves, device="cuda", dtype=torch.int64)[t // 16] * 16 + t % 16)
        pos = 0
        keys, values = [], []
        for L in lens:
            keys.append(key[pos : pos + L])
            values.append(value[pos : pos + L])
            pos += L
        write(store, page, st, A, torch.cat(keys), torch.cat(values), torch.cat(sl), kb, vb, None, 7)
        return store, page, kb, vb, bt

    H, G, A = 8, 4, 63
    lens = [1, 17, 100, 255, 700, 1500, 3000, 4100]
    store, page, kb, vb, bt = decode_pool(H, G, lens)
    gen = torch.Generator(device="cuda").manual_seed(1)
    scale = 1.0 / math.sqrt(128)
    B, pad = len(lens), 5
    q = (torch.randn(B, H * G, 128, device="cuda", generator=gen) * 2.0).to(torch.bfloat16)
    sl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    qp = torch.cat([q, (torch.randn(pad, H * G, 128, device="cuda", generator=gen) * 2.0).to(torch.bfloat16)])
    btp = torch.cat([bt, torch.full((pad, bt.shape[1]), 2**30, dtype=torch.int32, device="cuda")])  # (the padding rows' pages: nowhere)
    slp = torch.cat([sl, torch.zeros(pad, dtype=torch.int32, device="cuda")])
    same, zero, tried = True, True, 0
    _lib.kv_floor_set(1)  # (the automatic floor is from the batch's size: a number of CTAs a sequence has at least, fixed here; the next checks say what it is. The library's call, which a release library takes as a development one takes GLYD_KV_FLOOR)
    try:
        for sp in (1, 2, 4, 9, 16):
            ws = _lib.kv_decode_workspace(B + pad, H, G, sp)
            full, padded = torch.empty_like(q), torch.empty_like(qp)
            _lib.kv_decode(q, store, page, kb, vb, bt, sl, sp, H, G, A, None, scale, full, ws)
            _lib.kv_decode(qp, store, page, kb, vb, btp, slp, sp, H, G, A, None, scale, padded, ws)
            same &= torch.equal(padded[:B], full)
            zero &= not bool(padded[B:].any())
            for i in range(B):
                alone = torch.empty_like(q[i : i + 1])
                _lib.kv_decode(q[i : i + 1], store, page, kb, vb, bt[i : i + 1], sl[i : i + 1], sp, H, G, A, None, scale, alone, ws)
                same &= torch.equal(alone[0], full[i])
                tried += 1
    finally:
        _lib.kv_floor_set(0)
    check(same, f"decode attention's plan is a sequence's own: {B} sequences of 1 to {max(lens):,} tokens, each alone, in a batch of {B} and in a batch padded to {B + pad} with rows of no tokens, at 5 values of splits: the same bits ({tried} sequences' outputs compared)")
    check(zero, f"decode attention: rows of no tokens (their pages 2^30, nowhere) give zeros")

    # the plan's least pages a CTA (GLYD_KV_PPC_MIN: 16, 8, 4) and the last CTA's merge of the partials (GLYD_KV_FUSE=0: a kernel of its own merges them): at each value a sequence's bits are its own (alone, in a batch,
    # in a padded batch) and the merge by either gives the same bits; and the plans' outputs are within the reference's tolerance
    same, fused_same, pipe_same, tried = True, True, True, 0
    try:
        for ppc, fl in (("16", "1"), ("8", "1"), ("4", "1"), ("16", "16"), ("16", "4"), ("8", "16")):
            os.environ["GLYD_KV_PPC_MIN"] = ppc  # (a development library's: a release library has the plan's least pages fixed)
            _lib.kv_floor_set(int(fl))
            for sp in (1, 3, 8, 16, 64, 300):
                ws = _lib.kv_decode_workspace(B + pad, H, G, sp)
                full, padded, unfused, early = torch.empty_like(q), torch.empty_like(qp), torch.empty_like(q), torch.empty_like(q)
                _lib.kv_decode(q, store, page, kb, vb, bt, sl, sp, H, G, A, None, scale, full, ws)
                _lib.kv_decode(qp, store, page, kb, vb, btp, slp, sp, H, G, A, None, scale, padded, ws)
                for knob, val, out in (("GLYD_KV_FUSE", "0", unfused), ("GLYD_KV_PIPE", "1", early)):
                    os.environ[knob] = val
                    try:
                        _lib.kv_decode(q, store, page, kb, vb, bt, sl, sp, H, G, A, None, scale, out, ws)
                    finally:
                        os.environ.pop(knob)
                same &= torch.equal(padded[:B], full)
                fused_same &= torch.equal(unfused, full)
                pipe_same &= torch.equal(early, full)
                for i in range(B):
                    alone = torch.empty_like(q[i : i + 1])
                    _lib.kv_decode(q[i : i + 1], store, page, kb, vb, bt[i : i + 1], sl[i : i + 1], sp, H, G, A, None, scale, alone, ws)
                    same &= torch.equal(alone[0], full[i])
                    tried += 1
        _lib.kv_floor_set(1)
        worst = max((os.environ.__setitem__("GLYD_KV_PPC_MIN", ppc), decode_worst(8, 4, [1, 17, 100, 700, 2049, 4100]))[1] for ppc in ("8", "4"))
        os.environ["GLYD_KV_PPC_MIN"] = "16"
        _lib.kv_floor_set(16)
        worst_floor = decode_worst(8, 4, [1, 17, 100, 700, 2049, 4100])
    finally:
        os.environ.pop("GLYD_KV_PPC_MIN", None)
        _lib.kv_floor_set(0)
    check(same, f"decode attention at 3 values of the plan's least pages a CTA (16, 8, 4), 3 of its least CTAs a sequence (1, 4, 16) and 6 of splits (1 to 300): a sequence's bits its own, alone, in a batch of {B} and in a batch padded with {pad} rows of no tokens ({tried} outputs compared)")
    check(fused_same, "decode attention: the last CTA's merge of the partials gives the bits of the merge by a kernel of its own, at those plans")
    check(pipe_same, "decode attention: each warp loading its next page ahead (GLYD_KV_PIPE=1) or not (the default) gives the same bits, at those plans")
    check(worst < 0.012, f"decode attention with 8 and 4 pages a CTA at the least: within {worst:.4f} of an fp32 reference")
    check(worst_floor < 0.012, f"decode attention with 16 CTAs a sequence at the least (4 pages a CTA at the fewest): within {worst_floor:.4f} of an fp32 reference")

    # the CTAs a sequence has (the partials its CTAs wrote, counted in a fresh workspace) are the plan's: at least ppc_min pages a CTA, at most `splits`, and with a floor at least that many as far as 4 pages each; the automatic
    # floor is 16 on a GPU of 100 SMs or more for a batch of few (sequence, kv head) pairs (4 to an SM: fewer than the SMs) and 1 for others, and a batch padded to its size's next power of two has its plan
    def partials_of(ids, sp):
        n = len(ids)
        ws = _lib.kv_decode_workspace(n, H, G, sp)  # (a fresh one: zero, so that a split that wrote no partial reads sum 0)
        _lib.kv_decode(q[ids], store, page, kb, vb, bt[ids], torch.tensor([lens[i] for i in ids], dtype=torch.int32, device="cuda"), sp, H, G, A, None, scale, torch.empty_like(q[ids]), ws)
        torch.cuda.synchronize()
        ml = ws[_lib.KV_COUNTER_BYTES : _lib.KV_COUNTER_BYTES + n * H * sp * G * 2 * 4].view(torch.float32).view(n, H, sp, G, 2)
        return (ml[..., 1] > 0).any(dim=3).sum(dim=2)  # [n, H]: the splits that wrote a partial

    def want_partials(L, sp, fl):
        npg_ = (L + 15) // 16
        return min(sp, max(max(1, -(-npg_ // 16)), min(fl, -(-npg_ // 4))))

    cnt_ok, shown = True, []
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    try:
        for fl in (1, 4, 16):
            _lib.kv_floor_set(fl)
            for sp in (4, 16, 64):
                got = partials_of(list(range(len(lens))), sp)
                for b, L in enumerate(lens):
                    cnt_ok &= bool((got[b] == want_partials(L, sp, fl)).all())
        _lib.kv_floor_set(0)
        for nb in (1, 2, 3, 4, 8):  # (the automatic floor, by the batch's rows: the longest nb sequences)
            rows = 1 << max(0, (nb - 1).bit_length())
            fl = 16 if cus >= 100 and rows * H * 4 <= cus else 1
            sp = _lib.kv_decode_shares(rows, H, 257)
            ids = list(range(len(lens) - 1, len(lens) - 1 - nb, -1))
            got = partials_of(ids, sp)
            for k, i in enumerate(ids):
                cnt_ok &= bool((got[k] == want_partials(lens[i], sp, fl)).all())
            shown.append(f"{nb} rows {fl}")
    finally:
        _lib.kv_floor_set(0)
    check(cnt_ok, f"decode attention's CTAs a sequence are the plan's (at least 16 pages a CTA, at most the splits, and with a floor of 1, 4 and 16 at least that many as far as 4 pages each), counted from the partials written, at 3 values of splits; "
                  f"the automatic floor on this GPU ({cus} SMs, {H} KV heads; 16 on a GPU of 100 SMs or more for a batch of few sequences, else 1): {', '.join(shown)}")

    sp = _lib.kv_decode_shares(B, H, 257)
    wsg = _lib.kv_decode_workspace(B, H, G, sp)
    qg, slg, btg, outg = q.clone(), sl.clone(), bt.clone(), torch.empty_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _lib.kv_decode(qg, store, page, kb, vb, btg, slg, sp, H, G, A, None, scale, outg, wsg)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _lib.kv_decode(qg, store, page, kb, vb, btg, slg, sp, H, G, A, None, scale, outg, wsg)
    gsame = True
    for trial in range(4):
        gl = torch.Generator(device="cuda").manual_seed(100 + trial)
        qg.copy_((torch.randn(B, H * G, 128, device="cuda", generator=gl) * 2.0).to(torch.bfloat16))
        order = torch.randperm(B, device="cuda", generator=gl)
        nl = [min(lens[int(o)], max(0, lens[int(o)] - 17 * trial)) for o in order]  # (shorter lengths, rows shuffled; the last trial's last rows of no tokens: a batch padded)
        if trial == 3:
            nl[-3:] = [0, 0, 0]
        btg.copy_(bt[order])
        slg.copy_(torch.tensor(nl, dtype=torch.int32, device="cuda"))
        graph.replay()
        ref = torch.empty_like(q)
        _lib.kv_decode(qg, store, page, kb, vb, btg, slg, sp, H, G, A, None, scale, ref, torch.zeros_like(wsg))
        gsame &= torch.equal(outg, ref)
    check(gsame, f"decode attention in a CUDA graph (splits {sp}, {B} rows), replayed on four steps' lengths, block tables and queries written into its tensors (rows shuffled, shorter, the last three of no tokens): the bits of a call of the same arguments")

    # the write kernel in a CUDA graph: its tokens and slots read at the replay (a slot -1: padding, nothing written)
    Hw, nw = 8, 3
    kk = kv_natural(nw * 16, Hw, 41, "k", 0.0003)
    vv = kv_natural(nw * 16, Hw, 42, "v", 0.0003)
    kbw, vbw = kv_bases(kv_natural(4096, Hw, 43, "k")), kv_bases(kv_natural(4096, Hw, 44, "v"))
    poolw, pagew, stw = kv_pool(2 * nw + 2, Hw, A, seed=9)
    kg, vg, sg = torch.zeros_like(kk), torch.zeros_like(vv), torch.full((nw * 16,), -1, dtype=torch.int64, device="cuda")
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _lib.kv_write(kg, vg, sg, poolw, pagew, kbw, vbw, A, None, 5, stw)
    torch.cuda.current_stream().wait_stream(side)
    gw = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gw):
        _lib.kv_write(kg, vg, sg, poolw, pagew, kbw, vbw, A, None, 5, stw)
    wsame = True
    for trial in range(3):
        first = 2 * trial  # (pages first .. first + 2: this replay's)
        n_real = nw * 16 - 5 * trial  # (the last tokens of the batch are padding)
        kg.copy_(kk)
        vg.copy_(vv)
        t = torch.arange(nw * 16, device="cuda", dtype=torch.int64)
        sg.copy_(torch.where(t < n_real, first * 16 + t, -1))
        gw.replay()
        k, v = read(poolw, pagew, range(first, first + nw), Hw, A, kbw, vbw)
        wsame &= torch.equal(bits(k.view(nw * 16, Hw, 128))[:n_real], bits(kk)[:n_real]) and torch.equal(bits(v.view(nw * 16, Hw, 128))[:n_real], bits(vv)[:n_real])
    check(wsame and stw.tolist()[0] == 0, "the write kernel in a CUDA graph, replayed on three steps' tokens and slots (the last tokens padding, slot -1): every token written reads back bit for bit")

    # the staging put: a step's tokens' keys and values into the staging cache's slots (a slot -1, or past the cache, is skipped; the rows of a fused projection's output, a row stride wider than the heads'),
    # against torch's index_copy
    sput = True
    for trial, (Hs, nt, nb) in enumerate(((8, 300, 40), (4, 17, 3), (1, 1, 1), (2, 1000, 70), (8, 2048, 200))):
        gl = torch.Generator(device="cuda").manual_seed(200 + trial)
        fused = torch.randn(nt + 9, 2 * Hs * 128 + 64, device="cuda", generator=gl).to(torch.bfloat16)  # (nine rows more than the slots: not put; the 64 columns another tensor's)
        ks, vs = fused[:, : Hs * 128].view(nt + 9, Hs, 128), fused[:, Hs * 128 : 2 * Hs * 128].view(nt + 9, Hs, 128)
        sl = torch.randperm(nb * 16, device="cuda", generator=gl)[:nt].to(torch.int64)
        sl[::3] = -1
        sl[1::7] = nb * 16 + 5
        out = torch.zeros(nb, 16, Hs, 256, dtype=torch.bfloat16, device="cuda")
        _lib.kv_work_put(ks, vs, sl, out)
        live = (sl >= 0) & (sl < nb * 16)
        ref = torch.zeros(nb * 16, Hs, 256, dtype=torch.bfloat16, device="cuda")
        ref[sl[live], :, :128] = ks[:nt][live]
        ref[sl[live], :, 128:] = vs[:nt][live]
        sput &= torch.equal(bits(out.view(-1, Hs, 256)), bits(ref))
    check(sput, "the staging put (5 shapes: 1-2,048 tokens, 1-8 heads, rows of a fused projection's output, slots -1 and past the cache skipped): the staging cache is the bits torch's index_copy leaves")

    # 7. head size 256 (Qwen3.5 and Qwen3.8: 4 KV heads; 4 and 6 query heads a KV head): a head held as two of the page's heads of 128. Written and read back bit for bit (every 16-bit pattern as an
    # exception, the escape pool's chunks; raw pages), decoded into the staging cache whole (a head's 256 channels together, as FlashAttention reads them), the staging put, and decode attention against an
    # fp32 reference, its bits a sequence's own (alone, batched, padded; the merge in the kernel or apart; the next page ahead or not) and a CUDA graph's the bits of a call
    D2 = 256

    def nat2(T, H, seed, kind, outlier=0.0, erange=(1, 255)):  # [T, H, 256]: the pages' heads of 128, two a head
        return kv_natural(T, 2 * H, seed, kind, outlier, erange).view(T, H, D2)

    def bases2(x):
        return kv_bases(x.view(x.shape[0], -1, 128)).view(x.shape[1], D2)

    def read2(store, page, leaves, H, A, kb, vb, esc=None):
        leaves = torch.as_tensor(leaves, dtype=torch.int32, device="cuda")
        ok = torch.empty(leaves.numel(), 16, H, D2, dtype=torch.int16, device="cuda")
        ov = torch.empty_like(ok)
        _lib.kv_unpack(store, page, leaves, kb, vb, A, esc, ok, ov)
        return ok, ov

    H, n = 4, 24 if quick else 160
    T = n * 16
    g2 = torch.Generator(device="cuda").manual_seed(55)
    key, value = nat2(T, H, 1, "k"), nat2(T, H, 2, "v")
    kb, vb = bases2(key), bases2(value)
    allp = torch.randperm(65536, device="cuda", generator=g2).to(torch.int32)
    out = []
    for xx, b in ((key, kb), (value, vb)):
        e7 = (xx.to(torch.int32) >> 8) & 127
        bb = b.to(torch.int32)[None]
        uu = (xx.to(torch.int32) & 0xFFFF & ~(127 << 8)) | (torch.minimum(torch.maximum(e7, bb), bb + 7) << 8)
        pick = torch.rand(uu.shape, device="cuda", generator=g2) < 0.02
        uu[pick] = allp[torch.arange(int(pick.sum()), device="cuda") % 65536]
        out.append((uu - (uu >> 15 << 16)).to(torch.int16))
    key, value = out
    A2 = _lib.kv_free_bay(H, D2)
    for A, nc in ((1023, 0), (A2, 6000)):
        esc = _lib.Extra(nc) if nc else None
        store, page, st = kv_pool(n, 2 * H, A)
        assert page == _lib.kv_leaf_bytes(H, D2, A)
        write(store, page, st, A, key, value, torch.arange(T, device="cuda", dtype=torch.int64), kb, vb, esc, 2)
        k, v = read2(store, page, range(n), H, A, kb, vb, esc)
        s0 = st.tolist()
        check(torch.equal(bits(k.view(T, H, D2)), bits(key)) and torch.equal(bits(v.view(T, H, D2)), bits(value)) and s0[0] == 0,
              f"head size 256: every 16-bit pattern as an exception, ~2% of {T * H * 2 * D2:,} values, K and V, back bit for bit, {'in arenas of 1,023 entries' if not nc else f'in arenas of {A} entries and {nc:,} chunks of the escape pool'} ({s0[3]:,} entries)")
        if nc:
            pool_ok(esc, store, page, 2 * H, range(n), st, 2, "head size 256, after the first write")
    # raw pages: tokens whose every value is far from its window
    store, page, st = kv_pool(n, 2 * H, A2)
    esc = _lib.Extra(300, "cuda", 3000, 2 * H)
    hv = torch.rand(T, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)) < 0.04
    kh, vh = nat2(T, H, 11, "k", 0.0003, (60, 180)), nat2(T, H, 12, "v", 0.0003, (60, 180))
    kb, vb = bases2(nat2(4096, H, 21, "k")), bases2(nat2(4096, H, 22, "v"))
    kh[hv], vh[hv] = nat2(T, H, 81, "k", 1.0, (80, 175))[hv], nat2(T, H, 82, "v", 1.0, (80, 175))[hv]
    write(store, page, st, A2, kh, vh, torch.arange(T, device="cuda", dtype=torch.int64), kb, vb, esc, 3)
    k, v = read2(store, page, range(n), H, A2, kb, vb, esc)
    s0 = st.tolist()
    check(torch.equal(bits(k.view(T, H, D2)), bits(kh)) and torch.equal(bits(v.view(T, H, D2)), bits(vh)) and s0[10] > 0 and s0[11] == 0, f"head size 256: pages with heavy tokens raw ({s0[10]} pages made raw), back bit for bit")
    raw_books(esc, store, page, 2 * H, range(n), st, 3, "head size 256")
    # the staging cache: pages decoded into shuffled blocks, a head's keys at its 256 channels and its values after them
    blocks = n + 7
    dst = torch.randperm(blocks, generator=torch.Generator().manual_seed(3))[:n].to(torch.int32).cuda()
    cache = torch.zeros(blocks, 16, H, 2 * D2, dtype=torch.int16, device="cuda")
    _lib.kv_gather(store, page, torch.arange(n, dtype=torch.int32, device="cuda"), dst, kb, vb, A2, esc, cache.view(torch.bfloat16))
    got = cache[dst.long()]
    check(torch.equal(got[..., :D2], kh.view(n, 16, H, D2)) and torch.equal(got[..., D2:], vh.view(n, 16, H, D2)) and not bool(cache[[b for b in range(blocks) if b not in set(dst.tolist())]].any()),
          f"head size 256: {n} pages (raw ones among them) decoded into shuffled blocks of the staging cache [blocks, 16, {H}, 512], a head's keys and then its values whole: bit for bit, the other blocks untouched")
    sput = True
    for trial, (Hs, nt, nb) in enumerate(((4, 300, 40), (2, 17, 3), (1, 1, 1), (4, 1000, 70))):
        gl = torch.Generator(device="cuda").manual_seed(300 + trial)
        fused = torch.randn(nt + 9, 2 * Hs * D2 + 64, device="cuda", generator=gl).to(torch.bfloat16)
        ks, vs = fused[:, : Hs * D2].view(nt + 9, Hs, D2), fused[:, Hs * D2 : 2 * Hs * D2].view(nt + 9, Hs, D2)
        sl = torch.randperm(nb * 16, device="cuda", generator=gl)[:nt].to(torch.int64)
        sl[::3] = -1
        sl[1::7] = nb * 16 + 5
        outp = torch.zeros(nb, 16, Hs, 2 * D2, dtype=torch.bfloat16, device="cuda")
        _lib.kv_work_put(ks, vs, sl, outp)
        live = (sl >= 0) & (sl < nb * 16)
        ref = torch.zeros(nb * 16, Hs, 2 * D2, dtype=torch.bfloat16, device="cuda")
        ref[sl[live], :, :D2] = ks[:nt][live]
        ref[sl[live], :, D2:] = vs[:nt][live]
        sput &= torch.equal(bits(outp.view(-1, Hs, 2 * D2)), bits(ref))
    check(sput, "head size 256: the staging put (4 shapes, slots -1 and past the cache skipped): the bits torch's index_copy leaves")

    def decode2(H, G, lens, A=63, outlier=0.0003, nc=0, shares=None, nr=0, heavy=0.0, seed=0):
        """(the worst error against an fp32 reference, the pool, its page, bases, block table, queries): sequences of lens tokens at head size 256 on pages of a previous owner's bytes."""
        npg = [(L + 15) // 16 for L in lens]
        Tn = sum(npg) * 16
        esc = _lib.Extra(nc, "cuda", nr, 2 * H) if (nc or nr) else None
        key, value = nat2(Tn, H, 31 + seed, "k", outlier, (60, 180)), nat2(Tn, H, 32 + seed, "v", outlier, (60, 180))
        if heavy:
            hv = torch.rand(Tn, device="cuda", generator=torch.Generator(device="cuda").manual_seed(G + 5)) < heavy
            key[hv], value[hv] = nat2(Tn, H, 81, "k", 1.0, (80, 175))[hv], nat2(Tn, H, 82, "v", 1.0, (80, 175))[hv]
        kb, vb = bases2(nat2(max(Tn, 4096), H, 33, "k")), bases2(nat2(max(Tn, 4096), H, 34, "v"))
        store, page, st = kv_pool(sum(npg) + 8, 2 * H, A, seed=G)
        perm = torch.randperm(sum(npg) + 8, generator=torch.Generator().manual_seed(G)).tolist()
        bt = torch.zeros(len(lens), max(npg), dtype=torch.int32, device="cuda")
        sl, pos, truth = [], 0, []
        for b, L in enumerate(lens):
            leaves = [perm.pop() for _ in range(npg[b])]
            bt[b, : npg[b]] = torch.tensor(leaves, dtype=torch.int32, device="cuda")
            t = torch.arange(L, device="cuda", dtype=torch.int64)
            sl.append(torch.tensor(leaves, device="cuda", dtype=torch.int64)[t // 16] * 16 + t % 16)
            truth.append((key[pos : pos + L], value[pos : pos + L]))
            pos += L
        write(store, page, st, A, torch.cat([t[0] for t in truth]), torch.cat([t[1] for t in truth]), torch.cat(sl), kb, vb, esc, 7)
        gen = torch.Generator(device="cuda").manual_seed(G + seed)
        q = (torch.randn(len(lens), H * G, D2, device="cuda", generator=gen) * 2.0).to(torch.bfloat16)
        o = torch.empty_like(q)
        scale = 1.0 / math.sqrt(D2)
        sp = shares or _lib.kv_decode_shares(len(lens), H, max(npg))
        _lib.kv_decode(q, store, page, kb, vb, bt, torch.tensor(lens, dtype=torch.int32, device="cuda"), sp, H, G, A, esc, scale, o, _lib.kv_decode_workspace(len(lens), H, G, sp, None, D2))
        torch.cuda.synchronize()
        if st.tolist()[11] or (heavy and not st.tolist()[10]):
            return float("inf"), None
        worst = 0.0
        for b, L in enumerate(lens):
            kk, vv = truth[b][0].view(torch.bfloat16).float(), truth[b][1].view(torch.bfloat16).float()
            sc = torch.einsum("hgd,lhd->hgl", q[b].float().view(H, G, D2), kk) * scale
            ref = torch.einsum("hgl,lhd->hgd", torch.softmax(sc, -1), vv).reshape(H * G, D2)
            worst = max(worst, ((o[b].float() - ref).abs().max() / ref.abs().max()).item()) if not torch.isnan(o[b]).any() else float("inf")
        return worst, (store, page, kb, vb, bt, q, esc)

    lens = [1, 15, 17, 100, 255, 700] if quick else [1, 15, 16, 17, 31, 33, 100, 255, 1000, 2049]
    worst = max(decode2(4, G, lens)[0] for G in ((1, 4, 6, 8) if quick else (1, 2, 3, 4, 5, 6, 7, 8)))
    check(worst < 0.012, f"head size 256: decode attention, {len(lens)} sequences of 1 to {max(lens):,} tokens, 4 KV heads, {'1, 4, 6 and 8' if quick else '1 to 8'} query heads a KV head: within {worst:.4f} of an fp32 reference")
    worst = max(decode2(H, 4, lens[:5])[0] for H in (1, 2, 16))
    check(worst < 0.012, f"head size 256: decode attention with 1, 2 and 16 KV heads: within {worst:.4f} of an fp32 reference")
    worst = max(decode2(4, G, lens, A2, 0.004, 4000)[0] for G in (4, 6))
    check(worst < 0.012, f"head size 256: decode attention over pages whose exceptions (0.4% of the values) run into the escape pool's chunks: within {worst:.4f} of an fp32 reference")
    worst = max(decode2(4, G, lens, A2, 0.0003, 300, nr=3000, heavy=0.04)[0] for G in (4, 6))
    check(worst < 0.012, f"head size 256: decode attention over pages with heavy tokens (raw pages): within {worst:.4f} of an fp32 reference")
    long_lens = [1, 33, 700, 2049, 4097]
    worst = max(decode2(4, 4, long_lens, shares=sp)[0] for sp in (1, 2, 3, 7, 64))
    check(worst < 0.012, f"head size 256: decode attention with the most splits fixed at 1, 2, 3, 7 and 64: within {worst:.4f} of an fp32 reference")
    H, G = 4, 6
    lens = [1, 17, 100, 255, 700, 1500, 3000, 4100]
    _, (store, page, kb, vb, bt, q, _) = decode2(H, G, lens, seed=3)
    gen = torch.Generator(device="cuda").manual_seed(1)
    scale = 1.0 / math.sqrt(D2)
    B, pad = len(lens), 5
    sl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    qp = torch.cat([q, (torch.randn(pad, H * G, D2, device="cuda", generator=gen) * 2.0).to(torch.bfloat16)])
    btp = torch.cat([bt, torch.full((pad, bt.shape[1]), 2**30, dtype=torch.int32, device="cuda")])
    slp = torch.cat([sl, torch.zeros(pad, dtype=torch.int32, device="cuda")])
    same, zero, fused_same, pipe_same, tried = True, True, True, True, 0
    try:
        for fl in (1, 16):
            _lib.kv_floor_set(fl)
            for sp in (1, 3, 9, 16, 64):
                ws = _lib.kv_decode_workspace(B + pad, H, G, sp, None, D2)
                full, padded, unfused, early = torch.empty_like(q), torch.empty_like(qp), torch.empty_like(q), torch.empty_like(q)
                _lib.kv_decode(q, store, page, kb, vb, bt, sl, sp, H, G, 63, None, scale, full, ws)
                _lib.kv_decode(qp, store, page, kb, vb, btp, slp, sp, H, G, 63, None, scale, padded, ws)
                for knob, val, o_ in (("GLYD_KV_FUSE", "0", unfused), ("GLYD_KV_PIPE", "1", early)):
                    os.environ[knob] = val
                    try:
                        _lib.kv_decode(q, store, page, kb, vb, bt, sl, sp, H, G, 63, None, scale, o_, ws)
                    finally:
                        os.environ.pop(knob)
                same &= torch.equal(padded[:B], full)
                zero &= not bool(padded[B:].any())
                fused_same &= torch.equal(unfused, full)
                pipe_same &= torch.equal(early, full)
                for i in range(B):
                    alone = torch.empty_like(q[i : i + 1])
                    _lib.kv_decode(q[i : i + 1], store, page, kb, vb, bt[i : i + 1], sl[i : i + 1], sp, H, G, 63, None, scale, alone, ws)
                    same &= torch.equal(alone[0], full[i])
                    tried += 1
    finally:
        _lib.kv_floor_set(0)
    check(same and zero, f"head size 256: decode attention's bits a sequence's own, alone, in a batch of {B} and in a batch padded with {pad} rows of no tokens (zeros), at 2 floors and 5 values of splits ({tried} outputs compared)")
    check(fused_same and pipe_same, "head size 256: the merge in the kernel or by a kernel of its own, and the next page loaded ahead or not, give the same bits")
    sp = _lib.kv_decode_shares(B, H, 257)
    wsg = _lib.kv_decode_workspace(B, H, G, sp, None, D2)
    qg, slg, btg, outg = q.clone(), sl.clone(), bt.clone(), torch.empty_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _lib.kv_decode(qg, store, page, kb, vb, btg, slg, sp, H, G, 63, None, scale, outg, wsg)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _lib.kv_decode(qg, store, page, kb, vb, btg, slg, sp, H, G, 63, None, scale, outg, wsg)
    gsame = True
    for trial in range(4):
        gl = torch.Generator(device="cuda").manual_seed(100 + trial)
        qg.copy_((torch.randn(B, H * G, D2, device="cuda", generator=gl) * 2.0).to(torch.bfloat16))
        order = torch.randperm(B, device="cuda", generator=gl)
        nl = [min(lens[int(o)], max(0, lens[int(o)] - 17 * trial)) for o in order]
        if trial == 3:
            nl[-3:] = [0, 0, 0]
        btg.copy_(bt[order])
        slg.copy_(torch.tensor(nl, dtype=torch.int32, device="cuda"))
        graph.replay()
        ref = torch.empty_like(q)
        _lib.kv_decode(qg, store, page, kb, vb, btg, slg, sp, H, G, 63, None, scale, ref, torch.zeros_like(wsg))
        gsame &= torch.equal(outg, ref)
    check(gsame, f"head size 256: decode attention in a CUDA graph (splits {sp}, {B} rows), replayed on four steps' lengths, block tables and queries: the bits of a call of the same arguments")

    # torch's synchronization errors (what the metadata builder is run under: no step's build waits for the GPU) do raise where the host waits: a copy to the host, .item(), a blocking copy from
    # pageable memory (the staging plan's first version put its tensors on the device so), torch.unique (its size is the device's); and none for what the builder does (device ops, non-blocking copies)
    import numpy as np

    x = torch.arange(8, device="cuda", dtype=torch.int32)
    idx, vals, dst = torch.tensor([1, 2], device="cuda"), torch.tensor([5, 6], device="cuda"), torch.arange(4, device="cuda")  # (made before the errors are on: a tensor made from a list is a blocking copy)
    raised = []
    torch.cuda.set_sync_debug_mode("error")
    try:
        for name, f in (("copy to the host", lambda: x.cpu()), (".item()", lambda: x.sum().item()), ("blocking copy from pageable memory", lambda: torch.from_numpy(np.arange(8, dtype=np.int32)).to("cuda")), ("torch.unique", lambda: torch.unique(x))):
            try:
                f()
            except RuntimeError:
                raised.append(name)
        quiet = True
        try:
            torch.sort(torch.where(x > 2, x, 99), stable=True)[0].cumsum(0)
            dst.scatter_(0, idx, vals)
            torch.empty(8, device="cuda", dtype=torch.int32).copy_(x, non_blocking=True)
        except RuntimeError:
            quiet = False
    finally:
        torch.cuda.set_sync_debug_mode("default")
    check(len(raised) == 4 and quiet, f"torch's synchronization errors raise at {', '.join(raised)} ({len(raised)} of 4) and at none of the plan's sort, where, cumsum, scatter and a non-blocking copy")
    return res


STUDY = ["chat_qa", "doc_qa_3k", "wiki_4k", "code_4k", "prose_8k", "list_1k", "table_2k", "json_3k", "code_chat_8k", "chat_reason"]


def kv_prompts(tok, max_tokens, n_docs, study=None):
    """The real prompts: with study (the kv-study branch's research/kv-cache/code, which has prompts.py) its ten prompts (a chat, a document question, Wikipedia markup, Python, Dickens,
    a list, a table, JSON, a code review and a reasoning chat: 614 to 8,192 tokens), else PROMPTS and up to n_docs of this repository's documents cut at 1k, 2k, 4k ... max_tokens tokens: [(name, ids)]."""
    if study:
        sys.path.insert(0, os.path.expanduser(study))
        import prompts as PR

        return [(pid, tok(PR.text_of(pid, tok), add_special_tokens=False).input_ids[:max_tokens]) for pid in STUDY]
    out = [(f"short-{i}", tok.encode(p)) for i, p in enumerate(PROMPTS)]
    n = 1024
    for name in DOCS[:n_docs]:
        path = os.path.join(REPO, name)
        if os.path.exists(path):
            ids = tok.encode(open(path, encoding="utf-8").read())
            out.append((name, ids[: min(len(ids), n, max_tokens)]))
            n = min(n * 2, max_tokens)
    return out


_CAP = {"on": False, "cap": {}}


def kv_capture(model):
    """A forward pre-hook on every decoder Attention layer: while _CAP["on"], the key and value it is handed (after the key norm and the rotary embedding: what vLLM stores),
    appended to _CAP["cap"][layer index] as CPU int16 [tokens, KV heads, channels] (LLM.apply_model(kv_capture): the layers' names)."""
    import torch
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    names = []
    for m in model.modules():
        if not isinstance(m, Attention) or m.attn_type != AttentionType.DECODER:
            continue
        li = len(names)
        names.append(m.layer_name)

        def hook(mod, args, kwargs, li=li):
            if not _CAP["on"]:
                return
            k = args[1] if len(args) > 1 else kwargs.get("key")
            v = args[2] if len(args) > 2 else kwargs.get("value")
            k = k.detach().reshape(-1, mod.num_kv_heads, mod.head_size).contiguous().view(torch.int16).cpu()
            v = v.detach().reshape(-1, mod.num_kv_heads, mod.head_size_v).contiguous().view(torch.int16).cpu()
            old = _CAP["cap"].get(li)
            _CAP["cap"][li] = (k, v) if old is None else (torch.cat([old[0], k]), torch.cat([old[1], v]))

        m.register_forward_pre_hook(hook, with_kwargs=True)
    return names


def kv_real(spec):
    """Stock vLLM (eager) on real prompts: the keys and values every decoder layer hands the cache (glyd_gpu.kvcal's hooks: after the key norm and
    the rotary embedding) through Glyd's write kernel into pages of a previous owner's bytes, as vLLM writes them (a prompt in one step, then a token a step),
    every page read back: the values that differ from what vLLM would have stored (none), and what the exception arenas held."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from glyd_gpu import _lib, kernels as g, kvcal

    g.lib()
    t = kvcal.Tables(spec["tables"])
    tok = AutoTokenizer.from_pretrained(spec["model"])
    prompts = kv_prompts(tok, spec.get("max_tokens", 8192), spec.get("docs", 6), spec.get("study"))
    llm = LLM(model=spec["model"], dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=spec.get("max_tokens", 8192) + spec.get("n_gen", 130) + 16, max_num_batched_tokens=spec.get("max_tokens", 8192) + spec.get("n_gen", 130) + 16, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False, max_num_seqs=1, seed=0)
    names = llm.apply_model(kv_capture)[0]
    H = t.k.shape[1]
    A = spec["bay"] if spec.get("bay") is not None else _lib.kv_free_bay(H)
    esc = _lib.Extra(spec.get("chunks", 40000), None, spec.get("lumps", 2000), H)  # (the pools for every layer, as the plugin makes them)
    page = _lib.kv_leaf_bytes(H, 128, A)
    store = {}  # layer: its pages, reused by every prompt (a recycled page holds the last prompt's bytes)
    st = torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device="cuda")
    out = {"spec": spec, "bay": A, "leaf_bytes": page, "prompts": [], "differing": 0, "values": 0, "layers": len(names)}
    for pi, (name, ids) in enumerate(prompts):
        _CAP["cap"].clear()
        _CAP["on"] = True
        llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=spec.get("n_gen", 130), temperature=0.0, ignore_eos=True), use_tqdm=False)
        _CAP["on"] = False
        P = len(ids)
        T = _CAP["cap"][0][0].shape[0]
        npg = (T + 15) // 16
        bad = 0
        for li, ln in enumerate(names):
            k, v = (x.to("cuda") for x in _CAP["cap"][li])
            kb, vb = (torch.from_numpy(a).to("cuda") for a in t.layer(ln))
            if li not in store:
                gen = torch.Generator(device="cuda").manual_seed(li)
                store[li] = torch.randint(0, 256, (720 * page,), dtype=torch.uint8, device="cuda", generator=gen)
            leaves = torch.randperm(720, generator=torch.Generator().manual_seed(pi * 100 + li))[:npg].to(torch.int32).to("cuda")  # (a shuffled set of the pool's pages)
            tt = torch.arange(T, device="cuda", dtype=torch.int64)
            cells = leaves.to(torch.int64)[tt // 16] * 16 + tt % 16
            _lib.kv_write(k[:P], v[:P], cells[:P], store[li], page, kb, vb, A, esc, li, st)  # the prompt: one step
            for j in range(P, T):  # then a token a step
                _lib.kv_write(k[j : j + 1], v[j : j + 1], cells[j : j + 1], store[li], page, kb, vb, A, esc, li, st)
            ok = torch.empty(npg, 16, H, 128, dtype=torch.int16, device="cuda")
            ov = torch.empty_like(ok)
            _lib.kv_unpack(store[li], page, leaves, kb, vb, A, esc, ok, ov)
            bad += int((ok.view(-1, H, 128)[:T] != k).sum()) + int((ov.view(-1, H, 128)[:T] != v).sum())
        s0 = st.tolist()
        out["differing"] += bad
        out["values"] += 2 * T * H * 128 * len(names)
        out["prompts"].append({"name": name, "prompt_tokens": P, "tokens": T, "differing": bad})
        print(f"kv_real: {name}: {P} prompt tokens + {T - P} generated, {len(names)} layers, {2 * T * H * 128 * len(names):,} values: {bad} differ; status so far {s0[:4]}", flush=True)
    s0 = st.tolist()
    out.update({"overflow": s0[0], "page_high_water": s0[1], "arena_pages": s0[2], "entries": s0[3], "store_chunks": esc.chunks, "pool_taken": s0[4], "pool_high_water": s0[5], "pool_pages": s0[6], "raw_pages": s0[10], "void_pages": s0[11]})
    return out


def kv_agree_child(spec):
    """One vLLM instance, stock or kv lossless (spec: model, kv, tables, max_num_seqs, prompts: token ids, n, teacher: a continuation a prompt to force), eager: its greedy continuations
    of the prompts in one batch (n tokens, with the chosen tokens' logprobs), or, forced (the teacher's tokens sampled, whatever the model says: a logits processor), the model's own top-1 token
    and its logprob of the teacher's at every position: what a run's decode steps say of a continuation, whatever the run's own tokens."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.sample.logits_processor import LogitsProcessor, MoveDirectionality

    class Force(LogitsProcessor):
        """Request i's step-th token forced to its teacher's (SamplingParams.extra_args["teacher"]): its logits -inf but that token's."""

        def __init__(self, vllm_config, device, is_pin_memory):
            self.reqs = {}

        def is_argmax_invariant(self):
            return False

        def update_state(self, batch_update):
            if not batch_update:
                return
            for i in batch_update.removed:
                self.reqs.pop(i, None)
            for i, params, _, out in batch_update.added:
                t = (params.extra_args or {}).get("teacher")
                if t is not None:
                    self.reqs[i] = (t, out)
                else:
                    self.reqs.pop(i, None)
            for a, b, d in batch_update.moved:
                if d == MoveDirectionality.SWAP:
                    x, y = self.reqs.pop(a, None), self.reqs.pop(b, None)
                    if x is not None:
                        self.reqs[b] = x
                    if y is not None:
                        self.reqs[a] = y
                else:
                    x = self.reqs.pop(a, None)
                    self.reqs.pop(b, None)
                    if x is not None:
                        self.reqs[b] = x

        def apply(self, logits):
            for i, (t, out) in self.reqs.items():
                if len(out) < len(t):
                    tok = t[len(out)]
                    logits[i] = float("-inf")
                    logits[i, tok] = 0.0
            return logits

    kw = dict(model=spec["model"], dtype="bfloat16", enforce_eager=True, max_model_len=spec.get("max_model_len", 5120), gpu_memory_utilization=0.9, seed=0, max_num_seqs=spec["max_num_seqs"],
              max_num_batched_tokens=max(16384, spec.get("max_model_len", 5120)), enable_prefix_caching=False, enable_chunked_prefill=False)
    if spec.get("kv"):
        kw.update(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless"}})
    if spec.get("teacher"):
        kw["logits_processors"] = [Force]
    llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config
    out = {"spec": {k: v for k, v in spec.items() if k not in ("prompts", "teacher")}, "blocks": cfg.cache_config.num_gpu_blocks, "block_size": cfg.cache_config.block_size}
    prompts = [TokensPrompt(prompt_token_ids=p) for p in spec["prompts"]]
    if spec.get("teacher"):
        sps = [SamplingParams(temperature=0, max_tokens=len(t), logprobs=1, ignore_eos=True, extra_args={"teacher": t}) for t in spec["teacher"]]
    else:
        sps = SamplingParams(temperature=0, max_tokens=spec["n"], logprobs=0, ignore_eos=True)
    outs = llm.generate(prompts, sps, use_tqdm=False)
    out["tokens"] = [list(o.outputs[0].token_ids) for o in outs]
    out["logprobs"] = [[d[t].logprob for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)] for o in outs]
    if spec.get("teacher"):
        out["top1"] = [[next(tk for tk, lp in d.items() if lp.rank == 1) for d in o.outputs[0].logprobs] for o in outs]
    if spec.get("kv"):
        from glyd_gpu import vllm_kv

        out["kv_status"] = llm.apply_model(lambda m: vllm_kv.kv_status())[0]
    return out


def kv_verify_child(spec):
    """kv lossless with verify (every write checked against a bf16 shadow of each layer's cache, in vLLM, on the pages and slots vLLM gives) on real prompts of this repository's
    documents and some generated tokens each: the values compared (none differ: a difference stops the run), the most entries a page was given, the pools' use and the raw and void pages, and the tokens and their
    logprobs, for the parent to hold against stock's."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from glyd_gpu import vllm_kv

    tok = AutoTokenizer.from_pretrained(spec["model"])
    prompts = kv_prompts(tok, spec.get("max_tokens", 8192), spec.get("docs", 6), spec.get("study"))
    llm = LLM(model=spec["model"], dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.9, seed=0, max_model_len=spec.get("max_tokens", 8192) + spec["n_gen"] + 16, max_num_seqs=2,
              max_num_batched_tokens=spec.get("max_tokens", 8192) + spec["n_gen"] + 16, enable_prefix_caching=False, enable_chunked_prefill=False, num_gpu_blocks_override=spec["blocks"],
              quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless", "verify": True}})
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids in prompts], SamplingParams(temperature=0, max_tokens=spec["n_gen"], ignore_eos=True, logprobs=0), use_tqdm=False)
    st = llm.apply_model(lambda m: vllm_kv.kv_status())[0]
    dev = next(v for k, v in st.items() if k != "verified")
    return {"prompts": [{"name": n, "prompt_tokens": len(i)} for n, i in prompts], "generated": spec["n_gen"], "values_compared": st["verified"][0], "pages_checked": st["verified"][1], "overflow": dev[0],
            "page_high_water": dev[1], "arena_pages": dev[2], "entries": dev[3], "store_chunks": next(iter(st["store"].values())), "pool_taken": dev[4], "pool_high_water": dev[5], "pool_pages": dev[6], "raw_pages": dev[10], "void_pages": dev[11],
            "tokens": [list(o.outputs[0].token_ids) for o in outs]}


def kv_heavy_child(spec):
    """Prompts of the tokens whose pages the cache holds raw (the tables' sweep: those the tokenizer makes from text, those only a request of ids sends, the heaviest it holds), through vLLM (kv lossless,
    or stock: "plain"; GLYD_KV_TABLES names the tables) in the engine's own process. mode "serve": each prompt alone: its tokens, logprobs and prompt logprobs, how it finished. mode "exhaust": the 8 ordinary prompts alone, then `n` heavy
    prompts of `len` tokens at once with the 8 ordinary prompts among them (added to the engine together and stepped to the end, a request at a time: how each finished, or that a 400 refused it, and the seconds each took, the flood's wall time
    and prompt tokens a second), then the ordinary prompts alone again, then one heavy prompt alone ("prefix": prefix caching and chunked prefill on; "graphs": vLLM's default mode, compiled with CUDA graphs). Returns the rows, the
    status words, what the scheduler held, preempted and ended, and what the engine raised (nothing)."""
    import random
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.exceptions import VLLMClientError
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from glyd_gpu import kvcal, vllm_kv

    meta = kvcal.Tables(os.environ["GLYD_KV_TABLES"]).meta
    pl = meta["store"]
    text, ids, worst = pl["hard_text_ids"], pl["hard_ids_sample"], pl["worst_token"]
    tok = AutoTokenizer.from_pretrained(spec["model"])
    rng = random.Random(3)
    nat = kv_prompts(tok, 1024, 2)[-1][1]
    mix = text + ids or [worst]  # (the heavy tokens, drawn at random; where there are none, the one a page of which takes the most chunks of the escape pool, repeated)
    kw = dict(model=spec["model"], dtype="bfloat16", enforce_eager=not spec.get("graphs"), gpu_memory_utilization=0.9, seed=0, max_model_len=4096, max_num_seqs=spec.get("seqs", 8), max_num_batched_tokens=8192, enable_prefix_caching=bool(spec.get("prefix")), enable_chunked_prefill=bool(spec.get("prefix")))
    if spec.get("kv"):
        kw.update(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless", "verify": bool(spec.get("verify"))}})
    if spec.get("blocks"):
        kw["num_gpu_blocks_override"] = spec["blocks"]
    if spec.get("sync"):  # (vLLM's scheduler without async scheduling: a step's output is processed before the next is scheduled)
        kw["async_scheduling"] = False
    llm = LLM(**kw)
    if spec.get("no_front"):  # (the check of a prompt where vLLM validates its input off: what the scheduler does with a prompt that could never fit)
        from glyd_gpu import vllm_admit

        vllm_admit.check = lambda ip, ids: 0
    out = {"spec": spec, "blocks": llm.llm_engine.vllm_config.cache_config.num_gpu_blocks, "rows": [], "raised": None, "heavy": {"text": len(text), "ids": pl["hard_ids"], "worst": worst},
           "async_scheduling": llm.llm_engine.vllm_config.scheduler_config.async_scheduling}
    counter = iter(range(10**9))

    def drive(items, params_of):
        """Requests [(name, token ids)] added to the engine together and stepped until none is left (a prompt the API server's check would refuse is refused as it is there: a 400): {name: row}, the seconds it took. A row: how it finished,
        its tokens and logprobs, the seconds from the start to its end."""
        eng = llm.llm_engine
        t0 = time.perf_counter()
        by_id, rows = {}, {}
        for name, p in items:
            rid = f"{name}-{next(counter)}"
            try:
                eng.add_request(rid, TokensPrompt(prompt_token_ids=p), params_of(name))
                by_id[rid] = (name, p)
            except VLLMClientError as e:
                rows[name] = {"name": name, "prompt_tokens": len(p), "refused": str(e)[:240]}
        while eng.has_unfinished_requests():
            for o in eng.step():
                if o.finished:
                    name, p = by_id[o.request_id]
                    c = o.outputs[0]
                    rows[name] = {"name": name, "prompt_tokens": len(p), "finish": c.finish_reason, "stop_reason": c.stop_reason, "tokens": list(c.token_ids), "logprobs": [d[t].logprob for d, t in zip(c.logprobs or [], c.token_ids)],
                                  "latency": time.perf_counter() - t0}
        return rows, time.perf_counter() - t0

    def row(name, ids_, o):
        c = o.outputs[0]
        r = {"name": name, "prompt_tokens": len(ids_), "finish": c.finish_reason, "stop_reason": c.stop_reason, "tokens": list(c.token_ids), "logprobs": [d[t].logprob for d, t in zip(c.logprobs or [], c.token_ids)]}
        if o.prompt_logprobs:
            r["prompt_logprobs"] = [d[t].logprob for d, t in zip(o.prompt_logprobs[1:], ids_[1:])]
        return r

    try:
        if spec["mode"] == "serve":
            work = [(f"token {t} (from text) x 700", [t] * 700) for t in text[:4]] + [(f"token {t} (ids only) x 700", [t] * 700) for t in ids[:4]] + [(f"the heaviest token a page holds, {worst}, x 700", [worst] * 700)]
            if text:
                work.append((f"1,000 tokens drawn from the {len(text)} heavy tokens the tokenizer makes from text", rng.choices(text, k=1000)))
            if ids:
                work.append((f"1,000 tokens drawn from {len(ids)} heavy ids", rng.choices(ids, k=1000)))
            work.append(("a text prompt of 1,024 tokens", nat[:1024]))
            work.append(("a text prompt with two runs of 200 heavy tokens in it", nat[:300] + rng.choices(mix, k=200) + nat[300:600] + rng.choices(mix, k=200) + nat[600:]))
            for name, p in work:
                o = llm.generate([TokensPrompt(prompt_token_ids=p)], SamplingParams(temperature=0, max_tokens=spec.get("new", 24), ignore_eos=True, logprobs=0, prompt_logprobs=1), use_tqdm=False)[0]
                out["rows"].append(row(name, p, o))
                print(f"kv_heavy: {name}: {len(p)} tokens, {out['rows'][-1]['finish']}", flush=True)
        else:
            n, ln = spec.get("n", 40), spec.get("len", 1400)
            # (a model with no heavy token: the worst one repeated, the same prompt n times; with prefix caching on, behind a token of each prompt's own, so that no two prompts share a block)
            heavy_items = [(f"heavy{i}", rng.choices(mix, k=ln) if text + ids else ([1000 + i] + [worst] * (ln - 1) if spec.get("prefix") else [worst] * ln)) for i in range(n)]
            ord_items = [(f"ord{i}", tok.encode(q)) for i, q in enumerate(PROMPTS)]
            final = RequestOutputKind.FINAL_ONLY
            gen = spec.get("gen", 8)  # (tokens generated after a heavy prompt)
            params = lambda name: SamplingParams(temperature=0, max_tokens=gen, ignore_eos=True, output_kind=final) if name.startswith("heavy") else SamplingParams(temperature=0, max_tokens=TOKENS, logprobs=0, output_kind=final)
            alone, w_alone = drive(ord_items, params)  # (the ordinary prompts alone: the run without the flood)
            heavy_alone = {k: drive([(k, p_)], params)[0][k] for k, p_ in heavy_items} if spec.get("alone_first") else {}  # (each heavy prompt alone first: what a request that is preempted and run again must come to)
            fl, wall = drive(heavy_items + ord_items, params)
            hv = [fl[k] for k, _ in heavy_items]
            og = [fl[k] for k, _ in ord_items]
            bad = lambda r: "refused" in r or r.get("finish") == "error"
            ptok = n * ln + sum(len(p) for _, p in ord_items)
            out["flood"] = {"requests": n, "finished": sum(not bad(r) for r in hv), "failed": sum(r.get("finish") == "error" for r in hv), "refused": sum("refused" in r for r in hv), "ordinary_failed": sum(bad(r) for r in og),
                            "messages": sorted({str(r.get("stop_reason") or r.get("refused")) for r in hv + og if bad(r)}), "wall_seconds": round(wall, 2), "prompt_tokens": ptok, "prompt_tokens_per_second": round(ptok / wall, 1)}
            same = [x.get("tokens") == y.get("tokens") and x.get("logprobs") == y.get("logprobs") and "finish" in x for x, y in zip(og, [alone[k] for k, _ in ord_items])]
            out["during_same_as_alone"] = sum(same)
            if heavy_alone:
                out["flood"]["heavy_same_tokens_as_alone"] = sum(r.get("tokens") == heavy_alone[k].get("tokens") and "finish" in r for (k, _), r in zip(heavy_items, hv))
            out["latency"] = {"ordinary_alone": [round(alone[k]["latency"], 3) for k, _ in ord_items if "latency" in alone[k]], "ordinary_flood": [round(r["latency"], 3) for r in og if "latency" in r],
                              "heavy": sorted(round(r["latency"], 3) for r in hv if "latency" in r)}
            print(f"kv_heavy: {n} prompts of {ln} tokens at once with 8 ordinary ones: {out['flood']}", flush=True)
            if spec.get("sizing"):  # (a run that measures what the prompts take of the reserves: nothing more is run, so that what is taken at most is theirs)
                out["tokens"], out["logprobs"], out["post_failed"], out["heavy_after"] = [], [], 0, "n/a"
            else:
                post, _ = drive(ord_items, params)
                out["tokens"] = [post[k].get("tokens") for k, _ in ord_items]
                out["logprobs"] = [post[k].get("logprobs") for k, _ in ord_items]
                out["post_failed"] = sum(bad(post[k]) for k, _ in ord_items)
                ha, _ = drive([("heavy_after", [mix[0]] * 700)], params)  # (one prompt of a heavy token alone after the flood: the slabs of the requests that ended or were preempted are given back)
                out["heavy_after"] = ha["heavy_after"].get("finish") or "refused"
            try:  # (the engine's scheduler, in this process: what it held, preempted and ended)
                out["scheduler"] = dict(llm.llm_engine.engine_core.engine_core.scheduler.glyd_stats)
            except Exception as e:
                out["scheduler_error"] = f"{type(e).__name__}: {e}"
    except BaseException as e:  # (the engine in this process: what its step raised)
        import traceback

        out["raised"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[-3000:]
    if spec.get("kv"):
        try:
            st = llm.apply_model(lambda m: vllm_kv.kv_status())[0]
            dev = next(v for k, v in st.items() if k not in ("verified", "store", "second"))
            out["status"] = {"entries_with_no_place": dev[0], "most_entries_a_page_was_given": dev[1], "pool_taken_at_most": dev[5], "pages_with_a_chunk": dev[6], "slabs_taken_at_most": dev[9], "pages_made_raw": dev[10], "pages_made_void": dev[11],
                             "store_chunks": next(iter(st["store"].values())), "lumps": next(iter(st["second"].values())), "values_compared": st["verified"][0], "pages_checked": st["verified"][1]}
        except BaseException as e:
            out["status_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["failed_by_scheduler"] = vllm_kv.LOST[0]
    return out


def kv_ext_child(spec):
    """Prompts one request at a time through vLLM (stock, or kv lossless with bf16 weights; eager) with chunked prefill (spec["chunk"]: the step's tokens) and/or prefix caching: for each
    prompt its prompt_logprobs and 64 greedy tokens with their logprobs ("logprobs" mode: the prompt in chunks of `chunk` tokens, each reading the chunks before as context); or ("prefix"
    mode) the prompt run once (it writes its blocks) and then a prompt that shares its first `share` tokens and goes on with another text: its tokens and logprobs, and how many of its tokens
    were cached (the hit: its first `share` tokens' blocks are read as context)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(spec["model"])
    prompts = kv_prompts(tok, spec["max_tokens"], 6, spec.get("study"))
    chunk = spec.get("chunk")
    mml = spec["max_tokens"] + spec.get("tail", 0) + 128
    kw = dict(model=spec["model"], dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.9, seed=0, max_model_len=mml, max_num_seqs=spec.get("seqs", 1), enable_prefix_caching=bool(spec.get("prefix")),
              enable_chunked_prefill=bool(chunk), max_num_batched_tokens=chunk or mml)
    if spec.get("kv"):
        kw.update(quantization="glyd", additional_config={"glyd": {"fraction": 0, "kv": "lossless", "verify": bool(spec.get("verify")), **({"exact": True} if spec.get("exact") else {})}})
        if spec.get("blocks"):
            kw["num_gpu_blocks_override"] = spec["blocks"]
    if spec.get("nosync"):  # (every step's metadata is built with the host not allowed to wait for the GPU: torch raises at a call that synchronizes)
        import torch
        from glyd_gpu import vllm_kv

        orig = vllm_kv.GlydMetadataBuilder.build

        def build(self, *a, **k):
            torch.cuda.set_sync_debug_mode("error")
            try:
                return orig(self, *a, **k)
            finally:
                torch.cuda.set_sync_debug_mode("default")

        vllm_kv.GlydMetadataBuilder.build = build
    llm = LLM(**kw)
    out = {"spec": spec, "prompts": [], "kv_tokens": llm.llm_engine.vllm_config.cache_config.num_gpu_blocks * 16}
    greedy = dict(temperature=0, max_tokens=64, ignore_eos=True, logprobs=0)
    if spec["mode"] == "batch":  # several requests that share a long prefix in one batch (each reads the one cached copy of it): one warms the cache, then four go together
        base = prompts[1][1] if len(prompts[1][1]) >= 1700 else max((p for _, p in prompts), key=len)
        share, nreq = spec.get("share", 1600), spec.get("n", 4)
        # distinct: a prefix of its own for each request, each cached by a request of its own first: the rows of the batch share no page, so their contexts do not fit the staging cache together and are staged in groups
        bases = [p for _, p in prompts if len(p) >= share + 100][:nreq] if spec.get("distinct") else [base] * nreq
        nreq = len(bases)  # (a tokenizer that makes fewer tokens of the same documents gives fewer prompts of that length)
        assert nreq >= 2, f"{nreq} prompts of {share + 100} tokens"
        tails = [prompts[(i + 3) % len(prompts)][1][100 : 100 + 150 + 40 * i] for i in range(nreq)]
        for b in (bases if spec.get("distinct") else [base]):
            llm.generate([TokensPrompt(prompt_token_ids=b[:share])], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        outs = llm.generate([TokensPrompt(prompt_token_ids=b[:share] + t) for b, t in zip(bases, tails)], SamplingParams(**greedy), use_tqdm=False)
        for i, o in enumerate(outs):
            c = o.outputs[0]
            out["prompts"].append({"name": f"{'distinct' if spec.get('distinct') else 'shared'}-{i}", "prompt_tokens": len(o.prompt_token_ids), "shared": share, "cached_tokens": o.num_cached_tokens, "tokens": list(c.token_ids), "logprobs": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]})
            print(f"kv_ext: {'distinct' if spec.get('distinct') else 'shared'}-{i}: {len(o.prompt_token_ids)} tokens, {o.num_cached_tokens} cached", flush=True)
        return out
    for name, ids in prompts:
        if spec["mode"] == "logprobs":
            o = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(prompt_logprobs=1, **greedy), use_tqdm=False)[0]
            row = {"name": name, "prompt_tokens": len(ids), "prompt_logprobs": [d[t].logprob for d, t in zip(o.prompt_logprobs[1:], ids[1:])]}
        else:
            share = (int(len(ids) * 0.8) // 16) * 16
            other = prompts[(([n for n, _ in prompts].index(name)) + 1) % len(prompts)][1][: spec["tail"]]
            second = ids[:share] + other
            o1 = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)[0]
            o = llm.generate([TokensPrompt(prompt_token_ids=second)], SamplingParams(**greedy), use_tqdm=False)[0]
            row = {"name": name, "prompt_tokens": len(second), "shared": share, "cached_tokens": o.num_cached_tokens, "first_cached_tokens": o1.num_cached_tokens}  # (the first request's own hits: blocks an earlier request's steps wrote)
        c = o.outputs[0]
        row.update({"tokens": list(c.token_ids), "logprobs": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]})
        out["prompts"].append(row)
        print(f"kv_ext: {name}: {row['prompt_tokens']} tokens" + (f", {row['cached_tokens']} cached" if "cached_tokens" in row else ""), flush=True)
    return out


def first_token_spread(a, b):
    """The largest difference between the first generated token's logprob of the same prompts in two runs."""
    return max((abs(x["logprobs"][0] - y["logprobs"][0]) for x, y in zip(a["prompts"], b["prompts"])), default=0.0)


def flash_attention_3():
    """Whether vLLM's own prompt attention is FlashAttention 3 on this machine's GPU (Hopper), from nvidia-smi (this process does not take the GPU's memory)."""
    try:
        cap = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.split()[0]
    except Exception:
        return False
    return cap.split(".")[0] == "9"


def kv_ext_compare(a, b):
    """Rows of the same prompts from two runs: the prompt logprobs bit for bit, the first generated token's logprob bit for bit, the greedy tokens' prefix the same, the logprobs of the 64 bit for bit."""
    r = {"prompt_logprobs_bitwise": 0, "prompt_logprobs": 0, "first_token_bitwise": 0, "tokens_identical": 0, "all_bitwise": 0, "prefix": [], "n": len(a["prompts"]), "cached": []}
    for x, y in zip(a["prompts"], b["prompts"]):
        if "prompt_logprobs" in x:
            r["prompt_logprobs"] += 1
            r["prompt_logprobs_bitwise"] += x["prompt_logprobs"] == y["prompt_logprobs"]
        r["first_token_bitwise"] += x["tokens"][:1] == y["tokens"][:1] and x["logprobs"][:1] == y["logprobs"][:1]
        r["tokens_identical"] += x["tokens"] == y["tokens"]
        r["all_bitwise"] += x["tokens"] == y["tokens"] and x["logprobs"] == y["logprobs"]
        r["prefix"].append(same_prefix(x["tokens"], y["tokens"]))
        if "cached_tokens" in x:
            r["cached"].append((x["cached_tokens"], y["cached_tokens"]))
        if "first_cached_tokens" in x:  # (prefix mode: a prompt whose first request read blocks that an earlier request's decode steps wrote holds their keys and values as that side's own decode attention made them)
            hit = x["first_cached_tokens"] > 0 or y["first_cached_tokens"] > 0
            r.setdefault("fresh_n", 0), r.setdefault("fresh_first_bitwise", 0), r.setdefault("hit_rows", [])
            if hit:
                r["hit_rows"].append((x["name"], x["first_cached_tokens"], x["tokens"][:1] == y["tokens"][:1] and x["logprobs"][:1] == y["logprobs"][:1]))
            else:
                r["fresh_n"] += 1
                r["fresh_first_bitwise"] += x["tokens"][:1] == y["tokens"][:1] and x["logprobs"][:1] == y["logprobs"][:1]
    return r


def kv_main():
    """--kv: the lossless KV cache. Its kernels through the library (write, unpack, decode: the layout's every pattern, recycled pages, rewrites, an arena that
    overflows, attention against an fp32 reference); then vLLM's own keys and values on real prompts through the write kernel and back (kv_real): none differ;
    verify in vLLM on the same prompts; the KV cache's size against stock's at the flags the design logged its figure at; prompt_logprobs bit for bit stock's.
    --sanitize: the kernels' checks under compute-sanitizer (COMPUTE_SANITIZER names it). --ext-only: the checks of prefix caching, chunked prefill and exact alone. --no-ext: all but those of prefix caching, chunked prefill and exact. --no-kernels: all but the kernels' (run those with --kernels-only: a process that has run them holds 1.4 GiB of the GPU for the rest of its life). --study DIR: the KV study's ten prompts (614 to 8,192 tokens, 512 generated) in place of
    this repository's documents."""
    args = [a for a in sys.argv[1:] if a != "--kv"]
    out_dir = args[args.index("--out") + 1] if "--out" in args else "check_vllm_kv_results"
    kernels_only = "--kernels-only" in args or "--quick" in args or "--sanitize" in args
    ext_only = "--ext-only" in args  # (the extension rows' and exact's checks alone, no kernel checks)
    no_kernels = "--no-kernels" in args  # (every check but the kernels': this process then never takes the GPU's memory from the runs of vLLM it starts)
    no_ext = "--no-ext" in args  # (and not prefix caching, chunked prefill and exact's)
    model_checks = "--model-checks" in args  # (a model other than the one the design measured: its tables, values through the write kernel, verify, the cache's size, eager against stock's, prompts of the heavy tokens and floods of them; not the compiled and graph runs)
    tables_only = "--tables-only" in args  # (the tables alone: a model that does not fit this GPU's memory twice over, calibrated with KV_CAL_ENGINE's offload)
    flood_extras = "--no-flood-extras" not in args  # (the floods under batch invariance, with prefix caching, with the pools forced, and at the smallest pools: not with --no-flood-extras)
    study = args[args.index("--study") + 1] if "--study" in args else None
    models = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--out", "--study"))] or ["Qwen/Qwen3-8B"]
    os.makedirs(out_dir, exist_ok=True)
    report, failed = [], []

    def check(ok, what):
        report.append(("PASS " if ok else "FAIL ") + what)
        print(report[-1], flush=True)
        if not ok:
            failed.append(what)

    if "--sanitize" in args:  # the kernels' quick checks under compute-sanitizer (memcheck, initcheck, racecheck, synccheck), each a process of its own
        san = os.environ.get("COMPUTE_SANITIZER", "compute-sanitizer")
        for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
            r = subprocess.run([san, "--tool", tool, "--error-exitcode", "99", sys.executable, os.path.abspath(__file__), "--kv", "--quick", "--out", out_dir], capture_output=True, text=True, timeout=7200)
            summary = next((l for l in (r.stdout + r.stderr).splitlines() if "SUMMARY" in l), "no summary")  # (ERROR SUMMARY: 0 errors; racecheck's RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings))
            with open(os.path.join(out_dir, f"sanitizer-{tool}.txt"), "w") as f:
                f.write(r.stdout + r.stderr)
            clean = "ERROR SUMMARY: 0 errors" in summary or "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in summary
            check(r.returncode == 0 and clean, f"compute-sanitizer --tool {tool} over the write, unpack and decode kernels (the quick checks): {summary.strip()}")
        kernels_only = True
    elif not (ext_only or no_kernels):
        for ok, what in kv_kernel_checks("--quick" in args):
            report.append(("PASS " if ok else "FAIL ") + what)
            if not ok:
                failed.append(what)
        if not kernels_only:  # (the runs of vLLM that follow take the GPU's memory: this process gives back what its checks held)
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
    if not kernels_only:
        from glyd_gpu import kvcal

        def pool_note(meta):  # (the tables' worst single layer tensor, and the sweep's figures with the share of the cache the pools make)
            pt, pl = meta.get("per_tensor"), meta.get("store")
            if not pt or not pl:
                return ""
            from glyd_gpu import vllm_kv

            return (f" (the worst single layer's tensor: K {pt['k']['rate'] * 100:.4f}%, V {pt['v']['rate'] * 100:.4f}%); the sweep of {pl['swept']:,} tokens, {pl['seconds']:.0f} s: "
                    f"{pl['hard']} fill a page past the {pl['capacity']:,} entries it holds ({pl['hard_text']} made by the tokenizer from text, {pl['hard_ids']} only by token ids"
                    f"{', up to ' + format(pl['hard_worst_leaf_need'], ',') + ' entries, in up to ' + str(pl['second_layers']) + ' layers' if pl['hard'] else ''}); a layer-page of any takes up to {pl['worst_chunks_per_layer_leaf']} chunks "
                    f"(token {pl['worst_token']}; median {pl['median_chunks_per_layer_leaf']}): the escape pool {vllm_kv.store_percent(meta):.2f}% of the cache, the raw pool {vllm_kv.second_percent(meta):.2f}%")

        def cap_bar(meta):  # (the cache's size against stock's the model's page and pools make possible, less 2% for the staging cache, the decode workspace and the rounding of the allocator, which do not shrink with the model)
            from glyd_gpu import vllm_kv

            H = meta["heads"]
            return 0.98 * 8192 * H / (vllm_kv.leaf_bytes(H, vllm_kv.free_bay(H)) * (1 + (vllm_kv.store_percent(meta) + vllm_kv.second_percent(meta)) / 100))

        def pools_text(meta):
            from glyd_gpu import vllm_kv

            return f"the escape pool ({vllm_kv.store_percent(meta):g}% of the cache), the raw pool ({vllm_kv.second_percent(meta):g}%)"

        for model in models:
            tag = model.split("/")[-1]
            tables = os.path.join(out_dir, f"{tag}-tables.npz")
            if not os.path.exists(tables):  # (made as a first start makes them: a vLLM of its own, on the bundled text; KV_CAL_ENGINE: its LLM arguments, JSON, KV_CAL_POSITIONS: 16,384)
                wrap = os.environ["KV_WRAP"].split() if os.environ.get("KV_WRAP") else []  # (the calibration's vLLM holds the GPU as every run does: under the box's lock too)
                r = subprocess.run([*wrap, sys.executable, "-m", "glyd_gpu", "kvcal", model, "--out", tables, "--positions", os.environ.get("KV_CAL_POSITIONS", "16384"), *(["--engine", os.environ["KV_CAL_ENGINE"]] if os.environ.get("KV_CAL_ENGINE") else [])],
                                   env=dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", GLYD_KV="off"), timeout=3600)  # (exit 1: the tables written and the model refused: not a failure of the run)
                if not os.path.exists(tables):
                    raise RuntimeError(f"kvcal for {model} exited {r.returncode} and wrote no tables")
            t = kvcal.Tables(tables)
            refused = bool(t.meta["refused"])  # (the values fall outside the windows too often: the plugin refuses the model)
            check(True, f"{tag}: base tables from the bundled text ({t.meta['tokens']:,} tokens, {t.meta['seconds']:.0f} s): worst kind of text left out K {t.meta['held_out_rate']['k'] * 100:.4f}%, V {t.meta['held_out_rate']['v'] * 100:.4f}% of the values outside the windows{pool_note(t.meta)}; "
                  f"a page needed at most {t.meta['leaves']['bay_need_max']} entries past its heads' inline ones; {'REFUSED: ' + kvcal.refusal(t.meta)[:600] if refused else 'within the line'}")
            if tables_only:
                check(bool(t.meta.get("store")), f"{tag}: the tables carry the sweep (calibrated with the engine arguments {os.environ.get('KV_CAL_ENGINE') or 'by default'}: a model whose weights do not fit this GPU twice over)")
                continue
            if refused:
                continue
            pl = t.meta.get("store") or {}

            def step(what, spec, name, env=None):  # (one run's failure is a FAIL of its own, not the end of the checks)
                if model_checks and any(x in name for x in ("-default", "-det", "-fullgraph", "-inv-")):
                    return None
                try:
                    res = run(spec, out_dir, name, env)
                except Exception as e:
                    check(False, f"{tag}: {what}: {type(e).__name__}: {e}")
                    return None
                if res.get("error"):
                    check(False, f"{tag}: {what}: {res['error'][:400]}")
                    return None
                return res

            kenv = {"GLYD_KV_TABLES": tables}
            if not ext_only:
                r = step("vLLM's keys and values of real prompts through the write kernel and back", {"kv_real": True, "model": model, "tables": tables, "n_gen": 512 if study else 130, "study": study}, f"{tag}-kv-real")
                if r:
                    check(r["differing"] == 0, f"{tag}: vLLM's keys and values of {len(r['prompts'])} real prompts ({sum(p['tokens'] for p in r['prompts']):,} tokens, {r['values']:,} values over {r['layers']} layers, a prompt in one step then a token a step, into pages of a previous owner's bytes): {r['differing']} differ")
                    check(r["void_pages"] == 0, f"{tag}: the exception entries on those prompts: at most {r['page_high_water']} a page (arenas of {r['bay']}), {r['entries']:,} entries, {r['pool_pages']:,} pages took chunks of the escape pool ({r['pool_high_water']} of {r['store_chunks']:,} taken at most), {r['raw_pages']} pages made raw, {r['void_pages']} lost")
                v = step("verify in vLLM", {"kv_verify": True, "model": model, "n_gen": 512, "blocks": 800, "study": study}, f"{tag}-kv-verify", kenv)
                if v:
                    check(v["values_compared"] > 0 and v["void_pages"] == 0, f"{tag}: verify in vLLM, {len(v['prompts'])} prompts of {min(p['prompt_tokens'] for p in v['prompts'])} to {max(p['prompt_tokens'] for p in v['prompts']):,} tokens and {v['generated']} generated each, every write checked against a bf16 shadow of each layer's cache: {v['values_compared']:,} values over {v['pages_checked']:,} page checks, none differ (a difference stops the run); the most entries a page was given {v['page_high_water']}, {v['pool_pages']:,} pages took chunks of the escape pool ({v['pool_high_water']} of {v['store_chunks']:,} taken at most), {v['raw_pages']} pages made raw, {v['void_pages']} lost")
                s1 = step("stock's KV cache at the design's flags", {"model": model, "plain": True, "eager": True, "r1": True}, f"{tag}-stock-r1")
                k1 = step("kv lossless's KV cache at the design's flags", {"model": model, "kv": True, "eager": True, "r1": True}, f"{tag}-kv-r1", kenv)
                st = step("stock, eager (prompts, a long continuation fed back)", {"model": model, "plain": True, "eager": True, "make_long": True}, f"{tag}-stock-eager")
                ks = step("kv lossless, eager", {"model": model, "kv": True, "eager": True, "long_ids": st["long_ids"]}, f"{tag}-kv-eager", kenv) if st else None
                if s1 and k1:
                    n1, m1 = k1["blocks"] * k1["block_size"], s1["blocks"] * s1["block_size"]
                    small = f" (max-model-len 4096, 0.85: {ks['blocks'] * ks['block_size']:,} against stock's {st['blocks'] * st['block_size']:,}, {ks['blocks'] * ks['block_size'] / (st['blocks'] * st['block_size']):.4f}x)" if st and ks else ""
                    check(n1 >= cap_bar(t.meta) * m1, f"{tag}: KV cache {n1:,} tokens against stock's {m1:,} ({n1 / m1:.4f}x, {16 / (n1 / m1):.2f} bits a value; at least {cap_bar(t.meta):.3f}x), eager, max-model-len 8704, 8704 tokens a step, 0.9, bf16 weights, {pools_text(t.meta)} and the staging cache in it;{small}")
                if st and ks:
                    c = compare(ks, st)
                    check(c["long_bit_identical"], f"{tag}: prompt_logprobs of a {len(st['long_ids']):,}-token prompt (stock's own continuation fed back: prefill through the staging cache and vLLM's FlashAttention) bit for bit stock's")
                    check(all(a[0] == b[0] for a, b in zip(ks["logprobs"], st["logprobs"])), f"{tag}: the 8 prompts' first generated token's logprob bit for bit stock's; their {TOKENS} greedy tokens the same as stock's from the start for {c['prefix']} tokens, mean |logprob difference| on the shared prefixes {c['mean_abs_dlogprob']:.4f}")
                # vLLM's default mode (torch.compile, piecewise CUDA graphs around the attention layers, which run eager). Inductor picks some kernels' variants by timing them, so a compile on an empty cache lands on one of a few
                # variants: stock's own compiles gave two sets of first-token logprobs for the 8 prompts, up to 0.053 apart, the plugin's gave three, and each of stock's two was one of the plugin's, bit for bit.
                # So the default mode is checked for its tokens and its logprobs within that spread, and compiled bits against stock's in inductor's deterministic mode, both compiled in it.
                sd = step("stock, vLLM's default mode", {"model": model, "plain": True}, f"{tag}-stock-default")
                kd = step("kv lossless, vLLM's default mode", {"model": model, "kv": True}, f"{tag}-kv-default", kenv)
                if sd and kd:
                    cd = compare(kd, sd)
                    first = max(abs(a[0] - b[0]) for a, b in zip(kd["logprobs"], sd["logprobs"]))
                    flips = [i for i, (a, b) in enumerate(zip(kd["tokens"], sd["tokens"])) if a[0] != b[0]]  # (a near tie that the compile's variant decides the other way: stock's own compiles do it too)
                    check(all(abs(kd["logprobs"][i][0] - sd["logprobs"][i][0]) < 0.15 for i in flips) and cd["mean_abs_dlogprob"] < 0.02 and first < 0.15,
                          f"{tag}: vLLM's default mode (torch.compile, piecewise CUDA graphs around the attention layers, which run eager): the 8 prompts' first generated token stock's{' but for prompts ' + str(flips) + ' (a near tie)' if flips else ''}, its logprob within {first:.3f} of stock's "
                          f"(inductor picks kernels by timing them: compiles of stock's own differ from one another as far); their {TOKENS} greedy tokens the same as stock's from the start for {cd['prefix']} tokens, mean |logprob difference| on the shared prefixes {cd['mean_abs_dlogprob']:.4f} (eager's 0.0100)")
                sdt = step("stock, compiled in inductor's deterministic mode", {"model": model, "plain": True, "make_long": True, "compilation_config": DETERMINISTIC}, f"{tag}-stock-det")
                kdt = step("kv lossless, compiled in inductor's deterministic mode", {"model": model, "kv": True, "long_ids": sdt["long_ids"], "compilation_config": DETERMINISTIC}, f"{tag}-kv-det", kenv) if sdt else None
                if kd:
                    check(kd.get("cudagraph_mode") == "FULL_AND_PIECEWISE" and kd.get("graphs_captured", 0) > 0, f"{tag}: vLLM's default mode runs kv lossless with FULL decode graphs and piecewise graphs for the rest (cudagraph mode {kd.get('cudagraph_mode')}, {kd.get('graphs_captured')} graphs captured)")
                kg = step("kv lossless, FULL decode graphs (no torch.compile)", {"model": model, "kv": True, "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}}, f"{tag}-kv-fullgraph", kenv)
                if kg and ks:
                    same = [x == y and lx == ly for x, y, lx, ly in zip(kg["tokens"], ks["tokens"], kg["logprobs"], ks["logprobs"])]
                    check(kg.get("cudagraph_mode") == "FULL_DECODE_ONLY" and kg.get("graphs_captured", 0) > 0 and all(same),
                          f"{tag}: kv lossless's FULL decode graphs ({kg.get('graphs_captured')} captured, cudagraph mode {kg.get('cudagraph_mode')}) replayed: the 8 prompts' {TOKENS} greedy tokens and their logprobs bit for bit those of eager ({sum(same)} of {len(same)} prompts)")
                # batch invariance (VLLM_BATCH_INVARIANT=1, bf16 weights): a sequence's outputs the same one prompt at a time and in a batch of 8, eager and in FULL decode graphs (no torch.compile); vLLM's own
                # bf16 under the variable is the control: it is invariant in those two here, and not in its compiled default mode (0 of 8: an inductor kernel's variants follow the batch's shape), so that mode is not asked
                ienv = {**kenv, "VLLM_BATCH_INVARIANT": "1"}
                for nm, what, extra in (("eager", "eager", {"eager": True}), ("full", "FULL decode graphs, no torch.compile", {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}})):
                    ib = step(f"kv lossless, VLLM_BATCH_INVARIANT=1, {what}, 8 prompts in one batch", {"model": model, "kv": True, **extra}, f"{tag}-kv-inv-{nm}-batch", ienv)
                    i1 = step(f"kv lossless, VLLM_BATCH_INVARIANT=1, {what}, one prompt at a time", {"model": model, "kv": True, "one_by_one": True, **extra}, f"{tag}-kv-inv-{nm}-one", ienv)
                    sib = step(f"stock, VLLM_BATCH_INVARIANT=1, {what}, 8 prompts in one batch", {"model": model, "plain": True, **extra}, f"{tag}-stock-inv-{nm}-batch", {"VLLM_BATCH_INVARIANT": "1"})
                    si1 = step(f"stock, VLLM_BATCH_INVARIANT=1, {what}, one prompt at a time", {"model": model, "plain": True, "one_by_one": True, **extra}, f"{tag}-stock-inv-{nm}-one", {"VLLM_BATCH_INVARIANT": "1"})
                    if ib and i1:
                        same = [x == y and lx == ly for x, y, lx, ly in zip(ib["tokens"], i1["tokens"], ib["logprobs"], i1["logprobs"])]
                        ctl = sum(x == y and lx == ly for x, y, lx, ly in zip(sib["tokens"], si1["tokens"], sib["logprobs"], si1["logprobs"])) if sib and si1 else "?"
                        check(all(same), f"{tag}: under VLLM_BATCH_INVARIANT=1 (bf16 weights, kv lossless, {what}) each of 8 prompts' {TOKENS} greedy tokens and their logprobs the same bits alone and in a batch of 8 "
                              f"({sum(same)} of {len(same)} prompts; vLLM's own bf16 under the same variable and mode, the control: {ctl} of 8)")
                if sdt and kdt:
                    c = compare(kdt, sdt)
                    n1 = sum(a[0] == b[0] for a, b in zip(kdt["logprobs"], sdt["logprobs"]))
                    check(n1 == len(PROMPTS) and c["long_bit_identical"], f"{tag}: compiled in inductor's deterministic mode (both): the 8 prompts' first generated token's logprob bit for bit stock's ({n1} of {len(PROMPTS)}), and the logprobs of a {len(sdt['long_ids']):,}-token prompt (stock's own continuation fed back) "
                          f"bit for bit stock's ({c['long_bit_identical']}); their {TOKENS} greedy tokens the same from the start for {c['prefix']} tokens, mean |logprob difference| on the shared prefixes {c['mean_abs_dlogprob']:.4f}")
            if not no_ext:
                # prefix caching and chunked prefill (the extension rows: a context's pages decoded into the staging cache, vLLM's FlashAttention over it), one request at a time against vLLM's own
                for chunk, prefix, mode, tail, what in ((512, False, "logprobs", 0, "chunked prefill, chunks of 512 tokens"), (2048, False, "logprobs", 0, "chunked prefill, chunks of 2,048 tokens"),
                                                        (None, True, "prefix", 300, "prefix hits (a prompt after another that shares its first 80%)"), (512, True, "prefix", 1300, "prefix hits and chunks of 512 tokens (a shared 80%, then 1,300 new tokens)")):
                    sp = {"kv_ext": True, "model": model, "mode": mode, "chunk": chunk, "prefix": prefix, "tail": tail, "max_tokens": 4096, "study": study}
                    tg = f"{tag}-ext-{mode}-c{chunk}-p{int(prefix)}"
                    es, ek = step(f"stock, {what}", sp, f"{tg}-stock"), step(f"kv lossless, {what}", {**sp, "kv": True}, f"{tg}-kv", kenv)
                    if es and ek:
                        r = kv_ext_compare(ek, es)
                        # (a prompt whose first request read blocks that an earlier request's decode steps wrote is not counted: those keys and values are kv's own decode attention's, which rounds as vLLM's does not;
                        # the same scenario with `exact` below must be bit for bit in every prompt)
                        first_ok = r["fresh_first_bitwise"] == r["fresh_n"] if prefix else r["first_token_bitwise"] == r["n"]
                        check(r["prompt_logprobs_bitwise"] == r["prompt_logprobs"] and first_ok and (not prefix or (all(c[0] == c[1] for c in r["cached"]) and any(c[0] > 0 for c in r["cached"]))),
                              f"{tag}: {what}, {r['n']} prompts of up to {max(p['prompt_tokens'] for p in ek['prompts']):,} tokens one request at a time, eager: the prompt logprobs bit for bit stock's ({r['prompt_logprobs_bitwise']} of {r['prompt_logprobs']} prompts), the first generated "
                              f"token's logprob bit for bit ({r['first_token_bitwise']} of {r['n']}"
                              + (f"; {r['fresh_first_bitwise']} of the {r['fresh_n']} that read no blocks an earlier request wrote; the others, {[(n, c, 'bit for bit' if b else 'not') for n, c, b in r['hit_rows']]}" if prefix else "")
                              + f"), the 64 greedy tokens the same as stock's in {r['tokens_identical']} of {r['n']} (from the start for {r['prefix']}), their logprobs bit for bit in {r['all_bitwise']}"
                              + (f"; tokens cached (kv, stock): {r['cached']}" if prefix else "") + f"; KV cache {ek['kv_tokens']:,} tokens against stock's {es['kv_tokens']:,}")
                        if prefix and chunk is None:  # (the same scenario through vLLM's own FlashAttention for every row: the decode steps' keys and values are stock's too)
                            ex = step("kv lossless exact, the same prefix hits", {**sp, "kv": True, "exact": True}, f"{tg}-kv-exact", kenv)
                            if ex:
                                rx = kv_ext_compare(ex, es)
                                check(rx["first_token_bitwise"] == rx["n"] and rx["tokens_identical"] == rx["n"],
                                      f"{tag}: {what}, `exact` (every row's attention through vLLM's own FlashAttention): the first token's logprob bit for bit stock's ({rx['first_token_bitwise']} of {rx['n']}) and the 64 greedy tokens and logprobs ({rx['all_bitwise']} of {rx['n']}), "
                                      f"tokens cached (kv, stock) {rx['cached']}")
                sp = {"kv_ext": True, "model": model, "mode": "batch", "chunk": 2048, "prefix": True, "tail": 400, "seqs": 8, "max_tokens": 4096, "study": study}
                bs, bk = step("stock, four requests sharing a prefix in one batch", sp, f"{tag}-ext-batch-stock"), step("kv lossless, four requests sharing a prefix in one batch", {**sp, "kv": True}, f"{tag}-ext-batch-kv", kenv)
                if bs and bk:
                    r = kv_ext_compare(bk, bs)
                    check(r["first_token_bitwise"] == r["n"] and (all(c[0] == c[1] for c in r["cached"]) and any(c[0] > 0 for c in r["cached"])),
                          f"{tag}: four requests that share their first 1,600 tokens, in one batch (one copy of the prefix in the staging cache, four rows reading it), eager: the first generated token's logprob bit for bit stock's ({r['first_token_bitwise']} of {r['n']}), "
                          f"tokens cached (kv, stock) {r['cached']}, the 64 greedy tokens the same in {r['tokens_identical']} of {r['n']} (from the start for {r['prefix']})")
                for what, extra in (("requests (up to six) with a cached prefix of 2,048 tokens each of their own (no page shared: their contexts do not fit the staging cache together, so they are staged in groups that the step's first staged layer makes)", {"distinct": True, "n": 6, "share": 2048}),
                                    ("eight requests that share a cached prefix of 1,600 tokens (the host's bound is four groups of two rows, and the plan has one, the rows sharing the prefix's blocks: the first staged layer reads that one has rows and launches only it)", {"n": 8})):
                    sd = {**sp, **extra}
                    ds, dk = step(f"stock, {what[:40]}", sd, f"{tag}-ext-n{extra['n']}-stock"), step(f"kv lossless, {what[:40]}", {**sd, "kv": True}, f"{tag}-ext-n{extra['n']}-kv", kenv)
                    if ds and dk and len(ds["prompts"]) == len(dk["prompts"]) >= 2:
                        r = kv_ext_compare(dk, ds)
                        cached_ok = all(c[0] == c[1] for c in r["cached"]) and any(c[0] > 0 for c in r["cached"])
                        ok, control = r["first_token_bitwise"] == r["n"] and cached_ok, ""
                        if not ok and cached_ok and extra.get("distinct") and flash_attention_3():
                            # FlashAttention 3 schedules a call by its batch (its splits), so a row's bits depend on the rows beside it: stock's four rows in one call against kv's groups of two rows. The control is stock itself with its batch
                            # changed (two requests a step): where that moves stock's own first tokens, kv's are held to the same spread.
                            cs = step("stock, the same requests two a step (the control: vLLM's FlashAttention with its batch changed)", {**sd, "seqs": 2}, f"{tag}-ext-n{extra['n']}-stock2")
                            if cs and len(cs["prompts"]) == len(ds["prompts"]):
                                rc, spread, ours = kv_ext_compare(cs, ds), first_token_spread(cs, ds), first_token_spread(dk, ds)
                                ok = rc["first_token_bitwise"] < rc["n"] and ours <= 2 * spread
                                control = (f"; FlashAttention 3 schedules a call by its batch: stock's own first token's logprob is bit for bit its four-row batch's in {rc['first_token_bitwise']} of {rc['n']} with two requests a step (largest difference {spread:.3g}), "
                                           f"kv's largest {ours:.3g} ({'within twice the control: not held to the bits' if ok else 'beyond the control'})")
                            # and the same rows with a staging cache that holds their contexts together: one group, one FlashAttention call as stock's, so its bits are expected (the groups are what differ)
                            d1 = step("kv lossless, the same requests with a staging cache that holds their contexts together (one group)", {**sd, "kv": True}, f"{tag}-ext-n{extra['n']}-kv-onegroup", {**kenv, "GLYD_KV_STAGING_BLOCKS": "1500"})
                            if d1 and len(d1["prompts"]) == len(ds["prompts"]):
                                r1 = kv_ext_compare(d1, ds)
                                check(r1["first_token_bitwise"] == r1["n"] and all(c[0] == c[1] for c in r1["cached"]),
                                      f"{tag}: the same requests with a staging cache of 1,500 blocks (their contexts together: one group, one FlashAttention call as stock's), eager: the first generated token's logprob bit for bit stock's ({r1['first_token_bitwise']} of {r1['n']}), "
                                      f"the 64 greedy tokens the same in {r1['tokens_identical']} of {r1['n']} (from the start for {r1['prefix']})")
                        check(ok,
                              f"{tag}: {what}, in one batch, eager: the first generated token's logprob bit for bit stock's ({r['first_token_bitwise']} of {r['n']}), tokens cached (kv, stock) {r['cached']}, the 64 greedy tokens the same in {r['tokens_identical']} of {r['n']} "
                              f"(from the start for {r['prefix']}){control}")
                nk = step("kv lossless, the same four requests with every step's metadata built under torch's synchronization errors", {**sp, "kv": True, "nosync": True}, f"{tag}-ext-nosync", kenv)
                if nk and bk:
                    check(all(x["tokens"] == y["tokens"] and x["logprobs"] == y["logprobs"] for x, y in zip(nk["prompts"], bk["prompts"])),
                          f"{tag}: the steps that stage rows with shared pages (chunks of 2,048, a prefix of 1,600 tokens cached) build their metadata with the host not waiting for the GPU: torch's synchronization errors on for every build raised none, and the outputs are the run without them's")
                xs, xk = step("stock, eager, 8 prompts in one batch (exact's reference)", {"model": model, "plain": True, "eager": True, "max_num_seqs": 8}, f"{tag}-exact-stock"), step("kv lossless exact, eager, the same", {"model": model, "kv": True, "exact": True, "eager": True, "max_num_seqs": 8}, f"{tag}-exact-kv", kenv)
                if xs and xk:
                    cx = compare(xk, xs)
                    check(cx["bit_identical"] == len(PROMPTS), f"{tag}: kv lossless exact (every row's attention through vLLM's own FlashAttention on its decoded pages): the 8 prompts' {TOKENS} greedy tokens and their logprobs bit for bit stock's, {cx['bit_identical']} of {len(PROMPTS)} (tokens the same from the start for {cx['prefix']})")
            if not ext_only:
                # the tokens whose pages the cache holds raw, a flood of them, and pools too small: through the engine, a request is served or failed with the retryable error, and the engine never stops
                heavy = {"kv_heavy": True, "mode": "serve", "model": model}
                hs = step("stock, prompts of the heavy tokens", {**heavy, "plain": True}, f"{tag}-heavy-stock", kenv)
                hv = step("kv lossless with verify, prompts of the heavy tokens", {**heavy, "kv": True, "verify": True, "blocks": 800}, f"{tag}-heavy-verify", kenv)
                hk = step("kv lossless, prompts of the heavy tokens", {**heavy, "kv": True}, f"{tag}-heavy-kv", kenv)
                for what, h in (("verify", hv), ("default", hk)):
                    if h:
                        s_ = h.get("status", {})
                        check(h["raised"] is None and all(r["finish"] != "error" for r in h["rows"]) and s_.get("pages_made_void", 1) == 0 and (what != "verify" or s_.get("values_compared", 0) > 0) and (not pl.get("hard") or s_.get("pages_made_raw", 0) > 0),
                              f"{tag}: {len(h['rows'])} prompts of the model's heavy tokens ({pl.get('hard_text', 0)} made by the tokenizer from text, {pl.get('hard_ids', 0)} only by token ids; each alone x 700, drawn at random x 1,000, runs of 200 among text) "
                              f"{'with every write compared against a bf16 shadow of each layer: ' + format(s_.get('values_compared', 0), ',') + ' values over ' + format(s_.get('pages_checked', 0), ',') + ' page checks, none differ; ' if what == 'verify' else 'at the default pools: '}"
                              f"{s_.get('pages_made_raw', 0):,} pages made raw (the raw pool {s_.get('slabs_taken_at_most', 0):,} of {s_.get('lumps', 0):,} slabs taken at most), {s_.get('pages_made_void', 0)} lost, {h.get('failed_by_scheduler', 0)} requests failed, the engine raised {h['raised']}")
                if hs and hk:
                    same = [x.get("prompt_logprobs") == y.get("prompt_logprobs") for x, y in zip(hk["rows"], hs["rows"])]
                    first = [x["tokens"][:1] == y["tokens"][:1] and x["logprobs"][:1] == y["logprobs"][:1] for x, y in zip(hk["rows"], hs["rows"])]
                    pre = [same_prefix(x["tokens"], y["tokens"]) for x, y in zip(hk["rows"], hs["rows"])]
                    check(all(same) and all(first), f"{tag}: those prompts' prompt logprobs bit for bit stock's ({sum(same)} of {len(same)}), their first generated token and its logprob bit for bit ({sum(first)} of {len(first)}); the 24 greedy tokens the same as stock's from the start for {pre} tokens")
                fl = {"kv_heavy": True, "mode": "exhaust", "model": model, "kv": True}
                sf = {"kv_heavy": True, "mode": "exhaust", "model": model, "plain": True}
                big = {"seqs": 48, "n": 40, "len": 1400}  # (40 prompts of 1,400 heavy tokens at once with the 8 ordinary ones)
                f1 = step("kv lossless, one prompt of 900 heavy tokens (the pools it takes)", {**fl, "seqs": 8, "n": 1, "len": 900, "sizing": True}, f"{tag}-heavy-one", kenv)
                fd = step("kv lossless, a flood of prompts of heavy tokens at the default pools", {**fl, **big}, f"{tag}-heavy-flood", kenv)
                fst = step("stock, the same flood", {**sf, **big}, f"{tag}-heavy-flood-stock", kenv)
                if not flood_extras:
                    fi = fsi = fp = fx = fxs = fxp = fxi = fs = fsb = None
                else:
                    fi = step("kv lossless, the same flood under VLLM_BATCH_INVARIANT=1", {**fl, **big}, f"{tag}-heavy-flood-inv", {**kenv, "VLLM_BATCH_INVARIANT": "1"})
                    fsi = step("stock, the same flood under VLLM_BATCH_INVARIANT=1", {**sf, **big}, f"{tag}-heavy-flood-stock-inv", {**kenv, "VLLM_BATCH_INVARIANT": "1"})
                    fp = step("kv lossless, the same flood with prefix caching and chunked prefill on", {**fl, **big, "prefix": True}, f"{tag}-heavy-flood-prefix", kenv)
                    st1 = (f1 or {}).get("status", {})
                    pools = {"GLYD_KV_RESERVE_1_UNITS": str(max(1024, int(1.25 * st1.get("pool_taken_at_most", 0)))), "GLYD_KV_RESERVE_2_UNITS": str(max(256, int(1.25 * st1.get("slabs_taken_at_most", 0))))}
                    fx = step("kv lossless with verify, 12 prompts of 900 heavy tokens at once, admission off, pools that hold one of them and not all", {**fl, "verify": True, "blocks": 800, "seqs": 8, "n": 12, "len": 900, "gen": 24, "alone_first": True}, f"{tag}-heavy-flood-forced",
                              {**kenv, "GLYD_KV_ADMIT": "off", **pools})
                    fxs = step("kv lossless with verify, the same with async scheduling off", {**fl, "verify": True, "blocks": 800, "seqs": 8, "n": 12, "len": 900, "sync": True, "gen": 24, "alone_first": True}, f"{tag}-heavy-flood-forced-sync", {**kenv, "GLYD_KV_ADMIT": "off", **pools})
                    pools_p = pools if pl.get("hard") else {"GLYD_KV_RESERVE_1_UNITS": "1024", "GLYD_KV_RESERVE_2_UNITS": "256"}  # (prefix caching: a model with no heavy token has prompts of its own behind a token each, which take less)
                    fxp = step("kv lossless with verify, the same with prefix caching and chunked prefill on", {**fl, "verify": True, "blocks": 800, "seqs": 8, "n": 12, "len": 900, "prefix": True, "gen": 24, "alone_first": True}, f"{tag}-heavy-flood-forced-prefix", {**kenv, "GLYD_KV_ADMIT": "off", **pools_p})
                    fxi = step("kv lossless with verify, the same under VLLM_BATCH_INVARIANT=1", {**fl, "verify": True, "blocks": 800, "seqs": 8, "n": 12, "len": 900, "gen": 24, "alone_first": True}, f"{tag}-heavy-flood-forced-inv", {**kenv, "GLYD_KV_ADMIT": "off", **pools, "VLLM_BATCH_INVARIANT": "1"})
                    fs = step("kv lossless with verify, a flood at the smallest pools (128 chunks, 16 slabs)", {**fl, "verify": True, "blocks": 800, "seqs": 8, "n": 12, "len": 900}, f"{tag}-heavy-flood-small", {**kenv, "GLYD_KV_RESERVE_1_UNITS": "128", "GLYD_KV_RESERVE_2_UNITS": "16"})
                    fsb = step("kv lossless with verify, the same without the check where vLLM validates the input: the scheduler's own", {**fl, "verify": True, "blocks": 800, "seqs": 8, "n": 12, "len": 900, "no_front": True}, f"{tag}-heavy-flood-small-sched",
                               {**kenv, "GLYD_KV_RESERVE_1_UNITS": "128", "GLYD_KV_RESERVE_2_UNITS": "16"})

                def flood_line(h):  # (what a flood run did, in a line)
                    f_, s_, sc_ = h.get("flood", {}), h.get("status", {}), h.get("scheduler", {})
                    lat = h.get("latency", {})
                    med = lambda xs: f"{sorted(xs)[len(xs) // 2]:.2f}" if xs else "?"
                    return (f"{f_.get('finished')} of the {f_.get('requests')} served, {f_.get('failed')} ended with an error ({f_.get('messages')}), {f_.get('refused')} refused with a 400, {f_.get('ordinary_failed')} of the 8 ordinary ones failed, "
                            f"in {f_.get('wall_seconds')} s ({f_.get('prompt_tokens_per_second', 0):,.0f} prompt tokens/s); the ordinary prompts' latency alone {med(lat.get('ordinary_alone', []))} s (median), in the flood {med(lat.get('ordinary_flood', []))} s; "
                            f"held {sc_.get('held', '-')}, preempted {sc_.get('preempted', '-')} times, ended for never fitting {sc_.get('refused', '-')}, alone {sc_.get('failed', '-')}; {s_.get('pages_made_raw', 0):,} pages made raw, {s_.get('pages_made_void', 0):,} lost "
                            f"({s_.get('slabs_taken_at_most', 0):,} of {s_.get('lumps', 0):,} slabs and {s_.get('pool_taken_at_most', 0):,} of {s_.get('store_chunks', 0):,} chunks taken at most)"
                            f"{'; ' + format(s_.get('values_compared', 0), ',') + ' values compared, none differ' if s_.get('values_compared') else ''}")

                for what, h, strict in (("40 prompts of 1,400 heavy tokens at once with 8 ordinary ones, the default pools", fd, True),
                                        ("40 prompts of 1,400 heavy tokens at once with 8 ordinary ones, the default pools, prefix caching and chunked prefill on", fp, False)):
                    if h:
                        f_ = h.get("flood", {})
                        same = ks is not None and (h.get("tokens") == ks["tokens"] and h.get("logprobs") == ks["logprobs"] if strict else True)  # (with prefix hits the ordinary prompts' first blocks are the keys and values of the flood's batch, as vLLM's own prefix hits are: not compared bit for bit)
                        check(h["raised"] is None and f_.get("finished") == f_.get("requests") and f_.get("failed") == 0 and f_.get("refused") == 0 and f_.get("ordinary_failed") == 0 and h.get("post_failed", 1) == 0 and same and h.get("heavy_after") == "length",
                              f"{tag}: {what}: the engine raised {h['raised']}; {flood_line(h)}; the 8 ordinary prompts alone after it: {('the same tokens and logprobs, bit for bit, as without it' if strict else 'served (not compared: their first blocks are prefix hits)') if same else 'NOT the same as without it'}; a heavy prompt alone after it: {h.get('heavy_after')}")
                if fst:
                    f_ = fst.get("flood", {})
                    check(fst["raised"] is None and f_.get("finished") == f_.get("requests"), f"{tag}: stock, the same flood (40 prompts of 1,400 heavy tokens at once with 8 ordinary ones): the engine raised {fst['raised']}; {flood_line(fst)}")
                    if fd and fst.get("flood") and fd.get("flood"):
                        a_, b_ = fd["flood"], fst["flood"]
                        report.append(f"INFO {tag}: the flood's wall time, kv lossless against stock: {a_['wall_seconds']} s against {b_['wall_seconds']} s ({a_['prompt_tokens_per_second']:,.0f} against {b_['prompt_tokens_per_second']:,.0f} prompt tokens/s: {a_['prompt_tokens_per_second'] / b_['prompt_tokens_per_second']:.2f}x)")
                        print(report[-1], flush=True)
                for what, h, c in (("kv lossless", fi, fsi), ("stock", fsi, None)):
                    if h:
                        f_ = h.get("flood", {})
                        if c is None:  # (the control: what stock does under the same variable)
                            report.append(f"INFO {tag}: under VLLM_BATCH_INVARIANT=1, stock's 8 ordinary prompts in the flood the same bits as alone: {h.get('during_same_as_alone')} of 8 (the control)")
                            print(report[-1], flush=True)
                        else:
                            check(h["raised"] is None and f_.get("finished") == f_.get("requests") and f_.get("ordinary_failed") == 0 and h.get("during_same_as_alone") == 8,
                                  f"{tag}: under VLLM_BATCH_INVARIANT=1 (bf16 weights, kv lossless, eager) the flood of 40 prompts of 1,400 heavy tokens with 8 ordinary ones: {flood_line(h)}; the 8 ordinary prompts' {TOKENS} greedy tokens and their logprobs in the flood the same bits as alone: "
                                  f"{h.get('during_same_as_alone')} of 8 (stock's, the control: {c.get('during_same_as_alone') if c else '?'} of 8)")
                for what, fx_ in (("async scheduling on (vLLM's default)", fx), ("async scheduling off", fxs), ("prefix caching and chunked prefill on", fxp), ("under VLLM_BATCH_INVARIANT=1", fxi)):
                    if fx_:
                        f_, s_, sc_ = fx_.get("flood", {}), fx_.get("status", {}), fx_.get("scheduler", {})
                        check(fx_["raised"] is None and f_.get("finished") == f_.get("requests") and f_.get("failed") == 0 and f_.get("refused") == 0 and f_.get("ordinary_failed") == 0 and s_.get("pages_made_void", 0) > 0 and sc_.get("preempted", 0) > 0 and s_.get("values_compared", 0) > 0 and fx_.get("post_failed", 1) == 0
                              and (fx_ is not fxi or f_.get("heavy_same_tokens_as_alone") == f_.get("requests")),  # (under batch invariance a request that was preempted and run again comes to the tokens it had alone)
                              f"{tag}: 12 prompts of 900 heavy tokens at once with 8 ordinary ones, verify on, admission off, {what}, pools of {pools['GLYD_KV_RESERVE_1_UNITS']} chunks and {pools['GLYD_KV_RESERVE_2_UNITS']} slabs (1.25 times what one of them takes): the engine raised {fx_['raised']}; {flood_line(fx_)}; "
                              f"the 24 greedy tokens after each prompt the same as when it ran alone first: {f_.get('heavy_same_tokens_as_alone')} of {f_.get('requests')}")
                for what, fs_ in (("", fs), (", without the check where vLLM validates the input (the scheduler ends them)", fsb)):
                    if fs_:
                        f_, s_ = fs_.get("flood", {}), fs_.get("status", {})
                        check(fs_["raised"] is None and f_.get("ordinary_failed") == 0 and fs_.get("post_failed", 1) == 0 and f_.get("finished", 0) + f_.get("failed", 0) + f_.get("refused", 0) == f_.get("requests") and s_.get("values_compared", 0) > 0,
                              f"{tag}: 12 prompts of 900 heavy tokens at once with 8 ordinary ones, verify on, the smallest pools (128 chunks, 16 slabs: no prompt fits them alone){what}: the engine raised {fs_['raised']}; {flood_line(fs_)}")
                za = step("kv lossless with verify, no arena and no escape pool: every page with an exception past its heads' inline ones is raw", {**heavy, "kv": True, "verify": True, "blocks": 800}, f"{tag}-heavy-noarena", {**kenv, "GLYD_KV_SPARE": "0", "GLYD_KV_RESERVE_1": "0", "GLYD_KV_RESERVE_2": "25"})
                if za:
                    s_ = za.get("status", {})
                    check(za["raised"] is None and all(r["finish"] != "error" for r in za["rows"]) and s_.get("values_compared", 0) > 0 and s_.get("pages_made_void", 1) == 0,
                          f"{tag}: pages with no arena and no escape pool (every page whose heads have more than three exceptions is raw: {s_.get('pages_made_raw', 0):,} pages) serve the same {len(za['rows'])} prompts and a 1,024-token text prompt with verify on: {s_.get('values_compared', 0):,} values compared, none differ, "
                          f"{s_.get('pages_made_void', 0)} lost, the engine raised {za['raised']}")
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"check_vllm --kv: {'all passed' if not failed else f'{len(failed)} failed'} ({sum(l.startswith(('PASS', 'FAIL')) for l in report)} checks)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        result = child(json.load(open(sys.argv[2][1:])) if sys.argv[2].startswith("@") else json.loads(sys.argv[2]))
        with open(sys.argv[3], "w") as f:
            json.dump(result, f)
    elif "--kv" in sys.argv[1:]:
        kv_main()
    elif "--towers" in sys.argv[1:]:
        towers_main()
    else:
        main()
