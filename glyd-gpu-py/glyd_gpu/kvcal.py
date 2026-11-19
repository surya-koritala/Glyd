"""The lossless KV cache's per-model tables (vLLM's kv: lossless, glyd_gpu.vllm_kv), calibrated on vLLM itself.

    python -m glyd_gpu kvcal MODEL [--positions N] [--out FILE] [--engine JSON]

runs vLLM in this process (eager, its own attention backend, the stock bf16 cache) on the text in kvcorpus.bin, takes the keys and values each
layer hands the cache (what vLLM stores), and writes the tables. The vLLM plugin calls it as a child process the first time a model is served with
kv: lossless (about 1-4 minutes for an 8B model on an L4, by the window) and keeps the tables under $GLYD_CACHE/kv (default ~/.cache/glyd/kv), one file a
(model, weights, tensor parallel, vLLM, corpus, positions): later starts read the file.

The text is run as two sequences of as many tokens as the model will be served at (TIERS: 4,096, 8,192, 16,384, 32,768 or 40,960: the least that covers
--max-model-len; read again from the start where it is shorter), each kind of text at every depth of them. Before it writes, it measures what the tables leave on each kind of text left out in
turn (prose, docs, code, chat, tables), recorded in the tables file and logged; the model is refused (REFUSED) where the worst kind leaves more than 0.15%
of either tensor's values outside 12 bits. A model's head of 256 channels is counted as the cache holds it, two heads of 128 (the tables' [2 H, 128]). A last pass sweeps every row of the model's embedding matrix and sizes the cache's reserves from it
(glyd_gpu.vllm_kv): the tables record which tokens need the second reserve and how much each takes (what the scheduler predicts a prompt's needs from
before it runs).

The corpus (kvcorpus.bin, zlib-compressed, made by gpu/vllm/make_kvcorpus.py) is public domain prose (Project Gutenberg's A Tale of Two
Cities, The Adventures of Sherlock Holmes, On the Origin of Species and Pride and Prejudice), excerpts of this repository's
own documents, short modules and conversations written for it, and a table and JSON records generated from fixed seeds.
Under the Business Source License 1.1 (LICENSE), as the rest of Glyd's GPU code."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

VERSION = 8  # the tables' format and rule: part of the key (6: the pool is sized from a sweep of every row of the embedding matrix; 7: and each row's exceptions in each layer are kept; 8: the file's keys under neutral names)
BANDS = (4096, 8192, 16384, 32768, 40960)  # the positions a table is calibrated for (40,960: the Qwen3 models' window)
LONG_BANDS = (65536, 131072)  # not yet served and checked at those lengths on a GPU that holds them: with GLYD_KV_LONG=1 only (the sequences' whole cache sits in vLLM beside the weights: 16 GiB of it for an 8B model at 131,072)
CHUNKED = 16384  # a sequence of more positions is run in steps of STEP tokens, and the sweep of the vocabulary in batches of SWEEP_STEPPED requests: the activations of a step of all the sequence and the sampler's room for 1,024
STEP = 4096  # requests would not leave room for the sequence's own cache beside the weights on a 24 GB GPU
SWEEP_STEPPED = 256
LAP = 211  # the offset (tokens) a stream is read again from when the text is shorter than a sequence
WINDOW = 8  # values of exponent >> 1 a base covers (the code's 3-bit offset)
REFUSED = 0.0015  # the most of either tensor's values a kind of text may leave outside the windows
INLINE = 3  # exceptions a head-page holds in its header: the page's arena and the escape pool hold the rest
CH = 64  # entries in a chunk of the escape pool (gpu/glyd_gpu.cu)
SWEEP_SEQS = 1024  # one-token requests of the sweep a batch


def corpus_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "kvcorpus.bin")


def corpus_text():
    """The corpus's text, UTF-8 bytes (kvcorpus.bin holds it zlib-compressed)."""
    import zlib

    with open(corpus_path(), "rb") as f:
        return zlib.decompress(f.read())


def corpus():
    """The sections of the corpus: [(kind, name, body)]."""
    out = []
    for part in corpus_text().decode("utf-8").split("@@@ ")[1:]:
        head, _, body = part.partition("\n")
        kind, name = head.split(" ", 1)
        out.append((kind, name, body.strip()))
    return out


def corpus_digest():
    return hashlib.sha256(corpus_text()).hexdigest()[:16]


def cache_dir():
    root = os.environ.get("GLYD_CACHE") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "glyd")
    return os.path.join(root, "kv")


def weights_fingerprint(model):
    """What tells one checkpoint of a model's name from another: a local directory's weight files by name and size (a Hub
    repo: nothing more than its name and revision, which are in the key)."""
    if not os.path.isdir(model):
        return []
    return sorted((n, os.path.getsize(os.path.join(model, n))) for n in os.listdir(model) if n.endswith((".safetensors", ".bin", ".pt", ".json")) and os.path.isfile(os.path.join(model, n)))


def key(model, revision, hf_config, tp=1, pp=1, positions=8192, extra=None):
    """The 16 hex digits that name a model's tables: the model and its revision, its config, its weight files (a directory), tensor and
    pipeline parallel sizes, the positions calibrated, the corpus, this rule's version, and vLLM's and Glyd's (extra: anything else that
    changes the keys and values)."""
    from . import __version__

    try:
        import vllm

        v = vllm.__version__
    except Exception:
        v = "?"
    cfg = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config or {})
    cfg.pop("_name_or_path", None)
    cfg.pop("transformers_version", None)
    blob = json.dumps({"model": model, "revision": revision, "config": cfg, "files": weights_fingerprint(model), "tp": tp, "pp": pp, "positions": positions, "corpus": corpus_digest(),
                       "version": VERSION, "vllm": v, "glyd": __version__, "extra": extra}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Tables:
    """A tables file: layers (their vLLM names), k and v uint8 [layers, kv heads, channels], and meta (the measures)."""

    def __init__(self, path):
        import numpy as np

        z = np.load(path, allow_pickle=False)
        self.path = path
        self.layers = [str(x) for x in z["layers"]]
        self.k, self.v = z["k"], z["v"]
        self.meta = json.loads(str(z["meta"]))
        self.exc = z["counts"] if "counts" in z.files else None  # int16 [rows of the embedding matrix, layers]: the exceptions a token's keys and values have in each layer (the sweep's)
        self.index = {n: i for i, n in enumerate(self.layers)}

    def layer(self, name):
        """(k, v) uint8 [kv heads, channels] of the layer vLLM calls name; KeyError where the file has none."""
        i = self.index[name]
        return self.k[i], self.v[i]


def save(path, layers, k, v, meta, exc=None):
    import numpy as np

    tmp = f"{path}.{os.getpid()}.tmp.npz"
    extra = {} if exc is None else {"counts": np.ascontiguousarray(exc, dtype=np.int16)}
    np.savez_compressed(tmp, layers=np.array(layers), k=k, v=v, meta=np.array(json.dumps(meta, sort_keys=True)), **extra)
    os.replace(tmp, path)


def best_base(hist):
    """uint8 [H, D]: the start of the window of WINDOW consecutive values of exponent >> 1 holding the most (the first such) of a
    histogram [H, D, 128] of them."""
    import torch

    c = torch.cumsum(torch.nn.functional.pad(hist.to(torch.int64), (1, 0)), -1)
    return (c[..., WINDOW:] - c[..., :-WINDOW]).argmax(-1).to(torch.uint8)


def histogram(x):
    """[H, D, 128] counts of exponent >> 1 (bf16 bits 8-14) of x, int16 bits [T, H, D] on the GPU."""
    import torch

    T, H, D = x.shape
    e7 = (x.to(torch.int32) >> 8) & 127
    idx = (torch.arange(H * D, device=x.device, dtype=torch.int32).view(1, H, D) * 128 + e7).view(-1)
    return torch.bincount(idx, minlength=H * D * 128).view(H, D, 128)


def outside(x, base):
    """Bool [T, H, D]: the values of x (int16 bits) whose exponent >> 1 is outside their channel's window."""
    import torch

    e7 = (x.to(torch.int32) >> 8) & 127
    b = base.to(torch.int32)[None]
    return (e7 < b) | (e7 >= b + WINDOW)


def leaf_needs(ek, ev):
    """(total, beyond) exceptions of each page of 16 tokens from the bool masks [T, H, D] of K and V: a page's total, and what it needs beyond its heads' inline
    entries (its arena and the escape pool), the exceptions of a head-page (its K and V) beyond its three inline summed over the heads."""
    import torch

    n = ek.shape[0] // 16
    if n == 0:
        z = torch.zeros(0, dtype=torch.int64, device=ek.device)
        return z, z
    hp = ek[: n * 16].reshape(n, 16, *ek.shape[1:]).sum((1, 3)) + ev[: n * 16].reshape(n, 16, *ev.shape[1:]).sum((1, 3))  # [n, H]
    return hp.sum(1), torch.clamp(hp - INLINE, min=0).sum(1)


SLACK = 0.9  # a page is predicted raw where its exceptions come to this share of what a page holds (a token run through the model took 0.5 to 1.2 times the sweep's figure)


class Predictor:
    """What a prompt's pages will need, from the sweep's exceptions of each token in each layer (Tables.exc), before the prompt runs: the layer-pages that will be raw. A page of 16 tokens
    has at most the sum of its tokens' exceptions, and past its heads' inline ones at least that less INLINE a head; it is raw where that leaves more than a page holds (SLACK of it)."""

    def __init__(self, exc, heads, capacity):
        import numpy as np

        self.exc = np.concatenate([np.ascontiguousarray(exc, dtype=np.int16), np.zeros((1, exc.shape[1]), dtype=np.int16)])  # (one row more, of zeros: the places after a prompt's last token)
        self.rows, self.layers = exc.shape
        self.emax = self.exc.max(axis=1).astype(np.int32)
        self.thr = int(SLACK * int(capacity)) + INLINE * int(heads)
        self.active = 16 * int(self.emax.max()) > self.thr  # (False: no page of any tokens of this vocabulary is raw)

    def blocks(self, ids):
        """int32 [ceil(n / 16)]: the layer-pages each block of 16 tokens of `ids` (token ids) is predicted to make raw."""
        import numpy as np

        ids = np.minimum(np.asarray(ids, dtype=np.int64), self.rows - 1)
        n = ids.size
        nb = -(-n // 16)
        out = np.zeros(nb, dtype=np.int32)
        if not nb or not self.active:
            return out
        grid = np.full(nb * 16, self.rows, dtype=np.int64)  # (the zero row in the places past the last token)
        grid[:n] = ids
        grid = grid.reshape(nb, 16)
        cand = np.nonzero(self.emax[grid].sum(1) > self.thr)[0]  # (a layer's sum is at most the sum of the tokens' most: only these blocks can have a raw page)
        if cand.size:
            t = np.add.reduce(self.exc[grid[cand]], axis=1, dtype=np.int32)  # [candidates, layers]
            out[cand] = (t > self.thr).sum(1)
        return out


# ---- the child: vLLM in this process --------------------------------------------------------------------------------------------
_STATE = {"mode": None, "segs": [], "off": [], "hist": {}, "loo": {}, "ex": {}, "ex_layer": {}, "leaves": [], "layers": [], "base": None, "bay": 0, "sweep": None}
CHUNK = 512  # tokens of a kind's text a sequence takes at a time: every kind at every depth of it


def leaf_shape(H, head_size):
    """(heads, channels) of a layer's keys as the cache's pages hold them: a head of 256 channels is two of 128 (its tables are [2 H, 128])."""
    return (2 * H, head_size // 2) if head_size == 256 else (H, head_size)


def _install(model):
    """A forward pre-hook on every decoder Attention layer: the key and value it is handed (after the key norm and the rotary embedding: what vLLM stores), by the
    sequence's segments (kind, start, end), counted as _STATE["mode"] says: "hist" the exponents' histograms of each kind; "eval" the values outside the windows the
    other kinds' histograms make, and the pages' needs."""
    import torch
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    names = []
    for m in model.modules():
        if not isinstance(m, Attention) or m.attn_type != AttentionType.DECODER:
            continue
        if m.sliding_window is not None:
            raise ValueError(f"glyd kv: {m.layer_name} is a sliding window layer: the lossless KV cache does not hold those yet")
        li = len(names)
        names.append(m.layer_name)

        def hook(mod, args, kwargs, li=li):
            mode = _STATE["mode"]
            if mode is None:
                return
            k = args[1] if len(args) > 1 else kwargs.get("key")
            v = args[2] if len(args) > 2 else kwargs.get("value")
            if k is None or v is None:
                return
            hp, d = leaf_shape(mod.num_kv_heads, mod.head_size)  # (a head of 256 channels: two of the pages' heads of 128, as the cache holds it)
            k = k.detach().reshape(-1, hp, d).contiguous().view(torch.int16)
            v = v.detach().reshape(-1, hp, d).contiguous().view(torch.int16)
            if mode == "sweep":  # (a batch of one-token requests, in order: the entries a page of each token repeated needs, and the exceptions the token has, in this layer)
                sw = _STATE["sweep"]
                ek_, ev_ = outside(k, _STATE["base"]["k"][li]), outside(v, _STATE["base"]["v"][li])
                need = sweep_needs(ek_, ev_)
                r0 = sw["rows"][li]
                sw["need"][r0 : r0 + need.numel(), li] = need.to(torch.int32)
                sw["counts"][r0 : r0 + need.numel(), li] = sweep_exc(ek_, ev_)
                sw["rows"][li] = r0 + need.numel()
                return
            off, n = _STATE["off"][li], k.shape[0]  # (a sequence run in steps: this step's tokens are the sequence's off to off + n)
            _STATE["off"][li] = off + n
            for kind, s0, e0 in _STATE["segs"]:
                s0, e0 = max(s0, off), min(e0, off + n)
                if s0 >= e0:
                    continue
                ks, vs = k[s0 - off : e0 - off], v[s0 - off : e0 - off]
                if mode == "hist":
                    for w, x in (("k", ks), ("v", vs)):
                        key_ = (kind, w, li)
                        _STATE["hist"][key_] = _STATE["hist"].get(key_, 0) + histogram(x)
                else:
                    ek, ev = outside(ks, _STATE["loo"][(kind, "k", li)]), outside(vs, _STATE["loo"][(kind, "v", li)])
                    acc = _STATE["ex"].setdefault(kind, [0, 0, 0])
                    nk, nv = int(ek.sum()), int(ev.sum())
                    acc[0] += nk
                    acc[1] += nv
                    acc[2] += ek.numel()
                    al = _STATE["ex_layer"].setdefault((kind, li), [0, 0, 0])
                    al[0] += nk
                    al[1] += nv
                    al[2] += ek.numel()
                    j0, j1 = -(-s0 // 16), e0 // 16  # (the pages of the sequence, 16 tokens from its start, that lie in this segment and step)
                    if j1 > j0:
                        tot, need = leaf_needs(ek[16 * j0 - s0 : 16 * j1 - s0], ev[16 * j0 - s0 : 16 * j1 - s0])
                        _STATE["leaves"].append((kind, torch.arange(j0, j1) * 16, tot.cpu(), need.cpu()))

        m.register_forward_pre_hook(hook, with_kwargs=True)
    _STATE["layers"] = names
    _STATE["off"] = [0] * len(names)
    return names


def tokens(tok, kind, body):
    """A document's token ids: a chat section through the chat template (a model without one: its turns as plain lines)."""
    if kind == "chat":
        msgs = json.loads(body)
        if getattr(tok, "chat_template", None):
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            return list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        body = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    return list(tok.encode(body))


def sequences(docs_ids, length):
    """Two sequences of `length` tokens, each a document stream's chunks in turn (a kind's documents one after the other, CHUNK tokens of a kind at a time, the kinds round
    and round): A in the order of the kinds by name, B in the reverse with each kind's stream entered half a chunk late, so that the same text is at other depths. Returns
    [(ids, [(kind, start, end)])]: the segments each kind's chunks fill."""
    streams = {}
    for kind, _, ids in docs_ids:
        streams.setdefault(kind, []).extend(ids)
    out = []
    for order, shift in ((sorted(streams), 0), (sorted(streams, reverse=True), CHUNK // 2)):
        seq, segs, lap = [], [], 0
        while len(seq) < length:  # (a text shorter than the sequence is read again, from another offset a lap: the sequence is as long as the table says)
            ptr = {k: shift if lap == 0 else (shift + lap * LAP) % max(1, len(streams[k])) for k in streams}
            had = len(seq)
            while len(seq) < length and any(ptr[k] < len(streams[k]) for k in streams):
                for kind in order:
                    part = streams[kind][ptr[kind] : ptr[kind] + CHUNK][: length - len(seq)]
                    if part:
                        segs.append((kind, len(seq), len(seq) + len(part)))
                        seq += part
                        ptr[kind] += CHUNK
            if len(seq) == had:  # (no text at all)
                break
            lap += 1
        out.append((seq, segs))
    return out


def bands():
    """The positions a table can be made for: TIERS, and LONG_TIERS with GLYD_KV_LONG=1."""
    # dev: GLYD_KV_LONG=1 asks for the longer bands (not yet checked)
    if os.environ.get("GLYD_KV_LONG", "").strip().lower() in ("1", "true", "yes", "on"):
        return BANDS + LONG_BANDS
    # end dev
    return BANDS


def band(max_model_len):
    """The positions the tables are calibrated for to serve max_model_len: the least of bands() that covers it; None above the last."""
    return next((t for t in bands() if t >= max_model_len), None)


def _norms_of(m):
    """The L2 norm of each row of an embedding layer's table (CPU float32 [rows]): of its weight, or, where the plugin holds it packed (the weight is empty), of the pack decoded a
    piece at a time."""
    import torch

    if getattr(m, "glyd_words", None) is None:
        w = m.weight.detach()
        return torch.cat([w[i : i + 8192].float().norm(dim=1) for i in range(0, w.shape[0], 8192)]).cpu()  # (a piece at a time: the whole matrix in float32 is 2.5 GiB, which a GPU the cache has filled does not have)
    from . import kernels as g
    from .vllm_plugin import _pack_of

    p, K = _pack_of(m), m.embedding_dim
    buf = torch.empty(8192 * K, dtype=torch.bfloat16, device=m.glyd_data.device)
    out = [g.mma_unpack(p, buf[: min(8192, p.shape[0] - r) * K], r, min(8192, p.shape[0] - r)).float().norm(dim=1) for r in range(0, p.shape[0], 8192)]
    return torch.cat(out)[: m.num_embeddings_per_partition].cpu()


def _embed_norms(model):
    """The L2 norm of each row of the model's input embedding matrix (CPU float32 [vocab rows]); None where the model has no such module."""
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    for m in model.modules():
        if isinstance(m, VocabParallelEmbedding):
            return _norms_of(m)
    return None


def sweep_needs(ek, ev):
    """int32 [n]: for each token (bool masks [n, H, D] of its keys' and values' exceptions), the entries past its heads' inline ones that a page of 16 of it needs."""
    import torch

    hp = (ek.sum(2) + ev.sum(2)) * 16
    return torch.clamp(hp - INLINE, min=0).sum(1).to(torch.int32)


def sweep_exc(ek, ev):
    """int16 [n]: for each token (bool masks [n, H, D] of its keys' and values' exceptions), the exceptions it has in all its heads."""
    import torch

    return (ek.sum((1, 2)) + ev.sum((1, 2))).to(torch.int16)


def sweep_stats(need, bay, capacity):
    """From need [rows, layers] (the entries a page of 16 of each token needs past its heads' inline ones, in each layer): the chunks a layer-page of each token takes on average over the layers (a page
    that needs more than `capacity` is made raw: no chunks), the layers in which it needs more than a page holds, and the most entries it needs in any layer."""
    import torch

    over = need > capacity
    chunks = torch.where(over, 0, (torch.clamp(need - bay, min=0) + CH - 1) // CH)
    return chunks.float().mean(1), over.sum(1), need.max(1).values


def store_meta(chunks, over, maxneed, bay, capacity, rows, reach=lambda i: True, decode=str):
    """The tables' meta["pool"] from the sweep's figures per token: the chunks a layer-page of the worst token takes (what sizes the escape pool), the tokens whose pages need more than a page holds
    (`hard`: their pages are raw), by whether the tokenizer makes them from text (`reach`), and the most layers of such a page."""
    heavy = (over > 0).nonzero().view(-1).tolist()
    text = [i for i in heavy if reach(i)]
    ids = sorted(set(heavy) - set(text))
    top = chunks.topk(min(8, chunks.numel()))
    return {"store": {"swept": rows, "bay": bay, "capacity": capacity, "worst_chunks_per_layer_leaf": round(float(chunks.max()), 4), "worst_token": int(chunks.argmax()),
                     "median_chunks_per_layer_leaf": round(float(chunks.median()), 4), "top": [[int(i), round(float(c), 4)] for c, i in zip(top.values.tolist(), top.indices.tolist())],
                     "hard": len(heavy), "hard_text": len(text), "hard_ids": len(ids), "hard_text_ids": text[:32], "hard_text_tokens": [decode(i) for i in text[:8]],
                     "hard_ids_sample": ids[:64], "hard_worst_leaf_need": int(maxneed[heavy].max()) if heavy else 0, "second_layers": int(over.max())}}


def heavy(meta):
    """The tokens of a model that need the second reserve, from its tables' meta: {"text": ids the tokenizer makes from text, "ids": a sample of the ids only a request of token ids reaches,
    "count": how many in all, "worst": the one that takes most of it}."""
    p = (meta or {}).get("store") or {}
    return {"text": p.get("hard_text_ids", []), "ids": p.get("hard_ids_sample", []), "count": p.get("hard_ids"), "worst": p.get("worst_token", 1)}


def refusal(meta):
    """Why a model's tables refuse it (meta["refused"]): its values fall outside 12 bits too often."""
    h = meta["held_out_rate"]
    return f"the model's values compress too poorly (worst kind of text left out: K {h['k'] * 100:.3f}%, V {h['v'] * 100:.3f}%, the limit {REFUSED * 100:.2f}%)"


def lengths(positions, window=None):
    """(the engine's max_model_len, the sequences' tokens) for a table of `positions`: 64 tokens more than the sequences, but no more than the model's own window (vLLM refuses a longer one), and a prompt is
    at most one token shorter than that."""
    mml = positions + 64 if not window or positions + 64 <= window else int(window)
    return mml, min(positions, mml - 1)


def make_tables(model, engine, out, positions, log=print):
    """Runs the corpus through `model` on vLLM (engine: LLM keyword arguments; in this process) as two sequences of `positions` tokens (every kind of text at every depth,
    for a key's exponents drift with its position), and writes the tables file `out`: pass one counts each kind's exponents, pass two the values outside the windows the other kinds'
    counts make, and the pages' needs."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t0 = time.time()
    docs = corpus()
    stepped = positions > CHUNKED
    tok = AutoTokenizer.from_pretrained(model, **{k: engine[k] for k in ("revision", "trust_remote_code") if k in engine})
    try:  # (a model's own window: a prompt is at most one token shorter than it, so the last tier of a model whose window it is runs sequences of one token less)
        from transformers import AutoConfig

        window = getattr(AutoConfig.from_pretrained(model, **{k: engine[k] for k in ("revision", "trust_remote_code") if k in engine}).get_text_config(), "max_position_embeddings", None)
    except Exception:
        window = None
    mml, length = lengths(positions, window)
    kw = dict(dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False, max_num_seqs=SWEEP_SEQS, gpu_memory_utilization=0.92 if positions > 8192 else 0.85, seed=0)
    kw.update(engine)
    kw["enable_chunked_prefill"] = stepped
    kw["max_model_len"] = mml
    kw["max_num_batched_tokens"] = STEP if stepped else mml
    if stepped:
        kw["max_num_seqs"] = SWEEP_STEPPED
    seqs = sequences([(kd, nm, tokens(tok, kd, body)) for kd, nm, body in docs], length)
    llm = LLM(model, **kw)
    names = llm.apply_model(_install)[0]
    log(f"glyd kvcal: {len(docs)} documents, sequences of {positions:,} tokens, {len(names)} layers ({time.time() - t0:.0f} s to load)")

    def run_pass(engine_, mode):  # (the engine is an argument, not a name of this function's: it is deleted below, which a name used by a nested function cannot be)
        _STATE["mode"] = mode
        for ids, segs in seqs:
            _STATE["segs"], _STATE["off"] = segs, [0] * len(names)
            engine_.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
        _STATE["mode"] = None

    t_pass = time.time()
    run_pass(llm, "hist")
    log(f"glyd kvcal: pass 1 of 3 ({time.time() - t_pass:.0f} s)")
    kinds = sorted({k for k, _, _ in docs})
    hist = _STATE["hist"]
    tot = {(w, li): sum(hist[(kd, w, li)] for kd in kinds if (kd, w, li) in hist) for w in "kv" for li in range(len(names))}
    base = {w: torch.stack([best_base(tot[(w, li)]) for li in range(len(names))]) for w in "kv"}  # [L, H, D]
    for kd in kinds:
        for w in "kv":
            for li in range(len(names)):
                _STATE["loo"][(kd, w, li)] = best_base(tot[(w, li)] - hist.get((kd, w, li), 0))
    t_pass = time.time()
    run_pass(llm, "eval")
    log(f"glyd kvcal: pass 2 of 3 ({time.time() - t_pass:.0f} s)")
    per_kind = {kd: {"k": _STATE["ex"][kd][0] / max(1, _STATE["ex"][kd][2]), "v": _STATE["ex"][kd][1] / max(1, _STATE["ex"][kd][2]), "values": _STATE["ex"][kd][2]} for kd in kinds if kd in _STATE["ex"]}
    pos = torch.cat([p for _, p, _, _ in _STATE["leaves"]])
    pt, pn = torch.cat([t for _, _, t, _ in _STATE["leaves"]]), torch.cat([n for _, _, _, n in _STATE["leaves"]])
    q = lambda x, p: int(torch.quantile(x.float(), p).item()) if x.numel() else 0
    per_tensor = {w: [0.0, ""] for w in "kv"}  # (the worst single layer's K and V of a kind of text: the figure of the tensor that is least held)
    for (kd, li), (nk, nv, nn) in _STATE["ex_layer"].items():
        for w, x in (("k", nk), ("v", nv)):
            if x / max(1, nn) > per_tensor[w][0]:
                per_tensor[w] = [x / max(1, nn), f"{names[li]} ({kd})"]
    # pass three: every row of the embedding matrix swept (a page of each token repeated: the entries it needs)
    from .vllm_kv import ND, free_bay

    t1 = time.time()
    bay = free_bay(int(base["k"].shape[1]))
    _STATE["base"], _STATE["bay"] = base, bay
    capacity = bay + ND * CH  # (the most entries past its heads' inline ones a page holds: its arena and the pool's chunks of the directory)
    norms = llm.apply_model(_embed_norms)[0]
    rows = norms.numel() if norms is not None else len(tok)
    _STATE["mode"] = "sweep"
    _STATE["sweep"] = {"need": torch.zeros(rows, len(names), dtype=torch.int32, device="cuda"), "counts": torch.zeros(rows, len(names), dtype=torch.int16, device="cuda"), "rows": [0] * len(names)}
    nb = SWEEP_STEPPED if stepped else SWEEP_SEQS
    for b0 in range(0, rows, nb):
        llm.generate([TokensPrompt(prompt_token_ids=[i]) for i in range(b0, min(rows, b0 + nb))], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    _STATE["mode"] = None
    chunks, over, maxneed = sweep_stats(_STATE["sweep"]["need"], bay, capacity)
    exc = _STATE["sweep"]["counts"].cpu().numpy()
    _STATE["sweep"] = None
    skip = set(i for i in tok.all_special_ids if i is not None) | set(getattr(tok, "get_added_vocab", lambda: {})().values())
    reach = lambda i: i not in skip and tok.encode(tok.decode([i]), add_special_tokens=False) == [i]  # (a token of the vocabulary that the tokenizer makes from its own string)
    by_depth = []
    for lo in range(0, positions, 4096):
        m = (pos >= lo) & (pos < lo + 4096)
        if m.any():
            by_depth.append({"from": lo, "leaves": int(m.sum()), "bay_need_mean": round(float(pn[m].float().mean()), 3), "bay_need_p999": q(pn[m], 0.999), "bay_need_max": int(pn[m].max())})
    meta = {
        "positions": positions,
        "held_out_rate": {"k": max(r["k"] for r in per_kind.values()), "v": max(r["v"] for r in per_kind.values())},
        "per_kind": per_kind,
        "leaves": {"n": int(pt.numel()), "total_mean": float(pt.float().mean()), "total_p99": q(pt, 0.99), "total_max": int(pt.max()), "bay_need_p99": q(pn, 0.99), "bay_need_p999": q(pn, 0.999), "bay_need_max": int(pn.max())},
        "by_depth": by_depth,
    }
    meta["per_tensor"] = {w: {"rate": v[0], "worst": v[1]} for w, v in per_tensor.items()}
    meta.update(store_meta(chunks, over, maxneed, bay, capacity, rows, reach, lambda i: tok.decode([i])))
    meta["store"]["seconds"] = round(time.time() - t1, 1)
    meta["refused"] = max(meta["held_out_rate"].values()) > REFUSED
    meta.update({"model": model, "layers": len(names), "heads": int(base["k"].shape[1]), "channels": int(base["k"].shape[2]), "tokens": sum(len(i) for i, _ in seqs), "corpus": corpus_digest(), "version": VERSION,
                 "seconds": round(time.time() - t0, 1), "vllm": __import__("vllm").__version__})
    save(out, names, base["k"].cpu().numpy(), base["v"].cpu().numpy(), meta, exc)
    pl = meta["store"]
    log(f"glyd kvcal: {out}: {positions:,} positions; values that compress poorly, worst kind left out: K {meta['held_out_rate']['k'] * 100:.4f}%, V {meta['held_out_rate']['v'] * 100:.4f}%; "
        f"{rows:,} tokens checked ({pl['seconds']:.0f} s): {pl['hard']} need the lossless cache's second extra memory ({pl['hard_text']} made from text)"
        f"{'; REFUSED: ' + refusal(meta) if meta['refused'] else ''} ({time.time() - t0:.0f} s)")
    return meta


# ---- the parent: the plugin asking for a model's tables --------------------------------------------------------------------------
def minutes(positions):
    """What a first start's calibration takes, for the log's line (an L4, the weights in the OS file cache; Qwen3-8B packed: 4,096 positions 85 s, 8,192 130 s, 40,960 217 s)."""
    return "1-2" if positions <= 4096 else "2-3" if positions <= 8192 else "3-4"


def ensure(model, key_of, engine, log, revision=None, need=4096):
    """The path of a tables file that covers `need` positions: the cache's, for the least tier that does (key_of(positions): the key of the file
    for that many: a larger tier's file serves a smaller need), else made by a child process (this module's main, vLLM in it, the GPU free)
    while this one waits, under a lock so that two servers starting together make it once. The child's log: next to it."""
    import fcntl

    bands_ = [t for t in bands() if t >= need]
    if not bands_:
        raise ValueError(f"glyd: the KV cache's tables cover up to {bands()[-1]:,} positions, and {need:,} are asked: use --max-model-len {bands()[-1]} or less, or GLYD_KV=off")
    d = cache_dir()
    os.makedirs(d, exist_ok=True)
    for t in bands_:
        if os.path.exists(os.path.join(d, f"kvtab-{key_of(t)}.npz")):
            return os.path.join(d, f"kvtab-{key_of(t)}.npz")
    key_ = key_of(bands_[0])
    path = os.path.join(d, f"kvtab-{key_}.npz")
    with open(os.path.join(d, f"kvtab-{key_}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(path):  # (another server made it while this one waited)
            return path
        log(f"glyd: the lossless KV cache takes tables of {model} for {bands_[0]:,} positions, made once on vLLM itself (first start: about {minutes(bands_[0])} minutes on an L4, then kept in {d}); its log: kvtab-{key_}.log")
        env = dict(os.environ, GLYD_KV="off", VLLM_ENABLE_V1_MULTIPROCESSING="0")
        cmd = [sys.executable, "-m", "glyd_gpu", "kvcal", model, "--out", path, "--positions", str(bands_[0]), "--engine", json.dumps(engine)]
        if revision:
            cmd += ["--revision", revision]
        t0 = time.time()
        with open(os.path.join(d, f"kvtab-{key_}.log"), "w") as lg:
            r = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT)
        if r.returncode or not os.path.exists(path):
            tail = open(os.path.join(d, f"kvtab-{key_}.log")).read()[-1500:]
            raise RuntimeError(f"glyd: the KV cache's tables for {model} could not be made for {bands_[0]:,} positions (exit {r.returncode}; a smaller --max-model-len needs less of the GPU); its log {os.path.join(d, f'kvtab-{key_}.log')} ends:\n{tail}")
        log(f"glyd: the KV cache's tables for {model} made in {time.time() - t0:.0f} s: {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m glyd_gpu kvcal", description="The lossless KV cache's tables for a model, made on vLLM.")
    ap.add_argument("model")
    ap.add_argument("--out", help="the tables file (default: the cache's, by the model's name)")
    ap.add_argument("--engine", default="{}", help="vLLM LLM keyword arguments, JSON (quantization, trust_remote_code, ...)")
    ap.add_argument("--revision")
    ap.add_argument("--positions", type=int, default=8192, help=f"the positions to make tables for: one of {', '.join(map(str, BANDS))} (8192)")
    a = ap.parse_args(argv)
    engine = json.loads(a.engine)
    if a.revision:
        engine["revision"] = a.revision
    out = a.out or os.path.join(cache_dir(), "kvtab-" + hashlib.sha256(a.model.encode()).hexdigest()[:16] + ".npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    meta = make_tables(a.model, engine, out, a.positions, log=lambda s: print(s, flush=True))
    return 1 if meta["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
