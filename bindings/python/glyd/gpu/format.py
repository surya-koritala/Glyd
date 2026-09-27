"""glyd-v1: a model saved with its Linear weights packed, loaded back
without packing them again.

    glyd.save_pretrained(model, "qwen3-8b-glyd")
    model = glyd.from_pretrained("qwen3-8b-glyd", verify=True)

A directory (or a Hugging Face repo) of:
- model.safetensors, or shards and model.safetensors.index.json as
  transformers names them: every packed Linear's buffers under its module
  path, NAME.glyd_data and NAME.glyd_blocks (uint8) and
  NAME.glyd_block_base (int32), in the tiered layout (the smallest); a
  merged group's (q, k, v; gate, up) under its first member's path; the
  rest as the model holds it, in bf16: embeddings (an output layer tied to
  one saved as it), norms, biases;
- glyd.json: the format, the glyd version, the source repo and revision,
  the layout, and for every pack its matrix's shape, its tiers and the
  tensors it holds: their names, shapes and the sha256 of their bf16
  bytes;
- the source's config.json, generation_config.json and tokenizer files.

The names and the manifest need only the standard library; saving needs
PyTorch and safetensors.
"""
import glob
import json
import os
import shutil

FORMAT = "glyd-v1"
MANIFEST = "glyd.json"
BUFFERS = ("data", "blocks", "block_base")  # a tiered pack's tensors (kernels.Mma)
DTYPES = {"data": "U8", "blocks": "U8", "block_base": "I32"}
FILES = ("config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab*", "merges.txt", "*.model", "chat_template*", "preprocessor_config.json", "processor_config.json")  # copied from the source


def key(module, buffer):
    """The safetensors name of a pack's buffer: its module path, .glyd_, the buffer."""
    return f"{module}.glyd_{buffer}"


def entry(shape, tiers, tensors):
    """A pack's manifest entry: its matrix's shape, its tiers, the tensors it holds as [(name, shape, sha256)], in
    their order in its rows."""
    return {"layout": "mma", "shape": list(shape), "tiers": [int(t) for t in tiers], "tensors": [{"name": n, "shape": list(s), "sha256": h} for n, s, h in tensors]}


def members(e):
    """A pack's Linears (module paths) and their rows, in order."""
    return [t["name"][: -len(".weight")] for t in e["tensors"]], [t["shape"][0] for t in e["tensors"]]


def manifest(source, packs, version):
    return {"format": FORMAT, "glyd": version, "source": source, "layout": "mma", "packs": packs}


def read_manifest(directory):
    """directory's glyd.json, or None where it has none."""
    path = os.path.join(directory, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    if m.get("format") != FORMAT:
        raise ValueError(f"{path}: format {m.get('format')!r}; this glyd reads {FORMAT}")
    return m


def header(path):
    """A safetensors file's header: {name: {"dtype", "shape", "data_offsets"}}, and "__metadata__"."""
    with open(path, "rb") as f:
        return json.loads(f.read(int.from_bytes(f.read(8), "little")))


def stored(files):
    """The pack buffers in safetensors files: {name: (shape, dtype)}."""
    return {k: (t["shape"], t["dtype"]) for f in files for k, t in header(f).items() if ".glyd_" in k}


def shard_names(n):
    """n shards' file names, as transformers gives them (one: model.safetensors)."""
    return ["model.safetensors"] if n == 1 else [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]


def fetch_manifest(name_or_path, **hub):
    """A Hub repo's glyd.json fetched into its snapshot, beside its weights (where transformers finds them), when the
    repo has one. hub: revision, token, cache_dir, local_files_only."""
    if os.path.isdir(name_or_path):
        return
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(name_or_path, MANIFEST, **hub)
    except (OSError, ValueError):  # none (a bf16 checkpoint), or no such repo: transformers says which
        pass


def source_dir(repo, revision):
    """The source checkpoint's directory: repo itself, or its Hub snapshot of FILES (fetched as needed); None when
    neither is at hand."""
    if not repo or os.path.isdir(repo):
        return repo or None
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, revision=revision, allow_patterns=list(FILES))
    except (OSError, ValueError):
        return None


def save_pretrained(model, path, shard_bytes=5 * 10**9):
    """model (from glyd.from_pretrained or glyd.gpu.compress) saved in the
    directory path as glyd-v1, in shards of about shard_bytes: its packs in
    the tiered layout (a 12-bit pack decoded and packed again), each tensor
    of each decoded for the sha256 of its bf16 bytes in glyd.json; the rest
    as the model holds it; the source's config, generation config and
    tokenizer files (the model's config where the source is not at hand).
    The host holds a shard at a time, the GPU a matrix more than the model."""
    import torch
    from safetensors.torch import save_file
    from .. import __version__
    from . import kernels as g, model as gm

    if any(getattr(m, "glyd_packs", None) for m in model.modules()):
        raise NotImplementedError("glyd: glyd-v1 holds no packed experts yet; a mixture of experts is packed as it loads (from_pretrained)")
    old = glob.glob(os.path.join(path, "model*.safetensors")) + glob.glob(os.path.join(path, "model.safetensors.index.json"))
    if old and not os.path.exists(os.path.join(path, MANIFEST)):
        raise ValueError(f"glyd: {path} holds another checkpoint; save into a directory of its own")
    os.makedirs(path, exist_ok=True)
    tied = set(getattr(model, "all_tied_weights_keys", None) or {})
    inside = {id(m.lin) for m in model.modules() if isinstance(m, gm.Merged)}
    packs, shards, part = {}, [], {}
    size = [0]

    def flush():
        if part:
            name = f"glyd-part-{len(shards) + 1:05d}.safetensors"  # renamed once the number of shards is known
            save_file(part, os.path.join(path, name), metadata={"format": "pt"})
            shards.append((name, list(part), size[0]))
            part.clear()
            size[0] = 0

    def put(name, t):
        part[name] = t.detach().to("cpu", copy=True).contiguous()
        size[0] += t.numel() * t.element_size()
        if size[0] >= shard_bytes:
            flush()

    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, gm.Merged):
                parent = name.rpartition(".")[0]
                order = sorted((c.i, f"{parent}.{n}" if parent else n) for n, c in model.get_submodule(parent).named_children() if isinstance(c, gm.Part) and c.group[0] is m)
                lin, paths, rows = m.lin, [n for _, n in order], m.sizes
            elif isinstance(m, gm.GLinear) and id(m) not in inside:
                lin, paths, rows = m, [name], [m.p.shape[0]]
            elif isinstance(m, gm.GEmbedding):
                put(f"{name}.weight", gm.unpack(m.p))
                continue
            else:
                continue
            if paths[0] + ".weight" in tied:  # an output layer tied to the embedding: saved as the embedding
                continue
            w = gm.unpack(lin.p)
            p = lin.p if type(lin.p) is g.Mma else g.pack_mma(w)
            for b in BUFFERS:
                put(key(paths[0], b), getattr(p, b))
            packs[paths[0]] = entry(w.shape, p.tiers, [(f"{n}.weight", x.shape, gm.sha256(x)) for n, x in zip(paths, w.split(rows))])
            if lin.bias is not None:
                for n, b in zip(paths, lin.bias.split(rows)):
                    put(f"{n}.bias", b)
            del w, p
        for k, t in model.state_dict().items():
            put(k, t)
        flush()

    for f in old:
        os.remove(f)  # an earlier glyd save's
    names = shard_names(len(shards))
    for (tmp, _, _), name in zip(shards, names):
        os.replace(os.path.join(path, tmp), os.path.join(path, name))
    if len(shards) > 1:
        index = {"metadata": {"total_size": sum(n for _, _, n in shards)}, "weight_map": {k: name for (_, ks, _), name in zip(shards, names) for k in ks}}
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    q = getattr(model.config, "quantization_config", None)
    source = getattr(q, "source", None) or {"repo": model.config.name_or_path or None, "revision": getattr(model.config, "_commit_hash", None)}
    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(manifest(source, packs, __version__), f, indent=1)
    src = source_dir(model.config.name_or_path, getattr(model.config, "_commit_hash", None))
    if src and os.path.realpath(src) != os.path.realpath(path):
        for pattern in FILES:
            for f in glob.glob(os.path.join(src, pattern)):
                shutil.copy(f, path)
    if not os.path.exists(os.path.join(path, "config.json")):  # no source at hand: the model's own
        config = model.config.to_diff_dict()
        config.pop("quantization_config", None)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(path)
    return path
