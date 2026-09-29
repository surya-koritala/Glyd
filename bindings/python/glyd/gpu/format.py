"""glyd-v1: a model saved with its Linear weights packed, loaded back
without packing them again.

    glyd.save_pretrained(model, "qwen3-8b-glyd")
    model = glyd.from_pretrained("qwen3-8b-glyd", verify=True)
    glyd.save_pretrained(model, "qwen3-8b-glyd12", layout="mma12")   # glyd-v3: the 12-bit layout, as A10, A100 and H100 load it

A directory (or a Hugging Face repo) of:
- model.safetensors, or shards and model.safetensors.index.json as
  transformers names them: every packed Linear's buffers under its module
  path, NAME.glyd_data and NAME.glyd_blocks (uint8) and
  NAME.glyd_block_base (int32), in the tiered layout (the smallest); a
  merged group's (q, k, v; gate, up) under its first member's path; a
  mixture of experts' weight (a layer's experts, one matrix of their
  matrices stacked, [E out, in]) under the module holding it,
  NAME.glyd_WEIGHT_data and so on; the rest as the model holds it, in
  bf16: embeddings (an output layer tied to one saved as it), norms,
  biases;
- in the 12-bit layout instead (save_pretrained's layout="mma12":
  glyd-v3, which glyd 0.23 and before refuse by the format): each pack's
  NAME.glyd_data (uint8), NAME.glyd_exc and NAME.glyd_exc_base (int32), its
  words of symbols in glyd.json ("sym"); loaded as saved where the 12-bit
  layout is the one, else decoded and packed again;
- glyd.json: the format (glyd-v2 where it holds a mixture of experts'
  packs: glyd 0.21 reads glyd-v1 alone, and refuses it by the format),
  the glyd version, the source repo and revision,
  the layout, and for every pack its matrix's shape, its tiers and the
  tensors it holds: their names, shapes and the sha256 of their bf16
  bytes (an experts' weight: its E experts, and whether the model holds
  their matrices transposed, [in, out]; its one tensor as the model holds
  it); and ("tensors", glyd 0.25 on) the sha256 of every tensor saved as
  it is, which verify checks with the packs (readers before ignore it);
- the source's config.json, generation_config.json and tokenizer files.

The names and the manifest need only the standard library; saving needs
PyTorch and safetensors.
"""
import glob
import hashlib
import json
import os
import re
import shutil

FORMAT = "glyd-v1"
# glyd-v2: glyd-v1 with a mixture of experts' packs (glyd 0.21 reads glyd-v1 alone); glyd-v3: the packs in the 12-bit
# layout (glyd 0.23 reads glyd-v1 and glyd-v2)
FORMATS = (FORMAT, "glyd-v2", "glyd-v3")
MANIFEST = "glyd.json"
BUFFERS = ("data", "blocks", "block_base")  # a tiered pack's tensors (kernels.Mma)
LAYOUTS = {"mma": BUFFERS, "mma12": ("data", "exc", "exc_base")}  # each layout's (kernels.Mma12's)
WORDS = {"mma": "tiers", "mma12": "sym"}  # a pack's words, in glyd.json
GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))  # the Linears merged, a layer's self_attn's and mlp's (model.groups)
KEYS = ("format", "glyd", "source", "layout", "packs", "tensors")  # glyd.json's (glyd 0.21 on; "tensors" from 0.25)
MAP_FROM = (0, 25)  # the first glyd whose saves carry "tensors": a glyd-v1 or v2 it saved without them is refused
DTYPES = {"data": "U8", "blocks": "U8", "block_base": "I32", "exc": "I32", "exc_base": "I32"}
FILES = ("config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab*", "merges.txt", "*.model", "chat_template*", "preprocessor_config.json", "processor_config.json")  # copied from the source


def key(module, buffer, weight=None):
    """The safetensors name of a pack's buffer: its module path, .glyd_, the buffer (an experts' weight: .glyd_, the
    weight's name, _, the buffer)."""
    return f"{module}.glyd_{weight + '_' if weight else ''}{buffer}"


def entry(shape, tiers, tensors, experts=None, transposed=False, layout="mma"):
    """A pack's manifest entry: its layout, its matrix's shape, its words (the tiered layout's tiers, the 12-bit
    one's sym), the tensors it holds as [(name, shape, sha256)], in their order in its rows; experts: E, an experts'
    weight's (transposed: its matrices held [in, out])."""
    e = {"layout": layout, "shape": list(shape), WORDS[layout]: [int(t) for t in tiers], "tensors": [{"name": n, "shape": list(s), "sha256": h} for n, s, h in tensors]}
    return e if experts is None else dict(e, experts=int(experts), transposed=bool(transposed))


def members(e):
    """A pack's Linears (module paths) and their rows, in order."""
    return [t["name"][: -len(".weight")] for t in e["tensors"]], [t["shape"][0] for t in e["tensors"]]


def manifest(source, packs, version, layout="mma", tensors=None):
    """glyd.json: the format, version, source, layout, packs; tensors: {name: sha256} of the tensors saved as they
    are, in their order (None: left out, as glyd 0.24 and before wrote it)."""
    form = "glyd-v3" if layout == "mma12" else "glyd-v2" if any("experts" in e for e in packs.values()) else FORMAT
    m = {"format": form, "glyd": version, "source": source, "layout": layout, "packs": packs}
    return m if tensors is None else dict(m, tensors=tensors)


def check_files(directory, m):
    """A saved checkpoint's files against its glyd.json m (verify, before its packs are decoded): glyd.json's keys its
    own (KEYS: a damaged one refused), its glyd a version, its map of sha256 ("tensors") an object, and there wherever
    its format or its glyd says it is (glyd-v3; glyd-v1 and v2 saved by glyd 0.25 on, MAP_FROM); each file's tensors
    back to back from its data's start to its end, as the safetensors library reads them; the index, where there are
    shards, naming the shard of each tensor; every tensor a pack's buffer or one whose sha256 glyd.json holds (a save
    of glyd 0.24 or before has none: those are not checked); each pack's tensors its own (its module's .weight, a
    merged group's q, k, v or gate, up under the first's path, an experts' weight's own name), in no other pack and
    none also saved as it is; and the sha256 of every tensor saved as it is. The number of tensors checked by sha256,
    and of those not checked; ValueError where any is not so. The standard library alone."""
    unknown = [k for k in m if k not in KEYS]
    if unknown:
        raise ValueError(f"glyd.json: {unknown[0]!r}, a key glyd.json does not have (damaged?)")
    version = re.match(r"(\d+)\.(\d+)", str(m.get("glyd")))
    if not version:
        raise ValueError(f"glyd.json: glyd {m.get('glyd')!r}, not a version")
    hashes = m.get("tensors")
    if "tensors" in m and not isinstance(hashes, dict):
        raise ValueError('glyd.json: "tensors" is not an object of sha256')
    if hashes is None and (m.get("format") == "glyd-v3" or (int(version[1]), int(version[2])) >= MAP_FROM):
        raise ValueError(f"glyd.json: no sha256 for the tensors saved as they are, which a {m.get('format')} of glyd {m.get('glyd')} has")
    index = os.path.join(directory, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    where = {}  # name: (file, where its bytes start, how many)
    for name in dict.fromkeys(weight_map.values()) if weight_map is not None else ["model.safetensors"]:
        if name in ("", ".", "..") or os.path.basename(name) != name:
            raise ValueError(f"{index}: a shard not a file of its directory: {name!r}")
        path = os.path.join(directory, name)
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n > min(size - 8, 100_000_000):
                raise ValueError(f"{path}: a header of {n} bytes, past the file's {size} (or 100 MB)")
            head = json.loads(f.read(n))
        pos = 0
        for a, b, k in sorted((t["data_offsets"][0], t["data_offsets"][1], k) for k, t in head.items() if k != "__metadata__"):
            if a != pos or b < a:
                raise ValueError(f"{path}: {k}'s bytes do not follow the tensor's before (at {pos}): {[a, b]}")
            if k in where:
                raise ValueError(f"{path}: {k}, in two of the shards")
            where[k], pos = (path, 8 + n + a, b - a), b
            if weight_map is not None and weight_map.get(k) != name:
                raise ValueError(f"{index}: {k} not named in {name}, its shard")
        if 8 + n + pos != size:
            raise ValueError(f"{path}: {size - 8 - n - pos} bytes past its last tensor")
    if weight_map is not None and set(weight_map) != set(where):
        raise ValueError(f"{index}: names {sorted(set(weight_map) - set(where))[:3]} that no shard holds")
    buffers, names = set(), []
    for p, e in m["packs"].items():
        owner, _, weight = p.rpartition(".")
        buffers.update(key(owner, b, weight) if "experts" in e else key(p, b) for b in LAYOUTS.get(e.get("layout"), ()))
        ts = [t.get("name") for t in e.get("tensors", [])]
        own = [p] if "experts" in e else [p + ".weight"] if len(ts) == 1 else [f"{owner}.{c}.weight" for g in GROUPS if g[0] == weight and len(g) == len(ts) for c in g]
        if ts != own:
            raise ValueError(f"glyd.json: {p}'s tensors are not its own: {ts[:3]}")
        names += ts
    if len(set(names)) != len(names):
        raise ValueError("glyd.json: a tensor held by two packs, or twice by one")
    if set(names) & set(hashes or ()):
        raise ValueError(f"glyd.json: {sorted(set(names) & set(hashes))[0]} both packed and saved as it is")
    for b in buffers - where.keys():
        raise ValueError(f"{b}: a pack's buffer, not in the safetensors")
    unchecked = 0
    for k in where.keys() - buffers:
        if hashes is None:
            unchecked += 1
        elif k not in hashes:
            raise ValueError(f"{k}: in the safetensors, but glyd.json neither packs it nor holds its sha256")
    for k, want in (hashes or {}).items():
        if k not in where:
            raise ValueError(f"{k}: glyd.json's sha256, but not in the safetensors")
        path, at, n = where[k]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            f.seek(at)
            while n:
                b = f.read(min(n, 1 << 24))
                if not b:
                    raise ValueError(f"{path}: cut short in {k}")
                h.update(b)
                n -= len(b)
        if h.hexdigest() != want:
            raise ValueError(f"{k} is other bytes than glyd.json's sha256")
    return len(hashes or {}), unchecked


def read_manifest(directory):
    """directory's glyd.json, or None where it has none."""
    path = os.path.join(directory, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    if m.get("format") not in FORMATS:
        raise ValueError(f"{path}: format {m.get('format')!r}; this glyd reads {' and '.join(FORMATS)}")
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


def copy_source_files(src, path):
    """The source's config, generation config and tokenizer files (FILES) into path: each once, its content only. The
    Hub's cache keeps them read-only, and a copy of the mode made the saved ones so (tokenizer.model, which two
    patterns match, then failed on its second copy)."""
    for f in sorted({f for pattern in FILES for f in glob.glob(os.path.join(src, pattern))}):
        shutil.copyfile(f, os.path.join(path, os.path.basename(f)))


def save_pretrained(model, path, shard_bytes=5 * 10**9, layout="mma"):
    """model (from glyd.from_pretrained or glyd.gpu.compress) saved in the
    directory path as glyd-v1, in shards of about shard_bytes: its packs in
    the tiered layout (a 12-bit pack decoded and packed again; layout="mma12":
    the 12-bit layout, glyd-v3, a tiered pack packed again), each tensor
    of each decoded for the sha256 of its bf16 bytes in glyd.json (a
    mixture of experts' weight as the model holds it); the rest as the
    model holds it; the source's config, generation config and tokenizer
    files (the model's config where the source is not at hand). The host
    holds a shard at a time, the GPU a matrix more than the model."""
    import torch
    from safetensors.torch import save_file
    from .. import __version__
    from . import kernels as g, model as gm, moe

    kind, pack = (g.Mma12, g.pack_mma12) if layout == "mma12" else (g.Mma, g.pack_mma)  # (the type exactly: an Mma12 is an Mma)
    words = WORDS[layout]
    old = glob.glob(os.path.join(path, "model*.safetensors")) + glob.glob(os.path.join(path, "model.safetensors.index.json"))
    if old and not os.path.exists(os.path.join(path, MANIFEST)) and not stored([f for f in old if f.endswith(".safetensors")]):
        raise ValueError(f"glyd: {path} holds another checkpoint; save into a directory of its own")  # (a glyd save cut short: saved over)
    for f in glob.glob(os.path.join(path, "glyd-part-*.safetensors")):
        os.remove(f)  # a save cut short's
    os.makedirs(path, exist_ok=True)
    ties = getattr(model, "all_tied_weights_keys", None) or {}
    tied, sources = set(ties), set(ties.values())  # a weight tied to another; one others are tied to (saved in bf16: tied as it loads)
    inside = {id(m.lin) for m in model.modules() if isinstance(m, gm.Merged)}
    packs, shards, part, packed = {}, [], {}, set()
    size = [0]
    hashes = {}  # the sha256 of each tensor saved as it is, in the order put

    def flush():
        if part:
            name = f"glyd-part-{len(shards) + 1:05d}.safetensors"  # renamed once the number of shards is known
            save_file(part, os.path.join(path, name), metadata={"format": "pt"})
            shards.append((name, list(part), size[0]))
            part.clear()
            size[0] = 0

    def put(name, t, buffer=False):
        part[name] = t.detach().to("cpu", copy=True).contiguous()
        if not buffer:  # saved as it is, not a pack's buffer: its sha256 in glyd.json
            hashes[name] = gm.sha256(part[name])
        size[0] += t.numel() * t.element_size()
        if size[0] >= shard_bytes:
            flush()

    with torch.no_grad():
        for name, m in model.named_modules():
            for weight, p in (getattr(m, "glyd_packs", None) or {}).items():  # a mixture of experts' weight
                packed.add(f"{name}.{weight}")  # its parameter left empty: not saved
                if f"{name}.{weight}" in tied:  # tied to another's (DiffusionGemma's encoder's to its decoder's)
                    continue
                if f"{name}.{weight}" in sources:
                    put(f"{name}.{weight}", moe.decoded(m, p, weight, gm.unpack(p)))
                    continue
                q = p if type(p) is kind else pack(gm.unpack(p))
                for b in LAYOUTS[layout]:
                    put(key(name, b, weight), getattr(q, b), True)
                shape, transposed = m.glyd_held[weight]
                packs[f"{name}.{weight}"] = entry(p.shape, getattr(q, words), [(f"{name}.{weight}", shape, moe.sha256(m, p, weight))], moe.held(m)[1], transposed, layout)
                del q
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
            if paths[0] + ".weight" in sources:  # (not merged)
                put(paths[0] + ".weight", w)
                if lin.bias is not None:
                    put(paths[0] + ".bias", lin.bias)
                continue
            p = lin.p if type(lin.p) is kind else pack(w)
            for b in LAYOUTS[layout]:
                put(key(paths[0], b), getattr(p, b), True)
            packs[paths[0]] = entry(w.shape, getattr(p, words), [(f"{n}.weight", x.shape, gm.sha256(x)) for n, x in zip(paths, w.split(rows))], layout=layout)
            if lin.bias is not None:
                for n, b in zip(paths, lin.bias.split(rows)):
                    put(f"{n}.bias", b)
            del w, p
        for k, t in model.state_dict().items():
            if k not in packed:
                put(k, t)
        flush()

    if os.path.exists(os.path.join(path, MANIFEST)):
        os.remove(os.path.join(path, MANIFEST))  # first: a save cut short from here on has shards and no manifest, which a load refuses
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
        json.dump(manifest(source, packs, __version__, layout, hashes), f, indent=1)
    src = source_dir(model.config.name_or_path, getattr(model.config, "_commit_hash", None))
    if src and os.path.realpath(src) != os.path.realpath(path):
        copy_source_files(src, path)
    if not os.path.exists(os.path.join(path, "config.json")):  # no source at hand: the model's own
        config = model.config.to_diff_dict()
        config.pop("quantization_config", None)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(path)
    return path
