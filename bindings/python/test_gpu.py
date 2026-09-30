"""glyd.gpu's parts that need no GPU and no PyTorch: fit against the site's
answers, the glyd-v1 names and manifest, import glyd without torch, and the
library's C header against the package's calls.
Where a CUDA GPU, PyTorch and transformers are at hand (else skipped):
every mixture-of-experts family of transformers as a tiny random model,
packed and saved (test_moe_families), generate() compiled where
transformers' static cache works, and the CLI's pack and verify on one.

    python test_gpu.py              (or pytest test_gpu.py)

test_gpu_site.json: for each model the site lists (getglyd.com's
data/sizes.json, 2026-09-27), the Hub metadata fit reads, trimmed (the
config's attention fields; the safetensors at the repo's top, summed into
one entry where the repo has no consolidated copy), and the site's
answers."""
import ast
import ctypes
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glyd.gpu import fit, format as fmt  # noqa: E402

fitmod = importlib.import_module("glyd.gpu.fit")


def hub(data):
    """fit's Hub reads served from test_gpu_site.json (repo ids in any case, as the Hub takes them)."""
    def get(url):
        for repo, d in data.items():
            if url.lower() == f"/api/models/{repo}".lower():
                return {"safetensors": {"parameters": d["parameters"]}}
            if url.lower() == f"/api/models/{repo}/tree/main".lower():
                return [{"type": "file", "path": p, "size": n} for p, n in d["files"]] + [{"type": "directory", "path": "original"}]
            if url.lower() == f"/{repo}/resolve/main/config.json".lower():
                return d["config"]
        raise AssertionError(url)
    return get


def test_fit_as_the_site():
    data = json.load(open(os.path.join(HERE, "test_gpu_site.json")))
    fitmod._get = hub(data)
    for repo, d in data.items():
        s = d["site"]
        for tier in fitmod.GPUS:
            f = fit(repo, gpu=tier)
            n = int(tier[:-2])
            assert f.bf16_fits == (s["gb"] is not None and s["gb"] <= n), (repo, tier, "bf16")
            assert f.glyd_fits == (s["gg"] is not None and s["gg"] <= n), (repo, tier, "glyd")
        assert round(f.kv_cache / 1e9, 3) == s["kv8"] and f.format == s["fmt"] and f.measured == (not s["est"]), repo
        # the site's GB: a measured ratio (18 models), an end-to-end run's weights (3), else 0.673
        assert (round(f.bf16_weights / 1e9, 2), round(f.glyd_weights / 1e9, 2)) == (s["bf"], s["gl"]), (repo, f.bf16_weights, f.glyd_weights)
    f = fit("Qwen/Qwen3-32B", gpu="48 GB")
    assert repr(f) == "Qwen3-32B on a 48 GB GPU: bf16 needs 69.3 GB, no; Glyd 48.3 GB, fits", repr(f)
    assert repr(fit("qwen/qwen3-32b")) == repr(f).replace("Qwen3-32B", "qwen3-32b")  # the Hub's ids in any case
    assert fit("Qwen/Qwen3-32B", gpu=81559 * 2**20).bf16_fits and repr(fit("Qwen/Qwen3-8B", gpu=16 * 10**9)).startswith("Qwen3-8B on a 16.0 GB GPU:")
    try:
        fit("Qwen/Qwen3-8B", gpu="40GB")
        raise AssertionError("an unknown GPU")
    except ValueError:
        pass


def safetensors(path, tensors, metadata=None, data=None):
    """A safetensors file: tensors {name: (dtype, shape)}, their bytes data {name: bytes}, else zeros."""
    size = {"BF16": 2, "F32": 4, "U8": 1, "I32": 4, "F8_E4M3": 1}
    h, at, body = {}, 0, b""
    for name, (dtype, shape) in tensors.items():
        n = size[dtype]
        for d in shape:
            n *= d
        h[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}
        body += (data or {}).get(name, bytes(n))
        at += n
    if metadata:
        h["__metadata__"] = metadata
    b = json.dumps(h).encode()
    with open(path, "wb") as f:
        f.write(len(b).to_bytes(8, "little") + b + body)


def test_fit_a_directory():
    with tempfile.TemporaryDirectory() as d:
        config = {"num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512, "sliding_window": 256, "layer_types": ["sliding_attention", "full_attention"] * 2}
        json.dump(config, open(os.path.join(d, "config.json"), "w"))
        safetensors(os.path.join(d, "model.safetensors"), {"a.weight": ("BF16", [1024, 512]), "b.weight": ("F32", [512])}, {"format": "pt"})
        safetensors(os.path.join(d, "consolidated.safetensors"), {"a.weight": ("BF16", [1024, 512])})
        os.makedirs(os.path.join(d, "original"))
        safetensors(os.path.join(d, "original", "model.safetensors"), {"a.weight": ("BF16", [1024, 512])})
        f = fit(d, gpu=2 * 10**9, context=1000)
        weights = os.path.getsize(os.path.join(d, "model.safetensors"))
        per = 2 * 2 * 64 * 2  # keys and values, 2 KV heads of 64, bf16: a token's bytes a layer
        assert f.bf16_weights == weights and f.format == "bf16" and not f.measured
        assert f.glyd_weights == 1024 * 512 * 2 * fitmod.RATIO + (weights - 1024 * 512 * 2)
        assert f.kv_cache == 2 * per * 1000 + 2 * per * 256  # 2 full layers for 1000 tokens, 2 sliding ones for 256
        assert f.bf16_needs == weights + f.kv_cache + fitmod.RUNTIME and f.bf16_fits and f.model == os.path.basename(d)
        json.dump({"format": fmt.FORMAT}, open(os.path.join(d, fmt.MANIFEST), "w"))
        try:
            fit(d)
            raise AssertionError("a glyd-v1 checkpoint")
        except ValueError:
            pass


def test_kv_cache():
    mla = {"num_hidden_layers": 10, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_attention_heads": 16, "hidden_size": 2048, "layer_types": ["linear_attention"] * 2 + ["full_attention"] * 8}
    assert fitmod.kv_bytes(mla, 100) == 8 * (512 + 64) * 2 * 100
    qwen2 = {"num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "hidden_size": 256, "sliding_window": 131072}  # use_sliding_window off
    assert fitmod.kv_bytes(qwen2, 10) == 2 * (2 * 4 * 64 * 2) * 10
    nextgen = {"text_config": {"num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 128, "hidden_size": 256, "full_attention_interval": 4}}
    assert fitmod.kv_bytes(nextgen, 10) == 2 * (2 * 1 * 128 * 2) * 10
    gemma = {"num_hidden_layers": 12, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "hidden_size": 256, "sliding_window": 16, "sliding_window_pattern": 6}
    assert fitmod.kv_bytes(gemma, 100) == 2 * (2 * 2 * 64 * 2) * 100 + 10 * (2 * 2 * 64 * 2) * 16
    mha = {"num_hidden_layers": 1, "num_attention_heads": 4, "hidden_size": 256}  # no num_key_value_heads: every head its own
    assert fitmod.kv_bytes(mha, 1) == 2 * 4 * 64 * 2


def test_manifest_and_names():
    assert fmt.key("model.layers.0.self_attn.q_proj", "block_base") == "model.layers.0.self_attn.q_proj.glyd_block_base"
    assert fmt.key("model.layers.0.mlp.experts", "data", "gate_up_proj") == "model.layers.0.mlp.experts.glyd_gate_up_proj_data"
    e = fmt.entry((6144, 4096), [1, 2, 0xFFFFFFFF], [("m.q_proj.weight", (4096, 4096), "aa"), ("m.k_proj.weight", (1024, 4096), "bb"), ("m.v_proj.weight", (1024, 4096), "cc")])
    assert fmt.members(e) == (["m.q_proj", "m.k_proj", "m.v_proj"], [4096, 1024, 1024])
    assert e["shape"] == [6144, 4096] and e["tiers"][2] == 0xFFFFFFFF and e["layout"] == "mma"
    m = fmt.manifest({"repo": "Qwen/Qwen3-8B", "revision": "abc"}, {"m.q_proj": e}, "0.20.0")
    assert m["format"] == "glyd-v1"  # a dense model's: glyd 0.21 reads it
    assert fmt.shard_names(1) == ["model.safetensors"] and fmt.shard_names(3)[2] == "model-00003-of-00003.safetensors"
    with tempfile.TemporaryDirectory() as d:
        assert fmt.read_manifest(d) is None
        json.dump(m, open(os.path.join(d, fmt.MANIFEST), "w"))
        assert fmt.read_manifest(d) == m  # a JSON round trip keeps the members' order
        buffers = {fmt.key("m.q_proj", b): (fmt.DTYPES[b], [n]) for b, n in zip(fmt.BUFFERS, [30720, 5000, 25])}
        safetensors(os.path.join(d, "model.safetensors"), dict(buffers, **{"m.norm.weight": ("BF16", [4096])}), {"format": "pt"})
        assert fmt.stored([os.path.join(d, "model.safetensors")]) == {k: (shape, dtype) for k, (dtype, shape) in buffers.items()}
        x = fmt.entry((1024, 256), [1, 2, 3], [("m.experts.down_proj", (8, 256, 128), "dd")], experts=8, transposed=True)  # E 8, held [E, in, out]
        assert x["experts"] == 8 and x["transposed"] and fmt.manifest(None, {"m.q_proj": e, "m.experts.down_proj": x}, "0.22.0")["format"] == "glyd-v2"
        json.dump(fmt.manifest(None, {"m.experts.down_proj": x}, "0.22.0"), open(os.path.join(d, fmt.MANIFEST), "w"))
        assert fmt.read_manifest(d)["packs"]["m.experts.down_proj"] == x  # glyd-v2, which glyd 0.21 refuses by its format
        # the 12-bit layout (glyd-v3): a pack's base hb (split byte), its buffers data, exc and exc_base
        y = fmt.entry((1024, 256), 58, [("m.o_proj.weight", (1024, 256), "ee")], layout="mma12")
        assert list(y) == ["layout", "shape", "hb", "tensors"] and y["hb"] == 58 and fmt.LAYOUTS["mma12"] == ("data", "exc", "exc_base")
        json.dump(fmt.manifest(None, {"m.o_proj": y}, "0.24.0", "mma12"), open(os.path.join(d, fmt.MANIFEST), "w"))
        assert fmt.read_manifest(d)["format"] == "glyd-v3" and fmt.read_manifest(d)["layout"] == "mma12"
        json.dump(dict(m, format="glyd-v9"), open(os.path.join(d, fmt.MANIFEST), "w"))
        try:
            fmt.read_manifest(d)
            raise AssertionError("another format")
        except ValueError:
            pass


def test_check_files():
    """verify's checks of a saved checkpoint's files (format.check_files), before its packs are decoded: a save passes
    (one file, and in two shards); then each refused: a byte of a tensor saved as it is flipped, bytes appended to a
    shard, a merged pack's member renamed in glyd.json (to the other's name, by a letter, to a saved tensor's), a pack
    holding another's tensor or one saved as it is, a tensor neither packed nor hashed there, a pack's buffer missing,
    the index naming another shard; glyd.json's map damaged (its key, its value) or gone where its glyd (0.25 on) or
    its format (glyd-v3) says it is there, its glyd not a version, a key glyd.json does not have in a save of this
    glyd or an older one. A save of glyd 0.24 without the map passes with its tensors unchecked; a newer glyd's key is
    skipped with a warning, the rest checked."""
    import glyd
    import hashlib
    import warnings
    sha = lambda b: hashlib.sha256(b).hexdigest()
    norm, emb = bytes(range(256)) * 2, bytes((7 * i) % 251 for i in range(4096))
    tensors = {fmt.key("m.gate_proj", "data"): ("U8", [2560]), fmt.key("m.gate_proj", "blocks"): ("U8", [400]), fmt.key("m.gate_proj", "block_base"): ("I32", [3]), "m.norm.weight": ("BF16", [256]), "m.emb.weight": ("BF16", [16, 128])}
    data = {"m.norm.weight": norm, "m.emb.weight": emb}
    e = fmt.entry((128, 16), [1, 2, 3], [("m.gate_proj.weight", (64, 16), "aa"), ("m.up_proj.weight", (64, 16), "bb")])
    m = fmt.manifest(None, {"m.gate_proj": e}, "0.25.0", tensors={"m.norm.weight": sha(norm), "m.emb.weight": sha(emb)})
    assert list(m) == ["format", "glyd", "source", "layout", "packs", "tensors"]

    def refused(d, mm, why):
        try:
            fmt.check_files(d, mm)
        except ValueError as err:
            assert why in str(err), err
            return
        raise AssertionError(f"not refused: {why}")

    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "model.safetensors")
        safetensors(f, tensors, {"format": "pt"}, data)
        assert fmt.check_files(d, m) == (2, 0)
        # the map gone: a save of glyd 0.24 (which has none) passes, its tensors unchecked; one of glyd 0.25, or a
        # glyd-v3, is refused; the map's key damaged ("densors": one bit), its value not an object, glyd not a version
        old = {k: v for k, v in m.items() if k != "tensors"}
        assert fmt.check_files(d, dict(old, glyd="0.24.0")) == (0, 2)
        refused(d, old, "no sha256 for the tensors saved as they are, which a glyd-v1 of glyd 0.25.0 has")
        refused(d, dict(old, glyd="0.24.0", format="glyd-v3"), "which a glyd-v3 of glyd 0.24.0 has")
        for by in (glyd.__version__, "0.21.0"):  # this glyd's save, an older one's
            refused(d, {("densors" if k == "tensors" else k): v for k, v in dict(m, glyd=by).items()}, "'densors', a key glyd.json does not have")
        newer = dict(m, glyd="99.0.0", future={"x": 1})  # a newer glyd's save: its key skipped, with a warning
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            assert fmt.check_files(d, newer) == (2, 0)
        assert any("'future', a key of glyd 99.0.0, newer than this one" in str(w.message) for w in said), [str(w.message) for w in said]
        for damaged in (None, [], "x"):
            refused(d, dict(m, tensors=damaged), '"tensors" is not an object of sha256')
        for v in ("0.x", "0/25.0", 0.25):  # (a separator one bit off; a number)
            refused(d, dict(m, glyd=v), "not a version")
        # glyd.json's glyd read as save.rs's version reads it (its test holds the same list)
        for v, want in (("0.25.0", (0, 25, 0)), ("0.25", (0, 25, 0)), ("0.25rc1.3", (0, 25, 0)), ("0.25.0.dev0", (0, 25, 0)), ("0.25.", (0, 25, 0)), ("12.3.4", (12, 3, 4))):
            assert fmt.version(v) == want, v
        for v in ("0/25.0", "0,25.0", "0x.25.0", ".25.0", "0.x", "", "v0.25", "0", "0.\u0663"):
            assert fmt.version(v) is None, v
        b = bytearray(open(f, "rb").read())
        b[-4096 - 100] ^= 1  # a byte of m.norm.weight, saved as it is
        open(f, "wb").write(bytes(b))
        refused(d, m, "m.norm.weight is other bytes")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refused(d, newer, "m.norm.weight is other bytes")  # (the rest of a newer glyd's save checked)
        safetensors(f, tensors, {"format": "pt"}, data)
        open(f, "ab").write(b"\0" * 16)
        refused(d, m, "16 bytes past its last tensor")
        safetensors(f, tensors, {"format": "pt"}, data)
        for i, other in ((1, "m.gate_proj.weight"), (1, "m.up_prok.weight"), (1, "m.norm.weight"), (0, "m.up_proj.weight")):
            mm = json.loads(json.dumps(m))
            mm["packs"]["m.gate_proj"]["tensors"][i]["name"] = other
            refused(d, mm, "m.gate_proj's tensors are not its own")
        for p, why in (("m.up_proj", "two packs, or twice"), ("m.norm", "both packed and saved as it is")):
            refused(d, dict(m, packs=dict(m["packs"], **{p: fmt.entry((64, 16), [1, 2, 3], [(p + ".weight", (64, 16), "cc")])})), why)
        refused(d, dict(m, tensors={"m.norm.weight": sha(norm)}), "m.emb.weight: in the safetensors, but glyd.json neither packs it")
        del tensors[fmt.key("m.gate_proj", "blocks")]
        safetensors(f, tensors, {"format": "pt"}, data)
        refused(d, m, "a pack's buffer, not in the safetensors")
    with tempfile.TemporaryDirectory() as d:  # two shards and their index
        names = list(tensors)
        safetensors(os.path.join(d, "model-00001-of-00002.safetensors"), {k: tensors[k] for k in names[:2]}, {"format": "pt"}, data)
        safetensors(os.path.join(d, "model-00002-of-00002.safetensors"), {k: tensors[k] for k in names[2:]}, {"format": "pt"}, data)
        tensors[fmt.key("m.gate_proj", "blocks")] = ("U8", [400])
        safetensors(os.path.join(d, "model-00001-of-00002.safetensors"), {k: tensors[k] for k in [names[0], fmt.key("m.gate_proj", "blocks"), names[1]]}, {"format": "pt"}, data)
        wm = {k: "model-00001-of-00002.safetensors" for k in [names[0], fmt.key("m.gate_proj", "blocks"), names[1]]}
        wm.update({k: "model-00002-of-00002.safetensors" for k in names[2:]})
        json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
        assert fmt.check_files(d, m) == (2, 0)
        json.dump({"metadata": {"total_size": 0}, "weight_map": dict(wm, **{"m.norm.weight": "model-00001-of-00002.safetensors"})}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
        refused(d, m, "not named in model-00002-of-00002.safetensors")


def test_c_header():
    """gpu/glyd_gpu.h, the library's C API, as _lib.py calls it: every function by the same arguments (their ctypes
    types, the stream last), its version API_VERSION; and the functions glyd_gpu.cu defines, which includes it (the
    compiler holds each definition to its declaration there). Read as text: _lib.py's argument lists run alone, as
    it imports torch."""
    gpu = os.path.join(HERE, "..", "..", "gpu")
    h = re.sub(r"/\*.*?\*/", "", open(os.path.join(gpu, "glyd_gpu.h")).read(), flags=re.S)
    names = {"_P", "_I64", "_U64", "_SZ", "_W", "_PACK", "_FAST", "_DENSE", "_ARGS", "_SIZES", "_PLAIN", "_RING", "API_VERSION", "BIG"}
    body = [n for n in ast.parse(open(os.path.join(HERE, "glyd", "gpu", "_lib.py")).read()).body if isinstance(n, ast.Assign) and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} <= names]
    lib = {"ctypes": ctypes}
    exec(compile(ast.Module(body, []), "_lib.py", "exec"), lib)
    c, P = ctypes, ctypes.c_void_p
    types = {"int64_t": c.c_int64, "uint64_t": c.c_uint64, "size_t": c.c_size_t, "double": c.c_double, "int": c.c_int, "cudaStream_t": P, "size_t*": c.POINTER(c.c_size_t)}

    def ctype(a):  # "const uint8_t* data": P; "const uint32_t tiers[3]": the host words
        t, name = " ".join(a.split()).replace(" *", "*").rsplit(" ", 1)
        t = t.replace("const ", "")
        return c.POINTER(c.c_uint32) if name.endswith("]") and t == "uint32_t" else types.get(t, P if t.endswith("*") else t)

    declared = {name: (ret, [ctype(a) for a in args.split(",")] if args.strip() != "void" else []) for ret, name, args in re.findall(r"(int|const char\*) (glyd_gpu_\w+)\(([^)]*)\);", " ".join(h.split()))}
    called = {f"glyd_gpu_{n}": ("int", a + [P]) for n, a in lib["_ARGS"].items()}
    called.update({f"glyd_gpu_{n}_workspace": ("int", [c.c_int64] * k + [c.POINTER(c.c_size_t)]) for n, k in lib["_SIZES"].items()})
    called.update({f"glyd_gpu_{n}": ("int", a) for n, a in lib["_PLAIN"].items()})  # (no stream: the routes)
    called.update({f"glyd_gpu_{n}": ("int", a) for n, a in lib["_RING"].items()})  # (the route SPLIT's ring: each list whole)
    called.update(glyd_gpu_api_version=("int", []), glyd_gpu_cuda_version=("int", []), glyd_gpu_error_string=("const char*", [c.c_int]))
    assert declared == called, [n for n in sorted(set(declared) | set(called)) if declared.get(n) != called.get(n)]
    assert int(re.search(r"#define GLYD_GPU_API_VERSION (\d+)", h).group(1)) == lib["API_VERSION"], "GLYD_GPU_API_VERSION is not _lib.py's API_VERSION"
    # the routes' numbers and a GPU's classes: kernels.py's and _lib.py's the header's
    defines = {k: int(v) for k, v in re.findall(r"#define (GLYD_GPU_\w+) (\d+)", h)}
    kern = {}
    names = {"DECODE", "GEMM", "MID", "WG", "BIG", "AHEAD", "SPLIT", "GEFORCE", "A10", "L4", "L40S", "PCIE", "GH200", "WITH_SPLIT"}
    body = [n for n in ast.parse(open(os.path.join(HERE, "glyd", "gpu", "kernels.py")).read()).body if isinstance(n, ast.Assign) and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} <= names]
    exec(compile(ast.Module(body, []), "kernels.py", "exec"), kern)
    for r in ("DECODE", "GEMM", "MID", "WG", "BIG", "AHEAD", "SPLIT"):
        assert kern[r] == defines[f"GLYD_GPU_ROUTE_{r}"], r
    assert lib["BIG"] == defines["GLYD_GPU_ROUTE_BIG"] and (kern["GEFORCE"], kern["A10"], kern["L4"], kern["L40S"], kern["PCIE"], kern["GH200"], kern["WITH_SPLIT"]) == (
        defines["GLYD_GPU_GEFORCE"], defines["GLYD_GPU_A10"], defines["GLYD_GPU_L4"], defines["GLYD_GPU_L40S"], defines["GLYD_GPU_PCIE"], defines["GLYD_GPU_GH200"], defines["GLYD_GPU_WITH_SPLIT"])
    cu = open(os.path.join(gpu, "glyd_gpu.cu")).read()
    assert set(re.findall(r"GLYD_GPU_API [^(]*?(glyd_gpu_\w+)\(", cu)) == set(declared), "glyd_gpu.cu's C API is not glyd_gpu.h's"


def test_split_route():
    """The route SPLIT (option 2) through the library where it and a GPU are here (else skipped): its rule's pins (an
    A100 SXM from 769 to 8192 tokens, a matrix over 2 x 50 M weights to 4096; a GH200 from 2048 to 8192, O and K at
    least 4096; no H100 SXM, H200 or PCIe card; its decode's SMs), asked for (GLYD_GPU_WITH_SPLIT), and without the flag
    today's routes (v0.25.1's); a GLinear made as on an A100 by it at 1024 tokens: within 1e-2 of
    fp32 and the same bits run to run, its decode bit for bit the pack's, exact bit for bit F.linear, and today's route
    (decoded, then cuBLAS) where the route cannot run."""
    torch = cuda()
    if torch is None:
        print("  (no GPU: skipped)")
        return
    import torch.nn.functional as F
    from glyd.gpu import kernels as g, model as gm

    if g.lib() is None:
        print("  (no library: skipped)")
        return
    if any(os.environ.get(v) for v in ("GLYD_SPLIT_MIN", "GLYD_SPLIT_MAX", "GLYD_SPLIT_SMS")):
        print("  (GLYD_SPLIT_* set: pins skipped)")
    else:
        w = (torch.randn(512, 1024, device="cuda") * 0.02).to(torch.bfloat16)
        q = g.pack_mma12(w)
        big = g.Mma12((131072, 1024), q.data, q.exc, q.exc_base, q.hb)  # (its shape alone read by the routes)
        wide = g.Mma12((4096, 4096), q.data, q.exc, q.exc_base, q.hb)  # (a large matrix on Hopper)
        for p, gpu, M, route, sms in [(q, 80, 768, g.BIG, 0), (q, 80, 769, g.SPLIT, 12), (q, 80, 1536, g.SPLIT, 8), (q, 80, 8192, g.SPLIT, 4), (q, 80, 8193, g.DECODE, 0),
                                      (q, 5080, 769, g.DECODE, 0), (q, 90, 2048, g.DECODE, 0), (wide, 6090, 1024, g.WG, 0), (wide, 6090, 2047, g.DECODE, 0),
                                      (wide, 6090, 2048, g.SPLIT, 12), (wide, 6090, 6144, g.SPLIT, 4), (wide, 6090, 8193, g.DECODE, 0), (wide, 5090, 2048, g.DECODE, 0),
                                      (wide, 90, 2048, g.DECODE, 0), (wide, 90, 8192, g.DECODE, 0), (q, 3089, 4096, g.DECODE, 0), (q, 4089, 4096, g.AHEAD, 0),
                                      (q, 86, 4096, g.BIG, 0), (q, 89, 4096, g.BIG, 0)]:
            assert g.route(p, gpu | g.WITH_SPLIT, M)[0] == route and g.split_sms(p, gpu | g.WITH_SPLIT, M) == sms, (p.shape, gpu, M)
            assert g.split_sms(p, gpu, M) == 0 and g.route(p, gpu, M)[0] == (g.DECODE if route == g.SPLIT else route), ("not asked: v0.25.1's route", p.shape, gpu, M)
        assert g.route(big, 80 | g.WITH_SPLIT, 4096)[0] == g.SPLIT and g.route(big, 80 | g.WITH_SPLIT, 4097)[0] == g.DECODE and g.route(big, 6090 | g.WITH_SPLIT, 8192)[0] == g.DECODE
        assert g.route(q, 80, 1024)[0] == g.DECODE and g.route(wide, 6090, 2048)[0] == g.DECODE  # (not asked: v0.25.1's routes)
        assert g.route(g.pack_mma(w), 80, 1024)[0] == g.BIG  # (the tiered layout's: never)
    torch.manual_seed(0)
    w = (torch.randn(3072, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    q = g.pack_mma12(w)
    assert torch.equal(g.mma_unpack_split(q, 16).view(torch.int16), w.view(torch.int16)), "the route SPLIT's decode: the pack's bits"
    cap, name = torch.cuda.get_device_capability, torch.cuda.get_device_name
    torch.cuda.get_device_capability, torch.cuda.get_device_name = lambda device=None: (8, 0), lambda device=None: "NVIDIA A100-SXM4-40GB"
    try:
        lin, ex = gm.GLinear(q, None), gm.GLinear(q, None, exact=True)
    finally:
        torch.cuda.get_device_capability, torch.cuda.get_device_name = cap, name
    gm.set_scratch(torch.nn.ModuleList([lin, ex]), False)
    x = torch.randn(1024, 2048, dtype=torch.bfloat16, device="cuda")
    ys = [lin(x) for _ in range(3)]
    d = w.device
    if gm.Split.of.get(d) is False:
        print("  (the route SPLIT cannot run on this GPU: its fallback alone checked)")
    ref = F.linear(x.float(), w.float())
    assert ((ys[0].float() - ref).abs().max() / ref.abs().max()).item() < 1e-2
    assert all(torch.equal(y.view(torch.int16), ys[0].view(torch.int16)) for y in ys), "the route SPLIT: the same bits run to run"
    assert torch.equal(ex(x).view(torch.int16), F.linear(x, w).view(torch.int16)), "exact never takes the route SPLIT"
    was = gm.Split.of.get(d)
    gm.Split.of[d] = False
    try:
        assert torch.equal(lin(x).view(torch.int16), F.linear(x, g.mma_unpack(q)).view(torch.int16)), "the route SPLIT off: today's route"
    finally:
        if was is None:
            del gm.Split.of[d]
        else:
            gm.Split.of[d] = was


def test_split_stress():
    """The route SPLIT's ring under stress (gpu/split_stress.py, quick): Qwen3-0.6B's, 8B's and 14B's layers through the
    ring at 769 and 2048 tokens, rings of 3 slots and of the planned count, the order queued whole and a few ahead, 3
    passes each, then GLinear's recording pass and 3 after: every product the same bits across layers, passes and slot
    counts, within 1e-2 of fp32 on the pack's weights. Where a GPU from Ampere, the prebuilt library, its split (green
    contexts) and the repository's gpu/ are here (else skipped)."""
    torch = cuda()
    script = os.path.normpath(os.path.join(HERE, "..", "..", "gpu", "split_stress.py"))
    if torch is None or not os.path.exists(script) or torch.cuda.get_device_capability()[0] < 8:
        print("  (no GPU from Ampere, or not in the repository: skipped)")
        return
    from glyd.gpu import _lib, kernels as g

    if g.lib() is None:
        print("  (no prebuilt library: the ring is the library's; skipped)")
        return
    buf = torch.empty(3 << 20, dtype=torch.uint8, device="cuda")
    ring = _lib.ring_create(buf, 1 << 20)
    r = _lib.ring_split(ring, 4)[0]
    _lib.ring_destroy(ring)
    if r:
        print(f"  (the split cannot run here, {_lib.error_string(r)}: skipped)")
        return
    out = subprocess.run([sys.executable, "-u", script, "--models", "0.6B,8B,14B", "--ms", "769,2048", "--passes", "3", "--quick"], capture_output=True, text=True, cwd=os.path.dirname(script))
    assert out.returncode == 0 and " 0 failures" in out.stdout, (out.stdout[-4000:], out.stderr[-2000:])
    print("  " + next(l for l in out.stdout.splitlines() if l.startswith("split_stress:")))


def test_split_order():
    """The route SPLIT's order (model.Split.follow, taken from model.py with no torch, its queue faked): recorded from a
    prompt, then one queue a prompt; where the Linears that take the route change with the prompt's length (an A100's
    biggest matrices to 4096 tokens alone), the queue leaves out those that do not, and the order takes in those it
    lacks (a queue again from each such one, that prompt alone)."""
    tree = ast.parse(open(os.path.join(HERE, "glyd", "gpu", "model.py")).read())
    split = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Split")
    ns = {}
    top = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("ring_chunks", "ring_slot", "ring_plan") or isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SPLIT_SLOT" for t in n.targets)]
    exec(compile(ast.Module(top + [n for n in split.body if isinstance(n, ast.FunctionDef) and n.name == "follow"], []), "model.py", "exec"), ns)

    class Fake:
        follow = ns["follow"]

        def __init__(self):
            self.order, self.rec, self.run_, self.pos, self.at = None, [], [], 0, -1

        def plan(self):
            pass

        def top_up(self, sms):
            return 0

        def start(self, M, sms, i):  # as Split.start: the order's Linears from its i-th on that take the route now
            self.starts += 1
            self.run_, self.pos = [j for j in range(i, len(self.order)) if self.order[j] in self.takes], 0
            return 0

    def prompt(s, calls):  # the queues a prompt's calls made (the Linears that take the route: calls)
        s.takes, s.starts = set(calls), 0
        for h in calls:
            assert s.follow(h, 1024, 12) == 0
        return s.starts

    full = ["qkv0", "o0", "gate_up0", "down0", "qkv1", "o1", "gate_up1", "down1"]
    short = [h for h in full if not h.startswith("gate_up")]
    s = Fake()
    assert prompt(s, full) == 0 and s.rec == full, "recorded"
    assert [prompt(s, full) for _ in range(2)] == [1, 1] and s.order == full, "followed"
    assert [prompt(s, short) for _ in range(2)] == [1, 1], "the queue leaves out those that do not take the route"
    assert prompt(s, full) == 1, "and takes them again"
    s = Fake()
    prompt(s, short)
    assert [prompt(s, short), prompt(s, full), prompt(s, full), prompt(s, short)] == [1, 3, 1, 1] and s.order == full, "the order takes in those it lacks"
    # the ring (ring_slot, ring_plan): Qwen3-8B's layers merged (q k v, gate up) in 100 MiB slots, gate up in two
    # chunks, a layer's 5 chunks ahead, its lm_head (once) not setting the slot; not merged, the gap to the last of a
    # shape up to 7; 14B's in slots of half its gate up; Qwen3-0.6B's matrices whole
    assert ns["ring_chunks"](24576, 4096, 100 << 20) == [(0, 12288), (12288, 12288)]
    layer = [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)]
    assert ns["ring_slot"](layer * 36) == ns["ring_slot"](layer * 36 + [(151936, 4096)]) == 100 << 20
    assert ns["ring_plan"](layer * 36, 100 << 20) == (6, 6)
    loose = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (12288, 4096), (4096, 12288)] * 36
    assert ns["ring_slot"](loose) == 12288 * 4096 * 2 and ns["ring_plan"](loose, 12288 * 4096 * 2) == (8, 9)
    big = [(7168, 5120), (5120, 5120), (34816, 5120), (5120, 17408)] * 40
    assert ns["ring_slot"](big) == 17408 * 5120 * 2 and ns["ring_plan"](big, 17408 * 5120 * 2) == (6, 6)
    small = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)] * 28
    assert ns["ring_slot"](small) == 6144 * 1024 * 2 and ns["ring_plan"](small, 6144 * 1024 * 2) == (5, 6)


def test_names_defined_once():
    """No module of the package defines a function or class twice at its top: the later one would take every call
    meant for the earlier (read as text, no torch)."""
    for top, _, files in os.walk(os.path.join(HERE, "glyd")):
        for f in (os.path.join(top, f) for f in files if f.endswith(".py")):
            defs = [n.name for n in ast.parse(open(f).read()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != "_"]
            assert len(defs) == len(set(defs)), (f, sorted({d for d in defs if defs.count(d) > 1}))


def test_gpu_class_by_name():
    """A GPU's class by its name as the library gives it (glyd_gpu.cu's has_word and gpu_class, compiled here alone by
    the host's C++ compiler, the classes glyd_gpu.h's) and as GLinear does (model.gpu_code, taken from model.py with
    no torch, the classes kernels.py's): the same over check_capi's names and their neighbours, "A10" among other
    letters, digits, '_', punctuation and non-ASCII (skipped where there is no C++ compiler)."""
    import types
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        print("  (no C++ compiler: skipped)")
        return
    gpu = os.path.join(HERE, "..", "..", "gpu")
    cu, h = open(os.path.join(gpu, "glyd_gpu.cu")).read(), open(os.path.join(gpu, "glyd_gpu.h")).read()
    a = cu.index("static bool has_word(")
    b = cu.index("\n", cu.index("static int gpu_class("))
    classes = re.findall(r"#define (GLYD_GPU_(?:GEFORCE|A10|L4|L40S|PCIE|GH200)) (\d+)", h)
    assert len(classes) == 6
    prog = "#include <cctype>\n#include <cstdio>\n#include <cstring>\n" + "".join(f"#define {k} {v}\n" for k, v in classes) + cu[a:b]
    prog += '\nint main() { char s[512]; while (fgets(s, sizeof s, stdin)) { s[strcspn(s, "\\n")] = 0; printf("%d\\n", gpu_class(s)); } }\n'
    tree = ast.parse(open(os.path.join(HERE, "glyd", "gpu", "model.py")).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gpu_code")
    kern = {}
    body = [n for n in ast.parse(open(os.path.join(HERE, "glyd", "gpu", "kernels.py")).read()).body if isinstance(n, ast.Assign) and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} <= {"GEFORCE", "A10", "L4", "L40S", "PCIE", "GH200"}]
    exec(compile(ast.Module(body, []), "kernels.py", "exec"), kern)
    py = {"re": re, "g": types.SimpleNamespace(GEFORCE=kern["GEFORCE"], A10=kern["A10"], L4=kern["L4"], L40S=kern["L40S"], PCIE=kern["PCIE"], GH200=kern["GH200"])}
    exec(compile(ast.Module([fn], []), "model.py", "exec"), py)
    names = ["NVIDIA A10", "NVIDIA A10-24GB", "NVIDIA A10G", "NVIDIA A100-SXM4-80GB", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4080 SUPER", "A10", "NVIDIA A10_X",
             "NVIDIA A16", "NVIDIA A2", "NVIDIA A10 PCIe", "A10 A10G", "A10G A10", "xA10", "A10x", "A10M", "NVIDIA GeForce A10", "(A10)", "A10.", "A10é", "éA10", "A10\u00a0", "", "NVIDIA H100 80GB HBM3",
             "NVIDIA L4", "L4", "NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L4 L40S", "L4-24GB", "xL4", "L4x", "NVIDIA GeForce L4", "NVIDIA A10 L4", "L4_X", "(L4)",
             "L40S", "NVIDIA L40S-48GB", "xL40S", "L40Sx", "NVIDIA L40S L4", "NVIDIA GeForce L40S", "NVIDIA L40SX", "L40S_",
             "NVIDIA H100 PCIe", "NVIDIA A100-PCIE-40GB", "pcie", "PCI", "PCIé", "xPCIEx", "NVIDIA GH200 480GB", "NVIDIA GeForce RTX 4090 PCIe", "NVIDIA L4 PCIe", "NVIDIA L40S PCIe",
             "GH200", "NVIDIA GH200 144G HBM3e", "xGH200", "GH200x", "NVIDIA GH200-96GB", "NVIDIA GH200 PCIe", "NVIDIA H200", "NVIDIA H200 NVL"]
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "cls.cpp"), "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++17", "-O1", "-o", os.path.join(d, "cls"), os.path.join(d, "cls.cpp")], check=True)
        out = subprocess.run([os.path.join(d, "cls")], input="\n".join(names) + "\n", capture_output=True, text=True, encoding="utf-8", check=True).stdout.split()
    got = dict(zip(names, map(int, out)))
    want = {n: py["gpu_code"]((8, 6), n) - 86 for n in names}
    assert len(out) == len(names) and got == want, [(n, got.get(n), want[n]) for n in names if got.get(n) != want[n]]
    pinned = ("NVIDIA A10", "NVIDIA A10G", "NVIDIA GeForce RTX 4080 SUPER", "NVIDIA L4", "NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation",
              "NVIDIA H100 PCIe", "NVIDIA A100-PCIE-40GB", "NVIDIA A10 PCIe", "NVIDIA GH200 480GB", "NVIDIA H100 80GB HBM3", "NVIDIA H200")
    assert [want[n] for n in pinned] == [kern["A10"], 0, kern["GEFORCE"], kern["L4"], kern["L40S"], 0, 0, kern["PCIE"], kern["PCIE"], kern["A10"], kern["GH200"], 0, 0], [(n, want[n]) for n in pinned]


def test_route_env():
    """The library's route variables (GLYD_WG_MIN, GLYD_WG_MAX, GLYD_MID_MIN, GLYD_DEC_MIN, GLYD_SPLIT_*) as it reads them (glyd_gpu.cu's
    route_mins parse, compiled here alone with the host's C++ compiler) and as the package holds them at import
    (model.route_env, taken from model.py with no torch): where the library reads a number, the package takes the same
    one; where it would take the value as unset, the package refuses it (ValueError), as glyd 0.24 did at int() (skipped
    where there is no C++ compiler)."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        print("  (no C++ compiler: skipped)")
        return
    cu = open(os.path.join(HERE, "..", "..", "gpu", "glyd_gpu.cu")).read()
    a = cu.index("    auto get = [](const char* name, int64_t fallback) {", cu.index("static const RouteMins& route_mins()"))
    b = cu.index("    };\n", a) + len("    };\n")
    prog = "#include <cctype>\n#include <cerrno>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <string>\n#include <iostream>\nint main() {\n" + cu[a:b]
    prog += """    std::string line;
    while (std::getline(std::cin, line)) {
        std::string v;
        for (size_t i = 0; i + 2 < line.size(); i += 3) v += (char)std::stoi(line.substr(i + 1, 2), nullptr, 16);  // a byte: its two hex digits after a backslash
        setenv("GLYD_ROUTE_TEST", v.c_str(), 1);
        long long x = get("GLYD_ROUTE_TEST", 1), y = get("GLYD_ROUTE_TEST", 2);
        if (x == y) printf("%lld\\n", x); else printf("unset\\n");
    }
}
"""
    tree = ast.parse(open(os.path.join(HERE, "glyd", "gpu", "model.py")).read())
    py = {"os": os, "re": re}
    for n in tree.body:
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROUTE_ENV" for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name == "route_env"):
            exec(compile(ast.Module([n], []), "model.py", "exec"), py)
    values = ["", " ", "12", " 12", "12 ", "\t12\n", "+12", "-12", "1e3", "2k", "0x10", "1_000", "\u0663", "12.0", "9223372036854775807", "9223372036854775808", "-9223372036854775808",
              "-9223372036854775809", "- 5", "+", "-", "++1", "1 2", "\v7\f", "0", "00017", "\u00a012", "12\u00a0", "\r\n", "17 x", " -0 "]
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "env.cpp"), "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++17", "-O1", "-o", os.path.join(d, "env"), os.path.join(d, "env.cpp")], check=True)
        escaped = ["".join(f"\\{b:02x}" for b in v.encode()) for v in values]  # every byte as \\hh
        out = subprocess.run([os.path.join(d, "env")], input="\n".join(escaped) + "\n", capture_output=True, text=True, check=True).stdout.split()
    assert len(out) == len(values), out
    for v, c in zip(values, out):
        try:
            py["route_env"]({n: v for n in py["ROUTE_ENV"]})
            got = str(int(v))
        except ValueError:
            got = "unset"
        assert got == c, (v, c, got)


def test_import_without_torch_or_library():
    """import glyd and glyd.gpu, fit, and the errors, with neither PyTorch nor the codec's library: the package
    copied without its libraries, torch blocked."""
    code = """
import sys
sys.modules["torch"] = None  # as if not installed
import glyd, glyd.gpu
assert callable(glyd.fit) and glyd.fit is glyd.gpu.fit and "torch" not in [m for m in sys.modules if sys.modules[m]]
assert glyd.compress.__module__ == "glyd" and glyd.compress.__code__.co_varnames[:4] == ("data", "level", "records", "threads")  # the codec's
for call in (lambda: glyd.from_pretrained("Qwen/Qwen3-8B"), lambda: glyd.gpu.compress(None)):
    try:
        call()
        raise AssertionError("the GPU half without torch")
    except ImportError as e:
        assert "glyd[gpu]" in str(e), e
for call, error in ((glyd.version, OSError), (lambda: glyd.no_such_name, AttributeError)):
    try:
        call()
        raise AssertionError(error)
    except error:
        pass
print("ok")
"""
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(HERE, "glyd"), os.path.join(d, "glyd"), ignore=shutil.ignore_patterns("*.dylib", "*.so", "*.dll", "__pycache__"))
        env = dict(os.environ, GLYD_LIB=os.path.join(d, "no-such-library"))
        r = subprocess.run([sys.executable, "-c", code], cwd=d, env=env, capture_output=True, text=True)
    assert r.stdout.strip() == "ok", r.stderr


def test_source_files_copied_once_and_writable():
    names = ("config.json", "tokenizer.json", "tokenizer.model")  # tokenizer.model matches "tokenizer*" and "*.model"
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        for name in names:
            with open(os.path.join(src, name), "w") as f:
                f.write(name)
            os.chmod(os.path.join(src, name), 0o444)  # as the Hub's cache keeps them
        fmt.copy_source_files(src, dst)
        for name in names:
            with open(os.path.join(dst, name)) as f:
                assert f.read() == name
            assert os.access(os.path.join(dst, name), os.W_OK)


# Every family of transformers (5.17) whose layers hold a mixture of experts: the model type a tiny config is made of
# (an expert's matrices 384 x 128 and 128 x 192: none square, so a transposition taken wrong shows in the logits),
# and what the config needs besides TINY (text: its text config's; a vision or audio tower's own, small). compress:
# packed by glyd.gpu.compress, not saved: a model glyd.from_pretrained does not load (token classification), or one
# whose checkpoint does not run in bf16.
TINY = dict(vocab_size=512, hidden_size=128, intermediate_size=192, moe_intermediate_size=192, shared_expert_intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=8, num_local_experts=8, n_routed_experts=8, moe_num_experts=8,
            num_experts_per_tok=2, moe_topk=2, n_group=1, topk_group=1, first_k_dense_replace=0, n_shared_experts=1, mlp_only_layers=[], decoder_sparse_step=1,
            kv_lora_rank=32, q_lora_rank=64, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32, expert_ffn_hidden_size=192, sliding_window=64,
            max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2, linear_key_head_dim=32, linear_value_head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_head_dim=32, linear_num_heads=4, index_head_dim=32, index_n_heads=4, index_topk=16, index_kpool=4)
TOWER = dict(depth=1, num_hidden_layers=1, hidden_size=64, num_heads=2, num_attention_heads=2, intermediate_size=128, out_hidden_size=128, projection_intermediate_size=128)
LIN, SPARSE = ["linear_attention", "full_attention"], ["sparse", "sparse"]
ROPE = dict(rope_type="default", rope_theta=10000.0)
FAMILIES = {
    "afmoe": {}, "axk1": {}, "axk2": dict(num_key_value_heads=4), "cohere2_moe": {}, "deepseek_v2": {}, "deepseek_v3": {}, "deepseek_v32": dict(num_key_value_heads=4),
    "deepseek_v4": dict(compress=True),  # (its bf16 checkpoint as transformers loads it hands fp32 norms' outputs to bf16 Linears)
    "diffusion_gemma": dict(text=dict(top_k_experts=2)), "dots1": {}, "ernie4_5_moe": {}, "exaone_moe": {}, "flex_olmo": {},
    "gemma4": dict(text=dict(enable_moe_block=True, top_k_experts=2)), "glm4_moe": {}, "glm4_moe_lite": dict(num_key_value_heads=4), "glm_moe_dsa": dict(num_key_value_heads=4),
    "glm4v_moe": dict(text=dict(rope_parameters=dict(ROPE, mrope_section=[2, 3, 3], partial_rotary_factor=0.5))),
    "glm5_next": dict(text=dict(num_key_value_heads=4, qk_rope_head_dim=0, qk_nope_head_dim=32, mlp_layer_types=SPARSE, layer_types=LIN, indexer_types=["full", "full"])),
    "gpt_oss": {}, "granitemoe": {}, "granitemoeshared": {}, "granitemoe_swa": {}, "hunyuan_v1_moe": {}, "hy_v3": {}, "hy_v4": {}, "inkling_text": {},
    "granitemoehybrid": dict(layer_types=["mamba", "attention"], mamba_n_heads=8, mamba_d_head=32, mamba_d_state=16, mamba_chunk_size=16),
    "jamba": dict(attn_layer_period=2, attn_layer_offset=1, expert_layer_period=1, expert_layer_offset=0, use_mamba_kernels=False, mamba_d_state=16),
    "kimi_linear": dict(layer_types=LIN, mlp_layer_types=SPARSE), "laguna": {}, "lfm2_moe": dict(layer_types=["conv", "full_attention"], num_dense_layers=0),
    "mellum": {}, "mimo_v2_flash": {}, "minimax": {}, "minimax_m2": {}, "mistral4": {}, "mixtral": {}, "nemotron_h": {}, "olmoe": {}, "phimoe": {},
    "minimax_m3_vl": dict(text_config=dict({k: TINY[k] for k in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "num_experts_per_tok", "num_local_experts", "pad_token_id", "bos_token_id", "eos_token_id")},
                                                model_type="minimax_m3_vl_text", dense_intermediate_size=128, shared_intermediate_size=64, index_n_heads=2, index_head_dim=32, index_block_size=4, index_topk_blocks=2, index_local_blocks=1),
                          vision_config=dict(model_type="minimax_m3_vl_vision", hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2)),
    "openai_privacy_filter": dict(compress=True), "qwen2_moe": {}, "qwen3_5_moe": dict(text=dict(layer_types=LIN)), "qwen3_moe": {}, "qwen3_next": dict(layer_types=LIN),
    "qwen3_omni_moe_thinker": dict(audio_config=dict(encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=128, d_model=64, output_dim=128, downsample_hidden_size=32),
                                   vision_config=dict(TOWER, deepstack_visual_indexes=[0])),
    "qwen3_vl_moe": {}, "solar_open": {}, "zaya": dict(num_experts_per_tok=1),
    "qwen4_exp": dict(text=dict(layer_types=["linear_attention", "qwen_sparse_attention"], indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=16, indexer_compress_ratio=4)),
    "deepseek_ocr2": dict(text=dict(mlp_layer_types=SPARSE), vision_config=dict(sam_config=dict(hidden_size=64, output_channels=64, num_hidden_layers=1, num_attention_heads=2, mlp_dim=128, global_attn_indexes=[0], downsample_channels=[64, 128]),
                                                                            encoder_config=dict(hidden_size=128, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))),
    "ernie4_5_vl_moe": dict(text=dict(moe_intermediate_size=[192, 192], moe_k=2, rope_parameters=dict(ROPE, mrope_section=[6, 6, 4]))),
    # run by the family's own code (moe.OWN)
    "aria_text": {}, "jetmoe": {}, "llama4_text": dict(intermediate_size_mlp=128, interleave_moe_layer_step=1, num_experts_per_tok=1),
    "dbrx": dict(d_model=128, n_heads=4, n_layers=2, max_seq_len=512, attn_config=dict(kv_n_heads=2, rope_theta=10000.0, clip_qkv=8.0), ffn_config=dict(hidden_size=128, ffn_hidden_size=64, moe_num_experts=8, moe_top_k=2)),
    "longcat_flash": dict(num_layers=1, qk_nope_head_dim=32, qk_rope_head_dim=16, head_dim=16, num_key_value_heads=4, zero_expert_num=2),
    "step3p7": dict(text=dict(share_expert_dim=64, mlp_layer_types=SPARSE), vision_config=dict(TOWER, image_size=56, patch_size=14)),
    # experts as Linears (encoder-decoder models: packed by compress)
    "switch_transformers": dict(compress=True, d_model=128, d_kv=32, d_ff=128, num_layers=2, num_sparse_encoder_layers=1, num_decoder_layers=2, num_sparse_decoder_layers=1, num_heads=4),
    "nllb-moe": dict(compress=True, d_model=128, encoder_layers=2, decoder_layers=2, encoder_ffn_dim=128, decoder_ffn_dim=128, encoder_attention_heads=4, decoder_attention_heads=4, encoder_sparse_step=1, decoder_sparse_step=1),
}  # Doge's experts, rows of two nn.Embedding, are packed as embeddings; its MoE layer does not run in 5.17 (a tuple where its layer takes a tensor)


def cuda():
    """torch with a CUDA GPU and transformers, else None (the tests that need them skipped)."""
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def tiny(kind, over):
    """A tiny config of model type kind: TINY's fields it has, over's (text: its text config's), its towers' TOWER's."""
    import dataclasses
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    def small(cls, values):
        names = ({f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()) | set(getattr(cls, "attribute_map", {}))
        return {k: v for k, v in values.items() if k in names}

    cls, over = CONFIG_MAPPING[kind], dict(over)
    kw = small(cls, TINY)
    for s, sub in (getattr(cls, "sub_configs", None) or {}).items():
        if s == "text_config":
            kw[s] = dict(small(sub, TINY), **over.pop("text", {}))
        elif s == "vision_config":
            kw[s] = small(sub, TOWER)
    over.pop("compress", None)
    return AutoConfig.for_model(kind, **dict(kw, **over))


def experts(model):
    """model's experts' weights: [(name, packed)]."""
    import torch.nn as nn
    from glyd.gpu import moe
    from glyd.gpu.model import GLinear

    out = []
    for path, m in model.named_modules():
        if moe.held(m):
            out += [(f"{path}.{n}", n in (getattr(m, "glyd_packs", None) or {}) and getattr(m, n).numel() == 0) for n in moe.held(m)[0]]
        elif isinstance(m, nn.ModuleDict) and type(m).__name__.endswith("Experts"):  # Linears (Switch Transformers, NLLB-MoE)
            out += [(f"{path}.{n}", isinstance(c, GLinear)) for n, c in m.named_modules() if isinstance(c, (nn.Linear, GLinear))]
    return out


def tiny_model(torch, kind, over):
    """A tiny random model of kind on the GPU in fp32 (its weights N(0, 0.05), seeded): (the model, its Auto class,
    its config, token ids [4, 16], the forward's other arguments)."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForSeq2SeqLM, AutoModelForTokenClassification

    cfg = tiny(kind, over)
    torch.manual_seed(0)  # what the model's initialization draws (attention sinks ...)
    with torch.device("cuda"):
        for auto in (AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForTokenClassification, AutoModelForSeq2SeqLM):
            try:
                model = auto.from_config(cfg, dtype=torch.float32).eval()
                break
            except ValueError:
                pass
    gen = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, cfg.get_text_config().vocab_size, (4, 16), device="cuda", generator=gen)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2 and p.is_floating_point():
                p.normal_(0, 0.05, generator=gen)
    kw = dict(decoder_input_ids=ids) if kind == "diffusion_gemma" or cfg.is_encoder_decoder else {}  # a decoder's tokens (DiffusionGemma's canvas, else drawn at random)
    return model, auto, cfg, ids, kw


def moe_family(torch, kind, over, d):
    """A tiny random model of kind on the GPU (fp32, then bf16) saved as a checkpoint in d, loaded by
    glyd.from_pretrained: every expert's weight packed, in fewer bytes; logits as close to fp32's as bf16's (the
    median over positions of a position's largest difference, as a routing near-tie moves a whole position); exact
    bit for bit the bf16 model transformers loads from it; saved as glyd-v1 and loaded (verify: every tensor's
    sha256), its logits the model's packed at load bit for bit, in both layouts; exact from it bf16's again. The two
    medians, Glyd's (the tiered layout's) and bf16's."""
    import copy
    import glyd
    from glyd.gpu import format as fmt, model as gm, moe

    fp32, auto, cfg, ids, kw = tiny_model(torch, kind, over)
    run = lambda model: model(ids, **kw).logits.float()
    with torch.no_grad():
        l32 = run(fp32)
        bf16 = fp32.to(torch.bfloat16)
        experts_bytes = sum(p.numel() * 2 for n, p in bf16.named_parameters() if n in {e for e, _ in experts(bf16)})
        err = lambda l: (l - l32).abs().amax(-1).median().item()
        src, dst = os.path.join(d, kind), os.path.join(d, kind + "-glyd")
        if over.get("compress"):  # not a model from_pretrained loads: packed in place, exact from a copy
            ref, lb = copy.deepcopy(bf16), run(bf16)
            import glyd.gpu as gg
            g = gg.compress(bf16)
            packed = experts(g)
            assert packed and all(p for _, p in packed), (kind, packed)
            assert not moe.nbytes(g) or moe.nbytes(g) < 0.8 * experts_bytes, (kind, moe.nbytes(g), experts_bytes)  # (Linears: their GLinears' packs)
            eg = err(run(g))
            assert eg <= 1.5 * err(lb), (kind, eg, err(lb))
            assert torch.equal(run(gg.compress(ref, exact=True)), lb), kind
            return eg, err(lb)
        bf16.save_pretrained(src)
        del fp32, bf16
        lb = run(auto.from_pretrained(src, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval())  # (LongCat's differs from the model saved)
        for layout in ("mma", "mma12"):
            g = glyd.from_pretrained(src, layout=layout)
            packed = experts(g)
            assert packed and all(p for _, p in packed), (kind, packed)
            assert moe.nbytes(g) < 0.8 * experts_bytes, (kind, moe.nbytes(g), experts_bytes)
            for m in g.modules():  # the sha256 glyd.json holds, an expert at a time: the whole weight's as held
                for n, p in (getattr(m, "glyd_packs", None) or {}).items():
                    assert moe.sha256(m, p, n) == gm.sha256(moe.decoded(m, p, n, gm.unpack(p))), (kind, n)
            lg = run(g)
            assert err(lg) <= 1.5 * err(lb), (kind, layout, err(lg), err(lb))
            if layout == "mma":
                medians = err(lg), err(lb)
            glyd.save_pretrained(g, dst)
            del g
            g = glyd.from_pretrained(dst, layout=layout, verify=True)
            assert g.config.quantization_config.verified >= sum(len(e["tensors"]) for e in fmt.read_manifest(dst)["packs"].values()), kind  # and the embeddings packed again
            assert torch.equal(run(g), lg), (kind, layout)
            del g
        for path in (src, dst):
            assert torch.equal(run(glyd.from_pretrained(path, exact=True)), lb), (kind, path)
        # a bf16 model of a class Glyd took over (moe.OWN) runs the class's own code
        assert torch.equal(run(auto.from_pretrained(src, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()), lb), kind
        torch.cuda.empty_cache()
    return medians


def test_moe_families():
    torch = cuda()
    if torch is None:
        print("test_moe_families: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    with tempfile.TemporaryDirectory() as d:
        for kind, over in FAMILIES.items():
            moe_family(torch, kind, over, d)
            shutil.rmtree(d)
            os.makedirs(d)


def test_compiled_generate_where_the_static_cache_works():
    """glyd.from_pretrained's generate() compiled (model.fast_generate) but where transformers 5.17's static cache
    fails, which runs eager from the start (no failing first call, no warning): Llama 4, and a model with multi-head
    latent attention whose config has fewer key/value heads than heads (DeepSeek V3's as the tiny config makes it);
    with as many (as the released checkpoints have), and Qwen3-MoE, compiled."""
    torch = cuda()
    if torch is None:
        print("test_compiled_generate_where_the_static_cache_works: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    import warnings
    import glyd
    from glyd.gpu import model as gm

    with tempfile.TemporaryDirectory() as d:
        for kind, kv, fast in (("deepseek_v3", 2, False), ("deepseek_v3", 4, True), ("llama4_text", None, False), ("qwen3_moe", None, True)):
            model, auto, cfg, ids, kw = tiny_model(torch, kind, dict(FAMILIES[kind], **({"num_key_value_heads": kv} if kv else {})))
            model.to(torch.bfloat16).save_pretrained(d)
            del model
            g = glyd.from_pretrained(d)
            x = ids[:1, :8]
            with warnings.catch_warnings(record=True) as w, torch.no_grad():
                warnings.simplefilter("always")
                g.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=4, do_sample=False, pad_token_id=0)
            said = [str(m.message) for m in w if "glyd" in str(m.message)]
            fast = fast and torch.__version__ >= "2.13"  # (below it, eager as in glyd 0.23)
            assert ("glyd_fast" in g.__dict__, g in gm._COMPILED, said) == (fast, fast, []), (kind, kv, said)
            del g
            torch._dynamo.reset()


def test_compiled_generate_from_torch_2_13():
    """generate() compiled (model.fast_generate) from PyTorch 2.13 on: with torch.__version__ an older one's, the model
    is left as it was (its generate() eager, as in glyd 0.23); with the one installed (2.13 or later) it is set up."""
    torch = cuda()
    if torch is None:
        print("test_compiled_generate_from_torch_2_13: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    import glyd.gpu as gg
    from glyd.gpu import model as gm
    from torch.torch_version import TorchVersion

    model, auto, cfg, ids, kw = tiny_model(torch, "qwen3_moe", FAMILIES["qwen3_moe"])
    g = gg.compress(model.to(torch.bfloat16), compile=False)
    installed = torch.__version__
    try:
        for old in ("2.5.1", "2.11.0+cu128", "2.12.1", "2.13.0a0+git1234"):
            torch.__version__ = TorchVersion(old)
            assert "glyd_fast" not in gm.fast_generate(g).__dict__, old
    finally:
        torch.__version__ = installed
    assert ("glyd_fast" in gm.fast_generate(g).__dict__) == (installed >= "2.13"), installed


def test_compiled_generate_eager_where_transformers_differs():
    """fast_generate's probe of the helpers the gate reads (transformers 5.17's _prepare_generation_config and
    get_generation_mode): with the first gone, one warning, and the model left as it was (its generate() eager);
    below PyTorch 2.13 no probe, nothing said. A generation of the model's own with no search modes (DiffusionGemma's
    config refuses get_generation_mode): left as it was, nothing said."""
    torch = cuda()
    if torch is None:
        print("test_compiled_generate_eager_where_transformers_differs: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    import warnings
    import glyd.gpu as gg
    from glyd.gpu import model as gm

    model, auto, cfg, ids, kw = tiny_model(torch, "qwen3_moe", FAMILIES["qwen3_moe"])
    g = gg.compress(model.to(torch.bfloat16), compile=False)
    g._prepare_generation_config = None  # (the model's own, over its class's: gone)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gm.fast_generate(g)
    said = [str(m.message) for m in w if "glyd" in str(m.message)]
    assert "glyd_fast" not in g.__dict__ and len(said) == (torch.__version__ >= "2.13"), said
    del g._prepare_generation_config
    assert ("glyd_fast" in gm.fast_generate(g).__dict__) == (torch.__version__ >= "2.13")
    model, auto, cfg, ids, kw = tiny_model(torch, "diffusion_gemma", FAMILIES["diffusion_gemma"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = gg.compress(model.to(torch.bfloat16))
    said = [str(m.message) for m in w if "glyd" in str(m.message)]
    assert "glyd_fast" not in g.__dict__ and not said, said


def test_hooks_put_before_the_packs():
    """The families run by their own code (moe.OWN) packed after accelerate's hooks were put on their modules, as a
    device map over several GPUs puts them before the packs are made (from_pretrained): the logits as without the
    hooks, fused and exact. In a process of its own, where no class has been taken over yet (a hook put after
    captures the forward that took over, the case that works anyway)."""
    torch = cuda()
    if torch is None:
        print("test_hooks_put_before_the_packs: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    code = """
import copy, torch
import test_gpu as t
import glyd.gpu as gg
from accelerate.hooks import ModelHook, add_hook_to_module
for kind in ("llama4_text", "dbrx", "aria_text", "jetmoe", "step3p7", "longcat_flash", "qwen3_moe"):
    model, auto, cfg, ids, kw = t.tiny_model(torch, kind, t.FAMILIES[kind])
    run = lambda m: m(ids, **kw).logits.float()
    with torch.no_grad():
        bf16 = model.to(torch.bfloat16)
        hooked = [copy.deepcopy(bf16), copy.deepcopy(bf16)]
        for m in hooked:
            for mod in m.modules():
                add_hook_to_module(mod, ModelHook())
        gg.compress(hooked[0])
        gg.compress(hooked[1], exact=True)
        plain = gg.compress(copy.deepcopy(bf16))
        assert torch.equal(run(hooked[0]), run(plain)), kind
        assert torch.equal(run(hooked[1]), run(bf16)), kind
print("ok")
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env, capture_output=True, text=True)
    assert r.stdout.strip().endswith("ok"), r.stderr[-3000:]


def test_gpu_code_loads_the_library():
    """_lib.gpu() and the routes as a process's first calls, before any kernel has loaded the library: they load it
    (GLYD_GPU_LIB, as the kernels do) and give the code gpu_code gives and a route; with no library anywhere, an OSError
    that says so, not a KeyError; with the kernels a JIT build's (a module without the library's functions), an OSError
    that says that."""
    torch = cuda()
    if torch is None:
        print("test_gpu_code_loads_the_library: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    import glyd.gpu.kernels as g
    from glyd.gpu import model as gm

    if g.library() is None:
        print("test_gpu_code_loads_the_library: skipped (no prebuilt library: GLYD_GPU_LIB)")
        return
    first = "from glyd.gpu import _lib; print(_lib.gpu(), _lib.mma12_route(_lib.gpu(), 512, 1024, 1)[0])"
    r = subprocess.run([sys.executable, "-c", first], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    code, route = map(int, r.stdout.split())
    assert code == gm.gpu_code(torch.cuda.get_device_capability(), torch.cuda.get_device_name()) and route == g.GEMM, r.stdout
    env = {k: v for k, v in os.environ.items() if k != "GLYD_GPU_LIB"}
    env["PYTHONPATH"] = os.pathsep.join([HERE] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    r = subprocess.run([sys.executable, "-c", first], capture_output=True, text=True, env=env, cwd=tempfile.gettempdir())
    beside = os.path.exists(os.path.join(os.path.dirname(g.__file__), f"libglyd_gpu_cuda{torch.version.cuda.split('.')[0]}.so"))
    assert beside or (r.returncode != 0 and "OSError: no Glyd GPU library" in r.stderr and "KeyError" not in r.stderr), r.stderr[-2000:]
    jit = "import types; from glyd.gpu import _lib, kernels; kernels._ext = types.ModuleType('glyd_gpu'); _lib.gpu()"
    r = subprocess.run([sys.executable, "-c", jit], capture_output=True, text=True, env=env, cwd=tempfile.gettempdir())
    assert r.returncode != 0 and "OSError: glyd_gpu_gpu: the Glyd GPU library is not loaded" in r.stderr, r.stderr[-2000:]


def test_packed_weight_view():
    """A packed Linear's and embedding's .weight as a model's own code reads it: its dtype, device and shape; rows by
    index; a torch function on the matrix decoded, its arguments nested too (torch.cat's list)."""
    torch = cuda()
    if torch is None:
        print("test_packed_weight_view: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    import torch.nn as nn
    import glyd.gpu as gg

    torch.manual_seed(0)
    with torch.device("cuda"):
        m = nn.Sequential(nn.Embedding(256, 128), nn.Linear(128, 64)).to(torch.bfloat16)
    e, w = m[0].weight.detach().clone(), m[1].weight.detach().clone()
    emb, lin = gg.compress(m)
    assert (lin.weight.dtype, lin.weight.shape, lin.weight.device) == (torch.bfloat16, w.shape, w.device) and not lin.weight.requires_grad
    assert torch.equal(torch.cat([lin.weight, lin.weight]), torch.cat([w, w])) and torch.equal(lin.weight.float(), w.float())
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(nn.functional.linear(x, lin.weight), nn.functional.linear(x, w))
    assert torch.equal(emb.weight[3, :], e[3]) and torch.equal(emb.weight[-1], e[-1]) and torch.equal(emb.weight[torch.tensor([1, 5], device="cuda")], e[[1, 5]])


def test_quantized_checkpoint_refused():
    """A checkpoint quantized already (its config's quantization_config: gpt-oss's MXFP4, the FP8 releases) refused by
    glyd.from_pretrained with what to load instead, before a weight is read."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("test_quantized_checkpoint_refused: skipped (no PyTorch or transformers)")
        return
    import glyd
    with tempfile.TemporaryDirectory() as d:
        json.dump({"model_type": "qwen3_moe", "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}, open(os.path.join(d, "config.json"), "w"))
        try:
            glyd.from_pretrained(d)
            raise AssertionError("a quantized checkpoint")
        except ValueError as e:
            assert "quantized already (fp8)" in str(e) and "bf16 release" in str(e), e


def test_cli_pack_and_verify():
    """python -m glyd.gpu pack, verify and fit on a tiny mixture of experts (Qwen3-MoE's): fit counts its experts'
    bytes, and refuses the glyd-v1 checkpoint (fit the source); a save cut short (packs, no glyd.json) refused by
    verify, and saved over by pack."""
    torch = cuda()
    if torch is None:
        print("test_cli_pack_and_verify: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    from transformers import AutoModelForCausalLM
    with tempfile.TemporaryDirectory() as d:
        with torch.device("cuda"):
            AutoModelForCausalLM.from_config(tiny("qwen3_moe", {}), dtype=torch.bfloat16).save_pretrained(os.path.join(d, "src"))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        for args, says in ((["pack", os.path.join(d, "src"), os.path.join(d, "out")], "tensors packed and checked"), (["verify", os.path.join(d, "out")], "tensors decode to glyd.json's sha256")):
            r = subprocess.run([sys.executable, "-m", "glyd.gpu", *args], env=env, capture_output=True, text=True)
            assert r.returncode == 0 and says in r.stdout, r.stderr[-2000:]
        assert sum("experts" in e for e in fmt.read_manifest(os.path.join(d, "out"))["packs"].values()) == 4  # 2 layers' gate and up, down
        f = fit(os.path.join(d, "src"), gpu=10**9)
        assert f.bf16_weights == sum(os.path.getsize(os.path.join(d, "src", n)) for n in os.listdir(os.path.join(d, "src")) if n.endswith(".safetensors"))
        r = subprocess.run([sys.executable, "-m", "glyd.gpu", "fit", os.path.join(d, "out")], env=env, capture_output=True, text=True)
        assert r.returncode != 0 and "fit the source" in r.stderr, r.stderr[-2000:]
        os.remove(os.path.join(d, "out", fmt.MANIFEST))  # as a save over it cut short once the manifest is gone: refused, then saved over
        for args, ok in ((["verify", os.path.join(d, "out")], False), (["pack", os.path.join(d, "src"), os.path.join(d, "out")], True)):
            r = subprocess.run([sys.executable, "-m", "glyd.gpu", *args], env=env, capture_output=True, text=True)
            assert (r.returncode == 0) == ok and (ok or "a save cut short" in r.stderr), r.stderr[-2000:]
        # saved in the 12-bit layout (glyd-v3): verified; loaded for the 12-bit layout, its packs are the saved ones as
        # they are, each the pack of its weights there (and loaded tiered, packed again, its tensors as saved)
        for args, says in ((["pack", os.path.join(d, "src"), os.path.join(d, "out12"), "--layout", "mma12"], "tensors packed and checked"), (["verify", os.path.join(d, "out12")], "tensors decode to glyd.json's sha256")):
            r = subprocess.run([sys.executable, "-m", "glyd.gpu", *args], env=env, capture_output=True, text=True)
            assert r.returncode == 0 and says in r.stdout, r.stderr[-2000:]
        m12 = fmt.read_manifest(os.path.join(d, "out12"))
        assert m12["format"] == "glyd-v3" and all(e["layout"] == "mma12" and type(e["hb"]) is int and "sym" not in e for e in m12["packs"].values())
        from glyd.gpu import hf, kernels as g, model as gm
        saved, fresh = hf.from_pretrained(os.path.join(d, "out12"), layout="mma12"), hf.from_pretrained(os.path.join(d, "src"), layout="mma12")
        packs = lambda model: [m.p for m in model.modules() if isinstance(m, gm.GLinear)] + [p for m in model.modules() for p in (getattr(m, "glyd_packs", None) or {}).values()]
        assert len(packs(saved)) == len(packs(fresh)) == len(m12["packs"])
        for a, b in zip(packs(saved), packs(fresh)):
            assert type(a) is g.Mma12 and a.hb == b.hb and a.sym == b.sym and all(torch.equal(getattr(a, t), getattr(b, t)) for t in ("data", "exc", "exc_base"))
        del saved, fresh
        tiered = hf.from_pretrained(os.path.join(d, "out12"), layout="mma", verify=True)
        assert all(type(p) is g.Mma for p in packs(tiered))  # packed again, tiered
        assert tiered.config.quantization_config.verified >= sum(len(e["tensors"]) for e in m12["packs"].values())  # (and the embeddings packed as it loads)
        del tiered
        # a glyd-v3 save of the 12-bit layout before split byte (never released): its packs' words "sym", no "hb";
        # refused as it loads, never decoded as split byte
        old = os.path.join(d, "old12")
        shutil.copytree(os.path.join(d, "out12"), old)
        mm = json.loads(open(os.path.join(old, fmt.MANIFEST)).read())
        for e in mm["packs"].values():
            e["sym"] = [e.pop("hb")] * 4
        json.dump(mm, open(os.path.join(old, fmt.MANIFEST), "w"))
        try:
            hf.from_pretrained(old, layout="mma12")
            raise AssertionError("a 12-bit save without hb loaded")
        except ValueError as e:
            assert "no hb" in str(e), e
        shutil.rmtree(old)
        # verify on a changed copy: a byte of a tensor saved as it is flipped, bytes appended to a shard, a merged
        # pack's member renamed by a letter in glyd.json (k_proj to k_prok), the map's key renamed: each refused
        m = fmt.read_manifest(os.path.join(d, "out"))
        assert "model.norm.weight" in m["tensors"] and m["format"] == "glyd-v2"
        group = next(p for p, e in m["packs"].items() if len(e["tensors"]) > 1)
        for change, says in (("flip", "model.norm.weight is other bytes"), ("append", "bytes past its last tensor"), ("rename", f"{group}'s tensors are not its own"), ("key", "'densors', a key glyd.json does not have")):
            bad = os.path.join(d, "bad-" + change)
            shutil.copytree(os.path.join(d, "out"), bad)
            f = os.path.join(bad, "model.safetensors")
            if change == "flip":
                h = fmt.header(f)
                at = 8 + int.from_bytes(open(f, "rb").read(8), "little") + h["model.norm.weight"]["data_offsets"][0]
                b = bytearray(open(f, "rb").read())
                b[at] ^= 0x10
                open(f, "wb").write(bytes(b))
            elif change == "append":
                open(f, "ab").write(bytes(8))
            elif change == "rename":
                mm = json.loads(open(os.path.join(bad, fmt.MANIFEST)).read())
                mm["packs"][group]["tensors"][1]["name"] = mm["packs"][group]["tensors"][1]["name"].replace("k_proj", "k_prok")
                assert "k_prok" in mm["packs"][group]["tensors"][1]["name"]
                json.dump(mm, open(os.path.join(bad, fmt.MANIFEST), "w"))
            else:  # the map's key "densors" (one bit), which a save before glyd 0.25 would not have had either
                mm = json.loads(open(os.path.join(bad, fmt.MANIFEST)).read())
                mm = {("densors" if k == "tensors" else k): v for k, v in mm.items()}
                json.dump(mm, open(os.path.join(bad, fmt.MANIFEST), "w"), indent=1)
            r = subprocess.run([sys.executable, "-m", "glyd.gpu", "verify", bad], env=env, capture_output=True, text=True)
            assert r.returncode != 0 and says in r.stderr, (change, r.stderr[-2000:])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
