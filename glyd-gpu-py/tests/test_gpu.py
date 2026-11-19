"""glyd_gpu's parts that need no GPU and no PyTorch: fit against the site's
answers, the glyd-v1 names and manifest, import glyd without torch, and the
library's C header against the package's calls.
Where a CUDA GPU, PyTorch and transformers are at hand (else skipped):
every mixture-of-experts family of transformers as a tiny random model,
packed and saved (test_moe_families), generate() compiled where
transformers' static cache works, and the CLI's pack and verify on one.

    python test_gpu.py              (or pytest test_gpu.py)

The glyd_gpu it runs: the installed one (a wheel's compiled modules), or this checkout's sources with PYTHONPATH=.. (the tests that read
the sources as text, and the C header and .cu, need the checkout either way).

test_gpu_site.json: for each model the site lists (getglyd.com's
data/sizes.json, 2026-09-27), the Hub metadata fit reads, trimmed (the
config's attention fields; the safetensors at the repo's top, summed into
one entry where the repo has no consolidated copy), and the site's
answers."""
import ast
import ctypes
import importlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "glyd_gpu")  # this checkout's sources: the tests that read them as text
import glyd_gpu  # noqa: E402 (PYTHONPATH=.. runs this checkout's package; else the one installed, a wheel's compiled one)
from glyd_gpu import fit, format as fmt  # noqa: E402

PKG = os.path.dirname(os.path.abspath(glyd_gpu.__file__))  # the package that imported
ROOT = os.path.dirname(PKG)  # what PYTHONPATH needs for a subprocess to import the same one
fitmod = importlib.import_module("glyd_gpu.fit")


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
        for band in fitmod.GPUS:
            f = fit(repo, gpu=band)
            n = int(band[:-2])
            assert f.bf16_fits == (s["gb"] is not None and s["gb"] <= n), (repo, band, "bf16")
            assert f.glyd_fits == (s["gg"] is not None and s["gg"] <= n), (repo, band, "glyd")
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
    assert m["format"] == "glyd-v4" and fmt.OLD_FORMATS == ("glyd-v1", "glyd-v2")  # the tiered layout of glyd 0.28, a dense model's or an experts' (glyd-v1 and glyd-v2 are the packs of 0.20 to 0.27: packed again as they load)
    assert fmt.shard_names(1) == ["model.safetensors"] and fmt.shard_names(3)[2] == "model-00003-of-00003.safetensors"
    with tempfile.TemporaryDirectory() as d:
        assert fmt.read_manifest(d) is None
        json.dump(m, open(os.path.join(d, fmt.MANIFEST), "w"))
        assert fmt.read_manifest(d) == m  # a JSON round trip keeps the members' order
        buffers = {fmt.key("m.q_proj", b): (fmt.DTYPES[b], [n]) for b, n in zip(fmt.BUFFERS, [30720, 5000, 25])}
        safetensors(os.path.join(d, "model.safetensors"), dict(buffers, **{"m.norm.weight": ("BF16", [4096])}), {"format": "pt"})
        assert fmt.stored([os.path.join(d, "model.safetensors")]) == {k: (shape, dtype) for k, (dtype, shape) in buffers.items()}
        x = fmt.entry((1024, 256), [1, 2, 3], [("m.experts.down_proj", (8, 256, 128), "dd")], experts=8, transposed=True)  # E 8, held [E, in, out]
        assert x["experts"] == 8 and x["transposed"] and fmt.manifest(None, {"m.q_proj": e, "m.experts.down_proj": x}, "0.28.0")["format"] == "glyd-v4"
        json.dump(fmt.manifest(None, {"m.experts.down_proj": x}, "0.22.0"), open(os.path.join(d, fmt.MANIFEST), "w"))
        assert fmt.read_manifest(d)["packs"]["m.experts.down_proj"] == x  # glyd-v4, which glyd 0.27 and before refuse by their format
        # the 12-bit layout (glyd-v3): a pack's base hb (split byte), its buffers data, exc and exc_base
        y = fmt.entry((1024, 256), 58, [("m.o_proj.weight", (1024, 256), "ee")], layout="mma12")
        assert list(y) == ["layout", "shape", "hb", "tensors"] and y["hb"] == 58 and fmt.LAYOUTS["mma12"] == ("data", "exc", "exc_base")
        json.dump(fmt.manifest(None, {"m.o_proj": y}, "0.24.0", "mma12"), open(os.path.join(d, fmt.MANIFEST), "w"))
        assert fmt.read_manifest(d)["format"] == "glyd-v3" and fmt.read_manifest(d)["layout"] == "mma12"
        for lay in ("mma", "mma12"):  # a table among the packs (an embedding's): glyd-v5 in either layout, which glyd 0.29 and before refuse
            t = fmt.entry((8192, 128), 58, [("m.embed_tokens.weight", (8192, 128), "ff")], layout="mma12", table=True)
            assert t["table"] is True and "table" not in y
            json.dump(fmt.manifest(None, {"m.embed_tokens": t}, "0.29.0", lay, {}, tables=True), open(os.path.join(d, fmt.MANIFEST), "w"))
            assert fmt.read_manifest(d)["format"] == "glyd-v5" and fmt.FORMATS[-3:] == ("glyd-v5", "glyd-v6", "glyd-v7") and "glyd-v5" not in fmt.FORMATS[:-3]
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
    its format (glyd-v3, glyd-v4) says it is there, its glyd not a version, a key glyd.json does not have in a save of this
    glyd or an older one. A save of glyd 0.24 without the map passes with its tensors unchecked; a newer glyd's key is
    skipped with a warning, the rest checked."""
    import glyd_gpu as glyd
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
        assert fmt.check_files(d, dict(old, glyd="0.24.0", format="glyd-v1")) == (0, 2)
        refused(d, dict(old, format="glyd-v1"), "no sha256 for the tensors saved as they are, which a glyd-v1 of glyd 0.25.0 has")
        refused(d, dict(old, glyd="0.24.0", format="glyd-v3"), "which a glyd-v3 of glyd 0.24.0 has")
        refused(d, dict(old, glyd="0.28.0"), "which a glyd-v4 of glyd 0.28.0 has")
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
    names = {"_P", "_I64", "_U64", "_SZ", "_W", "_PACK", "_FAST", "_DENSE", "_ARGS", "_SIZES", "_PLAIN", "_AUX", "API_VERSION", "X1"}
    body = [n for n in ast.parse(open(os.path.join(SRC, "_lib.py")).read()).body if isinstance(n, ast.Assign) and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} <= names]
    lib = {"ctypes": ctypes}
    exec(compile(ast.Module(body, []), "_lib.py", "exec"), lib)
    c, P = ctypes, ctypes.c_void_p
    types = {"int64_t": c.c_int64, "uint64_t": c.c_uint64, "size_t": c.c_size_t, "double": c.c_double, "int": c.c_int, "cudaStream_t": P, "size_t*": c.POINTER(c.c_size_t)}

    def ctype(a):  # "const uint8_t* data": P; "const uint32_t tiers[3]": the host words
        t, name = " ".join(a.split()).replace(" *", "*").rsplit(" ", 1)
        t = t.replace("const ", "")
        return c.POINTER(c.c_uint32) if name.endswith("]") and t == "uint32_t" else types.get(t, P if t.endswith("*") else t)

    declared = {name: (ret, [ctype(a) for a in args.split(",")] if args.strip() != "void" else []) for ret, name, args in re.findall(r"(int|const char\*|const void\*) (glyd_gpu_\w+)\(([^)]*)\);", " ".join(h.split()))}
    called = {f"glyd_gpu_{n}": ("int", a + [P]) for n, a in lib["_ARGS"].items()}
    called.update({f"glyd_gpu_{n}_workspace": ("int", [c.c_int64] * k + [c.POINTER(c.c_size_t)]) for n, k in lib["_SIZES"].items()})
    called.update({f"glyd_gpu_{n}": ("int", a) for n, a in lib["_PLAIN"].items()})  # (no stream: the routes)
    called.update({f"glyd_gpu_{n}": ("int", a) for n, a in lib["_AUX"].items()})  # (the route SPLIT's ring: each list whole)
    called.update(glyd_gpu_api_version=("int", []), glyd_gpu_cuda_version=("int", []), glyd_gpu_error_string=("const char*", [c.c_int]), glyd_gpu_table=("const void*", [c.c_int]))
    assert declared == called, [n for n in sorted(set(declared) | set(called)) if declared.get(n) != called.get(n)]
    assert int(re.search(r"#define GLYD_GPU_API_VERSION (\d+)", h).group(1)) == lib["API_VERSION"], "GLYD_GPU_API_VERSION is not _lib.py's API_VERSION"
    # the routes' numbers and a GPU's classes: kernels.py's and _lib.py's the header's
    defines = {k: int(v) for k, v in re.findall(r"#define (GLYD_GPU_\w+) (\d+)", h)}
    kern = {}
    names = {"DECODE", "GEMM", "X2", "X3", "X1", "EARLY", "PART", "GEMM_CC12", "GEFORCE", "A10", "L4", "L40S", "PCIE", "GH200", "H100", "WITH_PART"}
    body = [n for n in ast.parse(open(os.path.join(SRC, "kernels.py")).read()).body if isinstance(n, ast.Assign) and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} <= names]
    exec(compile(ast.Module(body, []), "kernels.py", "exec"), kern)
    for r, c_name in (("DECODE", "DECODE"), ("GEMM", "GEMM"), ("X2", "MID"), ("X3", "WG"), ("X1", "BIG"), ("EARLY", "AHEAD"), ("PART", "SPLIT"), ("GEMM_CC12", "GEMM_CC12")):  # (the header's names for the routes: the package's are neutral)
        assert kern[r] == defines[f"GLYD_GPU_ROUTE_{c_name}"], r
    assert lib["X1"] == defines["GLYD_GPU_ROUTE_BIG"] and (kern["GEFORCE"], kern["A10"], kern["L4"], kern["L40S"], kern["PCIE"], kern["GH200"], kern["H100"], kern["WITH_PART"]) == (
        defines["GLYD_GPU_GEFORCE"], defines["GLYD_GPU_A10"], defines["GLYD_GPU_L4"], defines["GLYD_GPU_L40S"], defines["GLYD_GPU_PCIE"], defines["GLYD_GPU_GH200"], defines["GLYD_GPU_H100"], defines["GLYD_GPU_WITH_SPLIT"])
    cu = open(os.path.join(gpu, "glyd_gpu.cu")).read()
    assert set(re.findall(r"GLYD_GPU_(?:API|INTERNAL) [^(]*?(glyd_gpu_\w+)\(", cu)) == set(declared), "glyd_gpu.cu's C API is not glyd_gpu.h's"
    table = {f"glyd_gpu_{n}" for n in re.findall(r"^    T\((\w+)\)$", cu, flags=re.M)}  # (glyd_gpu_table's list: every function but itself, each once)
    assert table == set(declared) - {"glyd_gpu_table"} and len(re.findall(r"^    T\(\w+\)$", cu, flags=re.M)) == len(table), sorted(table ^ (set(declared) - {"glyd_gpu_table"}))


def check_wheel():
    """glyd-gpu-py/check_wheel.py, the release gate (a script of the checkout, not of the package), as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_wheel", os.path.join(HERE, "..", "check_wheel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def elf_fixture(machine, exports, symtab=False, debug=False, needs=(), rodata=b""):
    """A minimal ELF64 shared object with the dynamic symbols exports (and a symbol table, or a debug section, if asked; and a dynamic section naming the libraries `needs`; and a loaded section
    that is not code, `.rodata`, holding `rodata`: where a program's strings are)."""
    import struct

    dynstr, offsets, wanted = b"\0", [], []
    for e in exports:
        offsets.append(len(dynstr))
        dynstr += e.encode() + b"\0"
    for n in needs:
        wanted.append(len(dynstr))
        dynstr += n.encode() + b"\0"
    dynsym = b"\0" * 24 + b"".join(struct.pack("<IBBHQQ", o, 0x12, 0, 1, 0, 0) for o in offsets)
    dynamic = b"".join(struct.pack("<qQ", 1, o) for o in wanted) + struct.pack("<qQ", 0, 0)
    secs = [(b".dynsym", 11, dynsym, 2), (b".dynstr", 3, dynstr, 0)] + ([(b".dynamic", 6, dynamic, 2)] if needs else []) + ([(b".symtab", 2, b"\0" * 24, 0)] if symtab else []) + ([(b".debug_info", 1, b"x", 0)] if debug else []) + ([(b".rodata", 1, rodata, 0)] if rodata else [])
    shstr, names = b"\0", []
    for n, *_ in secs + [(b".shstrtab",)]:
        names.append(len(shstr))
        shstr += n + b"\0"
    secs.append((b".shstrtab", 3, shstr, 0))
    body, heads, at = b"", [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], 64
    for name, (_, typ, data, link) in zip(names, secs):
        flags = 2 if shstr[name : shstr.index(b"\0", name)] == b".rodata" else 0  # (SHF_ALLOC: loaded, and not code)
        heads.append(struct.pack("<IIQQQQIIQQ", name, typ, flags, 0, at + len(body), len(data), link, 0, 1, 24 if typ == 11 else 0))
        body += data
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    head = ident + struct.pack("<HHIQQQIHHHHHH", 3, machine, 1, 0, 0, at + len(body), 0, 64, 0, 0, 64, len(heads), len(heads) - 1)
    return head + body + b"".join(heads)


def lzss_fixture(marker):
    """The bytes of a string table as Cython stores it (LZSS: the literals as they are, a repeat as a pair of bytes) that holds marker and the word
    `tiered`, which they do not show: its first three letters and its last three are literals apart, and two repeats put them together."""
    lits = b"tie" + b"XYZ" + b"re"  # (the first group of 8: flags 0xFF)
    first = bytes([0xFF]) + lits
    # then: a literal d, a repeat of lits[0:3] ("tie": 6 back, 3 long), a repeat of "red" (lits[6:8] + "d": 3 back, 3 long), the marker as literals
    second_items = [("lit", ord("d")), ("rep", 6), ("rep", 3)] + [("lit", c) for c in marker.encode()]
    out, i = b"", 0
    while i < len(second_items):
        group = second_items[i : i + 8]
        flags, body = 0, b""
        for k, (kind, v) in enumerate(group):
            if kind == "lit":
                flags |= 1 << k
                body += bytes([v])
            else:
                body += bytes([v, 0])
        out += bytes([flags]) + body
        i += 8
    return b"\0\0\0\0" + first + out + b"\0\0\0\0"


def engine_fixture(exports=(), needs=("libc.so.6", "libm.so.6"), fat=True, extra=b"", payload=bytes(8), machine=62, rodata=b"", **kw):
    """A stand-in for the engine's program: an ELF64 file that needs the system's libraries, with `rodata` in its data (where its strings are), a fat binary's header and `payload` where its
    kernels are, and `extra` after (the pieces of names in its instructions are not in a section of data)."""
    import struct

    blob = b"\x50\xed\x55\xba" + struct.pack("<HHQ", 1, 16, len(payload)) + payload if fat else b""
    return elf_fixture(machine, list(exports), needs=needs, rodata=rodata, **kw) + blob + extra


def cubin_fixture(sections=(), symbols=()):
    """A minimal ELF64 file as a cubin has it: sections named `sections` (and .shstrtab, .strtab, .symtab), and in the symbol table the names `symbols`."""
    import struct

    strtab = b"\0" + b"".join(n.encode() + b"\0" for n in symbols)
    symtab = b"\0" * 24 + b"".join(struct.pack("<IBBHQQ", 1 + len(b"".join(m.encode() + b"\0" for m in symbols[:i])), 0x12, 0, 1, 0, 0) for i in range(len(symbols)))
    secs = [(n.encode(), 1, b"", 0) for n in sections] + [(b".strtab", 3, strtab, 0), (b".symtab", 2, symtab, 0)]
    shstr, names = b"\0", []
    for n, *_ in secs + [(b".shstrtab",)]:
        names.append(len(shstr))
        shstr += n + b"\0"
    secs.append((b".shstrtab", 3, shstr, 0))
    link = [i for i, (n, *_) in enumerate(secs, 1) if n == b".strtab"][0]
    body, heads, at = b"", [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)], 64
    for name, (n, typ, data, _) in zip(names, secs):
        heads.append(struct.pack("<IIQQQQIIQQ", name, typ, 0, 0, at + len(body), len(data), link if typ == 2 else 0, 0, 1, 24 if typ == 2 else 0))
        body += data
    head = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8 + struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, at + len(body), 0, 64, 0, 0, 64, len(heads), len(heads) - 1)
    return head + body + b"".join(heads)


NEUTRAL_CUBIN = [".text.gk_3", ".nv.info.gk_3", ".nv.shared.gk_3", ".nv.constant0.gk_3", ".nv.capmerc.text.gk_3", ".nv.callgraph", ".debug_frame", ".rela.debug_frame", ".nv.merc.rela.text.gk_3"]


def machine_code(archs=("sm_80", "sm_86", "sm_89", "sm_90a", "sm_100", "sm_120"), names=NEUTRAL_CUBIN, symbols=("gk_3", "$__internal_0_$__cuda_sm20_div_u64"), ptx=False):
    """What a stand-in for cuobjdump (below) reads out of a fat binary of the tests: its payload says which PTX and which cubins it has."""
    import base64
    import json

    return json.dumps({"ptx": ptx, "cubins": {a: base64.b64encode(cubin_fixture(names, symbols)).decode() for a in archs}}).encode()


FAKE_CUOBJDUMP = """#!{python}
import base64, json, os, sys
a = sys.argv[1:]
data = open(a[-1], "rb").read()
payload = json.loads(data[16:])  # (the fat binary's header is 16 bytes)
base = os.path.basename(a[-1]).rsplit(".", 1)[0]
if a[0] == "--list-ptx":
    print("PTX file    1: %s.1.sm_80.ptx" % base if payload["ptx"] else "cuobjdump info    : No PTX file found to extract from '%s'." % a[-1])
elif a[0] == "--list-elf":
    for i, arch in enumerate(payload["cubins"], 1):
        print("ELF file    %d: %s.%d.%s.cubin" % (i, base, i, arch))
elif a[0] == "-xelf":
    for i, (arch, b64) in enumerate(payload["cubins"].items(), 1):
        open("%s.%d.%s.cubin" % (base, i, arch), "wb").write(base64.b64decode(b64))
else:
    sys.exit(2)
"""


def fake_cuobjdump(d):
    """The path of an executable that does what the gate asks of cuobjdump (--list-ptx, --list-elf, -xelf all) for the fat binaries `machine_code` makes."""
    path = os.path.join(d, "cuobjdump")
    open(path, "w").write(FAKE_CUOBJDUMP.format(python=sys.executable))
    os.chmod(path, 0o755)
    return path


def test_check_wheel():
    """The release gate on wheels made for it: one that passes, and each way of not passing: Python source, a stray file, a module
    that exports more than its init, a library with a symbol table or another export, a docstring left in a module, a header that is
    not the public one or says what it should not, a method word in a string, the Python source or the page of the wheel, or the plain text of
    a module, the library the gpu-lib job did not check, a dependency without an extra."""
    import hashlib
    import zipfile

    cw = check_wheel()
    header = os.path.join(HERE, "..", "include", "glyd_gpu.h")
    public = open(header).read()
    licence = os.path.join(HERE, "..", "..", "gpu", "LICENSE")
    api = re.findall(r"\b(glyd_gpu_\w+)\(", re.sub(r"/\*.*?\*/", "", public, flags=re.S))
    doc = "Rows of the table, decoded: the lookup of an embedding by its ids, whole."
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "src"))
        for name, text in {"__init__": '"""The package."""\n', "__main__": "", "_trace": "", "fit": f'def f():\n    """{doc}"""\n'}.items():
            open(os.path.join(d, "src", name + ".py"), "w").write(text)
        meta = "Metadata-Version: 2.4\nName: glyd-gpu\nVersion: 9.9.9\nLicense-Expression: BUSL-1.1\nLicense-File: LICENSE\nProvides-Extra: gpu\nProvides-Extra: vllm\nRequires-Dist: torch>=2.5; extra == 'gpu'\n"
        tag = "cpython-312-x86_64-linux-gnu"

        def files(**over):
            f = {
                "glyd_gpu/__main__.py": b"", "glyd_gpu/_trace.py": b"", "glyd_gpu/chat.html": b"<html></html>", "glyd_gpu/kvcorpus.bin": b"text", "glyd_gpu/glyd_gpu.h": public.encode(),
                f"glyd_gpu/__init__.{tag}.so": elf_fixture(62, ["PyInit_glyd_gpu"]), f"glyd_gpu/fit.{tag}.so": elf_fixture(62, ["PyInit_fit"]),
                "glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api + ["glyd_gpu_table"]) + b"12", "glyd_gpu/libglyd_gpu_cuda13.so": elf_fixture(62, api + ["glyd_gpu_table"]) + b"13",
                "glyd_gpu/glyd-engine": engine_fixture(),
                "glyd_gpu-9.9.9.dist-info/METADATA": meta.encode(), "glyd_gpu-9.9.9.dist-info/WHEEL": b"", "glyd_gpu-9.9.9.dist-info/RECORD": b"",
                "glyd_gpu-9.9.9.dist-info/entry_points.txt": b"", "glyd_gpu-9.9.9.dist-info/top_level.txt": b"glyd_gpu\n", "glyd_gpu-9.9.9.dist-info/licenses/LICENSE": open(licence, "rb").read(),
            }
            f.update(over)
            return {k: v for k, v in f.items() if v is not None}

        def run(mode=0o755, kw=None, unsummed=(), **over):
            kw = {"cuobjdump": False, **(kw or {})}  # (the fixtures' libraries are not CUDA's: the machine code is not read, unless a case says so)
            path = os.path.join(d, "glyd_gpu-9.9.9-cp312-cp312-manylinux_2_28_x86_64.whl")
            with zipfile.ZipFile(path, "w") as z:
                for k, v in files(**over).items():
                    info = zipfile.ZipInfo(k)
                    info.external_attr = (0o644 if k != "glyd_gpu/glyd-engine" or mode is None else mode) << 16  # (the program: executable, as build.sh and setup.py make it)
                    z.writestr(info, v)
            sums = os.path.join(d, "sums")
            with open(sums, "w") as f:
                for lib in ("libglyd_gpu_cuda12.so", "libglyd_gpu_cuda13.so"):
                    if lib not in unsummed:
                        f.write(f"{hashlib.sha256(files()['glyd_gpu/' + lib]).hexdigest()}  lib/{lib}\n")
                if "glyd-engine" not in unsummed:
                    f.write(f"{hashlib.sha256(files()['glyd_gpu/glyd-engine']).hexdigest()}  engine/glyd-engine\n")
            return cw.check(path, "9.9.9", "manylinux_2_28_x86_64", os.path.join(d, "src"), header, licence, [sums], **kw)

        assert run() == []
        # the machine code is read with cuobjdump, or it is the bytes the jobs that read it vouch for (--sums): where neither, the gate fails (it does not skip what it cannot read)
        which, cw.shutil.which = cw.shutil.which, lambda name, *a, **k: None if name == "cuobjdump" else which(name, *a, **k)
        try:
            assert run(kw={"cuobjdump": None}) == [], run(kw={"cuobjdump": None})
            got = run(kw={"cuobjdump": None}, unsummed=("glyd-engine",))
            assert any("machine code of the program was not read" in line for line in got), got
            got = run(kw={"cuobjdump": None}, unsummed=("libglyd_gpu_cuda13.so",))
            assert any("machine code of the library was not read" in line and "cuda13" in line for line in got), got
        finally:
            cw.shutil.which = which
        # and read with it where it is there: the engine's names, here through a stand-in that reads each fat binary's payload
        tool = fake_cuobjdump(d)
        got = run(kw={"cuobjdump": tool}, **{"glyd_gpu/glyd-engine": engine_fixture(payload=machine_code(symbols=("gk_3", "kv_hist")))})
        assert any("[kernel] kv_hist" in line for line in got), got
        stable = os.path.join(d, "glyd_gpu-9.9.9-cp310-abi3-manylinux_2_28_x86_64.whl")  # (the stable ABI: its modules are named .abi3.so)
        with zipfile.ZipFile(stable, "w") as z:
            for k, v in files(**{f"glyd_gpu/__init__.{tag}.so": None, f"glyd_gpu/fit.{tag}.so": None, "glyd_gpu/__init__.abi3.so": elf_fixture(62, ["PyInit_glyd_gpu"]), "glyd_gpu/fit.abi3.so": elf_fixture(62, ["PyInit_fit"])}).items():
                info = zipfile.ZipInfo(k)
                info.external_attr = (0o755 if k == "glyd_gpu/glyd-engine" else 0o644) << 16
                z.writestr(info, v)
        assert cw.check(stable, "9.9.9", "manylinux_2_28_x86_64", os.path.join(d, "src"), header, licence, [os.path.join(d, "sums")], cuobjdump=False) == []
        cases = {
            "Python source": (dict(**{"glyd_gpu/model.py": b"x = 1"}), "Python source"),
            "a stray file": (dict(**{"glyd_gpu/glyd_gpu.cu": b"x"}), "is not a file"),
            "a second header": (dict(**{"glyd_gpu/glyd_gpu_internal.h": b"x"}), "is not a file"),
            "an internal header": (dict(**{"glyd_gpu/glyd_gpu.h": open(os.path.join(HERE, "..", "..", "gpu", "glyd_gpu.h"), "rb").read()}), "is not"),
            "a header that tells": (dict(**{"glyd_gpu/glyd_gpu.h": (public + "/* route SPLIT */").encode()}), "the header says"),
            "a module's other export": (dict(**{f"glyd_gpu/fit.{tag}.so": elf_fixture(62, ["PyInit_fit", "helper"])}), "exports"),
            "a symbol table": (dict(**{f"glyd_gpu/fit.{tag}.so": elf_fixture(62, ["PyInit_fit"], symtab=True)}), "symbol table"),
            "debug sections": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api + ["glyd_gpu_table"], debug=True)}), "debug"),
            "a library's other export": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api + ["glyd_gpu_table", "cudaMalloc"])}), "cudaMalloc is exported"),
            "a library's own function, exported": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api + ["glyd_gpu_table", "glyd_gpu_hold"])}), "glyd_gpu_hold is exported"),
            "a library without the header's functions": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, ["glyd_gpu_table"])}), "glyd_gpu_mma_linear is not exported"),
            "a library without its table": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api)}), "glyd_gpu_table is not exported"),
            "a kernel's readable name": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api + ["glyd_gpu_table"]) + b"\0_Z15mma_gemm_kernelILi1EEvv\0"}), "[kernel] mma_gemm_kernel"),
            "a method word in a library's name": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api + ["glyd_gpu_table"]) + b"\0kv_ring_slot_base\0"}), "[name]"),
            "a tuning variable in a library": (dict(**{"glyd_gpu/libglyd_gpu_cuda12.so": elf_fixture(62, api + ["glyd_gpu_table"]) + b"\0GLYD_WG_MAX\0"}), "GLYD_WG_MAX"),
            "a tuning variable in a module's table": (dict(**{f"glyd_gpu/fit.{tag}.so": elf_fixture(62, ["PyInit_fit"]) + b"GLYD_KV_RESERVE_1"}), "[table]"),
            "a word the table hides from a read of the bytes": (dict(**{f"glyd_gpu/fit.{tag}.so": elf_fixture(62, ["PyInit_fit"]) + lzss_fixture("glyd_gpu/fit.py")}), "'tiered'"),
            "another architecture": (dict(**{f"glyd_gpu/fit.{tag}.so": elf_fixture(183, ["PyInit_fit"])}), "ELF machine"),
            "a docstring": (dict(**{f"glyd_gpu/fit.{tag}.so": elf_fixture(62, ["PyInit_fit"]) + doc.encode()}), "docstring"),
            "a library that was not checked": (dict(**{"glyd_gpu/libglyd_gpu_cuda13.so": elf_fixture(62, api + ["glyd_gpu_table"]) + b"x"}), "gpu-lib job"),
            "no library": (dict(**{"glyd_gpu/libglyd_gpu_cuda13.so": None}), "no libglyd_gpu_cuda13.so"),
            "no engine": (dict(**{"glyd_gpu/glyd-engine": None}), "no glyd-engine"),
            "an engine that is not executable": (dict(mode=0o644), "glyd-engine is not executable"),
            "an engine for another architecture": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(machine=183)}), "is for ELF machine 183"),
            "an engine with a symbol table": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(symtab=True)}), "symbol table"),
            "an engine with an export": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(exports=["main"])}), "main is exported"),
            "an engine that needs CUDA's runtime": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(needs=("libc.so.6", "libcudart.so.13"))}), "needs libcudart.so.13"),
            "an engine with no machine code": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(fat=False)}), "no machine code"),
            "an engine that says how": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(extra=b"\0the tiered layout\0")}), "'tiered layout'"),
            "an engine that names a tuning variable": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(extra=b"\0GLYD_ENGINE_TILE\0")}), "GLYD_ENGINE_TILE"),
            "an engine that was not checked": (dict(**{"glyd_gpu/glyd-engine": engine_fixture(extra=b"x")}), "engine job"),
            "a dependency": (dict(**{"glyd_gpu-9.9.9.dist-info/METADATA": (meta + "Requires-Dist: torch>=2.5\n").encode()}), "without an extra"),
        }
        for what, (over, expect) in cases.items():
            got = run(**over)
            assert got and any(expect in line for line in got), (what, got)
        assert cw.told(open(os.path.join(HERE, "..", "..", "gpu", "glyd_gpu.h")).read()), "told() finds nothing in the internal header"  # (its positive control)
        # a wheel without the libraries has no engine either (CI's compile check), and an engine where there is none to be is a stray file
        got = run(**{"glyd_gpu/libglyd_gpu_cuda12.so": None, "glyd_gpu/libglyd_gpu_cuda13.so": None, "glyd_gpu/glyd-engine": None}, kw={"libs": False})
        assert got == [], got
        got = run(**{"glyd_gpu/libglyd_gpu_cuda12.so": None, "glyd_gpu/libglyd_gpu_cuda13.so": None}, kw={"libs": False})
        assert any("glyd-engine where there is none to be" in line for line in got), got
        assert run(**{"glyd_gpu/glyd-engine": None}, kw={"engine": False}) == []  # (the libraries without the engine: --no-engine)
        # method words (check_wheel.METHOD): a string of a source, the wheel's Python source and page, the plain text of a compiled module; not what a user types, a key, a name or a web page
        fit = os.path.join(d, "src", "fit.py")
        keep = open(fit).read()
        for what, text in {"a string": 'MSG = "the KV cache in 12 bits"\n', "a message's piece": 'MSG = f"glyd: {n} blocks of the second reserve"\n', "a word": 'MSG = f"{n} slots of a ring"\n', "an abbreviation": 'MSG = "the TMA copies"\n'}.items():
            open(fit, "w").write(keep + text)
            got = run()
            assert got and any("a string of fit.py says" in line for line in got), (what, got)
        open(fit, "w").write(keep + 'MSG = "GLYD_KV sets it"\nKEY = "tiers"\nPAGE = "the chat page"\nSMALL = "an sms"\nLATER = "the reserved memory"\n')
        assert run() == [], run()  # (the variable a user sets, a key a format has, a web page, an SMS, a word that only begins like one)
        open(fit, "w").write(keep)
        for what, over, word in (("Python source", {"glyd_gpu/_trace.py": b"# the tiered layout\n"}, "tiered layout"), ("the page", {"glyd_gpu/chat.html": b"<p>12 bits</p>"}, "12 bits"), ("the description", {"glyd_gpu-9.9.9.dist-info/METADATA": (meta + "\nthe tiered layout\n").encode()}, "tiered layout"),
                                 ("plain text of a module", {f"glyd_gpu/fit.{tag}.so": elf_fixture(62, ["PyInit_fit"]) + b"the cache in 12 bits"}, "12 bits")):
            got = run(**over)
            assert got and any(repr(word) in line for line in got), (what, got)


def test_check_lib():
    """The gpu-lib job's gate on a library alone: the eight and the table exported and nothing else, an entry in the table for each function the internal
    header declares (an id the package's own table_id gives), a name that is not neutral, a variable that tunes the code."""
    import struct

    cw = check_wheel()
    header = os.path.join(HERE, "..", "..", "gpu", "glyd_gpu.h")
    names = sorted(set(re.findall(r"\b(glyd_gpu_\w+)\(", re.sub(r"/\*.*?\*/", "", open(header).read(), flags=re.S))) - {"glyd_gpu_table"})

    def fnv(name):
        h = 2166136261
        for c in name[len("glyd_gpu_") :].encode():
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
        return h or 1

    if importlib.util.find_spec("torch"):  # (the package's own: the two readings of an id agree)
        from glyd_gpu import _lib

        assert all(_lib.table_id(n[len("glyd_gpu_") :]) == fnv(n) for n in names)
    entries = b"".join(struct.pack("<II", fnv(n), 0) + bytes(8) for n in names)
    exports = sorted(cw.PUBLIC)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "libglyd_gpu_cuda13.so")

        def run(data, **kw):
            open(p, "wb").write(data)
            return cw.check_lib(p, table_header=header, cuobjdump=False, **kw)

        assert run(elf_fixture(62, exports) + entries) == []
        got = run(elf_fixture(62, exports))
        assert len(got) == len(names) and all("[export] the table has no entry for glyd_gpu_" in line for line in got), got
        for what, data, expect in (("another export", elf_fixture(62, exports + ["glyd_gpu_hold"]) + entries, "glyd_gpu_hold is exported"),
                                   ("no table", elf_fixture(62, [e for e in exports if e != "glyd_gpu_table"]) + entries, "glyd_gpu_table is not exported"),
                                   ("a symbol table", elf_fixture(62, exports, symtab=True) + entries, "symbol table"),
                                   ("a variable that tunes", elf_fixture(62, exports) + entries + b"\0GLYD_KV_RESERVE_2\0", "GLYD_KV_RESERVE_2"),
                                   ("the variable it may name", elf_fixture(62, exports) + entries + b"\0GLYD_SPLIT_MIN\0", None)):
            got = run(data)
            assert (expect is None and got == []) or (expect and any(expect in line for line in got)), (what, got)
        open(p, "wb").write(elf_fixture(62, exports) + entries + b"\0kv_ring_slot_base\0")
        assert cw.main(["--lib", p, "--summary"]) == 1  # (the command line, as the job runs it)
        # where no demangler is at hand the name a mangled one starts with is read, by its length
        which, cw.shutil.which = cw.shutil.which, lambda name, *a, **k: None if "filt" in name else which(name, *a, **k)
        try:
            for mangled, expect in ((b"_Z15mma_gemm_kernelILi1EEvv", "[kernel] mma_gemm_kernel"), (b"_ZN3kvc10decode_rawILi0EEEvv", "[kernel] kvc"), (b"_Z4gk_17ILi112ELi2EEv14CUtensorMap_st5gt_15l", None)):
                got = run(elf_fixture(62, exports) + entries + b"\0" + mangled + b"\0")
                assert (expect is None and got == []) or (expect and any(expect in line for line in got)), (mangled, got)
        finally:
            cw.shutil.which = which


def test_check_engine():
    """The gate on the engine's program (the engine job's): an ELF file for the architecture with no symbol table, no export, the system's libraries only and machine code in it;
    no method word or phrase in its text, no development command or flag, no internal count or wording of how, no variable of a development build, no path of the machine that built it, no kernel
    of its sources or function of the library that its table hides (the ones the release build reads, run together as Rust stores strings, are fine, and so are the dependencies' own words and the
    paths of their sources), and the machine code (not text) not read as if it were. The machine code is read with cuobjdump, which the gate requires: here a stand-in that reads a payload."""
    cw = check_wheel()
    ok = engine_fixture()
    no_tool = dict(cuobjdump=False)  # (the byte-level checks: the stand-in's machine code is not the point)
    assert cw.engine_problems("e", ok, "x86_64", **no_tool) == [] and cw.engine_problems("e", ok, **no_tool) == []
    assert cw.needed(engine_fixture(needs=("libc.so.6", "libgcc_s.so.1"))) == ["libc.so.6", "libgcc_s.so.1"] and cw.fatbins(ok) == [(len(ok) - 24, 24)]
    for what, data, expect in (
        ("another architecture", engine_fixture(machine=183), "is for ELF machine 183"),
        ("a symbol table", engine_fixture(symtab=True), "symbol table"),
        ("debug sections", engine_fixture(debug=True), "debug"),
        ("an export", engine_fixture(exports=["main"]), "main is exported"),
        ("a library that is not the system's", engine_fixture(needs=("libc.so.6", "libcudart.so.13")), "needs libcudart.so.13"),
        ("no machine code", engine_fixture(fat=False), "no machine code"),
        ("a method phrase", engine_fixture(extra=b"\0a cache of 12 bits\0"), "'12 bits'"),
        ("a word of the table", engine_fixture(extra=b"\0the mantissa\0"), "'mantissa'"),
        ("a method word of a message", engine_fixture(rodata=b"\0the warps of the kernel\0"), "'warps'"),
        ("a method word between others", engine_fixture(rodata=b"{n} slots in the pool of a ring"), "'pool'"),
        ("a method word that is a name in the data", engine_fixture(rodata=b"worst_chunks tiers pages\0"), "'pages'"),
        ("a variable that tunes", engine_fixture(extra=b"\0GLYD_ENGINE_RESERVE_1\0"), "GLYD_ENGINE_RESERVE_1"),
        ("a variable that tunes, run together", engine_fixture(extra=b"GLYD_CACHEGLYD_ENGINE_TILE"), "GLYD_ENGINE_TILE"),
        ("a variable that tunes, as immediates", engine_fixture(extra=b"H\xb8GLYD_ENGH\xb8INE_TILE"), "GLYD_ENG"),
        ("a development command, advised", engine_fixture(extra=b"\0make tables for more: glyd-engine kvcal MODEL OUT.safetensors\0"), "'kvcal'"),
        ("a development command's name", engine_fixture(extra=b"\0rowcheck\0"), "'rowcheck'"),
        ("a development command of everyday words", engine_fixture(extra=b"\0run glyd-engine pack first\0"), "glyd-engine pack"),
        ("a development flag", engine_fixture(extra=b"\0no-graph\0"), "'no-graph'"),
        ("a development flag, as immediates", engine_fixture(extra=b"H\xb8no-graphH\xb8"), "'no-graph'"),
        ("a key of a development build's stats", engine_fixture(extra=b'{"steps":1,"lost_blocks":2}'), "'lost_blocks'"),
        ("how, in other words", engine_fixture(extra=b"\0its keys and values need more of the cache's extra memory than there is\0"), "'extra memory'"),
        ("a pass count", engine_fixture(extra=b"\0pass 2 of 3 (12 s)\0"), "pass 2 of 3"),
        ("a block shape", engine_fixture(extra=b"\0this layout wants rows a multiple of 64 and columns of 16\0"), "'multiple of 64'"),
        ("the machine that built it", engine_fixture(extra=b"\0/github/home/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ureq-3.4.2/src/lib.rs\0"), "[path]"),
        ("the home of a developer", engine_fixture(extra=b"\0/Users/dev/glyd/glyd-engine/src/model.rs\0"), "/Users/dev/glyd"),
    ):
        got = cw.engine_problems("e", data, "x86_64", **no_tool)
        assert got and any(expect in line for line in got), (what, got)
    # what a kernel of the sources or a function of the library that its table hides says: found where the sources or the header are given
    for what, extra, kw, expect in (
        ("a kernel of the sources", b"\0kv_hist\0", dict(kernels={"kv_hist", "copy_row"}), "[kernel] names kv_hist"),
        ("a kernel in names run together", b"\0gk_4gemm3_128x256_s3gk_5\0", dict(kernels={"gemm3_128x256_s3"}), "gemm3_128x256_s3"),
        ("a function of the library", b"\0glyd_gpu_kv_second_bytes\0", dict(internal={"glyd_gpu_kv_second_bytes", "kv_second_bytes"}), "[name] names glyd_gpu_kv_second_bytes"),
        ("a function of the library, bare", b"\0kv_work_put\0", dict(internal={"glyd_gpu_kv_work_put", "kv_work_put"}), "kv_work_put"),
    ):
        got = cw.engine_problems("e", engine_fixture(extra=extra), "x86_64", **no_tool, **kw)
        assert got and any(expect in line for line in got), (what, got)
        assert cw.engine_problems("e", ok, "x86_64", **no_tool, **kw) == [], what  # (and not where it is not named)
    # what a release build has: its variables as Rust stores the strings of a table (run together), the template language's message, a word where the kernels are, the dependencies' words and
    # the paths of their sources (remapped by the build), the file names a panic gives, a property of Unicode that has a command's letters in it, and the public header's functions
    fine = engine_fixture(extra=b"GLYD_API_KEYVLLM_API_KEYGLYD_CACHEGLYD_GPU_LIBHF_ENDPOINTexponent must be an integer", payload=b"tiered!!")
    fine += b"H\xb8GLYD_CACH\x00D$8f\xc7D$@HE" + b"\x48\xb8GLYD_GPUH$" + b"H\xb8GLYD_ENGH\xb8INE_ASKEH\x89"  # (x86-64: the names the code compares, as immediates of 8 bytes)
    fine += b"\0cargo/registry/src/index.crates.io-1949cf8c6b5b557f/ring-0.17.14/src/hmac.rs\0/rustc/48a229ceaefd4985c50990b14116b6d856af0985/library/core/src/fmt/mod.rs\0glyd-engine/src/kvcal.rs\0"
    fine += b"\0nfkcquickcheck NFKC_Quick_Check\0ring::signature::ED25519\0invalid escape sequence found in character class\0The global thread pool has not been initialized.\0Kamu SM SSL\0glyd_gpu_api_version\0glyd_gpu_table\0"
    fine += b"\0\0tier\0ring\0pool\0H\xb8slot\0"  # (the pieces of a name an x86-64 program compares, in its instructions: not text, and not read for a word)
    assert cw.engine_problems("e", fine, "x86_64", **no_tool, kernels={"kv_hist"}, internal={"kv_write", "glyd_gpu_kv_write"}) == [], cw.engine_problems("e", fine, "x86_64", **no_tool)
    fine = engine_fixture(rodata=b"GLYD_CACHE\0nfkcquickcheck NFKC_Quick_Check ring::signature::ED25519 invalid escape sequence found in character class\0The global thread pool has not been initialized.\0Kamu SM SSL\0", extra=b"tier")
    assert cw.engine_problems("e", fine, "x86_64", **no_tool) == [], cw.engine_problems("e", fine, "x86_64", **no_tool)
    # a message may say the way to vLLM, which is a name the engine does not read: GLYD_ENGINE=0, and no other of its names
    said = engine_fixture(extra=b"\0n is not supported here (GLYD_ENGINE=0 runs vLLM, which has it)\0")
    assert cw.engine_problems("e", said, "x86_64", **no_tool) == [], cw.engine_problems("e", said, "x86_64", **no_tool)
    for other in (b"GLYD_ENGINE=1", b"GLYD_ENGINE", b"GLYD_ENGINE_TILE=0", b"GLYD_ENGINE=0GLYD_ENGINE_TILE"):
        got = cw.engine_problems("e", engine_fixture(extra=b"\0" + other + b"\0"), "x86_64", **no_tool)
        assert got and any("GLYD_ENGINE" in line for line in got), (other, got)
    # the machine code, read with cuobjdump (a stand-in here): neutral names, no line information, no PTX, the six architectures
    with tempfile.TemporaryDirectory() as d:
        tool = fake_cuobjdump(d)
        assert cw.engine_problems("e", engine_fixture(payload=machine_code()), "x86_64", tool) == [], cw.engine_problems("e", engine_fixture(payload=machine_code()), "x86_64", tool)
        # (a function that is not inlined, under a neutral name, is a C++ name that spells it: fine)
        neutral = machine_code(names=NEUTRAL_CUBIN + [".text._Z4gd_2f", ".nv.info._Z4gd_2f"], symbols=("gk_3", "_Z4gd_2f", "$gk_3$_ZN5gt_374gd_5ILi0EEEvv"))
        assert cw.engine_problems("e", engine_fixture(payload=neutral), "x86_64", tool) == [], cw.engine_problems("e", engine_fixture(payload=neutral), "x86_64", tool)
        for what, code, expect in (
            ("a kernel under its readable name", machine_code(names=NEUTRAL_CUBIN + [".text.kv_hist", ".nv.info.kv_hist"], symbols=("gk_3", "kv_hist")), "[kernel] .text.kv_hist"),
            ("a kernel's name in its symbol alone", machine_code(symbols=("gk_3", "rmsnorm_add")), "[kernel] rmsnorm_add"),
            ("a tile and its stages in a name", machine_code(names=NEUTRAL_CUBIN + [".nv.shared.gemm3_128x256_b64_s2"]), "gemm3_128x256_b64_s2"),
            ("a device function", machine_code(symbols=("gk_3", "block_sum")), "[kernel] block_sum"),
            ("a device function that is not inlined, under its readable name", machine_code(names=NEUTRAL_CUBIN + [".text._Z9block_sumf"], symbols=("gk_3", "_Z9block_sumf")), "[kernel] block_sum"),
            ("a name that is some of ours and some of gen_names's", machine_code(names=NEUTRAL_CUBIN + [".nv.info.gk_3_hist"]), "[kernel] .nv.info.gk_3_hist"),
            ("line information", machine_code(names=NEUTRAL_CUBIN + [".debug_line", ".nv_debug_line_sass"]), "line or debug information"),
            ("debug information", machine_code(names=NEUTRAL_CUBIN + [".debug_info"]), "has the section .debug_info"),
            ("PTX", machine_code(ptx=True), "holds PTX"),
            ("an architecture short", machine_code(archs=("sm_80", "sm_86", "sm_89", "sm_90a", "sm_100")), "machine code for"),
            ("another architecture", machine_code(archs=("sm_80", "sm_86", "sm_89", "sm_90a", "sm_100", "sm_120", "sm_70")), "machine code for"),
            ("no machine code a tool can read", machine_code(archs=()), "machine code"),
        ):
            got = cw.engine_problems("e", engine_fixture(payload=code), "x86_64", tool)
            assert got and any(expect in line for line in got), (what, got)
        # the machine code of the program is read where the program is given, whatever else it holds
        assert cw.engine_problems("e", engine_fixture(payload=machine_code(), extra=b"x" * 40), "x86_64", tool) == []
        # cuobjdump is required: not on PATH, the gate fails; where the very bytes were read by the engine job (--sums) it does not read them again
        which, cw.shutil.which = cw.shutil.which, lambda name, *a, **k: None if name == "cuobjdump" else which(name, *a, **k)
        try:
            for kw, expect in (({}, "cuobjdump is not on PATH"), ({"vouched": True}, None)):
                got = cw.engine_problems("e", engine_fixture(payload=machine_code()), "x86_64", **kw)
                assert (expect is None and got == []) or (expect and any(expect in line for line in got)), (kw, got)
            p = os.path.join(d, "glyd-engine")  # the command line, as the engine job runs it
            open(p, "wb").write(ok)
            assert cw.main(["--engine", p, "--summary"]) == 1, "no cuobjdump: it must fail, not pass"
            lib = os.path.join(d, "libglyd_gpu_cuda13.so")
            open(lib, "wb").write(elf_fixture(62, sorted(cw.PUBLIC)))
            assert any("cuobjdump is not on PATH" in line for line in cw.check_lib(lib)), "a library's machine code is read with cuobjdump too"
        finally:
            cw.shutil.which = which
        which, cw.shutil.which = cw.shutil.which, lambda name, *a, **k: tool if name == "cuobjdump" else which(name, *a, **k)
        try:
            p = os.path.join(d, "glyd-engine")
            open(p, "wb").write(engine_fixture(payload=machine_code()))
            assert cw.main(["--engine", p, "--summary"]) == 0
            open(p, "wb").write(engine_fixture(payload=machine_code(), extra=b"GLYD_ENGINE_VERBOSE"))
            assert cw.main(["--engine", p, "--summary"]) == 1
            open(p, "wb").write(engine_fixture(payload=machine_code(symbols=("gk_3", "kv_sweep"))))
            assert cw.main(["--engine", p, "--summary"]) == 1
        finally:
            cw.shutil.which = which


def test_check_engine_leaks_script():
    """check_engine_leaks.py (release.yml's engine job runs it on the program it built: nvcc, cuobjdump and objcopy are there): it loads, builds the leaks of every kind it injects, and where one
    of its tools is missing it fails and says so, not passes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_engine_leaks", os.path.join(HERE, "..", "check_engine_leaks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cw = mod.cw
    assert "extern \"C\" __global__ void kv_hist(" in mod.probe("kv_hist") and "__noinline__ int block_sum" in mod.probe("gk_1", "block_sum")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "glyd-engine")
        open(p, "wb").write(engine_fixture())
        which, mod.shutil.which = mod.shutil.which, lambda name, *a, **k: None
        try:
            assert mod.main([p]) == 1
        finally:
            mod.shutil.which = which
    assert cw.engine_kernel_names(os.path.join(HERE, "..", "..", "glyd-engine")), "the sources' kernels are what the leak that names one uses"


def test_check_engine_sources():
    """The gate on the engine's Rust sources as its release build compiles them: a module under #[cfg(feature = "dev")], an item, a statement or a match arm under it, a
    #[cfg(test)] tail, tune!("NAME"), a comment and the names of inline format arguments are not read; a text that says a method word, a name or a key that carries one (a
    format's own key excepted), a variable that tunes the code and the page the engine serves are."""
    cw = check_wheel()
    quiet = r'''
use std::fmt;
const OK: &str = "{n} blocks of the cache, {tier} positions";
const KEY: &str = "tiers";
const VARS: [&str; 3] = ["GLYD_CACHE", "GLYD_GPU_LIB", "HF_TOKEN"];
const RAW: &str = r#"{"layout":"mma","shape":[1,2]}"#;
fn f<'a>(x: &'a str, c: char) -> bool { c == '"' || x == "it's" || c == '\'' }
fn g(v: Option<&str>) -> usize {
    let a = crate::tune!("GLYD_ENGINE_TILE").map_or(0, |v| v.len());
    #[cfg(feature = "dev")]
    let b = if a > 1 { "a ring of slots" } else { "pool" };
    match v {
        #[cfg(feature = "dev")]
        Some("pool") => { eprintln!("the staging cache"); 1 }
        #[cfg(feature = "dev")]
        Some("arena") => 2,
        Some(_) => 3,
        None => a,
    }
}
#[cfg(feature = "dev")]
pub fn check() { println!("the tiered layout, the second reserve"); }
fn h(a: &Args) -> (&'static str, bool) {
    let m = if cfg!(feature = "dev") { "glyd kvcal: pass 1 of 3 ({:.0} s), the ring of slots" } else { "setting up" };
    let g = !(cfg!(feature = "dev") && a.has("no-graph"));
    (if cfg!(feature = "dev") {
        let n = "}"; // (a brace in a string, and a block in a block)
        "the arena"
    } else { m }, g)
}
// the tiered layout, a comment
/// and a doc comment about the 12-bit layout
'''
    tail = '''#[cfg(test)]
mod tests { fn t() { assert!("tiered layout" != ""); } }
'''
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        os.makedirs(src)
        os.makedirs(os.path.join(d, "data"))
        w = lambda name, text: open(os.path.join(src, name), "w").write(text)
        w("lib.rs", 'pub mod cli;\n#[cfg(feature = "dev")]\npub mod bench;\n')
        w("main.rs", "fn main() {}\n")
        w("bench.rs", 'fn bench() { println!("the tiered layout and the arena"); }\n')  # (a module only a development build has)
        w("cli.rs", quiet + tail)
        assert cw.engine_dev_modules(src) == {"bench.rs"}
        assert cw.engine_source_problems(d) == [], cw.engine_source_problems(d)
        w("cli.rs", quiet + 'const SAYS: &str = "not done here: GLYD_ENGINE=0 runs vLLM";\n' + tail)  # (the way to vLLM, which a message may say)
        assert cw.engine_source_problems(d) == [], cw.engine_source_problems(d)
        w("cli.rs", quiet + tail)
        for what, line, expect in (
            ("a text", 'const A: &str = "a block of the second reserve";\n', "[words]"),
            ("a word", 'const A: &str = "{} pages a layer";\n', "'pages'"),
            ("a name", 'const A: &str = "arena";\n', "[identifier] arena"),
            ("a key", 'const A: &str = "worst_chunks_per_layer_page";\n', "[identifier]"),
            ("a variable that tunes", 'const A: &str = "GLYD_ENGINE_TILE=w4b64";\n', "[env] GLYD_ENGINE_TILE"),
            ("a variable in a raw string", 'const A: &str = r#"set GLYD_KV_RESERVE_1"#;\n', "[env]"),
            ("the way to vLLM and a variable that tunes", 'const A: &str = "GLYD_ENGINE=0 runs vLLM, GLYD_ENGINE_TILE=3";\n', "[env] GLYD_ENGINE_TILE"),
            ("another value of the way to vLLM", 'const A: &str = "GLYD_ENGINE=1";\n', "[env] GLYD_ENGINE"),
            ("a byte string", 'const A: &[u8] = b"in 12 bits";\n', "[words]"),
            ("a string continued", 'const A: &str = "a line \\\n     with the tiered layout";\n', "[words]"),
            ("a development command", 'const A: &str = "make them for more: glyd-engine kvcal MODEL OUT";\n', "[dev]"),
            ("a development command's name", 'const A: &str = "kvcal";\n', "[dev]"),
            ("a development flag", 'const A: &str = "no-graph";\n', "[dev]"),
            ("a development flag, read with a flag that is not a dev's", 'const A: &str = "x"; fn f(a: &Args) -> bool { a.has("no-graph") }\n', "[dev]"),
            ("a count of a development build's stats", 'const A: &str = "{{\\"lost_blocks\\":{}}}";\n', "lost_blocks"),
            ("how, in other words", 'const A: &str = "it needs more of the cache\'s extra memory";\n', "extra memory"),
            ("a pass count", 'const A: &str = "pass 2 of 3";\n', "pass 2 of 3"),
            ("a block's shape", 'const A: &str = "rows a multiple of 64 and columns of 16";\n', "[dev]"),
        ):
            w("cli.rs", quiet + line + tail)
            got = cw.engine_source_problems(d)
            assert got and any(expect in l for l in got), (what, got)
            assert all(l.startswith("engine src/cli.rs:") for l in got), got
        w("cli.rs", quiet + tail)
        open(os.path.join(d, "data", "chat.html"), "w").write("<p>the slots of a ring</p>")
        got = cw.engine_source_problems(d)
        assert got and "data/chat.html" in got[0], got
        os.remove(os.path.join(d, "data", "chat.html"))
        # the line a problem is on is the file's (what is left out keeps its lines)
        w("cli.rs", quiet + 'const A: &str = "the second reserve";\n' + tail)
        got = cw.engine_source_problems(d)
        assert got and got[0].startswith(f"engine src/cli.rs:{quiet.count(chr(10)) + 1}:"), got
    # the repository's own engine: its release build says no method word
    assert cw.engine_source_problems(os.path.join(HERE, "..", "..", "glyd-engine")) == []
    # every development command devcmd.rs has is one the gate does not let a release build name (a new one is not, until ENGINE_DENY has it)
    devcmd = open(os.path.join(HERE, "..", "..", "glyd-engine", "src", "devcmd.rs"), encoding="utf-8").read()
    commands = set(re.findall(r'^\s*"([a-z0-9]+)"\s*=>', devcmd[devcmd.index("pub fn run("):], re.M))
    assert {"pack", "kvcal", "bench", "stress", "recompute", "rowcheck", "calcheck", "topkcheck", "kcheck", "bandwidth", "kvtest", "info", "check", "score", "generate", "nhtoy", "nhdecode", "glmkernels", "glmlayers", "glmtoy", "glmmtpreal", "glmreal", "glmmtp", "glmspec", "glmprof", "glmdequant", "glmdecode", "glmserve", "glmcbtest", "glmcbblock", "cb2check", "cb2prefill", "sampletime", "nhoffload", "cpuexperts", "hbcheck", "hbtoy", "diskcheck"} <= commands, commands
    assert [c for c in sorted(commands) if not cw.ENGINE_DENY.search(f"try glyd-engine {c} MODEL")] == []
    # and the development flags of the expert cache's host half (a release build takes --cpu-threads alone of them)
    assert [f for f in ("offload-mode", "offload-window", "offload-promote", "cpu-pin") if not cw.ENGINE_DENY.search(f"glyd-engine run MODEL --{f} 1")] == []
    assert not cw.ENGINE_DENY.search("glyd-engine run MODEL --cpu-threads 8")
    # and --host-memory (the computer's memory for the experts, the SSD tier's budget) is the user's, in a release build too
    assert not cw.ENGINE_DENY.search("glyd-engine run MODEL --host-memory 20") and not cw.ENGINE_DENY.search("glyd-engine serve MODEL --host-memory auto")


def test_gate_reads_the_sources_as_a_release_build_compiles_them():
    """check_wheel reads a source without its dev blocks, as setup.py compiles it (the same function, here on every module and on a few lines);
    an identifier of a method word, a variable that tunes the code, a name that only begins like one: each is found where it is, and not in a dev block."""
    cw = check_wheel()
    tree = ast.parse(open(os.path.join(HERE, "..", "setup.py"), encoding="utf-8").read())
    ns = {"re": re}
    exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "strip_dev"], []), "setup.py", "exec"), ns)
    samples = ["a = 1\n# dev\nb = 2\n# end dev\nc = 3\n", "# dev: for the tuning\nX = 1\nY = 2\n# end dev\n", "no block\n", "x = 1 # dev\n"]
    samples += [open(os.path.join(SRC, f), encoding="utf-8").read() for f in sorted(os.listdir(SRC)) if f.endswith(".py")]
    assert all(ns["strip_dev"](t, "t") == cw.shipped(t) for t in samples)
    assert cw.shipped(samples[0]) == "a = 1\n\n\n\nc = 3\n", "the lines of what is left keep their numbers"
    with tempfile.TemporaryDirectory() as d:
        def source(text):
            open(os.path.join(d, "m.py"), "w").write(text)
            return cw.ident_problems(d) + cw.env_problems(d)

        assert source("x = 1\n") == []
        for what, text, word in (("a function", "def route_big():\n    pass\n", "route"), ("a constant", "KV_RAW_RING_AT = 1\n", "ring"), ("a class", "class Escape:\n    pass\n", "escape"),
                                 ("a name made of words", "_sign_mantissa = 1\n", "mantissa"), ("an attribute", "import os\nos.arena_x\n", "arena"), ("a keyword", "f(tile=1)\n", "tile"),
                                 ("a string that is a name", "KEY = 'arena_pool'\n", "arena"), ("a word inside a word", "nslabs = 1\n", "slab"),
                                 ("a variable that tunes", "import os\nos.environ.get('GLYD_WG_MAX')\n", "GLYD_WG_MAX"), ("one in a message", "print('set GLYD_KV_RESERVE_2 to 4')\n", "GLYD_KV_RESERVE_2")):
            got = source(text)
            assert got and any(word in line for line in got), (what, got)
        for what, text in (("the variables a user sets", "import os\nos.environ.get('GLYD_KV'), os.environ.get('GLYD_SPLIT_MIN')\n"), ("another library's names", "x.split('a')\nslot_mapping = 1\nblock_pool = 2\n"),
                           ("a whole name that is a plain word", "def split(a):\n    return a\n"), ("a dev block", "# dev\nimport os\nos.environ.get('GLYD_WG_MAX')\ndef route_big():\n    pass\n# end dev\n"),
                           ("a docstring", '"""the tiered layout, GLYD_WG_MAX"""\n')):
            assert source(text) == [], (what, source(text))


def test_gate_reads_a_table_that_cython_compresses():
    """The decoder of check_wheel (Cython's LZSS) gives the text of a table whose words the module's bytes do not show, and finds it among other bytes."""
    cw = check_wheel()
    blob = lzss_fixture("glyd_gpu/fit.py")
    assert b"tiered" not in blob
    filler = random.Random(875).randbytes  # (other bytes, the same on every run: bytes drawn afresh can decode to more words and fail a release at random)
    text = cw.string_table(filler(300) + blob + filler(300), b"glyd_gpu/fit.py")
    assert text is not None and "tiered" in text and "glyd_gpu/fit.py" in text, text
    assert cw.string_table(filler(2000), b"glyd_gpu/fit.py") is None
    assert [m.group() for m in cw.STABLE.finditer(text)] == ["tiered"]
    # the real thing: a compiled module of this package (when this runs on a wheel's) shows the names of its source in its table, or the reader is blind
    if not glyd_gpu.__file__.endswith(".py"):
        for name in ("kvcal", "vllm_kv", "model"):
            f = importlib.util.find_spec(f"glyd_gpu.{name}").origin  # (found, not imported: some need numpy or torch)
            data = open(f, "rb").read()
            names = cw.source_names(cw.shipped(open(os.path.join(SRC, name + ".py"), encoding="utf-8").read())) if os.path.exists(os.path.join(SRC, name + ".py")) else set()
            table = cw.string_table(data, f"glyd_gpu/{name}.py".encode(), names)
            assert table is not None, name
            assert not names or sum(x in table for x in names) >= 0.9 * len(names), (name, sorted(x for x in names if x not in table)[:10])


def test_gen_names():
    """gpu/gen_names.py, which a release library's names come from: the declarations of a source, and glyd_gpu.cu's own: each kernel and device function,
    none of the C API's, the same header for the same source."""
    import importlib.util

    gpu = os.path.join(HERE, "..", "..", "gpu")
    spec = importlib.util.spec_from_file_location("gen_names", os.path.join(gpu, "gen_names.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    sample = """
    namespace room { struct Tiered { int x; }; __device__ __forceinline__ int helper(int a) { return a; } }
    template <int N> __global__ void __launch_bounds__(128) work_kernel(const int* p) { __shared__ int tile[16]; }
    __device__ int lookup_table[4];  // __device__ int in_a_comment() {}
    extern "C" int glyd_gpu_thing(int a) { return a; }
    int main() { return 0; }
    """
    found = gen.names(sample)
    assert found["k"] == ["work_kernel"] and found["d"] == ["helper"] and found["v"] == ["lookup_table"] and set(found["t"]) == {"room", "Tiered"}, found
    # a macro defined as __device__ qualifies as it does (kernels/gemm_idx.h's GEMM_HD), and its own definition is no declaration
    macro = """
    #ifdef __CUDACC__
    #define HD __host__ __device__ __forceinline__
    #else
    #define HD inline
    #endif
    template <int BK> HD int swz_key(int r) { return r; }
    struct Fill { static HD int row(int tid) { return tid; } };
    template <int BK> HD int frag_a_off(int base_row) { return base_row; }
    """
    found = gen.names(macro)
    assert found["d"] == ["frag_a_off", "swz_key"] and found["t"] == ["Fill"] and found["k"] == [], found
    cu = open(os.path.join(gpu, "glyd_gpu.cu"), encoding="utf-8").read()
    found = gen.names(cu)
    every = [n for kind in found.values() for n in kind]
    assert len(every) == len(set(every)), "a name of two kinds"
    assert not [n for n in every if n.startswith(("glyd_gpu_", "GLYD_", "__"))], "the C API's names stay"
    kernels = set(re.findall(r"__global__[^{;]*?\b(\w+_kernel)\s*\(", cu))
    assert kernels and kernels <= set(found["k"]), sorted(kernels - set(found["k"]))
    text, table = gen.header(found)
    assert text == gen.header(gen.names(cu))[0] and len({new for new, *_ in table}) == len(table) == len(every)
    assert all(re.fullmatch(r"g[kdvt]_\d+", new) for new, *_ in table)


def test_engine_names():
    """The engine's kernels under neutral names in a release build (build.rs runs gpu/gen_names.py on kernels/gemm_idx.h and engine.cu): it reads every kernel (each written out, not made by a macro) and
    every device function (those of gemm_idx.h too, which a macro qualifies), and every kernel that kern.rs names through kname! is one of them, so that a release build's lookups find them."""
    cw = check_wheel()
    engine = os.path.join(HERE, "..", "..", "glyd-engine")
    gen_spec = importlib.util.spec_from_file_location("gen_names", os.path.join(HERE, "..", "..", "gpu", "gen_names.py"))
    gen = importlib.util.module_from_spec(gen_spec)
    gen_spec.loader.exec_module(gen)
    src = "".join(open(os.path.join(engine, "kernels", f), encoding="utf-8").read() for f in ("gemm_idx.h", "engine.cu"))
    found = gen.names(src)
    declared = set(re.findall(r'extern "C" __global__ void (?:__launch_bounds__\([^)]*\) )?(\w+)\(', src))
    assert declared and declared == set(found["k"]) == cw.engine_kernel_names(engine), (sorted(declared ^ set(found["k"])), "a kernel the generator does not see: it would keep its name")
    assert {"swz_key", "swzk", "frag_a_off", "frag_b_off", "block_sum", "prefill_attn_t", "gemm3_nt_t"} <= set(found["d"]) and {"Fill", "bf16"} <= set(found["t"]), found
    assert "name" not in found["k"], "a macro's parameter is no kernel"
    used = set(re.findall(r'kname!\("(\w+)"\)', open(os.path.join(engine, "src", "kern.rs"), encoding="utf-8").read()))
    assert used and used <= set(found["k"]), sorted(used - set(found["k"]))
    assert not [k for k in found["k"] if k in used and not re.fullmatch(r"\w+", k)]
    header, table = gen.header(found)
    assert header == gen.header(gen.names(src))[0] and {new for new, *_ in table} >= {f"gk_{i}" for i in range(len(found["k"]))}, "the same source gives the same header"


def test_strings_say_no_method():
    """What a user reads in the logs, the errors and the help (the strings of glyd_gpu's modules, the Python source and the page the wheel
    ships) says what Glyd does, never how it is built: check_wheel.METHOD's words are in none of them (the wheel's gate, here on the sources)."""
    cw = check_wheel()
    src = os.path.join(HERE, "..", "glyd_gpu")
    found = {mod: sorted({w for t in texts for w in cw.method(t)}) for mod, texts in cw.prose(src).items()}
    found.update({f: cw.method(open(os.path.join(src, f), encoding="utf-8").read()) for f in ("__main__.py", "_trace.py", "chat.html")})
    assert not {m: w for m, w in found.items() if w}, found
    assert cw.ident_problems(src) == [] and cw.env_problems(src) == [], (cw.ident_problems(src), cw.env_problems(src))  # (no identifier of a method word, no variable that tunes the code, as a release build compiles them)


def test_public_header():
    """include/glyd_gpu.h, the C header the wheel ships, against the library's: each function it declares is one gpu/glyd_gpu.h
    declares, by the same arguments (pointers as pointers, the rest the same), and one glyd_gpu.cu defines; the
    version the same; and nothing in it of how the library is built (the names of its routes and GPU classes, tile and
    ring sizes, byte layouts, the environment variables that tune it), as what it does not say is not in the wheel. It
    compiles as C and as C++ (where a compiler is at hand, against a stub of CUDA's header)."""
    gpu = os.path.join(HERE, "..", "..", "gpu")
    text = open(os.path.join(HERE, "..", "include", "glyd_gpu.h")).read()
    bare = lambda t: re.sub(r"/\*.*?\*/", "", t, flags=re.S)

    def declared(h):  # {name: (return type, [each argument: "P" for a pointer or an array, else its type])}
        out = {}
        for ret, name, args in re.findall(r"(int|const char\*) (glyd_gpu_\w+)\(([^)]*)\);", " ".join(bare(h).split())):
            kinds = []
            for a in ([] if args.strip() == "void" else args.split(",")):
                t = " ".join(a.split()).replace(" *", "*").rsplit(" ", 1)[0].replace("const ", "")
                kinds.append("P" if t.endswith("*") or t == "cudaStream_t" or "[" in a else t)
            out[name] = (ret, kinds)
        return out

    pub, internal = declared(text), declared(open(os.path.join(gpu, "glyd_gpu.h")).read())
    assert len(pub) == 8 and set(pub) <= set(internal), sorted(set(pub) - set(internal))
    assert {n: pub[n] for n in pub} == {n: internal[n] for n in pub}, [n for n in pub if pub[n] != internal[n]]
    cu = open(os.path.join(gpu, "glyd_gpu.cu")).read()
    assert set(pub) <= set(re.findall(r"GLYD_GPU_API [^(]*?(glyd_gpu_\w+)\(", cu))
    exports = set(re.findall(r"glyd_gpu_\w+", re.search(r"global: ([^\n]*?) local", open(os.path.join(gpu, "build_lib.sh")).read()).group(1)))  # (a release library's: the eight and the table)
    assert exports == set(pub) | {"glyd_gpu_table"}, sorted(exports ^ (set(pub) | {"glyd_gpu_table"}))
    assert check_wheel().PUBLIC == exports, "check_wheel.PUBLIC is not the version script's"
    assert set(re.findall(r"GLYD_GPU_API [^(]*?(glyd_gpu_\w+)\(", cu)) == set(pub) | {"glyd_gpu_table"}, "a function other than the public eight and the table is GLYD_GPU_API: it would be exported"
    version = lambda h: int(re.search(r"#define GLYD_GPU_API_VERSION (\d+)", h).group(1))
    assert version(text) == version(open(os.path.join(gpu, "glyd_gpu.h")).read()), "GLYD_GPU_API_VERSION is not gpu/glyd_gpu.h's"
    assert re.findall(r"#define (GLYD_\w+)", text) == ["GLYD_GPU_H", "GLYD_GPU_API_VERSION", "GLYD_GPU_DONE_COUNTERS", "GLYD_GPU_AUTO"]
    told = check_wheel().told(text)  # (the release gate's list)
    assert not told, told
    # the counters a product asks of its done array: at most DONE_COUNTERS(O, M) (the tile counts of glyd_gpu.cu: (M + 127) / 128 x O / 64, at least 1024)
    bound = lambda O, M: 1024 + O * (M // 1024 + 1)
    for O in (64, 4096, 151936):
        for M in (1, 64, 128, 1023, 1024, 4096, 100000):
            assert bound(O, M) >= max(1024, (M + 127) // 128 * (O // 64)), (O, M)
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc:
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "stub"))
            with open(os.path.join(d, "stub", "cuda_runtime_api.h"), "w") as f:
                f.write("typedef struct CUstream_st* cudaStream_t;\n")
            with open(os.path.join(d, "t.c"), "w") as f:
                f.write('#include "glyd_gpu.h"\nint main(void) { size_t n; return glyd_gpu_api_version() + (int)(sizeof(GLYD_GPU_DONE_COUNTERS(3, 4)) + GLYD_GPU_AUTO) + glyd_gpu_mma_linear_workspace(64, 64, 1, GLYD_GPU_AUTO, &n); }\n')
            inc = ["-I", os.path.join(HERE, "..", "include"), "-I", os.path.join(d, "stub")]
            for lang in ("c", "c++"):
                r = subprocess.run([cc, "-fsyntax-only", "-Wall", "-Werror", "-x", lang, *inc, os.path.join(d, "t.c")], capture_output=True, text=True)
                assert r.returncode == 0, (lang, r.stderr)


def test_public_header_runs():
    """include/glyd_gpu.h by itself, as a C user has it: public_header_check.c, compiled against it and linked to the library, unpacks
    the arrays of a pack in both layouts bit for bit and multiplies by them as the header says (the library's own way, counters of
    GLYD_GPU_DONE_COUNTERS, the words as the header builds them from a save's). Needs a GPU, the library, a C compiler and CUDA's
    headers and runtime (the nvidia packages PyTorch brings), else skipped."""
    torch = cuda()
    cc = shutil.which("cc") or shutil.which("gcc")
    if torch is None or cc is None:
        return print("test_public_header_runs: skipped (no CUDA GPU, PyTorch, transformers or C compiler)")
    import glob
    import struct
    from glyd_gpu import kernels as g

    major = torch.version.cuda.split(".")[0]
    lib = os.environ.get("GLYD_GPU_LIB") or os.path.join(PKG, f"libglyd_gpu_cuda{major}.so")
    where = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
    headers = glob.glob(os.path.join(where, "nvidia", "*", "include", "cuda_runtime_api.h"))
    runtime = glob.glob(os.path.join(where, "nvidia", "*", "lib*", "libcudart.so.*"))
    if not os.path.exists(lib) or not headers or not runtime:
        return print("test_public_header_runs: skipped (no library, CUDA headers or runtime to link)")
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(0)
        O, K = 384, 512
        w = (torch.randn(O, K, device="cuda") * 0.05).to(torch.bfloat16)
        w[::7, ::5] = 0  # (zeros: a 12-bit pack's exceptions)

        def raw(name, t):
            with open(os.path.join(d, name), "wb") as f:
                f.write(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())

        raw("w.bf16", w)
        p, q = g.pack_mma(w), g.pack_mma12(w)
        for name, t in (("mma.data", p.data), ("mma.blocks", p.blocks), ("mma.block_base", p.block_base), ("mma12.data", q.data), ("mma12.exc", q.exc), ("mma12.exc_base", q.exc_base)):
            raw(name, t)
        with open(os.path.join(d, "mma.tiers"), "wb") as f:
            f.write(struct.pack("<3I", *[int(t) for t in p.tiers]))
        with open(os.path.join(d, "mma12.hb"), "w") as f:
            f.write(str(q.hb))  # (glyd.json's hb: the header builds the four words from it)
        for M in (1, 3, 40, 200):
            x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
            raw(f"x{M}.bf16", x)
            raw(f"y{M}.f32", x.float() @ w.float().T)
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{O} {K}\n")
        exe = os.path.join(d, "check")
        cudart = sorted(runtime)[-1]
        subprocess.run([cc, "-O2", "-Wall", "-Werror", "-I", os.path.join(HERE, "..", "include"), "-I", os.path.dirname(headers[0]), os.path.join(HERE, "public_header_check.c"),
                        lib, cudart, f"-Wl,-rpath,{os.path.dirname(lib)}:{os.path.dirname(cudart)}", "-lm", "-o", exe], check=True)
        r = subprocess.run([exe, d], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.rstrip().endswith("ok") and "DIFFERENT" not in r.stdout, (r.stdout, r.stderr)
        assert r.stdout.count("bit for bit") == 2, r.stdout


def test_split_route():
    """The route SPLIT (option 2) through the library where it and a GPU are here (else skipped): its rule's pins (an
    A100 SXM from 769 to 4096 tokens, to 8192 for O and K at least 5120; a GH200 or an H100 SXM from 2048 to 8192 for
    such a matrix; no H200, H100 NVL or PCIe card; its decode's SMs) and the SM counts glyd_gpu's Linears ask by
    (Split.measured), asked for (GLYD_GPU_WITH_SPLIT), and without the flag today's routes (v0.25.1's); a GLinear
    made as on an A100 SXM by it at 1024 tokens (Split.run called, where the split can run here): within 1e-2 of fp32
    and the same bits run to run, its decode bit for bit the pack's, exact never by it (F.linear's bits), today's route
    (decoded, then cuBLAS) where the route cannot run, and its ring let go with its model, the next one's made anew."""
    torch = cuda()
    if torch is None:
        print("  (no GPU: skipped)")
        return
    import torch.nn.functional as F
    from glyd_gpu import kernels as g, model as gm

    if g.lib() is None:
        print("  (no library: skipped)")
        return
    if any(os.environ.get(v) for v in ("GLYD_SPLIT_MIN", "GLYD_SPLIT_MAX", "GLYD_SPLIT_SMS")):
        print("  (GLYD_SPLIT_* set: pins skipped)")
    else:
        w = (torch.randn(512, 1024, device="cuda") * 0.02).to(torch.bfloat16)
        q = g.pack_mma12(w)
        big = g.Mma12((131072, 1024), q.data, q.exc, q.exc_base, q.hb)  # (its shape alone read by the routes)
        wide = g.Mma12((5120, 5120), q.data, q.exc, q.exc_base, q.hb)  # (a large matrix on Hopper)
        eight = g.Mma12((4096, 12288), q.data, q.exc, q.exc_base, q.hb)  # (Qwen3-8B's class there: not large)
        for p, gpu, M, way, cus in [(q, 80, 768, g.X1, 0), (q, 80, 769, g.PART, 12), (q, 80, 1536, g.PART, 8), (q, 80, 4096, g.PART, 4), (q, 80, 4097, g.DECODE, 0),
                                      (q, 80, 8192, g.DECODE, 0), (wide, 80, 4097, g.PART, 4), (wide, 80, 8192, g.PART, 4), (wide, 80, 8193, g.DECODE, 0), (eight, 80, 4097, g.DECODE, 0),
                                      (q, 5080, 769, g.DECODE, 0), (q, 90, 2048, g.DECODE, 0), (wide, 6090, 1024, g.X3, 0), (wide, 6090, 2047, g.DECODE, 0),
                                      (wide, 6090, 2048, g.PART, 12), (wide, 6090, 6144, g.PART, 4), (wide, 6090, 8192, g.PART, 4), (wide, 6090, 8193, g.DECODE, 0),
                                      (eight, 6090, 2048, g.DECODE, 0), (eight, 6090, 8192, g.DECODE, 0), (wide, 5090, 2048, g.DECODE, 0),
                                      (wide, 7090, 1024, g.X3, 0), (wide, 7090, 2047, g.DECODE, 0), (wide, 7090, 2048, g.PART, 12), (wide, 7090, 6144, g.PART, 4),
                                      (wide, 7090, 8192, g.PART, 4), (wide, 7090, 8193, g.DECODE, 0), (eight, 7090, 2048, g.DECODE, 0), (eight, 7090, 8192, g.DECODE, 0),
                                      (wide, 90, 2048, g.DECODE, 0), (wide, 90, 8192, g.DECODE, 0), (q, 3089, 4096, g.DECODE, 0), (q, 4089, 4096, g.EARLY, 0),
                                      (q, 86, 4096, g.X1, 0), (q, 89, 4096, g.X1, 0)]:
            assert g.way(p, gpu | g.WITH_PART, M)[0] == way and g.part_cus(p, gpu | g.WITH_PART, M) == cus, (p.shape, gpu, M)
            assert g.part_cus(p, gpu, M) == 0 and g.way(p, gpu, M)[0] == (g.DECODE if way == g.PART else way), ("not asked: v0.25.1's route", p.shape, gpu, M)
        assert g.way(big, 80 | g.WITH_PART, 4096)[0] == g.PART and g.way(big, 80 | g.WITH_PART, 4097)[0] == g.DECODE and g.way(big, 6090 | g.WITH_PART, 8192)[0] == g.DECODE
        assert g.way(big, 7090 | g.WITH_PART, 8192)[0] == g.DECODE
        assert g.way(q, 80, 1024)[0] == g.DECODE and g.way(wide, 6090, 2048)[0] == g.DECODE and g.way(wide, 7090, 2048)[0] == g.DECODE  # (not asked: v0.25.1's routes)
        assert g.way(g.pack_mma(w), 80, 1024)[0] == g.X1  # (the tiered layout's: never)
        # Split.measured, the SM count glyd_gpu's Linears ask by (the properties faked, no PCI address: the name alone): an A100 SXM4's 108, a GH200's and an
        # H100 SXM's 132; not a MIG slice (7g.80gb's 132 included), an H100 PCIe's 114, an H200 or an H100 NVL (code 90), nor another count
        import types
        props = torch.cuda.get_device_properties
        try:
            for name, cus, code, takes in [("NVIDIA A100-SXM4-40GB", 108, 80, True), ("NVIDIA A100-SXM4-40GB", 98, 80, False), ("NVIDIA A100-SXM4-40GB MIG 3g.20gb", 42, 80, False),
                                           ("NVIDIA GH200 480GB", 132, 6090, True), ("NVIDIA H100 80GB HBM3", 132, 7090, True), ("NVIDIA H100 80GB HBM3", 114, 7090, False),
                                           ("NVIDIA H100 80GB HBM3 MIG 7g.80gb", 132, 7090, False), ("NVIDIA H100 PCIe", 114, 5090, False), ("NVIDIA H200", 132, 90, False),
                                           ("NVIDIA H100 NVL", 132, 90, False)]:
                torch.cuda.get_device_properties = lambda d, name=name, cus=cus: types.SimpleNamespace(name=name, multi_processor_count=cus)
                assert gm.AuxRun.measured(0, code) == takes, ("Split.measured", name, cus, code)
        finally:
            torch.cuda.get_device_properties = props
    torch.manual_seed(0)
    w = (torch.randn(3072, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    q = g.pack_mma12(w)
    assert torch.equal(g.mma_unpack_part(q, 16).view(torch.int16), w.view(torch.int16)), "the route SPLIT's decode: the pack's bits"
    cap, name = torch.cuda.get_device_capability, torch.cuda.get_device_name
    torch.cuda.get_device_capability, torch.cuda.get_device_name = lambda device=None: (8, 0), lambda device=None: "NVIDIA A100-SXM4-40GB"
    try:
        lin, ex = gm.GLinear(q, None), gm.GLinear(q, None, exact=True)
    finally:
        torch.cuda.get_device_capability, torch.cuda.get_device_name = cap, name
    gm.set_scratch(torch.nn.ModuleList([lin, ex]), False)
    measured, run, ran = gm.AuxRun.measured, gm.AuxRun.run, []
    gm.AuxRun.measured = staticmethod(lambda d, gpu: True)  # (as an A100 SXM's: its 108 SMs)
    gm.AuxRun.run = lambda s, lin_, x_: (ran.append(lin_.handle), run(s, lin_, x_))[1]
    d = w.device
    try:
        x = torch.randn(1024, 2048, dtype=torch.bfloat16, device="cuda")
        ys = [lin(x) for _ in range(3)]
        runs = gm.AuxRun.of.get(d) is not False
        if not runs:
            print("  (the route SPLIT cannot run on this GPU: its fallback alone checked)")
        assert ran == ([lin.handle] * 3 if runs else []), ("GLinear by the route SPLIT where it can run", ran)
        ref = F.linear(x.float(), w.float())
        assert ((ys[0].float() - ref).abs().max() / ref.abs().max()).item() < 1e-2
        assert all(torch.equal(y.view(torch.int16), ys[0].view(torch.int16)) for y in ys), "the route SPLIT: the same bits run to run"
        ran.clear()
        assert torch.equal(ex(x).view(torch.int16), F.linear(x, w).view(torch.int16)) and not ran, "exact never takes the route SPLIT"
        was = gm.AuxRun.of.get(d)
        gm.AuxRun.of[d] = False
        try:
            assert torch.equal(lin(x).view(torch.int16), F.linear(x, g.mma_unpack(q)).view(torch.int16)) and not ran, "the route SPLIT off: today's route"
        finally:
            if was is None:
                del gm.AuxRun.of[d]
            else:
                gm.AuxRun.of[d] = was
        if runs:  # its model deleted: the ring let go at once (the order's first Linear's finalizer), the next model's made anew
            s = gm.AuxRun.of[d]
            del lin
            import gc
            gc.collect()
            assert d not in gm.AuxRun.of and s.aux is None and s.buf is None, "the route SPLIT's ring let go with its model"
            torch.cuda.get_device_capability, torch.cuda.get_device_name = lambda device=None: (8, 0), lambda device=None: "NVIDIA A100-SXM4-40GB"
            try:
                lin = gm.GLinear(q, None)
            finally:
                torch.cuda.get_device_capability, torch.cuda.get_device_name = cap, name
            assert torch.equal(lin(x).view(torch.int16), ys[0].view(torch.int16)) and isinstance(gm.AuxRun.of.get(d), gm.AuxRun) and gm.AuxRun.of[d] is not s, "a new ring, the same bits"
    finally:
        gm.AuxRun.measured, gm.AuxRun.run = measured, run
        gm.AuxRun.stop(d)
        gm.AuxRun.of.pop(d, None)


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
    from glyd_gpu import _lib, kernels as g

    if g.lib() is None:
        print("  (no prebuilt library: the ring is the library's; skipped)")
        return
    buf = torch.empty(3 << 20, dtype=torch.uint8, device="cuda")
    aux = _lib.aux_create(buf, 1 << 20)
    r = _lib.aux_part(aux, 4)[0]
    _lib.aux_destroy(aux)
    if r:
        print(f"  (the split cannot run here, {_lib.error_string(r)}: skipped)")
        return
    out = subprocess.run([sys.executable, "-u", script, "--models", "0.6B,8B,14B", "--ms", "769,2048", "--passes", "3", "--quick"], capture_output=True, text=True,
                         cwd=os.path.dirname(script), env=dict(os.environ, GLYD_GPU_LIB=g.library()))  # (the library this process found: the script imports the checkout's package)
    assert out.returncode == 0 and " 0 failures" in out.stdout, (out.stdout[-4000:], out.stderr[-2000:])
    print("  " + next(l for l in out.stdout.splitlines() if l.startswith("split_stress:")))


def test_split_order():
    """The route SPLIT's order (model.Split.follow, taken from model.py with no torch, its queue faked): recorded from a
    prompt, then one queue a prompt; where the Linears that take the route change with the prompt's length (an A100's
    biggest matrices to 4096 tokens alone), the queue leaves out those that do not, and the order takes in those it
    lacks (a queue again from each such one, that prompt alone)."""
    tree = ast.parse(open(os.path.join(SRC, "model.py")).read())
    split = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuxRun")
    ns = {}
    top = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("aux_chunks", "aux_cell", "aux_plan") or isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "PART_CELL" for t in n.targets)]
    exec(compile(ast.Module(top + [n for n in split.body if isinstance(n, ast.FunctionDef) and n.name == "follow"], []), "model.py", "exec"), ns)

    class Fake:
        follow = ns["follow"]

        def __init__(self):
            self.order, self.rec, self.run_, self.pos, self.at, self.cus, self.hi, self.end = None, [], [], 0, -1, 0, -1, 0

        def plan(self):
            pass

        def top_up(self, cus):
            return 0

        def start(self, M, cus, i):  # as Split.start: the order's Linears from its i-th on that take the route now
            self.starts += 1
            self.run_, self.pos, self.cus = [j for j in range(i, len(self.order)) if self.order[j] in self.takes], 0, cus
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
    assert ns["aux_chunks"](24576, 4096, 100 << 20) == [(0, 12288), (12288, 12288)]
    layer = [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)]
    assert ns["aux_cell"](layer * 36) == ns["aux_cell"](layer * 36 + [(151936, 4096)]) == 100 << 20
    assert ns["aux_plan"](layer * 36, 100 << 20) == (6, 6)
    loose = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (12288, 4096), (4096, 12288)] * 36
    assert ns["aux_cell"](loose) == 12288 * 4096 * 2 and ns["aux_plan"](loose, 12288 * 4096 * 2) == (8, 9)
    big = [(7168, 5120), (5120, 5120), (34816, 5120), (5120, 17408)] * 40
    assert ns["aux_cell"](big) == 17408 * 5120 * 2 and ns["aux_plan"](big, 17408 * 5120 * 2) == (6, 6)
    small = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)] * 28
    assert ns["aux_cell"](small) == 6144 * 1024 * 2 and ns["aux_plan"](small, 6144 * 1024 * 2) == (5, 6)


def test_split_queue():
    """The route SPLIT's queue: model.Split's own follow, start and top_up (taken from model.py, no torch) against a
    fake library that keeps the ring's queue as glyd_gpu.cu does (a queue on one split; a product takes the front
    where it is its pack, else drops the queue and decodes its own). A model's queue left by a prompt cut short (a
    Linear followed, its product never run), the model deleted and another loaded at the same addresses: dropped
    before the other's first product, which would take the dead model's matrix as its own. A prompt whose Linears run
    on two splits (a vision tower's at 2000 tokens, the language model's at 1000, an A100's 8 and 12 SMs): every call
    by the route, none refused. A Linear one prompt called (a full-logits pass's lm_head) and the next did not: not
    queued again, until a prompt calls it."""
    import types
    tree = ast.parse(open(os.path.join(SRC, "model.py")).read())
    split = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AuxRun")
    fns = [n for n in split.body if isinstance(n, ast.FunctionDef) and n.name in ("follow", "start", "top_up")]
    mods = {}

    class Lib:  # the ring's queue, as glyd_gpu.cu keeps it: each entry the pack and the Linear it was queued for
        def __init__(self):
            self.q, self.split, self.queued, self.stale = [], None, [], []

        def aux_reset(self, aux):
            self.q.clear()
            return 0

        def mma12_aux_queue(self, aux, cus, data, exc, exc_base, sym, O, K):
            if cus != self.split and self.q:
                return 1  # (cudaErrorInvalidValue: a queue's matrices on one split)
            self.split = cus
            key = (data, exc, exc_base, tuple(sym))
            self.q.append((key, next(h for h, m in mods.items() if (m.p.data, m.p.exc, m.p.exc_base, tuple(m.p.sym)) == key)))
            self.queued.append(self.q[-1][1])
            return 0

        def product(self, h):  # mma12_ring_linear: the front where it is h's pack whole, else the queue dropped and W alone
            p = mods[h].p
            if self.q and self.q[0][0] == (p.data, p.exc, p.exc_base, tuple(p.sym)):
                if self.q.pop(0)[1] != h:
                    self.stale.append(h)
            else:
                self.q.clear()

    lib = Lib()
    g = types.SimpleNamespace(WITH_PART=1 << 20, part_cus=lambda p, gpu, M: 12 if M < 1536 else 8)  # (an A100's SMs by M)
    ns = {"_lib": lib, "_modules": mods, "g": g}
    exec(compile(ast.Module(fns, []), "model.py", "exec"), ns)

    class S:
        follow, start, top_up = ns["follow"], ns["start"], ns["top_up"]

        def __init__(self):
            self.aux, self.order, self.rec, self.run_, self.pos, self.queued, self.early, self.cus, self.at, self.hi, self.end = 1, None, [], [], 0, 0, 3, 0, -1, -1, 0

        def plan(self):
            pass

    def model(first, base, n=8):  # n Linears, their packs at base on (another model's may take the same)
        hs = list(range(first, first + n))
        for i, h in enumerate(hs):
            mods[h] = types.SimpleNamespace(p=types.SimpleNamespace(data=base + i, exc=base + 100 + i, exc_base=base + 200 + i, sym=[0, 0, 0, 0], shape=(4096, 4096)), gpu=80)
        return hs

    def call(s, h, M):  # a call by the route: the order's bookkeeping, then the product
        assert s.follow(h, M, g.part_cus(None, 0, M)) == 0, ("refused", h, M)
        lib.product(h)

    # a prompt cut short after its first Linear's follow, the model deleted, another loaded at its addresses
    s = S()
    a = model(0, 1000)
    for h in a:  # recorded
        call(s, h, 1000)
    assert s.follow(a[0], 1000, 12) == 0 and lib.q and lib.q[0][1] == a[0]  # (the next prompt cut short: its head followed, its product never run)
    for h in a:
        del mods[h]
    b = model(100, 1000)
    for h in b + b:
        call(s, h, 1000)
    assert not lib.stale, ("a dead model's matrix taken as another's", lib.stale)
    # a prompt on two splits: a vision tower at 2000 tokens (8 SMs), the language model at 1000 (12)
    s, lib.q = S(), []
    tower, lm = model(200, 2000, 4), model(300, 3000, 4)
    for p in range(3):
        for h in tower:
            call(s, h, 2000)
        for h in lm:
            call(s, h, 1000)
    assert not lib.stale
    # an lm_head called in one prompt (a full-logits pass), not in the next: not queued again, until called
    s, lib.q = S(), []
    *body, head = model(400, 4000, 4)
    for h in body + [head] + body:  # recorded with it; then a prompt without it
        call(s, h, 1000)
    lib.queued.clear()
    for h in body:
        call(s, h, 1000)
    assert head not in lib.queued and set(lib.queued) == set(body), lib.queued
    for h in body + [head]:
        call(s, h, 1000)
    assert head in lib.queued and not lib.stale


def test_names_defined_once():
    """No module of the package defines a function or class twice at its top: the later one would take every call
    meant for the earlier (read as text, no torch)."""
    for top, _, files in os.walk(SRC):
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
    classes = re.findall(r"#define (GLYD_GPU_(?:GEFORCE|A10|L4|L40S|PCIE|GH200|H100)) (\d+)", h)
    assert len(classes) == 7
    prog = "#include <cctype>\n#include <cstdio>\n#include <cstring>\n" + "".join(f"#define {k} {v}\n" for k, v in classes) + cu[a:b]
    prog += '\nint main() { char s[512]; while (fgets(s, sizeof s, stdin)) { s[strcspn(s, "\\n")] = 0; printf("%d\\n", gpu_class(s)); } }\n'
    tree = ast.parse(open(os.path.join(SRC, "model.py")).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gpu_code")
    kern = {}
    body = [n for n in ast.parse(open(os.path.join(SRC, "kernels.py")).read()).body if isinstance(n, ast.Assign) and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} <= {"GEFORCE", "A10", "L4", "L40S", "PCIE", "GH200", "H100"}]
    exec(compile(ast.Module(body, []), "kernels.py", "exec"), kern)
    py = {"re": re, "g": types.SimpleNamespace(GEFORCE=kern["GEFORCE"], A10=kern["A10"], L4=kern["L4"], L40S=kern["L40S"], PCIE=kern["PCIE"], GH200=kern["GH200"], H100=kern["H100"])}
    exec(compile(ast.Module([fn], []), "model.py", "exec"), py)
    names = ["NVIDIA A10", "NVIDIA A10-24GB", "NVIDIA A10G", "NVIDIA A100-SXM4-80GB", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4080 SUPER", "A10", "NVIDIA A10_X",
             "NVIDIA A16", "NVIDIA A2", "NVIDIA A10 PCIe", "A10 A10G", "A10G A10", "xA10", "A10x", "A10M", "NVIDIA GeForce A10", "(A10)", "A10.", "A10é", "éA10", "A10\u00a0", "", "NVIDIA H100 80GB HBM3",
             "NVIDIA L4", "L4", "NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L4 L40S", "L4-24GB", "xL4", "L4x", "NVIDIA GeForce L4", "NVIDIA A10 L4", "L4_X", "(L4)",
             "L40S", "NVIDIA L40S-48GB", "xL40S", "L40Sx", "NVIDIA L40S L4", "NVIDIA GeForce L40S", "NVIDIA L40SX", "L40S_",
             "NVIDIA H100 PCIe", "NVIDIA A100-PCIE-40GB", "pcie", "PCI", "PCIé", "xPCIEx", "NVIDIA GH200 480GB", "NVIDIA GeForce RTX 4090 PCIe", "NVIDIA L4 PCIe", "NVIDIA L40S PCIe",
             "GH200", "NVIDIA GH200 144G HBM3e", "xGH200", "GH200x", "NVIDIA GH200-96GB", "NVIDIA GH200 PCIe", "NVIDIA H200", "NVIDIA H200 NVL",
             "NVIDIA H100", "H100", "NVIDIA H100 NVL", "NVIDIA H100 NVL 94GB", "NVL H100", "NVIDIA H100-80C", "NVIDIA H100 80GB HBM3 MIG 3g.40gb", "xH100", "H100x", "NVIDIA H1000", "NVIDIA H100_X", "NVIDIA H100NVL",
             "NVIDIA H100 SXM5 80GB", "NVIDIA H100 PCIe NVL", "NVIDIA GH200 H100", "NVIDIA H100 L4", "NVIDIA A10 H100", "NVIDIA H800", "NVIDIA H20", "NVIDIA GeForce H100"]
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "cls.cpp"), "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++17", "-O1", "-o", os.path.join(d, "cls"), os.path.join(d, "cls.cpp")], check=True)
        out = subprocess.run([os.path.join(d, "cls")], input="\n".join(names) + "\n", capture_output=True, text=True, encoding="utf-8", check=True).stdout.split()
    got = dict(zip(names, map(int, out)))
    want = {n: py["gpu_code"]((8, 6), n) - 86 for n in names}
    assert len(out) == len(names) and got == want, [(n, got.get(n), want[n]) for n in names if got.get(n) != want[n]]
    pinned = ("NVIDIA A10", "NVIDIA A10G", "NVIDIA GeForce RTX 4080 SUPER", "NVIDIA L4", "NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation",
              "NVIDIA H100 PCIe", "NVIDIA A100-PCIE-40GB", "NVIDIA A10 PCIe", "NVIDIA GH200 480GB", "NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA H100 NVL")
    assert [want[n] for n in pinned] == [kern["A10"], 0, kern["GEFORCE"], kern["L4"], kern["L40S"], 0, 0, kern["PCIE"], kern["PCIE"], kern["A10"], kern["GH200"], kern["H100"], 0, 0], [(n, want[n]) for n in pinned]


def test_hf_cache_files():
    """gpu/hf_cache.py: a repo's files by name from a cache that holds a layer's shards alone (what a job's download of one layer leaves, which snapshot_download(local_files_only=True)
    refuses on huggingface_hub 1.x: IncompleteSnapshotError), a missing shard named in the error, a directory in place of an id, and moe_layer.py asking for no snapshot."""
    gpu = os.path.join(HERE, "..", "..", "gpu")
    if not os.path.exists(os.path.join(gpu, "hf_cache.py")):
        return print("  (gpu/hf_cache.py is not in the repository: skipped)")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return print("  (no huggingface_hub: skipped)")
    spec = importlib.util.spec_from_file_location("hf_cache", os.path.join(gpu, "hf_cache.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    index = {"weight_map": {"model.layers.0.mlp.experts.0.up_proj.weight": "model-00001-of-00003.safetensors", "model.layers.9.mlp.experts.0.up_proj.weight": "model-00002-of-00003.safetensors",
                            "lm_head.weight": "model-00003-of-00003.safetensors"}}
    with tempfile.TemporaryDirectory() as d:
        base, commit = os.path.join(d, "models--org--model"), "a" * 40
        snap = os.path.join(base, "snapshots", commit)
        os.makedirs(snap)
        os.makedirs(os.path.join(base, "refs"))
        open(os.path.join(base, "refs", "main"), "w").write(commit)
        for name, text in (("config.json", "{}"), ("model.safetensors.index.json", json.dumps(index)), ("model-00001-of-00003.safetensors", "shard")):
            open(os.path.join(snap, name), "w").write(text)
        for name in ("config.json", "model.safetensors.index.json", "model-00001-of-00003.safetensors"):
            assert mod.cached("org/model", name, cache_dir=d) == os.path.join(snap, name), name
        try:
            mod.cached("org/model", "model-00002-of-00003.safetensors", cache_dir=d)
            raise AssertionError("a shard that is not in the cache was found")
        except FileNotFoundError as e:
            assert "model-00002-of-00003.safetensors" in str(e), str(e)
        assert mod.cached(snap, "config.json") == os.path.join(snap, "config.json")  # (a directory in place of an id)
        try:
            mod.cached(snap, "model-00002-of-00003.safetensors")
            raise AssertionError("a file that is not in the directory was found")
        except FileNotFoundError as e:
            assert "model-00002-of-00003.safetensors" in str(e), str(e)
    assert "snapshot_download" not in open(os.path.join(gpu, "moe_layer.py")).read(), "moe_layer.py asks for a snapshot again"


def moe_ws_source_problems(cu):
    """What glyd_gpu.cu's sizing of the 12-bit experts' products lacks of the wgmma route's workspace, read as text: nothing where glyd_gpu_mma12_moe_workspace asks
    mma_moe_run for the 12-bit layout's own size (not the tiered layout's function's) and mma_moe_run's 12-bit branch adds the route's tail to it."""
    bad = []
    a = cu.find("int glyd_gpu_mma12_moe_workspace(")
    body = cu[a : cu.find("\n}\n", a)] if a >= 0 else ""
    if "mma_moe_run(Nib{}" not in body or "glyd_gpu_mma_moe_workspace" in body:
        bad.append("glyd_gpu_mma12_moe_workspace does not size with the 12-bit layout")
    r = cu.find("static int mma_moe_run(")
    b = cu.find("is_same_v<Fmt, Nib>", r) if r >= 0 else -1
    if b < 0 or "wg_tail(" not in cu[b : cu.find("if (need) {", b)]:
        bad.append("mma_moe_run's 12-bit branch does not add the route's tail to the size")
    return bad


def test_moe_wg_workspace():
    """The 12-bit experts' products' workspace holds the wgmma route's (an H100's default from 17 tokens: the tiles' records, the pairs' rows of X where the product has the gate, and
    the weighted product's fp32 pair rows). A product given less takes the other kernels without a word, and a release build has no variable that makes that an error, so a merge
    that gave the 12-bit layout the tiered one's size function would lose the route unseen. Read as text, anywhere (the checkout's glyd_gpu.cu): the function sizes with the
    12-bit layout, and the check itself refuses the tiered layout's function and a size without the tail; on an H100 through the library (else skipped): the bytes it gives for
    Qwen3-30B-A3B's two products and a small layer's, at 17 tokens (the route's first), 232 and 1,152, cover the route's."""
    cu = open(os.path.join(HERE, "..", "..", "gpu", "glyd_gpu.cu")).read()
    assert moe_ws_source_problems(cu) == [], moe_ws_source_problems(cu)
    a = cu.index("int glyd_gpu_mma12_moe_workspace(")
    z = cu.index("\n}\n", a)
    delegated = cu[:a] + "int glyd_gpu_mma12_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, size_t* bytes) {\n    return glyd_gpu_mma_moe_workspace(E, O, K, T, k, act, weighted, bytes);" + cu[z:]
    assert moe_ws_source_problems(delegated), "the check misses the tiered layout's size function"
    assert moe_ws_source_problems(cu.replace("wg_tail(E, K, P, act || gather)", "0")), "the check misses a size without the route's tail"
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0) or not re.search(r"\bH100\b", torch.cuda.get_device_name()) or os.environ.get("GLYD_GPU_MOE_WG"):
        print("  (no H100, or GLYD_GPU_MOE_WG set: the library's sizes not asked)")
        return
    from glyd_gpu import _lib, kernels as g

    if g.lib() is None:
        print("  (no library: skipped)")
        return
    align = lambda n: (n + 255) & ~255
    dev = torch.cuda.current_device()
    for E, I, H, k in ((128, 768, 2048, 8), (16, 256, 512, 4)):
        for T in (17, 232, 1152):
            P = T * k
            for O, K, act, weighted in ((2 * I, H, 1, 0), (H, I, 0, 1)):  # (gate and up with SiLU; down with the router's weights)
                need = (P * O * 4 if weighted else 0) + align((1 + min(E, P) + (P + 15) // 16) * 16) + (align(P * K * 2) if act else 0)
                got = _lib._need("mma12_moe", dev, (E, O, K, T, k, act, weighted))
                assert got >= need, (E, O, K, T, k, act, weighted, got, need)


def test_moe_tiled_epilogue_source():
    """mma_gemm_big_kernel's epilogue was 17,000 instructions long when it chose the gate's activation at each of its 128 products (SiLU and GELU's code, one among the other) and divided
    with the IEEE division's slow path: an instruction cache that missed, 54% of the epilogue's stalls and 30% of the kernel's warp samples (an L4, Qwen3-30B-A3B's shapes), a gate and up tile
    that spent two fifths of its time there. The activation is now chosen once, outside the loops, with the approximate reciprocal, and the down's pair rows and their weights are asked for together
    before the first store (a load that might alias a store waits for it). No result changes with either, so a merge that undid one would lose a fifth of the tiled kernel's time unseen: read
    as text (the checkout's glyd_gpu.cu), anywhere."""
    cu = open(os.path.join(HERE, "..", "..", "gpu", "glyd_gpu.cu")).read()
    a = cu.index("mma_gemm_big_kernel(Fmt f, int64_t O, int64_t K, int64_t M, int64_t stages_per_split")
    k = cu[a : cu.index("// A prompt's product on GeForce Ada", a)]
    assert "auto epilogue = [&](auto kind)" in k and "epilogue(std::integral_constant<int, 1>())" in k and "epilogue(std::integral_constant<int, 2>())" in k, "the activation is chosen once"
    assert "silu_fast(gate)" in k and "silu(gate)" not in k.replace("silu_fast(gate)", ""), "the gate's SiLU by the approximate reciprocal"
    assert "moe.act == 1 ? silu" not in k and "moe.act == 1 ? silu_fast" not in k, "no activation chosen at each product"
    assert "jr[mt][h] = (int)__ldg(order + p0 + m);" in k and "__ldg((const float*)moe.w + jr[mt][h])" in k, "the down's rows and weights asked for before the stores"
    assert "__fdividef(x, 1.f + __expf(-x))" in cu, "silu_fast's reciprocal"


def test_moe_tiled_kernel_shapes():
    """The experts' products by the tiled kernel (mma_gemm_big_kernel, the prompts' kernel on the GPUs with no wgmma route: an A100's, an L4's) in each shape it has, each against an fp32 product
    (gpu/moe_layer.py --check on small layers, a process a run): tiles of 128 tokens and of 256, the consumers' m-tiles with no pairs left out and multiplied, the gate's SiLU and GELU, the
    weighted down product, uneven loads (an expert with many tiles, one with a few pairs), a layer with exceptions by the dozen, and a layer whose K is 4 stages. A development library
    reads these variables at each call; a release library does not, and checks one shape under each name (a gate on its code, not its choices). Skipped without a GPU or a prebuilt library."""
    torch = cuda()
    if torch is None:
        print("test_moe_tiled_kernel_shapes: skipped (no CUDA GPU or PyTorch)")
        return
    from glyd_gpu import kernels

    if kernels.library() is None:
        print("  (no prebuilt library: skipped, not built here)")
        return
    gpu_dir = os.path.join(os.path.dirname(HERE), "..", "gpu")
    tiled = "tiled:GLYD_GPU_MOE_BIG=0;tiled128:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=128;tm256:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256;noskip:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_SKIP=0;tm256ns:GLYD_GPU_MOE_BIG=0,GLYD_GPU_BIG_TM=256,GLYD_GPU_BIG_SKIP=0"
    runs = (
        ("--synthetic 8,2,512,256 --tokens '1 5 17 40 100 333 1000'", ""),
        ("--synthetic 8,2,512,256 --gelu --tokens '1 17 64 300'", ""),
        ("--synthetic 6,3,256,192 --skew 1.5 --tokens '64 232 700'", ""),
        ("--synthetic 4,2,512,256 --wild 0.02 --tokens '17 232'", ""),
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for args, _ in runs:
        cmd = f"{sys.executable} -u moe_layer.py {args} --layout mma12 --check --no-time --variants \"new:;{tiled}\""
        r = subprocess.run(cmd, shell=True, cwd=gpu_dir, env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and "FAILED" not in r.stdout and r.stdout.count("checked") >= 6, (args, r.returncode, r.stdout[-1500:], r.stderr[-800:])


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
    tree = ast.parse(open(os.path.join(SRC, "model.py")).read())
    py = {"os": os, "re": re}
    for n in tree.body:
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WAY_ENV" for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name == "way_env"):
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
            py["way_env"]({n: v for n in py["WAY_ENV"]})
            got = str(int(v))
        except ValueError:
            got = "unset"
        assert got == c, (v, c, got)


def test_import_without_torch_or_library():
    """import glyd_gpu, fit, and the errors, with no PyTorch and no library: the package copied without its libraries,
    torch blocked."""
    code = """
import sys
sys.modules["torch"] = None  # as if not installed
import glyd_gpu
assert callable(glyd_gpu.fit) and "torch" not in [m for m in sys.modules if sys.modules[m]]
for call in (lambda: glyd_gpu.from_pretrained("Qwen/Qwen3-8B"), lambda: glyd_gpu.compress(None)):
    try:
        call()
        raise AssertionError("the GPU half without torch")
    except ImportError as e:
        assert "glyd[gpu]" in str(e), e
try:
    glyd_gpu.no_such_name
    raise AssertionError("AttributeError")
except AttributeError:
    pass
print("ok")
"""
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(PKG, os.path.join(d, "glyd_gpu"), ignore=shutil.ignore_patterns("libglyd*", "__pycache__"))
        r = subprocess.run([sys.executable, "-c", code], cwd=d, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, capture_output=True, text=True)
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
# packed by glyd_gpu.compress, not saved: a model glyd.from_pretrained does not load (token classification), or one
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


def test_hist_without_widening():
    """kernels._hist counts the exponents of int16 bits HIST_CHUNK a pass, without widening them: the counts the int32
    widening gave (what it did before), whatever the chunk, signs, zeros, subnormals, infinities and NaNs among them. The
    packers' int64 temporaries are between 10 MiB (below it PyTorch's allocator takes a 20 MiB segment for each) and the
    20 MiB that vLLM's load sets its max_split_size_mb to (PACK_CHUNK weights at 8 bytes); _hist's under it (HIST_CHUNK at 2)."""
    try:
        import torch
    except ImportError:
        return print("test_hist_without_widening: skipped (no torch)")
    from glyd_gpu import kernels as g

    gen = torch.Generator().manual_seed(0)
    w = (torch.randn(5000, generator=gen) * 10).to(torch.bfloat16)
    w[::7] = 0
    w[3::11] = float("inf")
    w[5::13] = -float("inf")
    w[9::17] = float("nan")
    w[2::19] = 1e-40
    w[4::23] *= -1
    u = w.view(torch.int16)
    want = torch.bincount(((u.to(torch.int32) & 0xFFFF) >> 7) & 0xFF, minlength=256)
    was = g.HIST_CHUNK
    try:
        for chunk in (1, 7, 4999, 5000, 5001, 1 << 22):
            g.HIST_CHUNK = chunk
            assert torch.equal(g._hist(u), want), chunk
    finally:
        g.HIST_CHUNK = was
    assert 10 * 2**20 <= g.PACK_CHUNK * 8 < 20 * 2**20 and g.HIST_CHUNK * 2 < 20 * 2**20


def mma_weights(n, wild, seed):
    """glyd-gpu/src/pack.rs's tests' weights (xorshift64): a trained matrix's spread, `wild` of them anywhere, as uint16 [n]."""
    import numpy as np

    M, x, out = (1 << 64) - 1, ((seed * 0x9E3779B97F4A7C15) & (1 << 64) - 1) | 1, np.empty(n, dtype=np.uint16)
    for i in range(n):
        x ^= (x << 13) & M
        x ^= x >> 7
        x ^= (x << 17) & M
        if ((x >> 8) % 1_000_000) / 1e6 < wild:
            out[i] = (x & 1) << 15 | (x >> 32) & 0x7FFF
        else:
            out[i] = (x & 1) << 15 | (115 + (x >> 40) % 12 + (x >> 48) % 3) << 7 | (x >> 20) & 0x7F
    return out


def mma_profile(rb, ks, t3, lit, t1, ballast):
    """pack.rs's tests' profile, uint16 [64 (rb + ballast), 16 ks]: step s of the first rb row blocks has t3(s) weights of ranks 6-9,
    lit(s) literals (rank 10 on) and t1(s) escapes (rank 3 on); the ballast row blocks hold the six commonest exponents."""
    import numpy as np

    RB, q = rb + ballast, np.arange(1024)
    arr = np.zeros((RB * ks, 1024), dtype=np.uint16)
    for s in range(RB * ks):
        a, b, c = min(t1(s), 1024), t3(s), lit(s)
        if s >= rb * ks:
            k = (q + 3 * s) % 12
            e = np.where(k < 9, 100 + k % 3, 94 + k)
        else:
            e = np.where(q < c, 150 + (7 * q + s) % 100, np.where(q < c + b, 106 + (q + s) % 4, np.where(q < max(a, c + b), 103 + (q + s) % 3, 100 + (q + s) % 3)))
        arr[s, (389 * q + 5 * s) % 1024] = (((q + s) & 1) << 15) | (e << 7) | ((13 * q + 5 * s) & 0x7F)
    return arr.reshape(RB, ks, 8, 4, 8, 2, 2).transpose(0, 4, 2, 1, 5, 3, 6).reshape(64 * RB, 16 * ks)  # [step, lane = 4g + t, i = 4n + 2jh + jl] -> W


def mma_matrices():
    """The matrices of pack.rs's tiered_is_pack_mma, as (name, uint16 [rows, cols]): a trained matrix's spread with 0-100% wild
    weights, every bf16 bit pattern, and matrices whose steps are at the edges of what the packer's flag holds (more than 512
    escapes or 32 literals, fields at the end of the first 124 bytes of a block's window)."""
    import numpy as np

    every = (np.arange(65536, dtype=np.int64) ^ 0x8000).astype(np.uint16)
    near = lambda f: np.where(np.arange(65536) % 10 != 0, f(every.astype(np.int64)), every.astype(np.int64)).astype(np.uint16).reshape(256, 256)
    out = [(f"weights({o * k}, {wild}, {o * k})", mma_weights(o * k, wild, o * k).reshape(o, k)) for o, k, wild in [(64, 16, 0.0), (64, 64, 0.5), (128, 48, 0.02), (192, 1040, 0.001), (256, 4096, 0.1), (64, 16, 1.0)]]
    out += [("every pattern", every.reshape(256, 256)), ("near v & 0x807F", near(lambda v: v & 0x807F)), ("near v | 0x7F00", near(lambda v: v | 0x7F00))]
    out += [("profile 440 + s * 3 / 4, s % 3, 512", mma_profile(1, 96, lambda s: 440 + s * 3 // 4, lambda s: s % 3, lambda s: 512, 3)),
            ("profile 150 + s, s * 2 / 5, 420", mma_profile(1, 90, lambda s: 150 + s, lambda s: s * 2 // 5, lambda s: 420, 3)),
            ("profile s, 0, 480 + s", mma_profile(1, 64, lambda s: s, lambda s: 0, lambda s: 480 + s, 3)),
            ("profile 20, 10 + s % 23, 300", mma_profile(1, 100, lambda s: 20, lambda s: 10 + s % 23, lambda s: 300, 3))]
    return out


def test_mma_is_the_rust_packers():
    """kernels.pack_mma against the glyd-gpu crate's pack_tiered, byte for byte (glyd-gpu/src/pack.rs's tiered_is_pack_mma has the
    same table): the sha256 of each pack's data, blocks, block_base and tiers (little-endian), the steps that have the flag and the
    length of the blocks, here on the CPU; and every pack decoded by the library's unpack on a GPU (where there is one)."""
    try:
        import torch
    except ImportError:
        return print("test_mma_is_the_rust_packers: skipped (no torch)")
    import hashlib
    import numpy as np
    from glyd_gpu import kernels as g

    want = [
        ("weights(1024, 0.0, 1024)", 1, 977, "b3600cb10d2363283b301f3d46c851733a7e789d0504cae9c9870e32cf993349"),
        ("weights(4096, 0.5, 4096)", 4, 6674, "d59d2f07b5b94191bc6362bfbfb830cbfe1a3e065cb57e445a513db2d229dbaf"),
        ("weights(6144, 0.02, 6144)", 6, 4604, "76ab1ebc4c36559143b24140d832b27dd06f447371d90e7de953a9b0d7d0e57a"),
        ("weights(199680, 0.001, 199680)", 195, 130862, "d7e06f0db432b53e01e3116eb1af0f77b71a8d537241efc59e82e4ff41b0d245"),
        ("weights(1048576, 0.1, 1048576)", 1024, 870909, "2d2985701f83455f599b0764592c3fc6c936a52d10ef77581016a0a2bf8e1300"),
        ("weights(1024, 1.0, 1024)", 1, 2742, "354f42722d6718934a449f54a1cb98ceb0572d2622f0e2ca15beffcd3db4b81d"),
        ("every pattern", 64, 158528, "441de428786dc3bb2ed61c92dbc1bef88078c0ccba3643b92182f45cc7d286c0"),
        ("near v & 0x807F", 64, 16396, "8f57b9d5727595b3ebd047cfa7a3e0282eb7f6d125a2d9ae9f77889b358a020b"),
        ("near v | 0x7F00", 64, 16400, "4b30cf8613f689d2f3ceb23f360cbb33c6ab307b0c451d6af624fcfa9cbf7a4a"),
        ("profile 440 + s * 3 / 4, s % 3, 512", 50, 43094, "2ad49cf6d8cbf0c9cef2b07514713cedc65b7bba37534560918c386a80d0a7a5"),
        ("profile 150 + s, s * 2 / 5, 420", 14, 35621, "174601964156b6998b701967f4014d58c6c4b1da204a83571edf46bf41517123"),
        ("profile s, 0, 480 + s", 31, 21696, "7ef96bd486edfa18de91bb349f74b8bde9ea768904b41b29ff9a96dc5d31bd3a"),
        ("profile 20, 10 + s % 23, 300", 0, 32610, "80b5e600dcd410ef985feb5e354287e9c2bf4e0804affc27e9fb081d4e7a82ad"),
    ]
    cases = mma_matrices()
    assert [n for n, _ in cases] == [n for n, *_ in want]
    gpu = torch.cuda.is_available()
    for (name, w), (_, flagged, blocks, sha) in zip(cases, want):
        wb = torch.from_numpy(w.astype(np.int16)).view(torch.bfloat16)
        p = g.pack_mma(wb)
        h = hashlib.sha256()
        for t in (p.data, p.blocks):
            h.update(t.numpy().tobytes())
        h.update(p.block_base.numpy().astype("<i4").tobytes())
        h.update(np.array(p.tiers, dtype="<u4").tobytes())
        assert (p.flagged(), p.blocks.numel(), h.hexdigest()) == (flagged, blocks, sha), name
        assert p.tiers[0] >> 24 == g.MMA_TAG and p.tiers[1] >> 24 == 0xFF
        if gpu and g.library() is not None:
            q = g.pack_mma(wb.cuda())
            assert all(torch.equal(getattr(p, a), getattr(q, a).cpu()) for a in ("data", "blocks", "block_base")) and q.tiers == p.tiers, name  # the GPU's pack is the CPU's
            assert torch.equal(g.mma_unpack(q).view(torch.int16).cpu(), wb.view(torch.int16)), name  # and the library decodes every step


def test_step_kernel_flags():
    """The product for 1-16 tokens (kernels.mma_gemm, the step kernel) on the pinned matrices of test_mma_is_the_rust_packers, where a GPU and
    the library are at hand: within 1e-2 of an fp32 product, the same bits twice, and the same bits where the flag (bit 31 of block_base) is set on
    every step, every 5th and every 64th, which sends those steps through the general decode: the short form of the decode, for the steps the
    packer does not flag, is that decode's."""
    try:
        import torch
    except ImportError:
        return print("test_step_kernel_flags: skipped (no torch)")
    import copy
    import numpy as np
    from glyd_gpu import kernels as g
    if not (torch.cuda.is_available() and g.library() is not None):
        return print("test_step_kernel_flags: skipped (no GPU or library)")
    torch.manual_seed(2)
    for name, w in mma_matrices():
        wb = torch.from_numpy(w.astype(np.int16)).view(torch.bfloat16).cuda()
        p = g.pack_mma(wb)
        O, K = wb.shape
        tame = bool(torch.isfinite(wb.double()).all()) and wb.double().abs().max().item() < 1e12
        x = torch.randn(16, K, device="cuda").to(torch.bfloat16)
        variants = []
        for every in (1, 5, 64):
            q, bb = copy.copy(p), p.block_base.clone()
            sel = torch.zeros(bb.numel(), dtype=torch.bool, device="cuda")
            sel[: bb.numel() - 1 : every] = True
            bb[sel] = bb[sel] | torch.iinfo(torch.int32).min
            q.block_base = bb
            variants.append(q)
        for M in (1, 2, 3, 8, 9, 16):
            y = g.mma_gemm(p, x[:M])
            if tame:
                ref = x[:M].double() @ wb.double().t()
                assert (y.double() - ref).abs().max().item() <= 1e-2 * ref.abs().max().item() + 1e-6, (name, M)
            assert torch.equal(y.view(torch.int16), g.mma_gemm(p, x[:M]).view(torch.int16)), (name, M)
            for q in variants:
                assert torch.equal(y.view(torch.int16), g.mma_gemm(q, x[:M]).view(torch.int16)), (name, M, "flag")


def test_rows_of_the_tiered_layout():
    """kernels.mma_rows (an embedding's lookup, one launch for any ids) on the pinned matrices of test_mma_is_the_rust_packers, where a GPU and the
    library are at hand: row ids[i] of the matrix bit for bit (repeats and the ends among the ids, an id outside [0, rows) a row of zeros), whether the
    packer flagged a step or the flag is set on every step (the lookup decodes every step as the unpack does), and a pack of the layout of 0.27 and
    before (the tag 0xFF in tiers[0]) refused."""
    try:
        import torch
    except ImportError:
        return print("test_rows_of_the_tiered_layout: skipped (no torch)")
    import copy
    import numpy as np
    from glyd_gpu import _lib, kernels as g
    if not (torch.cuda.is_available() and g.library() is not None):
        return print("test_rows_of_the_tiered_layout: skipped (no GPU or library)")
    torch.manual_seed(3)
    for name, w in mma_matrices():
        wb = torch.from_numpy(w.astype(np.int16)).view(torch.bfloat16).cuda()
        O, K = wb.shape
        p = g.pack_mma(wb)
        q = copy.copy(p)
        q.block_base = p.block_base | torch.iinfo(torch.int32).min
        ids = torch.cat([torch.randint(0, O, (100,), device="cuda"), torch.tensor([0, O - 1, O - 1], device="cuda")])
        for pack in (p, q):
            assert torch.equal(g.mma_rows(pack, ids).view(torch.int16), wb[ids].view(torch.int16)), name
        assert not g.mma_rows(p, torch.tensor([-1, O, 2**40], device="cuda")).view(torch.int16).any(), name
        old = [p.tiers[0] & 0xFFFFFF | 0xFF << 24, p.tiers[1], p.tiers[2]]
        try:
            _lib.mma_rows(p.data, p.blocks, p.block_base, old, K, O, ids, torch.empty(ids.numel(), K, dtype=torch.bfloat16, device="cuda"))
        except RuntimeError:
            continue
        raise AssertionError((name, "a pack of the earlier layout was read"))


def test_pack_smallest():
    """pack_smallest: the tiered layout of a matrix whose exponents are peaked; the 12-bit layout where its exponents are spread
    wide (it takes fewer bytes) and where more than a quarter of the tiered pack's steps have the flag though it is the smaller
    (600 escapes a step); the pack of tiers given (another's) is refused where they are another layout's; a pack's words are
    checked where it is made."""
    try:
        import torch
    except ImportError:
        return print("test_pack_smallest: skipped (no torch)")
    import numpy as np
    from glyd_gpu import kernels as g

    bf = lambda w: torch.from_numpy(w.astype(np.int16)).view(torch.bfloat16)
    peaked = bf(mma_profile(4, 64, lambda s: 5, lambda s: 1, lambda s: 70, 0))
    p = g.pack_smallest(peaked)
    assert type(p) is g.Mma and p.flagged() == 0 and p.nbytes() < 12.04 / 8 * peaked.numel()
    assert type(g.pack_smallest(bf(mma_weights(256 * 1024, 1.0, 9).reshape(256, 1024)))) is g.Mma12  # (the tiered pack of it is bigger)
    # 600 escapes in every step (ranks 3-5, the ballast making them so): the tiered pack is the smaller and nearly all its steps have the flag
    heavy = bf(mma_profile(4, 64, lambda s: 0, lambda s: 0, lambda s: 600, 3))
    q = g.pack_mma(heavy)
    twelve = g.pack_mma12(heavy)
    assert q.nbytes() < twelve.nbytes() and q.flagged() * 4 > 7 * 64, (q.nbytes(), twelve.nbytes(), q.flagged())
    assert type(g.pack_smallest(heavy)) is g.Mma12
    for words in ([1, 2, 3], [0x01000000, 2, 3]):
        try:
            g.Mma((64, 16), None, None, None, words)
            raise AssertionError("words of no layout read")
        except ValueError:
            pass
    try:
        g.pack_mma(peaked, tiers=[0xFF000000 | 1, 0xFF000002, 0xFF000003])
        raise AssertionError("another layout's words")
    except ValueError:
        pass
    assert g.pack_mma(peaked, tiers=p.tiers).tiers == p.tiers  # another pack's, as it stands


def test_earlier_packs_are_packed_again():
    """A pack of the tiered layout of glyd 0.20 to 0.27 (legacy_mma.json: v0.27's pack_mma of mma_weights(4096, 0.3, 11) as [64, 64],
    tiers[0]'s top byte 0xFF) is unpacked by the library as it was, and Mma() packs it again as pack_mma packs the weights; a product
    refuses it (the library takes the new layout's words alone). Needs a GPU and the library."""
    try:
        import torch
    except ImportError:
        return print("test_earlier_packs_are_packed_again: skipped (no torch)")
    from glyd_gpu import kernels as g

    if not torch.cuda.is_available() or g.library() is None:
        return print("test_earlier_packs_are_packed_again: skipped (no CUDA GPU or library)")
    import numpy as np

    assert g.lib() is not None  # (the library loaded)
    fx = json.load(open(os.path.join(HERE, "legacy_mma.json")))
    w = torch.from_numpy(mma_weights(64 * 64, 0.3, 11).astype(np.int16).reshape(64, 64)).view(torch.bfloat16).cuda()
    data, blocks = (torch.frombuffer(bytearray(bytes.fromhex(fx[k])), dtype=torch.uint8).cuda() for k in ("data", "blocks"))
    base = torch.tensor(fx["block_base"], dtype=torch.int32).cuda()
    assert fx["tiers"][0] >> 24 == g.OLD_TAG
    from glyd_gpu import _lib

    out = torch.empty(64 * 64, dtype=torch.int16, device="cuda")
    _lib.mma_unpack(data, blocks, base, fx["tiers"], 64, 0, 64, out, 0)  # as the library unpacked it
    assert torch.equal(out.view(64, 64), w.view(torch.int16))
    p = g.Mma((64, 64), data, blocks, base, fx["tiers"])
    q = g.pack_mma(w)
    assert p.tiers == q.tiers and all(torch.equal(getattr(p, a), getattr(q, a)) for a in ("data", "blocks", "block_base")) and torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16))
    try:
        _lib.mma_linear(data, blocks, base, fx["tiers"], 64, 64, torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda"), torch.empty(0, dtype=torch.bfloat16, device="cuda"), torch.empty(1, 64, dtype=torch.bfloat16, device="cuda"), -1)
        raise AssertionError("a product from the earlier layout")
    except RuntimeError as e:
        assert "invalid argument" in str(e), e


def embedding_classes(torch):
    """nn.Embedding classes as transformers has them: scaled (Gemma's, NLLB-MoE's: a float), normed (Inkling's, Muse
    Glimmer's), with a parameter of their own (T5Gemma 2's end-of-image row), positional (a mask in), both a scale and a norm."""
    import torch.nn as nn

    class Norm(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(n))

        def forward(self, x):
            return self.weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)

    class Scaled(nn.Embedding):
        def __init__(self, n, d):
            super().__init__(n, d)
            self.register_buffer("embed_scale", torch.tensor(d**0.5), persistent=False)

        def forward(self, ids):
            return super().forward(ids) * self.embed_scale.to(self.weight.dtype)

    class FloatScaled(nn.Embedding):
        def __init__(self, n, d):
            super().__init__(n, d)
            self.embed_scale = 2.0

        def forward(self, ids):
            return super().forward(ids) * self.embed_scale

    class Normed(nn.Embedding):
        def __init__(self, n, d):
            super().__init__(n, d)
            self.embed_norm = Norm(d)

        def forward(self, ids):
            return self.embed_norm(super().forward(ids))

    class Row(Scaled):
        def __init__(self, n, d):
            super().__init__(n, d)
            self.eoi_embedding = nn.Parameter(torch.zeros(d))

        def forward(self, ids):
            rows = super().forward(ids)
            rows[ids == 3] = self.eoi_embedding.to(rows.dtype)
            return rows

    class Positional(nn.Embedding):
        def forward(self, mask, past=0):
            return super().forward(mask.cumsum(1) + past)

    class Both(Normed):
        def __init__(self, n, d):
            super().__init__(n, d)
            self.embed_scale = 2.0

    return dict(plain=nn.Embedding, scaled=Scaled, float=FloatScaled, normed=Normed, row=Row, positional=Positional, both=Both)


def test_embedding_lookups():
    """pack_modules takes an embedding that is a lookup of its rows (nn.Embedding's forward; a scale or a norm of them,
    as transformers' Gemma, NLLB-MoE, Inkling and Muse Glimmer hold them) and leaves the rest as they are."""
    try:
        import torch
    except ImportError:
        return print("test_embedding_lookups: skipped (no torch)")
    from glyd_gpu import model as gm

    got = {k: gm.lookup(c(16, 8)) for k, c in embedding_classes(torch).items()}
    assert got == dict(plain=True, scaled=True, float=True, normed=True, row=False, positional=False, both=False), got


def test_embedding_norm_after_the_lookup():
    """A packed embedding gives what the module it took the place of gave, bit for bit: its rows scaled (embed_scale)
    or normed (embed_norm: a transformers 5.18's Inkling's), its norm's weights where the module held them."""
    torch = cuda()
    if torch is None:
        return print("test_embedding_norm_after_the_lookup: skipped (no CUDA GPU, PyTorch or transformers)")
    import torch.nn as nn
    from glyd_gpu import model as gm

    classes = embedding_classes(torch)
    gen = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, 512, (3, 5), device="cuda", generator=gen)
    for kind in ("plain", "scaled", "float", "normed"):
        emb = classes[kind](512, 128).to(torch.bfloat16).cuda()
        with torch.no_grad():
            emb.weight.normal_(0, 0.05, generator=gen)
            if kind == "normed":
                emb.embed_norm.weight.normal_(1, 0.1, generator=gen)
        want = emb(ids)
        net = nn.Sequential(emb)
        gm.pack_modules(net, lambda w, linear: gm.pack(w, linear, "mma"), lambda m: m.weight.device)
        assert isinstance(net[0], gm.GEmbedding), kind
        assert torch.equal(net[0](ids), want), kind
        assert [k for k in net.state_dict() if "embed_norm" in k] == (["0.embed_norm.weight"] if kind == "normed" else []), kind
        got = torch.compile(net)(ids)  # (a node of the graph, the norm traced after it: close, not the bits of the eager norm)
        assert torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2), (kind, (got.float() - want.float()).abs().max().item())
        torch._dynamo.reset()


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
    from glyd_gpu import moe
    from glyd_gpu.model import GLinear

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
    import glyd_gpu as glyd
    from glyd_gpu import format as fmt, model as gm, moe

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
            import glyd_gpu as gg
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
            was = fmt.TABLE_MIN
            fmt.TABLE_MIN = 0 if layout == "mma12" else was  # the 12-bit save with every table it can pack packed (glyd-v5); the tiered one as saves are by default (glyd-v4)
            try:
                glyd.save_pretrained(g, dst, tables=layout == "mma12")
            finally:
                fmt.TABLE_MIN = was
            saved = fmt.read_manifest(dst)
            assert saved["format"] == ("glyd-v5" if any(e.get("table") for e in saved["packs"].values()) else "glyd-v3" if layout == "mma12" else "glyd-v4") and (layout == "mma12" or saved["format"] == "glyd-v4"), (kind, saved["format"])
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
    import glyd_gpu as glyd
    from glyd_gpu import model as gm

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
    import glyd_gpu as gg
    from glyd_gpu import model as gm
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
    import glyd_gpu as gg
    from glyd_gpu import model as gm

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
import glyd_gpu as gg
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
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
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
    import glyd_gpu.kernels as g
    from glyd_gpu import model as gm

    if g.library() is None:
        print("test_gpu_code_loads_the_library: skipped (no prebuilt library: GLYD_GPU_LIB)")
        return
    first = "from glyd_gpu import _lib; print(_lib.gpu(), _lib.mma12_way(_lib.gpu(), 512, 1024, 1)[0])"
    r = subprocess.run([sys.executable, "-c", first], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    code, way = map(int, r.stdout.split())
    assert code == gm.gpu_code(torch.cuda.get_device_capability(), torch.cuda.get_device_name()) and way == g.GEMM, r.stdout
    env = {k: v for k, v in os.environ.items() if k != "GLYD_GPU_LIB"}
    env["PYTHONPATH"] = os.pathsep.join([ROOT] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    r = subprocess.run([sys.executable, "-c", first], capture_output=True, text=True, env=env, cwd=tempfile.gettempdir())
    beside = os.path.exists(os.path.join(os.path.dirname(g.__file__), f"libglyd_gpu_cuda{torch.version.cuda.split('.')[0]}.so"))
    assert beside or (r.returncode != 0 and "OSError: no Glyd GPU library" in r.stderr and "KeyError" not in r.stderr), r.stderr[-2000:]
    jit = "import types; from glyd_gpu import _lib, kernels; kernels._ext = types.ModuleType('glyd_gpu'); _lib.gpu()"
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
    import glyd_gpu as gg

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


def test_tables_saved_packed():
    """A save with tables=True packs a model's tables too (glyd-v5), in either layout: a tiny Gemma 4's embedding, tied
    to its output layer (which is not saved), and its per-layer one, here of 144 columns (not an embedding packed for
    the GPU: it stays bf16 there), each one pack marked "table", its sha256 glyd.json's; verify's file checks pass;
    loaded with verify, each decoded bit for bit, the output layer tied to the embedding again, the logits those of the
    model the save was made from; saved again, the same packs. By default (tables=False) the same tables are saved as
    they are (glyd-v4), as is a table under TABLE_MIN weights (test_cli_pack_and_verify)."""
    torch = cuda()
    if torch is None:
        print("test_tables_saved_packed: skipped (no CUDA GPU, PyTorch or transformers)")
        return
    import glyd_gpu as gg
    from glyd_gpu import hf
    from glyd_gpu.model import GEmbedding, GLinear
    from transformers import AutoModelForCausalLM

    cfg = tiny("gemma4", {"text": dict(vocab_size=8192, vocab_size_per_layer_input=8192, hidden_size_per_layer_input=72)})
    torch.manual_seed(0)
    emb, per = "model.language_model.embed_tokens", "model.language_model.embed_tokens_per_layer"
    with tempfile.TemporaryDirectory() as d:
        with torch.device("cuda"):
            src = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        with torch.no_grad():
            for p in src.parameters():
                if p.dim() >= 2:
                    p.normal_(0, 0.05)
        src.save_pretrained(os.path.join(d, "src"))
        del src
        ids = torch.randint(0, 8192, (2, 12), device="cuda")
        for layout in ("mma", "mma12"):
            a = hf.from_pretrained(os.path.join(d, "src"), layout=layout, verify=True, compile=False)
            gg.save_pretrained(a, os.path.join(d, "plain"), layout=layout)
            m = fmt.read_manifest(os.path.join(d, "plain"))
            assert m["format"] == ("glyd-v3" if layout == "mma12" else "glyd-v4") and {emb + ".weight", per + ".weight"} <= set(m["tensors"]) and not any(e.get("table") for e in m["packs"].values()), layout
            gg.save_pretrained(a, os.path.join(d, "out"), layout=layout, tables=True)
            m = fmt.read_manifest(os.path.join(d, "out"))
            assert m["format"] == "glyd-v5" and {emb, per} <= set(m["packs"]) and [t["name"] for t in m["packs"][per]["tensors"]] == [per + ".weight"], layout
            assert m["packs"][emb].get("table") is True and m["packs"][per].get("table") is True and sum(bool(e.get("table")) for e in m["packs"].values()) == 2, layout
            assert not any(k.startswith("lm_head") for k in [*m["packs"], *m["tensors"]]) and emb + ".weight" not in m["tensors"], layout
            assert fmt.check_files(os.path.join(d, "out"), m)[0] == len(m["tensors"])
            b = hf.from_pretrained(os.path.join(d, "out"), layout=layout, verify=True, compile=False)
            assert b.config.quantization_config.verified >= sum(len(e["tensors"]) for e in m["packs"].values())
            assert isinstance(b.get_submodule(emb), GEmbedding) and isinstance(b.lm_head, GLinear) and b.all_tied_weights_keys.get("lm_head.weight") == emb + ".weight"
            assert type(b.get_submodule(per)).forward is type(a.get_submodule(per)).forward and torch.equal(b.get_submodule(per).weight, a.get_submodule(per).weight)  # (144 columns: bf16 on the GPU)
            with torch.no_grad():
                assert torch.equal(a(ids).logits, b(ids).logits), layout
            gg.save_pretrained(b, os.path.join(d, "again"), layout=layout, tables=True)
            again = fmt.read_manifest(os.path.join(d, "again"))
            assert again["packs"] == m["packs"] and again["tensors"] == m["tensors"], layout
            del a, b
            for x in ("plain", "out", "again"):
                shutil.rmtree(os.path.join(d, x))


def test_quantized_checkpoint_refused():
    """A checkpoint quantized already (its config's quantization_config: gpt-oss's MXFP4, the FP8 releases) refused by
    glyd.from_pretrained with what to load instead, before a weight is read."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("test_quantized_checkpoint_refused: skipped (no PyTorch or transformers)")
        return
    import glyd_gpu as glyd
    with tempfile.TemporaryDirectory() as d:
        json.dump({"model_type": "qwen3_moe", "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}, open(os.path.join(d, "config.json"), "w"))
        try:
            glyd.from_pretrained(d)
            raise AssertionError("a quantized checkpoint")
        except ValueError as e:
            assert "quantized already (fp8)" in str(e) and "bf16 release" in str(e), e


def test_cli_pack_and_verify():
    """python -m glyd_gpu pack, verify and fit on a tiny mixture of experts (Qwen3-MoE's): fit counts its experts'
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
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        for args, says in ((["pack", os.path.join(d, "src"), os.path.join(d, "out")], "tensors packed and checked"), (["verify", os.path.join(d, "out")], "tensors decode to glyd.json's sha256")):
            r = subprocess.run([sys.executable, "-m", "glyd_gpu", *args], env=env, capture_output=True, text=True)
            assert r.returncode == 0 and says in r.stdout, r.stderr[-2000:]
        assert sum("experts" in e for e in fmt.read_manifest(os.path.join(d, "out"))["packs"].values()) == 4  # 2 layers' gate and up, down
        f = fit(os.path.join(d, "src"), gpu=10**9)
        assert f.bf16_weights == sum(os.path.getsize(os.path.join(d, "src", n)) for n in os.listdir(os.path.join(d, "src")) if n.endswith(".safetensors"))
        r = subprocess.run([sys.executable, "-m", "glyd_gpu", "fit", os.path.join(d, "out")], env=env, capture_output=True, text=True)
        assert r.returncode != 0 and "fit the source" in r.stderr, r.stderr[-2000:]
        os.remove(os.path.join(d, "out", fmt.MANIFEST))  # as a save over it cut short once the manifest is gone: refused, then saved over
        for args, ok in ((["verify", os.path.join(d, "out")], False), (["pack", os.path.join(d, "src"), os.path.join(d, "out")], True)):
            r = subprocess.run([sys.executable, "-m", "glyd_gpu", *args], env=env, capture_output=True, text=True)
            assert (r.returncode == 0) == ok and (ok or "a save cut short" in r.stderr), r.stderr[-2000:]
        # saved in the 12-bit layout (glyd-v3): verified; loaded for the 12-bit layout, its packs are the saved ones as
        # they are, each the pack of its weights there (and loaded tiered, packed again, its tensors as saved)
        for args, says in ((["pack", os.path.join(d, "src"), os.path.join(d, "out12"), "--layout", "mma12"], "tensors packed and checked"), (["verify", os.path.join(d, "out12")], "tensors decode to glyd.json's sha256")):
            r = subprocess.run([sys.executable, "-m", "glyd_gpu", *args], env=env, capture_output=True, text=True)
            assert r.returncode == 0 and says in r.stdout, r.stderr[-2000:]
        m12 = fmt.read_manifest(os.path.join(d, "out12"))
        assert m12["format"] == "glyd-v3" and all(e["layout"] == "mma12" and type(e["hb"]) is int and "sym" not in e for e in m12["packs"].values())
        from glyd_gpu import hf, kernels as g, model as gm
        saved, fresh = hf.from_pretrained(os.path.join(d, "out12"), layout="mma12"), hf.from_pretrained(os.path.join(d, "src"), layout="mma12")
        packs = lambda model: [m.p for m in model.modules() if isinstance(m, gm.GLinear)] + [p for m in model.modules() for p in (getattr(m, "glyd_packs", None) or {}).values()]
        assert len(packs(saved)) == len(packs(fresh)) == len(m12["packs"])
        for a, b in zip(packs(saved), packs(fresh)):
            assert type(a) is g.Mma12 and a.hb == b.hb and a.sym == b.sym and all(torch.equal(getattr(a, t), getattr(b, t)) for t in ("data", "exc", "exc_base"))
        del saved, fresh
        banded = hf.from_pretrained(os.path.join(d, "out12"), layout="mma", verify=True)
        assert all(type(p) in (g.Mma, g.Mma12) and type(p) is type(g.pack_smallest(gm.unpack(p))) for p in packs(banded))  # packed again, each the smaller of its two layouts
        assert banded.config.quantization_config.verified >= sum(len(e["tensors"]) for e in m12["packs"].values())  # (and the embeddings packed as it loads)
        del banded
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
        assert "model.norm.weight" in m["tensors"] and m["format"] == "glyd-v4"
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
            r = subprocess.run([sys.executable, "-m", "glyd_gpu", "verify", bad], env=env, capture_output=True, text=True)
            assert r.returncode != 0 and says in r.stderr, (change, r.stderr[-2000:])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
