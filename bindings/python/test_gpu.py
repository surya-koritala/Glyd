"""glyd.gpu's parts that need no GPU and no PyTorch: fit against the site's
answers, the glyd-v1 names and manifest, and import glyd without torch.

    python test_gpu.py              (or pytest test_gpu.py)

test_gpu_site.json: for each model the site lists (getglyd.com's
data/sizes.json, 2026-09-27), the Hub metadata fit reads, trimmed (the
config's attention fields; the safetensors at the repo's top, summed into
one entry where the repo has no consolidated copy), and the site's
answers."""
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glyd.gpu import fit, format as fmt  # noqa: E402

fitmod = importlib.import_module("glyd.gpu.fit")


def hub(data):
    """fit's Hub reads served from test_gpu_site.json."""
    def get(url):
        for repo, d in data.items():
            if url == f"/api/models/{repo}":
                return {"safetensors": {"parameters": d["parameters"]}}
            if url == f"/api/models/{repo}/tree/main":
                return [{"type": "file", "path": p, "size": n} for p, n in d["files"]] + [{"type": "directory", "path": "original"}]
            if url == f"/{repo}/resolve/main/config.json":
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
        assert round(f.kv_cache / 1e9, 3) == s["kv8"] and f.format == s["fmt"], repo
        assert abs(f.bf16_weights / 1e9 - s["bf"]) < 0.05, repo  # the site's measured totals (3 models) round the checkpoint's bytes
        if s["est"]:  # else the site has a measured ratio: 0.671-0.681 against fit's 0.673
            assert round(f.glyd_weights / 1e9, 2) == s["gl"], repo
    f = fit("Qwen/Qwen3-32B", gpu="48 GB")
    assert repr(f) == "Qwen3-32B on a 48 GB GPU: bf16 needs 69.3 GB, no; Glyd 47.9 GB, fits", repr(f)
    assert fit("Qwen/Qwen3-32B", gpu=81559 * 2**20).bf16_fits and repr(fit("Qwen/Qwen3-8B", gpu=16 * 10**9)).startswith("Qwen3-8B on a 16.0 GB GPU:")
    try:
        fit("Qwen/Qwen3-8B", gpu="40GB")
        raise AssertionError("an unknown GPU")
    except ValueError:
        pass


def safetensors(path, tensors, metadata=None):
    """A safetensors file of zeros: tensors {name: (dtype, shape)}."""
    size = {"BF16": 2, "F32": 4, "U8": 1, "I32": 4, "F8_E4M3": 1}
    h, at = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = size[dtype]
        for d in shape:
            n *= d
        h[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}
        at += n
    if metadata:
        h["__metadata__"] = metadata
    b = json.dumps(h).encode()
    with open(path, "wb") as f:
        f.write(len(b).to_bytes(8, "little") + b + bytes(at))


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
        assert f.bf16_weights == weights and f.format == "bf16"
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
    e = fmt.entry((6144, 4096), [1, 2, 0xFFFFFFFF], [("m.q_proj.weight", (4096, 4096), "aa"), ("m.k_proj.weight", (1024, 4096), "bb"), ("m.v_proj.weight", (1024, 4096), "cc")])
    assert fmt.members(e) == (["m.q_proj", "m.k_proj", "m.v_proj"], [4096, 1024, 1024])
    assert e["shape"] == [6144, 4096] and e["tiers"][2] == 0xFFFFFFFF and e["layout"] == "mma"
    m = fmt.manifest({"repo": "Qwen/Qwen3-8B", "revision": "abc"}, {"m.q_proj": e}, "0.20.0")
    assert fmt.shard_names(1) == ["model.safetensors"] and fmt.shard_names(3)[2] == "model-00003-of-00003.safetensors"
    with tempfile.TemporaryDirectory() as d:
        assert fmt.read_manifest(d) is None
        json.dump(m, open(os.path.join(d, fmt.MANIFEST), "w"))
        assert fmt.read_manifest(d) == m  # a JSON round trip keeps the members' order
        buffers = {fmt.key("m.q_proj", b): (fmt.DTYPES[b], [n]) for b, n in zip(fmt.BUFFERS, [30720, 5000, 25])}
        safetensors(os.path.join(d, "model.safetensors"), dict(buffers, **{"m.norm.weight": ("BF16", [4096])}), {"format": "pt"})
        assert fmt.stored([os.path.join(d, "model.safetensors")]) == {k: (shape, dtype) for k, (dtype, shape) in buffers.items()}
        json.dump(dict(m, format="glyd-v9"), open(os.path.join(d, fmt.MANIFEST), "w"))
        try:
            fmt.read_manifest(d)
            raise AssertionError("another format")
        except ValueError:
            pass


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


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
