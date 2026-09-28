"""glyd.gpu's parts that need no GPU and no PyTorch: fit against the site's
answers, the glyd-v1 names and manifest, import glyd without torch, and the
library's C header against the package's calls.
Where a CUDA GPU, PyTorch and transformers are at hand (else skipped):
every mixture-of-experts family of transformers as a tiny random model,
packed and saved (test_moe_families), and the CLI's pack and verify on one.

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
        json.dump(dict(m, format="glyd-v9"), open(os.path.join(d, fmt.MANIFEST), "w"))
        try:
            fmt.read_manifest(d)
            raise AssertionError("another format")
        except ValueError:
            pass


def test_c_header():
    """gpu/glyd_gpu.h, the library's C API, as _lib.py calls it: every function by the same arguments (their ctypes
    types, the stream last), its version API_VERSION; and the functions glyd_gpu.cu defines, which includes it (the
    compiler holds each definition to its declaration there). Read as text: _lib.py's argument lists run alone, as
    it imports torch."""
    gpu = os.path.join(HERE, "..", "..", "gpu")
    h = re.sub(r"/\*.*?\*/", "", open(os.path.join(gpu, "glyd_gpu.h")).read(), flags=re.S)
    names = {"_P", "_I64", "_U64", "_SZ", "_W", "_PACK", "_FAST", "_DENSE", "_ARGS", "_SIZES", "_PLAIN", "API_VERSION"}
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
    called.update(glyd_gpu_api_version=("int", []), glyd_gpu_cuda_version=("int", []), glyd_gpu_error_string=("const char*", [c.c_int]))
    assert declared == called, [n for n in sorted(set(declared) | set(called)) if declared.get(n) != called.get(n)]
    assert int(re.search(r"#define GLYD_GPU_API_VERSION (\d+)", h).group(1)) == lib["API_VERSION"], "GLYD_GPU_API_VERSION is not _lib.py's API_VERSION"
    cu = open(os.path.join(gpu, "glyd_gpu.cu")).read()
    assert set(re.findall(r"GLYD_GPU_API [^(]*?(glyd_gpu_\w+)\(", cu)) == set(declared), "glyd_gpu.cu's C API is not glyd_gpu.h's"


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


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
