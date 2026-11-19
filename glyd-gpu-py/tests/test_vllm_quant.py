"""glyd_gpu's vLLM plugin for quantized checkpoints (vllm_quant.py), its logic that needs no GPU: --quantization glyd on a checkpoint with
a quantization_config of its own, for each format of the 2026 checkpoints measured (FP8 with block scales, ModelOpt NVFP4 and its mixed
precision, compressed-tensors FP8-dynamic, W4A16, AWQ, W8A8 (INT8) and MXFP4, GPTQ, AWQ): claimed (vLLM's own probe, run on its real code, keeps glyd), and
vLLM's own config class for it with Glyd's put in front (the class vLLM's probe picks without --quantization, its name, pickled and
copied as vLLM does); a method Glyd does not cover refused, with why; which layers get Glyd's methods (the Linears, embedding and LM head
a checkpoint leaves 16-bit, under the options) and which keep the checkpoint's; kv lossless refused and kv off written for the workers;
bf16 only; ModelConfig.is_nvfp4_quantized as vLLM answers it for the checkpoint; GLYD_FP8=1's FP8 path as it was. Needs vLLM (no GPU),
else skipped with a line saying so (GLYD_REQUIRE_VLLM=1 makes it a failure).

    python test_vllm_quant.py              (or pytest test_vllm_quant.py)

The glyd_gpu it runs: the installed one, or this checkout's sources with PYTHONPATH=.."""
import copy
import os
import pickle
import types

import torch

try:
    import vllm  # noqa: F401
    from glyd_gpu import vllm_plugin as vp, vllm_quant as vq
except ImportError as e:
    if os.environ.get("GLYD_REQUIRE_VLLM") == "1":
        raise SystemExit(f"GLYD_REQUIRE_VLLM=1 and vLLM or the plugin does not import: {type(e).__name__}: {e}")
    vp = vq = None

ns = types.SimpleNamespace
# The quantization_configs of the checkpoints measured, their ignore lists cut (config.json, 2026-10-06)
QWEN_FP8 = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128],  # Qwen/Qwen3.6-35B-A3B-FP8, Qwen/Qwen3.8-27B-FP8
            "modules_to_not_convert": ["lm_head", "model.embed_tokens", "model.language_model.layers.0.mlp.gate", "model.visual.blocks.0.attn.qkv", "visual.blocks.0.attn.qkv"]}
GEMMA_NVFP4 = {"config_groups": {"group_0": {"input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},  # nvidia/Gemma-4-31B-IT-NVFP4
                                             "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}, "targets": ["Linear"]}},
               "ignore": ["lm_head", "model.embed_vision*", "model.language_model.layers.0.self_attn*", "model.vision_tower*"], "quant_algo": "NVFP4",
               "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"}, "producer": {"name": "modelopt", "version": "0.37.0"}, "quant_method": "modelopt"}
QWEN_NVFP4 = {"config_groups": {"group_0": {"input_activations": {"dynamic": False, "num_bits": 8, "type": "float"}, "weights": {"dynamic": False, "num_bits": 8, "type": "float"},  # nvidia/Qwen3.6-35B-A3B-NVFP4
                                            "targets": ["model.language_model.layers.0.linear_attn.in_proj_qkv"]}},
              "quantized_layers": {"model.language_model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"}, "model.language_model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                                   "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}},
              "ignore": ["mtp.layers.0*", "mtp*"], "quant_algo": "MIXED_PRECISION", "producer": {"name": "modelopt", "version": "0.37.0"}, "quant_method": "modelopt"}
_CT_ARGS = {"actorder": None, "block_structure": None, "observer_kwargs": {}, "scale_dtype": None, "zp_dtype": None}
CT_FP8 = {"config_groups": {"group_0": {"format": "float-quantized", "targets": ["Linear"], "output_activations": None,  # RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic
                                        "input_activations": {**_CT_ARGS, "dynamic": True, "group_size": None, "num_bits": 8, "observer": None, "strategy": "token", "symmetric": True, "type": "float"},
                                        "weights": {**_CT_ARGS, "dynamic": False, "group_size": None, "num_bits": 8, "observer": "memoryless_minmax", "strategy": "channel", "symmetric": True, "type": "float"}}},
          "format": "float-quantized", "global_compression_ratio": None, "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed", "sparsity_config": {},
          "transform_config": {}, "ignore": ["model.language_model.layers.0.router.proj", "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear", "lm_head"]}
CT_W4A16 = {**CT_FP8, "format": "pack-quantized", "ignore": ["model.embed_vision.patch_dense", "lm_head"],  # google/gemma-4-12B-it-qat-w4a16-ct
            "config_groups": {"group_0": {"format": "pack-quantized", "targets": ["Linear"], "input_activations": None, "output_activations": None,
                                          "weights": {**_CT_ARGS, "dynamic": False, "group_size": 32, "num_bits": 4, "observer": "memoryless_minmax", "strategy": "group", "symmetric": True, "type": "int"}}}}
CT_AWQ = {**CT_W4A16, "ignore": ["model.visual.blocks.0.attn.qkv", "lm_head"],  # cyankiwi/Qwen3.6-27B-AWQ-INT4 (fp16)
          "config_groups": {"group_0": {**CT_W4A16["config_groups"]["group_0"], "weights": {**CT_W4A16["config_groups"]["group_0"]["weights"], "observer": "mse", "symmetric": False, "zp_dtype": "torch.int8"}}}}
GPTQ = {"bits": 4, "group_size": 128, "damp_percent": 0.01, "desc_act": False, "static_groups": False, "sym": True, "true_sequential": True, "quant_method": "gptq",  # Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
        "dynamic": {"lm_head": {}, "model.language_model.embed_tokens": {}, "-:.*attn.*": {}, "-:.*shared_expert.*": {}, "-:.*mtp.*": {}, "-:.*visual.*": {}}, "modules_to_not_convert": []}
CT_W8A8 = {**CT_FP8, "format": "int-quantized", "ignore": ["model.visual.blocks.0.attn.qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv", "lm_head", "re:^mtp.*"],  # RedHatAI/Qwen3.5-9B-quantized.w8a8
           "config_groups": {"group_0": {"format": "int-quantized", "targets": ["Linear"], "output_activations": None,
                                         "input_activations": {**_CT_ARGS, "dynamic": True, "group_size": None, "num_bits": 8, "observer": None, "strategy": "token", "symmetric": True, "type": "int"},
                                         "weights": {**_CT_ARGS, "dynamic": False, "group_size": None, "num_bits": 8, "observer": "memoryless_minmax", "strategy": "channel", "symmetric": True, "type": "int"}}}}
CT_MXFP4 = {**CT_FP8, "format": "mxfp4-pack-quantized", "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*", "re:.*vision_tower.*", "re:.*mm_projector.*"],  # moonshotai/Kimi-K3
            "config_groups": {"group_0": {"format": "mxfp4-pack-quantized", "targets": ["Linear"], "input_activations": None, "output_activations": None,
                                          "weights": {**_CT_ARGS, "dynamic": False, "group_size": 32, "num_bits": 4, "observer": "minmax", "scale_dtype": "torch.uint8", "strategy": "group", "symmetric": True, "type": "float"}}}}
AWQ = {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True, "version": "gemm", "modules_to_not_convert": ["visual"]}  # (QuantTrio/Qwen3.6-35B-A3B-AWQ's kind)
FORMATS = {"fp8 block": QWEN_FP8, "modelopt nvfp4": GEMMA_NVFP4, "modelopt mixed": QWEN_NVFP4, "ct fp8-dynamic": CT_FP8, "ct w4a16": CT_W4A16, "ct awq": CT_AWQ, "gptq": GPTQ, "awq": AWQ, "ct w8a8": CT_W8A8, "ct mxfp4": CT_MXFP4}


def raises(f, text):
    try:
        f()
    except (ValueError, RuntimeError) as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"not refused: {text}")


def verify(user, cfg):
    """vLLM's own probe of the quantization methods (ModelConfig._verify_quantization, its real code): the method in effect."""
    from vllm.config.model import ModelConfig

    fake = ns(quantization=user, model_arch_config=ns(quantization_config=copy.deepcopy(cfg)), hf_config=None)
    ModelConfig._verify_quantization(fake)
    return fake.quantization


def lin(prefix, O=4096, K=4096):
    layer = object.__new__(vp.LinearBase)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.input_size, layer.output_size = prefix, K, O
    return layer


def table(cls, K=4096):
    t = object.__new__(cls)
    torch.nn.Module.__init__(t)
    t.embedding_dim, t.num_embeddings_padded, t.num_embeddings, t.tp_size = K, 248320, 248320, 1
    return t


def opened(c, exact=False, fraction=1.0, tied=False):
    """c's GlydConfig as resolve leaves it (no GPU here): the options in effect and the tables' plan."""
    g = c.glyd
    g.opts = {"layout": "mma", "exact": exact, "verify": False, "fraction": fraction, "kv": "off"}
    g.tied = tied
    g.plan = g.table_plan(ns(speculative_config=None, model_config=ns(head_dtype=torch.bfloat16)), fraction, exact, False)
    return c


def kind(c, layer, prefix):
    m = c.get_quant_method(layer, prefix)
    return None if m is None else type(m).__name__


def test_claims():
    """Every format: claimed under --quantization glyd (vLLM's probe keeps glyd), vLLM's own method without it; from_config gives vLLM's own class for
    the checkpoint (as its probe picks it) with Glyd in front, its name vLLM's, bf16 only, a GlydConfig of its own, pickled and deep-copied as vLLM does."""
    if vq is None:
        return print("test_claims: skipped (no vLLM)")
    from vllm.model_executor.layers.quantization import get_quantization_config

    vp.register()
    for name, cfg in FORMATS.items():
        own = verify(None, cfg)
        assert verify("glyd", cfg) == "glyd", name
        assert vp.GlydConfig.override_quantization_method(copy.deepcopy(cfg), "glyd") == "glyd" and vp.GlydConfig.override_quantization_method(copy.deepcopy(cfg), None) is None, name
        assert vp.GlydConfig.override_quantization_method(copy.deepcopy(cfg), "fp8") is None, name  # (only where the user asked for glyd)
        c = vp.GlydConfig.from_config(copy.deepcopy(cfg))
        inner = get_quantization_config(own)
        assert isinstance(c, inner) and isinstance(c, vq.GlydQuantConfig) and type(c).glyd_inner is inner and get_quantization_config(c.get_name()) is inner, (name, type(c), own)
        assert vq.inner_class(cfg) is inner and vq.inner_name(cfg) == own and type(c).__name__ == f"GlydQuant{inner.__name__}", name
        assert type(c.glyd) is vp.GlydConfig and c.glyd.quantized and c.glyd.given == {}, name
        assert c.get_supported_act_dtypes() == [torch.bfloat16] and c.get_min_capability() >= 80, name
        r = pickle.loads(pickle.dumps(c))  # (a round trip of our own object in memory, as vLLM does for its config)
        assert type(r) is type(c) and r.glyd.quantized and r.get_name() == c.get_name() and vars(r).keys() == vars(c).keys(), name
        d = copy.deepcopy(c)  # (vLLM's GPTQ methods deep-copy the config for every Linear)
        assert type(d) is type(c) and d.glyd.quantized, name
    assert type(vp.GlydConfig.from_config(QWEN_FP8)) is type(vp.GlydConfig.from_config(QWEN_FP8))  # (one class made for a vLLM class)
    q = vp.GlydConfig.from_config(QWEN_FP8)
    assert q.weight_block_size == [128, 128] and q.ignored_layers == QWEN_FP8["modules_to_not_convert"]  # (vLLM's own state: VllmConfig reads it)
    # a bf16 checkpoint's options and a glyd save are as before; a glyd save is not claimed
    assert type(vp.GlydConfig.from_config({"quant_method": "glyd", "layout": "mma"})) is vp.GlydConfig and type(vp.GlydConfig.from_config({"fraction": 0.5})) is vp.GlydConfig
    for cfg in (None, {}, {"quant_method": "glyd", "layout": "mma12"}, {"layout": "mma"}):
        assert vp.GlydConfig.override_quantization_method(cfg, "glyd") is None
    assert verify("glyd", None) == "glyd" and verify("glyd", {"quant_method": "glyd", "layout": "mma"}) == "glyd"


def test_refused():
    """A quantization vLLM serves with a method Glyd does not cover, and one vLLM has no method for: refused, with why, where --quantization glyd asks."""
    if vq is None:
        return print("test_refused: skipped (no vLLM)")
    mx = {"weight": {"dtype": "fp4", "is_dynamic": False, "qscheme": "per_group", "ch_axis": -1, "group_size": 32, "scale_format": "e8m0"}}
    quark = {"quant_method": "quark", "global_quant_config": {**mx, "input_tensors": {**mx["weight"], "is_dynamic": True}, "output_tensors": None, "bias": None},  # amd/MiniMax-M3-MXFP4
             "exclude": ["language_model.lm_head", "language_model.model.layers.0.self_attn.q_proj"]}
    assert vq.inner_name(quark) == verify(None, quark) == "quark"
    raises(lambda: vp.GlydConfig.override_quantization_method(copy.deepcopy(quark), "glyd"), "Glyd packs the 16-bit parts of")
    raises(lambda: verify("glyd", quark), "Glyd packs the 16-bit parts of")
    bnb = {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}  # (vLLM 0.30 has no bitsandbytes method)
    for cfg in (bnb, {"quant_method": "nonsense"}):
        raises(lambda: vp.GlydConfig.override_quantization_method(cfg, "glyd"), "not one vLLM serves")
        raises(lambda: verify("glyd", cfg), "not one vLLM serves")
        assert vp.GlydConfig.override_quantization_method(cfg, None) is None  # (not asked for: nothing refused)


def test_layers():
    """Which layers get Glyd's methods, on the real configs: the Linears a checkpoint leaves 16-bit (its ignore lists, GPTQ's negative dynamic rules,
    ModelOpt's excluded attention), its embedding and LM head (ModelOpt's excluded head gets vLLM's UnquantizedLinearMethod, taken too); a router
    (fewer than 1 M weights) keeps vLLM's; a quantized Linear keeps the checkpoint's method, untouched."""
    if vq is None:
        return print("test_layers: skipped (no vLLM)")
    import vllm.model_executor.layers.quantization.fp8 as fp8_module

    E, H = vp.VocabParallelEmbedding, vp.ParallelLMHead
    real = fp8_module.get_current_vllm_config
    try:  # (vLLM's FP8 method reads the model's dtype from the config being made)
        fp8_module.get_current_vllm_config = lambda: ns(model_config=ns(dtype=torch.bfloat16))
        c = opened(vp.GlydConfig.from_config(copy.deepcopy(QWEN_FP8)))
        assert kind(c, lin("visual.blocks.0.attn.qkv", 3456, 1152), "visual.blocks.0.attn.qkv") == "GlydLinearMethod"
        assert kind(c, lin("model.language_model.layers.0.mlp.gate", 256, 2048), "model.language_model.layers.0.mlp.gate") == "UnquantizedLinearMethod"  # (a router)
        assert kind(c, lin("model.language_model.layers.0.self_attn.o_proj"), "model.language_model.layers.0.self_attn.o_proj") == "Fp8LinearMethod"  # (quantized: vLLM's own)
        assert kind(c, table(E), "model.language_model.embed_tokens") == "GlydEmbeddingMethod" and kind(c, table(H), "lm_head") == "GlydEmbeddingMethod"
    finally:
        fp8_module.get_current_vllm_config = real
    c = opened(vp.GlydConfig.from_config(copy.deepcopy(GEMMA_NVFP4)), tied=True)
    assert kind(c, lin("model.language_model.layers.0.self_attn.qkv_proj", 16384, 5376), "model.language_model.layers.0.self_attn.qkv_proj") == "GlydLinearMethod"  # (bf16 attention)
    assert kind(c, table(E, 5376), "model.language_model.embed_tokens") == "GlydEmbeddingMethod" and kind(c, table(H, 5376), "lm_head") == "GlydEmbeddingMethod"
    for cfg in (CT_FP8, CT_W4A16):
        c = opened(vp.GlydConfig.from_config(copy.deepcopy(cfg)))
        assert kind(c, table(E, 2816), "model.language_model.embed_tokens") == "GlydEmbeddingMethod" and kind(c, table(H, 2816), "lm_head") == "GlydEmbeddingMethod"
    c = opened(vp.GlydConfig.from_config(copy.deepcopy(CT_FP8)))
    assert kind(c, lin("model.language_model.layers.0.router.proj", 128, 2816), "model.language_model.layers.0.router.proj") == "UnquantizedLinearMethod"  # (ignored, and a router)
    assert kind(c, lin("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear", 1152, 1152), "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear") == "GlydLinearMethod"
    c = opened(vp.GlydConfig.from_config(copy.deepcopy(CT_W8A8)))
    for p in ("model.language_model.layers.0.linear_attn.in_proj_qkv", "model.visual.blocks.0.attn.qkv"):
        assert kind(c, lin(p), p) == "GlydLinearMethod", p
    assert kind(c, table(H, 4096), "lm_head") == "GlydEmbeddingMethod"
    c = opened(vp.GlydConfig.from_config(copy.deepcopy(CT_MXFP4)))
    for p in ("language_model.model.layers.0.self_attn.q_proj", "language_model.model.layers.1.mlp.shared_experts.gate_up_proj", "vision_tower.encoder.blocks.0.mlp.fc1"):
        assert kind(c, lin(p), p) == "GlydLinearMethod", p
    assert kind(c, table(H, 7168), "language_model.lm_head") == "GlydEmbeddingMethod"
    c = opened(vp.GlydConfig.from_config(copy.deepcopy(GPTQ)))
    for p in ("model.language_model.layers.0.self_attn.qkv_proj", "model.language_model.layers.0.linear_attn.in_proj_qkvz", "model.language_model.layers.0.mlp.shared_expert.gate_up_proj", "visual.blocks.0.attn.qkv"):
        assert kind(c, lin(p), p) == "GlydLinearMethod", p
    assert kind(c, table(H, 2048), "lm_head") == "GlydEmbeddingMethod" and kind(c, table(E, 2048), "model.language_model.embed_tokens") == "GlydEmbeddingMethod"


class Stub(vp.QuantizationConfig if vp else object):
    """A checkpoint's config as far as the composite meets it: a quantized method for the layers named "quantized", vLLM's own 16-bit methods elsewhere
    (an LM head the way ModelOpt leaves it: UnquantizedLinearMethod)."""

    QUANTIZED = object()

    def __init__(self):
        super().__init__()

    def get_name(self):
        return "stub"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 75

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        if "quantized" in prefix:
            return Stub.QUANTIZED
        if isinstance(layer, vp.ParallelLMHead):
            return vp.UnquantizedLinearMethod()
        return vp.UnquantizedLinearMethod() if isinstance(layer, vp.LinearBase) else None


def stub():
    c = vq._restore(Stub, vars(Stub()))
    c.glyd = vp.GlydConfig()
    c.glyd.quantized = True
    return c


def test_options():
    """The options as for a bf16 model: a fraction's layers (the others vLLM's own 16-bit method), exact (the LM head, and a tied table, as vLLM runs them;
    an untied embedding packed), what the checkpoint quantizes untouched whatever the options; only bf16 (an fp16 checkpoint: vLLM refuses it)."""
    if vq is None:
        return print("test_options: skipped (no vLLM)")
    E, H = vp.VocabParallelEmbedding, vp.ParallelLMHead
    c = opened(stub())
    assert c.get_quant_method(lin("model.layers.0.mlp.quantized"), "model.layers.0.mlp.quantized") is Stub.QUANTIZED
    assert kind(c, lin("model.layers.0.self_attn.o_proj"), "model.layers.0.self_attn.o_proj") == "GlydLinearMethod"
    assert kind(c, table(E), "model.embed_tokens") == "GlydEmbeddingMethod" and kind(c, table(H), "lm_head") == "GlydEmbeddingMethod"
    assert c.get_quant_method(object(), "model.layers.0.self_attn.attn") is None  # (not a Linear or a table: the checkpoint's)
    opened(c, fraction=0.5)
    assert [kind(c, lin(f"model.layers.{i}.self_attn.o_proj"), f"model.layers.{i}.self_attn.o_proj") for i in range(4)] == ["UnquantizedLinearMethod", "GlydLinearMethod"] * 2
    assert kind(c, table(H), "lm_head") == "UnquantizedLinearMethod" and kind(c, table(E), "model.embed_tokens") is None  # (tables: at fraction 1 only)
    assert c.get_quant_method(lin("model.layers.0.mlp.quantized"), "model.layers.0.mlp.quantized") is Stub.QUANTIZED
    opened(c, exact=True)
    assert kind(c, table(H), "lm_head") == "UnquantizedLinearMethod" and kind(c, table(E), "model.embed_tokens") == "GlydEmbeddingMethod"
    opened(c, exact=True, tied=True)
    assert kind(c, table(H), "lm_head") == "UnquantizedLinearMethod" and kind(c, table(E), "model.embed_tokens") is None
    assert c.get_supported_act_dtypes() == [torch.bfloat16] and c.get_min_capability() == 80 and c.get_name() == "stub"
    assert vq.sixteen_bit(table(H), vp.UnquantizedLinearMethod()) and vq.sixteen_bit(table(E), None) and not vq.sixteen_bit(lin("x"), Stub.QUANTIZED) and not vq.sixteen_bit(object(), None)


def test_update_config():
    """In the engine's process: kv off written into additional_config's "glyd" for the workers, the other options kept; kv lossless refused; what Glyd refuses
    for a bf16 model refused here too (LoRA, exact compiled without inductor's deterministic mode); a draft model's config untouched."""
    if vq is None:
        return print("test_update_config: skipped (no vLLM)")

    def vc(extra=None, hf="H", eager=True, lora=None):
        cc = ns(inductor_compile_config={}, mode=vp.CompilationMode.NONE if eager else vp.CompilationMode.VLLM_COMPILE, cudagraph_mode=vp.CUDAGraphMode.NONE if eager else vp.CUDAGraphMode.PIECEWISE)
        return ns(model_config=ns(hf_config=hf, enforce_eager=eager, enable_sleep_mode=False), parallel_config=ns(use_ubatching=False), lora_config=lora, compilation_config=cc,
                  offload_config=ns(uva=ns(cpu_offload_gb=0), prefetch=ns(offload_group_size=0)), additional_config={} if extra is None else {"glyd": extra})

    real, env = vp._building_config, os.environ.pop("GLYD_KV", None)
    try:
        built = {}
        vp._building_config = lambda: built["vc"]
        c = stub()
        built["vc"] = v = vc()
        c.maybe_update_config("m", hf_config="H")
        assert v.additional_config == {"glyd": {"kv": "off"}} and c.glyd.hf_config == "H"
        built["vc"] = v = vc({"layout": "mma12", "exact": True})
        c.maybe_update_config("m", hf_config="H")
        assert v.additional_config["glyd"] == {"layout": "mma12", "exact": True, "kv": "off"}
        built["vc"] = vc({"kv": "lossless"})
        raises(lambda: c.maybe_update_config("m", hf_config="H"), "kv lossless is not held for a quantized checkpoint")
        built["vc"] = vc({"kv": "auto"})
        c.maybe_update_config("m", hf_config="H")
        assert built["vc"].additional_config["glyd"]["kv"] == "off"
        built["vc"] = vc(lora=object())
        raises(lambda: c.maybe_update_config("m", hf_config="H"), "LoRA")
        built["vc"] = vc({"exact": True}, eager=False)
        raises(lambda: c.maybe_update_config("m", hf_config="H"), "exact mode")
        built["vc"] = vc({"layot": "mma"})
        raises(lambda: c.maybe_update_config("m", hf_config="H"), "not an option")
        built["vc"] = v = vc(hf="OTHER")  # (a draft model's config: not this model's)
        c.maybe_update_config("m", hf_config="H")
        assert v.additional_config == {}
    finally:
        vp._building_config = real
        if env is not None:
            os.environ["GLYD_KV"] = env


def test_nvfp4_flag():
    """ModelConfig.is_nvfp4_quantized (vLLM turns its activation-quantization fusion on by it) answers for the method vLLM would pick: an NVFP4 checkpoint
    under --quantization glyd is one, an FP8 or GPTQ one and a bf16 model are not; the method in effect stays glyd."""
    if vq is None:
        return print("test_nvfp4_flag: skipped (no vLLM)")
    from vllm.config.model import ModelConfig

    vp.register()
    vp.register()  # (again: patched once)
    assert getattr(ModelConfig.is_nvfp4_quantized, "glyd", False)
    for cfg, want in ((GEMMA_NVFP4, True), (QWEN_FP8, False), (GPTQ, False), (None, False), ({"quant_method": "glyd", "layout": "mma"}, False)):
        fake = ns(quantization="glyd", model_arch_config=ns(quantization_config=copy.deepcopy(cfg)), hf_config=None)
        assert ModelConfig.is_nvfp4_quantized(fake) is want and fake.quantization == "glyd", cfg
    assert ModelConfig.is_nvfp4_quantized(ns(quantization="modelopt_fp4", model_arch_config=ns(quantization_config=GEMMA_NVFP4), hf_config=None))  # (stock's own answer)


def test_after_load():
    """What vLLM builds without the quantization config (Qwen3.5's embedding: vLLM's own method, no config) is packed once a quantized checkpoint's
    model has loaded, as a bf16 model's (vllm_plugin._after_load finds the GlydConfig behind the checkpoint's own config); nothing where the model
    was made without Glyd."""
    if vq is None:
        return print("test_after_load: skipped (no vLLM)")
    nn = torch.nn

    def model(qc):
        emb = table(vp.VocabParallelEmbedding, 64)
        emb.num_embeddings = emb.num_embeddings_padded = 8192
        emb.quant_method = vp.UnquantizedEmbeddingMethod()
        emb.weight = nn.Parameter(torch.zeros(8192, 64, dtype=torch.bfloat16), requires_grad=False)
        o = lin("model.layers.0.self_attn.o_proj")
        o.quant_config = qc
        m = nn.Module()
        m.embed_tokens, m.o_proj = emb, o
        return m

    calls, saved = [], (vp._bf16_matrix, vp.GlydEmbeddingMethod.pack_loaded)
    try:
        vp._bf16_matrix = lambda w, cols=16: isinstance(w, torch.Tensor) and w.dim() == 2 and w.dtype == torch.bfloat16 and w.shape[1] % cols == 0  # (on the CPU)
        vp.GlydEmbeddingMethod.pack_loaded = lambda self, layer, name: calls.append((name, self.config))
        c = opened(stub())
        vp._after_load(model(c))
        assert calls == [("embed_tokens", c.glyd)], calls
        calls.clear()
        vp._after_load(model(Stub()))  # (the checkpoint's own config alone: not a glyd model)
        assert calls == []
    finally:
        vp._bf16_matrix, vp.GlydEmbeddingMethod.pack_loaded = saved


def test_fp8_dev_path():
    """GLYD_FP8=1 (a development build): an FP8 checkpoint takes vllm_fp8.py's path as before; any other quantized checkpoint this one."""
    if vq is None:
        return print("test_fp8_dev_path: skipped (no vLLM)")
    from glyd_gpu import vllm_fp8

    old, gpu = os.environ.get("GLYD_FP8"), vllm_fp8.check_gpu
    try:
        vllm_fp8.check_gpu = lambda: None  # (the route is checked here, not the FP8 path's GPU rule: this runs on any GPU's pod, or none)
        os.environ["GLYD_FP8"] = "1"
        assert isinstance(vp.GlydConfig.from_config(copy.deepcopy(QWEN_FP8)), vllm_fp8.GlydFp8Config)
        assert isinstance(vp.GlydConfig.from_config(copy.deepcopy(GPTQ)), vq.GlydQuantConfig)
        os.environ.pop("GLYD_FP8")
        c = vp.GlydConfig.from_config(copy.deepcopy(QWEN_FP8))
        assert isinstance(c, vq.GlydQuantConfig) and not isinstance(c, vllm_fp8.GlydFp8Config)
    finally:
        vllm_fp8.check_gpu = gpu
        os.environ.pop("GLYD_FP8", None)
        if old is not None:
            os.environ["GLYD_FP8"] = old


DS_FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}  # (deepseek-ai/DeepSeek-V4-Flash's kind)


def model_config(model, model_type, cfg):
    """A ModelConfig as far as vLLM's weight_utils.get_quant_config and VllmConfig.get_quantization_config read it, under --quantization glyd."""
    return ns(quantization="glyd", quantization_config=None, hf_config=ns(model_type=model_type, quantization_config=copy.deepcopy(cfg)), hf_overrides={}, model=model,
              revision=None, dtype=torch.bfloat16)


def test_method_of_the_model():
    """The method vLLM's own config is made for is the one stock vLLM gives the model being configured, read where vLLM makes the config (its
    get_quant_config, run here on its real code): a DeepSeek-V4 FP8 checkpoint gets deepseek_v4_fp8 (picked by its model_type), not fp8, with no
    VllmConfig being built; a speculative draft in the target's checkpoint (MTP: get_draft_quant_config, as in a worker) the target's, as vLLM
    aligns a draft's quantization with its target's, though its own model_type is not deepseek_v4; a draft of another checkpoint its own."""
    if vq is None:
        return print("test_method_of_the_model: skipped (no vLLM)")
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.models.utils import get_draft_quant_config

    vp.register()
    ds, fp8, lc = get_quantization_config("deepseek_v4_fp8"), get_quantization_config("fp8"), ns(download_dir=None)
    assert ds is not fp8 and vq.inner_class(DS_FP8) is fp8  # (without the model's hf_config: fp8)
    target = model_config("deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4", DS_FP8)
    t = weight_utils.get_quant_config(target, lc)
    assert type(t).glyd_inner is ds and t.glyd_method == "deepseek_v4_fp8" and target.quantization == "glyd", type(t)
    assert pickle.loads(pickle.dumps(t)).glyd_method == "deepseek_v4_fp8"  # (the workers' copy says it too)
    draft = model_config("deepseek-ai/DeepSeek-V4-Flash", "deepseek_mtp", DS_FP8)
    vc = ns(model_config=target, quant_config=pickle.loads(pickle.dumps(t)), load_config=lc, speculative_config=ns(draft_model_config=draft))
    d = get_draft_quant_config(vc)
    assert type(d).glyd_inner is ds and d.glyd.quantized and d is not vc.quant_config, type(d)
    vc.speculative_config.draft_model_config = model_config("someone/another-draft", "deepseek_mtp", DS_FP8)
    assert type(get_draft_quant_config(vc)).glyd_inner is fp8  # (another checkpoint: its own probe)


def test_modelopt_mixed_file():
    """A ModelOpt mixed-precision checkpoint whose config.json holds no quantized_layers: read from its hf_quant_config.json, as stock vLLM reads it
    for modelopt_mixed (vLLM's own get_quant_config, on its real code, reaches it under --quantization glyd too); refused, with why, where no model
    is being configured."""
    if vq is None:
        return print("test_modelopt_mixed_file: skipped (no vLLM)")
    import json
    import tempfile

    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.model_loader import weight_utils

    vp.register()
    bare = {k: v for k, v in QWEN_NVFP4.items() if k != "quantized_layers"}
    assert vq.inner_name(bare) == "modelopt_mixed" and verify("glyd", bare) == "glyd"
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "hf_quant_config.json"), "w") as f:
            json.dump({"producer": QWEN_NVFP4["producer"], "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None, "exclude_modules": QWEN_NVFP4["ignore"],
                                                                           "quantized_layers": QWEN_NVFP4["quantized_layers"]}}, f)
        mc = model_config(d, "qwen3_5_moe", bare)
        c = weight_utils.get_quant_config(mc, ns(download_dir=None))
    assert type(c).glyd_inner is get_quantization_config("modelopt_mixed") and c.quantized_layers == QWEN_NVFP4["quantized_layers"] and mc.quantization == "glyd", type(c)
    raises(lambda: vp.GlydConfig.from_config(copy.deepcopy(bare)), "without quantized_layers")


if __name__ == "__main__":
    tests = [t for name, t in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(test.__name__, "ok")
    if vq is None:
        print(f"NOTE: vLLM is not installed, so these {len(tests)} tests did not run (GLYD_REQUIRE_VLLM=1 makes that a failure)")
