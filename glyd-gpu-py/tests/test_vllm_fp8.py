"""glyd_gpu's vLLM plugin for FP8 checkpoints (vllm_fp8.py, f8codec.py), its logic that needs no GPU: the FP8 schemes told from a
checkpoint's quantization_config (fp8 with and without blocks, mxfp8, ModelOpt, compressed-tensors) and what is refused with
why (a GPU that is not Ada or Hopper, FP8 checkpoints that are not covered, the bf16 path's options); --quantization glyd's
claim of such a checkpoint and vLLM's own config class for it with Glyd's in front (a subclass, isinstance of vLLM's own,
pickled as vLLM pickles its config); the identity codec's interface and the codec registry; the layouts of a weight vLLM's
kernel is fed (e4m3 [N, K], its transposed view, Marlin's int32 tiles) packed and decoded back into the one scratch every
packed Linear shares, grown as the layers come; the op glyd::f8_linear (decode, then the inner method's apply) on the CPU, and
traced by torch.compile. Needs vLLM (glyd[vllm]; no GPU), else skipped with a line saying so (GLYD_REQUIRE_VLLM=1 makes it a
failure, as test_vllm.py's).

    python test_vllm_fp8.py              (or pytest test_vllm_fp8.py)

The glyd_gpu it runs: the installed one, or this checkout's sources with PYTHONPATH=.. The FP8 path is a development build's (GLYD_FP8=1, set here; a release
build reads no such variable and refuses an FP8 checkpoint as 0.29 did): run these on the sources. test_off checks the default."""
import os
import pickle
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from glyd_gpu import f8codec  # noqa: E402 (PYTHONPATH=.. runs this checkout's package; else the one installed)

try:
    import vllm  # noqa: F401
    from glyd_gpu import vllm_fp8 as f8, vllm_plugin as vp
except ImportError as e:
    if os.environ.get("GLYD_REQUIRE_VLLM") == "1":
        raise SystemExit(f"GLYD_REQUIRE_VLLM=1 and vLLM or the plugin does not import: {type(e).__name__}: {e}")
    f8 = vp = None

QWEN = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128], "modules_to_not_convert": ["model.visual.blocks.0.attn.proj", "mtp.fc"]}  # (Qwen3.8-27B-FP8's, the lists cut)
MISTRAL = {"quant_method": "fp8", "activation_scheme": "static"}  # (what vLLM makes of a per-tensor Mistral checkpoint: transformers_utils/configs/mistral.py)
MINIMAX = {"quant_method": "mxfp8", "ignored_layers": ["lm_head", "model.layers.0.self_attn.q_proj"], "kv_cache_quant_algo": None}
MODELOPT = {"quant_method": "modelopt", "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": None, "exclude_modules": ["lm_head"]}}
os.environ["GLYD_FP8"] = "1"  # (the FP8 path's opt-in, a development build's: test_off unsets it)
os.environ["GLYD_F8_CODEC"] = "identity"  # (the CPU's codec, a development build's choice: the lossless one's kernels need a GPU; test_lossless runs its glue on fp8ref)
CT_W = {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True, "dynamic": False}
CT_A = {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True, "dynamic": True}
CT = {"quant_method": "compressed-tensors", "format": "float-quantized", "ignore": ["lm_head"], "quantization_status": "compressed", "kv_cache_scheme": None,
      "config_groups": {"group_0": {"targets": ["Linear"], "weights": CT_W, "input_activations": CT_A, "output_activations": None}}}


def compiled():
    """Whether glyd_gpu under test is the wheel's (its modules compiled: a release build, which reads no tuning variable) and not the sources'."""
    import glyd_gpu

    return not glyd_gpu.__file__.endswith(".py")


def raises(f, text):
    try:
        f()
    except (ValueError, RuntimeError) as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"not refused: {text}")


def test_codec():
    """The identity codec: the bytes kept (no copy), decoded exactly into a buffer made beforehand, whatever the byte dtype it is given;
    the registry (a codec by name, the default, the environment's choice, one added: the whole swap)."""
    w = torch.randint(0, 256, (37, 64), dtype=torch.uint8)
    p = f8codec.pack(w, "identity")
    assert p.codec == "identity" and p.shape == (37, 64) and p.nbytes() == 37 * 64 and p.tensors[0].data_ptr() == w.data_ptr()  # (no copy: its own memory)
    out = torch.zeros(37, 64, dtype=torch.uint8)
    f8codec.decode(p, out)
    assert torch.equal(out, w)
    f8 = w.view(torch.float8_e4m3fn)  # (e4m3 as the checkpoint holds it: NaN's 0x7F, 0xFF and both zeros among the bytes)
    p = f8codec.pack(f8, "identity")
    o8 = torch.zeros(37, 64, dtype=torch.float8_e4m3fn)
    f8codec.decode(p, o8)
    assert torch.equal(o8.view(torch.uint8), w) and p.shape == (37, 64)
    allb = torch.arange(256, dtype=torch.uint8).view(16, 16)  # (every byte pattern)
    out = torch.zeros(16, 16, dtype=torch.uint8)
    f8codec.decode(f8codec.pack(allb, "identity"), out)
    assert torch.equal(out, allb)
    raises(lambda: f8codec.pack(w.t()), "contiguous [rows, cols]")  # (a transposed view is not a byte matrix: the caller says which bytes)
    raises(lambda: f8codec.pack(w.view(torch.int32)), "one byte a value")
    raises(lambda: f8codec.pack(w.view(-1)), "contiguous [rows, cols]")
    assert f8codec.DEFAULT == "f8" and f8codec.get("f8") is f8codec.Lossless and f8codec.get("identity") is f8codec.Identity
    assert f8codec.get().name == ("identity" if not compiled() else "f8")  # (this file's GLYD_F8_CODEC: a release build reads none)
    raises(lambda: f8codec.get("nope"), "FP8 codec 'nope': one of f8, identity")
    old = os.environ.get("GLYD_F8_CODEC")
    try:
        if not compiled():  # (a release build reads no variable here: the codec is the additional_config option's, or DEFAULT's)
            os.environ["GLYD_F8_CODEC"] = "twice"
        else:
            os.environ["GLYD_F8_CODEC"] = "ignored"
            assert f8codec.get().name == f8codec.DEFAULT

        class Twice:  # (a codec added: registered, selected by name, by the environment (a build from the sources) or by DEFAULT: nothing else changes)
            name, version = "twice", 1

            @staticmethod
            def pack(w):
                b = w.view(torch.uint8)
                return f8codec.Packed("twice", b.shape, (b.clone(), b.clone()))

            @staticmethod
            def decode(p, out):
                out.view(torch.uint8).copy_(p.tensors[1])

        if not compiled():
            raises(lambda: f8codec.get(), "FP8 codec 'twice'")
        f8codec.register(Twice)
        q = f8codec.pack(w, "twice")
        assert q.codec == "twice" and q.nbytes() == 2 * 37 * 64
        out = torch.zeros(37, 64, dtype=torch.uint8)
        f8codec.decode(q, out)
        assert torch.equal(out, w)
        assert f8codec.pack(w, "identity").codec == "identity"
    finally:
        f8codec._CODECS.pop("twice", None)
        if old is None:
            os.environ.pop("GLYD_F8_CODEC", None)
        else:
            os.environ["GLYD_F8_CODEC"] = old


def test_experts():
    """The stacked experts' pack and decode_experts: the experts the ids name, in the ids' order, into out [len(ids), O, K]; an id outside [0, E) leaves
    its matrix of out as it was; the ids are not read on the host (the call is traced by torch.compile with them a tensor)."""
    E, O, K = 6, 64, 32
    stack = torch.randint(0, 256, (E * O, K), dtype=torch.uint8)
    p = f8codec.pack(stack, "identity")
    ids = torch.tensor([4, 1, 9, 5, -1, 0], dtype=torch.int32)
    out = torch.full((6, O, K), 7, dtype=torch.uint8)
    f8codec.decode_experts(p, E, ids, out)
    full = stack.view(E, O, K)
    for j, e in enumerate(ids.tolist()):
        assert torch.equal(out[j], full[e] if 0 <= e < E else torch.full((O, K), 7, dtype=torch.uint8)), (j, e)
    o8 = torch.zeros(2, O, K, dtype=torch.float8_e4m3fn)
    f8codec.decode_experts(p, E, torch.tensor([2, 3], dtype=torch.int32), o8)
    assert torch.equal(o8.view(torch.uint8), full[2:4])
    try:
        fn = torch.compile(lambda i, o: f8codec.decode_experts(p, E, i, o), backend="aot_eager", fullgraph=True)
        out2 = torch.zeros(3, O, K, dtype=torch.uint8)
        fn(torch.tensor([5, 0, 3], dtype=torch.int32), out2)
        assert torch.equal(out2, full[[5, 0, 3]])
    except RuntimeError as e:
        if "not supported" not in str(e) and "Python 3.1" not in str(e):
            raise


def test_scheme():
    """The schemes told from a checkpoint's quantization_config, by quant_method and its fields; None for what is not an FP8
    checkpoint's (as before: vLLM's own message for a mismatch), a refusal with why for what is FP8 but not covered."""
    if f8 is None:
        return print("test_scheme: skipped (no vLLM)")
    assert f8.scheme(QWEN) == "fp8-block" and f8.scheme({**QWEN, "weight_block_size": [32, 32]}) == "fp8-block"  # (DeepSeek V4's [32, 32] too)
    assert f8.scheme(MISTRAL) == "fp8-tensor" and f8.scheme({"quant_method": "FP8", "activation_scheme": "dynamic"}) == "fp8-tensor"
    assert f8.scheme(MINIMAX) == "mxfp8"
    assert f8.scheme(MODELOPT) == "modelopt-fp8" and f8.scheme({"quant_method": "modelopt", "quant_algo": "FP8_PB_WO"}) == "modelopt-fp8"  # (hf_quant_config.json's flat form)
    assert f8.scheme({"quant_method": "modelopt", "quantization": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}}) == "modelopt-fp8"
    assert f8.scheme({"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP8"}}) == "modelopt-mxfp8"
    assert f8.scheme(CT) == "ct-fp8"
    ct = {**CT, "config_groups": {"group_0": {"targets": ["Linear"], "weights": {**CT_W, "strategy": "group", "group_size": 32, "scale_dtype": "uint8"}, "input_activations": None}}}
    assert f8.scheme(ct) == "ct-mxfp8"
    for none in (None, {}, {"quant_method": ""}, {"quant_method": "glyd", "layout": "mma"}, {"layout": "mma"}, {"quant_method": "gptq", "bits": 4}, {"quant_method": "awq"}, "fp8", []):
        assert f8.scheme(none) is None, none
    raises(lambda: f8.scheme({"quant_method": "modelopt", "quantization": {"quant_algo": "NVFP4"}}), "ModelOpt NVFP4 checkpoint")
    raises(lambda: f8.scheme({"quant_method": "modelopt", "quantization": {"quant_algo": "MIXED_PRECISION"}}), "MIXED_PRECISION")
    raises(lambda: f8.scheme({"quant_method": "fbgemm_fp8", "activation_scheme": "dynamic"}), "FBGEMM FP8 is not covered")
    raises(lambda: f8.scheme({"quant_method": "quark"}), "Quark checkpoints are not covered")
    w8a16 = {**CT, "config_groups": {"group_0": {"targets": ["Linear"], "weights": CT_W, "input_activations": None}}}
    raises(lambda: f8.scheme(w8a16), "quantizes the weights to FP8 but not the activations")
    int4 = {**CT, "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "int", "strategy": "group", "group_size": 128}, "input_activations": None}}}
    raises(lambda: f8.scheme(int4), "4-bit int weights")
    mixed = {**CT, "config_groups": {"g0": CT["config_groups"]["group_0"], "g1": ct["config_groups"]["group_0"]}}
    raises(lambda: f8.scheme(mixed), "mixing FP8 and MXFP8")
    raises(lambda: f8.scheme({**CT, "config_groups": {}}), "no config_groups")
    one_int = {**CT, "config_groups": {"g0": CT["config_groups"]["group_0"], "g1": int4["config_groups"]["group_0"]}}
    raises(lambda: f8.scheme(one_int), "Glyd packs the 8-bit float ones")  # (every group must be: one that is not refuses the checkpoint)


def test_gpu_refusal():
    """The GPU: Ada (sm_89), Hopper (sm_90) and the RTX 50 series (sm_120) only, refused with why elsewhere (Ampere serves FP8 weight-only: not
    covered; B200/B300: not measured); where vLLM's platform cannot tell (no GPU here), the config goes on."""
    if f8 is None:
        return print("test_gpu_refusal: skipped (no vLLM)")
    real = f8.current_platform
    try:
        for cap, ok in (((8, 9), True), ((9, 0), True), ((8, 0), False), ((8, 6), False), ((7, 5), False), ((10, 0), False), ((12, 0), True)):
            f8.current_platform = types.SimpleNamespace(get_device_capability=lambda cap=cap: types.SimpleNamespace(major=cap[0], minor=cap[1]))
            if ok:
                assert f8.claims(QWEN) is True
            else:
                raises(lambda: f8.claims(QWEN), f"this GPU is sm_{cap[0]}{cap[1]}")
                raises(lambda: vp.GlydConfig.override_quantization_method(QWEN, "glyd"), "Ada (sm_89), Hopper (sm_90) and RTX 50 series (sm_120)")
                raises(lambda: vp.GlydConfig.from_config(QWEN), "Ada (sm_89), Hopper (sm_90) and RTX 50 series (sm_120)")
                assert vp.GlydConfig.override_quantization_method(QWEN, None) is None and vp.GlydConfig.override_quantization_method({}, "glyd") is None  # (not asked for, or not an FP8 checkpoint: no refusal of the GPU)
        f8.current_platform = types.SimpleNamespace(get_device_capability=lambda: None)
        assert f8.claims(QWEN) is True
        assert f8.claims({"quant_method": "gptq"}) is False
    finally:
        f8.current_platform = real


def test_override_and_config():
    """--quantization glyd on an FP8 checkpoint: GlydConfig claims it (vLLM's probe finds its name first) and builds vLLM's own config class
    for it, with Glyd's put in front: vLLM's fields and isinstance checks hold, and a bf16 checkpoint, a glyd save and every other
    quantization are not claimed."""
    if f8 is None:
        return print("test_override_and_config: skipped (no vLLM)")
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptMxFp8Config

    claim = vp.GlydConfig.override_quantization_method
    for cfg in (QWEN, MISTRAL, MINIMAX, MODELOPT, CT):
        assert claim(cfg, "glyd") == "glyd" and claim(cfg, None) is None and claim(cfg, "fp8") is None and claim(cfg, "gptq") is None  # (only where the user asked for glyd)
    for cfg in (None, {}, {"quant_method": "glyd", "layout": "mma12"}):
        assert claim(cfg, "glyd") is None  # (a glyd save, bf16 with options: as before; other quantizations are vllm_quant.py's: test_vllm_quant.py)
    raises(lambda: claim({"quant_method": "modelopt", "quantization": {"quant_algo": "NVFP4"}}, "glyd"), "ModelOpt NVFP4")
    for cfg, inner, name, s in ((QWEN, Fp8Config, "fp8", "fp8-block"), (MISTRAL, Fp8Config, "fp8", "fp8-tensor"), (MINIMAX, ModelOptMxFp8Config, "modelopt_mxfp8", "mxfp8"),
                                (MODELOPT, ModelOptFp8Config, "modelopt", "modelopt-fp8"), (CT, CompressedTensorsConfig, "compressed-tensors", "ct-fp8")):
        c = vp.GlydConfig.from_config(cfg)
        assert isinstance(c, inner) and isinstance(c, f8.GlydFp8Config) and type(c).glyd_inner is inner and c.glyd_scheme == s, (cfg, type(c))
        assert c.get_name() == name and c.get_min_capability() == 89 and type(c).__name__ == f"Glyd{inner.__name__}"
        assert isinstance(c.packed_modules_mapping, dict) and c.get_config_filenames() == inner.get_config_filenames()
    q = vp.GlydConfig.from_config(QWEN)
    assert q.weight_block_size == [128, 128] and q.activation_scheme == "dynamic" and q.ignored_layers == QWEN["modules_to_not_convert"] and q.use_deep_gemm is None  # (vLLM's: VllmConfig reads these off it)
    assert vp.GlydConfig.from_config(MISTRAL).weight_block_size is None and vp.GlydConfig.from_config(MISTRAL).activation_scheme == "static"
    assert vp.GlydConfig.from_config(MINIMAX).exclude_modules == ["lm_head", "model.layers.0.self_attn.q_proj"]
    assert vp.GlydConfig.from_config(MODELOPT).exclude_modules == ["lm_head"] and vp.GlydConfig.from_config(CT).get_supported_act_dtypes() == inner_acts(CompressedTensorsConfig, CT)
    assert type(vp.GlydConfig.from_config(QWEN)) is type(q)  # (one class made for a vLLM class)
    # a model's own packed-module mapping: updated in place, as the models do
    q.packed_modules_mapping.update({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
    assert q.packed_modules_mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
    # pickled as vLLM pickles its config to start its engine's process: the class made here is not importable by name, the state is kept
    r = pickle.loads(pickle.dumps(q))  # (a round trip of our own object in memory, as vLLM does for its own config: no untrusted data)
    assert type(r) is type(q) and r.weight_block_size == [128, 128] and r.glyd_scheme == "fp8-block" and r.packed_modules_mapping == q.packed_modules_mapping and isinstance(r, Fp8Config)
    import copy

    assert type(copy.deepcopy(q)) is type(q)
    # the bf16 path is as it was: from_config of the options, nothing claimed
    b = vp.GlydConfig.from_config({"quant_method": "glyd", "layout": "mma"})
    assert type(b) is vp.GlydConfig and b.given == {"layout": "mma"}
    assert type(vp.GlydConfig.from_config({"fraction": 0.5})) is vp.GlydConfig
    raises(lambda: vp.GlydConfig.from_config({"layot": "mma"}), "not an option")


def inner_acts(cls, cfg):
    return cls.from_config(cfg).get_supported_act_dtypes()


def test_modelopt_file():
    """A ModelOpt checkpoint whose quantization is in hf_quant_config.json alone: vLLM reads it from there for modelopt only, so --quantization glyd
    (which reads config.json's quantization_config) would pack its FP8 weights as bf16's, without their scales: refused, with why, by the bf16
    config; a directory without the file, and one that is a glyd save, as before."""
    if f8 is None:
        return print("test_modelopt_file: skipped (no vLLM)")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        assert not f8.modelopt_file(d) and not f8.modelopt_file("no/such-repo-here")
        c = vp.GlydConfig()
        c.maybe_update_config(d, hf_config=None)  # (a bf16 directory: nothing to refuse; a config of its own)
        with open(os.path.join(d, "hf_quant_config.json"), "w") as f:
            f.write("{}")
        assert f8.modelopt_file(d)
        raises(lambda: c.maybe_update_config(d, hf_config=None), "hf_quant_config.json")


def test_refusals():
    """The runtime refusals (vLLM's config): LoRA, weight offloading, sleep mode, dual-batch overlap, and the options of the bf16 path
    given to an FP8 checkpoint; the codec chosen in additional_config's "glyd" (f8_codec) validated, and the options in effect (the codec
    and its version) written there, for vLLM's compile cache to key by."""
    if f8 is None:
        return print("test_refusals: skipped (no vLLM)")

    def vc(**kw):
        ns = dict(parallel_config=types.SimpleNamespace(use_ubatching=False), model_config=types.SimpleNamespace(enable_sleep_mode=False, hf_config="H"), lora_config=None,
                  compilation_config=types.SimpleNamespace(mode=f8.CompilationMode.VLLM_COMPILE, custom_ops=[]),
                  offload_config=types.SimpleNamespace(uva=types.SimpleNamespace(cpu_offload_gb=0), prefetch=types.SimpleNamespace(offload_group_size=0)), additional_config={})
        ns.update(kw)
        return types.SimpleNamespace(**ns)

    f8.refusals(vc())
    raises(lambda: f8.refusals(vc(parallel_config=types.SimpleNamespace(use_ubatching=True))), "dual-batch overlap")
    raises(lambda: f8.refusals(vc(lora_config=object())), "LoRA")
    raises(lambda: f8.refusals(vc(offload_config=types.SimpleNamespace(uva=types.SimpleNamespace(cpu_offload_gb=4), prefetch=types.SimpleNamespace(offload_group_size=0)))), "weight offloading")
    raises(lambda: f8.refusals(vc(model_config=types.SimpleNamespace(enable_sleep_mode=True, hf_config="H"))), "sleep mode")
    raises(lambda: f8.refusals(vc(additional_config=None)), "not a dict")
    c = vp.GlydConfig.from_config(QWEN)
    built = {}
    real = vp._building_config
    try:
        vp._building_config = lambda: built["vc"]
        built["vc"] = v = vc(additional_config={"glyd": {"layout": "mma"}})
        raises(lambda: c.maybe_update_config("m", hf_config="H"), "not an option of an FP8 checkpoint")
        built["vc"] = v = vc(additional_config={"glyd": {"f8_codec": "nope"}})
        raises(lambda: c.maybe_update_config("m", hf_config="H"), "FP8 codec 'nope'")
        built["vc"] = v = vc()
        c.maybe_update_config("m", hf_config="H")
        assert v.additional_config["glyd"]["f8codec"] == "identity-1" and v.additional_config["glyd"]["glyd"] and c.glyd_codec == "identity" and v.additional_config["glyd"]["exact"] is False
        c.maybe_update_config("m", hf_config="H")  # (again, as vLLM's config is made again: its own keys are not options)
        assert v.additional_config["glyd"]["f8codec"] == "identity-1"
        built["vc"] = v = vc(additional_config={"glyd": {"f8_codec": "identity"}})
        c.maybe_update_config("m", hf_config="H")
        assert v.additional_config["glyd"] == {"f8_codec": "identity", "glyd": v.additional_config["glyd"]["glyd"], "f8codec": "identity-1", "exact": False}
        # a mixture of experts' checkpoint: the shared experts' side stream is turned off (one scratch, one writer at a time)
        old_env = os.environ.pop("VLLM_DISABLE_SHARED_EXPERTS_STREAM", None)
        try:
            built["vc"] = vc()
            c.maybe_update_config("m", hf_config="H")
            assert "VLLM_DISABLE_SHARED_EXPERTS_STREAM" not in os.environ
            moe = types.SimpleNamespace(num_experts=8)
            built["vc"] = vc(model_config=types.SimpleNamespace(enable_sleep_mode=False, hf_config=moe))
            c.maybe_update_config("m", hf_config=moe)
            assert os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] == "1"
        finally:
            os.environ.pop("VLLM_DISABLE_SHARED_EXPERTS_STREAM", None)
            if old_env is not None:
                os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = old_env
        built["vc"] = v = vc(model_config=types.SimpleNamespace(enable_sleep_mode=True, hf_config="OTHER"))  # (a draft model's config: not this model's: not checked)
        c.maybe_update_config("m", hf_config="H")
        assert v.additional_config == {}
    finally:
        vp._building_config = real


def test_exact():
    """exact: on a per-tensor or per-channel checkpoint refused, with why, under inductor unless vLLM's quantization op is on (custom_ops +quant_fp8) and
    its norm and activation fusion passes are off: else the quantization is fused into vLLM's graph and runs apart inside Glyd's product. What is not
    measured against stock is refused under exact whatever the compile: an MXFP8 checkpoint (config time), a Linear whose kernel is a block-scaled W8A8
    one of vLLM's (load time, by the kernel's class; Marlin's weight-only kernel kept), ModelOpt FP8 experts (test_experts_flow). Which schemes quantize
    with torch ops follows the checkpoint's config."""
    if f8 is None:
        return print("test_exact: skipped (no vLLM)")
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFp8BlockScaledMMKernel
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel

    ct_block ={**CT, "config_groups": {"g": {"targets": ["Linear"], "weights": {**CT_W, "strategy": "block", "block_structure": [128, 128]}, "input_activations": {**CT_A, "strategy": "group", "group_size": 128}}}}
    for cfg, native in ((QWEN, False), (MISTRAL, True), (MINIMAX, False), (MODELOPT, True), (CT, True), (ct_block, False), ({"quant_method": "modelopt", "quant_algo": "FP8_PB_WO"}, False),
                        ({"quant_method": "modelopt", "quantization": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}}, True), ({"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP8", "kv_cache_quant_algo": None, "exclude_modules": []}}, False)):
        assert f8.native_quant(cfg) is native, cfg
        assert vp.GlydConfig.from_config(cfg).glyd_native_quant is native

    def vc(mode, ops, eager=False, ac=None, fused=None):
        return types.SimpleNamespace(parallel_config=types.SimpleNamespace(use_ubatching=False), lora_config=None, offload_config=None, additional_config=ac if ac is not None else {"glyd": {"exact": True}},
                                     model_config=types.SimpleNamespace(enable_sleep_mode=False, hf_config="H", enforce_eager=eager),
                                     compilation_config=types.SimpleNamespace(mode=mode, custom_ops=ops, pass_config=types.SimpleNamespace(fuse_norm_quant=fused, fuse_act_quant=fused)))

    pt, blk = vp.GlydConfig.from_config(MISTRAL), vp.GlydConfig.from_config(QWEN)
    built, real = {}, vp._building_config
    try:
        vp._building_config = lambda: built["vc"]
        C = f8.CompilationMode.VLLM_COMPILE
        for name, v, refused in (("compiled", vc(C, []), True), ("compiled, quant op on (its fusions on by default)", vc(C, ["+quant_fp8"]), True),
                                 ("compiled, quant op on, fusions on", vc(C, ["+quant_fp8"], fused=True), True), ("compiled, fusions off, quant op off", vc(C, [], fused=False), True),
                                 ("compiled, quant op on, fusions off", vc(C, ["+quant_fp8"], fused=False), False),
                                 ("enforce-eager", vc(C, [], eager=True), False), ("no compile (CUDA graphs alone)", vc(f8.CompilationMode.NONE, []), False)):
            built["vc"] = v
            if refused:
                raises(lambda: pt.maybe_update_config("m", hf_config="H"), "Serve with --enforce-eager, or with --compilation-config")
                raises(lambda: pt.maybe_update_config("m", hf_config="H"), "fuses it with the RMSNorm or the activation before it")
            else:
                pt.maybe_update_config("m", hf_config="H")
                assert v.additional_config["glyd"]["exact"] is True, name
            cc = v.compilation_config
            built["vc"] = v = vc(cc.mode, cc.custom_ops, eager=v.model_config.enforce_eager, ac={"glyd": {"exact": True}}, fused=cc.pass_config.fuse_norm_quant)
            blk.maybe_update_config("m", hf_config="H")  # (a block-scaled checkpoint: nothing refused at config time, whatever the compile; its Linears at load, by their kernel)
            raises(lambda: vp.GlydConfig.from_config(MINIMAX).maybe_update_config("m", hf_config="H"), "exact on an MXFP8 checkpoint (mxfp8) is not measured")
            for exact in (True, False):
                blk.glyd_exact = exact
                for layout, kernel in (("i32", MarlinFP8ScaledMMLinearKernel), ("nk", CutlassFp8BlockScaledMMKernel)):  # (Marlin's weight-only kernel: measured; a block-scaled W8A8 one: not)
                    m = f8.GlydFp8LinearMethod(fake_inner(layout, kernel), blk)
                    layer = fake_layer("model.layers.0.mlp.down_proj", 64, 128)
                    m.create_weights(layer, 128, [64], 128, 64, torch.bfloat16)
                    if exact and kernel is CutlassFp8BlockScaledMMKernel:
                        raises(lambda: m.process_weights_after_loading(layer), "exact with vLLM's CutlassFp8BlockScaledMMKernel (a block-scaled W8A8 product) is not measured")
                    else:
                        m.process_weights_after_loading(layer)
                        assert layer.glyd_f8_id is not None, (name, layout, exact)
            f8._SCRATCH.clear()
            f8._REG.clear()
            built["vc"] = vc(cc.mode, cc.custom_ops, ac={"glyd": {"exact": False}})
            pt.maybe_update_config("m", hf_config="H")  # (exact off: nothing refused)
            vp.GlydConfig.from_config(MINIMAX).maybe_update_config("m", hf_config="H")
        built["vc"] = vc(f8.CompilationMode.VLLM_COMPILE, [], ac={"glyd": {"exact": "ture"}})
        raises(lambda: pt.maybe_update_config("m", hf_config="H"), "true or false")
    finally:
        vp._building_config = real


def fake_inner(layout, kernel=None):
    """vLLM's FP8 Linear method, as far as the wrapper meets it: create_weights makes layer.weight (e4m3 [N, K]); process_weights_after_loading
    leaves it as [N, K] ("nk"), as its transposed view [K, N] ("kn": per-tensor), or as int32 tiles ("i32": Marlin); apply reads it back from the
    layer, as the kernel does, and takes the scale from the layer. kernel: the class of vLLM's kernel object it holds (fp8_linear), None for none."""
    class Fp8LinearMethod:
        def __init__(self):
            self.fp8_linear = None if kernel is None else object.__new__(kernel)

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra):
            layer.weight = torch.nn.Parameter(torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.float8_e4m3fn), requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)

        def process_weights_after_loading(self, layer):
            w = layer.weight.data
            if layout == "kn":
                layer.weight = torch.nn.Parameter(w.t(), requires_grad=False)
            elif layout == "i32":
                layer.weight = torch.nn.Parameter(w.contiguous().view(torch.uint8).view(torch.int32).view(w.shape[0], -1).clone(), requires_grad=False)

        def apply(self, layer, x, bias=None):
            w = layer.weight
            if layout == "kn":
                w = w.t()
            elif layout == "i32":
                w = w.contiguous().view(torch.uint8).view(torch.float8_e4m3fn).view(w.shape[0], -1)
            y = torch.nn.functional.linear(x.float(), w.float() * layer.weight_scale)
            return (y if bias is None else y + bias).to(x.dtype)

    return Fp8LinearMethod()


def fake_layer(prefix, O, K):
    layer = object.__new__(f8.LinearBase)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.input_size, layer.output_size = prefix, K, O
    return layer


def test_wrap():
    """get_quant_method's wrapping: vLLM's own method for an FP8 Linear wrapped; an unquantized Linear (an ignored layer), a layer that is
    not a Linear, a Linear of few weights, no method: as they are; a Linear whose method is not an FP8 one refused, naming it."""
    if f8 is None:
        return print("test_wrap: skipped (no vLLM)")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod, WEIGHT_LOADER_V2_SUPPORTED
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod

    c = vp.GlydConfig.from_config(QWEN)
    lin = fake_layer("model.layers.0.mlp.down_proj", 5120, 17408)
    for cls in (Fp8LinearMethod, ModelOptLinearMethod, CompressedTensorsLinearMethod):
        inner = object.__new__(cls)
        if cls is CompressedTensorsLinearMethod:
            lin.scheme = type("CompressedTensorsW8A8Fp8", (), {})()
        w = f8.wrap(c, inner, lin, lin.prefix)
        assert type(w) is f8.GlydFp8LinearMethod and w.inner is inner and w.codec == "identity"
    assert "GlydFp8LinearMethod" in WEIGHT_LOADER_V2_SUPPORTED  # (the Linear's loader is chosen by the method's class name)
    lin.scheme = type("CompressedTensorsW4A16Fp8", (), {})()
    raises(lambda: f8.wrap(c, object.__new__(CompressedTensorsLinearMethod), lin, "model.layers.0.mlp.down_proj"), "CompressedTensorsW4A16Fp8 is not an FP8 one")
    plain = UnquantizedLinearMethod()
    assert f8.wrap(c, plain, lin, "x") is plain and f8.wrap(c, None, lin, "x") is None  # (an ignored layer; no method)
    inner = object.__new__(Fp8LinearMethod)
    assert f8.wrap(c, inner, object(), "model.embed_tokens") is inner  # (an embedding, an LM head: not a LinearBase)
    gate = fake_layer("model.layers.3.mlp.gate", 128, 2048)  # (a router: 0.25 M weights)
    assert f8.wrap(c, inner, gate, gate.prefix) is inner
    bmm = fake_layer("model.layers.3.self_attn.wo_a", 4096, 4096)
    bmm.is_bmm = True
    assert f8.wrap(c, inner, bmm, bmm.prefix) is inner
    raises(lambda: f8.wrap(c, object.__new__(type("Fp8PerTensorOnlineLinearMethod", (), {})), lin, "model.layers.0.mlp.down_proj"), "Fp8PerTensorOnlineLinearMethod for this FP8 checkpoint")
    # the config's own get_quant_method: vLLM's, then wrapped (an Attention or a mixture of experts layer: vLLM's own, untouched here)
    import vllm.model_executor.layers.quantization.fp8 as fp8_module

    real = fp8_module.get_current_vllm_config
    try:  # (vLLM's method reads the model's dtype from the config being made)
        fp8_module.get_current_vllm_config = lambda: types.SimpleNamespace(model_config=types.SimpleNamespace(dtype=torch.bfloat16))
        got = c.get_quant_method(lin, "model.layers.0.mlp.down_proj")
        assert type(got) is f8.GlydFp8LinearMethod and type(got.inner).__name__ == "Fp8LinearMethod"
        assert got.block_quant is True and got.weight_block_size == [128, 128] and got.quant_config is c  # (what vLLM reads of the method: through the wrapper)
        assert type(c.get_quant_method(lin, "model.visual.blocks.0.attn.proj")).__name__ == "UnquantizedLinearMethod"  # (modules_to_not_convert)
        assert type(vp.GlydConfig.from_config(MISTRAL).get_quant_method(lin, "model.layers.0.mlp.down_proj").inner).__name__ == "Fp8LinearMethod"
        # an MLA model's kv_b_proj (DeepSeek-V3's FP8 config): vLLM reads its weight directly after the load (dequantized with the scales where its method is
        # vLLM's own Fp8LinearMethod): that Linear keeps vLLM's own method, 16 M weights as it is; its neighbours of the same size are wrapped
        mla = vp.GlydConfig.from_config({"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]})
        kvb = fake_layer("model.layers.3.self_attn.kv_b_proj", 32768, 512)
        assert type(mla.get_quant_method(kvb, kvb.prefix)).__name__ == "Fp8LinearMethod"
        o_proj = fake_layer("model.layers.3.self_attn.o_proj", 32768, 512)
        assert type(mla.get_quant_method(o_proj, o_proj.prefix)).__name__ == "GlydFp8LinearMethod"
        assert f8.DIRECT == ("kv_b_proj",)
    finally:
        fp8_module.get_current_vllm_config = real


def test_matrix():
    """A weight as vLLM's kernel is fed it, as the bytes of its storage and the view back: e4m3 [N, K], its transposed view [K, N], int32 tiles;
    refused for a part of a larger storage or a layout that is neither."""
    if f8 is None:
        return print("test_matrix: skipped (no vLLM)")
    b = torch.randint(0, 256, (24, 64), dtype=torch.uint8)
    for w in (b.view(torch.float8_e4m3fn), b.view(torch.float8_e4m3fn).t(), b.view(torch.int32), b.view(torch.int32).view(2, 12, 16)):
        m, (dt, shape, stride) = f8._matrix(w, "l")
        assert m.dtype == torch.uint8 and m.dim() == 2 and m.data_ptr() == w.data_ptr() and m.numel() == b.numel() and (dt, shape, stride) == (w.dtype, tuple(w.shape), tuple(w.stride()))
        back = m.reshape(-1).view(dt).as_strided(shape, stride)  # (the storage's bytes, viewed as the kernel was fed them)
        assert back.shape == w.shape and back.stride() == w.stride() and torch.equal(back.contiguous().view(torch.uint8), w.contiguous().view(torch.uint8))
        assert torch.equal(m.reshape(-1), b.reshape(-1))  # (the storage's bytes, in order)
    raises(lambda: f8._matrix(b.view(torch.float8_e4m3fn)[:, :32], "l"), "not a layout Glyd packs")
    raises(lambda: f8._matrix(b.view(torch.float8_e4m3fn)[2:], "l"), "part of a larger storage")
    raises(lambda: f8._matrix(b.view(torch.float8_e4m3fn).view(4, 6, 64).transpose(0, 1), "l"), "not a layout Glyd packs")


def test_scratch_and_flow():
    """A packed Linear's life as vLLM calls it: create_weights, the load, process_weights_after_loading, then products. The weight is packed and its
    memory freed (the pack keeps it: the identity codec) and becomes a view of one scratch, grown to the largest layer's size as the
    layers come (the first Linears' weights pointed at the new one); each product decodes the Linear's own bytes into it first (the scratch
    overwritten between products, as another layer's would), then vLLM's method's apply: the result is that method's on the loaded weight,
    bit for bit, for every layout; the layer vLLM calls process_weights_after_loading for again is left alone; the scratch held once a
    product ran."""
    if f8 is None:
        return print("test_scratch_and_flow: skipped (no vLLM)")
    f8.register()
    cfg = types.SimpleNamespace(glyd_codec=None)
    f8._SCRATCH.clear()
    torch.manual_seed(0)
    layers = []
    for layout, O, K in (("nk", 64, 96), ("kn", 128, 64), ("i32", 192, 128), ("nk", 256, 256), ("kn", 64, 32)):  # (the third needs more than the first two: the scratch grows; the fourth more again)
        m = f8.GlydFp8LinearMethod(fake_inner(layout), cfg)
        layer = fake_layer(f"model.layers.{len(layers)}.mlp.down_proj", O, K)
        m.create_weights(layer, K, [O // 2, O // 2], K, O, torch.bfloat16)
        assert layer.requires_unquantized_input is True and layer.glyd_f8_id is None  # (as vLLM runs it, until packed)
        raw = torch.randint(0, 256, (O, K), dtype=torch.uint8)
        raw[raw == 0x7F] = 0x3C  # (no NaN's: a result of them is not equal to itself)
        raw[raw == 0xFF] = 0xBC
        layer.weight.data.copy_(raw.view(torch.float8_e4m3fn))  # (the load)
        x = torch.randn(5, K, dtype=torch.bfloat16)
        m.process_weights_after_loading(layer)
        w0 = layer.weight  # (the Parameter vLLM's kernels read: kept, its data swapped)
        want = {"nk": (O, K), "kn": (K, O), "i32": (O, K // 4)}[layout]
        assert layer.glyd_f8_id is not None and tuple(w0.shape) == want and (w0.stride() == (1, K) if layout == "kn" else w0.is_contiguous()) and w0.dtype == (torch.int32 if layout == "i32" else torch.float8_e4m3fn)
        m.process_weights_after_loading(layer)  # (again: nothing done twice)
        assert layer.weight is w0
        layers.append((m, layer, raw, layout, O, K, x))
    sc = f8._SCRATCH[torch.device("cpu")]
    assert sc.buf.numel() == 256 * 256 and not sc.sealed and len([r for r in sc.users if r() is not None]) == 5
    for m, layer, raw, layout, O, K, x in layers:  # (every layer's weight points into the one scratch, the same address)
        assert layer.weight.data_ptr() == sc.buf.data_ptr()
    n, packed, whole, scr = f8.stats(types.SimpleNamespace(modules=lambda: [types.SimpleNamespace(quant_method=m, glyd_f8_id=layer.glyd_f8_id, glyd_f8_nbytes=layer.glyd_f8_nbytes) for m, layer, *_ in layers]))
    assert n == 5 and packed == whole == 64 * 96 + 128 * 64 + 192 * 128 + 256 * 256 + 64 * 32 and scr == 256 * 256
    # products, out of order, the scratch overwritten with noise between them: each equals vLLM's method on the loaded weight
    for m, layer, raw, layout, O, K, x in layers[::-1] + layers:
        sc.buf.random_(0, 255)
        ref = fake_inner("nk")
        loaded = types.SimpleNamespace(weight=raw.view(torch.float8_e4m3fn), weight_scale=layer.weight_scale)
        want = ref.apply(loaded, x, None)
        got = torch.ops.glyd.f8_linear(x, None, layer.glyd_f8_id)
        assert got.dtype == want.dtype and got.shape == (5, O) and torch.equal(got, want), (layout, O, K)
        assert torch.equal(m.apply(layer, x, None), want)  # (the method's apply: the same op)
        b = torch.randn(O, dtype=torch.bfloat16)
        assert torch.equal(torch.ops.glyd.f8_linear(x, b, layer.glyd_f8_id), ref.apply(loaded, x, b))
        assert torch.ops.glyd.f8_linear(x[:0], None, layer.glyd_f8_id).shape == (0, O)  # (no tokens)
        assert torch.ops.glyd.f8_linear(x[None], None, layer.glyd_f8_id).shape == (1, 5, O)
    assert sc.sealed
    late = fake_layer("model.layers.9.mlp.down_proj", 512, 512)
    m = f8.GlydFp8LinearMethod(fake_inner("nk"), cfg)
    m.create_weights(late, 512, [512], 512, 512, torch.bfloat16)
    raises(lambda: m.process_weights_after_loading(late), "made after the model ran")
    f8._SCRATCH.clear()
    # a Linear whose layout is not met stays vLLM's own: the wrapper's apply is the inner method's
    odd = f8.GlydFp8LinearMethod(fake_inner("nk"), cfg)
    lay = fake_layer("model.layers.0.mlp.up_proj", 64, 64)
    odd.create_weights(lay, 64, [64], 64, 64, torch.bfloat16)
    odd.inner.process_weights_after_loading = lambda layer: setattr(layer, "weight", torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.float8_e4m3fn)[:, :32], requires_grad=False))
    odd.process_weights_after_loading(lay)
    assert lay.glyd_f8_id is None and not f8._SCRATCH
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    assert torch.equal(odd.apply(lay, x, None), torch.zeros(3, 64, dtype=torch.bfloat16))


def make_linear(prefix, O=64, K=96, codec=None):
    """A layer of the fake method, loaded and processed (packed): (method, layer)."""
    m = f8.GlydFp8LinearMethod(fake_inner("nk"), types.SimpleNamespace(glyd_codec=codec))
    layer = fake_layer(prefix, O, K)
    m.create_weights(layer, K, [O], K, O, torch.bfloat16)
    layer.weight.data.copy_(torch.randint(0, 0x70, layer.weight.shape, dtype=torch.uint8).view(torch.float8_e4m3fn))
    m.process_weights_after_loading(layer)
    return m, layer


def test_ids():
    """A packed Linear's id is its prefix (a graph vLLM compiled and keeps names it, so every process must give the same one); another live layer of the
    same prefix (a draft model's) is told apart by a number; the registry's entry, and the pack it holds, go with the layer."""
    if f8 is None:
        return print("test_ids: skipped (no vLLM)")
    import gc

    f8.register()
    f8._SCRATCH.clear()
    f8._REG.clear()
    m1, l1 = make_linear("model.layers.0.mlp.down_proj")
    m2, l2 = make_linear("model.layers.0.mlp.down_proj")
    assert l1.glyd_f8_id == "model.layers.0.mlp.down_proj" and l2.glyd_f8_id == "model.layers.0.mlp.down_proj#2" and set(f8._REG) == {l1.glyd_f8_id, l2.glyd_f8_id}
    x = torch.randn(3, 96, dtype=torch.bfloat16)
    assert torch.ops.glyd.f8_linear(x, None, l1.glyd_f8_id).shape == (3, 64)
    del m2, l2
    gc.collect()
    assert set(f8._REG) == {"model.layers.0.mlp.down_proj"}
    del m1, l1
    gc.collect()
    assert not f8._REG
    m3, l3 = make_linear("model.layers.0.mlp.down_proj")  # (built again: the same id)
    assert l3.glyd_f8_id == "model.layers.0.mlp.down_proj"
    f8._SCRATCH.clear()
    f8._REG.clear()


def test_poison():
    """GLYD_F8_POISON (a dev build's): the scratch overwritten after each product; the next product, which decodes first, is not moved by it."""
    if f8 is None or compiled():
        return print("test_poison: skipped (no vLLM, or a release build: it has no such variable)")
    f8.register()
    f8._SCRATCH.clear()
    m, layer = make_linear("model.layers.0.mlp.up_proj")
    x = torch.randn(4, 96, dtype=torch.bfloat16)
    clean = torch.ops.glyd.f8_linear(x, None, layer.glyd_f8_id)
    os.environ["GLYD_F8_POISON"] = "1"
    try:
        for _ in range(2):
            assert torch.equal(torch.ops.glyd.f8_linear(x, None, layer.glyd_f8_id), clean)
            assert bool((f8._SCRATCH[torch.device("cpu")].buf == 0xFF).all())
    finally:
        del os.environ["GLYD_F8_POISON"]
    f8._SCRATCH.clear()


def test_storage_freed():
    """A codec that copies (the compressing one): once a Linear is packed nothing holds the memory its weight was loaded in (the storage's last user is
    the test's own handle); the identity codec keeps it, as it is the pack."""
    if f8 is None:
        return print("test_storage_freed: skipped (no vLLM)")

    class Copy:
        name, version = "copy", 1

        @staticmethod
        def pack(w):
            b = w.view(torch.uint8)
            return f8codec.Packed("copy", b.shape, (b.clone(),))

        @staticmethod
        def decode(p, out):
            out.view(torch.uint8).copy_(p.tensors[0])

    f8codec.register(Copy)
    f8.register()
    try:
        counts = {}
        for codec in ("copy", "identity"):
            f8._SCRATCH.clear()
            m = f8.GlydFp8LinearMethod(fake_inner("nk"), types.SimpleNamespace(glyd_codec=codec))
            layer = fake_layer(f"model.layers.1.mlp.{codec}", 64, 96)
            m.create_weights(layer, 96, [64], 96, 64, torch.bfloat16)
            st = layer.weight.untyped_storage()  # (the test's own handle on the loaded weight's memory)
            m.process_weights_after_loading(layer)
            counts[codec] = torch._C._storage_Use_Count(st._cdata)
            assert layer.weight.untyped_storage().data_ptr() != st.data_ptr()  # (the Parameter is the scratch's now)
        assert counts == {"copy": 1, "identity": 2}, counts
    finally:
        f8codec._CODECS.pop("copy", None)
        f8._SCRATCH.clear()
        f8._REG.clear()


def test_probe():
    """vLLM's own probe of the quantization methods (ModelConfig._verify_quantization, run on its real code): --quantization glyd on an FP8 checkpoint stays
    "glyd" for every scheme, without it the checkpoint's own method as before, another quantization still vLLM's own mismatch, a bf16 checkpoint and a glyd save glyd."""
    if f8 is None:
        return print("test_probe: skipped (no vLLM)")
    from vllm.config.model import ModelConfig

    vp.register()

    def verify(user, cfg, hf_config=None):
        fake = types.SimpleNamespace(quantization=user, model_arch_config=types.SimpleNamespace(quantization_config=cfg), hf_config=hf_config)
        ModelConfig._verify_quantization(fake)
        return fake.quantization

    for cfg, own in ((QWEN, "fp8"), (MISTRAL, "fp8"), (MINIMAX, "mxfp8"), (MODELOPT, "modelopt"), (CT, "compressed-tensors")):
        assert verify("glyd", cfg) == "glyd" and verify(None, cfg) == own, cfg
    assert verify("glyd", None) == "glyd" and verify("glyd", {"quant_method": "glyd", "layout": "mma"}) == "glyd"
    assert verify("glyd", {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False, "sym": True}) == "glyd"  # (another quantization: its 16-bit parts, vllm_quant.py)
    raises(lambda: verify("glyd", {"quant_method": "modelopt", "quantization": {"quant_algo": "NVFP4"}}), "ModelOpt NVFP4")


def test_off():
    """Without GLYD_FP8=1 (and in every release build) the FP8 path is off: --quantization glyd on an FP8 checkpoint takes the path of any quantized
    checkpoint (vllm_quant.py: its FP8 Linears vLLM's own, its 16-bit parts packed), nothing of vllm_fp8's is built, and a directory with ModelOpt's
    hf_quant_config.json is not looked at."""
    if f8 is None:
        return print("test_off: skipped (no vLLM)")
    import tempfile

    from vllm.config.model import ModelConfig

    vp.register()
    old = os.environ.pop("GLYD_FP8")
    try:
        assert not vp._fp8_on()
        def verify(user, cfg):
            fake = types.SimpleNamespace(quantization=user, model_arch_config=types.SimpleNamespace(quantization_config=cfg), hf_config=None)
            ModelConfig._verify_quantization(fake)
            return fake.quantization

        for cfg in (QWEN, MISTRAL, MINIMAX, MODELOPT, CT):  # (claimed as any quantized checkpoint: vllm_quant.py's config, not the FP8 path's)
            assert vp.GlydConfig.override_quantization_method(cfg, "glyd") == "glyd" and verify("glyd", cfg) == "glyd", cfg
            c = vp.GlydConfig.from_config(cfg)
            assert isinstance(c, vp.vllm_quant.GlydQuantConfig) and not isinstance(c, f8.GlydFp8Config), (cfg, type(c))
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "hf_quant_config.json"), "w") as f:
                f.write("{}")
            seen, mf = [], f8.modelopt_file
            f8.modelopt_file = lambda *a, **k: seen.append(a) or True
            try:
                vp.GlydConfig().maybe_update_config(d, hf_config=None)
            finally:
                f8.modelopt_file = mf
            assert seen == [], seen  # (off: ModelOpt's file is never looked for)
    finally:
        os.environ["GLYD_FP8"] = old


def test_compile():
    """The product traced by torch.compile (Dynamo reads _trace.fp8_linear_method_apply's source, the op's fake gives its shape): one node a Linear,
    the same bits as eager for any number of tokens (the batch dimension dynamic), two Linears on one input as a model has them."""
    if f8 is None:
        return print("test_compile: skipped (no vLLM)")
    f8.register()
    f8._SCRATCH.clear()
    cfg = types.SimpleNamespace(glyd_codec=None)
    ms, layers = [], []
    for j in range(2):
        m = f8.GlydFp8LinearMethod(fake_inner("nk"), cfg)
        layer = fake_layer(f"model.layers.{j}.mlp.down_proj", 64 + 32 * j, 96)
        m.create_weights(layer, 96, [64 + 32 * j], 96, 64 + 32 * j, torch.bfloat16)
        layer.weight.data.copy_(torch.randint(0, 0x70, layer.weight.shape, dtype=torch.uint8).view(torch.float8_e4m3fn))
        m.process_weights_after_loading(layer)
        ms.append(m)
        layers.append(layer)

    def model(x):  # (two Linears on the one input, then one result of each)
        a, b = ms[0].apply(layers[0], x, None), ms[1].apply(layers[1], x, None)
        return (a[..., :64] + b[..., :64]) * 2

    sc = f8._SCRATCH[torch.device("cpu")]
    try:
        for backend in ("aot_eager", "inductor"):  # (inductor: the graph vLLM compiles with, whose scheduling can reorder independent calls and fuse around them)
            fn = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
            for n in (5, 1, 17):
                x = torch.randn(n, 96, dtype=torch.bfloat16)
                sc.buf.random_(0, 255)
                want = model(x)
                sc.buf.random_(0, 255)
                assert torch.equal(fn(x), want), (backend, n)
    except RuntimeError as e:
        if "not supported" not in str(e) and "Python 3.1" not in str(e):  # (a Python torch.compile does not run on: the plain op is the test)
            raise
        print(f"test_compile: skipped ({str(e).splitlines()[0]})")
    f8._SCRATCH.clear()

class FakeKernels:
    """kernels.f8_pack, f8_unpack_into, f8_unpack_experts and f8_linear on the CPU, by fp8ref (the reference the GPU's kernels give the same bytes as) and
    torch: the lossless codec's glue (its forms, its Packed, its decoders' views) runs on them here, the kernels themselves on a GPU (gpu/fp8/check_f8.py)."""

    def __enter__(self):
        from glyd_gpu import fp8ref, kernels

        self.k, self.saved = kernels, {n: getattr(kernels, n) for n in ("f8_pack", "f8_unpack_into", "f8_unpack_experts", "f8_linear")}
        self.calls = []

        def ref(p):
            return torch.from_numpy(fp8ref.unpack(fp8ref.Packed(p.shape, p.core.numpy(), p.heads.numpy(), p.var.numpy())))

        def pack(w):
            q = fp8ref.pack(w.contiguous().numpy())
            return kernels.F8Packed(q.shape, *(torch.from_numpy(a) for a in (q.core, q.heads, q.var)))

        def unpack_into(p, out):
            assert tuple(out.shape) == p.shape
            out.view(torch.uint8).copy_(ref(p))

        def unpack_experts(p, E, ids, out):
            full = ref(p).view(E, p.shape[0] // E, p.shape[1])
            assert tuple(out.shape[1:]) == tuple(full.shape[1:]) and ids.dtype == torch.int32
            for j, e in enumerate(ids.tolist()):
                if 0 <= e < E:
                    out[j].view(torch.uint8).copy_(full[e])
            return out

        def linear(p, kind, scale, x, out=None):
            self.calls.append((kind, x.shape[0]))
            w = ref(p).view(torch.float8_e4m3fn).float()
            if kind == "block":
                w = w * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
            else:
                w = w * scale
            return (x.float() @ w.t()).to(torch.bfloat16)

        kernels.f8_pack, kernels.f8_unpack_into, kernels.f8_unpack_experts, kernels.f8_linear = pack, unpack_into, unpack_experts, linear
        return self

    def __exit__(self, *a):
        for n, f in self.saved.items():
            setattr(self.k, n, f)


def e4m3(*shape, gen=None):
    """Random e4m3 bytes with no NaN (a NaN's result is not equal to itself), as a checkpoint's weights."""
    b = torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)
    b[(b & 0x7F) == 0x7F] = 0x38
    return b


def test_lossless():
    """The lossless codec's glue: the shapes it packs a matrix as (itself where its rows are whole patches' columns, else its bytes 128 at a time), the
    experts' stack each matrix whole patches of 64 rows (else None), its pack and decode bit for bit through the reference, and its size."""
    from glyd_gpu import fp8ref

    assert f8codec.form(5120, 17408) == (5120, 17408) and f8codec.form(2816, 704) == (2816 * 704 // 128, 128) and f8codec.form(3, 5) == (3, 5)
    assert f8codec.experts_form(256, 1024, 2048) == (256 * 1024, 2048) and f8codec.experts_form(128, 2816, 704) == (128 * 15488, 128)
    assert f8codec.experts_form(128, 1408, 2816) == (128 * 1408, 2816) and f8codec.experts_form(4, 32, 128) is None
    assert f8codec.experts_form(8, 16, 2048 * 4) == (8 * 1024, 128)  # (Marlin's int32 tiles of an expert [K / 16, 2N ...], as bytes)
    assert f8codec.experts_form(1, 64, 129) == (64, 129) and f8codec.experts_form(2, 64, 129) is None  # (the stack's bytes a multiple of 128, one matrix's not: pack would shape it otherwise)
    g = torch.Generator().manual_seed(1)
    with FakeKernels():
        for O, K in ((64, 256), (100, 200), (37, 64), (64, 704)):
            w = e4m3(O, K, gen=g)
            p = f8codec.pack(w.view(torch.float8_e4m3fn), "f8")
            assert p.codec == "f8" and p.shape == (O, K) and p.meta[0] == f8codec.form(O, K)
            assert p.nbytes() == fp8ref.pack(w.reshape(p.meta[0]).numpy()).nbytes() - 8  # (fp8ref counts the shape's 8 bytes too)
            out = torch.zeros(O, K, dtype=torch.float8_e4m3fn)
            f8codec.decode(p, out)
            assert torch.equal(out.view(torch.uint8), w), (O, K)
        E, O, K = 4, 128, 704
        stack = e4m3(E, O, K, gen=g)
        f = f8codec.experts_form(E, O, K)
        p = f8codec.pack(stack.reshape(f), "f8")
        out = torch.full((E, O, K), 7, dtype=torch.uint8)
        f8codec.decode_experts(p, E, torch.tensor([2, -1, 0, 9], dtype=torch.int32), out.view(E, *f8codec.form(O, K)))
        assert torch.equal(out[0], stack[2]) and torch.equal(out[2], stack[0]) and bool((out[1] == 7).all()) and bool((out[3] == 7).all())


def fake_moe(packable=True):
    """vLLM's FP8 method of a mixture of experts' experts, as far as the wrapper meets it: create_weights makes w13 [E, 2I, H] and w2 [E, H, I] (e4m3);
    process_weights_after_loading leaves them as they are; apply reads them from the layer, as the kernel does: each token's k experts, by topk_ids."""
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

    class Fp8MoEMethod(FusedMoEMethodBase):
        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra):
            E, H, I = num_experts, hidden_size, intermediate_size_per_partition
            layer.w13_weight = torch.nn.Parameter(torch.empty(E, 2 * I, H, dtype=torch.float8_e4m3fn), requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(torch.empty(E, H, I, dtype=torch.float8_e4m3fn), requires_grad=False)

        def get_fused_moe_quant_config(self, layer):
            return None

        def process_weights_after_loading(self, layer):
            self.processed = getattr(self, "processed", 0) + 1
            if not packable:
                layer.w2_weight = torch.nn.Parameter(layer.w2_weight.data.transpose(1, 2), requires_grad=False)

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            w13, w2 = layer.w13_weight.float(), layer.w2_weight.float()
            if not packable:
                w2 = w2.transpose(1, 2)
            y = torch.zeros(x.shape, dtype=torch.float32)
            for t in range(x.shape[0]):
                for e, c in zip(topk_ids[t].tolist(), topk_weights[t].tolist()):
                    g, u = (w13[e] @ x[t].float()).chunk(2)
                    y[t] += c * (w2[e] @ (torch.nn.functional.silu(g) * u))
            return y.to(x.dtype)

    return Fp8MoEMethod(types.SimpleNamespace(moe_parallel_config=types.SimpleNamespace(use_ep=False)))


def test_experts_flow():
    """A mixture of experts' FP8 method wrapped (a subclass of vLLM's own class, its state): the experts packed once vLLM has processed them, the layer's
    weights views of the one experts buffer (grown to the largest layer's), each call decoding the experts its tokens are routed to and only those (the
    others' bytes in the buffer left as they were), then vLLM's apply: its result on the loaded weights, bit for bit, with both codecs; not wrapped under
    expert parallel or a kernel that routes itself; experts of a layout not packed stay as vLLM runs them."""
    if f8 is None:
        return print("test_experts_flow: skipped (no vLLM)")
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

    cfg = types.SimpleNamespace(glyd_codec=None)
    m = fake_moe()
    other = object()
    assert f8.wrap_experts(cfg, None, "x") is None and f8.wrap_experts(cfg, other, "x") is other
    ep = fake_moe()
    ep.moe = types.SimpleNamespace(moe_parallel_config=types.SimpleNamespace(use_ep=True))
    assert f8.wrap_experts(cfg, ep, "model.layers.0.mlp.experts") is ep
    mono = fake_moe()
    mono.experts_cls = types.SimpleNamespace(is_monolithic=lambda: True)
    assert f8.wrap_experts(cfg, mono, "model.layers.0.mlp.experts") is mono
    mo = fake_moe()
    mo.__class__ = type("ModelOptFp8MoEMethod", (type(mo),), {})  # (ModelOpt's FP8 experts: not measured, refused under exact)
    raises(lambda: f8.wrap_experts(types.SimpleNamespace(glyd_codec=None, glyd_exact=True), mo, "model.layers.0.mlp.experts"), "exact on ModelOpt FP8 experts is not measured")
    assert type(f8.wrap_experts(types.SimpleNamespace(glyd_codec=None, glyd_exact=False), mo, "model.layers.0.mlp.experts")).__name__ == "GlydModelOptFp8MoEMethod"
    g = torch.Generator().manual_seed(2)
    with FakeKernels():
        for codec in ("identity", "f8"):
            f8._EXPERTS.clear()
            cfg = types.SimpleNamespace(glyd_codec=codec)
            layers = []
            for j, (E, H, I) in enumerate(((6, 128, 64), (8, 128, 128))):
                w = f8.wrap_experts(cfg, fake_moe(), f"model.layers.{j}.mlp.experts")
                assert isinstance(w, FusedMoEMethodBase) and type(w).__name__ == "GlydFp8MoEMethod" and type(w).__mro__[2].__name__ == "Fp8MoEMethod" and w.glyd_codec == codec
                layer = torch.nn.Module()
                layer.layer_name = f"model.layers.{j}.mlp.experts"
                w.create_weights(layer, E, H, I, torch.bfloat16)
                w13, w2 = e4m3(E, 2 * I, H, gen=g) // 4, e4m3(E, H, I, gen=g) // 4  # (small values: the reference's sums stay finite)
                layer.w13_weight.data.copy_(w13.view(torch.float8_e4m3fn))
                layer.w2_weight.data.copy_(w2.view(torch.float8_e4m3fn))
                w.process_weights_after_loading(layer)
                w.process_weights_after_loading(layer)  # (again: nothing done twice)
                assert w.processed == 1 and layer.glyd_f8_moe and layer.glyd_f8_nbytes == w13.numel() + w2.numel()
                layers.append((w, layer, w13, w2, E, H))
            sc = f8._EXPERTS[torch.device("cpu")]
            assert sc.buf.numel() == 8 * 2 * 128 * 128 + 8 * 128 * 128 and not sc.sealed
            n, packed, whole, scr = f8.stats(types.SimpleNamespace(modules=lambda: [x[1] for x in layers]))
            assert n == 2 and whole == sum(x[2].numel() + x[3].numel() for x in layers) and (packed == whole if codec == "identity" else packed < whole)
            for w, layer, w13, w2, E, H in layers[::-1] + layers:
                assert layer.w13_weight.data_ptr() == sc.buf.data_ptr()
                x = torch.randn(3, H, dtype=torch.bfloat16, generator=g)
                ids = torch.tensor([[1, 4], [4, 0], [1, 1]], dtype=torch.int32)
                tw = torch.rand(3, 2, generator=g)
                sc.buf.fill_(0xFF)
                got = w.apply(layer, x, tw, ids, None, None)
                ref = fake_moe()
                loaded = types.SimpleNamespace(w13_weight=w13.view(torch.float8_e4m3fn), w2_weight=w2.view(torch.float8_e4m3fn))
                assert torch.equal(got, ref.apply(loaded, x, tw, ids, None, None)), codec
                b13 = sc.buf[: w13.numel()].view(w13.shape)
                for e in range(E):  # (the routed experts decoded, the others not touched)
                    assert torch.equal(b13[e], w13[e]) if e in (0, 1, 4) else bool((b13[e] == 0xFF).all()), (codec, e)
                assert torch.equal(w.apply(layer, x[:0], tw[:0], ids[:0], None, None), x[:0])  # (no tokens: nothing decoded)
            assert sc.sealed
    f8._EXPERTS.clear()
    odd = f8.wrap_experts(cfg, fake_moe(packable=False), "model.layers.0.mlp.experts")
    layer = torch.nn.Module()
    odd.create_weights(layer, 4, 128, 64, torch.bfloat16)
    odd.process_weights_after_loading(layer)
    assert layer.glyd_f8_moe is False and not f8._EXPERTS and layer.w2_weight.shape == (4, 64, 128)


def test_verify_and_poison():
    """The checks' GLYD_F8_VERIFY (a development build's): every pack decoded back at load and compared with the bytes it was made from (a Linear's, and every
    expert of a stack), a codec that does not give them back refused; GLYD_F8_POISON: the experts buffer e4m3 NaN before each call, the routed experts
    decoded over it, the result vLLM's apply's on the loaded weights."""
    if f8 is None or compiled():
        return print("test_verify_and_poison: skipped (no vLLM, or a release build: it has no such variables)")
    f8.register()

    class Bad:
        name, version = "bad", 1

        @staticmethod
        def pack(w):
            return f8codec.Packed("bad", w.shape, (w.clone(),))

        @staticmethod
        def decode(p, out):
            o = out.view(torch.uint8).view(-1)
            o.copy_(p.tensors[0].view(-1))
            o[0] = o[0] ^ 1

    f8codec.register(Bad)
    os.environ["GLYD_F8_VERIFY"] = "1"
    try:
        f8._SCRATCH.clear()
        n0 = f8.VERIFIED[0]
        make_linear("model.layers.0.mlp.up_proj")
        assert f8.VERIFIED[0] == n0 + 1
        raises(lambda: make_linear("model.layers.1.mlp.up_proj", codec="bad"), "does not decode to the bytes it was made from")
        g = torch.Generator().manual_seed(3)
        with FakeKernels():
            for codec in ("identity", "f8"):
                f8._EXPERTS.clear()
                w = f8.wrap_experts(types.SimpleNamespace(glyd_codec=codec), fake_moe(), "model.layers.0.mlp.experts")
                layer = torch.nn.Module()
                E, H, I = 6, 128, 64
                w.create_weights(layer, E, H, I, torch.bfloat16)
                w13, w2 = e4m3(E, 2 * I, H, gen=g) // 4, e4m3(E, H, I, gen=g) // 4
                layer.w13_weight.data.copy_(w13.view(torch.float8_e4m3fn))
                layer.w2_weight.data.copy_(w2.view(torch.float8_e4m3fn))
                n0 = f8.VERIFIED[0]
                w.process_weights_after_loading(layer)
                assert f8.VERIFIED[0] == n0 + 2, codec  # (w13 and w2, every expert of each)
                os.environ["GLYD_F8_POISON"] = "1"
                try:
                    x, ids, tw = torch.randn(2, H, dtype=torch.bfloat16, generator=g), torch.tensor([[1, 3], [3, 1]], dtype=torch.int32), torch.rand(2, 2, generator=g)
                    got = w.apply(layer, x, tw, ids, None, None)
                finally:
                    del os.environ["GLYD_F8_POISON"]
                loaded = types.SimpleNamespace(w13_weight=w13.view(torch.float8_e4m3fn), w2_weight=w2.view(torch.float8_e4m3fn))
                assert torch.equal(got, fake_moe().apply(loaded, x, tw, ids, None, None)), codec
                b13 = f8._EXPERTS[torch.device("cpu")].buf[: w13.numel()].view(w13.shape)
                assert all(torch.equal(b13[e], w13[e]) if e in (1, 3) else bool((b13[e] == 0x7F).all()) for e in range(E)), codec
    finally:
        del os.environ["GLYD_F8_VERIFY"]
        f8codec._CODECS.pop("bad", None)
        f8._SCRATCH.clear()
        f8._EXPERTS.clear()
        f8._REG.clear()


def test_fused():
    """f8_linear's product for a step of few tokens (_fused): only where a check's GLYD_F8_FUSED_MAX is above 0 (a development build's; off otherwise), never under exact, and only
    for the lossless pack of the row-major e4m3 weight with one fp32 scale or fp32 128 x 128 blocks and no static activation scale; the op takes it for steps of up to that many
    tokens and the decode and vLLM's product for more."""
    if f8 is None:
        return print("test_fused: skipped (no vLLM)")
    f8.register()
    with FakeKernels() as fk:
        for exact, env, want in ((False, None, 0), (True, None, 0), (True, "8", 0), (False, "8", 8), (False, "99", 64)):  # (no table read: off unless a check asks)
            if env is None:
                os.environ.pop("GLYD_F8_FUSED_MAX", None)
            elif compiled():
                continue
            else:
                os.environ["GLYD_F8_FUSED_MAX"] = env
            assert f8.fused_max(types.SimpleNamespace(glyd_exact=exact)) == want, (exact, env)
        if compiled():
            return print("test_fused: the rest skipped (a release build reads no GLYD_F8_FUSED_MAX)")
        os.environ["GLYD_F8_FUSED_MAX"] = "4"
        try:
            f8._SCRATCH.clear()
            for codec, layout, kind, static in (("f8", "nk", "tensor", False), ("f8", "nk", "block", False), ("f8", "kn", "tensor", False), ("f8", "i32", None, False), ("identity", "nk", None, False), ("f8", "nk", None, True)):
                inner = fake_inner(layout)
                if kind == "block":
                    inner.weight_block_size = [128, 128]
                m = f8.GlydFp8LinearMethod(inner, types.SimpleNamespace(glyd_codec=codec, glyd_exact=False))
                O, K = 256, 384
                layer = fake_layer("model.layers.0.mlp.down_proj", O, K)
                m.create_weights(layer, K, [O], K, O, torch.bfloat16)
                layer.input_size_per_partition = K
                if static:  # (a static activation scale, which f8_linear would not apply)
                    layer.input_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
                raw = e4m3(O, K) // 4
                layer.weight.data.copy_(raw.view(torch.float8_e4m3fn))
                if kind == "block":
                    layer.weight_scale_inv = torch.nn.Parameter(torch.rand(2, 3) + 0.5, requires_grad=False)
                m.process_weights_after_loading(layer)
                r = f8._REG[layer.glyd_f8_id]
                assert (r.fused is None) == (kind is None) and (kind is None or (r.fused[0], r.fused[2]) == (kind, 4)), (codec, layout, r.fused)
                if kind is None:
                    continue
                fk.calls.clear()
                for n in (1, 4, 5):
                    x = torch.randn(n, K, dtype=torch.bfloat16)
                    y = torch.ops.glyd.f8_linear(x, None, layer.glyd_f8_id)
                    assert y.shape == (n, O) and y.dtype == torch.bfloat16
                    if kind == "tensor":  # (the fake's sums are the reference's: the same as the decoded path's here)
                        assert torch.equal(y, fake_inner("nk").apply(types.SimpleNamespace(weight=raw.view(torch.float8_e4m3fn), weight_scale=layer.weight_scale), x, None))
                assert fk.calls == [(kind, 1), (kind, 4)]  # (5 tokens: the decode and vLLM's product)
            f8._SCRATCH.clear()
            f8._REG.clear()
        finally:
            os.environ.pop("GLYD_F8_FUSED_MAX", None)


if __name__ == "__main__":
    tests = [t for name, t in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(test.__name__, "ok")
    if vp is None:
        import inspect

        n = sum("if f8 is None" in inspect.getsource(t) for t in tests)
        print(f"NOTE: vLLM is not installed, so {n} of these {len(tests)} tests did not run (GLYD_REQUIRE_VLLM=1 makes that a failure)")
