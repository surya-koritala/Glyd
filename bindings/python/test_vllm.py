"""glyd.gpu's vLLM plugin, its logic that needs no GPU: the entry point's version rule; the options' precedence and
what is refused among them; the layers a fraction packs (how many, how spread); a save's packs by vLLM's layer names;
the pieces a checkpoint gave; what is refused in vLLM's config; and a draft model asked for --quantization glyd (its
own size, its packs in the digest). Needs vLLM (glyd[vllm]) but for the version rule, else skipped.

    python test_vllm.py              (or pytest test_vllm.py)"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glyd.gpu import vllm_entry  # noqa: E402

try:
    import vllm  # noqa: F401
    from glyd.gpu import vllm_plugin as vp
except ImportError:
    vp = None


def raises(f, text):
    try:
        f()
    except (ValueError, RuntimeError) as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"not refused: {text}")


def test_version_rule():
    assert vllm_entry.tested("0.30.0") and vllm_entry.tested("0.30.1rc1") and vllm_entry.tested("0.30.2.dev3+g1234")
    assert not vllm_entry.tested("0.31.0") and not vllm_entry.tested("0.3.0") and not vllm_entry.tested("1.30.0")


def test_options():
    if vp is None:
        return print("test_options: skipped (no vLLM)")
    none = {"layout": None, "exact": None, "verify": None, "fraction": None}
    assert vp._options({}, {}, none) == ("auto", False, False, 1.0)
    assert vp._options({"layout": "MMA12"}, {"layout": "mma"}, dict(none, layout="mma")) == ("mma12", False, False, 1.0)  # --additional-config first
    assert vp._options({}, {"exact": True}, dict(none, exact="0")) == ("auto", True, False, 1.0)  # then the quantization_config
    assert vp._options({}, {}, {"layout": "mma", "exact": "Yes", "verify": "off", "fraction": "0.5"}) == ("mma", True, False, 0.5)  # then the environment
    assert vp._options({"fraction": 0.25}, {"fraction": 0.5}, dict(none, fraction="0.75"))[3] == 0.25  # (fraction as the others)
    assert vp._options({}, {"fraction": 0.5}, dict(none, fraction="0.75"))[3] == 0.5 and vp._options({}, {}, dict(none, fraction="0.75"))[3] == 0.75
    assert vp._options({"fraction": 0}, {}, dict(none, fraction="1"))[3] == 0.0 and vp._options({}, {}, dict(none, fraction="0"))[3] == 0.0  # (0 is a value, not unset)
    assert vp._options({"fraction": 1}, {}, none)[3] == 1.0 and vp._options({}, {}, dict(none, fraction=""))[3] == 1.0 and vp._options({"fraction": " 0.5 "}, {}, none)[3] == 0.5
    assert vp._options({"layout": "mma", "glyd": "0.26.0", "packs": "0123"}, {}, none)[0] == "mma"  # (resolve's own keys)
    raises(lambda: vp._options({"exat": True}, {}, none), "not an option")
    raises(lambda: vp._options("mma12", {}, none), "not an object of options")
    raises(lambda: vp._options({}, {}, dict(none, exact="ture")), "true or false")
    raises(lambda: vp._options({"layout": "fast"}, {}, none), "one of auto, mma, mma12")
    for bad in (1.5, -0.1, "half", True, "nan", "inf", [0.5], {"a": 1}):  # (a fraction: a number from 0 to 1)
        raises(lambda: vp._options({"fraction": bad}, {}, none), "a number from 0 to 1")
    raises(lambda: vp._options({}, {}, dict(none, fraction="1.01")), "fraction '1.01': a number from 0 to 1")
    assert vp.GlydConfig.from_config({"quant_method": "glyd", "layout": "mma", "merge": True}).given == {"layout": "mma"}
    assert vp.GlydConfig.from_config({"fraction": 0.5}).given == {"fraction": 0.5}
    raises(lambda: vp.GlydConfig.from_config({"layot": "mma"}), "not an option")


def test_fraction():
    """The layers a fraction packs: floor((i + 1) f) > floor(i f). Of the first L layers exactly floor(L f), spread evenly
    (any run of w layers holds floor(w f) or one more), none at 0, all at 1; 0.29 of 100 layers is 29 (a float's
    0.29 * 100 is 28.999999999999996); a module's layer by the first number in its name; what packs() says of it."""
    if vp is None:
        return print("test_fraction: skipped (no vLLM)")
    from fractions import Fraction
    from math import floor

    packed = lambda f, L: [i for i in range(L) if vp._packs(i, f)]
    assert packed(0.5, 36) == list(range(1, 36, 2)) and packed(0.25, 36) == list(range(3, 36, 4))  # (Qwen3-8B's 36 layers)
    assert packed(0.75, 8) == [1, 2, 3, 5, 6, 7] and packed(0.34, 9) == [2, 5, 8]
    for L in (1, 2, 3, 28, 36, 40, 64, 80, 100, 126):
        assert packed(0.0, L) == [] and packed(1.0, L) == list(range(L)), L  # 0: none, 1: every layer, today's
    for f in (0.0, 0.05, 0.1, 0.25, 0.29, 0.3, 0.33, 0.5, 0.57, 0.58, 0.6, 0.7, 0.75, 0.9, 0.99, 1.0):
        q = Fraction(repr(f))
        for L in (1, 3, 28, 36, 64, 100):
            assert len(packed(f, L)) == floor(L * q), (f, L)
        for w in (1, 2, 5, 7, 16):  # (spread: no run holds more than one over its share, or fewer than it)
            for a in range(0, 40):
                n = sum(vp._packs(i, f) for i in range(a, a + w))
                assert floor(w * q) <= n <= floor(w * q) + 1, (f, a, w, n)
    assert len(packed(0.29, 100)) == 29 and len(packed(0.57, 100)) == 57 and len(packed(0.58, 100)) == 58
    assert vp._fraction(0.3) == 0.3 and vp._fraction("1") == 1.0 and vp._fraction(None) == 1.0
    assert [vp._layer_of(n) for n in ("model.layers.12.mlp.down_proj", "transformer.h.3.attn.c_attn", "model.decoder.layers.0.self_attn.qkv_proj", "model.layers.36.self_attn.o_proj", "model.layers.3.mlp.experts", "lm_head", "model.embed_tokens", "multi_modal_projector.linear_1", "model.layers.\u0663.x", "")] == [12, 3, 0, 36, 3, None, None, None, None, None]
    c = vp.GlydConfig()
    c.opts = {"fraction": 0.5}
    assert [c.packs(f"model.layers.{i}.self_attn.qkv_proj") for i in range(4)] == [False, True, False, True]
    assert all(c.packs(f"model.layers.1.{m}") for m in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj", "mlp.experts"))  # (a layer's together)
    assert not any(c.packs(f"model.layers.0.{m}") for m in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj", "mlp.experts"))
    assert not c.packs("multi_modal_projector.linear_1")  # (outside the numbered layers: only at fraction 1)
    c.opts = {"fraction": 1.0}
    assert c.packs("multi_modal_projector.linear_1") and c.packs("model.layers.0.mlp.down_proj")
    c.opts = {"fraction": 0.0}
    assert not c.packs("multi_modal_projector.linear_1") and not any(c.packs(f"model.layers.{i}.mlp.down_proj") for i in range(64))
    lin, other = 1000.0, 100.0  # (the fit: the Linears not packed count their bf16 bytes)
    assert vp._estimate(lin, other, "mma", 1, fraction=0.0) == (lin + other, lin + other) and vp._estimate(lin, other, "mma", 1)[0] == (lin * 10.80 / 16 + other)
    full, half, bf16 = (vp._estimate(lin, other, "mma12", 1, fraction=f)[0] for f in (1.0, 0.5, 0.0))
    assert full < half < bf16 and abs(half - (lin * 0.5 * 12.04 / 16 + lin * 0.5 + other)) < 1e-9
    assert vp._estimate(lin, other, "mma", 2, draft=True, fraction=0.5)[0] == (lin * 0.5 * 10.80 / 16 + lin * 0.5) / 2


def test_quant_method():
    """get_quant_method under a fraction: a layer's Linears go to Glyd or, left out, to vLLM's own
    UnquantizedLinearMethod (what vLLM gives a layer with no quantization), by the layer's number; every Linear of a
    layer alike; the embeddings and the LM head as without a fraction."""
    if vp is None:
        return print("test_quant_method: skipped (no vLLM)")
    c = vp.GlydConfig()
    lin = object.__new__(vp.LinearBase)  # (a Linear, without its weights: get_quant_method reads its class and its prefix)
    kinds = lambda parts: [[type(c.get_quant_method(lin, f"model.layers.{i}.{p}")).__name__ for p in parts] for i in range(4)]
    parts = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
    glyd, plain = ["GlydLinearMethod"] * 4, ["UnquantizedLinearMethod"] * 4
    c.opts = {"layout": "mma", "exact": False, "verify": False, "fraction": 0.5}
    assert kinds(parts) == [plain, glyd, plain, glyd]
    assert type(c.get_quant_method(lin, "visual.merger.linear_fc1")).__name__ == "UnquantizedLinearMethod"  # (outside the layers)
    c.opts["fraction"] = 0.0
    assert kinds(parts) == [plain] * 4
    c.opts["fraction"] = 1.0
    assert kinds(parts) == [glyd] * 4 and type(c.get_quant_method(lin, "visual.merger.linear_fc1")).__name__ == "GlydLinearMethod"
    assert c.get_quant_method(object.__new__(vp.VocabParallelEmbedding), "lm_head") is None  # (a bf16 checkpoint's: vLLM's own, as before)


def test_saved():
    if vp is None:
        return print("test_saved: skipped (no vLLM)")
    c = vp.GlydConfig()
    one, two, three = ({"tensors": [{}] * n} for n in (1, 2, 3))
    c.manifest = {"packs": {
        "model.layers.0.self_attn.q_proj": three, "model.layers.0.mlp.gate_proj": two, "model.layers.0.self_attn.o_proj": one,
        "model.layers.1.self_attn.qkv_proj": one,  # (a family that keeps q, k and v fused on disk: Phi-3)
        "model.layers.2.mlp.gate_proj": one,  # (a count vLLM's layer does not have)
    }}
    assert c.saved("model.layers.0.self_attn.qkv_proj") is three and c.saved("model.layers.0.mlp.gate_up_proj") is two
    assert c.saved("model.layers.0.self_attn.o_proj") is one and c.saved("model.layers.1.self_attn.qkv_proj") is one
    assert c.saved("model.layers.3.self_attn.o_proj") is None
    raises(lambda: c.saved("model.layers.2.mlp.gate_up_proj"), "holds 1 tensors, vLLM's layer 2")


def test_pieces():
    if vp is None:
        return print("test_pieces: skipped (no vLLM)")
    assert vp._missing("qkv", ["q", "k", "v"]) == [] and vp._missing("qkv", [None]) == []
    assert vp._missing("qkv", ["q", "v"]) == ["k"] and vp._missing("qkv", []) == ["k", "q", "v"]
    assert vp._missing("merged", [0], 2) == [1] and vp._missing("merged", [(0, 1)], 2) == []
    assert vp._missing("one", []) == [None] and vp._missing("one", [None]) == []
    full = [(e, s) for e in range(2) for s in ("w1", "w3", "w2")]
    assert vp._experts_missing(full, 2) == [] and vp._experts_missing(full[:4], 2) == ["expert 1's w2", "expert 1's w3"]
    assert vp._experts_missing([(0, "w13")], 1) is None  # (another loader's way)


def test_refusals():
    if vp is None:
        return print("test_refusals: skipped (no vLLM)")
    ns = types.SimpleNamespace
    compiled = next(m for m in vp.CompilationMode if m != vp.CompilationMode.NONE)
    graphs = next(m for m in vp.CUDAGraphMode if m != vp.CUDAGraphMode.NONE)

    def vc(eager=False, icc=None, tp=1, ubatching=False, lora=None, offload=0, sleep=False):
        return ns(parallel_config=ns(use_ubatching=ubatching, tensor_parallel_size=tp, pipeline_parallel_size=1), lora_config=lora,
                  offload_config=ns(uva=ns(cpu_offload_gb=offload), prefetch=ns(offload_group_size=0)),
                  model_config=ns(enforce_eager=eager, enable_sleep_mode=sleep),
                  compilation_config=ns(inductor_compile_config=icc or {}, mode=compiled, cudagraph_mode=graphs))

    det = {"deterministic": True, "combo_kernels": True, "benchmark_combo_kernel": False}
    vp._refusals(vc(), False, None)
    vp._refusals(vc(eager=True), True, None)
    vp._refusals(vc(icc=det), True, None)
    raises(lambda: vp._refusals(vc(), True, None), "--enforce-eager")
    raises(lambda: vp._refusals(vc(icc={"deterministic": True}), True, None), "--enforce-eager")  # (the combo kernels' benchmark on)
    raises(lambda: vp._refusals(vc(ubatching=True), False, None), "dual-batch overlap")
    raises(lambda: vp._refusals(vc(lora=object()), False, None), "LoRA")
    raises(lambda: vp._refusals(vc(offload=4), False, None), "weight offloading")
    raises(lambda: vp._refusals(vc(sleep=True), False, None), "sleep mode")
    raises(lambda: vp._refusals(vc(tp=2), False, {"packs": {}}), "one GPU")
    vp._refusals(vc(), False, {"packs": {}}, 1.0)  # (a save at fraction 1; a bf16 checkpoint at any)
    vp._refusals(vc(), False, None, 0.5)
    raises(lambda: vp._refusals(vc(), False, {"packs": {}}, 0.5), "fraction 0.5 leaves some layers bf16")
    raises(lambda: vp._refusals(vc(), False, {"packs": {"m.experts.w13": {"experts": 8}}}), "mixture of experts' packs")
    was = os.environ.get("VLLM_BATCH_INVARIANT")
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    try:
        raises(lambda: vp._refusals(vc(), False, None), "VLLM_BATCH_INVARIANT")
        vp._refusals(vc(eager=True), True, None)  # (exact: vLLM's batch-invariant GEMM on the decoded weights)
    finally:
        os.environ.pop("VLLM_BATCH_INVARIANT") if was is None else os.environ.update(VLLM_BATCH_INVARIANT=was)


def test_draft():
    """A draft asked for --quantization glyd too (vLLM builds it a GlydConfig of its own): sized by its own config, not
    the target's, and its packs added to the process's digest, never resetting the target's."""
    if vp is None:
        return print("test_draft: skipped (no vLLM)")
    import tempfile
    import torch

    ns = types.SimpleNamespace
    target = ns(hidden_size=4096, num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128, intermediate_size=12288, vocab_size=151936, tie_word_embeddings=False)
    draft = ns(**{**vars(target), "num_hidden_layers": 1})
    lt, ot, mt = vp._linear_bytes(target)
    ld, od, md = vp._linear_bytes(draft)
    assert lt == 36 * ld and not mt and not md and ot == od
    assert vp._linear_bytes(ns(hidden_size=8)) == (0, 0, False)  # (a config it cannot read: no estimate)
    need, low = vp._estimate(ld, od, "mma", 1)
    dneed, dlow = vp._estimate(ld, od, "mma", 1, draft=True)  # (a draft: its Linears alone)
    assert dneed == ld * 10.80 / 16 and dlow == ld * 2 / 3 * 10.80 / 16 and need == dneed + od and low == dlow + od
    assert vp._estimate(lt, ot, "mma12", 2)[0] == (lt * 12.04 / 16 + ot) / 2
    c = vp.GlydConfig()
    with tempfile.TemporaryDirectory() as d:
        c.maybe_update_config(d, hf_config=draft)  # (a directory without glyd.json: not a save)
    assert c.hf_config is draft and c.manifest is None
    was = dict(vp._PACKS)
    vp._PACKS.clear()
    try:
        key = {"glyd": {"packs": ""}}
        a, b = vp.GlydConfig(), vp.GlydConfig()
        a._vc = b._vc = ns(additional_config=key)
        a.packed("model.layers.0.self_attn.qkv_proj", "mma", [1, 2, 3], [torch.empty(4)])
        first = key["glyd"]["packs"]
        b.packed("model.layers.36.self_attn.qkv_proj", "mma", [1, 2, 3], [torch.empty(4)])  # (the draft's layers: their own names)
        assert first and key["glyd"]["packs"] != first and len(vp._PACKS) == 2 and key["glyd"]["packs"] == vp._digest()
    finally:
        vp._PACKS.clear()
        vp._PACKS.update(was)


def test_moe_route():
    """The tokens a step from which a mixture of experts' layer takes its routed experts decoded: GLYD_MOE_DECODE_MIN,
    else the measured default; negative, never; anything but a number refused."""
    if vp is None:
        return print("test_moe_route: skipped (no vLLM)")
    was = os.environ.pop("GLYD_MOE_DECODE_MIN", None)
    try:
        assert vp._moe_decode_min() == vp.MOE_DECODE_MIN > 0
        os.environ["GLYD_MOE_DECODE_MIN"] = "1"
        assert vp._moe_decode_min() == 1
        os.environ["GLYD_MOE_DECODE_MIN"] = "-1"
        assert vp._moe_decode_min() is None
        os.environ["GLYD_MOE_DECODE_MIN"] = "lots"
        raises(vp._moe_decode_min, "a number of tokens a step")
    finally:
        os.environ.pop("GLYD_MOE_DECODE_MIN", None)
        if was is not None:
            os.environ["GLYD_MOE_DECODE_MIN"] = was


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
