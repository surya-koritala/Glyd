"""glyd_gpu's vLLM plugin, its logic that needs no GPU: the entry point's version rule; the options' precedence and
what is refused among them; the layers a fraction packs (how many, how spread); a save's packs by vLLM's layer names;
the pieces a checkpoint gave; what is refused in vLLM's config; and a draft model asked for --quantization glyd (its
own size, its packs in the digest); the load's allocator trim and the warning where there is no nvcc. Needs vLLM
(glyd[vllm]; no GPU) but for the version rule, else skipped with a line saying so: GLYD_REQUIRE_VLLM=1 (the CI job that
installs vLLM) makes it a failure, so a run without vLLM cannot pass as if it had run.

    python test_vllm.py              (or pytest test_vllm.py)

The glyd_gpu it runs: the installed one, or this checkout's sources with PYTHONPATH=.."""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
from glyd_gpu import vllm_entry  # noqa: E402 (PYTHONPATH=.. runs this checkout's package; else the one installed, a wheel's compiled one)

try:
    import vllm  # noqa: F401
    from glyd_gpu import vllm_plugin as vp
except ImportError as e:
    if os.environ.get("GLYD_REQUIRE_VLLM") == "1":
        raise SystemExit(f"GLYD_REQUIRE_VLLM=1 and vLLM or the plugin does not import: {type(e).__name__}: {e}")
    vp = None


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


def test_version_rule():
    assert vllm_entry.tested("0.30.0") and vllm_entry.tested("0.30.1rc1") and vllm_entry.tested("0.30.2.dev3+g1234")
    assert not vllm_entry.tested("0.31.0") and not vllm_entry.tested("0.3.0") and not vllm_entry.tested("1.30.0")


def test_options():
    if vp is None:
        return print("test_options: skipped (no vLLM)")
    none = {"layout": None, "exact": None, "verify": None, "fraction": None, "kv": None}
    assert vp._options({}, {}, none) == ("auto", False, False, 1.0, "auto", False)  # (kv's default is auto: lossless where it is measured faster, vLLM's own cache elsewhere)
    assert vp._options({"layout": "MMA12"}, {"layout": "mma"}, dict(none, layout="mma")) == ("mma12", False, False, 1.0, "auto", False)  # --additional-config first
    assert vp._options({}, {"exact": True}, dict(none, exact="0")) == ("auto", True, False, 1.0, "auto", False)  # then the quantization_config
    assert vp._options({}, {}, {"layout": "mma", "exact": "Yes", "verify": "off", "fraction": "0.5", "kv": None}) == ("mma", True, False, 0.5, "auto", False)  # then the environment
    # cc12: off by default, asked by --additional-config or the quantization_config (no environment variable)
    assert vp._options({"cc12": True}, {}, none)[5] and vp._options({}, {"cc12": "yes"}, none)[5] and not vp._options({"cc12": False}, {"cc12": True}, none)[5]
    raises(lambda: vp._options({"cc12": "maybe"}, {}, none), "true or false")
    # the route it gives a Linear's product: GEMM_CC12 for the mma layout's GEMM where asked, else the library's own (-1)
    ways, cc12 = dict(vp._WAYS), vp._CC12[0]
    try:
        vp._WAYS.update({(0, n, 4096, 4096, M): vp.g.GEMM if M <= 64 else vp.g.X1 for n in (3, 4) for M in (16, 128)})  # (the library's routes, as cached)
        vp._CC12[0] = False
        assert vp._way_given(0, [1, 2, 3], 4096, 4096, 16) == -1
        vp._CC12[0] = True
        assert vp._way_given(0, [1, 2, 3], 4096, 4096, 16) == vp.g.GEMM_CC12 == 7
        assert vp._way_given(0, [1, 2, 3], 4096, 4096, 128) == -1 and vp._way_given(0, [1, 2, 3, 4], 4096, 4096, 16) == -1  # (a prompt; the mma12 layout)
    finally:
        vp._WAYS.clear()
        vp._WAYS.update(ways)
        vp._CC12[0] = cc12
    assert vp._options({"kv": "Lossless"}, {"kv": "off"}, dict(none, kv="off"))[4] == "lossless" and vp._options({}, {"kv": "lossless"}, dict(none, kv="off"))[4] == "lossless"  # (kv: as the others)
    assert vp._options({}, {}, dict(none, kv=" LOSSLESS "))[4] == "lossless" and vp._options({"kv": "off"}, {}, dict(none, kv="lossless"))[4] == "off" and vp._options({"kv": ""}, {}, none)[4] == "auto"
    assert vp._options({"kv": "lossless", "kv_tables": {"path": "x", "bay": 63}, "glyd": "0.26.0"}, {}, none)[4] == "lossless"  # (resolve's own keys)
    raises(lambda: vp._options({"kv": "lossy"}, {}, none), "one of auto, lossless, off")
    raises(lambda: vp._options({}, {}, dict(none, kv="1")), "one of auto, lossless, off")
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
    assert vp.GlydConfig.from_config({"fraction": 0.5}).given == {"fraction": 0.5} and vp.GlydConfig.from_config({"kv": "lossless"}).given == {"kv": "lossless"}
    assert vp.GlydConfig.from_config({"cc12": True}).given == {"cc12": True}
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
    # the embedding and the LM head: packed at fraction 1 (a plain lookup, a hidden size of a multiple of 64), else as without Glyd
    c.opts = {"layout": "mma", "exact": False, "verify": False, "fraction": 1.0}
    c.plan = (True, True, None)
    table = lambda cls=vp.VocabParallelEmbedding, K=4096: _table(cls, K)
    kind = lambda layer, prefix: type(c.get_quant_method(layer, prefix)).__name__
    assert kind(table(), "model.embed_tokens") == "GlydEmbeddingMethod" and kind(table(vp.ParallelLMHead), "lm_head") == "GlydEmbeddingMethod"
    assert c.get_quant_method(table(K=100), "model.embed_tokens") is None and "multiple of 64" in " ".join(c.noted)  # (a hidden size the head's product does not take)
    plain = type("ScaledEmbedding", (vp.VocabParallelEmbedding,), {})
    assert c.get_quant_method(table(plain), "model.embed_tokens") is None and any("ScaledEmbedding" in n for n in c.noted)  # (an embedding class of its own)
    assert kind(table(), "") == "GlydEmbeddingMethod" and c.get_quant_method(table(), "").name() == "embed_tokens" and c.get_quant_method(table(vp.ParallelLMHead), "").name() == "lm_head"
    c.plan = (True, False, "exact")  # (exact, an untied model: the embedding packed, the LM head as vLLM runs it)
    assert kind(table(), "model.embed_tokens") == "GlydEmbeddingMethod" and c.get_quant_method(table(vp.ParallelLMHead), "lm_head") is None
    c.plan = (False, False, "packed at fraction 1 only")
    assert c.get_quant_method(table(), "model.embed_tokens") is None and c.get_quant_method(table(vp.ParallelLMHead), "lm_head") is None
    assert not c.noted - {n for n in c.noted if "ScaledEmbedding" in n or "multiple of 64" in n}  # (the plan's own reason is the resolve's line, not one a layer adds)

    # a mixture of experts' router (128 x 2,048) stays vLLM's; a Linear of the sizes of a model's qkv does not
    router = object.__new__(vp.LinearBase)
    router.input_size, router.output_size = 2048, 128
    assert type(c.get_quant_method(router, "model.layers.0.mlp.gate")).__name__ == "UnquantizedLinearMethod"
    router.output_size = 5120
    assert type(c.get_quant_method(router, "model.layers.0.self_attn.qkv_proj")).__name__ == "GlydLinearMethod"
    # an H100 or an A100 and a model with routed experts: its attention's Linears stay vLLM's (a run from the sources: GLYD_MOE_DENSE=packed packs them), its other Linears and any other GPU's do not
    import torch

    ns = types.SimpleNamespace
    big = object.__new__(vp.LinearBase)
    big.input_size, big.output_size = 2048, 5120
    kind = lambda prefix: type(c.get_quant_method(big, prefix)).__name__
    was, real = os.environ.pop("GLYD_MOE_DENSE", None), (torch.cuda.is_available, torch.cuda.get_device_name)
    try:
        torch.cuda.is_available = lambda: True
        for name, experts, want in (("NVIDIA H100 80GB HBM3", 128, "UnquantizedLinearMethod"), ("NVIDIA H100 PCIe", 8, "UnquantizedLinearMethod"), ("NVIDIA H100 80GB HBM3", 0, "GlydLinearMethod"),
                                    ("NVIDIA A100-SXM4-80GB", 128, "UnquantizedLinearMethod"), ("NVIDIA A100-SXM4-40GB", 64, "UnquantizedLinearMethod"), ("NVIDIA A100 80GB PCIe", 8, "UnquantizedLinearMethod"),
                                    ("NVIDIA A100-SXM4-80GB", 0, "GlydLinearMethod"), ("NVIDIA A1000", 128, "GlydLinearMethod"), ("NVIDIA GH200 480GB", 128, "GlydLinearMethod"), ("NVIDIA L4", 128, "GlydLinearMethod")):
            torch.cuda.get_device_name = lambda n=name: n
            c._vc = ns(model_config=ns(hf_config=ns(num_experts=experts, get_text_config=lambda e=experts: ns(num_experts=e))))
            assert kind("model.layers.0.self_attn.qkv_proj") == want, (name, experts, kind("model.layers.0.self_attn.qkv_proj"))
            assert kind("model.layers.0.attention.wqkv") == want  # (a family that names it attention)
            assert kind("model.layers.0.mlp.shared_expert.gate_up_proj") == "GlydLinearMethod"  # (a shared expert's Linear is packed on any GPU)
        torch.cuda.get_device_name = lambda: "NVIDIA H100 80GB HBM3"
        c._vc = ns(model_config=ns(hf_config=ns(num_experts=128, get_text_config=lambda: ns(num_experts=128))))
        os.environ["GLYD_MOE_DENSE"] = "packed"
        assert kind("model.layers.0.self_attn.qkv_proj") == ("UnquantizedLinearMethod" if compiled() else "GlydLinearMethod")  # (a release build reads no tuning variable)
        torch.cuda.get_device_name = lambda: "NVIDIA A100-SXM4-40GB"
        assert kind("model.layers.0.self_attn.qkv_proj") == ("UnquantizedLinearMethod" if compiled() else "GlydLinearMethod")  # (the A/B on an A100: packed)
        torch.cuda.get_device_name = lambda: "NVIDIA L4"
        os.environ["GLYD_MOE_DENSE"] = "bf16"  # (the A/B the other way: another GPU's attention kept bf16)
        assert kind("model.layers.0.self_attn.qkv_proj") == ("GlydLinearMethod" if compiled() else "UnquantizedLinearMethod")
        assert kind("model.layers.0.mlp.shared_expert.gate_up_proj") == "GlydLinearMethod"
        c._vc = ns(model_config=ns(hf_config=ns(num_experts=0, get_text_config=lambda: ns(num_experts=0))))
        assert kind("model.layers.0.self_attn.qkv_proj") == "GlydLinearMethod"  # (a model with no routed experts: packed, whatever the variable says)
    finally:
        torch.cuda.is_available, torch.cuda.get_device_name = real
        os.environ.pop("GLYD_MOE_DENSE", None)
        if was is not None:
            os.environ["GLYD_MOE_DENSE"] = was
        c._vc = None


def _table(cls, K):
    """An embedding or an LM head as get_quant_method sees it while vLLM makes it: its class, hidden size, padded vocabulary and shard count."""
    t = object.__new__(cls)
    t.embedding_dim, t.num_embeddings_padded, t.tp_size = K, 151936, 1
    return t


def test_table_plan():
    """Which of the embedding and the LM head are packed, from the options and vLLM's config (table_plan): both at fraction 1; none under
    a fraction below 1, a draft, a speculative target (its draft shares them); exact keeps the LM head and with it a tied table (the
    embedding of an untied model is packed); a head_dtype that is not bf16 keeps the head (a tied table with it); the same answer for a
    tied model's two layers; and the bytes that come to (what the fit counts)."""
    if vp is None:
        return print("test_table_plan: skipped (no vLLM)")
    import torch

    ns = types.SimpleNamespace
    vc = lambda spec=None, hd=torch.bfloat16: ns(speculative_config=spec, model_config=ns(head_dtype=hd))
    c = vp.GlydConfig()

    def plan(fraction=1.0, exact=False, tied=False, flagged=False, draft=False, v=None):
        c.tied, c.untied_by_checkpoint = tied, flagged
        return c.table_plan(v or vc(), fraction, exact, draft)[:2]

    assert plan() == (True, True) and plan(tied=True) == (True, True)
    assert plan(fraction=0.5) == (False, False) and plan(fraction=0.0) == (False, False) and plan(draft=True) == (False, False)
    assert plan(v=vc(object())) == (False, False)  # (a speculative config)
    assert plan(exact=True) == (True, False) and plan(exact=True, tied=True) == (False, False) and plan(exact=True, flagged=True) == (False, False)
    assert plan(v=vc(hd=torch.float32)) == (True, False) and plan(v=vc(hd=torch.float32), tied=True) == (False, False)
    for tied in (False, True):  # (a tied model's two layers are one answer: packed alone is never the head's, nor the embedding's)
        for exact in (False, True):
            e, h = plan(exact=exact, tied=tied)
            assert not tied or e == h
    assert "exact" in c.table_plan(vc(), 1.0, True, False)[2] and "fraction 1" in c.table_plan(vc(), 0.5, False, False)[2] and c.table_plan(vc(), 1.0, False, False)[2] is None
    # the bytes: both tables where there are two (other counts both), the one where tied; exact's embedding alone is half of two
    assert vp._held(100, (True, True, None), False) == 100 and vp._held(100, (True, True, None), True) == 100
    assert vp._held(100, (True, False, "exact"), False) == 50 and vp._held(100, (False, False, "exact"), True) == 0 and vp._held(100, (False, False, "x"), False) == 0
    lin, other = 1000.0, 100.0
    full = vp._estimate(lin, other, "mma", 1, held=other)
    assert full[0] == (lin + other) * 10.80 / 16 and full[1] == (lin * 2 / 3 + other) * 10.80 / 16
    assert vp._estimate(lin, other, "mma", 1, held=50)[0] == (lin + 50) * 10.80 / 16 + 50 and vp._estimate(lin, other, "mma", 2, draft=True, held=other)[0] == lin * 10.80 / 16 / 2
    assert abs(vp._estimate(lin, other, "mma12", 1, fraction=0.5, held=0)[0] - (lin * 0.5 * 12.04 / 16 + lin * 0.5 + other)) < 1e-9


def test_embedding_ops():
    """On the GPU (else skipped): the lookup and the LM head's product over a table's pack, in both layouts: rows by id (a table of rows
    padded to 64 with zeros: a tensor-parallel shard's, a vocabulary's) bit for bit the table's, an id outside it a row of zeros, ids of
    any shape; the product for 1 to 1,000 rows (the library's own kernels, never the matrix decoded whole), the padding cut off, with and
    without a bias, within 1e-2 of F.linear on the table. And the method's steps on a bare layer: its weight padded to 64 rows and made
    on the meta device, packed once loaded, its bf16 gone; a head tied to its embedding holding the embedding's buffers (one pack), or,
    where the embedding stays as vLLM runs it, as vLLM ties it; where the checkpoint holds both tables of a tied model, the second to be
    packed taking the first's pack if the same bits, and its own if not."""
    if vp is None:
        return print("test_embedding_ops: skipped (no vLLM)")
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return print("test_embedding_ops: skipped (no GPU)")
    from glyd_gpu import kernels as g
    from vllm.model_executor.model_loader.reload.layerwise import _layerwise_process, get_layerwise_info

    try:
        g.lib()
        vp._define()
    except OSError as e:
        return print(f"test_embedding_ops: skipped ({e})")
    torch.manual_seed(0)
    dev = "cuda"
    parts = lambda p: ((p.data, p.exc, p.exc_base), list(p.sym)) if isinstance(p, g.Mma12) else ((p.data, p.blocks, p.block_base), list(p.tiers))
    for pack in (g.pack_mma, g.pack_mma12):
        for V, K in ((1000, 256), (448, 128), (50257, 128)):
            rows = -(-V // 64) * 64
            w = torch.zeros(rows, K, dtype=torch.bfloat16, device=dev)
            w[:V] = (torch.randn(V, K, device=dev) * 0.05).to(torch.bfloat16)
            t, words = parts(pack(w))
            ids = torch.cat([torch.randint(0, V, (300,), device=dev), torch.tensor([0, V - 1], device=dev)]).view(2, -1)
            y = torch.ops.glyd.vllm_embed(ids, *t, words, rows, K)
            assert tuple(y.shape) == (2, 151, K) and torch.equal(y.view(torch.int16), w[ids].view(torch.int16)), (pack.__name__, V, K)
            assert torch.equal(torch.ops.glyd.vllm_embed(ids.int(), *t, words, rows, K), y)  # (ids of another integer type)
            z = torch.ops.glyd.vllm_embed(torch.tensor([-1, rows, 3], device=dev), *t, words, rows, K)
            assert not z[:2].view(torch.int16).any() and torch.equal(z[2], w[3])  # (outside the table: zeros; nothing read)
            bias = torch.randn(V, dtype=torch.bfloat16, device=dev)
            for M in (1, 7, 64, 65, 700, 1000):
                x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
                for b in (None, bias):
                    y = torch.ops.glyd.vllm_logits(x, *t, words, b, rows, V)
                    ref = F.linear(x.float(), w[:V].float(), None if b is None else b.float())
                    assert tuple(y.shape) == (M, V) and ((y.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2, (pack.__name__, V, K, M, b is None)
            assert tuple(torch.ops.glyd.vllm_logits(x.view(2, M // 2, K), *t, words, None, rows, V).shape) == (2, M // 2, V)
    # the method's steps
    cfg = vp.GlydConfig()
    cfg.opts = {"layout": "mma", "exact": False, "verify": True, "fraction": 1.0, "kv": "off"}
    loader = lambda param, loaded: (param[: loaded.shape[0]].data.copy_(loaded), param[loaded.shape[0] :].data.fill_(0))  # (VocabParallelEmbedding.weight_loader's)
    V, K = 1000, 128

    def make(head, w=None):
        m = vp.GlydEmbeddingMethod(cfg, head, prefix="lm_head" if head else "model.embed_tokens")
        layer = torch.nn.Module()
        layer.embedding_dim, layer.num_embeddings_per_partition, layer.quant_method = K, V, m
        with torch.device(dev):
            m.create_weights(layer, K, [V], K, V, torch.bfloat16, weight_loader=loader)
            cfg.tables.append(vp.weakref.ref(layer))
            if w is not None:
                layer.weight.weight_loader(layer.weight, w)  # (recorded: fewer elements than the layer holds, so processed once told to)
                _layerwise_process(layer, get_layerwise_info(layer))
        return m, layer

    table = lambda: (torch.randn(V, K, device=dev) * 0.05).to(torch.bfloat16)
    w = table()
    m, e = make(False)
    assert e.weight.is_meta and tuple(e.weight.shape) == (1024, K) and m.rows == 1024 and m.uses_meta_device  # (padded to 64 rows, on the meta device)
    m, e = make(False, w)
    ids = torch.randint(0, V, (200,), device=dev)
    assert e.glyd_out == 1024 and e.glyd_verified and tuple(e.weight.shape) == (0, K) and e.glyd_words and not [n for n, _ in e.named_parameters() if n != "weight"]
    assert torch.equal(m.embedding(e, ids).view(torch.int16), w[ids].view(torch.int16))  # (its method's own lookup)
    assert tuple(m.apply(e, torch.randn(3, K, dtype=torch.bfloat16, device=dev)).shape) == (3, V)  # (the shard's rows, not the pack's)
    from glyd_gpu import kvcal

    nm = kvcal._norms_of(e)
    assert nm.shape == (V,) and torch.allclose(nm, w.float().norm(dim=1).cpu(), rtol=1e-5)  # (the packed table's row norms, which the KV cache's calibration sweeps the vocabulary by: the shard's rows, the weight's norms)
    assert vp._same_rows(e, torch.cat([w, torch.zeros(24, K, dtype=torch.bfloat16, device=dev)])) and not vp._same_rows(e, torch.cat([table(), torch.zeros(24, K, dtype=torch.bfloat16, device=dev)]))
    hm, head = make(True)
    assert hm.tie_weights(head, e) is head and hm.src is e and head.weight.numel() == 0  # (tied to Glyd's embedding)
    _layerwise_process(head, get_layerwise_info(head))
    assert all(getattr(head, n) is getattr(e, n) for n in ("glyd_data", "glyd_a", "glyd_b")) and head.glyd_words == e.glyd_words and head.glyd_out == e.glyd_out  # (one pack)
    assert torch.equal(hm.apply(head, torch.ones(2, K, dtype=torch.bfloat16, device=dev)), m.apply(e, torch.ones(2, K, dtype=torch.bfloat16, device=dev)))
    plain = torch.nn.Module()  # (an embedding vLLM runs as it does without Glyd: the head is its weight)
    plain.weight, plain.quant_method = torch.nn.Parameter(table(), requires_grad=False), vp.UnquantizedEmbeddingMethod()
    hm2, head2 = make(True)
    assert hm2.tie_weights(head2, plain) is head2 and isinstance(head2.quant_method, vp.UnquantizedEmbeddingMethod) and head2.weight is plain.weight
    # the checkpoint holds both tables of a tied model: the second to be packed takes the first's pack if the same bits
    cfg.untied_by_checkpoint, cfg.tables = True, []
    m1, a = make(True, w)
    m2, b = make(False, w.clone())
    w3 = table()
    m3, c = make(False, w3)
    assert b.glyd_data is a.glyd_data and b.glyd_out == a.glyd_out and b.glyd_words == a.glyd_words and tuple(b.weight.shape) == (0, K)
    assert c.glyd_data is not a.glyd_data and torch.equal(m3.embedding(c, ids).view(torch.int16), w3[ids].view(torch.int16))  # (another table: its own pack)
    cfg.untied_by_checkpoint = False
    m4, d = make(False, w.clone())
    assert d.glyd_data is not a.glyd_data  # (not for a model whose config does not tie them: they are compared only where vLLM built them apart)
    cfg.tables = []


def test_scratch_rule():
    """On the GPU (else skipped): the buffer matrices are decoded into for cuBLAS is made at load only where a CUDA graph can capture the decode (it keeps
    the buffer's address): exact's, with graphs; a product whose route decodes for a count of tokens up to vLLM's max_cudagraph_capture_size; every
    product that decodes where vLLM's config is not known; none without graphs. _unpack makes a buffer for the call where the device's does not hold
    the matrix, and decodes the same bits into either."""
    if vp is None:
        return print("test_scratch_rule: skipped (no vLLM)")
    import torch

    if not torch.cuda.is_available():
        return print("test_scratch_rule: skipped (no GPU)")
    from glyd_gpu import kernels as g

    try:
        g.lib()
    except OSError as e:
        return print(f"test_scratch_rule: skipped ({e})")
    ns = types.SimpleNamespace
    graphs = next(m for m in vp.CUDAGraphMode if m != vp.CUDAGraphMode.NONE)

    def method(exact=False, eager=False, mode=graphs, cap=512, top=8192, known=True):
        cfg = vp.GlydConfig()
        cfg.opts = {"layout": "mma", "exact": exact, "verify": False, "fraction": 1.0, "kv": "off"}
        cfg._vc = ns(scheduler_config=ns(max_num_batched_tokens=top), model_config=ns(enforce_eager=eager), compilation_config=ns(cudagraph_mode=mode, max_cudagraph_capture_size=cap)) if known else None
        return vp.GlydLinearMethod(cfg, None)

    torch.manual_seed(0)
    W = (torch.randn(256, 128, device="cuda") * 0.05).to(torch.bfloat16)
    packs = {"mma": g.pack_mma(W), "mma12": g.pack_mma12(W)}
    dev = packs["mma"].data.device
    for name, p in packs.items():
        words = list(p.sym if name == "mma12" else p.tiers)
        for O, K in ((24576, 4096), (4096, 4096), (256, 128)):
            any_ = method()._decodes(dev, words, O, K)
            assert method()._kept(dev, words, O, K, False) == method()._decodes(dev, words, O, K, 512), (name, O, K)  # (graphs to 512)
            assert method(cap=8192)._kept(dev, words, O, K, False) == any_ and method(cap=1)._kept(dev, words, O, K, False) == method()._decodes(dev, words, O, K, 1)
            assert method(known=False)._kept(dev, words, O, K, False) == any_ and method(known=False)._kept(dev, words, O, K, True)
            assert method()._kept(dev, words, O, K, True)  # (exact: every product's decode is in a graph)
            assert not method(eager=True)._kept(dev, words, O, K, True) and not method(eager=True, cap=8192)._kept(dev, words, O, K, False)  # (no graph)
            assert not method(mode=vp.CUDAGraphMode.NONE)._kept(dev, words, O, K, False) and not method(mode=None)._kept(dev, words, O, K, True)
        assert method()._decodes(dev, words, 4096, 4096, 512) <= method()._decodes(dev, words, 4096, 4096)  # (a cap only narrows)
        # _unpack: the device's buffer where it holds the matrix, else one made for the call: the same bits
        saved = dict(vp._SCRATCH)
        vp._SCRATCH.clear()
        try:
            a, b = (p.exc, p.exc_base) if name == "mma12" else (p.blocks, p.block_base)
            y = vp._unpack(p.data, a, b, words, 256, 128)
            assert torch.equal(y.view(torch.int16), W.view(torch.int16)) and not vp._SCRATCH  # (none made for it)
            vp._SCRATCH[dev] = torch.empty(256 * 128, dtype=torch.bfloat16, device=dev)
            y = vp._unpack(p.data, a, b, words, 256, 128)
            assert y.data_ptr() == vp._SCRATCH[dev].data_ptr() and torch.equal(y.view(torch.int16), W.view(torch.int16))
            vp._SCRATCH[dev] = torch.empty(64, dtype=torch.bfloat16, device=dev)  # (one that holds less)
            y = vp._unpack(p.data, a, b, words, 256, 128)
            assert y.data_ptr() != vp._SCRATCH[dev].data_ptr() and torch.equal(y.view(torch.int16), W.view(torch.int16)) and vp._SCRATCH[dev].numel() == 64
        finally:
            vp._SCRATCH.clear()
            vp._SCRATCH.update(saved)


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
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as d:  # a save with its tables packed (glyd-v5): refused, not loaded untested
        json.dump({"format": "glyd-v5", "packs": {}}, open(os.path.join(d, "glyd.json"), "w"))
        raises(lambda: vp.GlydConfig().maybe_update_config(d, hf_config=types.SimpleNamespace(architectures=["Qwen3ForCausalLM"])), "glyd-v5")


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
    assert vp._experts_missing([(e, s) for e in range(2) for s in ("w1", "w2")], 2, gated=False) == []  # (no gate: w13 is the up's alone, loaded as w1)


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
        vp._refusals(vc(), False, None, 0.0)  # (fraction 0, kv lossless alone: bf16 weights, vLLM's own products)
    finally:
        os.environ.pop("VLLM_BATCH_INVARIANT") if was is None else os.environ.update(VLLM_BATCH_INVARIANT=was)


def test_linear_bytes_2026():
    """The plugin's count is glyd run's, layer by layer (v0.29.0's counted every layer as attention and a gated MLP: 33.6 GiB for
    NVIDIA-Nemotron-3-Nano-4B, refused on a 24 GB card), and it reads Gemma 4's config, which raises on num_key_value_heads (a per-layer
    attribute of transformers' heterogeneous configs: v0.29.0 stopped there, glyd run and glyd serve alike)."""
    if vp is None:
        return print("test_linear_bytes_2026: skipped (no vLLM)")
    from glyd_gpu import preflight as pf

    nemo = types.SimpleNamespace(**{k: v for k, v in pf.ladder_model("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16").config.items() if k != "safetensors_bytes"})
    assert vp._linear_bytes(nemo)[0] == 6300299264  # (its checkpoint's Linears, from its safetensors headers)

    class Guarded:  # (a heterogeneous config as transformers 5.18 makes Gemma 4's: the global values, the full layers' own in overrides)
        def __init__(self, flat, overrides):
            vars(self).update(flat)
            vars(self)["_heterogeneity_spec"] = types.SimpleNamespace(per_layer_overrides=overrides)

        def __getattribute__(self, key):
            if key in ("num_key_value_heads", "head_dim"):
                raise RuntimeError(f"'{key}' is a per-layer attribute")
            return object.__getattribute__(self, key)

    g12 = dict(pf.ladder_model("google/gemma-4-12B-it").config)
    types_ = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(48)]
    flat = {k: v for k, v in g12.items() if k not in ("global_head_dim", "num_global_key_value_heads", "sliding_window_pattern", "safetensors_bytes")}
    cfg = Guarded({**flat, "layer_types": types_}, {i: {"head_dim": 512, "num_key_value_heads": 1} for i, t in enumerate(types_) if t == "full_attention"})
    assert vp._linear_bytes(cfg) == pf.linear_bytes(g12)[:2] + (False,) and vp._flat(cfg)["global_head_dim"] == 512


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
    import torch

    was = os.environ.pop("GLYD_MOE_DECODE_MIN", None)
    try:
        assert vp._moe_decode_min() == vp._moe_decode_min_here() > 0
        # by GPU: an H100's, another Hopper GPU's, any other's
        real = torch.cuda.is_available, torch.cuda.get_device_capability, torch.cuda.get_device_name
        try:
            torch.cuda.is_available = lambda: True
            for cap, name, want in (((9, 0), "NVIDIA H100 80GB HBM3", vp.MOE_DECODE_MIN_H100), ((9, 0), "NVIDIA H100 PCIe", vp.MOE_DECODE_MIN_H100), ((9, 0), "NVIDIA GH200 480GB", vp.MOE_DECODE_MIN_HOPPER),
                                    ((8, 0), "NVIDIA A100-SXM4-80GB", vp.MOE_DECODE_MIN_A100), ((8, 0), "NVIDIA A100-PCIE-40GB", vp.MOE_DECODE_MIN_A100), ((8, 0), "NVIDIA A800 80GB", vp.MOE_DECODE_MIN), ((8, 9), "NVIDIA L4", vp.MOE_DECODE_MIN)):
                torch.cuda.get_device_capability, torch.cuda.get_device_name = (lambda c=cap: c), (lambda n=name: n)
                assert vp._moe_decode_min_here() == want, (name, vp._moe_decode_min_here(), want)
        finally:
            torch.cuda.is_available, torch.cuda.get_device_capability, torch.cuda.get_device_name = real
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


def _gpu_lib(name):
    """The GPU and the library, for a test's GPU part: (torch, kernels) or None, with a line saying why it is skipped."""
    import torch

    if not torch.cuda.is_available():
        return print(f"{name}: its GPU part skipped (no GPU)")
    from glyd_gpu import kernels as g

    try:
        g.lib()
        vp._define()
    except OSError as e:
        return print(f"{name}: its GPU part skipped ({e})")
    return torch, g


def test_moe_activations():
    """A mixture of experts' activation (vLLM's MoEActivation and the clamp, alpha and beta its config carries): the grouped product
    applies SiLU (1) or GELU (tanh) (2) itself only where vLLM's are plain, and every other activation vLLM's MoE layers run (a clamped
    SwiGLU: Hy4, GLM-5.3-Flash; SwiGLU-OAI: MiniMax-M3; swiglustep: Step 3.5; GELU (tanh): Gemma 4; ReLU squared without a gate:
    Nemotron-H) goes through vLLM's own activation op (0): a clamp is never dropped. Those layers are packed once measured (a development
    build's GLYD_MOE_ACTS=all; test_moe_unsupported), an activation vLLM's op does not run on its own never; a layer without a gate needs
    its experts' w1 and w2 alone. On the GPU (else skipped): the
    grouped products with each activation against the experts in float32 and vLLM's own activation, within 1e-2."""
    if vp is None:
        return print("test_moe_activations: skipped (no vLLM)")
    import torch

    if vp.RoutedExperts is None:
        raise AssertionError(f"vLLM's mixture of experts internals did not import: {vp._NO_MOE}")
    A, C = vp.MoEActivation, vp.ApplyMoEActivationConfig
    ns = types.SimpleNamespace
    cfg_of = lambda limit=None, alpha=None, beta=None: C.from_configs(ns(swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta, activation_situ_beta=None, activation_situ_linear_beta=None), vp.FusedMoEQuantConfig.make())
    fused = lambda a, I=768, limit=None, alpha=None, beta=None: vp._fused_act(ns(activation=a, intermediate_size_per_partition=I, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta))
    assert fused(A.SILU) == 1 and fused(A.GELU_TANH) == 2
    assert fused(A.SILU, limit=10.0) == 0 and fused(A.SILU, alpha=1.702) == 0 and fused(A.SILU, beta=1.0) == 0  # (the clamp: Hy4's, GLM-5.3-Flash's)
    assert fused(A.GELU_TANH, limit=7.0) == 0
    for a in (A.GELU, A.SWIGLUOAI, A.SWIGLUOAI_UNINTERLEAVE, A.SWIGLUSTEP, A.RELU2_NO_MUL, A.SILU_NO_MUL, A.GELU_TANH_NO_MUL):
        assert fused(a, limit=7.0 if a == A.SWIGLUOAI_UNINTERLEAVE else None) == 0, a

    def layer(act=A.SILU, H=2048, I=768, ep=False, bias=False, on_input=False, dtype=torch.bfloat16):
        mc = ns(moe_parallel_config=ns(use_ep=ep), has_bias=bias, activation=act, hidden_dim=H, intermediate_size_per_partition=I, hidden_dim_unpadded=H, intermediate_size_per_partition_unpadded=I)
        return ns(moe_config=mc, apply_router_weight_on_input=on_input, params_dtype=dtype)

    assert vp.GlydMoEMethod.unsupported(layer()) is None
    if not compiled():  # (the development build's switch, as their checks set it: every one of them taken)
        was, os.environ["GLYD_MOE_ACTS"] = os.environ.get("GLYD_MOE_ACTS"), "all"
        try:
            for a, I in ((A.SILU, 768), (A.GELU_TANH, 704), (A.RELU2_NO_MUL, 1856), (A.SWIGLUOAI_UNINTERLEAVE, 1536), (A.SWIGLUSTEP, 1280), (A.SWIGLUOAI, 512)):
                assert vp.GlydMoEMethod.unsupported(layer(a, I=I)) is None, (a, vp.GlydMoEMethod.unsupported(layer(a, I=I)))  # (Gemma 4 26B-A4B's 704, Nemotron-3-Nano's 1856)
            assert "not a size" in vp.GlydMoEMethod.unsupported(layer(I=100)) and "not a size" in vp.GlydMoEMethod.unsupported(layer(A.RELU2_NO_MUL, I=96))
        finally:
            os.environ.pop("GLYD_MOE_ACTS", None)
            if was is not None:
                os.environ["GLYD_MOE_ACTS"] = was
    assert "relu2" in vp.GlydMoEMethod.unsupported(layer(A.RELU2))  # (a gated ReLU squared: vLLM's op has none on its own)
    assert vp.GlydMoEMethod.unsupported(layer(ep=True)) == "expert parallel" and "biases" in vp.GlydMoEMethod.unsupported(layer(bias=True))
    assert "input" in vp.GlydMoEMethod.unsupported(layer(on_input=True)) and "float16" in vp.GlydMoEMethod.unsupported(layer(dtype=torch.float16))
    # a layer without a gate: w13 is w1 alone
    nogate = [(e, s) for e in range(3) for s in ("w1", "w2")]
    assert vp._experts_missing(nogate, 3, gated=False) == [] and vp._experts_missing(nogate[:-1], 3, gated=False) == ["expert 2's w2"]
    assert vp._experts_missing(nogate, 3) == [f"expert {e}'s w3" for e in range(3)]  # (gated: its up is missing)
    # on the GPU: the grouped products with each activation against float32
    have = _gpu_lib("test_moe_activations")
    if have is None:
        return
    torch, g = have
    torch.manual_seed(0)
    dev, E, k, H, I = "cuda", 8, 2, 256, 192
    cases = ((A.SILU, {}), (A.GELU_TANH, {}), (A.SILU, {"limit": 1.0}), (A.SWIGLUOAI_UNINTERLEAVE, {"limit": 1.0, "alpha": 1.702, "beta": 1.0}),
             (A.SWIGLUSTEP, {}), (A.RELU2_NO_MUL, {}), (A.GELU_TANH_NO_MUL, {}), (A.SWIGLUOAI, {}))
    for pack in (g.pack_mma, g.pack_mma12):
        for a, kw in cases:
            ac = cfg_of(**kw)
            O13 = 2 * I if a.is_gated else I
            W1 = (torch.randn(E * O13, H, device=dev) * 0.1).to(torch.bfloat16)  # (sums of about 1.6: past a limit of 1 often)
            W2 = (torch.randn(E * H, I, device=dev) * 0.05).to(torch.bfloat16)
            up, down = pack(W1), pack(W2)
            for T in (1, 7, 64, 300):
                x = torch.randn(T, H, device=dev).to(torch.bfloat16)
                ids = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(T)])
                w = torch.softmax(torch.randn(T, k, device=dev), dim=-1)
                act = fused(a, I, **kw)
                y = vp._experts(up, down, E, x, g.moe_way(ids, E), ids, w, act, a, ac)
                gu = torch.einsum("tkoh,th->tko", W1.float().view(E, O13, H)[ids], x.float())  # [T, k, O13]
                if act:
                    gate, u = gu[..., :I], gu[..., I:]
                    h = ((torch.nn.functional.silu(gate) if act == 1 else torch.nn.functional.gelu(gate, approximate="tanh")) * u).to(torch.bfloat16)
                else:
                    flat = gu.to(torch.bfloat16).reshape(T * k, O13).contiguous()
                    h = flat.new_empty(T * k, O13 // 2 if a.is_gated else O13)
                    vp.apply_moe_activation(a, h, flat.clone(), activation_config=ac)
                    h = h.view(T, k, -1)
                ref = torch.einsum("tkhi,tki->tkh", W2.float().view(E, H, I)[ids], h.float()).mul(w[..., None]).sum(1)
                err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                assert tuple(y.shape) == (T, H) and err < 1e-2, (pack.__name__, a, act, T, err)
                assert torch.equal(y, vp._experts(up, down, E, x, g.moe_way(ids, E), ids, w, act, a, ac))  # (the same bits again)
            if a == A.SILU and ac.clamp_limit is not None:  # (the clamp matters here: plain SiLU's sums are not the clamped ones)
                plain = vp._experts(up, down, E, x, g.moe_way(ids, E), ids, w, 1, a, cfg_of())
                assert not torch.equal(plain, y)


def test_padded_rows():
    """A Linear whose rows are not a multiple of 64 (a SigLIP MLP's 4,304, a Mamba2 in_proj's 17,504) is packed: its weight made on the meta
    device as any other's ([O, K]: vLLM's loaders write it as they do), the pack's rows padded to 64 with zeros; columns not a multiple of 16
    stay vLLM's. On the GPU (else skipped): the op for 1 to 2,000 tokens (fused, the matrix decoded, exact) returns [M, O] within 1e-2 of
    F.linear, with and without a bias; the pack decodes to the weight and zero rows."""
    if vp is None:
        return print("test_padded_rows: skipped (no vLLM)")
    import torch

    assert [vp._padded(n) for n in (1, 63, 64, 65, 4304, 17504, 288)] == [64, 64, 64, 128, 4352, 17536, 320]
    from vllm.model_executor import parameter as P

    cfg = vp.GlydConfig()
    cfg.opts = {"layout": "mma", "exact": False, "verify": True, "fraction": 1.0, "kv": "off"}
    tp = P.get_tensor_model_parallel_rank, P.get_tensor_model_parallel_world_size
    P.get_tensor_model_parallel_rank, P.get_tensor_model_parallel_world_size = (lambda: 0), (lambda: 1)  # (no distributed state here: one rank)
    try:
        for O, K, plain in ((4304, 1152, False), (17504, 3136, False), (4096, 100, True)):
            m = vp.GlydLinearMethod(cfg, None)
            lay = torch.nn.Module()
            with torch.device("cpu"):
                m.create_weights(lay, K, [O], K, O, torch.bfloat16, weight_loader=lambda param, loaded: None)
            assert (m.plain is not None) == plain and tuple(lay.weight.shape) == (O, K) and lay.weight.is_meta == (not plain), (O, K)
    finally:
        P.get_tensor_model_parallel_rank, P.get_tensor_model_parallel_world_size = tp
    have = _gpu_lib("test_padded_rows")
    if have is None:
        return
    torch, g = have
    import torch.nn.functional as F

    torch.manual_seed(0)
    was = list(vp._BF16)
    vp._BF16[:] = [types.SimpleNamespace(apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias))]  # (exact's GEMM: F.linear here, vLLM's config not being current)
    try:
        _padded_ops(torch, g, F)
    finally:
        vp._BF16[:] = was


def _padded_ops(torch, g, F):
    for pack in (g.pack_smallest, g.pack_mma12):
        for O, K in ((200, 128), (4304, 1152), (288, 256)):
            w = (torch.randn(O, K, device="cuda") * 0.05).to(torch.bfloat16)
            p = vp._pack_rows(w, pack, True, "w")
            rows = vp._padded(O)
            assert p.shape == (rows, K) and torch.equal(g.mma_unpack(p)[:O].view(torch.int16), w.view(torch.int16)) and not g.mma_unpack(p)[O:].view(torch.int16).any()
            t = (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base)
            words = list(p.sym if isinstance(p, g.Mma12) else p.tiers)
            bias = torch.randn(O, device="cuda").to(torch.bfloat16)
            for M in (1, 7, 64, 300, 2000):
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                for b in (None, bias):
                    y = torch.ops.glyd.vllm_linear(x, *t, words, b, O, False)
                    ref = F.linear(x.float(), w.float(), None if b is None else b.float())
                    assert tuple(y.shape) == (M, O) and y.is_contiguous() and ((y.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2, (pack.__name__, O, K, M, b is None)
                    ex = torch.ops.glyd.vllm_linear(x, *t, words, b, O, True)  # (exact: the GEMM on the decoded matrix, the Linear's rows)
                    assert torch.equal(ex, F.linear(x, w, b))
            assert tuple(torch.ops.glyd.vllm_linear(torch.randn(2, 3, K, device="cuda").to(torch.bfloat16), *t, words, None, O, False).shape) == (2, 3, O)


def _modules(**kids):
    """A bare module holding kids by name (a model's tree, for _after_load)."""
    import torch

    m = torch.nn.Module()
    for n, c in kids.items():
        setattr(m, n, c)
    return m


def test_after_load():
    """What vLLM builds without the quant config, packed once the model has loaded (_after_load): its embedding and LM head on vLLM's own method
    where a table given the config would be (a tied pair one pack; a position table of few rows, a fraction below 1, exact's tied table: kept),
    the Linears of Jamba's Mamba mixers (not a router, not one under 1M weights, not one Glyd itself left bf16), and a tower's (LinearBase and
    plain nn.Linear) only where _towers_on says so (a release build: never yet), never a plain one under exact; nothing in a model without
    Glyd's config. And the hook on vLLM's loader: its load_model reads the step by name, so the hook takes; installed once."""
    if vp is None:
        return print("test_after_load: skipped (no vLLM)")
    import torch

    nn = torch.nn
    cfg = vp.GlydConfig()

    def table(cls, V, K, w=None):
        t = object.__new__(cls)
        nn.Module.__init__(t)
        t.embedding_dim, t.num_embeddings, t.num_embeddings_padded, t.tp_size = K, V, V, 1
        t.quant_method, t.use_fused_embedding = vp.UnquantizedEmbeddingMethod(), False
        t.weight = w if w is not None else nn.Parameter(torch.zeros(V, K, dtype=torch.bfloat16), requires_grad=False)
        return t

    def linear(O, K, qc=None, bias=False, method=None):
        m = object.__new__(vp.LinearBase)
        nn.Module.__init__(m)
        m.quant_config, m.quant_method, m.skip_bias_add = qc, method or vp.UnquantizedLinearMethod(), False
        m.weight = nn.Parameter(torch.zeros(O, K, dtype=torch.bfloat16), requires_grad=False)
        m.bias = nn.Parameter(torch.zeros(O, dtype=torch.bfloat16), requires_grad=False) if bias else None
        return m

    mixer = type("MambaMixer", (nn.Module,), {})()
    mixer.in_proj, mixer.x_proj, mixer.dt_proj = linear(4096, 1024), linear(160, 2048), linear(2048, 512, bias=True)
    mixer.dt_proj.skip_bias_add = True
    router = type("NemotronHMoE", (nn.Module,), {})()
    router.gate = linear(512, 4096)  # (a router of 2M weights, built without the config: stays)
    emb = table(vp.VocabParallelEmbedding, 65536, 1024)
    head = table(vp.ParallelLMHead, 65536, 1024, emb.weight)  # (tied: one weight)
    layer0 = _modules(mamba=mixer, moe=router, mlp=_modules(down_proj=linear(1024, 4096, qc=cfg)))  # (Glyd's own bf16, a fraction's: not taken)
    layer1 = _modules(mlp=_modules(down_proj=linear(1024, 4096, qc=cfg, method=vp.GlydLinearMethod(cfg))))
    lm = _modules(model=_modules(embed_tokens=emb, layers=nn.ModuleList([layer0, layer1])), lm_head=head)
    fc1 = linear(4304, 1152, bias=True)  # (a SigLIP MLP vLLM gives no config)
    plain = nn.Linear(1152, 1152, dtype=torch.bfloat16)
    tower = _modules(blocks=nn.ModuleList([_modules(fc1=fc1, proj=plain)]), pos_embed=table(vp.VocabParallelEmbedding, 1024, 1152))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model, self.visual = lm, tower

        def get_language_model(self):
            return self.language_model

    model = Model()
    calls = []
    saved = (vp._bf16_matrix, vp.GlydEmbeddingMethod.pack_loaded, vp.GlydEmbeddingMethod.share, vp.GlydEmbeddingMethod.adopted, vp.GlydLinearMethod.pack_loaded)
    was = os.environ.pop("GLYD_PACK_TOWERS", None)
    try:
        vp._bf16_matrix = lambda w, cols=16: isinstance(w, torch.Tensor) and w.dim() == 2 and w.numel() > 0 and w.dtype == torch.bfloat16 and w.shape[1] % cols == 0  # (on the CPU)
        vp.GlydEmbeddingMethod.pack_loaded = lambda self, layer, name: calls.append(("table", name))
        vp.GlydEmbeddingMethod.share = lambda self, layer, src, name: calls.append(("share", name, src is emb))
        vp.GlydEmbeddingMethod.adopted = lambda self, layer: None
        vp.GlydLinearMethod.pack_loaded = lambda self, layer, name: calls.append(("linear", name))

        def run(fraction=1.0, exact=False, plan=(True, True, None), towers=None):
            cfg.opts = {"layout": "mma", "exact": exact, "verify": False, "fraction": fraction, "kv": "off"}
            cfg.plan, cfg.compiled, cfg.noted = plan, False, set()
            os.environ.pop("GLYD_PACK_TOWERS", None) if towers is None else os.environ.__setitem__("GLYD_PACK_TOWERS", towers)
            for m in model.modules():  # (as loaded: every one on vLLM's own method again)
                if isinstance(m, vp.LinearBase) and m.quant_config is None:
                    m.quant_method = vp.UnquantizedLinearMethod()
                if type(m) is nn.Linear:
                    m.__dict__.pop("forward", None)
            calls.clear()
            vp._after_load(model)
            return sorted(calls)

        mix = [("linear", f"language_model.model.layers.0.mamba.{n}") for n in ("dt_proj", "in_proj")]  # (x_proj: 0.3M weights, stays)
        tables = [("share", "language_model.lm_head", True), ("table", "language_model.model.embed_tokens")]
        assert run() == sorted(tables + mix), run()
        assert any("visual.pos_embed" in n and "too few" in n for n in cfg.noted)  # (a position table: kept, with why)
        assert isinstance(fc1.quant_method, vp.UnquantizedLinearMethod) and "forward" not in plain.__dict__
        towers = [("linear", "visual.blocks.0.fc1"), ("linear", "visual.blocks.0.proj")]
        assert run(towers="1") == sorted(tables + mix + ([] if compiled() else towers))  # (a release build reads no tuning variable: towers not yet)
        if not compiled():
            assert "forward" in plain.__dict__ and isinstance(fc1.quant_method, vp.GlydLinearMethod)
            assert run(towers="1", exact=True, plan=(False, False, "exact")) == sorted(mix + towers[:1])  # (exact: the tied table kept, the plain Linear kept)
        lm.model.replicated_embed = True  # (a model that gathers from the bf16 table itself: the table and the head tied to it stay)
        assert run() == sorted(mix) and any("gathers its rows" in n for n in cfg.noted)
        lm.model.replicated_embed = False
        assert run(fraction=0.5, plan=(False, False, "packed at fraction 1 only")) == []  # (layer 0 is not a fraction 0.5's; the tables stay)
        assert run(fraction=0.0, plan=(False, False, "packed at fraction 1 only")) == []
        other = _modules(lin=linear(4096, 4096), emb=table(vp.VocabParallelEmbedding, 65536, 1024))
        calls.clear()
        vp._after_load(other)  # (no layer with Glyd's config: not a glyd model)
        assert calls == []
    finally:
        vp._bf16_matrix, vp.GlydEmbeddingMethod.pack_loaded, vp.GlydEmbeddingMethod.share, vp.GlydEmbeddingMethod.adopted, vp.GlydLinearMethod.pack_loaded = saved
        os.environ.pop("GLYD_PACK_TOWERS", None)
        if was is not None:
            os.environ["GLYD_PACK_TOWERS"] = was
    # the hook on vLLM's loader
    from vllm.model_executor.model_loader import base_loader

    import inspect

    assert "process_weights_after_loading" in inspect.unwrap(base_loader.BaseModelLoader.load_model).__code__.co_names  # (read from the module at each load: the hook takes)
    vp._hook_load()
    hooked = base_loader.process_weights_after_loading
    vp._hook_load()
    assert hooked.glyd and base_loader.process_weights_after_loading is hooked  # (once)
    hooked(nn.Module(), types.SimpleNamespace(word_embeddings_untied_by_checkpoint=False, dtype=torch.bfloat16, quantization="glyd"), torch.device("cpu"))  # (vLLM's own step, then nothing to pack)
    # on the GPU: a table and its tied head packed after the load, one pack, the rows bit for bit; a Linear's product
    have = _gpu_lib("test_after_load")
    if have is None:
        return
    torch, g = have
    torch.manual_seed(0)
    cfg.opts, cfg.plan, cfg.tables, cfg.compiled = {"layout": "mma", "exact": False, "verify": True, "fraction": 1.0, "kv": "off"}, (True, True, None), [], False
    with torch.device("cuda"):
        V, K = 5000, 256
        w = (torch.randn(V, K) * 0.05).to(torch.bfloat16)
        emb = table(vp.VocabParallelEmbedding, V, K, nn.Parameter(w.clone(), requires_grad=False))
        head = table(vp.ParallelLMHead, V, K, emb.weight)
        head.num_embeddings_per_partition = emb.num_embeddings_per_partition = V
        mixer2 = type("MambaMixer", (nn.Module,), {})()
        mixer2.in_proj = lin = linear(6000, 256)  # (1.5M weights; 6,000 rows: padded to 6,016)
        lw = (torch.randn(6000, 256) * 0.05).to(torch.bfloat16)
        lin.weight.data.copy_(lw)
        gm = _modules(model=_modules(embed_tokens=emb, layers=nn.ModuleList([_modules(mamba=mixer2, mlp=_modules(o=linear(64, 64, qc=cfg)))])), lm_head=head)
    vp._after_load(gm)
    assert isinstance(emb.quant_method, vp.GlydEmbeddingMethod) and isinstance(head.quant_method, vp.GlydEmbeddingMethod) and head.glyd_data is emb.glyd_data and emb.glyd_verified
    ids = torch.randint(0, V, (300,), device="cuda")
    assert torch.equal(emb(ids).view(torch.int16), w[ids].view(torch.int16)) and emb.weight.numel() == 0 and head.weight.numel() == 0
    assert tuple(head.quant_method.apply(head, torch.randn(3, K, device="cuda").to(torch.bfloat16)).shape) == (3, V)
    assert isinstance(lin.quant_method, vp.GlydLinearMethod) and lin.glyd_out == 6000 and lin.glyd_verified and lin.weight.numel() == 0
    x = torch.randn(9, 256, device="cuda").to(torch.bfloat16)
    y, ref = lin.quant_method.apply(lin, x), x.float() @ lw.float().t()
    assert tuple(y.shape) == (9, 6000) and ((y.float() - ref).abs().max() / ref.abs().max()).item() < 1e-2


def test_ple_plain():
    """Qwen4Exp's n-gram table: vLLM refuses any quantization config but FP8's for it (Qwen3.8-Flash-Next did not load with --quantization glyd);
    under Glyd's it takes vLLM's own bf16 method, another config as before; nothing where the module is not loaded; once."""
    if vp is None:
        return print("test_ple_plain: skipped (no vLLM)")
    name = "vllm.models.qwen4_exp.nvidia.ngram_embedding"
    real = sys.modules.get(name)
    if real is None:
        try:
            import importlib

            real = importlib.import_module(name)
        except Exception as e:  # (its imports need what this machine lacks: the stand-in below checks the rest)
            print(f"test_ple_plain: vLLM's module did not import ({type(e).__name__}: {e}); a stand-in")
    if real is not None:
        cls = real.Qwen4ExpPLEEmbeddingMethod
        own = cls.__dict__["from_quant_config"]
        try:
            raises_ = False
            try:
                cls.from_quant_config(vp.GlydConfig(), "model.layers.2.ngram.ngram_embedding")
            except NotImplementedError:
                raises_ = True
            assert raises_ or getattr(cls.from_quant_config, "glyd", False)  # (vLLM's own: refused)
            vp._ple_plain()
            m = cls.from_quant_config(vp.GlydConfig(), "model.layers.2.ngram.ngram_embedding")
            assert type(m).__name__ == "Qwen4ExpPLEUnquantizedEmbeddingMethod" and cls.from_quant_config(None, "x").__class__ is m.__class__
        finally:
            cls.from_quant_config = own
    stand = types.ModuleType(name)

    class Method:
        @staticmethod
        def from_quant_config(quant_config, prefix, embedding_dtype=None):
            if quant_config is not None:
                raise NotImplementedError("Qwen4Exp PLE embedding does not support quantization config")
            return "bf16"

    stand.Qwen4ExpPLEEmbeddingMethod = Method
    sys.modules[name] = stand
    try:
        vp._ple_plain()
        vp._ple_plain()
        assert Method.from_quant_config(vp.GlydConfig(), "p") == "bf16" and Method.from_quant_config(None, "p", "bfloat16") == "bf16"
        try:
            Method.from_quant_config(object(), "p")
            raise AssertionError("another config not refused")
        except NotImplementedError:
            pass
    finally:
        if real is not None:
            sys.modules[name] = real
        else:
            sys.modules.pop(name, None)


def test_moe_unsupported():
    """The experts the grouped products take: plain SiLU; any other activation vLLM's experts apply between their two
    products (GLM-5.3-Flash's and Hy4's SiLU with a clamp of 10, Gemma 4's GELU tanh, Nemotron-H's relu2 without a gate,
    MiniMax-M3's swigluoai, Step's swiglustep) only once measured (a development build's switch; a clamped SiLU in exact
    mode too), the gate and up rows a multiple of 64 (I alone without a gate); the product's own activation (mma_moe's
    act) for SiLU and GELU tanh with no clamp, else 0 (vLLM's activation on the sums)."""
    if vp is None or vp.RoutedExperts is None:
        return print("test_moe_unsupported: skipped (no vLLM)")
    import torch

    A, ns = vp.MoEActivation, types.SimpleNamespace
    cfg = lambda **c: ns(**{"moe_parallel_config": ns(use_ep=False), "has_bias": False, "activation": A.SILU, "hidden_dim": 4096, "intermediate_size_per_partition": 2048, "swiglu_limit": None, "swiglu_alpha": None, "swiglu_beta": None, **c})
    layer = lambda **c: ns(moe_config=cfg(**c), apply_router_weight_on_input=False, params_dtype=torch.bfloat16)
    u = vp.GlydMoEMethod.unsupported
    others = ({"swiglu_limit": 10.0}, {"activation": A.GELU_TANH, "hidden_dim": 2816, "intermediate_size_per_partition": 704}, {"activation": A.RELU2_NO_MUL, "hidden_dim": 2688, "intermediate_size_per_partition": 1856},
              {"activation": A.SWIGLUOAI_UNINTERLEAVE, "hidden_dim": 6144, "intermediate_size_per_partition": 3072, "swiglu_limit": 7.0, "swiglu_alpha": 1.702, "swiglu_beta": 1.0}, {"activation": A.SWIGLUSTEP, "intermediate_size_per_partition": 1280})
    was = os.environ.pop("GLYD_MOE_ACTS", None)
    try:
        assert u(layer()) is None and "not measured" in u(layer(intermediate_size_per_partition=1824))  # (2I a multiple of 64, I not: the product's sums, then vLLM's SiLU, not measured)
        for c in others:  # (not measured yet: kept bf16, but a clamped SiLU in exact mode)
            assert "not measured" in u(layer(**c)), c
        assert "limit 10.0" in u(layer(swiglu_limit=10.0)) and u(layer(swiglu_limit=10.0), True) is None and "not measured" in u(layer(activation=A.GELU_TANH), True)
        if not compiled():  # (the development build's switch: every one of them taken)
            os.environ["GLYD_MOE_ACTS"] = "all"
            for c in others + ({"intermediate_size_per_partition": 1824},):
                assert u(layer(**c)) is None, c
            assert "not a size" in u(layer(activation=A.RELU2_NO_MUL, intermediate_size_per_partition=1824)) and "not a size" in u(layer(intermediate_size_per_partition=1800))
    finally:
        os.environ.pop("GLYD_MOE_ACTS", None)
        if was is not None:
            os.environ["GLYD_MOE_ACTS"] = was
    assert "expert parallel" in u(layer(moe_parallel_config=ns(use_ep=True))) and "biases" in u(layer(has_bias=True))
    act = lambda **c: vp._fused_act(cfg(**c))
    assert act() == 1 and act(activation=A.GELU_TANH) == 2 and act(swiglu_limit=10.0) == 0 and act(swiglu_beta=1.0) == 0 and act(intermediate_size_per_partition=1824) == 0
    assert act(activation=A.RELU2_NO_MUL) == act(activation=A.SWIGLUSTEP) == act(activation=A.SWIGLUOAI_UNINTERLEAVE, swiglu_limit=7.0) == 0


def test_trim_and_drop():
    """The load's allocator trim: PyTorch's unused blocks handed back to the driver only where they outweigh the
    driver's free memory; a packed bf16 weight's memory dropped at once."""
    if vp is None:
        return print("test_trim_and_drop: skipped (no vLLM)")
    import torch

    flushed = []
    cuda = torch.cuda
    saved = (cuda.memory_reserved, cuda.memory_allocated, cuda.mem_get_info, cuda.empty_cache)
    try:
        cuda.memory_allocated = lambda *a: 100
        cuda.empty_cache = lambda: flushed.append(1)
        for reserved, free, trimmed in ((100, 50, False), (150, 100, False), (200, 100, False), (201, 100, True), (900, 100, True), (900, 5000, False)):
            cuda.memory_reserved = lambda *a, r=reserved: r
            cuda.mem_get_info = lambda *a, f=free: (f, 10_000)
            flushed.clear()
            vp._trim()
            assert bool(flushed) == trimmed, (reserved, free)  # (cached = reserved - 100: over the free memory only)
    finally:
        cuda.memory_reserved, cuda.memory_allocated, cuda.mem_get_info, cuda.empty_cache = saved
    w = torch.nn.Parameter(torch.ones(4, 8), requires_grad=False)
    vp._drop(w)
    assert w.numel() == 0
    vp._drop(None)


def test_no_nvcc():
    """One warning where vLLM's FlashInfer sampler has no nvcc to compile with, naming the fix; none where the sampler
    is off, where nvcc is found (PATH, CUDA_HOME, /usr/local/cuda), where FlashInfer's precompiled kernels are
    installed or FlashInfer is not; one a process."""
    if vp is None:
        return print("test_no_nvcc: skipped (no vLLM)")
    import importlib.util
    import tempfile

    msgs = []
    warning, find_spec, which, isfile = vp.log.warning, importlib.util.find_spec, vp.shutil.which, os.path.isfile
    os.path.isfile = lambda p: False if p == "/usr/local/cuda/bin/nvcc" else isfile(p)  # (a machine with a toolkit there is not the case being tested)
    env = {k: os.environ.pop(k, None) for k in ("VLLM_USE_FLASHINFER_SAMPLER", "CUDA_HOME")}
    try:
        vp.log.warning = lambda m, *a: msgs.append(m % a if a else m)

        def run(sampler=None, flashinfer=True, jit_cache=False, nvcc=None, cuda_home=None):
            importlib.util.find_spec = lambda name, *a: object() if (name == "flashinfer" and flashinfer) or (name == "flashinfer_jit_cache" and jit_cache) else None
            vp.shutil.which = lambda name: nvcc
            for k, v in (("VLLM_USE_FLASHINFER_SAMPLER", sampler), ("CUDA_HOME", cuda_home)):
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            vp._NVCC_WARNED.clear()
            msgs.clear()
            vp._warn_no_nvcc()
            return list(msgs)

        got = run()
        assert len(got) == 1 and "no nvcc" in got[0] and "VLLM_USE_FLASHINFER_SAMPLER=0" in got[0], got
        vp._warn_no_nvcc()
        assert len(msgs) == 1  # (once a process)
        assert run(sampler="0") == [] and run(sampler="1") != []
        assert run(nvcc="/usr/bin/nvcc") == [] and run(jit_cache=True) == [] and run(flashinfer=False) == []
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "bin"))
            assert run(cuda_home=d) != []  # (CUDA_HOME with no nvcc in it)
            open(os.path.join(d, "bin", "nvcc"), "w").close()
            assert run(cuda_home=d) == []
    finally:
        vp.log.warning, importlib.util.find_spec, vp.shutil.which, os.path.isfile = warning, find_spec, which, isfile
        for k, v in env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        vp._NVCC_WARNED.clear()


def test_kv_page():
    """The lossless KV cache's page: its bytes as the plugin says them (16 tokens of H KV heads, an arena of A entries: the headers and the
    arena up to 128 bytes, then 6,144 a head) and as the library does; what vLLM's allocator sees of it (uint8 cells), and the capacity that is."""
    if vp is None:
        return print("test_kv_page: skipped (no vLLM)")
    import torch
    from glyd_gpu import vllm_kv as kv
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    assert kv.leaf_bytes(8, 15) == 49408 and kv.cell_bytes(8, 15) == 386 and kv.leaf_bytes(8, 0) == 49408  # (Qwen3-8B: bf16's page is 65,536; a header of 256 bytes holds an arena of 15 entries)
    assert kv.leaf_bytes(8, 47) == 49536 and kv.cell_bytes(8, 47) == 387 and kv.leaf_bytes(8, 111) == 49792 and kv.cell_bytes(8, 111) == 389 and kv.leaf_bytes(8, 175) == 50048 and kv.leaf_bytes(8, 143) == 49920 and kv.leaf_bytes(8, 79) == 49664
    assert [kv.leaf_bytes(h, 31) for h in (1, 2, 4)] == [6400, 12544, 24832] and [kv.cell_bytes(h, 31) for h in (1, 2, 4)] == [400, 392, 388]
    assert [kv.free_bay(h) for h in (1, 2, 3, 4, 8, 16, 32)] == [11, 7, 3, 31, 15, 15, 15] and all(kv.leaf_bytes(h, kv.free_bay(h)) == kv.leaf_bytes(h, 0) < kv.leaf_bytes(h, kv.free_bay(h) + 1) for h in (1, 2, 3, 4, 5, 8, 16, 32))
    for h in (1, 2, 3, 4, 5, 8, 16):
        for a in (0, 15, 31, 63, 127, 255):
            assert kv.cell_bytes(h, a) * 16 * h >= kv.leaf_bytes(h, a) > kv.cell_bytes(h, a) * 16 * h - 16 * h
    try:
        from glyd_gpu import _lib, kernels

        kernels.lib()
        for h in (1, 2, 3, 4, 8, 32):
            for a in (0, 1, 15, 31, 63, 127, 1023):
                assert _lib.kv_leaf_bytes(h, 128, a) == kv.leaf_bytes(h, a), (h, a)
            assert _lib.kv_free_bay(h) == kv.free_bay(h)
        esc = lambda n: ((n + 31) // 32 + 3) // 4 * 16 + (n + 3) // 4 * 16 + n * 256  # (a bitmap, the owners, the chunks, each up to 16 bytes)
        assert [_lib.kv_extra_bytes(n) for n in (0, 1, 32, 33, 1000)] == [0, 288, 8336, 8608, 260128] == [esc(n) for n in (0, 1, 32, 33, 1000)]
        for bad in ((0, 128, 63), (33, 128, 63), (8, 64, 63), (8, 128, -1), (8, 128, 1024)):
            try:
                _lib.kv_leaf_bytes(*bad)
            except RuntimeError:
                continue
            raise AssertionError(bad)
    except OSError:
        print("test_kv_page: the library's own page sizes skipped (no library)")
    vcs = lambda mb, ms, mml: types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_batched_tokens=mb, max_num_seqs=ms), model_config=types.SimpleNamespace(max_model_len=mml))
    assert kv.work_blocks(vcs(8704, 256, 8704)) == 608 and kv.work_blocks(vcs(2048, 256, 32768)) == 2048 and kv.work_blocks(vcs(8192, 256, 4096)) == 576 and kv.work_blocks(vcs(8192, 8, 4096)) == 520  # (a step's tokens and a block over for up to 64 rows, or one request of max-model-len)
    assert kv.work_blocks(vcs(8192, 8, 4096), exact=True) == 8 * 256 and kv.work_blocks(vcs(8704, 256, 8704), exact=True) == 256 * 544  # (exact: every request's whole context at once)
    kv._STATE.bay = None  # (the header's: 15 entries for 8 heads)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    assert spec.page_size_bytes == 65536
    c = kv.GlydBackend.customize_spec(spec)
    assert c.dtype == torch.uint8 and c.page_size_bytes == 49408 and c.state_content_bytes == 386 and kv.GlydBackend.customize_spec(c) is c
    assert kv.GlydBackend.customize_spec(FullAttentionSpec(block_size=32, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)).page_size_bytes == 2 * 49408  # (two pages of 16)
    assert round(65536 / c.page_size_bytes, 4) == 1.3264
    kv._STATE.bay = 111  # (GLYD_KV_SPARE)
    assert kv.GlydBackend.customize_spec(spec).page_size_bytes == 49792
    kv._STATE.bay = None
    # head size 256 (Qwen3.5: 4 KV heads): two of the page's heads of 128 a head, the page of 8 heads; a hybrid model's attention block is many pages (vLLM sizes it from the spec of one token)
    assert kv.leaf_heads(4, 256) == 8 and kv.leaf_heads(8, 128) == 8 and kv.cell_bytes(8, 15, 4) == 772 and kv.cell_bytes(8, 15) == 386
    s256 = kv.GlydBackend.customize_spec(FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=256, dtype=torch.bfloat16))
    assert s256.dtype == torch.uint8 and s256.state_content_bytes == 772 and s256.page_size_bytes == 49408  # (bf16's: 65,536)
    assert kv.GlydBackend.customize_spec(FullAttentionSpec(block_size=1, num_kv_heads=4, head_size=256, dtype=torch.bfloat16)).page_size_bytes == 3088  # (a token: what vLLM aligns a hybrid model's blocks by)
    assert kv.GlydBackend.customize_spec(FullAttentionSpec(block_size=704, num_kv_heads=4, head_size=256, dtype=torch.bfloat16)).page_size_bytes == 44 * 49408
    assert kv.leaves_of([0, 2], 44) == list(range(0, 44)) + list(range(88, 132)) and kv.leaves_of([5, 3]) == [5, 3]
    try:
        from glyd_gpu import _lib, kernels

        kernels.lib()
        for h in (1, 2, 4, 8, 16):
            for a in (0, 15, 31, 1023):
                assert _lib.kv_leaf_bytes(h, 256, a) == kv.leaf_bytes(2 * h, a), (h, a)
            assert _lib.kv_free_bay(h, 256) == kv.free_bay(2 * h)
        for bad in ((17, 256, 0), (4, 192, 0), (4, 512, 0)):
            try:
                _lib.kv_leaf_bytes(*bad)
            except RuntimeError:
                continue
            raise AssertionError(bad)
    except OSError:
        print("test_kv_page: the library's head size 256 skipped (no library)")


def test_kv_backend():
    """The backend's answers to vLLM's checks (AttentionBackend.validate_configuration): what it holds (decoder attention of head size 128 on
    Ampere, Ada and Hopper, bf16) and each reason vLLM is told where it does not."""
    if vp is None:
        return print("test_kv_backend: skipped (no vLLM)")
    import torch
    from glyd_gpu import vllm_kv as kv
    from vllm.platforms.interface import DeviceCapability

    ok = dict(head_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto", block_size=None, use_mla=False, has_sink=False, use_sparse=False, use_mm_prefix=False, use_per_head_quant_scales=False,
              device_capability=DeviceCapability(8, 9), attn_type="decoder")
    why = lambda **kw: kv.GlydBackend.validate_configuration(**{**ok, **kw})
    assert why() == [] and why(device_capability=DeviceCapability(8, 0)) == [] and why(block_size=32) == [] and why(kv_cache_dtype="bfloat16") == [] and why(head_size=256) == [] and why(block_size=704) == []
    assert why(head_size=64) == ["head_size not supported"] and why(head_size=192) == ["head_size not supported"] and why(dtype=torch.float16) == ["dtype not supported"] and why(kv_cache_dtype="fp8") == ["kv_cache_dtype not supported"]
    assert why(block_size=8) == ["block_size not supported"] and why(has_sliding_window=True) == ["sliding window not supported"] and why(has_sink=True) == ["attention sinks not supported"]
    assert why(device_capability=DeviceCapability(9, 0)) == [] and why(device_capability=DeviceCapability(10, 0)) == ["compute capability not supported"] and why(use_kv_connector=True) == ["KV connector not supported"]  # (Hopper: the prompt rows' FlashAttention is 2)
    assert why(use_batch_invariant=True) == [] and why(attn_type="encoder") == ["attention type encoder not supported"] and why(use_mla=True) == ["MLA not supported"]  # (the decode kernel's plan is the sequence's alone)
    assert kv.GlydMetadataBuilder.get_cudagraph_support(None, None).name == "UNIFORM_SINGLE_TOKEN_DECODE"  # (vLLM's FULL graphs for decode steps)
    assert kv.GlydBackend.get_name() == "CUSTOM" and kv.GlydBackend.get_supported_kernel_block_sizes() == [16]
    assert all(l.is_block_compact for l in kv.GlydBackend.supported_kv_cache_layouts())  # (a page one contiguous run: the kernels take the block stride)


def test_kv_fa3_schedule():
    """FlashAttention 3's scheduler metadata of a staged group (Hopper): the arguments vLLM's own builder gives get_scheduler_metadata, for the group's rows at the staged rows' longest query and context, 16-token
    pages, causal, no window, vLLM's own choice of splits; none for version 2 and under VLLM_BATCH_INVARIANT (as vLLM's builder leaves it)."""
    if vp is None:
        return print("test_kv_fa3_schedule: skipped (no vLLM)")
    import torch
    from glyd_gpu import vllm_kv as kv

    seen = []
    b = object.__new__(kv.GlydMetadataBuilder)
    b.fa3 = dict(num_heads_q=32, num_heads_kv=8, headdim=128, qkv_dtype=torch.bfloat16, page_size=16, causal=True, window_size=(-1, -1))
    b.fa3_graph = None
    b.sched_fn = lambda **kw: seen.append(kw) or "metadata"
    p = {"lens": torch.tensor([300, 0, 50], dtype=torch.int32), "cu": torch.tensor([0, 100, 100, 100], dtype=torch.int32)}
    from unittest import mock

    with mock.patch.object(kv.envs, "VLLM_BATCH_INVARIANT", False, create=True):
        assert b.schedule(p, 100, 300) == "metadata"
        (kw,) = seen
        assert kw["batch_size"] == 3 and kw["max_seqlen_q"] == 100 and kw["max_seqlen_k"] == 300 and kw["cache_seqlens"] is p["lens"] and kw["cu_seqlens_q"] is p["cu"]
        assert (kw["num_heads_q"], kw["num_heads_kv"], kw["headdim"], kw["page_size"], kw["causal"], kw["window_size"], kw["num_splits"]) == (32, 8, 128, 16, True, (-1, -1), 0)
        assert b.shares(100) == 0 and b.shares(100000) == 0  # (no CUDA graphs' bound: the kernel's own choice)
        b.fa3_graph = (32, 512)  # (the full graphs on: vLLM's builder gives a step that fits them the bound the graphs are captured with, a larger one the kernel's own choice)
        assert b.shares(512) == 32 and b.shares(513) == 0 and b.schedule(p, 100, 300, b.shares(100)) == "metadata" and seen[-1]["num_splits"] == 32
        b.fa3_graph = (32, None)
        assert b.shares(10) == 0
        b.fa3 = None
        assert b.schedule(p, 100, 300) is None and len(seen) == 2  # (FlashAttention 2: no schedule)
    with mock.patch.object(kv.envs, "VLLM_BATCH_INVARIANT", True, create=True):
        b.fa3 = dict(num_heads_q=32, num_heads_kv=8, headdim=128, qkv_dtype=torch.bfloat16, page_size=16, causal=True, window_size=(-1, -1))
        b.fa3_graph = (32, 512)
        assert b.schedule(p, 100, 300) is None and len(seen) == 2  # (vLLM's builder leaves it out under batch invariance)
        assert b.shares(10) == 1 and b.shares(10000) == 1  # (and one split)


def test_kv_refusals():
    """What kv: lossless refuses in vLLM's config, each with what to do (vllm_plugin._kv_refusals)."""
    if vp is None:
        return print("test_kv_refusals: skipped (no vLLM)")
    import torch

    ns = types.SimpleNamespace
    qwen = ns(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, head_dim=128, sliding_window=None, use_sliding_window=False)
    cfg = ns(get_text_config=lambda: qwen)

    def vc(tp=1, pp=1, dcp=1, pcp=1, prefix=False, chunked=False, spec=None, cache_dtype="auto", block=16, offload=0, conn=None, dtype=torch.bfloat16, backend=None, mml=8192, backend_x=None):
        return ns(parallel_config=ns(tensor_parallel_size=tp, pipeline_parallel_size=pp, decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp, distributed_executor_backend=backend_x), model_config=ns(dtype=dtype, hf_config=cfg, max_model_len=mml),
                  cache_config=ns(enable_prefix_caching=prefix, cache_dtype=cache_dtype, block_size=block, kv_offloading_size=offload), scheduler_config=ns(enable_chunked_prefill=chunked),
                  speculative_config=spec, kv_transfer_config=conn, attention_config=ns(backend=backend))

    def why(v=None, c=cfg):
        return vp._kv_refusals(v or vc(), c)

    assert why() == []
    assert len(why(vc(tp=2))) == 1 and "tensor, pipeline and context parallel" in why(vc(tp=2))[0] and len(why(vc(pp=2))) == 1 and len(why(vc(dcp=2))) == 1
    assert why(vc(prefix=True, chunked=True)) == []  # (prefix hits and the chunks of a prompt read their context through the staging cache)
    assert "speculative" in why(vc(spec=object()))[0] and "--kv-cache-dtype fp8" in why(vc(cache_dtype="fp8"))[0] and "multiple of 16" in why(vc(block=24))[0]
    assert "KV connectors" in why(vc(conn=object()))[0] and "KV connectors" in why(vc(offload=4))[0] and "bf16 only" in why(vc(dtype=torch.float16))[0]
    assert "MLA" in why(c=ns(get_text_config=lambda: ns(kv_lora_rank=512)))[0]
    assert "sliding window" in why(c=ns(get_text_config=lambda: ns(**{**vars(qwen), "sliding_window": 4096, "use_sliding_window": True})))[0]
    assert "sliding window" in why(c=ns(get_text_config=lambda: ns(**{**vars(qwen), "layer_types": ["full_attention", "sliding_attention"]})))[0]
    assert "head size 64: 128 or 256 only" in why(c=ns(get_text_config=lambda: ns(**{**vars(qwen), "head_dim": 64})))[0] and "16 query heads to a KV head" in why(c=ns(get_text_config=lambda: ns(**{**vars(qwen), "num_attention_heads": 128})))[0]
    qwen35 = ns(hidden_size=4096, num_attention_heads=16, num_key_value_heads=4, head_dim=256, sliding_window=None, layer_types=["linear_attention"] * 3 + ["full_attention"])  # (Qwen3.5-9B: Gated DeltaNet beside full attention)
    assert why(c=ns(get_text_config=lambda: qwen35)) == []  # (its linear-attention layers' state is vLLM's; the full-attention layers' cache is held)
    assert "sliding window and state-space layers" in why(c=ns(get_text_config=lambda: ns(**{**vars(qwen35), "layer_types": ["mamba", "full_attention"]})))[0]
    gemma4 = ns(hidden_size=3840, num_attention_heads=16, num_key_value_heads=8, head_dim=256, sliding_window=1024, layer_types=["sliding_attention"] * 5 + ["full_attention"])
    assert "sliding window" in why(c=ns(get_text_config=lambda: gemma4))[0]
    assert "--max-model-len 65,536" in why(vc(mml=65536))[0] and "40,960 positions" in why(vc(mml=65536))[0] and why(vc(mml=40960)) == [] and why(vc(mml=32768)) == [] and why(vc(mml=4096)) == []
    assert len(why(vc(tp=2, spec=object(), cache_dtype="fp8"))) == 3  # (every reason, not the first)
    assert why(vc(backend_x="uni")) == [] and "--distributed-executor-backend mp" in why(vc(backend_x="mp"))[0]
    raises(lambda: vp._kv_prepare(vc(tp=2), ("auto", False, False, 1.0, "lossless")), "kv lossless is not held here")


def test_kv_auto():
    """kv's default, auto: lossless on an A100, on Ada and on Hopper where nothing refuses it and the tables cover the window, vLLM's own KV cache with one line otherwise (other GPUs, a model the tables do not cover,
    a config that is refused); lossless is the user's own say (refused with why, except a window the tables do not cover: vLLM's cache serves it, whole, with a line); off is off. And what the engine's process
    writes for the workers, with the tables that cannot be had (vllm_plugin._kv_decide, _kv_prepare)."""
    if vp is None:
        return print("test_kv_auto: skipped (no vLLM)")
    import torch
    from unittest import mock
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    from glyd_gpu import kvcal

    ns = types.SimpleNamespace
    qwen = ns(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, head_dim=128, sliding_window=None, use_sliding_window=False)
    cfg = ns(get_text_config=lambda: qwen)

    def vc(tp=1, mml=8192):
        return ns(parallel_config=ns(tensor_parallel_size=tp, pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1, distributed_executor_backend=None),
                  model_config=ns(dtype=torch.bfloat16, hf_config=cfg, max_model_len=mml, model="m", revision=None, trust_remote_code=False, tokenizer="m", quantization=None),
                  cache_config=ns(enable_prefix_caching=False, cache_dtype="auto", block_size=16, kv_offloading_size=0), scheduler_config=ns(enable_chunked_prefill=False, scheduler_cls=None),
                  speculative_config=None, kv_transfer_config=None, attention_config=ns(backend=None, backend_per_kind={}), load_config=ns(download_dir=None), additional_config={})

    import contextlib

    NAMES = {(8, 0): "NVIDIA A100-SXM4-40GB", (8, 9): "NVIDIA L4", (9, 0): "NVIDIA H100 80GB HBM3"}

    @contextlib.contextmanager
    def on(major, minor, name=None):  # (the GPU: its compute capability and the name the driver gives it)
        nm = NAMES.get((major, minor), "NVIDIA Some GPU") if name is None else name
        with mock.patch.object(current_platform, "get_device_capability", lambda device_id=0: DeviceCapability(major, minor)), mock.patch.object(current_platform, "get_device_name", lambda device_id=0: nm):
            yield

    # (`auto` holds the cache on an A100, an L4, an H100 and a GH200 by name, where it is measured: the same architecture under another name is not yet)
    for cap, name, auto in (((8, 0), "NVIDIA A100-SXM4-40GB", True), ((8, 0), "NVIDIA A100 80GB PCIe", True), ((8, 9), "NVIDIA L4", True), ((9, 0), "NVIDIA H100 80GB HBM3", True), ((9, 0), "NVIDIA H100 PCIe", True),
                            ((9, 0), "NVIDIA H100 NVL", True), ((8, 9), "NVIDIA L40S", False), ((8, 9), "NVIDIA GeForce RTX 4090", False), ((8, 9), "NVIDIA RTX 6000 Ada Generation", False), ((9, 0), "NVIDIA GH200 480GB", True),
                            ((9, 0), "NVIDIA H200", False), ((8, 6), "NVIDIA A10G", False), ((8, 6), "NVIDIA A10", False), ((8, 0), "NVIDIA A30", False), ((8, 6), "NVIDIA GeForce RTX 3090", False),
                            ((7, 5), "Tesla T4", False), ((8, 9), "", False)):
        with on(*cap, name=name):
            eff, msg, level = vp._kv_decide(vc(), "auto", cfg)
            assert eff == ("lossless" if auto else "off") and (msg is None) == auto, (cap, name, eff, msg)
            assert auto or (level == "info" and "stays vLLM's" in msg and "kv lossless turns it on" in msg and (name in msg if name else "unknown name" in msg)), (name, msg)
    with on(8, 9):
        eff, msg, level = vp._kv_decide(vc(mml=65536), "auto", cfg)  # (a window the tables do not cover: vLLM's own cache, whole, and a line)
        assert eff == "off" and level == "info" and "--max-model-len 65,536" in msg and "40,960 positions" in msg
        eff, msg, level = vp._kv_decide(vc(mml=40960), "auto", cfg)  # (the Qwen3 models' window: covered)
        assert (eff, msg) == ("lossless", None)
        eff, msg, level = vp._kv_decide(vc(mml=32768), "auto", cfg)
        assert (eff, msg) == ("lossless", None)
        eff, msg, level = vp._kv_decide(vc(tp=2), "auto", cfg)  # (a config that is refused: not an error for the default)
        assert eff == "off" and "tensor, pipeline and context parallel" in msg
        eff, msg, level = vp._kv_decide(vc(mml=65536), "lossless", cfg)  # (the user's own say: the window is served on vLLM's cache with a warning, not shortened and not refused)
        assert eff == "off" and level == "warning" and "serve with --max-model-len 40960 or less" in msg
        raises(lambda: vp._kv_decide(vc(tp=2), "lossless", cfg), "kv lossless is not held here")
        assert vp._kv_decide(vc(), "lossless", cfg) == ("lossless", None, None) and vp._kv_decide(vc(tp=2), "off", cfg) == ("off", None, None)
    q35 = ns(get_text_config=lambda: ns(hidden_size=4096, num_attention_heads=16, num_key_value_heads=4, head_dim=256, sliding_window=None, layer_types=["linear_attention", "full_attention"]))
    with on(9, 0):  # (head size 256 and linear-attention layers: held where kv lossless asks, not by auto until measured faster)
        eff, msg, level = vp._kv_decide(vc(), "auto", q35)
        assert eff == "off" and level == "info" and "head size 256 and linear-attention layers" in msg and "kv lossless turns it on" in msg, msg
        assert vp._kv_decide(vc(), "lossless", q35) == ("lossless", None, None)
        assert "head size 256" in vp._kv_decide(vc(), "auto", ns(get_text_config=lambda: ns(hidden_size=4096, num_attention_heads=16, num_key_value_heads=4, head_dim=256, sliding_window=None)))[1]
    with on(8, 6):
        assert vp._kv_decide(vc(), "lossless", cfg)[0] == "lossless"  # (a GPU auto does not hold it on, an A10: on where the user asks)
    with on(10, 0):
        raises(lambda: vp._kv_decide(vc(), "lossless", cfg), "Ampere, Ada and Hopper only")  # (and not on one the backend does not hold)
    with on(8, 9):
        eff, msg, level = vp._kv_decide(vc(), "auto", cfg, exact=True)  # (exact and verify carry the staging route and a shadow of the cache: not for a default)
        assert eff == "off" and level == "info" and "exact is on" in msg and "kv lossless asks for that" in msg
        eff, msg, level = vp._kv_decide(vc(), "auto", cfg, verify=True)
        assert eff == "off" and "verify is on" in msg
        assert vp._kv_decide(vc(), "lossless", cfg, exact=True, verify=True) == ("lossless", None, None) and vp._kv_decide(vc(), "off", cfg, exact=True) == ("off", None, None)
    with on(8, 9), mock.patch.dict(os.environ):  # (what the engine's process writes for the workers; a job's own GLYD_KV_TABLES is not the tables this makes)
        os.environ.pop("GLYD_KV_TABLES", None)
        v = vc(mml=65536)
        assert vp._kv_prepare(v, ("auto", False, False, 1.0, "auto")) == "off" and v.additional_config["glyd"]["kv"] == "off" and v.attention_config.backend_per_kind == {} and v.scheduler_config.scheduler_cls is None
        v = vc()
        with mock.patch.object(kvcal, "ensure", side_effect=RuntimeError("no room on the GPU\nmore")):
            assert vp._kv_prepare(v, ("auto", False, False, 1.0, "auto")) == "off" and v.additional_config["glyd"]["kv"] == "off"  # (auto: no tables, vLLM's cache, a line)
            raises(lambda: vp._kv_prepare(vc(), ("auto", False, False, 1.0, "lossless")), "no room on the GPU")  # (lossless: raised)
        meta = {"held_out_rate": {"k": 0.0001, "v": 0.0002}}
        with mock.patch.object(kvcal, "ensure", return_value="/t.npz"), mock.patch.object(kvcal, "Tables", lambda path: ns(meta=meta, path=path)):
            v = vc()
            assert vp._kv_prepare(v, ("auto", False, False, 1.0, "auto")) == "lossless"
            ac = v.additional_config["glyd"]
            assert ac["kv"] == "lossless" and ac["kv_tables"] == {"path": "/t.npz"} and ac["scheduler_cls"] is None and v.scheduler_config.scheduler_cls == vp.SCHEDULER
            assert vp._options(v.additional_config["glyd"], {}, {"layout": None, "exact": None, "verify": None, "fraction": None, "kv": None})[4] == "lossless"  # (a worker reads it as given)
        poor = {"held_out_rate": {"k": 0.5, "v": 0.5}}
        with mock.patch.object(kvcal, "ensure", return_value="/t.npz"), mock.patch.object(kvcal, "Tables", lambda path: ns(meta=poor, path=path)), mock.patch.object(kvcal, "refusal", lambda m: "too many outside"):
            v = vc()
            assert vp._kv_prepare(v, ("auto", False, False, 1.0, "auto")) == "off" and v.additional_config["glyd"]["kv"] == "off"  # (auto leaves a model the tables refuse to vLLM's cache)


def test_kv_refused_tables():
    """A model whose base tables put it over the line is refused where its tables load, with why, and one that is under it is not (vllm_plugin._kv_setup)."""
    if vp is None:
        return print("test_kv_refused_tables: skipped (no vLLM)")
    import tempfile

    import numpy as np
    import torch
    from glyd_gpu import kvcal, vllm_kv

    ns = types.SimpleNamespace
    qwen = ns(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, head_dim=128, sliding_window=None, use_sliding_window=False)
    vc = ns(parallel_config=ns(tensor_parallel_size=1, pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1, distributed_executor_backend=None),
            model_config=ns(dtype=torch.bfloat16, hf_config=ns(get_text_config=lambda: qwen), max_model_len=8192, model="org/model"), cache_config=ns(enable_prefix_caching=False, cache_dtype="auto", block_size=16, kv_offloading_size=0),
            scheduler_config=ns(enable_chunked_prefill=False), speculative_config=None, kv_transfer_config=None, attention_config=ns(backend=None, backend_per_kind={}))
    kb = np.zeros((1, 8, 128), dtype=np.uint8)
    ok = {"positions": 16384, "heads": 8, "layers": 1, "refused": False, "held_out_rate": {"k": 0.0004, "v": 0.0003}, **kvcal.store_meta(torch.tensor([2.0]), torch.tensor([0]), torch.tensor([90]), 15, 1039, 1)}
    heavy = {**ok, **kvcal.store_meta(torch.tensor([2.0, 0.0]), torch.tensor([0, 1]), torch.tensor([90, 32744]), 15, 1039, 2, lambda i: True, lambda i: f"t{i}")}  # (a token whose page has more exceptions than a page holds: served)
    state = dict(vars(vllm_kv._STATE))
    env = {k: os.environ.pop(k, None) for k in ("GLYD_KV_TABLES", "GLYD_KV_SPARE", "GLYD_KV_RESERVE_1", "GLYD_KV_RESERVE_2", "GLYD_KV_RESERVE_1_UNITS", "GLYD_KV_RESERVE_2_UNITS")}  # (a job's own: the config's path is what is tested)
    try:
        with tempfile.TemporaryDirectory() as d:
            def setup(meta):
                path = os.path.join(d, "t.npz")
                kvcal.save(path, ["model.layers.0.self_attn.attn"], kb, kb, meta)
                return vp._kv_setup(vc, {"path": path})

            assert setup(ok)["path"].endswith("t.npz")
            raises(lambda: setup({**ok, "held_out_rate": {"k": 0.002, "v": 0.0003}}), "the model's values compress too poorly (worst kind of text left out: K 0.200%")
            raises(lambda: setup({**ok, "positions": 4096}), "made for 4,096 positions")
            assert setup(heavy)["path"].endswith("t.npz") and vllm_kv.hard_tokens(vllm_kv._STATE.tables.meta) == 1  # (served)
            raises(lambda: setup({**heavy, "held_out_rate": {"k": 0.002, "v": 0.0003}}), "the model's values compress too poorly")
    finally:
        vars(vllm_kv._STATE).update(state)
        for k, v in env.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})


class _Req:
    """A request as vLLM's scheduler keeps one (the fields vllm_sched reads)."""

    def __init__(self, i, n=16, heavy=False, status=None):
        from vllm.v1.request import RequestStatus

        self.request_id, self.client_index, self.status = i, 0, status or RequestStatus.WAITING
        self.all_token_ids = [1000 if heavy else 1] * n  # (token ids from 1000 are the heavy ones of the stub predictor)
        self.num_tokens, self.num_computed_tokens, self.num_in_flight_tokens, self.prompt_embeds = n, 0, 0, None


class _Stub:
    """A predictor: a block of 16 tokens with a heavy one (id from 1000) takes 10 slabs."""

    def blocks(self, ids):
        import numpy as np

        ids = np.asarray(ids)
        pad = -len(ids) % 16
        grid = np.concatenate([ids, np.zeros(pad, dtype=ids.dtype)]).reshape(-1, 16)
        return ((grid >= 1000).any(1) * 10).astype(np.int32)


def _sched(held=None, lumps=100, running=(), waiting=(), max_running=4, predictor=_Stub(), kcc=None):
    """vllm_sched.LostPages over a fake of vLLM's scheduler (its queues, running list, requests, preempt, finish and block bookkeeping): returns it with its fake pools and monitor."""
    from glyd_gpu import vllm_kv, vllm_sched
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.request import RequestStatus

    class Blk:
        ref_cnt = 0

    class Pool:
        def __init__(self):
            self.blocks = [Blk() for _ in range(32)]
            self.next = 20

        def get_new_blocks(self, n):  # (vLLM's: the blocks, KVCacheBlock-like, taken from the free queue)
            out = [types.SimpleNamespace(block_id=self.next + i) for i in range(n)]
            self.next += n
            return out

    class Cache:
        def __init__(self):
            self.held, self.evicted, self.reset = held if held is not None else {}, set(), False
            self.block_pool = Pool()

        def get_block_ids(self, rid):
            return (list(self.held[rid]),)

        def evict_blocks(self, blocks):
            self.evicted |= set(blocks)

        def reset_prefix_cache(self):
            self.reset = True
            return True

        def free(self, rid):
            for b in self.held.get(rid, ()):
                self.block_pool.blocks[b].ref_cnt -= 1

    class Mon:
        def __init__(self):
            self.b, self.o, self.r, self.t = set(), False, set(), 0

        def lost(self):
            b, self.b = self.b, set()
            return b, self.o

        def second_new(self):
            r, self.r = self.r, set()
            return r

        def taken(self):
            return self.t, 0

    class Base:
        max_num_running_reqs = max_running
        num_waiting_for_streaming_input = 0

        def __init__(self):
            self.waiting, self.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS), create_request_queue(SchedulingPolicy.FCFS)
            self.running, self.requests, self.prev_step_scheduled_req_ids, self.preempted, self.finished = [], {}, set(), [], []
            self.kv_cache_manager = Cache()
            if kcc is not None:
                self.kv_cache_config = kcc
            for rid, bl in self.kv_cache_manager.held.items():
                for b in bl:
                    self.kv_cache_manager.block_pool.blocks[b].ref_cnt += 1

        def schedule(self, *a, **k):  # (vLLM's loop in short: from the head of the queues while there is a place, a block each from the pool)
            out = []
            for q in (self.skipped_waiting, self.waiting):
                while q and len(self.running) < self.max_num_running_reqs:
                    r = q.pop_request()
                    r.status = RequestStatus.RUNNING
                    self.running.append(r)
                    out.append(r.request_id)
                    self.kv_cache_manager.block_pool.get_new_blocks(1)
            return out

        def update_from_output(self, so, mro):
            return {0: EngineCoreOutputs()}

        def _preempt_request(self, r, now, drop_stale_output=False):
            assert r.status == RequestStatus.RUNNING and r not in self.running
            self.kv_cache_manager.free(r.request_id)
            r.status, r.num_computed_tokens = RequestStatus.PREEMPTED, 0
            self.preempted.append((r.request_id, drop_stale_output))
            self.waiting.prepend_request(r)

        def finish_requests(self, ids, status):
            ids, done = set(ids), []
            for r in list(self.requests.values()):
                if r.request_id in ids:
                    r.status = status
                    if r in self.running:
                        self.running.remove(r)
                    for q in (self.waiting, self.skipped_waiting):
                        if r in q:
                            q.remove_request(r)
                    self.kv_cache_manager.free(r.request_id)
                    del self.requests[r.request_id]
                    done.append(r)
                    self.finished.append(r.request_id)
            return done

        def finish(self, rid):  # (a request that ends: its blocks are free)
            self.finish_requests([rid], RequestStatus.FINISHED_STOPPED)

    class Sched(vllm_sched.LostLeaves, Base):
        pass

    saved = (dict(vllm_kv._MONITORS), dict(vllm_kv._EXTRA), list(vllm_kv._RELEASE), vllm_kv.LOST[0], vllm_kv._STATE.tables, vllm_kv._STATE.reuse)
    vllm_kv._MONITORS.clear()
    vllm_kv._EXTRA.clear()
    vllm_kv._RELEASE.clear()
    mon = vllm_kv._MONITORS[0] = Mon()
    vllm_kv._EXTRA[0] = types.SimpleNamespace(lumps=lumps, chunks=64)
    vllm_kv._STATE.tables = None
    s = Sched()
    s._glyd_predictor = predictor
    for r in list(running) + list(waiting):
        s.requests[r.request_id] = r
    for r in running:
        r.status = RequestStatus.RUNNING
        s.running.append(r)
    for r in waiting:
        s.waiting.add_request(r)
    return s, mon, saved


def _restore(saved):
    from glyd_gpu import vllm_kv

    vllm_kv._MONITORS.clear()
    vllm_kv._MONITORS.update(saved[0])
    vllm_kv._EXTRA.clear()
    vllm_kv._EXTRA.update(saved[1])
    vllm_kv._RELEASE[:] = saved[2]
    vllm_kv.LOST[0] = saved[3]
    vllm_kv._STATE.tables = saved[4]
    vllm_kv._STATE.reuse = saved[5]


def test_kv_lost_pages():
    """vllm_sched.LostPages after a step: the running requests that hold a block whose page was lost are preempted (their blocks out of the prefix cache, the request back in the queue to run again when the others that were running
    have finished), the others run on; a request that is alone twice or has caused six exhaustions ends with the error finish; the blocks that have had a raw or a lost page are handed over to be cleared once no request holds them."""
    if vp is None:
        return print("test_kv_lost_pages: skipped (no vLLM)")
    from glyd_gpu import vllm_kv, vllm_sched
    from vllm.v1.engine import FinishReason
    from vllm.v1.request import RequestStatus

    held = {"a": [1, 2], "b": [3, 4], "c": [2, 5]}
    reqs = [_Req("a", heavy=True), _Req("b", heavy=True), _Req("c", heavy=True)]
    s, mon, saved = _sched(held, running=reqs)
    try:
        s._glyd_run = {r.request_id: vllm_sched.Pred(16, [10]) for r in reqs}  # (three heavy requests let in)
        assert s.update_from_output(None, None)[0].outputs == [] and s.preempted == [] and vllm_kv._RELEASE == []  # (nothing lost, nothing raw)
        mon.r = {4, 9}
        s.update_from_output(None, None)
        assert s.preempted == [] and vllm_kv._RELEASE == [9] and s.kv_cache_manager.evicted == {9}  # (block 9 is held by none: cleared at once; 4 by b: it waits)
        mon.b = {2}
        out = s.update_from_output(None, None)[0].outputs
        assert [x for x, _ in s.preempted] == ["c", "a"] and all(d for _, d in s.preempted)  # (a and c hold block 2; b does not: preempted in reverse, with the output of the step dropped)
        assert out == [] and vllm_kv.LOST[0] == saved[3] and s.glyd_stats["preempted"] == 2
        assert [r.request_id for r in s.running] == ["b"] and [r.request_id for r in s.waiting] == ["a", "c"] and all(r.status == RequestStatus.PREEMPTED for r in s.waiting)  # (a first: back in the queue in the order they ran)
        assert s._glyd_after == {"a": {"b"}, "c": {"b", "a"}}  # (a waits for b, which holds some of the reserve; c for b and a)
        assert vllm_kv._RELEASE == [9, 2] and s.kv_cache_manager.evicted == {2, 9}  # (2 is free now and handed over; 4 still waits for b; 1 and 5, held by a and c only, have no raw page)
        s.schedule()  # (b still runs: a and c stay in the queue)
        assert [r.request_id for r in s.running] == ["b"] and [r.request_id for r in s.waiting] == ["a", "c"]
        s.finish("b")
        s.update_from_output(None, None)
        assert vllm_kv._RELEASE == [9, 2, 4] and 3 not in vllm_kv._RELEASE
        assert s.schedule() == ["a"] and [r.request_id for r in s.waiting] == ["c"]  # (b is gone: a runs, c waits for a)
        assert s._glyd_after == {"c": {"a"}} and s._glyd_tries == {"a": 1, "c": 1}
        # a alone exhausts the reserve a second time with nothing else holding it: the error finish, the engine goes on
        mon.b = {7}
        s.kv_cache_manager.held["a"] = [7]
        s.kv_cache_manager.block_pool.blocks[7].ref_cnt += 1
        out = s.update_from_output(None, None)[0].outputs
        assert [o.request_id for o in out] == ["a"] and out[0].finish_reason == FinishReason.ERROR and out[0].stop_reason == vllm_sched.MESSAGE and vllm_kv.LOST[0] == saved[3] + 1 and s.glyd_stats["failed"] == 1
        assert s.finished.count("a") == 1 and "a" not in s.requests
        assert s.schedule() == ["c"]  # (c no longer waits for a)
    finally:
        _restore(saved)
    # nothing is predicted to hold the reserve (admission is off, or the model has no such token): the requests running with the one that exhausted it are the ones it waits for
    held = {"a": [1, 2], "b": [3, 4]}
    s, mon, saved = _sched(held, running=[_Req("a"), _Req("b")])
    try:
        mon.b = {1}
        s.update_from_output(None, None)
        assert [x for x, _ in s.preempted] == ["a"] and s._glyd_after == {"a": {"b"}} and [r.request_id for r in s.running] == ["b"]  # (b runs on; a waits for it)
        assert s.schedule() == [] and s.glyd_stats["failed"] == 0
    finally:
        _restore(saved)
    held = {"a": [1, 2], "b": [3, 4]}
    s, mon, saved = _sched(held, running=[_Req("a"), _Req("b")])
    try:
        mon.o = True  # (the status ring overflowed: more pages lost than it names: every running request is preempted, and the prefix cache dropped, for the blocks that were not named may hold a lost page)
        s.update_from_output(None, None)
        assert sorted(x for x, _ in s.preempted) == ["a", "b"] and s.running == [] and s.kv_cache_manager.reset
        assert [r.request_id for r in s.waiting] == ["a", "b"] and s._glyd_after == {"a": set(), "b": {"a"}}  # (one after the other)
    finally:
        _restore(saved)


def test_kv_hybrid_blocks():
    """A hybrid model (vLLM zeroes the KV blocks a step takes; its linear-attention layers share the pool): a block is block_size / 16 of the cache's pages, the rings' pages are read as their blocks, a free block's
    pages are handed over whole, and every block a step takes from the pool is handed over to be cleared before the step writes it (vllm_sched.LostLeaves)."""
    if vp is None:
        return print("test_kv_hybrid_blocks: skipped (no vLLM)")
    import torch
    from glyd_gpu import vllm_kv
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    ns = types.SimpleNamespace
    attn = FullAttentionSpec(block_size=48, num_kv_heads=4, head_size=256, dtype=torch.bfloat16)  # (3 pages a block)
    mamba = ns(block_size=48)  # (a linear-attention group's spec: not an AttentionSpec)
    kcc = ns(kv_cache_groups=[ns(kv_cache_spec=attn), ns(kv_cache_spec=mamba)], needs_kv_cache_zeroing=True)
    held = {"a": [1, 2], "b": [3, 4]}
    s, mon, saved = _sched(held, running=[_Req("a"), _Req("b")], waiting=[_Req("c")], kcc=kcc)
    try:
        assert s._glyd_ratio == 3 and vllm_kv._STATE.reuse
        assert s.schedule() == ["c"] and vllm_kv._RELEASE == [60, 61, 62]  # (c took block 20: its three pages, cleared before the step writes them)
        assert s._glyd_new == []
        vllm_kv._RELEASE.clear()
        mon.r = {13}  # (page 13: block 4's, held by b: it waits)
        mon.b = {4}  # (page 4: block 1's, held by a: a is preempted, its blocks freed and handed over whole)
        s.update_from_output(None, None)
        assert [x for x, _ in s.preempted] == ["a"] and s.kv_cache_manager.evicted == {1} and sorted(vllm_kv._RELEASE) == [3, 4, 5]  # (block 1's pages; block 2 had no raw or lost page: nothing to clear)
        assert s._glyd_pending == {4}
    finally:
        _restore(saved)
    s, mon, saved = _sched({"a": [1]}, running=[_Req("a")], waiting=[_Req("c")])  # (one group of 16-token blocks, nothing zeroed: a block is a page, and the blocks a step takes are not handed over)
    try:
        assert s._glyd_ratio == 1 and not vllm_kv._STATE.reuse
        assert s.schedule() == ["c"] and vllm_kv._RELEASE == []
        mon.b = {1}
        s.update_from_output(None, None)
        assert [x for x, _ in s.preempted] == ["a"] and vllm_kv._RELEASE == [1]
    finally:
        _restore(saved)


def test_kv_release_early():
    """A hybrid model's pages handed over are cleared at the start of the step's execute_model (vLLM's worker, wrapped), before the model runner zeroes or copies into the step's new blocks, all of them, and
    none kept for later where no layer has written yet (nothing to clear; by then another group may hold the block); anyone else's where the step's metadata is built, at most RELEASE_MAX a step, as before."""
    if vp is None:
        return print("test_kv_release_early: skipped (no vLLM)")
    from unittest import mock

    from glyd_gpu import vllm_kv
    from vllm.v1.worker import gpu_worker

    calls = []

    class W:
        def execute_model(self, so):
            calls.append(("run", list(vllm_kv._RELEASE)))
            return "out"

    state = dict(vars(vllm_kv._STATE))
    saved = (list(vllm_kv._RELEASE), dict(vllm_kv._CACHES))
    try:
        with mock.patch.object(gpu_worker, "Worker", W), mock.patch.object(vllm_kv, "release_pending", lambda: calls.append(("clear", list(vllm_kv._RELEASE))) or vllm_kv._RELEASE.clear()):
            assert vllm_kv.release_early() and vllm_kv._STATE.early and W.execute_model._glyd_releases
            vllm_kv.release_early()  # (wrapped once)
            vllm_kv._STATE.reuse = False
            vllm_kv._RELEASE[:] = [7, 8]
            assert W().execute_model("so") == "out" and calls == [("run", [7, 8])]  # (one group: cleared where the step's metadata is built, as before)
            calls.clear()
            vllm_kv._STATE.reuse = True
            assert W().execute_model("so") == "out" and calls == [("clear", [7, 8]), ("run", [])]
            calls.clear()
            assert W().execute_model("so") == "out" and calls == [("run", [])]  # (nothing handed over: nothing cleared)
        with mock.patch.object(vllm_kv.torch.cuda, "is_current_stream_capturing", lambda: False):
            vllm_kv._CACHES.clear()
            vllm_kv._STATE.layers, vllm_kv._STATE.reuse = 2, True
            vllm_kv._RELEASE[:] = [1, 2, 3]
            assert vllm_kv.release_pending() == 0 and vllm_kv._RELEASE == []  # (no layer has written: nothing to clear, and nothing kept)
            vllm_kv._STATE.reuse = False
            vllm_kv._RELEASE[:] = [1, 2, 3]
            assert vllm_kv.release_pending() == 0 and vllm_kv._RELEASE == [1, 2, 3]  # (one group: kept until every layer has written, as before)
    finally:
        vars(vllm_kv._STATE).update(state)
        vllm_kv._RELEASE[:] = saved[0]
        vllm_kv._CACHES.clear()
        vllm_kv._CACHES.update(saved[1])


def test_kvcal_leaf_shape():
    """kvcal counts a layer's keys as the cache's pages hold them: a head of 256 channels as two of 128 (Qwen3.5's 4 KV heads: tables [8, 128])."""
    from glyd_gpu import kvcal

    assert kvcal.leaf_shape(4, 256) == (8, 128) and kvcal.leaf_shape(8, 128) == (8, 128) and kvcal.leaf_shape(2, 64) == (2, 64)


def test_kv_admission():
    """vllm_sched.LostPages before a step: a request whose predicted raw pages the pool has no room for (beside those the running ones have taken and will take) stays in the queue; those behind it that take none go on, those that take
    some wait for it; the held are back where they were; a prompt that could never fit is ended with the error finish."""
    if vp is None:
        return print("test_kv_admission: skipped (no vLLM)")
    from glyd_gpu import vllm_sched
    from vllm.v1.engine import FinishReason

    ids = lambda q: [r.request_id for r in q]
    s, mon, saved = _sched(waiting=[_Req("h1", 64, True), _Req("h2", 64, True), _Req("h3", 64, True), _Req("o1"), _Req("o2")], max_running=4)  # (a heavy request takes 40 slabs: 4 blocks of 10; the pool has 100)
    try:
        assert s.schedule() == ["h1", "h2", "o1", "o2"] and ids(s.waiting) == ["h3"]  # (h3 does not fit beside h1 and h2: the ordinary ones behind it go on)
        assert set(s._glyd_run) == {"h1", "h2"} and s.glyd_stats["held"] == 1
        assert s.schedule() == [] and ids(s.waiting) == ["h3"]  # (no place to run: nothing is looked at)
        s.finish("h1")
        s.finish("o1")
        s.requests["h2"].num_computed_tokens = 64  # (h2's prompt is all written: what the status counts is all of it)
        mon.t = 40
        assert s.schedule() == ["h3"]  # (h1's 40 are free)
        s.finish("h3")
        s.waiting.add_request(r := _Req("h4", 64, True))
        s.requests["h4"] = r
        assert s.schedule() == ["h4"] and set(s._glyd_run) == {"h2", "h4"}  # (40 + 40 of 100)
        # in progress: a request that has written half of a prompt of 64 tokens has 20 more slabs to take
        s.finish("h4")
        s.requests["h2"].num_computed_tokens = 32
        mon.t = 20
        for r in (_Req("h5", 112, True), _Req("h6", 96, True)):  # (70 and 60 slabs: 20 taken, 20 to come, 60 left)
            s.waiting.add_request(r)
            s.requests[r.request_id] = r
        assert s.schedule() == [] and ids(s.waiting) == ["h5", "h6"]  # (h5 does not fit; h6 would, and waits behind it)
        s.requests["h2"].num_computed_tokens = 48  # (a block more written: 10 of its 20 are still to come)
        mon.t = 30
        assert s.schedule() == [] and ids(s.waiting) == ["h5", "h6"]  # (30 taken and 10 to come: 60 left, not the 70)
        s.finish("h5")
        assert s.schedule() == ["h6"]  # (60 fit exactly)
    finally:
        _restore(saved)
    # strict order among the heavy: a small one does not pass a large one that waits; the ones that take none do
    s, mon, saved = _sched(running=[_Req("r", 64, True)], waiting=[_Req("big", 112, True), _Req("small", 16, True), _Req("o")], max_running=4)
    try:
        s._glyd_run = {"r": vllm_sched.Pred(64, [10, 10, 10, 10])}
        assert s.schedule() == ["o"] and ids(s.waiting) == ["big", "small"]  # (40 + 70 > 100: big waits; small (10) would fit, and waits behind it)
    finally:
        _restore(saved)
    # the held go back where they were
    s, mon, saved = _sched(waiting=[_Req("a"), _Req("h0", 112, True), _Req("h1", 112, True), _Req("b"), _Req("c")], max_running=2)
    try:
        assert s.schedule() == ["a", "h0"] and ids(s.waiting) == ["h1", "b", "c"]  # (h0 takes 70 of 100; no place for more: all in the order they came)
        s.finish("a")
        assert s.schedule() == ["b"] and ids(s.waiting) == ["h1", "c"]  # (h1 is held, b goes on, and h1 is back before c)
    finally:
        _restore(saved)
    # a prompt that could never fit: 200 predicted against a pool of 100 (a 1.5 times margin: 150); one between them runs alone
    s, mon, saved = _sched(waiting=[_Req("never", 320, True), _Req("big", 192, True), _Req("ok")], max_running=4)
    try:
        s._glyd_run = {}
        assert s.schedule() == ["big", "ok"] and "never" not in s.requests and s.glyd_stats["refused"] == 1  # (120 predicted: it runs, with all the pool)
        out = s.update_from_output(None, None)[0].outputs
        assert [o.request_id for o in out] == ["never"] and out[0].finish_reason == FinishReason.ERROR
        s.waiting.add_request(r := _Req("big2", 192, True))
        s.requests["big2"] = r
        assert s.schedule() == [] and ids(s.waiting) == ["big2"]  # (and nothing else that is heavy runs beside it)
    finally:
        _restore(saved)
    # no predictor (the model has no token whose page is raw, or admission is off): nothing is held
    s, mon, saved = _sched(waiting=[_Req("h1", 64, True), _Req("h2", 64, True), _Req("h3", 64, True)], predictor=None)
    try:
        assert s.schedule() == ["h1", "h2", "h3"]
    finally:
        _restore(saved)


def test_kv_predict():
    """What a prompt's pages will need before it runs (kvcal.Predictor, from the sweep's exceptions of each token in each layer) and the frontend's check of it (vllm_admit): a block is raw in a layer where its tokens' exceptions
    sum past what a page holds; one heavy token among ordinary ones is not; the 400, decided by each process from its own config and the KV cache's size as vLLM reports it."""
    if vp is None:
        return print("test_kv_predict: skipped (no vLLM)")
    import tempfile
    import numpy as np
    from glyd_gpu import kvcal, vllm_admit

    exc = np.zeros((6, 3), dtype=np.int16)  # 6 tokens, 3 layers
    exc[1] = [100, 0, 30]  # (a heavy token: 16 of it is 1,600 in layer 0: over a page's 1,039; 480 in layer 2)
    exc[2] = [70, 70, 70]  # (16 of it: 1,120, over, in every layer)
    exc[3] = [10, 10, 10]
    p = kvcal.Predictor(exc, 8, 1039)
    assert p.active and p.thr == int(0.9 * 1039) + 24
    assert p.blocks([1] * 16).tolist() == [1] and p.blocks([2] * 16).tolist() == [3]  # (a block of the token alone: its layers over)
    assert p.blocks([1] + [3] * 15).tolist() == [0]  # (one heavy token among ordinary ones: 100 + 150 in layer 0)
    assert p.blocks([1] * 8 + [3] * 8).tolist() == [0] and p.blocks([1] * 10 + [3] * 6).tolist() == [1]  # (10 x 100 + 60 = 1,060 > 959)
    assert p.blocks([1] * 16 + [2] * 16 + [0] * 5 + [2] * 3).tolist() == [1, 3, 0]  # (the last block: 5 + 3 tokens, 210 at most)
    assert p.blocks([]).tolist() == [] and p.blocks([2] * 17).tolist() == [3, 0]  # (the second block holds one token: 70)
    assert p.blocks([9999] * 16).tolist() == [0]  # (an id past the sweep's rows is the last row's)
    assert not kvcal.Predictor(np.full((4, 2), 5, dtype=np.int16), 8, 1039).active  # (16 x 5 = 80: no page of this vocabulary is raw)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        kb = np.zeros((2, 8, 128), dtype=np.uint8)
        kvcal.save(path, ["l0", "l1"], kb, kb, {"heads": 8, "store": {}}, exc)
        t = kvcal.Tables(path)
        assert (t.exc == exc).all() and t.exc.dtype == np.int16
        kvcal.save(path, ["l0", "l1"], kb, kb, {"heads": 8})
        assert kvcal.Tables(path).exc is None and vllm_admit.predictor(kvcal.Tables(path)) is None  # (a file with no sweep of exceptions: nothing is predicted)
        kvcal.save(path, ["l0", "l1"], kb, kb, {"heads": 8, "store": {}}, exc)
        vllm_admit._PRED.clear()
        assert vllm_admit.predictor(kvcal.Tables(path)).blocks([2] * 16).tolist() == [3]
        vllm_admit._PRED.clear()
        # the API server's check, from what its own process reads (the tables by their path in its config, the KV cache's size as vLLM reports it to it): nothing handed over by the engine, in a file or the environment
        was = {k: os.environ.pop(k, None) for k in ("GLYD_KV_SPARE", "GLYD_KV_RESERVE_2", "GLYD_KV_RESERVE_2_UNITS", "GLYD_KV_ADMIT")}
        from vllm.exceptions import VLLMClientError
        from vllm.v1.engine.input_processor import InputProcessor as IP

        def ip(blocks, tables=path, block_size=16):
            return types.SimpleNamespace(vllm_config=types.SimpleNamespace(additional_config={"glyd": {"kv_tables": {"path": tables}}}, cache_config=types.SimpleNamespace(num_gpu_blocks=blocks, block_size=block_size)))

        def refused(i, ids):
            try:
                vllm_admit.check(i, ids)
            except VLLMClientError as e:
                return str(e)
            return None

        try:
            kvcal.save(path, ["l0", "l1", "l2"], kb[:1].repeat(3, 0), kb[:1].repeat(3, 0), {"heads": 8, "layers": 3, "store": {"hard": 3}}, exc)
            vllm_admit._PRED.clear()
            meta = kvcal.Tables(path).meta
            near = lambda x, y: abs(x - y) <= 2
            assert near(vllm_admit.room(ip(1000).vllm_config, meta), 1500) and vllm_admit.room(ip(10).vllm_config, meta) == 256 and vllm_admit.room(ip(None).vllm_config, meta) is None  # (1.5 a block of the cache; at least 256)
            assert near(vllm_admit.room(ip(500, block_size=32).vllm_config, meta), 1500) and vllm_admit.room(ip(1000).vllm_config, {**meta, "store": {}}) == 256  # (a block of 32 tokens is two pages; a model with no such token: 0.25 a block, the least)
            if not compiled():  # (a release build reads no tuning variable: the variables of this part are a run from the sources's)
                os.environ["GLYD_KV_RESERVE_2"] = "50"
                assert vllm_admit.room(ip(1000).vllm_config, meta) == int(0.5 * 1000 * 3 * 49408 / 16384)
                os.environ["GLYD_KV_RESERVE_2"] = "0"
                assert vllm_admit.room(ip(1000).vllm_config, meta) == 0
                del os.environ["GLYD_KV_RESERVE_2"]
                os.environ["GLYD_KV_RESERVE_2_UNITS"] = "100"
                assert vllm_admit.room(ip(1000).vllm_config, meta) == 100 and vllm_admit.room(ip(None).vllm_config, meta) == 100  # (the size itself)
            else:
                os.environ["GLYD_KV_RESERVE_2"] = "50"
                os.environ["GLYD_KV_RESERVE_2_UNITS"] = "100"
                assert near(vllm_admit.room(ip(1000).vllm_config, meta), 1500)  # (neither is read)
                del os.environ["GLYD_KV_RESERVE_2"], os.environ["GLYD_KV_RESERVE_2_UNITS"]
            a = ip(1000)
            assert vllm_admit.check(a, [1] * 16) == 1 and vllm_admit.check(a, [2] * 48) == 9  # (9 pages against 100)
            if not compiled():
                msg = refused(a, [2] * 16 * 60)  # (180 against 100: past 1.5 x)
                assert msg and "180" in msg and "100" in msg and "send a shorter prompt" in msg
            assert vllm_admit.check(a, [2] * 16 * 50) == 150  # (1.5 x is allowed: the prediction is not that exact)
            os.environ.pop("GLYD_KV_RESERVE_2_UNITS", None)
            b = ip(1000)  # (no other process's file or variable: the same tables and the same cache's size, the same answer)
            assert refused(b, [2] * 16 * 800) and not refused(b, [2] * 16 * 400) and vllm_admit.check(b, [2] * 16 * 400) == 1200  # (about 1,500 slabs: over 2,250 is refused)
            assert not refused(ip(None), [2] * 16 * 800) and vllm_admit.check(ip(None), [2] * 16 * 800) == 2400  # (the cache's size not known yet: nothing is refused)
            # the tables cannot be read: nothing is refused here (the engine ends such a prompt), once logged; kv off for the server: not touched; the check off
            c = ip(1000, tables=os.path.join(d, "none.npz"))
            assert vllm_admit.front(c) is None and vllm_admit.check(c, [2] * 16 * 800) == 0 and c.__dict__["_glyd_front"] is None
            nokv = types.SimpleNamespace(vllm_config=types.SimpleNamespace(additional_config={}, cache_config=types.SimpleNamespace(num_gpu_blocks=1000, block_size=16)))
            assert vllm_admit.front(nokv) is None and vllm_admit.check(nokv, [2] * 16 * 800) == 0
            os.environ["GLYD_KV_ADMIT"] = "off"
            assert (vllm_admit.front(ip(1000)) is None) == (not compiled())  # (a release build does not read it)
            del os.environ["GLYD_KV_ADMIT"]
            # the wrapper of vLLM's input check: after vLLM's own, for a decoder's token ids (not embeddings)
            orig = IP._validate_model_input
            try:
                IP._validate_model_input = lambda self, pi, pt: None
                assert vllm_admit.install_front() and vllm_admit.install_front()  # (once)
                m = ip(1000)
                IP._validate_model_input(m, {"type": "token", "prompt_token_ids": [2] * 16 * 50}, "decoder")
                for bad in ({"type": "token", "prompt_token_ids": [2] * 16 * 800}, ):
                    try:
                        IP._validate_model_input(m, bad, "decoder")
                        raise AssertionError("not refused")
                    except VLLMClientError:
                        pass
                IP._validate_model_input(m, {"type": "embeds", "prompt_embeds": None}, "decoder")
                IP._validate_model_input(m, {"type": "token", "prompt_token_ids": [2] * 16 * 800}, "encoder")
            finally:
                IP._validate_model_input = orig
        finally:
            vllm_admit._PRED.clear()
            for k, v in was.items():
                os.environ.pop(k, None) if v is None else os.environ.update({k: v})
    from glyd_gpu import vllm_kv

    vc = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_batched_tokens=8704, max_num_seqs=256))
    assert vllm_kv.aux_places(vc) == 3 * (544 + 512) + 64 and vllm_kv.aux_places(types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=8))) == vllm_kv.AUX  # (three steps' worth of the blocks a step can name, at least the default)


def test_kv_warmup_skip():
    """vLLM's warmup of its kernels (steps of dummy requests with the token ids 0, 1, 2 ...) writes nothing to the KV cache of a model whose tables count tokens a page cannot hold (vllm_kv.skip_warmup_writes)."""
    if vp is None:
        return print("test_kv_warmup_skip: skipped (no vLLM)")
    from glyd_gpu import vllm_kv as kv
    from vllm.v1.worker import gpu_worker

    orig, seen = gpu_worker.warmup_kernels, []
    try:
        gpu_worker.warmup_kernels = lambda *a, **k: seen.append(kv._STATE.warming) or ("ran", a, k)
        assert kv.skip_warmup_writes() and kv._STATE.warming == 0
        assert gpu_worker.warmup_kernels(1, x=2) == ("ran", (1,), {"x": 2}) and seen == [1] and kv._STATE.warming == 0  # (the call goes through, with the flag up meanwhile)
        assert kv.skip_warmup_writes() and gpu_worker.warmup_kernels() and seen == [1, 1]  # (not wrapped twice: the flag is 1, not 2)

        def boom(*a, **k):
            raise RuntimeError("x")

        gpu_worker.warmup_kernels = boom
        assert kv.skip_warmup_writes()
        try:
            gpu_worker.warmup_kernels()
            raise AssertionError("raised")
        except RuntimeError:
            pass
        assert kv._STATE.warming == 0  # (the flag comes down where the warmup raises)
        del gpu_worker.warmup_kernels
        assert not kv.skip_warmup_writes()  # (a vLLM with no such function: nothing wrapped)
    finally:
        gpu_worker.warmup_kernels = orig
        kv._STATE.warming = 0


def _stage_layout(bt, ql, kl, r0, r1):
    """The host's layout of rows r0 .. r1 in the staging cache (numpy, np.unique: the first version of the plugin's planner, kept as the oracle of stage_plan's): the table (rows' blocks), the
    tokens' slots, the distinct whole pages of the contexts (sorted), and for the rows whose context ends inside a page: its row, the page, the block it goes to, the tokens of it that are context."""
    import numpy as np

    PT = 16
    ql, kl = ql[r0:r1], kl[r0:r1]
    G, nt = r1 - r0, sum(ql)
    ctx = np.array([k - q for k, q in zip(kl, ql)], dtype=np.int64)
    nfull = ctx // PT
    blocks = np.array([-(-k // PT) for k in kl], dtype=np.int64)
    cat = np.concatenate([bt[r0 + i, : nfull[i]] for i in range(G)]) if nfull.any() else np.zeros(0, dtype=np.int32)
    uniq, inv = np.unique(cat, return_inverse=True)
    U = len(uniq)
    pstart = U + np.concatenate([[0], np.cumsum(blocks - nfull)[:-1]])
    table = pstart[:, None] + np.arange(int(blocks.max()))[None, :] - nfull[:, None]
    off = 0
    for i in range(G):
        table[i, : nfull[i]] = inv[off : off + nfull[i]]
        off += nfull[i]
    row = np.repeat(np.arange(G), ql)
    pos = ctx[row] + np.arange(nt) - np.repeat(np.cumsum([0, *ql[:-1]]), ql)
    cells = table[row, pos // PT].astype(np.int64) * PT + pos % PT
    part = np.nonzero(ctx % PT)[0]
    return table, cells, uniq, part, bt[r0 + part, nfull[part]], pstart[part], ctx[part] % PT


def _stage_groups(bt, ql, kl, cap):
    """The host's groups (the first version of the planner): rows in turn while the distinct whole pages of their contexts and their own blocks fit cap: [(r0, r1)]."""
    PT, out, r0 = 16, [], 0
    while r0 < len(ql):
        seen, used, r1 = set(), 0, r0
        while r1 < len(ql):
            nfull = (kl[r1] - ql[r1]) // PT
            new = set(bt[r1, :nfull].tolist()) - seen if nfull else set()
            need = len(new) + -(-kl[r1] // PT) - nfull
            if r1 > r0 and used + need > cap:
                break
            seen |= new
            used += need
            r1 += 1
        out.append((r0, r1))
        r0 = r1
    return out


def test_kv_staging_groups():
    """The staging cache's plan for rows with a prompt to run (stage_plan: torch operations that wait for nothing, here on the CPU's tensors): rows that share a prefix's pages share their staging
    blocks, each row's own blocks (the partly written last page of its context, its tokens) follow, the tokens' slots and the pages to decode are right, and the rows go in groups as the host's
    planner (the oracle: np.unique and its greedy groups) put them: by hand for four rows, and for random rows with shared prefixes and a staging cache small enough for several groups."""
    if vp is None:
        return print("test_kv_staging_groups: skipped (no vLLM)")
    import numpy as np
    import torch
    from glyd_gpu import vllm_kv as kv

    T = lambda x: torch.tensor(x, dtype=torch.int32)
    # three rows: A and B share their first 3 pages (48 tokens of context each: a prefix hit), A adds 10 tokens, B 20 (its second block holds 4 of them); C has 20 tokens of context (a page and 4
    # of the next) and 5 new ones, in pages of its own; a fourth, fresh, of 7 tokens
    bt = np.zeros((4, 8), dtype=np.int32)
    bt[0, :4] = [11, 12, 13, 14]
    bt[1, :4] = [11, 12, 13, 15]
    bt[2, :2] = [21, 22]
    bt[3, :1] = [31]
    ql, kl = [10, 20, 5, 7], [58, 68, 25, 7]
    (g,) = kv.work_plan(torch.from_numpy(bt), T(kl), T([0, 10, 30, 35, 42]), 1, 42, 3, 5, 64)
    assert g["rows"][0].item() == 0 and g["rows"][1].item() == 4 and g["cu"].tolist() == [0, 10, 30, 35, 42] and g["lens"].tolist() == [58, 68, 25, 7]
    # shared: pages 11, 12, 13 and C's full page 21 (sorted: 11, 12, 13, 21 -> blocks 0 .. 3); A's own blocks: 1 (its context is 3 whole pages; its 10 tokens are the fourth block), B's: 2, C's: 1 (the rest of its page 22 and its 5), D's: 1
    nl = 12  # (the places for the whole pages: 4 rows x the most pages a context has, 3; then a place a row for its partly written page)
    assert g["src"].numel() == 16 and g["src"].tolist()[:4] == [11, 12, 13, 21] and g["src"].tolist()[4:nl] == [-1] * 8 and g["dst"].tolist()[:4] == [0, 1, 2, 3] and g["valid"].tolist()[:4] == [16, 16, 16, 16]
    # C's context ends 4 tokens into its second page (page 22): that page is decoded into C's first own block, 4 tokens of it context
    assert g["src"].tolist()[nl:] == [-1, -1, 22, -1] and g["valid"].tolist()[nl + 2] == 4
    t = g["table"].tolist()
    assert t[0][:3] == [0, 1, 2] == t[1][:3] and t[2][0] == 3  # (the prefix's blocks are one copy)
    own = [t[0][3], t[1][3], t[1][4], t[2][1], t[3][0]]
    assert sorted(own) == list(range(4, 9)) and g["dst"].tolist()[nl + 2] == t[2][1]  # (every row's own blocks are its alone and follow the shared ones)
    # a token's slot: row A's token 0 is position 48: its 4th block's first slot; row C's token 0 is position 20: its second block, offset 4
    sl = g["cells"].tolist()
    assert sl[0] == t[0][3] * 16 and sl[9] == t[0][3] * 16 + 9 and sl[10] == t[1][3] * 16 and sl[10 + 19] == t[1][4] * 16 + 3 and sl[30] == t[2][1] * 16 + 4 and sl[34] == t[2][1] * 16 + 8 and sl[35] == t[3][0] * 16
    # a staging cache of 8 blocks: A and B together need 3 shared + 1 + 2 = 6, C 1 + 1 more (page 21's and its own): 8; D: the next group (two groups the host's bound allows: 4 rows of 4, 5, 2 and 1 blocks, 8 at most each)
    two = kv.work_plan(torch.from_numpy(bt), T(kl), T([0, 10, 30, 35, 42]), 3, 42, 3, 5, 8)
    assert [(x["rows"][0].item(), x["rows"][1].item()) for x in two] == [(0, 3), (3, 4), (4, 4)] and _stage_groups(bt, ql, kl, 8) == [(0, 3), (3, 4)]
    assert two[0]["cu"].tolist() == [0, 10, 30, 35, 35] and two[1]["cu"].tolist() == [35, 35, 35, 35, 42] and two[1]["lens"].tolist() == [0, 0, 0, 7] and two[1]["cells"].tolist()[:35] == [-1] * 35 and two[1]["cells"].tolist()[35:] == list(range(7))
    assert two[2]["lens"].tolist() == [0, 0, 0, 0] and set(two[2]["cells"].tolist()) == {-1}  # (a group no rows are left for)

    # random rows against the oracle: prefixes shared in several groups of rows, contexts that end inside a page or on one, fresh rows, one-token queries, a staging cache small enough for several groups
    rng = np.random.default_rng(7)
    for trial in range(120):
        n = int(rng.integers(1, 10))
        prefixes = [rng.permutation(1000)[: int(rng.integers(0, 12))].astype(np.int32) for _ in range(3)]
        ql, kl, rows = [], [], []
        for i in range(n):
            pre = prefixes[int(rng.integers(0, 3))] if rng.random() < 0.7 else np.zeros(0, dtype=np.int32)
            ctx = int(rng.integers(0, 30 * 16)) if rng.random() < 0.7 else 0
            q = int(rng.integers(1, 60)) if rng.random() < 0.8 else 1
            nb = -(-(ctx + q) // 16)
            leaves = np.concatenate([pre, 2000 + 100 * i + np.arange(nb)])[:nb].astype(np.int32)  # (its first pages are the prefix's, then its own)
            ql.append(q)
            kl.append(ctx + q)
            rows.append(leaves)
        W = max(len(r) for r in rows) + 2
        bt = np.zeros((n, W), dtype=np.int32)
        for i, r in enumerate(rows):
            bt[i, : len(r)] = r
        cap = max(-(-k // 16) for k in kl) + int(rng.integers(0, 40))  # (one row at least fits)
        ntok, wc, w = sum(ql), max((k - q) // 16 for k, q in zip(kl, ql)), max(-(-k // 16) for k in kl)
        # the groups the host's bound allows (as if no page were shared): the plan has that many, the last ones empty where sharing leaves fewer
        no_share = _stage_groups(np.arange(n * W, dtype=np.int32).reshape(n, W), ql, kl, cap)
        want = _stage_groups(bt, ql, kl, cap)
        plans = kv.work_plan(torch.from_numpy(bt), T(kl), T([0, *np.cumsum(ql)]), len(no_share), ntok, wc, w, cap)
        assert len(plans) == len(no_share) >= len(want), (trial, len(plans), len(want))
        assert int(torch.stack([x["rows"][1] > x["rows"][0] for x in plans]).sum()) == len(want), trial  # (the groups with rows: what a step's first staged layer reads, so that the empty ones are not launched)
        starts = [0, *np.cumsum(ql)]
        for k, pl in enumerate(plans):
            a, b = (pl["rows"][0].item(), pl["rows"][1].item())
            if k >= len(want):
                assert a == b == n and not pl["lens"].any() and (pl["cells"] == -1).all(), (trial, k)
                continue
            assert (a, b) == want[k], (trial, k, (a, b), want[k])
            table, cells, uniq, part, ppage, pdst, pvalid = _stage_layout(bt, ql, kl, a, b)
            blocks = [-(-x // 16) for x in kl]
            for i in range(a, b):
                assert pl["table"][i, : blocks[i]].tolist() == table[i - a, : blocks[i]].tolist() and not pl["table"][i, blocks[i]:].any(), (trial, k, i, "table")
            assert pl["cells"][starts[a] : starts[b]].tolist() == cells.tolist() and (pl["cells"][: starts[a]] == -1).all() and (pl["cells"][starts[b] :] == -1).all(), (trial, k, "cells")
            assert pl["lens"][a:b].tolist() == kl[a:b] and not pl["lens"][:a].any() and not pl["lens"][b:].any()
            assert pl["cu"].tolist() == [min(max(x, starts[a]), starts[b]) for x in starts], (trial, k, "cu")
            if wc or any(kk > q for kk, q in zip(kl, ql)):
                U = len(uniq)
                nl = pl["src"].numel() - n
                assert pl["src"].tolist()[:U] == uniq.tolist() and all(x == -1 for x in pl["src"].tolist()[U:nl]) and pl["dst"].tolist()[:U] == list(range(U)) and pl["valid"].tolist()[:U] == [16] * U, (trial, k, "leaves")
                sp = pl["src"].tolist()[nl:]
                assert [i - a for i, x in enumerate(sp) if x >= 0] == part.tolist() and [sp[a + i] for i in part] == ppage.tolist(), (trial, k, "partial pages")
                assert [pl["dst"].tolist()[nl + a + i] for i in part] == pdst.tolist() and [pl["valid"].tolist()[nl + a + i] for i in part] == pvalid.tolist(), (trial, k, "partial pages' blocks")


def test_kvcal():
    """The base tables' rules: the window of 8 values of exponent >> 1 that holds the most (the first on a tie), the pages' needs measured,
    a page's needs beyond three inline entries a head, the bundled text's sections and its key, and a tables file through save and load."""
    if vp is None:
        return print("test_kvcal: skipped (no vLLM)")
    import tempfile
    import torch
    from glyd_gpu import kvcal

    h = torch.zeros(2, 3, 128, dtype=torch.int64)
    h[0, 0, 10:14] = torch.tensor([5, 1, 1, 5])  # (every window from 6 to 13 to 10 to 17 holds all 12: the first start, 6)
    h[0, 1, 120:128] = 3
    h[1, 2, 50] = 9
    h[1, 2, 57] = 9  # (50 to 57 hold both)
    b = kvcal.best_base(h)
    assert b.dtype == torch.uint8 and b.shape == (2, 3) and int(b[0, 0]) == 6 and int(b[0, 1]) == 120 and int(b[1, 2]) == 50 and int(b[0, 2]) == 0
    x = torch.zeros(4, 1, 2, dtype=torch.int16)  # exponent >> 1 is bits 8-14 of bf16
    x[:, 0, 0] = torch.tensor([20, 21, 22, 23]) << 8
    x[:, 0, 1] = (torch.tensor([20, 21, 22, 23]) << 8) | torch.tensor([-32768, 0, 0, 0], dtype=torch.int16)
    hist = kvcal.histogram(x)
    assert hist.shape == (1, 2, 128) and hist[0, 0, 20:24].tolist() == [1, 1, 1, 1] and int(hist.sum()) == 8  # (the sign bit is not the exponent)
    ex = kvcal.outside(x, torch.tensor([[20, 18]], dtype=torch.uint8))
    assert ex[:, 0, 0].tolist() == [False] * 4 and ex[:, 0, 1].tolist() == [False, False, False, False]  # (18 to 25 holds 20 to 23)
    assert kvcal.outside(x, torch.tensor([[22, 22]], dtype=torch.uint8))[:, 0, 0].tolist() == [True, True, False, False]
    ek = torch.zeros(32, 2, 128, dtype=torch.bool)
    ev = torch.zeros(32, 2, 128, dtype=torch.bool)
    ek[0, 0, :5] = True  # (head 0 of page 0: 5 in K)
    ev[3, 0, :2] = True  # (and 2 in V: 7 of a head-page, 4 past its three inline)
    ek[20, 1, :3] = True  # (page 1, head 1: 3, all inline)
    tot, need = kvcal.leaf_needs(ek, ev)
    assert tot.tolist() == [7, 3] and need.tolist() == [4, 0]
    # the sweep of the vocabulary: the entries a page of 16 of a token needs from the masks of its exceptions, the chunks and the layers over what a page holds, the tables' meta
    mk, mv = torch.zeros(2, 2, 128, dtype=torch.bool), torch.zeros(2, 2, 128, dtype=torch.bool)
    mk[0, 0, :5] = True  # (token 0: 7 exceptions of head 0 in K and V: 112 in a page, 109 past the inline three)
    mv[0, 0, :2] = True
    mk[1, 1, :1] = True  # (token 1: one: 16 in a page, 13 past the inline three)
    assert kvcal.sweep_needs(mk, mv).tolist() == [109, 13]
    need = torch.tensor([[0, 79, 1039], [2000, 15, 90], [32744, 32744, 32744]], dtype=torch.int32)  # (a token a row, a layer a column)
    chunks, over, maxneed = kvcal.sweep_stats(need, 15, 1039)
    assert [round(c, 3) for c in chunks.tolist()] == [5.667, 0.667, 0.0] and over.tolist() == [0, 1, 3] and maxneed.tolist() == [1039, 2000, 32744]  # (a page past what it holds is raw: no chunks)
    pl = kvcal.store_meta(chunks, over, maxneed, 15, 1039, 3, lambda i: i == 1, lambda i: f"t{i}")["store"]  # (token 1 is made from text; 2 only by a request of ids)
    assert pl["hard"] == 2 and pl["hard_text"] == 1 and pl["hard_text_ids"] == [1] and pl["hard_text_tokens"] == ["t1"] and pl["hard_ids"] == 1 and pl["hard_ids_sample"] == [2]
    assert pl["hard_worst_leaf_need"] == 32744 and pl["second_layers"] == 3 and pl["worst_token"] == 0 and pl["worst_chunks_per_layer_leaf"] == 5.6667 and pl["median_chunks_per_layer_leaf"] == 0.6667
    assert pl["top"] == [[0, 5.6667], [1, 0.6667], [2, 0.0]] and pl["swept"] == 3 and pl["bay"] == 15 and pl["capacity"] == 1039
    none = kvcal.store_meta(torch.tensor([1.0, 2.0]), torch.tensor([0, 0]), torch.tensor([10, 20]), 15, 1039, 2)["store"]
    assert none["hard"] == 0 and none["hard_worst_leaf_need"] == 0 and none["second_layers"] == 0 and none["worst_token"] == 1
    from glyd_gpu import vllm_kv as kv

    meta = lambda w, h=8: {"heads": h, "store": {"worst_chunks_per_layer_leaf": w}}
    assert kv.store_percent({}) == kv.STORE_PERCENT == kv.store_percent({"heads": 8}) and kv.store_percent(meta(2.03)) == 1.07  # (the worst repeated token takes 2.03 chunks a layer-page of 49,408 bytes, as many as the sweep found)
    assert kv.store_percent(meta(2.0, 4)) == 2.09 and kv.store_percent(meta(0.0)) == kv.STORE_MIN_PERCENT and kv.store_percent(meta(500.0)) == kv.STORE_MAX_PERCENT  # (a page of 4 heads is half as large: twice the share; between 0.5 and 15)
    assert kv.hard_tokens({}) == 0 and kv.hard_tokens({"store": {"hard": 0}}) == 0
    rm = lambda k, h=8, L=36: {"heads": h, "layers": L, "store": {"hard": k}}
    assert kv.second_percent(rm(0)) == 0.23 and kv.second_percent(rm(3)) == 1.38 and kv.second_percent(rm(3, 4, 28)) == 1.77  # (planes for 1.5 layer-pages of every block where some token fills a page, 0.25 where none)
    pm = {"heads": 8, "layers": 36, "store": pl}
    assert kv.store_percent(pm) == 2.98 and kv.hard_tokens(pm) == 2  # (the pool follows the worst token's 5.67 chunks a layer-page)
    ho = {"k": 0.0001, "v": 0.0002}
    assert kvcal.refusal({"held_out_rate": {"k": 0.002, "v": 0.0}}).startswith("the model's values compress too poorly") and "0.200%" in kvcal.refusal({"held_out_rate": {"k": 0.002, "v": 0.0}})
    sections = kvcal.corpus()
    kinds = {k for k, _, _ in sections}
    assert kinds == {"prose", "docs", "code", "table", "chat"} and len(sections) >= 10 and all(body for _, _, body in sections)
    assert 150_000 < sum(len(b) for _, _, b in sections) < 220_000 and len(kvcal.corpus_digest()) == 16
    import json

    assert all(isinstance(json.loads(b), list) for k, _, b in sections if k == "chat")
    cfg = {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 36}
    k1 = kvcal.key("m", None, cfg, 1, 1, 8192)
    assert k1 == kvcal.key("m", None, dict(cfg), 1, 1, 8192) and len({k1, kvcal.key("m2", None, cfg, 1, 1, 8192), kvcal.key("m", "abc", cfg, 1, 1, 8192), kvcal.key("m", None, cfg, 2, 1, 8192), kvcal.key("m", None, {**cfg, "num_hidden_layers": 28}, 1, 1, 8192), kvcal.key("m", None, cfg, 1, 1, 16384)}) == 6
    assert [kvcal.band(n) for n in (1, 4096, 4097, 8704, 16384, 32768, 32769, 40960, 40961)] == [4096, 4096, 8192, 16384, 16384, 32768, 40960, 40960, None]  # (the positions a table covers: the least tier that holds max_model_len)
    was = os.environ.pop("GLYD_KV_LONG", None)
    try:
        os.environ["GLYD_KV_LONG"] = "1"  # (the tiers not yet checked at those lengths: asked for)
        if compiled():  # (a release build does not read it)
            assert kvcal.band(65536) is None and kvcal.bands() == kvcal.BANDS
        else:
            assert [kvcal.band(n) for n in (40961, 65536, 65537, 131072, 131073)] == [65536, 65536, 131072, 131072, None] and kvcal.bands()[-1] == 131072
    finally:
        os.environ.pop("GLYD_KV_LONG", None)
        if was is not None:
            os.environ["GLYD_KV_LONG"] = was
    assert kvcal.band(65536) is None and kvcal.bands() == kvcal.BANDS
    assert kvcal.lengths(8192) == (8256, 8192) and kvcal.lengths(8192, 40960) == (8256, 8192) and kvcal.lengths(40960, 40960) == (40960, 40959) and kvcal.lengths(40960, 131072) == (41024, 40960)  # (a model's own window caps the engine)
    assert [kvcal.minutes(n) for n in (4096, 8192, 16384, 32768, 40960)] == ["1-2", "2-3", "3-4", "3-4", "3-4"]
    assert kvcal.lengths(32768, 40960.0) == (32832, 32768) and kvcal.lengths(40960, 40960.0) == (40960, 40959) and kvcal.lengths(4096, None) == (4160, 4096)
    ids = [("prose", "a", list(range(3000))), ("prose", "b", list(range(3000, 4000))), ("docs", "d", list(range(10000, 10700))), ("chat", "c", list(range(20000, 20300)))]
    (a, sa), (b, sb) = kvcal.sequences(ids, 2048)
    assert len(a) == len(b) == 2048 and sa[0] == ("chat", 0, 300)
    assert [k for k, _, _ in sa[:3]] == ["chat", "docs", "prose"] and [k for k, _, _ in sb[:3]] == ["prose", "docs", "chat"]  # (the kinds round and round, A by name, B in reverse)
    assert all(e > s and e - s <= kvcal.CHUNK for _, s, e in sa + sb) and sa[-1][2] == 2048 == sb[-1][2] and all(x[2] == y[1] for x, y in zip(sa, sa[1:]))  # (segments follow one another)
    assert {k for k, _, _ in sa} == {k for k, _, _ in sb} == {"prose", "docs", "chat"} and a != b
    (a2, sa2), (b2, sb2) = kvcal.sequences(ids, 5000)  # (a text of 5,000 tokens in all: A is all of it; B starts each kind half a part late, so its last 768 tokens are the text read again)
    (a3, sa3), (b3, sb3) = kvcal.sequences(ids, 9000)  # (shorter than the sequence: read again, so that the sequence is as long as its table says)
    assert len(a2) == len(b2) == 5000 and len(a3) == len(b3) == 9000 and a3[:5000] == a2 and b3[:5000] == b2 and sa3[: len(sa2) - 1] == sa2[:-1]
    assert all(x[2] == y[1] for x, y in zip(sa3, sa3[1:])) and sa3[-1][2] == 9000 and sb3[-1][2] == 9000 and all(e > s for _, s, e in sa3 + sb3) and sa3[0][1] == 0
    assert kvcal.sequences([("prose", "x", [])], 100) == [([], []), ([], [])]  # (no text at all: nothing to read, and no endless loop)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        kb = (torch.arange(2 * 8 * 128) % 121).to(torch.uint8).view(2, 8, 128).numpy()
        kvcal.save(path, ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"], kb, kb[::-1].copy(), {"bay": 127, "refused": False})  # (a file made under the earlier rule: its arena is not read)
        t = kvcal.Tables(path)
        assert t.layers[1].endswith("1.self_attn.attn") and (t.layer("model.layers.1.self_attn.attn")[0] == kb[1]).all() and (t.layer("model.layers.0.self_attn.attn")[1] == kb[1]).all() and not hasattr(t, "bay")
        try:
            t.layer("model.layers.2.self_attn.attn")
            raise AssertionError("a layer the file has none of")
        except KeyError:
            pass
    was = {k: os.environ.pop(k, None) for k in ("GLYD_CACHE", "XDG_CACHE_HOME")}
    try:
        os.environ["GLYD_CACHE"] = "/x"
        assert kvcal.cache_dir() == "/x/kv"
        del os.environ["GLYD_CACHE"]
        os.environ["XDG_CACHE_HOME"] = "/y"
        assert kvcal.cache_dir() == "/y/glyd/kv"
    finally:
        for k, v in was.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})


def test_kvcal_steps():
    """A sequence run in steps (a long one, past kvcal.CHUNKED positions, so that its activations leave room for its own cache) is counted as run whole: the segments' parts in each step, the pages
    that lie inside them (kvcal._install's hook)."""
    if vp is None:
        return print("test_kvcal_steps: skipped (no vLLM)")
    import types
    import torch
    from unittest import mock
    from vllm.model_executor.layers import attention as attn_mod
    from vllm.v1.attention.backend import AttentionType
    from glyd_gpu import kvcal

    hooks = []

    class FakeAttention:
        attn_type, sliding_window, layer_name, num_kv_heads, head_size, head_size_v = AttentionType.DECODER, None, "l0", 2, 8, 8

        def register_forward_pre_hook(self, hook, with_kwargs=False):
            hooks.append(hook)

    g = torch.Generator().manual_seed(1)
    n = 160
    k = (torch.randn(n, 2, 8, generator=g) * torch.exp(torch.randn(n, 2, 8, generator=g) * 3)).to(torch.bfloat16)
    v = (torch.randn(n, 2, 8, generator=g) * torch.exp(torch.randn(n, 2, 8, generator=g) * 3)).to(torch.bfloat16)
    segs = [("a", 0, 48), ("b", 48, 100), ("a", 100, 160)]  # (page 6, tokens 96 to 111, is in two of them: in none)
    fake = FakeAttention()
    model = types.SimpleNamespace(modules=lambda: [fake])
    saved = dict(kvcal._STATE)
    try:
        with mock.patch.object(attn_mod, "Attention", FakeAttention):
            assert kvcal._install(model) == ["l0"]
        hook = hooks[0]

        def run(mode, steps):
            kvcal._STATE.update(mode=mode, segs=segs, off=[0], hist={}, ex={}, ex_layer={}, leaves=[])
            a = 0
            for m in steps:
                hook(fake, (None, k[a : a + m], v[a : a + m]), {})
                a += m
            assert a == n and kvcal._STATE["off"] == [n]
            return kvcal._STATE

        whole = run("hist", [n])
        hist1 = {key: val.clone() for key, val in whole["hist"].items()}
        assert sum(int(x.sum()) for x in hist1.values()) == 2 * (48 + 52 + 60) * 16  # (K and V of every token of every segment, 2 heads of 8)
        parts = run("hist", [64, 64, 32])  # (steps of 64 tokens: their boundaries are page boundaries)
        assert hist1.keys() == parts["hist"].keys() and all(torch.equal(hist1[key], parts["hist"][key]) for key in hist1)
        odd = run("hist", [16, 80, 48, 16])
        assert all(torch.equal(hist1[key], odd["hist"][key]) for key in hist1)
        kvcal._STATE["loo"] = {(kind, w, 0): torch.full((2, 8), 61, dtype=torch.uint8) for kind in ("a", "b") for w in "kv"}
        r1 = run("eval", [n])
        ref = (dict(r1["ex"]), dict(r1["ex_layer"]), sorted((kd, int(p_), int(t), int(nd)) for kd, ps, ts, ns in r1["leaves"] for p_, t, nd in zip(ps, ts, ns)))
        assert ref[0]["a"][2] == 108 * 16 and ref[0]["b"][2] == 52 * 16 and 0 < ref[0]["a"][0] and sorted({p_ for _, p_, _, _ in ref[2]}) == [0, 16, 32, 48, 64, 80, 112, 128, 144]  # (nine pages whole in a segment)
        for steps in ([64, 64, 32], [16, 80, 48, 16], [48, 112]):
            r2 = run("eval", steps)
            got = (dict(r2["ex"]), dict(r2["ex_layer"]), sorted((kd, int(p_), int(t), int(nd)) for kd, ps, ts, ns in r2["leaves"] for p_, t, nd in zip(ps, ts, ns)))
            assert got == ref, steps
    finally:
        kvcal._STATE.clear()
        kvcal._STATE.update(saved)


if __name__ == "__main__":
    tests = [t for name, t in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(test.__name__, "ok")
    if vp is None:
        import inspect

        n = sum("if vp is None" in inspect.getsource(t) for t in tests)
        print(f"NOTE: vLLM is not installed, so {n} of these {len(tests)} tests did not run (GLYD_REQUIRE_VLLM=1 makes that a failure)")
