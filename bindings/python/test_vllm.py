"""glyd.gpu's vLLM plugin, its logic that needs no GPU: the entry point's version rule; the options' precedence and
what is refused among them; a save's packs by vLLM's layer names; the pieces a checkpoint gave; and what is refused
in vLLM's config. Needs vLLM (glyd[vllm]) but for the version rule, else skipped.

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
    none = {"layout": None, "exact": None, "verify": None}
    assert vp._options({}, {}, none) == ("auto", False, False)
    assert vp._options({"layout": "MMA12"}, {"layout": "mma"}, dict(none, layout="mma")) == ("mma12", False, False)  # --additional-config first
    assert vp._options({}, {"exact": True}, dict(none, exact="0")) == ("auto", True, False)  # then the quantization_config
    assert vp._options({}, {}, {"layout": "mma", "exact": "Yes", "verify": "off"}) == ("mma", True, False)  # then the environment
    assert vp._options({"layout": "mma", "glyd": "0.26.0", "packs": "0123"}, {}, none)[0] == "mma"  # (resolve's own keys)
    raises(lambda: vp._options({"exat": True}, {}, none), "not an option")
    raises(lambda: vp._options("mma12", {}, none), "not an object of options")
    raises(lambda: vp._options({}, {}, dict(none, exact="ture")), "true or false")
    raises(lambda: vp._options({"layout": "fast"}, {}, none), "one of auto, mma, mma12")
    assert vp.GlydConfig.from_config({"quant_method": "glyd", "layout": "mma", "merge": True}).given == {"layout": "mma"}
    raises(lambda: vp.GlydConfig.from_config({"layot": "mma"}), "not an option")


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
    raises(lambda: vp._refusals(vc(), False, {"packs": {"m.experts.w13": {"experts": 8}}}), "mixture of experts' packs")
    was = os.environ.get("VLLM_BATCH_INVARIANT")
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    try:
        raises(lambda: vp._refusals(vc(), False, None), "VLLM_BATCH_INVARIANT")
        vp._refusals(vc(eager=True), True, None)  # (exact: vLLM's batch-invariant GEMM on the decoded weights)
    finally:
        os.environ.pop("VLLM_BATCH_INVARIANT") if was is None else os.environ.update(VLLM_BATCH_INVARIANT=was)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
