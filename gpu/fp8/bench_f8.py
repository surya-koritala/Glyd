"""The speed gate of the fused FP8 product (Task 5): our time against vLLM's own FP8 GEMM, as vLLM 0.30 installs and picks it on this GPU, at the
Linear shapes of the five 2026 FP8 checkpoints (tensor parallel 1, and 2, 4, 8 for the big ones) and 1, 8, 32, 64 tokens. CUDA-graph timed.

    python gpu/fp8/bench_f8.py [--samples DIR] [--models a,b] [--tp 1,2,4,8] [--ms 1,8,32,64] [--out bench.json] [--no-ours] [--no-unpack]

vLLM's side (its kernel objects, built and processed as Fp8LinearMethod / ModelOptLinearMethod build them; the kernel chosen by vLLM's own choice for
this GPU and shape, its class name recorded):
    block   128 x 128 fp32 scales (GLM-5.3-Flash, Qwen3.8-27B-FP8): init_fp8_linear_kernel(dynamic 1x128 groups, static 128x128 blocks): Marlin W8A16 on
            Ada; FlashInfer's block-scale GEMM (fewer than 32 tokens) / DeepGEMM (32 on) on Hopper.
    tensor  one fp32 scale, static input scale (Mistral-Small-4): init_fp8_linear_kernel(static per tensor): CUTLASS scaled_mm on both.
    mx      e8m0 per 1 x 32 (MiniMax-M3-MXFP8): init_mxfp8_linear_kernel(): Marlin MXFP8 (W8A16) on both.
    ds32    e8m0 per 32 x 32 (DeepSeek-V4.1-Flash): vLLM serves it as MXFP8 with each 32-row block's scales repeated for its rows (KMxfp8Static's
            loader): the same Marlin MXFP8 kernel at the same shape.
    channel fp32 a row, activations e4m3 a token (compressed-tensors FP8-dynamic: gemma-4-26B-A4B-it-FP8-dynamic): init_fp8_linear_kernel(dynamic
            per token, static per channel), the weight as CompressedTensorsW8A8Fp8 leaves it (a transposed view).
Ours: glyd_gpu.kernels.f8_linear on the pack of the same bytes (when the library has it); for reference, f8_unpack of the matrix (the prompt
path's decode) alone; and "route": vllm_fp8.py's way, the bytes vLLM's kernel is fed (its layer.weight's storage) packed, decoded before each product
into one scratch every copy shares, then vLLM's kernel on it (both in the graph).

vLLM's choice reads the model type (DeepGEMM is off on compute capability 10.x/12.x for qwen3_5_text and qwen3_5_moe_text), so each model's own
types stand in (MODEL_TYPES, from research/models-2026/models.json); --served first starts stock vLLM on each checkpoint (dummy weights, nothing
downloaded but its config) and keeps the "Selected X for ..." lines: a shape whose kernel is not one of them gets no vLLM time (the row's err says why).

Timing: each of C copies of the weights (C such that they hold at least 200 MB and 3 times the GPU's L2 together, with no cap, so that no product
finds its weights in L2, as in a model's layers) is a product in one CUDA graph, replayed 20 times after 3; a product's time is the graph's over C;
the best of 5 such. The weights are the model's own patches (tiled from the samples of fetch_samples.py) where --samples is given, else Gaussian
weights scaled per 128 x 128 block. Shapes vLLM refuses at tensor-parallel splits (block scales: a split N or K not a multiple of 128) are left out."""
import argparse, json, math, os, sys, time, traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "glyd-gpu-py"))

# (role, N, K, how tensor parallel splits it: "col" N, "row" K, "rep" not at all). N and K as vLLM serves them (q/k/v and gate/up merged).
MODELS = {
    "GLM-5.3-Flash": ("block", "zai-org/GLM-5.3-Flash", True, [
        ("qkv_a", 2048, 4096, "rep"), ("q_b", 16384, 1536, "col"), ("o", 4096, 16384, "row"),
        ("mlp_gate_up", 24576, 4096, "col"), ("mlp_down", 4096, 12288, "row"),
        ("expert_gate_up", 4096, 4096, "col"), ("expert_down", 4096, 2048, "row")]),
    "Qwen3.8-27B-FP8": ("block", "Qwen/Qwen3.8-27B-FP8", False, [
        ("qkv", 14336, 5120, "col"), ("o", 5120, 6144, "row"), ("gdn_in_qkvz", 16384, 5120, "col"), ("gdn_in_qkv", 10240, 5120, "col"),
        ("gdn_in_z", 6144, 5120, "col"), ("gdn_out", 5120, 6144, "row"), ("gate_up", 34816, 5120, "col"), ("down", 5120, 17408, "row")]),
    "MiniMax-M3-MXFP8": ("mx", "MiniMaxAI/MiniMax-M3-MXFP8", True, [
        ("qkv", 9216, 6144, "col"), ("o", 6144, 8192, "row"), ("mlp_gate_up", 24576, 6144, "col"), ("mlp_down", 6144, 12288, "row"),
        ("expert_gate_up", 6144, 6144, "col"), ("expert_down", 6144, 3072, "row"), ("index_q", 512, 6144, "rep")]),
    "Mistral-Small-4-119B-2603": ("tensor", "mistralai/Mistral-Small-4-119B-2603", True, [
        ("qkv_a", 1344, 4096, "rep"), ("q_b", 4096, 1024, "col"), ("kv_b", 6144, 256, "col"), ("o", 4096, 4096, "row"),
        ("expert_gate_up", 4096, 4096, "col"), ("expert_down", 4096, 2048, "row")]),
    "DeepSeek-V4.1-Flash": ("ds32", "deepseek-ai/DeepSeek-V4.1-Flash", True, [
        ("wq_a", 1280, 5120, "rep"), ("wkv", 512, 5120, "rep"), ("wq_b", 32768, 1280, "col"), ("wo_a", 8192, 4096, "col"),
        ("wo_b", 5120, 8192, "row"), ("indexer_wq_b", 4096, 1280, "col"), ("shared_w13", 4608, 5120, "col"), ("shared_w2", 5120, 2304, "row")]),
    "Qwen3.6-35B-A3B-FP8": ("block", "Qwen/Qwen3.6-35B-A3B-FP8", True, [
        ("qkv", 9216, 2048, "col"), ("o", 2048, 4096, "row"), ("gdn_in_qkvz", 12288, 2048, "col"), ("gdn_in_qkv", 8192, 2048, "col"),
        ("gdn_in_z", 4096, 2048, "col"), ("gdn_out", 2048, 4096, "row"),
        ("expert_gate_up", 1024, 2048, "col"), ("expert_down", 2048, 512, "row")]),  # (the shared expert's: the same shapes)
    "gemma-4-26B-A4B-it-FP8-dynamic": ("channel", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", False, [
        ("qkv_sliding", 8192, 2816, "col"), ("qk_global", 9216, 2816, "col"), ("o_sliding", 2816, 4096, "row"), ("o_global", 2816, 8192, "row"),
        ("mlp_gate_up", 4224, 2816, "col"), ("mlp_down", 2816, 2112, "row"), ("expert_gate_up", 1408, 2816, "col"), ("expert_down", 2816, 704, "row")]),
}


# (hf_config.model_type, hf_text_config.model_type) of each checkpoint, from research/models-2026/models.json (the FP8 twins share their originals')
MODEL_TYPES = {
    "GLM-5.3-Flash": ("glm5_next", "glm5_next_text"), "Qwen3.8-27B-FP8": ("qwen3_5", "qwen3_5_text"), "MiniMax-M3-MXFP8": ("minimax_m3_vl", "minimax_m3_vl"),
    "Mistral-Small-4-119B-2603": ("mistral3", "mistral4"), "DeepSeek-V4.1-Flash": ("deepseek_v41", "deepseek_v41_text"),
    "Qwen3.6-35B-A3B-FP8": ("qwen3_5_moe", "qwen3_5_moe_text"), "gemma-4-26B-A4B-it-FP8-dynamic": ("gemma4", "gemma4_text"),
}


def shapes(models, tps):
    out, seen = [], set()
    for m in models:
        kind, repo, big, rows = MODELS[m]
        for tp in tps:
            if tp > 1 and not big:
                continue
            for role, N, K, how in rows:
                if tp > 1 and how == "rep":
                    continue
                n, k = (N // tp, K) if how == "col" else (N, K // tp) if how == "row" else (N, K)
                if kind == "block" and tp > 1 and (n % 128 or k % 128):  # (vLLM's validate_fp8_block_shape refuses it)
                    continue
                key = (m, n, k)
                if key in seen:
                    continue
                seen.add(key)
                out.append(dict(model=m, kind=kind, repo=repo, role=role, tp=tp, N=n, K=k))
    return out


# ---- the weights ----

def gaussian_e4m3(N, K, gen):
    """e4m3 bytes of Gaussian weights quantized per 128 x 128 block (each block's largest to 448), as a block-scaled checkpoint's."""
    w = torch.randn(N, K, generator=gen, device="cuda", dtype=torch.float32)
    w *= torch.exp(torch.randn(N, 1, generator=gen, device="cuda") * 0.3)  # rows of other sizes
    nb, kb = -(-N // 128), -(-K // 128)
    pad = torch.zeros(nb * 128, kb * 128, device="cuda")
    pad[:N, :K] = w
    blk = pad.view(nb, 128, kb, 128)
    amax = blk.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    q = (blk / amax * 448.0).to(torch.float8_e4m3fn).view(nb * 128, kb * 128)[:N, :K]
    return q.contiguous().view(torch.uint8)


class Samples:
    """A model's sampled e4m3 matrices (fetch_samples.py's), whose 64 x 128 patches fill a matrix of any shape (patches drawn at random)."""

    def __init__(self, d):
        self.d = d
        self.man = json.load(open(os.path.join(d, "manifest.json"))) if d and os.path.exists(os.path.join(d, "manifest.json")) else []
        self.cache = {}

    def patches(self, repo):
        if repo not in self.cache:
            ps = []
            for e in self.man:
                if e["repo"] != repo:
                    continue
                a = np.load(os.path.join(self.d, e["file"]))
                O, K = a.shape
                O, K = O // 64 * 64, K // 128 * 128
                ps.append(a[:O, :K].reshape(O // 64, 64, K // 128, 128).transpose(0, 2, 1, 3).reshape(-1, 64, 128))
            self.cache[repo] = torch.from_numpy(np.concatenate(ps)).cuda() if ps else None
        return self.cache[repo]

    def fill(self, repo, N, K, gen):
        P = self.patches(repo)
        if P is None:
            return None
        rt, kt = -(-N // 64), -(-K // 128)
        idx = torch.randint(0, P.shape[0], (rt * kt,), generator=gen, device="cuda")
        m = P[idx].view(rt, kt, 64, 128).permute(0, 2, 1, 3).reshape(rt * 64, kt * 128)
        return m[:N, :K].contiguous()


def scales(kind, N, K, gen):
    """The weight's scales as the checkpoint holds them (kind's layout)."""
    if kind == "block":
        return (torch.rand(-(-N // 128), -(-K // 128), generator=gen, device="cuda") * 2e-3 + 1e-3).float()
    if kind == "tensor":
        return torch.tensor([2.5e-3], device="cuda", dtype=torch.float32)
    if kind == "channel":
        return (torch.rand(N, 1, generator=gen, device="cuda") * 2e-3 + 1e-3).float()
    if kind == "mx":
        return torch.randint(117, 124, (N, -(-K // 32)), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    if kind == "ds32":
        return torch.randint(117, 124, (-(-N // 32), -(-K // 32)), generator=gen, device="cuda", dtype=torch.int32).to(torch.uint8)
    raise ValueError(kind)


# ---- vLLM's FP8 GEMM, as installed ----

_VCFG = {}


def vllm_ctx(model=None):
    """vLLM's config context for its kernels' choice: the model's config types stand in (DeepGEMM's choice reads the text config's model type), bf16."""
    from types import SimpleNamespace
    from vllm.config import VllmConfig, set_current_vllm_config
    if model not in _VCFG:
        c = VllmConfig()
        if c.model_config is None:
            mt, tt = MODEL_TYPES.get(model, (None, None))
            object.__setattr__(c, "model_config", SimpleNamespace(dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type=tt), hf_config=SimpleNamespace(model_type=mt)))
        _VCFG[model] = c
    return set_current_vllm_config(_VCFG[model])


def served_kernels(models, timeout=900):
    """{model: [kernel class names]}: the "Selected X for ..." lines of stock vLLM started on each checkpoint here (dummy weights: only its config and
    tokenizer are downloaded; read up to "Model loading took", or the end: a checkpoint larger than the GPU stops at its allocation, after the lines)."""
    import re, signal, subprocess
    out = {}
    for m in models:
        repo = MODELS[m][1]
        code = (f"from vllm import LLM; LLM(model={repo!r}, load_format='dummy', enforce_eager=True, max_model_len=1024, max_num_seqs=1, "
                f"gpu_memory_utilization=0.85, limit_mm_per_prompt={{'image': 0, 'video': 0}})")
        log = []
        p = subprocess.Popen(["timeout", str(timeout), sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                             env=dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0"))  # (the engine in this process: the kill frees its GPU memory)
        try:
            for line in p.stdout:
                log.append(line)
                if "Model loading took" in line:
                    break
        finally:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        sel = sorted({k for k, _ in re.findall(r"Selected (\w+) for (\w+)", "".join(log))})
        out[m] = dict(kernels=sel, tail=[l.rstrip()[:300] for l in log if "Selected" in l or "DeepGemm" in l or "Error" in l][-12:])
        print(f"# served {m}: {sel or 'no Selected line'}", flush=True)
    return out


def served_ok(res, model, vname):
    """None where vname (bench's kernel class, maybe 'X[...]') is one stock vLLM selected for the model, or there is nothing to check against (no --served, or
    its run printed no line: res["served"] says so), else why not."""
    sv = (res.get("served") or {}).get(model)
    if not sv or not sv["kernels"]:
        return None
    return None if vname.split("[")[0] in sv["kernels"] else f"vLLM kernel {vname}, a served run selects {'/'.join(sv['kernels'])}"


def vllm_report():
    """What decides vLLM's FP8 kernels on this GPU (DeepGEMM and FlashInfer present?)."""
    out = {}
    for name, call in (("has_deep_gemm", "vllm.utils.import_utils:has_deep_gemm"), ("deep_gemm_supported", "vllm.utils.deep_gemm:is_deep_gemm_supported"),
                       ("deep_gemm_e8m0", "vllm.utils.deep_gemm:is_deep_gemm_e8m0_used"), ("flashinfer_blockscale", "vllm.utils.flashinfer:is_flashinfer_fp8_blockscale_gemm_supported"),
                       ("cutlass_block", "vllm.model_executor.layers.quantization.utils.w8a8_utils:cutlass_block_fp8_supported")):
        try:
            import importlib
            mod, fn = call.split(":")
            with vllm_ctx():
                out[name] = bool(getattr(importlib.import_module(mod), fn)())
        except Exception as e:  # noqa: BLE001
            out[name] = f"{type(e).__name__}: {e}"[:120]
    return out


def vllm_runner(kind, w8, sc, N, K, model=None):
    """(fn(x) -> y [M, N] bf16, the kernel's class name, the layer it reads): vLLM's own product for weights w8 [N, K] (uint8) and scales sc, processed as vLLM does at
    load, under the model's config types."""
    import torch.nn as nn
    from vllm.model_executor.kernels import linear as KL

    layer = nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = N, K
    layer.logical_widths, layer.orig_dtype = [N], torch.bfloat16
    P = lambda t: nn.Parameter(t, requires_grad=False)
    with vllm_ctx(model):
        if kind == "block":
            from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key
            layer.weight = P(w8.clone().view(torch.float8_e4m3fn))
            layer.weight_scale_inv = P(sc.clone())
            layer.weight_block_size = [128, 128]
            layer.input_scale = None
            kern = KL.init_fp8_linear_kernel(activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
                                             weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
                                             input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(N, K), module_name="bench")
        elif kind == "tensor":
            from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym
            layer.weight = P(w8.clone().view(torch.float8_e4m3fn).t())  # [K, N], as Fp8LinearMethod leaves it (a view of the [N, K] rows)
            layer.weight_scale = P(sc.clone().reshape(()))
            layer.input_scale = P(torch.tensor(0.05, device="cuda", dtype=torch.float32))
            layer.input_scale_ub = None
            kern = KL.init_fp8_linear_kernel(activation_quant_key=kFp8StaticTensorSym, weight_quant_key=kFp8StaticTensorSym,
                                             input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(N, K), module_name="bench")
        elif kind == "channel":  # (as CompressedTensorsW8A8Fp8 with channel weights and dynamic token activations leaves the layer)
            from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym
            layer.weight = P(w8.clone().view(torch.float8_e4m3fn).t())
            layer.weight_scale = P(sc.clone())
            layer.input_scale = None
            layer.input_scale_ub = None
            kern = KL.init_fp8_linear_kernel(activation_quant_key=kFp8DynamicTokenSym, weight_quant_key=kFp8StaticChannelSym,
                                             input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(N, K), module_name="bench")
        else:  # mx, ds32 (ds32: its 32 x 32 scales repeated for each row, as vLLM's MXFP8 loader does)
            s = sc if kind == "mx" else sc.repeat_interleave(32, dim=0)[:N]
            layer.weight = P(w8.clone().view(torch.float8_e4m3fn))
            layer.weight_scale = P(s.clone().contiguous())
            layer.weight_block_size = [1, 32]
            kern = KL.init_mxfp8_linear_kernel()
        kern.process_weights_after_loading(layer)
    name = type(kern).__name__
    if hasattr(kern, "base") and hasattr(kern, "fallback"):  # the dynamic Hopper kernel: its torch.cond (fewer than 32 tokens: base) taken here, by M
        def fn(x):
            with vllm_ctx(model):
                return (kern.base if x.shape[0] < 32 else kern.fallback).apply_weights(layer, x)
        return fn, f"{name}[{type(kern.base).__name__}<32<={type(kern.fallback).__name__}]", layer

    def fn(x):
        with vllm_ctx(model):
            return kern.apply_weights(layer, x)
    return fn, name, layer


def route_runner(fn, layer, scratch):
    """vllm_fp8.py's route for vLLM's product fn on layer: the bytes of layer.weight (as vLLM's kernel is fed them) packed, the weight made a view of
    scratch (a dict holding the one buffer every copy shares, grown here, before any graph), and each product the decode into it, then fn."""
    from glyd_gpu import kernels as g
    from glyd_gpu.vllm_fp8 import _matrix
    mat, (dtype, shape, stride) = _matrix(layer.weight.data, "bench")
    p = g.f8_pack(mat)
    n = mat.numel()
    if scratch.get("buf") is None or scratch["buf"].numel() < n:
        scratch["buf"] = torch.empty(n, dtype=torch.uint8, device="cuda")
    buf = scratch["buf"]
    layer.weight.data = buf[:n].view(dtype).as_strided(shape, stride)  # (its own memory: the pack's now)
    out = buf[:n].view(mat.shape)

    def f(x):
        g.f8_unpack_into(p, out)
        return fn(x)
    return f


# ---- ours ----

def ours_runner(kind, w8, sc, N, K):
    """(fn(x), the pack): f8_linear on the pack of w8."""
    from glyd_gpu import kernels as g
    p = g.f8_pack(w8)
    if not hasattr(g, "f8_linear"):
        return None, p
    if kind == "channel":
        sr = sc.reshape(-1).contiguous()
        return (lambda x: g.f8_linear(p, "channel", sr, x)), p
    s = sc.reshape(()) if kind == "tensor" else sc
    return (lambda x: g.f8_linear(p, kind, s, x)), p


# ---- numerics: each product against an exact one of the same dequantized weights ----

def dequant(w8, kind, sc):
    """W as the checkpoint means it, float64 on the GPU."""
    N, K = w8.shape
    W = w8.view(torch.float8_e4m3fn).to(torch.float64)
    if kind == "block":
        s = sc.to(torch.float64).repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    elif kind == "tensor":
        s = sc.to(torch.float64).reshape(())
    elif kind == "channel":
        s = sc.to(torch.float64).reshape(N, 1)
    else:
        e = (2.0 ** (sc.to(torch.float64) - 127))
        s = (e if kind == "mx" else e.repeat_interleave(32, 0)).repeat_interleave(32, 1)[:N, :K]
    return W * s


def _need_clear():
    """The library's workspace sizes forgotten (a development build's plan, and so its workspace, is the tuning variables')."""
    try:
        from glyd_gpu import _lib
        _lib._need.cache_clear()
    except Exception:  # noqa: BLE001
        pass


def numerics(a, smp, gen, res):
    """vLLM's FP8 kernel and ours against an exact product of the same weights (real samples' bytes, each kind of scales), 1 to 64 tokens: the error's
    largest share of the sum of |x w| for its output, and its RMS over the RMS of the exact outputs."""
    rows = []
    for m in a.models.split(","):
        kind, repo, _, shp = MODELS[m]
        for role, N, K, _ in shp[:3]:
            w8 = smp.fill(repo, N, K, gen)
            w8 = w8 if w8 is not None else gaussian_e4m3(N, K, gen)
            sc = scales(kind, N, K, gen)
            W = dequant(w8, kind, sc)
            try:
                fv, vname, _ = vllm_runner(kind, w8, sc, N, K, m)
                why = served_ok(res, m, vname)
                if why:
                    fv, vname = None, why
            except Exception as e:  # noqa: BLE001
                fv, vname = None, f"failed: {type(e).__name__}: {str(e)[:200]}"
            fo = ours_runner(kind, w8, sc, N, K)[0]
            for M in (1, 8, 32, 64):
                x = (torch.randn(M, K, device="cuda", generator=gen) * (1 + 3 * torch.rand(M, 1, device="cuda", generator=gen))).to(torch.bfloat16)
                X = x.to(torch.float64)
                ref, mag = X @ W.T, X.abs() @ W.abs().T
                row = dict(model=m, role=role, N=N, K=K, M=M, kind=kind, vllm_kernel=vname)
                for nm, f, env in [("vllm", fv, {}), ("ours", fo, {})] + [(f"ours[{sp}]", fo, dict(kv.split("=", 1) for kv in sp.split(","))) for sp in a.variant]:
                    if f is None:
                        continue
                    os.environ.update(env)
                    _need_clear()
                    try:
                        y = f(x).to(torch.float64)
                    except Exception as e:  # noqa: BLE001
                        row[nm + "_err"] = f"{type(e).__name__}: {str(e)[:200]}"
                        continue
                    finally:
                        for k in env:
                            del os.environ[k]
                        _need_clear()
                    err = (y - ref).abs()
                    row[nm + "_max"] = float((err / mag.clamp_min(1e-30)).max())
                    row[nm + "_rms"] = float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
                rows.append(row)
                var = "".join(f" | ours[{sp}]: max {row.get(f'ours[{sp}]_max', float('nan')):.2e} rms {row.get(f'ours[{sp}]_rms', float('nan')):.2e}" for sp in a.variant)
                print(f"numerics {m:<26} {role:<14} {N:>6} x {K:<6} M={M:<3} vLLM ({vname[:40]}): max {row.get('vllm_max', float('nan')):.2e} rms {row.get('vllm_rms', float('nan')):.2e}"
                      f" | ours: max {row.get('ours_max', float('nan')):.2e} rms {row.get('ours_rms', float('nan')):.2e}{var}", flush=True)
            del fv, fo, W
            torch.cuda.empty_cache()
    return rows


# ---- timing ----

def time_graph(fns, x, reps=20, best_of=5):
    """Microseconds a call of the fns (each on its own weights) takes, one after the other in one CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        for f in fns:
            f(x)
    for _ in range(3):
        gph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(best_of):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            gph.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / reps / len(fns))
    del gph
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("GLYD_FP8_SAMPLES", ""))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tp", default="1,2,4,8")
    ap.add_argument("--ms", default="1,8,32,64")
    ap.add_argument("--out", default="bench_f8.json")
    ap.add_argument("--no-ours", action="store_true")
    ap.add_argument("--no-unpack", action="store_true")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--footprint", type=float, default=200e6, help="bytes of weights the copies hold together (at least 3 times the GPU's L2)")
    ap.add_argument("--served", action="store_true", help="stock vLLM started on each checkpoint first; its kernels checked against (served_kernels)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-route", action="store_true")
    ap.add_argument("--variant", action="append", default=[], help="NAME=VALUE[,NAME=VALUE]: ours timed again with these tuning variables set (a development library's); again for more")
    ap.add_argument("--numerics", action="store_true", help="the error of vLLM's kernel and ours against an exact product, then the timing")
    a = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    gen = torch.Generator(device="cuda").manual_seed(0)
    smp = Samples(a.samples)
    dev = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()
    ms = [int(v) for v in a.ms.split(",")]
    sh = shapes(a.models.split(","), [int(v) for v in a.tp.split(",")])
    if a.limit:
        sh = sh[: a.limit]
    print(f"# {dev} (sm_{cc[0]}{cc[1]}), {len(sh)} shapes, tokens {ms}, samples: {'yes' if smp.man else 'no (Gaussian)'}", flush=True)
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    res = dict(gpu=dev, cc=list(cc), l2=l2, rows=[])
    if os.path.exists(a.out):
        try:
            res = json.load(open(a.out))
        except Exception:
            pass
    if a.served and not res.get("served"):
        res["served"] = served_kernels(list(dict.fromkeys(s["model"] for s in sh)))
        json.dump(res, open(a.out, "w"), indent=0)
    if not a.no_vllm:
        res["vllm"] = vllm_report()
        print(f"# vLLM: {res['vllm']}", flush=True)
    if a.numerics and not res.get("numerics"):
        res["numerics"] = numerics(a, smp, gen, res)
        json.dump(res, open(a.out, "w"), indent=0)
    done = {(r["model"], r["N"], r["K"], r["M"]) for r in res["rows"]}
    for s in sh:
        N, K, kind = s["N"], s["K"], s["kind"]
        if all((s["model"], N, K, M) in done for M in ms):
            continue
        C = max(2, math.ceil(max(a.footprint, 3 * l2) / (N * K)))  # (no cap: a small shape's copies hold 3 L2s too)
        t0 = time.time()
        try:
            ws = []
            for c in range(C):
                w8 = smp.fill(s["repo"], N, K, gen)
                ws.append((w8 if w8 is not None else gaussian_e4m3(N, K, gen), scales(kind, N, K, gen)))
            vl, layers, ours, unp, packs, vname, err = [], [], [], [], [], "", ""
            if not a.no_vllm:
                try:
                    for w8, sc in ws:
                        f, vname, lay = vllm_runner(kind, w8, sc, N, K, s["model"])
                        vl.append(f), layers.append(lay)
                    why = served_ok(res, s["model"], vname)
                    if why:  # (not the kernel stock vLLM serves this checkpoint with: no vLLM or route time)
                        err, vl, layers = why, [], []
                except Exception as e:
                    err = f"vllm: {type(e).__name__}: {e}"
                    traceback.print_exc()
                    vl, layers = [], []
            if not (a.no_ours and a.no_unpack):
                for w8, sc in ws:
                    f, p = ours_runner(kind, w8, sc, N, K)
                    packs.append(p)
                    if f is not None and not a.no_ours:
                        ours.append(f)
            if not a.no_unpack and packs:
                from glyd_gpu import kernels as g
                outs = [torch.empty(N, K, dtype=torch.uint8, device="cuda") for _ in range(min(2, C))]
                unp = [(lambda x, p=p, o=outs[i % len(outs)]: g.f8_unpack_into(p, o)) for i, p in enumerate(packs)]
            bpv = sum(p.bits_per_value() for p in packs) / len(packs) if packs else None
            del ws
            xs = {M: torch.randn(M, K, device="cuda", dtype=torch.bfloat16) for M in ms}
            rows = []
            for M in ms:
                x = xs[M]
                row = dict(s, M=M, copies=C, vllm_kernel=vname, bits=bpv, err=err)
                for nm, fs in (("vllm_us", vl), ("ours_us", ours), ("unpack_us", unp)):
                    if fs:
                        try:
                            row[nm] = time_graph(fs, x)
                        except Exception as e:
                            row[nm] = None
                            row["err"] = (row["err"] + f" {nm}: {type(e).__name__}: {e}").strip()
                            traceback.print_exc()
                if row.get("vllm_us") and row.get("ours_us"):
                    row["ratio"] = row["ours_us"] / row["vllm_us"]
                for spec in a.variant if ours else []:  # ours again with a development build's tuning variables set (its plan is read at each call, kept by the graph)
                    from glyd_gpu import _lib
                    env = dict(kv.split("=", 1) for kv in spec.split(","))
                    os.environ.update(env)
                    _lib._need.cache_clear()  # (the workspace's size is the plan's)
                    try:
                        row.setdefault("variants", {})[spec] = time_graph(ours, x)
                    except Exception as e:
                        row["err"] = (row["err"] + f" variant {spec}: {type(e).__name__}: {e}").strip()
                    finally:
                        for k in env:
                            del os.environ[k]
                        _lib._need.cache_clear()
                rows.append(row)
            if layers and not a.no_route:  # (last: it takes vLLM's layers over, their weights views of the scratch)
                try:
                    scratch = {}
                    rt = [route_runner(f, lay, scratch) for f, lay in zip(vl, layers)]
                    for row in rows:
                        row["route_us"] = time_graph(rt, xs[row["M"]])
                        if row.get("vllm_us"):
                            row["route_ratio"] = row["route_us"] / row["vllm_us"]
                    del rt, scratch
                except Exception as e:
                    for row in rows:
                        row["err"] = (row["err"] + f" route: {type(e).__name__}: {e}").strip()
                    traceback.print_exc()
            for row in rows:
                M = row["M"]
                res["rows"].append(row)
                var = "".join(f" ({k}: {v:.1f})" for k, v in row.get("variants", {}).items())
                print(f"{s['model']:<26} {s['role']:<15} tp{s['tp']} {N:>6} x {K:<6} M={M:<3} vllm {row.get('vllm_us') or float('nan'):8.1f} us ({vname})"
                      f"  ours {row.get('ours_us') or float('nan'):8.1f}{var}  unpack {row.get('unpack_us') or float('nan'):8.1f}"
                      f"  route {row.get('route_us') or float('nan'):8.1f}  ratio {row.get('ratio', float('nan')):.3f} {row['err'][:120]}", flush=True)
            del vl, layers, ours, unp, packs
        except Exception as e:
            traceback.print_exc()
            res["rows"].append(dict(s, M=-1, err=f"{type(e).__name__}: {e}"))
        torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=0)
        print(f"   ({time.time() - t0:.1f} s, {C} copies)", flush=True)
    rows = [r for r in res["rows"] if r.get("ratio")]
    if rows:
        ok = sum(r["ratio"] <= 1.0 for r in rows)
        print(f"gate: {ok} of {len(rows)} at most vLLM's time; worst {max(r['ratio'] for r in rows):.3f}")
    for spec in a.variant:
        rs = [r["variants"][spec] / r["vllm_us"] for r in res["rows"] if r.get("vllm_us") and spec in r.get("variants", {})]
        if rs:
            print(f"gate with {spec}: {sum(v <= 1.0 for v in rs)} of {len(rs)}; worst {max(rs):.3f}")
    rs = [r for r in res["rows"] if r.get("route_us") and r.get("ours_us")]
    if rs:
        print(f"ours at most the route's time: {sum(r['ours_us'] <= r['route_us'] for r in rs)} of {len(rs)}")


if __name__ == "__main__":
    main()
