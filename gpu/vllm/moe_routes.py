"""A mixture of experts' layer in vLLM with --quantization glyd, its two routes by tokens a step: "grouped" (the
library's grouped products on the packs) against "decoded" (the experts the tokens are routed to decoded, then vLLM's
Triton MoE kernel on them, exact mode's way), or bf16's own (vLLM's Triton kernel on the bf16 experts,
with --bf16). Each T: T random tokens, each routed to the layer's k experts at random (the top k of random logits),
each route's GPU time a call (CUDA events, the median of REPS after a warm-up), on the model's first MoE layer; and the
two routes' outputs against each other (the largest relative difference).

    VLLM_ENABLE_V1_MULTIPROCESSING=0 GLYD_MOE_DECODE_MIN=1 python moe_routes.py OUT.json [MODEL] [--bf16]

GLYD_MOE_DECODE_MIN=1 makes the decoded route's kernel and scratch buffer at load; --bf16 times bf16's layer instead.
Env: TOKENS ("1 2 4 8 16 32 64 128 256 512 1024 2048 4096"), REPS (20), UTIL (0.85)."""
import json
import os
import sys

import torch


def _time(f, reps):
    for _ in range(3):
        f()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        f()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def _layer(model):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    return next(m for m in model.modules() if isinstance(m, RoutedExperts))


def _measure(model):
    """Each T's GPU ms a call by route (the packed layer's: grouped and decoded; bf16's: its own), and the two routes'
    largest relative difference."""
    torch.manual_seed(0)
    m = _layer(model)
    qm, E, k = m.quant_method, m.global_num_experts, m.top_k
    H = m.moe_config.hidden_dim
    out = {"layer": m.layer_name, "E": E, "k": k, "H": H, "I": m.moe_config.intermediate_size_per_partition, "method": type(qm).__name__, "rows": []}
    reps = int(os.environ.get("REPS", "20"))
    for T in [int(t) for t in os.environ.get("TOKENS", "1 2 4 8 16 32 64 128 256 512 1024 2048 4096").split()]:
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1
        logits = torch.randn(T, E, device="cuda")
        w, ids = torch.topk(torch.softmax(logits, dim=-1), k, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).to(torch.float32)
        ids = ids.to(getattr(qm, "topk_indices_dtype", None) or torch.int32)
        row = {"T": T}
        if getattr(m, "glyd_moe", None) is None:  # bf16's layer
            row["bf16_ms"] = _time(lambda: qm.apply(m, x, w, ids, None, None), reps)
        else:
            keep = qm.decode_min
            try:
                qm.decode_min = 1 << 30
                row["grouped_ms"] = _time(lambda: qm.apply(m, x, w, ids, None, None), reps)
                yg = qm.apply(m, x, w, ids, None, None).float()
                if qm.ref is not None:
                    qm.decode_min = 0
                    row["decoded_ms"] = _time(lambda: qm.apply(m, x, w, ids, None, None), reps)
                    yd = qm.apply(m, x, w, ids, None, None).float()
                    row["rel_diff"] = ((yg - yd).abs().max() / yd.abs().max()).item()
            finally:
                qm.decode_min = keep
        out["rows"].append(row)
    return out


def main():
    out = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    model = args[0] if args else "ibm-granite/granite-3.1-3b-a800m-instruct"
    from vllm import LLM

    bf16 = "--bf16" in sys.argv
    llm = LLM(model=model, quantization=None if bf16 else "glyd", dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("UTIL", "0.85")), max_model_len=4096, enforce_eager=True, seed=0)
    res = llm.apply_model(_measure)[0]
    res.update(model=model, mode="bf16" if bf16 else "glyd", gpu=torch.cuda.get_device_name(), glyd=(llm.llm_engine.vllm_config.additional_config or {}).get("glyd"))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    for r in res["rows"]:
        print(r, flush=True)


if __name__ == "__main__":
    main()
