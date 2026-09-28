"""A model's Linear weights packed in both mma layouts, each unpacked and
compared bit for bit: bits a weight and bytes against bf16.

    python sizes.py MODEL_DIR [MODEL_DIR ...]"""
import json, os, sys, torch
from safetensors import safe_open
import glyd_gpu as g

for d in sys.argv[1:]:
    idx = os.path.join(d, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else [f for f in os.listdir(d) if f.endswith(".safetensors")]
    n = b16 = b12 = 0
    for fn in files:
        with safe_open(os.path.join(d, fn), "pt", device="cuda") as f:
            for k in f.keys():
                # a projection's matrix (q_proj.weight, in_proj_qkvz.weight ...), or all of a layer's
                # experts as one tensor (experts.gate_up_proj in Gemma 4 and Llama 4; input_linear and
                # output_linear in granite's checkpoints and JetMoE's), or an expert's (experts.0.w1.weight in
                # Mixtral's): a matrix an expert
                parts = k.split(".")
                if not ((parts[-1] == "weight" and len(parts) > 1 and ("proj" in parts[-2] or parts[-2] in ("input_linear", "output_linear")))
                        or parts[-1].endswith("_proj") or ("experts" in parts[:-1] and not parts[-1].endswith("bias"))):
                    continue
                t = f.get_tensor(k)
                if t.dtype != torch.bfloat16 or t.dim() not in (2, 3):
                    continue
                for w in (t if t.dim() == 3 else [t]):
                    if w.shape[0] % 64 or w.shape[1] % 16:
                        w = w.t().contiguous()  # an expert tensor may keep a matrix as (in, out)
                        if w.shape[0] % 64 or w.shape[1] % 16:
                            continue
                    for pack in (g.pack_mma, g.pack_mma12):
                        q = pack(w)
                        assert torch.equal(g.mma_unpack(q).view(torch.int16), w.view(torch.int16)), (d, k, pack.__name__)
                        if pack is g.pack_mma:
                            b16 += q.nbytes()
                        else:
                            b12 += q.nbytes()
                        del q
                    n += w.numel()
    print(f"{os.path.basename(d.rstrip('/'))}: {n / 1e9:.2f} B Linear weights, {2 * n / 1e9:.2f} GB in bf16; mma {b16 / 1e9:.2f} GB ({8 * b16 / n:.2f} bits, {100 * (1 - b16 / (2 * n)):.1f}% smaller), mma12 {b12 / 1e9:.2f} GB ({8 * b12 / n:.2f} bits, {100 * (1 - b12 / (2 * n)):.1f}% smaller); every tensor bit for bit", flush=True)
