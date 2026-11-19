"""A per-tensor FP8 model directory from a block-scaled FP8 slice (make_slice.py's), to try vLLM's per-tensor FP8 path (CUTLASS on Ada and
Hopper: the weight a transposed [K, N] view, a fused Linear's shards requantized to the largest scale) on the same real weights: each FP8
weight of the slice is dequantized with its block scales and quantized again to e4m3 with one scale for the tensor (amax / 448), written as
`weight` and a 0-d float32 `weight_scale`; the config's quantization_config becomes {"quant_method": "fp8", "activation_scheme": "dynamic"}.
Everything else is copied. Not a checkpoint anyone released: the weights are the real ones, the scales are this script's, for checking one
vLLM path against another.

    python pertensor_slice.py SLICE_DIR OUT_DIR       (needs torch and safetensors: vLLM's venv has them)"""
import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0


def main(src, out):
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*")):
        if not f.endswith((".safetensors", "config.json", ".index.json", "SLICE.json")) and os.path.isfile(f):
            shutil.copy(f, out)
    cfg = json.load(open(os.path.join(src, "config.json")))
    q = cfg["quantization_config"]
    block = q.get("weight_block_size")
    assert q.get("quant_method") == "fp8" and block, "the slice's quantization_config is a block-scaled fp8 one"
    cfg["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", **({"modules_to_not_convert": q["modules_to_not_convert"]} if "modules_to_not_convert" in q else {})}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    weight_map, n_requant = {}, 0
    for path in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        tensors = {}
        with safe_open(path, "pt", device="cpu") as f:
            keys = list(f.keys())
            for k in keys:
                if k.endswith(".weight_scale_inv"):
                    continue  # (replaced by the per-tensor scale)
                t = f.get_tensor(k)
                sk = k[: -len(".weight")] + ".weight_scale_inv" if k.endswith(".weight") else None
                if t.dtype == torch.float8_e4m3fn and sk in keys:
                    s = f.get_tensor(sk).float()  # [N/128, K/128]
                    w = t.float().cuda() if torch.cuda.is_available() else t.float()
                    s = s.to(w.device).repeat_interleave(block[0], 0)[: w.shape[0]].repeat_interleave(block[1], 1)[:, : w.shape[1]]
                    w = w * s
                    scale = (w.abs().max() / FP8_MAX).clamp(min=1e-12)
                    tensors[k] = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).cpu()
                    tensors[k[: -len(".weight")] + ".weight_scale"] = scale.float().reshape(()).cpu()
                    n_requant += 1
                    del w, s
                else:
                    tensors[k] = t
        name = os.path.basename(path)
        save_file(tensors, os.path.join(out, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in tensors})
        print(f"{name}: {len(tensors)} tensors", flush=True)
    json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    json.dump({"from": src, "per_tensor_requantized": n_requant}, open(os.path.join(out, "SLICE.json"), "w"))
    print(f"{out}: {n_requant} weights requantized to one scale a tensor")


if __name__ == "__main__":
    main(*sys.argv[1:3])
