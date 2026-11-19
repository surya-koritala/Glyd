"""The engine's residual dump against vLLM's (tools/ref_dump.py): per layer, the relative error and the largest difference, after attention and after the MLP.

    python dumpcmp.py ENGINE.bin VLLM.npy [VLLM2.npy]       (the engine's bf16 rows [attn_0, mlp_0, attn_1, mlp_1, ...]; vLLM's in call order)
"""
import sys

import numpy as np


def bf16_rows(path, hid=4096):
    a = np.fromfile(path, dtype=np.uint16).reshape(-1, hid).astype(np.uint32) << 16
    return a.view(np.float32)


def vllm_rows(path):
    r = np.load(path)  # call order: post_attention_layernorm 0 (attn_0), input_layernorm 1 (mlp_0), post_attention_layernorm 1 (attn_1), ..., model.norm (mlp_35)
    return r


def stats(a, b):
    d = a - b
    rel = np.linalg.norm(d) / max(np.linalg.norm(b), 1e-9)
    return rel, np.abs(d).max(), int(np.abs(d).argmax()), float(np.abs(b).max())


def main(engine, ref, ref2=None):
    e, v = bf16_rows(engine), vllm_rows(ref)
    v2 = vllm_rows(ref2) if ref2 else None
    n = min(len(e), len(v))
    print(f"{len(e)} engine rows, {len(v)} vLLM rows")
    for i in range(n):
        l, kind = i // 2, ("attention" if i % 2 == 0 else "mlp")
        rel, mx, ch, big = stats(e[i], v[i])
        line = f"layer {l:2d} after {kind:9s}: engine vs vLLM rel {rel:.5f} max {mx:9.3f} (channel {ch:4d}, vLLM's largest value {big:9.2f})"
        if v2 is not None and i < len(v2):
            r2, m2, _, _ = stats(v[i], v2[i])
            line += f" | vLLM vs vLLM2 rel {r2:.5f} max {m2:9.3f}"
        print(line)


if __name__ == "__main__":
    main(*sys.argv[1:4])
