"""bf16 tensors of a file, by role: the exponent entropies and what Glyd's bf16 layouts spend on them (counted from the
stored bits; nothing is lossy). This is the bf16 part of Kimi K3 and gpt-oss that the lossless codecs in gpu/ already handle.

    python bf16.py FILE.safetensors OUT.json [--min 1000000]

Per role (attention, shared experts, router, ...): n weights; sum of n * entropy (a table a tensor) of the whole 16-bit
word, of the exponent field, the share outside the best window of 16 exponents from an even exponent (mma12's
exceptions), the bits of the tiered code (3 tiers of 2-bit digits, then the exponent raw) and of 3-bit-plus-escape.
"""
import json, re, sys, time
import numpy as np
from st import ST
from ent import H, Hc, fast_bits, tiered_bits

BLOCK = 1 << 26


def role(name):
    n = name
    for key, r in (("embed_tokens", "embed"), ("lm_head", "lm_head"), ("shared_expert", "shared_experts"),
                   ("routed_expert_", "expert_latent_proj"), ("block_sparse_moe.gate", "router"), (".router.", "router"),
                   ("self_attn", "attention"), ("vision", "vision"), ("mm_projector", "vision"), (".mlp.", "dense_mlp")):
        if key in n:
            return r
    return "other"


def counts(w):
    w = w.reshape(-1)
    word = np.zeros(65536, np.int64)
    for i in range(0, w.size, BLOCK):
        word += np.bincount(w[i:i + BLOCK], minlength=65536)
    return word


def tensor_stats(word):
    n = int(word.sum())
    cube = word.reshape(2, 256, 128)  # sign, exponent, mantissa: the word is s eeeeeeee mmmmmmm
    ex = cube.sum((0, 2))
    hword = H(word) + ((word > 0).sum() - 1) / (2 * n * np.log(2))
    hexp = H(ex)
    # mma12: a window of 16 exponents from an even one holding the most weights; the rest are exceptions
    best = max(ex[b:b + 16].sum() for b in range(0, 241, 2))
    fb, fe = fast_bits(ex, 8, 3)
    tb, tr = tiered_bits(ex, 8, 3)
    return {"n": n, "H_word": hword, "H_exp": hexp, "exc12": 1 - best / n, "fast3_bits": fb + 8, "fast3_esc": fe,
            "tier3_bits": tb + 8, "tier3_raw": tr, "ex": ex.tolist()}


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    mn = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 1_000_000
    f = ST(path)
    res, t0 = {}, time.time()
    for k in f.keys():
        dt, sh = f.info(k)
        n = int(np.prod(sh))
        r = role(k)
        a = res.setdefault(r, {"tensors": 0, "small_bytes": 0, "big_n": 0, "sums": {}})
        if dt != "BF16" or n < mn:
            a["small_bytes"] += n * {"BF16": 2, "F32": 4}.get(dt, 1)
            continue
        s = tensor_stats(counts(f.get(k)))
        a["tensors"] += 1
        a["big_n"] += n
        for key in ("H_word", "H_exp", "exc12", "fast3_bits", "fast3_esc", "tier3_bits", "tier3_raw"):
            a["sums"][key] = a["sums"].get(key, 0.0) + n * s[key]
    res["_meta"] = {"file": path.split("/")[-1], "seconds": time.time() - t0, "min_elements": mn}
    json.dump(res, open(out, "w"), indent=1)
    for r, a in res.items():
        if r == "_meta" or not a["big_n"]:
            continue
        s = a["sums"]
        print(f"{r:20s} {a['tensors']:4d} tensors {a['big_n'] / 1e6:9.1f} M weights  H(word) {s['H_word'] / a['big_n']:.3f}  H(exp) {s['H_exp'] / a['big_n']:.3f}"
              f"  exc12 {100 * s['exc12'] / a['big_n']:.3f}%  fast3 {s['fast3_bits'] / a['big_n']:.3f}  tiered {s['tier3_bits'] / a['big_n']:.3f}")
