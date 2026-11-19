"""The lossless KV cache's kernels at head size 128, their outputs as hashes, for two builds of the library on the same inputs (that a change for head size 256 leaves
head size 128's bits as they were):

    GLYD_GPU_LIB=A.so python kv_bits.py OUT_A.json;  GLYD_GPU_LIB=B.so python kv_bits.py OUT_B.json;  python kv_bits.py compare OUT_A.json OUT_B.json

The pages written (8 KV heads, natural values with outliers, heavy tokens whose pages are raw, an escape pool and a raw pool), read back (unpack), decoded into the staging cache (gather), and decode attention
at 1, 4 and 8 query heads a KV head, the library's splits and 1, 7 and 64, the merge in the kernel or apart, the next page ahead or not: each output's sha256, and the status words'.
Run as check_vllm.py is."""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(out):
    import torch
    import check_vllm as cv
    from glyd_gpu import _lib, kernels

    kernels.lib()
    h = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    res = {}
    H, A = 8, _lib.kv_free_bay(8)
    lens = [1, 15, 17, 100, 255, 700, 2049, 4100]
    npg = [(L + 15) // 16 for L in lens]
    T = sum(npg) * 16
    key, value = cv.kv_natural(T, H, 31, "k", 0.002, (60, 180)), cv.kv_natural(T, H, 32, "v", 0.002, (60, 180))
    hv = torch.rand(T, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5)) < 0.02
    key[hv], value[hv] = cv.kv_natural(T, H, 81, "k", 1.0, (80, 175))[hv], cv.kv_natural(T, H, 82, "v", 1.0, (80, 175))[hv]
    kb, vb = cv.kv_bases(cv.kv_natural(4096, H, 33, "k")), cv.kv_bases(cv.kv_natural(4096, H, 34, "v"))
    store, page, st = cv.kv_pool(sum(npg) + 8, H, A, seed=1)
    esc = _lib.Extra(600, "cuda", 2000, H)
    perm = torch.randperm(sum(npg) + 8, generator=torch.Generator().manual_seed(1)).tolist()
    bt = torch.zeros(len(lens), max(npg), dtype=torch.int32, device="cuda")
    sl, pages = [], []
    for b, L in enumerate(lens):
        leaves = [perm.pop() for _ in range(npg[b])]
        pages += leaves
        bt[b, : npg[b]] = torch.tensor(leaves, dtype=torch.int32, device="cuda")
        t = torch.arange(npg[b] * 16, device="cuda", dtype=torch.int64)
        sl.append(torch.tensor(leaves, device="cuda", dtype=torch.int64)[t // 16] * 16 + t % 16)
    _lib.kv_write(key, value, torch.cat(sl), store, page, kb, vb, A, esc, 7, st)
    torch.cuda.synchronize()
    res["status"] = st.tolist()[:12]
    res["pool"] = h(store)
    pg = torch.tensor(pages, dtype=torch.int32, device="cuda")
    ok = torch.empty(len(pages), 16, H, 128, dtype=torch.int16, device="cuda")
    ov = torch.empty_like(ok)
    _lib.kv_unpack(store, page, pg, kb, vb, A, esc, ok, ov)
    res["unpack"] = h(torch.cat([ok, ov]))
    cache = torch.zeros(len(pages) + 3, 16, H, 256, dtype=torch.bfloat16, device="cuda")
    _lib.kv_gather(store, page, pg, torch.arange(len(pages), dtype=torch.int32, device="cuda") + 3, kb, vb, A, esc, cache)
    res["gather"] = h(cache)
    scale = 1.0 / math.sqrt(128)
    sll = torch.tensor(lens, dtype=torch.int32, device="cuda")
    for G in (1, 4, 8):
        q = (torch.randn(len(lens), H * G, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(G)) * 2.0).to(torch.bfloat16)
        for sp in (_lib.kv_decode_shares(len(lens), H, max(npg)), 1, 7, 64):
            for knob in ("", "GLYD_KV_FUSE=0", "GLYD_KV_PIPE=1"):
                if knob:
                    os.environ[knob.split("=")[0]] = knob.split("=")[1]
                try:
                    o = torch.empty_like(q)
                    _lib.kv_decode(q, store, page, kb, vb, bt, sll, sp, H, G, A, esc, scale, o, _lib.kv_decode_workspace(len(lens), H, G, sp))
                    torch.cuda.synchronize()
                finally:
                    if knob:
                        os.environ.pop(knob.split("=")[0])
                res[f"decode G{G} splits{sp} {knob or 'default'}"] = h(o)
    json.dump(res, open(out, "w"), indent=1)
    print(f"kv_bits: {len(res)} outputs hashed ({out})")


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    diff = [k for k in A if A[k] != B.get(k)]
    print(f"kv_bits: {len(A) - len(diff)} of {len(A)} outputs the same bits" + (f"; differ: {diff}" if diff else ""))
    return 1 if diff else 0


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    main(sys.argv[1])
