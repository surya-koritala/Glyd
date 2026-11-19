"""The lossless KV cache's decode attention and write kernels, timed alone against vLLM's own: one layer of Qwen3-8B (32 query heads over 8 KV heads, head 128, pages of 16 tokens)
at the shapes of kv_tps.py, B sequences of L tokens: Glyd's kv_decode on 12-bit pages against vLLM's FlashAttention 2 (flash_attn_varlen_func, as vLLM calls it for decode rows: a paged
bf16 cache of the same values, num_splits left to it) and Glyd's kv_write of one new token a sequence against vLLM's reshape_and_cache_flash. A call's time is the median of REPEAT
batches of CALLS calls (CUDA events); each shape is held in as many copies as make the pages larger than the L2 cache, taken in turn, as a step takes its layers'; the last two columns are what
36 layers add up to a step (from the graph's times). Each time is taken twice: inside a CUDA graph (kv_decode_us ..., the GPU's time: a graph step's) and as a loop of calls from Python (..._eager_us: an eager step's, with each launch). How close the two outputs are (max_abs_difference_to_fa2, against max_abs_output) is measured on finite inputs of the same kinds, apart from the timing data: that has outliers up to exponent 254,
where a score overflows and both outputs are NaN.

    python kv_decode_bench.py [--calls 40] [--repeat 5] [--arena A] [--chunks N] [--order seq|random] [--shapes 32x1024,4x8192,...] [--sweep 16:1,8:1,4:1] [--out FILE.json]
                              [--head 256 --kv-heads 4 --group 4 --layers 8]   (another model's layer: Qwen3.5-9B's 8 full-attention layers of 4 KV heads of 256, 16 query heads)

--sweep: the decode kernel timed (in the graph) at each plan given as PPC:FUSE[:PIPE[:FLOOR]] (the least pages a CTA: GLYD_KV_PPC_MIN; 1, the last CTA of a (sequence, head) merges the partials, or 0, a kernel of its own does;
PIPE 1: each warp loads its next page ahead, GLYD_KV_PIPE=1, 0 only when it has computed the one in hand; FLOOR: the CTAs a sequence has at least, GLYD_KV_FLOOR, 1: none; each the library's own where it is left out),
with the splits the library gives for it (the "sweep" of each row).

--shapes: BxL shapes in place of the L4's eight (a bigger GPU holds more: 32x8192,64x8192,32x32768,128x4096). FlashAttention is the version vLLM picks for the GPU (2; 3 on Hopper, with no scheduler metadata: its own).

--order: the pages of a sequence consecutive in the pool, as vLLM's allocator hands them out on a fresh server (seq, the default), or in a random order.

Run as check_vllm.py is (the library and vLLM's venv)."""
import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_vllm as cv  # noqa: E402
from glyd_gpu import _lib  # noqa: E402

H, G, D, PT, LAYERS = 8, 4, 128, 16, 36


def natural(T, seed, kind, **kw):
    """bf16 bits [T, H, D] (int16) as check_vllm's kv_natural makes them, a head of 256 two of its heads of 128."""
    return cv.kv_natural(T, H * D // 128, seed, kind, **kw).view(T, H, D)


def bases(x):
    return cv.kv_bases(x.view(x.shape[0], -1, 128)).view(H, D)


def timed(fns, calls, repeat):
    """The median microseconds of one call: `calls` calls (cycling through fns) between two CUDA events, `repeat` times."""
    for i in range(len(fns) * 2):
        fns[i % len(fns)]()
    torch.cuda.synchronize()
    res = []
    for _ in range(repeat):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for i in range(calls):
            fns[i % len(fns)]()
        b.record()
        torch.cuda.synchronize()
        res.append(a.elapsed_time(b) * 1000 / calls)
    return statistics.median(res), min(res), max(res)


def timed_graph(fns, calls, repeat):
    """The median microseconds of one call with no host in it (what a CUDA-graph step has): `calls` calls (cycling through fns) captured in a CUDA graph, replayed between two CUDA events, `repeat` times. A call made from Python
    (timed's) also costs its launch: a FlashAttention call's wrapper takes about 100 us of the host, which is all of a small shape's time on a fast GPU."""
    for i in range(len(fns) * 2):
        fns[i % len(fns)]()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(calls):
            fns[i % len(fns)]()
    gr.replay()
    torch.cuda.synchronize()
    res = []
    for _ in range(repeat):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gr.replay()
        b.record()
        torch.cuda.synchronize()
        res.append(a.elapsed_time(b) * 1000 / calls)
    del gr
    return statistics.median(res), min(res), max(res)


def timed_cold(fns, flush, n):
    """The median microseconds of one call made on a cold GPU: before each of n calls a read of `flush` (larger than the L2 cache: its lines are clean, as the weights' are in a step) is queued, then two events
    around the call (everything queued before the first runs, so the call does not wait for the host); the pages a write touches are then in no cache, as in a step's layer. The events' own cost is in every time
    (vLLM's write, 2 us a kernel, reads 5 to 7 us here): the difference of two calls is what the call itself costs."""
    for i in range(4):
        fns[i % len(fns)]()
    torch.cuda.synchronize()
    ev = []
    for i in range(n):
        flush.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fns[i % len(fns)]()
        b.record()
        ev.append((a, b))
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1000 for a, b in ev)


def main():
    global H, G, D, LAYERS
    a = sys.argv[1:]
    for flag, name in (("--head", "D"), ("--kv-heads", "H"), ("--group", "G"), ("--layers", "LAYERS")):
        if flag in a:
            globals()[name] = int(a[a.index(flag) + 1])
    calls = int(a[a.index("--calls") + 1]) if "--calls" in a else 40
    repeat = int(a[a.index("--repeat") + 1]) if "--repeat" in a else 5
    A = int(a[a.index("--arena") + 1]) if "--arena" in a else _lib.kv_free_bay(H, D)
    chunks = int(a[a.index("--chunks") + 1]) if "--chunks" in a else 4096  # (the escape pool the pages' entries past the arena take: one for all)
    out = a[a.index("--out") + 1] if "--out" in a else None
    order = a[a.index("--order") + 1] if "--order" in a else "seq"
    sweep = a[a.index("--sweep") + 1].split(",") if "--sweep" in a else []
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
    from vllm import _custom_ops as ops

    fa_version = get_flash_attn_version(head_size=D)
    dev = "cuda"
    page = _lib.kv_leaf_bytes(H, D, A)
    esc = _lib.Extra(chunks)
    scale = D ** -0.5
    rows = []
    shapes = [tuple(int(x) for x in s.split("x")) for s in a[a.index("--shapes") + 1].split(",")] if "--shapes" in a else [(32, 1024), (4, 8192), (1, 8192), (8, 4096), (3, 8192), (1, 1024), (44, 1024), (5, 8192)]
    for B, L in shapes:
        npg = -(-L // PT)
        N = B * npg
        T = B * L
        k = natural(T, 1, "k", outlier=0.0003)
        v = natural(T, 2, "v", outlier=0.0004)
        kb, vb = bases(k), bases(v)
        copies = max(1, min(64, math.ceil(256e6 / (N * PT * H * 2 * D * 2))))
        g = torch.Generator().manual_seed(B * 100000 + L)
        fns_kv, fns_fa, fns_w, fns_wfa, keep = [], [], [], [], []
        q = (torch.randn(B, H * G, D, device=dev) * 0.5).to(torch.bfloat16)
        seq_lens = torch.full((B,), L, dtype=torch.int32, device=dev)
        cu = torch.arange(B + 1, dtype=torch.int32, device=dev)
        ones = torch.ones(1, dtype=torch.float32, device=dev).expand(B, H)
        out_kv, out_fa = torch.empty_like(q), torch.empty_like(q)
        shares = _lib.kv_decode_shares(B, H, npg)  # (the plan from each sequence's own length: the most CTAs a (sequence, head) is split over)
        ws = _lib.kv_decode_workspace(B, H, G, shares, None, D)
        # how close the two are, on finite inputs: K and V of the same kinds with their outliers in exponents 110..134 (a score stays finite); their own pool and cache, freed before the timing data's are made
        kf, vf = natural(T, 3, "k", outlier=0.0003, erange=(110, 135)), natural(T, 4, "v", outlier=0.0004, erange=(110, 135))
        kbf, vbf = bases(kf), bases(vf)
        bt0 = torch.arange(N, dtype=torch.int32).view(B, npg).contiguous().to(dev)
        tt = torch.arange(L, dtype=torch.int64)
        slots0 = (bt0.cpu().to(torch.int64)[:, (tt // PT)] * PT + (tt % PT)[None, :]).reshape(-1).to(dev)
        pool0 = torch.zeros(N * page, dtype=torch.uint8, device=dev)
        _lib.kv_write(kf, vf, slots0, pool0, page, kbf, vbf, A, esc, 0, torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device=dev))
        cache0 = torch.zeros(N, PT, H, 2 * D, dtype=torch.bfloat16, device=dev)
        cache0.view(-1, H, 2 * D)[slots0, :, :D] = kf.view(torch.bfloat16)
        cache0.view(-1, H, 2 * D)[slots0, :, D:] = vf.view(torch.bfloat16)
        _lib.kv_decode(q, pool0, page, kbf, vbf, bt0, seq_lens, shares, H, G, A, esc, scale, out_kv, ws)
        flash_attn_varlen_func(q=q, k=cache0[..., :D], v=cache0[..., D:], out=out_fa, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seq_lens, max_seqlen_k=L, softmax_scale=scale, causal=True, alibi_slopes=None,
                               window_size=[-1, -1], block_table=bt0, softcap=0, scheduler_metadata=None, fa_version=fa_version, q_descale=None, k_descale=ones, v_descale=ones, num_splits=0, s_aux=None)
        torch.cuda.synchronize()
        diff, top = (out_kv.float() - out_fa.float()).abs().max().item(), out_fa.float().abs().max().item()
        finite = bool(torch.isfinite(out_kv).all().item() and torch.isfinite(out_fa).all().item())
        del kf, vf, pool0, cache0, bt0, slots0
        torch.cuda.empty_cache()
        for c in range(copies):
            perm = (torch.randperm(N, generator=g) if order == "random" else torch.arange(N)).to(torch.int32)
            bt = perm.view(B, npg).contiguous().to(dev)
            tt = torch.arange(L, dtype=torch.int64)
            cells = (bt.cpu().to(torch.int64)[:, (tt // PT)] * PT + (tt % PT)[None, :]).reshape(-1).to(dev)
            store = torch.zeros(N * page, dtype=torch.uint8, device=dev)
            st = torch.zeros(_lib.KV_STATUS_WORDS, dtype=torch.int32, device=dev)
            _lib.kv_write(k, v, cells, store, page, kb, vb, A, esc, 0, st)
            cache = torch.zeros(N, PT, H, 2 * D, dtype=torch.bfloat16, device=dev)
            cache.view(-1, H, 2 * D)[cells, :, :D] = k.view(torch.bfloat16)
            cache.view(-1, H, 2 * D)[cells, :, D:] = v.view(torch.bfloat16)
            keep += [store, cache, bt, cells]
            fns_kv.append(lambda store=store, bt=bt: _lib.kv_decode(q, store, page, kb, vb, bt, seq_lens, shares, H, G, A, esc, scale, out_kv, ws))
            fns_fa.append(lambda cache=cache, bt=bt: flash_attn_varlen_func(q=q, k=cache[..., :D], v=cache[..., D:], out=out_fa, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seq_lens, max_seqlen_k=L,
                                                                              softmax_scale=scale, causal=True, alibi_slopes=None, window_size=[-1, -1], block_table=bt, softcap=0, scheduler_metadata=None,
                                                                              fa_version=fa_version, q_descale=None, k_descale=ones, v_descale=ones, num_splits=0, s_aux=None))
            # the write of one new token a sequence (at the position L - 1 of each): Glyd's kernel, and vLLM's own
            ns = (bt.to(torch.int64)[:, (L - 1) // PT] * PT + (L - 1) % PT).contiguous()
            nk, nv = k[:B].contiguous(), v[:B].contiguous()
            nkb, nvb = nk.view(torch.bfloat16), nv.view(torch.bfloat16)
            kc, vc = cache[..., :D], cache[..., D:]
            fns_w.append(lambda store=store, ns=ns, st=st: _lib.kv_write(nk, nv, ns, store, page, kb, vb, A, esc, 0, st))
            fns_wfa.append(lambda kc=kc, vc=vc, ns=ns: ops.reshape_and_cache_flash(nkb, nvb, kc, vc, ns, "auto", ones[0, 0].reshape(1), ones[0, 0].reshape(1)))
        row = {"B": B, "L": L, "copies": copies, "max_abs_difference_to_fa2": round(diff, 5), "max_abs_output": round(top, 5), "inputs_finite": finite}
        for name, fns in (("kv_decode", fns_kv), ("fa2_decode", fns_fa), ("kv_write", fns_w), ("vllm_write", fns_wfa)):
            for suffix, timer in (("", timed_graph), ("_eager", timed)):  # (the graph's time: the GPU's; the eager loop's: with each call's launch from Python)
                try:
                    med, lo, hi = timer(fns, calls, repeat)
                    row[name + suffix + "_us"] = round(med, 2)
                    row[name + suffix + "_range_us"] = [round(lo, 2), round(hi, 2)]
                except Exception as e:  # (vLLM's write op's signature differs by version: not fatal)
                    row[name + suffix + "_error"] = f"{type(e).__name__}: {e}"[:200]
        if sweep:
            row["sweep"] = {}
            pools, bts = keep[0::4], keep[2::4]
            saved = {k: os.environ.get(k) for k in ("GLYD_KV_PPC_MIN", "GLYD_KV_FUSE", "GLYD_KV_PIPE", "GLYD_KV_FLOOR")}
            for cfg in sweep:
                ppc, fuse, *more = cfg.split(":")
                pipe, floor = (more + ["", ""])[:2]
                os.environ["GLYD_KV_PPC_MIN"], os.environ["GLYD_KV_FUSE"] = ppc, fuse
                for knob, val in (("GLYD_KV_PIPE", pipe), ("GLYD_KV_FLOOR", floor)):
                    if val:
                        os.environ[knob] = val
                    else:
                        os.environ.pop(knob, None)
                try:
                    sp = _lib.kv_decode_shares(B, H, npg)
                    ws2 = _lib.kv_decode_workspace(B, H, G, sp, None, D)
                    fns = [lambda store=store, bt=bt: _lib.kv_decode(q, store, page, kb, vb, bt, seq_lens, sp, H, G, A, esc, scale, out_kv, ws2) for store, bt in zip(pools, bts)]
                    med, lo, hi = timed_graph(fns, calls, repeat)
                    row["sweep"][cfg] = {"shares": sp, "us": round(med, 2), "range_us": [round(lo, 2), round(hi, 2)]}
                except Exception as e:
                    row["sweep"][cfg] = {"error": f"{type(e).__name__}: {e}"[:200]}
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val
        flush = torch.zeros(256 * 1024 * 1024 // 4 * 2, dtype=torch.float32, device=dev)  # (512 MB: ten L2 caches)
        for name, fns in (("kv_write_cold", fns_w), ("vllm_write_cold", fns_wfa)):
            try:
                row[name + "_us"] = round(timed_cold(fns, flush, 60), 2)
            except Exception as e:
                row[name + "_error"] = f"{type(e).__name__}: {e}"[:200]
        del flush
        if "kv_write_cold_us" in row and "vllm_write_cold_us" in row:
            row["step_ms_write_extra_cold_36_layers"] = round((row["kv_write_cold_us"] - row["vllm_write_cold_us"]) * LAYERS / 1000, 3)
        if "kv_decode_eager_us" in row and "fa2_decode_eager_us" in row:
            row["decode_ratio_fa2_over_kv_eager"] = round(row["fa2_decode_eager_us"] / row["kv_decode_eager_us"], 3)
        if "kv_decode_us" in row and "fa2_decode_us" in row:
            row["decode_ratio_fa2_over_kv"] = round(row["fa2_decode_us"] / row["kv_decode_us"], 3)
            row["step_ms_saved_36_layers"] = round((row["fa2_decode_us"] - row["kv_decode_us"]) * LAYERS / 1000, 3)
        if "kv_write_us" in row and "vllm_write_us" in row:
            row["step_ms_write_extra_36_layers"] = round((row["kv_write_us"] - row["vllm_write_us"]) * LAYERS / 1000, 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del keep, fns_kv, fns_fa, fns_w, fns_wfa
        torch.cuda.empty_cache()
    if out:
        with open(out, "w") as f:
            json.dump({"bay": A, "leaf_bytes": page, "order": order, "gpu": torch.cuda.get_device_name(), "head": D, "kv_heads": H, "group": G, "layers": LAYERS, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
