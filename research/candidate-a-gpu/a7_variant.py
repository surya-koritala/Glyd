"""Seven-bit variant, diagnostic only: Candidate A weights rounded to e3m3 with one power-of-two scale a row and 128
columns (the candidate-a7 arm of the quality screen, research/quant-benchmark), stored in seven bits a weight and turned
into e4m3 on the device by bit operations; X as e4m3 with one scale a token, fp32 sums, each 128 columns' sums times
their block's scale. Timed beside the FP8 lower bound of validate.py on the same matrices. No gate of the frozen protocol.
research/quant-benchmark/check_e3m3.py checks the encoding against the screen's transform on CI."""
import argparse
import ctypes as C
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

VARIANTS = (1, 2, 3, 4, 5, 6, 7)
BF16 = 3  # BF16 arithmetic from X itself, up to 64 tokens
# Column of each byte of a 16-column group in the B fragments' order (byte 4t + j: columns 2t, 2t + 1, 2t + 8, 2t + 9).
ORDER = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
BITS = [0, 1, 2, 3, 4, 5, 7]  # bit of the eighth word's bytes kept in bit 6 of word i's bytes
# Language-model matrix shapes of the pinned checkpoint and their counts (64 layers; the fixed-width census's tensor
# list). The 48 x 5120 shape is not a case and is left out, as in the earlier records' step sums.
STEP = {(1024, 5120): 32, (5120, 6144): 64, (5120, 17408): 64, (6144, 5120): 48, (10240, 5120): 48, (12288, 5120): 16, (17408, 5120): 128}


def encode(words):
    """Candidate A words (int16, rows x cols, cols a multiple of 128) as e4m3 bytes of each e3m3 value times 2^-4 and
    int8 block exponents: weight = byte's value * 2^exp, exp - 4 the smallest k with the block's peak <= 30 * 2^k."""
    import torch
    rows, cols = words.shape
    v = words.view(torch.bfloat16).float()
    if not bool(torch.isfinite(v).all()):
        raise ValueError("non-finite weight")
    peak = v.abs().view(rows, cols // 128, 128).amax(dim=2)
    f, e = torch.frexp(peak)  # peak = f 2^e, f in [0.5, 1); 30 = 0.9375 2^5
    k = torch.where(peak > 0, e - 5 + (f > 0.9375).int(), torch.zeros_like(e))
    if bool(((k + 4 > 7) | (k + 4 < -100)).any()):  # the device's normal-range powers of two (BF16 path: exp + 120 <= 127)
        raise ValueError("block scale out of range")
    q = torch.ldexp(v, -(k + 4).repeat_interleave(128, dim=1))
    codes = q.to(torch.float8_e4m3fn).view(torch.uint8)
    return codes, (k + 4).to(torch.int8)


def decode(codes, exps):
    import torch
    return torch.ldexp(codes.view(torch.float8_e4m3fn).float(), exps.int().repeat_interleave(128, dim=1))


def exps_blocks(exps):
    """Device order of the block exponents: [row block][column / 128][g][nn] for row nn * 8 + g of the block."""
    rows, kb = exps.shape
    return exps.view(rows // 64, 8, 8, kb).permute(0, 3, 2, 1).contiguous()


def fragments(codes):
    """Bytes [row block][pair][register][lane][4]: register 2nn + h of lane 4g + t holds row nn * 8 + g, columns
    2t, 2t + 1, 2t + 8, 2t + 9 of the pair's step h (the decoded path's B8)."""
    rows, cols = codes.shape
    x = codes.view(rows // 64, 8, 8, cols // 32, 2, 16)[..., ORDER].reshape(rows // 64, 8, 8, cols // 32, 2, 4, 4)
    return x.permute(0, 3, 1, 4, 2, 5, 6).reshape(rows // 64, cols // 32, 16, 32, 4)


def pack7(frag):
    """A step's eight registers (2nn + h) as seven words: uint32 [row block][pair][step][7][lane] as bytes."""
    import torch
    rb, npairs = frag.shape[:2]
    if bool((frag & 0x40).any()):
        raise ValueError("bit 6 set")
    f = frag.view(rb, npairs, 8, 2, 32, 4).permute(0, 1, 3, 2, 4, 5)  # [.., step h, nn, lane, byte]
    bits = torch.tensor(BITS, dtype=torch.uint8).view(1, 1, 1, 7, 1, 1)
    return (f[:, :, :, :7] | (((f[:, :, :, 7:] >> bits) & 1) << 6)).contiguous().view(-1)


def unpack7(packed, rb, npairs):
    """The device's bit operations, on bytes."""
    import torch
    w = packed.view(rb, npairs, 2, 7, 32, 4)
    bits = torch.tensor(BITS, dtype=torch.uint8).view(1, 1, 1, 7, 1, 1)
    last = (((w >> 6) & 1) << bits).sum(dim=3, keepdim=True, dtype=torch.uint8)
    return torch.cat([w & 0xBF, last], dim=3).permute(0, 1, 3, 2, 4, 5).reshape(rb, npairs, 16, 32, 4)


def api(path):
    from validate import I64, PTR, SIZE, bind
    lib = C.CDLL(str(path.resolve()))
    return {
        "quantize": bind(lib, "glyd_candidate_a8_quantize", [PTR, I64, I64, PTR, PTR, PTR]),
        "unpack": bind(lib, "glyd_candidate_a7_unpack", [PTR, I64, I64, PTR, PTR]),
        "workspace": bind(lib, "glyd_candidate_a7_workspace", [I64, I64, I64, I64, C.POINTER(SIZE), C.POINTER(SIZE)]),
        "product": bind(lib, "glyd_candidate_a7_product", [PTR, PTR, I64, I64, PTR, PTR, PTR, I64, I64, PTR, PTR, PTR, SIZE, PTR, PTR]),
    }


def case_run(root, case, output):
    import torch
    from protocol import POLICY, TOKENS
    from validate import SIZE, ok
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261008)
    funcs = api(root/"libcandidate_a.so")
    rows, cols = case["shape"]
    words = torch.frombuffer(bytearray((root/"fixtures"/case["id"]/"rounded.bf16").read_bytes()), dtype=torch.int16).reshape(rows, cols)
    stream = lambda: torch.cuda.current_stream().cuda_stream
    ptr = lambda t: t.data_ptr() if t is not None else None
    codes, exps = encode(words)  # on the CPU, as the self-check ran
    packed = pack7(fragments(codes)).cuda()
    exps_d = exps_blocks(exps).cuda()
    w7 = decode(codes, exps).cuda()
    wbf = words.view(torch.bfloat16).cuda()
    out = torch.empty((rows, cols), dtype=torch.uint8, device="cuda")
    ok(funcs["unpack"](ptr(packed), rows, cols, ptr(out), stream()))
    torch.cuda.synchronize()
    result = {"case": case["id"], "shape": case["shape"], "unpack_bytes_mismatch": int((out != codes.cuda()).sum()),
              "weights_changed_by_e3m3": int((w7 != wbf.float()).sum()), "packed_bytes": packed.numel(), "exps_bytes": exps.numel(),
              "scale_exponents": [int(exps.min()) - 4, int(exps.max()) - 4], "products": [], "complete": False}
    output.write_text(json.dumps(result, indent=2)+"\n")
    del out, codes
    order = torch.tensor(ORDER, device="cuda")
    copies = lambda nbytes: min(POLICY["max_copies"], max(2, -(-POLICY["rotation_bytes"]//nbytes)))
    k7 = copies(packed.numel() + exps.numel())
    c7 = [(packed, exps_d)]+[(packed.clone(), exps_d.clone()) for _ in range(k7-1)]
    f8 = torch.float8_e4m3fn
    sw8 = (wbf.float().abs().amax().clamp(min=1e-30)/448).reshape(())
    w8 = (wbf.float()/sw8).to(f8)
    c8 = [w8]+[w8.clone() for _ in range(copies(w8.numel())-1)]
    result["rotation"] = {"a7_copies": k7, "fp8_copies": len(c8)}

    def timing(fn, k):
        ops = max(POLICY["repetitions"], k)
        for _ in range(POLICY["warmup"]):
            for i in range(k):
                fn(i)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(ops):
                fn(i % k)
        graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(POLICY["timing_groups"]):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b)/ops)
        del graph
        return samples

    def nrms(y, ref):
        d = y.float()-ref
        return float(d.square().mean().sqrt())/max(float(ref.square().mean().sqrt()), 1e-30), float(d.abs().max())

    for tokens in TOKENS:
        for has_bias in (False, True):
            x = torch.randn(tokens, cols, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            bias = (torch.randn(rows, device="cuda")*0.1).to(torch.bfloat16) if has_bias else None
            x8 = torch.empty((tokens, cols), dtype=torch.uint8, device="cuda")
            sx = torch.empty(tokens, dtype=torch.float32, device="cuda")
            ok(funcs["quantize"](ptr(x), tokens, cols, ptr(x8), ptr(sx), stream()))
            torch.cuda.synchronize()
            amax = x.float().abs().amax(dim=1)
            full = lambda v: torch.full_like(amax, v)
            # Tensor-by-tensor division rounds once, as the device does; a Python-number divisor multiplies by its
            # reciprocal instead (the earlier harness's reference, kept for comparison).
            inv = torch.where(amax > 0, full(448.0)/amax, torch.zeros_like(amax))
            q = (x.float()*inv[:, None]).to(f8)
            got = x8.view(tokens, cols//16, 16)
            permuted = lambda t: t.view(torch.uint8).view(tokens, cols//16, 16)[:, :, order]
            old_inv = torch.where(amax > 0, 448.0/amax, torch.zeros_like(amax))
            row = {"tokens": tokens, "bias": has_bias,
                   "quantize_bytes_mismatch": int((got != permuted(q)).sum()), "scale_mismatch": int((sx != amax/full(448.0)).sum()),
                   "quantize_bytes_mismatch_reciprocal_reference": int((got != permuted((x.float()*old_inv[:, None]).to(f8))).sum()),
                   "scale_mismatch_reciprocal_reference": int((sx != amax/448.0).sum()), "variants": {}}
            ref = (q.float()*sx[:, None]) @ w7.T
            exact = x.float() @ w7.T
            if bias is not None:
                ref.add_(bias.float())
                exact.add_(bias.float())
            for v in VARIANTS:
                if v == BF16 and tokens > 64:
                    continue
                wb, cb = SIZE(), SIZE()
                ok(funcs["workspace"](rows, cols, tokens, v, C.byref(wb), C.byref(cb)))
                work = torch.empty(max(1, wb.value), dtype=torch.uint8, device="cuda")
                counters = torch.zeros(cb.value//4, dtype=torch.int32, device="cuda")
                y = torch.empty((tokens, rows), dtype=torch.bfloat16, device="cuda")
                run = lambda bufs: ok(funcs["product"](ptr(bufs[0]), ptr(bufs[1]), rows, cols, ptr(x), ptr(x8), ptr(sx), tokens, v, ptr(bias), ptr(y),
                                                      ptr(work), wb.value, ptr(counters), stream()))
                run(c7[0])
                torch.cuda.synchronize()
                first = y.view(torch.int16).clone()
                run(c7[0])
                torch.cuda.synchronize()
                own = exact if v == BF16 else ref  # the variant's own arithmetic: BF16 activations, or their e4m3 bytes
                ref_nrms, ref_max = nrms(y, own)
                entry = {"repeat_bit_exact": bool(torch.equal(first, y.view(torch.int16))) and not bool(counters.any()),
                         "nrms_vs_reference": ref_nrms, "max_vs_reference": ref_max, "reference_max": float(own.abs().max()),
                         "nrms_vs_bf16_activations": nrms(y, exact)[0], "workspace_bytes": wb.value}
                if not has_bias:
                    entry["ms"] = timing(lambda i: run(c7[i]), len(c7))
                    torch.cuda.synchronize()
                    entry["replay_bit_exact"] = bool(torch.equal(first, y.view(torch.int16)))
                row["variants"][str(v)] = entry
                del work, counters, y, first
            if not has_bias:
                xq8 = (x.float()/(x.float().abs().amax().clamp(min=1e-30)/448)).to(f8)
                sxt = (x.float().abs().amax().clamp(min=1e-30)/448).reshape(())
                o8 = torch.empty((tokens, rows), dtype=torch.bfloat16, device="cuda")
                row["fp8_ms"] = timing(lambda i: torch._scaled_mm(xq8, c8[i].t(), scale_a=sxt, scale_b=sw8, out_dtype=torch.bfloat16,
                                                                  use_fast_accum=True, out=o8), len(c8))
                row["quantize_ms"] = timing(lambda i: ok(funcs["quantize"](ptr(x), tokens, cols, ptr(x8), ptr(sx), stream())), 1)
                del xq8, o8
            result["products"].append(row)
            output.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps({"case": case["id"], "tokens": tokens, "bias": has_bias}), flush=True)
            del x, bias, x8, sx, q, ref, exact
            gc.collect()
    result["complete"] = True
    output.write_text(json.dumps(result, indent=2)+"\n")


def summary(cases, results):
    """Text-decoder step: each language-model shape's median times its count, per row count."""
    out = {}
    for case in cases:
        r = results.get(case["id"])
        if tuple(case["shape"]) not in STEP or not r or not r.get("complete"):
            continue
        n = STEP[tuple(case["shape"])]
        for p in r["products"]:
            if p["bias"]:
                continue
            s = out.setdefault(str(p["tokens"]), {"shapes": 0, "fp8_lower_bound": 0.0, "a7_best": 0.0, "quantize": 0.0,
                                                  **{f"a7_variant_{v}": 0.0 for v in VARIANTS if v != BF16 or p["tokens"] <= 64}})
            s["shapes"] += 1
            s["fp8_lower_bound"] += n*statistics.median(p["fp8_ms"])
            s["quantize"] += n*statistics.median(p["quantize_ms"])
            times = {v: statistics.median(p["variants"][str(v)]["ms"]) for v in VARIANTS if str(v) in p["variants"]}
            for v, ms in times.items():
                s[f"a7_variant_{v}"] += n*ms
            s["a7_best"] += n*min(times.values())
    for s in out.values():
        for k, val in s.items():
            if isinstance(val, float):
                s[k] = round(val, 3)
        s["a7_best_over_fp8"] = round(s["a7_best"]/s["fp8_lower_bound"], 4) if s["fp8_lower_bound"] else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case")
    args = ap.parse_args()
    from protocol import check_inputs
    manifest = check_inputs(args.bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in manifest["cases"] if c["kind"] == "model" and c["shape"][1] % 128 == 0]
    if args.case:
        case_run(args.bundle, next(c for c in cases if c["id"] == args.case), args.out/(args.case+".json"))
        return
    start, status = time.monotonic(), {}
    for case in cases:
        try:
            done = subprocess.run([sys.executable, __file__, "--bundle", str(args.bundle), "--out", str(args.out), "--case", case["id"]], timeout=600)
            status[case["id"]] = done.returncode
        except subprocess.TimeoutExpired:
            status[case["id"]] = "timeout"
    results = {}
    for case in cases:
        path = args.out/(case["id"]+".json")
        results[case["id"]] = json.loads(path.read_text()) if path.exists() else None
    report = {"scope": "Seven-bit diagnostic variant and the FP8 lower bound (per-tensor e4m3, cuBLASLt fast accumulation, activation quantization excluded) on the same matrices; not native serving or end-to-end speed",
              "variants": {"1": "64 tokens by two row blocks", "2": "128 tokens by one row block", "3": "BF16 arithmetic from X, the engine's small-M product (up to 64 tokens)",
                           "4": "64 tokens by two row blocks, eight 16-row consumers", "5": "128 tokens by two row blocks", "6": "256 tokens by one row block",
                           "7": "128 tokens by one row block, eight 16-row consumers"},
              "exit_status": status, "elapsed_s": time.monotonic()-start, "text_decoder_step_ms": summary(cases, results), "results": results}
    (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"cases": len(cases), "status": status}))


if __name__ == "__main__":
    main()
