"""FP8-arithmetic variant, diagnostic only: compact Candidate A weights converted on the device to e4m3 with one
power-of-two scale a row (values below e4m3's range round to nearest even), X as e4m3 with one scale a token,
fp32 sums. It is not the exact Candidate A path and has no gate of the frozen protocol; the exact matrices run first."""
import argparse
import ctypes as C
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

from protocol import POLICY, TOKENS, check_inputs
from validate import I64, PTR, SIZE, bind, ok

VARIANTS = (1, 2, 3)
# Column of each byte of a 16-column group in the B fragments' order (byte 4t + j: columns 2t, 2t + 1, 2t + 8, 2t + 9).
ORDER = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
SCALE_SHIFT = [0, 8, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 4]  # largest t with M 2^t <= 448, M = 1..15


def api(path):
    lib = C.CDLL(str(path.resolve()))
    return {
        "quantize": bind(lib, "glyd_candidate_a8_quantize", [PTR, I64, I64, PTR, PTR, PTR]),
        "unpack": bind(lib, "glyd_candidate_a8_unpack", [PTR]*4+[I64, I64, PTR, PTR, PTR]),
        "workspace": bind(lib, "glyd_candidate_a8_workspace", [I64, I64, I64, I64, C.POINTER(SIZE), C.POINTER(SIZE)]),
        "product": bind(lib, "glyd_candidate_a8_product", [PTR]*4+[I64, I64, PTR, PTR, PTR, I64, I64, PTR, PTR, PTR, SIZE, PTR, PTR]),
    }


def row_scales(words):
    """2^k a row: the smallest power of two putting the row's largest Candidate A magnitude at or below 448."""
    import torch
    w = words.to(torch.int64) & 0xFFFF
    e, m = (w >> 7) & 0xFF, (w >> 4) & 7
    sig = torch.where(e > 0, 8 + m, m)
    power = torch.where(e > 0, e - 130, torch.full_like(e, -129))
    k = power - torch.tensor(SCALE_SHIFT)[sig]
    k = torch.where((sig > 0) & (e != 0xFF), k, torch.full_like(k, -(1 << 20))).amax(dim=1)
    k = torch.where(k == -(1 << 20), torch.zeros_like(k), k)
    return torch.ldexp(torch.ones(k.shape, dtype=torch.float64), k).float()


def case_run(root, case, output):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261008)
    funcs = api(root/"libcandidate_a.so")
    directory = root/"fixtures"/case["id"]
    meta = json.loads((directory/"pack.json").read_text())
    rows, cols = case["shape"]
    tiers = (C.c_uint32*3)(*meta["tiers"])
    stream = lambda: torch.cuda.current_stream().cuda_stream
    load = lambda name, dtype: torch.frombuffer(bytearray((directory/name).read_bytes()), dtype=dtype).clone()
    ptr = lambda t: t.data_ptr() if t is not None else None
    words = load("rounded.bf16", torch.int16).reshape(rows, cols)
    data, blocks, bases = (load(n, d).cuda() for n, d in (("compact.data", torch.uint8), ("blocks.bin", torch.uint8), ("bases.i32", torch.int32)))
    wbf = words.view(torch.bfloat16).cuda()
    sw = row_scales(words).cuda()
    w8 = (wbf.float()/sw[:, None]).to(torch.float8_e4m3fn)
    out = torch.empty((rows, cols), dtype=torch.uint8, device="cuda")
    ok(funcs["unpack"](ptr(data), ptr(blocks), ptr(bases), tiers, rows, cols, ptr(sw), ptr(out), stream()))
    torch.cuda.synchronize()
    unpack_mismatch = int((out != w8.view(torch.uint8)).sum())
    changed = int((w8.float()*sw[:, None] != wbf.float()).sum())
    peak = w8.float().abs().amax(dim=1)
    smallest_cover = bool(((peak > 224) & (peak <= 448) | (wbf.float().abs().amax(dim=1) == 0)).all())
    result = {"case": case["id"], "shape": case["shape"], "unpack_bytes_mismatch": unpack_mismatch,
              "weights_rounded_by_e4m3": changed, "scale_smallest_cover": smallest_cover, "scale_exponents": [int(torch.log2(sw).min()), int(torch.log2(sw).max())],
              "products": [], "complete": False}
    output.write_text(json.dumps(result, indent=2)+"\n")
    order = torch.tensor(ORDER, device="cuda")
    copies = min(POLICY["max_copies"], max(2, -(-POLICY["rotation_bytes"]//(data.numel()+blocks.numel()+4*bases.numel()))))
    cw = [(data, blocks, bases)]+[(data.clone(), blocks.clone(), bases.clone()) for _ in range(copies-1)]

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
            inv = torch.where(amax > 0, 448.0/amax, torch.zeros_like(amax))
            q = (x.float()*inv[:, None]).to(torch.float8_e4m3fn)
            quantize_mismatch = int((x8.view(tokens, cols//16, 16) != q.view(torch.uint8).view(tokens, cols//16, 16)[:, :, order]).sum())
            scale_mismatch = int((sx != amax/448.0).sum())
            ref8 = (q.float()*sx[:, None]) @ (w8.float()*sw[:, None]).T
            exact = x.float() @ wbf.float().T
            if bias is not None:
                ref8.add_(bias.float())
                exact.add_(bias.float())
            row = {"tokens": tokens, "bias": has_bias, "quantize_bytes_mismatch": quantize_mismatch, "scale_mismatch": scale_mismatch, "variants": {}}
            for v in VARIANTS:
                wb, cb = SIZE(), SIZE()
                ok(funcs["workspace"](rows, cols, tokens, v, C.byref(wb), C.byref(cb)))
                work = torch.empty(max(1, wb.value), dtype=torch.uint8, device="cuda")
                counters = torch.zeros(cb.value//4, dtype=torch.int32, device="cuda")
                y = torch.empty((tokens, rows), dtype=torch.bfloat16, device="cuda")
                run = lambda bufs: ok(funcs["product"](ptr(bufs[0]), ptr(bufs[1]), ptr(bufs[2]), tiers, rows, cols, ptr(sw), ptr(x8), ptr(sx),
                                                      tokens, v, ptr(bias), ptr(y), ptr(work), wb.value, ptr(counters), stream()))
                run(cw[0])
                torch.cuda.synchronize()
                first = y.view(torch.int16).clone()
                run(cw[0])
                torch.cuda.synchronize()
                repeat = bool(torch.equal(first, y.view(torch.int16))) and not bool(counters.any())
                ref_nrms, ref_max = nrms(y, ref8)
                exact_nrms, _ = nrms(y, exact)
                entry = {"repeat_bit_exact": repeat, "nrms_vs_fp8_reference": ref_nrms, "max_vs_fp8_reference": ref_max,
                         "reference_max": float(ref8.abs().max()), "nrms_vs_exact": exact_nrms, "workspace_bytes": wb.value}
                if not has_bias:
                    entry["ms"] = timing(lambda i: run(cw[i]), len(cw))
                    torch.cuda.synchronize()
                    entry["replay_bit_exact"] = bool(torch.equal(first, y.view(torch.int16)))
                row["variants"][str(v)] = entry
                del work, counters, y, first
            if not has_bias:
                row["quantize_ms"] = timing(lambda i: ok(funcs["quantize"](ptr(x), tokens, cols, ptr(x8), ptr(sx), stream())), 1)
            result["products"].append(row)
            output.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps({"case": case["id"], "tokens": tokens, "bias": has_bias}), flush=True)
            del x, bias, x8, sx, q, ref8, exact
            gc.collect()
    result["complete"] = True
    output.write_text(json.dumps(result, indent=2)+"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case")
    args = ap.parse_args()
    manifest = check_inputs(args.bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in manifest["cases"] if c["kind"] == "model" and c["shape"][1] % 64 == 0]
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
    report = {"scope": "FP8-arithmetic diagnostic variant; not the exact Candidate A path, native FP8 serving or end-to-end speed",
              "variants": {"1": "64 tokens by two row blocks", "2": "128 tokens by two row blocks", "3": "256 tokens by one row block"},
              "exit_status": status, "elapsed_s": time.monotonic()-start, "results": results}
    (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"cases": len(cases), "status": status}))


if __name__ == "__main__":
    main()
