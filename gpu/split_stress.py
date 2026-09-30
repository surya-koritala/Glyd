"""The route SPLIT's ring under stress: every slot written only after its last reader finished and read only after
its writer finished, on every schedule. Each model's layer (Qwen3 0.6B to 32B, q k v and gate up merged; weights of a
trained matrix's spread, a few far out) packed in the 12-bit layout, LAYERS copies of it queued as a prompt's order and
multiplied in order through the ring (glyd_gpu_mma12_ring_*, PyTorch's cuBLAS on the split's SMs), at 769, 1024, 2048
and 4096 tokens, in rings of 3 slots to 16 (the exact fit ring_plan gives, where a chunk fills its slot, and others),
the order queued whole or a few matrices ahead of the products, PASSES passes each: every copy's product the same
bits as the first copy's and pass to pass, and within 1e-2 of fp32 on the decoded weights (W bit for bit the pack's).
Then GLinear (made as on an A100, model.Split: the ring planned for the order) over the same copies, pass to pass.

    python split_stress.py [LIBRARY] [--models 0.6B,8B,14B,32B] [--passes 6] [--layers 4] [--ms 769,1024,2048,4096]"""
import argparse, os, sys, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("library", nargs="?")
ap.add_argument("--models", default="0.6B,1.7B,4B,8B,14B,32B")
ap.add_argument("--ms", default="769,1024,2048,4096")
ap.add_argument("--passes", type=int, default=6)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--quick", action="store_true", help="the rings of ring_plan's slots alone (3 and planned), 3 passes")
args = ap.parse_args()
if args.library:
    os.environ["GLYD_GPU_LIB"] = args.library
import glyd_gpu as g  # noqa: E402
from glyd.gpu import _lib as lib, model as gm  # noqa: E402

LAYERS = {  # Qwen3's layers: q k v, o, gate up, down ([O, K] each)
    "0.6B": [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)],
    "1.7B": [(4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144)],
    "4B": [(6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728)],
    "8B": [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)],
    "14B": [(7168, 5120), (5120, 5120), (34816, 5120), (5120, 17408)],
    "32B": [(10240, 5120), (5120, 8192), (51200, 5120), (5120, 25600)],
}
dev = torch.device("cuda", torch.cuda.current_device())
bf = torch.bfloat16
torch.manual_seed(0)


def weights(O, K):
    w = torch.randn(O, K, device=dev) * 0.02
    m = torch.rand(O, K, device=dev) < 0.002  # a few far out: the pack's exceptions
    w[m] = torch.randn(int(m.sum()), device=dev) * torch.exp2(torch.randint(-30, 10, (int(m.sum()),), device=dev).float())
    return w.to(bf)


def bits(t):
    return t.contiguous().view(-1).view(torch.int16)


def ring_for(shapes):
    """(slot bytes, slots, matrices queued ahead) for an order, by model.py's rule (written out here, so that this
    runs against a package before it)."""
    again = [sh for sh in set(shapes) if shapes.count(sh) > 1] or shapes
    O, K = max(again, key=lambda sh: sh[0] * sh[1])
    slot = O * K * 2 if O * K * 2 <= 100 << 20 else max(100 << 20, ((O + 1) // 2 + 63) // 64 * 64 * K * 2)
    slot = (slot + 255) // 256 * 256

    def chunks(O, K):
        per = max(64, min(O, slot // (2 * K) // 64 * 64))
        n = -(-O // per)
        rows = (-(-O // n) + 63) // 64 * 64
        return [(r0, min(rows, O - r0)) for r0 in range(0, O, rows)]

    def gap(keys):
        last, most = {}, 1
        for i, k in enumerate(keys):
            most, last[k] = max(most, i - last.get(k, i)), i
        return most

    return slot, max(3, min(16, gap([(sh, c) for sh in shapes for c in chunks(*sh)]) + 1)), gap(shapes) + 2


ws = torch.empty(32 << 20, dtype=torch.uint8, device=dev)  # (the device's context made first: cuBLAS's handle after it)
fns = gm.Split.blas_fns()
assert fns, "PyTorch's cuBLAS not found"
blas = lib.Blas(None, fns["cublasGemmEx"], fns["cublasSetStream_v2"], fns["cublasGetStream_v2"], fns["cublasSetWorkspace_v2"], fns["cublasSetSmCountTarget"], fns["cublasGetSmCountTarget"], ws.data_ptr(), 32 << 20)
S = torch.cuda.get_device_properties(dev).multi_processor_count
SMS = min(12, S // 4)
failures, checked = [], 0
t0 = time.time()
for name in args.models.split(","):
    shapes = LAYERS[name]
    W = [weights(O, K) for O, K in shapes]
    P = [g.pack_mma12(w) for w in W]
    for w, p in zip(W, P):
        assert torch.equal(bits(g.mma_unpack(p)), bits(w)), (name, "the pack")
    slot, slots, ahead = ring_for(shapes * args.layers)
    rings = [(slot, 3), (slot, slots)] if args.quick else [(slot, 3), (slot, slots), (slot, 16), (64 << 20, 4), (64 << 20, 16), (slot // 2 // 256 * 256, slots + 2)]
    order = [(i, c) for c in range(args.layers) for i in range(len(shapes))]
    for M in [int(m) for m in args.ms.split(",")]:
        X = [torch.randn(M, K, dtype=bf, device=dev) for _, K in shapes]
        ref = [F.linear(x.float(), w.float()) for x, w in zip(X, W)]
        firsts = {}  # by slot size (the same chunks, the same products): the first pass's first layer's products
        for slot_bytes, n in rings:
            n = max(3, min(16, n))
            if slot_bytes < 64 * 2 * max(K for _, K in shapes):
                continue
            buf = torch.empty(n * slot_bytes, dtype=torch.uint8, device=dev)
            ring = lib.ring_create(buf, slot_bytes)
            for how in ("whole", "ahead"):
                for pas in range(args.passes // (1 if args.quick else 2) or 1):
                    assert lib.ring_reset(ring) == 0
                    queued = 0
                    if how == "whole":
                        for i, c in order:
                            q = P[i]
                            assert lib.mma12_ring_queue(ring, SMS, q.data, q.exc, q.exc_base, q.sym, *q.shape) == 0
                        queued = len(order)
                    ys = []
                    for pos, (i, c) in enumerate(order):
                        while how == "ahead" and queued < min(len(order), pos + 1 + ahead):
                            q = P[order[queued][0]]
                            assert lib.mma12_ring_queue(ring, SMS, q.data, q.exc, q.exc_base, q.sym, *q.shape) == 0
                            queued += 1
                        q = P[i]
                        y = torch.full((M, q.shape[0]), float("nan"), dtype=bf, device=dev)
                        blas.handle = torch.cuda.current_blas_handle()
                        r = lib.mma12_ring_linear(ring, SMS, q.data, q.exc, q.exc_base, q.sym, *q.shape, X[i], None, y, blas)
                        assert r == 0, (name, M, lib.error_string(r))
                        ys.append(y)
                    torch.cuda.synchronize()
                    first = firsts.setdefault(slot_bytes, [])
                    for (i, c), y in zip(order, ys):
                        checked += 1
                        err = ((y.float() - ref[i]).abs().max() / ref[i].abs().max()).item()
                        if len(first) < len(shapes) and c == 0:
                            first.append(y)
                        tag = f"{name} M={M} ring {n} x {slot_bytes >> 20} MiB ({how}), pass {pas}, layer {c}, matrix {shapes[i]}"
                        if not err < 1e-2:
                            failures.append(f"{tag}: {err:.2e} off fp32")
                        elif not torch.equal(bits(y), bits(first[i])):
                            d = (bits(y) != bits(first[i])).nonzero()
                            o = sorted({int(v) % shapes[i][0] for v in d[:4096]})
                            failures.append(f"{tag}: not the first product's bits ({d.numel()} of {y.numel()} differ, W's rows {o[0]}-{o[-1]}), {err:.2e} off fp32")
                    del ys, y
            assert lib.ring_destroy(ring) == 0
            del buf
        # GLinear by the route (made as on an A100): model.Split plans the ring for the order and follows it
        cc, nm = torch.cuda.get_device_capability, torch.cuda.get_device_name
        torch.cuda.get_device_capability, torch.cuda.get_device_name = lambda device=None: (8, 0), lambda device=None: "NVIDIA A100-SXM4-40GB"
        try:
            lins = [[gm.GLinear(p, None) for p in P] for _ in range(args.layers)]
        finally:
            torch.cuda.get_device_capability, torch.cuda.get_device_name = cc, nm
        gm.set_scratch(torch.nn.ModuleList([m for c in lins for m in c]), False)
        gm.Split.of.pop(dev, None)
        outs = None
        for pas in range(args.passes + 1):  # (the first records the order: its products too the same bits as every pass after)
            ys = [lin(x) for layer in lins for lin, x in zip(layer, X)]
            torch.cuda.synchronize()
            for k, y in enumerate(ys):
                i, c = k % len(shapes), k // len(shapes)
                checked += 1
                ok = torch.equal(bits(y), bits(ys[i])) and (outs is None or torch.equal(bits(y), bits(outs[i])))
                err = ((y.float() - ref[i]).abs().max() / ref[i].abs().max()).item() if c == 0 else 0.0
                if not ok or not err < 1e-2:
                    failures.append(f"{name} M={M} GLinear, pass {pas}{' (the recording)' if pas == 0 else ''}, layer {c}, matrix {shapes[i]}: "
                                    + ("not the same bits as " + ("the first layer's" if not torch.equal(bits(y), bits(ys[i])) else "the first pass's") if not ok else f"{err:.2e} off fp32"))
            outs = outs or ys[: len(shapes)]
        s = gm.Split.of.get(dev)
        print(f"{name} M={M}: {sum(1 for f in failures if f.startswith(f'{name} M={M}'))} failures; GLinear's ring {getattr(s, 'slots', '-')} x {getattr(s, 'slot', 0) >> 20} MiB ({time.time() - t0:.0f} s)", flush=True)
        gm.Split.stop(dev)
        gm.Split.of.pop(dev, None)
        if s:
            lib.ring_destroy(s.ring)
        del lins, X, ref
    del W, P
    torch.cuda.empty_cache()
print(f"split_stress: {checked} products checked, {len(failures)} failures")
for f in failures[:40]:
    print("  " + f)
sys.exit(1 if failures else 0)
