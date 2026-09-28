# The products whose code the branch touched without meaning to change their bits, through the library in
# GLYD_GPU_LIB (GLYD_TREE's Python): saved to OUT.pt, or compared with OLD.pt.  python bitcmp.py OUT [OLD]
import os, sys, torch
sys.path.insert(0, os.path.expanduser(os.environ.get("GLYD_TREE", "~/glyd") + "/gpu"))
import glyd_gpu as g
torch.manual_seed(0)
out = {}
for O, K, wild in [(64, 64, 0), (192, 128, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (6144, 4096, 0.0003)]:
    w = torch.randn(O, K, device="cuda") * 0.02
    m = torch.rand(O, K, device="cuda") < wild
    w[m] = torch.randn(int(m.sum()), device="cuda") * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device="cuda").float())
    w = w.to(torch.bfloat16)
    q, t = g.pack_mma12(w), g.pack_mma(w)
    b = torch.randn(O, device="cuda").to(torch.bfloat16)
    for M in [1, 7, 16, 17, 24, 32, 33, 48, 64]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        out[f"mid {O}x{K} {M}"] = g.mma_gemm_mid(q, x, b)
        out[f"mma12 {O}x{K} {M}"] = g.mma_gemm(q, x, b)
    for M in [65, 128, 300]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        out[f"big12 {O}x{K} {M}"] = g.mma_gemm_big(q, x, b)
        out[f"big {O}x{K} {M}"] = g.mma_gemm_big(t, x, b)
    out[f"unpack12 {O}x{K}"] = g.mma_unpack(q)
E, O, K = 8, 256, 512
w = (torch.randn(E * O, K, device="cuda") * 0.02).to(torch.bfloat16)
q = g.pack_mma12(w)
for T in [3, 40, 200]:
    ids = torch.randint(0, E, (T, 2), device="cuda")
    x = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    plan = g.moe_route(ids, E)
    out[f"moe12 {T}"] = g.mma_moe(q, E, x, plan, ids)
    out[f"moe12 act {T}"] = g.mma_moe(q, E, x, plan, ids, act=1)
if len(sys.argv) > 2:
    old = torch.load(sys.argv[2])
    bad = [k for k in out if not torch.equal(out[k].view(torch.int16), old[k].cuda().view(torch.int16))]
    print(f"{len(out)} products compared bit for bit with {sys.argv[2]}: {len(out) - len(bad)} identical" + (f"; differ: {bad}" if bad else ""))
else:
    torch.save({k: v.cpu() for k, v in out.items()}, sys.argv[1])
    print(f"{len(out)} products saved to {sys.argv[1]}")
