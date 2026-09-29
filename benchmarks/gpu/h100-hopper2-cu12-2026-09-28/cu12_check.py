"""One library's Hopper products on the self-test's matrices (odd row blocks, split tiles, exceptions few and many, past
a stage's copy), 1-2100 tokens, with and without bias: each within 1e-2 of the fp32 product and the same every run, and
each output's sha256 into OUT (JSON) so that libraries can be compared bit for bit.
    GLYD_GPU_DIR=tree/gpu GLYD_GPU_LIB=lib.so python cu12_check.py OUT.json
Off Hopper (a smoke test) the same for mma_gemm_mid to CHECK_MAX tokens (600).
"""
import hashlib, json, os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, os.environ["GLYD_GPU_DIR"])
import glyd_gpu as g

hopper = torch.cuda.get_device_capability() == (9, 0)
name, prod = ("mma_gemm_wg", g.mma_gemm_wg) if hopper else ("mma_gemm_mid", g.mma_gemm_mid)
most = int(os.environ.get("CHECK_MAX", 2100 if hopper else 600))
torch.manual_seed(0)
sha, bad = {}, []
for O, K, wild in [(64, 64, 0), (192, 128, 0), (128, 4096, 0), (1024, 2048, 0), (5120, 1024, 0.001), (192, 4096, 0.1), (3072, 5120, 0.02), (17408, 1024, 0.01)]:
    w = torch.randn(O, K, device="cuda") * 0.02
    m = torch.rand(O, K, device="cuda") < wild  # this share of weights at exponents far from the commonest 15
    w[m] = torch.randn(int(m.sum()), device="cuda") * torch.exp2(torch.randint(-40, 20, (int(m.sum()),), device="cuda").float())
    w = w.to(torch.bfloat16)
    q = g.pack_mma12(w)
    bias = torch.randn(O, device="cuda").to(torch.bfloat16)
    for M in [1, 7, 16, 17, 32, 33, 64, 65, 100, 128, 129, 160, 256, 257, 384, 600, 1024, 1100, 2100]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")  # (drawn at every M, run or not: the same inputs for every CHECK_MAX)
        if M > most:
            continue
        for b in (None, bias):
            ref = F.linear(x.float(), w.float(), None if b is None else b.float())
            y = prod(q, x, b)
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            key = f"{O}x{K} M={M} bias={int(b is not None)}"
            if not (err < 1e-2 and torch.equal(y, prod(q, x, b))):
                bad.append([key, err])
            sha[key] = hashlib.sha256(y.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]
json.dump({"product": name, "library": os.environ.get("GLYD_GPU_LIB"), "bad": bad, "sha": sha}, open(sys.argv[1], "w"), indent=1)
print(f"{name}: {len(sha)} products, {len(bad)} not within 1e-2 or not the same every run")
sys.exit(1 if bad else 0)
