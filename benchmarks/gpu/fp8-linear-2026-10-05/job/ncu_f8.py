# one shape's f8_linear, eager (for ncu): python ncu_f8.py N K M kind
import sys, torch
from glyd_gpu import kernels as g
N, K, M, kind = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
gen = torch.Generator(device="cuda").manual_seed(0)
w = torch.randn(N, K, device="cuda", generator=gen)
w = (w / w.abs().amax() * 448).to(torch.float8_e4m3fn).view(torch.uint8)
p = g.f8_pack(w)
sc = {"block": torch.full((-(-N // 128), -(-K // 128)), 1e-3, device="cuda"), "tensor": torch.tensor(1e-3, device="cuda"),
      "mx": torch.full((N, -(-K // 32)), 120, dtype=torch.uint8, device="cuda"), "ds32": torch.full((-(-N // 32), -(-K // 32)), 120, dtype=torch.uint8, device="cuda")}[kind]
x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
for _ in range(3):
    y = g.f8_linear(p, kind, sc, x)
torch.cuda.synchronize()
print("ok", y.shape, p.bits_per_value())
