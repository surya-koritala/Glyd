"""Every bf16 tensor of a safetensors file packed on the GPU, unpacked,
compared bit for bit; bits a weight; decode speed on the largest."""
import sys, time, torch
from safetensors.torch import load_file
import glyd_gpu as g

tensors = {k: v for k, v in load_file(sys.argv[1], device="cuda").items() if v.dtype == torch.bfloat16}
n_all, bytes_all = 0, 0
biggest = None
for name, w in tensors.items():
    p = g.pack(w)
    back = g.unpack(p)
    assert torch.equal(back.view(torch.int16), w.view(torch.int16)), name
    n_all += p.n
    bytes_all += p.nbytes()
    if biggest is None or p.n > biggest[1].n:
        biggest = (name, p, w)
print(f"{len(tensors)} tensors, {n_all/1e6:.0f}M weights, all exact: {bytes_all*8/n_all:.3f} bits a weight ({bytes_all/1e6:.0f} MB against {n_all*2/1e6:.0f} MB bf16, {100*bytes_all/(n_all*2):.1f}%)")
name, p, w = biggest
out = torch.empty(p.n, dtype=torch.bfloat16, device="cuda")
copy = torch.empty_like(out)
for _ in range(3):
    g.unpack(p, out)
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(20):
    g.unpack(p, out)
e1.record()
torch.cuda.synchronize()
t = e0.elapsed_time(e1) / 20 / 1e3
e0.record()
for _ in range(20):
    copy.copy_(w.flatten())
e1.record()
torch.cuda.synchronize()
tc = e0.elapsed_time(e1) / 20 / 1e3
print(f"decode {name} ({p.n/1e6:.0f}M weights): {t*1e3:.3f} ms, {p.n*2/t/1e9:.0f} GB/s of bf16 out; a plain bf16 copy of it: {tc*1e3:.3f} ms, {p.n*2/tc/1e9:.0f} GB/s")
