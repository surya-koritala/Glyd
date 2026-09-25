"""Every bf16 tensor of a safetensors file packed on the GPU, unpacked,
compared bit for bit; bits a weight; decode and fused matrix-vector speed
against bf16."""
import sys, torch
import torch.nn.functional as F
from safetensors.torch import load_file
import glyd_gpu as g

tensors = {k: v for k, v in load_file(sys.argv[1], device="cuda").items() if v.dtype == torch.bfloat16}
n_all, bytes_all = 0, 0
packs = {}
for name, w in tensors.items():
    p = g.pack(w)
    assert torch.equal(g.unpack(p).view(torch.int16), w.view(torch.int16)), name
    if p.rows_per_tile:
        idx = torch.randint(0, w.shape[0], (7,), device="cuda")
        assert torch.equal(g.rows(p, idx).view(torch.int16), w[idx].view(torch.int16)), name + " rows"
    n_all += p.n
    bytes_all += p.nbytes()
    packs[name] = (p, w)
print(f"{len(tensors)} tensors, {n_all/1e6:.0f}M weights, all exact: {bytes_all*8/n_all:.3f} bits a weight ({bytes_all/1e6:.0f} MB against {n_all*2/1e6:.0f} MB bf16, {100*bytes_all/(n_all*2):.1f}%)")


def timed(f, reps=50):
    for _ in range(3):
        f()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        f()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps / 1e3


shown = set()
for name, (p, w) in sorted(packs.items(), key=lambda kv: -kv[1][0].n):
    if not p.rows_per_tile or w.shape in shown:
        continue
    shown.add(w.shape)
    x = torch.randn(w.shape[1], dtype=torch.bfloat16, device="cuda")
    ref = F.linear(x, w)
    y = g.gemv(p, x)
    err = ((y.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    out = torch.empty(p.n, dtype=torch.bfloat16, device="cuda")
    t_ref, t_gemv = timed(lambda: F.linear(x, w)), timed(lambda: g.gemv(p, x))
    f = g.pack_fast(w)
    assert torch.equal(g.fast_unpack(f).view(torch.int16), w.view(torch.int16)), name + " fast"
    idx = torch.randint(0, w.shape[0], (5,), device="cuda")
    assert torch.equal(g.fast_rows(f, idx).view(torch.int16), w[idx].view(torch.int16)), name + " fast rows"
    yf = g.fast_gemv(f, x)
    errf = ((yf.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    t_fast = timed(lambda: g.fast_gemv(f, x))
    print(f"{name.split('.')[-2]:>14} {tuple(w.shape)}: bf16 {t_ref*1e6:6.1f} us ({w.numel()*2/t_ref/1e9:4.0f} GB/s) | huffman {p.bits_per_weight():.2f} bits {t_gemv*1e6:6.1f} us ({t_ref/t_gemv:.2f}x) | fast {f.bits_per_weight():.2f} bits {t_fast*1e6:6.1f} us ({t_ref/t_fast:.2f}x, {f.nbytes()/t_fast/1e9:.0f} GB/s) | rel diff {err:.0e} {errf:.0e}")
