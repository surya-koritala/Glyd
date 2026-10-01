# FlashInfer's top-k/top-p sampler against vLLM's PyTorch one, on the same logits: their draws' frequencies against the
# exact distribution (top-k 20 then top-p 0.8 at temperature 0.7), 400,000 draws each.
import time
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, flashinfer_sample

torch.manual_seed(0)
V, B, N = 151936, 256, 400_000 // 256 * 256
logits = (torch.randn(1, V, device="cuda") * 2.5)
logits[0, :6] += torch.tensor([9, 8.5, 8, 7.5, 7, 6.5], device="cuda")
T, K, P = 0.7, 20, 0.8
L = (logits / T).expand(B, V).contiguous()
k = torch.full((B,), K, device="cuda", dtype=torch.int32)
p = torch.full((B,), P, device="cuda", dtype=torch.float32)

def exact():
    l = apply_top_k_top_p(L[:1].clone(), k[:1], p[:1])
    return torch.softmax(l.float(), -1)[0]

def draw(kind):
    cnt = torch.zeros(V, device="cuda", dtype=torch.float64)
    t0 = time.perf_counter()
    for _ in range(N // B):
        if kind == "flashinfer":
            ids = flashinfer_sample(L.clone(), k, p, {})
        else:
            l = apply_top_k_top_p(L.clone(), k, p)
            ids = torch.multinomial(torch.softmax(l.float(), -1), 1).squeeze(1)
        cnt += torch.bincount(ids, minlength=V).double()
    torch.cuda.synchronize()
    return cnt / cnt.sum(), (time.perf_counter() - t0) / (N // B) * 1e3

ex = exact()
print("support of the exact distribution:", int((ex > 0).sum()), "tokens")
res = {}
for kind in ("pytorch", "flashinfer"):
    try:
        f, ms = draw(kind)
    except Exception as e:
        print(kind, "unavailable:", type(e).__name__, str(e)[:160]); continue
    res[kind] = f
    outside = float(f[ex == 0].sum())
    tv = 0.5 * float((f - ex).abs().sum())
    print(f"{kind:10s} total variation from the exact {tv:.4f}, mass outside the support {outside:.1e}, {ms:.3f} ms a batch of {B}")
if len(res) == 2:
    print(f"flashinfer against pytorch: total variation {0.5 * float((res['flashinfer'] - res['pytorch']).abs().sum()):.4f}")
