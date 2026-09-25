"""End to end: a Hugging Face causal LM generating with its weights held
compressed in VRAM (glyd_gpu), against the same model in bf16.

    python e2e.py MODEL_DIR --format fast|huffman [--fused] [--baseline] [--tokens N]

Exact path (default): each matrix decoded into a scratch buffer, then
PyTorch's own matmul: logits and tokens bit-identical to bf16. --fused:
one-token steps multiply straight from the packed weights (decoded in
registers, never written out); their sums are in another order than
cuBLAS's, as between any two GEMM kernels, so late tokens may differ.
The bf16 model is never held on the GPU: every Linear is packed from
the CPU copy, one at a time.
"""
import argparse, time, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import glyd_gpu as g

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--format", default="fast", choices=["fast", "huffman"])
ap.add_argument("--fused", action="store_true")
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--tokens", type=int, default=128)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
prompt = "The history of data compression began"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
SCRATCH = 128 << 20  # weights; bigger matrices are decoded in row blocks


def measure(model, label):
    torch.cuda.synchronize()
    with torch.no_grad():
        logits = model(ids, logits_to_keep=1).logits
        model.generate(ids, max_new_tokens=4, do_sample=False)  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        out = model.generate(ids, max_new_tokens=args.tokens, min_new_tokens=args.tokens, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter() - t
    print(f"{label}: {args.tokens / t:.1f} tokens/s, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return logits, out


class Scratch:
    buf = None


class GLinear(nn.Module):
    def __init__(self, p, bias):
        super().__init__()
        self.p, self.bias = p, bias
        O, K = p.shape
        step = getattr(p, "rows_per_tile", 1) or 1
        # Whole when it fits the scratch (a split matmul sums in another order).
        self.block = O if O * K <= SCRATCH else max(step, SCRATCH // K // step * step)

    def decode_rows(self, r0, r1):
        p, K = self.p, self.p.shape[1]
        out = Scratch.buf[: (r1 - r0) * K]
        if isinstance(p, g.Fast):
            g._ext.fast_decode(p.sm, p.planes, p.exc, p.exc_base, p.top, r0, r1 - r0, g._none(out.device), K, out.view(torch.int16))
        elif p.split:  # tiles split its rows: decoded whole (it fits the scratch)
            assert r0 == 0 and r1 == p.shape[0]
            g.unpack(p, Scratch.buf)
        else:
            T = p.rows_per_tile
            tiles = torch.arange(r0 // T, (r1 + T - 1) // T, device=out.device)
            full = Scratch.buf[: tiles.numel() * p.tw]
            g.decode_tiles(p, tiles, full)
            out = full[: (r1 - r0) * K]
        return out.view(r1 - r0, K)

    def forward(self, x):
        O, K = self.p.shape
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if args.fused and x2.shape[0] == 1:
            f = g.fast_gemv if isinstance(self.p, g.Fast) else g.gemv
            return f(self.p, x2[0], self.bias).view(*lead, O)
        if self.block >= O:
            return F.linear(x2, self.decode_rows(0, O), self.bias).view(*lead, O)
        y = torch.empty(x2.shape[0], O, dtype=x.dtype, device=x.device)
        for r0 in range(0, O, self.block):
            r1 = min(O, r0 + self.block)
            y[:, r0:r1] = F.linear(x2, self.decode_rows(r0, r1), None if self.bias is None else self.bias[r0:r1])
        return y.view(*lead, O)


class GEmbedding(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ids):
        rows = g.fast_rows(self.p, ids) if isinstance(self.p, g.Fast) else g.rows(self.p, ids)
        return rows.view(*ids.shape, -1)


model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
weights_bf16 = sum(p.numel() * p.element_size() for p in model.parameters())
if args.baseline:
    model.cuda()
    logits_a, out_a = measure(model, f"bf16 (weights {weights_bf16 / 1e9:.2f} GB)")
    model.cpu()
    torch.cuda.empty_cache()

pack = g.pack_fast if args.format == "fast" else g.pack
packed, biggest, t0 = {}, 0, time.perf_counter()
with torch.no_grad():
    for name, m in list(model.named_modules()):
        for cname, child in list(m.named_children()):
            if isinstance(child, (nn.Linear, nn.Embedding)):
                key = child.weight.data_ptr()
                if key not in packed:
                    packed[key] = pack(child.weight.data.cuda())
                p = packed[key]
                bias = child.bias.data.cuda() if isinstance(child, nn.Linear) and child.bias is not None else None
                setattr(m, cname, GLinear(p, bias) if isinstance(child, nn.Linear) else GEmbedding(p))
                biggest = max(biggest, min(p.n, SCRATCH))
    model.cuda()
    Scratch.buf = torch.empty(biggest + 16384 * 8, dtype=torch.bfloat16, device="cuda")
torch.cuda.empty_cache()
packed_bytes = sum(p.nbytes() for p in packed.values())
other = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(m.bias.numel() * 2 for m in model.modules() if isinstance(m, GLinear) and m.bias is not None)
print(f"{args.format}{' fused' if args.fused else ''}: packed in {time.perf_counter() - t0:.0f} s; weights {(packed_bytes + other) / 1e9:.2f} GB against {weights_bf16 / 1e9:.2f} GB bf16 ({100 * (packed_bytes + other) / weights_bf16:.1f}%), scratch {Scratch.buf.numel() * 2 / 1e9:.2f} GB, VRAM in use {torch.cuda.memory_allocated() / 1e9:.2f} GB")
logits_b, out_b = measure(model, f"glyd {args.format}{' fused' if args.fused else ''}")
if args.baseline:
    print("logits bit-identical:", torch.equal(logits_a.cuda().view(torch.int16), logits_b.view(torch.int16)))
    same = (out_a.cuda() == out_b).all(0).long().cumprod(0).sum().item() - ids.shape[1]
    print(f"generated tokens identical to bf16: {same} of {args.tokens}")
print("text:", tok.decode(out_b[0][ids.shape[1]:ids.shape[1] + 40]).replace("\n", " "))
