"""End to end: a Hugging Face causal LM generating with its weights held
compressed in VRAM (glyd_gpu), against the same model in bf16.

    python e2e.py MODEL_DIR --format fast|huffman|mma [--fused] [--baseline] [--tokens N]

Exact path (default): each matrix decoded into a scratch buffer, then
PyTorch's own matmul: logits and tokens bit-identical to bf16. --fused:
one-token steps multiply straight from the packed weights (decoded in
registers, never written out); their sums are in another order than
cuBLAS's, as between any two GEMM kernels, so late tokens may differ.
The bf16 model is never held on the GPU: every Linear is packed from
the CPU copy, one at a time.
"""
import argparse, math, time, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import glyd_gpu as g

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--format", default="fast", choices=["fast", "huffman", "mma"], help="mma: the Linears in the mma layout (the embedding in fast), up to 64 tokens a step multiplied straight from it")
ap.add_argument("--fused", action="store_true")
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--prefill", type=str, default="", help="prompt lengths to time one forward pass at, e.g. 128,512,2048")
ap.add_argument("--gemm-max", type=int, default=64, help="fused: steps of up to this many tokens multiply straight from the packed weights (fast format)")
ap.add_argument("--batch", type=str, default="1", help="generate for this many copies of the prompt at once (comma list: each measured)")
ap.add_argument("--gpus", type=int, default=1, help="spread the layers over this many GPUs (bf16: accelerate's device map; glyd: layers balanced by packed size)")
ap.add_argument("--ppl", default="", help="a text file: perplexity over 200 windows of 64 tokens (a forward pass each; from its 10th MB), and how often the next-token choice is bf16's")
ap.add_argument("--gpu-mem", type=float, default=0, help="GiB a GPU may hold of bf16 weights (the baseline's device map); default: all but 2 GiB")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
prompt = "The history of data compression began"
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
SCRATCH = 128 << 20  # weights; bigger matrices are decoded in row blocks


def prefill(model, label):
    """One forward pass over a prompt of each length: the prompt's tokens a second."""
    out = []
    for n in [int(x) for x in args.prefill.split(",") if x]:
        x = torch.randint(0, 150000, (1, n), device="cuda")
        with torch.no_grad():
            model(x, logits_to_keep=1)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(3):
                model(x, logits_to_keep=1)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t) / 3
        out.append(f"{n} tokens {t * 1e3:.0f} ms ({n / t:.0f} tokens/s)")
    if out:
        print(f"{label} prefill: " + ", ".join(out))


def perplexity(model, label):
    if not args.ppl:
        return None
    text = open(args.ppl, "rb").read()[10_000_000:10_400_000].decode("utf-8", "ignore")
    windows = tok(text, return_tensors="pt").input_ids[0][: 200 * 64].view(200, 64).cuda()
    nll, top = 0.0, []
    with torch.no_grad():
        for w in windows:
            lg = model(w[None]).logits[0, :-1].float()
            nll += F.cross_entropy(lg, w[1:], reduction="sum").item()
            top.append(lg.argmax(-1))
    top = torch.stack(top)
    print(f"{label} perplexity: {math.exp(nll / top.numel()):.4f} ({top.numel()} tokens)")
    return top


def measure(model, label):
    torch.cuda.synchronize()
    out = None
    with torch.no_grad():
        logits = model(ids, logits_to_keep=1).logits
        for b in [int(x) for x in args.batch.split(",")]:
            batch = ids.repeat(b, 1)
            model.generate(batch, max_new_tokens=4, do_sample=False)  # warm-up
            torch.cuda.synchronize()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            t = time.perf_counter()
            o = model.generate(batch, max_new_tokens=args.tokens, min_new_tokens=args.tokens, do_sample=False)
            torch.cuda.synchronize()
            t = time.perf_counter() - t
            peaks = [torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())]
            used = [f"{x:.1f}" for x in peaks if x > 0.05]
            print(f"{label}: batch {b}: {b * args.tokens / t:.1f} tokens/s ({args.tokens / t:.1f} a sequence), peak VRAM {sum(peaks):.2f} GB" + (f" ({' + '.join(used)} GB on {len(used)} GPUs)" if len(used) > 1 else ""))
            out = o[:1] if out is None else out
    return logits, out


class Scratch:
    buf = None  # one a device: {device: tensor}


class GLinear(nn.Module):
    def __init__(self, p, bias):
        super().__init__()
        self.p, self.bias = p, bias
        O, K = p.shape
        step = 64 if isinstance(p, g.Mma) else getattr(p, "rows_per_tile", 1) or 1
        # Whole when it fits the scratch (a split matmul sums in another order).
        self.block = O if O * K <= SCRATCH else max(step, SCRATCH // K // step * step)

    def decode_rows(self, r0, r1):
        p, K = self.p, self.p.shape[1]
        buf = Scratch.buf[p.sm.device]
        out = buf[: (r1 - r0) * K]
        if isinstance(p, g.Mma):
            g.mma_unpack(p, out, r0, r1 - r0)
        elif isinstance(p, g.Fast):
            g._ext.fast_decode(p.sm, p.planes, p.exc, p.exc_base, p.top, r0, r1 - r0, g._none(out.device), K, out.view(torch.int16))
        elif p.split:  # tiles split its rows: decoded whole (it fits the scratch)
            assert r0 == 0 and r1 == p.shape[0]
            g.unpack(p, buf)
        else:
            T = p.rows_per_tile
            tiles = torch.arange(r0 // T, (r1 + T - 1) // T, device=out.device)
            full = buf[: tiles.numel() * p.tw]
            g.decode_tiles(p, tiles, full)
            out = full[: (r1 - r0) * K]
        return out.view(r1 - r0, K)

    def forward(self, x):
        O, K = self.p.shape
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if args.fused and isinstance(self.p, g.Mma) and x2.shape[0] <= 1024:
            return (g.mma_gemm if x2.shape[0] <= 64 else g.mma_gemm_big)(self.p, x2, self.bias).view(*lead, O)
        if args.fused and x2.shape[0] == 1:
            f = g.fast_gemv if isinstance(self.p, g.Fast) else g.gemv
            return f(self.p, x2[0], self.bias).view(*lead, O)
        n = x2.shape[0]
        if args.fused and isinstance(self.p, g.Fast) and 1 < n <= 16 and K % 512 == 0:
            # A few tokens: the batched product, the tokens padded to 2, 4, 8 or 16.
            m = 1 << (n - 1).bit_length()
            xp = x2 if m == n else torch.cat([x2, x2.new_zeros(m - n, K)])
            return g.fast_bgemv(self.p, xp, self.bias)[:n].view(*lead, O)
        if args.fused and isinstance(self.p, g.Fast) and n <= args.gemm_max and K % 64 == 0:
            return g.fast_gemm(self.p, x2, self.bias).view(*lead, O)
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
    if args.gpus > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.hooks import remove_hook_from_module
        cap = args.gpu_mem or torch.cuda.get_device_properties(0).total_memory / 2**30 - 2
        dm = infer_auto_device_map(model, max_memory={i: f"{cap}GiB" for i in range(args.gpus)}, no_split_module_classes=model._no_split_modules)
        dispatch_model(model, device_map=dm)
    else:
        model.cuda()
    logits_a, out_a = measure(model, f"bf16 (weights {weights_bf16 / 1e9:.2f} GB)")
    prefill(model, "bf16")
    top_a = perplexity(model, "bf16")
    if args.gpus > 1:
        remove_hook_from_module(model, recurse=True)
    model.cpu()
    torch.cuda.empty_cache()

# Where every decoder layer's weights go: contiguous runs of layers, balanced
# by their bytes, over args.gpus GPUs; the embedding on the first, the final
# norm and the output layer on the last.
layers = model.model.layers
layer_bytes = [sum(p.numel() for p in l.parameters()) for l in layers]
per_gpu, acc, gpu_of = sum(layer_bytes) / args.gpus, 0, []
for b in layer_bytes:
    gpu_of.append(min(args.gpus - 1, int(acc // per_gpu)))
    acc += b
layer_of = {id(m): gpu_of[i] for i, l in enumerate(layers) for m in l.modules()}
last = args.gpus - 1



def pack(w, linear):
    if args.format == "mma" and linear and w.shape[0] % 64 == 0 and w.shape[1] % 16 == 0:
        return g.pack_mma(w)
    return (g.pack if args.format == "huffman" else g.pack_fast)(w)


packed, biggest, t0 = {}, {}, time.perf_counter()
with torch.no_grad():
    for name, m in list(model.named_modules()):
        for cname, child in list(m.named_children()):
            if isinstance(child, (nn.Linear, nn.Embedding)):
                dev = torch.device("cuda", layer_of.get(id(child), 0 if isinstance(child, nn.Embedding) else last))
                key = (child.weight.data_ptr(), dev)
                if key not in packed:
                    packed[key] = pack(child.weight.data.to(dev), isinstance(child, nn.Linear))
                p = packed[key]
                bias = child.bias.data.to(dev) if isinstance(child, nn.Linear) and child.bias is not None else None
                setattr(m, cname, GLinear(p, bias) if isinstance(child, nn.Linear) else GEmbedding(p))
                biggest[dev] = max(biggest.get(dev, 0), min(p.n, SCRATCH))
    if args.gpus > 1:
        from accelerate import dispatch_model
        dm = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": last, "lm_head": last}
        dm.update({f"model.layers.{i}": d for i, d in enumerate(gpu_of)})
        dispatch_model(model, device_map=dm)
    else:
        model.cuda()
    Scratch.buf = {dev: torch.empty(n + 16384 * 8, dtype=torch.bfloat16, device=dev) for dev, n in biggest.items()}
torch.cuda.empty_cache()
packed_bytes = sum(p.nbytes() for p in packed.values())
other = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(m.bias.numel() * 2 for m in model.modules() if isinstance(m, GLinear) and m.bias is not None)
in_use = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
scratch = sum(b.numel() * 2 for b in Scratch.buf.values())
print(f"{args.format}{' fused' if args.fused else ''}: packed in {time.perf_counter() - t0:.0f} s; weights {(packed_bytes + other) / 1e9:.2f} GB against {weights_bf16 / 1e9:.2f} GB bf16 ({100 * (packed_bytes + other) / weights_bf16:.1f}%), scratch {scratch / 1e9:.2f} GB, VRAM in use {in_use / 1e9:.2f} GB on {args.gpus} GPU{'s' if args.gpus > 1 else ''}")
logits_b, out_b = measure(model, f"glyd {args.format}{' fused' if args.fused else ''}")
prefill(model, f"glyd {args.format}")
top_b = perplexity(model, f"glyd {args.format}")
if args.baseline and top_b is not None:
    print(f"next-token choice as bf16's: {(top_a.cuda() == top_b).float().mean().item() * 100:.2f}%")
if args.baseline:
    print("logits bit-identical:", torch.equal(logits_a.cuda().view(torch.int16), logits_b.view(torch.int16)))
    same = (out_a.cuda() == out_b).all(0).long().cumprod(0).sum().item() - ids.shape[1]
    print(f"generated tokens identical to bf16: {same} of {args.tokens}")
print("text:", tok.decode(out_b[0][ids.shape[1]:ids.shape[1] + 40]).replace("\n", " "))
