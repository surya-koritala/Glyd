"""A model end to end in the 12-bit layout as a user loads it (glyd.from_pretrained(MODEL, layout="mma12"), eager):
exact mode's steps (generate(): TOKENS greedy tokens after a 16-token prompt) and long prompts (a forward pass over
each of PROMPTS tokens, logits_to_keep=1), exact and then fused (merged, as it loads by default). Every time the median
of REPS, the variants in turn within a repetition (the order rotated each time): the decode kernel's load orders
(GLYD_DEC_ORDER, this tree's library) or, with --orders '', the library as it is (main's, v0.25.0's). Each variant's
outputs as sha256 lines (the generated tokens, each prompt's logits), to be the same in every variant and tree.

    GLYD_COMPILE=0 PYTHONPATH=TREE/bindings/python GLYD_GPU_LIB=LIB python dec_e2e.py MODEL [--orders 0,1,2,3] [--tokens 64] [--reps 5] [--prompts 1280,2048,4096] [--modes exact,fused]"""
import argparse, gc, hashlib, os, time
import torch
import glyd
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--orders", default="0,1,2,3")
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--prompts", default="1280,2048,4096")
ap.add_argument("--modes", default="exact,fused")
args = ap.parse_args()
orders = [o for o in args.orders.split(",") if o] or [None]
lens = [int(n) for n in args.prompts.split(",") if n]
tok = AutoTokenizer.from_pretrained(args.model)
ids = tok("The history of data compression began with Morse code, which gave the commonest letters the shortest codes.", return_tensors="pt").input_ids[:, :16].cuda()


def use(o):
    if o is not None:
        os.environ["GLYD_DEC_ORDER"] = o  # read by this tree's library at each call


def label(mode, o):
    return f"{mode} {'as built' if o is None else 'order ' + o}"


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def wall(f):
    torch.cuda.synchronize()
    t = time.perf_counter()
    f()
    torch.cuda.synchronize()
    return time.perf_counter() - t


print(f"{torch.cuda.get_device_name()}, {args.model}, library {os.environ.get('GLYD_GPU_LIB')}", flush=True)
for mode in args.modes.split(","):
    t0 = time.perf_counter()
    m = glyd.from_pretrained(args.model, layout="mma12", exact=mode == "exact", compile=False).eval()
    print(f"{mode}: loaded in {time.perf_counter() - t0:.0f} s", flush=True)
    X = {n: torch.randint(0, m.config.vocab_size, (1, n), generator=torch.Generator().manual_seed(n)).cuda() for n in lens}
    gen = lambda: m.generate(ids, max_new_tokens=args.tokens, min_new_tokens=args.tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    T = {}
    with torch.no_grad():
        for o in orders:  # untimed: each variant's kernels loaded, its outputs' bits
            use(o)
            if mode == "exact":
                print(f"{label(mode, o)}: {args.tokens} tokens sha256 {sha(gen()[0, ids.shape[1]:])}", flush=True)
            for n in lens:
                m(X[n], logits_to_keep=1)
                print(f"{label(mode, o)}: {n}-token prompt's logits sha256 {sha(m(X[n], logits_to_keep=1).logits)}", flush=True)
        for r in range(args.reps):
            for o in orders[r % len(orders):] + orders[: r % len(orders)]:
                use(o)
                if mode == "exact":
                    T.setdefault((o, "steps"), []).append(wall(gen) / args.tokens)
                for n in lens:
                    T.setdefault((o, n), []).append(wall(lambda: m(X[n], logits_to_keep=1)))
    med = {k: sorted(v)[len(v) // 2] for k, v in T.items()}
    for o in orders:
        if mode == "exact":
            t = med[o, "steps"]
            print(f"{label(mode, o)}: steps {t * 1e3:.3f} ms a token ({1 / t:.2f} tokens/s)  [runs of {args.tokens}, ms a token: {' '.join(f'{x * 1e3:.3f}' for x in T[o, 'steps'])}]", flush=True)
        for n in lens:
            print(f"{label(mode, o)}: prompt {n} tokens {med[o, n] * 1e3:.2f} ms  [runs, ms: {' '.join(f'{x * 1e3:.2f}' for x in T[o, n])}]", flush=True)
    del m, gen
    gc.collect()
    torch.cuda.empty_cache()
print(f"median of {args.reps}", flush=True)
