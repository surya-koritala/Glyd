"""A prompt's routes on this GPU, end to end: a forward pass over a prompt of each length (logits_to_keep=1: the pass
before its first token), Glyd by route in one process, bf16 in another (the same prompts: a fixed text's first N
tokens). Glyd as glyd.from_pretrained loads it (q, k, v and gate, up merged; eager: compile=False), each prompt route
forced on every GLinear:
  fused    the prompt kernel (mma_gemm_big: the weights decoded in its registers), the route the library gives it
           past 64 tokens here unless told otherwise;
  decoded  each matrix decoded whole, then cuBLAS, on the current stream (the route DECODE);
  ahead    each matrix decoded ahead, on a second stream beside the products before it, then cuBLAS (model.Ahead,
           the route AHEAD: GeForce Ada's and an A10's prompts), the scratch buffer holding two of the largest (as
           set_scratch sizes it where a GLinear decodes ahead);
  default  the library's own routes, as the package takes them (nothing forced).
Per length and route: two passes untimed (Ahead's first records the order), then REPS timed, the median; each
window's wall-clock start and end, for the GPU's clock and power beside it (the job's nvidia-smi samples).

    GLYD_GPU_LIB=LIB PYTHONPATH=TREE/bindings/python python route_e2e.py MODEL --mode glyd --layout mma|mma12 --out R.json
    python route_e2e.py MODEL --mode bf16 --out R.json          [--lengths 128,...,8192] [--routes fused,decoded,ahead] [--reps 3]"""
import argparse, json, os, statistics, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = ("The history of data compression begins long before computers. Telegraph operators shortened common words to "
        "save time on the wire, and Morse gave the most frequent letters the shortest codes. Shannon later showed that "
        "the average length of any code is bounded below by the entropy of its source, and Huffman found the optimal "
        "prefix code for a known distribution. ")
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--mode", required=True, choices=["glyd", "bf16"])
ap.add_argument("--layout", default="mma", choices=["mma", "mma12"])
ap.add_argument("--out", required=True)
ap.add_argument("--lengths", default="128,256,512,768,1024,2048,4096,8192")
ap.add_argument("--routes", default="fused,decoded,ahead")
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
lengths = [int(x) for x in args.lengths.split(",") if x]
tok = AutoTokenizer.from_pretrained(args.model)
ids = tok(TEXT * (max(lengths) // 50 + 2), return_tensors="pt").input_ids[0]
assert ids.numel() >= max(lengths)
R = {"model": args.model, "mode": args.mode, "layout": args.layout if args.mode == "glyd" else "bf16", "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": []}
if args.mode == "bf16":
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    routes = {"bf16": lambda: None}
else:
    import glyd
    from glyd.gpu import model as gm
    model = glyd.from_pretrained(args.model, layout=args.layout, compile=False).eval()
    R["glyd"], R["gpu_code"] = glyd.__version__, gm.gpu_code(torch.cuda.get_device_capability(), torch.cuda.get_device_name())
    lins = [m for m in model.modules() if isinstance(m, gm.GLinear)]
    own = {id(m): m.ahead for m in lins}
    own_get, own_decoded = gm.Ahead.get, gm.GLinear.decoded

    def route(name):
        def set_():
            for m in lins:
                m.ahead = own[id(m)] if name in ("fused", "default") else 65  # (a prompt's fused product below ahead: the step's C call too)
                m.step = m._step()
            if name == "ahead":  # the scratch buffer as a GPU whose route is AHEAD has it: two of its largest matrices
                gm.set_scratch(model, False)  # (else Ahead leaves a matrix it cannot place twice to the fused kernel)
            gm.Ahead.get = staticmethod((lambda d: None) if name == "decoded" else own_get)
            gm.GLinear.decoded = (lambda self, M: True) if name == "decoded" else own_decoded
        return set_

    routes = {r: route(r) for r in args.routes.split(",") if r}
R["weights_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
print(f"{args.model} {args.mode} {R['layout']} on {R['gpu']}: {R['weights_gb']} GB", flush=True)
for L in lengths:
    x = ids[:L].cuda()[None]
    for name, set_ in routes.items():
        set_()
        try:
            with torch.no_grad():
                for _ in range(2):
                    model(x, logits_to_keep=1)
                torch.cuda.synchronize()
                ts, t_start = [], time.time()
                for _ in range(args.reps):
                    t = time.perf_counter()
                    model(x, logits_to_keep=1)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t)
            row = {"route": name, "tokens": L, "ms": statistics.median(ts) * 1e3, "runs_ms": [t * 1e3 for t in ts], "window": [t_start, time.time()]}
        except torch.cuda.OutOfMemoryError as e:
            row = {"route": name, "tokens": L, "error": f"out of memory: {str(e)[:160]}"}
            torch.cuda.empty_cache()
        R["rows"].append(row)
        print(f"  {L:5} tokens, {name:7}: " + (f"{row['ms']:.1f} ms" if "ms" in row else row["error"]), flush=True)
        with open(args.out, "w") as f:
            json.dump(R, f, indent=1)
