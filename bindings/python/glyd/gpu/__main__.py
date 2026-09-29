"""python -m glyd.gpu fit MODEL [--gpu 48GB] [--context 8192]
    will MODEL (a Hugging Face repo id or a directory) fit the GPU, bf16 against Glyd
python -m glyd.gpu pack MODEL OUT [--no-merge] [--layout mma|mma12] [--device cuda:0]
    MODEL packed on the GPU, each pack checked against its weights, and saved in OUT as glyd-v1 (glyd-v2: a mixture of experts;
    --layout mma12: the 12-bit layout, glyd-v3, which an A10, A100 or H100 loads without packing again)
python -m glyd.gpu verify PATH [--device cuda:0]
    a saved checkpoint loaded, every packed tensor decoded and its sha256 checked against glyd.json, and every tensor
    saved as it is by its own there (a save of glyd 0.25 on)"""
import argparse
import glyd.gpu as gg

ap = argparse.ArgumentParser(prog="python -m glyd.gpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="command", required=True)
a = sub.add_parser("fit")
a.add_argument("model")
a.add_argument("--gpu", default="48GB", help="16GB, 24GB, 32GB, 48GB, 80GB, 96GB, 141GB, or the memory in bytes")
a.add_argument("--context", type=int, default=8192, help="tokens in the KV cache")
a = sub.add_parser("pack")
a.add_argument("model")
a.add_argument("out")
a.add_argument("--no-merge", action="store_true", help="q, k, v and gate, up saved as packs of their own")
a.add_argument("--layout", default="mma", choices=["mma", "mma12"], help="the packs' layout: mma, tiered (glyd-v1, glyd-v2); mma12, 12-bit (glyd-v3)")
a.add_argument("--device", default="cuda:0")
a = sub.add_parser("verify")
a.add_argument("path")
a.add_argument("--device", default="cuda:0")
args = ap.parse_args()

if args.command == "fit":
    print(gg.fit(args.model, gpu=int(args.gpu) if args.gpu.isdigit() else args.gpu, context=args.context))
elif args.command == "pack":
    model = gg.from_pretrained(args.model, device=args.device, layout=args.layout, merge=not args.no_merge, verify=True)  # the layout saved
    gg.save_pretrained(model, args.out, layout=args.layout)
    print(f"{args.model}: {model.config.quantization_config.verified} tensors packed and checked, saved in {args.out}")
else:
    model = gg.from_pretrained(args.path, device=args.device, verify=True)
    q = model.config.quantization_config
    rest = f"{q.hashed} saved as they are match theirs" if not q.unhashed else f"{q.unhashed} saved as they are not checked (saved before glyd 0.25: no sha256 for them)"
    print(f"{args.path}: {q.verified} tensors decode to glyd.json's sha256, {rest}")
