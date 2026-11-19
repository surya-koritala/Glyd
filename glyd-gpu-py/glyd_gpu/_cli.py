"""The commands of `python -m glyd_gpu` and of glyd's GPU half (glyd run, serve, doctor, login, pack, verify, fetch)."""
import argparse
import os
import subprocess
import sys

GPU = ("run", "serve", "doctor", "login")  # glyd run's: run.py
USAGE = """python -m glyd_gpu fit MODEL [--gpu 48GB] [--context 8192]
    will MODEL (a Hugging Face repo id or a directory) fit the GPU, bf16 against Glyd
python -m glyd_gpu pack MODEL OUT [--no-merge] [--layout mma|mma12] [--tables] [--device cuda:0]
    MODEL packed on the GPU, each pack checked against its weights, and saved in OUT as glyd-v4 (each matrix in the smaller of its two layouts;
    --layout mma12: the faster layout, glyd-v3, which an A10, A100 or H100 loads without packing again; --tables: its embeddings and
    other large tables packed in the save too, a smaller save, glyd-v5, which glyd 0.29 and before do not load)
python -m glyd_gpu verify PATH [--device cuda:0]
    a saved checkpoint loaded, every packed tensor decoded and its sha256 checked against glyd.json, and every tensor
    saved as it is by its own there (a save of glyd 0.25 on)
python -m glyd_gpu fetch MODEL[:LEVEL]
    MODEL's files (or its level's pre-packed repo) downloaded into the Hugging Face cache, no GPU needed
python -m glyd_gpu run MODEL | serve MODEL | doctor | login
    the local chat (glyd run --help)
python -m glyd_gpu kvcal MODEL [--positions N] [--out FILE] [--engine JSON]
    the lossless KV cache's tables for MODEL, made on vLLM itself (the vLLM plugin makes them on a first start)"""


def main(argv=None, prog="python -m glyd_gpu"):
    """The exit status of a command: argv's (default sys.argv[1:]), prog the name usage texts give."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in GPU:
        from . import run  # (the chat's half is imported where it is asked for)

        code = run.main(args[0], args[1:])
        if code in (129, 130, 143):  # (Ctrl-C, SIGHUP, SIGTERM: huggingface_hub's download threads are not daemons, and a normal exit would wait for the shard in flight)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
        return code
    if args and args[0] == "kvcal":  # a model's tables for the KV cache: the plugin runs this as a child process on a first start (a compiled module cannot be run with -m)
        from . import kvcal

        return kvcal.main(args[1:])
    import glyd_gpu as gg

    ap = argparse.ArgumentParser(prog=prog, description=USAGE.replace("python -m glyd_gpu", prog), formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("fit")
    a.add_argument("model")
    a.add_argument("--gpu", default="48GB", help="16GB, 24GB, 32GB, 48GB, 80GB, 96GB, 141GB, or the memory in bytes")
    a.add_argument("--context", type=int, default=8192, help="tokens in the KV cache")
    a = sub.add_parser("pack")
    a.add_argument("model")
    a.add_argument("out")
    a.add_argument("--no-merge", action="store_true", help="q, k, v and gate, up saved as packs of their own")
    a.add_argument("--layout", default="mma", choices=["mma", "mma12"], help="the packs' layout: mma, the smallest (glyd-v4); mma12, the faster (glyd-v3)")
    a.add_argument("--tables", action="store_true", help="embeddings and other large tables packed in the save too (glyd-v5: smaller; glyd 0.29 and before do not load it)")
    a.add_argument("--device", default="cuda:0")
    a = sub.add_parser("verify")
    a.add_argument("path")
    a.add_argument("--device", default="cuda:0")
    a = sub.add_parser("fetch")
    a.add_argument("model")
    args = ap.parse_args(args)
    from . import engine

    if args.command == "fetch":  # (the engine's download: the Hub cache, resumed where it stopped, every file's sha256)
        exe = engine.find()
        if exe is None:
            print(f"glyd fetch: {engine.NAME} is not installed here", file=sys.stderr)
            return 1
        return subprocess.run([exe, "fetch", args.model]).returncode

    if args.command == "fit":
        print(gg.fit(args.model, gpu=int(args.gpu) if args.gpu.isdigit() else args.gpu, context=args.context))
    elif args.command == "pack":
        model = gg.from_pretrained(args.model, device=args.device, layout=args.layout, merge=not args.no_merge, verify=True)  # the layout saved
        gg.save_pretrained(model, args.out, layout=args.layout, tables=args.tables)
        print(f"{args.model}: {model.config.quantization_config.verified} tensors packed and checked, saved in {args.out}")
    else:
        # a level's save (glyd-v7) or a Hub name: the engine's verify, which reads every level and the Hub's cache, on the CPU (a Hub name not downloaded yet is fetched first)
        manifest = os.path.join(args.path, "glyd.json")
        newer = not os.path.isdir(args.path) or ('"glyd-v7"' in open(manifest, encoding="utf-8").read(4096) if os.path.isfile(manifest) else False)
        exe = engine.find() if newer else None
        if exe:
            if not os.path.isdir(args.path):
                got = subprocess.run([exe, "fetch", args.path]).returncode
                if got != 0:
                    return got
            return subprocess.run([exe, "verify", args.path]).returncode
        model = gg.from_pretrained(args.path, device=args.device, verify=True)
        q = model.config.quantization_config
        rest = f"{q.hashed} saved as they are match theirs" if not q.unhashed else f"{q.unhashed} saved as they are not checked (saved before glyd 0.25: no sha256 for them)"
        print(f"{args.path}: {q.verified} tensors decode to glyd.json's sha256, {rest}")
    return 0
