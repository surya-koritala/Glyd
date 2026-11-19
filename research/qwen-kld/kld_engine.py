#!/usr/bin/env python3
"""Our engine's logits on the same chunks: `glyd-engine qwenserve CKPT PACK [--precise]` (JSON lines on stdin) reads each chunk's first
n/2+1 tokens, then the rest teacher-forced one at a time ("score_from"), and appends each scored row of logits to OUT ("dump"):
the same positions as kld_ref.py; then a decode speed on one fixed prompt.   python kld_engine.py --tokens chunks.json --out ours.f32 -- ENGINE qwenserve CKPT PACK [--precise]
--ref REF.f32 [--top K]: each chunk scored against kld_ref.py's rows as it is made (kld_compare.py's numbers, the whole vocabulary and, with --top, its K most likely
tokens) and its rows dropped, so that OUT never holds more than a chunk: the numbers in OUT.kl.json ({"full": ..., "topK": ...})."""
import argparse, json, os, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--tokens", required=True); ap.add_argument("--out", required=True); ap.add_argument("cmd", nargs=argparse.REMAINDER)
ap.add_argument("--from", dest="start", type=int, help="as kld_ref.py's --from")
ap.add_argument("--rows", type=int, default=1, help="ids read this many a pass, the way drafts are verified (the multi-row kernels)")
ap.add_argument("--prefill-steps", action="store_true", help="the first ids read one at a time too (the per-pair products in place of the prompt kernels)")
ap.add_argument("--ref", help="kld_ref.py's rows: score each chunk against them as it is made, keep no rows")
ap.add_argument("--top", type=int, help="with --ref: also the KL over the reference's K most likely tokens")
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
docs = json.load(open(a.tokens))["docs"]
out = os.path.abspath(a.out)
open(out, "wb").close()
if a.ref:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kld_compare as C
    rm = json.load(open(a.ref + ".json"))
    V, per = rm["vocab"], rm["rows_per_chunk"]
    ref = np.memmap(a.ref, dtype="<f4", mode="r").reshape(-1, V)
    assert ref.shape[0] == len(docs) * per, (ref.shape, len(docs), per)
    parts = {"full": [], f"top{a.top}": []} if a.top else {"full": []}
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
t0, nll = time.time(), []
for i, d in enumerate(docs):
    p.stdin.write(json.dumps({"ids": d, "score_from": (len(d) // 2 if a.start is None else a.start) + 1, "dump": out, "rows": a.rows, "prefill_steps": a.prefill_steps}) + "\n"); p.stdin.flush()
    r = json.loads(p.stdout.readline())
    if "error" in r:
        sys.exit(f"chunk {i}: {r['error']}")
    nll.append(r["nll"])
    if a.ref:
        ours = np.memmap(out, dtype="<f4", mode="r")
        ours = ours.reshape(per, -1)
        rows, tg = ref[i * per:(i + 1) * per], np.array(d[len(d) - per:])
        # (128 positions at a time: the engine holds most of the memory)
        for k in parts:
            bl = [C.chunk(rows[j:j + 128], ours[j:j + 128, :V], tg[j:j + 128], a.top if k != "full" else None) for j in range(0, per, 128)]
            parts[k].append(tuple(np.concatenate([b[x] for b in bl]) for x in range(3)) + (sum(b[3] for b in bl),))
        del ours
        open(out, "wb").close()
    print(f"chunk {i + 1}/{len(docs)}: nll {r['nll']:.4f}, {time.time() - t0:.0f} s", flush=True)
if a.ref:
    json.dump({k: C.summary(v, len(docs[0]), a.top if k != "full" else None) for k, v in parts.items()}, open(out + ".kl.json", "w"), indent=1)
# decode speed on the same prompt for every build: greedy, the default drafts, 512 tokens (the first 64 tokens of chunk 0; the prompt's
# own time, from a 1-token request, taken off). Best of 3.
def ask(o):
    p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())
speed = []
for _ in range(3):
    base = {"ids": docs[0][:64], "ignore_eos": True, "reuse": False}
    one, full = ask({**base, "max_new": 1}), ask({**base, "max_new": 512})
    speed.append(511 / ((full["ms"] - one["ms"]) / 1e3))
print(f"decode {max(speed):.1f} tok/s (best of 3: {', '.join(f'{x:.1f}' for x in speed)})", flush=True)
p.stdin.close(); p.wait()
json.dump({"chunks": len(docs), "nll_per_chunk": nll, "seconds": round(time.time() - t0), "cmd": cmd, "decode_tok_s": round(max(speed), 1)}, open(out + ".json", "w"))
