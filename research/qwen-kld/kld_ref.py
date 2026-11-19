#!/usr/bin/env python3
"""The bf16 reference's logits for KL divergence: research/qwen-data/qwen_ref.py's layer-at-a-time forward of the BF16 checkpoint
(each layer's weights read only while it runs, the GPU lock held per layer), QWEN_BATCH chunks per forward, the LM head in fp32.
For each chunk of n tokens the rows of positions n/2 .. n-2 (the logits that predict tokens n/2+1 .. n-1, as llama.cpp's
perplexity tool scores a chunk's second half), as little-endian float32 rows of the whole vocabulary, appended to OUT.
  QWEN_SNAP=.../flash-next python kld_ref.py --tokens chunks.json --out ref.f32
--pack DIR: the routed experts of --pack-layers as that pack rounds them (codes and codebooks back to bf16, the rotation undone), the rest
exact: the KL of the experts' rounding alone, apart from the engine's other approximations. --routes FILE: without --pack, each layer's
expert picks saved there; with it, those picks forced (the weights still from the router), to take routing flips out of the KL.
A layer with format q rows (q, q3, q4) is decoded by the engine (QXTQ_ENGINE: glyd-engine qxtq decode) into /dev/shm and read back.
--requant cb2|cb3|cb4 in place of --pack: every routed expert rounded on the fly as qwen_pack.py packs it (nearest entry of each row's
fitted codebook, in the rotated basis), for KL against the size of the error.
  QWEN_SNAP=.../flash-next python kld_ref.py --tokens chunks.json --out q.f32 --pack PACK [--pack-layers 0:48] [--routes R.pt]"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen-data"))
import qwen_ref as R
import qwen_pack as Q
from safetensors import safe_open

ap = argparse.ArgumentParser(); ap.add_argument("--tokens", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--from", dest="start", type=int, help="the first position whose next token is scored (default the chunk's middle; 0 scores every one)")
ap.add_argument("--pack"); ap.add_argument("--pack-layers", default="0:48"); ap.add_argument("--routes")
ap.add_argument("--requant", choices=["cb2", "cb3", "cb4"])
a = ap.parse_args()
docs = json.load(open(a.tokens))["docs"]
B = int(os.environ.get("QWEN_BATCH", "16"))
ck = R.Ckpt(R.SNAP)
model, tc = R.build(ck)
if a.pack or a.requant:
    meta, (l0, l1), get, done = json.load(open(f"{a.pack}/meta.json"))["layers"] if a.pack else None, map(int, a.pack_layers.split(":")), ck.get, {}

    def rounded(name):
        """The pack's experts for a checkpoint name: (512, 1280, 2560) gate_up_proj or (512, 2560, 640) down_proj, bf16 on the GPU."""
        m = re.fullmatch(rf"{re.escape(R.P)}layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)", name)
        if not m or not l0 <= int(m[1]) < l1:
            return get(name)
        up, s01 = m[2] == "gate_up_proj", Q.layer_signs(int(m[1]))
        if a.pack and any(f.startswith("q") for f in meta[m[1]]["f"]):
            out = f"/dev/shm/qxtq_{m[1]}.safetensors"
            if not os.path.exists(out):
                subprocess.run([os.environ["QXTQ_ENGINE"], "qxtq", "decode", a.pack, m[1], out], check=True)
            t = safe_open(out, "pt", device="cpu").get_tensor(m[2]).to(R.DEV)
            done.setdefault(m[1], set()).add(m[2])
            if len(done[m[1]]) == 2:
                os.remove(out)
            return t
        if a.pack:
            lm = meta[m[1]]
            f, s01 = safe_open(f"{a.pack}/{lm['file']}", "pt", device="cpu"), torch.tensor(lm["s"], dtype=torch.uint8)
        else:
            W = get(name).to(R.DEV)
        sgn = s01.to(R.DEV).float() * 2 - 1
        out = torch.empty((Q.NE, 2 * Q.INTER, Q.HID) if up else (Q.NE, Q.HID, Q.INTER), dtype=torch.bfloat16, device=R.DEV)
        for e in range(Q.NE):
            part = "up" if up else "downT"
            if a.pack:
                q = f.get_tensor(Q.tname(e, part, "codebook")).to(R.DEV).float().gather(1, Q.unpack_bits(f.get_tensor(Q.tname(e, part, "codes")).to(R.DEV), Q.FMT_IN[lm["f"][e]]).long())
            else:
                x = Q.rotate(W[e] if up else W[e].T, s01)
                cb = Q.fit_codebooks(x, Q.K_ENTRIES[a.requant])
                q = cb.gather(1, Q.codes_of(x, cb).long())
            w = ((q.reshape(-1, 5, 512) @ Q.hadamard()) * 512 ** -0.5).reshape(q.shape) * sgn   # rows back from the rotated basis
            out[e] = w if up else w.T
        return out
    ck.get = rounded
routes, nb = (torch.load(a.routes) if a.pack and a.routes else {}), [0]
for l, layer in enumerate(model.layers if a.routes else ()):
    def route(m, i, o, l=l):
        if not a.pack:
            routes.setdefault(l, []).append(o[2].to(torch.int16).cpu())
            return
        idx = routes[l][nb[0]].to(o[2].device).long()
        w = o[0].float().softmax(-1).gather(1, idx)
        return o[0], (w / w.sum(-1, keepdim=True) if m.norm_topk_prob else w).to(o[1].dtype), idx
    layer.mlp.gate.register_forward_hook(route)
t0 = time.time()
open(a.out, "wb").close()
with torch.no_grad():
    for b in range(0, len(docs), B):
        batch = torch.tensor(docs[b:b + B], dtype=torch.long)
        n = batch.shape[1]
        s = n // 2 if a.start is None else a.start
        final = model(input_ids=batch.to(R.DEV), use_cache=False).last_hidden_state[:, s:n - 1].float()
        with R.Lock():
            head = ck.get("lm_head.weight").to(R.DEV).float()
            with open(a.out, "ab") as f:
                for row in final:
                    f.write((row @ head.T).cpu().numpy().astype("<f4").tobytes())
            vocab = head.shape[0]
            del head
            torch.cuda.empty_cache()
        print(f"chunks {b + len(batch)}/{len(docs)}, {time.time() - t0:.0f} s", flush=True)
        nb[0] += 1
if a.routes and not a.pack:
    torch.save(routes, a.routes)
json.dump({"vocab": vocab, "rows_per_chunk": n - 1 - s, "chunks": len(docs), "seconds": round(time.time() - t0)}, open(a.out + ".json", "w"))
