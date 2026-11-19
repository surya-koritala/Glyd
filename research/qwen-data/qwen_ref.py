#!/usr/bin/env python3
"""Qwen3.8-Flash-Next reference forward, one decoder layer at a time on the GPU: transformers' qwen4_exp modules, with each layer's weights
read from the BF16 safetensors only while it runs (the GPU lock held per layer), and the 51B n-gram table read row by row through memory maps.
Produces the stages the engine's checks compare against, and the experts' usage profile for the pack.

  python qwen_ref.py dump    --tokens T.json --out DIR [--every 0]   L{l}.{stage}.bin files (--every N: `out` and the logits only every N-th
                                                                     position and the last 8; 0 = all positions)
  python qwen_ref.py profile --tokens T.json --out P.npz             routed counts and weight sums per (layer, expert), the shared gate's mean
T.json: {"docs": [[ids...], ...]} (each document its own forward, no cache between them).
"""
import argparse, fcntl, json, os, struct, time
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as M

SNAP = os.environ.get("QWEN_SNAP", "/home/surya-koritala/glyd-scratch/qwen/flash-next")
LOCK = os.environ.get("QWEN_GPU_LOCK", "/home/surya-koritala/glyd-scratch/gpu.lock")
DEV = "cuda"
P = "model.language_model."


class Ckpt:
    def __init__(self, snap):
        self.snap = snap
        self.map = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
        self.files = {}

    def get(self, name):
        fn = self.map[name]
        if fn not in self.files:
            self.files[fn] = safe_open(f"{self.snap}/{fn}", "pt", device="cpu")
        return self.files[fn].get_tensor(name)

    def raw(self, name):
        """(file, byte offset, shape, dtype) of a tensor, for a memory map."""
        path = f"{self.snap}/{self.map[name]}"
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        m = hdr[name]
        return path, 8 + n + m["data_offsets"][0], m["shape"], m["dtype"]


class NGramRows(torch.nn.Module):
    """The n-gram table (the checkpoint's shard_0..shard_K row blocks, in order) gathered row by row from memory maps; never loaded whole."""

    def __init__(self, ck, prefix, rows_expected):
        super().__init__()
        self.maps, starts, k = [], [0], 0
        while f"{prefix}.shard_{k}.weight" in ck.map:
            path, off, shape, dt = ck.raw(f"{prefix}.shard_{k}.weight")
            assert dt == "BF16", dt
            self.maps.append(np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=tuple(shape)))
            starts.append(starts[-1] + shape[0])
            k += 1
        assert starts[-1] == rows_expected, (starts[-1], rows_expected)
        self.starts = np.array(starts[:-1])
        self.register_buffer("weight", torch.zeros(1), persistent=False)   # (the caller reads the device of `weight`)
        self.last = None

    def forward(self, ids):
        flat = ids.reshape(-1).cpu().numpy()
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        out = np.empty((flat.size, self.maps[0].shape[1]), np.uint16)
        for j in np.unique(shard):
            sel = shard == j
            out[sel] = self.maps[j][flat[sel] - self.starts[j]]
        t = torch.from_numpy(out.view(np.int16)).view(torch.bfloat16).reshape(*ids.shape, -1)
        self.last = (ids.detach().cpu(), t)
        return t


class Lock:
    def __enter__(self):
        self.fh = open(LOCK, "a")
        fcntl.flock(self.fh, fcntl.LOCK_EX)

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def build(ck):
    tc = AutoConfig.from_pretrained(SNAP).text_config
    tc._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = M.Qwen4ExpTextModel(tc)
    model.rotary_emb = M.Qwen4ExpTextRotaryEmbedding(config=tc).to(DEV)
    model.embed_tokens = torch.nn.Embedding.from_pretrained(ck.get(P + "embed_tokens.weight").to(DEV), freeze=True)
    mixer = model.hyper_connection_mixer
    mixer.load_state_dict({k: ck.get(P + "hyper_connection_mixer." + k).to(DEV) for k in mixer.state_dict()}, assign=True)
    for l, layer in enumerate(model.layers):
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            emb.ngram_embedding = NGramRows(ck, f"{P}layers.{l}.ple.ple_embedding.ngram_embedding", emb.ngram_embedding.num_embeddings)
        orig = layer.forward

        def fwd(*a, _l=l, _layer=layer, _orig=orig, **k):
            with Lock():
                # (assign: each tensor keeps the checkpoint's dtype, as transformers loads it)
                _layer.load_state_dict({n: ck.get(f"{P}layers.{_l}.{n}").to(DEV) for n in _layer.state_dict()}, strict=True, assign=True)
                out = _orig(*a, **k)
                torch.cuda.synchronize()
                _layer.to_empty(device="meta")
            torch.cuda.empty_cache()
            return out

        layer.forward = fwd
    return model, tc


def bits(t):
    t = t.detach().contiguous().cpu()
    return t.view(torch.int16).numpy() if t.dtype == torch.bfloat16 else t.numpy()


def run_dump(model, tc, ck, docs, out, every):
    os.makedirs(out, exist_ok=True)
    ids = torch.tensor(docs[0], dtype=torch.long)
    T = ids.numel()
    pos = np.arange(T) if every == 0 else np.unique(np.concatenate([np.arange(0, T, every), np.arange(max(0, T - 8), T)]))
    keep_all = every == 0
    st = {}

    def save(name, t, rows_all=True):
        if not keep_all and not name.endswith(".out") and name != "final":
            return   # (the long case keeps the streams after each layer at the dumped positions, the final mixer output and the logits)
        a = bits(t.reshape(T, -1) if rows_all else t)
        if not keep_all and name.endswith(".out"):
            a = a[pos]
        a.tofile(f"{out}/{name}.bin")

    hooks = []
    for l, layer in enumerate(model.layers):
        def ahc(m, i, o, l=l):
            save(f"L{l}.a_in", o[0]); save(f"L{l}.a_inj", o[2].float())
        def mhc(m, i, o, l=l):
            save(f"L{l}.mid", i[0]); save(f"L{l}.m_in", o[0]); save(f"L{l}.m_inj", o[2].float())
        def att(m, i, o, l=l):
            save(f"L{l}.a_out", o[0] if isinstance(o, tuple) else o)
        def gate(m, i, o, l=l):
            save(f"L{l}.r_w", o[1].float()); save(f"L{l}.r_ids", o[2].int())
        def mlp(m, i, o, l=l):
            save(f"L{l}.m_out", o)
        def lay(m, i, o, l=l):
            save(f"L{l}.out", o)
        hooks += [layer.attn_hyper_connection.register_forward_hook(ahc), layer.mlp_hyper_connection.register_forward_hook(mhc),
                  (layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn).register_forward_hook(att),
                  layer.mlp.gate.register_forward_hook(gate), layer.mlp.register_forward_hook(mlp), layer.register_forward_hook(lay)]
        if layer.layer_type != "linear_attention" and not keep_all:
            def picks(m, i, o, l=l):
                mask = o[0] if isinstance(o, tuple) else o   # the indexer's token mask [B, 1, T, T], before the causal and
                mk = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
                rows = []
                for p in pos:
                    nz = torch.nonzero(mk[int(p)]).flatten().int().cpu().numpy()[:2051]
                    rows.append(np.pad(nz, (0, 2051 - len(nz)), constant_values=-1))
                np.stack(rows).astype(np.int32).tofile(f"{out}/L{l}.sel.bin")
            hooks.append(layer.self_attn.indexer.register_forward_hook(picks))
        if layer.ple is not None:
            def ple(m, i, o, l=l):
                ng_ids, ng = m.ple_embedding.ngram_embedding.last
                save(f"L{l}.ng_ids", ng_ids.reshape(T, -1).long()); save(f"L{l}.ng_emb", ng.reshape(T, -1)); save(f"L{l}.ple", o)
            hooks.append(layer.ple.register_forward_hook(ple))
    t0 = time.time()
    with torch.no_grad():
        final = model(input_ids=ids[None].to(DEV), use_cache=False).last_hidden_state[0]
        save("final", final)
        head = ck.get("lm_head.weight").to(DEV)
        logits = (final[torch.as_tensor(pos, device=DEV)] @ head.T).float()
    logits.cpu().numpy().tofile(f"{out}/logits.bin")
    np.asarray(pos, np.int32).tofile(f"{out}/pos.bin")
    ids.int().numpy().tofile(f"{out}/ids.bin")
    bits(model.embed_tokens(ids.to(DEV))).tofile(f"{out}/embed.bin")
    for h in hooks:
        h.remove()
    json.dump({"T": T, "positions": len(pos), "every": every, "seconds": round(time.time() - t0, 1)}, open(f"{out}/meta.json", "w"))
    print(f"dumped T={T} in {time.time() - t0:.0f} s to {out}", flush=True)


def run_profile(model, tc, docs, out):
    L, E = tc.num_hidden_layers, tc.num_experts
    cnt, wsum = np.zeros((L, E), np.int64), np.zeros((L, E), np.float64)
    hooks = []
    for l, layer in enumerate(model.layers):
        def gate(m, i, o, l=l):
            idx, w = o[2].reshape(-1).cpu().numpy(), o[1].float().reshape(-1).cpu().numpy()
            np.add.at(cnt[l], idx, 1); np.add.at(wsum[l], idx, w)
        hooks.append(layer.mlp.gate.register_forward_hook(gate))
    t0, n, B = time.time(), 0, int(os.environ.get("QWEN_PROFILE_BATCH", "16"))
    with torch.no_grad():
        for b in range(0, len(docs), B):
            batch = torch.tensor(docs[b:b + B], dtype=torch.long)   # (equal-length documents, one forward for B of them)
            model(input_ids=batch.to(DEV), use_cache=False)
            n += batch.numel()
            np.savez(out, counts=cnt, weights=wsum, tokens=n)
            print(f"profile: {n} tokens, {time.time() - t0:.0f} s", flush=True)
    for h in hooks:
        h.remove()


def tower(ck, dtype=torch.bfloat16, attn="sdpa"):
    """transformers' vision tower with the checkpoint's weights (0.9 GB in bf16), on the GPU."""
    vc = AutoConfig.from_pretrained(SNAP).vision_config
    vc._attn_implementation = attn
    with torch.device("meta"):
        v = M.Qwen4ExpVisionModel(vc)
    v.load_state_dict({n: ck.get("model.visual." + n).to(DEV, dtype) for n in v.state_dict()}, strict=True, assign=True)
    v.rotary_pos_emb = M.Qwen4ExpVisionRotaryEmbedding(vc).to(DEV)
    return v.eval()


def run_vision(ck, stems):
    """For each DIR/NAME of the engine's `qxvision` (NAME.patches.bf16, NAME.json): the tower's rows from the same patches, in bf16 (NAME.ref.bf16, the reference) and in fp32 (NAME.ref32.f32,
    the noise floor's yardstick), and how far the engine's rows (NAME.rows.bf16) are from each beside how far the bf16 reference is from fp32."""
    with Lock():
        towers = {"bf16": tower(ck), "f32": tower(ck, torch.float32)}
        for stem in stems:
            g = torch.tensor([json.load(open(stem + ".json"))["grid_thw"]], device=DEV)
            px = torch.from_numpy(np.fromfile(stem + ".patches.bf16", np.int16)).view(torch.bfloat16).reshape(-1, 1536).to(DEV)
            with torch.no_grad():
                ref = towers["bf16"](px, grid_thw=g).pooler_output.float()
                ref32 = towers["f32"](px.float(), grid_thw=g).pooler_output.float()
            bits(ref.to(torch.bfloat16)).tofile(stem + ".ref.bf16")
            ref32.cpu().numpy().tofile(stem + ".ref32.f32")
            ours = torch.from_numpy(np.fromfile(stem + ".rows.bf16", np.int16)).view(torch.bfloat16).reshape(ref.shape).to(DEV).float()
            rel = lambda a, b: ((a - b).norm() / b.norm()).item()
            cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
            print(f"{stem}: {ref.shape[0]} rows; to fp32: engine {rel(ours, ref32):.2e} (worst row cosine {cos(ours, ref32):.5f}), transformers bf16 {rel(ref, ref32):.2e} "
                  f"({cos(ref, ref32):.5f}); engine to transformers bf16 {rel(ours, ref):.2e}, largest difference {(ours - ref).abs().max().item():.3f} of {ref.abs().max().item():.1f}", flush=True)
        del towers
        torch.cuda.empty_cache()


def run_answer(ck, cases_path, out):
    """The engine's greedy answers to image prompts, teacher-forced through the bf16 model. CASES: [{"prompt": ids (an image's place its tokens), "answer": the engine's ids, "images":
    [DIR/NAME of qxvision files], "ours": its logits at the answer's positions (f32 file, the JSON-lines "score_from" dump), ...}]. The image rows are transformers' tower's on the engine's
    patches, the positions get_rope_index's, the n-gram ids the prompt's. Per case: the model's own pick at each answer position against the engine's token, and the KL of the engine's
    logits to the model's there; the reference's logits to OUT.NAME.f32."""
    cfg = AutoConfig.from_pretrained(SNAP)
    cases = json.load(open(cases_path))
    feats = []
    with Lock():
        v = tower(ck)
        with torch.no_grad():
            for c in cases:
                rows = []
                for stem in c["images"]:
                    g = torch.tensor([json.load(open(stem + ".json"))["grid_thw"]], device=DEV)
                    px = torch.from_numpy(np.fromfile(stem + ".patches.bf16", np.int16)).view(torch.bfloat16).reshape(-1, 1536).to(DEV)
                    rows.append(v(px, grid_thw=g).pooler_output.cpu())
                feats.append(rows)
        del v
        torch.cuda.empty_cache()
    model, tc = build(ck)

    class Stub:
        config = cfg

        def get_vision_position_ids(self, *a, **k):
            return M.Qwen4ExpModel.get_vision_position_ids(self, *a, **k)

    # every case through each layer while its weights are read (once for all): Qwen4ExpTextModel.forward's steps, the inputs as Qwen4ExpModel.forward makes them for images
    states = []
    with torch.no_grad():
        for c, rows in zip(cases, feats):
            ids = torch.tensor(c["prompt"] + c["answer"], dtype=torch.long, device=DEV)
            grids = torch.tensor([json.load(open(s + ".json"))["grid_thw"] for s in c["images"]], device=DEV)
            emb = model.embed_tokens(ids)
            place = ids == cfg.image_token_id
            emb[place] = torch.cat(rows).to(DEV, emb.dtype)
            pos, _ = M.Qwen4ExpModel.get_rope_index(Stub(), ids[None], place.int()[None], image_grid_thw=grids)
            mk = {"config": tc, "inputs_embeds": emb[None], "attention_mask": None, "past_key_values": None, "position_ids": None, "allow_is_causal_skip": False}
            conv = M.create_recurrent_attention_mask(**mk)
            ple = ids[None] if conv is None else torch.where(conv.bool(), ids[None], tc.eos_token_id)
            states.append({"h": emb[None].repeat(1, 1, tc.hc_count), "pe": model.rotary_emb(emb[None], pos), "mask": M.create_causal_mask(**mk), "conv": conv, "ple": ple})
        for l, layer in enumerate(model.layers):
            with Lock():
                layer.load_state_dict({n: ck.get(f"{P}layers.{l}.{n}").to(DEV) for n in layer.state_dict()}, strict=True, assign=True)
                for s in states:
                    s["h"] = M.Qwen4ExpTextDecoderLayer.forward(layer, s["h"], position_embeddings=s["pe"], attention_mask=s["mask"], conv_mask=s["conv"], past_key_values=None, ple_input_ids=s["ple"])
                torch.cuda.synchronize()
                layer.to_empty(device="meta")
            torch.cuda.empty_cache()
            print(f"layer {l} done", flush=True)
    report = []
    for c, s in zip(cases, states):
        Pn, A = len(c["prompt"]), len(c["answer"])
        with torch.no_grad():
            final = model.hyper_connection_mixer(s["h"])[0, Pn - 1:Pn + A - 1].float()
            with Lock():
                head = ck.get("lm_head.weight").to(DEV).float()
                ref = (final @ head.T).cpu()
                del head
                torch.cuda.empty_cache()
        name = c.get("name", f"case{len(report)}")
        ref.numpy().astype("<f4").tofile(f"{out}.{name}.f32")
        picks = ref.argmax(-1).tolist()
        r = {"name": name, "answer_tokens": A, "same_picks": sum(int(p == t) for p, t in zip(picks, c["answer"])), "first_other": next((i for i, (p, t) in enumerate(zip(picks, c["answer"])) if p != t), None)}
        if c.get("ours"):
            ours = torch.from_numpy(np.fromfile(c["ours"], "<f4")).reshape(A, -1)[:, :ref.shape[1]].double()
            lp, lq = torch.log_softmax(ref.double(), -1), torch.log_softmax(ours, -1)
            r["mean_kld"] = float((lp.exp() * (lp - lq)).sum(-1).mean())
            r["same_top1"] = float((lp.argmax(-1) == lq.argmax(-1)).double().mean())
        print(json.dumps(r), flush=True)
        report.append(r)
    json.dump(report, open(out + ".json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "profile", "vision", "answer"])
    ap.add_argument("--tokens")
    ap.add_argument("--out")
    ap.add_argument("--every", type=int, default=0)
    ap.add_argument("--cases", help="answer: the cases' JSON")
    ap.add_argument("stems", nargs="*", help="vision: DIR/NAME of the engine's qxvision files")
    a = ap.parse_args()
    if a.cmd == "vision":
        return run_vision(Ckpt(SNAP), a.stems)
    if a.cmd == "answer":
        return run_answer(Ckpt(SNAP), a.cases, a.out)
    docs = json.load(open(a.tokens))["docs"]
    ck = Ckpt(SNAP)
    model, tc = build(ck)
    if a.cmd == "dump":
        run_dump(model, tc, ck, docs, a.out, a.every)
    else:
        run_profile(model, tc, docs, a.out)


if __name__ == "__main__":
    main()
