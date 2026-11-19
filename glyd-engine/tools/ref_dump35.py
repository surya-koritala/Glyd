"""A Qwen3.5 model's hidden states in transformers (bf16, on the GPU) for the last token of a prompt, layer by layer, against the engine's (`glyd-engine dump35`):
where the two part, if they do. Each decoder layer's output is taken by a hook (the residual stream after the layer), and the final norm's.

    python ref_dump35.py MODEL_DIR IDS.json ENGINE.json [OUT.json]     IDS.json: {"ids": [...]}; ENGINE.json: dump35's output

Printed per layer: the relative difference |engine - transformers| / |transformers| over the hidden values, and the largest absolute difference; then both sides' next token.
"""
import json
import sys

import torch


def main(model_dir, ids_file, engine_file, out_file=None):
    ids = json.load(open(ids_file))["ids"]
    eng = json.load(open(engine_file))
    import transformers

    m = None
    for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            m = getattr(transformers, cls).from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")
            break
        except Exception as e:  # (the model with its vision tower, else the text model alone)
            print(f"{cls}: {type(e).__name__}: {str(e)[:200]}")
    if m is None:
        raise SystemExit("transformers did not load the model")
    lm = None
    for mod in m.modules():
        if hasattr(mod, "layers") and hasattr(mod, "norm") and isinstance(getattr(mod, "layers"), torch.nn.ModuleList):
            lm = mod
            break
    got = []

    def keep(_mod, _inp, out):
        o = out[0] if isinstance(out, tuple) else out
        got.append(o[0, -1].float().cpu())

    hooks = [ly.register_forward_hook(keep) for ly in lm.layers[:-1]] + [lm.norm.register_forward_hook(keep)]
    with torch.no_grad():
        logits = m(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()
    for h in hooks:
        h.remove()
    lp = torch.log_softmax(logits, -1)
    ref_tok = int(torch.argmax(logits))
    rows = []
    for i, (e, r) in enumerate(zip(eng["layers"], got)):
        e = torch.tensor(e)
        rel = float((e - r).norm() / r.norm().clamp_min(1e-30))
        rows.append({"layer": i, "rel": rel, "max_abs": float((e - r).abs().max()), "ref_norm": float(r.norm())})
        what = f"after layer {i}" if i < len(got) - 1 else "the final norm"
        print(f"{what:16s} relative difference {rel:.2e}, largest {rows[-1]['max_abs']:.3e} (|ref| {rows[-1]['ref_norm']:.1f})")
    print(f"{len(ids)} prompt tokens; next token: engine {eng['token']} (logprob {eng['logprob']:.4f}), transformers {ref_tok} (logprob {float(lp[ref_tok]):.4f}); "
          f"transformers' logprob of the engine's token {float(lp[eng['token']]):.4f}")
    if out_file:
        json.dump({"layers": rows, "engine_token": eng["token"], "ref_token": ref_tok, "transformers": transformers.__version__}, open(out_file, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
