"""Reference logits of a Hub model from transformers, as a glyd-logits/1 fixture (glyd-engine/src/logits.rs reads it; tests/logits/CASE.json.gz).

    python logits_ref_hf.py CASE MODEL_ID_OR_DIR OUT.json.gz [--bound penguin=kl:MEAN:MAX:TOP1 ...] [--device cuda|cpu] [--date YYYY-MM-DD] [--note TEXT] [--vocab N --label TEXT]

The tokens are a fixed text's first 190 (the prompt: 24, then 12 decode steps fed the text's next tokens, then a long prompt of 150 that crosses the engine's chunk of 64), the model in bf16 as
the release has it, one forward over each (a step's logits are the full pass's logits at that position: the same tokens, teacher forced). Each row is the last position's 64 largest logits,
the log-sum-exp of the vocabulary and the hash of the logits' bits, as src/logits.rs makes them (largest first, a tie by the smaller token; FNV-1a over the f32 bits).
Take the GPU lock (~/glyd-scratch/gpu.lock) for --device cuda.
"""
import gzip
import json
import sys

import numpy as np

TEXT = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, a lantern swinging from one hand and a notebook tucked under the other arm. "
    "From the gallery he could see the whole harbour: the fishing boats nosing back to their moorings, the ferry drawing a long white line across the grey water, "
    "and beyond the breakwater the open sea, which that night was calm enough to hold the first stars like coins in a dish. He wrote down the wind, the swell and the "
    "visibility in a small neat hand, then trimmed the wick, wound the clockwork and listened for the soft click that told him the great lens had begun to turn. "
    "Somewhere below a door closed, a kettle began to sing, and the beam swept out over the dark water, steady as a promise, reaching for ships that would never know his name."
)
PROMPT, DECODE, LONG = 24, 12, 150
TOPK = 64


def fnv(row):
    h = 0xCBF29CE484222325
    for b in row.astype("<f4").tobytes():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def make_row(logits):
    l = logits.astype(np.float32)
    assert np.isfinite(l).all(), "a logit that is not a number"
    order = np.lexsort((np.arange(len(l)), -l))[:TOPK]  # largest first, a tie by the smaller token
    m = float(l.max())
    lse = m + float(np.log(np.exp(l.astype(np.float64) - m).sum()))
    return {"ids": [int(i) for i in order], "logits": [float(l[i]) for i in order], "lse": lse, "hash": "%016x" % fnv(l)}


def main(argv):
    pos = [a for a in argv if not a.startswith("--")]
    case, model_id, out = pos[0], pos[1], pos[2]
    opt = {}
    bounds = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--bound", "--device", "--date", "--note", "--family", "--vocab", "--label"):
            if argv[i] == "--bound":
                bounds.append(argv[i + 1])
            else:
                opt[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    device = opt.get("device", "cuda")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "vocab" in opt:
        # a random model of src/logits.rs (no tokenizer): its fixed tokens, as token_list makes them (a first token 2, then 37 i + 11 modulo the vocabulary)
        v = int(opt["vocab"])
        ids = [2 if i == 0 else ((i - 1) * 37 + 11) % v for i in range(LONG)]
    else:
        tok = AutoTokenizer.from_pretrained(model_id)
        ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    assert len(ids) >= LONG + 8 or "vocab" in opt, len(ids)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    rev = getattr(model.config, "_commit_hash", None) or ""

    def last_logits(n, upto=None):
        x = torch.tensor([ids[:n]], device=device)
        with torch.no_grad():
            lg = model(input_ids=x).logits[0].float().cpu().numpy()
        return lg

    steps = last_logits(PROMPT + DECODE)
    rows = [make_row(steps[PROMPT - 1 + k]) for k in range(DECODE + 1)]
    rows.append(make_row(last_logits(LONG)[-1]))
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    bl = []
    for b in bounds or ["penguin=kl:1:1:0"]:
        level, spec = b.split("=", 1)
        p = spec.split(":")
        bl.append({"level": level, "kind": "bits"} if p[0] == "bits" else {"level": level, "kind": "kl", "mean": float(p[1]), "max": float(p[2]), "top1": float(p[3])})
    fx = {
        "format": "glyd-logits/1",
        "case": case,
        "family": opt.get("family", case),
        "model": opt.get("label", model_id),
        "provenance": {
            "source": f"transformers {transformers.__version__}, torch {torch.__version__}, {model.dtype} on {dev_name}, the model's own implementation (not the engine's)",
            "revision": rev,
            "recorded": opt.get("date", ""),
            "note": opt.get("note", ""),
            "tokens": "the fixed list of src/logits.rs token_list: prompt 24, 12 decode steps teacher-forced, a long prompt of 150" if "vocab" in opt else "the first tokens of the text in tools/logits_ref_hf.py: prompt 24, 12 decode steps teacher-forced, a long prompt of 150",
        },
        "bounds": bl,
        "prompt": ids[:PROMPT],
        "decode": ids[PROMPT : PROMPT + DECODE],
        "long": ids[:LONG],
        "rows": rows,
    }
    with gzip.open(out, "wt", compresslevel=9) as f:
        json.dump(fx, f, separators=(",", ":"))
    print(out, len(rows), "rows,", model_id, dev_name)


if __name__ == "__main__":
    main(sys.argv[1:])
