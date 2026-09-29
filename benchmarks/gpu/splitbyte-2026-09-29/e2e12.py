"""A model loaded in the 12-bit layout as a user loads it (glyd.from_pretrained(MODEL, layout="mma12"): fused, then
exact=True), run eager: its logits for prompts of 1, 17, 64, 300 and 2100 tokens (the library's routes for them on
this GPU: step, mid and prompt kernels, decoded ahead) and 32 greedy tokens, printed as the sha256 of the logits' bits
and the tokens, to compare main's package and library (the 12-bit layout before split byte) with this tree's (split
byte): the same kernels on the same bits, so the same lines.

    GLYD_COMPILE=0 PYTHONPATH=TREE/bindings/python GLYD_GPU_LIB=LIB python e2e12.py MODEL [MODEL ...]"""
import hashlib, sys
import torch
import glyd
from transformers import AutoTokenizer

TEXT = "The history of data compression began with Morse code, which gave the commonest letters the shortest codes. "
for name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(name)
    base = tok(TEXT * 200, return_tensors="pt").input_ids[0]
    assert base.numel() >= 2100
    for exact in (False, True):
        m = glyd.from_pretrained(name, layout="mma12", exact=exact)
        assert m.config.quantization_config.layout == "mma12"
        with torch.no_grad():
            for n in (1, 17, 64, 300, 2100):
                logits = m(base[:n].cuda()[None]).logits
                print(f"{name} exact={exact} {n:4} tokens: logits {hashlib.sha256(logits.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]}", flush=True)
            out = m.generate(base[:16].cuda()[None], max_new_tokens=32, min_new_tokens=32, do_sample=False)
            print(f"{name} exact={exact} 32 tokens: {out[0, 16:].tolist()}", flush=True)
        del m
        torch.cuda.empty_cache()
