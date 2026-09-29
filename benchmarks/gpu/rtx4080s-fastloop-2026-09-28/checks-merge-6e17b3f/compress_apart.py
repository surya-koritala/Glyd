# The 1.7B compress check apart: from_pretrained's model against compress's, their eager logits, their compiled
# tokens and their eager tokens (disable_compile), in a process with its own inductor cache (argv[1]).
import os, sys, torch, glyd, glyd.gpu
from transformers import AutoModelForCausalLM, AutoTokenizer

name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(name)
ids = tok("The history of data compression began", return_tensors="pt").input_ids.cuda()


def run(model):
    with torch.no_grad():
        logits = model(ids, logits_to_keep=1).logits
        out = model.generate(ids, max_new_tokens=32, min_new_tokens=32, do_sample=False)
        eager = model.generate(ids, max_new_tokens=32, min_new_tokens=32, do_sample=False, disable_compile=True)
    return logits, out[0, ids.shape[1]:], eager[0, ids.shape[1]:]


same = lambda a, b: (a == b).long().cumprod(0).sum().item()
m = glyd.from_pretrained(name)
lb, ob, eb = run(m)
c = glyd.gpu.compress(AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16))
lc, oc, ec = run(c)
lb2, ob2, eb2 = run(m)
print(f"{os.environ.get('TORCHINDUCTOR_CACHE_DIR')}: logits exact {torch.equal(lb.view(torch.int16), lc.view(torch.int16))}; compiled tokens m/c {same(ob, oc)} of 32, m again {same(ob, ob2)}; eager tokens m/c {same(eb, ec)} of 32; compiled/eager {same(ob, eb)}", flush=True)
