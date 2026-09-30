"""A one-minute smoke of `fraction` on a small model, eager: which decoder layers are packed and which are vLLM's own method,
the options written into additional_config, and the tokens and logprobs against another run (python smoke.py MODEL FRACTION
OUT.json [bf16])."""
import json, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

model, frac, out = sys.argv[1], sys.argv[2], sys.argv[3]
bf16 = "bf16" in sys.argv[4:]
kw = {} if bf16 else dict(quantization="glyd", additional_config={"glyd": {"fraction": float(frac)}})
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.3, max_model_len=1024, enforce_eager=True, seed=0, **kw)
cfg = llm.llm_engine.vllm_config


def layers(m):
    from glyd.gpu.vllm_plugin import _layer_of
    packed, plain = set(), set()
    for name, mod in m.named_modules():
        i = _layer_of(name)
        qm = getattr(mod, "quant_method", None)
        if i is not None and qm is not None:
            (packed if getattr(mod, "glyd_words", None) is not None or getattr(mod, "glyd_moe", None) is not None else plain).add(i)
    kinds = sorted({type(getattr(mod, "quant_method", None)).__name__ for _, mod in m.named_modules() if getattr(mod, "quant_method", None) is not None})
    return {"packed": sorted(packed), "plain": sorted(plain), "kinds": kinds}


res = {"fraction": frac, "bf16": bf16, "additional_config": cfg.additional_config, "blocks": cfg.cache_config.num_gpu_blocks}
if not bf16:
    res["layers"] = llm.apply_model(layers)[0]
prompts = ["The history of data compression began", "def fibonacci(n):", "The capital of Australia is"]
o = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=24, logprobs=0), use_tqdm=False)
res["tokens"] = [list(x.outputs[0].token_ids) for x in o]
res["logprobs"] = [[d[t].logprob for d, t in zip(x.outputs[0].logprobs, x.outputs[0].token_ids)] for x in o]
json.dump(res, open(out, "w"))
print("SMOKE", json.dumps({k: v for k, v in res.items() if k not in ("tokens", "logprobs")}))
