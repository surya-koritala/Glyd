"""vLLM 0.30's residual stream at one position through the layers (eager mode), for the engine's debugging dump (GLYD_ENGINE_DUMP=POSITION,
GLYD_ENGINE_DUMP_FILE=...): after each layer's attention and after each layer's MLP, the residual (hidden + residual of vLLM's pair) as float32.

    python ref_dump.py MODEL MODE TEXTFILE POSITION OUT.npy      MODE: bf16 | glyd1       (the first 1,536 tokens of the text, as ref_vllm.py's score)
"""
import os
import sys

import numpy as np

SCORE = 1536
ROWS = []  # (name, float32 row) in call order


def main(model, mode, textfile, position, out):
    position = int(position)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if mode == "glyd1":
        os.environ.update(GLYD_FRACTION="1", GLYD_KV="off")
    elif mode != "bf16":
        raise SystemExit(f"mode {mode}")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, seed=0, enforce_eager=True, enable_prefix_caching=False)
    if mode != "bf16":
        kw["quantization"] = "glyd"
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    ids = tok.encode(open(textfile).read())[:SCORE]

    def install(m):
        for name, mod in m.named_modules():
            if name.endswith("input_layernorm") or name.endswith("post_attention_layernorm") or name == "model.norm" or name.endswith(".norm") and name.count(".") == 1:
                def hook(mod, args, output, name=name):
                    if isinstance(output, tuple) and output[1].shape[0] == len(ids):
                        ROWS.append((name, output[1][position].float().cpu().numpy().copy()))
                mod.register_forward_hook(hook)
        return True

    llm.apply_model(install)
    llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    # the residual after each layer's attention is post_attention_layernorm's; after its MLP, the next layer's input_layernorm's (and the final norm's)
    names = [n for n, _ in ROWS]
    print("captured", len(ROWS), names[:4], names[-2:])
    np.save(out, np.stack([r for _, r in ROWS]))


if __name__ == "__main__":
    main(*sys.argv[1:6])
