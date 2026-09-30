"""Where two compiled Glyd processes part. Qwen3-1.7B, tiered, one pass over the check's continuation (1,543 tokens,
prompt_logprobs), torch.compile with CUDA graphs off (so every product runs its Python): each of the library's
products hashed in call order, its input X and its output Y (their bits, on the GPU), and the device's done counters
checked zero after the pass; FlashAttention's calls too (their query, key, value and output), in the same order.
    python dbg_hash.py compiled|eager OUT.json LONG.json ['{JSON: more of vLLM's compilation_config}']"""
import json
import sys

import torch
from glyd.gpu import _lib
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

how, out, long_json = sys.argv[1:4]
cc = {"cudagraph_mode": "NONE", **(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})}
calls, on, W = [], [False], {}


def h(t):
    """Two sums of t's bits (int16), one weighted: a fingerprint, computed on the GPU in stream order."""
    v = t.reshape(-1).view(torch.int16).to(torch.int64)
    w = W.get(v.numel())
    if w is None:
        w = W[v.numel()] = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) * 2654435761 + 12345) % 2147483647
    return torch.stack([(v * w).sum(), v.sum()])


for name in ("mma_linear", "mma12_linear"):
    def hooked(data, a, b, words, O, K, x, bias, y, route, f=getattr(_lib, name)):
        f(data, a, b, words, O, K, x, bias, y, route)
        if on[0]:
            calls.append((O, K, x.shape[0], h(x), h(y)))
    setattr(_lib, name, hooked)

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl  # noqa: E402

attn, fa = [], FlashAttentionImpl.forward


def fa_hooked(self, layer, query, key, value, kv_cache, attn_metadata, output, *rest, **kw):
    if on[0]:
        seen = [h(t) for t in (query, key, value) if t is not None]
    r = fa(self, layer, query, key, value, kv_cache, attn_metadata, output, *rest, **kw)
    if on[0]:
        attn.append((len(calls), seen, h(output)))
    return r


FlashAttentionImpl.forward = fa_hooked

ids = json.load(open(long_json))["long_ids"]
import os  # noqa: E402

llm = LLM(model="Qwen/Qwen3-1.7B", quantization=os.environ.get("DBG_Q", "glyd") or None, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, seed=0, enable_prefix_caching=False, enforce_eager=how == "eager", **({} if how == "eager" else {"compilation_config": cc}))
on[0] = True
o = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)[0]
on[0] = False
torch.cuda.synchronize()
json.dump({
    "calls": [[O, K, M, hx.tolist(), hy.tolist()] for O, K, M, hx, hy in calls],
    "attn": [[i, [t.tolist() for t in ins], o.tolist()] for i, ins, o in attn],
    "long_logprobs": [d[t].logprob for d, t in zip(o.prompt_logprobs[1:], ids[1:])],
    "counters_nonzero": {str(k): int(v[0].count_nonzero()) for k, v in _lib._done.items()},
}, open(out, "w"))
print(f"{how}: {len(calls)} products hashed; counters not zero: {sum(int(v[0].count_nonzero()) for v in _lib._done.values())}", flush=True)
