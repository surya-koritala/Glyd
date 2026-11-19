"""plan() on a tiny Llama on the meta device: an output layer tied to the embedding is left out, other tied weights are refused:
PYTHONPATH=glyd-gpu-py python3 huggingface/test_plan.py"""
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from plan import plan


def tiny(tie):
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=tie), dtype=torch.bfloat16)


def put(m, lossy=False):
    puts = plan(m, {k: ("BF16", list(v.shape)) for k, v in m.state_dict().items() if k != "lm_head.weight" or not m.config.tie_word_embeddings}, lossy)[0]
    return [x for p in puts for x in (p.get("linear") or [p["copy"]])]


m = tiny(True)
assert set(m.all_tied_weights_keys) == {"lm_head.weight"}
for lossy in (False, True):
    names = put(m, lossy)
    assert not any("lm_head" in n for n in names) and ("model.embed_tokens" in names) == lossy and ("model.embed_tokens.weight" in names) != lossy, names
assert "lm_head" in put(tiny(False))
m.all_tied_weights_keys["model.norm.weight"] = "model.embed_tokens.weight"
try:
    plan(m, {}, False)
    raise AssertionError("other tied weights planned")
except SystemExit as e:
    assert "weights tied to others" in str(e)
print("ok")
