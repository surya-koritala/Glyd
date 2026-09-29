# Tiny random bf16 checkpoints of the dense families glyd pack takes and GraniteMoe (transformers 5.17 saves them), for
# Python's pack and Rust's to be compared on: Llama (tied and not), Qwen2 (q, k, v biases), Mistral, Granite, Qwen3 with
# a hidden size no multiple of 128 (its embedding kept in bf16) and with k, v rows no multiple of 64 (not merged), and
# GraniteMoe. rtx4080s-rust-2026-09-28/tiny.py's, the output directory an argument.
#   python tiny.py OUT
import os, sys, torch, transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM

out = sys.argv[1]
base = dict(num_hidden_layers=2, hidden_size=256, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2, vocab_size=512, max_position_embeddings=256)
cfgs = {
    "llama": transformers.LlamaConfig(**base, tie_word_embeddings=False),
    "llama-tied": transformers.LlamaConfig(**base, tie_word_embeddings=True),
    "qwen2": transformers.Qwen2Config(**base, tie_word_embeddings=False),
    "mistral": transformers.MistralConfig(**base, tie_word_embeddings=False),
    "granite": transformers.GraniteConfig(**base, tie_word_embeddings=True),
    "qwen3-odd": transformers.Qwen3Config(**dict(base, hidden_size=192, num_attention_heads=6, num_key_value_heads=1, head_dim=32, intermediate_size=320), tie_word_embeddings=True),
    "qwen3-untied": transformers.Qwen3Config(**dict(base, head_dim=64), tie_word_embeddings=False),
    "granitemoe": transformers.GraniteMoeConfig(**dict(base, intermediate_size=128), num_local_experts=4, num_experts_per_tok=2, tie_word_embeddings=True),
}
for name, cfg in cfgs.items():
    torch.manual_seed(len(name))
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    for p in m.parameters():  # a trained model's spread, not the init's (zeros, ones)
        with torch.no_grad():
            p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.02).to(p.dtype))
    d = os.path.join(out, name)
    m.save_pretrained(d)
    names = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), "pt") as s:
                names += list(s.keys())
    print(name, type(m).__name__, sorted(n for n in names if "layers.0." in n or "layers" not in n))
