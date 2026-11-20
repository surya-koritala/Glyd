"""Records the transformers fixtures the runtime's CPU reference kinds are checked against (glyd-engine/src/runtime/reference).

Each case builds one transformers module (an attention, a linear attention, ...) from a recent model's real config at reduced widths,
fills every parameter and the input with values both sides generate from the tensor's name (xorshift64 from its FNV-1a hash; see
reference/testing.rs), runs a causal prompt of a few tokens in float32 and writes the output. A fixture holds each tensor's name, shape
and range, never its values, so it stays a few KB.

    python glyd-engine/tests/runtime/record.py [CASE ...]

Needs torch and transformers (the version is written into each fixture). Run with -I from a directory of its own.
"""

import json
import math
import os
import sys

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
T = 7  # tokens of the prompt


def seed(name):
    h = 0xCBF29CE484222325
    for b in name.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h | 1


def fill(name, n, lo, hi):
    r = seed(name)
    out = np.empty(n, dtype=np.float32)
    lo32, span = np.float32(lo), np.float32(hi) - np.float32(lo)
    for i in range(n):
        r ^= (r << 13) & 0xFFFFFFFFFFFFFFFF
        r ^= r >> 7
        r ^= (r << 17) & 0xFFFFFFFFFFFFFFFF
        out[i] = lo32 + span * np.float32((r >> 40) / float(1 << 24))
    return out


assert seed("input") == 0x1EBBAE8F5810B65B


def ranges(name, shape):
    """A tensor's range by what it is: norm weights near 1 (or near 0 for (1+w) norms), products by their fan-in, the rest small."""
    leaf = name.split(".")[-1]
    if leaf in ("A_log",):
        return (-1.0, 0.5)
    if leaf in ("dt_bias",):
        return (-0.5, 0.5)
    if "norm" in name and len(shape) == 1:
        return (0.8, 1.2)
    if len(shape) >= 2:
        a = math.sqrt(3.0 / shape[-1])
        return (-a, a)
    return (-0.5, 0.5)


def fill_module(m, plus_one_norms=False, prefix=""):
    specs = {}
    with torch.no_grad():
        # (the routers' choice biases are buffers)
        params = list(m.named_parameters()) + [(n, b) for n, b in m.named_buffers() if n.endswith("e_score_correction_bias")]
        for name, p in params:
            name = prefix + name
            lo, hi = ranges(name, list(p.shape))
            if plus_one_norms and "norm" in name and p.dim() == 1:
                lo, hi = (-0.2, 0.2)
            v = fill(name, p.numel(), lo, hi)
            p.copy_(torch.from_numpy(v).view(p.shape))
            specs[name] = {"shape": list(p.shape), "lo": lo, "hi": hi}
    return specs


def causal(window=None):
    m = torch.full((T, T), float("-inf"))
    for i in range(T):
        for j in range(i + 1):
            if window is None or i - j < window:
                m[i, j] = 0.0
    return m.view(1, 1, T, T)


def inputs(hidden):
    x = torch.from_numpy(fill("input", T * hidden, -1.0, 1.0)).view(1, T, hidden)
    return x, torch.arange(T).view(1, T)


def write(case, what, source, hidden, specs, out, extra=None):
    j = {
        "what": what,
        "source": f"transformers {transformers.__version__}, torch {torch.__version__}: {source}",
        "tokens": T,
        "hidden": hidden,
        "positions": list(range(T)),
        "input": {"lo": -1.0, "hi": 1.0},
        "tensors": specs,
        "output": [float(v) for v in out.detach().reshape(-1).tolist()],
    }
    j.update(extra or {})
    with open(os.path.join(HERE, f"{case}.json"), "w") as f:
        json.dump(j, f, indent=1)
        f.write("\n")
    print(case, out.shape, f"{float(out.abs().max()):.3f}")


def qwen38_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    # Qwen3.8-27B's layout (Qwen3_5ForConditionalGeneration): gated GQA with head dim 256 and partial rope 64/256 at theta 1e7, (1+w)
    # norms, Gated DeltaNet with 16 key and 48 value heads of 128 and a conv of 4; here at widths 64/32 and 2 key / 4 value heads of 16.
    c = Qwen3_5TextConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_hidden_layers=4, intermediate_size=128,
        vocab_size=256, rms_norm_eps=1e-6, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25},
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    c._attn_implementation = "eager"
    return c


def qwen38_attention():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    c = qwen38_config()
    m = M.Qwen3_5Attention(c, 3).eval()
    specs = fill_module(m, plus_one_norms=True)
    x, pos = inputs(c.hidden_size)
    rot = M.Qwen3_5TextRotaryEmbedding(c)
    cos, sin = rot(x, pos.expand(3, 1, T))  # M-RoPE: a text position is the same on all three axes
    out, _ = m(x, (cos, sin), causal())
    write("qwen38_attention", "Qwen3.8 dense full-attention layer: gated GQA, (1+w) q/k norm, partial rope 0.25 at theta 1e7",
          "Qwen3_5Attention", c.hidden_size, specs, out)


def qwen38_gdn():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    c = qwen38_config()
    m = M.Qwen3_5GatedDeltaNet(c, 0).eval()
    specs = fill_module(m)
    x, _ = inputs(c.hidden_size)
    out = m(x)
    write("qwen38_gdn", "Qwen3.8 dense linear-attention layer: Gated DeltaNet, 2 key and 4 value heads, conv 4",
          "Qwen3_5GatedDeltaNet (torch chunked path, chunk 64)", c.hidden_size, specs, out)


def gemma4_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    # gemma-4-12B's layout: sliding layers of head dim 256 (window 1,024, rope theta 1e4) and global layers of head dim 512 with one K=V head
    # and proportional rope 0.25 at theta 1e6, q/k norm with a weight, v norm without; scaling 1. Here head dims 16 and 32, window 4.
    c = Gemma4TextConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32, num_global_key_value_heads=1,
        attention_k_eq_v=True, num_hidden_layers=2, intermediate_size=128, vocab_size=256, rms_norm_eps=1e-6, sliding_window=4,
        layer_types=["sliding_attention", "full_attention"], num_kv_shared_layers=0, hidden_size_per_layer_input=0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25},
        },
    )
    c._attn_implementation = "eager"
    return c


def gemma4_attention(layer, case, what):
    from transformers.models.gemma4 import modeling_gemma4 as M

    c = gemma4_config()
    m = M.Gemma4TextAttention(c, layer).eval()
    specs = fill_module(m)
    x, pos = inputs(c.hidden_size)
    lt = c.layer_types[layer]
    cos, sin = M.Gemma4TextRotaryEmbedding(c)(x, pos, lt)
    window = m.sliding_window
    out, _ = m(x, (cos, sin), causal(window), {})
    write(case, what, f"Gemma4TextAttention layer {layer} ({lt}, window {window})", c.hidden_size, specs, out, {"window": window})


def glm53flash_kda():
    from transformers.models.glm5_next import modeling_glm5_next as M
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    # GLM-5.3-Flash's KDA layers: 64 heads of 128, conv 4, decay bounded below by -5 (as Kimi-K3's), low-rank gate and decay. Here 4 heads of 16.
    c = Glm5NextTextConfig(hidden_size=64, linear_num_heads=4, linear_head_dim=16, linear_conv_kernel_dim=4, linear_lower_bound=-5.0,
                           num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], rms_norm_eps=1e-5)
    m = M.Glm5NextTextLinearAttention(c, 0).eval()
    specs = fill_module(m)
    x, _ = inputs(c.hidden_size)
    out = m(x)
    write("glm53flash_kda", "GLM-5.3-Flash linear-attention layer: Kimi delta attention, decay bound -5, 4 heads of 16",
          "Glm5NextTextLinearAttention (torch chunked path)", c.hidden_size, specs, out)


def glm53_mla():
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    # GLM-5.3's MLA (GlmMoeDsa): q_lora, kv_lora 512, nope 192 + rope 64 (interleaved), v 256; the DSA indexer keeps 2,048 tokens, so a
    # prompt shorter than that attends to every token. Here q_lora 32, kv_lora 16, nope 8 + rope 8, v 8, 4 heads.
    c = GlmMoeDsaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=8,
                        qk_rope_head_dim=8, v_head_dim=8, num_hidden_layers=1, index_topk=64, index_n_heads=2, index_head_dim=16,
                        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0}, rms_norm_eps=1e-6)
    c._attn_implementation = "eager"
    m = M.GlmMoeDsaAttention(c, 0).eval()
    specs = fill_module(m)
    x, pos = inputs(c.hidden_size)
    cos, sin = M.GlmMoeDsaRotaryEmbedding(c)(x, pos)
    out, _, _ = m(x, (cos, sin), causal(), position_ids=pos)
    write("glm53_mla", "GLM-5.3 latent attention: q_lora 32, kv_lora 16, nope 8 + interleaved rope 8, v 8, 4 heads",
          "GlmMoeDsaAttention (indexer top-k past the prompt)", c.hidden_size, specs, out)


def mlp_case(case, what, source, module, hidden, plus_one_norms=False):
    specs = fill_module(module, plus_one_norms)
    x, _ = inputs(hidden)
    write(case, what, source, hidden, specs, module(x))


def gemma4_mlp():
    from transformers.models.gemma4 import modeling_gemma4 as M

    c = gemma4_config()
    mlp_case("gemma4_mlp", "gemma-4-12B MLP: GELU-tanh gated, 64 -> 128", "Gemma4TextMLP", M.Gemma4TextMLP(c, 0).eval(), c.hidden_size)


def glm53_moe():
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    # GLM-5.3's MoE: sigmoid scores with a choice bias, top-8 of 256 in groups, renormed, x2.5, one shared expert. Here top-2 of 8 in 4
    # groups keeping 2 (so the group limit removes experts), inter 32.
    c = GlmMoeDsaConfig(hidden_size=64, n_routed_experts=8, num_experts_per_tok=2, n_group=4, topk_group=2, routed_scaling_factor=2.5,
                        norm_topk_prob=True, n_shared_experts=1, moe_intermediate_size=32, num_hidden_layers=1, hidden_act="silu")
    m = M.GlmMoeDsaMoE(c).eval()
    specs = fill_module(m)
    x, _ = inputs(c.hidden_size)
    write("glm53_moe", "GLM-5.3 MoE: sigmoid + choice bias, 4 groups keep 2, top-2 of 8, renorm x2.5, one shared expert",
          "GlmMoeDsaMoE", c.hidden_size, specs, m(x))


def dsv4_moe(layer, case):
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    # DeepSeek-V4's MoE: sqrt-softplus scores, a choice bias, hash routing (a token-id table) in the first layers, SwiGLU limit 10,
    # renormed, one shared expert. Here top-2 of 8, inter 32, a vocabulary of 32.
    c = DeepseekV4Config(hidden_size=64, num_local_experts=8, num_experts_per_tok=2, intermediate_size=32, vocab_size=32, num_hidden_layers=2,
                         mlp_layer_types=["hash_moe", "moe"], scoring_func="sqrtsoftplus", swiglu_limit=1.0, routed_scaling_factor=1.5)
    m = M.DeepseekV4SparseMoeBlock(c, layer).eval()
    specs = fill_module(m)
    hash_table = [[(t * 5 + j * 3) % 8 for j in range(2)] for t in range(32)]
    if m.is_hash:
        m.gate.tid2eid.copy_(torch.tensor(hash_table))
    ids = [(t * 11 + 3) % 32 for t in range(T)]
    x, _ = inputs(c.hidden_size)
    out = m(x, torch.tensor(ids).view(1, T))
    write(case, f"DeepSeek-V4 MoE layer ({c.mlp_layer_types[layer]}): sqrt-softplus, top-2 of 8, SwiGLU limit 1, x1.5, one shared expert",
          "DeepseekV4SparseMoeBlock", c.hidden_size, specs, out, {"token_ids": ids, "hash": hash_table if m.is_hash else None})


def qwen38_moe():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    # Qwen3.8-2.4T-A95B's MoE (the Qwen3.5 MoE architecture): softmax top-k, renormed, a shared expert gated by sigmoid of its own
    # router. Here top-3 of 8, inter 32, shared 48.
    c = Qwen3_5MoeTextConfig(hidden_size=64, num_experts=8, num_experts_per_tok=3, moe_intermediate_size=32, shared_expert_intermediate_size=48,
                             num_hidden_layers=1)
    mlp_case("qwen38_moe", "Qwen3.8 MoE: softmax top-3 of 8, renormed, sigmoid-gated shared expert", "Qwen3_5MoeSparseMoeBlock",
             M.Qwen3_5MoeSparseMoeBlock(c).eval(), c.hidden_size)


def rope_scaling():
    """Each rope_type's inverse frequencies and attention factor, from transformers' own ROPE_INIT_FUNCTIONS."""
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    short = [round(1.0 + 0.05 * i, 4) for i in range(36)]
    long_ = [round(1.0 + 0.9 * i, 4) for i in range(36)]
    cases = [
        ("linear", 64, {"rope_type": "linear", "rope_theta": 10_000.0, "factor": 4.0}, None, 2048),
        ("dynamic", 64, {"rope_type": "dynamic", "rope_theta": 10_000.0, "factor": 2.0}, 4096, 2048),
        ("dynamic_short", 64, {"rope_type": "dynamic", "rope_theta": 10_000.0, "factor": 2.0}, 1000, 2048),
        # DeepSeek-V4's compressed layers: YaRN factor 16 over 65,536 positions on the 64 rotated of 512
        ("yarn", 64, {"rope_type": "yarn", "rope_theta": 10_000.0, "factor": 16.0, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1}, None, 1048576),
        ("yarn_mscale", 64, {"rope_type": "yarn", "rope_theta": 10_000.0, "factor": 40.0, "original_max_position_embeddings": 4096, "beta_fast": 32, "beta_slow": 1,
                             "mscale": 1.0, "mscale_all_dim": 0.707}, None, 163840),
        ("llama3", 128, {"rope_type": "llama3", "rope_theta": 500_000.0, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}, None, 131072),
        ("longrope_short", 96, {"rope_type": "longrope", "rope_theta": 10_000.0, "short_factor": short, "long_factor": long_, "original_max_position_embeddings": 4096, "partial_rotary_factor": 0.75}, 2048, 131072),
        ("longrope_long", 96, {"rope_type": "longrope", "rope_theta": 10_000.0, "short_factor": short, "long_factor": long_, "original_max_position_embeddings": 4096, "partial_rotary_factor": 0.75}, 8192, 131072),
    ]
    out = []
    for name, hd, rp, seq_len, max_pos in cases:
        c = LlamaConfig(hidden_size=hd * 4, num_attention_heads=4, head_dim=hd, max_position_embeddings=max_pos, rope_parameters=dict(rp))
        inv, af = ROPE_INIT_FUNCTIONS[rp["rope_type"]](c, None, seq_len=seq_len)
        out.append({"name": name, "head_dim": hd, "rope_parameters": rp, "seq_len": seq_len, "max_position_embeddings": max_pos,
                    "inv_freq": [float(v) for v in inv.tolist()], "attention_factor": float(af)})
        print(name, len(inv), float(af))
    with open(os.path.join(HERE, "rope_scaling.json"), "w") as f:
        json.dump({"what": "rope scaling variants", "source": f"transformers {transformers.__version__}: ROPE_INIT_FUNCTIONS", "cases": out}, f, indent=1)
        f.write("\n")


def glm53_mtp():
    """GLM-5.3's MTP head as tools/glm_ref.py's `mtp_forward` computes it (bf16 between steps), its latent attention and MoE being
    transformers' GLM-5.3 modules (checked on their own by glm53_mla and glm53_moe)."""
    import importlib.util

    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    spec = importlib.util.spec_from_file_location("glm_ref", os.path.join(HERE, "..", "..", "tools", "glm_ref.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    c = GlmMoeDsaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=8,
                        qk_rope_head_dim=8, v_head_dim=8, num_hidden_layers=1, index_topk=64, index_n_heads=2, index_head_dim=16,
                        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0}, rms_norm_eps=1e-5, n_routed_experts=8,
                        num_experts_per_tok=2, n_group=4, topk_group=2, routed_scaling_factor=2.5, norm_topk_prob=True, n_shared_experts=1,
                        moe_intermediate_size=32, hidden_act="silu")
    c._attn_implementation = "eager"
    attn, moe = M.GlmMoeDsaAttention(c, 0).eval(), M.GlmMoeDsaMoE(c).eval()
    specs = fill_module(attn, prefix="attn.")
    specs.update(fill_module(moe, prefix="moe."))
    d, vocab = c.hidden_size, 32
    W = {}
    for name, shape, lo, hi in [("enorm", [d], 0.8, 1.2), ("hnorm", [d], 0.8, 1.2), ("eh_proj", [d, 2 * d], -math.sqrt(1.5 / d), math.sqrt(1.5 / d)),
                                ("norm_a", [d], 0.8, 1.2), ("norm_f", [d], 0.8, 1.2), ("head_norm", [d], 0.8, 1.2), ("embed", [vocab, d], -1.0, 1.0)]:
        W[name] = torch.from_numpy(fill(name, int(np.prod(shape)), lo, hi)).view(*shape)
        specs[name] = {"shape": shape, "lo": lo, "hi": hi}
    pos = torch.arange(T).view(1, T)
    cos, sin = M.GlmMoeDsaRotaryEmbedding(c)(torch.zeros(1, T, d), pos)
    g.mla_mixer = lambda h, W, cfg, stages=None, sel=None: g.bf(attn(g.f32(h).view(1, T, d), (cos, sin), causal(), position_ids=pos)[0].view(T, d))
    g.moe_mixer = lambda h, W, cfg, stages=None: g.bf(moe(g.f32(h).view(1, T, d)).view(T, d))
    toks = [(t * 7 + 2) % vocab for t in range(T)]
    prev = g.bf(torch.from_numpy(fill("input", T * d, -1.0, 1.0)).view(T, d))
    out = g.mtp_forward(prev, torch.tensor(toks), g.bf(W["embed"]), {k: g.bf(v) for k, v in W.items()}, {"eps": 1e-5})
    write("glm53_mtp", "GLM-5.3 MTP head: eh_proj of the normed embedding and hidden state, an MLA + MoE layer, the head norm",
          "tools/glm_ref.py mtp_forward with GlmMoeDsaAttention and GlmMoeDsaMoE inside", d, specs, g.f32(out), {"token_ids": toks})


CASES = {
    "glm53_mtp": glm53_mtp,
    "rope_scaling": rope_scaling,
    "gemma4_mlp": gemma4_mlp,
    "glm53_moe": glm53_moe,
    "dsv4_hash_moe": lambda: dsv4_moe(0, "dsv4_hash_moe"),
    "dsv4_moe": lambda: dsv4_moe(1, "dsv4_moe"),
    "qwen38_moe": qwen38_moe,
    "qwen38_attention": qwen38_attention,
    "qwen38_gdn": qwen38_gdn,
    "gemma4_sliding": lambda: gemma4_attention(0, "gemma4_sliding", "gemma-4-12B sliding layer: GQA, head dim 16, window 4, rope theta 1e4, q/k/v norm"),
    "gemma4_global": lambda: gemma4_attention(1, "gemma4_global", "gemma-4-12B global layer: one K=V head of 32, proportional rope 0.25 at theta 1e6"),
    "glm53flash_kda": glm53flash_kda,
    "glm53_mla": glm53_mla,
}

if __name__ == "__main__":
    torch.manual_seed(0)
    for name in sys.argv[1:] or list(CASES):
        with torch.no_grad():
            CASES[name]()
