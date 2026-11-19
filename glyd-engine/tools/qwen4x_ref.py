"""Qwen3.8-Flash-Next (qwen4_exp) references for the engine's checks, from transformers 5.19.0's own modules (src/kcheck_qwen4x.rs compares).

  qwen4x_ref.py tiny OUT [--seed S] [--tokens T]   a random model in the real layer pattern (8 layers: 3 GDN + QSA, twice; PLE at layer 1; 16 experts; an indexer of 2 blocks so that 12 tokens or more prune), its
                                                   tensors in the checkpoint's names and shards (OUT/model.safetensors, OUT/config.json), the experts and shared experts on the six-bit grid (the engine's packs
                                                   hold them exactly), and OUT/ref: the stages of every layer at every position of T random tokens (the names src/kcheck_qwen4x.rs reads; the real model's references are
                                                   ~/glyd-scratch/qwen/qwen_ref.py's, in the same names) and the logits of every position.
  qwen4x_ref.py selftest                           the tiny model's own consistency: the stages' layer loop gives the model's logits.

Stage files (raw little-endian, rows = positions): ids.bin i32 [T]; embed.bin bf16 [T, H]; L{l}.{a_in, a_out, m_in, m_out} bf16 [T, H], {mid, out} bf16 [T, 4H], {a_inj, m_inj} f32 [T, 4], r_ids i32 [T, k],
r_w f32 [T, k]; at the PLE layer ng_ids i64 [T, 16], ng_emb bf16 [T, H], ple bf16 [T, 4H]; final.bin bf16 [T, H]; logits.bin f32 [T, V]; pos.bin i32 [T].
"""
import json, math, os, sys

import torch

BF = torch.bfloat16


def tiny_text_config(vocab=2048, eos=2000):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    kinds = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]
    return Qwen4ExpTextConfig(
        vocab_size=vocab, hidden_size=256, num_hidden_layers=8, layer_types=kinds, num_attention_heads=24, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        hc_count=4, hc_lowrank=64, num_experts=16, num_experts_per_tok=4, moe_intermediate_size=128, shared_expert_intermediate_size=128,
        ple_layer_ids=[2], ple_embed_dim=256, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000, split_ngram_parts=128,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=8, indexer_compress_ratio=4, output_gate_type="sigmoid",
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        rms_norm_eps=1e-6, max_position_embeddings=4096, eos_token_id=eos, bos_token_id=eos, tie_word_embeddings=False, dtype="bfloat16")


def grid(g, rows, cols, std, kmax=31):
    """rows x cols weights of about that spread that the six-bit block layout holds exactly: integers of at most kmax times a power of two, a code of kmax in every run of 32 along a row (the packer's scale is then
    that power of two)."""
    scale = 2.0 ** math.ceil(math.log2(3 * std / kmax))
    k = torch.round(torch.randn(rows, cols, generator=g) * std / scale).clamp(-kmax, kmax)
    k[:, ::32] = kmax * (torch.randint(0, 2, (rows, (cols + 31) // 32), generator=g) * 2 - 1)
    return (k * scale).to(BF)


def tiny_weights(model, cfg, seed):
    """Random weights for every tensor of the tiny model (in place), each of a spread that keeps a block's output about the size of the streams."""
    g = torch.Generator().manual_seed(seed)
    H, HW = cfg.hidden_size, cfg.hidden_size * cfg.hc_count
    with torch.no_grad():
        for name, p in model.named_parameters():
            shape = tuple(p.shape)
            if name.endswith("norm.weight") or name.endswith("layernorm.weight") or "hc_norm" in name or "norm_key" in name or "norm_query" in name or "norm_conv" in name:
                v = torch.rand(shape, generator=g) - 0.5  # (the (1 + w) norms; the GDN's gated norm is plain w: centred on 1 below)
                if ".linear_attn.norm." in name:
                    v = v + 1.0
            elif name.endswith("A_log"):
                v = torch.log(torch.rand(shape, generator=g) * 15 + 1)
            elif name.endswith("dt_bias"):
                v = torch.randn(shape, generator=g)
            elif "conv1d" in name:
                v = torch.randn(shape, generator=g) * 0.4
            elif "experts.gate_up_proj" in name:
                v = torch.stack([grid(g, shape[1], shape[2], 1 / math.sqrt(shape[2])) for _ in range(shape[0])]).float()
            elif "experts.down_proj" in name:
                v = torch.stack([grid(g, shape[1], shape[2], 1 / math.sqrt(shape[2])) for _ in range(shape[0])]).float()
            elif "shared_expert." in name:
                v = grid(g, shape[0], shape[1], 1 / math.sqrt(shape[1])).float()
            elif "ngram_embedding" in name or "embed_tokens" in name:
                v = torch.randn(shape, generator=g)
            elif len(shape) == 2:
                v = torch.randn(shape, generator=g) / math.sqrt(shape[1])
            else:
                raise SystemExit(f"no rule for {name} {shape}")
            p.copy_(v.to(p.dtype))


def put(path, t):
    t = t.detach().contiguous().cpu()
    if t.dtype == BF:
        t.view(torch.int16).numpy().tofile(path)
    else:
        t.numpy().tofile(path)


def write_safetensors(path, tensors):
    """{name: tensor} in the safetensors format (bf16, f32, i64), header in sorted order."""
    names = sorted(tensors)
    header, off, blobs = {}, 0, []
    for n in names:
        t = tensors[n].detach().contiguous().cpu()
        dt = {BF: "BF16", torch.float32: "F32", torch.int64: "I64"}[t.dtype]
        b = (t.view(torch.int16) if t.dtype == BF else t).numpy().tobytes()
        header[n] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for b in blobs:
            f.write(b)


def checkpoint_tensors(model, cfg):
    """The model's tensors in the real checkpoint's names: model.language_model.*, lm_head.weight, the n-gram table in `split_ngram_parts` shards."""
    out = {}
    for name, t in list(model.state_dict().items()):
        n = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        if n.endswith("ngram_embedding.weight"):
            parts = cfg.split_ngram_parts
            rows = t.shape[0] // parts
            assert rows * parts == t.shape[0]
            for k in range(parts):
                out[n.replace("ngram_embedding.weight", f"ngram_embedding.shard_{k}.weight")] = t[k * rows:(k + 1) * rows]
            continue
        out[n] = t
    return out


def build_tiny(seed=1):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    cfg = tiny_text_config()
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(cfg).to(BF).eval()
    tiny_weights(model, cfg, seed)
    return model, cfg


def stages(model, cfg, ids, out_dir):
    """The model on ids [T] with every stage captured by hooks on its own modules, written to out_dir; returns the logits [T, V] (f32)."""
    m = model.model
    os.makedirs(out_dir, exist_ok=True)
    T = ids.shape[0]
    put(f"{out_dir}/ids.bin", ids.to(torch.int32))
    put(f"{out_dir}/pos.bin", torch.arange(T, dtype=torch.int32))
    hooks, k = [], cfg.num_experts_per_tok
    hooks.append(m.embed_tokens.register_forward_hook(lambda mod, a, o: put(f"{out_dir}/embed.bin", o[0])))
    for l, layer in enumerate(m.layers):
        def h(stage, idx=None, f32=False, l=l):
            def fn(mod, a, o):
                v = o if idx is None else o[idx]
                v = v[0]
                put(f"{out_dir}/L{l}.{stage}.bin", v.float() if f32 else v)
            return fn
        hooks.append(layer.attn_hyper_connection.register_forward_hook(h("a_in", 0)))
        hooks.append(layer.attn_hyper_connection.register_forward_hook(h("a_inj", 2, True)))
        mixer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        hooks.append(mixer.register_forward_hook(h("a_out", None if layer.layer_type == "linear_attention" else 0)))
        if layer.layer_type != "linear_attention":
            def picks(mod, a, o, l=l):
                # the tokens the indexer picks for each query (its mask before the causal one), ascending, padded with -1
                mask = o[0, 0] if o.dtype == torch.bool else o[0, 0] == 0
                tmax = cfg.indexer_budget + cfg.indexer_compress_ratio - 1
                sel = torch.full((mask.shape[0], tmax), -1, dtype=torch.int32)
                for q in range(mask.shape[0]):
                    nz = torch.nonzero(mask[q]).flatten()[:tmax]
                    sel[q, :nz.numel()] = nz.to(torch.int32)
                put(f"{out_dir}/L{l}.sel.bin", sel)
            hooks.append(layer.self_attn.indexer.register_forward_hook(picks))
        hooks.append(layer.mlp_hyper_connection.register_forward_pre_hook(lambda mod, a, l=l: put(f"{out_dir}/L{l}.mid.bin", a[0][0])))
        hooks.append(layer.mlp_hyper_connection.register_forward_hook(h("m_in", 0)))
        hooks.append(layer.mlp_hyper_connection.register_forward_hook(h("m_inj", 2, True)))
        def route(mod, a, o, l=l):
            put(f"{out_dir}/L{l}.r_w.bin", o[1].float())
            put(f"{out_dir}/L{l}.r_ids.bin", o[2].to(torch.int32))
        hooks.append(layer.mlp.gate.register_forward_hook(route))
        hooks.append(layer.mlp.register_forward_hook(h("m_out")))
        hooks.append(layer.register_forward_hook(h("out")))
        if layer.ple is not None:
            hooks.append(layer.ple.register_forward_hook(h("ple")))
            hooks.append(layer.ple.ple_embedding.register_forward_hook(h("ng_emb")))
            hooks.append(layer.ple.ple_embedding.ngram_embedding.register_forward_pre_hook(lambda mod, a, l=l: put(f"{out_dir}/L{l}.ng_ids.bin", a[0][0].to(torch.int64))))
    hooks.append(m.hyper_connection_mixer.register_forward_hook(lambda mod, a, o: put(f"{out_dir}/final.bin", o[0])))
    with torch.no_grad():
        logits = model(input_ids=ids[None], use_cache=False).logits[0].float()
    for hk in hooks:
        hk.remove()
    put(f"{out_dir}/logits.bin", logits)
    json.dump({"T": T, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "vocab": cfg.vocab_size, "top_k": k}, open(f"{out_dir}/meta.json", "w"))
    return logits


def tiny(out, seed=1, tokens=200):
    model, cfg = build_tiny(seed)
    os.makedirs(out, exist_ok=True)
    tc = cfg.to_dict()
    for key in ("_attn_implementation", "_attn_implementation_internal", "transformers_version", "architectures", "dtype", "torch_dtype"):
        tc.pop(key, None)
    tc["dtype"] = "bfloat16"
    json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "text_config": tc, "tie_word_embeddings": False}, open(f"{out}/config.json", "w"), indent=1)
    write_safetensors(f"{out}/model.safetensors", checkpoint_tensors(model, cfg))
    g = torch.Generator().manual_seed(seed + 7)
    ids = torch.randint(0, cfg.vocab_size, (tokens,), generator=g)
    ids[tokens // 3] = cfg.eos_token_id  # (an end token inside: the n-gram contexts restart after it)
    logits = stages(model, cfg, ids, f"{out}/ref")
    print(f"{out}: a random qwen4_exp of {cfg.num_hidden_layers} layers, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters; ref: {tokens} positions, logits top-1 {logits.argmax(-1)[:8].tolist()} ...")


def selftest():
    model, cfg = build_tiny(3)
    ids = torch.randint(0, cfg.vocab_size, (40,), generator=torch.Generator().manual_seed(5))
    d = "/tmp/qwen4x_selftest"
    a = stages(model, cfg, ids, d)
    with torch.no_grad():
        b = model(input_ids=ids[None], use_cache=False).logits[0].float()
        # prefill then decode with the cache: the same logits (the engine is held to the full pass)
        out = model(input_ids=ids[None, :25], use_cache=True)
        steps = [out.logits[0, -1].float()]
        pkv = out.past_key_values
        for t in range(25, 39):
            o = model(input_ids=ids[None, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            steps.append(o.logits[0, -1].float())
    assert torch.equal(a, b), "the hooked pass is the model's"
    dec = torch.stack(steps)
    print("prefill + decode vs the full pass: largest logit difference", (dec - b[24:39]).abs().max().item(), "top-1 agree", (dec.argmax(-1) == b[24:39].argmax(-1)).float().mean().item())


if __name__ == "__main__":
    args = sys.argv[1:]
    flag = lambda name, d: type(d)(args[args.index(name) + 1]) if name in args else d
    if not args:
        print(__doc__)
    elif args[0] == "tiny":
        tiny(args[1], flag("--seed", 1), flag("--tokens", 200))
    elif args[0] == "selftest":
        selftest()
    else:
        print(__doc__)
