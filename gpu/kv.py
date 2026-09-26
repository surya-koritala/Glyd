"""A Hugging Face KV cache held compressed in GPU memory, bit for bit.

Each layer keeps its newest keys and values as they are (bf16) and its
older ones in the mma layout's tiered code (glyd_gpu.cu), a page of 64
tokens at a time: every full page of the tail is packed as it fills, all
of the batch's sequences and heads together. Keys go in as [pages x batch
x heads x 64 tokens, head_dim], values transposed, [pages x batch x heads
x head_dim, 64 tokens] (the operands of the two products of attention).
A layer's exponent tiers are taken from its first page and kept, so its
pages append to one pack; an exponent outside them goes in as itself.
Attention gets the keys and values back exactly: the pages packed before
the call decoded, then the tail and the call's own tokens as they came.
With fused=True (and use_fused_attention(model)), a step of one new token
a sequence instead hands attention the layer itself, and attn_decode
(glyd_gpu.cu) reads the packed pages in place: keys and values decoded in
registers, the product on the tensor cores.

    cache = GlydKVCache(model.config)
    model.generate(ids, past_key_values=cache, ...)
    cache.nbytes(), cache.bf16_bytes()
"""
import torch
from transformers.cache_utils import Cache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import glyd_gpu as g

PAGE = 64  # tokens a page
CHUNK = 1 << 19  # values packed at a time (the packer's scratch: some 50 MB)


class Paged:
    """What a fused layer hands attention for its keys and for its values."""

    def __init__(self, layer):
        self.layer = layer


class GlydKVLayer(DynamicLayer):
    def __init__(self, fused=False):
        super().__init__()
        self.k = self.v = None  # the packed pages (glyd_gpu.Mma)
        self.pages = 0
        self.cumulative_length = 0
        self.fused = fused

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]
        tk = torch.cat([self.keys, key_states], dim=-2) if self.keys.numel() else key_states
        tv = torch.cat([self.values, value_states], dim=-2) if self.values.numel() else value_states
        B, H, T, D = tk.shape
        P, n = self.pages, T // PAGE
        if n:
            # The tail's full pages packed: keys by token, values transposed.
            k = tk[:, :, : n * PAGE].reshape(B, H, n, PAGE, D).permute(2, 0, 1, 3, 4).reshape(-1, D).contiguous()
            v = tv[:, :, : n * PAGE].reshape(B, H, n, PAGE, D).permute(2, 0, 1, 4, 3).reshape(-1, PAGE).contiguous()
            kp = g.pack_mma(k, self.k.tiers if self.k is not None else None, CHUNK)
            vp = g.pack_mma(v, self.v.tiers if self.v is not None else None, CHUNK)
            del k, v
            self.k = kp if self.k is None else g.mma_cat(self.k, kp)
            self.v = vp if self.v is None else g.mma_cat(self.v, vp)
            self.pages += n
        self.keys = tk[:, :, n * PAGE :].clone()  # a copy: a view would hold the call's whole tensor
        self.values = tv[:, :, n * PAGE :].clone()
        if self.fused and key_states.shape[-2] == 1 and self.pages:
            return Paged(self), Paged(self)
        if not P:
            return tk, tv
        # The pages packed before this call, decoded; the rest as it came.
        k = g.mma_unpack(self.k, rows=P * B * H * PAGE).view(P, B, H, PAGE, D).permute(1, 2, 0, 3, 4).reshape(B, H, P * PAGE, D)
        v = g.mma_unpack(self.v, rows=P * B * H * D).view(P, B, H, D, PAGE).permute(1, 2, 0, 4, 3).reshape(B, H, P * PAGE, D)
        return torch.cat([k, tk], dim=-2), torch.cat([v, tv], dim=-2)

    def materialize(self):
        """The layer's keys and values, decoded."""
        B, H, _, D = self.keys.shape
        P = self.pages
        k = g.mma_unpack(self.k).view(P, B, H, PAGE, D).permute(1, 2, 0, 3, 4).reshape(B, H, P * PAGE, D)
        v = g.mma_unpack(self.v).view(P, B, H, D, PAGE).permute(1, 2, 0, 4, 3).reshape(B, H, P * PAGE, D)
        return torch.cat([k, self.keys], dim=-2), torch.cat([v, self.values], dim=-2)

    def get_seq_length(self):
        return self.cumulative_length

    def nbytes(self):
        tail = sum(t.numel() * t.element_size() for t in (self.keys, self.values) if t is not None)
        return tail + sum(p.nbytes() for p in (self.k, self.v) if p is not None)


def attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    """SDPA's, but for a fused layer's step of one new token a sequence:
    attn_decode over its packed pages and tail."""
    if isinstance(key, Paged):
        lay = key.layer
        B, Hq, T, D = query.shape
        if attention_mask is None and T == 1 and D in (64, 128):
            H = lay.keys.shape[1]
            out = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)
            k, v = lay.k, lay.v
            g._ext.attn_decode(query.reshape(B, Hq, D).contiguous(), k.data, k.blocks, k.block_base, k.tiers, v.data, v.blocks, v.block_base, v.tiers, lay.keys, lay.values, lay.keys.shape[2], B * H, Hq // H, lay.pages, scaling if scaling is not None else D**-0.5, out)
            return out.view(B, 1, Hq, D), None
        key, value = lay.materialize()
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)


def use_fused_attention(model):
    """The model's attention through attention() (SDPA's masks)."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register("glyd", attention)
    AttentionMaskInterface.register("glyd", sdpa_mask)
    model.set_attn_implementation("glyd")


class GlydKVCache(Cache):
    def __init__(self, config, fused=False):
        n = config.get_text_config(decoder=True).num_hidden_layers
        super().__init__(layers=[GlydKVLayer(fused) for _ in range(n)])

    def nbytes(self):
        """Bytes the cache holds: packed pages and bf16 tails."""
        return sum(l.nbytes() for l in self.layers)

    def bf16_bytes(self):
        """Bytes the same cache takes in bf16."""
        return sum(2 * l.cumulative_length * l.keys.shape[0] * l.keys.shape[1] * l.keys.shape[-1] * 2 for l in self.layers if l.is_initialized)


if __name__ == "__main__":
    # Every page decodes to its keys and values bit for bit, through appends and a tail.
    torch.manual_seed(0)
    lay, ks, vs = GlydKVLayer(), [], []
    for t in [100, 1, 1, 30, 70, 1, 64, 5]:
        k = (torch.randn(2, 4, t, 128, device="cuda") * torch.randn(1, 1, 1, 128, device="cuda").exp()).to(torch.bfloat16)
        v = torch.randn(2, 4, t, 128, device="cuda").to(torch.bfloat16)
        ks.append(k)
        vs.append(v)
        K, V = lay.update(k, v)
        assert torch.equal(K.view(torch.int16), torch.cat(ks, -2).view(torch.int16)) and torch.equal(V.view(torch.int16), torch.cat(vs, -2).view(torch.int16))
    print(f"{lay.get_seq_length()} tokens in {lay.pages} pages and a tail of {lay.keys.shape[-2]}: {lay.nbytes() / (2 * 2 * lay.get_seq_length() * 4 * 128 * 2) * 100:.1f}% of bf16, bit for bit")
    # attn_decode against SDPA on the decoded keys and values: pages and tails of every length, 1-16 queries a head.
    F = torch.nn.functional
    for D, H, G in [(128, 4, 7), (128, 8, 8), (64, 2, 7), (128, 2, 16), (128, 1, 1)]:
        for T in [1, 63, 64, 65, 200, 1024, 4101]:
            lay = GlydKVLayer(fused=True)
            k = (torch.randn(2, H, T, D, device="cuda") * torch.randn(1, H, 1, D, device="cuda").exp()).to(torch.bfloat16)
            v = torch.randn(2, H, T, D, device="cuda").to(torch.bfloat16)
            lay.update(k[:, :, :-1], v[:, :, :-1]) if T > 1 else None
            K, V = lay.update(k[:, :, -1:], v[:, :, -1:])
            q = torch.randn(2, H * G, 1, D, device="cuda").to(torch.bfloat16)
            ref = F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(G, 1), v.float().repeat_interleave(G, 1)).transpose(1, 2)
            if isinstance(K, Paged):
                got, _ = attention(None, q, K, V, None, scaling=D**-0.5)
                err = ((got.float() - ref).abs().max() / ref.abs().max()).item()
                assert err < 2e-2, (D, H, G, T, err)
                assert torch.equal(got, attention(None, q, K, V, None, scaling=D**-0.5)[0])  # the same every run
        print(f"attn_decode D={D} H={H} G={G}: pages and tails 1-{T} tokens, within 2e-2 of SDPA, the same every run")
