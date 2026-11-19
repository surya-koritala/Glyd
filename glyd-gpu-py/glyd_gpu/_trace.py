"""The functions torch.compile traces. Dynamo reads a function's bytecode, which a compiled module has none of, so these
stay Python source in the wheel: each does no more than call an op (glyd::linear, glyd::experts ...) or hand over to the
eager code, which the other modules hold compiled and which model.py and moe.py put here by name."""
import torch

try:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
except ImportError:  # (no transformers: nothing of the experts' side runs)
    ALL_EXPERTS_FUNCTIONS = None


def _detached(x):
    return x.detach() if x.requires_grad else x  # (no gradient, as eager)


# --- model.py: a packed Linear, an embedding, the Linears merged into one product ----------------------------------------------

def linear(self, x):
    if torch.compiler.is_compiling():  # one node of the graph (glyd::linear), which runs the eager call
        return torch.ops.glyd.linear(_detached(x), self.handle, self.out_features)
    return self.eager(x)


def embedding(self, ids):
    if torch.compiler.is_compiling():  # one node of the graph (glyd::embedding), which runs the eager call
        rows = torch.ops.glyd.embedding(ids, self.handle, self.embedding_dim)
    else:
        rows = self.eager(ids)
    return rows if self.embed_norm is None else self.embed_norm(rows)  # (Inkling's, Muse Glimmer's: the norm, after the lookup)


def part(self, i, x):
    h = self.held
    if h[0] is not x:
        h[0], h[1], h[2] = x, self.lin(x).split(self.sizes, -1), len(self.sizes)
    h[2] -= 1
    y = h[1][i]
    if h[2] == 0:
        h[0] = h[1] = None
    return y


def part_forward(self, x):
    return self.group[0].part(self.i, x)


class Weight:
    """A packed module's weight as a model's own code reads it (Llama 4: the embedding's device; HunYuan V4, DeepSeek
    V3.2: a Linear's dtype; Gemma 4: the pad token's row): its device, dtype and shape as they are; its rows by index
    (an embedding's: those rows alone decoded), anything else and a torch function on the matrix decoded."""

    dtype, requires_grad = torch.bfloat16, False

    def __init__(self, m):
        self.m = m

    device = property(lambda self: self.m.p.core.device)
    shape = property(lambda self: torch.Size(self.m.p.shape))

    def tensor(self):
        from .model import unpack

        return unpack(self.m.p)

    def __getitem__(self, i):
        from .model import GEmbedding

        i = i if isinstance(i, tuple) else (i,)
        if isinstance(self.m, GEmbedding) and isinstance(i[0], (int, torch.Tensor)):
            ids = torch.as_tensor(i[0], device=self.device).remainder(self.shape[0])
            return self.m.rows(ids.reshape(-1)).view(*ids.shape, -1)[(slice(None),) * ids.dim() + i[1:]]
        return self.tensor()[i]

    def __getattr__(self, name):
        return getattr(self.tensor(), name)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        args, kwargs = torch.utils._pytree.tree_map(lambda a: a.tensor() if isinstance(a, cls) else a, (args, kwargs or {}))  # (nested: torch.cat's list)
        return func(*args, **kwargs)


# --- moe.py: the experts' forwards; moe.py sets eager_forward, eager_whole, eager_llama4, eager_grouped, decoded_reference and
# captures_grouped_mm here ----------------------------------------------------------------------------------------------------

eager_forward = eager_whole = eager_llama4 = eager_grouped = decoded_reference = captures_grouped_mm = None

def own_forward(cls):
    """cls's forward before glyd took it over."""
    return getattr(cls.forward, "glyd_own", cls.forward)


def owner(m):
    """The module holding m's packs: m, DBRX's Experts module's mlp."""
    return m.mlp if type(m).__name__ == "DbrxExperts" else m


def take_over(own):
    """The forward a class gets in place of its own (own): the module's glyd_run where it has one."""

    def forward_(self, *args, **kwargs):
        return (getattr(self, "glyd_run", None) or own)(self, *args, **kwargs)

    forward_.glyd_own = own
    return forward_


def reference(self, hidden_states, top_k_index, top_k_weights):
    """The implementation the module ran by before "glyd" (a family of OWN: its class's own forward), on its matrices
    decoded where they are packed (decoded_reference); else as it is."""
    ref = getattr(self, "glyd_ref", "eager")  # (a module "glyd" never installed: another model made from a packed one's config)
    packs = getattr(owner(self), "glyd_packs", None)
    if ref == "grouped_mm" and getattr(self, "glyd_decoding", False):
        ref = "batched_mm"  # as generate() decodes bf16's
    elif ref == "grouped_mm" and packs and torch.cuda.is_current_stream_capturing() and not captures_grouped_mm(hidden_states.device):
        ref = "batched_mm"  # (a CUDA graph's capture, where grouped_mm copies to the host)
    fn = own_forward(type(self)) if ref is None else ALL_EXPERTS_FUNCTIONS.get_interface(ref, type(self).forward.__wrapped__)
    if not packs:
        return fn(self, hidden_states, top_k_index, top_k_weights)
    return decoded_reference(fn, self, hidden_states, top_k_index, top_k_weights)


def moe_forward(self, hidden_states, top_k_index, top_k_weights):
    if torch.compiler.is_compiling():
        handle = getattr(self, "glyd_handle", None)
        if handle is not None:  # packed: one node of the graph (glyd::experts), which runs the eager forward
            return torch.ops.glyd.experts(_detached(hidden_states), top_k_index, _detached(top_k_weights), handle)
        return reference(self, hidden_states, top_k_index, top_k_weights)  # not packed: bf16's, compiled as bf16's
    return eager_forward(self, hidden_states, top_k_index, top_k_weights)


def moe_whole(self, *args):
    if torch.compiler.is_compiling():  # Llama 4's experts: one node of the graph (glyd::whole), run as eager
        return torch.ops.glyd.whole(_detached(args[0]), self.glyd_handle)
    return eager_whole(self, *args)


def moe_llama4(self, hidden_states):
    if torch.compiler.is_compiling():  # one node of the graph (glyd::moe_block), run as eager
        return tuple(torch.ops.glyd.moe_block(_detached(hidden_states), self.glyd_handle))
    return eager_llama4(self, hidden_states)


def moe_grouped(self, inputs, counts):
    if torch.compiler.is_compiling():  # one node of the graph (glyd::grouped), run as eager
        return torch.ops.glyd.grouped(_detached(inputs), torch.as_tensor(counts, device=inputs.device), self.glyd_handle)
    return eager_grouped(self, inputs, counts)


# --- vllm_plugin.py: vLLM's compile traces a Linear's apply, an embedding's lookup ----------------------------------------------------------------------

def linear_method_apply(self, layer, x, bias=None):
    if self.plain is not None:
        return self.plain.apply(layer, x, bias)
    return torch.ops.glyd.vllm_linear(x, layer.glyd_data, layer.glyd_a, layer.glyd_b, layer.glyd_words, bias, layer.glyd_out, layer.glyd_exact)


def embedding_method_lookup(self, layer, ids):
    return torch.ops.glyd.vllm_embed(ids, layer.glyd_data, layer.glyd_a, layer.glyd_b, layer.glyd_words, layer.glyd_out, layer.embedding_dim)


def embedding_method_logits(self, layer, x, bias=None):
    return torch.ops.glyd.vllm_logits(x, layer.glyd_data, layer.glyd_a, layer.glyd_b, layer.glyd_words, bias, layer.glyd_out, layer.num_embeddings_per_partition)


def plain_linear_forward(self, x):
    """A plain nn.Linear's forward once its weight is packed (vllm_plugin._after_load): the same op as a packed LinearBase's."""
    return torch.ops.glyd.vllm_linear(x, self.glyd_data, self.glyd_a, self.glyd_b, self.glyd_words, self.bias, self.glyd_out, self.glyd_exact)


# --- vllm_fp8.py: an FP8 Linear's product, one op: its weight decoded into the scratch, then vLLM's own method's apply ------------------

def fp8_linear_method_apply(self, layer, x, bias=None):
    i = layer.glyd_f8_id
    if i is None:  # a Linear left as vLLM runs it (a layout Glyd does not pack)
        return self.inner.apply(layer, x, bias)
    return torch.ops.glyd.f8_linear(x, bias, i)
