"""A mixture-of-experts layer's experts packed. transformers (5.17 on) keeps
a layer's experts as 3-D parameters of an Experts module, gate_up_proj
[E, 2I, H] (or up_proj [E, I, H]) and down_proj [E, H, I] ([E, in, out]
where the class says is_transposed), and runs them through an experts
implementation chosen by name (use_experts_implementation; grouped_mm by
default). Here each of the two is packed as one matrix of its E experts'
stacked, [E out, in], in the model's layout, and "glyd", the implementation
registered here, multiplies them: each token's k choices (its pairs)
sorted by expert on the GPU, one grouped product for gate and up with the
activation applied as its sums are written out, one for down with the
routing weights applied and each token's k rows added; no host sync. A gate
of the model's own (gpt-oss's) or none: that product's rows written out,
then the module's _apply_gate or act_fn on them. exact: the layer's
experts the tokens are routed to decoded into the scratch buffer and run
by the implementation bf16 took (the model's before "glyd"), which reads
no other, so its outputs are bf16's bit for bit.

    moe.compress(model, "mma12", lambda m: torch.device("cuda"))   # a loaded model's experts, in place
"""
import torch
import torch.nn as nn
from transformers.activations import GELUTanh, SiLUActivation
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface, _default_apply_gate
from . import kernels as g, model as gm

NAME = "glyd"


def is_experts(m):
    """An Experts module transformers runs through an experts implementation (use_experts_implementation)."""
    return hasattr(m, "has_gate") and hasattr(m, "is_transposed") and hasattr(type(m).forward, "__wrapped__")


def names(m):
    """An Experts module's two 3-D weights: gate_up_proj (or up_proj), down_proj."""
    return ("gate_up_proj" if m.has_gate else "up_proj", "down_proj")


def weights(m):
    """The names of an Experts module's weights to pack: both of names(m) where their matrices [out, in] have rows
    a multiple of 64 and columns of 16; else none."""
    for n in names(m):
        w = getattr(m, n, None)
        if not isinstance(w, torch.Tensor) or w.dim() != 3:
            return ()
        o, i = (w.shape[2], w.shape[1]) if m.is_transposed else (w.shape[1], w.shape[2])
        if o % 64 or i % 16:
            return ()
    return names(m)


def targets(model):
    """{id(module): (its path, the names of its weights to pack)} for model's Experts modules packed."""
    return {id(m): (n, weights(m)) for n, m in model.named_modules() if is_experts(m) and weights(m)}


def packable_bytes(model):
    """The bf16 bytes of model's expert weights that are packed."""
    return sum(getattr(m, n).numel() * 2 for m in model.modules() if is_experts(m) for n in weights(m))


def stacked(w, transposed):
    """A layer's experts' matrices (w [E, out, in]; [E, in, out] transposed) as one matrix [E out, in]."""
    if transposed:
        w = w.transpose(1, 2)
    return w.contiguous().view(-1, w.shape[2])


@torch.no_grad()
def take(module, name, w, layout, verify=False):
    """module's 3-D weight `name`, w (bf16, on its GPU): packed onto the module (module.glyd_packs), the
    parameter left empty; verify: the pack decoded and compared bit for bit. The tensors verified."""
    m2 = stacked(w, module.is_transposed)
    p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(m2)
    if verify:
        gm.check(p, m2, f"{name} of a {tuple(w.shape)} Experts module")
    del m2
    if getattr(module, "glyd_packs", None) is None:
        module.glyd_packs = {}
    module.glyd_packs[name] = p
    # Off the meta device with no bytes, as the Linears' weights packed (hf.py).
    setattr(module, name, nn.Parameter(w.new_empty(0), requires_grad=False))
    module._is_hf_initialized = True
    return int(verify)


def _act(m, O):
    """The gate and up product's activation where its sums are written out (O: an expert's rows of it): 1 SiLU,
    2 GELU (tanh), for the default gate (gate then up, concatenated, halves of whole row blocks); else 0 (its rows
    as they are, then the module's own gate or activation)."""
    if not m.has_gate or not m.is_concatenated or getattr(type(m), "_apply_gate", None) is not _default_apply_gate or O % 128:
        return 0
    return 1 if type(m.act_fn) in (SiLUActivation, nn.SiLU) else 2 if type(m.act_fn) is GELUTanh else 0


def install(model, exact=False):
    """model's Experts modules run by "glyd", the implementation each ran by until now kept as its reference (the
    one exact runs, and a module not packed); exact: every one decodes its matrices and runs its reference. The
    Experts modules packed."""
    mods = [m for m in model.modules() if is_experts(m)]
    for m in mods:
        if m.config._experts_implementation != NAME:
            m.glyd_ref = m.config._experts_implementation
        packs = getattr(m, "glyd_packs", None)
        if packs:
            if set(packs) != set(names(m)):
                raise ValueError(f"glyd: an Experts module's {', '.join(sorted(set(names(m)) - set(packs)))} not in the checkpoint")
            up, down = (packs[n] for n in names(m))
            E = m.num_experts
            bias = (getattr(m, names(m)[0] + "_bias").contiguous(), m.down_proj_bias.contiguous()) if m.has_bias else (None, None)
            m.glyd, m.glyd_exact = (up, down, E, _act(m, up.shape[0] // E), *bias), exact
    if mods:
        if hasattr(model, "set_experts_implementation"):
            model.set_experts_implementation(NAME)
        for m in mods:
            if m.config._experts_implementation != NAME:  # a config the model's call did not reach
                m.config._experts_implementation_internal = NAME
    return sum(1 for m in mods if getattr(m, "glyd", None) is not None)


@torch.no_grad()
def compress(model, layout, device_of, exact=False):
    """Every Experts module of an already-loaded model with its weights packed (in `layout`, on the GPU
    device_of(module) gives) and run by "glyd". The Experts modules packed."""
    for m in model.modules():
        if is_experts(m):
            for name in weights(m):
                take(m, name, getattr(m, name).data.to(device_of(m)), layout)
    return install(model, exact)


def nbytes(model):
    """The bytes of model's experts' packs."""
    return sum(p.nbytes() for m in model.modules() for p in (getattr(m, "glyd_packs", None) or {}).values())


def scratch(model, exact):
    """{device: weights} the scratch buffer holds for model's Experts modules: exact, a layer's two matrices
    decoded at once; else none."""
    need = {}
    for m in model.modules() if exact else ():
        packs = getattr(m, "glyd_packs", None)
        if packs:
            d = next(iter(packs.values())).sm.device
            need[d] = max(need.get(d, 0), sum(p.n for p in packs.values()))
    return need


class _Decoded:
    """An Experts module as its reference implementation sees it, with its matrices decoded."""

    def __init__(self, m, w):
        self.__dict__.update(w)
        self._m = m

    def __getattr__(self, name):
        return getattr(self._m, name)


def _reference(self, hidden_states, top_k_index, top_k_weights):
    """The implementation the module ran by before "glyd", on its matrices decoded into the scratch buffer where
    they are packed: the experts the tokens are routed to, the only ones it reads (the rest of the buffer as it was)."""
    fn = ALL_EXPERTS_FUNCTIONS.get_interface(self.glyd_ref, type(self).forward.__wrapped__)
    packs = getattr(self, "glyd_packs", None)
    if not packs:
        return fn(self, hidden_states, top_k_index, top_k_weights)
    E = self.num_experts
    plan = g.moe_route(top_k_index if top_k_index.dtype == torch.int64 else top_k_index.long(), E)
    buf, at, w = gm.Scratch.buf[next(iter(packs.values())).sm.device], 0, {}
    for name, p in packs.items():
        x = g.mma_moe_unpack(p, E, plan, top_k_index.numel(), buf[at : at + p.n]).view(E, p.shape[0] // E, p.shape[1])
        w[name] = x.transpose(1, 2).contiguous() if self.is_transposed else x
        at += p.n
    return fn(_Decoded(self, w), hidden_states, top_k_index, top_k_weights)


def forward(self, hidden_states, top_k_index, top_k_weights):
    """The "glyd" experts implementation: hidden_states [T, H], each token's k experts (top_k_index [T, k]) and
    their weights (top_k_weights [T, k]) to [T, H]."""
    packs = getattr(self, "glyd", None)
    if packs is None or self.glyd_exact:
        return _reference(self, hidden_states, top_k_index, top_k_weights)
    up, down, E, act, bu, bd = packs
    x = hidden_states if hidden_states.dtype == torch.bfloat16 else hidden_states.to(torch.bfloat16)
    ids = top_k_index if top_k_index.dtype == torch.int64 else top_k_index.long()
    w = top_k_weights if top_k_weights.dtype in (torch.bfloat16, torch.float32) else top_k_weights.float()
    plan = g.moe_route(ids, E)
    h = g.mma_moe(up, E, x, plan, ids, act, bu)
    if not act:
        h = self._apply_gate(h) if self.has_gate else self.act_fn(h)
    return g.mma_moe(down, E, h, plan, ids, 0, bd, w, gather=False).to(hidden_states.dtype)


ExpertsInterface.register(NAME, forward)
