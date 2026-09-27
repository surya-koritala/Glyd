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
by the implementation bf16 runs (the model's before "glyd"; while
generate() decodes, batched_mm in grouped_mm's place, as transformers
switches bf16's), which reads no other, so its outputs are bf16's bit for
bit. Under torch.compile a packed module's forward is one node of the graph
(glyd::experts), run as eager, its workspaces made for the call alone and
its done counters before any capture, as model.py's ops: no graph break,
and CUDA graphs capture its kernels. A model with packed experts can't be
copied or pickled (a copy's op would name the first's module).

    moe.compress(model, "mma12", lambda m: torch.device("cuda"))   # a loaded model's experts, in place
"""
import contextlib
import torch
import torch.nn as nn
from transformers.activations import GELUTanh, SiLUActivation
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface, _default_apply_gate
from . import _lib, kernels as g, model as gm

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
            act = _act(m, up.shape[0] // E)
            m.glyd, m.glyd_exact = (up, down, E, act), exact
            if getattr(m, "glyd_handle", None) is None:  # its name for the op below
                m.glyd_handle = next(gm._handles)
                gm._modules[m.glyd_handle] = m
                # A copy would keep the handle, which names this module: copy.deepcopy, copy.copy and pickle refused.
                object.__setattr__(m, "__reduce_ex__", _uncopyable)
            if g.lib() is not None:  # the done counters made now: never in a CUDA graph's memory pool
                for p, units in ((up, up.shape[0] // E // (128 if act else 64)), (down, down.shape[0] // E // 64)):
                    _lib._counters("mma12_moe" if isinstance(p, g.Mma12) else "mma_moe", p.sm.get_device(), units * E, 1 << 16)
    if mods:
        if hasattr(model, "set_experts_implementation"):
            model.set_experts_implementation(NAME)
        for m in mods:
            if m.config._experts_implementation != NAME:  # a config the model's call did not reach
                m.config._experts_implementation_internal = NAME
        if hasattr(type(model), "_optimize_model_for_decode"):
            model._optimize_model_for_decode = _Decoding(model)
    return sum(1 for m in mods if getattr(m, "glyd", None) is not None)


def _uncopyable(protocol):
    raise TypeError("glyd: a model with packed experts can't be copied or pickled (copy.deepcopy, torch.save): load it again")


class _Decoding:
    """transformers' generate() decodes with batched_mm where a model's experts run by grouped_mm
    (GenerationMixin._optimize_model_for_decode, around its decoding loop): in its place on the model, so that the
    reference (exact, a module not packed) follows it, the model's Experts modules told while it lasts. An object
    holding its model, where a closure would hold the first: a copy's (copy.deepcopy, pickle) is the copy's."""

    def __init__(self, model):
        self.model = model

    @contextlib.contextmanager
    def __call__(self):
        mods = [m for m in self.model.modules() if is_experts(m)]
        for m in mods:
            m.glyd_decoding = True
        try:
            with type(self.model)._optimize_model_for_decode(self.model):
                yield
        finally:
            for m in mods:
                m.glyd_decoding = False


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


def _captures_grouped_mm(device):
    """Whether a CUDA graph captures torch's grouped_mm on device: its GPU-only path is taken on compute capability
    9.x and 10.x alone (10.x from torch 2.9); elsewhere it copies its offsets to the host, which a capture refuses."""
    major = torch.cuda.get_device_capability(device)[0]
    return major == 9 or (major == 10 and torch.__version__ >= "2.9")


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
    ref = self.glyd_ref
    packs = getattr(self, "glyd_packs", None)
    if ref == "grouped_mm" and getattr(self, "glyd_decoding", False):
        ref = "batched_mm"  # as generate() decodes bf16's
    elif ref == "grouped_mm" and packs and torch.cuda.is_current_stream_capturing() and not _captures_grouped_mm(hidden_states.device):
        # A CUDA graph captures (torch.compile's reduce-overhead) where grouped_mm copies to the host (bf16's own
        # compiled forward stops there): batched_mm, as generate() runs bf16's decoding steps.
        ref = "batched_mm"
    fn = ALL_EXPERTS_FUNCTIONS.get_interface(ref, type(self).forward.__wrapped__)
    if not packs:
        return fn(self, hidden_states, top_k_index, top_k_weights)
    E = self.num_experts
    # batched_mm reads expert E - 1 for an id past it (clamped): decoded too
    plan = g.moe_route(top_k_index.long().clamp(0, E - 1), E)
    d = next(iter(packs.values())).sm.device
    if _lib.local.fresh:  # a compiled graph's node: the buffer's address kept by its CUDA graph (model.set_scratch)
        gm.Scratch.graphed.add(d)
    gm.Ahead.stop(d)  # a prompt's decodes ahead into the buffer done first
    buf, at, w = gm.Scratch.buf[d], 0, {}
    for name, p in packs.items():
        x = g.mma_moe_unpack(p, E, plan, top_k_index.numel(), buf[at : at + p.n]).view(E, p.shape[0] // E, p.shape[1])
        w[name] = x.transpose(1, 2).contiguous() if self.is_transposed else x
        at += p.n
    return fn(_Decoded(self, w), hidden_states, top_k_index, top_k_weights)


def forward(self, hidden_states, top_k_index, top_k_weights):
    """The "glyd" experts implementation: hidden_states [T, H], each token's k experts (top_k_index [T, k]) and
    their weights (top_k_weights [T, k]) to [T, H]."""
    if torch.compiler.is_compiling():
        handle = getattr(self, "glyd_handle", None)
        if handle is not None:  # packed: one node of the graph (glyd::experts), which runs what follows (no gradient, as eager)
            x = hidden_states.detach() if hidden_states.requires_grad else hidden_states
            w = top_k_weights.detach() if top_k_weights.requires_grad else top_k_weights
            return torch.ops.glyd.experts(x, top_k_index, w, handle)
        return _reference(self, hidden_states, top_k_index, top_k_weights)  # not packed: bf16's, compiled as bf16's
    packs = getattr(self, "glyd", None)
    if packs is None or self.glyd_exact:
        return _reference(self, hidden_states, top_k_index, top_k_weights)
    up, down, E, act = packs
    # The biases as the module holds them now (accelerate's dispatch puts new tensors in their place).
    bu, bd = (getattr(self, names(self)[0] + "_bias"), self.down_proj_bias) if self.has_bias else (None, None)
    x = hidden_states if hidden_states.dtype == torch.bfloat16 else hidden_states.to(torch.bfloat16)
    ids = top_k_index if top_k_index.dtype == torch.int64 else top_k_index.long()
    w = top_k_weights if top_k_weights.dtype in (torch.bfloat16, torch.float32) else top_k_weights.float()
    plan = g.moe_route(ids, E)
    h = g.mma_moe(up, E, x, plan, ids, act, bu)
    if not act:
        h = self._apply_gate(h) if self.has_gate else self.act_fn(h)
    return g.mma_moe(down, E, h, plan, ids, 0, bd, w, gather=False).to(hidden_states.dtype)


@torch.library.custom_op("glyd::experts", mutates_args=())
def _experts(hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, handle: int) -> torch.Tensor:
    _lib.local.fresh = True  # workspaces for the call alone (a CUDA graph keeps the addresses it captured)
    try:
        return forward(gm._modules[handle], hidden_states, top_k_index, top_k_weights)
    finally:
        _lib.local.fresh = False


@_experts.register_fake
def _(hidden_states, top_k_index, top_k_weights, handle):
    return hidden_states.new_empty(hidden_states.shape)


ExpertsInterface.register(NAME, forward)
