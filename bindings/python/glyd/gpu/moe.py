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

The families whose experts transformers runs by their own code (OWN) are
packed the same way, their forward taken over where it multiplies: Step
3.7's and LongCat-Flash's Experts (as "glyd"'s; LongCat's zero-compute
experts adding their tokens' rows), DBRX's (its gate, up and down matrices
each a pack), Llama 4's MoE block (each token's rows times its experts'
scores through its experts: they scale the input, not the output), and
Aria's and JetMoE's grouped products over rows sorted by expert (JetMoE's
attention experts too). exact: the family's own code on the matrices
decoded (Llama 4, Aria, JetMoE: all of them, as they read them all).
Compiled, each is one node as well (glyd::experts; glyd::moe_block for
Llama 4's block, glyd::whole for its experts exact, glyd::grouped).

    moe.compress(model, "mma12", lambda m: torch.device("cuda"))   # a loaded model's experts, in place
"""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import GELUTanh, SiLUActivation
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface, _default_apply_gate
from . import _lib, kernels as g, model as gm

NAME = "glyd"
# Experts that transformers runs by the family's own code: the class holding the weights, their names, the
# attribute giving E, and those held transposed ([E, in, out]; DBRX's w2 an [in, out] matrix an expert).
OWN = {
    "Llama4TextExperts": (("gate_up_proj", "down_proj"), "num_experts", ("gate_up_proj", "down_proj")),
    "Step3p7Experts": (("gate_up_proj", "down_proj"), "num_experts", ()),
    "LongcatFlashExperts": (("gate_up_proj", "down_proj"), "num_routed_experts", ()),  # gate_up_proj's zero-compute experts' rows unused
    "DbrxExpertGLU": (("w1", "v1", "w2"), "moe_num_experts", ("w2",)),
    "AriaGroupedExpertsGemm": (("weight",), "groups", ("weight",)),
    "JetMoeParallelExperts": (("weight",), "num_experts", ()),
}


def is_experts(m):
    """An Experts module transformers runs through an experts implementation (use_experts_implementation)."""
    return hasattr(m, "has_gate") and hasattr(m, "is_transposed") and hasattr(type(m).forward, "__wrapped__")


def names(m):
    """An Experts module's two 3-D weights: gate_up_proj (or up_proj), down_proj."""
    return ("gate_up_proj" if m.has_gate else "up_proj", "down_proj")


def held(m):
    """How m holds experts' weights: (their names, E, those held transposed); None if it holds none."""
    if is_experts(m):
        return names(m), m.num_experts, names(m) if m.is_transposed else ()
    o = OWN.get(type(m).__name__)
    return o and (o[0], getattr(m, o[1]), o[2])


def _matrices(w, E):
    """A weight held as E experts' matrices, [E, a, b] (a 2-D one [E a, b]; LongCat's gate_up_proj past E unused)."""
    return w[:E] if w.dim() == 3 else w.view(E, -1, w.shape[1])


def weights(m):
    """The names of m's experts' weights to pack: all of them where their matrices [out, in] have rows a multiple of
    64 and columns of 16; else none."""
    h = held(m)
    if h is None:
        return ()
    ns, E, tr = h
    for n in ns:
        w = getattr(m, n, None)
        if not isinstance(w, torch.Tensor) or w.dim() not in (2, 3) or w.numel() == 0 or w.shape[0] < E:
            return ()
        a, b = _matrices(w, E).shape[1:]
        o, i = (b, a) if n in tr else (a, b)
        if o % 64 or i % 16:
            return ()
    return ns


def targets(model):
    """{id(module): (its path, the names of its weights to pack)} for model's modules holding experts' weights packed."""
    return {id(m): (n, weights(m)) for n, m in model.named_modules() if weights(m)}


def packable_bytes(model):
    """The bf16 bytes of model's expert weights that are packed."""
    return sum(_matrices(getattr(m, n), held(m)[1]).numel() * 2 for m in model.modules() for n in weights(m))


def stacked(w, E, transposed):
    """E experts' matrices (w as held) as one matrix [E out, in]."""
    w = _matrices(w, E)
    if transposed:
        w = w.transpose(1, 2)
    return w.contiguous().view(-1, w.shape[2])


@torch.no_grad()
def take(module, name, w, layout, verify=False):
    """module's experts' weight `name`, w (bf16, on its GPU): packed onto the module (module.glyd_packs; its shape as
    held, module.glyd_held), the parameter left empty; verify: the pack decoded and compared bit for bit. The
    tensors verified."""
    _, E, tr = held(module)
    m2 = stacked(w, E, name in tr)
    p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(m2)
    if verify:
        gm.check(p, m2, f"{name} of a {tuple(w.shape)} {type(module).__name__}")
    del m2
    put(module, name, p, (tuple(_matrices(w, E).shape) if w.dim() == 3 else tuple(w.shape), name in tr))
    # Off the meta device with no bytes, as the Linears' weights packed (hf.py).
    setattr(module, name, nn.Parameter(w.new_empty(0), requires_grad=False))
    module._is_hf_initialized = True
    return int(verify)


def put(module, name, p, how):
    """Pack p as module's weight `name`, held as how: (its shape, transposed)."""
    if getattr(module, "glyd_packs", None) is None:
        module.glyd_packs, module.glyd_held = {}, {}
    module.glyd_packs[name], module.glyd_held[name] = p, how


def decoded(module, p, name, out):
    """Pack p (module's weight `name`) decoded into out, as the module holds it (a transposed one copied)."""
    shape, transposed = module.glyd_held[name]
    E = held(module)[1]
    x = out.view(E, p.shape[0] // E, p.shape[1])
    return (x.transpose(1, 2).contiguous() if transposed else x).view(shape)


def _act(m, O, gate):
    """The gate and up product's activation where its sums are written out (O: an expert's rows of it): 1 SiLU,
    2 GELU (tanh), for the default gate (gate then up, concatenated, halves of whole row blocks); else 0 (its rows
    as they are, then the module's own gate or activation)."""
    if not getattr(m, "has_gate", True) or not getattr(m, "is_concatenated", True) or gate is not _default_apply_gate or O % 128:
        return 0
    return 1 if type(m.act_fn) in (SiLUActivation, nn.SiLU) or m.act_fn is F.silu else 2 if type(m.act_fn) is GELUTanh else 0


def _counters(p, E):
    """p's products' done counters made now: never in a CUDA graph's memory pool."""
    if g.lib() is not None:
        _lib._counters("mma12_moe" if isinstance(p, g.Mma12) else "mma_moe", p.sm.get_device(), p.shape[0] // E // 64 * E, 1 << 16)


def _take_over(m, run):
    """m's forward run(m, ...) (m.glyd_run): its class's taken over once, the class's own where a module has none (not
    packed, another model's). Nothing of m's refers to m, so a model let go of is freed with no garbage collection
    (but where accelerate's hook, which refers to its module, is on it)."""
    cls = type(m)
    if getattr(cls.forward, "glyd_own", None) is None:
        own = cls.forward

        def forward_(self, *args, **kwargs):
            return (getattr(self, "glyd_run", None) or own)(self, *args, **kwargs)

        forward_.glyd_own = own
        cls.forward = forward_
    m.glyd_run = run
    if hasattr(m, "_old_forward"):  # accelerate's hook (a device map over several GPUs, put before the packs) calls the forward it found
        m._old_forward = cls.forward.__get__(m)
    object.__setattr__(m, "__reduce_ex__", _uncopyable)  # (a copy would share the packs; refused, as a packed Experts module is)


def _own_forward(cls):
    """cls's forward before glyd took it over."""
    return getattr(cls.forward, "glyd_own", cls.forward)


def _owner(m):
    """The module holding m's packs: m, DBRX's Experts module's mlp."""
    return m.mlp if type(m).__name__ == "DbrxExperts" else m


def install(model, exact=False):
    """model's Experts modules run by "glyd", the implementation each ran by until now kept as its reference (the
    one exact runs, and a module not packed); exact: every one decodes its matrices and runs its reference. The
    families run by their own code (OWN) taken over where their packs are. The Experts modules packed."""
    mods = [m for m in model.modules() if is_experts(m)]
    n = 0
    for m in mods:
        if m.config._experts_implementation != NAME:
            m.glyd_ref = m.config._experts_implementation
        packs = getattr(m, "glyd_packs", None)
        if packs:
            if set(packs) != set(names(m)):
                raise ValueError(f"glyd: an Experts module's {', '.join(sorted(set(names(m)) - set(packs)))} not in the checkpoint")
            _unit(m, m, [packs[k] for k in names(m)], getattr(type(m), "_apply_gate", _default_apply_gate), None, exact)
            n += 1
    for m in model.modules():
        kind = type(m).__name__
        if kind in ("Step3p7Experts", "LongcatFlashExperts") and getattr(m, "glyd_packs", None):
            m.has_gate, m.has_bias, m.is_transposed, m.is_concatenated = True, False, False, True
            m.glyd_ref = None  # the class's own forward
            _unit(m, m, [m.glyd_packs["gate_up_proj"], m.glyd_packs["down_proj"]], getattr(type(m), "_apply_gate", _default_apply_gate), None, exact)
            _take_over(m, forward)
            n += 1
        elif kind == "DbrxExperts" and getattr(m.mlp, "glyd_packs", None):
            m.has_gate, m.has_bias, m.glyd_ref = True, False, None
            p = m.mlp.glyd_packs
            _unit(m, m.mlp, [p["v1"], p["w2"]], None, (p["w1"], m.mlp.activation_fn), exact)
            _take_over(m, forward)
            n += 1
        elif kind == "Llama4TextMoe" and getattr(m.experts, "glyd_packs", None):
            e, p = m.experts, m.experts.glyd_packs
            e.glyd = (p["gate_up_proj"], p["down_proj"], e.num_experts, _act(e, p["gate_up_proj"].shape[0] // e.num_experts, _default_apply_gate), None, False)
            for q in p.values():
                _counters(q, e.num_experts)
            if exact:  # all of them: its products read them all
                _take_over(e, _whole)
                m.glyd_run = None  # the block's own forward (installed again, fused before)
            else:
                _take_over(m, _llama4)
                e.glyd_run = None
            _handle(e if exact else m)
            n += 1
        elif kind in ("AriaGroupedExpertsGemm", "JetMoeParallelExperts") and getattr(m, "glyd_packs", None):
            m.glyd_exact = exact
            _counters(m.glyd_packs["weight"], held(m)[1])
            _take_over(m, _grouped)
            _handle(m)
            n += 1
    if mods:
        if hasattr(model, "set_experts_implementation"):
            model.set_experts_implementation(NAME)
        for m in mods:
            if m.config._experts_implementation != NAME:  # a config the model's call did not reach
                m.config._experts_implementation_internal = NAME
        if hasattr(type(model), "_optimize_model_for_decode"):
            _decoding(type(model))
    return n


def _unit(m, owner, packs, gate, own_gate, exact):
    """m (an Experts module, or one run as one) over owner's packs [up, down]: (up, down, E, act, a gate matrix of
    its own and its activation (DBRX) or None, zero-compute experts past E (LongCat's)), exact; its handle for the op
    below; its done counters."""
    E = held(owner)[1]
    up, down = packs
    zero = bool(getattr(m, "zero_expert_num", 0))
    m.glyd, m.glyd_exact = (up, down, E, 0 if own_gate else _act(m, up.shape[0] // E, gate), own_gate, zero), exact
    _handle(m)
    for p in owner.glyd_packs.values():
        _counters(p, E)


def _handle(m):
    """m's name for the ops below (a copy would keep it, naming this module: copy.deepcopy, copy.copy and pickle
    refused)."""
    if getattr(m, "glyd_handle", None) is None:
        m.glyd_handle = next(gm._handles)
        gm._modules[m.glyd_handle] = m
        object.__setattr__(m, "__reduce_ex__", _uncopyable)


def _uncopyable(protocol):
    raise TypeError("glyd: a model with packed experts can't be copied or pickled (copy.deepcopy, torch.save): load it again")


def _decoding(cls):
    """transformers' generate() decodes with batched_mm where a model's experts run by grouped_mm
    (GenerationMixin._optimize_model_for_decode, around its decoding loop): cls's taken over once to tell the
    model's Experts modules while it lasts, so that the reference (exact, a module not packed) follows it. The class's,
    not the model's: a copy (copy.deepcopy, pickle) follows it too, and nothing of the model's refers to it."""
    own = cls._optimize_model_for_decode
    if getattr(own, "glyd_own", None) is not None:
        return

    @contextlib.contextmanager
    def decoding(self):
        mods = [m for m in self.modules() if is_experts(m)]
        for m in mods:
            m.glyd_decoding = True
        try:
            with own(self):
                yield
        finally:
            for m in mods:
                m.glyd_decoding = False

    decoding.glyd_own = own
    cls._optimize_model_for_decode = decoding


@torch.no_grad()
def rest(model, layout, verify=False, device_of=lambda m: None):
    """model's experts' weights not packed yet, packed (in `layout`, on the GPU device_of(module) gives, else their
    own), a weight several modules share (DiffusionGemma's encoder's and decoder's) once. The tensors verified."""
    done, n = {}, 0
    for m in list(model.modules()):
        for name in weights(m):
            w = getattr(m, name)
            key = (w.data_ptr(), w.device)
            if key in done:
                put(m, name, *done[key])
                setattr(m, name, nn.Parameter(w.new_empty(0), requires_grad=False))
            else:
                n += take(m, name, w.data.to(device_of(m) or w.device), layout, verify)
                done[key] = (m.glyd_packs[name], m.glyd_held[name])
    return n


def compress(model, layout, device_of, exact=False):
    """Every mixture of experts' layer of an already-loaded model with its experts' weights packed (in `layout`, on
    the GPU device_of(module) gives) and run by "glyd" (the families of OWN by their own code taken over). The
    Experts modules packed."""
    rest(model, layout, device_of=device_of)
    return install(model, exact)


def nbytes(model):
    """The bytes of model's experts' packs (one several modules share, once)."""
    return sum(p.nbytes() for p in {id(p): p for m in model.modules() for p in (getattr(m, "glyd_packs", None) or {}).values()}.values())


def scratch(model, exact):
    """{device: weights} the scratch buffer holds for model's experts: exact, a module's matrices decoded at once;
    else none."""
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
    """A module as its reference implementation sees it, with its matrices decoded; called, its class's forward."""

    def __init__(self, m, w):
        self.__dict__.update(w)
        self._m = m

    def __getattr__(self, name):
        return getattr(self._m, name)

    def __call__(self, *args, **kwargs):
        return _own_forward(type(self._m))(self, *args, **kwargs)


def _buffer(device):
    """The scratch buffer on device (a compiled graph's node: its address kept by its CUDA graph, model.set_scratch)."""
    if _lib.local.fresh:
        gm.Scratch.graphed.add(device)
    return gm.Scratch.buf[device]


def _reference(self, hidden_states, top_k_index, top_k_weights):
    """The implementation the module ran by before "glyd" (a family of OWN: its class's own forward), on its matrices
    decoded into the scratch buffer where they are packed: the experts the tokens are routed to, the only ones it
    reads (the rest of the buffer as it was)."""
    ref = getattr(self, "glyd_ref", "eager")  # (a module "glyd" never installed: another model made from a packed one's config)
    owner = _owner(self)
    packs = getattr(owner, "glyd_packs", None)
    if ref == "grouped_mm" and getattr(self, "glyd_decoding", False):
        ref = "batched_mm"  # as generate() decodes bf16's
    elif ref == "grouped_mm" and packs and torch.cuda.is_current_stream_capturing() and not _captures_grouped_mm(hidden_states.device):
        # A CUDA graph captures (torch.compile's reduce-overhead) where grouped_mm copies to the host (bf16's own
        # compiled forward stops there): batched_mm, as generate() runs bf16's decoding steps.
        ref = "batched_mm"
    fn = _own_forward(type(self)) if ref is None else ALL_EXPERTS_FUNCTIONS.get_interface(ref, type(self).forward.__wrapped__)
    if not packs:
        return fn(self, hidden_states, top_k_index, top_k_weights)
    E = self.glyd[2]
    # batched_mm reads expert E - 1 for an id past it (clamped): decoded too
    plan = g.moe_route(top_k_index.long().clamp(0, E - 1), E)
    buf, at, w = _buffer(next(iter(packs.values())).sm.device), 0, {}
    for name, p in packs.items():
        w[name] = decoded(owner, p, name, g.mma_moe_unpack(p, E, plan, top_k_index.numel(), buf[at : at + p.n]))
        at += p.n
    # DBRX: its matrices its mlp's
    return fn(_Decoded(self, w) if owner is self else _Decoded(self, {"mlp": _Decoded(owner, w)}), hidden_states, top_k_index, top_k_weights)


def _whole(self, *args):
    """A module's own forward on all of its matrices decoded (exact: Llama 4's experts, Aria's and JetMoE's grouped
    products, which read them all)."""
    if torch.compiler.is_compiling():  # Llama 4's experts: one node of the graph (glyd::whole), run as eager
        return torch.ops.glyd.whole(_detached(args[0]), self.glyd_handle)
    buf, at, w = _buffer(next(iter(self.glyd_packs.values())).sm.device), 0, {}
    for name, p in self.glyd_packs.items():
        w[name] = decoded(self, p, name, g.mma_unpack(p, buf[at : at + p.n]))
        at += p.n
    return _own_forward(type(self))(_Decoded(self, w), *args)


def forward(self, hidden_states, top_k_index, top_k_weights):
    """The "glyd" experts implementation: hidden_states [T, H], each token's k experts (top_k_index [T, k]) and
    their weights (top_k_weights [T, k]) to [T, H]."""
    if torch.compiler.is_compiling():
        handle = getattr(self, "glyd_handle", None)
        if handle is not None:  # packed: one node of the graph (glyd::experts), which runs what follows (no gradient, as eager)
            return torch.ops.glyd.experts(_detached(hidden_states), top_k_index, _detached(top_k_weights), handle)
        return _reference(self, hidden_states, top_k_index, top_k_weights)  # not packed: bf16's, compiled as bf16's
    packs = getattr(self, "glyd", None)
    if packs is None or self.glyd_exact:
        return _reference(self, hidden_states, top_k_index, top_k_weights)
    up, down, E, act, own_gate, zero = packs
    # The biases as the module holds them now (accelerate's dispatch puts new tensors in their place).
    bu, bd = (getattr(self, names(self)[0] + "_bias"), self.down_proj_bias) if self.has_bias else (None, None)
    x = hidden_states if hidden_states.dim() == 2 else hidden_states.reshape(-1, hidden_states.shape[-1])  # DBRX's [B, S, H]
    x = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    ids = top_k_index if top_k_index.dtype == torch.int64 else top_k_index.long()
    w = top_k_weights if top_k_weights.dtype in (torch.bfloat16, torch.float32) else top_k_weights.float()
    plan = g.moe_route(ids, E)
    h = g.mma_moe(up, E, x, plan, ids, act, bu)
    if own_gate is not None:  # DBRX: a gate matrix of its own, act(gate) up
        h = own_gate[1](g.mma_moe(own_gate[0], E, x, plan, ids)) * h
    elif not act:
        h = getattr(type(self), "_apply_gate", _default_apply_gate)(self, h) if self.has_gate else self.act_fn(h)
    y = g.mma_moe(down, E, h, plan, ids, 0, bd, w, gather=False).to(hidden_states.dtype)
    if zero:  # LongCat's zero-compute experts, past E: their tokens' rows as they are
        y = y + x * (w * (ids >= E)).sum(-1, keepdim=True).to(x.dtype)
    return y if hidden_states.dim() == 2 else y.view(hidden_states.shape)


def _llama4(self, hidden_states):
    """Llama4TextMoe's forward, its experts packed: each token's top k experts as its router picks them, the token's
    row times each one's score (a sigmoid, as the router's) through that expert, added, and the shared expert's."""
    if torch.compiler.is_compiling():  # one node of the graph (glyd::moe_block), run as eager
        return tuple(torch.ops.glyd.moe_block(_detached(hidden_states), self.glyd_handle))
    x = hidden_states.reshape(-1, self.hidden_dim)
    logits = nn.Linear.forward(self.router, x)
    top, ids = torch.topk(logits, self.top_k, dim=1)
    rows = (x.unsqueeze(1) * torch.sigmoid(top.float()).to(x.dtype).unsqueeze(-1)).view(-1, x.shape[1])  # a pair's row
    up, down, E, act = self.experts.glyd[:4]
    pair = ids.reshape(-1, 1)  # each pair a token of its own, k = 1
    plan = g.moe_route(pair, E)
    h = g.mma_moe(up, E, rows, plan, pair, act)
    if not act:
        gate, u = h.chunk(2, dim=-1)
        h = u * self.experts.act_fn(gate)
    y = g.mma_moe(down, E, h, plan, pair, 0, None, torch.ones_like(pair, dtype=torch.float32), gather=False)
    out = self.shared_expert(x)
    out.add_(y.view(-1, self.top_k, x.shape[1]).sum(dim=1))
    return out, logits


def _grouped(self, inputs, counts):
    """AriaGroupedExpertsGemm's and JetMoeParallelExperts' forward, packed: inputs [N, in], grouped by expert in
    order (counts[e] rows of expert e), each by its expert's matrix. exact: the class's own on it decoded."""
    if torch.compiler.is_compiling():  # one node of the graph (glyd::grouped), run as eager
        return torch.ops.glyd.grouped(_detached(inputs), torch.as_tensor(counts, device=inputs.device), self.glyd_handle)
    if self.glyd_exact:
        return _whole(self, inputs, counts)  # (counts as the class takes them: JetMoE's a list)
    p, E = self.glyd_packs["weight"], held(self)[1]  # (weights applied by the family's own code)
    counts = torch.as_tensor(counts, device=inputs.device)
    ids = torch.repeat_interleave(torch.arange(E, device=inputs.device), counts, output_size=inputs.shape[0]).view(-1, 1)
    return g.mma_moe(p, E, inputs.to(torch.bfloat16), g.moe_route(ids, E), ids).to(inputs.dtype)  # rows in the order they came


def _detached(x):
    return x.detach() if x.requires_grad else x  # (no gradient, as eager)


def _run(f, handle, *args):
    """f(module handle names, *args) as eager, its workspaces for the call alone (a CUDA graph keeps the addresses it
    captured): the ops below."""
    _lib.local.fresh = True
    try:
        return f(gm._modules[handle], *args)
    finally:
        _lib.local.fresh = False


@torch.library.custom_op("glyd::experts", mutates_args=())
def _experts(hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, handle: int) -> torch.Tensor:
    return _run(forward, handle, hidden_states, top_k_index, top_k_weights)


@_experts.register_fake
def _(hidden_states, top_k_index, top_k_weights, handle):
    return hidden_states.new_empty(hidden_states.shape)


@torch.library.custom_op("glyd::moe_block", mutates_args=())
def _moe_block(hidden_states: torch.Tensor, handle: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _run(_llama4, handle, hidden_states)


@_moe_block.register_fake
def _(hidden_states, handle):
    m = gm._modules[handle]
    x = hidden_states.reshape(-1, m.hidden_dim)
    return x.new_empty(x.shape), x.new_empty(x.shape[0], m.num_experts)


@torch.library.custom_op("glyd::whole", mutates_args=())
def _whole_op(x: torch.Tensor, handle: int) -> torch.Tensor:
    return _run(_whole, handle, x)


@_whole_op.register_fake
def _(x, handle):
    return x.new_empty(x.shape)


@torch.library.custom_op("glyd::grouped", mutates_args=())
def _grouped_op(inputs: torch.Tensor, counts: torch.Tensor, handle: int) -> torch.Tensor:
    return _run(_grouped, handle, inputs, counts)


@_grouped_op.register_fake
def _(inputs, counts, handle):
    m = gm._modules[handle]
    p = m.glyd_packs["weight"]
    return inputs.new_empty(inputs.shape[0], p.shape[0] // held(m)[1])


ExpertsInterface.register(NAME, forward)
