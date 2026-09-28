"""from_pretrained(): a Hugging Face checkpoint loaded by transformers with
each Linear's weight packed on its GPU as it arrives.

transformers does the loading: it builds the model on the meta device,
streams the safetensors a tensor at a time (with a quantizer that packs
on the fly it loads them in turn, not ahead), renames and converts the
checkpoint's keys, places every tensor by the device map (several GPUs:
device_map="auto", with accelerate) and ties the embeddings. The
quantizer registered here as "glyd" packs: a Linear's weight as it
arrives, a merged group's (q, k, v; gate, up) when its last weight is in,
a mixture of experts' layer's experts as each 3-D weight arrives (moe.py),
the embeddings and an output layer tied to one once everything is. The
GPU holds the packed model and the weights not yet packed: the
embedding, a group's members. A glyd-v1 checkpoint (glyd.json beside its
safetensors, format.py) loads its packs' buffers in place of the weights.
"""
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.core_model_loading import ConversionOps
from transformers.quantizers import HfQuantizer, get_module_from_name, register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin
from . import format as fmt, kernels as g, model as gm, moe

BITS = {"mma": 10.80, "mma12": 12.04}  # a weight, measured (best_layout's): the sizes the device map is planned by
DTYPES = {"U8": torch.uint8, "I32": torch.int32}


def from_pretrained(name_or_path, *, device="cuda:0", layout="auto", exact=False, merge=True, verify=False, **hf_kwargs):
    """A model from the Hugging Face Hub or a directory with its weights
    packed on the GPU as it loads, ready for generate(): a causal LM, else
    an image-text-to-text one (a checkpoint transformers loads only with
    its vision tower). A bf16 checkpoint, or a glyd-v1 one (save_pretrained)
    whose packs load as saved.

    device: the GPU (hf_kwargs' device_map instead: several GPUs).
    layout: "auto" (best_layout's pick for the GPU), "mma" (tiered, 10.80
      bits a weight) or "mma12" (12.04, a lighter decode); a glyd-v1
      checkpoint is tiered, and packed again into the 12-bit layout where
      that is the one.
    exact: every product decodes its matrix whole and multiplies by
      F.linear, as nn.Linear does: logits bit for bit bf16's (and no
      merging); else products straight from the packed weights.
    merge: q, k, v and gate, up as one product each, as serving engines
      run them (not with exact; a glyd-v1 checkpoint's as it was saved).
    verify: every pack decoded and compared with its weights bit for bit
      as it is made; from a glyd-v1 checkpoint, every tensor decoded and
      its sha256 checked against glyd.json.
    hf_kwargs: transformers' from_pretrained's (revision, token,
      device_map, attn_implementation ...); the dtype is bf16.
    """
    fmt.fetch_manifest(name_or_path, **{k: hf_kwargs[k] for k in ("revision", "token", "cache_dir", "local_files_only") if k in hf_kwargs})
    hf_kwargs.setdefault("device_map", {"": device})
    hf_kwargs.pop("torch_dtype", None)
    hf_kwargs.update(dtype=torch.bfloat16, quantization_config=GlydConfig(layout=layout, exact=exact, merge=merge, verify=verify))
    try:
        return AutoModelForCausalLM.from_pretrained(name_or_path, **hf_kwargs)
    except ValueError as e:  # a checkpoint transformers loads only with its vision tower (Muse Glimmer)
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(name_or_path, **hf_kwargs)


@register_quantization_config("glyd")
class GlydConfig(QuantizationConfigMixin):
    """from_pretrained's options as transformers carries them (after the load, model.config.quantization_config): the
    layout the packs are in, the tensors verified, a glyd-v1 checkpoint's source."""

    def __init__(self, layout="auto", exact=False, merge=True, verify=False, verified=0, source=None, **kwargs):
        self.quant_method = "glyd"
        self.layout, self.exact, self.merge, self.verify, self.verified, self.source = layout, exact, merge, verify, verified, source


def _unparam(model, name):
    """Parameter name of model out of its module, for a pack in its place (nothing to load, place or initialize): an
    empty tensor there until then, as a model's own initialization may read it. Its module."""
    host = model.get_submodule(name.rpartition(".")[0])
    delattr(host, name.rpartition(".")[2])
    placeholder = torch.empty(0, device="meta")
    placeholder._is_hf_initialized = True
    setattr(host, name.rpartition(".")[2], placeholder)
    host._is_hf_initialized = True
    return host


def _cuda(d):
    return torch.device("cuda", d) if isinstance(d, int) else torch.device(d)


def _install(model, paths, p, mode):
    """A GLinear over pack p in place of the Linears at paths (module paths), their biases stacked: one Merged product
    over them where there are several."""
    lins = [model.get_submodule(n) for n in paths]
    bias = None if lins[0].bias is None else torch.cat([l.bias.data for l in lins])
    lin = gm.GLinear(p, bias, **mode)
    parent = model.get_submodule(paths[0].rpartition(".")[0])
    if len(paths) == 1:
        setattr(parent, paths[0].rpartition(".")[2], lin)
    else:
        parent.merged = gm.Merged(lin, [l.out_features for l in lins])
        for i, n in enumerate(paths):
            setattr(parent, n.rpartition(".")[2], gm.Part(parent.merged, i))


class _Pack(ConversionOps):
    """transformers' quantize op: a tensor to pack goes to the quantizer, the rest back to the loader."""

    def __init__(self, quantizer):
        self.quantizer = quantizer

    def convert(self, input_dict, full_layer_name=None, model=None, missing_keys=None, **kwargs):
        out = {}
        for name, value in input_dict.items():
            if self.quantizer.take(model, name, value[0] if isinstance(value, list) else value):
                if missing_keys is not None:
                    missing_keys.discard(name)
            else:
                out[name] = value
        return out


@register_quantizer("glyd")
class GlydQuantizer(HfQuantizer):
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.targets = {}  # id(nn.Linear): [its path, its merged group or None, its place there]
        self.groups = []  # [members' paths, {place: bf16 weight} until all are in, the pack]
        self.experts = {}  # id(Experts module): (its path, its 3-D weights packed) (moe.py)
        self.stored = None  # a glyd-v1 checkpoint's manifest

    def validate_environment(self, device_map=None, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("glyd: the weights are packed on a CUDA GPU, and none is available")
        if isinstance(device_map, dict) and {str(d) for d in device_map.values()} & {"cpu", "disk"}:
            raise ValueError("glyd: every layer on a GPU (a device map without cpu or disk)")

    def update_dtype(self, dtype):
        return torch.bfloat16  # a pack holds bf16's bits

    def update_device_map(self, device_map):
        return device_map if device_map is not None else {"": torch.cuda.current_device()}

    def _packed(self, model, param_name):
        module, name = get_module_from_name(model, param_name)
        return (name == "weight" and id(module) in self.targets) or name in self.experts.get(id(module), ((), ()))[1]

    def param_element_size(self, model, param_name, param):
        return BITS[self.quantization_config.layout] / 8 if self._packed(model, param_name) else param.element_size()

    def param_needs_quantization(self, model, param_name, **kwargs):
        return self._packed(model, param_name)

    def get_quantize_ops(self):
        return _Pack(self)

    def is_serializable(self):
        return False  # glyd.save_pretrained writes glyd-v1

    @property
    def is_trainable(self):
        return False

    @property
    def is_compileable(self):
        return True  # generate()'s compiled forward (a static cache): GLinear and GEmbedding are ops of its graph

    def _process_model_before_weight_loading(self, model, device_map=None, checkpoint_files=None, **kwargs):
        q = self.quantization_config
        devices = [_cuda(d) for d in device_map.values()] if isinstance(device_map, dict) else [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
        if q.layout == "auto":
            q.layout = gm.auto_layout(model, len(set(devices)), devices[0])[0]
        self.stored = fmt.read_manifest(os.path.dirname(checkpoint_files[0])) if checkpoint_files else None
        if self.stored is None and checkpoint_files and fmt.stored([f for f in checkpoint_files if f.endswith(".safetensors")]):
            raise ValueError(f"glyd: {os.path.dirname(checkpoint_files[0])} holds packs but no {fmt.MANIFEST}: a save cut short; save it again")
        if self.stored is not None:  # a glyd-v1 checkpoint: its packs' buffers load in place of their Linears' weights
            heads = fmt.stored(checkpoint_files)
            for path, e in self.stored["packs"].items():
                if "experts" in e:  # a mixture of experts' weight: its buffers on the module holding it, in its place
                    owner, _, weight = path.rpartition(".")
                    host = _unparam(model, path)
                    for b in fmt.BUFFERS:
                        shape, dtype = heads[fmt.key(owner, b, weight)]
                        host.register_buffer(f"glyd_{weight}_{b}", torch.empty(shape, dtype=DTYPES[dtype], device="meta"))
                    continue
                for member in fmt.members(e)[0]:
                    lin = model.get_submodule(member)
                    del lin.weight  # its matrix comes packed: nothing to load, place or initialize for it
                    lin._is_hf_initialized = True
                    getattr(model, "all_tied_weights_keys", {}).pop(member + ".weight", None)
                host = model.get_submodule(path)
                for b in fmt.BUFFERS:
                    shape, dtype = heads[fmt.key(path, b)]
                    host.register_buffer("glyd_" + b, torch.empty(shape, dtype=DTYPES[dtype], device="meta"))
            self.pre_quantized = True  # its buffers load as stored (a model's fp32 patterns, HunYuan V4's "base", would cast block_base)
            return model
        # A bf16 checkpoint: the Linears whose weights are packed as they arrive, and the groups packed as one. A weight
        # tied to another (an output layer to the embedding) is packed at the end, as it is tied then.
        tied = getattr(model, "all_tied_weights_keys", None) or {}
        tied = set(tied) | set(tied.values())
        # a mixture of experts' layers: each 3-D weight packed as it arrives (one tied: at the end)
        self.experts = {k: (n, ws) for k, (n, ws) in moe.targets(model).items() if not tied & {f"{n}.{w}" for w in ws}}
        paths = {}
        for name, m in model.named_modules():
            paths[id(m)] = name
            if gm.plain(m) and name + ".weight" not in tied and m.weight.shape[0] % 64 == 0 and m.weight.shape[1] % 16 == 0:
                self.targets[id(m)] = [name, None, 0]
        if q.merge and not q.exact:
            for mod, names in gm.groups(model):
                lins = [getattr(mod, c) for c in names]
                if all(id(l) in self.targets for l in lins) and len({l.bias is None for l in lins}) == 1:
                    group = [[paths[id(l)] for l in lins], {}, None]
                    self.groups.append(group)
                    for i, l in enumerate(lins):
                        self.targets[id(l)][1:] = [group, i]
        return model

    def take(self, model, name, w):
        """Tensor `name`, arrived on its GPU as w: packed onto its Linear, or held until its merged group is in (True);
        or not one to pack (False: loaded as it is)."""
        module, attr = get_module_from_name(model, name)
        if w.dtype != torch.bfloat16:  # a weight the model keeps in another dtype (HunYuan V4's output layer, fp32): as it is
            return False
        if attr in self.experts.get(id(module), ((), ()))[1]:
            self.quantization_config.verified += moe.take(module, attr, w, self.quantization_config.layout, self.quantization_config.verify)
            return True
        t = self.targets.get(id(module))
        if t is None or attr != "weight":
            return False
        path, group, i = t
        if group is None:
            module.glyd = self._pack(w, path)
        else:
            group[1][i] = w
            if len(group[1]) == len(group[0]):
                group[2] = self._pack(torch.cat([group[1][j] for j in range(len(group[0]))]), " + ".join(group[0]), len(group[0]))
                group[1].clear()
        # Off the meta device with no bytes, as accelerate places every tensor of a model dispatched over GPUs.
        module.weight = nn.Parameter(w.new_empty(0), requires_grad=False)
        module._is_hf_initialized = True
        return True

    def _pack(self, w, name, tensors=1):
        q = self.quantization_config
        p = gm.pack(w, True, q.layout)
        if q.verify:
            gm.check(p, w, name)
            q.verified += tensors
        return p

    def _rest(self, w, linear):
        """pack_modules' pack for what is left once the model is loaded: embeddings, an output layer tied to one."""
        q = self.quantization_config
        p = gm.pack(w, linear, q.layout)
        if p is not None and q.verify:
            gm.check(p, w, f"a {tuple(w.shape)} {'Linear' if linear else 'embedding'}")
            q.verified += 1
        return p

    def _unstore(self, model, mode):
        """A glyd-v1 checkpoint's packs from their buffers: installed as saved where this load wants them so (tiered,
        its merged groups merged), else decoded and packed again (the 12-bit layout; a group split for exact or
        merge=False). verify: every tensor's sha256 against glyd.json."""
        q = self.quantization_config
        q.source = self.stored.get("source")
        for path, e in self.stored["packs"].items():
            if "experts" in e:
                self._unstore_experts(path, e, model.get_submodule(path.rpartition(".")[0]), path.rpartition(".")[2])
                continue
            host = model.get_submodule(path)
            p = g.Mma(tuple(e["shape"]), host.glyd_data, host.glyd_blocks, host.glyd_block_base, e["tiers"])
            for b in fmt.BUFFERS:
                delattr(host, "glyd_" + b)
            paths, rows = fmt.members(e)
            split = len(paths) > 1 and (q.exact or not q.merge)
            w = gm.unpack(p) if q.verify or split or q.layout != "mma" else None
            if q.verify:
                for t, x in zip(e["tensors"], w.split(rows)):
                    if gm.sha256(x) != t["sha256"]:
                        raise ValueError(f"glyd: {t['name']} decodes to other bytes than glyd.json's sha256")
                    q.verified += 1
            if split:
                for member, x in zip(paths, w.split(rows)):
                    _install(model, [member], gm.pack(x.contiguous(), True, q.layout), mode)
            else:
                _install(model, paths, p if w is None or q.layout == "mma" else gm.pack(w, True, q.layout), mode)
            del w, p

    def _unstore_experts(self, path, e, host, weight):
        """A glyd-v1 checkpoint's experts' weight from its buffers, on the module holding it (moe.py), packed again
        into the 12-bit layout where that is the one; verify: its sha256 against glyd.json."""
        q = self.quantization_config
        _, E, tr = moe.held(host)
        if (e["experts"], e["transposed"]) != (E, weight in tr):  # (a transformers that holds the family otherwise)
            raise ValueError(f"glyd: {path} saved as {e['experts']} experts' matrices{' transposed' if e['transposed'] else ''}; this model holds {E}{' transposed' if weight in tr else ''}")
        p = g.Mma(tuple(e["shape"]), *(getattr(host, f"glyd_{weight}_{b}") for b in fmt.BUFFERS), e["tiers"])
        for b in fmt.BUFFERS:
            delattr(host, f"glyd_{weight}_{b}")
        moe.put(host, weight, p, (tuple(e["tensors"][0]["shape"]), e["transposed"]))
        w = gm.unpack(p) if q.verify or q.layout != "mma" else None
        if q.verify:
            if gm.sha256(moe.decoded(host, p, weight, w)) != e["tensors"][0]["sha256"]:
                raise ValueError(f"glyd: {path} decodes to other bytes than glyd.json's sha256")
            q.verified += 1
        if q.layout != "mma":
            moe.put(host, weight, gm.pack(w, True, q.layout), host.glyd_held[weight])
        setattr(host, weight, nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=p.sm.device), requires_grad=False))
        del w, p

    def _process_model_after_weight_loading(self, model, **kwargs):
        q = self.quantization_config
        mode = {"exact": q.exact}
        with torch.no_grad():
            if self.stored is not None:
                self._unstore(model, mode)
            for paths, _, p in self.groups:
                if p is None:
                    raise ValueError(f"glyd: {', '.join(paths)} not all in the checkpoint")
                _install(model, paths, p, mode)
            for m in list(model.modules()):
                if isinstance(m, nn.Linear) and hasattr(m, "glyd"):
                    _install(model, [self.targets[id(m)][0]], m.glyd, mode)
            gm.pack_modules(model, self._rest, lambda m: m.weight.device, **mode)
            q.verified += moe.rest(model, q.layout, q.verify)  # experts' weights tied to others (saved as they are)
            moe.install(model, q.exact)
            gm.set_scratch(model, q.exact)
        self.targets, self.groups, self.experts = {}, [], {}
        torch.cuda.empty_cache()
        return model
