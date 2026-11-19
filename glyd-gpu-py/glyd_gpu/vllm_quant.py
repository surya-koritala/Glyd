"""Glyd in vLLM for a quantized checkpoint (FP8, NVFP4, MXFP4, INT4 from GPTQ, AWQ or compressed-tensors, INT8): the checkpoint's own
quantization method runs its quantized layers exactly as vLLM runs them without Glyd, and Glyd packs the 16-bit parts the checkpoint
keeps (a Linear it leaves bf16, the embedding, the LM head, a vision tower's Linears where vLLM gives the tower the config), by the
rules and with the products of a bf16 model's (vllm_plugin.py: GlydLinearMethod, GlydEmbeddingMethod; the options layout, exact, verify
and fraction as there).

    vllm serve nvidia/Gemma-4-31B-IT-NVFP4 --quantization glyd                       # its bf16 attention, embedding and LM head packed
    vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --quantization glyd --enforce-eager --additional-config '{"glyd": {"exact": true}}'

How vLLM gets here: its probe of the quantization methods (ModelConfig._verify_quantization) asks a plugin's method first, so
--quantization glyd on a checkpoint with its own quantization_config reaches GlydConfig.override_quantization_method, which claims it
(claims below) instead of vLLM's "does not match" refusal; vLLM then builds the config with GlydConfig.from_config(the checkpoint's
quantization_config), which returns config(): vLLM's own config class for the checkpoint (the one its probe picks without
--quantization: inner_class), made one class lower, a subclass with GlydQuantConfig in front (as vllm_fp8.py does for FP8), so that
every isinstance check and get_name() of vLLM's holds and its from_config, ignore lists, packed-module mapping, KV-cache scales and
apply_vllm_mapper are its own. Its get_quant_method is the inner config's; where that gives a layer vLLM's own 16-bit method
(UnquantizedLinearMethod for a Linear, None or UnquantizedEmbeddingMethod for an embedding or an LM head, and UnquantizedLinearMethod
for an LM head ModelOpt leaves 16-bit), the method is a GlydConfig's (glyd, made with this config), as for that layer of a bf16 model:
packed where its rules pack it, vLLM's own where they keep it. Attention layers and the KV cache, and a mixture of experts' experts, are
the checkpoint's (experts a checkpoint leaves 16-bit stay vLLM's: none of the 2026 checkpoints measured has them).

What differs from a bf16 model under --quantization glyd: kv is off (the lossless KV cache is not held with a checkpoint's own KV-cache
scales: kv lossless refused, auto off); layout auto is the tiered layout on every GPU, and no fit check is made (which Linears a quantized
checkpoint keeps 16-bit is not told by its config, so their bytes are not estimated; the tiered layout is the smallest, and on the RTX 5090
the 12-bit one kept 98.4% of gemma-4-12B QAT's embedding); the shared experts' side stream is left as vLLM has it (a Glyd product there
never runs beside another: the routed experts beside it are the checkpoint's); the model runs in bf16 only (get_supported_act_dtypes: an
fp16 checkpoint is refused by vLLM, as a bf16 path's; with --dtype bfloat16 vLLM rounds its fp16 parts to bf16 before Glyd packs them, so
the checkpoint's own fp16 bits are not what is kept).
The method vLLM's own config is made for: the one stock vLLM gives the checkpoint (_method): its probe with the model's hf_config (read from
weight_utils.get_quant_config's model_config, the model being configured, in the engine's process or a worker's), and for a speculative
draft in the target's checkpoint (MTP, DSpark: get_draft_quant_config) the target's, as vLLM aligns a draft's quantization with its
target's. A ModelOpt mixed-precision checkpoint whose config.json holds no quantized_layers is read from its hf_quant_config.json, as
stock vLLM reads it for modelopt_mixed (_own).
ModelConfig.is_nvfp4_quantized answers for the method vLLM would pick (register), so an NVFP4 checkpoint keeps the compile passes vLLM
turns on for it. Where GLYD_FP8=1 (a development build) and the checkpoint is an FP8 one vllm_fp8.py takes, that path is taken instead.
What vLLM builds without the config is a bf16 model's case too (vllm_plugin._after_load): Qwen3.5's embedding packed once the model has loaded;
a vision tower vLLM gives no config (Gemma 4's, but in a compressed-tensors checkpoint) stays as vLLM runs it.

vLLM's internals it relies on, beyond vllm_plugin.py's: QUANTIZATION_METHODS, get_quantization_config and the override_quantization_method
of vLLM's methods, in the order of ModelConfig._verify_quantization's overrides (OVERRIDES); the names and locals of weight_utils.get_quant_config
(model_config, load_config) and models.utils.get_draft_quant_config (vllm_config); the config classes' from_config and their
objects' state (copied into the subclass), get_quant_method, maybe_update_config, get_min_capability, get_supported_act_dtypes;
UnquantizedLinearMethod, UnquantizedEmbeddingMethod; ModelConfig.is_nvfp4_quantized and model_arch_config.quantization_config.
It copies none of vLLM's code. Under the Business Source License 1.1 (LICENSE), as the rest of Glyd's GPU code."""
import copy
import sys

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding

log = init_logger("vllm.glyd")
NAME = "glyd"
# vLLM 0.30's own order of the quantization methods that can claim a checkpoint (config/model.py, _verify_quantization's overrides)
OVERRIDES = ("auto_gptq", "gptq", "gptq_marlin", "auto_awq", "awq", "awq_marlin", "inc", "moe_wna16", "modelopt", "modelopt_fp4", "modelopt_mxfp8", "mxfp8",
             "modelopt_mixed", "mxfp4", "gpt_oss_mxfp4", "deepseek_v4_fp8", "humming")
# the methods (vLLM's names for them) whose checkpoints' 16-bit parts Glyd packs: their loaders and get_quant_method leave a layer they do not quantize to
# vLLM's own 16-bit methods
COVERED = ("fp8", "modelopt", "modelopt_fp4", "modelopt_mixed", "modelopt_mxfp8", "mxfp4", "compressed-tensors", "auto_gptq", "auto_awq", "moe_wna16", "deepseek_v4_fp8")
_LOADER, _MODELS = "vllm.model_executor.model_loader.weight_utils", "vllm.model_executor.models.utils"  # (where vLLM makes a quantization config)
_SUBS = {}  # vLLM's class -> the subclass with GlydQuantConfig in front, made once


def inner_name(cfg, hf_config=None):
    """The quantization method vLLM picks for a checkpoint whose quantization_config is cfg, without --quantization, by its registered name (some of
    vLLM's classes answer get_name() on an object only): the first of its overriding methods that claims cfg, in vLLM's order, else cfg's own quant_method."""
    for name in OVERRIDES:
        try:
            cls = get_quantization_config(name)
        except (ValueError, ImportError):  # (a method this vLLM does not have)
            continue
        got = cls.override_quantization_method(cfg, None, hf_config=hf_config)
        if got:
            return got
    return str(cfg.get("quant_method") or "").strip().lower()


def _covered():
    """vLLM's config classes of COVERED's methods (those this vLLM has)."""
    out = set()
    for name in COVERED:
        try:
            out.add(get_quantization_config(name))
        except (ValueError, ImportError):
            pass
    return out


def inner_class(cfg, hf_config=None):
    """The quantization config class vLLM picks for that checkpoint (inner_name's)."""
    return get_quantization_config(inner_name(cfg, hf_config))


def claims(cfg, hf_config=None):
    """Whether cfg (a checkpoint's quantization_config) is a quantized checkpoint's that --quantization glyd serves here (True); refused, with why,
    where vLLM serves it with a method Glyd does not cover (ValueError); False where it is none (a bf16 checkpoint's options, a glyd save's)."""
    if not isinstance(cfg, dict):
        return False
    qm = str(cfg.get("quant_method") or "").strip().lower()
    if qm in ("", NAME):
        return False
    try:
        name = inner_name(cfg, hf_config)
        cls = get_quantization_config(name)
    except ValueError as e:  # (a quant_method vLLM has no method for)
        raise ValueError(f"glyd: this checkpoint's quantization ({qm}) is not one vLLM serves: {e}") from e
    if cls not in _covered():  # (by class: vLLM registers some classes under several names, MiniMax's "mxfp8" for ModelOpt's MXFP8)
        raise ValueError(f"glyd: a {name} checkpoint: Glyd packs the 16-bit parts of {', '.join(COVERED)} checkpoints, and bf16 checkpoints; serve this one without --quantization glyd")
    return True


class GlydQuantConfig:
    """vLLM's quantization config for a quantized checkpoint with Glyd in front of it (config(): a subclass of vLLM's own class, this one first):
    the checkpoint's methods for its quantized layers, a GlydConfig's (glyd) for the layers it leaves 16-bit."""

    glyd_inner = None  # vLLM's class this one is made from (set on the subclass)
    glyd = None  # the GlydConfig of the 16-bit parts (config())

    @classmethod
    def get_min_capability(cls):
        return max(80, super(GlydQuantConfig, cls).get_min_capability())  # (Glyd's library: Ampere on)

    def get_supported_act_dtypes(self):
        return [d for d in super(GlydQuantConfig, self).get_supported_act_dtypes() if d == torch.bfloat16]  # (Glyd's layouts are bf16's)

    def get_quant_method(self, layer, prefix):
        m = super(GlydQuantConfig, self).get_quant_method(layer, prefix)
        if not sixteen_bit(layer, m):
            return m
        if not self.__dict__.get("glyd_noted"):
            self.glyd_noted = True
            log.info("glyd: a %s checkpoint: its quantized layers as vLLM runs them, its 16-bit Linears, embedding and LM head packed where Glyd packs a bf16 model's", self.get_name())
        g = self.glyd.get_quant_method(layer, prefix)
        return m if g is None or type(g) is UnquantizedLinearMethod else g

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        """vLLM's own, then what only the VllmConfig under construction can say (in the engine's process): the options (kv off: refused where
        lossless is asked for) and what Glyd refuses there."""
        super(GlydQuantConfig, self).maybe_update_config(model_name, hf_config=hf_config, revision=revision)
        from .vllm_plugin import KEY, _building_config, _env, _options, _refusals, _warn_no_nvcc

        g = self.glyd
        g.hf_config = hf_config
        vc = _building_config()
        if vc is None or getattr(vc.model_config, "hf_config", None) is not hf_config or not isinstance(vc.additional_config, dict):
            return  # (a draft model's config, or none being built)
        o = _options(vc.additional_config.get(KEY) or {}, g.given, _env())
        if o[4] == "lossless":
            raise ValueError("glyd: kv lossless is not held for a quantized checkpoint (its KV cache stays vLLM's, with the checkpoint's own scales): leave kv auto or off")
        vc.additional_config[KEY] = {**(vc.additional_config.get(KEY) or {}), "kv": "off"}  # (the workers read it: vLLM's own KV cache, whatever their environment says)
        _refusals(vc, o[1], None, o[3])
        _warn_no_nvcc()

    def __reduce__(self):  # (vLLM pickles its config for its engine's and workers' processes, and deep-copies it; the class made here is not importable by name)
        return _restore, (type(self).glyd_inner, dict(self.__dict__))


def sixteen_bit(layer, m):
    """Whether m, vLLM's method for layer under the checkpoint's config, is vLLM's own 16-bit one for a Linear, an embedding or an LM head."""
    if isinstance(layer, VocabParallelEmbedding):  # (an LM head ModelOpt leaves 16-bit gets UnquantizedLinearMethod)
        return m is None or type(m) in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
    return isinstance(layer, LinearBase) and (m is None or type(m) is UnquantizedLinearMethod)


def _subclass(inner):
    cls = _SUBS.get(inner)
    if cls is None:
        cls = _SUBS[inner] = type(inner)(f"GlydQuant{inner.__name__}", (GlydQuantConfig, inner), {"glyd_inner": inner, "__module__": __name__})
    return cls


def _restore(inner, state):
    """An object of the subclass of inner (vLLM's class) with the given state: from vLLM's own object (config()), unpickled or copied."""
    cls = _subclass(inner)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


def _frame(module, fn):
    """The locals of the nearest calling frame of vLLM's function fn of that module, or {}."""
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_name == fn and f.f_globals.get("__name__") == module:
            return f.f_locals
        f = f.f_back
    return {}


def _method(cfg):
    """The quantization method (vLLM's registered name) stock vLLM gives the checkpoint whose quantization_config is cfg, where vLLM makes its
    config: a speculative draft in the target's checkpoint (MTP, DSpark: get_draft_quant_config, in a worker) the target's, as vLLM aligns a
    draft's quantization with its target's; else vLLM's probe with the hf_config of the model being configured (get_quant_config's
    model_config: some methods are picked by its model_type, deepseek_v4_fp8 by deepseek_v4), or of the VllmConfig being built."""
    from .vllm_plugin import _building_config

    vc = _frame(_MODELS, "get_draft_quant_config").get("vllm_config")
    if vc is not None and isinstance(getattr(vc, "quant_config", None), GlydQuantConfig) and vc.quant_config.__dict__.get("glyd_method"):
        d = getattr(getattr(vc, "speculative_config", None), "draft_model_config", None)
        if getattr(d, "model", None) == getattr(vc.model_config, "model", None):
            return vc.quant_config.glyd_method
    mc = _frame(_LOADER, "get_quant_config").get("model_config") or getattr(_building_config(), "model_config", None)
    return inner_name(cfg, getattr(mc, "hf_config", None))


def _own(name, cfg):
    """vLLM's own config object for the checkpoint under the method name, as stock vLLM makes it: from cfg; for a ModelOpt mixed-precision
    checkpoint whose config.json holds no quantized_layers, from its hf_quant_config.json, by vLLM's own get_quant_config run for modelopt_mixed
    on a copy of the model's config (the file stock vLLM reads; the --quantization glyd one it made skips it)."""
    inner = get_quantization_config(name)
    if name != "modelopt_mixed" or "quantized_layers" in cfg:
        return inner.from_config(cfg)
    at = _frame(_LOADER, "get_quant_config")
    if at.get("model_config") is None or at.get("load_config") is None:
        raise ValueError("glyd: a ModelOpt mixed-precision checkpoint without quantized_layers in its config.json: its hf_quant_config.json is read only where vLLM loads the model's quantization config")
    from vllm.model_executor.model_loader.weight_utils import get_quant_config

    mc = copy.copy(at["model_config"])
    object.__setattr__(mc, "quantization", name)
    return get_quant_config(mc, at["load_config"])


def config(cfg):
    """vLLM's own quantization config for the checkpoint whose quantization_config is cfg (the method stock vLLM gives it: _method, _own), with
    Glyd in front of it and a GlydConfig of its own (its options from --additional-config's "glyd" or the environment, resolved where the model
    is made)."""
    from .vllm_plugin import GlydConfig

    name = _method(cfg)
    inner = get_quantization_config(name)
    if inner not in _covered():  # (claims refuses it where vLLM probes the checkpoint; here too, for a config made without that probe)
        raise ValueError(f"glyd: a {name} checkpoint: Glyd packs the 16-bit parts of {', '.join(COVERED)} checkpoints, and bf16 checkpoints; serve this one without --quantization glyd")
    obj = _restore(inner, vars(_own(name, cfg)))
    obj.glyd_method = name
    obj.glyd = GlydConfig()
    obj.glyd.quantized = True
    return obj


def register():
    """ModelConfig.is_nvfp4_quantized as vLLM answers it for the method it picks without --quantization glyd (it reads the method's name, which is
    glyd here): an NVFP4 checkpoint keeps the activation-quantization fusion vLLM's compile turns on for it. Once a process."""
    from vllm.config.model import ModelConfig

    own = ModelConfig.is_nvfp4_quantized
    if getattr(own, "glyd", False):
        return

    def is_nvfp4_quantized(self):
        cfg = getattr(getattr(self, "model_arch_config", None), "quantization_config", None)
        if self.quantization != NAME or not isinstance(cfg, dict) or str(cfg.get("quant_method") or "").lower() in ("", NAME):
            return own(self)
        object.__setattr__(self, "quantization", inner_name(cfg, getattr(self, "hf_config", None)))
        try:
            return own(self)
        finally:
            object.__setattr__(self, "quantization", NAME)

    is_nvfp4_quantized.glyd = True
    ModelConfig.is_nvfp4_quantized = is_nvfp4_quantized
