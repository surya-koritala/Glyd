"""Glyd in vLLM for an FP8 checkpoint's Linears and experts (exact): the weights vLLM serves FP8 with, held packed by a codec
(f8codec.py: the lossless FP8 format, about 6.7 bits a value) on the GPU and decoded, bit for bit, into one scratch before each
product, which vLLM's own FP8 kernel then runs as it would on the weights it loaded.

    GLYD_FP8=1 vllm serve Qwen/Qwen3.8-27B-FP8 --quantization glyd   # the checkpoint's own quantization_config says fp8
    GLYD_FP8=1 vllm serve ./mistral-small-4 --quantization glyd      # per-tensor FP8
    GLYD_FP8=1 vllm serve ./minimax-m3-mxfp8 --quantization glyd     # mxfp8

A development build only: GLYD_FP8=1 is its opt-in (vllm_plugin._fp8_on); a release build reads no such variable and serves no FP8 checkpoint.

--quantization glyd on a checkpoint whose quantization_config is one of the FP8 ones is claimed by GlydConfig
(override_quantization_method) and built here: vLLM's own quantization config for the checkpoint (the class vLLM would pick
without --quantization: Fp8Config, ModelOptFp8Config, ModelOptMxFp8Config, CompressedTensorsConfig, the DeepSeek V4 one;
from_config, ignored layers, block sizes, KV-cache scales, packed-module mapping, apply_vllm_mapper all its own), made one
class lower: a subclass, GlydFp8Config in front of it, so that every isinstance check of vLLM's (a model's own FP8
special cases) holds. Its get_quant_method is the inner config's, with each FP8 Linear's method wrapped
(GlydFp8LinearMethod). The wrapper is vLLM's method until the layer is loaded: create_weights and the weight loaders are vLLM's,
and so are the process_weights_after_loading and the scales. After it the weight vLLM's kernel will be fed -- layer.weight as
the kernel left it: [N, K] e4m3 (Ada's Triton or Hopper's CUTLASS block kernels; DeepGEMM's, whose bytes it requantized), a
transposed view [K, N] (per-tensor, ModelOpt FP8, compressed-tensors: CUTLASS), Marlin's int32 tiles (block FP8 on Ada,
every MXFP8), with their strides and dtype -- is read as the bytes of its storage, packed with f8codec, and its memory
freed. The layer's weight becomes a view of a scratch, one byte buffer a device, shared by every packed Linear and made
at load (so that vLLM's memory profile counts it), grown to the largest layer's size as the layers are processed (a CUDA graph
holds its pointer: nothing may grow it once the model has run). A product is one op, glyd::f8_linear(x, bias, layer): the
decode into the scratch, then the inner method's own apply, which reads layer.weight, the scales and the rest from the layer as
it does on the weights it loaded. One op, so that the scratch's one writer and its reader cannot be put out of order by vLLM's
torch.compile, whatever the model's graph: it holds a decode and its product, which share a stream, and it is as capturable as
they are (no host synchronization, no allocation but the product's own, pointers that do not change). Its cost: what vLLM's
compile passes fuse into a Linear's input quantization stays out (each Linear takes the activation as it is: no pre-quantized
activation, layer.requires_unquantized_input), and a Linear's activation quantization (per-tensor and per-channel: torch ops, which inductor
fuses into stock's graph; block-scaled: vLLM's quantization op, on by default for blocked weights) runs unfused inside the op, and so does
what vLLM's norm and activation fusion passes fuse it with: the logits are then stock's compiled ones to rounding, not bit for bit
(RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic on an L40S, compiled with custom_ops +quant_fp8, the fusion passes on: they differ). exact is kept
for per-tensor and per-channel checkpoints eager, and compiled with custom_ops +quant_fp8 and the passes fuse_norm_quant and fuse_act_quant off
(compiled_exact: refused at config time otherwise). Under exact, what has not been compared with stock vLLM is refused, with why: an MXFP8
checkpoint (config time), a Linear whose kernel is one of vLLM's block-scaled W8A8 ones (Fp8BlockScaledMMLinearKernel's: CUTLASS's on the
RTX 50 series and Hopper, Triton, DeepGEMM, FlashInfer; load time, _kernel), ModelOpt FP8 experts (load time). Measured bit for bit
(benchmarks/gpu/fp8-in-vllm-2026-10-06; compiled with inductor's deterministic mode and CUDA graphs): block-scaled Linears under Marlin
(weight-only) on Ada (Qwen3.8-27B-FP8, Qwen3.6-35B-A3B-FP8 on an L40S), compressed-tensors W8A8 FP8 Linears under CUTLASS's scaled_mm and
the Fp8 and compressed-tensors methods' experts with the fusions off (gemma-4-26B-A4B-it-FP8-dynamic, Nemotron-3.5-Lightning FP8 on an L40S;
gemma also on an RTX 5090); per-tensor fp8 and ModelOpt FP8 Linears and Hopper are not measured.
Outside exact, a development build's GLYD_F8_FUSED_MAX=N gives steps of up to N tokens f8_linear's product from the pack instead (the weight
never written out), only where the pack is the row-major weight [N, K] (not Marlin's tiles) with one or 128 x 128 fp32 scales and no static
activation scale: f8_linear does not quantize the activations (W8A16), so its logits are not a W8A8 checkpoint's numerics; not measured end
to end in vLLM, and off in every other build.

A mixture of experts' experts (vLLM's Fp8MoEMethod, CompressedTensorsW8A8Fp8MoEMethod, ModelOptFp8MoEMethod) are packed the same
way once vLLM has processed them: w13 and w2 each the bytes of its E matrices as vLLM's kernel is fed them, stacked; the layer's
weights become views of the device's experts buffer (one, apart from the Linears' scratch, the size of the largest layer's
experts), into which each call decodes the experts its tokens are routed to (f8_unpack_experts: one launch, the ids read on the
GPU) before vLLM's own apply runs its kernel there. The experts no token is routed to are not decoded and not read.

Refused at config time, with why: a GPU that is not Ada (sm_89), Hopper (sm_90) or an RTX 50 (sm_120) (Ampere serves FP8 weight-only; not covered),
FP8-like quantization_configs that are not covered (compressed-tensors with another weight type or no activation
quantization, ModelOpt NVFP4, FBGEMM FP8 and Quark; a ModelOpt checkpoint whose quantization is in hf_quant_config.json alone),
LoRA, weight offloading, sleep mode, dual-batch overlap, options of the bf16 path (layout, verify ...), `exact` under inductor on a per-tensor or
per-channel checkpoint unless custom_ops +quant_fp8 and the norm and activation quantization fusions are off, `exact` where not measured (above), a
Linear whose scheme is not FP8; the attention
methods of the checkpoint are vLLM's own (and a mixture of experts' shared experts' side stream is turned off), and so is an MLA
attention's kv_b_proj, whose weight vLLM reads directly after the load, and so are experts under expert parallel or a kernel that
routes the tokens itself (a line says so).

vLLM's internals it relies on, beyond those of vllm_plugin.py: QuantizationConfig's subclasses (from_config,
get_quant_method, override_quantization_method, get_min_capability, maybe_update_config, packed_modules_mapping) through
their classes, which GlydFp8Config is put before; Fp8LinearMethod, ModelOptLinearMethod and CompressedTensorsLinearMethod
(with the scheme's name: CompressedTensorsW8A8Fp8, CompressedTensorsW8A8Mxfp8) as the methods of an FP8 Linear; LinearBase's
requires_unquantized_input and its input_size, output_size and prefix; FusedMoEMethodBase's FP8 methods (their moe config, is_monolithic,
process_weights_after_loading and apply, which reads layer.w13_weight and layer.w2_weight), RoutedExperts' w13_weight, w2_weight and layer_name; WEIGHT_LOADER_V2_SUPPORTED and
register_weight_loader_v2_supported_method (the wrapper's class name is looked up there); layer.weight, as a Parameter whose
data is swapped (the layer after process_weights_after_loading), layer.input_size_per_partition and the quantization method's
own attributes, read through the wrapper (use_marlin, block_quant, fp8_linear ...); the kernel object a Linear's method holds (Fp8LinearMethod's and a
compressed-tensors scheme's fp8_linear, ModelOptLinearMethod's kernel) and the name of vLLM's base class Fp8BlockScaledMMLinearKernel; the VllmConfig under construction
(vllm_plugin._building_config: its additional_config keys vLLM's compile cache; its parallel, lora, offload and model configs).
It copies none of vLLM's code. Under the Apache License 2.0 (LICENSE), as the rest of Glyd."""
import json
import os
import weakref

import torch
from vllm.config import CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED, LinearBase, LinearMethodBase, UnquantizedLinearMethod, register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.platforms import current_platform

from . import __version__, _trace, f8codec

log = init_logger("vllm.glyd")
KEY = "glyd"  # vLLM's additional_config key (as vllm_plugin.KEY): the options in effect, which vLLM's compile cache keys by
CAPS = ((8, 9), (9, 0), (12, 0))  # the GPUs, by compute capability: Ada, Hopper and Blackwell's RTX 50 series, FP8 tensor cores that vLLM's FP8 kernels run on
OPTIONS = ("f8_codec", "exact")  # --additional-config's "glyd", for an FP8 checkpoint (layout, verify, fraction and kv are the bf16 path's)
OURS = ("glyd", "f8codec")  # what this module writes there
SMALL_LINEAR = 1 << 20  # a Linear of fewer weights than this stays as vLLM runs it (a router's gate: packing it saves nothing)
# vLLM's own order of the quantization methods whose override_quantization_method can claim an FP8 checkpoint (config/model.py: the `overrides` list)
OVERRIDES = ("modelopt", "modelopt_fp4", "modelopt_mxfp8", "mxfp8", "modelopt_mixed", "deepseek_v4_fp8")
REFUSED = {  # quant_method: why not (FP8 checkpoints that are not covered)
    "fbgemm_fp8": "FBGEMM FP8 is not covered (its method is not vLLM's FP8 one)",
    "quark": "Quark checkpoints are not covered",
    "fp_quant": "FP-Quant checkpoints are not covered",
    "ptpc_fp8": "PTPC FP8 (ROCm) is not covered",
}
# The Linears whose weight vLLM reads directly once the load is done, not through the product: an MLA attention's kv_b_proj (its post-load step dequantizes the
# weight with the scales where its method is vLLM's Fp8LinearMethod, else by multiplying an identity matrix, which its own comment says a static activation scale
# spoils), ~16 MB a layer: left vLLM's own, as stock reads it.
DIRECT = ("kv_b_proj",)
CT_METHODS = ("CompressedTensorsW8A8Fp8", "CompressedTensorsW8A8Mxfp8")  # a compressed-tensors Linear's scheme, by class name
MOE_METHODS = ("Fp8MoEMethod", "CompressedTensorsW8A8Fp8MoEMethod", "ModelOptFp8MoEMethod")  # vLLM's FP8 methods of a mixture of experts' experts, by class name
_SCRATCH = {}  # device -> _Scratch
_EXPERTS = {}  # device -> _Scratch: the experts' one buffer, apart from the Linears' (a shared expert's Linear runs inside the experts' op)
_REG = {}  # layer id (its prefix) -> _Rec: what the op reads (a layer, its method, its pack); gone with its layer
_HF = {}  # a claimed checkpoint's config (its JSON) -> the model's hf_config, from the override that saw both (from_config is given the dict alone)


def scheme(cfg):
    """The FP8 scheme a checkpoint's quantization_config (a dict) is (fp8-block, fp8-tensor, mxfp8, modelopt-fp8, modelopt-mxfp8, ct-fp8, ct-mxfp8); None where it is not an FP8 one (a bf16
    checkpoint's, a glyd save's, GPTQ's: not this module's); ValueError, with why, where it is FP8 but not covered."""
    if not isinstance(cfg, dict):
        return None
    qm = str(cfg.get("quant_method") or "").strip().lower()
    if qm in ("", "glyd"):
        return None
    if qm == "fp8":
        return "fp8-block" if cfg.get("weight_block_size") else "fp8-tensor"
    if qm == "mxfp8":
        return "mxfp8"
    if qm.startswith("modelopt"):
        q = cfg.get("quantization") if isinstance(cfg.get("quantization"), dict) else cfg
        algo = str(q.get("quant_algo") or "").upper()
        if "MXFP8" in algo:
            return "modelopt-mxfp8"
        if algo in ("FP8", "FP8_PER_CHANNEL_PER_TOKEN", "FP8_PB_WO"):
            return "modelopt-fp8"
        raise ValueError(f"glyd: a ModelOpt {algo or 'unknown'} checkpoint: Glyd packs ModelOpt FP8 (FP8, FP8_PER_CHANNEL_PER_TOKEN, FP8_PB_WO) and MXFP8, and bf16 checkpoints")
    if qm == "compressed-tensors":
        return _ct_scheme(cfg)
    if qm in REFUSED:
        raise ValueError(f"glyd: {REFUSED[qm]}: Glyd packs FP8 checkpoints of quant_method fp8, mxfp8, modelopt (FP8) and compressed-tensors (W8A8 FP8), and bf16 checkpoints")
    return None


def _ct_scheme(cfg):
    """A compressed-tensors checkpoint's scheme (by its config groups' weights): ct-fp8 (8-bit float weights and activations:
    tensor, channel or block scales), ct-mxfp8 (groups of 32); refused where any group is another (integers, 4 bits, FP8
    weights with the activations left 16-bit, which vLLM runs another way) or the groups mix the two."""
    groups = cfg.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("glyd: a compressed-tensors checkpoint with no config_groups: Glyd packs compressed-tensors checkpoints whose weights are all 8-bit float")
    kinds = set()
    for name, g in groups.items():
        w, a = (g or {}).get("weights") or {}, (g or {}).get("input_activations")
        if not (w.get("type") == "float" and w.get("num_bits") == 8):
            raise ValueError(f"glyd: compressed-tensors config group {name!r} holds {w.get('num_bits')}-bit {w.get('type')} weights: Glyd packs the 8-bit float ones (W8A8 FP8, MXFP8), where all of them are")
        if w.get("strategy") == "group" and w.get("group_size") == 32:
            kinds.add("ct-mxfp8")
        elif a and a.get("type") == "float" and a.get("num_bits") == 8:
            kinds.add("ct-fp8")
        else:
            raise ValueError(f"glyd: compressed-tensors config group {name!r} quantizes the weights to FP8 but not the activations (W8A16: vLLM runs it with Marlin, weight only): not covered")
    if len(kinds) > 1:
        raise ValueError("glyd: a compressed-tensors checkpoint mixing FP8 and MXFP8 groups is not covered")
    return kinds.pop()


def native_quant(cfg):
    """Whether an FP8 checkpoint's Linears quantize their activations with vLLM's torch ops, not its quantization op: those with per-tensor or per-channel weights
    (a block-scaled Linear's activations vLLM quantizes with its op, which VllmConfig turns on for blocked weights; MXFP8's are not quantized at all)."""
    s = scheme(cfg)
    if s == "fp8-tensor":
        return True
    if s == "ct-fp8":
        return any(((g or {}).get("weights") or {}).get("strategy") in ("tensor", "channel") for g in cfg["config_groups"].values())
    return s == "modelopt-fp8" and str((cfg.get("quantization") if isinstance(cfg.get("quantization"), dict) else cfg).get("quant_algo")).upper() != "FP8_PB_WO"


def compiled_exact(vc):
    """exact on a per-tensor or per-channel checkpoint, refused with why, where vLLM compiles with inductor and the activation quantization is not a
    step of its own in stock's graph (vc: vLLM's config): with vLLM's quantization op off, inductor fuses its torch ops into the graph; with it on,
    vLLM's norm and activation fusion passes (on by default with it) fuse it with the RMSNorm or the activation before it. Glyd's product quantizes
    the activation inside its op."""
    mc, cc = vc.model_config, vc.compilation_config
    if getattr(mc, "enforce_eager", False) or cc.mode == CompilationMode.NONE:
        return
    pc = getattr(cc, "pass_config", None)
    if "+quant_fp8" in (cc.custom_ops or []) and getattr(pc, "fuse_norm_quant", None) is False and getattr(pc, "fuse_act_quant", None) is False:
        return
    raise ValueError("glyd: exact on an FP8 checkpoint whose Linears quantize their activations with per-tensor or per-channel scales "
                     "is vLLM FP8's bits only where that quantization is a step of its own, as in Glyd's product: "
                     "under torch.compile vLLM fuses it with the RMSNorm or the activation before it (inductor does, where vLLM's quantization op is off), so the last bit of a rounding moves. "
                     "Serve with --enforce-eager, or with --compilation-config '{\"custom_ops\": [\"+quant_fp8\"], \"pass_config\": {\"fuse_norm_quant\": false, \"fuse_act_quant\": false}}' "
                     "(the run exact is compared with too), or leave exact off")


def modelopt_file(model_name, revision=None):
    """Whether a checkpoint (a directory, or a Hub repo) has ModelOpt's hf_quant_config.json: its quantization is there, and vLLM reads it from there for
    modelopt alone."""
    if os.path.isdir(model_name):
        return os.path.isfile(os.path.join(model_name, "hf_quant_config.json"))
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_name, "hf_quant_config.json", revision=revision)
        return True
    except Exception:  # (no such file in the repo, not a repo, no network and none cached: not found)
        return False


def check_gpu():
    """Refused where the GPU is not one of CAPS (a known one: where vLLM's platform cannot tell, it goes on)."""
    cap = current_platform.get_device_capability()
    if cap is not None and (cap.major, cap.minor) not in CAPS:
        raise ValueError(f"glyd: an FP8 checkpoint is served on Ada (sm_89), Hopper (sm_90) and RTX 50 series (sm_120) GPUs, the ones vLLM's FP8 kernels run on in hardware; this GPU is sm_{cap.major}{cap.minor} (on Ampere vLLM serves FP8 weight-only, which Glyd does not cover yet, and B200/B300 are not measured)")


def claims(cfg, hf_config=None):
    """Whether cfg (a checkpoint's quantization_config) is an FP8 one that --quantization glyd serves (True); refused, with why,
    where it is FP8 but not covered or the GPU is not (ValueError); False where it is not an FP8 checkpoint's."""
    if scheme(cfg) is None:
        return False
    check_gpu()
    if hf_config is not None:
        _HF[_key(cfg)] = hf_config
    return True


def _key(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def inner_class(cfg, hf_config=None):
    """The quantization config class vLLM would pick for this checkpoint without --quantization (config/model.py's probe of the
    methods that override, in its order, else the checkpoint's own quant_method): vLLM's own for it, which GlydFp8Config is put before."""
    for name in OVERRIDES:
        try:
            cls = get_quantization_config(name)
        except Exception:  # (a method this vLLM does not have)
            continue
        picked = cls.override_quantization_method(cfg, None, hf_config=hf_config)
        if picked:
            return get_quantization_config(picked)
    return get_quantization_config(str(cfg["quant_method"]))


class GlydFp8Config:
    """vLLM's quantization config for an FP8 checkpoint with Glyd in front of it (made by config(): a subclass of vLLM's own class
    for the checkpoint, this one first): its get_quant_method wraps each FP8 Linear's method, the rest is vLLM's."""

    glyd_inner = None  # the class this one is made from (set on the subclass)
    glyd_scheme, glyd_codec, glyd_native_quant, glyd_exact = None, None, False, False  # the scheme (scheme()'s), the codec's name (None: DEFAULT's), whether the checkpoint's activation quantization is vLLM's torch ops (native_quant), exact

    @classmethod
    def get_min_capability(cls):
        return max(89, super(GlydFp8Config, cls).get_min_capability())

    def get_quant_method(self, layer, prefix):
        m = super(GlydFp8Config, self).get_quant_method(layer, prefix)
        w = wrap(self, m, layer, prefix)
        if w is not m:
            log.info_once("glyd: an FP8 checkpoint (%s): its Linears held packed (codec %s), vLLM's own products", self.glyd_scheme, w.codec)
            return w
        w = wrap_experts(self, m, prefix)
        if w is not m:
            log.info_once("glyd: an FP8 checkpoint (%s): its experts held packed (codec %s), the routed ones decoded before vLLM's own kernel runs them", self.glyd_scheme, w.glyd_codec)
        return w

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        """vLLM's own, then what only the VllmConfig under construction can say: the options in effect (the codec), and the refusals."""
        super(GlydFp8Config, self).maybe_update_config(model_name, hf_config=hf_config, revision=revision)
        from .vllm_plugin import _building_config, _flag, _has_experts

        vc = _building_config()
        if vc is not None and getattr(vc.model_config, "hf_config", None) is hf_config:  # (the model's own: not a draft's)
            refusals(vc)
            ac = vc.additional_config
            extra = {k: v for k, v in (ac.get(KEY) or {}).items() if k not in OURS}
            bad = [k for k in extra if k not in OPTIONS]
            if bad:
                raise ValueError(f"glyd: --additional-config's \"glyd\": {', '.join(map(repr, bad))}: not an option of an FP8 checkpoint ({', '.join(OPTIONS)}); layout, verify, fraction and kv are the bf16 path's")
            c = f8codec.get(extra.get("f8_codec") or self.glyd_codec or None)
            self.glyd_codec = c.name
            exact = _flag("exact", extra.get("exact", os.environ.get("GLYD_EXACT")) or False)
            if exact and "mxfp8" in (self.glyd_scheme or ""):
                raise ValueError(f"glyd: exact on an MXFP8 checkpoint ({self.glyd_scheme}) is not measured against stock vLLM yet: serve with exact off")
            if exact and self.glyd_native_quant:
                compiled_exact(vc)
            self.glyd_exact = exact
            if _has_experts(hf_config):  # (a mixture of experts' shared experts run on a side stream beside the routed ones: no wrapped Linear may run there while another does)
                os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
            ac[KEY] = {**extra, "glyd": __version__, "f8codec": f"{c.name}-{c.version}", "exact": exact}  # (vLLM's compile cache keys by it: another codec never finds another's graphs)

    def __reduce__(self):  # (vLLM pickles its config to start its engine's process; the class made here is not importable by name)
        return _restore, (type(self).glyd_inner, dict(self.__dict__))


def _subclass(inner):
    """GlydFp8Config in front of the class inner (vLLM's own), made once."""
    cls = _SUBS.get(inner)
    if cls is None:
        cls = _SUBS[inner] = type(inner)(f"Glyd{inner.__name__}", (GlydFp8Config, inner), {"glyd_inner": inner, "__module__": __name__})
    return cls


_SUBS = {}


def _restore(inner, state):
    """An object of the subclass of inner (vLLM's class) with the given state: from vLLM's own object (config()), or unpickled."""
    cls = _subclass(inner)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


def config(cfg, hf_config=None):
    """vLLM's own quantization config for the FP8 checkpoint whose quantization_config is cfg (refused where it is not one
    Glyd serves), with Glyd in front of it. hf_config: the model's, where known (the DeepSeek V4 family is picked by it)."""
    s = scheme(cfg)
    if s is None:
        raise ValueError(f"glyd: the quantization_config's quant_method {cfg.get('quant_method')!r} is not an FP8 one")
    check_gpu()
    if hf_config is None:
        hf_config = _HF.get(_key(cfg))
    if hf_config is None:
        from .vllm_plugin import _building_config

        vc = _building_config()
        hf_config = getattr(getattr(vc, "model_config", None), "hf_config", None)
    inner = inner_class(cfg, hf_config)
    obj = _restore(inner, vars(inner.from_config(cfg)))  # (vLLM's own config, whole, with Glyd's put in front)
    obj.glyd_scheme, obj.glyd_native_quant = s, native_quant(cfg)
    return obj


def refusals(vc):
    """What the FP8 path does not do yet, refused with why (vLLM's config vc): the checks of vllm_plugin._refusals that its packs and its
    scratch share with it (LoRA, weight offloading, sleep mode, dual-batch overlap, which run two batches' products at once)."""
    pc, mc = vc.parallel_config, vc.model_config
    if not isinstance(vc.additional_config, dict):
        raise ValueError("glyd: vLLM's additional_config is not a dict: Glyd keys vLLM's compile cache by it")
    if pc.use_ubatching:
        raise ValueError("glyd: dual-batch overlap (--enable-dbo, ubatching) runs two batches' products at once on two streams, and Glyd's FP8 Linears share one scratch: run without it")
    if vc.lora_config is not None:
        raise ValueError("glyd: LoRA on packed layers is not supported yet: serve without --enable-lora, or without --quantization glyd")
    oc = vc.offload_config
    if oc is not None and (oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0):
        raise ValueError("glyd: weight offloading (--cpu-offload-gb, prefetch offload) moves parameters, not Glyd's packs: not supported")
    if getattr(mc, "enable_sleep_mode", False):
        raise ValueError("glyd: sleep mode is not supported yet")


def wrap(cfg, m, layer, prefix):
    """vLLM's quantization method m for an FP8 Linear, wrapped (GlydFp8LinearMethod); m itself for what is not one it wraps: no
    method, an unquantized Linear (an ignored layer, the bf16 parts of a model), anything that is not a Linear (an embedding, an LM
    head), a Linear of few weights, a batched matrix (a bmm layer), a Linear vLLM reads the weight of directly after the load (DIRECT). Refused where a Linear's method is not an FP8 one."""
    if m is None or not isinstance(layer, LinearBase) or isinstance(m, UnquantizedLinearMethod):
        return m
    name = type(m).__name__
    if name == "CompressedTensorsLinearMethod":
        s = type(getattr(layer, "scheme", None)).__name__
        if s not in CT_METHODS:
            raise ValueError(f"glyd: {prefix}: compressed-tensors scheme {s} is not an FP8 one Glyd packs ({', '.join(CT_METHODS)})")
    elif name != "Fp8LinearMethod" and name != "ModelOptLinearMethod":
        raise ValueError(f"glyd: {prefix}: vLLM's {name} for this FP8 checkpoint's Linear is not one Glyd wraps")
    if name not in WEIGHT_LOADER_V2_SUPPORTED:
        raise ValueError(f"glyd: {prefix}: vLLM's {name} loads its weights the legacy way, which Glyd's wrapper does not")
    if getattr(layer, "is_bmm", False) or layer.input_size * layer.output_size < SMALL_LINEAR or prefix.rsplit(".", 1)[-1] in DIRECT:
        return m
    return GlydFp8LinearMethod(m, cfg)


def _kernel(method, layer):
    """vLLM's kernel object for an FP8 Linear (Fp8LinearMethod's and a compressed-tensors scheme's fp8_linear, ModelOptLinearMethod's kernel), None where none is found."""
    for o in (method, getattr(layer, "scheme", None)):
        k = getattr(o, "fp8_linear", None) or getattr(o, "kernel", None)
        if k is not None:
            return k
    return None


def _release(w):
    """The caching allocator's free blocks given back, before a pack is made on w's GPU: vLLM loads a model with max_split_size_mb 20, under which an
    allocation of 20 MiB or more takes a freed block of up to 20 MiB more whole (a freed weight's), and vLLM counts what it holds as the weights'
    (Qwen3.8-27B-FP8 on an L40S: 1.5 GiB of its Linears' packs)."""
    if w.is_cuda:
        torch.cuda.empty_cache()


def _matrix(w, name):
    """w (layer.weight, as vLLM's kernel is fed it) as the [rows, cols] uint8 matrix of its storage's bytes, on its own memory,
    and how to view them back as w: (dtype, shape, strides). A tensor of dense, contiguous storage (e4m3 [N, K], Marlin's
    int32 tiles) or its transposed view (per-tensor [K, N]); refused for another."""
    if w.storage_offset() != 0 or w.untyped_storage().nbytes() != w.numel() * w.element_size():
        raise ValueError(f"glyd: {name}: weight {w.dtype} {tuple(w.shape)} is a part of a larger storage: not a layout Glyd packs")
    view = (w.dtype, tuple(w.shape), tuple(w.stride()))
    if w.is_contiguous():
        return w.reshape(-1, w.shape[-1]).view(torch.uint8), view
    if w.dim() == 2 and w.t().is_contiguous():
        return w.t().view(torch.uint8), view
    raise ValueError(f"glyd: {name}: weight {w.dtype} {tuple(w.shape)} with strides {tuple(w.stride())} is not a layout Glyd packs (dense row-major, or a transposed view of it)")


class _Scratch:
    """A device's one byte buffer that packed weights are views of: made as the layers are, grown (each layer's weights pointed at the new one)
    to the largest layer's size, and held once the model has run (a CUDA graph keeps its pointer). A layer's weights in it: layer.glyd_f8_parts,
    (attribute, offset, bytes, (dtype, shape, strides)) each, layer.glyd_f8_nbytes in all."""

    def __init__(self):
        self.buf, self.users, self.sealed = None, [], False

    def add(self, layer):
        n = layer.glyd_f8_nbytes
        if self.buf is None or self.buf.numel() < n:
            if self.sealed:
                raise RuntimeError(f"glyd: {getattr(layer, 'prefix', '') or getattr(layer, 'layer_name', '')}: a packed FP8 layer made after the model ran: its scratch is held by CUDA graphs and cannot grow")
            new = torch.empty(n, dtype=torch.uint8, device=getattr(layer, layer.glyd_f8_parts[0][0]).device)
            self.users = [r for r in self.users if r() is not None]
            for r in self.users:
                self.point(r(), new)
            self.buf = new
        self.users.append(weakref.ref(layer))
        self.point(layer, self.buf)

    @staticmethod
    def point(layer, buf):
        for name, at, n, (dtype, shape, stride) in layer.glyd_f8_parts:
            getattr(layer, name).data = buf[at : at + n].view(dtype).as_strided(shape, stride)  # (the Parameter stays the one vLLM made)


def scratch(dev, where=None):
    where = _SCRATCH if where is None else where
    s = where.get(dev)
    if s is None:
        s = where[dev] = _Scratch()
    return s


class _Rec:
    """What glyd::f8_linear reads for a layer by its id: its prefix, which every process that builds the model gives the same (a graph vLLM compiled and
    keeps names it), told apart by a number from another live layer's of the same prefix (a draft model's); dropped when the layer is. fused: f8_linear's
    (kind, scale, most tokens) for it, or None (_fused)."""

    def __init__(self, layer, method, packed, sc, out, fused=None):
        base = getattr(layer, "prefix", "") or type(layer).__name__
        self.id, n = base, 1
        while self.id in _REG and _REG[self.id].layer() not in (None, layer):
            n += 1
            self.id = f"{base}#{n}"
        self.layer, self.method, self.packed, self.scratch, self.out, self.fused = weakref.ref(layer, lambda _, r=self: _REG.pop(r.id, None) if _REG.get(r.id) is r else None), method, packed, sc, out, fused
        _REG[self.id] = self


def fused_max(config):
    """The most tokens a step whose products f8_linear runs from the packs: 0 (the decode and vLLM's product for every step) unless a development build's
    GLYD_F8_FUSED_MAX says otherwise, and 0 under exact. f8_linear does not quantize the activations (W8A16: not a W8A8 checkpoint's numerics) and is not
    measured end to end in vLLM yet."""
    if getattr(config, "glyd_exact", False):
        return 0
    # dev: GLYD_F8_FUSED_MAX=N (the checks: f8_linear for steps of up to N tokens)
    v = os.environ.get("GLYD_F8_FUSED_MAX", "").strip()
    if v:
        return max(0, min(int(v), 64))
    # end dev
    return 0


def _fused(method, layer, packed, view):
    """(kind, scale, most tokens) where f8_linear can run this Linear's product from its pack (fused_max above 0, the lossless format's pack of the row-major
    e4m3 weight [N, K] itself, scales one fp32 or fp32 a 128 x 128 block as f8_linear takes them, no static activation scale, which it would not apply), else None."""
    n = fused_max(method.config)
    N, K = packed.shape
    if n <= 0 or getattr(layer, "input_scale", None) is not None or packed.codec != "f8" or packed.meta[0] != packed.shape or view[0] != getattr(torch, "float8_e4m3fn", None) or (N, K) != (layer.glyd_f8_out, getattr(layer, "input_size_per_partition", K)):
        return None
    s = getattr(layer, "weight_scale_inv", None)
    s = getattr(layer, "weight_scale", None) if s is None else s
    if not isinstance(s, torch.Tensor) or s.dtype != torch.float32:
        return None
    if s.numel() == 1:
        return "tensor", s.reshape(1), n
    if s.dim() == 2 and tuple(s.shape) == (-(-N // 128), -(-K // 128)) and s.stride(1) == 1 and list(getattr(method.inner, "weight_block_size", None) or []) == [128, 128]:
        return "block", s, n
    return None


@register_weight_loader_v2_supported_method
class GlydFp8LinearMethod(LinearMethodBase):
    """An FP8 Linear's product from its pack. vLLM's own method (inner) makes the parameters, loads them and processes them (the
    layer's weight as its kernel will be fed it); then the weight is packed and its memory freed, and every product decodes it into
    the scratch and runs the inner method's apply (one op), or, for a step of few tokens in a development build's check (GLYD_F8_FUSED_MAX, _fused),
    is f8_linear's from the pack. Scales, and the rest of the layer, are untouched."""

    def __init__(self, inner, config):
        self.inner, self.config = inner, config
        self.codec = f8codec.get(config.glyd_codec or None).name

    def __getattr__(self, name):  # (what vLLM reads of the method: use_marlin, block_quant, fp8_linear, quant_config ...)
        if name in ("inner", "config"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra):
        self.inner.create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra)
        layer.requires_unquantized_input = True  # (the op takes the activation as it is: no pre-quantized one from a fused producer)
        layer.glyd_f8_out, layer.glyd_f8_id, layer.glyd_f8_done = sum(output_partition_sizes), None, False  # (the id: None, as vLLM runs the layer, until it is packed)

    def process_weights_after_loading(self, layer):
        if layer.glyd_f8_done:  # (vLLM calls it again after the load: the weight is packed, and the inner method must not see the scratch)
            return
        layer.glyd_f8_done = True
        mem = []
        # dev: GLYD_F8_MEMLOG=1 (the checks: the GPU memory torch holds before vLLM's processing, after it, after the pack and after the scratch, a line a Linear)
        if os.environ.get("GLYD_F8_MEMLOG") == "1" and layer.weight.is_cuda:
            mem.append(torch.cuda.memory_allocated())
        # end dev
        self.inner.process_weights_after_loading(layer)
        # dev: GLYD_F8_MEMLOG
        if mem:
            mem.append(torch.cuda.memory_allocated())
        # end dev
        if getattr(layer, "is_bmm", False):  # (a batched matrix, not a Linear's product: as vLLM runs it)
            return
        name = getattr(layer, "prefix", "") or type(layer).__name__
        w = layer.weight
        try:
            mat, view = _matrix(w.data, name)
        except ValueError as e:  # (a layout not met yet: this Linear stays as vLLM runs it, and the log says so)
            log.warning("%s: stays as vLLM runs it", e)
            return
        k = _kernel(self.inner, layer)
        if getattr(self.config, "glyd_exact", False) and any(c.__name__ == "Fp8BlockScaledMMLinearKernel" for c in type(k).__mro__):
            raise ValueError(f"glyd: {name}: exact with vLLM's {type(k).__name__} (a block-scaled W8A8 product) is not measured against stock vLLM yet "
                             "(measured bit for bit: Marlin's weight-only kernel, on Ada): serve with exact off")
        layer.glyd_f8_parts, layer.glyd_f8_nbytes = [("weight", 0, mat.numel(), view)], mat.numel()
        _release(w)
        packed = f8codec.pack(mat, self.codec)
        # dev: GLYD_F8_VERIFY=1 (the checks: the pack decoded back and compared with the bytes it was made from, before they are freed)
        if os.environ.get("GLYD_F8_VERIFY") == "1":
            _verify(name, packed, mat)
        # end dev
        fused = _fused(self, layer, packed, view)
        if fused is not None and w.is_cuda:  # (a check's CPU layer: none to make)
            from . import _lib

            _lib._counters("f8_linear", w.device.index, None, 0, 1 << 16)  # (the device's done counters, made now: never in a CUDA graph's memory)
        sc = scratch(w.device)
        layer.glyd_f8_id = _Rec(layer, self, packed, sc, layer.glyd_f8_out, fused).id
        sc.add(layer)  # (the weight's old memory is the pack's now, or freed with the last reference to it)
        # dev: GLYD_F8_MEMLOG
        if mem:
            del w, mat
            log.info("glyd memlog %s: %d weight bytes, %d packed, allocated %d -> %d (processed) -> %d (packed; scratch %d)", name, layer.glyd_f8_nbytes, packed.nbytes(), mem[0], mem[1], torch.cuda.memory_allocated(), sc.buf.numel())
        # end dev

    apply = _trace.fp8_linear_method_apply  # (Python source: vLLM's torch.compile traces it)


def wrap_experts(cfg, m, prefix):
    """vLLM's FP8 method m of a mixture of experts' experts with Glyd's in front (a subclass of its class, m's state: isinstance of vLLM's own holds): the
    experts packed once vLLM has processed them (_Experts), the ones a call routes to decoded before vLLM's own apply; m itself for another method, and where
    the experts cannot be packed yet, with a line saying why: expert parallel, a kernel that routes the tokens itself (monolithic)."""
    if m is None or type(m).__name__ not in MOE_METHODS:
        return m
    if getattr(cfg, "glyd_exact", False) and type(m).__name__ == "ModelOptFp8MoEMethod":
        raise ValueError(f"glyd: {prefix}: exact on ModelOpt FP8 experts is not measured against stock vLLM yet: serve with exact off")
    why = "expert parallel" if m.moe.moe_parallel_config.use_ep else "vLLM's kernel for them picks each token's experts itself" if m.is_monolithic else None
    if why:
        log.warning_once("glyd: an FP8 checkpoint's experts stay as vLLM runs them: %s", why)
        return m
    cls = _MOE_SUBS.get(type(m))
    if cls is None:
        cls = _MOE_SUBS[type(m)] = type(type(m))(f"Glyd{type(m).__name__}", (GlydFp8MoEMethod, type(m)), {"__module__": __name__})
    obj = cls.__new__(cls)
    obj.__dict__.update(vars(m))
    obj.glyd_codec = f8codec.get(cfg.glyd_codec or None).name
    return obj


_MOE_SUBS = {}


class GlydFp8MoEMethod:
    """Put in front of vLLM's FP8 method of a mixture of experts' experts (wrap_experts): process_weights_after_loading is vLLM's, then the experts are packed
    (layer.glyd_f8_moe: an _Experts, or False where they stay as vLLM runs them); apply decodes the experts the call's tokens are routed to into the device's
    experts buffer, which the layer's weights are views of, then is vLLM's own (its kernel reads them there, and the scales as it loaded them)."""

    glyd_codec = None

    def process_weights_after_loading(self, layer):
        if getattr(layer, "glyd_f8_moe", None) is not None:  # (vLLM calls it again after the load)
            return
        super().process_weights_after_loading(layer)
        layer.glyd_f8_moe = _Experts.make(self, layer) or False

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        e = getattr(layer, "glyd_f8_moe", None)
        if e and x.shape[0]:
            e.decode(topk_ids)
        return super().apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)


class _Experts:
    """A layer's experts packed: w13's and w2's weights as vLLM's kernel is fed them (e4m3, or Marlin's int32 tiles), each the bytes of its E matrices stacked,
    in f8codec.experts_form; the layer's weights views of the device's experts buffer (layer.glyd_f8_parts), which decode fills with the routed experts."""

    PARTS = ("w13_weight", "w2_weight")

    def __init__(self, E, packs, dev):
        self.E, self.packs = E, packs
        self.all = torch.arange(E, dtype=torch.int32, device=dev)  # (made at load: never in a CUDA graph's memory)
        self.none = torch.full((E,), -1, dtype=torch.int32, device=dev)

    @classmethod
    def make(cls, method, layer):
        name = getattr(layer, "layer_name", "") or getattr(layer, "prefix", "") or type(layer).__name__
        ws = [getattr(getattr(layer, a, None), "data", None) for a in cls.PARTS]
        E = ws[0].shape[0] if isinstance(ws[0], torch.Tensor) and ws[0].dim() == 3 else None
        for a, w in zip(cls.PARTS, ws):
            if not isinstance(w, torch.Tensor) or w.dim() != 3 or w.shape[0] != E or not w.is_contiguous() or w.storage_offset() != 0 or w.untyped_storage().nbytes() != w.numel() * w.element_size():
                log.warning_once("glyd: an FP8 checkpoint's experts stay as vLLM runs them: %s's %s (%s) is not a layout Glyd packs (E contiguous matrices of their own storage)", name, a, None if w is None else f"{w.dtype} {tuple(w.shape)}")
                return None
        forms = [f8codec.experts_form(E, w.shape[1], w.shape[2] * w.element_size()) for w in ws]
        if None in forms:
            log.warning_once("glyd: an FP8 checkpoint's experts stay as vLLM runs them: experts of %s bytes are not whole patches of 64 rows", [tuple(w.shape) for w in ws])
            return None
        dev, packs, parts, at = ws[0].device, [], [], 0
        _release(ws[0])
        for a, w, f in zip(cls.PARTS, ws, forms):
            packs.append(f8codec.pack(w.view(torch.uint8).reshape(f), method.glyd_codec))
            # dev: GLYD_F8_VERIFY=1 (every expert of the stack decoded back and compared with its bytes)
            if os.environ.get("GLYD_F8_VERIFY") == "1":
                _verify(f"{name}.{a}", packs[-1], w.view(torch.uint8).reshape(f), E)
            # end dev
            n = w.numel() * w.element_size()
            parts.append((a, at, n, (w.dtype, tuple(w.shape), tuple(w.stride()))))
            at += n
        ws = w = None
        e = cls(E, packs, dev)
        e.parts, e.scratch = parts, scratch(dev, _EXPERTS)
        layer.glyd_f8_parts, layer.glyd_f8_nbytes = parts, at
        e.scratch.add(layer)  # (the weights' old memory freed with the last reference to it)
        return e

    def decode(self, ids):
        """The experts ids (topk_ids: a call's tokens' choices) names, decoded into their places of the buffer (the others left as they are: vLLM's kernel reads
        the experts a token is routed to alone). Device work only: a CUDA graph holds it."""
        self.scratch.sealed = True
        # dev: GLYD_F8_POISON (the checks: the buffer e4m3 NaN (0x7F) before each call, so that an expert no token is routed to, if vLLM's kernel read it, would move the logits)
        if os.environ.get("GLYD_F8_POISON"):
            self.scratch.buf.fill_(0x7F)
        # end dev
        E, i = self.E, ids.reshape(-1).long()
        hit = torch.zeros(E + 1, dtype=torch.int32, device=ids.device)
        hit.index_fill_(0, torch.where((i >= 0) & (i < E), i, E), 1)
        sel = torch.where(hit[:E] > 0, self.all, self.none)
        for p, (_, at, n, _v) in zip(self.packs, self.parts):
            f8codec.decode_experts(p, E, sel, self.scratch.buf[at : at + n].view(E, p.shape[0] // E, p.shape[1]))


# dev: GLYD_F8_VERIFY
VERIFIED = [0, 0]  # (the checks: packs decoded back bit for bit, and their bytes)


def _verify(name, p, want, E=None):
    """The pack p decoded back (a Linear's matrix, or all E experts of a stack) and compared with the bytes want it was made from: a RuntimeError where they differ."""
    got = torch.empty_like(want)
    if E is None:
        f8codec.decode(p, got)
    else:
        f8codec.decode_experts(p, E, torch.arange(E, dtype=torch.int32, device=want.device), got.view(E, want.shape[0] // E, want.shape[1]))
    if not torch.equal(got, want):
        raise RuntimeError(f"glyd: {name}: its pack ({p.codec}) does not decode to the bytes it was made from")
    VERIFIED[0] += 1
    VERIFIED[1] += want.numel()


# end dev
def _define():
    """glyd::f8_linear: x W^T (+ bias) for the FP8 Linear `layer` (its id, a prefix), by vLLM's own method: the weight decoded into the scratch, then
    the inner method's apply."""
    if _OPS:
        return
    lib = torch.library.Library("glyd", "FRAGMENT")
    lib.define("f8_linear(Tensor x, Tensor? bias, str layer) -> Tensor")

    def linear(x, bias, layer):  # (the layer's id: the schema's name)
        r = _REG[layer]
        f = r.fused
        if f is not None and x.dtype == torch.bfloat16 and 0 < x.numel() <= f[2] * x.shape[-1]:  # (a step of few tokens: f8_linear from the pack)
            from . import _lib, kernels

            fresh, _lib.local.fresh = _lib.local.fresh, True  # (its workspace the call's own: in a capture, the graph's memory)
            try:
                y = kernels.f8_linear(r.packed.obj, f[0], f[1], x.reshape(-1, x.shape[-1]))
            finally:
                _lib.local.fresh = fresh
            if bias is not None:
                y = y + bias
            return y.view(*x.shape[:-1], y.shape[-1])
        r.scratch.sealed = True  # (from here on a CUDA graph may hold the scratch's pointer)
        f8codec.decode(r.packed, r.scratch.buf[: r.packed.shape[0] * r.packed.shape[1]].view(r.packed.shape))
        y = r.method.inner.apply(r.layer(), x, bias)
        # dev: GLYD_F8_POISON
        if os.environ.get("GLYD_F8_POISON"):  # (the scratch overwritten after every product: whatever reads it outside the op, or overlaps it on another stream, moves the logits)
            r.scratch.buf.fill_(0xFF)
        # end dev
        return y

    for key in ("CUDA", "CPU"):  # (the CPU's: tests with a method of their own)
        lib.impl("f8_linear", linear, key)
    torch.library.register_fake("glyd::f8_linear", lambda x, bias, layer: x.new_empty((*x.shape[:-1], _REG[layer].out)), lib=lib)
    _OPS.append(lib)


_OPS = []


def register():
    """The op, in this process (again: the same). vllm_plugin.register() calls it."""
    _define()


def stats(model):
    """(Linears and experts' layers packed, bytes their packs hold, bytes of their weights as vLLM's kernel is fed them, the scratches' bytes) of a model: for the
    checks (LLM.apply_model)."""
    n = packed = whole = 0
    for m in model.modules():
        if isinstance(getattr(m, "quant_method", None), GlydFp8LinearMethod) and getattr(m, "glyd_f8_id", None) is not None:
            r = _REG[m.glyd_f8_id]
            n, packed, whole = n + 1, packed + r.packed.nbytes(), whole + m.glyd_f8_nbytes
        e = getattr(m, "glyd_f8_moe", None)
        if e:
            n, packed, whole = n + 1, packed + sum(p.nbytes() for p in e.packs), whole + m.glyd_f8_nbytes
    return n, packed, whole, sum(s.buf.numel() for d in (_SCRATCH, _EXPERTS) for s in d.values() if s.buf is not None)
