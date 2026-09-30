"""Glyd in vLLM: a model's Linears held packed on the GPU, bit for bit, as
vLLM loads them, and multiplied from there by the library's kernels.

    pip install "glyd[vllm]"
    vllm serve Qwen/Qwen3-8B --quantization glyd                  # a bf16 checkpoint, packed as it loads
    vllm serve ./qwen3-8b-glyd --quantization glyd                # a glyd save, its packs as saved
    vllm serve Qwen/Qwen3-8B --quantization glyd --additional-config '{"glyd": {"layout": "mma12"}}'

vLLM calls register() in every process it starts (the entry point
vllm.general_plugins: glyd): the quantization method "glyd".

Options, in --additional-config's "glyd" (or a checkpoint's, or
--hf-overrides', quantization_config; else GLYD_LAYOUT, GLYD_EXACT,
GLYD_VERIFY):
- layout: "auto" (best_layout's choice for the GPU: the tiered layout on
  Ada and where only it fits, the 12-bit one elsewhere), "mma" or "mma12";
- exact: each product's matrix decoded whole, then the GEMM vLLM runs for
  bf16 (F.linear): its logits bf16's bit for bit, with --enforce-eager
  alone (under vLLM's torch.compile the product runs inside Glyd's op, not
  in the compiled graph as bf16's does, and the logits differ from compiled
  bf16's, which is not always the same from one compile to the next
  either: exact is refused there, never silently inexact);
- verify: every pack decoded as it is made and compared with its weights
  bit for bit (a glyd save's by glyd.json's sha256).
The options in effect, with a digest of the packs (their layouts, words and
sizes), are written into vLLM's additional_config, which vLLM's compile
cache keys by: another layout, mode or checkpoint never finds another's
compiled graph.

A Linear's weight (bf16) is held on the meta device and packed a layer at a
time as vLLM's layerwise online processing completes it (the peak: the
packs and a layer); a glyd save's packs load as saved (TP 1, the same
layout; else decoded and packed again). A product is one op,
glyd::vllm_linear, over the pack's tensors: the library's route by the
batch's tokens (glyd_gpu_mma[12]_linear), and where it routes a prompt to
cuBLAS (DECODE, AHEAD) the matrix decoded into the GPU's scratch buffer,
then F.linear. vLLM's torch.compile takes it as one node, its CUDA graphs
capture its kernels. Embeddings, the LM head, norms, attention and the KV
cache stay vLLM's; a mixture of experts' experts too (bf16), for now.
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of
Glyd's GPU code; it uses vLLM's plugin interfaces, and copies none of its
code.
"""
import hashlib
import json
import os
import torch
import torch.nn.functional as F
from vllm.config import CompilationMode, CUDAGraphMode, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.reload.layerwise import initialize_online_processing
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from .. import __version__
from . import _lib, format as fmt, kernels as g

log = init_logger("vllm.glyd")
NAME = "glyd"
KEY = "glyd"  # vllm_config.additional_config's: the options (and the packs' digest), in vLLM's compile cache key
LAYOUTS = ("auto", "mma", "mma12")
BUFFERS = {"mma": ("glyd_data", "glyd_blocks", "glyd_block_base"), "mma12": ("glyd_data", "glyd_exc", "glyd_exc_base")}  # a save's names
_OPS = []  # the ops' library (they live as long as it does)
_SCRATCH = {}  # a device's buffer matrices are decoded into (bf16): made at load, never replaced after
_ROUTES = {}  # (gpu, words, O, K, M): the library's route
_GPU = {}  # a device's code, as the library's routes take it


def _flag(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _gpu(d):
    code = _GPU.get(d)
    if code is None:
        with torch.cuda.device(d):
            code = _GPU[d] = _lib.gpu()  # the library's own (glyd_gpu_gpu): its routes' classes by name, whatever they are
    return code


def _decoded(d, words, O, K, M):
    """Whether a product of M tokens decodes its matrix for cuBLAS: the library's route DECODE or AHEAD (a long prompt on
    a GPU where cuBLAS on the decoded matrix outruns the fused kernel), or a prompt kernel's where K is not a multiple
    of 64 (which it does not take)."""
    key = (d, len(words), O, K, M)
    r = _ROUTES.get(key)
    if r is None:
        route = (_lib.mma12_route if len(words) == 4 else _lib.mma_route)(_gpu(d), O, K, M)[0]
        r = _ROUTES[key] = route in (g.DECODE, g.AHEAD) or (route == g.BIG and K % 64 != 0)
    return r


def _unpack(data, a, b, words, O, K):
    """The matrix, bf16, decoded into the device's scratch buffer (O * K weights of it)."""
    w = _SCRATCH[data.device][: O * K].view(O, K)
    (_lib.mma12_unpack if len(words) == 4 else _lib.mma_unpack)(data, a, b, list(words), K, 0, O, w.view(torch.int16), 0)
    return w


def _define():
    """glyd::vllm_linear: X W^T (+ bias) over a pack (data and its two tensors; its words: the tiered layout's three
    tiers, the 12-bit one's four), fused by the library's route for X's rows on this GPU, or the matrix decoded, then
    F.linear (exact, or where the route says so)."""
    if _OPS:
        return
    lib = torch.library.Library("glyd", "FRAGMENT")
    lib.define("vllm_linear(Tensor x, Tensor data, Tensor a, Tensor b, int[] words, Tensor? bias, int out_features, bool exact) -> Tensor")

    def linear(x, data, a, b, words, bias, out_features, exact):
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        if exact or (M and _decoded(data.device.index, words, out_features, K, M)):
            return F.linear(x2, _unpack(data, a, b, words, out_features, K), bias).view(*x.shape[:-1], out_features)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.empty(M, out_features, dtype=torch.bfloat16, device=x.device)
        if M:
            fresh, _lib.local.fresh = _lib.local.fresh, True  # workspaces for the call alone (in a capture: the graph's pool)
            try:
                f = _lib.mma12_linear if len(words) == 4 else _lib.mma_linear
                f(data, a, b, list(words), out_features, K, x2, bias if bias is not None else _nobias(x.device), y, -1)
            finally:
                _lib.local.fresh = fresh
        return y.view(*x.shape[:-1], out_features)

    lib.impl("vllm_linear", linear, "CUDA")
    torch.library.register_fake("glyd::vllm_linear", lambda x, data, a, b, words, bias, out_features, exact: x.new_empty((*x.shape[:-1], out_features)), lib=lib)
    _OPS.append(lib)


_NOBIAS = {}


def _nobias(dev):
    t = _NOBIAS.get(dev)
    if t is None:
        t = _NOBIAS[dev] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    return t


def register():
    """vLLM's general plugin: the "glyd" quantization method and its op, in this process (again: the same)."""
    _define()
    register_quantization_config(NAME)(GlydConfig)


def _linear_bytes(mc):
    """A dense model's Linears (bf16 bytes) and the rest, from its config (for best_layout's fit); (0, 0) where unknown."""
    try:
        c = mc.hf_text_config
        h, L, I = c.hidden_size, c.num_hidden_layers, c.intermediate_size
        nh = c.num_attention_heads
        nkv = getattr(c, "num_key_value_heads", None) or nh
        hd = getattr(c, "head_dim", None) or h // nh
        per = h * (nh + 2 * nkv) * hd + nh * hd * h + 3 * h * I
        other = c.vocab_size * h * (1 if getattr(c, "tie_word_embeddings", False) else 2)
        return 2 * L * per, 2 * other
    except (AttributeError, TypeError):
        return 0, 0


class GlydConfig(QuantizationConfig):
    """The "glyd" quantization method: the options given (a checkpoint's or --hf-overrides' quantization_config), a
    glyd save's glyd.json where the model is one, and the options in effect once vLLM makes the model (resolve)."""

    def __init__(self, layout=None, exact=None, verify=None):
        super().__init__()
        self.given = {k: v for k, v in (("layout", layout), ("exact", exact), ("verify", verify)) if v is not None}
        self.manifest = None  # a glyd save's glyd.json
        self.opts = None  # {"layout", "exact", "verify"}: resolved at the model's first Linear
        self.packs = {}  # a packed layer's prefix: (layout, shape, words, sizes), for the digest

    def __getstate__(self):  # (vLLM pickles its config to start its engine's process: not the worker's own)
        return {**self.__dict__, "_vc": None}

    def get_name(self):
        return NAME

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config.get("layout"), config.get("exact"), config.get("verify"))

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        """A glyd save: its glyd.json, beside its weights (a Hub repo's fetched into its snapshot)."""
        d = model_name
        if not os.path.isdir(d):
            try:
                from huggingface_hub import hf_hub_download

                d = os.path.dirname(hf_hub_download(model_name, fmt.MANIFEST, revision=revision))
            except Exception:  # none: a bf16 checkpoint (or a repo transformers will name)
                return
        self.manifest = fmt.read_manifest(d)

    def resolve(self):
        """The options in effect: --additional-config's "glyd", else the given ones, else GLYD_LAYOUT, GLYD_EXACT,
        GLYD_VERIFY; the layout "auto" best_layout's for this GPU (a save's own where that is the one). What Glyd
        does not do yet refused, with why. The options written into vLLM's additional_config (its compile cache's
        key)."""
        vc = get_current_vllm_config_or_none()
        if vc is None:
            raise RuntimeError("glyd: vLLM's config is not set where the model is made")
        if not isinstance(vc.additional_config, dict):
            raise ValueError("glyd: vLLM's additional_config is not a dict: Glyd keys vLLM's compile cache by it")
        extra = vc.additional_config.get(KEY) or {}
        env = {"layout": os.environ.get("GLYD_LAYOUT"), "exact": os.environ.get("GLYD_EXACT"), "verify": os.environ.get("GLYD_VERIFY")}
        pick = {k: extra[k] if k in extra else self.given[k] if k in self.given else env[k] for k in env}
        layout, exact, verify = str(pick["layout"] or "auto").lower(), _flag(pick["exact"] or 0), _flag(pick["verify"] or 0)
        if layout not in LAYOUTS:
            raise ValueError(f"glyd: layout {layout!r}: one of {', '.join(LAYOUTS)}")
        pc, mc = vc.parallel_config, vc.model_config
        if pc.use_ubatching:
            raise ValueError("glyd: dual-batch overlap (--enable-dbo, ubatching) runs two batches' products at once on two streams, which Glyd's kernels do not share a GPU's done counters for yet: run without it")
        if vc.lora_config is not None:
            raise ValueError("glyd: LoRA on packed layers is not supported yet: serve without --enable-lora, or without --quantization glyd")
        oc = vc.offload_config
        if oc is not None and (oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0):
            raise ValueError("glyd: weight offloading (--cpu-offload-gb, prefetch offload) moves parameters, not Glyd's packs: not supported")
        if getattr(mc, "enable_sleep_mode", False):
            raise ValueError("glyd: sleep mode is not supported yet")
        if exact and not (mc.enforce_eager or (vc.compilation_config.mode == CompilationMode.NONE and vc.compilation_config.cudagraph_mode == CUDAGraphMode.NONE)):
            raise ValueError("glyd: exact mode gives vLLM's bf16 logits bit for bit only with --enforce-eager: under torch.compile the product runs inside Glyd's op, not in vLLM's compiled graph as bf16's does, and the logits differ from compiled bf16's (which is not always the same from one compile to the next either). Add --enforce-eager, or leave exact off")
        if self.manifest is not None:
            if pc.tensor_parallel_size > 1 or pc.pipeline_parallel_size > 1:
                raise ValueError("glyd: a glyd save loads on one GPU for now (tensor and pipeline parallel: from its bf16 checkpoint)")
            if any("experts" in e for e in self.manifest["packs"].values()):
                raise ValueError("glyd: a save with a mixture of experts' packs (glyd-v2) is not supported in vLLM yet: load its bf16 checkpoint")
        elif pc.tensor_parallel_size > 1:
            log.warning("glyd: tensor parallel %d: each rank packs its own shard (not measured yet)", pc.tensor_parallel_size)
        if layout == "auto":
            if self.manifest is not None:
                layout = self.manifest["layout"]
            else:
                lin, other = _linear_bytes(mc)
                layout = g.best_layout(lin // max(1, pc.tensor_parallel_size), other, 1, torch.cuda.current_device())[0]
        self.opts, self._vc = {"layout": layout, "exact": exact, "verify": verify}, vc
        vc.additional_config[KEY] = {**self.opts, "glyd": __version__, "packs": ""}
        log.info("glyd: %s layout%s%s", layout, ", exact" if exact else "", ", verified" if verify else "")

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, LinearBase):
            return None
        if self.opts is None:
            self.resolve()
        return GlydLinearMethod(self, self.saved(prefix))

    def saved(self, prefix):
        """A glyd save's manifest entry for vLLM's layer prefix, or None: its pack as saved (q, k, v and gate, up are one
        pack each, under q_proj's and gate_proj's paths, as vLLM's qkv_proj and gate_up_proj hold them)."""
        if self.manifest is None:
            return None
        packs = self.manifest["packs"]
        for merged, first, n in (("qkv_proj", "q_proj", 3), ("gate_up_proj", "gate_proj", 2)):
            if prefix.endswith("." + merged):
                e = packs.get(prefix[: -len(merged)] + first)
                return e if e is not None and len(e["tensors"]) == n else None
        e = packs.get(prefix)
        return e if e is not None and len(e["tensors"]) == 1 else None

    def packed(self, layer, layout, words, t):
        """A layer packed: into the digest of the packs, which vLLM's compile cache keys by (additional_config)."""
        self.packs[getattr(layer, "prefix", str(id(layer)))] = [layout, list(words), [int(x.numel()) for x in t]]
        digest = hashlib.sha256(json.dumps(sorted(self.packs.items())).encode()).hexdigest()[:16]
        vc = getattr(self, "_vc", None) or get_current_vllm_config_or_none()
        if vc is not None and isinstance(vc.additional_config, dict) and KEY in vc.additional_config:
            vc.additional_config[KEY]["packs"] = digest


def _take_whole(param, loaded_weight, *shard):
    """A saved pack's tensor, whole and as it is (a merged group's under its first member's name: its shard ignored)."""
    param.data = loaded_weight.to(param.device)


class GlydLinearMethod(LinearMethodBase):
    """A Linear's product from its pack. From a bf16 checkpoint the weight is made on the meta device and packed as
    vLLM's layerwise processing completes the layer (process_weights_after_loading); from a glyd save the pack's
    tensors load as saved. A matrix the mma layouts do not take (rows not a multiple of 64, columns not of 16), or
    not bf16, stays as vLLM runs it."""

    uses_meta_device = True

    def __init__(self, config, saved=None):
        self.config, self.saved, self.plain = config, saved, None
        if saved is not None:
            self.uses_meta_device = False

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra):
        O, K = sum(output_partition_sizes), input_size_per_partition
        if self.saved is not None:
            if list(self.saved["shape"]) != [O, K]:
                raise ValueError(f"glyd: {getattr(layer, 'prefix', '')}: the save's pack is {self.saved['shape']}, vLLM's layer [{O}, {K}]")
            for name in BUFFERS[self.saved["layout"]]:  # (their sizes the save's: made as they load)
                p = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8 if fmt.DTYPES[name[5:]] == "U8" else torch.int32), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": _take_whole})
                layer.register_parameter(name, p)
            return
        if O % 64 or K % 16 or params_dtype != torch.bfloat16:
            self.plain, self.uses_meta_device = UnquantizedLinearMethod(), False
            return self.plain.create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra)
        weight = ModelWeightParameter(data=torch.empty(O, K, device="meta", dtype=params_dtype), input_dim=1, output_dim=0, weight_loader=extra.get("weight_loader"))
        layer.register_parameter("weight", weight)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer):
        if self.plain is not None:
            return self.plain.process_weights_after_loading(layer)
        if getattr(layer, "glyd_words", None) is not None:  # (vLLM calls it again after the load)
            return
        opts = self.config.opts
        layout = opts["layout"]
        if self.saved is not None:
            p = self._saved_pack(layer)
            dev = p.data.device
            if isinstance(p, g.Mma12) != (layout == "mma12"):  # another layout asked for: decoded and packed again
                p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(g.mma_unpack(p))
        else:
            w = layer.weight.data
            dev = w.device
            p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(w)
            if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
                raise ValueError(f"glyd: {getattr(layer, 'prefix', '')} decoded to other bits than its weights")
        g.lib()  # the library loaded (the op's calls are its C API's)
        t = (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base)
        for name in {n for names in BUFFERS.values() for n in names}:
            layer._parameters.pop(name, None)
        for name, x in zip(("glyd_data", "glyd_a", "glyd_b"), t):
            layer.register_buffer(name, x, persistent=False)
        O, K = p.shape
        layer.glyd_words = list(p.sym if isinstance(p, g.Mma12) else p.tiers)
        layer.glyd_out, layer.glyd_exact, layer.glyd_verified = O, opts["exact"], opts["verify"]
        layer._parameters.pop("weight", None)
        layer.weight = torch.nn.Parameter(torch.empty(0, K, dtype=torch.bfloat16, device=dev), requires_grad=False)  # (read for its dtype and K)
        self.config.packed(layer, layout, layer.glyd_words, t)
        need = O * K if opts["exact"] or self._decodes(dev, layer.glyd_words, O, K) else 0
        if need and (dev not in _SCRATCH or _SCRATCH[dev].numel() < need):
            _SCRATCH[dev] = torch.empty(need, dtype=torch.bfloat16, device=dev)  # (at load: no CUDA graph holds the old one)
        for name in ("mma12_linear", "mma_linear"):  # the device's done counters, made now: never in a CUDA graph's pool
            _lib._counters(name, dev.index, None, 0, _lib._UNITS)

    def _saved_pack(self, layer):
        """The save's pack from the tensors loaded (its words from glyd.json), checked by its sha256s with verify."""
        e = self.saved
        data, a, b = (getattr(layer, n).data for n in BUFFERS[e["layout"]])
        if not data.numel():
            raise ValueError(f"glyd: {getattr(layer, 'prefix', '')}: the save's pack did not load")
        shape = tuple(e["shape"])
        p = g.Mma12(shape, data, a, b, int(e["hb"])) if e["layout"] == "mma12" else g.Mma(shape, data, a, b, [int(x) for x in e["tiers"]])
        if self.config.opts["verify"]:
            from .model import sha256

            w, r = g.mma_unpack(p), 0
            for t in e["tensors"]:
                if sha256(w[r : r + t["shape"][0]]) != t["sha256"]:
                    raise ValueError(f"glyd: {t['name']} decodes to other bits than glyd.json's sha256")
                r += t["shape"][0]
        return p

    def _decodes(self, dev, words, O, K):
        """Whether any product up to vLLM's batch of tokens decodes the matrix for cuBLAS (the scratch buffer it needs)."""
        vc = getattr(self.config, "_vc", None)
        top = vc.scheduler_config.max_num_batched_tokens if vc is not None else 8192
        M = 1
        while M <= top:
            route, last = (_lib.mma12_route if len(words) == 4 else _lib.mma_route)(_gpu(dev.index), O, K, M)
            if route in (g.DECODE, g.AHEAD) or (route == g.BIG and K % 64):
                return True
            M = last + 1
        return False

    def apply(self, layer, x, bias=None):
        if self.plain is not None:
            return self.plain.apply(layer, x, bias)
        return torch.ops.glyd.vllm_linear(x, layer.glyd_data, layer.glyd_a, layer.glyd_b, layer.glyd_words, bias, layer.glyd_out, layer.glyd_exact)
