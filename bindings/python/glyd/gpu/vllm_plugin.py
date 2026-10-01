"""Glyd in vLLM: a model's Linears, and a mixture of experts' experts, held
packed on the GPU, bit for bit, as vLLM loads them, and multiplied from there
by the library's kernels.

    pip install "glyd[vllm]"
    vllm serve Qwen/Qwen3-8B --quantization glyd                  # a bf16 checkpoint, packed as it loads
    vllm serve ./qwen3-8b-glyd --quantization glyd                # a glyd save, its packs as saved
    vllm serve Qwen/Qwen3-8B --quantization glyd --additional-config '{"glyd": {"layout": "mma12"}}'

vLLM calls glyd.gpu.vllm_entry.register() in every process it starts (the
entry point vllm.general_plugins: glyd); where vLLM is the minor release this
is tested with, it calls register() here: the quantization method "glyd".

Options, in --additional-config's "glyd" (or a checkpoint's, or
--hf-overrides', quantization_config; else GLYD_LAYOUT, GLYD_EXACT,
GLYD_VERIFY, GLYD_FRACTION); another key, or a flag not true or false,
refused:
- layout: "auto" (best_layout's choice for the GPU: the tiered layout on
  Ada, for a mixture of experts on an A10 too, and where only it fits; the
  12-bit one elsewhere), "mma" or "mma12";
- exact: each product's matrix decoded whole, then the GEMM vLLM runs for
  bf16 (UnquantizedLinearMethod's apply: F.linear by default, a FlashInfer
  --linear-backend's, VLLM_BATCH_INVARIANT's): its logits bf16's bit for
  bit, eager (--enforce-eager), or compiled with inductor's deterministic
  mode, with which vLLM's compiled bf16 is itself the same from one run to
  the next: --compilation-config '{"inductor_compile_config":
  {"deterministic": true, "combo_kernels": true, "benchmark_combo_kernel":
  false}}' (inductor otherwise times some of its kernels' variants on the
  GPU, and compiled logits, bf16's too, are not always the same from one
  run to the next). Refused compiled without it, and compiled where a packed
  Linear has a bias (inductor adds bf16's apart from its matmul, rounding
  before it), never silently inexact;
- verify: every pack decoded as it is made and compared with its weights
  bit for bit (a glyd save's by glyd.json's sha256, and its tensors saved as
  they are; a save packed again in the other layout against the save);
- fraction: the share of the decoder layers packed, 0 to 1 (default 1: every
  one). Layer i's Linears (qkv, o, gate_up, down; a mixture of experts'
  experts too) are packed where floor((i + 1) * fraction) > floor(i *
  fraction), so floor(L * fraction) of L layers, spread evenly over the
  depth; the rest stay vLLM's own bf16 (UnquantizedLinearMethod,
  UnquantizedFusedMoEMethod), exactly as it runs without --quantization
  glyd: 0 is bf16 itself, and a packed layer trades a rebuild each step for
  its memory. Layers are told by the first number in a module's name
  (model.layers.12.mlp.down_proj: 12); a Linear outside the numbered layers
  is packed at fraction 1 only. Embeddings and the LM head as without it. A
  glyd save's layers are packed on disk: with a save, only fraction 1.
Fused products (exact off) are refused under VLLM_BATCH_INVARIANT: the
library's kernels are chosen by the batch's tokens.
The options in effect, with a digest of the packs (their layouts, words and
sizes; one a process, a draft model's with the target's), are written into
vLLM's additional_config, which vLLM's compile cache keys by: another
layout, mode or checkpoint never finds another's compiled graph.

A Linear's weight (bf16) is held on the meta device and packed a layer at a
time as vLLM's layerwise online processing completes it (the peak: the
packs and a layer), once every piece of the weight has loaded (else refused,
naming the layer); its bf16 is dropped as soon as it is packed, and PyTorch's
unused blocks are handed back to the driver where they outweigh its free
memory (_trim: vLLM loads with the allocator's max_split_size_mb at 20, which
never splits a larger block, so on a full card the cache of many sizes of
block piled up until a cudaMalloc failed and the allocator warned, hundreds of
times on a 16 GB card); a glyd save's packs load as saved (TP 1, the same
layout, the families checked; else decoded and packed again). A product is one op,
glyd::vllm_linear, over the pack's tensors: the library's route by the
batch's tokens (glyd_gpu_mma[12]_linear), and where it routes a prompt to
cuBLAS (DECODE, AHEAD) the matrix decoded into the GPU's scratch buffer,
then F.linear. vLLM's torch.compile takes it as one node, its CUDA graphs
capture its kernels. A mixture of experts' experts: GlydMoEMethod.
Embeddings, the LM head, norms, attention and the KV cache stay vLLM's (a
save's packed LM head decoded to bf16 as it loads). A speculative draft:
vLLM keeps an EAGLE-3 draft bf16 under --quantization glyd, and packs it
where --speculative-config asks for "quantization": "glyd" (a GlydConfig of
its own, sized by the draft's config, its packs in the process's digest).

vLLM's internals it relies on, beyond its plugin entry point and
register_quantization_config (vLLM 0.30; the entry point refuses another
minor release):
- QuantizationConfig (from_config, maybe_update_config, get_quant_method);
  LinearBase, LinearMethodBase, QKVParallelLinear and
  MergedColumnParallelLinear (their loaders' shard ids), and
  UnquantizedLinearMethod (exact's GEMM); VocabParallelEmbedding and
  UnquantizedEmbeddingMethod; ModelWeightParameter and set_weight_attrs;
- the layerwise online processing of model_loader.reload.layerwise
  (initialize_online_processing; get_layerwise_info's loaded_weights and
  load counts, for the pieces loaded);
- get_current_vllm_config_or_none, and the config's additional_config
  (the compile cache's key), parallel_config (tensor and pipeline parallel,
  ubatching), lora_config, offload_config, model_config (enforce_eager,
  enable_sleep_mode, hf_text_config), compilation_config (mode,
  cudagraph_mode, inductor_compile_config) and scheduler_config
  (max_num_batched_tokens); vllm.envs (VLLM_BATCH_INVARIANT,
  VLLM_DISABLE_SHARED_EXPERTS_STREAM, VLLM_USE_FLASHINFER_SAMPLER: read, for
  a warning where FlashInfer's sampler has no nvcc; never set);
- for a mixture of experts (without them, experts stay bf16): RoutedExperts
  (moe_config, layer_name, top_k, activation, expert_map,
  apply_router_weight_on_input), OnlineMoEMethodBase,
  FusedMoEQuantConfig.make, MoEActivation, and UnquantizedFusedMoEMethod
  (unquantized_backend, _init_moe_kernel, moe_kernel.apply's keyword
  arguments) with UnquantizedMoeBackend.
It copies none of vLLM's code. Under the Business Source License 1.1
(LICENSE-glyd-gpu), as the rest of Glyd's GPU code.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import types
from fractions import Fraction
from math import floor
import torch
import torch.nn.functional as F
from vllm import envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, MergedColumnParallelLinear, QKVParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info, initialize_online_processing
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from .. import __version__
from . import _lib, format as fmt, kernels as g

try:  # a mixture of experts' internals: without them (another vLLM) a model's experts stay vLLM's bf16
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    from vllm.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase

    _NO_MOE = None
except ImportError as _e:
    RoutedExperts, OnlineMoEMethodBase, _NO_MOE = None, object, f"{type(_e).__name__}: {_e}"

log = init_logger("vllm.glyd")
NAME = "glyd"
KEY = "glyd"  # vllm_config.additional_config's: the options (and the packs' digest), in vLLM's compile cache key
LAYOUTS = ("auto", "mma", "mma12")
OPTIONS = ("layout", "exact", "verify", "fraction")
IGNORED = ("quant_method", "merge", "verified", "source", "hashed", "unhashed")  # (transformers' glyd config's own)
BITS = {"mma": 10.80, "mma12": 12.04}
# Tokens a step from which a mixture of experts' layer decodes its routed experts for vLLM's Triton kernel instead of the
# library's grouped products: granite-3.1-3b-a800m-instruct's layer on an L4, the decoded route the faster from 1,152
# tokens (2.89 against 2.99 ms; 1,024: 2.84 against 2.75), 0.65x at 8,192 (benchmarks/gpu/l4-vllm-moe-routes-2026-09-30)
MOE_DECODE_MIN = 1152
SAVES = ("Qwen3ForCausalLM", "LlamaForCausalLM")  # the families whose glyd saves are checked in vLLM (check_vllm.py --saves)
BUFFERS = {"mma": ("glyd_data", "glyd_blocks", "glyd_block_base"), "mma12": ("glyd_data", "glyd_exc", "glyd_exc_base")}  # a save's names
_OPS = []  # the ops' library (they live as long as it does)
_SCRATCH = {}  # a device's buffer matrices are decoded into (bf16): made at load, never replaced after
_ROUTES = {}  # (gpu, words, O, K, M): the library's route
_GPU = {}  # a device's code, as the library's routes take it
_BF16 = []  # exact: vLLM's bf16 method, whose GEMM exact's products run (made where vLLM's config is current)
_PACKS = {}  # every layer packed in this process (a draft model's too): (layout, words, sizes), for the digest


def _flag(name, v):
    """An option's true or false: a bool, 0 or 1, or 1, true, yes, on / 0, false, no, off (any case); else refused."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("", "0", "false", "no", "off"):
        return False
    raise ValueError(f"glyd: {name} {v!r}: true or false (1, true, yes, on / 0, false, no, off)")


def _fraction(v):
    """The option fraction: a number from 0 to 1 (a string or int too; 1 where not given); else refused."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return 1.0
    try:
        f = float("nan") if isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if not 0.0 <= f <= 1.0:  # (nan is neither)
        raise ValueError(f"glyd: fraction {v!r}: a number from 0 to 1 (the share of the decoder layers packed)")
    return f


def _layer_of(prefix):
    """The decoder layer a module is in, by the first number in its name (vLLM's prefix: model.layers.12.mlp.down_proj is
    12, transformer.h.3.attn.c_attn 3); None where it has none (outside the numbered layers)."""
    return next((int(p) for p in prefix.split(".") if p.isascii() and p.isdigit()), None)


def _packs(i, f):
    """Whether decoder layer i is packed at fraction f: floor((i + 1) f) > floor(i f), f as the decimal it was written
    (0.3 is 3/10, whatever its float). Of any L layers from the first, floor(L f) are, spread evenly: 0 packs none, 1 all."""
    q = Fraction(repr(f))
    return floor((i + 1) * q) > floor(i * q)


def _options(extra, given, env):
    """The options in effect (layout, exact, verify, fraction): --additional-config's "glyd" (extra), else the given ones
    (a checkpoint's or --hf-overrides' quantization_config), else the environment's; another key refused."""
    if not isinstance(extra, dict):
        raise ValueError(f"glyd: --additional-config's \"glyd\" is a {type(extra).__name__}, not an object of options ({', '.join(OPTIONS)})")
    extra = {k: v for k, v in extra.items() if k not in ("glyd", "packs")}  # (what resolve writes there: the key)
    bad = [k for k in extra if k not in OPTIONS]
    if bad:
        raise ValueError(f"glyd: --additional-config's \"glyd\": {', '.join(map(repr, bad))}, not an option ({', '.join(OPTIONS)})")
    pick = {k: extra[k] if k in extra else given[k] if k in given else env.get(k) for k in OPTIONS}
    layout = str(pick["layout"] or "auto").lower()
    if layout not in LAYOUTS:
        raise ValueError(f"glyd: layout {pick['layout']!r}: one of {', '.join(LAYOUTS)}")
    return layout, _flag("exact", pick["exact"] or False), _flag("verify", pick["verify"] or False), _fraction(pick["fraction"])


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
    F.linear (where the route says so), or (exact) the GEMM vLLM's bf16 runs."""
    if _OPS:
        return
    lib = torch.library.Library("glyd", "FRAGMENT")
    lib.define("vllm_linear(Tensor x, Tensor data, Tensor a, Tensor b, int[] words, Tensor? bias, int out_features, bool exact) -> Tensor")

    def linear(x, data, a, b, words, bias, out_features, exact):
        K = x.shape[-1]
        if exact:  # bf16's own GEMM (UnquantizedLinearMethod.apply: its linear backend, or batch-invariant) on the matrix
            return _BF16[0].apply(types.SimpleNamespace(weight=_unpack(data, a, b, words, out_features, K)), x, bias)
        torch._check(x.dtype == torch.bfloat16, lambda: f"glyd: bf16 activations, not {x.dtype}")
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        if M and _decoded(data.device.index, words, out_features, K, M):
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


def _drop(param):
    """A bf16 weight that has been packed: its memory goes back to PyTorch's allocator now, not when vLLM lets go of its
    Parameter (it keeps it, with the loads it replayed, until the layer is done)."""
    if param is not None:
        param.data = param.data.new_empty(0)


def _trim():
    """PyTorch's unused blocks handed back to the driver where they outweigh the driver's free memory. vLLM loads with the
    allocator's max_split_size_mb at 20, so a block past 20 MiB is never split: the packing's blocks of many sizes pile up
    unused until a cudaMalloc fails, the allocator frees them all and warns (hundreds of times on a 16 GB card)."""
    if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > torch.cuda.mem_get_info()[0]:
        torch.cuda.empty_cache()


_NVCC_WARNED = []  # (one warning a process)


def _warn_no_nvcc():
    """One warning where vLLM's FlashInfer sampler (its default for top-k and top-p, which Qwen3's own sampling settings
    use) has no nvcc to compile with: FlashInfer builds its sampling kernels on the first request that samples, found
    through CUDA_HOME, PATH, then /usr/local/cuda (flashinfer.jit.cpp_ext.get_cuda_path), and vLLM's warmup stops with
    "Could not find nvcc". Not where a flashinfer-jit-cache package has them built. vLLM's own settings stay as they are."""
    if _NVCC_WARNED or not envs.VLLM_USE_FLASHINFER_SAMPLER or importlib.util.find_spec("flashinfer") is None:
        return
    _NVCC_WARNED.append(1)
    home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    if shutil.which("nvcc") or os.path.isfile(os.path.join(home, "bin", "nvcc")) or importlib.util.find_spec("flashinfer_jit_cache"):
        return
    log.warning("glyd: no nvcc (not on PATH, nor in CUDA_HOME or /usr/local/cuda) and vLLM's FlashInfer sampler is on: it compiles its top-k and top-p kernels with nvcc on the first request that samples, and vLLM's warmup stops with 'Could not find nvcc'. Start the server with VLLM_USE_FLASHINFER_SAMPLER=0 (vLLM then samples with PyTorch and Triton), or install flashinfer-jit-cache (FlashInfer's precompiled kernels) or the CUDA toolkit")


def register():
    """The "glyd" quantization method and its op, in this process (again: the same). vllm_entry.register() calls it
    where vLLM is the release this is tested with."""
    _define()
    register_quantization_config(NAME)(GlydConfig)


def _linear_bytes(c):
    """A model's Linears and experts (bf16 bytes), the rest (its embeddings and LM head), and whether it has experts,
    from its (text) config c (for best_layout's fit and the memory check); (0, 0, False) where unknown. The Linears'
    count takes a gated MLP (gate, up and down; a mixture of experts' layer its E experts' matrices,
    moe_intermediate_size each, else intermediate_size): at most a third over for a model without a gate."""
    try:
        h, L = c.hidden_size, c.num_hidden_layers
        E = next((getattr(c, n) for n in ("num_experts", "num_local_experts", "n_routed_experts") if getattr(c, n, None)), 0)
        I = (getattr(c, "moe_intermediate_size", None) or c.intermediate_size) * E if E else c.intermediate_size
        nh = c.num_attention_heads
        nkv = getattr(c, "num_key_value_heads", None) or nh
        hd = getattr(c, "head_dim", None) or h // nh
        per = h * (nh + 2 * nkv) * hd + nh * hd * h + 3 * h * I
        other = c.vocab_size * h * (1 if getattr(c, "tie_word_embeddings", False) else 2)
        return 2 * L * per, 2 * other, bool(E)
    except (AttributeError, TypeError):
        return 0, 0, False


def _moe_decode_min():
    """Tokens a step from which a mixture of experts' layer decodes the experts its tokens are routed to and runs vLLM's
    Triton kernel on them, instead of the library's grouped products (fused mode): GLYD_MOE_DECODE_MIN, else
    MOE_DECODE_MIN; None (a negative number): never."""
    v = os.environ.get("GLYD_MOE_DECODE_MIN", "").strip()
    try:
        n = int(v) if v else MOE_DECODE_MIN
    except ValueError:
        raise ValueError(f"glyd: GLYD_MOE_DECODE_MIN {v!r}: a number of tokens a step (negative: never)") from None
    return None if n < 0 else n


def _estimate(lin, other, layout, n, draft=False, fraction=1.0):
    """A model's weights a GPU packed in layout, bytes, and at least (the Linears' count less a gate's third), over n
    GPUs: its Linears (lin, bf16 bytes) packed, but for the share (1 - fraction) left bf16, and the rest (other) as they
    are; a draft's Linears alone (it shares its target's embeddings; its LM head is its own smaller one)."""
    other = 0 if draft else other
    packed, plain = lin * fraction, lin * (1 - fraction)
    return (packed * BITS[layout] / 16 + plain + other) / n, (packed * 2 / 3 * BITS[layout] / 16 + plain + other) / n


def _missing(kind, loaded, n=0):
    """The pieces of a weight a checkpoint did not give, from its loader calls' shard ids (loaded: loaded_shard_id each;
    None, the tensor whole): kind "qkv" (q, k, v), "merged" (its n members by number) or "one"."""
    got = set()
    for s in loaded:
        got.update(s if isinstance(s, tuple) else (s,))
    if None in got:
        return []
    want = {"q", "k", "v"} if kind == "qkv" else set(range(n)) if kind == "merged" else {None}
    return sorted(want - got, key=str)


def _check_loaded(layer, name):
    """Refused where the checkpoint did not give every piece of layer's weight (held on the meta device: vLLM's layerwise
    processing hands over fresh memory for it, then replays the loads it recorded; their shard ids tell the pieces, a
    merged qkv's q, k, v or a merged Linear's members). Not the bias, a tensor of its own on the GPU that vLLM's loaders
    fill as they do bf16's (its first piece before the layer's loads are recorded); nor a layer whose weight's shard ids
    are another's, which vLLM's record cannot tell apart."""
    kind, n = ("qkv", 0) if isinstance(layer, QKVParallelLinear) else ("merged", len(layer.output_sizes)) if isinstance(layer, MergedColumnParallelLinear) else ("one", 0)
    ids = [args.arguments.get("loaded_shard_id") for p, args in get_layerwise_info(layer).loaded_weights if p == "weight"]
    if kind == "one" and any(s is not None for s in ids):
        return
    miss = ["weight" if s is None else f"weight {s}" for s in _missing(kind, ids, n)]
    if miss:
        raise ValueError(f"glyd: {name}: the checkpoint has no {', '.join(miss)} for it: its pack would hold memory never written")


def _experts_missing(loaded, E):
    """A mixture of experts' layer's pieces the checkpoint did not give, from its loader calls' (expert_id, shard_id)
    (w1, w3: w13's gate and up; w2), for its E experts; None where the loader took another way (not w1, w2, w3)."""
    got = set(loaded)
    if not {s for _, s in got} <= {"w1", "w2", "w3"}:
        return None
    return [f"expert {e}'s {s}" for e, s in sorted({(e, s) for e in range(E) for s in ("w1", "w3", "w2")} - got)]


class GlydConfig(QuantizationConfig):
    """The "glyd" quantization method: the options given (a checkpoint's or --hf-overrides' quantization_config), a
    glyd save's glyd.json where the model is one, and the options in effect once vLLM makes the model (resolve)."""

    def __init__(self, layout=None, exact=None, verify=None, fraction=None):
        super().__init__()
        self.given = {k: v for k, v in (("layout", layout), ("exact", exact), ("verify", verify), ("fraction", fraction)) if v is not None}
        self.manifest, self.dir, self.hf_config = None, None, None  # a glyd save's glyd.json and directory; the model's config
        self.opts = None  # {"layout", "exact", "verify", "fraction"}: resolved at the model's first Linear
        self.compiled, self.need, self.free, self.files_checked = False, 0, 0, False

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
        bad = [k for k in config if k not in OPTIONS + IGNORED]
        if bad:
            raise ValueError(f"glyd: the quantization_config's {', '.join(map(repr, bad))}: not an option ({', '.join(OPTIONS)})")
        return cls(config.get("layout"), config.get("exact"), config.get("verify"), config.get("fraction"))

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        """The model this config is for (vLLM's model or, for a draft asked to be packed, the draft's): its config, and
        where it is a glyd save its glyd.json, beside its weights (a Hub repo's fetched into its snapshot); refused
        where its family's saves are not checked in vLLM yet."""
        self.hf_config = hf_config
        d = model_name
        if not os.path.isdir(d):
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            try:
                d = os.path.dirname(hf_hub_download(model_name, fmt.MANIFEST, revision=revision))
            except EntryNotFoundError:  # none: a bf16 checkpoint
                d = None
        self.manifest = fmt.read_manifest(d) if d else None
        if self.manifest is not None:
            self.dir = d
            arch = (getattr(hf_config, "architectures", None) or [None])[0]
            if arch not in SAVES:
                src = (self.manifest.get("source") or {}).get("repo")
                raise ValueError(f"glyd: a glyd save of {arch} does not load in vLLM yet (saves of {', '.join(SAVES)} do): serve its bf16 checkpoint{f' ({src})' if src else ''} with --quantization glyd, which packs it as it loads")
        # Refused here too, in the engine's process before its workers start (vLLM 0.30 makes this config inside
        # VllmConfig.__post_init__, the config there to read): over several GPUs a worker's refusal reaches the engine
        # as "WorkerProc initialization failed". The workers' resolve refuses the same, with what only they know.
        vc = _building_config()
        if vc is not None and getattr(vc.model_config, "hf_config", None) is hf_config and isinstance(vc.additional_config, dict):
            o = _options(vc.additional_config.get(KEY) or {}, self.given, _env())
            _refusals(vc, o[1], self.manifest, o[3])  # (the model's own, not a draft's)
            _warn_no_nvcc()

    def resolve(self):
        """The options in effect (_options); the layout "auto" best_layout's for this GPU (a save's own where that is
        the one). What Glyd does not do yet refused, with why, and a model whose packs cannot fit the GPU. The options
        written into vLLM's additional_config (its compile cache's key)."""
        vc = get_current_vllm_config_or_none()
        if vc is None:
            raise RuntimeError("glyd: vLLM's config is not set where the model is made")
        if not isinstance(vc.additional_config, dict):
            raise ValueError("glyd: vLLM's additional_config is not a dict: Glyd keys vLLM's compile cache by it")
        layout, exact, verify, fraction = _options(vc.additional_config.get(KEY) or {}, self.given, _env())
        pc, mc = vc.parallel_config, vc.model_config
        _refusals(vc, exact, self.manifest, fraction)
        # A mixture of experts' shared experts run on a side stream beside the rest by default: their products and the
        # main stream's would share the device's done counters. Off, before vLLM makes them (and caches its env), where
        # anything is packed (fraction 0 is bf16 as vLLM runs it).
        if fraction > 0:
            os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
        cc = vc.compilation_config
        self.compiled = not (mc.enforce_eager or (cc.mode == CompilationMode.NONE and cc.cudagraph_mode == CUDAGraphMode.NONE))
        if not self.manifest and pc.tensor_parallel_size > 1:
            log.warning("glyd: tensor parallel %d: each rank packs its own shard (not measured yet)", pc.tensor_parallel_size)
        try:
            dev = torch.cuda.current_device()
            if fraction > 0:
                _gpu(dev)  # the library, loaded now: not at the first forward (fraction 0 packs nothing: it stays unloaded)
        except Exception as e:
            raise RuntimeError(f"glyd: Glyd's GPU library did not load ({type(e).__name__}: {e}): a CUDA build of PyTorch and the glyd wheel's library (or GLYD_GPU_LIB) are needed") from e
        hf = self.hf_config  # (this config's model: a draft's own, not the target's vc.model_config)
        tc = hf.get_text_config() if hf is not None and hasattr(hf, "get_text_config") else hf if hf is not None else mc.hf_text_config
        lin, other, moe = _linear_bytes(tc)
        n = max(1, pc.tensor_parallel_size) * max(1, pc.pipeline_parallel_size)
        packed = int(lin * fraction)  # (the Linears packed, bf16 bytes: the rest stay as vLLM runs them)
        if layout == "auto":
            layout = self.manifest["layout"] if self.manifest else g.best_layout(packed // n, (other + lin - packed) // n, 1, dev, moe=moe)[0]
        # Free: the driver's, and what PyTorch holds cached but unused (a draft is made after the target's packing freed
        # its temporaries into PyTorch's cache). A draft (a config of its own, not vLLM's model's) counted by its
        # Linears alone: it shares the target's embeddings, and its LM head is its own smaller one.
        self.free = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        draft = hf is not None and hf is not getattr(mc, "hf_config", None)
        self.need, low = _estimate(lin, other, layout, n, draft=draft, fraction=fraction)
        if lin and low > self.free:  # (they cannot fit: refused here, not by an OutOfMemoryError deep in the load)
            tiered = f"; the tiered layout (layout mma, {BITS['mma']} bits) takes about {_estimate(lin, other, 'mma', n, draft=draft, fraction=fraction)[0] / 2**30:.1f} GiB" if layout == "mma12" else ""
            share = f", fraction {fraction:g} of the layers, the rest bf16" if fraction < 1 else ""
            raise ValueError(f"glyd: the model's weights take about {self.need / 2**30:.1f} GiB a GPU packed in the {layout} layout ({BITS[layout]} bits a weight{share}), at least {low / 2**30:.1f}, and the GPU has {self.free / 2**30:.1f} GiB free{tiered}: serve it over more GPUs (--tensor-parallel-size), or a smaller model")
        if exact:
            _BF16[:] = [UnquantizedLinearMethod()]  # (vLLM's config current: its linear backend)
        self.opts, self._vc = {"layout": layout, "exact": exact, "verify": verify, "fraction": fraction}, vc
        vc.additional_config[KEY] = {**self.opts, "glyd": __version__, "packs": _digest()}  # (a draft's resolve keeps the packs made)
        L = getattr(tc, "num_hidden_layers", 0) or 0
        log.info("glyd: %s layout%s%s%s", layout, ", exact" if exact else "", ", verified" if verify else "", f", {sum(_packs(i, fraction) for i in range(L))} of {L} layers packed (fraction {fraction:g})" if fraction < 1 else "")

    def check_files(self):
        """verify: a glyd save's files against its glyd.json, its tensors saved as they are by their sha256 (once, when
        the first pack is loaded: the weights are there by then, a Hub repo's too)."""
        if self.files_checked or self.manifest is None or not self.opts["verify"]:
            return
        checked, unchecked = fmt.check_files(self.dir, self.manifest)
        self.files_checked = True
        log.info("glyd: %s: %d tensors saved as they are checked by sha256%s", self.dir, checked, f", {unchecked} not (saved before glyd 0.25)" if unchecked else "")

    def get_quant_method(self, layer, prefix):
        if RoutedExperts is not None and isinstance(layer, RoutedExperts):  # a mixture of experts' layer
            if self.opts is None:
                self.resolve()
            if not self.packs(prefix):  # (fraction: this layer's experts stay as vLLM runs them)
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
                raise RuntimeError("glyd: VLLM_DISABLE_SHARED_EXPERTS_STREAM is not on in this process (vLLM read its environment before Glyd set it): set VLLM_DISABLE_SHARED_EXPERTS_STREAM=1, since shared experts on a side stream would share the GPU's done counters with the rest")
            why = GlydMoEMethod.unsupported(layer)
            if why:
                log.warning("glyd: %s: experts kept bf16 (%s)", prefix, why)
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return GlydMoEMethod(self, layer.moe_config)
        if isinstance(layer, VocabParallelEmbedding):  # (the LM head too): a save's pack decoded to bf16 at load
            e = self.saved(prefix)
            if e is None:
                return None
            if self.opts is None:
                self.resolve()
            return GlydSavedEmbeddingMethod(self, e)
        if not isinstance(layer, LinearBase):
            return None
        if self.opts is None:
            self.resolve()
            if _NO_MOE:
                log.warning("glyd: vLLM's mixture of experts internals did not import (%s): a model's experts stay bf16", _NO_MOE)
        if not self.packs(prefix):  # (fraction: this layer's Linears stay as vLLM runs them without a quantization)
            return UnquantizedLinearMethod()
        return GlydLinearMethod(self, self.saved(prefix))

    def packs(self, prefix):
        """Whether the layer at vLLM's prefix is packed (fraction): its decoder layer's turn (_packs), or, outside the
        numbered layers, only where every layer is."""
        i, f = _layer_of(prefix), self.opts["fraction"]
        return f >= 1 if i is None else _packs(i, f)

    def saved(self, prefix):
        """A glyd save's manifest entry for vLLM's layer prefix, or None: its pack as saved (q, k, v and gate, up are one
        pack each, under q_proj's and gate_proj's paths, as vLLM's qkv_proj and gate_up_proj hold them; else the
        layer's own path, as a family that keeps them fused on disk saves them). Refused where the pack holds another
        count of tensors than vLLM's layer."""
        if self.manifest is None:
            return None
        packs = self.manifest["packs"]
        e, n = packs.get(prefix), 1
        for merged, first, members in (("qkv_proj", "q_proj", 3), ("gate_up_proj", "gate_proj", 2)):
            if prefix.endswith("." + merged) and packs.get(prefix[: -len(merged)] + first) is not None:
                e, n = packs[prefix[: -len(merged)] + first], members
        if e is not None and len(e["tensors"]) != n:
            raise ValueError(f"glyd: {prefix}: the save's pack holds {len(e['tensors'])} tensors, vLLM's layer {n}")
        return e

    def packed(self, name, layout, words, t):
        """A layer packed (name: its prefix): into the process's digest of the packs, which vLLM's compile cache keys by
        (additional_config)."""
        _PACKS[name] = [layout, list(words), [int(x.numel()) for x in t]]
        vc = getattr(self, "_vc", None) or get_current_vllm_config_or_none()
        if vc is not None and isinstance(vc.additional_config, dict) and KEY in vc.additional_config:
            vc.additional_config[KEY]["packs"] = _digest()

    def out_of_memory(self, name, e):
        """An OutOfMemoryError while packing name, with what was packed so far and the model's estimate."""
        return torch.OutOfMemoryError(f"glyd: {name}: out of GPU memory packing it, after {len(_PACKS)} layers packed; the model's weights take about {self.need / 2**30:.1f} GiB a GPU packed in the {self.opts['layout']} layout, with {self.free / 2**30:.1f} GiB free before loading: serve it over more GPUs (--tensor-parallel-size), in the tiered layout (layout mma), or a smaller model ({e})")


def _env():
    return {"layout": os.environ.get("GLYD_LAYOUT"), "exact": os.environ.get("GLYD_EXACT"), "verify": os.environ.get("GLYD_VERIFY"), "fraction": os.environ.get("GLYD_FRACTION")}


def _building_config():
    """The VllmConfig vLLM is building where it makes a quantization config (vLLM 0.30: in VllmConfig.__post_init__, in
    the engine's process), found up the calling frames; None elsewhere (a draft's config, made in a worker)."""
    f = sys._getframe(2)
    while f is not None:
        obj = f.f_locals.get("self")
        if isinstance(obj, VllmConfig):
            return obj
        f = f.f_back
    return None


def _digest():
    return hashlib.sha256(json.dumps(sorted(_PACKS.items())).encode()).hexdigest()[:16] if _PACKS else ""


def _refusals(vc, exact, manifest, fraction=1.0):
    """What Glyd does not do yet, refused with why: vLLM's config (vc), exact mode, a glyd save's glyd.json (and with it
    a fraction under 1: its layers are packed on disk)."""
    pc, mc = vc.parallel_config, vc.model_config
    if envs.VLLM_BATCH_INVARIANT and not exact:
        raise ValueError("glyd: VLLM_BATCH_INVARIANT asks for every product's bits not to depend on the batch, and Glyd's fused kernels are chosen by the batch's tokens: serve with exact (vLLM's batch-invariant GEMM on the decoded weights), or without VLLM_BATCH_INVARIANT")
    if pc.use_ubatching:
        raise ValueError("glyd: dual-batch overlap (--enable-dbo, ubatching) runs two batches' products at once on two streams, which Glyd's kernels do not share a GPU's done counters for yet: run without it")
    if vc.lora_config is not None:
        raise ValueError("glyd: LoRA on packed layers is not supported yet: serve without --enable-lora, or without --quantization glyd")
    oc = vc.offload_config
    if oc is not None and (oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0):
        raise ValueError("glyd: weight offloading (--cpu-offload-gb, prefetch offload) moves parameters, not Glyd's packs: not supported")
    if getattr(mc, "enable_sleep_mode", False):
        raise ValueError("glyd: sleep mode is not supported yet")
    icc, cc = vc.compilation_config.inductor_compile_config, vc.compilation_config
    # (in the engine's process vLLM has not settled the compile mode yet: unset counts as compiled, as it will be
    # unless eager)
    eager = mc.enforce_eager or (cc.mode == CompilationMode.NONE and cc.cudagraph_mode in (None, CUDAGraphMode.NONE))
    if exact and not (eager or (icc.get("deterministic") and icc.get("benchmark_combo_kernel") is False)):
        raise ValueError("glyd: exact mode gives vLLM's bf16 logits bit for bit eager (--enforce-eager), or compiled with inductor's deterministic mode, with which vLLM's compiled bf16 is itself the same from one run to the next: --compilation-config '{\"inductor_compile_config\": {\"deterministic\": true, \"combo_kernels\": true, \"benchmark_combo_kernel\": false}}' (the bf16 run to compare with the same). Without it inductor times some of its kernels' variants on the GPU, and compiled logits, bf16's too, are not always the same from one run to the next. Add one of the two, or leave exact off")
    if manifest is not None:
        if fraction < 1:
            raise ValueError(f"glyd: fraction {fraction:g} leaves some layers bf16, and a glyd save's layers are packed on disk: serve its bf16 checkpoint with a fraction (packed as it loads), or the save with fraction 1")
        if pc.tensor_parallel_size > 1 or pc.pipeline_parallel_size > 1:
            raise ValueError("glyd: a glyd save loads on one GPU for now (tensor and pipeline parallel: from its bf16 checkpoint)")
        if any("experts" in e for e in manifest["packs"].values()):
            raise ValueError("glyd: a save with a mixture of experts' packs (glyd-v2) is not supported in vLLM yet: load its bf16 checkpoint")


def _take_whole(param, loaded_weight, *shard):
    """A saved pack's tensor, whole and as it is (a merged group's under its first member's name: its shard ignored)."""
    param.data = loaded_weight.to(param.device)


def _placeholders(layer, layout):
    """A save's pack tensors as the layer's parameters, of the save's dtypes, their sizes the save's as they load."""
    for name in BUFFERS[layout]:
        p = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8 if fmt.DTYPES[name[5:]] == "U8" else torch.int32), requires_grad=False)
        set_weight_attrs(p, {"weight_loader": _take_whole})
        layer.register_parameter(name, p)


def _saved_pack(layer, e, verify, out=None):
    """A save's pack from the tensors loaded into layer (its words from glyd.json's entry e). With verify, or an out
    (bf16, the pack's rows by its columns), decoded (into out where given) and, with verify, checked by its sha256s."""
    data, a, b = (getattr(layer, n).data for n in BUFFERS[e["layout"]])
    if not data.numel():
        raise ValueError(f"glyd: {getattr(layer, 'prefix', '')}: the save's pack did not load")
    shape = tuple(e["shape"])
    p = g.Mma12(shape, data, a, b, int(e["hb"])) if e["layout"] == "mma12" else g.Mma(shape, data, a, b, [int(x) for x in e["tiers"]])
    if verify or out is not None:
        from .model import sha256

        w = g.mma_unpack(p, None if out is None else out.view(-1))
        r = 0
        for t in e["tensors"] if verify else ():
            if sha256(w[r : r + t["shape"][0]]) != t["sha256"]:
                raise ValueError(f"glyd: {t['name']} decodes to other bits than glyd.json's sha256")
            r += t["shape"][0]
    return p


class GlydSavedEmbeddingMethod(UnquantizedEmbeddingMethod):
    """A glyd save's packed LM head (a model's own, not tied to its embeddings) or embedding: the pack's tensors loaded
    as saved, then decoded into the bf16 weight vLLM runs it on, as it runs a bf16 checkpoint's (the LM head stays
    vLLM's for now)."""

    def __init__(self, config, saved):
        super().__init__()
        self.config, self.saved = config, saved

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra):
        super().create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra)
        _placeholders(layer, self.saved["layout"])

    def process_weights_after_loading(self, layer):
        if getattr(layer, "glyd_decoded", False):  # (vLLM calls it again after the load)
            return
        self.config.check_files()
        V, K = self.saved["shape"]
        if K != layer.weight.shape[1] or V > layer.weight.shape[0]:
            raise ValueError(f"glyd: {getattr(layer, 'prefix', '')}: the save's pack is {[V, K]}, vLLM's layer {list(layer.weight.shape)}")
        _saved_pack(layer, self.saved, self.config.opts["verify"], out=layer.weight.data[:V])  # (decoded in place)
        layer.weight.data[V:].zero_()  # (the vocabulary's padding)
        for name in BUFFERS[self.saved["layout"]]:
            layer._parameters.pop(name, None)
        layer.glyd_decoded = True
        super().process_weights_after_loading(layer)


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
            _placeholders(layer, self.saved["layout"])
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
        opts, name = self.config.opts, getattr(layer, "prefix", "")
        layout = opts["layout"]
        if opts["exact"] and self.config.compiled and getattr(layer, "bias", None) is not None:
            raise ValueError(f"glyd: {name} has a bias, and exact mode under torch.compile is not bit for bit there yet: inductor adds a bf16 Linear's bias apart from its matmul, rounding before the add, where exact's product adds it in the GEMM. Serve exact with --enforce-eager (vLLM's bf16 eager's bits), or compiled without exact")
        pack = g.pack_mma12 if layout == "mma12" else g.pack_mma
        try:
            if self.saved is not None:
                self.config.check_files()
                p = _saved_pack(layer, self.saved, opts["verify"])
                dev = p.data.device
                if isinstance(p, g.Mma12) != (layout == "mma12"):  # another layout asked for: decoded and packed again
                    w = g.mma_unpack(p)
                    p = pack(w)
                    if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
                        raise ValueError(f"glyd: {name}: the save packed again in the {layout} layout decoded to other bits than the save")
                    del w
            else:
                _check_loaded(layer, name)
                w = layer.weight.data
                dev = w.device
                p = pack(w)
                if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
                    raise ValueError(f"glyd: {name} decoded to other bits than its weights")
        except torch.OutOfMemoryError as e:
            raise self.config.out_of_memory(name, e) from e
        g.lib()  # the library loaded (the op's calls are its C API's)
        t = (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base)
        for n in {n for names in BUFFERS.values() for n in names}:
            layer._parameters.pop(n, None)
        for n, x in zip(("glyd_data", "glyd_a", "glyd_b"), t):
            layer.register_buffer(n, x, persistent=False)
        O, K = p.shape
        layer.glyd_words = list(p.sym if isinstance(p, g.Mma12) else p.tiers)
        layer.glyd_out, layer.glyd_exact, layer.glyd_verified = O, opts["exact"], opts["verify"]
        _drop(layer._parameters.pop("weight", None))
        w = None
        layer.weight = torch.nn.Parameter(torch.empty(0, K, dtype=torch.bfloat16, device=dev), requires_grad=False)  # (read for its dtype and K)
        self.config.packed(name, layout, layer.glyd_words, t)
        need = O * K if opts["exact"] or self._decodes(dev, layer.glyd_words, O, K) else 0
        if need and (dev not in _SCRATCH or _SCRATCH[dev].numel() < need):
            _SCRATCH[dev] = torch.empty(need, dtype=torch.bfloat16, device=dev)  # (at load: no CUDA graph holds the old one)
        for n in ("mma12_linear", "mma_linear"):  # the device's done counters, made now: never in a CUDA graph's pool
            _lib._counters(n, dev.index, None, 0, _lib._UNITS)
        _trim()

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


class GlydMoEMethod(OnlineMoEMethodBase):
    """A mixture of experts' layer's experts (vLLM's RoutedExperts: w13 [E, 2I, H], gate then up, and w2 [E, H, I]),
    each weight packed as one matrix of its E experts' stacked ([E 2I, H], [E H, I]) as vLLM's layerwise processing
    completes the layer, from its bf16 on the meta device (the peak: the packs and a layer's experts), once every
    expert's pieces have loaded (else refused, naming the layer). Their products the library's grouped ones: each
    token's k choices (its pairs) sorted by expert on the GPU (moe_route), gate and up with SiLU applied as its sums
    are written out, down with the router's weights applied and each token's k rows added (mma_moe); no host sync, in
    the CUDA graphs vLLM captures around its MoE op. From decode_min tokens a step (_moe_decode_min; exact: always)
    the experts the tokens are routed to are decoded into the device's scratch buffer (mma_moe_unpack) instead, then run
    by the kernel vLLM runs bf16's experts by (its Triton one; where vLLM picks another, whose weights it lays out
    otherwise, exact is refused and fused mode keeps the grouped products throughout)."""

    def __init__(self, config, moe):
        super().__init__(moe)
        self.config, self.ref = config, None
        self.decode_min = 0 if config.opts["exact"] else _moe_decode_min()
        if self.decode_min is not None:
            ref = UnquantizedFusedMoEMethod(moe)  # (bf16's: the backend vLLM picks for this layer)
            if ref.unquantized_backend == UnquantizedMoeBackend.TRITON:
                self.ref = ref
            elif config.opts["exact"]:
                raise ValueError(f"glyd: exact mode runs a mixture of experts' layers by vLLM's Triton kernel on their experts decoded; vLLM picks {ref.unquantized_backend.value} for bf16's here, whose weights it lays out otherwise: leave exact off")
            else:
                self.decode_min = None

    @property
    def topk_indices_dtype(self):
        return self.ref.topk_indices_dtype if self.ref is not None else None

    @staticmethod
    def unsupported(layer):
        """Why the library's grouped products do not take this layer's experts (they stay bf16), or None."""
        mc = layer.moe_config
        if mc.moe_parallel_config.use_ep:
            return "expert parallel"
        if mc.has_bias:
            return "experts with biases"
        if mc.activation != MoEActivation.SILU:
            return f"activation {mc.activation.value}, not SiLU"
        if layer.apply_router_weight_on_input:
            return "router weights applied to the input"
        if layer.params_dtype != torch.bfloat16:
            return f"{layer.params_dtype}, not bf16"
        H, I = mc.hidden_dim, mc.intermediate_size_per_partition
        if H != getattr(mc, "hidden_dim_unpadded", H) or I != getattr(mc, "intermediate_size_per_partition_unpadded", I):
            return f"hidden {H} or intermediate {I} padded for this backend"
        if (2 * I) % 128 or H % 64 or I % 16:
            return f"hidden {H} or intermediate {I} (a rank's) off the packs' multiples (2I of 128, H of 64, I of 16)"
        return None

    @property
    def supports_eplb(self):
        return False

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make()  # (bf16 activations, as vLLM's own bf16 experts)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "glyd_moe", None) is not None:  # (vLLM calls it again after the load)
            return
        opts = self.config.opts
        layout = opts["layout"]
        pack = g.pack_mma12 if layout == "mma12" else g.pack_mma
        E = layer.w13_weight.shape[0]
        info = get_layerwise_info(layer)
        miss = _experts_missing([(args.arguments.get("expert_id"), args.arguments.get("shard_id")) for p, args in info.loaded_weights if p in ("w13_weight", "w2_weight")], E)
        if miss is None and info.load_numel_total and info.load_numel < info.load_numel_total:  # (another loader's way: vLLM's count)
            miss = [f"{info.load_numel_total - info.load_numel:,} of its experts' weights"]
        if miss:
            raise ValueError(f"glyd: {layer.layer_name}: the checkpoint has no {', '.join(miss[:8])}{' ...' if len(miss) > 8 else ''} for it: its packs would hold memory never written")
        packs = []
        for name in ("w13", "w2"):
            w = getattr(layer, name + "_weight").data
            m = w.reshape(-1, w.shape[2])  # E experts' matrices stacked, [E out, in]
            try:
                p = pack(m)
                if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), m.view(torch.int16)):
                    raise ValueError(f"glyd: {layer.layer_name}'s {name} decoded to other bits than its weights")
            except torch.OutOfMemoryError as e:
                raise self.config.out_of_memory(f"{layer.layer_name}'s {name}", e) from e
            t = (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base)
            for part, x in zip(("data", "a", "b"), t):
                layer.register_buffer(f"glyd_{name}_{part}", x, persistent=False)
            dev = w.device
            _drop(layer._parameters.pop(name + "_weight", None))
            w = m = None
            setattr(layer, name + "_weight", torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=dev), requires_grad=False))
            self.config.packed(f"{layer.layer_name}.{name}", layout, list(p.sym if isinstance(p, g.Mma12) else p.tiers), t)
            packs.append(p)
        g.lib()
        layer.glyd_moe, layer.glyd_verified = (packs[0], packs[1], E), opts["verify"]
        for name in ("mma12_moe", "mma_moe"):  # the device's done counters, made now: never in a CUDA graph's pool
            _lib._counters(name, dev.index, None, 0, 1 << 16)
        if self.ref is not None:  # (the decoded route) bf16's kernel (it takes the weights at each call), the scratch for both
            self.ref._init_moe_kernel(layer)
            need = sum(p.shape[0] * p.shape[1] for p in packs)
            if dev not in _SCRATCH or _SCRATCH[dev].numel() < need:
                _SCRATCH[dev] = torch.empty(need, dtype=torch.bfloat16, device=dev)  # (at load: no CUDA graph holds the old one)
        _trim()

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        up, down, E = layer.glyd_moe
        if not x.shape[0]:
            return torch.empty_like(x)
        fresh, _lib.local.fresh = _lib.local.fresh, True  # workspaces for the call alone (in a capture: the graph's pool)
        try:
            ids = topk_ids if topk_ids.dtype == torch.int64 else topk_ids.long()
            plan = g.moe_route(ids, E)
            if self.ref is not None and x.shape[0] >= self.decode_min:  # the routed experts decoded (the rest of the buffer as it was, unread)
                buf, n13 = _SCRATCH[x.device], up.shape[0] * up.shape[1]
                w1 = g.mma_moe_unpack(up, E, plan, ids.numel(), buf[:n13]).view(E, -1, up.shape[1])
                w2 = g.mma_moe_unpack(down, E, plan, ids.numel(), buf[n13 : n13 + down.shape[0] * down.shape[1]]).view(E, -1, down.shape[1])
                return self.ref.moe_kernel.apply(hidden_states=x, w1=w1, w2=w2, topk_weights=topk_weights, topk_ids=topk_ids, activation=layer.activation, apply_router_weight_on_input=layer.apply_router_weight_on_input, global_num_experts=layer.global_num_experts, expert_map=layer.expert_map, shared_experts=shared_experts, shared_experts_input=shared_experts_input)
            h = g.mma_moe(up, E, x, plan, ids, 1)  # [T k, I]: SiLU(gate) up, the pairs in the plan's order
            return g.mma_moe(down, E, h, plan, ids, 0, None, topk_weights, gather=False)  # [T, H]
        finally:
            _lib.local.fresh = fresh
