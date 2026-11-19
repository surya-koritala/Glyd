"""Glyd in vLLM: a model's Linears, a mixture of experts' experts, and its
embedding and LM head, held packed on the GPU, bit for bit, as vLLM loads
them, and multiplied from there by the library's kernels (the embedding's rows
read from the pack).

    pip install "glyd[vllm]"
    vllm serve Qwen/Qwen3.5-9B --quantization glyd                # a bf16 checkpoint, packed as it loads
    vllm serve ./SAVE --quantization glyd                         # a glyd save, its packs as saved (Qwen3ForCausalLM/LlamaForCausalLM saves only; no 2026 model yet)
    vllm serve Qwen/Qwen3.5-9B --quantization glyd --additional-config '{"glyd": {"layout": "mma12"}}'
    vllm serve MODEL --quantization glyd --additional-config '{"glyd": {"kv": "lossless"}}'   # the KV cache in 12 bits a value (vllm_kv.py; head size 128, full attention)
    vllm serve nvidia/Gemma-4-31B-IT-NVFP4 --quantization glyd    # a quantized checkpoint (FP8, NVFP4, MXFP4, INT4, INT8): its quantized layers vLLM's own, the 16-bit parts it keeps packed (vllm_quant.py)
    GLYD_FP8=1 vllm serve Qwen/Qwen3.8-27B-FP8 --quantization glyd   # a development build only: an FP8 checkpoint (fp8, mxfp8, modelopt, compressed-tensors) through vllm_fp8.py;
                                                                     # without GLYD_FP8 (a release build) it is a quantized checkpoint as above (vllm_quant.py)

vLLM calls glyd_gpu.vllm_entry.register() in every process it starts (the
entry point vllm.general_plugins: glyd); where vLLM is the minor release this
is tested with, it calls register() here: the quantization method "glyd".

Options, in --additional-config's "glyd" (or a checkpoint's, or
--hf-overrides', quantization_config; else GLYD_LAYOUT, GLYD_EXACT,
GLYD_VERIFY, GLYD_FRACTION, GLYD_KV); another key, or a flag not true or false,
refused:
- layout: "auto" (best_layout's choice for the GPU: the tiered layout on
  Ada, for a mixture of experts on an A10 too, and where only it fits; the
  12-bit one elsewhere; a quantized checkpoint's 16-bit parts: the tiered
  layout on every GPU, vllm_quant.py), "mma" or "mma12";
- exact: each product's matrix decoded whole, then the GEMM vLLM runs for
  bf16 (UnquantizedLinearMethod's apply: F.linear by default, a FlashInfer
  --linear-backend's, VLLM_BATCH_INVARIANT's), the LM head kept as vLLM runs it (its
  matrix decoded whole would take a buffer of its size, more than packing it saves, and its
  product in pieces is another kernel's bits; a tied model's embedding, which is that matrix,
  stays too, and an untied model's embedding is packed: its rows are read back exactly): its
  logits bf16's bit for bit, eager (--enforce-eager), or compiled with inductor's deterministic
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
  is packed at fraction 1 only, and so are the embedding and the LM head (they stay
  as without it under a fraction). A glyd save's layers are packed on disk: with a save, only fraction 1.
- kv: "auto" (default), "lossless" or "off": the KV cache held in 12 bits a value instead of 16, every value read back bit for bit: an
  attention backend of this plugin (vllm_kv.py: its docstring has what it does, refuses and relies on), its bases calibrated once a model
  on vLLM itself (kvcal.py: the first start runs the model on a bundled text for a minute or two, then the tables are kept in $GLYD_CACHE/kv;
  GLYD_KV_TABLES names a tables file). It works with any fraction, 0 included (bf16 weights, the KV cache compressed). With verify
  every write is checked against a bf16 shadow of the cache (a small cache: --num-gpu-blocks-override). "auto" is lossless on an A100, an L4, an H100 and a GH200
  (the GPUs it is measured faster than vLLM's own cache on) when nothing refuses it and the tables cover the model's window
  (--max-model-len up to 40,960), else vLLM's own KV cache, with one line that says why; "lossless" is the user's own say on any GPU the
  backend holds, refused with why where it is not held, except for a window the tables do not cover, which vLLM's own cache serves, whole, with a line.
- cc12: true or false (default false; no environment variable): on a GPU of compute capability 12 (the RTX 50 series, the RTX PRO
  Blackwell cards), the mma layout's Linears take, at 9-64 tokens a step, the products measured faster there than the default ones
  (an RTX 5090, 2026 models' matrices: benchmarks/gpu/tiered-speed-2026-10-06), the same bits. Off by default until measured
  end to end; no effect on another GPU or in the mma12 layout.
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
cuBLAS (DECODE, AHEAD) the matrix decoded into a buffer, then F.linear (the
buffer is made at load, and kept, only where a CUDA graph can capture the
decode: exact's, or a route that decodes for a count of tokens vLLM captures;
else a call makes its own, and its bytes are the KV cache's). vLLM's
torch.compile takes it as one node, its CUDA graphs capture its kernels. A mixture of experts' experts: GlydMoEMethod.
The embedding and the LM head: GlydEmbeddingMethod, packed as the Linears, the
lookup an op (glyd::vllm_embed: the library's mma_rows, any ids in one launch, no buffer
of the table's size) and the head's product another (glyd::vllm_logits, the library's
product for the rows of the step, never the matrix decoded whole); a tied head's buffers
are the embedding's (one pack), and where the config ties them with a copy in the checkpoint
(vLLM builds them apart) the second to be packed takes the first's pack if it holds the same
bits. They stay as vLLM runs them (a line in the log says why) under a fraction below 1,
speculative decoding (its draft shares them), exact for the LM head (and so for a tied
table), a head_dtype other than bf16, an embedding class of its own, a hidden size
that is not a multiple of 64. A table vLLM builds without the quant config (Qwen3.5's,
Nemotron-H's, LFM2's, Jamba's ... embedding; Qwen3-Next's, Jamba's LM head; LongCat's n-gram
table) is packed by the same rules once the model has loaded (_after_load), and so are the
Linears vLLM builds without it in Jamba's Mamba mixers (MIXERS); a vision or audio tower's,
not yet by default (_towers_on: an image's prefill through them is measured first). A Linear's
rows need not be a multiple of 64 (the pack's are padded with zeros, the product returns the
Linear's). Qwen4Exp's n-gram table, which vLLM refuses under any quantization config but FP8's,
takes vLLM's own bf16 method (_ple_plain). Norms, attention and the KV cache are vLLM's. A speculative
draft: vLLM keeps an EAGLE-3 draft bf16 under --quantization glyd, and packs it
where --speculative-config asks for "quantization": "glyd" (a GlydConfig of
its own, sized by the draft's config, its packs in the process's digest).

vLLM's internals it relies on, beyond its plugin entry point and
register_quantization_config (vLLM 0.30; the entry point refuses another
minor release):
- QuantizationConfig (from_config, maybe_update_config, get_quant_method) and
  QuantizeMethodBase (embedding, tie_weights); LinearBase, LinearMethodBase,
  QKVParallelLinear and MergedColumnParallelLinear (their loaders' shard ids), and
  UnquantizedLinearMethod (exact's GEMM); VocabParallelEmbedding (its padded and per-shard
  sizes, weight_loader, use_fused_embedding) and ParallelLMHead, UnquantizedEmbeddingMethod;
  ModelWeightParameter and set_weight_attrs;
- model_loader.base_loader's process_weights_after_loading (the load's last step, which
  _after_load follows), a multimodal model's get_language_model, the Mamba (v1) mixer's class
  name, and vllm.models.qwen4_exp's Qwen4ExpPLEEmbeddingMethod.from_quant_config;
- the layerwise online processing of model_loader.reload.layerwise
  (initialize_online_processing; get_layerwise_info's loaded_weights and
  load counts, for the pieces loaded);
- get_current_vllm_config_or_none, and the config's additional_config
  (the compile cache's key), parallel_config (tensor and pipeline parallel,
  ubatching), lora_config, offload_config, model_config (enforce_eager,
  enable_sleep_mode, hf_text_config), compilation_config (mode,
  cudagraph_mode, inductor_compile_config) and scheduler_config
  (max_num_batched_tokens), speculative_config and the model config's head_dtype and
  word_embeddings_untied_by_checkpoint; vllm.envs (VLLM_BATCH_INVARIANT,
  VLLM_DISABLE_SHARED_EXPERTS_STREAM, VLLM_USE_FLASHINFER_SAMPLER: read, for
  a warning where FlashInfer's sampler has no nvcc; never set);
- for kv, vllm_kv.py's (its docstring lists them), and here AttentionConfig's backend and backend_per_kind, the cache config's
  enable_prefix_caching, cache_dtype, block_size and kv_offloading_size, the scheduler config's enable_chunked_prefill, speculative_config,
  kv_transfer_config, vllm.platforms.current_platform's device capability, and the Attention layer class (get_quant_method's layer);
- for a mixture of experts (without them, experts stay bf16): RoutedExperts
  (moe_config, layer_name, top_k, activation, expert_map,
  apply_router_weight_on_input), OnlineMoEMethodBase,
  FusedMoEQuantConfig.make, MoEActivation, apply_moe_activation and
  ApplyMoEActivationConfig.from_configs (the activation vLLM's experts apply
  between their two products), and UnquantizedFusedMoEMethod
  (unquantized_backend, _init_moe_kernel, moe_kernel.apply's keyword
  arguments) with UnquantizedMoeBackend.
It copies none of vLLM's code. Under the Business Source License 1.1
(LICENSE), as the rest of Glyd's GPU code.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import types
import weakref
from fractions import Fraction
from math import floor
import torch
import torch.nn.functional as F
from vllm import envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, MergedColumnParallelLinear, QKVParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod, VocabParallelEmbedding
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info, initialize_online_processing
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from . import __version__
from . import _lib, _trace, format as fmt, kernels as g

try:  # an FP8 checkpoint's Linears (vllm_fp8.py): without it (a vLLM whose FP8 internals moved) an FP8 checkpoint is refused, with why
    from . import vllm_fp8

    _NO_FP8 = None
except Exception as _e:  # (any failure of vllm_fp8's import, not ImportError alone: the bf16 path must load whatever vLLM's FP8 internals do)
    vllm_fp8, _NO_FP8 = None, f"{type(_e).__name__}: {_e}"

try:  # a quantized checkpoint's 16-bit parts (vllm_quant.py): without it (a vLLM whose quantization internals moved) such a checkpoint is refused, with why
    from . import vllm_quant

    _NO_QUANT = None
except Exception as _e:  # (any failure of its import: the bf16 path must load whatever vLLM's quantization internals do)
    vllm_quant, _NO_QUANT = None, f"{type(_e).__name__}: {_e}"

try:  # a mixture of experts' internals: without them (another vLLM) a model's experts stay vLLM's bf16
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation, apply_moe_activation, apply_moe_activation_supported
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
SCHEDULER = "glyd_gpu.vllm_sched.make"  # kv lossless: the engine's scheduler (SchedulerConfig.scheduler_cls)
LAYOUTS = ("auto", "mma", "mma12")
KVS = ("auto", "lossless", "off")
# The GPUs, by the name the driver gives (an A100: "NVIDIA A100-SXM4-40GB", "NVIDIA A100 80GB PCIe"; an L4: "NVIDIA L4"; an H100: "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"; a GH200: "NVIDIA GH200 480GB"), that `kv: auto` turns the lossless
# KV cache on for: the ones it was measured faster on than vLLM's own cache, on every load measured, and whose checks (every value back bit for bit, prompt logprobs and `exact` bit for bit vLLM's) pass. Another
# GPU of the same architecture (an L40S, an RTX 40, an H200, an A10) gets it when it is measured: until then `kv: lossless`, the user's own say, on any GPU the backend holds.
KV_AUTO = re.compile(r"\b(A100|L4|H100|GH200)\b")
H100 = re.compile(r"\bH100\b")
A100 = re.compile(r"\bA100\b")
OPTIONS = ("layout", "exact", "verify", "fraction", "kv", "cc12")
IGNORED = ("quant_method", "merge", "verified", "source", "hashed", "unhashed")  # (transformers' glyd config's own)
BITS = {"mma": 10.80, "mma12": 12.04}
# Tokens a step from which a mixture of experts' layer decodes its routed experts for vLLM's Triton kernel instead of the
# library's grouped products: granite-3.1-3b-a800m-instruct's layer on an L4, the decoded route the faster from 1,152
# tokens (2.89 against 2.99 ms; 1,024: 2.84 against 2.75), 0.65x at 8,192 (benchmarks/gpu/l4-vllm-moe-routes-2026-09-30)
MOE_DECODE_MIN = 1152
# A Linear of fewer weights than this stays as vLLM runs it (a router of 128 x 2,048: 0.5 MB, 20 us a product on an H100 at 232 tokens against bf16's 5)
SMALL_LINEAR = 1 << 20
# On Hopper: Qwen3-30B-A3B's layer on an H100 SXM, the decoded route the faster from 2,048 tokens (1,024: 1.15x the time, 2,048: 0.97x;
# benchmarks/gpu/vllm-m6-h100-2026-10-01) than the grouped products of the other GPUs' settings; on an H100, whose grouped products are the
# library's wgmma ones (its settings, chosen by the same layer's sweep, 2026-10-03), from 8,192 (4,096: 0.94x the time of the decoded route's,
# 8,192: 1.25x)
MOE_DECODE_MIN_HOPPER = 2048
MOE_DECODE_MIN_H100 = 8192
# On an A100: Qwen3-30B-A3B's layer on an A100 SXM4, the grouped products (the library's: the 12-bit layout's tiled GEMM at these sizes) the faster up to 2,048 tokens (the decoded route
# takes 1.19x their time at 2,048, 1.45x at 1,152) and the decoded route from about 3,000 (4,096: 0.90x their time, 8,192: 0.71x; benchmarks/gpu/a100-moe-v028-2026-10-03)
MOE_DECODE_MIN_A100 = 3072
SAVES = ("Qwen3ForCausalLM", "LlamaForCausalLM")  # the families whose glyd saves are checked in vLLM (check_vllm.py --saves)
BUFFERS = {"mma": ("glyd_data", "glyd_blocks", "glyd_block_base"), "mma12": ("glyd_data", "glyd_exc", "glyd_exc_base")}  # a save's names
_OPS = []  # the ops' library (they live as long as it does)
_SCRATCH = {}  # a device's buffer matrices are decoded into (bf16) where a CUDA graph can capture their decode: made at load, never replaced after (elsewhere a call makes its own)
_WAYS = {}  # (gpu, words, O, K, M): the library's route
_CC12 = [False]  # the option cc12 (resolve's): the mma layout's products of up to 64 tokens by the route GEMM_CC12
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
    """The options in effect (layout, exact, verify, fraction, kv, cc12): --additional-config's "glyd" (extra), else the given ones
    (a checkpoint's or --hf-overrides' quantization_config), else the environment's (none for cc12); another key refused."""
    if not isinstance(extra, dict):
        raise ValueError(f"glyd: --additional-config's \"glyd\" is a {type(extra).__name__}, not an object of options ({', '.join(OPTIONS)})")
    extra = {k: v for k, v in extra.items() if k not in ("glyd", "packs", "kv_tables", "scheduler_cls")}  # (what resolve writes there: the key)
    bad = [k for k in extra if k not in OPTIONS]
    if bad:
        raise ValueError(f"glyd: --additional-config's \"glyd\": {', '.join(map(repr, bad))}, not an option ({', '.join(OPTIONS)})")
    pick = {k: extra[k] if k in extra else given[k] if k in given else env.get(k) for k in OPTIONS}
    layout = str(pick["layout"] or "auto").lower()
    if layout not in LAYOUTS:
        raise ValueError(f"glyd: layout {pick['layout']!r}: one of {', '.join(LAYOUTS)}")
    kv = str(pick["kv"] or "auto").strip().lower()
    if kv not in KVS:
        raise ValueError(f"glyd: kv {pick['kv']!r}: one of {', '.join(KVS)}")
    return layout, _flag("exact", pick["exact"] or False), _flag("verify", pick["verify"] or False), _fraction(pick["fraction"]), kv, _flag("cc12", pick["cc12"] or False)


def _gpu(d):
    code = _GPU.get(d)
    if code is None:
        with torch.cuda.device(d):
            code = _GPU[d] = _lib.gpu()  # the library's own (glyd_gpu_gpu): its routes' classes by name, whatever they are
    return code


def _way(d, words, O, K, M):
    """The library's route for a product of M tokens on device d."""
    key = (d, len(words), O, K, M)
    r = _WAYS.get(key)
    if r is None:
        r = _WAYS[key] = (_lib.mma12_way if len(words) == 4 else _lib.mma_way)(_gpu(d), O, K, M)[0]
    return r


def _decoded(d, words, O, K, M):
    """Whether a product of M tokens decodes its matrix for cuBLAS: the library's route DECODE or AHEAD (a long prompt on
    a GPU where cuBLAS on the decoded matrix outruns the fused kernel), or a prompt kernel's where K is not a multiple
    of 64 (which it does not take)."""
    way = _way(d, words, O, K, M)
    return way in (g.DECODE, g.EARLY) or (way == g.X1 and K % 64 != 0)


def _way_given(d, words, O, K, M):
    """The route a Linear's product of M tokens is given: GEMM_CC12 for the mma layout's GEMM where the option cc12 asks (the
    library takes it as GEMM on a GPU of another compute capability), else -1 (the library's own)."""
    return g.GEMM_CC12 if _CC12[0] and len(words) == 3 and _way(d, words, O, K, M) == g.GEMM else -1


def _unpack(data, a, b, words, O, K):
    """The matrix, bf16, decoded into a buffer of O * K weights: the device's own where it has one that holds that many (a CUDA graph's capture of
    the decode keeps its address), else one made for the call, its bytes the allocator's again after it (vLLM's memory profile counts a step's peak:
    a buffer that is not held through it is not taken from the KV cache)."""
    buf = _SCRATCH.get(data.device)
    if buf is None or buf.numel() < O * K:
        buf = torch.empty(O * K, dtype=torch.bfloat16, device=data.device)
    w = buf[: O * K].view(O, K)
    (_lib.mma12_unpack if len(words) == 4 else _lib.mma_unpack)(data, a, b, list(words), K, 0, O, w.view(torch.int16), 0)
    return w


def _padded(n):
    """n rows padded to a multiple of 64 (a pack's rows)."""
    return -(-n // 64) * 64


# dev: a prompt's decoded products with each matrix decoded ahead on a second stream, beside the product before its own (GLYD_AHEAD_PROMPT=1:
# the products the route decodes take it; GLYD_AHEAD_PROMPT_WARPS: the decode's warps an SM, 0 the library's own launch). Its products are
# F.linear on the same bits in a buffer of the same shape, the decoded route's bits. A development build's alone, for measurements
# (benchmarks/gpu/prompt-speed-2026-10-06).
AHEAD_ON = os.environ.get("GLYD_AHEAD_PROMPT", "").strip() == "1"
AHEAD_WARPS = int(os.environ.get("GLYD_AHEAD_PROMPT_WARPS", "").strip() or 2)
AHEAD_HOLD = 5000  # ns the decode beside a product waits, so that the product's blocks are placed first (model.py's EARLY_HOLD)
_BIGGEST = {}  # a device's largest packed matrix, in weights (at load): the size of the two buffers


class _Ahead:
    """A device's order of a prompt's decoded products, recorded from the calls (from the first to its next), and two buffers taken
    in turn: product j + 1's matrix decoded on the side stream once product j is queued (it waits for the products queued before
    j, the last to read that buffer, then AHEAD_HOLD ns), waited for on the GPU before product j + 1; the order's first decoded on
    the current stream. A call off the order waits for the side stream, forgets what the buffers hold and records the order again."""

    of = {}

    @staticmethod
    def get(d):
        a = _Ahead.of.get(d)
        if a is None:
            a = _Ahead.of[d] = _Ahead(d)
        return a

    def __init__(self, d):
        self.d, self.side = d, torch.cuda.Stream(d, priority=-1)  # (high: its blocks placed as soon as an SM has room)
        self.cus = torch.cuda.get_device_properties(d).multi_processor_count
        self.order, self.rec, self.args, self.pos, self.held = None, [], {}, 0, [None, None]
        self.bufs = self.alloc(_BIGGEST.get(d, 0))  # (at the first prompt: vLLM's profile counts them)
        self.ready, self.gate = [torch.cuda.Event(), torch.cuda.Event()], torch.cuda.Event()

    def alloc(self, n):
        return [torch.empty(n, dtype=torch.bfloat16, device=self.d) for _ in range(2)] if n else None

    def drop(self):
        torch.cuda.current_stream(self.d).wait_stream(self.side)
        self.held = [None, None]

    def decode(self, key, buf, warps):
        data, a, b, words, O, K = self.args[key]
        (_lib.mma12_unpack if len(words) == 4 else _lib.mma_unpack)(data, a, b, words, K, 0, O, buf[: O * K].view(torch.int16), warps)

    def product(self, x2, data, a, b, words, bias, O, K, key=None):
        """F.linear(x2, W, bias), W decoded ahead where the order is known; None where the call is not taken here (a capture, the order
        being recorded): the caller decodes W itself. key: the call's in the order (default: the pack's address; a measurement's own
        where it calls one pack at several places)."""
        if torch.cuda.is_current_stream_capturing():
            return None
        key = data.data_ptr() if key is None else key
        self.args[key] = (data, a, b, list(words), O, K)
        if self.order is None:
            if key not in self.rec:
                self.rec.append(key)
                return None
            self.order, self.rec, self.pos = self.rec[self.rec.index(key) :], [], 0
            need = max(self.args[k][4] * self.args[k][5] for k in self.order)
            if self.bufs is None or self.bufs[0].numel() < need:
                self.drop()
                self.bufs = self.alloc(need)
        if self.order[self.pos] != key:
            self.drop()
            if key != self.order[0]:
                self.order, self.rec = None, [key]
                return None
            self.pos = 0
        j, s = self.pos, self.pos % 2
        main = torch.cuda.current_stream(self.d)
        if self.held[s] == key:
            main.wait_event(self.ready[s])
        else:
            self.decode(key, self.bufs[s], 0)
            self.held[s] = key
        nxt = self.order[j + 1] if j + 1 < len(self.order) else None
        if nxt is not None:
            self.gate.record(main)
        y = F.linear(x2, self.bufs[s][: O * K].view(O, K), bias)
        if nxt is not None:
            self.side.wait_event(self.gate)
            with torch.cuda.stream(self.side):
                _lib.hold(AHEAD_HOLD)
                self.decode(nxt, self.bufs[1 - s], AHEAD_WARPS * self.cus)
                self.ready[1 - s].record(self.side)
            self.held[1 - s] = nxt
        self.pos = j + 1 if nxt is not None else 0
        return y
# end dev


def _define():
    """glyd::vllm_linear: X W^T (+ bias) over a pack (data and its two tensors; its words: the tiered layout's three
    tiers, the 12-bit one's four), fused by the library's route for X's rows on this GPU, or the matrix decoded, then
    F.linear (where the route says so), or (exact) the GEMM vLLM's bf16 runs. out_features: the Linear's own rows; its
    pack's are padded to a multiple of 64 with zeros (_padded), and the product of the padding is never returned."""
    if _OPS:
        return
    lib = torch.library.Library("glyd", "FRAGMENT")
    lib.define("vllm_linear(Tensor x, Tensor data, Tensor a, Tensor b, int[] words, Tensor? bias, int out_features, bool exact) -> Tensor")

    def linear(x, data, a, b, words, bias, out_features, exact):
        K, rows = x.shape[-1], _padded(out_features)
        if exact:  # bf16's own GEMM (UnquantizedLinearMethod.apply: its linear backend, or batch-invariant) on the matrix
            return _BF16[0].apply(types.SimpleNamespace(weight=_unpack(data, a, b, words, rows, K)[:out_features]), x, bias)
        torch._check(x.dtype == torch.bfloat16, lambda: f"glyd: bf16 activations, not {x.dtype}")
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        if M and _decoded(data.device.index, words, rows, K, M):
            # dev
            if AHEAD_ON and rows == out_features:
                y = _Ahead.get(data.device).product(x2, data, a, b, words, bias, out_features, K)
                if y is not None:
                    return y.view(*x.shape[:-1], out_features)
            # end dev
            return F.linear(x2, _unpack(data, a, b, words, rows, K)[:out_features], bias).view(*x.shape[:-1], out_features)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.empty(M, rows, dtype=torch.bfloat16, device=x.device)
        if M:
            if bias is not None and bias.shape[0] != rows:
                bias = F.pad(bias, (0, rows - bias.shape[0]))
            fresh, _lib.local.fresh = _lib.local.fresh, True  # workspaces for the call alone (in a capture: the graph's pool)
            try:
                f = _lib.mma12_linear if len(words) == 4 else _lib.mma_linear
                f(data, a, b, list(words), rows, K, x2, bias if bias is not None else _nobias(x.device), y, _way_given(data.device.index, words, rows, K, M))
            finally:
                _lib.local.fresh = fresh
        if rows != out_features:
            y = y[:, :out_features].contiguous()
        return y.view(*x.shape[:-1], out_features)

    lib.impl("vllm_linear", linear, "CUDA")
    torch.library.register_fake("glyd::vllm_linear", lambda x, data, a, b, words, bias, out_features, exact: x.new_empty((*x.shape[:-1], out_features)), lib=lib)
    # an embedding's lookup and an LM head's product, over a table's pack (rows: the pack's, a multiple of 64; the head's out_features: the shard's)
    lib.define("vllm_embed(Tensor ids, Tensor data, Tensor a, Tensor b, int[] words, int rows, int dim) -> Tensor")
    lib.define("vllm_logits(Tensor x, Tensor data, Tensor a, Tensor b, int[] words, Tensor? bias, int rows, int out_features) -> Tensor")

    def embed(ids, data, a, b, words, rows, dim):
        flat = ids.reshape(-1)
        if flat.dtype != torch.int64 or not flat.is_contiguous():
            flat = flat.to(torch.int64).contiguous()
        out = torch.empty(flat.numel(), dim, dtype=torch.bfloat16, device=ids.device)
        if flat.numel():
            (_lib.mma12_rows if len(words) == 4 else _lib.mma_rows)(data, a, b, list(words), dim, rows, flat, out)
        return out.view(*ids.shape, dim)

    def logits(x, data, a, b, words, bias, rows, out_features):
        torch._check(x.dtype == torch.bfloat16, lambda: f"glyd: bf16 activations, not {x.dtype}")
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        y = torch.empty(M, rows, dtype=torch.bfloat16, device=x.device)
        if M:
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            if bias is not None and bias.shape[0] != rows:
                bias = F.pad(bias, (0, rows - bias.shape[0]))
            way = (_lib.mma12_way if len(words) == 4 else _lib.mma_way)(_gpu(data.device.index), rows, K, M)[0]
            fresh, _lib.local.fresh = _lib.local.fresh, True  # workspaces for the call alone (in a capture: the graph's pool)
            try:
                f = _lib.mma12_linear if len(words) == 4 else _lib.mma_linear
                # (a table's matrix is never decoded whole: its prompts take the kernel that multiplies from the pack)
                f(data, a, b, list(words), rows, K, x2, bias if bias is not None else _nobias(x.device), y, g.X1 if way in (g.DECODE, g.EARLY) else -1)
            finally:
                _lib.local.fresh = fresh
        if rows != out_features:
            y = y[:, :out_features].contiguous()
        return y.view(*x.shape[:-1], out_features)

    lib.impl("vllm_embed", embed, "CUDA")
    lib.impl("vllm_logits", logits, "CUDA")
    torch.library.register_fake("glyd::vllm_embed", lambda ids, data, a, b, words, rows, dim: ids.new_empty((*ids.shape, dim), dtype=torch.bfloat16), lib=lib)
    torch.library.register_fake("glyd::vllm_logits", lambda x, data, a, b, words, bias, rows, out_features: x.new_empty((*x.shape[:-1], out_features)), lib=lib)
    _OPS.append(lib)


_NOBIAS = {}


def _nobias(dev):
    t = _NOBIAS.get(dev)
    if t is None:
        t = _NOBIAS[dev] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    return t


def _pack_rows(w, pack, verify, name):
    """w (bf16 [O, K] on the GPU) packed by pack, its rows padded to a multiple of 64 with zeros first (a copy, for the call); with
    verify, decoded and compared with it bit for bit."""
    rows = _padded(w.shape[0])
    if rows != w.shape[0]:
        w = F.pad(w, (0, 0, 0, rows - w.shape[0]))
    p = pack(w)
    if verify and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
        raise ValueError(f"glyd: {name} decoded to other bits than its weights")
    return p


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
    """The "glyd" quantization method and its op, in this process (again: the same), and the lossless KV cache's attention backend
    (AttentionBackendEnum.CUSTOM, chosen only where kv is on). vllm_entry.register() calls it where vLLM is the release this is tested
    with."""
    _define()
    if _fp8_on():
        if vllm_fp8 is not None:
            vllm_fp8.register()
        else:
            log.warning("glyd: FP8 checkpoints are not available (vLLM's FP8 internals did not import: %s)", _NO_FP8)
    if vllm_quant is not None:
        vllm_quant.register()
    register_quantization_config(NAME)(GlydConfig)
    try:
        _hook_load()
    except Exception as e:  # (vLLM's loader moved: what it builds without the quant config stays bf16)
        log.warning("glyd: the tables and Linears vLLM builds without a quantization config stay bf16 (its loader did not take Glyd's step: %s: %s)", type(e).__name__, e)
    try:
        from . import vllm_kv

        if not vllm_kv.register():
            log.warning("glyd: another plugin holds vLLM's CUSTOM attention backend: kv lossless is not available")
        from . import vllm_admit

        vllm_admit.install_front()  # (in every process that takes requests: a prompt the KV cache could never hold is refused there; a server without kv lossless is not touched)
    except Exception as e:  # (the KV cache's internals moved: the weights' plugin still loads)
        log.warning("glyd: the lossless KV cache did not load (%s: %s): kv lossless is not available", type(e).__name__, e)


def _flat(c):
    """A (text) config's values as a dict, read past transformers' per-layer guard (a heterogeneous config, Gemma 4's from transformers 5.15
    on, raises on reading a per-layer attribute such as num_key_value_heads): its global values, and Gemma 4's full-attention layers' own
    head size and KV heads under the keys its config.json gives them (global_head_dim, num_global_key_value_heads)."""
    d = {k: v for k, v in vars(c).items() if not k.startswith("_")}
    types_ = d.get("layer_types") or []
    for i, o in (getattr(vars(c).get("_heterogeneity_spec"), "per_layer_overrides", None) or {}).items():
        if i < len(types_) and types_[i] == "full_attention":
            if "head_dim" in o:
                d.setdefault("global_head_dim", o["head_dim"])
            if "num_key_value_heads" in o:
                d.setdefault("num_global_key_value_heads", o["num_key_value_heads"])
    return d


def _linear_bytes(c):
    """A model's Linears and experts (bf16 bytes), the rest (its embeddings and LM head), and whether it has experts, from its (text)
    config c (for best_layout's fit and the memory check): glyd run's count (preflight.linear_bytes, layer by layer); (0, 0, False)
    where unknown."""
    from .preflight import linear_bytes

    try:
        return linear_bytes(_flat(c))
    except TypeError:  # (not a config: no vars())
        return 0, 0, False


def _h100_here():
    """Whether the current device is an H100 (SXM, PCIe or NVL: "H100" a word in its name)."""
    return torch.cuda.is_available() and bool(H100.search(torch.cuda.get_device_name() or ""))


def _a100_here():
    """Whether the current device is an A100 (SXM4 or PCIe, 40 or 80 GB: "A100" a word in its name)."""
    return torch.cuda.is_available() and bool(A100.search(torch.cuda.get_device_name() or ""))


def _has_experts(hf):
    """Whether a model's config (its text config) has routed experts."""
    tc = hf.get_text_config() if hf is not None and hasattr(hf, "get_text_config") else hf
    return any(getattr(tc, a, None) for a in ("num_experts", "num_local_experts", "n_routed_experts"))


def _moe_decode_min_here():
    """The tokens a step from which the decoded route takes over on this GPU: an H100's, another Hopper GPU's, an A100's, or the others'."""
    if not torch.cuda.is_available():
        return MOE_DECODE_MIN
    name = torch.cuda.get_device_name() or ""
    if torch.cuda.get_device_capability()[0] == 9:
        return MOE_DECODE_MIN_H100 if H100.search(name) else MOE_DECODE_MIN_HOPPER
    return MOE_DECODE_MIN_A100 if A100.search(name) else MOE_DECODE_MIN


def _fused_act(moe):
    """The activation the grouped gate and up product applies as it writes its sums (mma_moe's act: 1 SiLU, 2 GELU
    tanh, an expert's first I rows the gate's, I a multiple of 64), for a layer's FusedMoEConfig moe; else 0."""
    plain = all(getattr(moe, "swiglu_" + a, None) is None for a in ("limit", "alpha", "beta"))
    act = {MoEActivation.SILU: 1, MoEActivation.GELU_TANH: 2}.get(moe.activation, 0) if plain else 0
    return act if moe.intermediate_size_per_partition % 64 == 0 else 0


def _measured_acts():
    """Whether experts with an activation but plain SiLU are packed: not until their products are measured no slower than
    bf16's (their GPU checks set it in a development build)."""
    # dev: GLYD_MOE_ACTS=all packs them (gpu/vllm/moe_acts.py and check_vllm.py's runs of them; a release build reads none)
    if os.environ.get("GLYD_MOE_ACTS", "").strip().lower() == "all":
        return True
    # end dev
    return False


def _moe_decode_min():
    """Tokens a step from which a mixture of experts' layer decodes the experts its tokens are routed to and runs vLLM's
    Triton kernel on them, instead of the library's grouped products (fused mode): GLYD_MOE_DECODE_MIN, else
    MOE_DECODE_MIN (MOE_DECODE_MIN_H100 on an H100, MOE_DECODE_MIN_HOPPER on another Hopper GPU, MOE_DECODE_MIN_A100 on an A100); None (a negative number): never."""
    v = os.environ.get("GLYD_MOE_DECODE_MIN", "").strip()
    try:
        n = int(v) if v else _moe_decode_min_here()
    except ValueError:
        raise ValueError(f"glyd: GLYD_MOE_DECODE_MIN {v!r}: a number of tokens a step (negative: never)") from None
    return None if n < 0 else n


def _experts(up, down, E, x, plan, ids, weights, act, activation, config):
    """A mixture of experts' layer by the library's grouped products: x [T, H], each token's k experts (ids [T, k], int64; plan:
    moe_way's) and their weights [T, k], to [T, H]. act (_fused_act) 1 or 2: the gate and up product applies it; 0: vLLM's own
    activation op (apply_moe_activation, with config) on the product's sums, [T k, 2I] (gated) or [T k, I], in the plan's order."""
    if act:
        h = g.mma_moe(up, E, x, plan, ids, act)  # [T k, I]: act(gate) up, the pairs in the plan's order
    else:
        gu = g.mma_moe(up, E, x, plan, ids, 0)
        h = gu.new_empty(gu.shape[0], gu.shape[1] // 2 if activation.is_gated else gu.shape[1])
        apply_moe_activation(activation, h, gu, activation_config=config)
    return g.mma_moe(down, E, h, plan, ids, 0, None, weights, gather=False)  # [T, H]


def _estimate(lin, other, layout, n, draft=False, fraction=1.0, held=0):
    """A model's weights a GPU packed in layout, bytes, and at least (the Linears' count less a gate's third), over n
    GPUs: its Linears (lin, bf16 bytes) packed, but for the share (1 - fraction) left bf16, and the rest (other: its
    embedding and LM head) as they are but for the bytes of them packed too (held); a draft's Linears alone (it shares
    its target's embeddings; its LM head is its own smaller one)."""
    if draft:
        other = held = 0
    packed, plain = lin * fraction + held, lin * (1 - fraction) + other - held
    return (packed * BITS[layout] / 16 + plain) / n, ((lin * fraction * 2 / 3 + held) * BITS[layout] / 16 + plain) / n


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


def _experts_missing(loaded, E, gated=True):
    """A mixture of experts' layer's pieces the checkpoint did not give, from its loader calls' (expert_id, shard_id)
    (w1, w3: w13's gate and up, or w1 alone without a gate; w2), for its E experts; None where the loader took another
    way (not w1, w2, w3)."""
    got = set(loaded)
    if not {s for _, s in got} <= {"w1", "w2", "w3"}:
        return None
    return [f"expert {e}'s {s}" for e, s in sorted({(e, s) for e in range(E) for s in (("w1", "w3", "w2") if gated else ("w1", "w2"))} - got)]


def _fp8_on():
    """Whether FP8 checkpoints go through vllm_fp8.py: a development build's GLYD_FP8=1 alone. Off, an FP8 checkpoint under --quantization glyd is
    refused by vLLM (its quantization_config is not glyd), as in 0.29: the FP8 product is slower than vLLM's own FP8 GEMM."""
    # dev: GLYD_FP8=1 serves FP8 checkpoints through vllm_fp8.py (a release build reads none)
    if os.environ.get("GLYD_FP8", "").strip() == "1":
        return True
    # end dev
    return False


def _fp8(cfg, hf_config=None):
    """Whether cfg (a checkpoint's quantization_config) is an FP8 one Glyd serves (vllm_fp8.claims); refused with why where it is FP8 but not covered.
    False wherever the FP8 path is off (_fp8_on)."""
    if not _fp8_on():
        return False
    if not isinstance(cfg, dict) or str(cfg.get("quant_method") or "").strip().lower() in ("", NAME):
        return False
    if vllm_fp8 is None:
        if cfg.get("quant_method") in ("fp8", "mxfp8", "modelopt", "compressed-tensors"):
            raise ValueError(f"glyd: FP8 checkpoints are not available in this vLLM (its FP8 internals did not import: {_NO_FP8})")
        return False
    return vllm_fp8.claims(cfg, hf_config)


def _quant(cfg, hf_config=None):
    """Whether cfg (a checkpoint's quantization_config) is a quantized checkpoint's whose 16-bit parts Glyd packs (vllm_quant.claims); refused with why
    where vLLM serves it but Glyd does not cover it. False for a bf16 checkpoint's options and a glyd save's."""
    if vllm_quant is None:
        if isinstance(cfg, dict) and str(cfg.get("quant_method") or "").strip().lower() not in ("", NAME):
            raise ValueError(f"glyd: quantized checkpoints are not available in this vLLM (its quantization internals did not import: {_NO_QUANT})")
        return False
    return vllm_quant.claims(cfg, hf_config)


class GlydConfig(QuantizationConfig):
    """The "glyd" quantization method: the options given (a checkpoint's or --hf-overrides' quantization_config), a
    glyd save's glyd.json where the model is one, and the options in effect once vLLM makes the model (resolve)."""

    def __init__(self, layout=None, exact=None, verify=None, fraction=None, kv=None, cc12=None):
        super().__init__()
        self.given = {k: v for k, v in (("layout", layout), ("exact", exact), ("verify", verify), ("fraction", fraction), ("kv", kv), ("cc12", cc12)) if v is not None}
        self.manifest, self.dir, self.hf_config = None, None, None  # a glyd save's glyd.json and directory; the model's config
        self.opts = None  # {"layout", "exact", "verify", "fraction", "kv"} (and "cc12" where true): resolved at the model's first layer
        self.compiled, self.need, self.free, self.files_checked = False, 0, 0, False
        self.plan = (False, False, None)  # (the embedding packed, the LM head packed, why not: resolve's table_plan)
        self.tied, self.untied_by_checkpoint = False, False  # the config ties the LM head to the embedding; and vLLM built them apart for the checkpoint's own copy
        self.tables, self.noted = [], set()  # the packed tables' layers made so far (weak references); the lines logged once
        self.quantized = False  # the 16-bit parts of a quantized checkpoint (vllm_quant.py), not a bf16 model

    def __getstate__(self):  # (vLLM pickles its config to start its engine's process: not the worker's own)
        return {**self.__dict__, "_vc": None, "tables": [], "noted": set()}

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
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        """--quantization glyd on a checkpoint with a quantization_config of its own: this method, which vLLM probes before its own (a
        plugin's name first). An FP8 one under GLYD_FP8=1 (vllm_fp8.scheme; refused, with why, for an FP8 one it does not cover or on a GPU
        that is not Ada or Hopper), else any quantized one vLLM serves with a method Glyd covers (vllm_quant.py: its 16-bit parts packed;
        refused, with why, for another). None for everything else (a bf16 checkpoint, a glyd save: as before)."""
        if user_quant != NAME or not isinstance(hf_quant_cfg, dict):
            return None
        return NAME if _fp8(hf_quant_cfg, hf_config) or _quant(hf_quant_cfg, hf_config) else None

    @classmethod
    def from_config(cls, config):
        if _fp8(config):  # an FP8 checkpoint's quantization_config: vLLM's own FP8 config for it, with Glyd's wrapping of its Linears (vllm_fp8.py)
            return vllm_fp8.config(config)
        if _quant(config):  # a quantized checkpoint's: vLLM's own config for it, with Glyd's for the 16-bit parts it keeps (vllm_quant.py)
            return vllm_quant.config(config)
        bad = [k for k in config if k not in OPTIONS + IGNORED]
        if bad:
            raise ValueError(f"glyd: the quantization_config's {', '.join(map(repr, bad))}: not an option ({', '.join(OPTIONS)})")
        return cls(config.get("layout"), config.get("exact"), config.get("verify"), config.get("fraction"), config.get("kv"), config.get("cc12"))

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
        if self.manifest is None and _fp8_on() and vllm_fp8 is not None and vllm_fp8.modelopt_file(model_name, revision):
            raise ValueError("glyd: this checkpoint's quantization (ModelOpt's hf_quant_config.json) is not in its config.json's quantization_config, where --quantization glyd finds an FP8 checkpoint's: "
                             "packed as a bf16 one its FP8 weights would be read without their scales. Copy hf_quant_config.json's contents into config.json's quantization_config, or serve it with --quantization modelopt")
        if self.manifest is not None:
            self.dir = d
            fmt.refuse_int(self.manifest, "vLLM")
            if self.manifest.get("format") == fmt.TABLES:
                raise ValueError("glyd: a glyd save with its tables packed (glyd-v5, pack --tables) does not load in vLLM yet: save it without --tables, or serve its bf16 checkpoint with --quantization glyd")
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
            if o[4] != "off":
                _kv_prepare(vc, o)

    def resolve(self):
        """The options in effect (_options); the layout "auto" best_layout's for this GPU (a save's own where that is
        the one). What Glyd does not do yet refused, with why, and a model whose packs cannot fit the GPU. The options
        written into vLLM's additional_config (its compile cache's key)."""
        vc = get_current_vllm_config_or_none()
        if vc is None:
            raise RuntimeError("glyd: vLLM's config is not set where the model is made")
        if not isinstance(vc.additional_config, dict):
            raise ValueError("glyd: vLLM's additional_config is not a dict: Glyd keys vLLM's compile cache by it")
        layout, exact, verify, fraction, kv, cc12 = _options(vc.additional_config.get(KEY) or {}, self.given, _env())
        pc, mc = vc.parallel_config, vc.model_config
        _refusals(vc, exact, self.manifest, fraction)
        tables = None
        if kv != "off" and self.hf_config is mc.hf_config:  # (the model's own: not a speculative draft's, which takes no KV cache of this kind)
            kv = _kv_decide(vc, kv, self.hf_config, exact, verify)[0]  # (what the engine's process wrote into additional_config, or the same answer from the same config: kv auto is no more than lossless or off here)
            if kv != "off":
                tables = _kv_setup(vc, (vc.additional_config.get(KEY) or {}).get("kv_tables"), verify, exact)
        elif kv == "auto":  # (a draft's config: no such cache)
            kv = "off"
        # A mixture of experts' shared experts run on a side stream beside the rest by default: their products and the
        # main stream's would share the device's done counters. Off, before vLLM makes them (and caches its env), where
        # anything is packed (fraction 0 is bf16 as vLLM runs it).
        # (A quantized checkpoint's routed experts are its own: a Glyd product on the side stream never runs beside another, and the stream is left as vLLM has it.)
        if fraction > 0 and not self.quantized:
            os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
        cc = vc.compilation_config
        self.compiled = not (mc.enforce_eager or (cc.mode == CompilationMode.NONE and cc.cudagraph_mode == CUDAGraphMode.NONE))
        if not self.manifest and pc.tensor_parallel_size > 1:
            log.warning("glyd: tensor parallel %d: each rank packs its own shard (not measured yet)", pc.tensor_parallel_size)
        try:
            dev = torch.cuda.current_device()
            if fraction > 0 or kv != "off":
                _gpu(dev)  # the library, loaded now: not at the first forward (fraction 0 packs nothing: it stays unloaded, unless kv is on)
        except Exception as e:
            raise RuntimeError(f"glyd: Glyd's GPU library did not load ({type(e).__name__}: {e}): a CUDA build of PyTorch and the glyd wheel's library (or GLYD_GPU_LIB) are needed") from e
        hf = self.hf_config  # (this config's model: a draft's own, not the target's vc.model_config)
        tc = hf.get_text_config() if hf is not None and hasattr(hf, "get_text_config") else hf if hf is not None else mc.hf_text_config
        # (A quantized checkpoint's 16-bit Linears and tables are not told by its config: no estimate and no fit check; the tiered layout, the
        # smallest, on every GPU: below.)
        lin, other, moe = (0, 0, _has_experts(hf)) if self.quantized else _linear_bytes(tc)
        n = max(1, pc.tensor_parallel_size) * max(1, pc.pipeline_parallel_size)
        # (A quantized checkpoint's model made as a sub-model, under a copy of vLLM's config with its text config (Gemma 4's language model, vLLM's
        # init_vllm_registered_model), is not a draft: there is one only where vLLM speculates.)
        draft = hf is not None and hf is not getattr(mc, "hf_config", None) and (vc.speculative_config is not None or not self.quantized)
        # The embedding and the LM head: packed too where nothing keeps them as vLLM runs them (table_plan); where the config ties them
        # (or ties them with a copy in the checkpoint, which vLLM builds apart) they are one table.
        self.tied = bool(getattr(tc, "tie_word_embeddings", False))
        self.untied_by_checkpoint = bool(getattr(mc, "word_embeddings_untied_by_checkpoint", False)) and not draft
        self.plan = self.table_plan(vc, fraction, exact, draft)
        held = _held(other, self.plan, self.tied or self.untied_by_checkpoint)  # (their bf16 bytes that are packed)
        packed = int(lin * fraction)  # (the Linears packed, bf16 bytes: the rest stay as vLLM runs them)
        if layout == "auto":  # (a quantized checkpoint's: the tiered layout, whose saving is what serving one with Glyd is for; the RTX 5090's 12-bit one kept 98.4% of gemma-4-12B QAT's embedding)
            layout = self.manifest["layout"] if self.manifest else "mma" if self.quantized else g.best_layout((packed + held) // n, (other - held + lin - packed) // n, 1, dev, moe=moe)[0]
        # Free: the driver's, and what PyTorch holds cached but unused (a draft is made after the target's packing freed
        # its temporaries into PyTorch's cache). A draft (a config of its own, not vLLM's model's) counted by its
        # Linears alone: it shares the target's embeddings, and its LM head is its own smaller one.
        self.free = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        self.need, low = _estimate(lin, other, layout, n, draft=draft, fraction=fraction, held=held)
        if lin and low > self.free:  # (they cannot fit: refused here, not by an OutOfMemoryError deep in the load)
            banded = f"; the mma layout (the smallest) takes about {_estimate(lin, other, 'mma', n, draft=draft, fraction=fraction, held=held)[0] / 2**30:.1f} GiB" if layout == "mma12" else ""
            share = f" (fraction {fraction:g} of the layers, the rest bf16)" if fraction < 1 else ""
            raise ValueError(f"glyd: the model's weights take about {self.need / 2**30:.1f} GiB a GPU packed in the {layout} layout{share}, at least {low / 2**30:.1f}, and the GPU has {self.free / 2**30:.1f} GiB free{banded}: serve it over more GPUs (--tensor-parallel-size), or a smaller model")
        if exact:
            _BF16[:] = [UnquantizedLinearMethod()]  # (vLLM's config current: its linear backend)
        self.opts, self._vc = {"layout": layout, "exact": exact, "verify": verify, "fraction": fraction, "kv": kv, **({"cc12": True} if cc12 else {})}, vc
        _CC12[0] = cc12
        vc.additional_config[KEY] = {**self.opts, "glyd": __version__, "packs": _digest(), **({"kv_tables": tables} if tables else {})}  # (a draft's resolve keeps the packs made)
        L = getattr(tc, "num_hidden_layers", 0) or 0
        log.info("glyd: %s layout%s%s%s%s", layout, ", exact" if exact else "", ", verified" if verify else "", f", {sum(_packs(i, fraction) for i in range(L))} of {L} layers packed (fraction {fraction:g})" if fraction < 1 else "",
                 f", the KV cache lossless (tables {tables['path']})" if tables else "")
        if not draft:  # (what this plan keeps as vLLM runs it; the tables packed are named once the model has loaded: _after_load)
            e, h, why = self.plan
            if not (e and h):
                log.info("glyd: %s", f"the embedding is packed; the LM head stays as vLLM runs it ({why})" if e else f"the embedding and the LM head stay as vLLM runs them ({why})")
        _ple_plain()

    def table_plan(self, vc, fraction, exact, draft):
        """(the embedding packed, the LM head packed, why not) as the options and vLLM's config decide them for this model: both
        where nothing keeps them as vLLM runs them. A model's tables count as the layers outside the decoder's, so a fraction under
        1 leaves them; a draft's, and a target's whose draft shares them (speculative decoding), stay; exact keeps the LM head vLLM's
        own matrix and GEMM (its logits bit for bit vLLM's; decoding the matrix whole for the GEMM would take a buffer of its size,
        more than packing it saves, and the GEMM in pieces is another kernel's bits), and with it a tied table, which is that matrix; a
        head_dtype other than the model's runs the head as vLLM's own. The same answer for a tied model's two layers."""
        if draft:
            return False, False, "a speculative draft keeps its own"
        if fraction < 1:
            return False, False, "packed at fraction 1 only"
        if vc.speculative_config is not None:
            return False, False, "speculative decoding shares them with its draft"
        tied = self.tied or self.untied_by_checkpoint
        hd = getattr(vc.model_config, "head_dtype", torch.bfloat16)
        if hd != torch.bfloat16:
            return (False, False, f"the LM head runs in {hd}") if tied else (True, False, f"the LM head runs in {hd}")
        if exact:
            return (False, False, "exact mode keeps the LM head's matrix, which a tied embedding is, as vLLM runs it: its logits stay vLLM's bit for bit") if tied else (
                True, False, "exact mode keeps the LM head's matrix and product as vLLM runs them: its logits stay vLLM's bit for bit")
        # dev: GLYD_PACK_EMBED=none|embed|head|all (a tied model: all or none)
        only = os.environ.get("GLYD_PACK_EMBED", "").strip().lower()
        if only:
            e, h = only in ("embed", "all"), only in ("head", "all")
            return (e and h, e and h, "GLYD_PACK_EMBED") if tied else (e, h, "GLYD_PACK_EMBED")
        # end dev
        return True, True, None

    def table_why(self, layer, head):
        """Why this embedding or LM head stays as vLLM runs it, in a user's words; None where it is packed."""
        e, h, why = self.plan
        if not (h if head else e):
            return why
        if type(layer) not in (VocabParallelEmbedding, ParallelLMHead):
            return f"{type(layer).__name__} is an embedding of its own kind, not a plain lookup"
        if getattr(layer, "embedding_dim", 64) % 64:
            return f"its hidden size {layer.embedding_dim} is not a multiple of 64"
        if getattr(layer, "num_embeddings", 1 << 20) < 4096:  # (a position table, not a vocabulary: nothing to gain)
            return f"its {layer.num_embeddings} rows are too few to gain from packing"
        return None

    def note(self, line):
        """A line to the log, once a process."""
        if line not in self.noted:
            self.noted.add(line)
            log.info("glyd: %s", line)

    def table_method(self, layer, prefix):
        """The method of an embedding's or an LM head's table: Glyd's packed one (a glyd save's pack as saved), or None, vLLM's
        own, where the table stays as it is (why, once, in the log). A save's packed head of another row count than vLLM's (a
        vocabulary it padded) is decoded to bf16 as it loads."""
        head = isinstance(layer, ParallelLMHead)
        if self.opts is None:
            self.resolve()
        e, why = self.saved(prefix), self.table_why(layer, head)
        if why is not None:
            if why is not self.plan[2]:
                self.note(f"the {'LM head' if head else 'embedding'} stays as vLLM runs it ({why})")
            return GlydSavedEmbeddingMethod(self, e) if e is not None else None
        V = getattr(layer, "num_embeddings_padded", 0) // max(1, getattr(layer, "tp_size", 1))
        if e is not None:
            return GlydEmbeddingMethod(self, head, e, prefix) if e["shape"][0] == -(-V // 64) * 64 else GlydSavedEmbeddingMethod(self, e)
        self.tables.append(weakref.ref(layer))
        return GlydEmbeddingMethod(self, head, None, prefix)

    def twin(self, layer, w):
        """A packed table of this model that w (this layer's bf16 rows) equals bit for bit, else None: where the config ties the LM
        head to the embedding and the checkpoint holds both, vLLM builds them apart and loads both; one table is kept."""
        if not self.untied_by_checkpoint:
            return None
        for ref in self.tables:
            o = ref()
            if o is not None and o is not layer and getattr(o, "glyd_words", None) is not None and (o.glyd_out, o.embedding_dim) == tuple(w.shape) and _same_rows(o, w):
                return o
        return None

    def check_files(self):
        """verify: a glyd save's files against its glyd.json, its tensors saved as they are by their sha256 (once, when
        the first pack is loaded: the weights are there by then, a Hub repo's too)."""
        if self.files_checked or self.manifest is None or not self.opts["verify"]:
            return
        checked, unchecked = fmt.check_files(self.dir, self.manifest)
        self.files_checked = True
        log.info("glyd: %s: %d tensors saved as they are checked by sha256%s", self.dir, checked, f", {unchecked} not (saved before glyd 0.25)" if unchecked else "")

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, Attention):  # (a decoder's attention layer: its KV cache's tables, where kv is on)
            if self.opts is None:
                self.resolve()
            if self.opts["kv"] == "off":
                return None
            from . import vllm_kv

            return vllm_kv.GlydKVCacheMethod(self)
        if RoutedExperts is not None and isinstance(layer, RoutedExperts):  # a mixture of experts' layer
            if self.opts is None:
                self.resolve()
            if not self.packs(prefix):  # (fraction: this layer's experts stay as vLLM runs them)
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
                raise RuntimeError("glyd: VLLM_DISABLE_SHARED_EXPERTS_STREAM is not on in this process (vLLM read its environment before Glyd set it): set VLLM_DISABLE_SHARED_EXPERTS_STREAM=1, since Glyd's products cannot yet run beside vLLM's shared experts on a side stream")
            why = GlydMoEMethod.unsupported(layer, self.opts["exact"])
            if why:
                log.warning("glyd: %s: experts kept bf16 (%s)", prefix, why)
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return GlydMoEMethod(self, layer.moe_config)
        if isinstance(layer, VocabParallelEmbedding):  # an embedding, or an LM head
            return self.table_method(layer, prefix)
        if not isinstance(layer, LinearBase):
            return None
        if self.opts is None:
            self.resolve()
            if _NO_MOE:
                log.warning("glyd: vLLM's mixture of experts internals did not import (%s): a model's experts stay bf16", _NO_MOE)
        if not self.packs(prefix):  # (fraction: this layer's Linears stay as vLLM runs them without a quantization)
            return UnquantizedLinearMethod()
        saved = self.saved(prefix)
        if saved is None and getattr(layer, "input_size", SMALL_LINEAR) * getattr(layer, "output_size", 1) < SMALL_LINEAR:  # (a mixture of experts' router: packing saves nothing, and its product takes 3x bf16's time)
            return UnquantizedLinearMethod()
        if saved is None and self.dense_stays(prefix):
            return UnquantizedLinearMethod()
        return GlydLinearMethod(self, saved)

    def dense_stays(self, prefix):
        """Whether the attention Linear at vLLM's prefix stays as vLLM runs it: on an H100 or an A100, in a model with routed experts. Its products are a few percent of such a
        model's weights (0.4 GiB of Qwen3-30B-A3B's 44), and there cuBLAS's products take less time than Glyd's (an H100, 5120 x 2048 at one token: 9.9 us against 14.8; an A100,
        the serve of Qwen3-16B-A3B: 1.35x bf16's requests a second with them left bf16, 1.32x packed)."""
        if not ("attn" in prefix or "attention" in prefix):
            return False
        vc = getattr(self, "_vc", None) or get_current_vllm_config_or_none()
        experts = vc is not None and _has_experts(vc.model_config.hf_config)
        # dev: GLYD_MOE_DENSE=packed packs them, bf16 keeps them on any GPU: the A/B of this rule both ways (a release build reads none)
        a = os.environ.get("GLYD_MOE_DENSE", "").strip().lower()
        if a in ("packed", "bf16"):
            return experts and a == "bf16"
        # end dev
        return experts and (_h100_here() or _a100_here())

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
        return torch.OutOfMemoryError(f"glyd: {name}: out of GPU memory packing it, after {len(_PACKS)} layers packed; the model's weights take about {self.need / 2**30:.1f} GiB a GPU packed in the {self.opts['layout']} layout, with {self.free / 2**30:.1f} GiB free before loading: serve it over more GPUs (--tensor-parallel-size), in the mma layout (the smallest), or a smaller model ({e})")


def _env():
    return {"layout": os.environ.get("GLYD_LAYOUT"), "exact": os.environ.get("GLYD_EXACT"), "verify": os.environ.get("GLYD_VERIFY"), "fraction": os.environ.get("GLYD_FRACTION"), "kv": os.environ.get("GLYD_KV")}


def _building_config():
    """The VllmConfig vLLM is building where it makes a quantization config (vLLM 0.30: in VllmConfig.__post_init__, in
    the engine's process), found up the calling frames; None elsewhere (a draft's config, made in a worker)."""
    f = sys._getframe(0)  # (from the innermost Python frame: compiled code makes none of its own, so no frame is skipped by number)
    while f is not None:
        obj = f.f_locals.get("self")
        if isinstance(obj, VllmConfig):
            return obj
        f = f.f_back
    return None


def _digest():
    return hashlib.sha256(json.dumps(sorted(_PACKS.items())).encode()).hexdigest()[:16] if _PACKS else ""


def _hook_load():
    """vLLM's last step of a model's load (base_loader's process_weights_after_loading, its own processing of every layer's weights),
    followed by _after_load: what vLLM built without the quant config packed once the whole model has loaded. Once a process."""
    from vllm.model_executor.model_loader import base_loader

    own = base_loader.process_weights_after_loading
    if getattr(own, "glyd", False):
        return

    def process_weights_after_loading(model, model_config, target_device):
        own(model, model_config, target_device)
        _after_load(model)

    process_weights_after_loading.glyd = True
    base_loader.process_weights_after_loading = process_weights_after_loading


def _ple_plain():
    """Qwen4Exp's n-gram table (vllm.models.qwen4_exp, imported where the model is one): vLLM refuses any quantization config but FP8's for
    it; under Glyd's it takes vLLM's own bf16 method (the table stays bf16, the model loads). Nothing where the module is not loaded."""
    mod = sys.modules.get("vllm.models.qwen4_exp.nvidia.ngram_embedding")
    cls = getattr(mod, "Qwen4ExpPLEEmbeddingMethod", None)
    own = getattr(cls, "from_quant_config", None)
    if own is None or getattr(own, "glyd", False):
        return

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        return own(None if isinstance(quant_config, GlydConfig) else quant_config, prefix, embedding_dtype)

    from_quant_config.glyd = True
    cls.from_quant_config = staticmethod(from_quant_config)


def _towers_on():
    """Whether the Linears of a model's vision and audio towers that vLLM builds without the quant config are packed after the load: not yet
    (an image's prefill through them is to be measured first)."""
    # dev: GLYD_PACK_TOWERS=1 packs them (the A/B of an image's prefill; a release build reads none)
    if os.environ.get("GLYD_PACK_TOWERS", "").strip() == "1":
        return True
    # end dev
    return False


MIXERS = ("MambaMixer",)  # decoder modules whose Linears vLLM builds without the quant config and runs as modules: Jamba's Mamba (v1) mixer


def _glyd_config(c):
    """The GlydConfig in a layer's quant_config: the config itself, or the one for a quantized checkpoint's 16-bit parts (vllm_quant.py's
    GlydQuantConfig, vLLM's own config with Glyd's in front); None for any other."""
    if vllm_quant is not None and isinstance(c, vllm_quant.GlydQuantConfig):
        c = c.glyd
    return c if isinstance(c, GlydConfig) else None


def _after_load(model):
    """A model loaded with Glyd's config (its Linears' quant_config, or a quantized checkpoint's with Glyd's in front: vllm_quant.py): what
    vLLM built without the config, packed now, after vLLM's own processing (the bf16 was resident through the load, as it is without
    this; its memory goes back before vLLM sizes the KV cache):
    - its embedding and LM head (VocabParallelEmbedding, ParallelLMHead on vLLM's own method): by the plan and table_why, as a table given
      the config; a tied pair (one weight) one pack;
    - Linears built without it (LinearBase on vLLM's own method with no quant_config): a decoder's MIXERS', and with _towers_on the
      towers' (every module outside the language model's), plain nn.Linear there too (not under exact); 1M weights or more, bf16,
      columns a multiple of 16, and the fraction's layers (packs).
    One line in the log: what was packed."""
    cfg = next((g for m in model.modules() if (g := _glyd_config(getattr(m, "quant_config", None))) is not None), None)
    if cfg is None or cfg.opts is None:
        return
    with torch.no_grad():
        tables = _after_load_tables(model, cfg)
        lins = _after_load_linears(model, cfg)
    if tables or lins:
        log.info("glyd: packed once loaded (vLLM builds them without a quantization config): %s", "; ".join(x for x in (", ".join(tables), f"{len(lins)} Linears" if lins else "") if x))
    kept = [n or type(m).__name__ for n, m in model.named_modules() if isinstance(m, VocabParallelEmbedding) and getattr(m, "glyd_words", None) is None and m.num_embeddings >= 4096]
    if kept:
        log.info("glyd: as vLLM runs them: %s", ", ".join(kept))


def _bf16_matrix(w, cols=16):
    """Whether w is a bf16 matrix on the GPU whose columns are a multiple of cols."""
    return isinstance(w, torch.Tensor) and w.dim() == 2 and w.numel() > 0 and w.dtype == torch.bfloat16 and w.is_cuda and w.shape[1] % cols == 0


def _after_load_tables(model, cfg):
    """The embeddings and LM heads still on vLLM's own method, packed where a table given the config would be (table_why); those that share
    one weight (a tied head) one pack. Returns their names."""
    groups = {}
    fused = {id(getattr(m, "embed_tokens", None)) for m in model.modules() if getattr(m, "replicated_embed", False) is True}  # (a model that chose vLLM's fused gather of a plain table, which reads its weight)
    for name, m in model.named_modules():
        if isinstance(m, VocabParallelEmbedding) and type(m.quant_method) is UnquantizedEmbeddingMethod and _bf16_matrix(getattr(m, "weight", None)):
            groups.setdefault(id(m.weight), []).append((name, m))
    done = []
    for members in groups.values():
        members.sort(key=lambda nm: isinstance(nm[1], ParallelLMHead))  # (the embedding first: a head tied to it holds its weight)
        why = "the model gathers its rows from the bf16 table itself" if any(id(m) in fused for _, m in members) else None
        why = why or next((w for w in (cfg.table_why(m, isinstance(m, ParallelLMHead)) for _, m in members) if w), None)
        if why is not None:
            if why is not cfg.plan[2]:
                cfg.note(f"{', '.join(n for n, _ in members)} stay{'s' if len(members) == 1 else ''} as vLLM runs {'it' if len(members) == 1 else 'them'} ({why})")
            continue
        (name, first), rest = members[0], members[1:]
        method = GlydEmbeddingMethod(cfg, isinstance(first, ParallelLMHead), None, name)
        method.uses_meta_device = False
        method.pack_loaded(first, name)
        for n, m in rest:
            other = GlydEmbeddingMethod(cfg, isinstance(m, ParallelLMHead), None, n)
            other.uses_meta_device = False
            other.share(m, first, n)
            other.adopted(m)
        done.append(" and ".join(n for n, _ in members) + (" (one pack)" if rest else ""))
    return done


def _language_model(model):
    """The model's language model (vLLM's multimodal models: get_language_model()), or None where it has none of its own."""
    f = getattr(model, "get_language_model", None)
    try:
        lm = f() if callable(f) else None
    except Exception:
        return None
    return lm if isinstance(lm, torch.nn.Module) and lm is not model else None


def _after_load_linears(model, cfg):
    """The Linears vLLM built without the quant config that are packed after the load (_after_load), packed. Returns their names."""
    lm, towers = _language_model(model), _towers_on()
    inside = {id(m) for m in lm.modules()} if lm is not None else None
    parents = {id(c): p for p in model.modules() for c in p.children()}
    done = []
    for name, m in list(model.named_modules()):
        tower = towers and inside is not None and id(m) not in inside
        if isinstance(m, LinearBase):
            if getattr(m, "quant_config", None) is not None or type(m.quant_method) is not UnquantizedLinearMethod:
                continue
            if not (tower or type(parents.get(id(m))).__name__ in MIXERS):
                continue
        elif type(m) is not torch.nn.Linear or not tower or cfg.opts["exact"] or (m.bias is not None and m.bias.dtype != torch.bfloat16):  # (exact: a plain Linear's own F.linear is not vLLM's GEMM)
            continue
        w = getattr(m, "weight", None)
        if not _bf16_matrix(w) or w.numel() < SMALL_LINEAR or not cfg.packs(name):
            continue
        if cfg.opts["exact"] and cfg.compiled and getattr(m, "bias", None) is not None and not getattr(m, "skip_bias_add", False):
            continue  # (exact compiled: inductor adds such a bias apart from the matmul; it stays as vLLM runs it, where a Linear given the config is refused)
        method = GlydLinearMethod(cfg, None)
        method.pack_loaded(m, name)
        if isinstance(m, LinearBase):
            m.quant_method = method
        else:
            m.forward = types.MethodType(_trace.plain_linear_forward, m)
        done.append(name)
    return done


def _held(other, plan, one):
    """The bf16 bytes of a model's embedding and LM head that plan packs: other is both where the model has two tables, else
    (one: the config ties them) the one."""
    e, h, _ = plan
    return other * e if one else other // 2 * (e + h)


def _pack_of(layer):
    """A packed table's layer as the pack kernels.py's functions take: over its buffers."""
    shape, t, words = (layer.glyd_out, layer.embedding_dim), (layer.glyd_data, layer.glyd_a, layer.glyd_b), layer.glyd_words
    return g.Mma12(shape, *t, words[0] & 0xFF) if len(words) == 4 else g.Mma(shape, *t, words)


def _same_rows(layer, w):
    """Whether a packed table (layer's) decodes to w (bf16 [rows, K] on its GPU) bit for bit, a chunk of rows at a time (32 MB)."""
    p, K = _pack_of(layer), w.shape[1]
    step = max(64, (1 << 24) // K // 64 * 64)
    buf = torch.empty(step * K, dtype=torch.bfloat16, device=w.device)
    for r in range(0, w.shape[0], step):
        n = min(step, w.shape[0] - r)
        if not torch.equal(g.mma_unpack(p, buf[: n * K], r, n).view(torch.int16), w[r : r + n].view(torch.int16)):
            return False
    return True


def _ignore_load(param, loaded_weight):
    """The weight loader of a tied LM head's empty weight: what a checkpoint holds for it is the embedding's."""


def _refusals(vc, exact, manifest, fraction=1.0):
    """What Glyd does not do yet, refused with why: vLLM's config (vc), exact mode, a glyd save's glyd.json (and with it
    a fraction under 1: its layers are packed on disk)."""
    pc, mc = vc.parallel_config, vc.model_config
    if envs.VLLM_BATCH_INVARIANT and not exact and fraction > 0:
        raise ValueError("glyd: VLLM_BATCH_INVARIANT asks for every product's bits not to depend on the batch, which Glyd's products do not give: serve with exact (vLLM's batch-invariant GEMM on the decoded weights), with fraction 0 (bf16 weights: kv lossless alone is batch invariant), or without VLLM_BATCH_INVARIANT")
    if pc.use_ubatching:
        raise ValueError("glyd: dual-batch overlap (--enable-dbo, ubatching) runs two batches' products at once on two streams, which Glyd's products do not support yet: run without it")
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


def _kv_window(vc):
    """Why the tables do not cover the model's window (--max-model-len), or None: they are calibrated for up to kvcal.bands()[-1] positions."""
    from . import kvcal

    mml = vc.model_config.max_model_len
    return None if kvcal.band(mml) is not None else f"--max-model-len {mml:,}: the KV cache's tables cover up to {kvcal.bands()[-1]:,} positions"


def _kv_refusals(vc, hf_config, window=True):
    """What the lossless KV cache does not hold, from vLLM's config (vc) and the model's: the reasons (each with what to do), none where it is fine (window: the tables' reach among them).
    Its backend's own checks (head size, sliding window, attention sinks ...) follow when vLLM selects it, per layer."""
    from vllm.platforms import current_platform

    from . import kvcal, vllm_kv

    pc, mc, cc = vc.parallel_config, vc.model_config, vc.cache_config
    why = []
    if pc.tensor_parallel_size > 1 or pc.pipeline_parallel_size > 1 or pc.decode_context_parallel_size > 1 or pc.prefill_context_parallel_size > 1:
        why.append("tensor, pipeline and context parallel are not held yet (one GPU)")
    if vc.speculative_config is not None:
        why.append("speculative decoding is not held yet")
    if cc.cache_dtype not in ("auto", "bfloat16"):
        why.append(f"--kv-cache-dtype {cc.cache_dtype}: kv lossless is the KV cache's dtype (leave it auto)")
    if cc.block_size % vllm_kv.PT:
        why.append(f"--block-size {cc.block_size}: a multiple of {vllm_kv.PT}")
    if pc.distributed_executor_backend not in (None, "uni"):
        why.append(f"--distributed-executor-backend {pc.distributed_executor_backend}: the engine's scheduler and the worker must be one process")
    if getattr(cc, "kv_offloading_size", None) or vc.kv_transfer_config is not None:
        why.append("KV connectors and KV offloading move the cache as bf16 (not held)")
    if mc.dtype != torch.bfloat16:
        why.append(f"{mc.dtype}: bf16 only")
    tc = hf_config.get_text_config() if hf_config is not None and hasattr(hf_config, "get_text_config") else hf_config
    if getattr(tc, "kv_lora_rank", None):
        why.append("a model with multi-head latent attention (MLA) keeps another cache")
    sw, types_ = getattr(tc, "sliding_window", None), list(getattr(tc, "layer_types", None) or [])
    if (sw and getattr(tc, "use_sliding_window", True)) or any("sliding" in str(t) or "mamba" in str(t) for t in types_):
        why.append("sliding window and state-space layers are not held")  # (linear-attention layers are: their state is vLLM's, the full-attention layers' cache this one)
    f = _flat(tc) if tc is not None else {}  # (Gemma 4's config raises on reading num_key_value_heads)
    nh, nkv, hd = f.get("num_attention_heads") or 0, f.get("num_key_value_heads") or f.get("num_attention_heads") or 0, f.get("head_dim") or ((f.get("hidden_size") or 0) // max(1, f.get("num_attention_heads") or 1))
    if hd not in vllm_kv.HEAD_SIZES:
        why.append(f"head size {hd}: 128 or 256 only")
    if nkv and nh // nkv > vllm_kv.MAX_GROUP:
        why.append(f"{nh // nkv} query heads to a KV head: up to {vllm_kv.MAX_GROUP}")
    cap = current_platform.get_device_capability()
    if cap is not None and cap.major not in (8, 9):
        why.append(f"compute capability {cap.major}.{cap.minor}: Ampere, Ada and Hopper only so far")
    if window and _kv_window(vc):
        why.append(_kv_window(vc))
    if vc.attention_config.backend not in (None, AttentionBackendEnum.CUSTOM):
        why.append(f"--attention-backend {vc.attention_config.backend.name}: the KV cache's own backend takes the full-attention layers")
    if AttentionBackendEnum.CUSTOM.is_overridden() and AttentionBackendEnum.CUSTOM.get_path() != vllm_kv.BACKEND_PATH:
        why.append("another plugin holds vLLM's CUSTOM attention backend")
    return why


def _kv_unmeasured(hf_config):
    """What of a model `auto` does not hold yet, held where kv lossless asks (None: nothing): head size 256, and linear-attention layers beside the full-attention ones."""
    tc = hf_config.get_text_config() if hf_config is not None and hasattr(hf_config, "get_text_config") else hf_config
    hd = getattr(tc, "head_dim", None) or (getattr(tc, "hidden_size", 0) // max(1, getattr(tc, "num_attention_heads", 1)))
    out = (["head size 256"] if hd == 256 else []) + (["linear-attention layers"] if any("linear" in str(t) for t in (getattr(tc, "layer_types", None) or [])) else [])
    return " and ".join(out) or None


def _kv_decide(vc, kv, hf_config, exact=False, verify=False):
    """What kv comes to for this config: (the value in effect, "off" or "lossless"; a line to log or None; the line's level). `auto` is lossless on the GPU classes of KV_AUTO where nothing refuses it and the tables
    cover the model's window, else vLLM's own KV cache with the line that says why; `lossless` is refused with why where it is not held (as ever), except for the window: a model whose window the tables do
    not cover is served on vLLM's own KV cache, whole, with a line (the cache never shortens a window). `auto` also stays off where `exact` or `verify` is on: those carry the staging route for every row (exact) or a bf16
    shadow of the cache (verify), which `lossless` asks for and a default does not. A pure function of the config, the options and the GPU: every process of a server comes to the same."""
    if kv == "off":
        return "off", None, None
    if kv == "auto" and (exact or verify):
        return "off", f"glyd: the KV cache stays vLLM's (kv auto): {'exact' if exact else 'verify'} is on, and with the lossless KV cache it would {'run every request on a bf16 copy of its cache (slower, more memory)' if exact else 'keep a bf16 copy of the cache (more memory)'}: kv lossless asks for that", "info"
    from vllm.platforms import current_platform

    from . import kvcal

    why, window = _kv_refusals(vc, hf_config, window=False), _kv_window(vc)
    if kv == "auto":
        try:
            name = str(current_platform.get_device_name() or "")
        except Exception:
            name = ""
        if not KV_AUTO.search(name):
            return "off", f"glyd: the KV cache stays vLLM's here (kv auto holds it on an A100, an L4, an H100 and a GH200, where it is measured faster; this is {name or 'a GPU of unknown name'}: kv lossless turns it on)", "info"
        newer = _kv_unmeasured(hf_config)
        if newer:
            return "off", f"glyd: the KV cache stays vLLM's (kv auto): it is not measured faster than vLLM's own yet for a model of {newer}: kv lossless turns it on", "info"
        if why or window:
            return "off", "glyd: the KV cache stays vLLM's (kv auto): " + "; ".join(why + ([window] if window else [])), "info"
        return "lossless", None, None
    if why:
        raise ValueError("glyd: kv lossless is not held here: " + "; ".join(why + ([window] if window else [])))
    if window:
        return "off", (f"glyd: kv lossless is off for this run, and vLLM's own KV cache serves the model's window: {window}; serve with --max-model-len {kvcal.bands()[-1]} "
                       f"or less to use it"), "warning"
    return "lossless", None, None


def _kv_key(vc, positions):
    from . import kvcal

    mc, pc = vc.model_config, vc.parallel_config
    return kvcal.key(mc.model, mc.revision, mc.hf_config, pc.tensor_parallel_size, pc.pipeline_parallel_size, positions)


def _kv_backend(vc):
    """The lossless KV cache's backend chosen for the full-attention layers (vLLM picks a backend per layer as the layer is made)."""
    per = vc.attention_config.backend_per_kind
    if per.get("full_attention") is not AttentionBackendEnum.CUSTOM:
        vc.attention_config.backend_per_kind = {**per, "full_attention": AttentionBackendEnum.CUSTOM}


def _kv_prepare(vc, o):
    """kv lossless or auto, in the engine's process before its workers start (vLLM makes its config there, the GPU still free): what kv comes to for this config and GPU (_kv_decide: refused with why where
    `lossless` is not held, vLLM's own KV cache with a line where `auto` finds no reason to hold it), and, where it is held, the backend chosen and the model's base tables found or, on a first start, made (a
    child process runs the model on a bundled text: about 1-5 minutes, logged) and their place written into vLLM's additional_config for the workers, with the value in effect: "lossless" or "off". Where `auto` cannot
    have its tables (they cannot be made, or the model leaves too much outside 12 bits) it is off with a line; `lossless` raises. Returns the value in effect."""
    from . import kvcal

    mc = vc.model_config
    eff, msg, level = _kv_decide(vc, o[4], mc.hf_config, o[1], o[2])
    if msg:
        (log.warning if level == "warning" else log.info)(msg)
    ac = vc.additional_config.get(KEY)
    if not isinstance(ac, dict):
        ac = vc.additional_config[KEY] = {}
    if eff == "off":
        ac["kv"] = "off"  # (the workers: vLLM's own cache, whatever their own environment says)
        return "off"
    try:
        path = None
        # dev: a tables file named by the environment (a release build reads none)
        path = os.environ.get("GLYD_KV_TABLES") or None
        # end dev
        if path is None:
            eng = {"trust_remote_code": mc.trust_remote_code, "download_dir": vc.load_config.download_dir}
            if mc.tokenizer != mc.model:
                eng["tokenizer"] = mc.tokenizer
            if getattr(getattr(mc, "multimodal_config", None), "language_model_only", False):  # (the text model alone, as it is served: the vision tower is not loaded)
                eng["language_model_only"] = True
            if mc.quantization == NAME:  # (the model as served: its weights packed where they are, or it may not fit bf16)
                eng.update(quantization=NAME, additional_config={KEY: {"layout": o[0], "fraction": o[3], "kv": "off"}})
            path = kvcal.ensure(mc.model, lambda n: _kv_key(vc, n), {k: v for k, v in eng.items() if v is not None}, log.info, mc.revision, mc.max_model_len)
        t = kvcal.Tables(path)  # (a file that does not load is refused here, in the engine's process, not in a worker)
        if o[4] == "auto" and max(t.meta["held_out_rate"].values()) > kvcal.REFUSED:
            raise ValueError(kvcal.refusal(t.meta))
    except Exception as e:
        if o[4] != "auto":
            raise
        log.warning("glyd: the KV cache stays vLLM's (kv auto): its tables for %s could not be had (%s: %s); kv lossless asks for them and says why they are not there", mc.model, type(e).__name__, str(e).splitlines()[0][:300])
        ac["kv"] = "off"
        return "off"
    _kv_backend(vc)
    ac["kv"] = "lossless"
    ac["kv_tables"] = {"path": path}
    sc = vc.scheduler_config
    if sc.scheduler_cls != SCHEDULER:  # (the engine's scheduler: the one asked for, or vLLM's, that lets a request in when the pool has room for its pages and preempts those that hold a page it could not store)
        cur = sc.scheduler_cls
        ac["scheduler_cls"] = cur if isinstance(cur, str) or cur is None else f"{cur.__module__}.{cur.__qualname__}"
        sc.scheduler_cls = SCHEDULER
    return "lossless"


def _kv_setup(vc, given, verify=False, exact=False):
    """kv lossless in a worker, as the model is made: the tables the engine's process found (additional_config's kv_tables) or, run some other way,
    GLYD_KV_TABLES's or the cache's (never made here: the GPU is this process's), checked against the model; this process's arena and the backend
    set. Returns what vLLM's additional_config keeps for the compile cache and the workers: {"path"}."""
    from . import kvcal, vllm_kv

    why = _kv_refusals(vc, vc.model_config.hf_config)
    if why:
        raise ValueError("glyd: kv lossless is not held here: " + "; ".join(why))
    _kv_backend(vc)
    path = (given or {}).get("path") or next((q for q in (os.path.join(kvcal.cache_dir(), f"kvtab-{_kv_key(vc, n)}.npz") for n in kvcal.bands() if n >= vc.model_config.max_model_len) if os.path.exists(q)), "")
    # dev: a tables file named by the environment (a release build reads none)
    path = os.environ.get("GLYD_KV_TABLES") or path
    # end dev
    if not os.path.exists(path):
        raise ValueError(f"glyd: kv lossless has no tables for {vc.model_config.model} ({path}): they are made on a first start by vllm serve's engine process; make them by hand with python -m glyd_gpu kvcal MODEL")
    t = vllm_kv.use_tables(path, verify, exact)
    if t.meta["positions"] < vc.model_config.max_model_len:
        raise ValueError(f"glyd: kv lossless: the tables {path} are made for {t.meta['positions']:,} positions and --max-model-len is {vc.model_config.max_model_len:,}: "
                         f"make tables for that many (python -m glyd_gpu kvcal MODEL --positions N), or serve with --max-model-len {t.meta['positions']} or less")
    if max(t.meta["held_out_rate"].values()) > kvcal.REFUSED:
        raise ValueError(f"glyd: kv lossless refused for {vc.model_config.model}: {kvcal.refusal(t.meta)}: serve with kv off")
    return {"path": path}


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


class GlydEmbeddingMethod(QuantizeMethodBase):
    """An embedding's or an LM head's table held packed, in the layout of the Linears, bit for bit: the lookup reads rows from
    the pack (the library's mma_rows: any ids, one launch, no buffer of the table's size) and the LM head multiplies from it
    (the library's product for the rows of the step; never the matrix decoded whole). One class for both: a model that uses its
    embedding module as its head (Gemma's) is one layer. As a Linear's, the weight is made on the meta device and packed by
    vLLM's layerwise processing once loaded (the peak: the packs and this table), its bf16 dropped as it is packed; its rows
    are padded to a multiple of 64 with zeros (a tensor-parallel shard's, a vocabulary's padding: the lookup never reads them,
    the head's output is cut back to the shard's). A head tied to the embedding (tie_weights) holds the embedding's buffers:
    one pack. Where the config ties them and the checkpoint holds both (vLLM builds them apart), the second of them to be
    packed is compared with the first and, if the same bits, takes its pack. A glyd save's pack (its LM head's) is loaded as
    saved."""

    uses_meta_device = True

    def __init__(self, config, head, saved=None, prefix=""):
        self.config, self.head, self.saved, self.prefix, self.src, self.rows = config, head, saved, prefix, None, 0
        if saved is not None:
            self.uses_meta_device = False

    embedding = _trace.embedding_method_lookup  # (Python source: vLLM's torch.compile traces them)
    apply = _trace.embedding_method_logits

    def name(self):
        return self.prefix or ("lm_head" if self.head else "embed_tokens")

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra):
        V, K = sum(output_partition_sizes), input_size_per_partition
        self.rows = -(-V // 64) * 64
        if params_dtype != torch.bfloat16:
            raise ValueError(f"glyd: {self.name()}: {params_dtype}, bf16 only")
        if self.saved is not None:
            if list(self.saved["shape"]) != [self.rows, K]:
                raise ValueError(f"glyd: {self.name()}: the save's pack is {self.saved['shape']}, vLLM's layer [{self.rows}, {K}]")
            return _placeholders(layer, self.saved["layout"])
        weight = torch.nn.Parameter(torch.empty(self.rows, K, device="meta", dtype=params_dtype), requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra)
        initialize_online_processing(layer)

    def tie_weights(self, layer, embed_tokens):
        """vLLM ties this LM head to the embedding: its table is the embedding's pack (its weight, on the meta device, gives way
        to an empty one that nothing loads into). Where the embedding stays as vLLM runs it, the head is its weight as vLLM ties it."""
        em = getattr(embed_tokens, "quant_method", None)
        if not isinstance(em, GlydEmbeddingMethod) or em.head or self.saved is not None:
            plain = layer.quant_method = UnquantizedEmbeddingMethod()
            return plain.tie_weights(layer, embed_tokens)
        self.src = embed_tokens
        weight = torch.nn.Parameter(torch.empty(0, layer.embedding_dim, dtype=torch.bfloat16), requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0, "weight_loader": _ignore_load})
        layer.weight = weight
        return layer

    def process_weights_after_loading(self, layer):
        if getattr(layer, "glyd_words", None) is not None:  # (vLLM calls it again after the load)
            return
        opts, name, K = self.config.opts, self.name(), layer.embedding_dim
        layout = opts["layout"]
        pack = g.pack_mma12 if layout == "mma12" else g.pack_mma
        try:
            if self.src is not None:
                return self.share(layer, self.src, name)
            if self.saved is not None:
                self.config.check_files()
                p = _saved_pack(layer, self.saved, opts["verify"])
                if isinstance(p, g.Mma12) != (layout == "mma12"):  # another layout asked for: decoded and packed again
                    w = g.mma_unpack(p)
                    p = pack(w)
                    if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
                        raise ValueError(f"glyd: {name}: the save packed again in the {layout} layout decoded to other bits than the save")
                    del w
            else:
                _check_loaded(layer, name)
                w = layer.weight.data
                twin = self.config.twin(layer, w)
                if twin is not None:
                    log.info("glyd: %s holds the same table as the embedding or LM head packed before it: one pack", name)
                    _drop(layer._parameters.pop("weight", None))
                    return self.share(layer, twin, name)
                p = pack(w)
                if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
                    raise ValueError(f"glyd: {name} decoded to other bits than its weights")
        except torch.OutOfMemoryError as e:
            raise self.config.out_of_memory(name, e) from e
        t = (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base)
        self.publish(layer, name, t, list(p.sym if isinstance(p, g.Mma12) else p.tiers), p.shape[0])

    def pack_loaded(self, layer, name):
        """A table that vLLM built without the quant config and loaded in bf16, packed once the model has loaded (_after_load): its
        weight padded to 64 rows with zeros and packed (where the config ties the LM head to the embedding and the checkpoint holds
        both, the first one's pack taken if the same bits), and this method the layer's from now on."""
        opts, w = self.config.opts, layer.weight.data
        rows = _padded(w.shape[0])
        if rows != w.shape[0]:
            w = F.pad(w, (0, 0, 0, rows - w.shape[0]))
        try:
            twin = self.config.twin(layer, w)
            if twin is not None:
                log.info("glyd: %s holds the same table as the embedding or LM head packed before it: one pack", name)
                self.share(layer, twin, name)
            else:
                p = (g.pack_mma12 if opts["layout"] == "mma12" else g.pack_mma)(w)
                if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
                    raise ValueError(f"glyd: {name} decoded to other bits than its weights")
                t = (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base)
                self.publish(layer, name, t, list(p.sym if isinstance(p, g.Mma12) else p.tiers), rows)
        except torch.OutOfMemoryError as e:
            raise self.config.out_of_memory(name, e) from e
        self.adopted(layer)

    def adopted(self, layer):
        """layer (a table packed after the load) run by this method from now on: its lookup and product read the pack (vLLM's fused lookup
        over several GPUs reads a bf16 weight)."""
        layer.quant_method, layer.use_fused_embedding = self, False
        self.config.tables.append(weakref.ref(layer))

    def share(self, layer, src, name):
        """layer's table is src's pack: the same buffers."""
        if getattr(src, "glyd_words", None) is None:
            raise RuntimeError(f"glyd: {name} is tied to an embedding that is not packed: the checkpoint gave it no weight")
        self.publish(layer, name, (src.glyd_data, src.glyd_a, src.glyd_b), src.glyd_words, src.glyd_out)

    def publish(self, layer, name, t, words, rows):
        """The table's buffers and words on its layer, its bf16 weight gone (an empty one is read for its dtype and K)."""
        opts, K = self.config.opts, layer.embedding_dim
        g.lib()  # the library loaded (the op's calls are its C API's)
        dev = t[0].device
        for n in {n for names in BUFFERS.values() for n in names}:
            layer._parameters.pop(n, None)
        for n, x in zip(("glyd_data", "glyd_a", "glyd_b"), t):
            layer.register_buffer(n, x, persistent=False)
        layer.glyd_words, layer.glyd_out, layer.glyd_verified = list(words), rows, opts["verify"]
        _drop(layer._parameters.pop("weight", None))
        layer.weight = torch.nn.Parameter(torch.empty(0, K, dtype=torch.bfloat16, device=dev), requires_grad=False)
        self.config.packed(name, opts["layout"], layer.glyd_words, t)
        for n in ("mma12_linear", "mma_linear"):  # the device's done counters, made now: never in a CUDA graph's pool
            _lib._counters(n, dev.index, None, 0, _lib._UNITS)
        _trim()


class GlydLinearMethod(LinearMethodBase):
    """A Linear's product from its pack. From a bf16 checkpoint the weight is made on the meta device and packed as
    vLLM's layerwise processing completes the layer (process_weights_after_loading); from a glyd save the pack's
    tensors load as saved. Rows not a multiple of 64 are packed with zero rows after them (the product returns the
    Linear's own); a matrix whose columns are not a multiple of 16, or not bf16, stays as vLLM runs it. A Linear vLLM
    builds without the quant config takes this method after the load (pack_loaded: _after_load)."""

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
        if K % 16 or params_dtype != torch.bfloat16:
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
        self.no_bias_under_exact(layer, name)
        pack = g.pack_mma12 if layout == "mma12" else g.pack_smallest
        try:
            if self.saved is not None:
                self.config.check_files()
                p = _saved_pack(layer, self.saved, opts["verify"])
                if isinstance(p, g.Mma12) != (layout == "mma12"):  # another layout asked for: decoded and packed again
                    w = g.mma_unpack(p)
                    p = pack(w)
                    if opts["verify"] and not torch.equal(g.mma_unpack(p).view(torch.int16), w.view(torch.int16)):
                        raise ValueError(f"glyd: {name}: the save packed again in the {layout} layout decoded to other bits than the save")
                    del w
                O = p.shape[0]
            else:
                _check_loaded(layer, name)
                O = layer.weight.shape[0]
                p = _pack_rows(layer.weight.data, pack, opts["verify"], name)
        except torch.OutOfMemoryError as e:
            raise self.config.out_of_memory(name, e) from e
        self.publish(layer, name, p, O)

    def no_bias_under_exact(self, layer, name):
        """Refused: exact under torch.compile, a Linear with a bias."""
        if self.config.opts["exact"] and self.config.compiled and getattr(layer, "bias", None) is not None:
            raise ValueError(f"glyd: {name} has a bias, and exact mode under torch.compile is not bit for bit there yet: inductor adds a bf16 Linear's bias apart from its matmul, rounding before the add, where exact's product adds it in the GEMM. Serve exact with --enforce-eager (vLLM's bf16 eager's bits), or compiled without exact")

    def pack_loaded(self, layer, name):
        """A Linear vLLM built without the quant config and loaded in bf16 (a LinearBase on vLLM's own method, or a plain nn.Linear),
        packed once the model has loaded (_after_load): this method's pack of its weight, its bias kept as it is."""
        self.uses_meta_device = False
        pack = g.pack_mma12 if self.config.opts["layout"] == "mma12" else g.pack_smallest
        try:
            p = _pack_rows(layer.weight.data, pack, self.config.opts["verify"], name)
        except torch.OutOfMemoryError as e:
            raise self.config.out_of_memory(name, e) from e
        self.publish(layer, name, p, layer.weight.shape[0])

    def publish(self, layer, name, p, O):
        """The pack's buffers and words on its layer (O: the Linear's own rows; the pack's, a multiple of 64, may be more), its bf16 weight
        gone (an empty one is read for its dtype and K), and the scratch the decode takes where a CUDA graph captures it."""
        opts, layout = self.config.opts, self.config.opts["layout"]
        g.lib()  # the library loaded (the op's calls are its C API's)
        t = (p.data, p.exc, p.exc_base) if isinstance(p, g.Mma12) else (p.data, p.blocks, p.block_base)
        dev = t[0].device
        for n in {n for names in BUFFERS.values() for n in names}:
            layer._parameters.pop(n, None)
        for n, x in zip(("glyd_data", "glyd_a", "glyd_b"), t):
            layer.register_buffer(n, x, persistent=False)
        rows, K = p.shape
        # dev
        _BIGGEST[dev] = max(_BIGGEST.get(dev, 0), rows * K)
        # end dev
        layer.glyd_words = list(p.sym if isinstance(p, g.Mma12) else p.tiers)
        layer.glyd_out, layer.glyd_exact, layer.glyd_verified = O, opts["exact"], opts["verify"]
        _drop(layer._parameters.pop("weight", None))
        layer.weight = torch.nn.Parameter(torch.empty(0, K, dtype=torch.bfloat16, device=dev), requires_grad=False)  # (read for its dtype and K)
        self.config.packed(name, layout, layer.glyd_words, t)
        need = rows * K if self._kept(dev, layer.glyd_words, rows, K, opts["exact"]) else 0
        if need and (dev not in _SCRATCH or _SCRATCH[dev].numel() < need):
            _SCRATCH[dev] = torch.empty(need, dtype=torch.bfloat16, device=dev)  # (at load: no CUDA graph holds the old one)
        for n in ("mma12_linear", "mma_linear"):  # the device's done counters, made now: never in a CUDA graph's pool
            _lib._counters(n, dev.index, None, 0, _lib._UNITS)
        _trim()

    def _decodes(self, dev, words, O, K, cap=None):
        """Whether any product up to vLLM's batch of tokens (or up to cap of them) decodes the matrix for cuBLAS (the scratch buffer it needs)."""
        vc = getattr(self.config, "_vc", None)
        top = vc.scheduler_config.max_num_batched_tokens if vc is not None else 8192
        top = min(top, cap) if cap else top
        M = 1
        while M <= top:
            way, last = (_lib.mma12_way if len(words) == 4 else _lib.mma_way)(_gpu(dev.index), O, K, M)
            if way in (g.DECODE, g.EARLY) or (way == g.X1 and K % 64):
                return True
            M = last + 1
        return False

    def _kept(self, dev, words, O, K, exact):
        """Whether this product's decode can be captured by a CUDA graph, so that its buffer is made at load and kept (a graph keeps the address it
        captured): exact decodes every product; else one whose route decodes for a count of tokens vLLM captures (up to its max_cudagraph_capture_size);
        and where the options or vLLM's config leave that unknown, any that decodes. Where no graph captures it (eager, or the counts a graph takes
        never decode: an L4's from 896 tokens, graphs to 512) the buffer is made for the call (_unpack)."""
        vc = getattr(self.config, "_vc", None)
        cc = getattr(vc, "compilation_config", None)
        if vc is None or cc is None:
            return exact or self._decodes(dev, words, O, K)
        if vc.model_config.enforce_eager or getattr(cc, "cudagraph_mode", None) in (None, CUDAGraphMode.NONE):
            return False
        if exact:
            return True
        top = getattr(cc, "max_cudagraph_capture_size", None)
        return self._decodes(dev, words, O, K, top if isinstance(top, int) and top > 0 else None)

    apply = _trace.linear_method_apply  # (Python source: vLLM's torch.compile traces it)


class GlydMoEMethod(OnlineMoEMethodBase):
    """A mixture of experts' layer's experts (vLLM's RoutedExperts: w13 [E, 2I, H], gate then up, or [E, I, H] without
    a gate, and w2 [E, H, I]), each weight packed as one matrix of its E experts' stacked ([E 2I, H], [E H, I]) as vLLM's
    layerwise processing completes the layer, from its bf16 on the meta device (the peak: the packs and a layer's
    experts), once every expert's pieces have loaded (else refused, naming the layer). Their products the library's
    grouped ones: each token's k choices (its pairs) sorted by expert on the GPU (moe_route), gate and up with the
    activation applied as its sums are written out where the product has it (_fused_act: SiLU, GELU tanh, no clamp),
    else their sums written out (bf16) and vLLM's own activation run on them, as its experts do between their two
    products (any it applies: a clamp, swigluoai, swiglustep, relu2 without a gate ...); down with the router's weights
    applied and each token's k rows added (mma_moe); no host sync, in
    the CUDA graphs vLLM captures around its MoE op. From decode_min tokens a step (_moe_decode_min; exact: always)
    the experts the tokens are routed to are decoded into the device's scratch buffer (mma_moe_unpack) instead, then run
    by the kernel vLLM runs bf16's experts by (its Triton one; where vLLM picks another, whose weights it lays out
    otherwise, exact is refused and fused mode keeps the grouped products throughout)."""

    def __init__(self, config, moe):
        super().__init__(moe)
        self.config, self.ref = config, None
        self.act = _fused_act(moe)  # (mma_moe's act; 0: vLLM's activation on the sums, with its config as its bf16 experts make it)
        self.act_config = None if self.act else ApplyMoEActivationConfig.from_configs(moe, FusedMoEQuantConfig.make())  # (only where vLLM's activation runs)
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
    def unsupported(layer, exact=False):
        """Why the library's grouped products do not take this layer's experts (they stay bf16), or None. Until measured
        no slower than bf16's (_measured_acts), only SiLU with I a multiple of 64 as before: plain, or with a clamp or its
        own alpha and beta (moe_config's swiglu_limit, swiglu_alpha, swiglu_beta) in exact mode alone, whose decoded route
        runs vLLM's own kernel at every count of tokens."""
        mc = layer.moe_config
        if mc.moe_parallel_config.use_ep:
            return "expert parallel"
        if mc.has_bias:
            return "experts with biases"
        if not apply_moe_activation_supported(mc.activation):
            return f"activation {mc.activation.value}"
        clamp = {a: getattr(mc, "swiglu_" + a, None) for a in ("limit", "alpha", "beta")}
        if not _measured_acts() and not (mc.activation == MoEActivation.SILU and mc.intermediate_size_per_partition % 64 == 0 and (exact or all(v is None for v in clamp.values()))):
            return f"activation {mc.activation.value}" + "".join(f", {a} {v}" for a, v in clamp.items() if v is not None) + f", intermediate {mc.intermediate_size_per_partition}: its products are not measured yet"
        if layer.apply_router_weight_on_input:
            return "router weights applied to the input"
        if layer.params_dtype != torch.bfloat16:
            return f"{layer.params_dtype}, not bf16"
        H, I = mc.hidden_dim, mc.intermediate_size_per_partition
        if H != getattr(mc, "hidden_dim_unpadded", H) or I != getattr(mc, "intermediate_size_per_partition_unpadded", I):
            return f"hidden {H} or intermediate {I} padded for this backend"
        if (2 * I if mc.activation.is_gated else I) % 64 or H % 64 or I % 16:
            return f"hidden {H} or intermediate {I} (a rank's) is not a size the packs take"
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
        pack = g.pack_mma12 if layout == "mma12" else g.pack_smallest
        E = layer.w13_weight.shape[0]
        info = get_layerwise_info(layer)
        miss = _experts_missing([(args.arguments.get("expert_id"), args.arguments.get("shard_id")) for p, args in info.loaded_weights if p in ("w13_weight", "w2_weight")], E, layer.moe_config.is_act_and_mul)
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
            plan = g.moe_way(ids, E)
            if self.ref is not None and x.shape[0] >= self.decode_min:  # the routed experts decoded (the rest of the buffer as it was, unread)
                buf, n13 = _SCRATCH[x.device], up.shape[0] * up.shape[1]
                w1 = g.mma_moe_unpack(up, E, plan, ids.numel(), buf[:n13]).view(E, -1, up.shape[1])
                w2 = g.mma_moe_unpack(down, E, plan, ids.numel(), buf[n13 : n13 + down.shape[0] * down.shape[1]]).view(E, -1, down.shape[1])
                return self.ref.moe_kernel.apply(hidden_states=x, w1=w1, w2=w2, topk_weights=topk_weights, topk_ids=topk_ids, activation=layer.activation, apply_router_weight_on_input=layer.apply_router_weight_on_input, global_num_experts=layer.global_num_experts, expert_map=layer.expert_map, shared_experts=shared_experts, shared_experts_input=shared_experts_input)
            return _experts(up, down, E, x, plan, ids, topk_weights, self.act, layer.activation, self.act_config)
        finally:
            _lib.local.fresh = fresh
