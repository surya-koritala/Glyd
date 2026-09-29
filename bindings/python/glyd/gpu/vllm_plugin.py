"""vLLM's "glyd" quantization method (M1 spike): a model's Linears packed on
the GPU as vLLM loads them, their products by the library's kernels.

    vllm serve Qwen/Qwen3-8B --quantization glyd

vLLM calls register() in every process it starts (the entry point
vllm.general_plugins: glyd); from a script, call it before LLM(...).
GLYD_LAYOUT (auto, mma, mma12) chooses the layout, GLYD_EXACT=1 the exact
mode (each product's matrix decoded, then vLLM's own bf16 GEMM). A Linear's
product is one op, glyd::mma_linear, over its pack's tensors: torch.compile
sees it as one node and CUDA graphs capture its kernels. The classes are
this module's own (not made at the call): vLLM pickles its config, the
quantization's with it, to start its engine's process.
"""
import os
import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.reload.layerwise import initialize_online_processing
from vllm.model_executor.parameter import ModelWeightParameter
from . import _lib, kernels as g

NAME = "glyd"
_NOBIAS = {}  # a device's empty bf16 tensor: no bias
_OPS = []  # the ops' library (they live as long as it does)


def _define():
    """glyd::mma_linear: X W^T (+ bias) over a pack (data and its two tensors, and its words: the tiered layout's three
    tiers, the 12-bit one's four), by the library's route for X's rows on this GPU (-1); glyd::mma_unpack: its matrix,
    bf16."""
    if _OPS:
        return
    lib = torch.library.Library("glyd", "FRAGMENT")
    lib.define("mma_linear(Tensor x, Tensor data, Tensor a, Tensor b, int[] words, Tensor? bias, int out_features) -> Tensor")
    lib.define("mma_unpack(Tensor data, Tensor a, Tensor b, int[] words, int rows, int cols) -> Tensor")

    def linear(x, data, a, b, words, bias, out_features):
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.empty(x2.shape[0], out_features, dtype=torch.bfloat16, device=x.device)
        _NOBIAS.setdefault(x.device, torch.empty(0, dtype=torch.bfloat16, device=x.device))
        if x2.shape[0]:
            fresh, _lib.local.fresh = _lib.local.fresh, True  # workspaces for the call alone (a CUDA graph's pool)
            try:
                f = _lib.mma12_linear if len(words) == 4 else _lib.mma_linear
                f(data, a, b, list(words), out_features, K, x2, bias if bias is not None else _NOBIAS[x.device], y, -1)
            finally:
                _lib.local.fresh = fresh
        return y.view(*x.shape[:-1], out_features)

    def unpack(data, a, b, words, rows, cols):
        w = torch.empty(rows, cols, dtype=torch.bfloat16, device=data.device)
        (_lib.mma12_unpack if len(words) == 4 else _lib.mma_unpack)(data, a, b, list(words), cols, 0, rows, w.view(torch.int16), 0)
        return w

    lib.impl("mma_linear", linear, "CUDA")
    lib.impl("mma_unpack", unpack, "CUDA")
    torch.library.register_fake("glyd::mma_linear", lambda x, data, a, b, words, bias, out_features: x.new_empty((*x.shape[:-1], out_features)), lib=lib)
    torch.library.register_fake("glyd::mma_unpack", lambda data, a, b, words, rows, cols: data.new_empty((rows, cols), dtype=torch.bfloat16), lib=lib)
    _OPS.append(lib)


def register():
    """vLLM's general plugin: the "glyd" quantization method, registered in this process (again: the same)."""
    _define()
    register_quantization_config(NAME)(GlydConfig)


class GlydConfig(QuantizationConfig):
    def __init__(self, layout=None, exact=False):
        super().__init__()
        self.layout = layout or os.environ.get("GLYD_LAYOUT", "auto")
        self.exact = exact or os.environ.get("GLYD_EXACT", "0") == "1"

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
        return cls(config.get("layout"), bool(config.get("exact", False)))

    def get_quant_method(self, layer, prefix):
        return GlydLinearMethod(self) if isinstance(layer, LinearBase) else None


class GlydLinearMethod(LinearMethodBase):
    """The weight loaded in bf16 on the meta device, a layer at a time materialized and packed as vLLM's layerwise
    processing completes it (process_weights_after_loading); a matrix the mma layouts do not take (rows not a multiple
    of 64, columns not of 16) kept in bf16, as vLLM runs it."""

    uses_meta_device = True

    def __init__(self, config):
        self.config = config
        self.plain = None
        self.gemm = UnquantizedLinearMethod()._gemm_impl if config.exact else None  # exact: vLLM's bf16 GEMM

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra):
        O, K = sum(output_partition_sizes), input_size_per_partition
        if O % 64 or K % 16 or params_dtype != torch.bfloat16:
            self.plain, self.uses_meta_device = UnquantizedLinearMethod(), False
            return self.plain.create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra)
        weight = ModelWeightParameter(data=torch.empty(O, K, device="meta", dtype=params_dtype), input_dim=1, output_dim=0, weight_loader=extra.get("weight_loader"))
        layer.register_parameter("weight", weight)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer):
        if self.plain is not None:
            return self.plain.process_weights_after_loading(layer)
        if getattr(layer, "glyd_words", None) is not None:
            return
        w = layer.weight.data
        layout = self.config.layout
        if layout == "auto":
            layout = g.best_layout(0, 0, 1, w.device)[0]
        p = (g.pack_mma12 if layout == "mma12" else g.pack_mma)(w)
        g.lib()  # the library loaded (the ops' calls are its C API's)
        for name, t in zip(("glyd_data", "glyd_a", "glyd_b"), (p.data, p.exc, p.exc_base) if layout == "mma12" else (p.data, p.blocks, p.block_base)):
            layer.register_buffer(name, t, persistent=False)
        layer.glyd_words = list(p.sym if layout == "mma12" else p.tiers)
        layer.glyd_out = w.shape[0]
        layer.weight = torch.nn.Parameter(torch.empty(0, w.shape[1], dtype=w.dtype, device=w.device), requires_grad=False)
        for name in ("mma12_linear", "mma_linear"):  # the device's done counters, made now: never in a CUDA graph's pool
            _lib._counters(name, w.device.index, None, 0, _lib._UNITS)

    def apply(self, layer, x, bias=None):
        if self.plain is not None:
            return self.plain.apply(layer, x, bias)
        if self.gemm is not None:  # exact: the matrix decoded, then the GEMM vLLM runs for bf16
            w = torch.ops.glyd.mma_unpack(layer.glyd_data, layer.glyd_a, layer.glyd_b, layer.glyd_words, layer.glyd_out, layer.weight.shape[1])
            return self.gemm(layer, x, w, bias)
        return torch.ops.glyd.mma_linear(x, layer.glyd_data, layer.glyd_a, layer.glyd_b, layer.glyd_words, bias, layer.glyd_out)
