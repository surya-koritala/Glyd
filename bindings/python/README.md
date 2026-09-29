# Glyd for Python

Lossless AI compression: 33% less GPU memory, bit for bit.

    pip install "glyd[gpu]"   # models on the GPU: Linux x86_64 / aarch64, an NVIDIA GPU (Ampere or later)
    pip install glyd          # the codec alone: Linux x86_64 / aarch64, macOS arm64

```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3-8B")               # packed on the GPU as it loads
model = glyd.from_pretrained("Qwen/Qwen3-8B", exact=True)   # logits bit for bit bf16's
```

Every option: [on the GPU](https://github.com/surya-koritala/Glyd/tree/main/bindings/python#on-the-gpu-a-models-weights-held-compressed-bit-for-bit) below, and
[getglyd.com](https://getglyd.com/docs/) for which models fit which GPU.

## The codec

Wheels for Linux x86_64 and aarch64 and macOS arm64. From a checkout:
`bindings/python/build.sh && pip install bindings/python`.

```python
import glyd
c = glyd.compress(data)                       # --max; level="ultra" / "cold" / "default"
c = glyd.compress(log_bytes, records=True)    # logs, dumps, CSV, JSON lines as typed columns
data = glyd.decompress(c)

p = glyd.pack(events)                         # many small objects as one stream
event = glyd.unpack(p, 7)

with glyd.Store("bucket/") as s:              # objects compressed across each other
    i = s.put("wed.tar", data)                # a delta against the object it most resembles
    data = s.get(i)
with glyd.Store("meta/", s3="s3://bucket/prefix") as s:   # objects in S3
    ...
```

A thin ctypes layer over `include/glyd.h` (BSD-3-Clause OR GPL-2.0); no build step
beyond placing the shared library. `build.sh` places `libglyd_store`,
which carries the codec and the store; with `libglyd` alone (the codec
crate, BSD-3-Clause OR GPL-2.0) everything but `Store` works.

## On the GPU: a model's weights held compressed, bit for bit

Needs a CUDA GPU (Ampere or later) and `pip install "glyd[gpu]"`
(PyTorch 2.5+ built for CUDA 12 or 13, transformers 5.17+, accelerate,
safetensors, huggingface_hub). The Linux wheels (x86_64, aarch64; glibc 2.28 or later)
carry the kernels, `libglyd_gpu_cuda12.so` and `libglyd_gpu_cuda13.so`
(CUDA 12.8 and 13.0, their runtime linked in), and the one for PyTorch's
CUDA is taken; `GLYD_GPU_LIB` names another. From a checkout,
`bash gpu/build_lib.sh bindings/python/glyd/gpu` builds the one for your
nvcc. `fit` needs none of it. The kernels are a C library too, on every
release by themselves with their header, for engines in C, C++, Rust or any
language with a C FFI: [gpu/README.md](https://github.com/surya-koritala/Glyd/tree/main/gpu#the-library).

```python
import glyd
from transformers import AutoTokenizer
model = glyd.from_pretrained("Qwen/Qwen3-8B")            # any bf16 checkpoint, packed as it loads, on the GPU
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
out = model.generate(**tok("Hello", return_tensors="pt").to(model.device), max_new_tokens=64)

model = glyd.from_pretrained("Qwen/Qwen3-8B", exact=True) # logits bit-identical to bf16's
glyd.save_pretrained(model, "qwen3-8b-glyd")               # the packed format, loads without repacking
model = glyd.from_pretrained("qwen3-8b-glyd", verify=True) # re-hashes every weight against the source
print(glyd.fit("Qwen/Qwen3-32B", gpu="48GB"))              # will it fit, bf16 against Glyd
```

`glyd.from_pretrained(name_or_path, *, device="cuda:0", layout="auto", exact=False, merge=True, verify=False, **hf_kwargs)`
returns the transformers model (a causal LM, else an image-text-to-text
one), ready for `generate()`. transformers loads it a tensor at a time and
every Linear's weight is packed on the GPU as it arrives: the GPU holds
the packed model and the largest weight not packed yet, the host about a
shard.

- `device`: the GPU. `device_map="auto"` (with accelerate) or a device map
  in `hf_kwargs` spreads the layers over several.
- `layout`: `"auto"` picks for the GPU (the tiered layout on Ada, a
  mixture of experts' on an A10 too, and wherever only it fits; the
  12-bit one on A10, A100 and H100); `"mma"`, the tiered layout (10.8
  bits a weight); `"mma12"`, the 12-bit layout (12.0 bits, a lighter
  decode). Embeddings go in the fast format, their rows decoded as they
  are looked up.
- `exact`: every product decodes its matrix whole and multiplies as
  `nn.Linear` does, so the logits are bf16's bit for bit. By default the
  products run straight from the packed weights (decoded in registers):
  faster, their sums in another order than cuBLAS's, so late tokens may
  differ from bf16's as between any two GEMM kernels.
- `merge`: q, k, v and gate, up as one product each, as serving engines
  run them (not with `exact`).
- `verify`: every pack decoded and compared with its weights bit for bit
  as it is made; from a saved checkpoint, every packed tensor decoded and
  its sha256 checked against `glyd.json`, and every tensor saved as it is
  by its own there (a save of glyd 0.25 on), each file's tensors back to
  back, each a pack's buffer or of a sha256 there.
- `hf_kwargs`: transformers' `from_pretrained`'s (`revision`, `token`,
  `device_map`, `attn_implementation` ...); the dtype is bf16.

`glyd.gpu.compress(model, *, layout="auto", exact=False, merge=True)`
packs a model already loaded in bf16 in place, on the GPU its weights are
on (the current one for weights on the CPU), and returns it
(`glyd.compress` is the codec's, for bytes).

A mixture of experts is packed too, each layer's experts as one matrix:
every family of transformers 5.17, those whose Experts modules it runs
through an experts implementation (OLMoE, granite MoE, Qwen3-MoE,
gpt-oss, DeepSeek V3 ...) by `glyd`, one registered with it, and those
whose own code runs their experts (Llama 4, DBRX, Aria, JetMoE, Step 3.7,
LongCat-Flash) taken over where it multiplies; with `exact=True` the
experts are decoded and bf16's own implementation runs on them. glyd-v1
holds them packed, and a model with them can't be copied or pickled
(`copy.deepcopy`, `torch.save`): load it again.

Compiled: `model.generate(..., cache_implementation="static")` compiles
the forward as transformers does (`torch.compile`,
`mode="reduce-overhead"`: CUDA graphs), and `torch.compile(model.forward,
mode="reduce-overhead", fullgraph=True)` compiles it as it would the bf16
model's: each GLinear and GEmbedding is one op of the graph
(`glyd::linear`, `glyd::embedding`), with no graph break, and the CUDA
graph captures Glyd's kernels, so a step's host time goes and what is
left is its GPU time, less than bf16's. Tokens/s generating 128 tokens at
1 / 8 sequences on an RTX 4080 SUPER, `from_pretrained` as above against
transformers' bf16 model:

| | bf16 eager | Glyd eager | bf16 compiled | Glyd compiled |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 88.9 / 693 | 98.3 / 780 | 151.9 / 1020 | **187.8 / 1282** |
| Qwen3-4B-Instruct-2507 | 61.5 / 455 | 75.7 / 563 | 73.9 / 481 | **95.2 / 612** |
| Qwen3-8B | does not fit | 48.5 / 364 | does not fit | **55.6 / 386** |

Eager, a step's product is one C call, with less host time than
`nn.Linear`'s. Compile the whole forward, as these do: an op names its
module by a number the graph is specialized on, so compiling each layer
on its own compiles every layer anew (and meets torch._dynamo's
recompile limit). With `exact=True` each product in the graph is
`F.linear`'s, as eager, but the logits are the bf16 model's compiled the
same way bit for bit only with
`torch._inductor.config.emulate_precision_casts = True`: by default
Inductor keeps a bf16 value in fp32 across a fused kernel, and fuses
differently around Glyd's op than around bf16's matmul.

`glyd.save_pretrained(model, path)` writes glyd-v1: the packs in the
tiered layout as safetensors (each packed Linear's buffers under its
module path, `.glyd_data`, `.glyd_blocks`, `.glyd_block_base`; a mixture
of experts' weight's under the module holding it, `.glyd_gate_up_proj_data`
and so on), the rest of the model as it is, `glyd.json` (the format, the
source repo and revision, for every packed tensor its shape and the
sha256 of its bf16 bytes, and from glyd 0.25 the sha256 of every tensor
saved as it is), and the source's config, generation config
and tokenizer files. With a mixture of experts' packs the format is
glyd-v2, which glyd 0.21 refuses by its format ("this glyd reads
glyd-v1"): load it with 0.22 or later. `from_pretrained(path)` loads the packs as saved;
on a GPU where the 12-bit layout is the pick, it decodes and packs them
again. `save_pretrained(model, path, layout="mma12")` (`pack --layout
mma12`) saves the packs in the 12-bit layout instead, as an A10, A100 or
H100 runs them (`.glyd_data`, `.glyd_exc`, `.glyd_exc_base`; glyd-v3,
which glyd 0.23 and before refuse): loaded there as saved, 2.1-3.9x
faster than packing again (on an RTX 4080 SUPER, Qwen3-8B in 1.17-1.18 s
against 3.14-4.55 s from a tiered save and 3.54-3.61 s from the bf16
checkpoint; benchmarks/gpu/rtx4080s-rust-2026-09-28).

`glyd.fit(name_or_path, gpu="48GB", context=8192)`: whether the model
fits one GPU in bf16 and with Glyd, by the site's rule: the weights (with
Glyd, bf16 tensors at the ratio measured for the model, else 0.673 of
their bytes; FP8 and 4-bit as they are), a KV cache for `context` tokens
and 1.5 GiB for the runtime, against the memory nvidia-smi reports
(`16GB` ... `141GB`, or a number of bytes). From the Hub's metadata for a
repo id, from the files for a directory.

    python -m glyd.gpu fit Qwen/Qwen3-32B --gpu 48GB
    python -m glyd.gpu pack Qwen/Qwen3-8B qwen3-8b-glyd     # packed, checked, saved as glyd-v1
    python -m glyd.gpu verify qwen3-8b-glyd

`glyd pack` and `glyd verify`, the Rust CLI's (the glyd-gpu command),
do the same on the CPU with no Python, PyTorch or GPU, and save the same
bytes (Qwen3, Qwen2, Llama, Mistral, Granite and GraniteMoe for now; other
families: the commands above).

`glyd.gpu` is under the Business Source License 1.1 (`LICENSE-glyd-gpu`),
as the rest of Glyd's GPU code; the codec under BSD-3-Clause OR GPL-2.0.
