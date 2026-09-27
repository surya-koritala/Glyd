# Glyd for Python

    pip install glyd

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
nvcc. `fit` needs none of it.

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
- `layout`: `"auto"` picks for the GPU (the tiered layout on Ada, and
  wherever only it fits; the 12-bit one on A10, A100 and H100); `"mma"`,
  the tiered layout (10.8 bits a weight); `"mma12"`, the 12-bit layout
  (12.0 bits, a lighter decode). Embeddings go in the fast format, their
  rows decoded as they are looked up.
- `exact`: every product decodes its matrix whole and multiplies as
  `nn.Linear` does, so the logits are bf16's bit for bit. By default the
  products run straight from the packed weights (decoded in registers):
  faster, their sums in another order than cuBLAS's, so late tokens may
  differ from bf16's as between any two GEMM kernels.
- `merge`: q, k, v and gate, up as one product each, as serving engines
  run them (not with `exact`).
- `verify`: every pack decoded and compared with its weights bit for bit
  as it is made; from a saved checkpoint, every tensor decoded and its
  sha256 checked against `glyd.json`.
- `hf_kwargs`: transformers' `from_pretrained`'s (`revision`, `token`,
  `device_map`, `attn_implementation` ...); the dtype is bf16.

`glyd.gpu.compress(model, *, layout="auto", exact=False, merge=True)`
packs a model already loaded in bf16 in place, on the GPU its weights are
on (the current one for weights on the CPU), and returns it
(`glyd.compress` is the codec's, for bytes).

`glyd.save_pretrained(model, path)` writes glyd-v1: the packs in the
tiered layout as safetensors (each packed Linear's buffers under its
module path, `.glyd_data`, `.glyd_blocks`, `.glyd_block_base`), the rest
of the model as it is, `glyd.json` (the format, the source repo and
revision, and for every packed tensor its shape and the sha256 of its
bf16 bytes), and the source's config, generation config and tokenizer
files. `from_pretrained(path)` loads the packs as saved; on a GPU where
the 12-bit layout is the pick, it decodes and packs them again.

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

`glyd.gpu` is under the Business Source License 1.1 (`LICENSE-glyd-gpu`),
as the rest of Glyd's GPU code; the codec under BSD-3-Clause OR GPL-2.0.
