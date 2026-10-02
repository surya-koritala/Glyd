# Glyd for Python

Lossless AI compression: 33% less GPU memory, bit for bit.

    curl -LsSf https://getglyd.com/install.sh | sh   # a model on your GPU in two commands: Linux, an NVIDIA GPU, driver 580 or newer
    glyd run Qwen/Qwen3-8B                           # downloads it, starts it packed, and chats (glyd serve, glyd doctor)

    pip install "glyd[gpu]"   # models on the GPU: Linux x86_64 / aarch64, an NVIDIA GPU (Ampere or later)
    pip install "glyd[vllm]"  # and serving them: vllm serve MODEL --quantization glyd (vLLM 0.30)
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

`glyd.from_pretrained(name_or_path, *, device="cuda:0", layout="auto", exact=False, merge=True, verify=False, compile=True, **hf_kwargs)`
returns the transformers model (a causal LM, else an image-text-to-text
one), ready for `generate()`. transformers loads it a tensor at a time and
every Linear's weight is packed on the GPU as it arrives: the GPU holds
the packed model and the largest weight not packed yet, the host about a
shard.

- `device`: the GPU. `device_map="auto"` (with accelerate) or a device map
  in `hf_kwargs` spreads the layers over several.
- `layout`: `"auto"` picks for the GPU (`mma` on Ada, for a mixture of experts on an A10 too, and wherever only it
  fits; `mma12` on A10, A100 and H100); `"mma"`, the most memory off (10.8 bits a weight); `"mma12"`, 12.0 bits a
  weight, the faster one on an A10, A100 and H100. Embeddings are held compressed too, their rows rebuilt as they
  are looked up.
- `exact`: the logits are bf16's bit for bit (every product multiplies as `nn.Linear` does). By default the
  products run straight from the packed weights: faster, their sums in another order than bf16's, so late tokens
  may differ from bf16's as between any two GEMM kernels.
- `merge`: q, k, v and gate, up as one product each, as serving engines
  run them (not with `exact`).
- `verify`: every pack unpacked and compared with its weights bit for bit as it is made; from a saved checkpoint,
  every packed tensor unpacked and checked against the sha256 in `glyd.json`, and every other tensor against its
  own (a save of glyd 0.25 on).
- `compile`: `generate()` compiled (below); `False`, or `GLYD_COMPILE=0`
  in the environment, runs it eager, as transformers runs it.
- `hf_kwargs`: transformers' `from_pretrained`'s (`revision`, `token`,
  `device_map`, `attn_implementation` ...); the dtype is bf16.

`glyd.gpu.compress(model, *, layout="auto", exact=False, merge=True, compile=True)`
packs a model already loaded in bf16 in place, on the GPU its weights are
on (the current one for weights on the CPU), and returns it
(`glyd.compress` is the codec's, for bytes).

A mixture of experts is packed too: every family of transformers 5.17 (OLMoE, granite MoE, Qwen3-MoE, gpt-oss,
DeepSeek V3, Llama 4, DBRX, Aria, JetMoE, Step 3.7, LongCat-Flash ...); with `exact=True` the experts' products
are bf16's own. A saved model holds them packed, and a model with them can't be copied or pickled
(`copy.deepcopy`, `torch.save`): load it again.

On PyTorch 2.13.0 or later (measured on 2.14), `generate()` is compiled;
below it, a 2.13 pre-release included, it runs eager, as in glyd 0.23. A
compiled call runs as `generate(..., cache_implementation="static")`
asks transformers to run it, a static cache and the forward compiled
(`torch.compile`, `mode="reduce-overhead"`: CUDA graphs), so a step's
host time goes and what is left is its GPU time, less than bf16's. The
prompt runs eager (transformers compiles the steps after it). Tokens/s
generating 128 tokens at 1 / 8 sequences on an RTX 4080 SUPER (a Ryzen 9
7950X3D), each in a process of its own, the median of 3 runs (logs:
benchmarks/gpu/rtx4080s-fastloop-2026-09-28, `gen-main.txt` and
`gen-branch.txt`):

| | bf16 | bf16, `cache_implementation="static"` | Glyd, `compile=False` | **Glyd** |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 88.4 / 694 | 147.9 / 994 | 96.6 / 772 | **184.6 / 1260** |
| Qwen3-4B-Instruct-2507 | 61.8 / 456 | 73.1 / 477 | 75.1 / 563 | **94.3 / 610** |
| Qwen3-8B | does not fit | does not fit | 48.6 / 365 | **55.3 / 386** |
| granite-3.1-3b-a800m-instruct (a mixture of experts) | | | 90.3 / 699 | **232.3 / 1547** |

Greedy tokens compiled can differ from 0.23's eager loop's, as a
compiled bf16 model's can from its eager ones (the first 8 of 32 the
same on Qwen3-0.6B, 17 on Qwen3-1.7B, 32 on
granite-3.1-3b-a800m-instruct: `gpu/check_api.py`). In the default mode
they can also vary within a process, between calls whose cache sizes
compile differently (Qwen3-1.7B's first compiled call and its later
ones, after a longer cache, shared 13 of 32 in one run of
`gpu/check_api.py`:
benchmarks/gpu/rtx4080s-fastloop-2026-09-28/checks-merge-6e17b3f);
`exact=True` is never compiled and stays bit-identical to bf16. The
first call compiles and captures: Qwen3-8B's took 17.5 s with PyTorch's
compile caches empty, 6.7 s in a later process (4-6 s for the others); a
call whose cache is longer than any before it compiles again once, then
captures a graph (Qwen3-4B-Instruct-2507, a chat's turns: 15.0 s, then
5.6 s, then 0.8-0.9 s for 64 tokens). Before serving, warm up with one
short `generate()`: the first compiled step comes after the first token
is streamed, so a streamer's consumer waits through the compile (give a
`TextIteratorStreamer` a `timeout` longer than it, or use
`compile=False`). Only greedy and sampled calls compile: a call runs as
transformers runs it if it uses several beams, an assistant or another
assisted mode (prompt lookup, early exit, `use_mtp`), its own cache or a
`cache_implementation`, `use_cache=False`, `return_dict_in_generate`,
attentions or hidden states, `custom_generate`, or
`disable_compile=True` (one call eager); so does one whose static cache
would hold more positions in all (its sequences times the prompt and
`max_new_tokens`, or `max_cache_len` where longer) than 1280 on a
GeForce card and 2048 on another: past that the eager loop is as fast
(Qwen3-8B), sooner the faster the host's CPU. A step's ms compiled
against eager, Qwen3-8B, the static cache that long with 64 positions
used:

| | 256 | 1024 | 2048 | 4096 positions | 8 sequences |
| :--- | ---: | ---: | ---: | ---: | :--- |
| RTX 4080 SUPER, Ryzen 9 7950X3D (a desktop) | 18.2 / 20.5 | 20.4 / 20.5 | 23.0 / 20.5 | 27.6 / 20.5 | 19.5 / 21.8 with 80 each |
| A10, Xeon Platinum 8358 (a server) | 28.3 / 38.3 | 31.3 / 36.9 | 34.9 / 37.2 | 42.8 / 33.6 | 33.9 / 35.8 with 256 each, 52.6 / 33.6 with 1024 |

`GLYD_COMPILE_MAX` sets the cap on any GPU. Not with `exact=True`
(below), a family transformers does not compile whole (its
`_can_compile_fullgraph`), the model over several GPUs, or a
transformers whose generation helpers are not as 5.17 has them (one
warning at the load), which run eager, nor where transformers 5.17's
static cache fails (bf16's too), which run eager from the start: Llama 4
(transformers compiles none of its forwards), and a model with
multi-head latent attention whose config has fewer key/value heads than
heads (as tiny DeepSeek V2 and V3, Kimi Linear and AXK1 test models do;
the released checkpoints, with as many as heads, compile). A call whose
forward fails to compile anyway (torch._dynamo's or Inductor's error)
runs again eager, from its start (a streamer gets only what the failed
attempt had not streamed, and a sampled call draws again from the random
state it started with: its tokens and text are the eager run's), and so
do the model's later calls, with one warning; any other error (out of
memory included) is the call's own, and the next call compiles as
before. Several models in one process all compile (ten Qwen3-0.6B models one after another: the second to tenth at
296.5-301.3 tokens/s against 99.6 eager); the process's own `torch._dynamo` settings are left as they are. A
model that has generated compiled is freed at `del`, as an eager one.

A bf16 model, or one loaded with `compile=False`, runs transformers' own `generate()`. `compile=False`, or
`GLYD_COMPILE=0` before the load, takes nothing over. transformers sets `TOKENIZERS_PARALLELISM=0` for the process
where it compiles; Glyd puts back the value it had, or its absence, after each compiled call (two calls at once in
two threads can leave it 0, as transformers' own compiled calls do).

`torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)`
compiles the forward as it would the bf16 model's, likewise. Eager, a
step's product has less host time than `nn.Linear`'s. Compile the whole
forward, as these do: compiling each layer on its own compiles every
layer anew. With `exact=True` each product in the graph is `F.linear`'s,
as eager, but the logits are the bf16 model's compiled the same way bit
for bit only with `torch._inductor.config.emulate_precision_casts = True`
(so an exact model's `generate()` stays eager: its tokens are bf16's
eager ones).

`glyd.save_pretrained(model, path)` writes glyd-v1: the packed weights as safetensors, the rest of the model as it
is, `glyd.json` (the format, the source repo and revision, and for every packed tensor its shape and the sha256 of
its bf16 bytes; from glyd 0.25 the sha256 of every tensor saved as it is), and the source's config, generation config
and tokenizer files. With a mixture of experts' packs the format is glyd-v2, which glyd 0.21 refuses by its format
("this glyd reads glyd-v1"): load it with 0.22 or later. `from_pretrained(path)` loads the packs as saved; on a GPU
where `mma12` is the pick, it decodes and packs them again. `save_pretrained(model, path, layout="mma12")` (`pack
--layout mma12`) saves the `mma12` layout instead, as an A10, A100 or H100 runs it (glyd-v3, which glyd 0.24 and
before refuse): loaded there as saved, 2.7-3.9x faster than packing again (on an RTX 4080 SUPER, Qwen3-8B in
1.17-1.18 s against 3.14-4.55 s from an `mma` save and 3.54-3.61 s from the bf16 checkpoint;
benchmarks/gpu/rtx4080s-rust-2026-09-28).

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

### Serving with vLLM

`glyd run MODEL` and `glyd serve MODEL` (installed by the script above, or by
`pip install "glyd[vllm]"` in a virtual environment) start vLLM with the
plugin and the settings worked out from the GPU: memory share, context,
the tool-call and reasoning parsers (eager mode, which starts in under a
third of compiled's time and runs within 3% of its speed); `glyd doctor` checks
the machine. [gpu/vllm](https://github.com/surya-koritala/Glyd/tree/main/gpu/vllm)
has the steps. By hand, `pip install "glyd[vllm]"` installs vLLM 0.30 and
the plugin, which vLLM finds by itself (the package's `vllm.general_plugins`
entry point):

    vllm serve Qwen/Qwen3-8B --quantization glyd       # a bf16 checkpoint, packed as it loads
    vllm serve ./qwen3-8b-glyd --quantization glyd     # a glyd save, as saved
    vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager --additional-config '{"glyd": {"exact": true}}'

On a 16 GB card these defaults do not leave room for a chat (vLLM takes
0.92 of the memory and sizes the context to the model's 40,960 tokens):
`glyd run` works both out from the card, and
[gpu/vllm](https://github.com/surya-koritala/Glyd/tree/main/gpu/vllm#advanced-vllm-serve-by-hand)
has the flags by hand.

`layout`, `exact` and `verify` are `from_pretrained`'s options, given in
`--additional-config`'s `"glyd"` (or `GLYD_LAYOUT`, `GLYD_EXACT`,
`GLYD_VERIFY`). `fraction` (0 to 1, `GLYD_FRACTION`) packs only that share
of the decoder layers, spread evenly over the depth, and leaves the rest as
vLLM runs them: 0 is bf16, 1 (the default) every layer. vLLM sizes its KV
cache after the weights load, so the memory the packs save becomes KV
cache: 1.04-2.11x bf16's on an L4, an A10, an A100, a GH200 and an H100 SXM, at the same
`--gpu-memory-utilization`. With `exact` the logits are vLLM's bf16 ones
bit for bit, eager, or compiled in inductor's deterministic mode where the
packed Linears have no biases.
Throughput against bf16, exact mode
compiled, mixtures of experts and what is not supported yet:
[gpu/vllm](https://github.com/surya-koritala/Glyd/tree/main/gpu/vllm).

`glyd.gpu` is under the Business Source License 1.1 (`LICENSE-glyd-gpu`),
as the rest of Glyd's GPU code; the codec under BSD-3-Clause OR GPL-2.0.
