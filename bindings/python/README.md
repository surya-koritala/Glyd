# Glyd for Python

Lossless AI compression: 33% less GPU memory, bit for bit.

    curl -LsSf https://getglyd.com/install.sh | sh   # a model on your GPU in two commands: Linux, an NVIDIA GPU, driver 580 or newer
    glyd run Qwen/Qwen3.5-9B                         # downloads it, starts it packed, and chats (glyd serve, glyd doctor)
    glyd run Qwen/Qwen3.8-27B:swift                  # fewer bits, not lossless: :kestrel 6.5 bits per weight, :swift 5.5

    pip install "glyd[gpu]"   # models on the GPU: Linux x86_64 / aarch64, an NVIDIA GPU (Ampere or later)
    pip install "glyd[vllm]"  # and serving them: vllm serve MODEL --quantization glyd (vLLM 0.30)
    pip install glyd          # the codec alone: Linux x86_64 / aarch64, macOS arm64

```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3.5-9B")               # packed on the GPU as it loads
model = glyd.from_pretrained("Qwen/Qwen3.5-9B", exact=True)   # logits exactly bf16's
```

Every option: [on the GPU](https://github.com/surya-koritala/Glyd/tree/main/bindings/python#on-the-gpu-a-models-weights-held-compressed) below, and
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

A thin ctypes layer over `include/glyd.h`; no build step
beyond placing the shared library. `build.sh` places `libglyd_store`,
which carries the codec and the store; with `libglyd` alone (the codec
crate) everything but `Store` works.

## On the GPU: a model's weights held compressed

Needs a CUDA GPU (Ampere or later) and `pip install "glyd[gpu]"`
(PyTorch 2.5+ built for CUDA 12 or 13, transformers 5.17+, accelerate,
safetensors, huggingface_hub). The `glyd-gpu` Linux wheels it brings (x86_64, aarch64; glibc 2.28 or later)
carry the kernels, `libglyd_gpu_cuda12.so` and `libglyd_gpu_cuda13.so`
(CUDA 12.8 and 13.0, their runtime linked in), and the one for PyTorch's
CUDA is taken; `GLYD_GPU_LIB` names another. `fit` needs none of it. The
libraries are a C library too, with their header, `glyd_gpu.h` (8 functions), in the
wheel's `glyd_gpu` directory, for engines in C, C++, Rust or any language with
a C FFI.

```python
import glyd
from transformers import AutoTokenizer
model = glyd.from_pretrained("Qwen/Qwen3.5-9B")            # any bf16 checkpoint, packed as it loads, on the GPU
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
out = model.generate(**tok("Hello", return_tensors="pt").to(model.device), max_new_tokens=64)

model = glyd.from_pretrained("Qwen/Qwen3.5-9B", exact=True) # logits exactly bf16's
glyd.save_pretrained(model, "qwen3.5-9b-glyd")               # the packed format, loads without repacking
model = glyd.from_pretrained("qwen3.5-9b-glyd", verify=True) # re-hashes every weight against the source
print(glyd.fit("Qwen/Qwen3.5-27B", gpu="48GB"))              # will it fit, bf16 against Glyd
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
  fits; `mma12` on A10, A100 and H100); `"mma"`, the most memory off (10.8 bits per weight); `"mma12"`, 12.0 bits a
  weight, the faster one on an A10, A100 and H100. Embeddings are held compressed too, their rows rebuilt as they
  are looked up.
- `exact`: the logits are exactly bf16's (every product multiplies as `nn.Linear` does). By default the
  products run straight from the packed weights: faster, their sums in another order than bf16's, so late tokens
  may differ from bf16's as between any two GEMM kernels.
- `merge`: q, k, v and gate, up as one product each, as serving engines
  run them (not with `exact`).
- `verify`: every pack unpacked and compared with its weights as it is made; from a saved checkpoint,
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

A mixture of experts is packed too: every mixture-of-experts family of transformers 5.17; with `exact=True` the experts' products
are bf16's own. A saved model holds them packed, and a model with them can't be copied or pickled
(`copy.deepcopy`, `torch.save`): load it again.

On PyTorch 2.13.0 or later (measured on 2.14), `generate()` is compiled;
below it, a 2.13 pre-release included, it runs eager, as in glyd 0.23. A
compiled call runs as `generate(..., cache_implementation="static")`
asks transformers to run it, a static cache and the forward compiled
(`torch.compile`, `mode="reduce-overhead"`: CUDA graphs), so a step's
host time goes and what is left is its GPU time. The
prompt runs eager (transformers compiles the steps after it).

Greedy tokens compiled can differ from 0.23's eager loop's, as a
compiled bf16 model's can from its eager ones. In the default mode
they can also vary within a process, between calls whose cache sizes
compile differently;
`exact=True` is never compiled and stays exactly bf16's. The
first call compiles and captures (faster in a later process, from
PyTorch's compile caches); a
call whose cache is longer than any before it compiles again once, then
captures a graph. Before serving, warm up with one
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
GeForce card and 2048 on another: past that the eager loop is as fast,
sooner the faster the host's CPU.

`GLYD_COMPILE_MAX` sets the cap on any GPU. Not with `exact=True`
(below), a family transformers does not compile whole (its
`_can_compile_fullgraph`), the model over several GPUs, or a
transformers whose generation helpers are not as 5.17 has them (one
warning at the load), which run eager, nor where transformers 5.17's
static cache fails (bf16's too), which run eager from the start: a family none of whose forwards
transformers compiles, and a model with
multi-head latent attention whose config has fewer key/value heads than
heads (released checkpoints, with as many as heads, compile). A call whose
forward fails to compile anyway (torch._dynamo's or Inductor's error)
runs again eager, from its start (a streamer gets only what the failed
attempt had not streamed, and a sampled call draws again from the random
state it started with: its tokens and text are the eager run's), and so
do the model's later calls, with one warning; any other error (out of
memory included) is the call's own, and the next call compiles as
before. Several models in one process all compile; the process's own `torch._dynamo` settings are left as they are. A
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

`glyd.save_pretrained(model, path)` writes glyd-v4 (from glyd 0.28): the packed weights as safetensors, the rest of the
model as it is, `glyd.json` (the format, the source repo and revision, and for every packed tensor its shape and the
sha256 of its bf16 bytes; from glyd 0.25 the sha256 of every tensor saved as it is), and the source's config,
generation config and tokenizer files. Glyd 0.27 and before wrote glyd-v1, and glyd-v2 with a mixture of experts'
packs; glyd 0.28 loads both and packs them again as they load, and glyd 0.27 and before refuse a glyd-v4 save by its
format. `from_pretrained(path)` loads the packs as saved; on a GPU where `mma12` is the pick, it decodes and packs
them again. `save_pretrained(model, path, layout="mma12")` (`pack
--layout mma12`) saves the `mma12` layout instead, as an A10, A100 or H100 runs it (glyd-v3, which glyd 0.24 and
before refuse): loaded there as saved, without packing again.

`glyd.fit(name_or_path, gpu="48GB", context=8192)`: whether the model
fits one GPU in bf16 and with Glyd, by the site's rule: the weights (with
Glyd, bf16 tensors at the ratio measured for the model, else 0.673 of
their bytes; FP8 and 4-bit as they are), a KV cache for `context` tokens
and 1.5 GiB for the runtime, against the memory nvidia-smi reports
(`16GB` ... `141GB`, or a number of bytes). From the Hub's metadata for a
repo id, from the files for a directory.

    python -m glyd_gpu fit Qwen/Qwen3.5-27B --gpu 48GB
    python -m glyd_gpu pack Qwen/Qwen3.5-9B qwen3.5-9b-glyd     # packed, checked, saved as glyd-v4
    python -m glyd_gpu verify qwen3.5-9b-glyd

`glyd pack` and `glyd verify` are the same two commands (the `glyd` program passes them to the Python tool).

### Serving with vLLM

`glyd run MODEL` and `glyd serve MODEL` (installed by the script above, or by
`pip install "glyd[vllm]"` in a virtual environment) start vLLM with the
plugin and the settings worked out from the GPU: memory share, context,
the tool-call and reasoning parsers, compiled wherever the chat keeps 8,192 tokens of context
(eager only where that gives a longer chat); `glyd doctor` checks
the machine. [The docs](https://getglyd.com/docs/vllm/)
have the steps. By hand, `pip install "glyd[vllm]"` installs vLLM 0.30 and
the plugin, which vLLM finds by itself (the package's `vllm.general_plugins`
entry point):

    vllm serve Qwen/Qwen3.5-9B --quantization glyd       # a bf16 checkpoint, packed as it loads
    vllm serve ./SAVE --quantization glyd                # a glyd save, as saved (not yet of a 2026 model)
    vllm serve Qwen/Qwen3.5-9B --quantization glyd --enforce-eager --additional-config '{"glyd": {"exact": true}}'

vLLM's own settings need not fit the card: Qwen3.5-9B in bf16 ran out of memory at start
on an RTX 4090 (24 GB), even with the context capped at 4,096 tokens, and Glyd's
settings run it there packed with a 189,440-token context
([log](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/rtx4090-5090-qwen35-9b-v029-2026-10-06)).
`glyd run` works both out from the card, and
[the docs](https://getglyd.com/docs/vllm/#advanced-vllm-serve-by-hand)
have the flags by hand.

`layout`, `exact` and `verify` are `from_pretrained`'s options, given in
`--additional-config`'s `"glyd"` (or `GLYD_LAYOUT`, `GLYD_EXACT`,
`GLYD_VERIFY`). `fraction` (0 to 1, `GLYD_FRACTION`) packs only that share
of the decoder layers, spread evenly over the depth, and leaves the rest as
vLLM runs them: 0 is bf16, 1 (the default) every layer, and from v0.28 the
embedding and the output layer too (every embedding row comes back bit for
bit; `exact` keeps the output layer as vLLM runs it). vLLM sizes its KV
cache after the weights load, so the memory the packs save becomes KV
cache: Qwen3.8-27B on an H100, 204,117 tokens against bf16's 122,538 at the same
`--gpu-memory-utilization` (benchmarks/gpu/h100-qwen38-v028-2026-10-03). With `exact` the logits are exactly vLLM's bf16
ones, eager, or compiled in inductor's deterministic mode where the
packed Linears have no biases.
`kv` (`auto`, the default, `lossless` or `off`; `GLYD_KV`) holds vLLM's KV cache
in fewer bits, every value read back exactly, for models with head size 128 and
full attention only: none of the 2026 models measured so far (Qwen3.5, Qwen3.8, Gemma 4)
is one, so they keep vLLM's cache, with a line in the log. `auto` holds it on an A100, an L4, an H100 and a GH200 and
leaves vLLM's own cache on any other GPU, with a line in the log; `glyd run` keeps
vLLM's cache unless `GLYD_KV` is set, because the first start sets the cache up
for the model.
Throughput against bf16, exact mode
compiled, mixtures of experts and what is not supported yet:
[the docs](https://getglyd.com/docs/vllm/).

The GPU half is the `glyd-gpu` package (compiled wheels; `glyd.gpu` re-exports its API).

## License

From v0.29.0 all of Glyd is under the Business Source License 1.1 (`LICENSE`). Free forever for personal and non-commercial use (including education, research and nonprofits) on computers you own or rent for yourself, and for anyone to try, test and develop with. Commercial use needs a license: suryakoritala@getglyd.com. Each version becomes Apache-2.0 four years after its release.

Releases up to v0.28 keep the licenses they shipped with.
