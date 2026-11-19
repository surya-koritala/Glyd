# Glyd weights on the GPU

Glyd holds a bf16 model's weights, and its KV cache, in fewer bits in GPU memory and rebuilds the exact
values on the GPU as the model runs. The model is the same bit for bit, in 33% less GPU memory: a larger
model on the same GPU, or more room for KV cache and more requests at once.

Two layouts, chosen with `layout` (`--format` in the scripts):

| Layout  | What it is for                                                                           | Memory off bf16's weights |
| :------ | :--------------------------------------------------------------------------------------- | ------------------------: |
| `mma`   | The most memory off. Also the faster one on GeForce Ada GPUs at a few sequences.         |                       33% |
| `mma12` | Less memory off. The faster one on an A100 and an H100, and on an A10 at many sequences. |                       25% |
| `auto`  | The default: the layout for this GPU.                                                    |                           |

`fast` and `huffman` are earlier layouts, kept in `e2e.py` for comparison. No lossless coding takes more
than about 34% off bf16 weights: the floor measured is about 10.6 bits a weight.

## Requirements

- An NVIDIA GPU of the Ampere generation or newer: RTX 30 and 40 series, A10, A100, L4, L40S, H100, GH200
  and later. The library carries native code for Ampere, Ada, Hopper and Blackwell (sm_100, sm_120);
  Blackwell's has run on an RTX 5090 (sm_120).
- PyTorch 2.5 or later built for CUDA 12 or 13, and a driver for it (580 or newer for CUDA 13).
- Linux on x86_64 or aarch64, glibc 2.28 or later.
- transformers 5.17 or later, accelerate, safetensors and huggingface_hub (the `gpu` extra installs them).
- For serving: vLLM 0.30 (the `vllm` extra pins `vllm>=0.30,<0.31`).

## Install

```bash
curl -LsSf https://getglyd.com/install.sh | sh   # Glyd, vLLM 0.30 and PyTorch as one tool: Linux, an NVIDIA GPU, driver 580 or newer
pip install "glyd[gpu]"                          # models on the GPU (glyd.from_pretrained)
pip install "glyd[vllm]"                         # and serving them: vllm serve MODEL --quantization glyd (vLLM 0.30)
```

The wheels carry the prebuilt library, for PyTorch's CUDA 12 or 13. From a checkout, `bash build_lib.sh`
builds it (it needs nvcc) next to `glyd_gpu_dev.py`; without it the scripts here build the extension on first
import. `GLYD_GPU_LIB` names another library.

## Use

```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3.5-9B")               # packed on the GPU as it loads
model = glyd.from_pretrained("Qwen/Qwen3.5-9B", exact=True)   # logits bit for bit bf16's
```

```bash
glyd pack Qwen/Qwen3.5-9B qwen3.5-9b-glyd   # a model saved packed
glyd verify qwen3.5-9b-glyd                 # every pack and tensor checked
glyd run Qwen/Qwen3.5-9B                    # downloads it, starts it packed with vLLM, and chats
```

- Serving with vLLM, `vllm serve MODEL --quantization glyd`: [vllm/README.md](vllm/README.md).
- Every option of the Python API: [bindings/python/README.md](../bindings/python/README.md#on-the-gpu-a-models-weights-held-compressed-bit-for-bit).
- C, C++ and Rust: [the library](#the-library).

## Options and environment variables

| Option | `glyd.from_pretrained` | vLLM: `--additional-config '{"glyd": {...}}'`, or the environment | What it does |
| :--- | :--- | :--- | :--- |
| `layout` | `layout="auto"` | `layout`, `GLYD_LAYOUT` | `"auto"` takes `mma` on Ada (L4, L40S, RTX 40), for a mixture of experts on an A10 too, and wherever only it fits, `mma12` on an A10, A100, H100, GH200 and Blackwell (RTX 5090). `"mma"` and `"mma12"` force one. |
| `exact` | `exact=False` | `exact`, `GLYD_EXACT` | The logits are bf16's bit for bit, at a cost in speed. |
| `merge` | `merge=True` |  | q, k, v and gate, up as one product each, as serving engines run them. Not with `exact`. |
| `verify` | `verify=False` | `verify`, `GLYD_VERIFY` | Every pack unpacked and compared with its weights, bit for bit, as it is made. |
| `compile` | `compile=True`; `GLYD_COMPILE=0` |  | `generate()` compiled on PyTorch 2.13 or later. `False` runs it eager. |
| `fraction` |  | `fraction`, `GLYD_FRACTION` | The share of the decoder layers packed: 0 is bf16, 1 every layer. |
| `cc12` |  | `cc12` (no environment variable) | `true`: on compute capability 12 (RTX 50, RTX PRO Blackwell) `mma`'s Linears take, at 9-64 tokens a step, the products measured faster there (below). Off by default until measured end to end; the same bits. |

| Environment variable | What it does |
| :--- | :--- |
| `GLYD_GPU_LIB` | The library to load, in place of the one the package carries. |
| `GLYD_WG_MIN`, `GLYD_WG_MAX` | Raise or lower the token counts where Hopper's route for many tokens a step starts and stops (17 and 1024 by default). |
| `GLYD_MID_MIN` | Raises or lowers the token count where the route for many tokens a step starts on Ampere and Ada (17 by default). |
| `GLYD_DEC_MIN` | Raises or lowers the token count where an `mma12` prompt takes the long-prompt path, on any GPU (0 or unset: the GPU's own). |
| `GLYD_AHEAD_MIN` | Raises or lowers the token count where the long-prompt path of GeForce Ada, an A10 and an L40S starts. |
| `GLYD_AHEAD_WARPS`, `GLYD_AHEAD_RATE`, `GLYD_AHEAD_FLOPS` | Tune that path; the defaults are the measured ones. |
| `GLYD_SPLIT_MIN`, `GLYD_SPLIT_MAX` | Raise or lower the token counts where the extra long-prompt path of an A100 SXM, a GH200 and an H100 SXM starts and stops ([below](#long-prompts-on-an-a100-a-gh200-and-an-h100-sxm)). `GLYD_SPLIT_MIN=-1` turns it off. |
| `GLYD_SPLIT_SMS` | Tunes that path (0 or unset: the GPU's own). |
| `GLYD_MOE_DECODE_MIN` | Raises or lowers the token count a step where a mixture of experts' layer takes its other path (-1: never): [vllm/README.md](vllm/README.md). |

The route variables are read once a process, at the library's first route: set them in the environment
before the first model is loaded. The glyd package refuses a value that is not a whole number at import; the C
library takes it as unset.

## Measured

2026 models only.

### Under vLLM

vLLM 0.30.0, the weights as vLLM's log gives them (`glyd serve`, against `vllm serve` of the bf16 checkpoint):

| | bf16 | Glyd |
| :--- | ---: | ---: |
| Qwen3.5-9B, RTX 4090 24 GB: weights | 17.66 GiB | **12.68 GiB** (`mma`, −28.2%) |
| Qwen3.5-9B, RTX 5090 32 GB: weights | 17.66 GiB | **13.89 GiB** (`mma12`, −21.3%) |
| Qwen3.5-9B, RTX 4090: context | out of memory at start (even at a 4,096-token context) | **189,440 tokens** (`glyd serve`, compiled) |
| Qwen3.8-27B (text only), H100 80GB HBM3: weights | 50.22 GiB | **38.77 GiB** (`mma12`, −22.8%) |
| Qwen3.8-27B, H100: KV cache | 122,538 tokens | **204,117 tokens** (1.67x) |

- **Qwen3.5-9B, one user** (a prompt of about 1,000 tokens, 256 out, the median of three): `glyd run` compiled 77.3
  tokens a second on the RTX 4090 and 115.9 on the RTX 5090, against 26.6 and 28.7 eager; vLLM's bf16 95.0 on the RTX
  5090, the first token after 113 ms against Glyd's 131.
- **Qwen3.8-27B on the H100** (1,024 tokens in and 256 out): saturated, 5.05 requests a second against 4.72 and 4.70
  (1.07x), the first token 0.84x as late, each token 1.43x as long; at 1 request a second 0.92 against 0.92, each token
  21.7 ms against 23.4, the first token 227 ms against 199.
- **Under 33% on these models:** the 12-bit layout is the RTX 5090's and the H100's pick (`best_layout`), and in these
  runs Qwen3.5's embedding, which vLLM builds without a quantization config, stayed bf16 (the plugin packs it since,
  not measured yet; the vision MLP's first Linear too). Norms, biases and the linear-attention layers' small
  parameters stay bf16; vLLM does not load the multi-token-prediction head unless speculative decoding asks for it.
- **The lossless KV cache** holds head size 128 with full attention only: Qwen3.5, Qwen3.8 and Gemma 4 keep vLLM's cache.

Sources: CHANGELOG.md v0.29.0 (Qwen3.5-9B, 2026-10-05) and v0.28.0,
[h100-qwen38-v028-2026-10-03](../benchmarks/gpu/h100-qwen38-v028-2026-10-03) (Qwen3.8-27B).

<a id="two-layouts-the-most-memory-or-the-lightest-decode"></a>
<a id="which-layout-on-which-gpu"></a>
<a id="many-tokens-a-step-on-an-h100-the-copy-engine-and-wgmma"></a>

### Which layout on which GPU

`glyd_gpu.best_layout()` (and `e2e.py --format auto`) takes `mma`, the smallest, on Ada and wherever only it
fits, and `mma12` elsewhere (an A10, an A100, an H100, a GH200, Blackwell), where its decode is the faster one.

<a id="2026-models-on-an-rtx-5090-and-an-h100-sxm"></a>

#### On an RTX 5090 and an H100 SXM

The products of a step through each model's layers (the models' own matrices: a full-attention and a
linear-attention layer of Qwen3.5-9B and Qwen3.5-27B, a sliding-window and a global layer of Gemma-4-12B and
Gemma-4-31B, each counted as often as the model has it; q, k, v and gate, up joined as vLLM joins them), each
layout by its route on that GPU, under CUDA graphs with the weights from memory (`layouts.py`, 2026-10-06,
[tiered-speed-2026-10-06](../benchmarks/gpu/tiered-speed-2026-10-06)). `mma`'s GPU time over `mma12`'s, by
tokens a step, and a prompt's pass of 128, 512 and 2048 tokens:

| GPU | Model | 1 | 8 | 16 | 32 | 64 | 128 | 512 | 2048 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5090, v0.29.0's routes | Qwen3.5-9B | 0.94 | 0.99 | 1.06 | 0.96 | 1.17 | 1.07 | 1.03 | 1.03 |
| | Qwen3.5-27B | 0.94 | 0.96 | 1.00 | 0.94 | 1.24 | 1.07 | 1.03 | 1.04 |
| | Gemma-4-12B | 0.96 | 1.02 | 1.10 | 0.97 | 1.22 | 1.09 | 1.03 | 1.03 |
| | Gemma-4-31B | 0.93 | 0.95 | 0.98 | 0.96 | 1.22 | 1.07 | 1.03 | 1.04 |
| RTX 5090, `cc12` | Qwen3.5-9B | 0.93 | 0.99 | 0.96 | 0.96 | **0.80** | 1.06 | 1.02 | 1.02 |
| | Qwen3.5-27B | 0.92 | 0.95 | 0.95 | 0.93 | **0.81** | 1.06 | 1.02 | 1.01 |
| | Gemma-4-12B | 0.96 | 1.01 | 0.99 | 0.98 | **0.80** | 1.09 | 1.02 | 1.03 |
| | Gemma-4-31B | 0.92 | 0.95 | 0.95 | 0.95 | **0.86** | 1.07 | 1.02 | 1.03 |
| H100 SXM | Qwen3.5-9B | 1.12 | 1.15 | 1.26 | 2.08 | 2.48 | 2.21 | 1.42 | 0.95 |
| | Qwen3.5-27B | 1.05 | 1.17 | 1.26 | 1.98 | 2.33 | 2.42 | 1.42 | 0.94 |
| | Gemma-4-12B | 1.21 | 1.25 | 1.36 | 2.04 | 2.39 | 2.21 | 1.40 | 0.94 |
| | Gemma-4-31B | 1.06 | 1.17 | 1.27 | 1.97 | 2.33 | 2.47 | 1.47 | 0.94 |

- **The RTX 5090** (GDDR7, 1.79 TB/s for 170 SMs: about the bytes a clock an SM of an RTX 4090) streams
  `mma`'s fewer bytes at 1 to 8 tokens: 0.92-1.01x. At 9-16 tokens the step kernel took 0.98-1.10x and at 33-64
  the 4-m-tile product 1.08-1.24x, bound by their instructions there (from the L2 cache 0.71-0.97 of their time
  from memory). With the vLLM option `cc12` (the library's route `GLYD_GPU_ROUTE_GEMM_CC12`; off by default
  until `vllm serve` is measured with it at every load) the tiered layout's 9-32 tokens take the 2-m-tile product and
  33-64 the stream-K kernel with 8 producer warps decoding and 4 consumers multiplying (64 tokens a block):
  0.93-0.99x and 0.80-0.90x. Prompts stay 1.01-1.03x from 512 tokens and 1.06-1.09x at 128: no prompt shape
  measured (8 producers, 8 consumers, blocks of 256 by two row blocks, stream-K) was faster than the prompt
  kernel's own. Qwen3.5-9B's output layer (248,320 x 4,096) takes 1.06x at 1 token and 1.08x at 8, 0.92-0.95x
  at 16-64. A mixture of experts' layer (Qwen3.5-35B-A3B's, 256 experts, top 8) 0.92-1.00x at 1-1,024 tokens
  (1.00x at 1 and 1,024).
- **The H100 SXM** (HBM3, 3.35 TB/s) has `mma` bound by its decode's instructions at every count: 200-245 clk
  a step of 1,024 weights at 1 token against `mma12`'s 170-200 on the large matrices (`mma` as fast from the L2 cache as from memory), and
  `mma12` takes its TMA and wgmma kernel from 17 tokens. The stream-K shapes above took 1.9-2.2x `mma12`'s time
  there at 32-64 tokens. `mma` stays the slower layout on Hopper.

<a id="long-prompts-on-an-a100-a-gh200-and-an-h100-sxm"></a>

### Long prompts on an A100, a GH200 and an H100 SXM

Long `mma12` prompts can take the Python package's extra long-prompt path (`GLYD_SPLIT_*`): on an A100 SXM4 40 GB
prompts of 769 to 4,096 tokens (to 8,192 for 14B and larger models), and on a GH200 and an H100 SXM prompts of
2,048 to 8,192 tokens for 14B and larger models. The
Python package takes it by default where it applies. Other callers of the library (the C API, the Rust crate and
the vLLM plugin) get the default path unless they ask for it with `GLYD_GPU_WITH_SPLIT`. Not measured on a 2026
model yet.

- **Where it applies:** an A100-SXM4-40GB, a GH200 480GB and an H100 80GB HBM3, the SXM5; also an A100 SXM4 80 GB
  and an A800 SXM4, not measured there. An H200, an H100 NVL and the PCIe cards keep the default path until a
  session measures them. Nothing past 8192 tokens takes it. Never on a MIG slice.
- **Where it cannot run,** a prompt takes the default path, never an error: the JIT build (the prebuilt
  library only), a driver before CUDA 12.5 or one that refuses it (a warning says so), too little memory, a
  CUDA graph being captured or a torch.compile graph (a compiled `generate()` runs its prompt eager).
- **Exact mode never takes it.** Its products are the same bits run to run and prompt to prompt within a
  process, but not bit for bit a whole-matrix product.

<a id="popular-models"></a>
<a id="larger-models"></a>

### Every matrix packed and unpacked

`sizes.py MODEL_DIR ...` packs every Linear layer's matrix of a model in both layouts and unpacks it,
compared bit for bit (A10, 2026-09-27, [open-models-a10-2026-09-27](../benchmarks/gpu/open-models-a10-2026-09-27)):

| Model | Matrices | bf16 | `mma` | `mma12` |
| :--- | ---: | ---: | ---: | ---: |
| Muse Glimmer 30B | 25.66 B | 51.33 GB | 34.48 GB (10.75 bits, −32.8%) | 38.61 GB (12.04, −24.8%) |
| Qwen3.8 27B | 24.76 B | 49.52 GB | 33.28 GB (10.75 bits, −32.8%) | 37.25 GB (12.04, −24.8%) |
| Gemma 4 26B-A4B | 24.50 B | 49.00 GB | 32.92 GB (10.75 bits, −32.8%) | 36.87 GB (12.04, −24.8%) |

Qwen3.8 27B end to end on an H100 PCIe (`e2e.py --baseline --ppl --mmlu 300 --format auto --merge`, `mma12`): perplexity
15.1941 against bf16's 15.1946, the next token bf16's 98.82% of the time, MMLU (300) 80.00% against 79.67% (answers as
bf16's: 99.67%); 41,071 MiB of GPU memory against bf16's 51,771 (nvidia-smi), under a 48 GB card's 49,140; GPU time a
token 34.73 / 51.42 / 84.52 ms at 1 / 8 / 32 sequences against 40.21 / 50.60 / 85.19
([lambda-gpu_1x_h100_pcie-20260926-213544](../benchmarks/gpu/lambda-gpu_1x_h100_pcie-20260926-213544)).

## The KV cache

The keys and values a model keeps for the tokens it has seen are bf16 like its weights, and Glyd holds them
compressed too. `kv.py` holds a Hugging Face model's cache that way: `GlydKVCache(config)`. With `fused=True`
and `use_fused_attention(model)`, a step of one new token a sequence reads the compressed cache directly. Other
steps (a prompt) get the keys and values back exactly. Not measured on a 2026 model yet. Under vLLM:
[vllm/README.md](vllm/README.md#the-kv-cache-in-12-bits).

## Where it is slower

Measured on 2026 models, in the tables above:

- **The first token,** one user: Qwen3.5-9B on an RTX 5090 131 ms against vLLM's bf16 113.
- **Each token at full load:** Qwen3.8-27B on an H100 1.43x as long as bf16's (the first token 0.84x as late,
  the requests a second 1.07x), and at 1 request a second the first token 227 ms against 199.
- **A token at 8 sequences:** Qwen3.8-27B on an H100 PCIe, 51.42 ms against bf16's 50.60.
- **With vLLM:** [vllm/README.md](vllm/README.md#measured).
- **Not measured on a 2026 model:** speed on an A10, an L4, an L40S, an A100 and a GH200, the long-prompt path,
  prompts alone, and many sequences without vLLM.

## Reproduce

```
python check_capi.py [LIBRARY]            # every entry point of the library, bit for bit
python check_api.py [MODEL ...]           # glyd.from_pretrained, compress, save_pretrained, verify: against bf16, bit for bit where exact
python check.py model.safetensors         # every tensor packed, unpacked, compared; speeds
python shapes.py MODEL_DIR                # one layer's matrices against bf16, at one token
python gemm.py MODEL_DIR 1,16,64          # several tokens: one layer's matrices against bf16
python e2e.py MODEL_DIR --format mma --fused --baseline [--batch 1,8,32] [--compile] [--prefill 16,64] [--ppl TEXT] [--mmlu 1000] [--kv 1024,4096]
python kv.py                              # the KV cache packed and unpacked bit for bit; attention against SDPA
python sizes.py MODEL_DIR ...             # every Linear's matrix in both layouts, bit for bit: bits a weight, GB
python vllm/check_vllm.py [MODEL ...]     # vllm serve --quantization glyd against vLLM's own bf16 (glyd[vllm])
```

## The library

`build_lib.sh` builds the kernels alone, behind a C API: `libglyd_gpu_cuda12.so` or `libglyd_gpu_cuda13.so`
by nvcc's CUDA major version, with no PyTorch in it. It carries its own CUDA runtime, so it needs only the
driver. A program that calls it links its own runtime beside that one, so take the library whose CUDA major
version is the program's toolkit and runtime: `libglyd_gpu_cuda12.so` for CUDA 12, `libglyd_gpu_cuda13.so`
for CUDA 13. It has code for sm_80, sm_86, sm_89, sm_90a, and sm_100 and sm_120 where nvcc has them (CUDA 12.8
on), and PTX for the GPUs after them.

[glyd_gpu.h](glyd_gpu.h) declares every function and says what it takes: the arrays of a packed matrix (the
`mma` and `mma12` layouts, the `fast` and dense formats), a product's workspace query before its call, the
stream, the return codes. The glyd package calls the library through ctypes (`_lib.py`, whose argument lists
`glyd-gpu-py/tests/test_gpu.py` checks against the header). An engine in C, C++, Rust or any language with a C
FFI calls the same functions, Rust through the [glyd-gpu](../glyd-gpu) crate (the library loaded at run time
and held to its API version, each function typed, a product's workspace query first, errors as `Result`; a
test holds its declarations to the header).

Which kernel a product for M tokens takes on a GPU is the library's choice (`glyd_gpu_mma_route`,
`glyd_gpu_mma12_route`), as measured on an RTX 4080 SUPER, an L4, an L40S, an A10, an A100 SXM4 40 GB, an
H100 (SXM5 and PCIe) and a GH200 (logs: [benchmarks/gpu](../benchmarks/gpu)). A GPU with the same code takes
the same routes, not measured there: an A100 SXM4 80 GB and an A800 SXM4 have an A100 SXM4's; an H200 and an
H100 NVL take an H100 SXM's routes but for the long-prompt path (their code 90, its 7090).
`glyd_gpu_mma_linear` and `glyd_gpu_mma12_linear` run the product by the route for the tokens given. Where K
is not a multiple of 64, past 64 tokens (`mma12`: also from `GLYD_DEC_MIN` where that is lower), they return
`cudaErrorNotSupported`, nothing launched: unpack the matrix (`glyd_gpu_mma_unpack`) for a GEMM of your own.
The long-prompt path takes your cuBLAS handle through `glyd_gpu_mma12_ring_linear`.

A GPU's code, which the routes take, is its compute capability plus a class where the name tells GPUs apart
(`GLYD_GPU_GEFORCE`, `GLYD_GPU_A10`, `GLYD_GPU_L4`, `GLYD_GPU_L40S`, `GLYD_GPU_PCIE`, `GLYD_GPU_GH200`,
`GLYD_GPU_H100`: `glyd_gpu.h`). `GLYD_GPU_WITH_SPLIT` added to it asks for the long-prompt path, which only
glyd_gpu's Linears do by default (without it the routes and `linear` are v0.25.1's). The route variables
([above](#options-and-environment-variables)) are read once a process, at the library's first route.

Every release carries the library on its own for Linux x86_64 and aarch64 (glibc 2.28 or later), CUDA 12
(built with 12.8) and 13: `glyd-gpu-TAG-linux-ARCH-cudaN.tar.gz`, holding the library, glyd_gpu.h,
examples/unpack.c, this directory's LICENSE and a README ([README-lib.md](README-lib.md), with the example's
build line for that download's library), each with its `.sha256`.

[examples/unpack.c](examples/unpack.c), C and the C API alone: a matrix of a model saved by
`glyd.save_pretrained` (glyd-v1) unpacked on the GPU by `glyd_gpu_mma_unpack` and checked against the bf16
checkpoint it was packed from, bit for bit; a merged pack tensor by tensor (the runtime's lib64, or lib in a
toolkit from pip, as setup_env.sh's):

    bash build_lib.sh .
    gcc -O2 -I . -I $CUDA_HOME/include examples/unpack.c -o unpack \
        -L . -lglyd_gpu_cuda13 -L $CUDA_HOME/lib64 -lcudart -Wl,-rpath,$PWD:$CUDA_HOME/lib64:$CUDA_HOME/lib
    python -m glyd_gpu pack Qwen/Qwen3.5-0.8B qwen3.5-0.8b-glyd

It prints each pack's name, shape and bits a weight, and for each tensor in it whether the GPU decoded the
checkpoint's bits (a bit flipped in the checkpoint is found); a pack's name after the checkpoint decodes that
pack alone:

    ./unpack qwen3.5-0.8b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/ [PACK]

The glyd-gpu crate's examples do the same from Rust: `examples/unpack.rs` (this one), and
`examples/linear.rs`, a pack multiplied by `linear` for 1-2000 tokens, bit for bit the kernel of the route
this GPU takes:

    cargo run --release -p glyd-gpu --example unpack -- qwen3.5-0.8b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/
    cargo run --release -p glyd-gpu --example linear -- qwen3.5-0.8b-glyd PACK

(These three were run on Qwen3 saves; not yet on a 2026 model's.)

And the saved model itself comes from Rust too, on the CPU, byte for byte as `python -m glyd_gpu pack` saves
it (the architectures `glyd-gpu pack` lists: no 2026 model's yet, which `python -m glyd_gpu pack` saves on a GPU; the
glyd-gpu command):

    glyd-gpu pack MODEL OUT
    glyd-gpu verify OUT [--device cuda:0]

## License

The files under gpu/ (and the glyd-gpu crate) are under the [Apache License 2.0](LICENSE), as all of Glyd.
