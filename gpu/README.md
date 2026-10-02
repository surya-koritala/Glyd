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

`fast` and `huffman` are earlier layouts, kept in `e2e.py` for comparison (Qwen2.5-7B, one sequence: `fast`
11.05 GB at 55.1 tokens/s, `huffman` 10.60 GB at 52.0, bf16 15.25 GB at 43.3). No lossless coding takes more
than about 34% off bf16 weights: the floor measured is about 10.6 bits a weight.

## Requirements

- An NVIDIA GPU of the Ampere generation or newer: RTX 30 and 40 series, A10, A100, L4, L40S, H100, GH200
  and later. The library carries native code for Ampere, Ada, Hopper and Blackwell (sm_100, sm_120);
  Blackwell's is built and not yet run on a GPU.
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
builds it (it needs nvcc) next to `glyd_gpu.py`; without it the scripts here build the extension on first
import. `GLYD_GPU_LIB` names another library.

## Use

```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3-8B")               # packed on the GPU as it loads
model = glyd.from_pretrained("Qwen/Qwen3-8B", exact=True)   # logits bit for bit bf16's
```

```bash
glyd pack Qwen/Qwen3-8B qwen3-8b-glyd     # a model saved packed, on the CPU: no Python or GPU needed
glyd verify qwen3-8b-glyd                 # every pack and tensor checked
glyd run Qwen/Qwen3-8B                    # downloads it, starts it packed with vLLM, and chats
```

- Serving with vLLM, `vllm serve MODEL --quantization glyd`: [vllm/README.md](vllm/README.md).
- Every option of the Python API: [bindings/python/README.md](../bindings/python/README.md#on-the-gpu-a-models-weights-held-compressed-bit-for-bit).
- C, C++ and Rust: [the library](#the-library).

## Options and environment variables

| Option | `glyd.from_pretrained` | vLLM: `--additional-config '{"glyd": {...}}'`, or the environment | What it does |
| :--- | :--- | :--- | :--- |
| `layout` | `layout="auto"` | `layout`, `GLYD_LAYOUT` | `"auto"` takes `mma` on Ada (L4, L40S, RTX 40), for a mixture of experts on an A10 too, and wherever only it fits, `mma12` on an A10, A100, H100 and GH200. `"mma"` and `"mma12"` force one. |
| `exact` | `exact=False` | `exact`, `GLYD_EXACT` | The logits are bf16's bit for bit, at a cost in speed. |
| `merge` | `merge=True` |  | q, k, v and gate, up as one product each, as serving engines run them. Not with `exact`. |
| `verify` | `verify=False` | `verify`, `GLYD_VERIFY` | Every pack unpacked and compared with its weights, bit for bit, as it is made. |
| `compile` | `compile=True`; `GLYD_COMPILE=0` |  | `generate()` compiled on PyTorch 2.13 or later. `False` runs it eager. |
| `fraction` |  | `fraction`, `GLYD_FRACTION` | The share of the decoder layers packed: 0 is bf16, 1 every layer. |

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

Logs are in [benchmarks/gpu](../benchmarks/gpu). bf16 and Glyd ran in the same run, on the same machine,
unless a line says otherwise.

### Qwen2.5-7B-Instruct on an RTX 4080 SUPER

RTX 4080 SUPER (16 GB), PyTorch 2.14, CUDA 13.0; Qwen2.5-7B-Instruct, greedy
(`e2e.py MODEL --format mma --fused --baseline`): generating for 1 to 64 sequences at once, a prompt's
forward pass, and perplexity on Wikipedia text (enwik8 from its 10th MB, 200 windows of 64 tokens):

|                               |          bf16 |         Glyd `mma` |
| :---------------------------- | ------------: | -----------------: |
| Peak VRAM                     |      15.25 GB |       **10.61 GB** |
| 1 sequence                    | 43.4 tokens/s |   **55.7** (1.28x) |
| 4 sequences                   |         167.9 |  **217.5** (1.30x) |
| 16 sequences                  |         652.0 |  **814.2** (1.25x) |
| 32 sequences                  |        1153.7 | **1518.9** (1.32x) |
| 48 sequences                  |        1664.8 | **1882.5** (1.13x) |
| 64 sequences                  |        2160.0 | **2244.7** (1.04x) |
| Prompt of 16 tokens           |         24 ms |          **19 ms** |
| Prompt of 64 tokens           |         27 ms |          **26 ms** |
| Prompt of 128 tokens          |         29 ms |              29 ms |
| Prompt of 256 tokens          |         43 ms |              45 ms |
| Prompt of 512 tokens          |         79 ms |              85 ms |
| Prompt of 1024 tokens         |        154 ms |             163 ms |
| Prompt of 2048 tokens         |        301 ms |             330 ms |
| Prompt of 4096 tokens         |        645 ms |             702 ms |
| Perplexity, 64-token windows  |       17.0015 |            17.0052 |
| Perplexity, 512-token windows |        7.5677 |             7.5660 |

The weights are the model's to the bit. The products sum in another order than bf16's, which moves the
logits by a rounding: the next token chosen is bf16's 98.13% of the time (98.49% on the 512-token windows).
bf16 against itself, two windows a pass instead of one: perplexity 17.0153, the same next token 98.33% of
the time.

On 7B's matrices a product is 1.30-1.34x faster than bf16's at one token on the MLP's, and 1.06-1.19x at 64.
Past 128 tokens a prompt's products take at best bf16's time: they are within 5-10% of it.

<a id="two-layouts-the-most-memory-or-the-lightest-decode"></a>

## Two layouts: the most memory off, or the faster one

`mma` takes the most off, and where memory is the limit, as on an RTX 4080 SUPER at a few tokens a step,
that is the faster one too. Where the GPU's memory is quick (an H100's HBM3), or at many tokens a step,
`mma12` is. Qwen2.5-7B-Instruct, RTX 4080 SUPER, the same harness:

|  | bf16 | `mma` | `mma12` |
| :--- | ---: | ---: | ---: |
| Weights | 15.23 GB | **10.32 GB** | 11.42 GB |
| Tokens/s at 1 / 8 / 32 / 64 sequences | 43.4 / 332.4 / 1153.7 / 2160.0 | **55.7 / 424.2 / 1518.9** / 2244.7 | 51.9 / 398.6 / 1452.4 / **2455.9** |
| Prompt of 64 / 2048 tokens | 27 / 301 ms | 26 / 330 ms | **23** / 318 ms |
| Perplexity; MMLU (1,000) | 17.0015; 73.50% | 17.0052; 73.50% | 17.0052; 73.50% |

Per matrix (down_proj, 3584 x 18944): one token 141 us (`mma`), 152 (`mma12`), 198 (bf16); 64 tokens 193,
170, 236.

On an H100 SXM (HBM3, 3.35 TB/s) the order turns. Qwen3-32B's layer 0, one token: down_proj 75-79 us
(`mma12`), 126 (`mma`), 90 (bf16); gate and up 75-76, 130, 86-88; 16 tokens 80-82, 132-137, 89-91 (at 64
tokens bf16 leads: 95 against 141-143). End to end, GPU time a generated token (Hugging Face's loop, bound by
the CPU at about 68 ms a token for all three, over 16 tokens):

| H100 SXM                     |     bf16 |        `mma` |      `mma12` |
| :--------------------------- | -------: | -----------: | -----------: |
| Qwen3-32B: weights           | 65.52 GB | **44.45 GB** |     49.23 GB |
| Qwen3-32B: GPU time a token  | 28.22 ms |     40.58 ms | **26.39 ms** |
| Qwen3-32B: MMLU (1,000)      |    78.3% |        78.2% |        78.1% |
| Qwen2.5-7B: weights          | 15.23 GB | **10.32 GB** |     11.42 GB |
| Qwen2.5-7B: GPU time a token |  7.47 ms |     10.73 ms |  **7.35 ms** |
| Qwen2.5-7B: MMLU (1,000)     |    73.2% |        73.2% |        73.3% |

Logs: [lambda-gpu_1x_h100_sxm5-20260926-114529](../benchmarks/gpu/lambda-gpu_1x_h100_sxm5-20260926-114529)
for bf16 and `mma`, and [-120552](../benchmarks/gpu/lambda-gpu_1x_h100_sxm5-20260926-120552) for `mma12`.

<a id="which-layout-on-which-gpu"></a>

### Which layout on which GPU

Measured the way a server runs a model: q, k and v as one product and gate and up as another, for bf16 and
Glyd alike (`e2e.py --merge`, as vLLM runs them), and the GPU time of a generated token apart from the
prompt's (`--profile`: 17 steps less 1, over 16). Qwen2.5-7B-Instruct, GPU time a token at 1 / 8 / 32 / 64
sequences (logs: benchmarks/gpu/lambda-gpu_1x_*-2026092617*, -18*, and
[rtx4080s-layouts-2026-09-26](../benchmarks/gpu/rtx4080s-layouts-2026-09-26)):

| GPU | bf16 | `mma` | `mma12` |
| :--- | ---: | ---: | ---: |
| RTX 4080 SUPER 16 GB | 21.97 / 22.85 / 26.19 / 27.67 ms | **16.52 / 17.37 / 18.80** / 25.14 | 17.48 / 18.25 / 19.67 / **21.66** |
| A10 24 GB | 33.71 / 34.77 / 35.50 / 37.98 ms | 24.33 / 25.61 / 35.99 / 49.66 | **24.40 / 25.76 / 29.03 / 33.05** |
| A100 40 GB | 14.18 / 14.87 / 16.18 / 17.80 ms | 15.34 / 16.00 / 23.99 / 28.81 | **11.84 / 13.77** / 17.16 / 19.95 |
| H100 SXM 80 GB | 6.90 / 7.50 / 8.02 / 8.55 ms |  | **6.55 / 7.24** / 8.69 / 9.99 |
| H100, Qwen3-32B | 27.90 / 29.78 / 31.44 / 33.52 ms |  | **24.50 / 26.75** / 33.46 / 37.35 |

A blank cell is not measured.

So: on Ada, `mma`, the smallest, is also the fastest to 32 sequences (24-28% under bf16's time). On an A10,
`mma12` is 13-28% under at every count. On an A100 and an H100, `mma12` is 3-16% under to 8 sequences and
6-17% over at 32 and 64. Perplexity is bf16's everywhere (17.00 against 17.01 on the A10, for one).
`glyd_gpu.best_layout()` (and `e2e.py --format auto`) takes `mma` on Ada and wherever only it fits, `mma12`
elsewhere.

### Many sequences at once

Generating for 1 to 128 sequences at once: GPU time a step.

#### A100 SXM4 40 GB

GPU time a step (`e2e.py --format auto --fused --merge --profile 16`; Qwen3-14B's bf16 from the log's
`bf16prof.py`, as bf16's and Glyd's copies do not fit 40 GB at once):

| A100, GPU time a step |         1 |         8 |        32 |        64 | 128 sequences |
| :-------------------- | --------: | --------: | --------: | --------: | ------------: |
| Qwen3-8B, bf16        |     18.88 |     19.59 |     21.12 |     21.13 |      25.71 ms |
| Qwen3-8B, `mma12`     | **15.27** | **18.81** | **18.54** | **21.05** |         28.17 |
| Qwen3-14B, bf16       |     26.76 |     28.92 |     32.89 |     36.01 |      41.15 ms |
| Qwen3-14B, `mma12`    | **23.11** | **25.05** | **28.70** | **31.81** |         42.41 |

At 128 sequences Qwen3-8B generates 2993.6 tokens/s against bf16's 2997.4, Qwen3-14B 2594.0 against 2657.4.
Logs: [a100-ampere-2026-09-28](../benchmarks/gpu/a100-ampere-2026-09-28).

<a id="many-tokens-a-step-on-an-h100-the-copy-engine-and-wgmma"></a>

#### H100 PCIe

GPU time a step (`e2e.py --format auto --fused --merge --profile`, bf16's from the same
GPU; Qwen3-32B's bf16 row from the log's `bf16prof.py`, since `e2e.py`'s own bf16 profile runs out of memory at 32B):

| H100 PCIe, GPU time a step |         1 |         8 |        32 | 64 sequences |
| :------------------------- | --------: | --------: | --------: | -----------: |
| Qwen3-8B, bf16             |     12.30 |     13.34 |     14.57 |     15.66 ms |
| Qwen3-8B, `mma12`          | **11.26** | **12.46** | **13.55** | **14.86 ms** |
| Qwen3-32B, bf16            |     42.44 |     44.40 |     47.10 |     49.56 ms |
| Qwen3-32B, `mma12`         | **34.76** | **37.53** | **40.65** | **43.90 ms** |

A layer's products against bf16's (weights read from memory, as a model's step reads them), 32 / 64 / 96 /
128 tokens: Qwen3-8B 0.84x / 0.86x / 0.95x / 1.01x, Qwen3-32B 0.77x / 0.84x / 0.91x / 1.04x; the output
layer (151936 x 4096) 0.78x at 32 and 0.93x at 64. Logs: [h100-hopper-2026-09-28](../benchmarks/gpu/h100-hopper-2026-09-28).

#### RTX 4080 SUPER

128 sequences at once (`--batch 128 --tokens 64 --profile 16`, three runs in fresh processes):

| 128 sequences                   | tokens/s | GPU time a step, ms |
| :------------------------------ | -------: | ------------------: |
| Qwen3-1.7B, bf16                |     8452 |               12.18 |
| Qwen3-1.7B, `mma`               |     8384 |               12.40 |
| Qwen3-1.7B, `mma12`             |     8845 |               11.60 |
| Qwen3-4B-Instruct-2507, bf16    |     4702 |               23.19 |
| Qwen3-4B-Instruct-2507, `mma`   |     4971 |               21.73 |
| Qwen3-4B-Instruct-2507, `mma12` |     5069 |               21.32 |

Logs: [rtx4080s-prefill-2026-09-28](../benchmarks/gpu/rtx4080s-prefill-2026-09-28).

## Prompts

One forward pass over a prompt, ms (`e2e.py --prefill`).

### RTX 4080 SUPER

Qwen3-1.7B and Qwen3-4B-Instruct-2507, `--prefill --merge` (bf16 and Glyd merged alike; bf16 in the same
runs), each the mean of two runs in fresh processes:

| Prompt, tokens                  |  128 |  256 |  512 | 1024 |  2048 |  4096 |
| :------------------------------ | ---: | ---: | ---: | ---: | ----: | ----: |
| Qwen3-1.7B, bf16                | 10.8 | 14.1 | 24.4 | 42.3 |  86.5 | 184.6 |
| Qwen3-1.7B, `mma`               | 10.6 | 14.4 | 23.4 | 43.6 |  87.4 | 186.4 |
| Qwen3-1.7B, `mma12`             | 10.1 | 14.2 | 23.2 | 42.5 |  88.2 | 185.3 |
| Qwen3-4B-Instruct-2507, bf16    | 20.7 | 28.1 | 49.1 | 96.3 | 199.0 | 446.1 |
| Qwen3-4B-Instruct-2507, `mma`   | 19.4 | 28.9 | 52.3 | 98.3 | 199.9 | 449.6 |
| Qwen3-4B-Instruct-2507, `mma12` | 19.0 | 28.3 | 50.9 | 99.2 | 201.0 | 449.9 |

Logs: [rtx4080s-prefill-2026-09-28](../benchmarks/gpu/rtx4080s-prefill-2026-09-28).

### A100 SXM4 40 GB

One forward pass (`e2e.py --prefill`), ms (the time to first token through `generate()` is 3-5 ms more,
alike):

| Prompt             |      128 |  256 |   512 |  1024 |  2048 | 4096 tokens |
| :----------------- | -------: | ---: | ----: | ----: | ----: | ----------: |
| Qwen3-8B, bf16     |     41.0 | 41.1 |  48.4 |  90.1 | 174.3 |       347.9 |
| Qwen3-8B, `mma12`  | **40.9** | 45.5 |  64.5 | 112.0 | 195.7 |       368.2 |
| Qwen3-14B, bf16    |     45.1 | 48.9 |  82.7 | 151.2 | 291.1 |       584.6 |
| Qwen3-14B, `mma12` |     45.3 | 62.9 | 111.7 | 198.6 | 347.5 |       653.7 |

Logs: [a100-ampere-2026-09-28](../benchmarks/gpu/a100-ampere-2026-09-28). Longer prompts are faster with the
long-prompt path ([below](#long-prompts-on-an-a100-a-gh200-and-an-h100-sxm)).

### H100 SXM

One forward pass (`e2e.py --format auto --fused --merge --prefill`), ms:

| H100 SXM           |  128 |  512 |      1024 |  2048 | 4096 tokens |
| :----------------- | ---: | ---: | --------: | ----: | ----------: |
| Qwen3-8B, bf16     | 27.3 | 27.8 |      36.4 |  71.3 |       142.6 |
| Qwen3-8B, `mma12`  | 30.0 | 29.5 |  **45.9** |  82.2 |       155.1 |
| Qwen3-32B, bf16    | 51.1 | 73.7 |     135.5 | 276.9 |       544.0 |
| Qwen3-32B, `mma12` | 53.6 | 90.9 | **166.3** | 332.5 |       622.6 |

Validated in a third run, Qwen3-8B's layer takes 0.94 / 1.16 / 1.16 / 1.24 / 1.43 / 1.38x bf16's time at 17 /
128 / 129 / 256 / 512 / 1024 tokens (the CUDA 12 library's within 0.8%), and one forward pass over 1024 tokens
45.0 ms against bf16's 36.9 (over 256 tokens, 30.9 against 28.3). `e2e.py --exact` gave bf16's logits bit for
bit, 8 of 8 tokens. Prompts past 1024 tokens take the long-prompt path. Logs:
[h100-hopper2-2026-09-28](../benchmarks/gpu/h100-hopper2-2026-09-28),
[h100-hopper2-cu12-2026-09-28](../benchmarks/gpu/h100-hopper2-cu12-2026-09-28) and
[h100-hopper2-val-2026-09-28](../benchmarks/gpu/h100-hopper2-val-2026-09-28).

### A10, L4 and L40S

Qwen3-8B, one forward pass, over bf16's time in the same run (`e2e.py --prefill --merge`); a + is a
longer time.

An A10 (Lambda Cloud, 150 W), `mma12`. Each row is one run, over the bf16 of that run, with its own log
(the runs are in [lambda-a10-routes-2026-09-28](../benchmarks/gpu/lambda-a10-routes-2026-09-28)):

| Prompts of                 | Over bf16's time       | Log                                                                 |
| :------------------------- | :--------------------- | :------------------------------------------------------------------ |
| 128 and 512 tokens         | +2.2% / +15.8%         | [log](../benchmarks/gpu/lambda-a10-routes-2026-09-28/e2e-fused.txt) |
| 1024, 2048 and 4096 tokens | +10.0% / +5.2% / +2.6% | [log](../benchmarks/gpu/lambda-a10-routes-2026-09-28/e2e-ahead.txt) |

The A10G, the same chip at 300 W, stays at most +5.3% over bf16's to 4096 tokens
([log](../benchmarks/gpu/sweep-2026-09-28/a10g-aws-g5)).

An L4 (AWS g6.4xlarge, 72 W):

| Prompt  |   128 |    512 |   1024 |   2048 |  4096 |  8192 |
| :------ | ----: | -----: | -----: | -----: | ----: | ----: |
| `mma`   | +5.6% | +17.6% | +25.5% | +11.3% | +8.4% | +4.7% |
| `mma12` | -9.0% |  -0.8% |  +7.2% | +11.0% | +9.3% | +4.4% |

Qwen3-4B-Instruct-2507 in the same run, `mma` at 1024 / 2048 / 4096 / 8192 tokens: +12.9 / +14.0 / +10.1 /
+4.3%; `mma12` at 4096 / 8192 +6.8 / +5.0%. `mma` is the L4's default (33% less memory). `mma12`'s prompts
took 5-20% less time than `mma`'s to 1536 tokens on the L4, and about the same from 1792 (2.4% more to 3.6%
less; Qwen3-8B and Qwen3-4B-Instruct-2507): for the fastest short prompts, at 25% less memory, load with
`layout="mma12"`. The L4's clock falls as it heats at its power cap: the same prompt pass (Qwen3-8B `mma`,
2048 tokens) took 739 ms at 66 C and 1148 MHz and 794 ms at 82 C and 1035 MHz, so compare its runs at like
temperatures. Logs: [l4-routes-2026-09-29](../benchmarks/gpu/l4-routes-2026-09-29).

An L40S (AWS g6e.xlarge, 350 W):

| Prompt  |    512 |    768 |   1024 |   1536 |   2048 |  3072 |   4096 |  8192 |
| :------ | -----: | -----: | -----: | -----: | -----: | ----: | -----: | ----: |
| `mma`   | +20.6% | +27.9% | +30.3% | +13.9% | +11.9% | +8.0% | +10.9% | +3.8% |
| `mma12` |  -0.3% |  +4.8% | +13.6% |  +9.4% | +12.9% | +7.7% | +11.3% | +3.9% |

The L40S's `mma12` prompts took 17.3 / 18.1 / 12.8 / 4.0% less time than `mma`'s at 512 / 768 / 1024 / 1536
tokens and were within 0.9% of them from 2048; `mma` is its default too (33% less memory; `layout="mma12"`:
25%). The L40 and RTX 6000 Ada share its compute capability and are not measured: they take the default path.

<a id="long-prompts-on-an-a100-a-gh200-and-an-h100-sxm"></a>

### Long prompts on an A100, a GH200 and an H100 SXM

Long `mma12` prompts take less time on these GPUs with the Python package's extra long-prompt path
(`GLYD_SPLIT_*`) than without it (a forward pass at least 2% less):

- an A100 SXM4 40 GB: prompts of 769 to 4096 tokens, and up to 8192 tokens for 14B and larger models
  (Qwen3-14B's pass takes 0.968 of the time at 8192; Qwen3-8B's 0.994 there, so it is not taken);
- a GH200: 14B and larger models, prompts of 2048 to 8192 tokens (Qwen3-32B's pass takes 0.909 / 0.940 / 0.952
  of the time at 2048 / 4096 / 8192; Qwen3-8B's 0.978 / 0.994 / 0.994, 2.2% at 2048 alone, so it is not taken);
- an H100 SXM: 14B and larger models, prompts of 2048 to 8192 tokens (Qwen3-14B's pass takes 0.893 / 0.937 /
  0.938 of the time at 2048 / 4096 / 8192, run with `GLYD_SPLIT_MIN=2048`, which is also the package's own
  choice for a 14B).

The Python package takes it by default where it applies. Other callers of the library (the C API, the Rust
crate and the vLLM plugin) get the default path unless they ask for it with `GLYD_GPU_WITH_SPLIT`. One
forward pass (`e2e.py --prefill --merge`), ms, the medians of 3 rounds each way in turn:

| Prompt                         |   769 |  1024 |  2048 |  4096 |   8192 |
| :----------------------------- | ----: | ----: | ----: | ----: | -----: |
| A100-SXM4-40GB, Qwen3-8B, bf16 |  80.9 |  93.0 | 181.9 | 363.0 |  766.3 |
| Glyd, `GLYD_SPLIT_MIN=-1`      | 103.3 | 115.1 | 204.4 | 386.7 |  792.5 |
| Glyd                           |  92.9 | 101.2 | 196.6 | 375.8 |  787.1 |
| Qwen3-14B, bf16                | 128.9 | 156.7 | 303.5 | 611.6 | 1282.6 |
| Glyd, `GLYD_SPLIT_MIN=-1`      | 179.5 | 205.9 | 362.1 | 680.9 | 1373.9 |
| Glyd                           | 151.7 | 178.0 | 332.0 | 644.5 | 1329.6 |
| GH200, Qwen3-8B, bf16          |       |       |  72.1 | 146.7 |  306.7 |
| Glyd, `GLYD_SPLIT_MIN=-1`      |       |       |  81.5 | 155.4 |  314.8 |
| Glyd                           |       |       |  79.7 | 154.0 |  312.3 |
| Qwen3-32B, bf16                |       |       | 280.1 | 564.5 | 1190.1 |
| Glyd, `GLYD_SPLIT_MIN=-1`      |       |       | 335.7 | 636.2 | 1279.9 |
| Glyd                           |       |       | 305.1 | 598.7 | 1218.4 |
| H100 SXM, Qwen3-14B, bf16      |       |       | 119.2 | 245.7 |  504.5 |
| Glyd, `GLYD_SPLIT_MIN=-1`      |       |       | 150.5 | 279.0 |  551.4 |
| Glyd                           |       |       | 134.4 | 260.4 |  518.2 |

The A100's Qwen3-8B at 8192 tokens and the GH200's Qwen3-8B are not taken by default: they were run with
`GLYD_SPLIT_MIN` set, as a measurement. The ratios above are each round's time over the `GLYD_SPLIT_MIN=-1`
time, the median of the 3 (Qwen3-8B at 1024 tokens on the A100: rounds 0.867, 0.879, 0.876, median 0.876).

- **Where measured:** an A100-SXM4-40GB, a GH200 480GB and an H100 80GB HBM3, the SXM5. The package takes
  it on an A100 SXM4 80 GB and an A800 SXM4 too, not measured there. An H200, an H100 NVL and the PCIe cards
  (an A100 PCIe, an H100 PCIe) keep the default path until a session measures them. Nothing past 8192 tokens
  takes it, not measured. Never on a MIG slice.
- **Where it cannot run,** a prompt takes the default path, never an error: the JIT build (the prebuilt
  library only), a driver before CUDA 12.5 or one that refuses it (a warning says so), too little memory, a
  CUDA graph being captured or a torch.compile graph (a compiled `generate()` runs its prompt eager).
- **Exact mode never takes it.** Its products are the same bits run to run and prompt to prompt within a
  process, but not bit for bit a whole-matrix product.
- **A stress check** (`split_stress.py`) ran 16,512 products on an L4, an A100, a GH200 and an H100 SXM: the
  same bits every time, and within 1e-2 of fp32 (logs: [option2-2026-09-29](../benchmarks/gpu/option2-2026-09-29)).

<a id="popular-models"></a>

## Popular models

`sizes.py MODEL_DIR ...` packs every Linear layer's matrix of a model in both layouts and unpacks it,
compared bit for bit. Nineteen popular open models (H100 SXM, 2026-09-26,
[benchmarks/gpu/popular-h100-2026-09-26](../benchmarks/gpu/popular-h100-2026-09-26); A10, 2026-09-27,
[benchmarks/gpu/open-models-a10-2026-09-27](../benchmarks/gpu/open-models-a10-2026-09-27)):

| Model                        | Matrices |      bf16 |                          `mma` |                   `mma12` |
| :--------------------------- | -------: | --------: | -----------------------------: | ------------------------: |
| GLM-4.5-Air                  | 107.96 B | 215.92 GB | 144.58 GB (10.71 bits, −33.0%) | 162.43 GB (12.04, −24.8%) |
| Llama 4 Scout 17B-16E        | 105.97 B | 211.93 GB |      142.18 GB (10.73, −32.9%) | 159.42 GB (12.04, −24.8%) |
| Qwen3-Next 80B-A3B           |  80.64 B | 161.28 GB |      109.40 GB (10.85, −32.2%) | 123.34 GB (12.24, −23.5%) |
| Llama 3.3 70B Instruct       |  68.45 B | 136.90 GB |  91.93 GB (10.74 bits, −32.9%) | 103.00 GB (12.04, −24.8%) |
| Qwen3 30B-A3B (MoE)          |  29.90 B |  59.79 GB |       40.22 GB (10.76, −32.7%) |  44.98 GB (12.04, −24.8%) |
| Gemma 3 27B                  |  25.74 B |  51.48 GB |       34.58 GB (10.75, −32.8%) |  38.74 GB (12.04, −24.8%) |
| Muse Glimmer 30B             |  25.66 B |  51.33 GB |       34.48 GB (10.75, −32.8%) |  38.61 GB (12.04, −24.8%) |
| Qwen3.8 27B                  |  24.76 B |  49.52 GB |       33.28 GB (10.75, −32.8%) |  37.25 GB (12.04, −24.8%) |
| Gemma 4 26B-A4B              |  24.50 B |  49.00 GB |       32.92 GB (10.75, −32.8%) |  36.87 GB (12.04, −24.8%) |
| Mistral Small 3.2 24B        |  22.63 B |  45.26 GB |       30.34 GB (10.72, −33.0%) |  34.05 GB (12.04, −24.8%) |
| Phi-4                        |  13.63 B |  27.26 GB |       18.28 GB (10.73, −32.9%) |  20.51 GB (12.04, −24.8%) |
| DeepSeek-R1-Distill-Qwen 14B |  13.21 B |  26.42 GB |       17.99 GB (10.89, −31.9%) |  19.89 GB (12.05, −24.7%) |
| Gemma 3 12B                  |  10.90 B |  21.80 GB |       14.63 GB (10.74, −32.9%) |  16.41 GB (12.04, −24.8%) |
| Llama 3.1 8B Instruct        |   6.98 B |  13.96 GB |        9.39 GB (10.76, −32.8%) |  10.50 GB (12.04, −24.8%) |
| Mistral 7B Instruct v0.3     |   6.98 B |  13.96 GB |        9.40 GB (10.77, −32.7%) |  10.50 GB (12.04, −24.7%) |
| Qwen3 8B                     |   6.95 B |  13.89 GB |        9.44 GB (10.87, −32.1%) |  10.46 GB (12.05, −24.7%) |
| Qwen3 4B 2507                |   3.63 B |   7.27 GB |        4.93 GB (10.85, −32.2%) |   5.47 GB (12.04, −24.8%) |
| Llama 3.2 3B Instruct        |   2.82 B |   5.64 GB |        3.79 GB (10.75, −32.8%) |   4.24 GB (12.04, −24.8%) |
| SmolLM3 3B                   |   2.81 B |   5.62 GB |        3.77 GB (10.73, −32.9%) |   4.23 GB (12.04, −24.8%) |

Llama and Gemma from `unsloth/` (the same weights, ungated). bf16 against Glyd (`mma`) end to end, the same
run (`e2e.py --baseline --ppl --mmlu 300`), where it loads the model as a causal LM on one GPU:

| Model                        | Perplexity bf16 / Glyd | Next token as bf16's | MMLU (300) bf16 / Glyd | Answers as bf16's |
| :--------------------------- | ---------------------: | -------------------: | ---------------------: | ----------------: |
| Phi-4                        |      14.7888 / 14.7855 |               98.91% |        76.67% / 76.33% |            99.33% |
| DeepSeek-R1-Distill-Qwen 14B |      27.1955 / 27.1975 |               98.52% |        78.00% / 78.00% |              100% |
| Llama 3.1 8B Instruct        |      19.5915 / 19.5909 |               98.68% |        71.67% / 72.00% |            99.67% |
| Mistral 7B Instruct v0.3     |      12.1422 / 12.1473 |               99.05% |        60.67% / 60.67% |              100% |
| Qwen3 8B                     |      20.7490 / 20.7428 |               98.47% |        74.00% / 74.33% |            99.67% |
| SmolLM3 3B                   |      29.1467 / 29.1422 |               97.90% |        63.33% / 63.33% |            99.67% |
| Qwen3.8 27B (H100 PCIe)      |      15.1946 / 15.1941 |               98.82% |        79.67% / 80.00% |            99.67% |
| Gemma 3 12B (H100 PCIe)      |                        |               95.56% |        74.00% / 74.00% |            99.33% |
| Qwen3 4B 2507 (A10)          |      22.4636 / 22.4672 |               98.53% |        71.00% / 71.00% |            99.33% |
| Llama 3.2 3B Instruct (A10)  |      25.1298 / 25.1437 |               98.58% |        63.67% / 64.33% |            98.00% |

The last four with the `mma12` layout the GPU picks (`--format auto`) and q, k, v and gate, up merged
(`--merge`). Qwen3.8 27B with Glyd uses 41,071 MiB of GPU memory against bf16's 51,771 (nvidia-smi), under a
48 GB card's 49,140; GPU time a token 34.73 / 51.42 / 84.52 ms at 1 / 8 / 32 sequences against 40.21 / 50.60 /
85.19. Gemma 3's perplexity is left out: the windows start without the BOS token Gemma needs.

<a id="larger-models"></a>

## Larger models

On rented GPUs (`scripts/gpu_lambda.sh`: one Lambda Cloud instance a run, terminated at the end; raw logs in
`benchmarks/gpu/lambda-*`), bf16 and Glyd in the same run, 64 new tokens a sequence, MMLU on the same 1,000
questions (0-shot):

| Model, GPUs                     |      Weights | Tokens/s at 1 / 8 / 32 / 64 sequences | Prompt of 2048 | Perplexity |  MMLU |
| :------------------------------ | -----------: | ------------------------------------: | -------------: | ---------: | ----: |
| Qwen3-32B, bf16, 2x A6000 48 GB |     65.52 GB |            9.5 / 74.4 / 275.9 / 488.4 |        1332 ms |    17.0796 | 78.5% |
| Qwen3-32B, Glyd, **1x** A6000   | **44.45 GB** |       **11.8 / 94.9 / 289.9** / 400.5 |        2082 ms |    17.0788 | 78.0% |
| Qwen2.5-72B, bf16, 4x A6000     |    145.41 GB |            4.5 / 35.0 / 134.6 / 254.3 |        2689 ms |    10.6035 | 81.9% |
| Qwen2.5-72B, Glyd, **3x** A6000 | **97.80 GB** |        **6.4 / 49.0 / 153.8** / 218.1 |        4195 ms |    10.5996 | 81.8% |
| Qwen3-32B, bf16, H100 SXM 80 GB |     65.52 GB |          13.4 / 123.2 / 502.0 / 988.4 |         262 ms |    17.0814 | 78.2% |
| Qwen3-32B, Glyd, H100 SXM       | **44.45 GB** |          19.9 / 157.1 / 465.1 / 789.5 |         305 ms |    17.0849 | 78.2% |
| Qwen2.5-7B, bf16, H100 SXM      |     15.23 GB |         25.9 / 207.1 / 823.0 / 1670.2 |          55 ms |    17.0178 | 73.3% |
| Qwen2.5-7B, Glyd, H100 SXM      | **10.32 GB** |        57.4 / 455.5 / 1655.5 / 2807.1 |          64 ms |    17.0162 | 73.4% |

The MMLU answers are bf16's on 99.2% (Qwen3-32B, A6000), 99.4% (Qwen2.5-72B), 100% (Qwen3-32B, H100) and
99.9% (Qwen2.5-7B, H100) of the questions. Across GPUs the A6000s run Glyd's model on fewer of them, so
their pipeline has fewer stages. On the H100 Hugging Face's generation loop is bound by the CPU at few
sequences (Qwen3-32B, profiled over 16 tokens: 54.8 ms a token for bf16, 55.8 for Glyd), and `mma` shows no
gain there: 40.6 ms of GPU time a token against bf16's 28.2, Qwen3-32B's MLP matrices 128 us against 90 at
one token. Use `mma12` on an H100 (above).

## The KV cache

The keys and values a model keeps for the tokens it has seen are bf16 like its weights, and Glyd holds them
compressed too: 34% under bf16 at best (measured: 10.5-10.6 bits a value). `kv.py` holds a Hugging Face
model's cache that way: `GlydKVCache(config)`. With `fused=True` and `use_fused_attention(model)`, a step of
one new token a sequence reads the compressed cache directly. Other steps (a prompt) get the keys and values
back exactly.

Qwen2.5-7B-Instruct, weights in the `mma` layout, RTX 4080 SUPER, an enwik8 prompt then 128 new tokens
(`e2e.py --kv 1024,4096,16384`):

|        Prompt | KV cache, plain |             Packed | A step, plain |   Fused | Peak memory, plain |   Packed |
| ------------: | --------------: | -----------------: | ------------: | ------: | -----------------: | -------: |
|  1,024 tokens |           66 MB |  **46 MB** (70.4%) |       18.3 ms | 19.2 ms |           10.80 GB | 10.78 GB |
|  4,096 tokens |          242 MB | **167 MB** (69.1%) |       19.1 ms | 19.6 ms |           11.41 GB | 11.34 GB |
| 16,384 tokens |          947 MB | **651 MB** (68.7%) |       21.6 ms | 21.7 ms |           13.87 GB | 13.58 GB |

Decoded back, the cache is the plain cache's bit for bit: the same 128 tokens. The attention sums run in
another order than FlashAttention's, as with the weights: 256 tokens of the text fed one at a time after the
prompt, perplexity 2.9895 / 4.3106 / 2.4778 against the plain cache's 2.9950 / 4.3130 / 2.4809, the next
token the plain cache's 97.3% / 99.6% / 99.6% of the time.

## Where it is slower

Measured, and in the tables above:

- **Many sequences at once.** At 64 sequences Glyd's model on fewer GPUs makes fewer tokens a second than
  bf16's (400.5 against 488.4 for Qwen3-32B on A6000s) and still costs less a token. On an A100 and an H100,
  `mma12` takes 6-17% more GPU time a token than bf16 at 32 and 64 sequences. On an A100 at 128 sequences
  Qwen3-8B generates 2993.6 tokens/s against bf16's 2997.4, and Qwen3-14B 2594.0 against 2657.4.
- **An H100 at few sequences.** `mma` takes more GPU time a token than bf16 (Qwen3-32B: 40.58 ms against
  28.22): use `mma12`.
- **Prompts past 128 tokens.** On an A100, Qwen3-8B 6-33% and Qwen3-14B 12-35% longer than bf16's (less
  with the long-prompt path), and Qwen3-8B's steps of 97-128 tokens, 1.16x a layer. On an H100, Qwen3-8B's
  layer takes 1.16 / 1.16 / 1.24 / 1.43 / 1.38x bf16's time at 128 / 129 / 256 / 512 / 1024 tokens; the small
  matrices past 32 tokens (Qwen3-8B's q, k, v 1.04x and o 1.14x at 64) and 113-128 tokens are longer than
  bf16's. On an RTX 4080 SUPER, Qwen3-4B-Instruct-2507's `mma` prompt of 512 tokens takes 52.3 ms against
  bf16's 49.1; on an A10, an L4 and an L40S, long prompts take the percentages above.
- **With vLLM** at saturation on a GH200, an H100 SXM and two RTX A6000s: [vllm/README.md](vllm/README.md#measured).
- **Not measured:** Blackwell; the long-prompt path on an H200, an H100 NVL and the PCIe cards; the L40 and
  the RTX 6000 Ada, which take the default path.

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
`bindings/python/test_gpu.py` checks against the header). An engine in C, C++, Rust or any language with a C
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
glyd.gpu's Linears do by default (without it the routes and `linear` are v0.25.1's). The route variables
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
    python -m glyd.gpu pack Qwen/Qwen3-0.6B qwen3-0.6b-glyd

On an RTX 4080 SUPER (CUDA 13.0; a v0.25 library, C API 5: this release's prints 7), the first pack, then a
merged one (every one of Qwen3-0.6B's 112 packs, its 196 Linears, decodes to the checkpoint's bits so; a bit
flipped in the checkpoint is found):

    $ ./unpack qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/
    libglyd_gpu: C API 5, CUDA runtime 13000
    model.layers.0.self_attn.o_proj: [1024, 2048], 10.86 bits a weight packed, decoded on the GPU
      model.layers.0.self_attn.o_proj.weight [1024, 2048]: the checkpoint's, bit for bit
    $ ./unpack qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ model.layers.0.self_attn.q_proj
    libglyd_gpu: C API 5, CUDA runtime 13000
    model.layers.0.self_attn.q_proj: [4096, 1024], 10.79 bits a weight packed, decoded on the GPU
      model.layers.0.self_attn.q_proj.weight [2048, 1024]: the checkpoint's, bit for bit
      model.layers.0.self_attn.k_proj.weight [1024, 1024]: the checkpoint's, bit for bit
      model.layers.0.self_attn.v_proj.weight [1024, 1024]: the checkpoint's, bit for bit

The glyd-gpu crate's examples do the same from Rust: `examples/unpack.rs` (this one), and
`examples/linear.rs`, a pack multiplied by `linear` for 1-2000 tokens, bit for bit the kernel of the route
this GPU takes:

    cargo run --release -p glyd-gpu --example unpack -- qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/
    cargo run --release -p glyd-gpu --example linear -- qwen3-0.6b-glyd model.layers.0.mlp.gate_proj

And the saved model itself comes from Rust too, on the CPU, byte for byte as `python -m glyd.gpu pack` saves
it (Qwen3, Qwen2, Llama, Mistral, Granite and GraniteMoe; the glyd-gpu command, which the glyd CLI runs):

    glyd pack Qwen/Qwen3-0.6B qwen3-0.6b-glyd
    glyd verify qwen3-0.6b-glyd [--device cuda:0]

## License

The files under gpu/ (and the glyd-gpu crate) are under the [Business Source License 1.1](LICENSE):
source available, free for personal, educational, research and other
non-commercial use; any commercial production use needs a license
(suryakoritala1324@gmail.com); each version converts to Apache-2.0 four
years after its release. The Glyd codec is BSD-3-Clause OR GPL-2.0.
