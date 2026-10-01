# The quickstart on a machine with no CUDA toolkit: 0.26.0rc2's two failures, their fixes, and the acceptance runs (L4 as a 16 GB GeForce, 2026-09-30)

`gpu/vllm/README.md`'s "Local chat, like Ollama" run from nothing by a user with an RTX 4080 SUPER (16 GB, a desktop running,
Ubuntu 26.04, no CUDA toolkit), with 0.26.0rc2 from PyPI, failed two ways. This record reproduces both on the dev L4 in
the user's conditions, finds their causes, measures the fixes, and keeps the runs of `gpu/vllm/acceptance.sh` that fail on
0.26.0rc2 and pass on the fixed tree.

| | 0.26.0rc2 | the fixed tree |
| :--- | ---: | ---: |
| `vllm serve` as the owner ran it | stops at warmup: `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | the same machine runs it with `VLLM_USE_FLASHINFER_SAMPLER=0`; without it a warning at start names the setting |
| allocator warnings while loading (16 GB budget) | 254 | 0 |
| allocator warnings with 21.7 GiB free | 0 | 0 |
| weights, "Model loading took" | 11.83 GiB, 24 s | 11.39 GiB, 15 s |
| KV cache at the 13.71 GiB budget | 1.35 GiB, 9,856 tokens | 1.83 GiB, 13,280 tokens |
| the load's peak, PyTorch reserved / allocated | 14,772 / 12,511 MiB | 12,218 / 11,927 MiB |
| the least GPU memory free while loading | 1-3 MiB | 669 MiB |
| one user, tokens/s (greedy, top-p) | 21.28, 21.09 | 21.25, 21.00 |
| 8 users at once, tokens/s in all (greedy, top-p) | 162.1, 159.7 | 161.4, 159.5 |
| `acceptance.sh` | FAILED (3): no server (nvcc), 3 tracebacks, 256 allocator warnings | PASSED |

## The stand-in for the owner's card

The owner's rc2 log (their first run): the card's total as CUDA reports it 15.57 GiB (16,717,119,488 B), 14.48 GiB free at
vLLM's start (a desktop holds the rest), `--gpu-memory-utilization 0.88` a budget of 13.70 GiB, 11.83 GiB of weights, 1.34
GiB (9,744 tokens) of KV cache, and about 300 lines of "memory allocation failed with OOM" (20-201 MB each, 13-140 MB
free). With `VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: 11.31 GiB, 1.72 GiB (12,544
tokens).

- **A container** from `ubuntu:26.04` (the owner's OS) with gcc and libc6-dev and nothing else: no nvcc, no
  `/usr/local/cuda`, no `CUDA_HOME`, no Python; the host's driver is passed in (`--gpus all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`).
  Without gcc vLLM stops sooner, in its first sampling kernel (Triton builds its launchers with a C compiler:
  `serve-rc2-no-gcc.txt`, "Failed to find C compiler"); the owner's machine has one, so the image has gcc. `uv venv --python
  3.12` and `uv pip install "glyd[vllm]==0.26.0rc2"` from PyPI, as the owner did: vLLM 0.30.0, torch 2.13.0, flashinfer-python
  0.6.18.post1, triton 3.7.1, transformers 5.18.0 (`acceptance/rc2/freeze.txt`).
- **A hog** (`hog.py` of [l4-local-chat-2026-09-30](../l4-local-chat-2026-09-30)) holds the L4's memory but 15,015 MiB, and vLLM
  logs "Free memory on device (14.48/22.04 GiB) on startup", the owner's 14.48 GiB. **`--gpu-memory-utilization 0.622`** is
  0.88 of the owner's total scaled to the L4's 22.04 GiB, the same budget: "(0.622, 13.71 GiB)" against their 13.70.
- **The plugin reads the GPU as GeForce Ada** (`sitecustomize.py`, as in the earlier record), as it reads an RTX 4080 SUPER.
- **It matches the owner's log:** 0.26.0rc2 there took 11.83 GiB of weights and left 1.35 GiB of KV cache (9,856 tokens)
  against their 1.34 GiB (9,744), and with `expandable_segments` 1.74 GiB (12,640 tokens) against 1.72 GiB (12,544).
- Everything else of the load runs the same on both: the weights are the checkpoint's 15.26 GiB of safetensors read from
  disk (here the OS cache), the L4 is an Ada GPU as the 4080 SUPER is. What a desktop's own allocations do while the load
  takes the free memory is not in this test, and nothing here ran on an RTX 4080 SUPER.

## Failure 1: nvcc

vLLM 0.30's kernel warmup (`vllm/v1/worker/gpu/warmup.py`, `_warmup_kernels`) runs a request through `sample_tokens`, with
top-k and top-p set, which Qwen3's own `generation_config.json` gives every request (`top_k 20`, `top_p 0.95`), so vLLM's
sampler calls FlashInfer's top-k/top-p kernel (`Using FlashInfer for top-p & top-k sampling.`). FlashInfer builds that kernel
the first time it is called: `flashinfer.sampling.top_k_top_p_sampling_from_logits`, `get_sampling_module`,
`gen_sampling_module().build_and_load()`, with nvcc found through `CUDA_HOME`, `PATH`, then `/usr/local/cuda`
(`flashinfer/jit/cpp_ext.py`'s `get_cuda_path`). `serve-rc2.txt` (the owner's command, no toolkit): `RuntimeError: Could not
find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`, the owner's line, after the weights (11.83 GiB) had loaded
and the KV cache had been sized.

- **`VLLM_USE_FLASHINFER_SAMPLER=0`** (vLLM's `envs.py`, `topk_topp_sampler.py`): vLLM samples with PyTorch and Triton.
  The server comes up (`serve-fix-sampler-off.txt`).
- **FlashInfer's precompiled kernels** exist as separate packages for 0.6.18.post1: `flashinfer-jit-cache` for CUDA 13 (1.0
  GB, from `https://flashinfer.ai/whl/cu130/`) and `flashinfer-cubin` (1.57 GB; 6.3 GB installed). `flashinfer-jit-cache`
  alone is enough: with it and no nvcc the server comes up with FlashInfer's sampler on (`serve-fix-jit-cache.txt`).

Both ways measured, on the fixed tree, 16 GB budget, five answers of 256 tokens and then eight at once (`chatbench.py`):

| sampler | greedy, 1 user | top-p, 1 user | greedy, 8 users in all | top-p, 8 users in all | first token |
| :--- | ---: | ---: | ---: | ---: | ---: |
| PyTorch and Triton, second start | 21.25 | 21.00 | 161.4 | 159.5 | 53 ms |
| FlashInfer (jit-cache), second start | 21.12 | 20.98 | 161.0 | 159.9 | 53-54 ms |
| PyTorch and Triton, first start | 21.23 | 21.07 | 161.9 | 125.4 | 53 ms |
| FlashInfer (jit-cache), first start | 21.14 | 21.01 | 161.2 | 160.4 | 53-56 ms |
| rc2, PyTorch and Triton | 21.28 | 21.09 | 162.1 | 159.7 | 53 ms |

The one low figure is the first start's 8-user top-p burst (`after-fix-sampler-off.txt`), 125.4; the second start of the
same server gave 159.5 (`after-fix-sampler-off-2.txt`). Its cause was not examined (the first sampling of that batch
shape on a Triton cache that had not seen it is the likely one). The greedy answer's text is the same in every run (its
sha256 `00b02abc33a9`). `sampler-check.txt` (`sampler_check.py`, 399,872 draws each from the same logits, top-k 20 then
top-p 0.8 at temperature 0.7, 12 tokens in the exact distribution's support): PyTorch's sampler is 0.0010 from the exact
distribution in total variation, FlashInfer's 0.0016, and they are 0.0020 from each other; neither put mass outside the
support; 9.4 against 5.9 ms for a batch of 256.

**Chosen:** (a), `VLLM_USE_FLASHINFER_SAMPLER=0` in the quickstart's command: no extra download, no difference in speed or in
the draws. Glyd's plugin does not set it: `GlydConfig.maybe_update_config` logs one warning, in the engine's process before
the weights load, where vLLM's FlashInfer sampler is on, FlashInfer is installed, and no nvcc is found (PATH, `CUDA_HOME`
or `/usr/local/cuda`) and no `flashinfer-jit-cache` package is (`serve-fix-nvcc-warning.txt`, 09:31 against the failure at
09:58 in the same log): "glyd: no nvcc ... Start the server with VLLM_USE_FLASHINFER_SAMPLER=0 (vLLM then samples with
PyTorch and Triton), or install flashinfer-jit-cache (FlashInfer's precompiled kernels) or the CUDA toolkit".

Other JIT builds on a GeForce Ada card (vLLM 0.30's `kernel_warmup`, `jit_warmup`): `flashinfer_autotune` runs only at
compute capability 9.0 and up; FlashInfer's attention warmup only where its attention backend is chosen (Ada's is
FlashAttention, prebuilt); `deep_gemm_warmup` needs a Hopper or Blackwell card; the Triton kernels (the sampler's among
them) are compiled by Triton's own ptxas, with gcc for the launcher. The real run of the quickstart (below) passed warmup
and served chats, the OpenAI API's and Open WebUI's, with no nvcc: none remained.

## Failure 2: the allocator warnings

**What allocates 20-201 MB at a time** (the owner's sizes; this run's 254 are 5 of 18 MiB, 57 of 20, 18 of 30, 26 of 32, 42
of 64, 27 of 96, 67 of 128 and 12 of 192: `serve-rc2.txt`). The sizes name the allocators:

- **134,217,728 B (128 MiB), the owner's number**, is `kernels._hist`'s widening: `CHUNK = 1 << 25` weights widened to
  int32 and masked, four temporaries of 128 MiB a pass (`.to(torch.int32)`, `& 0xFFFF`, `>> 7`, `& 0xFF`), run at the start of
  every pack to count the exponents. 96 MiB and 64 MiB are the same temporaries for a matrix of 25.2M and of 16.8M weights
  (qkv, o_proj), 18 MiB and 30 MiB the packs' own passes and qkv's pack (31,457,280 B).
- **201,326,592 B (192 MiB)** is one layer's bf16 gate_up weight (24,576 x 4,096 x 2 B), and 96 MiB and 32 MiB those of down and
  o_proj: vLLM's layerwise loading makes each layer's weights on the GPU, and Glyd packs them from there.
- **20,971,520 B** is the allocator's own 20 MiB segment for any request of 1 to 10 MiB.
- **vLLM loads with PyTorch's allocator told not to split a block past 20 MiB**: `Worker.load_model` wraps the load in
  `_scoped_allocator_max_split(max_split_size_mb=20)` (`vllm/v1/worker/gpu_worker.py`, "to reduce allocator fragmentation").
  A cached block of 192 MiB is not cut for a 120 MiB request, so each size of temporary and each long-lived pack keeps blocks
  of its own, and the cache of blocks no request fits grows (14,772 MiB reserved at its peak against 12,511 allocated; at the
  end of the load 288 free blocks, 2.6 GiB, 100 of them 20 MiB segments) until a cudaMalloc fails. PyTorch then prints
  "memory allocation failed with OOM", frees every unused block and tries again. The warnings are that cache emptied 254 times
  as the load nears the card's end, not a shortage: the weights fit with 2.6 GiB to spare. The run with 21.7 GiB free has none
  because the cache never meets the card's end.
- **The 0.52 GiB**: vLLM's "Model loading took" is `max_memory_allocated` after `empty_cache` (`platforms/cuda.py`). With
  rc2 it was 11.83 GiB (12,112 MiB), 526 MiB over the 11,586 MiB the load asked for. A live block bigger than its tensor
  is the same rule at work: a 120 MiB `data` tensor took a cached 128 MiB block of the `_hist` temporaries whole (36 blocks,
  8 MiB over each), the 60 MiB ones 64 MiB blocks (36, 4 MiB over), the 30 MiB ones 32 MiB blocks (30, 2 MiB over): 526 MiB
  of slack, kept for the life of the server, which is 0.5 GiB of KV cache (with `expandable_segments`, which does not use
  `max_split_size_mb`, there was none).

**Options measured**, 16 GB budget, Qwen3-8B (`run-*.txt`, `serve-*.txt`: allocator warnings are lines with "with OOM"; the
probe, `probe_sitecustomize.py`, prints PyTorch's allocated and reserved after each packed layer and the allocator's blocks at
the end of the load):

| run | what | warnings | weights, load | KV cache | least free | peak reserved | slack |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rc2-sampler-off` | 0.26.0rc2 | 254 | 11.83 GiB, 23.9 s | 1.35 GiB, 9,856 | 1 MiB | 14,772 | 526 MiB |
| `rc2-expandable` | rc2, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 73 (mapping) | 11.31 GiB, 34.1 s | 1.74 GiB, 12,640 | 15 MiB | 14,768 | 0 |
| `rc2-trim` | rc2, `empty_cache()` after each packed layer | 0 | 11.84 GiB, 25.3 s | 1.36 GiB, 9,920 | 679 MiB | 12,998 | 536 MiB |
| `hist16` | `_hist` in int16, 4M weights a pass | 133 | 11.43 GiB, 22.9 s | 1.76 GiB, 12,768 | 1 MiB | 14,772 | 117 MiB |
| `hist16-pack2M` | and the packers' passes 2M weights | 1 | 11.38 GiB, 15.0 s | 1.83 GiB, 13,280 | 15 MiB | 14,758 | 72 MiB |
| `hist16-trim` | `hist16` and `empty_cache()` after each layer | 0 | 11.42 GiB, 24.5 s | 1.78 GiB, 12,992 | 679 MiB | 12,790 | 104 MiB |
| `fix` | **the fix:** `hist16-pack2M`, the bf16 weight dropped at once, the cache trimmed where it outweighs the driver's free memory | **0** | 11.39 GiB, 15.3 s | **1.83 GiB, 13,280** | 669 MiB | 12,218 | 73 MiB |
| `fix-chunks-4M-4M` | the fix with the packers' passes at 4M | 0 | 11.45 GiB, 22.4 s | 1.74 GiB, 12,640 | 673 MiB | 13,266 | 141 MiB |
| `fix-chunks-2M-1M` | the fix with `_hist` at 2M and the packers at 1M | 5,312 | 11.43 GiB, 36.8 s | 1.71 GiB, 12,480 | 1 MiB | 14,772 | 117 MiB |
| `fix-24gb` | the fix, no hog, `--gpu-memory-utilization 0.88` | 0 | 11.38 GiB, 15.1 s | 7.51 GiB, 54,688 | 2,185 MiB | 14,828 | 72 MiB |
| `rc2-24gb` | 0.26.0rc2, no hog | 0 | 11.83 GiB, 23.2 s | 7.04 GiB, 51,264 | | 19,622 | 526 MiB |

- **The packers' passes** (`packbench.py`, `packbench.txt`): the packs' bits are the same whatever the passes (every
  variant, both layouts, three shapes, "same bits as old"). With the passes at 2M weights a pack of gate_up (24,576 x 4,096)
  takes 166 ms against 284 (tiered) and 58 against 78 (12-bit), and its temporaries are half the size (261 MiB over the
  pack against 392). At 1M the allocator's 1-10 MiB requests each take a 20 MiB segment of their own: the load that logged 5,312
  warnings. An int64 temporary of 16 MiB (2M weights) is past 10 MiB and under the 20 MiB at which vLLM stops splitting.
- **Trimming the cache alone** (`rc2-trim`) removes the warnings and 1.8 GiB of the peak but not the slack; **the smaller
  passes alone** (`hist16-pack2M`) remove the slack and most of the warnings, and a load 9 s shorter, but leave the
  card at 15 MiB free and one warning at the end; the two together leave 669 MiB free and none.
- **The trim** (`vllm_plugin._trim`) is called after each packed layer (a Linear's, a mixture of experts' experts), where
  PyTorch's unused blocks outweigh the driver's free memory (`reserved - allocated > mem_get_info()[0]`): on a 24 GB card it
  never fires. A packed layer's bf16 weight is dropped (`_drop`) before it: vLLM keeps its Parameter, and the weight with
  it, until the layer is done.
- **`expandable_segments:True` is not the fix:** its warnings are of another kind, "expandable_segments: memory mapping
  failed with OOM" (73 of them, the other count's name not matching them), and the allocator still reached 15 MiB free.
  `acceptance.sh` counts both kinds (`with OOM`).
- The first attempts of this record's batch (`batch-summary.txt`) have two failed steps: a first overlay of the fix on rc2's
  libraries, whose C API (7) the branch's Python did not match, and the acceptance runs, whose work directory held a
  dangling symlink for uv's cache. Neither is the code's; both were run again, and what is above is the second run.

## Open WebUI

Found by driving it: the owner's `uvx --python 3.11 open-webui@latest serve --port 3000` started and listed no models.

- **Why:** Open WebUI 0.11 keeps its connections in its database (`PersistentConfig`): the variables are read on a first
  start only. A first start without `OPENAI_API_BASE_URL` stored OpenAI's address and an empty key, and a second, with them set,
  listed no models (`/openai/config` showed `https://api.openai.com/v1`, `/api/models` `[]`); with `ENABLE_PERSISTENT_CONFIG=False`
  on the same data directory it listed `['Qwen/Qwen3-8B', 'arena-model']`. A first start with the variables gave models.
  The uvx route's data directory is inside uv's cache, so an earlier try with other variables leaves it as it was.
- **`uvx` writes `.webui_secret_key` into its working directory** (`PermissionError: '/.webui_secret_key'` from `/`), and
  `serve` listens on 0.0.0.0 unless `--host 127.0.0.1`: with `WEBUI_AUTH=False` that is an admin page on the network.
- **The browser's chat request** (captured from the page: `POST /api/chat/completions` with `features`, `params`,
  `tool_servers`, `session_id`, `chat_id`, `user_message`, `background_tasks`) makes Open WebUI offer the model its built-in
  tools: the backend's request to the model carried 33-35 `tools` and no `tool_choice`, and a title, a tags and a follow-up
  request beside it. vLLM answers that with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to
  be set` (`vllm/renderers/online_renderer.py`) unless the server has both flags: `--tool-call-parser hermes` is Qwen3's.
  The chat endpoint without a `session_id` sends no tools, so the owner's earlier API test never met it: `acceptance.sh`
  sends the browser's fields with a `session_id` and the response streams back to it.
- **`--reasoning-parser qwen3`** (`reason_probe.py`, `after-fix-hermes.txt` and `after-fix-hermes-qwen3.txt`): thinking on,
  without it every delta has `content` only, `<think>` tags in it (`content has <think>: True`); with it `reasoning` (1,223
  deltas, 3,236 characters) and `content` (140) apart, no `<think>` in the answer. A tool call parsed both ways (`finish:
  tool_calls`, `get_weather {"city": "Paris"}`). Open WebUI 0.11.4 reads `delta.reasoning` (`utils/middleware.py`). In its page
  (below) the thinking is a collapsed "Thought for 7 seconds".
- **A real browser** (the built-in browser of this session over an SSH tunnel to the server of `acceptance/fix-reasoning-parser`,
  held after its checks): Open WebUI 0.11.4's page listed `Qwen/Qwen3-8B`; "What is lossless compression? Answer in two
  sentences." gave a collapsed "Thought for 7 seconds" and a two-sentence answer; "What is the current Unix timestamp? Use
  your tools to find out." showed "Explored get_current_timestamp", the tool's answer in the next step and the sentence "The current Unix
  timestamp is 1790815600 ...". The page named the chat itself. The server's log after it: no traceback, no allocator
  warning, 13 requests (the chats' and Open WebUI's background ones), two running at the most
  (`acceptance/fix-reasoning-parser/server.txt`).

## `acceptance.sh`

`gpu/vllm/acceptance.sh`, the README's blocks marked `<!-- acceptance: ... -->` as they are, in the clean container; the
card was `--card 4080s` (`--budget-mib 14828 --card-mib 15942 --geforce`). `acceptance/*.txt` are its outputs, the
directories its logs.

- **0.26.0rc2 with the owner's command** (`acceptance/rc2.txt`; `--version 0.26.0rc2`, `--command "vllm serve ..."`):
  FAILED (3): the server did not come up; 3 tracebacks (`RuntimeError: Could not find nvcc and default
  cuda_home='/usr/local/cuda' doesn't exist`); 256 allocator out-of-memory warnings.
- **The fixed tree** (`acceptance/fix.txt`, `acceptance/fix-reasoning-parser.txt`; `--wheel`, a wheel of release-0.26.0 at
  889f4e3 with this branch's change over rc2's libraries, `scripts/buildwheel.sh`, its version string set to 0.25.1 for the file
  name the batch had): PASSED both times, the second with the README as it
  now is: no nvcc, install 67-98 s, server up after 49-64 s, weights 11.39 GiB, KV cache 13,280 tokens, no traceback and no
  allocator warning; the OpenAI API's two streamed turns ("Paris", "Berlin", the second with the first as history, top-p)
  and a tool call (`get_weather {"city": "Paris"}`); Open WebUI 0.11.4 by `uvx` and by Docker (`ghcr.io/open-webui/open-webui:v0.11.4`):
  the model listed, the browser's chat request answered, a tool call (`get_current_timestamp`) in the stream. The first
  run's command was without `--reasoning-parser`: the empty `<think>` block of a `/no_think` answer is in its answers'
  content (`'<think>\n\n</think>\n\nParis'`).
- **The merged release tree** (`acceptance/release-16gb.txt`, `acceptance/release-24gb.txt`, `scripts/final.sh`): a wheel of
  `bindings/python` as merged into release-0.26.0 (`7f2b218`; its own version, 0.26.0rc2, over rc2's libraries) and the
  script of the branch's last commit, the README's command as it stands. At the 16 GB card: PASSED, both Open WebUI routes
  (install 94 s, up after 52 s, 11.39 GiB, 13,280 tokens, no traceback, no allocator warning). With no hog (21.85 GiB free at start, the
  plugin reading GeForce Ada, no Open WebUI): PASSED (up after 52 s, 11.38 GiB, 54,688 tokens, no warning).
- Its own checks first ran against a stand-in server (`scripts/stub_server.py`) on a machine without a GPU, which is
  how the working directory of `uvx`, the exit status of the checks and the Docker route's volume were found.

## The plugin against vLLM's bf16, after the change

`gpu/vllm/check_vllm.py --brief` on the L4 with no hog and the fixed plugin over rc2's libraries (`check_vllm/*.txt`):
Qwen3-1.7B, all 5 checks passed (every pack, 112, decoded to its weights bit for bit; each layer's product within 1e-2 of
F.linear, 4.06e-3; top-1 agreement with bf16's continuation 0.9942, bf16 eager's own 0.9929; exact eager bf16 eager's
tokens, logprobs and prompt_logprobs bit for bit); granite-3.1-3b-a800m-instruct (a mixture of experts), all 5 passed
(every pack, 64 and 32 layers' experts, bit for bit; products 3.79e-3; agreement 0.9896; exact eager bit for bit). The
unit tests: `test_vllm.py` 11 of 11 and `test_gpu.py` with vLLM installed and no GPU, on the box.

## Not measured

An RTX 4080 SUPER with the fixes, and what a desktop's allocations do in the seconds the load takes the card's free memory
(669 MiB free at the least with the fix). Other cards: an RTX 50 (Blackwell) 16 GB, whose layout `auto` is the 12-bit one. The
first start of a server that downloads the 16 GB model. An older driver than 595. A machine with no C compiler beyond the
one failure above. Open WebUI's page was driven with one browser, on the first chats of a fresh data directory.

## Run it again

On a machine with Docker and the NVIDIA container toolkit, from the repository: `bash gpu/vllm/acceptance.sh --version
0.26.0rc2 --card 4080s --command "vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager --max-model-len 8192
--gpu-memory-utilization 0.88 --host 127.0.0.1" --webui none` (the owner's run), and `bash gpu/vllm/acceptance.sh --wheel
glyd-....whl --card 4080s` (the fixed tree). The load's runs: `scripts/rv.sh` (`dkg.sh`, `run1.sh`) with
`probe_sitecustomize.py` on `PYTHONPATH`; `scripts/batch.sh` is the batch these came from.
