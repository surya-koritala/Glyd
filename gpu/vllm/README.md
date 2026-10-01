# Glyd in vLLM: `vllm serve MODEL --quantization glyd`

vLLM serves the model with its Linear layers held packed on the GPU, bit for bit, and a mixture of experts' experts
too. Glyd's kernels multiply them from there. vLLM sizes its KV cache after the weights load, so the memory the packs
save becomes KV cache: more requests at once on the same GPU.

```bash
pip install "glyd[vllm]"                                  # vLLM 0.30, and Glyd with its plugin
vllm serve Qwen/Qwen3-8B --quantization glyd              # a bf16 checkpoint, packed as it loads
vllm serve ./qwen3-8b-glyd --quantization glyd            # a glyd save (glyd pack, glyd.save_pretrained), as saved
vllm serve Qwen/Qwen3-8B --quantization glyd --additional-config '{"glyd": {"layout": "mma12"}}'
```

On a 16 GB card these defaults do not leave room for a chat: vLLM sizes the context to the model's own 40,960 tokens and
takes 0.92 of the memory, which a desktop shares. Start with `glyd run` ([below](#local-chat-like-ollama)), which works the
memory and the context out from the card, or give `vllm serve` the flags of [the by-hand section](#advanced-vllm-serve-by-hand).

The `glyd` package registers the plugin with vLLM through its `vllm.general_plugins` entry point; nothing else is
needed. It is tested with vLLM 0.30.0, and `glyd[vllm]` pins `vllm>=0.30,<0.31`. With another minor release of vLLM
the entry point logs one line and loads nothing, and `--quantization glyd` stops with why.

## Local chat, like Ollama

Two commands, on Linux with an NVIDIA GPU:

<!-- acceptance: install -->
```bash
curl -LsSf https://getglyd.com/install.sh | sh
```

<!-- acceptance: run -->
```bash
glyd run Qwen/Qwen3-8B
```

The script installs [uv](https://docs.astral.sh/uv/) if it is missing, then Glyd with vLLM 0.30 and PyTorch as one isolated
tool, on a Python 3.12 that uv fetches for it (about 8 GB of disk; no sudo, no virtual environment, no pip), and ends with
`glyd doctor`. `glyd run` downloads the model (Qwen3-8B is 16.4 GB the first time), checks that this machine can run it,
starts vLLM with settings it works out from the GPU, and opens a chat: type in the terminal (`/bye` leaves, `/clear`
starts over, `/think` turns the model's thinking on and off), or open http://localhost:8000.

**Needs:** Linux on x86_64 (the installer also takes aarch64, which this flow was not run on); an NVIDIA GPU of the
Ampere generation or newer (RTX 30 and 40 series, A10, A100, L4, H100 and later) and its driver, 580 or newer (the CUDA 13
PyTorch that vLLM 0.30 installs); curl; and disk for the packages and the model. It needs no CUDA toolkit and no sudo.
vLLM's Triton builds small launchers with a C compiler when the server starts: where the machine has no gcc or clang, the
installer adds ziglang, a compiler from PyPI, and `glyd run` hands it to vLLM. `glyd doctor` checks each of these and says
what to do about a line that fails.

`glyd run`, in order, stopping with a plain message and what to do where it must:

1. **Checks** the GPU and its driver (against the CUDA this PyTorch was built for), the compiler, vLLM's version, the model
   (it is on the Hugging Face Hub, bf16, and not gated without your token), whether the model with Glyd fits the memory free
   now (otherwise: how much it needs, what holds the GPU's memory, and the largest model of its family that fits), the disk
   the download needs, and the port.
2. **Downloads** the model, with one progress bar. A gated model (Llama, Gemma): accept its licence on its Hugging Face page,
   then `glyd login` (the `hf` command is not on the PATH of a tool install).
3. **Chooses** the settings below and prints them on one line.
4. **Starts** vLLM with its output in a log file, and shows what it is doing.
5. **Chats**, in the terminal and at the printed address. A conversation longer than the model's window is said so in both:
   "This conversation is longer than the model's window (10,240 tokens). Start a new chat with /clear."

`glyd run MODEL --prompt "Say hello"` prints one answer on stdout (the thinking and notices go to stderr) and exits; a
prompt on stdin does the same. `--context N` sets the context. Any vLLM flag after a lone `--` is passed on and wins over
what was chosen: `glyd run MODEL -- --max-model-len 4096`. A `glyd serve` of the same model already up on the port is used
instead of a second server.

### The settings

| Setting | `glyd run` and `glyd serve` choose | Why |
| :--- | :--- | :--- |
| `--gpu-memory-utilization` | the memory free now, less 0.55 GiB for the server's CUDA context (outside vLLM's share) and 0.4 GiB left for a desktop, over the card's total as CUDA reports it (not nvidia-smi's), rounded down to 1%; at most 0.92. `glyd run` (one chat) takes only what two windows of KV cache need, `glyd serve` the whole share | vLLM's share is of that total: an RTX 4080 SUPER's is 15.57 GiB, nvidia-smi's 16,376 MiB is 15.99 |
| `--max-model-len` | the model's own length, or the most the KV cache holds at that share (a multiple of 1,024). Under 4,096 the model does not fit | the KV cache is what is left after the weights |
| eager mode (`--enforce-eager`) | always; `-- --no-enforce-eager` compiles | on an L4 with Qwen3-8B compiled was 2-3% faster in all, for 1, 4 and 8 users (21.7, 84.8 and 165.8 tokens/s against 21.2, 82.3 and 161.2), the server up in 2 min 45 s against 47 s, and it needs 2.15 GiB beyond the weights where eager needs 0.5 |
| the layout (`GLYD_LAYOUT`) | the plugin's own choice for the GPU (the tiered layout, 10.80 bits a weight, on Ada and wherever only it fits; else the 12-bit one), the tiered one also where the 12-bit one leaves less room than an 8,192-token chat | the layout is counted into the memory above, so the plugin packs what was counted |
| the sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) | PyTorch's | FlashInfer's compiles with nvcc at the first request that samples, which stopped a server on a machine with no CUDA toolkit; the same tokens a second (21.1 and 21.2 for one user) |
| tool calls and thinking | by family: Qwen3 `hermes` and `qwen3`; Qwen3 Instruct-2507 and Qwen2.5 `hermes`; Qwen3-Coder `qwen3_coder`; DeepSeek-R1 distills `deepseek_r1`; Llama 3.x `llama3_json`; Mistral `mistral`. Another family chats without tool calls | the names are vLLM 0.30's |
| the allocator | PyTorch's default | `expandable_segments:True` gave 24 allocator warnings in a load at a 16 GB budget, and the default none |
| telemetry, the address | `VLLM_NO_USAGE_STATS=1`; the server listens on 127.0.0.1 | `glyd serve --host 0.0.0.0` opens it to the network, with no key unless `-- --api-key SECRET` is added |

The one line printed before loading is these, for example `Settings: 10,240-token context (the most that fits), eager
mode, 61% of GPU memory (14.4 GB), tool calls (hermes), thinking shown apart (qwen3); PyTorch sampler (no CUDA toolkit).` The model's weights with Glyd are counted from its config and file sizes before anything downloads
(Qwen3-8B: 12.2 GB, bf16's 16.4).

### `glyd serve`, `glyd doctor` and Open WebUI

`glyd serve MODEL` does the same checks and chooses the same settings, and leaves the server up for other programs: the
OpenAI API at http://localhost:8000/v1 and the chat page at http://localhost:8000. `glyd doctor` prints the GPU, driver,
CUDA, free memory, compilers and versions, and which of Qwen3-8B, 14B and 32B fit, at Glyd's sizes.

Open WebUI is a chat page with accounts, history and tools, which talks to that API. It uses no GPU memory. Pinned to
0.11.4, the version tested. Without Docker:

<!-- acceptance: webui-uvx -->
```bash
OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=none WEBUI_AUTH=False ENABLE_PERSISTENT_CONFIG=False \
  uvx --python 3.11 open-webui@0.11.4 serve --host 127.0.0.1 --port 3000
```

With Docker Engine on Linux, whose containers can share the host's network (`--network=host` is Linux's: Docker Desktop on
macOS and Windows does not give the container the host's 127.0.0.1):

<!-- acceptance: webui-docker -->
```bash
docker run -d --name open-webui --network=host -e PORT=3000 -e HOST=127.0.0.1 \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 -e OPENAI_API_KEY=none -e WEBUI_AUTH=False \
  -e ENABLE_PERSISTENT_CONFIG=False -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:v0.11.4
```

With Docker Desktop, or any Docker without host networking, the container reaches the host by name, and the server has to
listen on every interface, which opens it to the network: give it a key.

```bash
glyd serve Qwen/Qwen3-8B --host 0.0.0.0 -- --api-key YOUR_KEY
```

<!-- acceptance: webui-bridge -->
```bash
docker run -d --name open-webui -p 127.0.0.1:3000:8080 --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=YOUR_KEY -e WEBUI_AUTH=False \
  -e ENABLE_PERSISTENT_CONFIG=False -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:v0.11.4
```

Open http://localhost:3000. The third command was run on Docker Engine on Linux without host networking, which is the
case `host-gateway` makes work there; Docker Desktop is the case it is written for and was not run. Notes:

- `WEBUI_AUTH=False` leaves Open WebUI with no login, and it can run tools: anyone who can reach port 3000 uses it. The
  commands keep it on this computer (`HOST`, `--host` and `127.0.0.1:3000` publish nothing else); to let others in, take
  `WEBUI_AUTH=False` out and make an account.
- `ENABLE_PERSISTENT_CONFIG=False` makes these variables the settings on every start. Open WebUI otherwise keeps the
  connection it first started with (the default, OpenAI's) in its data directory, takes the variables only on a first
  start, and shows "No models available" on a later one that has them.
- The `uvx` route needs no Docker. `docker run` needs a user in the `docker` group (or `sudo`), which a Docker Engine install
  does not give you by itself. Run `uvx` from a directory you can write to: Open WebUI keeps a secret key file in it.
- Its chats offer the model Open WebUI's built-in tools, which is why `glyd serve` starts vLLM with a tool-call parser
  (Qwen3: `hermes`): without one every chat is answered with `"auto" tool choice requires --enable-auto-tool-choice and
  --tool-call-parser to be set`. The thinking arrives as a field of its own (`--reasoning-parser qwen3`), which Open WebUI
  shows as a collapsed "Thought".

### Measured

On an L4 (24 GB) with no CUDA toolkit, installed by the script from this tree's wheel, vLLM 0.30.0, Qwen3-8B unless it says
otherwise. The 16 GB row is the L4 with another process holding the GPU's memory down to what an RTX 4080 SUPER with a
desktop has free (14.48 GiB: the card's 15.57 GiB total, a desktop's share taken), and the plugin reading it as the
GeForce Ada card that it is; likewise the 8 GB row (7.8 GiB free). `glyd` prints GB (10^9 bytes), vLLM's log GiB.

| Free at start | `glyd run` chose | vLLM logged | Ready in |
| :--- | :--- | :--- | ---: |
| 23.7 GB (the whole L4) | 92% of GPU memory (21.8 GB), a 40,960-token context (the model's own limit) | weights 11.38 GiB, KV cache 61,104 tokens | 59 s |
| 15.5 GB (a 16 GB card, a desktop) | 61% (14.4 GB) of the L4's total, the share of 0.86 on a 4080 SUPER's, a 10,240-token context | weights 11.39 GiB, KV cache 11,360 tokens | 39 s |
| 8.2 GB (an 8 GB card) | Qwen3-8B refused: "needs about 14.7 GB of GPU memory with Glyd (12.2 GB of weights and room for a 4,096-token chat); your GPU has 8.2 GB free. Or try Qwen/Qwen3-1.7B, which needs about 5.1 GB" | | |
| the same, Qwen3-1.7B | 30% (7.1 GB), a 26,624-token context | weights 2.47 GiB, KV cache 35,248 tokens | 32 s |
| the same, Qwen3-4B, `-- --max-model-len 2048` | 30% (7.1 GB), the context given | weights 5.48 GiB, KV cache 4,528 tokens | 39 s |

The numbers `glyd run` predicts from the config were within 0.11 GiB of the weights vLLM logged (over, for the smaller
models) and 2-7% under its KV cache tokens: it never promised a context vLLM then refused. In the servers' logs, no
traceback and no allocator warning ("memory allocation failed with OOM" or "memory mapping failed with OOM"), 0 of 8 runs
in this table's logs. {{MEASURED_EXTRA}}

### Update, remove, logs

Run the install line again to update (it installs the release the script was written for). `uv tool uninstall glyd` removes
the tool; the models stay in the Hugging Face cache (`~/.cache/huggingface`, or where `HF_HOME` says), and a run's log is
in `~/.local/state/glyd/logs` (the last ten are kept). The `glyd` command is the compression program's too: the Rust
program is not in the wheel, so `glyd FILE` and `glyd pack` work where that program is on the PATH (Homebrew, `cargo
install glyd`, the release tarball), and `glyd` says where to get it where it is not.

`bash gpu/vllm/acceptance.sh` runs this section from nothing, in a container with no CUDA toolkit and no compiler: the
two commands above, the chat page, the API, a conversation longer than the window and Open WebUI, failing on a traceback,
an allocator warning or a missing answer. `--card 4080s` leaves the server a 16 GB card's memory, `--card 8gb` an 8 GB
card's, where it expects the refusal and runs the model suggested (`--help`).

### Advanced: `vllm serve` by hand

`glyd run` is `vllm serve` with its flags decided for you, so every vLLM flag works with it, and the plugin works with
`vllm serve` as it is. By hand, for Qwen3-8B on an RTX 4080 SUPER with a desktop, and what each flag is for:

**Needs**, none of it a CUDA toolkit: an NVIDIA driver 580 or newer (the CUDA 13 build of PyTorch that vLLM 0.30
installs; the tests ran on 595), [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12, with its headers), a C
compiler (`sudo apt install build-essential`: Triton builds its launchers with one, and without it vLLM stops at start
with "Failed to find C compiler"), and disk: about 8 GB for the packages and 16 GB for the model.

<!-- acceptance: setup -->
```bash
uv venv --python 3.12 ~/glyd-env && source ~/glyd-env/bin/activate
uv pip install "glyd[vllm]"
```

<!-- acceptance: serve -->
```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-8B --quantization glyd --enforce-eager \
  --max-model-len 8192 --gpu-memory-utilization 0.88 --host 127.0.0.1 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

The server answers on http://localhost:8000 once it logs `Application startup complete`.

- **`VLLM_USE_FLASHINFER_SAMPLER=0`**: vLLM 0.30 samples top-k and top-p, which Qwen3's own settings use, with
  FlashInfer. FlashInfer builds that kernel with nvcc on the first request that samples, and vLLM's warmup makes that
  request at start: with no CUDA toolkit the server stopped there, after loading the weights, with "Could not find nvcc
  and default cuda_home='/usr/local/cuda' doesn't exist". This makes vLLM sample with PyTorch and Triton instead, with
  the same speed: 21.25 tokens/s greedy and 21.00 with top-p for one user, 161.4 and 159.5 in all for 8 users at once,
  against 21.12, 20.98, 161.0 and 159.9 with FlashInfer's own kernel (`flashinfer-jit-cache`, 1 GB, installed in place of
  the toolkit); the greedy answer is the same text, and the two samplers' draws are the same distribution (about
  400,000 each: 0.0010 and 0.0016 from the exact one in total variation, 0.0020 from each other). Glyd logs one warning
  at start where it finds no nvcc and the sampler on, naming both ways out.
- **`--enable-auto-tool-choice --tool-call-parser hermes`**: Open WebUI offers the model its built-in tools in every
  chat. The request carries `tools`, which vLLM takes as `tool_choice` auto, and without these flags it answers every
  chat with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`. hermes is Qwen3's
  tool-call format.
- **`--reasoning-parser qwen3`**: Qwen3 thinks before it answers. With the parser the thinking is a field of its own
  (`reasoning`, which Open WebUI shows as a collapsed Thought), and the answer's `content` has no `<think>` in it; without
  it the thinking comes in the content, tags and all. A tool call parses either way.
- **`--gpu-memory-utilization 0.88`** is vLLM's share of the card's total memory, for the weights, the KV cache and the
  working memory, of the total CUDA reports (`python -c "import torch; print(torch.cuda.mem_get_info()[1] / 2**30)"`):
  an RTX 4080 SUPER's is 15.57 GiB (16,376 MiB is nvidia-smi's, 15.99 GiB), so 0.88 is 13.70 GiB. The server's CUDA
  context and the desktop live outside it: on that card with a desktop vLLM logged 14.48 GiB free at start. If something
  else holds more, vLLM stops at start with "Free memory on device ... is less than desired GPU memory utilization". Close
  that program, or lower the number.
- **`--enforce-eager`** runs without torch.compile and CUDA graphs. At this budget vLLM's defaults gave no server (-0.83
  GiB left for the KV cache, measured on an L4 at 14.1 GiB). A compiled server (context 4,096, 8 sequences, 512 tokens a
  step) started on its second try, from the compile cache. Eager started at once, and one user's tokens a second were
  within 2% of compiled's.
- **`--max-model-len 8192`** is the longest chat, in tokens. The KV cache holds 1.62 of them. A longer chat is refused
  with "maximum context length is 8192 tokens".
- **`--host 127.0.0.1`** keeps the server on this machine. Without it vLLM listens on every interface, with no key.

Open WebUI, as above, on the server's port 8000. To test the server alone:

```bash
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "What is lossless compression? Answer in one sentence. /no_think"}],
  "max_tokens": 100}'
```

Qwen3 thinks before it answers. `/no_think` at the end of a message skips that.

`bash gpu/vllm/acceptance.sh --flow vllm` runs this part from nothing, on a machine with no CUDA toolkit: a clean environment,
the install, the serve command above as written, a chat through the OpenAI API (two streamed turns and a tool call) and
through Open WebUI both ways, failing on a traceback, an allocator warning or a missing answer. Where the GPU is bigger
than a card it stands in for, `--card 4080s` leaves the server that card's memory and budget (`--help`).

**Measured, by hand,** on an L4 held to what an RTX 4080 SUPER with a desktop leaves (14.48 GiB free at start, a budget of 13.71
GiB: `--card 4080s`, which runs `--gpu-memory-utilization 0.88` as 0.622), with the plugin reading it as a GeForce Ada
card; vLLM 0.30.0, Open WebUI 0.11.4. 0.26.0rc2's load was also run on an RTX 4080 SUPER with a desktop (the
fixes of this section were not): the stand-in's numbers for 0.26.0rc2 match it, 11.83 GiB of weights and 1.35 GiB of KV
cache against 1.34 GiB there.

- **One user:** 21.25 tokens/s greedy and 21.00 with top-p (the median of five answers of 256 tokens each), the first
  token 53 ms after the request; 161.4 and 159.5 tokens/s in all for 8 users at once.
- **KV cache:** 1.83 GiB, 13,280 tokens, at the 13.71 GiB budget, with the weights at 11.39 GiB. 0.26.0rc2's weights took
  11.83 GiB and its KV cache 1.35 GiB (9,856 tokens) in the same run, 1.34 GiB (9,744 tokens) on the RTX 4080 SUPER.
- **bf16's weights leave no room:** its server did not start with the same flags (without `--quantization glyd`; the
  earlier record below). It ran out of memory loading the weights, 15.26 GiB into about 15.3 GiB free.
- **Loading** took 15 s for the weights (24 s with 0.26.0rc2) and 39-46 s to a running server. While it loads, the
  server takes most of the free GPU memory: 669 MiB were left free at the least, where 0.26.0rc2 took the card to 3 MiB
  and its log held 254 allocator warnings, "memory allocation failed with OOM", one for each time PyTorch's allocator
  could not get a block and freed its cache to try again. Now it holds none, at 14.48 GiB free and at the 21.7 GiB the L4
  has free with no hog (0.26.0rc2 held none there either: its KV cache 7.04 GiB, 51,264 tokens, against 7.51 GiB, 54,688).
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which gave 0.26.0rc2 1.74 GiB of KV cache, replaced its 254
  warnings with 73 of its own, "memory mapping failed with OOM". If PyTorch's warning shows in another setup it is
  harmless: it freed its cache and tried again, as it did each time here. No desktop ran in the test, so what one does
  in those seconds is not measured.
- **Open WebUI:** its model list showed Qwen/Qwen3-8B, a chat through its chat endpoint with the browser's request (its
  tools on) completed, and it passed the model's tool call, by both routes. In a browser (Open WebUI 0.11.4's page over
  a tunnel) a chat showed a collapsed "Thought for 7 seconds" and its answer, and a question about the time called the
  tool, "Explored get_current_timestamp", and answered with its result.

The packing's passes, the allocator and every run with its log:
[benchmarks/gpu/l4-quickstart-2026-09-30](../../benchmarks/gpu/l4-quickstart-2026-09-30). The hog that held the L4's
memory and the GeForce emulation of the first test: [benchmarks/gpu/l4-local-chat-2026-09-30](../../benchmarks/gpu/l4-local-chat-2026-09-30).

## What it does

- **At load.** Each Linear's bf16 weight is held on the meta device. As vLLM's layerwise loading completes a layer,
  its weights are packed on the GPU, so the load peaks at the packs plus one layer; the layer's bf16 is dropped at
  once, and PyTorch's unused blocks go back to the driver where they outweigh its free memory (vLLM loads with the
  allocator told not to split a block past 20 MiB, and on a card the weights nearly fill the unused blocks would pile up
  until an allocation failed and PyTorch warned, 254 times on a 16 GB card). The layout is Glyd's tiered one (10.80 bits a
  weight) or its 12-bit one (12.04). With `verify`, every pack is decoded and compared with its weights.
  - A layer whose checkpoint lacks a piece (a merged qkv's k, say, or an expert's up) is refused, naming the layer and
    the piece. vLLM's bf16 runs such a checkpoint with that piece's memory never written.
  - A model whose packs cannot fit the GPU's free memory is refused before it loads, with the numbers; running out of
    memory while packing says which layer and how much was packed.
- **A glyd save** loads as saved: its packs are the layers' buffers, with no bf16 at any point. Asked for the other
  layout, it is decoded and packed again. A save's packed LM head is decoded into the bf16 weight vLLM's LM head runs
  on. Saves of Qwen3 and Llama models are checked in vLLM; another family's save is refused, naming its bf16 source.
- **Each product** is one op, `glyd::vllm_linear`, which vLLM's torch.compile takes as one node and its CUDA graphs
  capture. The op takes the library's route for the step's tokens on this GPU: the fused kernels (the weights decoded
  in registers, straight into the tensor cores), or, for long prompts where cuBLAS is the faster, each matrix decoded
  and then cuBLAS.
- **A mixture of experts** (a model vLLM runs by its fused MoE layer): each layer's experts are packed as one matrix,
  the experts' stacked.
  - Their products are the library's grouped ones. Each token's choices are sorted by expert on the GPU; gate and up
    run with SiLU applied as their sums are written out; down runs with the router's weights, each token's rows
    added.
  - None of it syncs with the host, so vLLM's CUDA graphs capture it.
  - From 1,152 tokens a step (a prompt's), the layer decodes the experts its tokens are routed to and runs vLLM's own
    Triton MoE kernel on them instead, exact mode's way, the faster there: granite's layer on an L4 took 0.97x the
    grouped products' time at 1,152 tokens and 0.65x at 8,192. `GLYD_MOE_DECODE_MIN` moves the threshold (-1: never).
    It holds one layer's experts decoded in a scratch buffer
    ([benchmarks/gpu/l4-vllm-moe-routes-2026-09-30](../../benchmarks/gpu/l4-vllm-moe-routes-2026-09-30)).
  - These stay bf16, with a warning: experts with biases, activations other than SiLU, expert parallelism, and sizes
    off the packs' multiples.
- **Embeddings, the LM head, norms, attention and the KV cache stay vLLM's.**
- **The compile cache.** The options in effect, with a digest of the packs, go into vLLM's `additional_config`, which
  its compile cache is keyed by. Another layout, mode or checkpoint never finds another's compiled graph.

## Options

Each option is read from the first of these that sets it: `--additional-config '{"glyd": {...}}'`, the checkpoint's
`quantization_config` (or `--hf-overrides`'), or the environment (`GLYD_LAYOUT`, `GLYD_EXACT`, `GLYD_VERIFY`,
`GLYD_FRACTION`). Another key, a flag other than true or false (`1`, `true`, `yes`, `on`; `0`, `false`, `no`, `off`), or a
fraction outside 0 to 1 is refused.

| Option | Values | What it does |
| :--- | :--- | :--- |
| `layout` | `auto` (default), `mma`, `mma12` | `auto` takes `best_layout`'s choice for the GPU: the tiered layout on Ada (L4, L40S, RTX 40), for a mixture of experts on an A10 too, and wherever only it fits; else the 12-bit one (A10, A100, H100, GH200). A save loads in its own. |
| `exact` | `false` (default), `true` | Each product's matrix decoded whole, then the GEMM vLLM runs for bf16, so the logits are bf16's bit for bit (below). |
| `verify` | `false` (default), `true` | Every pack decoded at load and compared with its weights bit for bit. A save's packs by glyd.json's sha256, its other tensors too, and a save packed again in the other layout against the save. |
| `fraction` | `1` (default), a number from 0 to 1 | The share of the decoder layers packed; the rest stay vLLM's own bf16 ([below](#a-fraction-of-the-layers)). `0` is bf16, `1` every layer. A glyd save takes only `1`. |

## A fraction of the layers

A packed layer saves memory and costs a rebuild of its weights at every step; a layer left as it is saves nothing and
costs nothing extra. `fraction` says how many layers are packed: `0` packs none, which is vLLM's own bf16, and `1` every
layer, as before.

```bash
vllm serve Qwen/Qwen3-8B --quantization glyd --additional-config '{"glyd": {"fraction": 0.5}}'
GLYD_FRACTION=0.5 vllm serve Qwen/Qwen3-8B --quantization glyd
```

- **Which layers.** Layer i's Linears (qkv, o, gate and up, down) are packed where floor((i + 1) × fraction) >
  floor(i × fraction): floor(L × fraction) of a model's L layers, spread evenly over its depth (0.5 packs layers 1, 3,
  5 and so on; 0.25 packs 3, 7, 11). All of a layer's Linears go together, and a mixture of experts' layer's experts
  with them. The fraction counts as the decimal it is written in: 0.29 of 100 layers is 29. A layer is found by the
  first number in a module's name (`model.layers.12.mlp.down_proj`); a Linear outside the numbered layers is packed at
  `1` only. Embeddings and the LM head are as without the option.
- **The others** run vLLM's own methods, `UnquantizedLinearMethod` and `UnquantizedFusedMoEMethod`, as vLLM runs a model
  with no quantization. At `0` nothing is packed, Glyd's library is not loaded, and a mixture of experts' shared experts
  keep their side stream.
- **Exact mode** is unaffected: a layer left as it is gives bf16's bits, and a packed layer's decoded matrix goes through
  the same GEMM as before.
- **A glyd save** takes only `1`, its layers being packed on disk; a bf16 checkpoint takes any fraction. A fraction outside
  0 to 1 is refused.
- **The compile cache** is keyed by the fraction with the other options.

**Measured** on an L4 with Qwen3-8B (the tiered layout, `vllm bench serve` as above, servers warm, 64 prompts at 1 request
a second and 256 at once, against bf16 in the same session):

| | Weights | KV cache | Requests/s, at once | First token at 1 a second | Each token at 1 a second |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 27,024 tokens | 0.75 | 3,351 ms | 100.2 ms |
| fraction 0 | 15.27 GiB | 27,024 tokens | 0.76 | 3,539 ms | 100.8 ms |
| fraction 0.5 | 13.65 GiB | 37,904 tokens | 0.96 | 1,095 ms | 99.9 ms |
| fraction 1 | 11.83 GiB | 51,040 tokens | 1.04 | 772 ms | 95.6 ms |

- **Fraction 0 is bf16:** the same weights and KV cache, the same requests a second (0.755 against 0.754 at once), each
  token within 1% at 1 a second and at once. Its first token at 1 a second came 6% later, where bf16 is already queueing
  (0.66 requests a second completed of 1 offered).
- **Half the layers** gave 1.40x the KV cache and 1.27x the requests a second at once, against fraction 1's 1.89x and 1.38x.
  That is 71% of fraction 1's gain in requests a second, for 34% of its extra time per token at once (125.4 ms against
  bf16's 111.2 and fraction 1's 152.4); at 1 a second each token took as long as bf16's (99.9 against 100.2 ms).
- **Checked** (`check_vllm.py --quick --fraction 0.5`, Qwen3-8B, all 18 passed): every pack decoded to its weights bit for
  bit, every product within 4.2e-3 of its matrix decoded, the layers packed the rule's, exact eager and exact compiled in
  the deterministic mode bf16's bits. Fraction 0 in eager gave bf16 eager's tokens, logprobs and prompt_logprobs bit for
  bit, nothing packed, with the same KV cache. granite-3.1-3b-a800m-instruct (a mixture of experts, `--brief`, all 7
  passed): 16 of 32 layers packed, their experts too, the other 16 vLLM's own methods.
- **The GH200,** where fraction 1 served 0.88x bf16's requests a second with Qwen3-32B, is not measured with it yet.

Runs and logs: [benchmarks/gpu/l4-vllm-fraction-2026-09-30](../../benchmarks/gpu/l4-vllm-fraction-2026-09-30).

## Exact mode

The fused products sum in another order than cuBLAS's, so logits can differ from bf16's in their last bits. The same
happens between any two GEMM kernels, and within vLLM's own bf16, between CUDA graphs and eager.

With `exact`, each Linear's matrix is decoded into a scratch buffer and multiplied by the GEMM vLLM runs for bf16:
its `UnquantizedLinearMethod`'s, which is F.linear by default, a FlashInfer `--linear-backend`'s where one is asked for,
and the batch-invariant one under `VLLM_BATCH_INVARIANT`. A mixture of experts' layer decodes the experts its tokens are
routed to and runs vLLM's own bf16 MoE kernel.

- **Eager (`--enforce-eager`):** the logits are vLLM's bf16 eager's, bit for bit.
- **Compiled:** in inductor's deterministic mode, the logits are vLLM's compiled bf16's in that mode, bit for bit.
  That is the one mode in which compiled bf16 is itself the same from one run to the next:

  ```bash
  vllm serve MODEL --quantization glyd --additional-config '{"glyd": {"exact": true}}' \
    --compilation-config '{"inductor_compile_config": {"deterministic": true, "combo_kernels": true, "benchmark_combo_kernel": false}}'
  ```

  Otherwise inductor picks some of its kernels' variants by timing them on the GPU, among them the q and k norms and
  rotary embedding before attention. Compiled logits then vary from one process to the next, bf16's too, so exact
  refuses to start compiled without the deterministic mode. It never runs inexact without saying so.
- **Compiled, Linears with biases** (Qwen2.5's q, k and v, for one) are refused: in bf16's graph inductor adds a
  Linear's bias apart from its matmul, rounding before the add, where exact's product adds it in the GEMM. Eager, exact
  gives bf16's bits there too (Qwen2.5-1.5B-Instruct on an L4).
- **`VLLM_BATCH_INVARIANT`** asks for every product's bits not to depend on the batch. The fused kernels are chosen by
  the batch's tokens, so without exact it is refused; with exact the products are vLLM's batch-invariant GEMM on the
  decoded weights.
- **The deterministic mode on its own** makes compiled Glyd, fused too, the same from one run to the next. On an L4
  it cost nothing measurable: Qwen3-8B's tokens/s at 1, 8 and 32 sequences were within 1% of the default's, bf16's
  and Glyd's alike.
- **The cost of exact** is a decode per product.
- **For a mixture of experts,** exact needs vLLM's Triton kernel for bf16's experts, vLLM's pick on the L4. It is
  refused where vLLM picks another, which lays the weights out otherwise.

## Measured

`vllm bench serve`, bf16 against Glyd:

- the same `--gpu-memory-utilization 0.9`;
- servers started warm, on the compile cache their first start filled;
- the random dataset, 1,024 tokens in and 256 out;
- v0.25.1's library.

Low load is 1 request a second, 0.25 on the L4. Saturated is every request sent at once. A ratio or a percentage is
Glyd's against bf16's; for the times, less is better.

| GPU (Glyd's layout) | Model | KV cache | Requests/s, saturated | Low load: first token, each token | Saturated: first token, each token |
| :--- | :--- | ---: | ---: | :--- | :--- |
| L4 (tiered) | Qwen3-8B | 1.89x | 1.33x | +16%, -21% | -25%, +42% |
| A10 (12-bit) | Qwen3-8B | 1.73x | 1.31x | +16%, -21% | -25%, +30% |
| A100 40 GB (12-bit) | Qwen3-8B | 1.14x | 1.19x | +18%, -10% | -32%, -2% |
| A100 40 GB (12-bit) | Qwen3-14B | 1.77x | 1.65x | +18%, -13% | -57%, +4% |
| GH200 (12-bit) | Qwen3-8B | 1.04x | 0.92x | +4%, +1% | +4%, +9% |
| GH200 (12-bit) | Qwen3-32B | 1.66x | 0.88x | +28%, -6% | -52%, +82% |
| 2x RTX A6000, tensor parallel (tiered) | Qwen3-30B-A3B (a mixture of experts) | 1.67x | 0.87x | +30%, -7% | +34%, +28% |

- **Wins:**
  - 1.04-1.89x the KV cache, so more requests at once (1.67x for Qwen3-30B-A3B over two RTX A6000s).
  - 1.19-1.65x the requests a second saturated on the L4, A10 and A100.
  - At saturation, the first token 25-57% sooner there, and 52% sooner on the GH200 with Qwen3-32B.
  - At low load, each token 6-21% sooner, but for Qwen3-8B on the GH200 (1% slower).
- **Losses:**
  - At low load, the first token 4-28% later.
  - On the GH200 at saturation, 0.88-0.92x bf16's requests a second, with Qwen3-32B although bf16 ran short of KV
    cache there. Hopper's gap is not profiled yet.
  - At saturation, each token 30-42% slower on the L4 and A10, where each step carries more requests; within 4% on the
    A100. On the GH200, 9% slower with Qwen3-8B and 82% with Qwen3-32B.
  - Qwen3-30B-A3B over two RTX A6000s: at 1 request a second the same requests a second, each token 7% sooner; from 4
    a second 0.87-0.88x, each token 28-59% later, with the two modes running the same batches. The experts' grouped
    products at those batch sizes are the next work.
- **The L4's pair** ran back to back in one session. An earlier pair, Glyd's run within the hour after bf16's, gave
  1.39x saturated, +17% and -23% at low load, and -28% and +36% saturated (`l4-vllm-m2-2026-09-29`).

Every rate and percentile, and the logs: [L4](../../benchmarks/gpu/l4-vllm-m5-2026-09-30), [2x RTX A6000](../../benchmarks/gpu/vllm-m4-2xa6000-2026-09-30),
[A10](../../benchmarks/gpu/vllm-m3-a10-2026-09-30), [A100](../../benchmarks/gpu/vllm-m3-a100-40gb-2026-09-30),
[GH200](../../benchmarks/gpu/vllm-m3-gh200-2026-09-30).

**Against vLLM's own bf16** (`check_vllm.py`): Qwen3-1.7B, Qwen3-4B-Instruct-2507, Yi-1.5-6B-Chat (the Llama
architecture), granite-3.1-3b-a800m-instruct (a mixture of experts) and Qwen2.5-1.5B-Instruct (Linears with biases) on
an L4, and Qwen3-8B on an L4, A10, A100 and GH200:

- Every pack decodes to its weights bit for bit.
- Every product is within 6.2e-3 (relative) of the same product on its matrix decoded, with the same bits every run.
- The fused tokens are within vLLM's own bf16 noise, bf16 eager's against bf16 with CUDA graphs. Fed the 1,536-token
  continuation bf16 generated, Glyd ranks 0.988-0.997 of its tokens first, and bf16 eager 0.987-0.995.
  - Glyd's share is at or above bf16 eager's for 16 of the 18 model, GPU and layout pairs. The other two are 0.13 and
    0.20 points under: Yi 12-bit on the L4, and Qwen3-8B tiered on the A100.
  - Glyd's mean logprob difference from bf16's is at most 1.01x bf16 eager's.
- Exact mode gives bf16's bits: eager, and compiled in the deterministic mode (refused there for Qwen2.5's biases).
- Over two GPUs (tensor parallel, 2x RTX A6000): Qwen3-8B and granite, every check but one (below); Qwen3-30B-A3B,
  `--brief`, all 5, exact eager bit for bit. The one failed check: exact compiled was refused as it should be, but in
  the workers, where the check did not see Glyd's message.

## Speculative decoding

vLLM's speculation runs on Glyd's packs as on bf16's weights. Qwen3-8B on an L4, one user, greedy, the tiered layout,
in two mixes of five prompts: "edit" (fix, annotate, convert, rewrite or summarize a given text, code or data) and
"chat". The drafts are n-gram (5 tokens) and EAGLE-3 (`RedHatAI/Qwen3-8B-speculator.eagle3`, Apache-2.0, 3 tokens).

| Output tokens/s, one user | Edit | Chat | Draft tokens accepted (edit, chat) |
| :--- | ---: | ---: | :--- |
| bf16 | 16.6 | 16.7 | |
| bf16, n-gram | 27.1 | 16.7 | 40%, 12% |
| bf16, EAGLE-3 (`--gpu-memory-utilization 0.95 --max-num-batched-tokens 2048`) | 43.3 | 30.3 | 73%, 39% |
| Glyd | 21.2 | 21.5 | |
| Glyd, n-gram | 35.5 | 22.1 | 40%, 12% |
| Glyd, EAGLE-3 | 55.0 | 38.9 | 74%, 39% |

- **Speed.** Glyd with EAGLE-3 made 3.3x bf16's tokens a second on the edit mix, and 2.3x on chat. That is 1.27-1.28x
  bf16 with the same draft.
- **Memory.** bf16 with the draft did not fit the L4 at vLLM's defaults: no memory was left for the KV cache.
- **First token.** Glyd's came later on the edit mix's longer prompts (178 against 150 ms) and sooner on chat's (61
  against 82 ms).
- **Exact.** With speculation, exact eager gave bf16 eager's tokens, request for request: 10 of 10 with each draft.
- **Tokens against plain decoding.** A verify step multiplies up to 1 + k tokens at once, and GEMMs and attention round
  by that shape. So speculative decoding's greedy tokens differ from plain decoding's, bf16's too: eager bf16 parted in
  5 of 10 requests with n-gram and 7 of 10 with EAGLE-3. Under `VLLM_BATCH_INVARIANT=1` they are the same: bf16 with
  n-gram, 10 of 10, and Glyd exact with n-gram, 10 of 10.
- **The draft.** Under `--quantization glyd`, vLLM leaves an EAGLE-3 draft bf16. With `"quantization": "glyd"` in
  `--speculative-config` it is packed too, with the same speed and exact's same tokens.

Every run and its log: [benchmarks/gpu/l4-vllm-spec-2026-09-30](../../benchmarks/gpu/l4-vllm-spec-2026-09-30).
`spec_decode.py` runs one configuration; `spec_summary.py` makes the tables.

## Not supported yet

Each of these is refused at start, with a message saying why; none runs wrong. What vLLM's config tells is refused as
the engine builds it, before any worker starts, so over several GPUs too Glyd's message is the error the user sees:

- dual-batch overlap (`--enable-dbo`), whose two streams would share the GPU's done counters;
- LoRA;
- weight offloading (`--cpu-offload-gb`) and sleep mode;
- a glyd save over several GPUs, one with a mixture of experts' packs, or one of a family other than Qwen3's and
  Llama's; their bf16 checkpoints load;
- a glyd save with a `fraction` below 1: its layers are packed on disk, and its bf16 checkpoint takes a fraction;
- exact under torch.compile where a packed Linear has a bias (exact eager runs), and fused products under
  `VLLM_BATCH_INVARIANT`.

Tensor parallelism packs each rank's shard (measured over two RTX A6000s above).

## Files here

- `check_vllm.py [MODEL ...]`: the checks against vLLM's bf16.
  - Every pack against its weights.
  - Every layer's product against F.linear on its matrix decoded, and every mixture of experts' layer against its
    experts decoded.
  - Tokens and logprobs against bf16's own noise, and exact mode.
  - The compile cache: a graph for each layout and mode, each loaded again.
  - Flags: `--saves` (saves, as saved and in the other layout), `--quick`, `--brief`, `--fraction F` (Glyd's runs at
    that fraction, with which layers are packed checked against the rule, and fraction 0 in eager against bf16 eager's
    bits), `--tp N`, `--mp` (vLLM's workers in processes of their own, as over several GPUs, on one), `--out DIR`.
- `bench_serve.sh [MODEL]`, `bench_summary.py`: `vllm bench serve`, bf16 against Glyd at several rates. `WARM=1`
  notes the cold start and measures warm. The summary adds the GPU's clock and temperature. A mode `glyd@F` in `MODES`
  is Glyd at fraction F (`MODES="bf16 glyd@0.5 glyd@1"`), and the summary then adds each mode's weights, KV cache,
  requests a second, TTFT and TPOT against bf16's. `SERVE_ARGS` adds arguments to every `vllm serve`.
- `profile_steps.py [MODEL]`: a step's GPU time by kind of kernel (Glyd's, GEMMs, attention, the rest), bf16 against
  Glyd, at decode steps of B sequences and prompt steps of M tokens.
- `moe_routes.py [MODEL]`: a mixture of experts' layer by tokens a step, the grouped products against the routed
  experts decoded for vLLM's Triton kernel, and bf16's own layer (the threshold `GLYD_MOE_DECODE_MIN` routes by).
- `spec_decode.py`, `spec_summary.py`: one user's tokens/s, first token and speculation's acceptance for a configuration
  (bf16 or Glyd, n-gram or EAGLE-3, eager, exact), and the runs' tables and token-for-token comparisons.
- `bindings/python/test_vllm.py`: the plugin's logic that needs no GPU (options, the layers a fraction packs and the
  method each layer gets, a save's packs by vLLM's layer names, the pieces a checkpoint gave, what is refused, the entry
  point's version rule), with vLLM installed.

## vLLM's internals

The plugin is `bindings/python/glyd/gpu/vllm_plugin.py`, loaded by `vllm_entry.py`, the entry point, which checks
vLLM's version first. Beyond vLLM's plugin entry point and `register_quantization_config`, it relies on these of vLLM
0.30's internals:

- `QuantizationConfig` (`from_config`, `maybe_update_config`, `get_quant_method`).
- `LinearBase` and `LinearMethodBase`; `QKVParallelLinear` and `MergedColumnParallelLinear`, for their loaders' shard ids;
  `UnquantizedLinearMethod`, whose GEMM exact runs.
- `VocabParallelEmbedding` and `UnquantizedEmbeddingMethod`; `ModelWeightParameter` and `set_weight_attrs`.
- The layerwise online processing in `model_loader.reload.layerwise`: `initialize_online_processing`, and
  `get_layerwise_info`'s record of the weights loaded.
- The config through `get_current_vllm_config_or_none`:
  - `additional_config`, the compile cache's key;
  - the parallel, LoRA, offload, model, compilation and scheduler configs;
  - `vllm.envs`: `VLLM_BATCH_INVARIANT` and `VLLM_DISABLE_SHARED_EXPERTS_STREAM`.
- For a mixture of experts (without these, experts stay bf16, with a warning):
  - `RoutedExperts` and its `moe_config`;
  - `OnlineMoEMethodBase`, `FusedMoEQuantConfig.make` and `MoEActivation`;
  - `UnquantizedFusedMoEMethod` (its `unquantized_backend`, `_init_moe_kernel` and `moe_kernel.apply`'s arguments) with
    `UnquantizedMoeBackend`.

It copies none of vLLM's code. Like the rest of Glyd's GPU code it is under the Business Source License 1.1
(`gpu/LICENSE`).
