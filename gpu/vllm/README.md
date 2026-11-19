# Glyd in vLLM: `vllm serve MODEL --quantization glyd`

vLLM serves the model with its Linear layers held packed on the GPU, bit for bit, and a mixture of experts' experts
too. Glyd's kernels multiply them from there. vLLM sizes its KV cache after the weights load, so the memory the packs
save becomes KV cache: more requests at once on the same GPU.

```bash
pip install "glyd[vllm]"                                  # vLLM 0.30, and Glyd with its plugin
vllm serve Qwen/Qwen3.5-9B --quantization glyd            # a bf16 checkpoint, packed as it loads
vllm serve ./SAVE --quantization glyd                     # a glyd save (glyd pack, glyd.save_pretrained), as saved: not yet of a 2026 model
vllm serve Qwen/Qwen3.5-9B --quantization glyd --additional-config '{"glyd": {"layout": "mma12"}}'
```

vLLM's own settings need not fit the card: Qwen3.5-9B in bf16 did not start on an RTX 4090 (24 GB), out of memory at
start even with the context capped at 4,096 tokens. Start with `glyd run` ([below](#local-chat-like-ollama)), which works the
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
glyd run Qwen/Qwen3.5-9B
```

The script installs [uv](https://docs.astral.sh/uv/) if it is missing, then Glyd with vLLM 0.30 and PyTorch as one isolated
tool, on a Python 3.12 that uv fetches for it (about 8 GB of disk; no sudo, no virtual environment, no pip), and ends with
`glyd doctor`. `glyd run` downloads the model (Qwen3.5-9B is 19.3 GB the first time), checks that this machine can run it,
starts vLLM with settings it works out from the GPU, and opens a chat: type in the terminal (`/bye` leaves, `/clear`
starts over, `/think` turns the model's thinking on and off), or open http://localhost:8000.

What the script does to your machine, all of it under your home directory:

- **uv**, where there is none: uv's own installer at the version the script names (0.12.21), checked against that installer's sha256
  (which lists the sha256 of each file the installer downloads), and told to edit no shell startup file; into `~/.local/bin`.
- **Glyd**, at the release the script names, with its packages at the versions the acceptance run installed (the list of about 200 at the
  end of the script; `GLYD_CONSTRAINTS=none` in front of `sh` resolves them fresh, where one has been withdrawn from PyPI): in uv's tool
  directory (`~/.local/share/uv/tools/glyd`) and uv's cache (`~/.cache/uv`), with a link `~/.local/bin/glyd`.
- **Your PATH**, where `~/.local/bin` is not on it: `uv tool update-shell` adds a line to your shell's startup file. The script says so
  before it does, and uv names the file.
- Nothing else, and nothing of yours replaced. Where `~/.local/bin/glyd` is a program uv did not put there (the compression program, a pip
  install of glyd, one of your own), the script stops before it installs Glyd and says so, with the way to keep both: the tool in another
  folder, which the script then puts first on your PATH (`curl -LsSf https://getglyd.com/install.sh | UV_TOOL_BIN_DIR=$HOME/.glyd/bin sh`).

A `glyd` that comes before `~/.local/bin` on your PATH (Homebrew's, cargo's, a release tarball's: the compression program) is what typing
`glyd run` reaches. The script says so, with the line to add to your shell's startup file (`export PATH="$HOME/.local/bin:$PATH"`), and
`glyd doctor` does too (run it as `~/.local/bin/glyd doctor`). The Rust `glyd` of the release after 0.26.0rc3 passes `run`, `serve`, `doctor`
and `login` to the tool itself, found on PATH or in `~/.local/bin`; an earlier one takes `run` for a file name.

**On a Mac, or a Linux machine with no NVIDIA GPU,** `glyd run` has nothing to run on. The script says so (`glyd run needs Linux with an
NVIDIA GPU`) and installs what works there, the compression program (`glyd FILE -o OUT`, `glyd pack`): the release's tarball, checked
against its sha256, in `~/.local/share/glyd/cli`, with `glyd`, `glyd-store` and `glyd-gpu` linked from `~/.local/bin`. (Homebrew's
`brew install surya-koritala/glyd/glyd` is the other way.) `glyd run` there says the same, and gives no advice about drivers to a Mac.

**Needs:** Linux on x86_64 (the installer also takes aarch64, which this flow was not run on); an NVIDIA GPU of the
Ampere generation or newer (RTX 30 and 40 series, A10, A100, L4, H100 and later) and its driver, 580 or newer (the CUDA 13
PyTorch that vLLM 0.30 installs); curl; and disk for the packages (the script asks for 10 GB free in your home directory, and
stops with a plain message under that) and the model. It needs no CUDA toolkit and no sudo.
vLLM's Triton builds small launchers with a C compiler when the server starts: where the machine has no gcc or clang, the
installer adds ziglang, a compiler from PyPI, and `glyd run` hands it to vLLM. `glyd doctor` checks each of these and says
what to do about a line that fails.

`glyd run`, in order, stopping with a plain message and what to do where it must:

1. **Checks** the GPU and its driver (against the CUDA this PyTorch was built for; with several GPUs, the one Glyd can use: an Ampere
   card is taken over an older one beside it), the compiler, vLLM's version, the model (it is on the Hugging Face Hub, bf16 (a float16
   checkpoint is refused before its download, and `-- --dtype bfloat16` asks vLLM to convert it), and not gated without your token),
   whether the model with Glyd fits the memory free
   now (otherwise: how much it needs, what holds the GPU's memory, and the largest model of its family that fits, in
   the words "MODEL needs about N GB of GPU memory with Glyd (...); your GPU has M GB free" and "Or try MODEL, which
   needs about K GB"), the disk
   the download needs, and the port.
2. **Downloads** the model, with one progress bar. A gated model: accept its licence on its Hugging Face page,
   then `glyd login` (the `hf` command is not on the PATH of a tool install).
3. **Chooses** the settings below and prints them on one line.
4. **Starts** vLLM with its output in a log file, and shows what it is doing.
5. **Chats**, in the terminal and at the printed address. A conversation longer than the model's window is said so in both:
   "This conversation is longer than the model's window (10,240 tokens). Start a new chat with /clear."

`glyd run MODEL --prompt "Say hello"` prints one answer on stdout (the thinking and notices go to stderr) and exits; a
prompt on stdin does the same. `--context N` sets the context. Any vLLM flag after a lone `--` is passed on and wins over
what was chosen: `glyd run MODEL -- --max-model-len 4096`. On vLLM, a server that `glyd serve` started for the same model on the port is
used by `glyd run` instead of a second one (`glyd serve` of it again says "Nothing to start").

Glyd's own engine, `glyd-engine` (a program that ships in the glyd-gpu package, beside the GPU library), runs a model it supports for `glyd run` only on a GPU where it is measured at least as fast as vLLM and with
less memory, and only where it will certainly run the model (no 2026 model yet: Qwen3.5, Qwen3.8 and Gemma 4 run on vLLM); every other model, and every other GPU, runs on vLLM as described here. It is measured on no GPU yet, so until it is every model runs on vLLM, as in v0.28.
The engine is asked before anything is downloaded: it checks the GPU a run would use (the one with the most free memory on a computer with several; `CUDA_VISIBLE_DEVICES` names the ones to choose from),
its driver (580 or newer), the model, the memory and the first run's disks. `glyd serve` runs vLLM, which serves more users at once than the engine's 8, and so do `glyd run` where `GLYD_KV`, `GLYD_LAYOUT`,
`GLYD_EXACT`, `GLYD_VERIFY` or `GLYD_FRACTION` is set (vLLM reads them) and the flags after a lone `--` (vLLM's). `GLYD_ENGINE=1` asks for the engine for `glyd serve` (its checks still apply), `GLYD_ENGINE=0`
keeps vLLM for any model, and `glyd-engine run MODEL` and `glyd-engine serve MODEL` run it directly. Where the engine does not get up (it refuses, crashes, is killed or goes silent for 15 minutes), the command goes
on with vLLM, with one line saying why; once it is up, a crash is reported and the model is not started again. The first line `glyd run` or `glyd serve` prints says which one runs the model (`glyd serve: the engine
serves up to 8 users at once and vLLM more (GLYD_ENGINE=1 runs the engine); using vLLM.`, or `glyd run: MODEL on glyd-engine`, or why vLLM does: `glyd run: Qwen3.5-9B is a Qwen3_5ForConditionalGeneration model, and this
engine runs Qwen3 models so far; using vLLM.`). The engine chooses the context from the GPU's memory (8,192 tokens where it has room), decodes up
to 4 users' replies together (`--context N` and `--users N`, 1 to 8, ask for others) and serves the OpenAI API (chat and plain completions, streaming, a key, the thinking returned apart) and the chat page. Its API takes tool calls (`tools`, `tool_choice` of auto, none, a named function or required, `parallel_tool_calls`; the model's `<tool_call>` blocks read as `tool_calls`, whole and streamed), `logprobs` and `top_logprobs` (up to 20) and a plain completion's `logprobs`, `n` up to 8, `logit_bias`, the sampling and stopping fields (`stop_token_ids`, `min_tokens`, `include_stop_str_in_output`, `allowed_token_ids`, `skip_special_tokens`), `echo` and a list of prompts for a plain completion. A request for a `response_format` other than text (a JSON object or a schema) or for another field only vLLM does (`guided_*`, `structured_outputs`, `bad_words`, `prompt_logprobs`, ...) is refused with a 400 that says so (`GLYD_ENGINE=0` serves those from vLLM).

### The settings

| Setting | `glyd run` and `glyd serve` choose | Why |
| :--- | :--- | :--- |
| `--gpu-memory-utilization` | the memory free now, less 0.55 GiB for the server's CUDA context (outside vLLM's share) and 0.4 GiB left for a desktop, over the card's total as CUDA reports it (not nvidia-smi's), rounded down to 1%; at most 0.92. `glyd run` (one chat) takes only what two windows of KV cache need, `glyd serve` the whole share | vLLM's share is of that total: an RTX 4080 SUPER's is 15.57 GiB, nvidia-smi's 16,376 MiB is 15.99 |
| `--max-model-len` | the model's own length, or the most the KV cache holds at that share (a multiple of 1,024). Under 4,096 the model does not fit | the KV cache is what is left after the weights |
| compiled or eager (`--enforce-eager`) | compiled (vLLM's own default: torch.compile and CUDA graphs) wherever the chat keeps 8,192 tokens (or the model's own length, or the context asked for); eager only where that makes the chat longer. `-- --enforce-eager` and `-- --no-enforce-eager` are taken as they are | Qwen3.5-9B, one user: 77.3 tokens/s compiled on an RTX 4090 against 26.6 eager, 115.9 against 28.7 on an RTX 5090; the server up in 135 s compiled against 72 s eager (RTX 5090); compiled needs about 1.75 GiB more beyond the weights |
| the layout (`GLYD_LAYOUT`) | the plugin's own choice for the GPU (`mma` on Ada and wherever only it fits; else `mma12`), `mma` also where `mma12` leaves less room than an 8,192-token chat | the layout's size is counted into the memory above, so the plugin packs what was counted |
| the sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) | PyTorch's | FlashInfer's compiles with nvcc at the first request that samples, which stopped a server on a machine with no CUDA toolkit |
| tool calls and thinking | by family: Qwen3.5 and Qwen3.8 `qwen3_coder` and `qwen3` (the flags their model cards give vLLM). Gemma 4, and a family the table does not know, chats without tool calls | the names are vLLM 0.30's |
| the allocator | PyTorch's default | `expandable_segments:True` gave 24 allocator warnings in a load at a 16 GB budget, and the default none |
| telemetry, the address | `VLLM_NO_USAGE_STATS=1`; the server listens on 127.0.0.1 and answers this computer's programs and its own page only ([below](#who-can-reach-the-server)) | `glyd serve --host 0.0.0.0` opens it to the network, with no key unless `VLLM_API_KEY=KEY` is set |

The one line printed before loading is these, for example, Qwen3.5-9B on an RTX 4090 (the measured run's line, with the
parsers v0.29.0 adds): `Settings: 189,440-token context
(the most that fits), compiled mode (CUDA graphs), 92% of GPU memory (23.2 GB), tool calls (qwen3_coder), thinking shown
apart (qwen3).` The model's weights with Glyd are counted from its config and file sizes before anything downloads.

### Who can reach the server

`glyd run` and `glyd serve` listen on 127.0.0.1, which is this computer's own address, and answer this computer's programs and the page
they serve. A web page you open in your browser can still send requests to 127.0.0.1: a script from any site, where the server lets it
(vLLM's default allows every origin), or a page whose name an attacker points at 127.0.0.1, which is the server's own origin to the browser
and needs no permission at all (DNS rebinding). So the server checks each request, as the outermost layer of vLLM's app: a Host that is not
`localhost`, `127.0.0.1` or `[::1]` is refused with 421, an Origin that is not the server's own (`http://` and its Host) with 403, a preflight
and a websocket's handshake included, and vLLM's own CORS is limited to the server's addresses. `curl`, the OpenAI libraries and Open WebUI's
server send no Origin and an address as their Host, and are answered; the chat page is the server's own origin. `acceptance.sh` checks this
against a live server. A web app of your own on another origin (a dev server on localhost:5173) is refused too; to let it in, name the
address yourself, which turns these checks off and leaves the origins to vLLM: `glyd serve MODEL --host 127.0.0.1 -- --allowed-origins
'["http://localhost:5173"]'` (`glyd` says so where it starts a server on an address you named, a loopback one too).

Glyd's own engine, where it serves (`GLYD_ENGINE=1` or `glyd-engine serve`), makes the same checks, with the same answers (421 for the Host, 403 for the Origin and for a preflight from another origin, a JSON error body of type `Forbidden`),
and answers its own page's origin (a preflight from it too); an address you chose with `--host` turns them off there too, and every origin is answered then.

`glyd serve --host ADDRESS` (or `-- --host`) is the way onto the network, and the checks are off for an address you chose: anyone who can
reach this computer can send the model prompts. Give it a key, in the environment:

<!-- acceptance: serve-key -->
```bash
VLLM_API_KEY=YOUR_KEY glyd serve Qwen/Qwen3.5-9B --host 0.0.0.0
```

The environment, not `-- --api-key KEY`: a flag's value is in `ps` and in the log. (`-- --api-key KEY` still works: `glyd` moves the key to
the environment, says so, and keeps it out of the server's command line and of the log, but your own `glyd serve` command line shows it.
One key: vLLM takes several after `--api-key`, `VLLM_API_KEY` holds one, and `glyd` refuses the rest; `-- --hf-token TOKEN` is moved to
`HF_TOKEN` the same way.) The
key guards `/v1` only: the chat page, `/health`, `/metrics`, `/version`, `/tokenize`, `/detokenize` and the API's `/docs` stay open without it,
and the traffic is plain HTTP, so a key crosses the network readable: use a VPN or an SSH tunnel, or `-- --ssl-keyfile FILE --ssl-certfile
FILE`. `glyd serve` says all of this where it starts a server on another address.

### `glyd serve`, `glyd doctor` and Open WebUI

`glyd serve MODEL` does the same checks and chooses the same settings, and leaves the server up for other programs: the
OpenAI API at http://localhost:8000/v1 and the chat page at http://localhost:8000. `glyd doctor` prints the GPU, driver,
CUDA, free memory, compilers and versions, and which of Qwen3.5-4B, 9B and 27B fit, at Glyd's sizes.

Open WebUI is a chat page with accounts, history and tools, which talks to that API. It uses no GPU memory, and it is large:
the Docker image is 6.5 GB, and `uvx` puts 7.1 GB into uv's cache (it brings its own PyTorch). Pinned to 0.11.4, the version
tested. Without Docker:

<!-- acceptance: webui-uvx -->
```bash
DATA_DIR="$HOME/.open-webui" OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=none ENABLE_PERSISTENT_CONFIG=False \
  CORS_ALLOW_ORIGIN='http://localhost:3000;http://127.0.0.1:3000' \
  uvx --python 3.11 open-webui@0.11.4 serve --host 127.0.0.1 --port 3000
```

With Docker Engine on Linux, whose containers can share the host's network (`--network=host` is Linux's: Docker Desktop on
macOS and Windows does not give the container the host's 127.0.0.1):

<!-- acceptance: webui-docker -->
```bash
docker run -d --name open-webui --network=host -e PORT=3000 -e HOST=127.0.0.1 \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 -e OPENAI_API_KEY=none \
  -e 'CORS_ALLOW_ORIGIN=http://localhost:3000;http://127.0.0.1:3000' \
  -e ENABLE_PERSISTENT_CONFIG=False -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:v0.11.4
```

With Docker Desktop, or any Docker without host networking, the container reaches the host by name, and the server has to
listen on every interface, which opens it to the network: give it a key, as [above](#who-can-reach-the-server)
(`VLLM_API_KEY=YOUR_KEY glyd serve Qwen/Qwen3.5-9B --host 0.0.0.0`), and give Open WebUI the same one:

<!-- acceptance: webui-bridge -->
```bash
docker run -d --name open-webui -p 127.0.0.1:3000:8080 --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=YOUR_KEY \
  -e 'CORS_ALLOW_ORIGIN=http://localhost:3000;http://127.0.0.1:3000' \
  -e ENABLE_PERSISTENT_CONFIG=False -v open-webui:/app/backend/data ghcr.io/open-webui/open-webui:v0.11.4
```

Open http://localhost:3000 and make the first account at once: it is the administrator (until it exists, whoever reaches the
port first can make it), and Open WebUI asks for a login after that. The account and the chats are kept in the `open-webui`
volume (Docker) or in `~/.open-webui` (`uvx`). The third command was run on Docker Engine on Linux without host networking,
which is the case `host-gateway` makes work there; Docker Desktop is the case it is written for and was not run. Notes:

- `CORS_ALLOW_ORIGIN` is set to the page's own two addresses: Open WebUI's default, `*`, answers a script on any page you open with
  `access-control-allow-origin` naming that page, credentials allowed (measured on 0.11.4); with it set, another site's request gets no
  such header and its preflight is refused with 400 (`acceptance.sh` checks both).
- **If you want no login:** add `WEBUI_AUTH=False` (`-e WEBUI_AUTH=False` for Docker). Everyone who reaches port 3000 is then signed
  in as the administrator, whose tools and functions run Python code (as you, for `uvx`). `CORS_ALLOW_ORIGIN` stops another site's
  script (measured on 0.11.4: with the default `*`, a sign-in from `Origin: http://evil.example` came back with the administrator's
  token), but not **DNS rebinding**: Open WebUI does not check the Host header, so a web page whose name is pointed at 127.0.0.1 is
  Open WebUI's own origin to your browser and can use the administrator's session. Use it only on a computer that is yours alone.
- `ENABLE_PERSISTENT_CONFIG=False` makes these variables the settings on every start. Open WebUI otherwise keeps the
  connection it first started with (the default, OpenAI's) in its data directory, takes the variables only on a first
  start, and shows "No models available" on a later one that has them.
- The `uvx` route needs no Docker. `docker run` needs a user in the `docker` group (or `sudo`), which a Docker Engine install
  does not give you by itself. Run `uvx` from a directory you can write to: Open WebUI keeps a secret key file in it.
- Its chats offer the model Open WebUI's built-in tools. `glyd serve` starts vLLM with a tool-call parser (Qwen3.5: `qwen3_coder`): without one every chat is answered with `"auto" tool choice requires
  --enable-auto-tool-choice and --tool-call-parser to be set`. Glyd's engine, where `GLYD_ENGINE=1` or `glyd-engine serve` runs it, answers a request with `tools` too (it reads its models' tool calls itself). The thinking arrives as a field of its own (`reasoning`; `--reasoning-parser qwen3` on vLLM),
  which Open WebUI shows as a collapsed "Thought".

### Measured

Qwen3.5-9B, vLLM 0.30.0, `glyd run` and `glyd serve` with the settings they chose (CHANGELOG.md, v0.29.0):

| GPU | `glyd` chose | vLLM logged | One user, tokens/s |
| :--- | :--- | :--- | ---: |
| RTX 4090 (24 GB) | a 189,440-token context (the most that fits), compiled, 92% of GPU memory (23.2 GB) | weights 12.68 GiB | 77.3 (26.6 eager) |
| RTX 5090 (32 GB) | compiled | weights 13.89 GiB | 115.9 (28.7 eager) |

bf16's weights are 17.66 GiB; on the RTX 4090 bf16 did not start (out of memory at start, even with the context capped
at 4,096 tokens). One user: a prompt of about 1,000 tokens and 256 out, the median of three. A first start compiles: the server
was up in 135 s on the RTX 5090 (72 s eager).

### Update, remove, logs

Run the install line again to update: it installs the release the script was written for, at the package versions the acceptance
run installed, and does nothing where that is what is there. `uv tool uninstall glyd` removes the tool and its link, and leaves
what is not the tool's: the models in the Hugging Face cache (`~/.cache/huggingface`, or where `HF_HOME` says), a run's log (the
last ten are kept) and the compiler wrapper, in `~/.local/state/glyd`, uv itself (`~/.local/bin/uv`, `uv self uninstall`) and its cache
(`~/.cache/uv`, `uv cache clean`), and the line in your shell's startup file if the script added one. Delete those by hand. The `glyd` command
is the compression program's too: the Rust program is not in the wheel, so `glyd FILE` and `glyd pack` work where that program is on the PATH
(the script's tarball on a machine with no GPU, Homebrew, `cargo install --git https://github.com/surya-koritala/Glyd glyd glyd-store glyd-gpu`,
the release tarball), and `glyd` says where to get it where it is not.

`bash gpu/vllm/acceptance.sh` runs this section from nothing, in a container with no CUDA toolkit and no compiler: the two commands
above, the chat page, the API, a conversation longer than the window and Open WebUI, failing on a traceback, an allocator warning or a
missing answer; and what a user meets around them: a program of the user's own where uv puts its link, an update, a `glyd` ahead on
PATH, `nohup` and SIGHUP, another site's script and a rebinding Host refused, a server with a key, Ctrl-C while the model loads, the
engine killed under a chat, `uv tool uninstall glyd`. `--card 4080s` leaves the server a 16 GB card's memory, `--card 8gb` an 8 GB
card's, where it expects the refusal and runs the model suggested; `--flow cli` runs the installer on a machine with no GPU (`--help`).

### Advanced: `vllm serve` by hand

`glyd run` is `vllm serve` with its flags decided for you, so every vLLM flag works with it, and the plugin works with
`vllm serve` as it is. By hand, for Qwen3.5-9B on an RTX 4090 (24 GB) with the settings `glyd run` chose there, and
what each flag is for:

**Needs**, none of it a CUDA toolkit: an NVIDIA driver 580 or newer (the CUDA 13 build of PyTorch that vLLM 0.30
installs; the tests ran on 595), [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12, with its headers), a C
compiler (`sudo apt install build-essential`: Triton builds its launchers with one, and without it vLLM stops at start
with "Failed to find C compiler"), and disk: about 8 GB for the packages and 19.3 GB for the model.

<!-- acceptance: setup -->
```bash
uv venv --python 3.12 ~/glyd-env && source ~/glyd-env/bin/activate
uv pip install "glyd[vllm]"
```

<!-- acceptance: serve -->
```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3.5-9B --quantization glyd \
  --max-model-len 189440 --gpu-memory-utilization 0.92 --host 127.0.0.1 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

The server answers on http://localhost:8000 once it logs `Application startup complete`.

- **`VLLM_USE_FLASHINFER_SAMPLER=0`**: vLLM 0.30 samples top-k and top-p with FlashInfer, which builds that kernel
  with nvcc on the first request that samples, and vLLM's warmup makes that request at start: with no CUDA toolkit the
  server stopped there, after loading the weights, with "Could not find nvcc and default cuda_home='/usr/local/cuda'
  doesn't exist". This makes vLLM sample with PyTorch and Triton instead. Glyd logs one warning at start where it finds
  no nvcc and the sampler on, naming both ways out.
- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder`**: Open WebUI offers the model its built-in tools in
  every chat. The request carries `tools`, which vLLM takes as `tool_choice` auto, and without these flags it answers
  every chat with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`.
  `qwen3_coder` is the parser Qwen3.5's model card gives vLLM.
- **`--reasoning-parser qwen3`**: Qwen3.5 thinks before it answers. With the parser the thinking is a field of its own
  (`reasoning`, which Open WebUI shows as a collapsed Thought), and the answer's `content` has no `<think>` in it; without
  it the thinking comes in the content, tags and all.
- **`--gpu-memory-utilization 0.92`** is vLLM's share of the card's total memory, for the weights, the KV cache and the
  working memory, of the total CUDA reports (`python -c "import torch; print(torch.cuda.mem_get_info()[1] / 2**30)"`).
  The server's CUDA context and a desktop live outside it. If something else holds more, vLLM stops at start with
  "Free memory on device ... is less than desired GPU memory utilization". Close that program, or lower the number.
- **No `--enforce-eager`**: compiled, vLLM's default (torch.compile and CUDA graphs): 77.3 tokens a second for one
  user here against 26.6 eager. A first start compiles, and a later one loads the compile cache.
- **`--max-model-len 189440`** is the longest chat, in tokens: what Glyd's settings chose at that share (vLLM's cache
  held 197,225). A longer chat
  is refused with "maximum context length is 189440 tokens".
- **`--host 127.0.0.1`** keeps the server on this machine. Without it vLLM listens on every interface, with no key.

Open WebUI, as above, on the server's port 8000. To test the server alone:

```bash
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3.5-9B",
  "messages": [{"role": "user", "content": "What is lossless compression? Answer in one sentence."}],
  "max_tokens": 100}'
```

Qwen3.5 thinks before it answers: the thinking comes back in `reasoning`, apart from the answer's `content`.

`bash gpu/vllm/acceptance.sh --flow vllm` runs this part from nothing, on a machine with no CUDA toolkit: a clean environment,
the install, the serve command above as written, a chat through the OpenAI API (two streamed turns and a tool call) and
through Open WebUI both ways, failing on a traceback, an allocator warning or a missing answer. Where the GPU is bigger
than a card it stands in for, `--card 4080s` leaves the server that card's memory and budget (`--help`); the block above
is an RTX 4090's (24 GB), so a smaller card needs `--command` with a serve command that fits it.

**Measured** on an RTX 4090 with these settings but the two parsers (added after the run), vLLM 0.30.0: the weights
12.68 GiB against bf16's 17.66, whose server did not start (out of memory at start, even at a 4,096-token context); the
KV cache held 197,225 tokens; one user 77.3 tokens a second (CHANGELOG.md, v0.29.0).

## What it does

- **At load.** Each Linear's weight is packed on the GPU as vLLM loads it, layer by layer, so the load peaks at the
  packs plus one layer, and the layer's bf16 weight is dropped at once. The layout is `mma` or `mma12`. With
  `verify`, every pack is unpacked and compared with its weights.
  - A layer whose checkpoint lacks a piece (a merged qkv's k, say, or an expert's up) is refused, naming the layer and
    the piece. vLLM's bf16 runs such a checkpoint with that piece's memory never written.
  - A model whose packs cannot fit the GPU's free memory is refused before it loads, with the numbers; running out of
    memory while packing says which layer and how much was packed.
- **A glyd save** loads as saved, with no bf16 at any point. Asked for the other layout, it is decoded and packed
  again. A save's packed LM head loads as saved; its embedding is packed as the load of a bf16 checkpoint packs it. Saves of the
  Qwen3ForCausalLM and LlamaForCausalLM architectures are checked in vLLM (no 2026 model's yet); another family's save is
  refused, naming its bf16 source, which loads.
- **Each product** runs Glyd's kernels, inside vLLM's torch.compile and CUDA graphs.
- **A mixture of experts** (a model vLLM runs by its fused MoE layer): each layer's experts are packed, and Glyd's
  kernels multiply them.
  - From 1,152 tokens a step (a prompt's), the layer takes another path, which is faster there.
    `GLYD_MOE_DECODE_MIN` moves the threshold (-1: never). It holds one layer's experts decoded in a scratch buffer.
  - SiLU experts are packed (a SwiGLU with a clamp, Hy4's and GLM-5.3-Flash's, in `exact` mode only). The other activations
    vLLM's MoE layers run (GELU (tanh), a clamped SwiGLU, SwiGLU-OAI, swiglustep, ReLU squared without a gate) are built, each
    as vLLM computes it, and stay bf16 until their products are measured.
  - These stay bf16, with a warning: experts with biases, the activations above until measured, an activation vLLM's own op
    does not run on its own, expert parallelism, and sizes off the packs' multiples.
- **The embedding and the LM head are packed too**, in the same layout, so the whole model is about a third smaller in the smallest layout
  (a quarter in the 12-bit one) and the freed memory is KV cache. Every lookup returns the table's exact bf16 row, and the LM head's
  logits are as close to bf16's as any other product's. A tied model has one packed table for both.
  - They stay as vLLM runs them, with a line in the log saying why, under a `fraction` below 1, speculative decoding, `exact` (below), a
    `head_dtype` other than bf16, an embedding class that is not a plain lookup, and a hidden size that is not a multiple of 64.
  - Tables vLLM builds without a quantization config (Qwen3.5's, Nemotron-H's, LFM2's, Jamba's ... embedding; some LM heads;
    LongCat's n-gram table) are packed by the same rules once the model has loaded; the log names the tables packed.
- **Linears vLLM builds without a quantization config:** Jamba's Mamba mixers' are packed once the model has loaded. A vision or
  audio tower's stay bf16 for now (an image's prefill through them is to be measured: `check_vllm.py --towers`). A Linear's rows
  need not be a multiple of 64. Qwen3.8-Flash-Next's n-gram table, which vLLM refuses under any quantization config but FP8's,
  stays bf16 instead of the model being refused.
- **Norms and attention stay vLLM's**, and so does the KV cache unless `kv` holds it in 12 bits (`auto`, the default, does on an A100, an L4, an H100 and a GH200, for head size 128 with full attention only:
  Qwen3.5, Qwen3.8 and Gemma 4 keep vLLM's cache; [below](#the-kv-cache-in-12-bits)).
- **The compile cache** is keyed by the options in effect and the checkpoint: another layout, mode or checkpoint never
  finds another's compiled graph.

## A quantized checkpoint

`vllm serve nvidia/Gemma-4-31B-IT-NVFP4 --quantization glyd` (FP8, NVFP4, MXFP4, INT4 from GPTQ, AWQ or compressed-tensors, INT8): the
checkpoint's quantized layers run exactly as vLLM runs them without Glyd, with vLLM's own method for its format, and Glyd packs the
16-bit parts the checkpoint keeps (Linears it leaves bf16, such as Gemma-4-31B-IT-NVFP4's attention, the embedding, the LM head, a vision
tower's Linears where vLLM gives the tower the quantization config), with the same rules and options as for a bf16 model (`layout`,
`exact`, `verify`, `fraction`). `exact` gives the logits of vLLM's own run of the same checkpoint, bit for bit.

- The KV cache stays vLLM's (`kv: lossless` is refused).
- bf16 only: an fp16 checkpoint is refused, as for a bf16 one. With `--dtype bfloat16` vLLM rounds its fp16 parts to bf16 before Glyd
  packs them (with or without Glyd), so such a checkpoint's own bits are not what is kept.
- `layout: auto` is the tiered layout for a quantized checkpoint on every GPU; no fit estimate is made for one (which Linears it keeps
  16-bit is not in its config).
- A speculative draft in the checkpoint's own files (MTP) gets the target's quantization method, as vLLM gives it; not measured on a GPU.
- `glyd run` and `glyd serve` still refuse a quantized checkpoint; this is `vllm serve --quantization glyd` only.
- What vLLM builds without the quantization config is packed as for a bf16 model: Qwen3.5's embedding once the model has loaded; a
  vision tower vLLM gives no config (Gemma 4's, but in a compressed-tensors checkpoint) stays as vLLM runs it.
- Formats vLLM serves by other methods (AMD Quark, as amd/MiniMax-M3-MXFP4; torchao, experts_int8 and the rest) are refused, with why.

Measured on an L40S (`benchmarks/gpu/quant-16bit-parts-2026-10-06/`, each row's run and commit there), stock vLLM against Glyd on the
same checkpoint: the weights 2.4% to 19.0% smaller in the default mode (Gemma-4-31B-IT-NVFP4: 31.18 -> 25.25 GiB, twice the KV cache;
Qwen3.8-27B-FP8: 28.90 -> 27.13 GiB); exact mode bit for bit at the checkpoint's own bits on 5 of 8; the fp16 AWQ checkpoint equal to stock
at `--dtype bfloat16` (its fp16 parts rounded by vLLM: not lossless); no bit-for-bit result on the two whose experts run on vLLM's Marlin
MoE kernel, where stock vLLM differs from itself (their packs verified only). Default mode: 1.03-1.24x stock's tokens a second at one user
(parity within noise on Qwen3.6-35B-A3B-NVFP4) and 1.01-1.05x at 32 (Gemma-4-31B-IT-NVFP4's 32-user cell aside: stock queues for want of
KV cache). On an RTX 5090: Gemma-4-31B-IT-NVFP4, which stock vLLM cannot load on its 32 GB, serves with the tiered layout at
`--gpu-memory-utilization 0.95` and a 4,096-token context (25.25 GiB of weights, 5,559 tokens of KV cache, 42 tokens a second for one
user); with the 12-bit layout (the default for a quantized checkpoint then) gemma-4-12B QAT and Qwen3.6-35B-A3B-NVFP4 were 0.5-0.6% smaller
and 0.98-1.00x stock's tokens a second. NVFP4 weights and activations on the RTX 5090 are exact with vLLM's CUTLASS FP4 GEMM; with vLLM's
default there (FlashInfer's) one of two exact runs differed from stock: not shown exact.

## Options

Each option is read from the first of these that sets it: `--additional-config '{"glyd": {...}}'`, the checkpoint's
`quantization_config` (or `--hf-overrides`'), or the environment (`GLYD_LAYOUT`, `GLYD_EXACT`, `GLYD_VERIFY`,
`GLYD_FRACTION`, `GLYD_KV`). Another key, a flag other than true or false (`1`, `true`, `yes`, `on`; `0`, `false`, `no`, `off`), or a
fraction outside 0 to 1 is refused.

| Option | Values | What it does |
| :--- | :--- | :--- |
| `layout` | `auto` (default), `mma`, `mma12` | `auto` takes `best_layout`'s choice for the GPU: `mma` on Ada (L4, L40S, RTX 40), for a mixture of experts on an A10 too, and wherever only it fits; else `mma12` (A10, A100, H100, GH200). A save loads in its own. |
| `exact` | `false` (default), `true` | Each product's matrix decoded whole, then the GEMM vLLM runs for bf16, so the logits are bf16's bit for bit (below). |
| `verify` | `false` (default), `true` | Every pack decoded at load and compared with its weights bit for bit. A save's packs by glyd.json's sha256, its other tensors too, and a save packed again in the other layout against the save. |
| `fraction` | `1` (default), a number from 0 to 1 | The share of the decoder layers packed; the rest stay vLLM's own bf16 ([below](#a-fraction-of-the-layers)). `0` is bf16, `1` every layer. A glyd save takes only `1`. |
| `kv` | `auto` (default), `lossless`, `off` | The KV cache held in 12 bits a value instead of 16, every value read back bit for bit ([below](#the-kv-cache-in-12-bits)). `auto` holds it on an A100, an L4, an H100 and a GH200 (the GPUs it is measured on: any A100, H100 or GH200, by name), and leaves vLLM's own cache elsewhere (an L40S, an RTX 40, an H200, an A10 and the rest until each is measured); `lossless` holds it on any GPU it supports; `off` is vLLM's own. It works with any `fraction`, 0 included. |

## A fraction of the layers

A packed layer saves memory and costs some time at every step; a layer left as it is saves nothing and
costs nothing extra. `fraction` says how many layers are packed: `0` packs none, which is vLLM's own bf16, and `1` every
layer, as before.

```bash
vllm serve Qwen/Qwen3.5-9B --quantization glyd --additional-config '{"glyd": {"fraction": 0.5}}'
GLYD_FRACTION=0.5 vllm serve Qwen/Qwen3.5-9B --quantization glyd
```

- **Which layers.** floor(L × fraction) of a model's L layers are packed, spread evenly over its depth (0.5 packs
  layers 1, 3, 5 and so on; 0.25 packs 3, 7, 11). All of a layer's Linears go together, and a mixture of experts'
  layer's experts with them. The fraction counts as the decimal it is written in: 0.29 of 100 layers is 29. A layer is
  found by the first number in a module's name (`model.layers.12.mlp.down_proj`); a Linear outside the numbered
  layers is packed at `1` only, and so are the embedding and the LM head: under a fraction below 1 they are as without the option.
- **The others** run vLLM's own code, as vLLM runs a model with no quantization. At `0` nothing is packed and Glyd's
  library is not loaded.
- **Exact mode** is unaffected: a layer left as it is gives bf16's bits.
- **A glyd save** takes only `1`, its layers being packed on disk; a bf16 checkpoint takes any fraction. A fraction outside
  0 to 1 is refused.
- **The compile cache** is keyed by the fraction with the other options.

Not measured on a 2026 model yet.

## The KV cache in 12 bits

`kv: lossless` holds the KV cache in 12 bits a value where bf16 has 16, and every value is read back bit for bit. The memory vLLM gives its KV cache holds more tokens: more requests at once, or longer ones. It holds models with head size 128 and full attention only (below: what is refused), so Qwen3.5, Qwen3.8 and Gemma 4 keep vLLM's own cache, with a line in the log; no 2026 model is measured with it yet. It works with any `fraction` (0 is bf16 weights with the cache compressed). `kv: auto`, the default, holds the cache in 12 bits on an A100, an L4, an H100 and a GH200, where it is measured faster than vLLM's own, and leaves vLLM's own cache on any other GPU until it is measured there (`glyd run`, one user's chat, keeps vLLM's own cache too, since the first start's tables take minutes for about 1% for one user: `GLYD_KV=auto glyd run MODEL` turns it on; `glyd serve` and `vllm serve --quantization glyd` use `auto`); `kv: lossless` holds it on any GPU the cache supports, and `kv: off` never.

```bash
vllm serve MODEL --quantization glyd --additional-config '{"glyd": {"fraction": 0, "kv": "lossless"}}'
GLYD_KV=lossless vllm serve MODEL --quantization glyd
```

- **What is exact.** The stored values. `verify` keeps a bf16 shadow of each layer's cache and checks every write, and every context read back for a prompt, against it, bit for bit, in a small run
  (`--num-gpu-blocks-override 600`). A decode step's attention is not bit-equal to FlashAttention's (the sums are in
  another order: closer to an fp32 reference, but not the same bits): its outputs differ from vLLM's as far as vLLM's own
  differ when only the batch changes.
- **First start.** The cache is calibrated once for a model, on vLLM itself, logged, then kept in `$GLYD_CACHE/kv` (`~/.cache/glyd/kv`) for that model, vLLM and context
  length (4,096, 8,192, 16,384, 32,768 or 40,960 tokens: the least that covers `--max-model-len`). A longer context never has its window shortened: `auto` keeps vLLM's own cache for it, and `lossless` serves it on vLLM's own cache too, each with one line that says why. It also measures the model's whole vocabulary, to size the cache's reserves (below). A model whose values compress too poorly
  is refused, with why. The first start takes minutes, and the weights are loaded twice.
- **What works.** Decode steps run in vLLM's FULL decode CUDA graphs (its default, FULL_AND_PIECEWISE): a sequence's output does not depend on the other sequences in the batch nor on the batch's padding. Prefix caching and chunked prefill work: a prompt,
  a later step of a chunked one, or the new tokens after a prefix hit run on vLLM's own FlashAttention, so the row's logits and prompt logprobs are vLLM's bit for bit, and no step waits for the GPU on the host.
  The scheduler and block allocator are vLLM's (the scheduler is vLLM's own with one override, `SchedulerConfig.scheduler_cls`, an interface vLLM does not list as public, and needs the single-process executor). `exact` (the plugin's option) runs every row, decode rows too, as vLLM's
  own does, for logits that are vLLM's bf16 bit for bit, eager (refused where that is over 5% of the GPU's memory, so serve it with a small `--max-num-seqs`).
- **Content that does not store at 12 bits, and what happens when the room runs out.** Text and token ids are almost all held in 12 bits. A prompt of one repeated token, or of tokens whose embeddings were never trained (rare tokens), is held in more.
  The cache keeps two reserves for it, sized for the model at its first start from its whole vocabulary (most of the second for the models that have rare tokens). Its values stay exact either way.
  Ordinary text takes almost none of either.
  A request is let in when the second reserve has room for what its prompt will take (known from the prompt's token ids before it runs); otherwise it waits in vLLM's queue, as it does when KV blocks are short, and the requests behind it
  that need none go on. A prompt that could never fit even alone is refused when it is sent, with a 400 and why, as one longer than the model's context is. Every API server process decides this on its own, from the model's tables in the cache directory and the size of
  the KV cache vLLM reports: any number of API servers, in any environment, refuse the same prompts (where a process cannot read the tables it logs one line and the engine ends such a prompt with an error finish instead). Where generated tokens still outgrow the reserve, the request is preempted
  as vLLM preempts for blocks and runs again when there is room. The engine never stops for what a prompt holds, and a request that stock vLLM serves is not failed with a 500 for it. Log lines say when a reserve has had 50, 75 and 90%
  of its room taken at once, and when requests were preempted.
- **Known limit.** Prompts made mostly of rare tokens are served more slowly than stock's when many arrive at once, on
  a model that has such tokens. The ordinary requests sent with them are not held behind them.
- **Rare tokens.** A row of the embedding matrix that was never trained is zero or nearly, and a prompt of such tokens is held in more than 12 bits.
  The first start counts them (those the tokenizer makes from their own strings and those only a request of token ids reaches). vLLM's own warmup, which runs dummy requests with the token ids 0, 1, 2 ...,
  writes nothing to the cache for such a model.
- **Head size 256 and linear-attention layers** (Qwen3.5, Qwen3.6, Qwen3.8: Gated DeltaNet beside full attention, head 256): held where `kv: lossless` asks; `auto` keeps vLLM's own cache for them until
  they are measured faster. A head of 256 channels is two of the page's heads of 128 (the page is 49,408 bytes for 4 KV heads of 256, as for 8 of 128; the tables are [2 H, 128]); the decode kernel runs a
  page on two warps, each its half's channels, their scores summed in shared memory before the softmax. The linear-attention layers' states stay vLLM's, in the one pool of blocks they share with the cache:
  a block holds block_size / 16 of the cache's pages (vLLM sizes it from the page: 704 tokens for Qwen3.5-9B, 528 for bf16's), and every block a step takes from the free pool has its pages cleared at the
  start of the step (before vLLM zeroes it or another group's state is copied into it), so that the reserves' chunks and slabs a page held in an earlier life go back to them. Measured on an RTX 4090 with
  Qwen3.5-9B ([benchmarks/gpu/kv-2026-10-06](../../benchmarks/gpu/kv-2026-10-06)): 1.18x the KV cache's tokens at 8,192 (183,409 against 154,903), prompt logprobs and prefix hits bit for bit against vLLM's
  cache at the same block, verify clean under churn, tokens a second at least vLLM's at 1 to 256 requests at once (1.10x the requests a second at 256) but the first token 5% later for one request.
- **Refused at start, with why:** speculative decoding, tensor, pipeline and context parallel, sliding-window and
  state-space layers, MLA, LoRA, KV connectors and offload, `--kv-cache-dtype` other than auto, head sizes other than 128 and 256,
  GPUs other than Ampere, Ada and Hopper (measured on an L4, an A100, an H100 and a GH200 so far), another attention backend, and a model whose values compress too poorly in 12 bits. With bf16 weights (`fraction` 0) or `exact`, `VLLM_BATCH_INVARIANT=1` works: a
  sequence's decode outputs are the same bits alone and in a batch (Glyd's fused weight kernels are chosen by the batch's tokens, so they
  are refused under it).

## Exact mode

Glyd's products sum in another order than bf16's, so logits can differ from bf16's in their last bits. The same
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
- **Compiled, Linears with biases** (a q, k and v with biases, for one) are refused: in bf16's graph inductor adds a
  Linear's bias apart from its matmul, rounding before the add, where exact's product adds it in the GEMM. Eager, exact
  gives bf16's bits there too.
- **`VLLM_BATCH_INVARIANT`** asks for every product's bits not to depend on the batch. Glyd's kernels are chosen by
  the batch's tokens, so without exact it is refused; with exact the products are vLLM's batch-invariant GEMM on the
  decoded weights.
- **The deterministic mode on its own** makes compiled Glyd, in its default mode too, the same from one run to the
  next.
- **The embedding and the LM head under exact:** the LM head stays as vLLM runs it, matrix and GEMM, so its logits are bf16's bit for bit
  (decoding it whole for the GEMM would take a buffer of its size, more than packing it saves), and so does a tied model's one table, which is that
  matrix. An untied model's embedding is packed: its rows read back exactly. The log says so.
- **The cost of exact** is a decode per product.
- **For a mixture of experts,** exact needs vLLM's Triton kernel for bf16's experts. It is
  refused where vLLM picks another, which lays the weights out otherwise.

## Measured

2026 models, vLLM 0.30.0, bf16 against Glyd in the same session:

| GPU (Glyd's layout) | Model | Weights | KV cache | Saturated: requests/s, first token, each token | 1 request a second: requests/s, first token, each token |
| :--- | :--- | ---: | ---: | :--- | :--- |
| H100 80GB HBM3 (`mma12`) | Qwen3.8-27B (text only) | 38.77 GiB against 50.22 (−22.8%) | 204,117 tokens against 122,538 (1.67x) | 5.05 against 4.72 and 4.70 (1.07x), 0.84x as late, 1.43x as long | 0.92 against 0.92, 227 ms against 199, 21.7 ms against 23.4 |
| RTX 4090 (`mma`) | Qwen3.5-9B | 12.68 GiB against 17.66 (−28.2%) | bf16: out of memory at start (even at 4,096 tokens); Glyd: a 189,440-token context | | |
| RTX 5090 (`mma12`) | Qwen3.5-9B | 13.89 GiB against 17.66 (−21.3%) | | | |

- **Qwen3.8-27B:** 1,024 tokens in and 256 out, saturated = 256 requests at once. bf16's continuations scored on Glyd:
  top-1 99.22% (bf16 against itself started again: 99.22%). The lossless KV cache stays off for it (linear attention
  and head size 256), with a line in the log. Glyd's saturated passes drew more power and ran at a lower clock (1,852
  MHz against 1,980), so the run's own clock check marks it not comparable. Logs:
  [h100-qwen38-v028-2026-10-03](../../benchmarks/gpu/h100-qwen38-v028-2026-10-03).
- **Qwen3.5-9B, one user** (a prompt of about 1,000 tokens, 256 out, the median of three): 115.9 tokens a second on
  the RTX 5090 against vLLM's bf16 95.0, the first token after 131 ms against 113; 77.3 on the RTX 4090, where bf16
  did not start (CHANGELOG.md, v0.29.0).

## Speculative decoding

vLLM's speculation runs on Glyd's packs as on bf16's weights, n-gram and EAGLE-3 drafts alike. Not measured on a
2026 model yet.

- **Exact.** With speculation, exact eager gives bf16 eager's tokens, request for request.
- **Tokens against plain decoding.** A verify step multiplies up to 1 + k tokens at once, and GEMMs and attention round
  by that shape. So speculative decoding's greedy tokens differ from plain decoding's, bf16's too. Under
  `VLLM_BATCH_INVARIANT=1` they are the same.
- **The draft.** Under `--quantization glyd`, vLLM leaves an EAGLE-3 draft bf16. With `"quantization": "glyd"` in
  `--speculative-config` it is packed too.

`spec_decode.py` runs one configuration; `spec_summary.py` makes the tables.

## Not supported yet

Each of these is refused at start, with a message saying why; none runs wrong. What vLLM's config tells is refused as
the engine builds it, before any worker starts, so over several GPUs too Glyd's message is the error the user sees:

- dual-batch overlap (`--enable-dbo`);
- LoRA;
- weight offloading (`--cpu-offload-gb`) and sleep mode;
- a glyd save over several GPUs, one with a mixture of experts' packs, or one of an architecture other than
  Qwen3ForCausalLM and LlamaForCausalLM (every 2026 model so far has another architecture); their bf16 checkpoints load;
- a glyd save with a `fraction` below 1: its layers are packed on disk, and its bf16 checkpoint takes a fraction;
- exact under torch.compile where a packed Linear has a bias (exact eager runs), and Glyd's default products under
  `VLLM_BATCH_INVARIANT`;
- with `kv: lossless`, what [its section](#the-kv-cache-in-12-bits) lists.

Tensor parallelism packs each rank's shard.

## Files here

- `quant_parts.py MODEL --out DIR`: a quantized checkpoint, stock vLLM against `--quantization glyd` on one GPU: exact mode eager
  against stock eager bit for bit (tokens, top-5 logprobs, prompt_logprobs, an image's chat), the weights and KV cache vLLM logs, 1 and
  32 requests' tokens a second, and what Glyd packed.
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
- `moe_routes.py [MODEL]`: a mixture of experts' layer by tokens a step, Glyd's two paths against bf16's own layer
  (the threshold `GLYD_MOE_DECODE_MIN` moves).
- `spec_decode.py`, `spec_summary.py`: one user's tokens/s, first token and speculation's acceptance for a configuration
  (bf16 or Glyd, n-gram or EAGLE-3, eager, exact), and the runs' tables and token-for-token comparisons.
- `check_vllm.py --kv [--kernels-only | --quick | --sanitize | --ext-only] [--study DIR] [MODEL]`: the lossless KV cache.
  - Its kernels through the library: every bit pattern, every 16-bit pattern, recycled blocks and
    rewrites (with 3 to 32 KV heads too), the reserves (what is given back, one that runs out), attention against an fp32 reference.
    `--ext-only`: prefix caching, chunked prefill and `exact` against vLLM's own.
  - vLLM's own keys and values of real prompts through the write kernel and back, and `verify` in vLLM (`--study` is the
    KV study's ten prompts, 113 to 8,192 tokens and 512 generated each; else this repository's documents).
  - The cache's size against stock's at the standard flags, prompt logprobs and the first token's
    bit for bit stock's, eager and compiled in inductor's deterministic mode (in the default compiled mode within the
    compile's own variation), and floods of the model's rare-token prompts: all served, against stock's wall time, the ordinary requests among them the same bits
    (under `VLLM_BATCH_INVARIANT=1`), the reserve run out in flight and reserves too small (`--no-flood-extras` leaves all but the first flood out).
  - `--sanitize` runs the kernels' checks under compute-sanitizer (memcheck, initcheck, racecheck, synccheck).
  - `--model-checks MODEL`: a model other than the default one, its tables made as a first start makes them (a refusal is exit 1 of the calibration, and the plugin is asked for the model to see that it raises), then the values through
    the write kernel and back, `verify`, the cache's size against stock's, eager against stock's and the refusals, not the compiled and graph runs. `--tables-only`: the tables alone, for a model the GPU cannot hold
    (`KV_CAL_ENGINE`, the calibration's LLM arguments as JSON: `{"cpu_offload_gb": 16}`).
- `kv_stress.py capture|engine`: the content that stores worst (repeated tokens, first tokens, random ids,
  14 languages, tables) written through the kernel and read back, with the reserves sized as the plugin sizes them; and through
  vLLM's engine (`--verify`, `--ext`). `kv_stress_text.py`: its texts.
- `kv_capacity.py`, `kv_agree.py`, `kv_tps.py`: the KV cache's size at each reserve share, its outputs against vLLM's own
  (free-running and teacher-forced, with vLLM's own spread beside), and decode tokens a second at the same batches and
  at each side's largest (`--rounds` alternates the two processes, stock first and kv first in turn). `kv_decode_bench.py` and `kv_step_profile.py`: the kernels
  alone against FlashAttention 2 and vLLM's cache write, timed in a CUDA graph as a decode step has them (`--sweep` times the decode kernel at several plans), and where a decode step's time goes (`--graphs`: in vLLM's default mode).
  `kv_prefix_control.py`: the prefix-hit scenario of `--ext-only` with each attention layer's query, key, value and output compared between two runs (stock, kv, kv `exact`, stock with its decode steps' keys in one split), to see
  where two runs first part.
  `make_kvcorpus.py` made `glyd-gpu-py/glyd_gpu/kvcorpus.txt`, the text the calibration runs on.
- `kv_traffic.py tokens|kv`: how common the content that stores worst is in text people send (`tokens` needs a tokenizer, no GPU; `kv` runs the documents through a model), with an estimate of the decode kernel's extra cost. `kv_firststart.py`: a first start on an empty cache directory, its seconds.
  `kv_launches.py`: the kernels a decode step launches a layer, by kind, under torch's profiler (vLLM's cache or kv; compiled or eager).
- `head_sweep.py [MODEL]`: the LM head's product for 1 to 4,096 rows, packed against bf16's F.linear (and against the matrix decoded in
  pieces first), in both layouts, and the embedding's lookup for 1 to 8,192 tokens, on this GPU.
- `tables_check.py [--tp N]`: the embedding and the LM head on tiny random models (tied and untied, with a vocabulary vLLM pads or not and shards,
  tied with a copy in the checkpoint, equal and not), in vLLM over N GPUs: every table packed, every lookup the checkpoint's row bit for bit on each
  rank, each product within 1e-2, bf16's tokens, exact mode bit for bit. `check_vllm.py` does the same for a real model, and reports the weights vLLM logs.
- `tps_ab.py --a NAME:K=V --b NAME:K=V`: one user's tokens a second of two trees (or modes) alternated, each in a vLLM of its own, with the weights and KV
  cache vLLM logs. `tp2_job.sh`: `head_sweep.py`, `tables_check.py --tp 2` and `check_vllm.py --brief --tp 2` as one job on a two-GPU machine, in 30 minutes.
- `acct.py MODEL`: where the loaded model's GPU memory is, every block the allocator holds by owner (packs by kind, norm weights, rotary caches, the plugin's decode buffer,
  vLLM's own baseline), against the figure vLLM logs for the load. `routes.py [GPU CODE]`: the library's route table for a model's Linear shapes (for which counts of
  tokens a step a product decodes its matrix for cuBLAS, which is when the plugin needs its decode buffer; no GPU work).
- `glyd-gpu-py/tests/test_vllm.py`: the plugin's logic that needs no GPU (options, the layers a fraction packs and the
  method each layer gets, the embedding and the LM head's plan, a save's packs by vLLM's layer names, the pieces a checkpoint gave, what is refused, the entry
  point's version rule), with vLLM installed; and, with a GPU, the lookup and the LM head's product over a pack and the method's steps on a bare layer.

## The plugin's source

The plugin is `glyd-gpu-py/glyd_gpu/vllm_plugin.py`, loaded by `vllm_entry.py`, the entry point, which checks
vLLM's version first. It relies on vLLM 0.30's plugin and quantization interfaces, and copies none of vLLM's code.
`kv: lossless` is an attention backend of the plugin (`vllm_kv.py`, with `vllm_sched.py` and `vllm_admit.py`), which
relies on more of vLLM's internals, listed in their docstrings. Like the rest of Glyd's GPU code it is under the
Business Source License 1.1 (`gpu/LICENSE`).
