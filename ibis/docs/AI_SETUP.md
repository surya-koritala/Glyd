# Setting up Ibis (Qwen3.8-Flash-Next): steps for an AI assistant

You are setting up Glyd's Qwen3.8-Flash-Next (Hummingbird level) for a user: Qwen3.8-Flash-Next running on their own NVIDIA GPU, with a chat page and an OpenAI- and
Anthropic-compatible API. Follow these steps in order, and tell the user what you find at each one.

## 1. Check the computer

Run these and compare with the requirements:

```sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | awk '/^Mem:/ {print $2 " GiB RAM"}'
df -h ~ | tail -1
uname -sm
```

| Requirement | Needed |
|---|---|
| System | Linux on x86-64, Ubuntu 22.04 or newer (tested on 22.04, 24.04 and 26.04). These steps are Linux's; on Windows run `setup.ps1` (see `ibis/README.md`). WSL2 is untested. |
| GPU | NVIDIA, Ampere or newer (RTX 30, 40 or 50 series, A-series, H100), 16 GB or more |
| Driver | 580 or newer |
| RAM | 64 GB |
| Disk | about 90 GB free (75 GB for fast mode) |

If something is short, stop and tell the user which requirement, and by how much.

## 2. Install

```sh
git clone https://github.com/surya-koritala/Glyd && cd Glyd/ibis
tools/install.sh                  # Glyd's engine (prebuilt from this repository's source) under ~/.glyd-qwen-3.8-flash; no sudo
                                  # (tools/install.sh --source builds it here instead: needs Rust and the CUDA toolkit)
tools/download.sh                 # precise mode (recommended), about 82 GB; --fast for fast mode, about 68 GB
```

The download resumes if it stops: run it again.

## 3. Start it

```sh
tools/serve.sh                    # precise mode; tools/serve.sh --fast for fast mode
```

The page opens at once and says when the model is loaded (about a minute). Then:

- the chat page is at http://127.0.0.1:8080/;
- the OpenAI-compatible API is at http://127.0.0.1:8080/v1, model `qwen3.8-flash-next`, any API key;
- the Anthropic-compatible API is at http://127.0.0.1:8080/v1/messages.

Check that it answers:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello in five words."}]}'
```

## 4. Connect the user's tools

- **Any OpenAI-compatible app** (Open WebUI, Continue, Cline, LibreChat): base URL `http://127.0.0.1:8080/v1`, model `qwen3.8-flash-next`.
- **Anthropic-compatible tools:** base URL `http://127.0.0.1:8080`.
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude`.
- **MCP:** `claude mcp add glyd -- ~/.glyd-qwen-3.8-flash/bin/glyd-engine qwenmcp --repo "$PWD"` gives an AI tool `glyd_start`, `_stop`, `_status` and `_ask`.

## 5. Measure it

`tools/bench.sh` takes about 5 minutes, with the server running. It measures a chat and 4K, 16K and 32K prompts, and writes `bench-report.md` for the user.
