# Condor: a big MoE on one GPU

Condor runs Qwen3.8-Flash-Next, a 180B mixture of experts, on one 12-16 GB NVIDIA GPU with 32-64 GB of RAM. It has its own
setup script, its own server and chat page. `glyd-engine run` does not take this model yet: it refuses it and points here
(ROADMAP P8 moves it into `run`).

## What you need

- NVIDIA RTX 30, 40 or 50 series (or an Ampere workstation card), 12 GB or more, driver 580 or newer.
- 32 GB of RAM or more: 64 GB for the precise and best modes, 32 GB for fast.
- About 90 GB of free SSD (73 GB for fast, 97 for best).
- Linux x86_64 or Windows 10/11 x64.

## Install and run

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd/condor
./setup.sh
```

It checks the GPU, RAM and disk, installs the engine, downloads the model (rerun it if it stops), asks the mode and the
context, and opens `http://127.0.0.1:8080`. Run `./setup.sh` again to start next time. On Windows, in PowerShell:

```powershell
git clone https://github.com/surya-koritala/Glyd; cd Glyd\condor
powershell -ExecutionPolicy Bypass -File setup.ps1
```

## Modes

| Mode | Experts, bits | Download | KL vs bf16 | Writing, short prompt | RAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| precise (default) | about 2.9 | 82 GB | 0.15 | 91 tokens/s | 64 GB |
| best | about 3.4 | 90 GB | 0.12 | 79 tokens/s | 64 GB |
| fast | about 2.0 | 66 GB | 0.39 | 113 tokens/s | 32 GB |

RTX 4080 SUPER 16 GB, Ryzen 9 7950X3D, 64 GB RAM, Ubuntu 26.04; Condor v0.2.4 build of 2026-10-21; greedy, thinking off, the
median of three runs. Logs: [condor/results/v0.2.4](https://github.com/surya-koritala/Glyd/tree/main/condor/results/v0.2.4).
Switch with `tools/download.sh --best` (or `--fast`), then `tools/serve.sh --best`.

On other cards, fast mode, short prompt: RTX 5090 175 tokens/s, RTX 4090 116, RTX 3090 72 (rented machines, right after
install). The CPU matters nearly as much as the GPU, because the experts that miss the GPU's cache run on it.

## Using it

- Chat: `http://127.0.0.1:8080`. OpenAI API at `http://127.0.0.1:8080/v1` (any key, any model name), Anthropic API at `/v1/messages`.
- Context up to 262,144 tokens; the default is 32,768 (`GLYD_CTX=65536 tools/serve.sh` for more).
- Two NVIDIA cards: `GLYD_GPUS=0,1 tools/serve.sh`.
- 4 requests at once by default (`GLYD_USERS=N`, 1 to 8).

More: [condor/README.md](https://github.com/surya-koritala/Glyd/blob/main/condor/README.md),
[condor/docs/CLIENTS.md](https://github.com/surya-koritala/Glyd/blob/main/condor/docs/CLIENTS.md) for what is tested with tools.
