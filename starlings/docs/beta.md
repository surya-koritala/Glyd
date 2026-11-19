# Starlings beta

Starlings runs one model across two GPUs: two cards in one machine, or two machines on one LAN.
Each GPU holds a contiguous range of the model's layers. A request's tokens cross between them
once per step. The answers are the same tokens the whole model gives on one card.

This beta runs Glyd's Qwen3.5-family models (for example `glyd/Qwen3.8-27B-swift`, 18.7 GB) on
NVIDIA GPUs under Linux.

## What you need

- Linux x86_64 with glibc 2.28 or newer (Ubuntu 20.04 or newer, Debian 11, RHEL 8) and an NVIDIA
  driver for CUDA 13 (driver 580 or newer; `nvidia-smi` must work).
- For two machines: both on one LAN with private addresses (10.x, 172.16-31.x, 192.168.x), and room
  for the model on both disks. The first machine takes TCP 9741 and 9761-9762 from the second; the
  second takes 9770-9772 from the first.
- About half the model's size free on each GPU (the 27B: about 10 GB each), plus 1 GiB.

## Install

    tar xzf starlings-VERSION-linux-x86_64-cuda13.tar.gz
    starlings-VERSION-linux-x86_64-cuda13/install.sh

No root is needed. The command goes in `~/.local/bin/starlings`, and its files go in
`~/.local/lib/starlings`. `install.sh --uninstall` removes both; your state and downloaded models
stay.

## One machine

    starlings up --model glyd/Qwen3.8-27B-swift

The model downloads into the Hugging Face cache the first time; `up` asks before it downloads. It
then:
1. shows the engine, the model and where each half goes, and asks you to confirm;
2. starts everything;
3. prints a link to the web UI and the API's address.

With two GPUs, each takes a half. With one GPU, both halves go on it. That proves the setup, but
it is slower than running the model whole. Ctrl-C stops everything.

## Two machines

On the first machine:

    starlings up --model glyd/Qwen3.8-27B-swift --machines 2

It prints a **join code** and a **cluster check**. On the second machine, within five minutes:

    starlings join --model glyd/Qwen3.8-27B-swift

1. Paste the join code when the second machine asks for it. That machine shows the cluster
   check; it must be the same as the first machine's. It shows which layers it will take; answer
   y.
2. It prints an **approval code** and an **approval check**. Paste the code on the first
   machine, then type the check there when it asks.
3. Both machines load their halves and print READY. Open the link the first machine printed.

The checks are how you know the two machines are talking to each other and not to something in
between. Don't approve a check you didn't read off the other machine's screen.

If either machine has more than one network address, add `--host LAN_ADDRESS`. To start again
later with the same setup, run the same command; it asks once and reuses everything.

## Using it

- **Web UI:** the cluster (which GPU has which layers, memory, state), chat, and the last 50
  requests. Under each answer it shows the tokens, tok/s, the time to first text, each GPU's
  engine time, and the time spent between them.
- **API:** `http://127.0.0.1:9750/v1/chat/completions` on the first machine, OpenAI-style chat
  completions (stream or not). Send `temperature`, `top_p` and `seed`. The bearer token is in the
  file `up` prints and lasts one hour; the UI has its own.

Both listen on loopback only. To use them from another machine, tunnel with
`ssh -L 9751:127.0.0.1:9751`.

## What it measured

Qwen3.8-27B swift on two RTX 4090s (24 GB) in one data center (0.5 ms apart), 256 tokens, greedy:

| | tok/s |
| --- | --- |
| whole model on one 4090 | 46.0 |
| two machines | 43.5 |
| two machines, drafting on (the default) | 52-54 |
| two requests at once (`--sessions 2`) | about 50 each |

Each answer was the whole model's, word for word. From empty state to READY took 25 s with the
model already on disk. On a home LAN or Wi-Fi each token waits one round trip between the
machines, so a slower link means fewer tok/s; drafting hides part of it.

## Limits

- **Hardware:** two GPUs, two machines at most, NVIDIA only, Linux only, the Qwen3.5 family only.
- **Context:** 4096 tokens by default (`--ctx`), and at most 1024 tokens in an answer.
- **Concurrency:** one request at a time, unless the first machine starts with `--sessions 2`
  (the second takes it from the join code).
- **Network:** fixed ports (9741, 9750-9751, 9760-9762, 9770-9772), and private addresses only. Machines on
  public addresses need a private network (a VPN) first.
- **Failures:** if a stage's engine dies, `up` and `join` restart that stage; a request in
  flight then fails. If a machine goes away, its requests fail within a minute.
- **Shared machines:** another user on the same machine can hold all 4 local API connections.
- **Integrity:** the download has checksums but is not signed yet. The model check compares the
  model's description, not its weights.

## Reporting a problem

Send what you ran, what it printed, and:

    starlings cluster diagnostics --state-dir ~/.local/state/starlings/coordinator > diagnostics.json

On the second machine, use `--state-dir ~/.local/state/starlings/stage`. The bundle is redacted
and has no prompts or keys. Each daemon's log is `daemon.log` in its state directory. The first
machine's coordinator log holds the UI's secret link, so don't send that file.
