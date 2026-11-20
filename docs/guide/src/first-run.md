# First run

Three commands, from the folder you built in (or with `glyd-engine` on your PATH):

```bash
glyd-engine doctor                       # what this computer has: GPU, driver, library, disk, Hub
glyd-engine fit Qwen/Qwen3-8B            # the plan for this GPU, before anything is downloaded
glyd-engine run Qwen/Qwen3-8B            # download, pack, then a chat in the terminal and at http://localhost:8000
```

## doctor

`doctor` prints one line for each thing the engine needs and ends with "Everything glyd run needs is here." (exit 0) or
"N thing(s) to fix." (exit 1). [Troubleshooting](troubleshooting.md) goes through each line.

## fit

`fit` reads the model's `config.json` from the Hub and prints the plan: the level, where the weights go, the context, and
a predicted speed. With `--machine rtx4080s`, the preset for an RTX 4080 SUPER with 15.2 GiB free:

```
Plan: penguin (10.8 bits) · GPU 0 (NVIDIA GeForce RTX 4080 SUPER) · 8192 tokens of context · ~50 tokens/s
  penguin       10.3 GiB     8192 ctx  ~  50 tok/s  GPU 0 (NVIDIA GeForce RTX 4080 SUPER)  <- chosen
```

The speed is an estimate from memory bandwidth, for a short prompt and one user. `--machine NAME` plans for another
computer (`rtx4060-laptop`, `rtx4080s`, `rtx4090`, `2x3090`, `l4`, `a6000`, `a100-80g`, `h100`, `8xh100`, `rx7900xtx`,
`m4max-64g`, `dgx-spark`, `strix-halo-128g`), so you can see the plan before you buy or rent. `--context N` and `--users N` change what it plans for.
`glyd-engine preflight MODEL` says whether the engine will certainly run the model here (exit 0 yes, 3 no with the reason).

## run

The first run downloads the model into the Hugging Face cache, packs it for this GPU and makes the KV tables. Later runs
start from the packed copy. On an RTX 4080 SUPER with Qwen3-8B at penguin the record has: packed in 12 s, KV tables in 31 s,
ready in 45 s, 10.3 GiB on the GPU instead of bf16's 15.3 GiB (which does not load on that 16 GB card), 59.7 tokens/s for
one user, which is 90% of the card's memory-bandwidth ceiling for those weights. One run, one request:
[benchmarks/gpu/rtx4080s-qwen3-8b-engine-2026-11-20](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/rtx4080s-qwen3-8b-engine-2026-11-20).

- `--prompt "TEXT"` gives one answer and exits. `--no-think` turns a thinking model's thinking off.
- `MODEL:LEVEL` picks a level: [Choosing a level](levels.md). `--context N` sets the context.
- The chat page is at `http://localhost:8000` while it runs: the same server as [serve](serve.md).
- A gated model needs a Hugging Face token: `glyd-engine login`, or `HF_TOKEN` in the environment.
- The packed models are in `$GLYD_CACHE` (default `~/.cache/glyd`), the downloads in the Hub cache (`$HF_HOME`, default
  `~/.cache/huggingface`). Set either to put them on another drive.
