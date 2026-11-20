# First run

Three commands, from the folder you built in (or with `glyd-engine` on your PATH):

```bash
glyd-engine doctor                       # what this computer has: GPU, driver, library, disk, Hub
glyd-engine fit google/gemma-4-12B-it    # the plan for this GPU, before anything is downloaded
glyd-engine run google/gemma-4-12B-it    # download, pack, then a chat in the terminal and at http://localhost:8000
```

## doctor

`doctor` prints one line for each thing the engine needs and ends with "Everything glyd run needs is here." (exit 0) or
"N thing(s) to fix." (exit 1). [Troubleshooting](troubleshooting.md) goes through each line.

## fit

`fit` reads the model's `config.json` from the Hub and prints the plan: the level, where the weights go, the context, and a
predicted speed. With `--machine rtx4090`, the preset for an RTX 4090 with 23 GiB free:

```
Plan: penguin (10.8 bits) · GPU 0 (NVIDIA GeForce RTX 4090) · 32768 tokens of context · ~43 tokens/s
  penguin       14.9 GiB    32768 ctx  ~  43 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)  <- chosen
  kestrel        9.0 GiB    32768 ctx  ~  80 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)
  swift          7.6 GiB    32768 ctx  ~  94 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)
  wren           6.2 GiB    32768 ctx  ~ 114 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)
```

It lists every level the model runs at with its size on your GPU, and marks the one it chose. The speeds are estimates from
memory bandwidth, for a short prompt and one user. `--machine NAME` plans for another computer (`rtx4060-laptop`, `rtx4080s`,
`rtx4090`, `2x3090`, `l4`, `a6000`, `a100-80g`, `h100`, `8xh100`, `rx7900xtx`, `m4max-64g`, `dgx-spark`, `strix-halo-128g`), so
you can see the plan before you buy or rent. `--context N` and `--users N` change what it plans for.
`glyd-engine preflight MODEL` says whether the engine will certainly run the model here (exit 0 yes, 3 no with the reason).

## run

The first run downloads the model into the Hugging Face cache, packs it for this GPU and makes the KV tables. Later runs start
from the packed copy.

- `--prompt "TEXT"` gives one answer and exits. `--no-think` turns a thinking model's thinking off.
- `MODEL:LEVEL` picks a level: [Choosing a level](levels.md). `--context N` sets the context.
- The chat page is at `http://localhost:8000` while it runs: the same server as [serve](serve.md).
- A gated model needs a Hugging Face token: `glyd-engine login`, or `HF_TOKEN` in the environment.
- The packed models are in `$GLYD_CACHE` (default `~/.cache/glyd`), the downloads in the Hub cache (`$HF_HOME`, default
  `~/.cache/huggingface`). Set either to put them on another drive.
