# First run

A few commands (with a source build, `target/release/glyd`):

```bash
glyd doctor                       # what this computer has: GPU, driver, library, disk, Hub
glyd fit gemma4:12b               # the plan for this GPU, before anything is downloaded
glyd run gemma4:12b               # download, pack, then a chat in the terminal and at http://localhost:8000
```

## Names

A model is a Hub id (`google/gemma-4-12B-it`), a directory, a GGUF file, or a short name: `NAME[:SIZE][:LEVEL]`, for example
`gemma4`, `qwen3.8:27b`, `gemma4:12b:wren`. A tag of letters only is a level, so `:27b` is a size. Without a size, the family's
default is taken. `glyd ls --all` lists the names and what each is:

```bash
glyd ls --all        # name, level, size on disk, repo; "not pulled" for a name not downloaded
glyd pull gemma4:12b # download a model by name or Hub id
glyd rm gemma4:12b   # take it out of the cache
```

Short names offer models released in the last six months. An older model still works by its Hub id.

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
`glyd preflight MODEL` says whether the engine will certainly run the model here (exit 0 yes, 3 no with the reason).

When a model does not load at the level asked for, `fit` prints the degrade ladder: a shorter context, experts in the computer's memory, then a lower level, each with its cost in speed. `run` says the same steps in its first lines and asks before one that costs over 30% of the speed or lowers the level (`--yes`, or `GLYD_DEGRADE=yes`, takes them; `GLYD_DEGRADE=no` refuses); `serve` and the service never stop to ask and refuse without `--yes`.

`fit` reads the model's file headers by range requests (two small reads a shard, kept in the cache), so a model's sizes are exact before anything is downloaded; nothing of the weights is fetched to plan. The plan is what loads. For Qwen3 dense, Qwen3.5/3.6/3.8, Nemotron 3 Super, Gemma 4 and (once its files are here) the models that run over several GPUs, `fit` sizes the model with the
functions `run` and `serve` do, from the same GPU's free memory: the weights, the cache for the context and the users, the engine's work, and a
small reserve the GPU keeps. Where `run` would refuse, `fit` refuses in the same words and exits 3. `run` prints its plan before it loads
(`Plan: ...`). `fit --json` prints the plan as `glyd-plan/1` for scripts: `level`, `context`, `users`, the bytes of each part, and `experts`
(places a layer on the GPU, bytes in the computer's memory and on the disk) where a mixture of experts keeps them off the GPU.

## run

The first run downloads the model into the Hugging Face cache, packs it for this GPU and makes the KV tables. Later runs start
from the packed copy.

- `--prompt "TEXT"` gives one answer and exits. `--no-think` turns a thinking model's thinking off.
- `MODEL:LEVEL` picks a level: [Choosing a level](levels.md). `--context N` sets the context.
- The chat page is at `http://localhost:8000` while it runs: the same server as [serve](serve.md).
- A gated model needs a Hugging Face token: `glyd login`, or `HF_TOKEN` in the environment.
- The packed models are in `$GLYD_CACHE` (default `~/.cache/glyd`), the downloads in the Hub cache (`$HF_HOME`, default
  `~/.cache/huggingface`). Set either to put them on another drive.
