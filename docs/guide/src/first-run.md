# First run

A few commands (with a source build, `target/release/glyd`):

```bash
glyd doctor                       # what this computer has: GPU, driver, library, disk, Hub
glyd fit gemma4:12b               # the plan for this GPU, before anything is downloaded
glyd run gemma4:12b               # download, pack, then a chat in the terminal and at http://localhost:8000
```

## Names

A model is a Hub id (`google/gemma-4-12B-it`), a directory, a GGUF file, or a short name: `NAME[:SIZE][:LEVEL]`, for example
`gemma4`, `qwen3.8:27b`, `gemma4:12b:wren`. A tag of letters only is a level, so `:27b` is a size. Without a size, `glyd run`, `serve`, `fit` and
`pull` pick the largest size of the family whose plan on this computer reaches 10 tokens a second and 8,192 tokens of context (the
plan `glyd fit` shows, with the level it takes), and say so in one line:
`glyd: gemma4 is gemma4:12b (google/gemma-4-12B-it) at kestrel: the largest size that gets 10 tokens/s and 8192 tokens of context on ..., about 38 tokens/s, 32768 context; the others: 31b: 4 tokens/s, 8192 context. To choose another: gemma4:SIZE[:LEVEL]`.
`gemma4:31b` (a size, and `:wren` for its level) is taken as it is; a level alone (`gemma4:wren`) is asked of every size. Where no size
reaches the floors the fastest that fits is taken and the line says so; where none fits, the refusal says why for each. `glyd fit gemma4 --machine rtx4090`
shows the choice for another machine. `glyd ls --all` lists the names and what each is:

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
predicted speed. For gemma-4-12B with `--machine rtx4090`, the preset for an RTX 4090 with 23 GiB free (the plan line and the levels; the lines
between them and the ones after, on quality and on where the speeds come from, are left out):

```
Plan: penguin (10.8 bits) · GPU 0 (NVIDIA GeForce RTX 4090) · 32768 tokens of context · ~41 tokens/s (37-45) · prompts ~4583 tokens/s
  penguin       15.0 GiB    32768 ctx  ~  41 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)  <- chosen
  kestrel        9.1 GiB    32768 ctx  ~  83 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)
  swift          7.7 GiB    32768 ctx  ~  95 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)
  wren           6.3 GiB    32768 ctx  ~ 111 tok/s  GPU 0 (NVIDIA GeForce RTX 4090)
```

It lists every level the model runs at with its size on your GPU, and marks the one it chose. The speeds are predictions for
one user at about 1,000 positions, from the rate your GPU reads its memory at. `fit` measures that rate the first time it uses a GPU and
keeps it in the cache. The plan line also shows a p10-p90 band and the prompts' speed. Each
reply that runs alone is recorded with the plan that loaded it, and from then on the predictions on that GPU follow what
those runs reached. `fit --json` has the same figures (`tokens_per_second`, `tokens_per_second_p10`, `tokens_per_second_p90`,
`prefill_tokens_per_second`). `--machine NAME` plans for another computer (`rtx4060-laptop`, `rtx4080s`,
`rtx4090`, `2x3090`, `l4`, `a6000`, `a100-80g`, `h100`, `8xh100`, `rx7900xtx`, `m4max-64g`, `dgx-spark`, `strix-halo-128g`), so
you can see the plan before you buy or rent. `--context N` and `--users N` change what it plans for.
`glyd preflight MODEL` says whether the engine will certainly run the model here (exit 0 yes, 3 no with the reason).

When a model does not load at the level asked for, `fit` prints the degrade ladder: a shorter context, experts in the computer's memory, the last layers' dense matrices there too, then a lower level, each with its cost in speed. `run` says the same steps in its first lines and asks before one that costs over 30% of the speed or lowers the level (`--yes`, or `GLYD_DEGRADE=yes`, takes them; `GLYD_DEGRADE=no` refuses); `serve` and the service never stop to ask and refuse without `--yes`.

`fit` reads the model's file headers by range requests (two small reads a shard, kept in the cache), so a model's sizes are exact before anything is downloaded; nothing of the weights is fetched to plan. The plan is what loads. For every family of `glyd-engine/src/support.rs` (once its files are here, the models that run over several GPUs too), `fit` sizes the model with the
functions `run` and `serve` do, from the same GPU's free memory: the weights, the cache for the context and the users, the engine's work, and a
small reserve the GPU keeps. Where `run` would refuse, `fit` refuses in the same words and exits 3. `run` prints its plan before it loads
(`Plan: ...`). `fit --json` prints the plan as `glyd-plan/1` for scripts: `level`, `context`, `users`, the bytes of each part, `kv_format`
(`bf16`, or `12-bit` where `GLYD_KV=lossless` asks for the lossless 12-bit pages and they reserve no more than bf16 for the plan; `fit` and `run` say which in their plan line, "keys and values in bf16" or "in 12-bit pages", and `run` and `serve` load what the plan says), and `experts`
(places a layer on the GPU, bytes in the computer's memory and on the disk) where a mixture of experts keeps them off the GPU.

The one size `fit` estimates is the lossless level's (penguin), which depends on how well the weights compress: 10.75 bits a weight
(12.04 on A100, H100 and B200). Measured on an RTX 4080 SUPER, Qwen3-8B loaded 0.2% above that and Qwen3-0.6B 2.3% below it. After a
model's first load, `fit` and `run` plan the GPU's memory with what that load took beyond its plan (up to 1 GiB).

## run

The first run downloads the model into the Hugging Face cache and packs it for this GPU (and makes the 12-bit KV tables where its plan has those pages: `GLYD_KV=lossless` asks for them). Later runs start
from the packed copy.

- `--prompt "TEXT"` gives one answer and exits. `--no-think` turns a thinking model's thinking off.
- `MODEL:LEVEL` picks a level: [Choosing a level](levels.md). `--context N` sets the context.
- The chat page is at `http://localhost:8000` while it runs: the same server as [serve](serve.md).
- A gated model needs a Hugging Face token: `glyd login`, or `HF_TOKEN` in the environment.
- The packed models are in `$GLYD_CACHE` (default `~/.cache/glyd`), the downloads in the Hub cache (`$HF_HOME`, default
  `~/.cache/huggingface`). Set either to put them on another drive.
