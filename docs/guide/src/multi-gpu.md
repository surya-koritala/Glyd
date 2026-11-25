# Several GPUs and Starlings

Two things share the name "more than one GPU": the engine across the GPUs of one machine, and Starlings, which puts one
model across machines.

## The engine, one machine

`glyd run` and `serve` use the GPU with the most free memory. A model too big for it (Kimi-K3, GLM-5.3-Flash,
DeepSeek-V4, Muse Spark) is split over all the GPUs the process can see, by their free memory.

```bash
CUDA_VISIBLE_DEVICES=0,1 glyd serve MODEL                 # the GPUs to choose from
glyd serve MODEL --split CUTS --devices GPUS              # your own split, see below
```

- Which families split, and at which levels, is the [Models](models.md) table, column "Weights split over GPUs". Qwen3.8-Flash-Next runs on the one GPU
  with the most room: its layers are not split over GPUs on the executor yet, so `--split` and a `--devices` of more than one are refused in words.
- `--split CUTS --devices GPUS` is the layer cuts and the GPUs they go to. `fit MODEL` shows the placement the planner
  would choose. If the GPUs do not hold the model, `run` says how much it needs and stops (exit 3).
- With layers split, one GPU works at a time. On datacenter GPUs that reach each other directly (A100, H100, B200), Kimi-K3
  and DeepSeek-V4 put every layer across all of them, so every GPU works on every token. The start checks two gathers of known
  values and refuses where the GPUs' writes to each other arrive wrong.
- To see a split before you rent the machines: `glyd fit zai-org/GLM-5.3-Flash --machine 8xh100` plans GLM-5.3-Flash at
  penguin over the eight GPUs of the preset. Measurements of the split are in
  [research/tp-8xa100-2026-10-22](https://github.com/surya-koritala/Glyd/tree/main/research/tp-8xa100-2026-10-22).

## Starlings, one model across machines

Starlings is a beta. One model across two GPUs in one machine, or two machines on one LAN, NVIDIA on Linux, Qwen3.5, 3.6 and 3.8
models. Each GPU holds a contiguous range of layers and the output is the same tokens the whole model gives on one card.

Build it (from the repository root; the binary is `target/release/starlings`):

```bash
cargo build --release -p starlings-cli
```

Run it, from the [beta guide](https://github.com/surya-koritala/Glyd/blob/main/starlings/docs/beta.md):

```bash
starlings up --model glyd/Qwen3.8-27B-swift                 # one machine, its GPUs
starlings up --model glyd/Qwen3.8-27B-swift --machines 2    # first machine: prints a join code
starlings join --model glyd/Qwen3.8-27B-swift               # second machine: enter the code, approve on the first
```

- It opens a web UI and an OpenAI-style `/v1/chat/completions` on loopback (port 9750).
- Two machines need one LAN with private addresses and TCP ports 9741 and 9761-9762 open from the second to the first, 9770-9772
  the other way. Each GPU needs about half the model free (the 27B: about 10 GB) plus 1 GiB.
- Measurements are in [starlings/benchmarks](https://github.com/surya-koritala/Glyd/tree/main/starlings/benchmarks).
- One GPU for both halves works and is slower than the whole model: it proves the setup.

Starlings becomes `glyd cluster` in M4 ([ROADMAP](https://github.com/surya-koritala/Glyd/blob/main/ROADMAP.md)). It is not
part of the engine's `serve` yet.
