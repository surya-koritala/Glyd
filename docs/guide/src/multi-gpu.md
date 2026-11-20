# Several GPUs and Starlings

Two things share the name "more than one GPU": the engine across the GPUs of one machine, and Starlings, which puts one
model across machines.

## The engine, one machine

`glyd-engine run` and `serve` use the GPU with the most free memory. A model too big for it (Kimi-K3, DeepSeek-V4,
GLM-5.3-Flash, Muse Spark, Nemotron-H at penguin) is split over all the GPUs the process can see, by their free memory.

```bash
CUDA_VISIBLE_DEVICES=0,1 glyd-engine serve MODEL                 # the GPUs to choose from
glyd-engine serve MODEL --split CUTS --devices GPUS              # your own split, see below
```

- Which families split, and at which levels, is the [Models](models.md) table, column "Weights split over GPUs".
- `--split CUTS --devices GPUS` is the layer cuts and the GPUs they go to. `fit MODEL` shows the placement the planner
  would choose. If the GPUs do not hold the model, `run` says how much it needs and stops (exit 3).
- With layers split, one GPU works at a time. On datacenter GPUs that reach each other directly (A100, H100, B200), Kimi-K3
  and DeepSeek-V4 put every layer across all of them, so every GPU works on every token. The start checks two gathers of known
  values and refuses where the GPUs' writes to each other arrive wrong.
- Measured on 8x A100 80 GB with NVLink, DeepSeek-V4-Flash decode: 54.6 tokens/s split by layers on 2 GPUs, 59.0 across 2,
  62.8 across 4, 58.3 across 8. It gains little on A100: past the products, a layer's small kernels and gathers take the same
  time on every GPU. Log: [research/tp-8xa100-2026-10-22](https://github.com/surya-koritala/Glyd/tree/main/research/tp-8xa100-2026-10-22).

## Starlings, one model across machines

Starlings is a beta. One model across two GPUs in one machine, or two machines on one LAN, NVIDIA on Linux, Qwen3.5-family
models. Each GPU holds a contiguous range of layers and the output is the same tokens the whole model gives on one card.

Build it (own workspace and pinned toolchain):

```bash
(cd starlings && cargo build --release)
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
- Two RTX 4090 machines 0.4-0.6 ms apart: 43.5 tokens/s split against 46.0 on one card; 51.9-53.8 with drafting
  ([starlings/benchmarks](https://github.com/surya-koritala/Glyd/tree/main/starlings/benchmarks)).
- One GPU for both halves works and is slower than the whole model: it proves the setup.

Starlings becomes `glyd cluster` in M5 ([ROADMAP](https://github.com/surya-koritala/Glyd/blob/main/ROADMAP.md)). It is not
part of the engine's `serve` yet.
