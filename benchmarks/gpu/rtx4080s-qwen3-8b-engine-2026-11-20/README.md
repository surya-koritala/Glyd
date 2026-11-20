# Qwen3-8B on the engine, RTX 4080 SUPER (2026-11-20)

Machine: glyd-pc, RTX 4080 SUPER 16 GB (736 GB/s), 60 GB RAM, driver reporting CUDA 13.2, Ubuntu.
Engine: `glyd-engine 0.32.1` as built by `gpu.yml` for sm_89 at 2955ac4a (main f742347 plus #271, which only changes
the multi-GPU paths), run inside `nvidia/cuda:13.0.3-devel-ubuntu24.04` with `--gpus all`. Model: Qwen/Qwen3-8B,
snapshot b968826d, level penguin (lossless).

```bash
glyd-engine fit Qwen/Qwen3-8B
glyd-engine serve Qwen/Qwen3-8B --port 8000
# then one chat request from the chat page (thinking on), and GET /stats
```

```
fit     penguin (10.8 bits), 10.3 GiB, 8192 tokens of context, ~50 tokens/s predicted; GPU 14.7 GiB free of 15.6
serve   10.3 GiB on the GPU instead of 15.3 GiB (bf16); packed in 12 s (12.4 GB), KV tables in 31 s, ready in 45 s,
        8192 tokens of context, 4 users at once
request prompt 21 tokens in 27 ms; 1,214 tokens decoded in 20,332 ms = 59.7 tokens/s (one user, /stats)
ceiling 10.3 GiB = 11.06 GB of weights a token / 736 GB/s = 15.0 ms = 66.5 tokens/s; measured 90% of it
        (KV cache reads not counted)
```

One run, one request. bf16 does not load on this card (15.3 GiB of weights against 14.7 GiB free), so there is no
bf16 speed to put next to it.
