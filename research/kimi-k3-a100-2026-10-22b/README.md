# Kimi-K3 on 8x A100 80 GB: the device path, low-latency gathers, the fused layers (2026-10-22)

Lambda gpu_8x_a100_80gb_sxm4, us-east-1, $22.32/h, launched 21:36 EDT, terminated 22:43 EDT: 67 min, about $25. Branch a100-run (main
c5ca3d39 with #230, #241, #242 merged in). `go.sh` (launcher), `job.sh` (host), `inner.sh` (the CUDA 13 container), `req/` (the requests).
PREDICTIONS.md was written before the launch. The raw outputs (`out/`) are not in git ([benchmarks/README.md](../../benchmarks/README.md)).

## Against the predictions

| | predicted | measured | variance |
|---|---|---|---|
| decode, LL gathers, device path | 70.2 ms a token (14.2 tok/s) | 87 ms (11.5 tok/s), ±5% | +24% |
| found share, decode | 0.85 | 0.838 (the story's 511 steps) | −1.4% |
| a small gather at 8 GPUs, LL (1 x 896, graph) | ~5 µs | 8.8 µs | +76% |
| the same, regular | 19.5 µs | 15.4 µs | −21% |
| tpcheck on 2 GPUs: LL, regular, device path in copy mode | bit for bit | 41/41 each | — |
| answers | Paris; 48213; get_weather | all three | — |
| ready | ~35 min | 44m21s | +27% |

Decode is from the requests' times: 512 tokens in 47 s and 64 in 8 s, prompts of 43 and 33 tokens, so (47 − 8) / 448 = 87 ms. The steps are
logged in whole seconds, hence ±5%. The last 8-GPU run (before #237, #239, #241) did 118 ms.

## The formula with what the run measured: 86.4 ms, −0.7%

| term | predicted | measured inputs | what changed |
|---|---|---|---|
| weights and product starts | 22.7 ms | 22.7 | — |
| missing experts | 31.9 | 42.5 | the device path copies every miss (no host pairs: +8.1 ms); found 0.838, not 0.85 (+2.5) |
| exchanges | 3.4 | 9.0 | each gather at its size from tpgather (12.9 µs on average, not 4.8) |
| small kernels | 11.8 | 11.8 | — |
| total | 70.2 | 86.4 | measured 87 |

`ceiling` now times each gather at its size from a box's tpgather points (`--gathers ll|regular`); the device path is `--pair-ms 0`.

Not measured here: the product starts (12.7 µs) and the small kernels (11.8 ms a token) are the 4080's times 2.3. They could be off in ways
that cancel; the next calibration measures them on the A100 directly (a few real layers, no full load).

## The gathers (tpgather, µs in the step graph, GPU 0)

| a GPU's slice | 2 GPUs LL / regular | 4 GPUs | 8 GPUs |
|---|---|---|---|
| 1 x 896 | 3.6 / 7.9 | 5.6 / 10.3 | 8.8 / 15.4 |
| 1 x 4608 | 5.5 / 9.8 | 10.5 / 15.2 | 20.9 / 24.1 |
| 16 x 768 | 6.4 / 12.9 | 13.3 / 21.7 | 22.7 / 32.8 |
| 64 x 4608 | 48.2 / 45.2 | 98.2 / 98.2 | 202.9 / 200.3 |
| 1024 x 4608 | 227.4 / 228.6 | 482.4 / 483.0 | 1224.9 / 1210.9 |

LL saves 40-45% on one token's small gathers and nothing on large ones.

## The expert cache's policy against an ideal one (dcreplay, the prompts' 2,701 tokens)

| places a layer | CLOCK (dc_plan) | ideal (Belady) | CLOCK's layers |
|---|---|---|---|
| 306 (this run) | 0.900 | 0.959 | 0.688 to 0.983 |
| 400 | 0.938 | 0.976 | |
| 500 | 0.964 | 0.986 | |

The decode found 0.838: the prompts (the pass key's filler text above all) route more alike than a story does. Decode traces need the
device path to log its steps; the next run should record them.

## Cost

67 min: boot 8, download 10, checks 1, load 44, requests 3. The model's load and download were 81% of it.
