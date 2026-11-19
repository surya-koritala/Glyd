# Every layer across the GPUs, 8x A100 80 GB (2026-10-22)

Lambda gpu_8x_a100_80gb_sxm4 (us-midwest-1, NVLink between every pair: NV12), $22.32/h, launched 10:04 EDT. Branch gly-tp (GLY-358): each GPU holds a
row slice of every big matrix and of every expert, gathers the slices GPU to GPU after each product (kernels/tp.cu), and runs the small operations itself.
The layer split (main) has one GPU at work at a time. `job.sh` (host), `inner.sh` (the CUDA 13 container), `poll.sh`/`run.sh` (launch, upload, pull,
terminate; run.sh stopped by hand after the DeepSeek runs to add `extra.sh`, the instance terminated by hand at 13:21); `out/`, `out2/`. About $73.

## The gathers over NVLink (tpcheck, random models, two GPUs against one)

| model | | passes the same to the bit |
|---|---|---|
| Kimi-K3, experts resident | decode + 24-token prompt | 25 of 25 |
| | chunks of 8, 40 tokens | 41 of 41 |
| Kimi-K3, expert cache (4 of 16 placed), copies alone | chunks of 8 | 41 of 41 |
| the same, the host alone | chunks of 8 | 41 of 41 |
| DeepSeek-V4, every matrix FP8 | decode + prompt | 25 of 25 |
| | chunks of 16 | 41 of 41 |

## DeepSeek-V4-Flash, `glyd serve`

| | ready | 512-token story | tools | its own tools example |
|---|---|---|---|---|
| split by layers on 2 GPUs (main) | 151 s | 398 tokens in 7 s (57 tok/s) | `get_weather {"city": "Paris", "unit": "c"}` | `get_weather {"location": "Beijing"}` |
| across 2 GPUs | 160 s | 401 in 8 s (50) | the same | the same |
| across 4 | 145 s | 352 in 6 s (59) | the same | the same, with "unit": "celsius" |
| across 8 | 161 s | 323 in 6 s (54) | the same | the same |

(A request's time in whole seconds.) `tpbench`, decode steps with a graph after the first run, the build with a gather of a decode step's few vectors in
one block (`extra.sh`, `out2/`):

| | a step | tokens/s |
|---|---|---|
| split by layers on 2 GPUs | 18.32 ms | 54.6 |
| across 2 | 16.96 ms | 59.0 |
| across 4 | 15.92 ms | 62.8 |
| across 8 | 17.17 ms | 58.3 |

On A100 DeepSeek gains little: past its products, a layer's small kernels, launches and gathers take the same time on every GPU, and a gather's own
time grows with the GPUs (below).

## A gather's own time (`tpgather`, in a graph)

| GPUs | 1 x 896 a GPU | 1 x 4,608 | 16 x 768 | 64 x 4,608 | 1,024 x 4,608 |
|---|---|---|---|---|---|
| 2 | 8.9 us | 11.0 | 14.2 | 45.8 | 239 |
| 4 | 12.9 | 17.8 | 23.6 | 100 | 491 |
| 8 | 19.5 | 26.5 | 32.9 | 204 | 1,227 |

One system fence a block in place of one a thread: 8.0 / 15.5 us at 2 / 8 GPUs for the smallest (measured beside Kimi-K3's load). A Kimi-K3 layer
gathers about ten times (each sliced product's output), DeepSeek-V4's about eight: on 8 GPUs some 20 ms of a Kimi-K3 token and most of what DeepSeek
gains. Next: fewer exchanges a layer, the attention split by heads and the down products by their columns, one sum a block in place of a gather a
product (the sums' order then differs from one GPU's, as another GPU's split of K does).

## Kimi-K3, `glyd serve --context 32768`, the expert cache

| | across 8 GPUs | split by layers (main, #223/#224) | Oct 22 night, split (#219-#222) |
|---|---|---|---|
| ready | 54m08s (see the load below) | 28m13s | 34m34s |
| places a layer | 306 of 896 (each GPU an eighth of every place) | 314-425 | 318-430 |
| 512 tokens (its reasoning, a coherent story) | 60 s: **8.5 tokens/s** | 104 s: 4.9 tokens/s | 128 s: 4.0 |
| 64-token windows in decode | 6.8-7.7 s | | 14.7-16.4 s |
| expert reads found in places | 82.6-88.3% | 48-75% | 64-75% |
| a copy of an expert | 167-194 us (each GPU its slice over its own link) | | 700-840 us |
| chat | "The capital of France is Paris." | the same | the same |
| tools, tool_choice required | `get_weather {"city": "Paris", "unit": "c"}` | the same | the same |
| tools; the 2,472-token pass key | failed: "linear: invalid argument" (below) | the call; "The pass key is 48213." | the same |

Across the GPUs, a copy is a quarter of what it was, so the plan copies most misses (a window's 10-15 thousand copies, under 1,500 pairs on the
host) and the cache holds more of what the router asks for. The products of the bf16 matrices read an eighth of each matrix on each GPU.

The two failed requests were the prompts of more than 64 tokens: the experts' scratch sized its workspace for the shared expert's products by the
experts' layout, where the shared expert is in the 12-bit one (#224); sliced, its products at a prompt chunk's rows asked for more. Fixed in the branch
(either layout's) and run again across the 8 GPUs (`extra.sh`, `out2/`, ready in 43m11s off a disk again):

| | across 8 GPUs, the fix | split by layers |
|---|---|---|
| tools | `get_weather {"city": "Paris", "unit": "c"}`, with its reasoning | the same |
| the 2,472-token pass key | "The pass key is 48213." in 45 s | the same in 116 s |
| 512 tokens | 64 s: 8.0 tokens/s (windows 7.2-8.5 s) | 104 s: 4.9 |

The pass key's prompt went 2.6 times as fast: its chunks copy most of each layer's experts, an eighth of each over each GPU's link.

The load: #224 dropped the checkpoint from the page cache before the load, and this box's disk (a 19 TB virtio volume) reads about 0.5 GB/s: 1.35 TB of
experts read again from it, and every core busy packing them (a profile on the PC: 46% in the FP4 pack's check, 25% in the pack). The branch drops each
expert's bytes once read instead, and packs an expert's matrix 10x faster (a tile a step, encoded and checked; the same bytes): the split run's load,
with that build, took 28 minutes.
