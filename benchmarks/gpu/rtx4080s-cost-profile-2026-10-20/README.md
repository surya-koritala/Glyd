# Cost profiles and the tier predictor on an RTX 4080 SUPER, 2026-10-20

GLY-236 on the owner's PC (Ryzen 9 7950X3D, 60 GiB, RTX 4080 SUPER 16 GB, driver 595.91.07): `glyd-engine costs` of Qwen3.5-9B at linnet,
kestrel and penguin from their packed directories, `glyd-engine rates` on this machine, and the predictor (a token's time is the sum over the
memory tiers of the bytes it reads there over the tier's rate) against the engine's own bench and against the dynamic engine served. Nemotron 3
Super at hummingbird: the profile from the Hub's safetensors headers (nothing downloaded) and the predictor against the measured decode of the
2026-10-10 records, each pod's tiers taken from its record. Built from commit 3f634598 (build.txt). `record.sh` is the whole run, under the GPU
lock; every file below is its output as it came. docs/cost-profile.md has the fields. (The command and the flag were `tiers` and `--tiers` when this ran and were renamed `rates` and `--rates` before the merge; the JSON key `tiers` is unchanged.)

## Predicted against measured

This machine (`tiers.json`, 512 MiB buffers, 10 passes each): the GPU's memory 697.1 GB/s streamed by a kernel, 312.1 copied within it, **628.1 GB/s
as the engine's own products read the linnet directory's first layers' packs** (90% of the stream); the link 26.9 GB/s from pinned memory (27.1 back,
21.6 from pageable); the computer's memory 74.5 GB/s on 32 threads (51.9 on one); the disk under the Hub cache 6.78 GB/s in 2 MiB direct reads 16 at a
time (3.99 one at a time). The prediction takes the GPU's memory at the product rate, the keys and values of 1,152 positions (the prompt and half the
output) and the states read and written. One user, `--ctx 4096`, a 1,024-token prompt, 256 tokens; another process held 2,226 MiB of the GPU throughout.

| model, level | bytes a token from the GPU's memory | predicted | `glyd-engine bench` (the hand-written Qwen3.5 engine) | served (the dynamic engine, timed by the client) | predicted vs bench | the same prediction at the 697.1 GB/s stream |
| :--- | --: | --: | --: | --: | --: | --: |
| Qwen3.5-9B linnet | 5,637,877,620 | **111.41 tok/s** (8.976 ms) | **106.89 tok/s** (9.356 ms mean, p99 9.365; prefill 216.9 ms) | 107.16 tok/s (2.389 s for 256) | **+4.2%** | 123.7 (+15.7%) |
| Qwen3.5-9B kestrel | 6,629,044,596 | **94.76 tok/s** (10.553 ms) | **92.64 tok/s** (10.794 ms, p99 10.803; prefill 218.4 ms) | 92.89 tok/s (2.756 s) | **+2.3%** | 105.2 (+13.5%) |
| Qwen3.5-9B penguin (glyd-v4) | 10,825,132,869 | 58.03 tok/s (17.234 ms) | not run: the engine's Qwen3.5 path does not read a glyd-v4 directory (costs-penguin-loaded.err, bench-penguin.err: `no model.language_model.layers.0...weight`; serve went to the lossless route, which refuses the family) | not run | — | 64.6 |

The profile against what the engine allocated (`costs --load`, and the served `/v1/glyd/info`, `allocated` beside `memory`):

| | profile (the files) | allocated (the engine's counters) |
| :--- | --: | --: |
| linnet: weights | 6,200,345,704 | 6,200,345,608 (96 bytes less: one 4-byte block base per linear-attention layer, where two saved packs are joined at load) |
| linnet: KV at 4,096, a token; state; working memory; engine | 134,217,728; 32,768; 51,511,296; (unavailable before a load); 6,386,074,728 + working | 134,217,728; 32,768; 51,511,296; 601,399,076; 6,987,473,708 (the same bytes as the 2026-10-19 serve record; nvidia-smi 7,002 MiB) |
| kestrel: weights; engine | 7,318,652,008; 7,504,381,032 + working | 7,318,651,912; 8,105,780,012 (working 601,399,076) |

Nemotron 3 Super at hummingbird (`costs-nemotron-hummingbird.json`, the Hub's headers, 7 s): 48,681,614,344 bytes of tensors (the pack's directory is
48,744,769,593), the routed experts 42,108,272,640 (the record's 42.1 GB), a mean expert of 2,056,068 bytes (3-bit ones 2,494,464, 2-bit 1,763,328;
the record's "2.1 MB on average"), 40 mixture layers of 512 experts, 22 a token (880 reads a token, the records' 224,400 over 255 steps), 8 attention
layers (KV 8,192 bytes a token), 40 Mamba-2 layers (states 170,229,760 bytes a session: the record's 0.159 GiB), 3.22 bits a weight. A token reads
7,944,380,116 bytes: 6,135,040,276 of the dense part (attention, Mamba-2, the latent projections, the shared experts, the routers, the embedding's row,
the head) and 1,809,339,840 of experts. The predictor against the 2026-10-10 records' decode (255 greedy steps after a 42-token prompt), each pod's
tiers from its record (`tiers-*.json`: the links and the host rates measured there, the GPU's memory at NVIDIA's figure, 1,008 GB/s for the RTX 4090
and 1,792 for the 5090, which no record measured), the GPU's share of the experts and the cache's hit rate as recorded:

| record, pod | mode | GPU experts a layer, hits | tiers the prediction took | measured tok/s | predicted (the serial sum) | error | the overlapped variant |
| :--- | :--- | :--- | :--- | --: | --: | --: | --: |
| nh-hb-4090 (pod A, 22 CPUs) | misses copied | 190 of 512, 77.9% | link 24.8 GB/s | 32.39 | 41.08 | +26.8% | 41.08 |
| nh-hb-cpu-4090 (pod B, 28 CPUs) | misses copied | 190, 77.9% | link 13.0 | 22.19 | 25.65 | +15.6% | 25.65 |
| pod B | misses computed by the CPU (26 threads) | 190, 72.3% | the memory streamed, 44.8 | 45.46 | 52.88 | +16.3% | 89.39 |
| pod B | the same | the same | the memory as the expert kernel reads it, 33.3 | 45.46 | **43.91** | **-3.4%** | 66.44 |
| pod B, a 16 GB card's budget | misses copied | 84, 60.7% | link 13.0 | 14.48 | **15.89** | **+9.7%** | 15.89 |
| pod B, 16 GB | computed by the CPU | 84, 54.7% | streamed, 44.8 | 38.12 | **38.91** | **+2.1%** | 54.66 |
| pod B, 16 GB | the same | the same | the kernel's, 33.3 | 38.12 | 31.23 | -18.1% | 40.63 |
| nh-hb-ssd-4090 (pod C, 11 CPUs), 20 GiB of experts in memory | CPU, the rest from the SSD | 190, 72.3% (pod B's) | memory 117.9, SSD 6.9 (16 reads at once) | 17.17 | **15.90** | **-7.4%** | 16.14 |
| pod C, 16 GB | the same | 84, 54.7% | the same | 13.12 | **12.99** | **-1.0%** | 13.49 |
| nh-offload-4090 (pod E), wren (sizes estimated: no wren pack is published) | misses copied | 123, 67.7% | link 12.6 | 11.95 | **12.58** | **+5.3%** | 12.58 |
| nh-offload-5090 (pod D), wren (estimated) | misses copied | 190, 77.9% | link 50.3 | 43.59 | 58.15 | +33.4% | 58.15 |

How to read it:

- **The dense model on this card is predicted within 5% with the GPU's memory at the rate the engine's products read it** (628 GB/s, `vram_product`):
  +4.2% and +2.3% against the engine's bench, +4.0% and +2.0% against the served dynamic engine (the two engines are within 0.3% of each other). At the
  streaming read (697 GB/s) the same model over-predicts by 14 to 16%: the ceiling, not the engine. `tiers` gives both; the predictor takes the product
  rate where the directory is there to measure it.
- **The offloaded Nemotron runs are predicted within 10% where the slow tier is the time**: the SSD tier (-7.4%, -1.0%), the CPU mode at a 16 GB card's
  budget (+2.1% with the host's streaming rate), the CPU mode on the 24 GB card with the memory at the expert kernel's own rate (-3.4%; at the streaming
  rate +16%), the 16 GB copy mode (+9.7%) and the wren copy mode on the slow link (+5.3%, from estimated sizes).
- **The copy mode on a fast link is over-predicted by 16 to 33%.** The link's term is right (pod A: 0.400 GB a token at 24.8 GB/s is 16.1 ms; the record's
  own copying time is 16.3 ms a token). The GPU's term is not: 8.2 ms at 1,008 GB/s against the 14.6 ms a token the record leaves for the rest of the
  step. Two things the model has no term for: the 4090's products read below NVIDIA's figure (this card's read 90% of it, and that is a dense product;
  the expert cache's run on the old Nemotron engine is slower still), and the cache's per-layer round trip to the host (the router's choices read on the
  CPU before the layer's experts run). A measured product rate on that card, as `tiers --dir` gives here, and a per-layer host cost would close it; both
  need the second machine.
- The host's rate for the CPU mode is the question the two pod-B rows leave open: the kernel's rate (33.3 GB/s) fits the 24 GB card, the streaming rate
  (44.8) fits the 16 GB budget; the CPU's share of the step is 60% in one and 76% in the other, and the GPU's work overlaps it in part. The `overlapped`
  variant (the GPU under the CPU) is far too fast everywhere: the engine does not overlap them that well. The serial sum is the better model of the two.
- KV and state bytes are the config's formula and the engine allocates exactly them (the table above); the working memory is measured only (601 MB here
  for both levels: the same buffers), null in a profile without a load.

## Files

| file | what |
| :--- | :--- |
| record.sh | the run |
| build.txt | the engine's and the GPU library's SHA-256, the commit, rustc, nvcc, the kernel, the CPU, the driver, the memory |
| tiers.json, tiers.err | this machine's tiers: the GPU's memory (read, copy, the engine's products over the linnet directory's first layers), the link (pinned, pageable, both ways), the computer's memory (32 threads, one), the disk under the Hub cache (16 reads at once, one) |
| costs-LEVEL.json | the profile of Qwen3.5-9B at the level with the predictor's rows from tiers.json (`--predict --rates tiers.json --ctx 4096 --prompt 1024 --new 256`) |
| costs-LEVEL-loaded.json | the same model loaded on the GPU by the dynamic engine (`--load`): `allocated` is the engine's own counters |
| bench-LEVEL.txt, bench-LEVEL.err | `glyd-engine bench DIR --prompt 1024 --new 256 --ctx 4096 --reps 5`: the hand-written Qwen3.5 engine, each decode step timed on the host |
| info-LEVEL.json, info-memory-LEVEL.json | `/v1/glyd/info` of `glyd-engine serve DIR --ctx 4096`: the `costs` object as served (with `allocated`), its totals beside `memory` |
| serve-decode-LEVEL.json, serve-stream-LEVEL.txt, tokenize-LEVEL.txt | a streamed plain completion of 256 tokens after the 1,024-token prompt through the served dynamic engine, timed by the client between the first and the last token; the stream; the prompt's count |
| costs-nemotron-hummingbird.json | Nemotron 3 Super at hummingbird (glyd/nemotron-3-hummingbird) from the Hub's headers |
| predict-nemotron-NAME.json | the predictor's rows for each recorded pod and mode (`--rates tiers-*.json --placement ... --gpu-experts ... --hit ...`) |
| tiers-rtx4090-pod*.json, tiers-rtx5090-podD.json | the recorded pods' tiers, written by hand from each record's own lines (the source line in each) |
| predict-nemotron-wren-*.json | the wren level (the experts four-bit, the rest six-bit) sized from the base model's config alone (no wren pack is published): basis estimated |
| steps.txt, nvidia-smi.txt, nvidia-smi-full-before.txt, server-LEVEL.err, status-LEVEL | the steps with their times, the GPU at each step, the servers' own lines |

## Known limits

- One machine measured with the new code (this one). The second is pending (below). The Nemotron rows are the 2026-10-10 records' measurements against
  predictions made now from the same pack's sizes; their GPU rates are NVIDIA's figures, their hit rates the records'. The 5090 row and the wren rows use
  sizes estimated from the base model's config (the wren pack is not published: `glyd/nemotron-3-wren` is 404 on the Hub), at fit::bits a weight.
- The first run of record.sh (23:18 Eastern) inherited GLYD_GPU_LIB from the build environment (an older library, `no function 7c63f439`): every GPU
  step but `tiers` failed in seconds. Its tiers agreed with this run's (697.28 GB/s read, 26.90 link). Not kept; this is the corrected run (23:47).
- `bench` is the hand-written Qwen3.5 engine, `serve` the dynamic one; the served figure is timed by the client between the first and the last token's
  event (HTTP and the sampler included), so the 0.3% between them is within that.
- The penguin profile is of a glyd-v4 directory the engine does not run; basis `file` all the same (its embedding is held in bf16 there).
- The Hummingbird mean expert is the plain mean of a layer's 512; the 3-bit experts are the most-used, so a token's real mean is a little above it.
- Prefill is not predicted (the bench's prefill: 216.9 ms for 1,024 tokens at linnet, 218.4 at kestrel).
- The test suite (`cargo test --release -p glyd-engine --lib`, 399 tests) passed on this machine after the record (tests.log, not kept: 4.2 s).

## The second machine

Pending: a rented consumer card (the owner's go). The same script, on the pod, with its own paths:

```
git clone <the private repo> && cd <it> && GLYD_SOURCE_COMMIT=$(git rev-parse HEAD) cargo build --release -p glyd-engine     # or the release wheel's binary
glyd-engine fetch glyd/Qwen3.5-9B-linnet                                                        # and kestrel, penguin: the snapshots into HF_HUB_CACHE
OUT=/root/rec ENGINE=/root/glyd/target/release/glyd-engine HUB=/root/.cache/huggingface/hub GLYD_GPU_LIB=/root/glyd/target/release/libglyd_gpu_cuda13.so \
  COMMIT=$(git rev-parse HEAD) benchmarks/gpu/rtx4080s-cost-profile-2026-10-20/record.sh      # no lock on a pod of one's own
```

Then the predicted-vs-measured table as here: `costs-LEVEL.json`'s `predictions[0].tok_s` (and `tiers.rates.vram_product`) against `bench-LEVEL.txt`'s
`decode_tokens_per_s` and `serve-decode-LEVEL.json`'s. For Nemotron on a 24 GB card (hummingbird fits with the expert cache): `glyd-engine run
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:hummingbird` is the measurement (`tools/cards/nh_offload_job.sh` is the pod script of the records), and
`glyd-engine costs glyd/nemotron-3-hummingbird --predict --rates tiers.json --placement experts_ram_cpu --gpu-experts SLOTS/512 --hit HITS` the prediction,
with `tiers.json` from `glyd-engine rates --dir <the packed directory>` on that pod (its `ram_read` is the streaming bound; the CPU kernel's own rate is
`glyd-engine cpuexperts`'s, a development command).
