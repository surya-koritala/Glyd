# DeepSeek-V4-Flash and Kimi-K3 on 8x A100 80 GB, after #219-#222 (2026-10-22)

Lambda gpu_8x_a100_80gb_sxm4 (us-midwest-1; no 8x H100 capacity), $22.32/h, launched 00:11 EDT. AMD EPYC 7J13, 2 sockets of 120 cores, 2 NUMA
nodes of 907 GB. `job.sh` (host), `inner.sh` (the CUDA 13 container), `dltest.py`; `req/` the requests; `out/` everything the box wrote.

## Downloads (GLY-356), before any other traffic

| | one connection | many |
|---|---|---|
| curl through the Hub's bridge (resolve: what glyd reads) | 155 MB/s | 1,061 MB/s (16) |
| curl straight from the xorbs (what hf_xet reads) | 68 MB/s | 915 MB/s (16) |
| glyd's client (`dlrate`) | 151 MB/s | 682 MB/s (8), 1,188 MB/s (32) |

glyd's client is curl's rate. Oct 21's 2.3 MB/s a connection was measured beside `hf download`. `hf download` then: DeepSeek-V4-Flash in
93 s, Kimi-K3 in 579 s (1.56 TB, about 2.7 GB/s).

## DeepSeek-V4-Flash (GLY-354), `glyd serve`, ready in 1m55s

After #219 (prompt chunks past 64 rows), every tool prompt returns its call:

- DeepSeek's own tools example: `get_weather {"location": "Beijing"}`, with its reasoning. Before: degenerate ("c = 0; d = 0; ...").
- tools, no thinking: `get_weather {"city": "Paris", "unit": "c"}` as an OpenAI tool call. Before: JSON text after the answer.
- tools with thinking, tool_choice required: the same call.
- the 512-token story request: 398 tokens, coherent, about 10 s.

## The host's FP4 experts on the 7J13 (fp4bench, one Kimi expert, 16.8 MB)

| threads | 1 pair | 3 pairs (a pair) |
|---|---|---|
| 16 | 963 us | 884 us |
| 32 | 728 us | 710 us |
| 64 | 688 us | 616 us |
| 120 | 1,011 us | 815 us |
| 60 on node 0, memory on node 0 | 631 us | 506 us |
| 60 on node 0, memory on node 1 | 593 us | 511 us |

Remote memory costs nothing here, and 64 threads are only 1.4 times 16: at 16 threads about 1.4 cycles a weight, but a call of one to three
pairs gives each of 64 threads one or two row blocks, and the crew's wake-up and barriers weigh as much. The AVX2 kernel also held 64
accumulators in AVX2's 16 registers, so they spilled (fixed in the PR with these notes, the same bits: 6-15% on the PC). A pair on the host is
about a PCIe 4 copy (0.69 ms) and slower than the H100's PCIe 5 one (0.32 ms): beside the copies, not instead of them.

## Kimi-K3, `glyd serve --context 32768`

| | this run | Oct 21 |
|---|---|---|
| ready | 34m34s | 38m26s |
| plan | 575.0 GiB on the GPUs, 318-430 places of 896 a layer, 1,347 GiB of experts in host memory | the same |
| chat | "The capital of France is Paris.", 64 tokens in 25 s (cold cache) | |
| 512-token story | 128 s: **4.0 tokens/s**; 64-token windows 14.7-16.4 s | 162 s: 3.2 tokens/s |
| tools, tool_choice required | `get_weather {"city": "Paris", "unit": "c"}` both | the same |
| pass key, 2,472-token prompt, attention in tiles of 512 (GLYD_MLA_TILE) | its reasoning finds "The pass key is 48213." (117 s with the prompt) | |

The decode's gain is the host's crew of 64 pinned threads (#219): auto mode now gives the host most misses, 0.47-0.67 ms a pair against
0.70-0.84 ms a copy; a window's stage 0, for example, 2,437 experts on the host and 1,570 copied. Per token, about 98 ms of copying and 96 ms
of host experts side by side across the 8 stages, and 128 ms of the host waiting on the GPUs.

The early guess (#221), 0, 1 and 2 copies a layer in turn per window: no gain. Its copies were mostly experts the router then did not choose,
taking PCIe time from the real misses; 0 is the default now.

| early copies | window | hits | copied after the router | guessed |
|---|---|---|---|---|
| 0 | 14.7 s, 16.4 s | 75.0%, 64.4% | 1,137, 1,570 | 0 |
| 1 | 15.6 s, 15.1 s, 15.6 s | 78.5%, 76.6%, 75.6% | 943, 1,079, 1,111 | 684, 671, 692 |
| 2 | 15.6 s, 14.7 s | 75.8%, 80.1% | 362, 366 | 1,343, 1,157 |

The load (34.6 min) was mostly waiting on memory, not packing: between layers the process sat at 1 core while kswapd (82%) and kcompactd (95%)
reclaimed and compacted the download's 1.6 TB of page cache for each layer's 15 GB of pinned memory; packing a layer then took ~200 cores for
seconds. Next: the checkpoint dropped from the page cache before the load, each expert's bytes after they are read (posix_fadvise; in the PR with
these notes).

Not measured here (this box ran #219-#222): #223's depth mix and norm, and the 12-bit layout for Kimi on HBM GPUs (`../layout12-h100-2026-10-22`:
-22% a step on one H100).

About $24 (00:11-01:16 EDT; terminated by hand once the requests were done, a process holding 1.35 TB pinned takes ~10 min to exit).
