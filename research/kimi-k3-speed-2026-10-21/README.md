# Kimi-K3 speed with the host's experts, 8x A100 80 GB (Lambda us-midwest-1), 2026-10-21

20:44-21:58 EDT, $22.32/h, about $27. AMD EPYC 7J13: 2 sockets, 240 cores, AVX2. PR #219's build, expert cache in auto mode, `GLYD_KIMI_STATS`.

| | |
|---|---|
| plan | 573.9 GiB on the GPUs; 318 to 431 places a layer (equal places; was 290 to 815) |
| load | ready in 38m26s, including a 7.4 min drop of the page cache (below) |
| chat | "The capital of France is Paris." (64 tokens, 31 s, cold cache) |
| 512-token story | 162 s: 3.2 tokens/s (2.1 in the session before, without the host's experts) |
| tools, tool_choice required | `get_weather {"city": "Paris", "unit": "c"}` both |

The cache over the story's first 64 tokens, per stage: 75-97% of expert reads hit; 122 ms of PCIe copying per token (24.8 GB/s); 1-3 experts per call
computed on the host at 3.6-4.0 ms each, against 0.69 ms per copy, so auto mode mostly copies. In the prompt passes the host did much more
(2,500 experts per stage at 0.5-1.5 ms).

fp4bench (dev) on this box, one Kimi expert (16.8 MB): 4.2 ms on 240 threads (the crew the server used), 0.6-1.0 ms on 32-64, 0.48 ms per pair at
3 pairs on 64 pinned threads. The PC (7950X3D): 0.29 ms on 16 pinned threads. So: crew of at most 64, pinned (ba2c4886), and the host side is a gain
beside the copies rather than instead of them.

Load: kswapd and kcompactd at 100% reclaiming the 1.5 TB page cache of the download while 1.35 TB is pinned; dropping the cache took 7.4 min of system
time and did not speed the rest. perf during the load: 42% in TLB-shootdown interrupts (each layer's 15 GB joined buffer, its copy into pinned memory,
the per-expert reads' buffers). Next: pack straight into the pinned host tier.

DeepSeek-V4-Flash on DeepSeek's own tools example (system prompt, two tools, thinking): the reasoning degenerates ("c = 0; d = 0; ...") to the
length limit. Cause found after the session: prompt chunks of more than 64 rows went wrong in the grouped output product (serve prefills 256
at a time); fixed in #219 (GLY-354).
