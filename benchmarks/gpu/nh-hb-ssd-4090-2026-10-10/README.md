# Nemotron 3 Super at Hummingbird on one RTX 4090: the planner's fix and the SSD tier (2026-10-10)

`tools/cards/nh_offload_job.sh` (HB=1, `PACKED_S3`) on v030 01efcbd3: a pod of one NVIDIA GeForce RTX 4090 (24,564 MiB, sm_89, PCIe 4), **11 CPUs**
(9 threads for the experts), 57 GiB of memory for the pod, a European data center. 16 minutes of job (about $0.25). New since
benchmarks/gpu/nh-hb-cpu-4090-2026-10-10: the planner weighs the copy stream's backlog (27e18ef9), and the SSD tier (`--host-memory GIB`: the experts that
do not fit the budget stay in the packed directory and are read into slots of the host tier as the router asks for them, the page cache left alone).
Linear GLY-98, GLY-130.

| check | result |
|---|---|
| the tiny model with most experts on the disk (`nhoffload --host-memory`, copy, cpu, split and auto; 3, 5 and 8 places; cold, warm, hints, a predictor) | 99 checks a mode ok: every logit and token **to the bit** as with every expert in memory |
| the whole model, auto against every miss copied (97 teacher-forced positions) | top-1 the same at 100.0%, KL 0.00102 |
| the disk (`diskcheck`: 600 random experts of the directory) | **direct** reads (the page cache bypassed): 4.0 GB/s one at a time, **7.2 GB/s 16 at a time**, the same on a second pass |
| `run --prompt` | "The capital of France is Paris." (ready in 63 s) |

Decode, 255 greedy steps, one load a cache, cold at the start of each variant:

| experts in the computer's memory | GPU cache | mode | decode tok/s | 2nd half | prefill tok/s | disk a token |
|---|---|---|--:|--:|--:|---|
| all (39.2 GiB) | 24 GB card (190 a layer) | copy | 32.36 | 32.55 | 43.8 | — |
| | | cpu | 35.28 | 39.06 | 29.7 | — |
| | | split | 41.15 | 42.31 | 89.1 | — |
| | | **auto** | **43.64** | 43.84 | 93.9 | — |
| all | 16 GB card's budget (84) | copy | 22.86 | 23.01 | 43.0 | — |
| | | cpu | 33.83 | 35.22 | 33.5 | — |
| | | split | 33.63 | 33.59 | 88.5 | — |
| | | **auto** | **35.43** | 36.24 | 73.5 | — |
| **20 GiB (a 32 GB PC)** | 24 GB card | cpu | 17.17 | 20.85 | 15.1 | 71 reads, 150 MB, 52 ms |
| | | **auto** | **17.67** | 17.83 | 18.9 | 85 reads, 175 MB, 54 ms |
| **20 GiB** | 16 GB card's budget | **cpu** | **13.12** | 13.93 | 13.6 | 125 reads, 258 MB, 84 ms |
| | | auto | 12.52 | 12.25 | 18.8 | 128 reads, 264 MB, 85 ms |

Peak resident memory with 20 GiB for the experts: 20.4 GiB (24 GB card) and 20.5 GiB (16 GB budget): within a 32 GB PC.

How to read it:
- **The planner's fix makes auto, the default, the fastest mode** on this 11-CPU pod (43.6 against cpu's 35.3 on the 24 GB card, 35.4 against 33.8 at
  16 GB); before it auto ran 41.1 and 27.2 against cpu's 45.5 and 38.1 on a 28-CPU pod (nh-hb-cpu-4090-2026-10-10).
- **The SSD tier runs the model in a 32 GB PC's memory**, at 17.7 tokens a second on a 24 GB card and 13.1 at a 16 GB card's budget. The disk is the
  cost: 52 of a token's 56 ms on the 24 GB card. The drive gave about 3 GB/s of its 7.2: a layer's call reads about two experts, and the calls' reads
  wait on the work between them (no read ahead, no overlap with the CPU's other experts yet). That is the next work.
