# Nemotron 3 Super at Hummingbird on one RTX 3090, the CPU computing the missed experts (2026-10-10)

`tools/cards/nh_offload_job.sh` (HB=1, `PACKED_S3`) on v030 f6fc224b (before the planner's backlog fix, 27e18ef9): a pod of one NVIDIA GeForce RTX 3090
(24 GB, sm_86, PCIe 4), 32 CPUs, 116 GiB of memory. 17 minutes of job (about $0.15). The model is the directory the 4090 run packed and sent to the
private bucket (benchmarks/gpu/nh-hb-cpu-4090-2026-10-10: Nemotron 3 Super at the Hummingbird level, 48.7 GB), taken whole: 277 s at 175 MB/s, no
checkpoint download and no pack. Linear GLY-98, GLY-130.

| step | result |
|---|---|
| the tiny model through the cache (`nhoffload`), copy and cpu, split, auto | ok in every mode |
| verify (`glyd-gpu verify`) | every pack decodes to glyd.json's sha256 |
| load | 20.8 GiB on the GPU; 192 of 512 experts a layer there; 39.2 GiB of experts in host memory, 0.0 GiB more for the CPU half |

Decode, 255 greedy steps of the same prompt as the 4090's, one load, the cache cold at the start of each variant (`generate-card.log`):

| mode | decode tok/s | second half | prefill tok/s | reads on the GPU | missed a token: CPU / copied |
|---|--:|--:|--:|--:|---|
| copy | 23.18 | 22.96 | 35.3 | 73.9% | 0 / 229.7 |
| **cpu (30 threads)** | **37.01** | **38.41** | 47.2 | 72.5% | 242.1 / 0 |
| cpu (8 threads) | 27.34 | 28.97 | 20.5 | 72.5% | 242.1 / 0 |
| cpu (16 threads) | 32.86 | 35.29 | 36.7 | 72.5% | 242.1 / 0 |
| cpu (30, pinned to cores) | 29.96 | 32.00 | 31.7 | 72.5% | 242.1 / 0 |
| split | 31.59 | 31.64 | 80.3 | 74.9% | 113.6 / 107.4 |
| auto | 32.46 | 32.85 | 75.6 | 74.9% | 124.0 / 97.1 |
| split (16) | 32.37 | 32.77 | 78.3 | 77.1% | 114.3 / 87.7 |
| auto (16) | 31.26 | 31.14 | 76.8 | 75.3% | 131.8 / 85.9 |

| check | result |
|---|---|
| the CPU mode against the same model with every miss copied (`logits-compare.log`: 64 tokens prefilled, 96 steps teacher-forced) | top-1 the same at 100.0% of 97 positions, KL 0.00060 nats on average (worst 0.0147) |
| `run --prompt` | "The capital of France is Paris." (ready in 1 min 21 s) |

How to read it:
- **The CPU half lifts the 3090 from 23.2 to 37.0 tokens a second** (38.4 over the second half of the steps), as it lifts the 4090 from 22.2 to 45.5.
- **This CPU wants every thread**: 8 threads give 27.3 and 16 give 32.9 (the 4090 pod's CPU was within 6% at 8), and keeping the crew to cores costs
  19% here. The default (the pod's CPUs less two, not pinned) is the right one on both.
- auto and split are this build's, before the planner weighed the copy stream's backlog (27e18ef9); they copy about 100 misses a token and run 32 against
  cpu's 37 in decode, 76 to 80 against 47 in prefill.
