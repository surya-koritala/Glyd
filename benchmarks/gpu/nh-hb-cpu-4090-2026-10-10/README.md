# Nemotron 3 Super at Hummingbird on one RTX 4090, the CPU computing the missed experts (2026-10-10)

`tools/cards/nh_offload_job.sh` (HB=1) on v030 96803245: a pod of two NVIDIA GeForce RTX 4090 (24 GB, sm_89, PCIe 4), the job on GPU 0 alone; 28 CPUs,
114 GiB of memory (124 GiB total), a European data center (`RP_DC`). 19 minutes of job (about $0.60). Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
at 2dc98e2a at the Hummingbird level (the hbr40 calibration's codes: 3 bits for each layer's 205 most-used experts, 2 for the other 307; every other
matrix six-bit), as benchmarks/gpu/nh-hb-4090-2026-10-10. New since that run: the expert cache's CPU half (`ExpertStore::run_groups`): a missed expert's
pairs are computed by the CPU from the pinned host tier in place (`cpu_hb`, no second copy: 0.0 GiB more memory), the host's sums added before R^T and
the one rounding; the planner chooses per miss (`--offload-mode copy|cpu|split|auto`). Linear GLY-98, GLY-130.

| step | result |
|---|---|
| the calibration's codes from the private bucket | 42.1 GB in 204 s (205 MB/s) |
| the tiny model through the cache (`nhoffload`), copy and cpu, split, auto on 4 threads | ok in every mode (cpu, split, auto: every logit within two places of bf16 of the largest) |
| pack, verify | 152 s; every pack decodes to glyd.json's sha256 |
| the packed directory to the private bucket (`PACKED_UP`) | 48.7 GB in 48 s: later pods take it whole (`PACKED_S3`) |
| load | 20.7 GiB on the GPU; 190 of 512 experts a layer there; 39.2 GiB of experts in host memory, 0.0 GiB more for the CPU half |

Decode, 255 greedy steps of the same prompt, one load, the cache cold at the start of each variant (`generate-card.log`, `generate-16g.log`):

| cache | mode | decode tok/s | second half | prefill tok/s | reads on the GPU | missed a token: CPU / copied |
|---|---|--:|--:|--:|--:|---|
| 24 GB card (`auto`: 190 a layer) | copy | 22.19 | 22.33 | 23.8 | 77.9% | 0 / 194.2 |
| | **cpu (26 threads)** | **45.46** | **50.03** | 49.2 | 72.3% | 244.1 / 0 |
| | cpu (8 threads) | 42.75 | 47.13 | 34.2 | 72.3% | 244.1 / 0 |
| | cpu (16 threads) | 45.19 | 50.02 | 48.2 | 72.3% | 244.1 / 0 |
| | cpu (26, pinned to cores) | 45.22 | 50.12 | 49.0 | 72.3% | 244.1 / 0 |
| | split | 39.48 | 41.34 | 80.3 | 75.8% | 164.7 / 48.6 |
| | auto | 41.13 | 43.57 | 79.7 | 75.7% | 178.7 / 35.2 |
| | split (16) | 39.85 | 41.82 | 79.8 | 75.8% | 167.2 / 46.1 |
| | auto (16) | 41.83 | 44.90 | 82.3 | 75.5% | 184.7 / 31.3 |
| **16 GB card's budget** (`auto:15`: 84 a layer) | copy | 14.48 | 14.59 | 23.7 | 60.7% | 0 / 345.4 |
| | **cpu (26)** | **38.12** | **39.38** | 52.9 | 54.7% | 399.0 / 0 |
| | split | 28.70 | 29.38 | 82.9 | 56.8% | 301.3 / 78.7 |
| | auto | 27.19 | 26.83 | 83.0 | 50.5% | 355.0 / 80.4 |

Correctness of the whole model (`logits-compare.log`, `logits-check.log`: 64 tokens prefilled, 96 steps teacher-forced, eager):

| check | result |
|---|---|
| the CPU mode against the same model with every miss copied | top-1 the same at 100.0% of 97 positions, KL 0.00077 nats on average (worst 0.0095) |
| the CPU mode against the bf16 checkpoint | top-1 96.9% of 97 positions, KL 0.0139 nats on average (90th position 0.0168) |
| `run --prompt` | "The capital of France is Paris." (ready in 58 s) |

How to read it:
- **The CPU half doubles the 4090's decode**: 45.5 tokens a second (50.0 over the second half of the steps) against 22.2 with every miss copied, on
  the same pod and load. This pod's host-to-GPU copies ran at 13.0 GB/s; the first Hummingbird pod's at 24.8 (32.39 tokens a second copying):
  the CPU mode is 40% above that pod's copy figure too, and it depends less on the pod's PCIe link.
- **A 16 GB card's budget runs at 38.1 tokens a second** in the CPU mode (84 experts a layer on the GPU, 6.43 GiB of cache), against 14.5 copying.
- **The planner's split and auto are slower than cpu in decode** (27 to 44 against 38 to 50): their copies wait on the GPU (the host waits 17 to 19 ms
  a token at 16 GB against 7.6). In prefill they are faster (80 against 49). The planner's costs are the next tuning.
- **Threads:** 8 are within 6% of 26; pinning to cores changes nothing here.
- Greedy tokens of the CPU modes first differ from copy's at token 2 of 256: a near tie (the teacher-forced logits agree on every top-1, KL 0.0008).
