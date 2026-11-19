## Expert-routing skew: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ 2dc98e2a
40 MoE layers x 512 experts, top-22; 294912 calibration tokens in 6 domains, 49152 held-out; sequences of 2048. Hit = share of routed (token, expert) pairs served by the resident set.

### Top-k experts per layer: share of routed tokens covered (mean over layers; % of experts in the header)

| set | 5% | 10% | 13% | 17% | 20% | 30% | 32% | 46% | 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all domains (min-max over layers) | 11.0 (7.1-15.0) | 19.0 (13.2-24.6) | 23.7 (17.0-30.2) | 29.2 (21.7-36.5) | 33.0 (25.1-40.9) | 45.4 (36.5-54.0) | 47.6 (38.7-56.2) | 62.2 (53.5-70.2) | 65.9 (57.4-73.6) |
| wiki | 15.2 | 24.7 | 30.0 | 36.1 | 40.3 | 53.0 | 55.3 | 69.3 | 72.8 |
| code | 14.5 | 24.0 | 29.2 | 35.2 | 39.3 | 52.0 | 54.2 | 68.4 | 71.9 |
| chat | 10.9 | 18.8 | 23.3 | 28.7 | 32.6 | 44.9 | 47.1 | 61.9 | 65.7 |
| math | 22.4 | 34.4 | 40.5 | 47.1 | 51.5 | 64.1 | 66.1 | 78.5 | 81.4 |
| multi | 17.8 | 28.0 | 33.5 | 39.6 | 43.8 | 56.2 | 58.4 | 71.7 | 74.9 |
| reason | 13.7 | 23.0 | 28.3 | 34.3 | 38.5 | 51.4 | 53.7 | 68.3 | 71.8 |
| uniform routing | 5.0 | 10.0 | 13.0 | 17.0 | 20.0 | 30.0 | 32.0 | 46.0 | 50.0 |

A 2048-token chunk touches 98.9% of a layer's experts on average (prefill streaming).

### Static profile built on the calibration domains, hit rate on held-out text

| cache | per-layer split | one pool over layers | by mass | oracle (held-out's own top-k) | uniform | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17% (87/layer) | 28.9 | 29.1 | 28.4 | 30.6 | 17.0 | 31.2 | 25.4 | 22.7 | 37.4 | 30.8 | 26.2 |
| 32% (164/layer) | 47.4 | 47.4 | 46.5 | 49.5 | 32.0 | 49.7 | 43.9 | 40.0 | 56.4 | 49.4 | 45.0 |
| 46% (236/layer) | 62.1 | 62.2 | 61.1 | 64.2 | 46.1 | 64.5 | 59.0 | 54.6 | 70.0 | 64.6 | 60.2 |

Decision rule (Nemotron notes: 32% cache needs ~43% hit for 10 tok/s and ~75% for 20 tok/s on a 4090 + PCIe 4 (4 bpw)): static 32% profile on held-out = 47.4%.

### A profile that never saw the domain (built on the other domains, scored on this one)

| cache | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|
| 17% | 26.7 | 22.4 | 21.1 | 25.7 | 25.3 | 23.8 |
| 32% | 45.0 | 39.6 | 37.7 | 44.6 | 43.8 | 41.5 |
| 46% | 59.9 | 54.2 | 52.2 | 59.8 | 58.5 | 56.7 |

### Decode (20 generations, 2540 steps, thinking off): static profile vs per-layer caches replayed in step order

| cache | static profile | static, own top-k (oracle) | LRU cold | LRU seeded with profile | LFU cold |
|---|---:|---:|---:|---:|---:|
| 17% | 23.2 | 34.5 | 60.7 | 60.7 | 33.7 |
| 32% | 40.6 | 53.9 | 74.2 | 74.2 | 52.8 |
| 46% | 55.3 | 68.3 | 82.4 | 82.6 | 67.1 |

Wall clock: model load 59.7 s, prefill 1236.8 tokens/s over 84 batches, decode 0.097 s/step.
