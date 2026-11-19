## Expert-routing skew: zai-org/GLM-5.3-Flash @ eb9eb208
42 MoE layers x 288 experts, top-8; 294912 calibration tokens in 6 domains, 49152 held-out; sequences of 2048. Hit = share of routed (token, expert) pairs served by the resident set.

### Top-k experts per layer: share of routed tokens covered (mean over layers; % of experts in the header)

| set | 5% | 7% | 10% | 13% | 19% | 20% | 30% | 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all domains (min-max over layers) | 13.0 (7.5-19.3) | 16.9 (10.1-24.2) | 22.3 (14.0-30.5) | 26.6 (17.4-35.4) | 35.3 (24.6-44.8) | 36.7 (25.8-46.1) | 48.3 (36.7-57.5) | 68.4 (57.6-75.7) |
| wiki | 20.9 | 26.4 | 33.5 | 38.8 | 49.0 | 50.5 | 62.9 | 81.3 |
| code | 14.9 | 19.3 | 25.1 | 29.8 | 39.2 | 40.6 | 52.8 | 72.8 |
| chat | 14.7 | 19.0 | 24.7 | 29.4 | 38.9 | 40.4 | 52.8 | 73.2 |
| math | 26.4 | 32.4 | 39.7 | 45.1 | 55.0 | 56.5 | 67.9 | 84.3 |
| multi | 24.7 | 30.5 | 37.6 | 42.9 | 52.5 | 53.9 | 64.9 | 80.9 |
| reason | 22.5 | 27.9 | 34.9 | 40.3 | 50.5 | 52.0 | 64.0 | 81.6 |
| uniform routing | 5.0 | 7.0 | 10.0 | 13.0 | 19.0 | 20.0 | 30.0 | 50.0 |

A 2048-token chunk touches 97.5% of a layer's experts on average (prefill streaming).

### Static profile built on the calibration domains, hit rate on held-out text

| cache | per-layer split | one pool over layers | by mass | oracle (held-out's own top-k) | uniform | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7% (20/layer) | 16.1 | 16.6 | 16.3 | 17.7 | 6.9 | 19.0 | 8.0 | 11.8 | 22.9 | 16.7 | 18.0 |
| 13% (37/layer) | 25.7 | 26.2 | 25.6 | 27.8 | 12.8 | 30.3 | 14.6 | 19.5 | 34.3 | 28.2 | 27.2 |
| 19% (55/layer) | 34.4 | 34.7 | 34.1 | 36.8 | 19.1 | 40.0 | 21.6 | 27.0 | 43.9 | 38.0 | 35.7 |

Decision rule (GLM notes: static 13% profile transfers at >= 70% -> go on; < 55% -> stop): static 13% profile on held-out = 25.7%.

### A profile that never saw the domain (built on the other domains, scored on this one)

| cache | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|
| 7% | 13.4 | 7.4 | 10.0 | 12.6 | 12.8 | 11.9 |
| 13% | 22.6 | 13.4 | 17.0 | 21.8 | 21.9 | 20.0 |
| 19% | 31.9 | 19.6 | 24.1 | 30.9 | 30.0 | 27.4 |

### Decode (20 generations, 2540 steps, thinking off): static profile vs per-layer caches replayed in step order

| cache | static profile | static, own top-k (oracle) | LRU cold | LRU seeded with profile | LFU cold |
|---|---:|---:|---:|---:|---:|
| 7% | 13.9 | 21.9 | 40.7 | 40.7 | 21.0 |
| 13% | 22.4 | 33.3 | 52.9 | 53.0 | 32.4 |
| 19% | 30.6 | 43.3 | 61.9 | 61.9 | 42.2 |

Wall clock: model load 107.1 s, prefill 1675.3 tokens/s over 84 batches, decode 0.15 s/step.
