# Nemotron 3 Super on one RTX 5090 through the expert cache (2026-10-10)

`tools/cards/nh_offload_job.sh` on v030 1793b709: a pod of two NVIDIA GeForce RTX 5090 (32 GB, sm_120, PCIe 5), the job on GPU 0 alone; 55 CPUs,
175 GiB of memory. 16 minutes of job (about $0.65). Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a packed by the engine at wren
(the routed experts four-bit, the rest six-bit: 70.5 GB). The expert cache's first version, as on the RTX 4090 (benchmarks/gpu/nh-offload-4090-2026-10-10):
every expert in pinned host memory, the GPU holding as many as fit, a missed expert copied to the GPU before its layer runs. Linear GLY-130.

| step | result |
|---|---|
| the tiny model through the cache at levels 6, 5 and 4 (`nhoffload`) | every logit and token the same as with every expert on the GPU |
| the grouped products over Int6, Int5 and Int4 experts (`moecheck relu2-22of512`) | 24 checks ok, none failed |
| pack at wren | 241.34 GB of bf16 to 70.49 GB; 399 s on 55 CPUs |
| load (`generate --expert-cache auto`) | 96.2 s; 28.04 GiB of weights on the GPU, 29.74 of 31.36 GiB in use; 190 of 512 experts a layer on the GPU (37%) |
| prefill, 42 tokens | 0.72 s (58.2 tokens a second); every expert read copied in (33.3 GB at 51.1 GB/s) |
| decode, 255 steps (greedy) | **43.59 tokens a second**; 77.9% of 224,400 expert reads found on the GPU, 49,593 copied in: 153.6 GB in 3.06 s of copying (50.3 GB/s) |
| `run --prompt` | "The capital of France is Paris." (ready in 1 min 26 s, 32,768 tokens of context) |

How to read it:
- **Nemotron 3 Super (120B) at 43.6 tokens a second on one 32 GB consumer card**, its experts in the computer's memory, with the cache's first
  version: 3.6 times the RTX 4090's 11.95.
- **Why the 5090 is so much faster:** it holds 37% of the experts against the 4090's 24% (77.9% of reads hit, against 67.7%), and PCIe 5
  copies at 50.3 GB/s against 12.6 GB/s on the 4090's pod. Copying is 3.1 s of the decode's 5.85 s; the GPU's own work about 11 ms a token.
- **The hit rate is the replay's:** 77.9% with 37% of the experts on the GPU, between the measured per-layer LRU's 74.2% at 32% and 82.4% at
  46% (benchmarks/moe-skew-2026-10-09/nemotron).

Files: steps.txt, summary.txt, free.txt, machine.txt, gpu.txt, versions.log, pack.txt, pack.log, moecheck.log, tiny6-offload.log,
tiny5-offload.log, tiny4-offload.log, generate-card.log, run-prompt.log.
