# Nemotron 3 Super on one RTX 4090 through the expert cache (2026-10-10)

`tools/cards/nh_offload_job.sh` on v030 be614726: a pod of two NVIDIA GeForce RTX 4090 (24 GB, sm_89, PCIe 4), the job on GPU 0 alone; 22 CPUs,
186 GiB of memory. 35 minutes of job (about $1.05). Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a packed by the engine at wren
(the routed experts four-bit, the rest six-bit: 70.5 GB). The expert cache's first version: every expert in pinned host memory, the GPU
holding as many as fit, a missed expert copied to the GPU before its layer runs. Linear GLY-130.

| step | result |
|---|---|
| the tiny model through the cache at levels 6, 5 and 4 (`nhoffload`: 3 places a layer, cold and warm, with hints, the decode graph) | every logit and token the same as with every expert on the GPU |
| the grouped products over Int6, Int5 and Int4 experts (`moecheck relu2-22of512`) | all within their bounds |
| pack at wren | 241.34 GB of bf16 to 70.49 GB; 1,069 s on 22 CPUs |
| load (`generate --expert-cache auto`) | 288.7 s (59.1 GiB of experts pinned in host memory); 20.31 GiB of weights on the GPU, 21.89 of 23.52 GiB in use; 123 of 512 experts a layer on the GPU (24%) |
| prefill, 42 tokens | 2.74 s (15.3 tokens a second); every expert read copied in (10,766 reads, 33.3 GB at 12.6 GB/s) |
| decode, 255 steps (greedy) | **11.95 tokens a second**; 67.7% of 224,400 expert reads found on the GPU, 72,403 copied in: 224.2 GB in 17.8 s of copying (12.6 GB/s) |
| `run --prompt` | "The capital of France is Paris." (ready in 2 min 14 s, 32,768 tokens of context) |

How to read it:
- **Nemotron 3 Super (120B) runs on a 24 GB consumer card** with its experts in the computer's memory, the output the same as with every
  expert on the GPU (the tiny model's check, bit for bit).
- **The hit rate is the replay's:** 67.7% with 24% of the experts on the GPU, between the measured per-layer LRU's 60.7% at 17% and
  74.2% at 32% (benchmarks/moe-skew-2026-10-09/nemotron).
- **Copying is the time:** 17.8 s of the decode's 21.3 s, 0.88 GB a token at 12.6 GB/s (half of PCIe 4 x16's 25 GB/s). The rest, about
  14 ms a token, is the GPU's work. Computing the missed experts on the CPU (2 KB back each, the relu² zeros skipped) and a faster copy
  path are the next steps; this run measures the starting point.
- **Load** pins 59.1 GiB of host memory and takes 4.8 minutes; a second start took 2 min 14 s.

Files: steps.txt, summary.txt, free.txt, machine.txt, gpu.txt, versions.log, pack.txt, pack.log, moecheck.log, tiny6-offload.log,
tiny5-offload.log, tiny4-offload.log, generate-card.log, run-prompt.log.
