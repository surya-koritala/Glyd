# Nemotron 3 Super at Hummingbird on one RTX 4090 (2026-10-10)

`tools/cards/nh_offload_job.sh` (HB=1) on v030 c71dc64f: a pod of two NVIDIA GeForce RTX 4090 (24 GB, sm_89, PCIe 4), the job on GPU 0 alone; 22 CPUs,
186 GiB of memory. 17 minutes of job (about $0.50). Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a at the Hummingbird level: the
routed experts as codebook neuron rows from the hbr40 calibration (benchmarks/weights/moe-quality-2026-10-10/nemotron-hbr: 3 bits for each layer's
205 most-used experts, 2 for the other 307; 42.1 GB), every other matrix six-bit. The expert cache's first version (missed experts copied in), as
the wren runs on the same cards. Linear GLY-98, GLY-130.

| step | result |
|---|---|
| the calibration's codes from the private bucket | 42.1 GB in 313 s (134 MB/s) |
| the tiny model at Hummingbird through the cache (`nhoffload`: 3, 5 and 8 places a layer, cold and warm, hints, a predictor, the decode graph) | 26 checks ok: every logit and token the same as with every expert on the GPU |
| the kernels against float64 (`hbcheck`: 3 and 2 bits, up to the real 2,688 x 1,024 experts, a prompt of 24 tokens) | 8 checks ok: outputs within 1.6e-3 to 2.1e-3 of the reference's norm (bound 6e-3) |
| pack (`glyd-engine pack --layout hummingbird --codes`) | 41,234 packs (40,960 from the codes: 8,200 experts at 3 bits, 12,280 at 2), 133 s |
| verify (`glyd-gpu verify`) | every pack decodes to glyd.json's sha256 |
| load | 20.7 GiB on the GPU; 190 of 512 experts a layer there (37%); 39.2 GiB of experts in host memory (wren: 59.1) |
| prefill, 42 tokens | 0.96 s (43.6 tokens a second) |
| decode, 255 steps (greedy) | **32.39 tokens a second**; 77.9% of 224,400 expert reads found on the GPU, 49,512 copied in: 103.0 GB in 4.16 s of copying (24.8 GB/s) |
| `run --prompt` | "The capital of France is Paris." (ready in 1 min 32 s, 32,768 tokens of context) |

How to read it:
- **Against wren on the same card class** (benchmarks/gpu/nh-offload-4090-2026-10-10: 11.95 tokens a second): the smaller experts fit 190 a layer
  on the GPU against 123 (77.9% of reads found against 67.7%), each copy moves 2.1 MB on average against 3.1, and this pod copied at 24.8 GB/s
  against 12.6. The host tier is 39.2 GiB against 59.1.
- **Copying is still 4.2 s of the decode's 7.9 s.** Computing the missed experts on the CPU (both of a Hummingbird expert's matrices are rows the
  CPU kernels read in place) is the next step.
- **Quality** is the calibration's: KL 0.0946, perplexity +10.3% on held-out text (experts only).

Files: steps.txt, summary.txt, free.txt, machine.txt, gpu.txt, versions.log, fetch-export.log, pack.txt, pack.log, verify.log, hbcheck.log,
tinyhb-offload.log, generate-card.log, run-prompt.log.
