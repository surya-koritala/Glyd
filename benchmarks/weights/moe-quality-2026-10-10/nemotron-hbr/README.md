# Nemotron 3 Super at Hummingbird with neuron rows: the recipe the engine ships (2026-10-10)

`research/lossy-2026/nemotron_quality.py` (v030 6c6aac98) with VARIANTS="ref,hbr40,hb40,hbr40p" on four A100-SXM4-80GB, 69 minutes of job (about $9).
Model, text and method as benchmarks/weights/moe-quality-2026-10-10/nemotron: the bf16 model as released, glm_skew.py's held-out set (49,128
positions over six domains), only the routed experts rounded. Linear GLY-98.

| variant | experts at | bits/weight | experts GB | KL from bf16 | top-1 = bf16 | perplexity (bf16 3.021) |
|---|---|--:|--:|--:|--:|--:|
| ref | bf16 through the rounding path, unrounded | 16.00 | 225.5 | 0.0047 | 99.01% | 3.021 |
| **hbr40** | both matrices as 2,688 neuron rows of 1,024 values in one rotated latent basis a layer, per-row codebooks; the 40% most-used experts of each layer 3 bits, the rest 2; GPTQ act-order (over neurons for down) | 2.988 | 42.1 | 0.0946 | 93.38% | 3.334 (+10.3%) |
| hb40 | the earlier recipe: rotation on each matrix's input side (down's over the 2,688 neurons) | 2.961 | 41.7 | 0.0831 | 93.96% | 3.286 (+8.8%) |
| hbr40p | hbr40 with a rotation of the latent for each expert | 2.988 | 42.1 | 0.0968 | 93.43% | 3.344 (+10.7%) |

How to read it:
- **hbr40 is the format the engine runs** (kernels/moe_hb.cu, src/cpu_hb.rs): a token's second product reads only its active neurons' rows (92%
  of them are 0, benchmarks/weights/offload-probe-2026-10-10/nemotron), and the CPU reads both matrices from the host tier as they are. hb40's
  input-side rotation of down mixes all 2,688 neurons and would read every row. Keeping the sparsity costs 1.5 points of perplexity here.
- **One rotation a layer costs nothing:** hbr40 (shared) is not worse than hbr40p (one an expert).
- **Its codes** (42.1 GB, format glyd-hbr-codes/1, research/lossy-2026/nemotron_hbr.py) went to the private bucket at
  s3://glyd-models-712416033631/nemotron-3-super/hbr40/20261010-1138-EST/export/ in 61 s (690 MB/s).
- GPTQ fell back to rounding to nearest for 82 of 20,480 expert-layers (fewer than 2,048 calibration tokens).

Files: quality.md and quality.json, results.jsonl, meta.json, counts/, perpos/, steps.txt, versions.txt, gpus.txt, cpus.txt, run.log,
selftest.log, prep.json.
