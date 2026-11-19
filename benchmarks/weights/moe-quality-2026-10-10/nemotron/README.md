# Nemotron 3 Super with its routed experts at 2-4 bits (2026-10-10)

`research/lossy-2026/nemotron_quality.py` (nh-quality 899666c1) on four A100-SXM4-80GB, one pod, 29 min of job (about $3.50). Model:
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a, held bf16 on the four cards. Text: glm_skew.py's held-out set, 24 sequences
of 2,048 tokens over six domains, 49,128 positions scored against the bf16 model's own forward. Only the routed experts are rounded:
40 MoE layers, 512 experts each, 112.7B weights. The Mamba-2 and attention layers, the latent projections, the shared experts, the routers,
the embedding and the head stay bf16. Every layer of a variant sees the layers before it already rounded. Linear GLY-131.

| variant | experts at | bits/weight | experts GB | KL from bf16 | top-1 = bf16 | perplexity (bf16 3.021) |
|---|---|--:|--:|--:|--:|--:|
| ref | bf16, through the job's path | 16.00 | 225.5 | 0.0047 | 99.01% | 3.021 |
| int4 | every expert int4 | 4.50 | 63.4 | 0.0133 | 97.85% | 3.044 (+0.8%) |
| mix13 | the 13% most-used int4, the rest hlm3 | 3.71 | 52.2 | 0.0273 | 96.75% | 3.101 (+2.7%) |
| hlm3g | hlm3 with GPTQ act-order | 3.59 | 50.6 | 0.0310 | 96.73% | 3.105 (+2.8%) |
| hlm3 | rotation + per-row codebook, 3 bits | 3.59 | 50.6 | 0.0326 | 96.33% | 3.123 (+3.4%) |
| int3 | every expert int3 | 3.50 | 49.3 | 0.0445 | 95.68% | 3.159 (+4.6%) |
| cold50 | the coldest 50% hlm2, the rest int4 | 3.52 | 49.6 | 0.0564 | 95.02% | 3.177 (+5.2%) |

How to read it:
- **The floor.** ref is KL 0.0047 from the model, where ~1e-4 was expected. The job sums the 22 chosen experts in float32; the model's
  own grouped kernel sums them in bf16 at a routed scale of 5, over 40 layers. Every KL above includes that floor. Later runs also
  score each variant against the control (nh-quality b39536ad), which separates what rounding alone costs.
- **Whole-model size.** The other weights are 21.7 GB in bf16 and about 8.9 GB at six bits. With them at six bits the model is about
  72 GB with int4 experts and about 59.5 GB with hlm3g. Those six bits are not in the KL above.
- **Routing is flat.** The 13% most-used experts take 23.4% of the held-out routed tokens, and the 50% most-used take 65.9%. That is why
  mix13 sits between int4 and hlm3, nearer hlm3.
- **GPTQ helps 3 bits a little.** hlm3g is 0.0310 against hlm3's 0.0326, while its weight error is higher (0.221 against 0.167): it fits
  the outputs, not the weights. 81 of 20,480 expert-layers had fewer than 2,048 calibration tokens and were rounded to nearest.
- **The weakest domains are wiki and chat.** Code and the reasoning text lose least: at int4, 0.0058 and 0.0054, against wiki's 0.0239.
- **Copies between GPUs.** None of this pod's direct GPU-to-GPU copies came back wrong (the job stages them through the host anyway).

Files: quality.md and quality.json (the report), results.jsonl (one row a variant), meta.json, counts/ (routing counts in glm_skew.py's
format), perpos/ (the KL of every position), steps.txt, versions.txt (torch 2.11.0+cu128, transformers 5.19.0, accelerate 1.15.0),
gpus.txt, run.log, selftest.log, prep.json, cpus.txt.
