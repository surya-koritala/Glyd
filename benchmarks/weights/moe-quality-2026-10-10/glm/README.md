# GLM-5.3-Flash with its routed experts at 2-4 bits (2026-10-10)

`research/lossy-2026/glm_quality.py` (glm-quality 34a33a77) on four H100 NVL, one pod, 77 min of job (about $17). Model:
zai-org/GLM-5.3-Flash at eb9eb208, held FP8 on the four cards (transformers 5.19.0). Text: glm_skew.py's held-out set, 24 sequences
of 2,048 tokens over six domains, 49,128 positions. Only the routed experts are rounded: 42 MoE layers, 288 experts each, 304.5 GB at
FP8. Attention, shared experts, dense layers, routers, embedding and head stay as released. Every layer of a variant sees the layers
before it already rounded. Linear GLY-131.

The control (ref) is the FP8 experts dequantised to bf16 and run unrounded through the job's path. It is KL 0.0438 from the FP8 model,
whose kernels also round every activation to FP8 (a Glyd engine would not). The columns "from control" are what rounding the weights
alone costs.

| variant | experts at | bits/weight | experts GB | KL from control | top-1 = control | KL from FP8 | perplexity (FP8 3.858) |
|---|---|--:|--:|--:|--:|--:|--:|
| ref | FP8 dequantised, unrounded | 8.00 | 304.5 | | | 0.0438 | 3.870 (+0.3%) |
| int4 | every expert int4 | 4.50 | 171.2 | 0.0950 | 92.77% | 0.0934 | 3.866 (+0.2%) |
| mix13 | the 13% most-used int4, the rest hlm3 | 3.67 | 139.5 | 0.1533 | 89.91% | 0.1523 | 3.975 (+3.1%) |
| hlm3g | hlm3 with GPTQ act-order | 3.54 | 134.8 | 0.1651 | 89.83% | 0.1654 | 4.071 (+5.5%) |
| hlm3 | rotation + per-row codebook, 3 bits | 3.54 | 134.8 | 0.1770 | 89.04% | 0.1752 | 4.020 (+4.2%) |
| int3 | every expert int3 | 3.50 | 133.2 | 0.2124 | 88.03% | 0.2128 | 3.988 (+3.4%) |
| cold50 | the coldest 50% hlm2, the rest int4 | 3.51 | 133.6 | 0.2267 | 87.50% | 0.2247 | 3.979 (+3.2%) |

How to read it:
- **The reference is the checkpoint's model.** transformers 5.19.0's load report lists 90 `weight_scale_inv` of the KDA forget-gate
  projections (layers 0-44) as MISSING: it makes them FP8 linears, while the checkpoint holds them in bf16 without scales. Those scales
  are never used. The weights load as bf16, and its FP8 linear multiplies a weight that is not FP8 with a plain `F.linear`
  (integrations/finegrained_fp8.py). A rerun at 3:22 AM that would have put them back in bf16 found nothing to replace. It reproduced
  the control and int4 to four decimals and hlm3 to 0.1748 from the control, against 0.1770 here.
- **Much more sensitive than Nemotron 3 Super.** Under the same recipes on the same text, Nemotron's int4 is KL 0.0133 and its hlm3
  0.0326, both against bf16 and including a 0.0047 floor (benchmarks/weights/moe-quality-2026-10-10/nemotron). Nemotron's experts were
  trained at NVFP4; GLM's were not. The rerun above makes the difference real, not an artifact of the load.
- **Perplexity barely moves at int4 while KL and top-1 do.** It is +0.2% against KL 0.095 and 92.8% top-1: the probability shifts among
  the top tokens, not away from the token that comes. At 3 bits perplexity rises 3-5% and is not monotonic in KL (int3 +3.4%, hlm3g
  +5.5%). Chat text, at perplexity 14, is the noisiest domain. Task scores (GSM8K) are the next measure, not more KL.
- **Routing.** The 13% most-used experts take 25.8% of the held-out routed tokens, and the 50% most-used take 67.7%.
- **Copies between GPUs.** None of this pod's direct GPU-to-GPU copies came back wrong (the job stages them through the host anyway).

Files: quality.md and quality.json (the report), results.jsonl, meta.json, counts/ (routing counts, glm_skew.py's format), perpos/ (the
KL of every position), steps.txt, versions.txt, gpus.txt, run.log, selftest.log, prep.json.
