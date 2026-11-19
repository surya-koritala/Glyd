# Nemotron 3 Super with its routed experts at 3 bits or less: the Hummingbird recipes (2026-10-10)

`research/lossy-2026/nemotron_quality.py` (v030 6a814719) with VARIANTS="ref,hlm2g,hb40,hb20" on four A100-SXM4-80GB, 47 minutes of job
(about $6). Model, text and method as benchmarks/weights/moe-quality-2026-10-10/nemotron (the first run of this job): the bf16 model as
released, glm_skew.py's held-out set (24 sequences of 2,048 tokens over six domains, 49,128 positions), only the routed experts rounded
(112.7B weights), every layer seeing the layers before it rounded. Linear GLY-98 (Hummingbird) and GLY-131.

| variant | experts at | bits/weight | experts GB | KL from bf16 | top-1 = bf16 | perplexity (bf16 3.021) |
|---|---|--:|--:|--:|--:|--:|
| ref | bf16 through the rounding path, unrounded | 16.00 | 225.5 | 0.0047 | 99.01% | 3.021 (-0.02%) |
| hb40 | the 40% most-used experts of each layer hlm3g, the rest hlm2g | 2.96 | 41.7 | 0.0831 | 93.96% | 3.286 (+8.8%) |
| hb20 | the 20% most-used experts of each layer int4, the rest hlm2g | 2.93 | 41.3 | 0.1022 | 93.07% | 3.364 (+11.3%) |
| hlm2g | every expert hlm2 with GPTQ act-order | 2.54 | 35.8 | 0.1532 | 91.05% | 3.548 (+17.4%) |

The first run's rows on the same text, for comparison: int4 4.50 bits 0.0133 (97.85%, +0.8%); hlm3g 3.59 bits 0.0310 (96.73%, +2.8%); int3
3.50 bits 0.0445 (95.68%, +4.6%); cold50 3.52 bits 0.0564 (95.02%, +5.2%).

How to read it:
- **Below 3 bits the cost steps up.** The best recipe at 3 bits or less, hb40, is KL 0.083 and perplexity +8.8%: 2.7 times hlm3g's KL for
  9 GB less (41.7 GB of experts against 50.6). hlm3g at 3.59 bits stays the better point per GB lost.
- **Three bits on the hot experts beat four bits on fewer.** hb40 (40% hlm3g) is ahead of hb20 (20% int4) at about the same size.
- **The KL includes the job's 0.0047 floor** (the control; the "KL from control" column in quality.md is within 0.0002 of it).
- **Wiki and chat lose most:** hb40 is 0.211 on wiki and 0.122 on chat, 0.037 on code and 0.022 on the reasoning text (quality.md).
- GPTQ fell back to rounding to nearest for 83-92 of 20,480 expert-layers per variant (fewer than 2,048 calibration tokens routed to them).

Files: quality.md and quality.json (the report), results.jsonl, meta.json, counts/ (routing counts), perpos/ (the KL of every position),
steps.txt, versions.txt, gpus.txt, cpus.txt, run.log, selftest.log, prep.json.
