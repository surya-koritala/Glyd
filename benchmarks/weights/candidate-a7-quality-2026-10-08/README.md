# Candidate A rounded to seven bits: fixed-choice quality

The seven-bit variant of Candidate A completed the same 13,243 questions as the [quality baseline](../quant-core-qwen38-2026-10-07/README.md) on one A100-SXM4-80GB on 2026-10-08. Its six-suite mean is 78.70214%, against 78.54327% for original BF16 and 78.54877% for official FP8 weights from the retained baseline arms. The preset statistical decision is **inconclusive**, as it was for Candidate A.

## What the variant is

Candidate A values (BF16 rounded to three fraction bits), then every two-dimensional matrix rounded to the nearest e3m3 value (sign, three exponent bits with bias 3, three fraction bits, subnormals, no NaN; largest 30) times the smallest power-of-two scale covering each 1 x 128 row span, ties to even. Stored that way it would take about 7 + 8/128 ≈ 7.06 bits a weight, about 12% below eight bits; that size is arithmetic, not an encoded file. The transform changed 499,427,792 weights: exactly the [fixed-width census](../candidate-a-fixed-width-census-2026-10-07/README.md)'s e3m3 1 x 128 inexact count over all matrices (507,784,359) less the MTP matrices (8,356,567), which this model load does not include. CI checked the transform against an independent nearest-value search over every Candidate A word.

Scoring used the baseline's frozen questions, tokenizer, packages and common Transformers BF16/FP32 eager arithmetic; the BF16 and FP8 arms are the baseline's retained scores, so no reference was rescored. It measures weight quality only, not native FP8 execution, generated answers, memory or speed.

| Suite | Questions | BF16 | Official FP8 weights | Candidate A | Seven-bit |
| :-- | --: | --: | --: | --: | --: |
| ARC-Challenge | 1,172 | 59.04% | 59.22% | 58.19% | 58.36% |
| HellaSwag | 2,048 | 84.42% | 84.18% | 84.23% | 84.03% |
| WinoGrande | 1,267 | 75.61% | 75.85% | 76.72% | 76.72% |
| PIQA | 1,838 | 81.61% | 81.56% | 81.45% | 81.50% |
| BoolQ | 3,270 | 86.61% | 86.67% | 87.49% | 87.55% |
| MMLU | 3,648 | 83.96% | 83.83% | 83.99% | 84.05% |
| Equal-suite mean | 13,243 | 78.54% | 78.55% | 78.68% | 78.70% |

## Decision

The frozen policy permits at most 0.5 percentage point mean loss and 1 point per suite against both references, with simultaneous 95% bounds. The seven-bit mean is 0.15886 points above BF16 and 0.15337 above FP8; its mean accuracy-loss intervals are [-1.66276, +1.34535] points versus BF16 and [-1.74837, +1.44165] versus FP8 (positive loss means the variant scores lower). These do not establish the margins, and no comparison meets the refocus condition.

ARC-Challenge remains the observed regression: 0.68259 points below BF16 (19 harms, 11 gains) and 0.85324 below FP8 (23 harms, 13 gains), with intervals spanning both sides of the margin. BoolQ and MMLU meet the per-suite upper bound against both references; Candidate A met it on BoolQ only.

Paired with Candidate A, the seven-bit variant changes the chosen answer on 152 of 13,243 questions and answers 3 more correctly. The intervals are about as wide as the baseline's.

## Evidence

`candidate-a7/` holds every choice score, the header with the transform counts and `samples.jsonl`'s SHA256. `report.json` is the GPU job's report; `audit.json` records an independent recomputation of all 39,729 scored outcomes (this arm and both retained arms) and inversion of the binomial tails, matching the report's intervals within 1e-11. The questions (`manifest.json`, identical byte for byte), inputs, tokenizer identity and reference arms are in the [baseline record](../quant-core-qwen38-2026-10-07/). To reproduce the report with the recorded SciPy version:

```sh
python research/quant-benchmark/report.py \
  --prepared benchmarks/weights/quant-core-qwen38-2026-10-07 \
  --results benchmarks/weights/candidate-a7-quality-2026-10-08 \
  --references benchmarks/weights/quant-core-qwen38-2026-10-07 \
  --candidate candidate-a7 --out /tmp/candidate-a7-report.json
```

## Run

The source is [e1121dd3](https://github.com/Sigbound/glyd-gpu-private/tree/e1121dd3315e4623d803df3157ca5b159232123f), merged as #95. [CI run 37724482616](https://github.com/Sigbound/glyd-gpu-private/actions/runs/37724482616) built the bundle from the baseline's preparation artifact (SHA256 `2d2351c99f596edb1aaf2a985ae0c2894d2160f65c0978b33a2afa56a19fe325`), its prepared questions and environment unchanged, with this code and the retained reference arms. The artifact zip SHA256 `0986fbf4196a4e433c5bedfffff030f33260050912bc02191d50c7d5cea17327` matched CI; the bundle has SHA256 `26f938d27de0c0c2dde9552bc951e40fe1423f210d1b5a52e3df198a30081a16`, and all 189 members passed `BUNDLE-SHA256SUMS` before upload and on the pod. `bundle-proof.json` records the approval; `job.sh` is the bounded job.

Pod `0j4afux79bx6j3` was created at 11:35 UTC; the job ran 11:35:27–12:10:45 (`steps.txt`; scoring from 11:39:52) and all stages exited zero. The launcher deleted the pod at 12:11, and an authenticated listing at 12:12:01 UTC returned no pods (`deletion-proof.json`). The guard accounts 2,268 seconds at $1.59/hour, a conservative $1.0017 GPU estimate excluding storage, within the $1.60 cap; one attempt, no retry (`cost.json`). `RECORD-SHA256SUMS` covers this record.

Memory, a seven-bit product's speed against FP8, generated tasks and multimodal quality remain separate gates.
