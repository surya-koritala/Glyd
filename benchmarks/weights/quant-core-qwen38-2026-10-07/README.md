# Qwen3.8-27B fixed-choice quality baseline

All three arms completed the same 13,243 questions on one A100-SXM4-80GB on 2026-10-07. The six-suite mean is 78.67785% for Candidate A, 78.54327% for original BF16 and 78.54877% for official FP8 weights. The preset statistical decision is **inconclusive**, and the overall product decision remains incomplete.

Original BF16, Candidate A and dequantized official FP8 use the same Transformers BF16/FP32 eager arithmetic. This isolates weight quality. The evaluation is zero-shot fixed-choice likelihood scoring without a chat template; it does not measure native FP8 activation quantization, generated answers, compact GPU memory or serving speed.

| Suite | Questions | BF16 | Official FP8 weights | Candidate A |
| :-- | --: | --: | --: | --: |
| ARC-Challenge | 1,172 | 59.04% | 59.22% | 58.19% |
| HellaSwag | 2,048 | 84.42% | 84.18% | 84.23% |
| WinoGrande | 1,267 | 75.61% | 75.85% | 76.72% |
| PIQA | 1,838 | 81.61% | 81.56% | 81.45% |
| BoolQ | 3,270 | 86.61% | 86.67% | 87.49% |
| MMLU | 3,648 | 83.96% | 83.83% | 83.99% |
| Equal-suite mean | 13,243 | 78.54% | 78.55% | 78.68% |

ARC-Challenge, HellaSwag and PIQA use the pinned task definitions' length-normalized choice accuracy; WinoGrande, BoolQ and MMLU use choice accuracy. ARC-Challenge, WinoGrande, PIQA and BoolQ use their full selected test or validation splits. HellaSwag uses a frozen 2,048-question validation sample. MMLU uses 64 test questions from each of 57 subjects. The mean weights each of the six suites equally. Per-subject MMLU results are retained in `report.json`.

The model pair is `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, released in August 2026. Dataset revisions, selected original row indices, task definitions, tokenizer identity, question text and token sequences were frozen before the run. Upstream datasets and model materials retain their own terms; identities and available license metadata are in `manifest.json`.

The frozen policy permits at most 0.5 percentage point mean loss and 1 point per suite against both references, with simultaneous 95% uncertainty bounds. The observed Candidate A mean is 0.13458 points above BF16 and 0.12908 above FP8. Its mean accuracy-loss intervals are [-1.63348, +1.36421] points versus BF16 and [-1.74530, +1.48680] versus FP8. Positive loss means Candidate A scores lower. These intervals do not establish the preset margins.

ARC-Challenge is the main observed regression: Candidate A is 0.85324 points below BF16 and 1.02389 below FP8. It answers 12 fewer questions correctly than FP8 out of 1,172, with 23 harms and 11 gains in paired outcomes. The one-point suite margin is exceeded by that point estimate, while the paired uncertainty still spans both sides of the margin. No comparison meets the predefined statistical refocus condition; only BoolQ meets the per-suite upper-bound condition against both references.

The intervals bound paired harm and gain rates using exact binomial limits, with a union bound across six suites and both references. They rely on Bernoulli sampling assumptions; fixed public datasets and the stratified MMLU sample limit generalization. This result establishes a reusable screen, not quality equivalence, a general accuracy percentage for the model or a winner over FP8.

`manifest.json` identifies all 13,243 questions and the immutable protocol. `inputs/` retains all 62 frozen task request files, including prompts, choices and token IDs. Each arm retains every choice score in `samples.jsonl`, completion metadata and a SHA256 hash. `report.json` is the original GPU-job report; `audit.json` records independent recomputation of all 39,729 scored outcomes and inversion of binomial tails, matching the reported intervals within 1e-11. Original logs, environment locks, source and CI identities are retained. `BUNDLE-SHA256SUMS` identifies the preparation bundle; `RECORD-SHA256SUMS` covers this record. Tokenizer bytes and dependency wheels are in the separately preserved preparation bundle.

The source is [e7dfc48b](https://github.com/surya-koritala/glyd-gpu-private/tree/e7dfc48b08dca2c67f70cc54762deec328eeee5c), prepared by [CI run 37621438332](https://github.com/surya-koritala/glyd-gpu-private/actions/runs/37621438332). Its CI merge commit `3bd50e81849d45ed6d7e8ac336cb1ebe482f3f5a` has the same tree as the reviewed head. The 220,026,880-byte bundle has SHA256 `2d2351c99f596edb1aaf2a985ae0c2894d2160f65c0978b33a2afa56a19fe325`. The run's bounded wrapper is `job.sh`. Model/kernel execution occurred on the pod; subsequent local work only verified and recorded data.

The existing [benchmark harness](../../../research/quant-benchmark/README.md) defines how to prepare future model entries. To reproduce this report from the retained scores in an environment with the recorded SciPy version:

```sh
python research/quant-benchmark/report.py \
  --prepared benchmarks/weights/quant-core-qwen38-2026-10-07 \
  --results benchmarks/weights/quant-core-qwen38-2026-10-07 \
  --out /tmp/quant-core-qwen38-report.json
```

All stages exited zero. The workload ran from 12:58:36 to 14:24:39 UTC; the launcher deleted the pod at 14:25 UTC, and an authenticated listing independently confirmed its absence. The conservative GPU estimate is $2.338625 at $1.59/hour, using 5,295 accounted seconds including the guard's 120-second allowance. This is below the $4.25 allocation; storage and other fees are not an invoiced total. The run used one attempt and ended within its 150-minute cap. `cost.json` and `deletion-proof.json` retain the evidence.

Generated math, instruction following, executed code tasks, native-runtime outliers, multimodal quality, compact resident memory and matched serving speed remain separate gates. The [complete weight inventory](../candidate-a-2026-10-06/README.md) measures representation bytes independently. Production defaults and automatic engine admission are unchanged.
