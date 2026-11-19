# Candidate A and official FP8 matched quality screen

Qwen/Qwen3.8-27B, measured on one A100-SXM4-80GB on 2026-10-07. All three weight variants use the same Transformers BF16 eager arithmetic, tokenization, evaluation windows and target positions. Official FP8 weights are dequantized before scoring; this isolates weight quality and does not measure native FP8 execution, activation quantization or speed.

| Weight variant | Perplexity, lower is better | Change versus BF16 |
| :-- | --: | --: |
| Original BF16 | 6.02989756 | +0.000000% |
| Candidate A | 6.03488422 | +0.082699% |
| Official FP8 | 6.03593601 | +0.100142% |

The evaluation contains 32 disjoint 1,024-token WikiText test windows: 512 context tokens and 512 scored targets per window, 16,384 targets per arm. This is a language-likelihood screen, not task accuracy, multimodal acceptance or a demonstration of quality equivalence. No model retraining or calibration was performed. Vision and MTP task quality are not evaluated.

## Paired uncertainty

Negative changes favor the first variant. Intervals use 2,000 paired window bootstrap samples with the frozen protocol seed. All three weight-comparison intervals include zero; this screen establishes no winner.

| Comparison | Perplexity change | 95% interval for perplexity change |
| :-- | --: | --: |
| Candidate A vs BF16 | +0.082699% | [-0.174934%, +0.330775%] |
| Official FP8 vs BF16 | +0.100142% | [-0.088764%, +0.291400%] |
| Candidate A vs official FP8 | -0.017425% | [-0.326534%, +0.286536%] |

## Native runtime controls

The native engine is checked separately against the common-arithmetic reference for original BF16 and Candidate A. Both pass the predefined diagnostic bounds: top-1 agreement at least 95%, mean absolute target log-probability difference at most 0.05 nats, and every absolute window NLL delta at most 0.10. These controls do not assert identical forward-pass outputs or relax any lossless weight check.

| Runtime control | Top-1 agreement | Mean absolute log-probability difference | Maximum absolute target difference |
| :-- | --: | --: | --: |
| Original BF16 | 99.05396% | 0.02039526 nats | 0.71577829 nats |
| Candidate A | 98.77930% | 0.02111450 nats | 6.24339690 nats |

Candidate A has three individual targets with a native/reference log-probability difference above one nat, including a maximum of 6.24340 nats. Passing the aggregate diagnostic bounds does not resolve those outliers or establish exact runtime equivalence. The common-arithmetic weight-quality comparison above does not depend on the native runtime.

The native quality arms retain the existing GPU representation. Their logged weight allocation is not the compact Candidate A footprint. Scoring-stage durations are not comparable inference-throughput measurements: the common reference and native scorer use different execution schedules.

## Size evidence and remaining measurements

The separate [complete weight inventory](https://github.com/surya-koritala/glyd-gpu-private/blob/bd699c103423411ba921a66fb254625e75102b60/benchmarks/weights/candidate-a-2026-10-06/README.md) counts all 1,199 tensors, including vision and MTP: Candidate A reference encoding 24.43534 GiB, BF16 tensor payload 51.74694 GiB, official FP8 indexed files 28.74701 GiB. Candidate A is 52.77916% smaller than BF16 and 14.99867% smaller than those FP8 files. FP8's count includes headers and tensors retained as BF16. These are separate full-checkpoint byte measurements at the same pinned model revisions, not same-run GPU memory measurements.

Compact GPU correctness, resident size, whole-model integration, same-GPU tokens/s, time to first token and time per token remain unmeasured. The quality result alone does not establish a 24 GiB fit or a speed advantage.

## Reproduce and inspect

The immutable source is `88ea1ae57e34025a76f88a8e947b6ddc82bbeacb`, with prepared binaries from [quality CI](https://github.com/surya-koritala/glyd-gpu-private/actions/runs/37558294094). Its built merge commit and source have identical trees, recorded in `source-proof.json`. The reproducible job is `research/lossy-2026/quality_match_job.sh` at that source; the job verifies the prepared bundle, runs both runtime controls and creates the three paired comparisons.

`evaluation.json` records pinned model, tokenizer and dataset provenance with the exact numeric inputs. `reference-audit.json` records both checkpoint revisions, matching model configuration and tokenizer, parameter coverage, and FP8 conversion checks. All five arm JSON files preserve per-target log probabilities and predictions. The paired JSON files and logs are unmodified run outputs. `BUNDLE-SHA256SUMS` identifies the immutable prepared executables; `SHA256SUMS` covers the files in this record. The native scorer and all GPU checks ran on the pod; local processing only collected and checked the numerical evidence.

The batch ended after about 32 minutes under its 75-minute limit and the pod was deleted. The guard estimates GPU cost at $0.89084; the approved total allocation was $2.10. `cost.json` records the guard's GPU estimate and its accounting scope; storage is not represented as an invoiced measurement. No retry was used.
