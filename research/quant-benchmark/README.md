# Quantized-weight evaluation baseline

Compare original BF16, official FP8 checkpoint values and fixed Glyd Candidate A values using the same arithmetic, questions and scoring. This private research suite decides whether weight-quality evidence supports further implementation work. It does not turn a perplexity change into an accuracy percentage or infer GPU memory from encoded file bytes.

## Fixed quality profiles

The task definitions come from a pinned [EleutherAI evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). Its official request builders and choice metrics are retained. Every dataset revision, task name, prompt, answer, tokenizer and token input is frozen before inference. All tasks are zero-shot, without a chat template; these are fixed-choice measurements, not reasoning-enabled generation scores.

| Suite | Coverage | Primary metric | Screen profile |
| :-- | :-- | :-- | :-- |
| ARC-Challenge | Science questions | Length-normalized accuracy | Full labeled test split |
| HellaSwag | Plausible event completion | Length-normalized accuracy | 2,048 fixed validation examples |
| WinoGrande | Commonsense reference resolution | Accuracy | Full labeled validation split |
| PIQA | Physical reasoning | Length-normalized accuracy | Full labeled validation split |
| BoolQ | Passage comprehension | Accuracy | Full labeled validation split |
| MMLU | 57 subjects, including mathematics and computing | Accuracy | 64 fixed test examples per subject |

The full profile uses every labeled evaluation example in each suite. The screen uses a declared SHA256 ranking of dataset row indices; selection does not depend on answers, model predictions or outcomes. Every chosen question is required. Screen and full results have distinct manifest identities and must not be mixed or labeled as the same baseline. MMLU's screen equally samples subjects; it is not the full official MMLU aggregate. `manifest.json` contains measured counts and exact input-token work; no example count is silently assumed.

FP8 is dequantized into the same BF16/FP32 arithmetic used by the other arms. This isolates the weight transformation. Native FP8 activation quantization and execution are separate comparisons. Candidate A reuses the existing tested transform; this suite introduces no additional rounding or training.

## Decisions

The provisional strict policy permits at most 0.5 percentage point accuracy loss in the equal-weight average over six suites and 1 point on any individual suite, against each of BF16 and official FP8. The policy is stored in `suite.json` and included in the immutable manifest hash. It must be fixed before a measured run; changing it produces a different protocol identity.

Paired question outcomes retain both losses and gains. Per-suite loss intervals use exact Clopper-Pearson bounds for the two discordant probabilities and a union bound across their tails, all suites and both references. Their simultaneous coverage is at least 95% under the binomial assumption of independent, identically distributed paired questions within each suite. The public task collections and stratified MMLU screen need not satisfy that population model, so the intervals are decision aids rather than distribution-free guarantees. Equal-weight average bounds are derived from those simultaneous suite intervals. These conservative intervals retain uncertainty even when all observed answers agree. They do not prove representativeness of public benchmarks or remove training-data contamination concerns.

- **Continue quality work:** every upper confidence bound is within its declared loss margin.
- **Refocus:** at least one lower confidence bound exceeds its margin.
- **Inconclusive:** neither condition holds. More independent evidence is required; a small mean difference alone cannot pass.
- **Invalid:** a missing question, failed arm, non-finite score, changed input, truncated context or mismatched source prevents a decision.

The report preserves per-question scores, raw and normalized predictions, per-subject scores, suite accuracy, paired discordances, uncertainty, and macro averages. Accuracy is recomputed from the retained choice scores before aggregation. No weighting by a favorable task's size hides a weak suite. Thresholds guide research prioritization, not release approval.

## Additional weight variants

`candidate-a7` is a separate, pre-specified lossy variant: Candidate A's rounding, then every two-dimensional matrix as the nearest e3m3 value (seven bits: sign, three exponent bits with bias 3, three fraction bits, subnormals, no NaN) times the smallest power-of-two scale covering each 1 x 128 span of a row, ties to even. Values below e3m3's range relative to their span round onto its subnormal step; the scorer records how many changed. `check_e3m3.py` compares the transform with an independent nearest-value search in CI.

A single new arm is scored on the original prepared questions with `pod_job_arm.sh`, and `report.py --candidate ... --references ...` pairs it with the retained BF16 and FP8 arms of the same manifest and package environment. The `arm_bundle` workflow input packages the original screen bundle unchanged with the current scorer and those retained arms.

## Separate completion gates

The overall product decision stays incomplete even if this quality screen passes. Generated mathematics, instruction following, isolated code-execution evaluation, native-runtime probability outliers, multimodal quality, compact GPU residency and serving speed remain separate work. The current suite does not claim those capabilities have passed.

Runtime records must distinguish encoded bytes, allocated weights, peak total GPU memory, working buffers and KV cache. Performance records must retain GPU, software, model revision, input/output lengths, warmup, repetitions, and workloads at 1, 8, 32, 64, 128 and 256 concurrent users. Report tokens/s, time to first token and time per token at each load. A compact-format end-to-end run is required; scoring-stage duration is not inference speed. Broad public speed claims and automatic engine routing retain their existing all-load gates.

## Add a model

Add an immutable BF16/official-FP8 pair to `models.json`, both released in 2026, with its supported Transformers auto class and an independently verified tensor inventory. Declare the architecture/configuration fields that must agree. The preparation verifies Hub release dates, revision hashes, shard hashes, configuration and vocabulary identity, then produces a model-specific tokenizer/input manifest under the unchanged suite. An unsupported loader or missing FP8 counterpart must be reported explicitly. Do not substitute another model's FP8 or reuse its baseline.

The current adapter supports the pinned Qwen3.8-27B entry. Adding another architecture requires its loader/inventory audit and the same CI checks; a registry entry alone is not evidence of model support. Dataset text, task definitions, scoring and decision policy are reusable across entries. Adding tasks or changing prompts, sample selection, arithmetic or policy requires a new versioned comparison identity.

## Prepare and run

The `Quantization benchmark preparation` workflow runs protocol regressions, checks the model adapter on a small configuration of the actual 2026 architecture, freezes the selected task requests and builds the complete evaluation wheel set in the exact pinned pod image. It verifies a fresh offline virtual environment against the resolved package manifest before producing `quant-benchmark.tar` with checksums. No full model weights or GPU jobs run during preparation.

The separate pod job verifies the bundle, hardware and package identities, downloads and checks both pinned checkpoints, and executes all three arms on one A100 80GB. Its workload deadline is 140 minutes; the outer launcher must enforce a maximum 150 minutes and delete the pod on every exit. If the frozen workload estimate does not fit that cap, do not launch it unchanged. A timeout produces an incomplete record, never a score from the completed prefix. No silent retry, arm resumption, truncation or dependency replacement is allowed.

Each run retains its prepared manifest, source and bundle hashes, environment, raw scores, logs, completion markers, paired report, cost and pod-deletion evidence. Baselines are immutable. Keep previous results and compare only compatible protocol identities; an optimized engine later adds a new runtime record rather than overwriting this weight-quality reference.
