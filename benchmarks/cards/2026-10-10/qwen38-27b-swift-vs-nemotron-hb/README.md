# Qwen3.8-27B at swift against Nemotron 3 Super at Hummingbird, the same questions (2026-10-10)

`tools/cards/eval_job.sh` on v030 daefe3c3, glyd/Qwen3.8-27B-swift (main) through `glyd-engine serve` on one RTX 4090 (24,564 MiB), shard 1/2 of
Inspect AI's gsm8k and mmlu_pro (inspect_evals 0.24.0 as shipped, non-thinking, temperature 0): the questions of Nemotron's shard 1
(benchmarks/cards/2026-10-10/nemotron-3-hummingbird-evals/shard1). The pod's 60-minute cap ended MMLU-Pro after 80 questions; the comparison is on the
questions both answered, matched by Inspect's sample ids. About $0.90.

| same questions | Qwen3.8-27B swift (5.5 bits a weight) | Nemotron 3 Super hummingbird (2.99 bits a weight of the experts) |
|---|--:|--:|
| GSM8K (150) | 150 / 150 (100%) | 147 / 150 (98.0%) |
| MMLU-Pro (80) | 69 / 80 (86.3%) | 62 / 80 (77.5%) |

The bf16 Nemotron was not run on these questions: how much of the gap is the hummingbird level's (its perplexity is 10.3% over bf16's,
benchmarks/weights/moe-quality-2026-10-10/nemotron-hbr) is not measured.
