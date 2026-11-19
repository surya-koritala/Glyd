# Nemotron 3 Super at Hummingbird: GSM8K and MMLU-Pro (2026-10-10)

`tools/cards/eval_job.sh` on v030 32c57da7, two shards (every second question of each task), 150 questions a task a shard: Inspect AI 0.3.278 with
inspect_evals 0.24.0 as shipped, through `glyd-engine serve` of the packed Hummingbird directory (taken whole from the private bucket), one user at a
time, temperature 0, non-thinking (`enable_thinking: false`), at most 3,072 new tokens. Two one-GPU pods: shard 1 an "RTX 4090" of 49,140 MiB (a 48 GB
variant: 504 of 512 experts a layer on the GPU), shard 2 an RTX 4090 of 24,564 MiB (the experts not on the GPU computed by the CPU). The level's
outputs do not depend on where an expert is computed (the copied and CPU runs agree on every top-1 token: benchmarks/gpu/nh-hb-cpu-4090-2026-10-10).
About $2 in all. Linear GLY-98.

| task | shard 1 | shard 2 | together |
|---|--:|--:|--:|
| GSM8K (10-shot, numeric match) | 147 / 150 (98.0%) | 144 / 150 (96.0%) | **291 / 300 (97.0%)** |
| MMLU-Pro (0-shot chain of thought) | 117 / 150 (78.0%) | 53 / 72 (73.6%) | **170 / 222 (76.6%)** |

- Shard 2's MMLU-Pro stopped at the pod's 90-minute cap after 72 questions (the 24 GB card is about half the 48 GB one's speed on these prompts); its
  score is counted from the samples in its Inspect log (`shard2/*mmlu-pro*.eval`: 72 scored, 53 correct).
- The bf16 model was not run on this harness; these are the checkpoint's own scores.
