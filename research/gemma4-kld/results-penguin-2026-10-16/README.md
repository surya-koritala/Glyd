# Gemma 4 penguin against transformers, each vs float32 (2026-10-16, Lambda A100 40 GB for the 12B, H100 for the 31B "ours" row)

Wikitext-2 test, 32 chunks of 512 (kld_tokens.py), compared with transformers 5.18 in float32. Ours: the penguin pack (Google's bf16 weights
bit for bit, verified), branch penguin-gemma (d9151938 + cae5a43d), `g4score`. Recorded in Linear GLY-295 (run 4, 01:20-02:10 Eastern, ~$1.65).

| vs transformers fp32 | mean KLD | same top-1 | PPL ratio |
|---|--:|--:|--:|
| 12B transformers bf16 eager | 0.132 | 89.9% | 1.051 |
| 12B transformers bf16 sdpa | 0.143 | 90.4% | 1.058 |
| 12B ours, penguin (A100) | 0.170 | 88.8% | 1.062 |
| 31B transformers bf16 eager | 0.162 | 91.6% | 1.040 |
| 31B transformers bf16 sdpa | 0.179 | 92.3% | 1.008 |

| between bf16 runs | mean KLD |
|---|--:|
| transformers eager vs sdpa, 12B / 31B | 0.132 / 0.164 |
| ours vs transformers eager, 12B (A100) / 31B (H100) | 0.127 / 0.200 |

The instruct checkpoints on raw text are sensitive to any rounding: transformers' two bf16 paths are 0.13-0.16 apart. 64 chunks of the same
engine on the 12B QAT weights (results-2026-10-16) put ours at 0.109 against transformers bf16's 0.102 vs fp32, batches ±0.03 apart.
12B penguin on the A100 40 GB: 14.93 GiB of weights, 16.2 GiB in use at 1K context, pack 25.9 s, verify bit for bit.
