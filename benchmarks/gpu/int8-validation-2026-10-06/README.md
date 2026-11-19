# Initial INT8 code validation

Source 6ac8fb2e, RedHatAI/Qwen3.5-9B-quantized.w8a8, vLLM 0.30.0. The provider reports RTX 4090 / sm_89 with 49,140 MiB; this is not evidence of fitting a standard 24 GiB card.

| Mode | Weight GiB | Output tokens/s, 1 user | Output tokens/s, 32 users |
| :-- | --: | --: | --: |
| Stock | 12.73 | 76.91 | 1458.38 |
| Glyd BF16 parts only | 10.28 | 86.87 | 1582.22 |
| Glyd with INT8 codes | 10.37 | 52.16 | 1130.29 |

Exact mode uses 10.98 GiB and matches the stock-eager text, prompt log probabilities and image comparison. All 159 BF16 packs and 80 INT8 packs reconstruct exactly. Code tensors save 393,772,316 bytes after a 96 MiB scratch allocation, while reported total weight memory increases by 0.09 GiB versus BF16-parts-only Glyd. The later paired allocator record investigates this discrepancy. Default generated-token agreement is 0.2403; this is not a task-accuracy measure. No six-load speed or ordinary onboarding acceptance follows.

The raw summary records settings but does not pin the checkpoint revision in its invocation. Preserve this reproducibility limitation. The later allocator record uses an explicit pinned checkpoint.

Pod hesubezaqg62er was deleted after completion; launcher.log records the remaining, separate scale-validation pod. The guard recorded 2,001 seconds at $0.74/hour, an estimated $0.4113167 GPU charge excluding storage and other fees. Historical source validation does not establish current integration-head acceptance.
