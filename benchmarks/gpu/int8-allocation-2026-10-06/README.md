# INT8 allocation comparison

One RTX 4090, RedHatAI/Qwen3.5-9B-quantized.w8a8 at 4257ccf0e5c6f87246fd96ea2e4fb16817101558, vLLM 0.30.0. Baseline 7717c129c8 and candidate ae233ca8dd use the same inputs and settings; the CUDA implementation is unchanged. Full source hashes are in BASE_COMMIT and COMMIT.

| Mode | Baseline weight GiB | Candidate weight GiB |
| :-- | --: | --: |
| Stock checkpoint | 12.73 | 12.73 |
| Glyd BF16 parts only | 10.28 | 10.28 |
| Glyd INT8 codes, default | 10.37 | 9.92 |
| Glyd INT8 codes, exact output | 10.98 | 10.53 |

The allocation change saves 0.45 GiB in both packed-code modes. Exact mode matches the stock-eager text, prompt log probabilities and image comparisons; all 159 BF16 packs and 80 INT8 packs pass byte reconstruction. Both default-mode repeats confirm the memory result. Raw results and comparison.json retain allocator accounting separately from encoded tensor bytes.

Default throughput is essentially unchanged: 50.66 to 50.61 output tokens/s at one user and 1102.61 to 1101.74 at 32 users. Candidate stock reaches 75.16 and 1431.57 respectively. These measurements establish no speed improvement or six-load acceptance. Default-mode generated-token agreement is 0.2559 for both baseline and candidate; this is a token-agreement statistic, not a task-accuracy score or an exact-mode failure. It remains a separate quality concern.

Pod s2w3y6s9b8ampz was deleted after the batch; launcher.log preserves completion and the empty pod listing. Ledger duration 1,857 seconds at $0.74/hour gives an estimated GPU charge of $0.3817167, excluding storage and other fees. No current integration-head GPU validation is implied by this historical record.
