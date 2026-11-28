# Verify chunks and keep_previous on an RTX 4080 SUPER, 2026-10-22

Drafted tokens through range stages (docs/range-worker.md, "Drafted tokens"): `rangecheck --workers` of Qwen3.5-9B linnet
(glyd/Qwen3.5-9B-linnet 073a2132) on the owner's PC (RTX 4080 SUPER, driver 595.91.07), under ~/glyd-scratch/gpu.lock.
Binary built from the range-rewind branch's source (glyd-engine/ and gpu/ as committed), sha256 091fdfea...0c5fa; its GPU
library from the same tree, sha256 d60421a6...b88f. Each file here is a run's output (stdout and stderr) as it came.

```
glyd-engine rangecheck $M --workers --prompt 64 --new 128 --cut 16          # rangecheck-2-stages.txt
glyd-engine rangecheck $M --workers --prompt 64 --new 128 --cuts 11,23      # rangecheck-3-stages.txt
glyd-engine rangecheck $M --workers --prompt 1024 --new 256 --cut 16        # rangecheck-2-stages-long.txt
```

Drafting check: after the prompt, verify chunks of 8 rows `[last token, 7 drafts]` through every worker, the head on all 8
rows; the drafts are the plain run's next `a` tokens and then wrong ones, `a` = 0..7 in turn; the next chunk carries
`keep_previous: a + 1`. Every kept row's token and logits (all 248,320 bf16 words a row) against the plain whole-model run.

| run | verify chunks | chunks per a = 0..7 | result | a verify chunk | a token (plain step) |
| :--- | ---: | :--- | :--- | ---: | ---: |
| 2 stages, prompt 64, 128 tokens | 30 | 4, 4, 4, 5, 4, 3, 3, 3 | 128 steps, 31,784,960 logits equal, every token equal | 12.80 ms | 3.02 ms (9.89 ms) |
| 3 stages, prompt 64, 128 tokens | 30 | 4, 4, 4, 5, 4, 3, 3, 3 | the same | 13.19 ms | 3.12 ms |
| 2 stages, prompt 1,024, 256 tokens | 58 | 8, 8, 7, 7, 7, 7, 7, 7 | 256 steps, 63,569,920 logits equal, every token equal | 13.67 ms | 3.11 ms |

The refusals (each run, first worker): keep_previous 9 and 0, a position_start other than the verify chunk's + K, a verify
chunk of 9 rows, a verify prefill chunk, keep_previous after a plain chunk; keep_previous then a plain step is taken.

Cost (the whole model in one process, 20 repeats; prompt 64 / prompt 1,024):

| | prompt 64 | prompt 1,024 |
| :--- | ---: | ---: |
| a one-row step (the graph) | 9.28 ms | 9.39 ms |
| a verify pass of 8 rows | 10.26 ms | 10.69 ms |
| of it, the states' copy (51,511,296 bytes, 24 layers) | 0.170 ms | 0.171 ms |
| a rewind keeping 1, 2, .. 7 rows | 0.401, 0.433, 0.462, 0.493, 0.525, 0.557, 0.588 ms | 0.401 .. 0.587 ms |
| keeping all 8 | 0 | 0 |

The tokens of a random prompt repeat (15, 15, ...); the comparison is of every logit, so that changes nothing. Not run:
penguin, kestrel, wren (only linnet is on the PC now), the 27Bs, another GPU.
