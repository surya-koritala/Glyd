# Two sessions a range stage on an RTX 4080 SUPER, 2026-10-22

`rangeworker --sessions 2` (docs/range-worker.md, "Two sessions a stage"): `rangecheck --workers` of Qwen3.5-9B swift
(glyd/Qwen3.5-9B-swift 073a2132) on the owner's PC (RTX 4080 SUPER, driver 595.91.07), under ~/glyd-scratch/gpu.lock.
Binary built from the range-sessions branch's source, sha256 c22e236e...cd135; its GPU library a copy of range-rewind's
(gpu/ is the same tree), sha256 b452dd11...1b31c. Each file here is a run's output as it came.

```
glyd-engine rangecheck $M --workers --sessions 2                  # rangecheck-2-stages.txt (prompt 1,024, 256 tokens, cut 16)
glyd-engine rangecheck $M --workers --sessions 2 --cuts 11,23     # rangecheck-3-stages.txt
glyd-engine rangecheck $M --workers                               # rangecheck-1-session.txt (the workers as before)
glyd-engine rangecheck $M --workers --ctx 8192 --prompt 64 --new 48 [--sessions 2]   # rangecheck-ctx8192-{1-session,2-sessions}.txt (memory)
```

The two-session checks, on every worker, a pass of each session in turn (a prompt chunk, a step or a verify chunk). A runs
the random prompt, B the same backwards and a quarter shorter (768 tokens), so the two sit at other positions. Each
session's 256 tokens and every logit (248,320 bf16 words a row) against its own prompt's whole-model run:

| check | 2 stages | 3 stages |
| :--- | :--- | :--- |
| a third begin_session | refused | refused |
| A and B plain to 128 tokens; A cancelled (its status cancelled, B's idle), A's next chunk refused, B 4 steps alone, A ended (cancelled true on every worker), C begun on A's room and run from 0 in turn with B | B and C: 256 steps, 63,569,920 logits equal, every token equal | the same |
| A drafted (58 verify chunks of 8 rows, keep_previous, 0..7 drafts kept), B plain | both equal | both equal |
| A and B both drafted | both equal | both equal |
| the one-session checks before them (plain, repeats, cancel, sampling, head_rows 8, drafting) | passed | passed |

`rangecheck-1-session.txt`: the default `--sessions 1`, all checks passed as before. describe with `--sessions 1` is byte
for byte range-rewind's (both stages; incarnation, pid and build commit aside); with `--sessions 2` only `limits.sessions`
is 2.

Memory a session more costs (the engine's counters; a stage of 16 layers: 4 full-attention, 12 linear-attention):

| | one stage, ctx 1,296 | one stage, ctx 8,192 | whole model, ctx 8,192 |
| :--- | ---: | ---: | ---: |
| keys and values (16,384 bytes a token a stage) | 21,233,664 | 134,217,728 | 268,435,456 |
| states | 25,755,648 | 25,755,648 | 51,511,296 |
| prepare's engine_bytes, --sessions 1 to 2 | | 3,677,122,908 to 3,837,096,284 | |
| the device's free bytes, stage 0 | | 165,675,008 less | |

The device's free bytes go down 5.7 MB more than the counters on stage 0 (the graph and the driver's rounding). A
verify chunk's copy of the states (25,755,648 bytes a stage) and kept inputs (2,371,584) are made at each session's first
verify chunk, as before; the attention parts for 8 rows are shared.

Time (whole model in one process, 50 one-row steps each; then the workers, 2 stages on the one GPU, one client thread
sending the passes one after another, so nothing overlaps):

| | one session | two sessions in turn |
| :--- | ---: | ---: |
| whole model, a step (graph), ctx 1,296 | 9.34 ms | 9.34 ms |
| whole model, a step, ctx 8,192 | 9.23 ms | 9.23 ms |
| a second sequence's room and graph capture (once, at prepare) | | 11.9 to 12.2 ms |
| workers, 2 stages, a plain step | 9.92 ms | 9.87 ms |
| workers, 3 stages, a plain step | 10.10 ms | 10.06 ms |
| workers, 2 stages, a verify chunk of 8 rows | 13.29 ms | 13.31 ms (both drafted: 26.62 ms a turn of two) |

Not measured here: two stages on two GPUs with the passes overlapping (that is Starlings' pipeline), the 27Bs, another
GPU, levels other than swift.
