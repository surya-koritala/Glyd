# glyd bench on the RTX 4080 SUPER: the gate's loads in five sessions, and how much they move (2026-11-20)

The reference result of `glyd-engine bench` ([docs/bench.md](../../../docs/bench.md)) for the RTX 4080 SUPER, and the measurement the regression gate's noise band stands on. Machine: glyd-pc (AMD Ryzen 9 7950X3D,
RTX 4080 SUPER 16 GB, driver 595.91.07, 736 GB/s by the spec sheet; under load the GPU boosts to 2,670 MHz SM and 11,251 MHz memory in every session; clocks are not locked: there is no root here). Model: Qwen3-8B lossless,
10.3 GiB in the engine, a ceiling of 66.5 tokens/s a user (736 GB/s over those bytes, KV reads not counted). Program: `glyd-engine bench Qwen/Qwen3-8B --suite gate --runs 5 --weights-gib 10.3` of main at 0.33.0,
a development build (the CI build), starting a `glyd-engine serve --users 8` for each session, in the CI image (nvidia/cuda 13.0.3), with the GPU lock of glyd-pc held. The computer's load average was 1 to 2.2 (CI runners
on glyd-pc may add CPU load); the harness marked no session busy (GPU utilization before each: 0%), though a CI job's container was up when session 4 began.

## The five sessions

`session-1.json` to `session-5.json` are the files `bench` wrote (`glyd-bench/1`), one server started for each, one after another. The median of each session's five runs, and how far the sessions are from one another
(standard deviation over the mean) beside how far the runs of one session are:

| load | metric | session 1 | 2 | 3 | 4 | 5 | sessions | runs of one session |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short x1 | decode tok/s a user (% of ceiling) | 60.00 (90%) | 60.01 | 60.00 | 59.85 | 60.00 | 0.12% | 0.02% |
| short x8 | decode tok/s a user (% of ceiling) | 57.38 (86%) | 57.40 | 57.40 | 57.27 | 57.36 | 0.10% | 0.13% |
| short x8 | total tok/s | 446 | 450 | 450 | 450 | 446 | 0.52% | 0.58% |
| 1k x1 | decode tok/s a user (% of ceiling) | 59.37 (89%) | 59.39 | 59.38 | 59.38 | 59.34 | 0.03% | 0.06% |
| 4k x1 | decode tok/s a user (% of ceiling) | 57.71 (87%) | 57.73 | 57.56 | 57.71 | 57.70 | 0.12% | 0.05% |
| 1k x1 | prefill tok/s | 5,132 | 5,126 | 5,132 | 5,133 | 5,114 | 0.15% | 0.33% |
| 4k x1 | prefill tok/s | 5,915 | 5,894 | 5,885 | 5,899 | 5,907 | 0.19% | 0.50% |
| short x1 | first token s | 0.021 | 0.021 | 0.021 | 0.021 | 0.021 | 0.38% | 0.73% |
| short x8 | first token s | 0.085 | 0.068 | 0.068 | 0.068 | 0.085 | 12.65% | 16.13% |
| 1k x1 | first token s | 0.194 | 0.194 | 0.194 | 0.194 | 0.195 | 0.15% | 0.08% |
| 4k x1 | first token s | 0.678 | 0.681 | 0.682 | 0.680 | 0.679 | 0.21% | 0.17% |

Decode moves 0.03-0.12% from one session to the next. Prefill and the times of a single request move 0.15-0.4%. The loads that put several requests on the server at once move more: the first token of 8 short prompts is
0.068 s in three sessions and 0.085 s in two (the bar's record of 2026-11-20 shows two such values at 4 users as well, 0.036 and 0.0455; why the server's passes fall one way or the other is not investigated here), and its band is wide for that reason.
The gate's noise band (z = 3 on these spreads, with a floor of 0.3%) is therefore **0.3-0.5% for decode, 0.7-1.1% for the prefill of a long prompt and the first token, about 10% for a short prompt's first token and prefill rate
(milliseconds), and 40-47% for the first token and prefill of 8 at once**. The gate between two of these sessions (`bench --compare session-1.json session-2.json --history` the other three) is within the band at every figure, and fails
the same pair on 13 of the 16 figures for a synthetic 5% slowdown (`--synthetic 0.95`: the three that stay inside are the 8-at-once first token and prefill, and the short prompt's, whose bands are wider than 5%).

## What the numbers say about the server

- Decode is 86-90% of the ceiling at the gate's loads, the same as in [rtx4080s-bar-2026-11-20](../rtx4080s-bar-2026-11-20) (59.9 tok/s at short x1, 57.7 at 4k x1; 57.1 a user and 452.7 in all at short x8 in the bar, 57.4 and 449 here).
- `serve` takes 4 users by default: `bench` starts it with as many as its heaviest load (at most 8), else the 8 at once of the gate would be 4 and 4 waiting (a first token of 1.15 s).

## The full suite, once

`full.json`: `bench --suite full --runs 2 --warmup 0 --max-tokens 64`, the loads R8 lists, on the same server with its default context (8,192). Medians of 2 runs:

| load | first token s | decode tok/s a user | total tok/s | prefill tok/s |
| :--- | ---: | ---: | ---: | ---: |
| short x1 | 0.021 | 60.1 | 60.1 | 1,787 |
| short x8 | 0.076 | 57.5 | 438.9 | 3,899 |
| short x64 | 4.135 | 57.5 | 435.0 | 287 |
| short x256 | 18.052 | 57.5 | 434.5 | 261 |
| 1k x1 | 0.194 | 59.4 | 59.4 | 5,126 |
| 4k x1 | 0.679 | 57.7 | 57.7 | 5,910 |
| 32k x1 | failed: 2 requests | | | |

64 and 256 at once are taken as 8 at a time with the rest waiting (the first token of the 256th request is the 18 s it waits; the total stays at 435 tokens/s: the ceiling of 8 users, not of 256); a 32,000-token prompt is past
this server's context and both requests failed (the counts are in the file, not hidden). Both are R11's and the context work's to move; this is where they start.

## Files

| File | What |
| :--- | :--- |
| `session-1.json` ... `session-5.json` | the gate's loads, five runs each, one server each, one after another (what `bench --compare --history` reads) |
| `full.json` | the full suite once |
