# Where a decode step's time goes, RTX 4080, gemma-4-12B at wren (2026-11-22)

R4a (#138) plans an asynchronous step for the description executor: the step launched, the host free to schedule the next one while the GPU works, no wait for the
tokens each step. Before building it, this record measures what it could gain: how long the GPU sits idle between steps while the host schedules, samples and answers.

The measurement is the step profiler of a development build (`GLYD_ENGINE_STEP_PROF=1`, src/dynamic/rows.rs), which prints for every 256 steps the call's wall time; the GPU's
time from an event at the call's start to one after its work (busy with the step, or waiting for the host to queue the rest of it); and the host's time between one step's return
and the next one's call. A step today waits for its tokens, so the GPU is idle for all of the host's time.

**Result: while sequences decode, the host's time between steps is 0.055 ms against a step of 12.68 ms with one user (0.4%), and 0.053 ms against 16.58 ms with eight (0.3%); the
call returns 5 µs after the GPU's last work. An asynchronous step could hide at most that share on this model and card. The windows that hold a switch between the bench's loads
show 6.2 to 6.4 ms between steps: they include the time when there was no request at all, and are not the host's work.**

Machine: a rented RTX 4080 (16 GB, driver 590.48.01, 320 W, AMD EPYC 7742), nothing else on the GPU. Engine: main 7d3f37ff with M2 batch 16 (the profiler and the executor's
Gemma 4 lowered from its layer kinds), built by the GPU job's recipe (`scripts/ci_gpu_build.sh`, kernels for sm_89). Model: `google/gemma-4-12B-it:wren`. The server ran with
8 users, no draft model, the profiler on (`step-profile.log` is its whole stderr); `glyd-engine bench --against` it, one load at a time, a warm-up run and two runs a load, 128
new tokens a request, temperature 0; no request failed (`prof-*.json`).

| window of 256 steps (in the order printed) | rows on average | prompt tokens a step on average | the call | the GPU | the host between steps | the host's share |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| a short prompt alone | 1.0 | 4 | 12.675 ms | 12.670 | 0.055 | 0.4% |
| the switch to eight short prompts | 2.7 | 1 | 12.861 | 12.857 | 6.161 | (idle) |
| eight short prompts | 7.9 | 2 | 16.575 | 16.571 | 0.053 | 0.3% |
| the switch to a 1k prompt alone | 2.8 | 7 | 15.008 | 15.003 | 6.404 | (idle) |
| the switch to a 4k prompt alone | 1.0 | 4 | 13.043 | 13.038 | 6.151 | (idle) |
| a 4k prompt alone, its chunks beside the decode | 1.0 | 22 | 18.607 | 18.603 | 0.158 | 0.8% |

The bench's figures with the profiler on: a short prompt alone 0.024 s to the first token and 86.5 tokens/s; eight short prompts 0.112 s, 62.6 tokens/s a user and 481.1 in all;
a 1k prompt alone 0.263 s and 82.1; a 4k prompt alone 0.859 s and 79.0. The records before measured 86.6, 82.1 and 79.0 tokens/s alone and 484 in all for eight short prompts
(`../rtx4080-blocks-by-name-2026-11-22/standing.json`): no cost of the profiler that the runs can tell.

## What it does not say

- One model and one card. gemma-4-12B's step is long beside the host's work; a smaller model or a faster card has a shorter step, and the host's share is larger there. Not
  measured.
- The GPU's time is from the call's first work to its last. Gaps inside a step, where the GPU waits for the host to queue the rest of it, are in it and not apart.
- A window's averages mix the decode with the prompt work of the same steps (a short prompt's 38 tokens, a 4k prompt's chunks); the profiler does not split a step's time by
  what it does.
- The host's work that the expert cache adds (copying missed experts, R12) and the exchanges between GPUs are not on this path: a model split across GPUs, or with its experts in
  the computer's memory, can spend more time on the host between steps. Not measured here.
