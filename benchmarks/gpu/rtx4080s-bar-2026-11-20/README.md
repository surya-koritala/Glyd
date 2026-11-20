# The engine against vLLM on an RTX 4080 SUPER, five runs of each: it passes the speed bar (2026-11-20)

**The RTX 4080 SUPER is in `GPUS`** (`glyd-engine/src/machine.rs`): on the means of five runs of each program, at every load measured the engine's decode tokens/s, in all and for a user, and its time to first token
are at least vLLM's (within the 3% of `scripts/gpu_bar.py`), and its peak GPU memory is lower (14,796 MiB against 15,190). Machine, model, programs and loads are those of
[rtx4080s-engine-vs-vllm-2026-11-20](../rtx4080s-engine-vs-vllm-2026-11-20) (glyd-pc, RTX 4080 SUPER 16 GB, 736 GB/s, driver 595.91; Qwen3-8B lossless; `glyd-engine serve` of main with the changes below against
vLLM 0.30.0 with glyd-gpu 0.32.1, `--quantization glyd`; 1, 4 and 8 requests at once, short, ~2,000 and ~4,000-token prompts, 3 rounds a load); no CI job on the GPU during any run.

| load | decode tok/s a user: engine (% of ceiling) | vLLM (%) | total tok/s: engine | vLLM | first token s: engine (sd) | vLLM (sd) | engine / vLLM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short (~35 tokens), 1 at once | 59.9 (90%) | 60.0 (91%) | 59.9 | 60.0 | 0.021 (0.2%) | 0.024 (1.0%) | 0.88x |
| short (~35 tokens), 4 at once | 58.6 (88%) | 58.7 (89%) | 233.3 | 233.6 | 0.044 (9.9%) | 0.048 (0.6%) | 0.90x |
| short (~35 tokens), 8 at once | 57.1 (86%) | 57.0 (86%) | 452.7 | 452.1 | 0.064 (0.3%) | 0.068 (0.9%) | 0.93x |
| ~2,000 tokens, 1 at once | 58.7 (88%) | 58.7 (89%) | 58.7 | 58.7 | 0.353 (0.1%) | 0.357 (0.1%) | 0.99x |
| ~2,000 tokens, 4 at once | 49.0 (74%) | 48.6 (74%) | 177.3 | 176.3 | 0.892 (0.5%) | 0.882 (0.2%) | 1.01x |
| ~2,000 tokens, 8 at once | 40.2 (60%) | 39.5 (60%) | 267.5 | 263.2 | 1.604 (0.4%) | 1.587 (0.2%) | 1.01x |
| ~4,000 tokens, 1 at once | 57.7 (87%) | 56.8 (86%) | 57.7 | 56.8 | 0.678 (0.2%) | 0.735 (0.1%) | 0.92x |
| ~4,000 tokens, 4 at once | 42.6 (64%) | 41.3 (63%) | 144.6 | 138.8 | 1.696 (0.2%) | 1.830 (0.3%) | 0.93x |
| ~4,000 tokens, 8 at once | 34.3 (52%) | 31.3 (47%) | 137.8 | 130.5 | 3.076 (0.3%) | 3.302 (0.1%) | 0.93x |

Means of five runs each (the sd is a percent of the mean; the engine's first token at 4 short users is 0.036 s in the first run and 0.0455 in the other four). Ceiling: 736 GB/s over the weights the server holds
(engine 10.3 GiB: 66.5 tokens/s; vLLM 10.38 GiB: 66.0), KV reads not counted. Where first token is within 1% of vLLM's (2,000 tokens, 4 and 8 at once: 0.892 against 0.882 and 1.604 against 1.587) it is behind, inside
the 3% the bar allows; every other load is ahead.

## What got it here (all merged, Part of #214)

- Prompt tile by wave count (#312): a 2,000-token pass 372 to 353 ms, same bits. - Requests taken off the queue by a thread of their own, and the decode step skipped after a short pass for prompts that came in together (#313, #310).
- **Batched prompt passes on by default** (this change): `GLYD_ENGINE_BATCH_PREFILL=0` keeps every reply identical to solo; with several prompts at once a reply's last bits can differ from the same prompt's alone (a pass
  of more than 64 rows takes the products' route its rows give); a deterministic mode is #206 (branch `lever1-exact-batch-wip`).
- **The decode step after a long prompt pass waits for up to two passes while more prompts are pending** (this change): with 8 prompts of 2,000 tokens at once the scheduler trace shows one pass each (350 ms, 2,000 tokens,
  94 TFLOPS) with a 17 ms decode step between each pair, 7 of them in the mean first token (1.640 s against vLLM's 1.587: 3.3% at a spread of 0.1%, so not noise). Now every third round has the step: 1.604 s, decode a user
  40.2 tokens/s as before, total 267.5 against 265.2.

## Per family, and the age of the model

Since this record `GPUS` is per (GPU, family) and the measurement file names its model: this one is Qwen3 (`family Qwen3`), measured on Qwen3-8B, released 2025-04-27, 572 days before the run: the newest dense Qwen3
model there is, but older than the owner's 6 months, so `gpu_bar.py` passes it only with `--stale-ok`. [rtx4080s-gemma4-12b-2026-11-20](../rtx4080s-gemma4-12b-2026-11-20) measures the one recent model that fits.

## Files

`engine.json` and `vllm.json`: the means and spreads of the five runs of each, the measurement file `scripts/gpu_bar.py` takes (`python3 scripts/gpu_bar.py engine.json vllm.json "NVIDIA GeForce RTX 4080 SUPER" 8.9 Qwen3 --stale-ok`, exit 0 and the
`GPUS` line); made from the runs' own files by `gpu_bar.py --summarize` (the runs' files are not kept: `scripts/check_tree.sh`). The scripts that made the runs: the earlier record's `load.py`, `run.sh`, `until_clean.sh`.
