# Drafted decode across two stages (2026-10-22)

GLY-360 on the daemon pipeline: a greedy decode step carries the last accepted token and up to 7
guesses after it. Both stages make the same guesses (an n-gram lookup over the prompt IDs and the
accepted tokens), the last stage returns each row's greedy token, both keep the longest matching
prefix, and both engines rewind the rest (`keep_previous`). See
[the pipeline doc](../../docs/daemon-pipeline.md#drafted-decode-gly-360).

Every run below gives the whole model's tokens (same count and sha256), and the drafted and plain
runs of a prompt give the same tokens.

## One PC

Qwen3.5-9B swift, layers 0..16 and 16..32 on one RTX 4080 SUPER, loopback TLS, `draft_tokens` 7.

| run | tokens | decode steps, plain / drafted | drafts held | pipeline ms, plain / drafted | whole model ms |
| --- | --- | --- | --- | --- | --- |
| greedy-48 | 48 | 47 / 39 | 8 of 93 | 500 / 430 | 452 |
| greedy-long-prompt-200 | 200 | 199 / 152 | 47 of 512 | 2089 / 1708 | 1908 |
| greedy-stop-token | 145 | 144 / 129 | 15 of 418 | 1456 (earlier record) / 1400 | 1352 |

- A verify pass of up to 8 rows costs about one step on this GPU (the engine PR measured 10.26 ms
  for 8 rows against 9.28 for one), so a held guess is nearly a free token. With these prompts only
  9-11% of guesses hold, and the pipeline still runs faster than the whole model in one process on
  greedy-48 and long-200.
- Matches of at least 2 tokens (min-match-2 records) hold more often (15-22%) but make fewer
  guesses: 41 and 167 steps, 437 and 1762 ms. The default keeps 1-token matches.

A sampled request (temperature 0.7) on a route with `draft_tokens` 7 sends no guesses and no prompt
IDs, and gives the whole model's tokens (record `sampled-t0.7-64-drafts-7-configured`).

## Over a 43 ms link (simulated)

`STARLINGS_RANGE_LINK_MS=43`: a relay holds every byte 21.5 ms each way (no bandwidth limit), same PC.

| run | M4, real link (ms a token) | one round trip a token (ms a token) | drafted (ms a token) | drafted pipeline ms |
| --- | --- | --- | --- | --- |
| greedy-48 | 143.9 | 57.6 | 49.2 | 2362 |
| greedy-long-prompt-200 | 137.0 | 55.2 | 43.3 | 8652 |

The drafted long run spends less than one round trip a token.

## Builds and commands

Engine: glyd-gpu-private#232 (`range-rewind`, d815f8b3), release build, sha256 `3e3a2f30…`; GPU
library built from the same source, sha256 `b452dd11…`. Starlings: this branch, test-backend
release build. From `crates/starlings-cli` on the PC, under the GPU lock:

    GLYD_GPU_LIB=.../lib-rewind/libglyd_gpu_cuda13.so STARLINGS_RANGE_ENGINE=.../glyd-engine-rewind \
    STARLINGS_RANGE_MODEL=<Qwen3.5-9B-swift snapshot> STARLINGS_RANGE_NEW=48 STARLINGS_RANGE_DRAFTS=7 \
    [STARLINGS_RANGE_LINK_MS=43] [STARLINGS_RANGE_TEXT=...] \
      flock ~/glyd-scratch/gpu.lock target/release/deps/daemon-<hash> --exact \
      real_pipeline_daemons_generate_through_two_stages_like_the_whole_model --nocapture

Records: `record-*.json` (both requests' reports with `drafted` and `accepted`, the token digests,
the whole model's run).

Limits: one PC and one GPU; the link is a relay. Greedy only (the engine draws only a chunk's last
row, so sampled requests send no guesses). Only the swift level on Ada was run. Guesses come from
the request's own tokens only, so prompts that do not repeat themselves gain little.
