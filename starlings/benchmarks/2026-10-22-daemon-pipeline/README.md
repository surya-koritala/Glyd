# One request through two daemon stages (2026-10-22)

GLY-188 in the daemons. Three `starlings daemon` processes (a coordinator and two enrolled workers) on one PC.
A chat request goes to the coordinator over HTTP; the first worker's `glyd-engine rangeworker` tokenizes it and runs
layers 0..16 of Qwen3.5-9B (glyd linnet), the second runs 16..32 and samples, the token goes back to the first worker
over the data channel, and text streams back as SSE. Then the test stops the daemons and runs the same prompt on
the whole model (one rangeworker, 0..32) with the same chunking and settings.

Result: in every run the pipeline's tokens equal the whole model's (same count, same sha256).

| run | prompt | tokens | finish | equal | pipeline ms | whole model ms | client ms | engine ms, stage 0 + 1 | rows a stage | ended positions, 0 / 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy-48 | 21 | 48 | length | yes | 683 | 449 | 1540 | 206 + 255 | 68 | 68 / 68 |
| greedy-long-prompt-200 | 104 | 200 | length | yes | 2206 | 1902 | 3061 | 878 + 1095 | 303 | 303 / 303 |
| greedy-stop-token | 22 | 145 | stop | yes | 1657 | 1355 | 2501 | 622 + 773 | 166 | 166 / 166 |
| sampled-t0.7-64 | 21 | 64 | length | yes | 1934 | 1538 | 2780 | 277 + 1338 | 84 | 84 / 84 |

- pipeline ms: from the first poll to both stages' reports (prefill and decode). Whole model ms: the same work in
  one rangeworker, no Starlings. Client ms: the whole HTTP request; the ~0.85 s over pipeline ms is per-request
  setup (two generation connections, count, plan and issuers, Prepare/Commit/Begin, grants, data TLS). Not tuned.
- Each stage's report says what it executed: its layer range, prefill and decode steps, rows (prompt + tokens - 1:
  the last token is never run), the engine's own `end_session` count of positions (equal to the rows), and a
  `status` after it showing no open session.
- The long prompt ran as 4 chunks of 32; its 2.2 s run renewed both stages' 2 s route grants every 0.9 s.
- The sampled run draws on the last stage from (seed 7, step, logits) at temperature 0.7, top_p 0.9; the whole model
  with the same settings gives the same tokens. Its stage-1 time is the host-side top_p sort of the vocabulary.
- The stop run ended on a stop ID (248044 `<|endoftext|>` or 248046 `<|im_end|>`, from the model's tokenizer config).
- Each run sends the request twice, streamed and not; both give the same text. No data connection was
  refused in these runs (`refused_connections` 0 on both stages).

Limits: both stages on one RTX 4080 SUPER (one machine, loopback TLS), two stages, one request at a time,
one user message (the engine's chat tokenize).

Machine: Ryzen 9 7950X3D, RTX 4080 SUPER (driver 595.91.07), Ubuntu kernel 7.0.0-38. Engine: glyd-gpu-private
PR #226 (range-text-sampling, 66e6bdbe), dev build, binary sha256 `7c7261c3…` (in record.json), with the GPU library
built from the same source. Records: record.json (all four runs, including both stages' reports).

Command, from `crates/starlings-cli` on the PC, under the GPU lock (`pipe-run.sh`; test-backend builds are the
only ones that log the token digests the comparison needs):

    export GLYD_GPU_LIB=~/glyd-scratch/starlings/lib-pr/libglyd_gpu_cuda13.so
    export STARLINGS_RANGE_ENGINE=~/glyd-scratch/starlings/bin/glyd-engine-66e6bdbe
    export STARLINGS_RANGE_MODEL=~/glyd-baseline/hf/hub/models--glyd--Qwen3.5-9B-linnet/snapshots/073a2132abeaa41c119678b9687092871521eb4d
    export STARLINGS_RANGE_NEW=48 STARLINGS_RANGE_RECORD=record.json
    # optional: STARLINGS_RANGE_TEXT="..." STARLINGS_RANGE_TEMPERATURE=0.7
    cargo test --release -p starlings-cli --features test-backend --test daemon --no-run
    flock ~/glyd-scratch/gpu.lock target/release/deps/daemon-<hash> --exact \
      real_pipeline_daemons_generate_through_two_stages_like_the_whole_model --nocapture
