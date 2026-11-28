# One round trip a token, and no setup stall (2026-10-22)

Two fixes to the daemon pipeline in [2026-10-22-daemon-pipeline](../2026-10-22-daemon-pipeline/README.md)
and M4 ([2026-10-22-two-machine-pipeline](../2026-10-22-two-machine-pipeline/README.md)).

**A token crossed the link three times.** Each stage granted credit for its next input only after its
own output's receipt came back (the core admits the next step's input only once the running step has
drained, and a step drains after its receipt). So every hop waited for a credit, the data and the
receipt: ~3 round trips a token, ~140 ms on M4's 43 ms link. Now a stage may open its next receive
(reserve it and grant the credit) one step early, right after its own output is on the wire. The
check for that is `validate_data_input_next`: the step right after the pending one, as its completion
will leave the session, and nothing later. The bytes themselves are still read under
`validate_data_input`, after the running step drains. See [data flow](../../docs/data-flow.md).

**Setup took ~0.85 s on Linux.** Generation records went out as two writes (the 4-byte length, then
the JSON). With Nagle on, the second write waited for the peer's delayed ACK: ~80 ms an exchange on
Linux (`generation_small_record_tcp_comparison`, 4 exchanges: 330 ms before, 0.6 ms after). macOS
did not show it. Now each record is one write, and generation streams set `TCP_NODELAY`.

## Real engine, one PC

Qwen3.5-9B linnet, layers 0..16 and 16..32 on one RTX 4080 SUPER, loopback TLS. Every run's tokens
equal the whole model's (same count and sha256 as the earlier records).

| run | tokens | pipeline ms before | now | whole model ms | client ms before | now | setup ms now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| greedy-48 | 48 | 683 | 486 | 453 | 1540 | 506 | 9 |
| greedy-long-prompt-200 | 200 | 2206 | 2026 | 1899 | 3061 | 2051 | 14 |
| greedy-stop-token | 145 | 1657 | 1456 | 1355 | 2501 | 1483 | 9 |
| sampled-t0.7-64 | 64 | 1934 | 1663 | 1617 | 2780 | 1695 | 14 |

`setup_ms` is new in `PIPELINE_REPORT`: from the coordinator's first connection to the stages
until both have their grants (count, plan, Prepare/Commit/Begin, Start).

## Real engine over a 43 ms link (simulated)

`STARLINGS_RANGE_LINK_MS=43` puts a relay in front of each stage's data listener that holds every
byte 21.5 ms each way (no bandwidth limit). Same PC and GPU as above. M4 had a real 43 ms link
between two RTX 6000 Ada pods, so the GPUs differ (engine time is small next to the link).

| run | pipeline ms, M4 (real link, before) | now (simulated link) | ms a token, M4 | now | client ms, M4 | now |
| --- | --- | --- | --- | --- | --- | --- |
| greedy-48 | 6908 | 2767 | 143.9 | 57.6 | 8497 | 2799 |
| long-200 (104-token prompt) | 27391 | 11037 | 137.0 | 55.2 | not captured | 11058 |

A token now costs one round trip plus both stages' engine time (~10 ms). What is left of the rest
is per request: two data connections (TLS and the stage handshake, each several round trips) and
one round trip per prompt chunk.

## Stand-in engine

`pipeline_fake_engine_decodes_at_one_round_trip_a_token` (test-backend, no GPU) runs 4 and then 24
tokens over a 40 ms relay and fails above 1.5 round trips a token.

| machine, build | per token before | after | setup ms before | after |
| --- | --- | --- | --- | --- |
| Mac, debug | 145.4 | 52.8 | not logged (client minus pipeline 50-70) | not logged |
| PC, release | not run | 40.8 | 837 | 7-10 |

On the Mac, debug: 2.4 ms a token with no relay, 52.8 at 40 ms, 100.5 at 80 ms.

## Commands

From `crates/starlings-cli` on the PC, with the build and environment of the earlier record
(engine 66e6bdbe, dev, `GLYD_GPU_LIB` from the same source), under the GPU lock:

    STARLINGS_RANGE_NEW=48 [STARLINGS_RANGE_LINK_MS=43] [STARLINGS_RANGE_TEXT=...] \
      flock ~/glyd-scratch/gpu.lock target/release/deps/daemon-<hash> --exact \
      real_pipeline_daemons_generate_through_two_stages_like_the_whole_model --nocapture

Records: `record-*.json` (each has both requests' `PIPELINE_REPORT`s, the token digests and the
whole model's run).

Limits: one PC, so the 43 ms link is a relay, not a network (no loss, no bandwidth cap). Two stages,
one request at a time.
