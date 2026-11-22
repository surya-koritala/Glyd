# A start shared between requests, RTX 4080, gemma-4-12B at wren (2026-11-22)

On the models the description executor serves with paged keys and values, a request that begins as another one does now names that sequence's blocks for the start (whole blocks of
16 tokens) and runs only the rest, in a slot of its own; and a request whose start another is still running waits for it while at least a chunk of that start is left to run, then
shares it (R4c; `GLYD_ENGINE_KEEP=0` turns both off, with the kept sequences of the record beside this one). This record measures both with `glyd-engine bench`'s `same` loads: every
request has the same notes (1k or 4k tokens) and its own question after them, as requests with one system prompt do.

**Result: eight requests that begin with the same 4k tokens, arriving together at a server that has never seen them, have their first token in 1.37 s where it took 5.0 s, and make
298 tokens/s in all where they made 87. Once the start is in the server, a request's first token comes in 0.04 to 0.2 s. With a 1k start, waiting adds nothing to sharing (0.73 s
to the first token either way, 1.20 s with neither): there is less than two chunks of it to wait for.**

Machine: a rented RTX 4080 (16 GB, driver 595.99.02, Threadripper PRO 3955WX), nothing else on the GPU; the same machine as `../rtx4080-kept-sequences-2026-11-22`. Engine: main
d47bf927 with M2 batch 11, built by the GPU job's recipe (`scripts/ci_gpu_build.sh`, kernels for sm_89). Model: `google/gemma-4-12B-it:wren`. `bench` started the server itself with
8 users, no draft model; 128 new tokens a request, temperature 0; no request failed.

## A start the server has never seen, eight requests at once

One round with no warm-up round of the load before it (`--runs 1 --warmup 0`), a new server each time, three times (`new/`). "Shared, not held" is this batch's first commit, which
shares a start but lets every request in at once; the other two columns are the batch's head.

| load | figure | nothing kept or shared | shared, not held | shared and held (the default) |
| :--- | :--- | ---: | ---: | ---: |
| 4k start, 8 at once | first token | 5.018 s | 4.961, 4.886, 4.885 | 1.370, 1.369, 1.369 |
| | decode tokens/s a user | 20.0 | 23.4, 23.3, 23.3 | 41.2, 41.3, 41.2 |
| | tokens/s in all | 86.6 | 105.3, 105.5, 105.6 | 297.5, 298.0, 297.7 |
| | 99th percentile gap between tokens | 167 ms | 165, 164, 165 | 86, 86, 86 |
| | prompt tokens found | 0% | 52% | 83% |
| 1k start, 8 at once | first token | 1.201 s | 0.734, 0.733, 0.866 | 0.735, 0.734, 0.734 |
| | tokens/s in all | 222.5 | 304.7, 294.6, 281.4 | 294.2, 294.4, 329.2 |
| | prompt tokens found | 0% | 53%, 48%, 41% | 48%, 48%, 63% |

- Held, the first request runs the start and the other seven wait until less than a chunk of it (512 tokens on this model) is left: they share 3,584 of the 3,760 tokens and run
  the rest, which is the 83% (7 x 3,584 of 8 x 3,760).
- Not held, the eight are let in as they come: those in the first round each run the whole prompt beside one another, the later ones share what is run by then.
- A 1k start is less than two chunks: a request waits one round and shares the first 512 tokens either way, so holding changes nothing there.
- One of the six "not held" servers started while the GPU was still busy with the one before (27% utilized; `new/not-held-same4k-2.json` is marked contended); its figures are the
  other two runs'.

## A start the server holds

Two rounds after a warm-up round of the same load (`shared.json`, `not-shared.json`), the batch's head. The requests of a round have new questions, so each finds the notes and
not its question.

| load | figure | nothing kept or shared | the default |
| :--- | :--- | ---: | ---: |
| 1k start, 8 requests 250 ms apart | first token | 0.327 s | 0.043 |
| | decode tokens/s a user / in all | 36.1 / 222.6 | 58.1 / 267.6 |
| | 99th percentile gap between tokens | 144 ms | 35 |
| | prompt tokens found | 0% | 96% |
| 4k start, 8 requests 250 ms apart | first token | 4.157 s | 0.053 |
| | decode tokens/s a user / in all | 20.0 / 86.4 | 49.6 / 245.8 |
| | 99th percentile gap between tokens | 167 ms | 43 |
| | prompt tokens found | 0% | 99% |
| 4k start, 8 at once | first token | 5.039 s | 0.198 |
| | decode tokens/s a user / in all | 20.0 / 86.3 | 44.6 / 340.3 |
| | 99th percentile gap between tokens | 167 ms | 23 |
| | prompt tokens found | 0% | 99% |

**These rows are mostly the kept sequences of the record beside this one, not sharing.** After a round of eight requests the server holds eight finished sequences that begin with
the notes, and by the scheduler's rule each request of the next round goes on in one of them, cut back to the notes. (Main without this batch was not run on these loads, so the
two are not told apart by a measurement here.) Sharing is what serves a request when there is no kept sequence for it to take: the table above, and a request that comes while the
others still run.

## Other loads

As before, on the batch's head (`standing-1.json`, `standing-2.json`; against `../rtx4080-kept-sequences-2026-11-22`, same machine, same day): a short prompt alone 0.024 s to the
first token and 86.6 tokens/s; 1k and 4k prompts alone 0.264 and 1.091 s, 82.1 and 79.0 tokens/s; eight short prompts 478 tokens/s in all; eight 1k prompts 250 ms apart 0.330 s,
222.7 tokens/s in all, a 99th percentile gap of 144 ms; eight 4k prompts 500 ms apart 3.288 s, 86.3 and 167 ms. Conversations of 4 turns: a later turn's first token 0.076 s (1k, one
conversation), 0.105 s (1k, eight, 272.4 tokens/s in all), 0.090 s (4k, one). Eight conversations of 4k tokens still do not fit this card's cache kept (nothing found; a later turn's
first token 3.45 and 2.90 s in the two rounds, where the record beside this one has 3.45 and 2.67 kept and 3.14 and 2.76 not kept).

## What it does not say

- One model, one card. The at-once figures are three rounds of one each, the others two rounds.
- **A start is shared only from a sequence that has grown no more than a chunk past it, on a model with layers whose attention slides (Gemma 4, Muse Spark).** Those layers keep
  a window and a chunk behind a sequence's end and give the blocks before back as it grows, so a long conversation no longer holds what a step at its system prompt's end reads, and
  a new conversation with the same system prompt runs it again. `keepcheck` states and checks this limit. Blocks found by what they hold, with no sequence needed to hold them, are
  what lifts it (R4c's index, not built).
- A model with recurrent states (the Qwen3.5 layout) shares a start only where another request's prompt ended (its states are copied from there), and holds no request back; the
  latent-cache models (Kimi-K3, GLM-5.3-Flash) share nothing. Both were checked for their bits on random models (`glyd-engine keepcheck`), not timed.
- Whether a reply is the same as without sharing. It is not, to the bit, for the same cause as with kept sequences: a request that shares a start runs the rest of its prompt in
  passes of other sizes than the whole prompt's. What `keepcheck` holds, on every family's random model and on this model, is that a sequence that shared a start is, bit for bit, a
  sequence that ran the same tokens in the same passes on its own; that the sequence it shared from is unharmed; and that the shared blocks outlive either.
- Callers that must not learn of each other's prompts: a shared start shows in how soon the first token comes. `cache_salt` gives a caller a scope of its own (docs/serve-api.md).
