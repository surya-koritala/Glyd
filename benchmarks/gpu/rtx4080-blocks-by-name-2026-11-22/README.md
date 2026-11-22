# A start found by what it holds, RTX 4080, gemma-4-12B at wren (2026-11-22)

The description executor's blocks now have names (R4c: a full block of 16 tokens is named by its tokens and all before them, `kv::block_hash`), a pool finds its blocks by name,
and a named block that no sequence holds stays, with its bytes, until its memory is needed. So a prompt finds the start it has in common with an earlier request whoever holds
those blocks, or no one. The batch before this one (`../rtx4080-shared-starts-2026-11-22`) shared a start only from a sequence in a slot that had grown no more than a chunk past
it: on a model with sliding layers a conversation gives those blocks up as it grows, and a new conversation with the same system prompt ran it again.

This record measures that case with conversations: `glyd-engine bench`'s `same4k*4`, 4 turns each, every conversation beginning with the same 4k tokens of notes and its own
question, every later turn resending the conversation and adding a short question, as a chat client with one long system prompt does. A round is 8 conversations at once (or 1); two
rounds after a warm-up round, each round's conversations with new questions.

**Result: a new conversation that begins with 4k tokens the server has run before has its first token in 0.05 s alone and in 0.21 s with seven others, where it took 1.09 s and
3.44 s: the conversations that had run the start had grown past it or ended. Eight such conversations make 291 tokens/s in all (245 on the batch before, 80 with nothing kept or
shared) and find 97% of their prompt tokens. Loads whose prompts have nothing in common are as before.**

Machine: a rented RTX 4080 (16 GB, driver 595.99.02, Threadripper PRO 3955WX), nothing else on the GPU; the same machine and session as the second half of
`../rtx4080-shared-starts-2026-11-22`. Engine: main d47bf927 with M2 batches 11 and 12, built by the GPU job's recipe (`scripts/ci_gpu_build.sh`, kernels for sm_89). Model:
`google/gemma-4-12B-it:wren`. `bench` started the server itself with 8 users, no draft model; 128 new tokens a turn, temperature 0; no request failed. The batch before was
measured in the same session from its own build (`batch-11-conversations.json`).

| load | figure | nothing kept or shared | batch 11: a start shared from a sequence in a slot | blocks found by name (the default) |
| :--- | :--- | ---: | ---: | ---: |
| the same 4k start, 4 turns, 8 conversations | first token, first turn | 5.039 s | 3.438 | 0.208 |
| | first token, later turns | 2.748 s | 0.115 | 0.130 |
| | decode tokens/s a user / in all | 13.8 / 80.0 | 37.2 / 244.7 | 38.8 / 290.6 |
| | 99th percentile gap between tokens | 170 ms | 92 | 91 |
| | prompt tokens found | 0% | 88% | 97% |
| the same 4k start, 4 turns, 1 conversation | first token, first turn | not run | 1.092 s | 0.046 |
| | first token, later turns | | 0.091 s | 0.090 |
| | tokens/s in all | | 75.5 | 75.6 |
| | prompt tokens found | | 73% | 97% |

- The first turn is where the two differ: a round's conversations begin with notes that the conversations of the round before ran. Those have grown three turns past the
  notes: batch 11 can neither go on in one of them from the notes' end nor share the notes from it (its sliding layers gave those blocks up), and runs them; this batch finds
  the blocks.
- The later turns' first token is 0.130 s to batch 11's 0.115 with eight conversations (0.114 and 0.146 in the two rounds): the eight are in step now, and their later turns come
  together.
- Eight conversations that have 4k tokens in common fit this card's cache. Eight of 4k with nothing in common still do not (`standing.json`: 0.3% found, as in the records before).

## Other loads

Requests that begin alike, as in the record before (`shared.json`, `new/`): eight with a 4k start the server has never seen, at once, 1.370, 1.371 and 1.369 s to the first token
and 297.5, 297.9 and 297.8 tokens/s in all (83% found); with a 1k start 0.734, 0.733, 0.733 s and 294.4, 294.3, 294.8 (48%); with a start the server holds, 0.043 s and 267.6
tokens/s (1k, 250 ms apart), 0.052 s and 245.9 (4k, 250 ms apart), 0.192 s and 339.6 (4k, at once). The record before has the same figures, but for one of its three 1k runs
(329 tokens/s, 63% found).

Loads with nothing in common (`standing.json`, against `../rtx4080-kept-sequences-2026-11-22` and the record before): a short prompt alone 0.024 s and 86.6 tokens/s; 1k and 4k
prompts alone 0.263 and 1.091 s, 82.1 and 79.0 tokens/s; eight short prompts 484 tokens/s in all; eight 1k prompts 250 ms apart 0.330 s, 222.3 tokens/s, a 99th percentile gap of
144 ms; eight 4k prompts 500 ms apart 3.290 s, 86.2 and 168 ms. Conversations of 4 turns with nothing in common: a later turn's first token 0.076 s (1k, one conversation), 0.105 s
(1k, eight, 274.3 tokens/s in all), 0.090 s (4k, one); eight of 4k, which do not fit, 2.91 and 2.78 s.

## What it does not say

- One model, one card, two rounds a load (three single rounds for the start never seen).
- How long a block that no sequence holds stays: until its pool needs the memory, the oldest first. Under a load that fills the cache with other prompts a start is run again.
  This record's loads leave room; nothing here measures a cache under pressure from unrelated prompts.
- What the names cost. A prompt's names are one SHA-256 of 80 bytes for every 16 tokens, on the host, when the request is let in; the loads with nothing in common show no change
  that the runs can tell, and the cost was not measured apart.
- Models with a state: the Qwen3.5 layout still shares a start only where another request's prompt ended, from a sequence in a slot; Kimi-K3 and GLM-5.3-Flash share nothing.
  Their blocks are not named.
- Whether a reply is the same as without it. It is not, to the bit, for the same cause as in the records before: a request that finds a start runs the rest of its prompt in
  passes of other sizes than the whole prompt's. What `keepcheck` holds, on every family's random model and on this model, is that a sequence that found a start by name is, bit
  for bit, a sequence that ran the same tokens in the same passes on its own, after the sequence that ran them has ended and while it holds far more; and that that sequence is
  unharmed.
