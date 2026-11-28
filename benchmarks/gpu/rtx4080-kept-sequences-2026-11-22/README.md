# A finished sequence kept for the conversation's next turn, RTX 4080, gemma-4-12B at wren (2026-11-22)

On the models the description executor serves, a finished request's keys, values and states now stay in its slot, and a request that begins with its tokens goes on from them (R4c's
first step; `GLYD_ENGINE_KEEP=0` empties the slot at once, as before). This record measures both on conversations: `glyd-engine bench`'s `1k*4` and `4k*4` loads, 4 turns each, every
later turn resending the conversation and adding a short question, as a chat client does.

**Result: a later turn's first token comes in 0.08 to 0.11 s where it took 0.34 to 1.17 s, with 70% to 73% of the measured prompt tokens found in a slot, and eight conversations at
once make 47% more tokens/s in all. Where the cache cannot hold the conversations (eight of 4k tokens on this card) nothing is found and nothing is lost.**

Machine: a rented RTX 4080 (16 GB, driver 595.99.02, Threadripper PRO 3955WX), nothing else on the GPU; the same machine as the second half of
`../rtx4080-one-step-round-2026-11-22`. Engine: main fdc44922 with M2 batch 9, built by the GPU job's recipe (`scripts/ci_gpu_build.sh`, kernels for sm_89). Model:
`google/gemma-4-12B-it:wren`. `bench` started the server itself with 8 users, no draft model; 2 runs a load after a warm-up round, 128 new tokens a turn, temperature 0. Both columns
were measured in the same session, one after the other (`kept.json`, `not-kept.json`); no request failed.

| load | figure | not kept | kept (the default) |
| :--- | :--- | ---: | ---: |
| 1k, 4 turns, 1 conversation | first token, first turn | 0.263 s | 0.263 |
| | first token, later turns | 0.341 s | 0.076 |
| | tokens/s in all | 69.8 | 78.9 |
| | prompt tokens found | 0% | 70% |
| 1k, 4 turns, 8 conversations | first token, first turn | 1.206 s | 1.205 |
| | first token, later turns | 0.683 s | 0.105 |
| | decode tokens/s a user / in all | 28.0 / 187.5 | 39.3 / 275.2 |
| | prompt tokens found | 0% | 70% |
| 4k, 4 turns, 1 conversation | first token, first turn | 1.091 s | 1.092 |
| | first token, later turns | 1.169 s | 0.091 |
| | tokens/s in all | 49.8 | 75.5 |
| | prompt tokens found | 0% | 73% |
| 4k, 4 turns, 8 conversations | first token, first turn | 5.041 s | 5.039 |
| | first token, later turns | 2.950 s | 3.059 |
| | decode tokens/s a user / in all | 13.9 / 79.2 | 14.0 / 80.3 |
| | prompt tokens found | 0% | 0.2% |

- "Prompt tokens found" is the server's count over the measured rounds, first turns included: a first turn finds nothing, so three turns of four finding nearly all of theirs is about 70%.
- A later turn still runs what the slot does not hold: the new question, the template around it, and the end of the last reply where its text does not make the same tokens again
  (Gemma 4 is cut back to where the two part).
- **Eight conversations of 4k tokens do not fit this card's cache kept.** A kept sequence gives its room up before a request waits or a running sequence is sent back, so each was
  emptied before its next turn came: nothing found, and the figures are the ones without keeping (within 4%). Fewer bytes a token (12-bit pages, R4f) or moving a kept sequence to
  host memory (R4d) is what would keep them.
- Loads that are not conversations are as before (`gate-and-arrive.json`, kept by default, against `b-one-step-default.json` of the record beside this one, same machine, same day):
  1k and 4k prompts alone 0.264 and 1.092 s to the first token and 82.1 and 79.0 tokens/s; eight 1k prompts 250 ms apart 0.331 s, 222.5 tokens/s in all, a 99th percentile gap of
  144 ms; eight 4k prompts 500 ms apart 3.289 s, 86.3 and 168 ms. The bench's prompts differ from their first token, so nothing of them is found in a slot.

## What it does not say

- One model, one card, two runs a load. A model with recurrent states (the Qwen3.5 layout: its states are copied at each prompt's end, so it goes on from there or from the slot's
  end) and the latent-cache models (Kimi-K3, GLM-5.3-Flash: the whole slot or nothing) were checked for their bits (`glyd-engine keepcheck`, random models), not timed.
- Whether a reply is the same as without keeping. It is not, to the bit: a turn that goes on in a slot runs the rest of its prompt in a pass of fewer rows than the whole prompt's,
  which is another kernel's (the record beside this one measures that rounding). What `keepcheck` holds, on every family's random model and on this model, is that a slot gone on
  with is, bit for bit, a slot that ran the same tokens in the same passes and never ended.
- Two requests that begin with the same tokens at the same time: the second finds nothing while the first runs (blocks are not shared between sequences yet; R4c's index).
