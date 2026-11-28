# Recurrent states kept at block boundaries, RTX 4080, Qwen3.5-9B at linnet (2026-11-22)

A model with recurrent layers (Qwen3.5's Gated DeltaNet, Nemotron-H's Mamba-2) carries a sequence's whole past in its states, which are what they are after its last
token. Until this change the description executor could go on from such a sequence only at its mark, its prompt's end, and only while the sequence was still in a slot;
blocks found by name (the record `../rtx4080-blocks-by-name-2026-11-22`) were off for these models. Now the states are copied, every chunk of a prompt, into a pool of
fixed size made at load (512 MB by default, at most 16 copies; src/dynamic/snaps.rs), under the name of the block ending there; the scheduler ends a prompt's pieces on
those boundaries, so a copy is the states after that token exactly, and a request takes the newest boundary of its start whose states are kept and whose attention
blocks are held by name (#142). That is the Qwen3.5 / 3.8 layout today; Nemotron-H's attention is not paged yet.

**Result: on Qwen3.5-9B, eight conversations that begin with the same 4k tokens have their later turns' first token in 0.134 s, where they took 1.141 s, and make
233.2 tokens/s in all, where they made 107.7; one conversation's later turns 0.098 to 0.125 s, where they took 0.93 s. Loads with nothing in common are as before.**

Machine: a rented RTX 4080 (16 GB, driver 595.91.07, 350 W, AMD Ryzen 9 3900X), nothing else on the GPU. Engine: main 6c598eca with M2 batch 20, built by the GPU
job's recipe (`scripts/ci_gpu_build.sh`, kernels for sm_89). Model: `Qwen/Qwen3.5-9B:linnet` (6.2 GB). `bench` started the server itself (8 users, no draft model);
128 new tokens a turn, temperature 0, a warm-up round and two rounds a load; no request failed. "Off" is the same binary with `GLYD_ENGINE_STATE_SNAP_MB=0` (no pool:
the behaviour before this change), in the same session.

| load | figure | off | the pool (the default) |
| :--- | :--- | ---: | ---: |
| the same 4k start, 4 turns, 8 conversations (`same4k*4:8`) | first token, first turn | 3.895 s | 2.059 |
| | first token, later turns | 1.141 s | 0.134 |
| | decode tokens/s a user / in all | 16.9 / 107.7 | 46.0 / 233.2 |
| | 99th percentile gap between tokens | 466 ms | 460 |
| | prompt tokens found | 0% | 72% |
| the same 4k start, 4 turns, 1 conversation (`same4k*4:1`) | first token, first turn | 0.868 s | 0.463 |
| | first token, later turns | 0.937 s | 0.125 |
| | tokens/s in all | 63.1 | 87.1 |
| a 4k prompt, 4 turns, 1 conversation (`4k*4:1`) | first token, later turns | 0.932 s | 0.098 |
| | tokens/s in all | 60.2 | 86.2 |
| | prompt tokens found | 0% | 61% |

Loads with nothing in common (`standing.json` with the pool, `standing-off.json` without): a short prompt alone 0.021 / 0.020 s and 104.0 tokens/s both; eight short
prompts 599.6 / 600.8 tokens/s in all; 1k and 4k prompts alone 0.224 / 0.224 s and 0.859 / 0.855 s, 102.9 and 99.8 tokens/s both.

- Without the pool, a single conversation's later turns found nothing: its kept slot can go on only from its mark, and Qwen3.5's chat template ends a generation prompt
  with `<|im_start|>assistant\n<think>\n\n</think>\n\n` but writes an earlier reply back as `<|im_start|>assistant\n` and its text, so the next turn's tokens part from
  the kept ones just before the mark (tests/fixtures/chat_qwen35_tools.jinja). With the pool, the boundaries before that place are found by name.
- A first turn finds the 4k start that the round before's conversations ran (each round's conversations begin with the same notes and ask new questions): 0.868 s to
  0.463 s alone.
- The checks on the same machine: `keepcheck qwen35` with boundaries 32 tokens apart (a conversation that begins as another does: 32 of its 90 prompt tokens found,
  its reply the same passes' by hand, token for token; and every case before it), every family's keep, round and user checks, and the logits suite as on main's runs.

## What it does not say

- One model and one card, two rounds a load. Nemotron-H does not take the pool yet: its attention is not on the paged pools (src/dynamic/rows.rs `pageable`), so its
  states are kept at a prompt's end alone, as before this change.
- What the copies cost: a copy is the sequence's states, 49.1 MB on Qwen3.5-9B (its 24 linear layers' convolution and recurrent states, from its config by the
  executor's sizes), so the default pool holds 10; one is made on the GPU's stream at each boundary a prompt reaches. The loads with nothing in common show no change the
  runs can tell; the copies' time was not measured apart.
- A boundary every chunk (the server's) means a start is found to the last chunk's end within it; `GLYD_ENGINE_STATE_EVERY` sets another spacing, not measured here.
- A pool under pressure: with more starts than copies the least recently used is written over; these loads have a few. Not measured.
