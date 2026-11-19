# glyd-engine on glyd-v4 packs and the library of C API 15 (2026-10-04)

The library's smallest layout has a new format (glyd-v4): each matrix is held in the smaller of Glyd's two layouts, the tiered one or the 12-bit one where that takes fewer bytes, and a pack names its own. glyd-engine loads each pack in
the layout it names (a directory of glyd-v4 mixes them), packs the embedding in the smaller one too, and runs against the library of C API 15; a copy of the earlier tiered layout (glyd 0.27 and before, glyd-v1 and glyd-v2), which the library no longer
reads, is packed again, once, the first time it is used from the engine's cache, with its KV tables kept, and refused in words where it is a directory the user keeps. Everything here ran on the L4 (AWS g6.4xlarge).

## What the directories hold

```
Qwen3-8B: format glyd-v4; 139 packs in the tiered layout and 6 in the 12-bit layout; the embedding in the tiered layout; in the 12-bit layout: layer 1 down_proj, layer 1 gate_proj, layer 2 down_proj, layer 2 gate_proj, layer 3 down_proj, layer 3 gate_proj
Qwen3-4B: format glyd-v4; 139 packs in the tiered layout and 5 in the 12-bit layout; the embedding in the tiered layout; in the 12-bit layout: layer 1 gate_proj, layer 2 down_proj, layer 2 gate_proj, layer 3 down_proj, layer 3 gate_proj
```

## Against the engine before the change

`check` runs the reference's 8 short prompts (5 to 31 tokens) and 3 long ones (1,030, 2,100 and 3,500 tokens), 64 greedy tokens each, and writes the tokens and the score (log probability) of each; `checks/check*.json`.

| the two runs | tokens the same | each prompt's largest difference of scores |
| :--- | ---: | :--- |
| Qwen3-8B: the engine before (earlier library, tiered layout) against the engine now (glyd-v4) | 704 of 704 | 31 tokens: 0.101; 18 tokens: 0.098; the other 9 prompts: identical |
| Qwen3-4B: the same | 704 of 704 | 31 tokens: 0.117; 18 tokens: 0.066; the other 9 prompts: identical |
| Qwen3-8B in the 12-bit layout alone: earlier library against the new one | 704 of 704 | all 11 prompts: identical |
| Qwen3-8B, earlier library: the tiered layout against the 12-bit layout alone | 704 of 704 | 31 tokens: 0.189; 18 tokens: 0.103; the other 9 prompts: identical |
| Qwen3-8B, new library: glyd-v4 against the 12-bit layout alone | 704 of 704 | 31 tokens: 0.189; 18 tokens: 0.065; the other 9 prompts: identical |

- The tokens are the same in all of them. The scores of the prompts of 5 to 16 tokens and of 1,030 tokens and more are identical, to six digits, between the engine before and the engine now; the two prompts of 18 and 31 tokens differ, by at most 0.10 on the 8B and 0.12 on
  the 4B. The comparisons below them say where from: with the earlier library alone, a matrix held in the 12-bit layout instead of the tiered one changes the scores of those two prompts (0.19 and 0.10), and the 12-bit layout alone
  gives the same scores on the earlier library and on the new one. Six of the 8B's matrices (the gate and the down of layers 1 to 3) are in the 12-bit layout in glyd-v4, five of the 4B's: what changed the two prompts' scores is the layout a matrix is held in, for prompts of 17 to 64 tokens.
- One sequence's logits in a decode step with 0 to 7 others (`checks/rowcheck8-*.out`): the same bit for bit at every number of rows and in either row, and the same hash before (4351ad4cbd71c125) and now (4351ad4cbd71c125).
- The cache's known-answer test on the new library (`checks/kvtest8-after.out`): 53,084,160 values written and read back, **0 differ**.
- A 1,024-token prompt and 64 tokens (`checks/bench8-*.out`, three timed runs): the weights take 10.365 GiB before and 10.340 now; a decode step 46.05 ms before (21.72 tokens a second) and 45.05 ms now (22.20); the prompt 385 ms before and 377 ms now.

## A copy of the earlier layout, used for the first time

A copy of Qwen3-8B in the engine's cache, as an earlier run left it (glyd-v1, with its KV tables), run with `glyd-engine run Qwen/Qwen3-8B --context 4096` twice (`checks/first-use.out`):

```
-- run 1
The packed copy of Qwen3-8B is of the layout of glyd 0.27 and before, which this glyd does not read: packing it again for this GPU (mma), once: (the cache's directory)
Packed in 2m16s (12.4 GB).
Ready in 2m19s: Qwen3-8B on the NVIDIA L4, 10.3 GiB of weights, 4096 tokens of context.
Hello, how can I help?
  its glyd.json: glyd-v4
  its tables: ['kv-tables-4096.safetensors']
-- run 2
Ready in 2s: Qwen3-8B on the NVIDIA L4, 10.3 GiB of weights, 4096 tokens of context.
Hello, how can I help?
  its glyd.json: glyd-v4
  its tables: ['kv-tables-4096.safetensors']
```

The first run packs the model again (the same packing as `glyd-engine pack`, which took 30 s for the 8B when run on its own a few minutes before: `checks/pack.out`), keeps the tables, and answers; the second finds it made.
