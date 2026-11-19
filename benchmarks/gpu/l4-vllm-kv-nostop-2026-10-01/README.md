# The engine does not stop on content that stores poorly: a reserve in place of a stop

Qwen3-8B, bf16 weights, vLLM 0.30.0, the dev L4 (g6.4xlarge, 22 GiB), 2026-10-01.

**The question.** The first version of `kv: lossless` stopped the engine, fail closed, when a prompt's content did not store at 12 bits in the room it had for it; a serving engine that stops on a prompt's content is not shippable. What bounds that content, what holds it, and what does it cost in KV capacity?

**What was found.** Natural text, chat prompts, text in 14 languages, code and structured text, and random token ids all fit the first version's room. A long run of one repeated token did not: it needs far more, over all 151,936 token ids of the embedding matrix and in a cache filled entirely with one such token (the engine would have stopped on a long repetition).
The cache now keeps a reserve of its own for this content, as a share of the KV cache (1.3% in this version), allocated at load so that vLLM's profile counts it. The engine stops only where the reserve is out.

## Capacity (vLLM's "GPU KV cache size" at the standard flags: eager, `--max-model-len 8704`, 8,704 tokens a step, 0.9, bf16 weights; stock 25,232 tokens; `capacity/stock.txt`)

| the reserve's share of the cache | tokens | blocks | against stock's |
| :--- | ---: | ---: | ---: |
| 1.3% (the default) | 32,560 | 2,035 | 1.2904x |
| 1% | 32,656 | 2,041 | 1.2942x |
| 0.5% | 32,816 | 2,051 | 1.3006x |
| none (the first version) | 32,480 | 2,030 | 1.2873x |

The first version's figure was 32,576 tokens, **1.2911x**. The default (a reserve of 1.3% of the cache: 45.4 MiB) is **1.2904x: the same capacity, the stop closed**. A reserve of 1.0% (1.2942x) is short for 101 of the 151,936 tokens when they fill the whole cache; 0.5% gives 1.3006x.

## The stress test (`gpu/vllm/kv_stress.py`)

**Capture** (`kv_stress.py capture --pct 1.3 --vocab`): vLLM's own keys and values, as every layer hands them to its cache (after the key norm and the rotary embedding), written by Glyd's kernel into 1,000 blocks a layer (36 layers; blocks of a previous owner's random bytes; written round and round as vLLM's block allocator reuses them) with a reserve of 1.3% of those blocks' bytes, and read back and compared bit for bit, step by step.

| population | prompts x tokens | values | differ |
| :--- | :--- | ---: | ---: |
| natural | 5,948 x 64 | 28,066,185,216 | 0 |
| chat | 1,470 x 64 | 6,936,330,240 | 0 |
| multilingual | 574 x 64 | 2,708,471,808 | 0 |
| code and structured | 460 x 64 | 2,170,552,320 | 0 |
| random: the embedding matrix | 2,000 x 64 | 9,437,184,000 | 0 |
| random: the vocabulary | 2,000 x 64 | 9,437,184,000 | 0 |
| random as vllm bench makes them (decoded, encoded) | 2,000 x 64 | 9,437,184,000 | 0 |
| one token repeated (256) | 203 x 256 | 3,831,496,704 | 0 |
| short cycles of random tokens (256) | 200 x 256 | 3,774,873,600 | 0 |
| a phrase repeated (256) | 150 x 256 | 2,831,155,200 | 0 |
| one token repeated (4096) | 24 x 4,096 | 7,247,757,312 | 0 |
| natural, 12k tokens | 2 x 12,288 | 1,811,939,328 | 0 |
| every token id repeated 32 times | 151,936 x 32 | 358,461,997,056 | 0 |
| **total** | | **460,647,825,408** | **0** |

Nothing was dropped or lost in any of these runs. Then **the cache filled entirely** with one token repeated (4 prompts of 4,096 tokens each, more blocks than the cache holds, so that at the end every block of every layer is that token's), for the 12 tokens that took the most room: the reserve of 1.3% held them all, and its books agreed with the blocks' afterwards.
With a reserve of **1.0%** the same fill ran it out for the tokens that need more: values were read back wrong, and the engine's step would have raised. That is what the stop is for, and where the 1.3% comes from.

**Engine** (`kv_stress.py engine`): vLLM with `kv: lossless` on the same kinds of prompts through `generate()`: nothing raised `GlydKVOverflow` or `GlydKVVerifyError`. With `--verify` (a bf16 shadow of every layer's cache, every write compared, 700 blocks): 32,157,573,120 values compared, none differ. Without it, the full cache (2,378 blocks), no engine stop either.

| workload | prompts | values compared (verify run) |
| :--- | ---: | ---: |
| sinks: chat prompts (the first tokens of every sequence), 16 new tokens | 147 | 2,018,746,368 |
| natural windows, 32 new tokens | 300 | 9,096,634,368 |
| text in fourteen languages, 64 new tokens | 126 | 14,596,153,344 |
| code, tables, hex, base64, 32 new tokens | 100 | 16,955,449,344 |
| random token ids over the embedding matrix, 32 new tokens | 200 | 21,674,041,344 |
| random 1,024-token prompts as vllm bench makes them, 128 new tokens (ignore EOS) | 48 | 29,410,172,928 |
| a phrase or a few characters repeated (480 tokens), 256 new tokens (its continuation repeats) | 5 | 30,324,400,128 |
| the cache filled entirely: 10 prompts of one token repeated 1,000 times, 16 new tokens | 10 | 31,156,051,968 |
| one token repeated 4,000 times, 64 new tokens | 3 | 32,157,573,120 |

(The counts are cumulative over the run.)

## Kernels

`check_vllm.py --kv --kernels-only`: 27 checks passed, 0 failed: every bit pattern; every 16-bit pattern; recycled blocks and rewrites into blocks of a previous owner's bytes (8, 3, 16 and 32 heads); the reserve's cases (taking, giving back, running out); decode against an fp32 reference.

`compute-sanitizer` over the kernels' quick checks (the same cases): `memcheck`: ========= ERROR SUMMARY: 0 errors; `initcheck`: ========= ERROR SUMMARY: 0 errors; `racecheck`: ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings); `synccheck`: ========= ERROR SUMMARY: 0 errors (`san/`).

Kernel times (`kv_decode_bench.py`, the same shapes as the first version's, `decode/`; each timed as a loop of calls from Python, so a call's time includes its launch, which is most of a small kernel's: the graph-timed numbers are in [`l4-vllm-kv-graphs-2026-10-01`](../l4-vllm-kv-graphs-2026-10-01)): the write of a decode step's token 10.9-11.6 us warm (the first version 10.4-11.2), 9.2-19.5 us cold (the first version 9.2-23.6); decode attention unchanged (32 x 1,024: 392.7 us, the first version 393.3; 4 x 8,192: 396.0).

Decode attention over content that stores poorly, by how much of it needs extra storage (32 x 1,024 / 8 x 4,096): none 1.00x / 1.00x, 0.1% of the values 1.03x / 1.03x, 0.3% 1.09x / 1.09x, 1.0% 1.51x / 1.46x. A cache of one repeated token's blocks costs decode attention a tenth to a half more; natural text nothing measurable (+3%).

## Files

`capacity/` (stock's run of the cache's size), `decode/` (the kernel timings), `san/` (the sanitizer's report). The tools are in [`gpu/vllm/`](../../../gpu/vllm). Logs that carry the plugin's internal diagnostics are not part of this record.
