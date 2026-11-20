# The prompt products' tiles on an RTX 4080 SUPER (2026-11-20)

Lever 3 of the follow-up to [rtx4080s-batched-prefill-2026-11-20](../rtx4080s-batched-prefill-2026-11-20): the engine's 2,000-token prompt pass was 372 ms against vLLM's 357 (88 of the card's ~104
TFLOPS at fp32 accumulation; 32.8 TFLOP a pass). **It is now 353 ms (93 TFLOPS, 89% of that peak; the first token with one 2,000-token prompt 0.353 s against vLLM's 0.357), by the choice of tile alone:
the same sums, the same bits.**

## What was measured

`glyd-engine bench Qwen/Qwen3-8B --prompt N --chunk 2048` (`GPU time by kind of work`, `--profile-prefill`): 2,000 tokens, 372 ms, 95% in the four dense products (gate and up 168 ms, down 97, qkv 42,
o 33), attention 17 ms. `GLYD_ENGINE_TILE` (a development build) forces a tile; `scripts/tile_sweep.sh` is the sweep. Pass time, ms, tile forced for the plain products:

| prompt | 256x128 s3 (the pick from 1,536 tokens) | 128x128 s4 | 128x128 b64 s3 | 128x256 s3 |
| :--- | ---: | ---: | ---: | ---: |
| 600 | 156.3 | 124.0 | 124.2 | 129.3 |
| 1,000 | 200.3 | 194.5 | 194.5 | 200.8 |
| 1,500 | 284.1 | 266.7 | 266.7 | 284.4 |
| 1,800 | 367.3 | 330.8 | 331.0 | 338.6 |
| 2,048 | 372.8 | 353.3 | 353.2 | 372.8 |
| 4,000 | 771.1 | 731.4 | 731.4 | 771.6 |

Below 1,536 tokens the engine already picked 128-wide tiles (the defaults equal the 128x128 column there). The gain is in the 4,096 and 6,144-column products (o, down, qkv): with 80 multiprocessors and a block
each, 2,000 x 4,096 in 256x128 tiles is 256 blocks, 4 waves with the last at 0.2 full, in 128x128 tiles 512 blocks in 7 waves (14% less padded work). The fused gate-and-up product has two tiles of one size (256x128 and
128x256): from 1,536 tokens the pick is the one with fewer waves (1,800 tokens: 315.5 ms against 330.8; 2,000 and 2,048 the same either way). `Kern::pick_tile` and `gemm_silu` choose by that count, the bigger tile
kept unless the smaller has more than 3% less padded work. A block adds its k steps in the same order whatever its tile, so the logits are the same bits: 10 prompts of 96 tokens, one of them 2,000 tokens, alone and
together, cold and warm, are bit for bit the engine of main's (tokens and logprobs).

An overlap of the matrices' decoding (a matrix decoded a product ahead on a second stream, `unpack` 4 ms to 8 ms a matrix) was tried first and measured: 0.3% (370.4 against 371.4 ms); the products' blocks fill
the multiprocessors, so the decode waits for them. Not kept.

## First token and decode, engine against vLLM (two runs of the engine; vLLM's the record's)

| load | before | after | vLLM | after / vLLM |
| :--- | ---: | ---: | ---: | ---: |
| ~2000 tokens, 1 at once | 0.371 s | 0.353 | 0.357 | 0.99x |
| ~2000 tokens, 4 at once | 0.949 | 0.903 | 0.881 | 1.02x |
| ~2000 tokens, 8 at once | 1.718 | 1.636 | 1.584 | 1.03x |
| ~4000 tokens, 1 at once | 0.696 | 0.676 | 0.734 | 0.92x |
| ~4000 tokens, 4 at once | 1.761 | 1.705 | 1.825 | 0.93x |
| ~4000 tokens, 8 at once | 3.190 | 3.114 | 3.300 | 0.94x |

Decode is untouched (58.7 / 49.0 / 40.3 tokens/s a user at 2,000 tokens with 1 / 4 / 8 users, 1.00x to 1.02x vLLM's); the rest of the table is the batched-prefill record's.
Remaining against the bar: 2,000 tokens with 8 at once 3% behind (the decode steps between the prompts), and the short prompts at 4 and 8 at once.
