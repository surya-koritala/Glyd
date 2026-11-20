# The speed bar on the RTX 4080 SUPER with models of the last 6 months: Gemma 4 12B fails it, so the list is per family (2026-11-20)

The owner's rule: the models a measurement uses were released within the last 6 months (cutoff 2026-05-20; the Hub's `createdAt`). Qwen3-8B (2025-04-27), on which the 4080 SUPER's line in `GPUS` was measured,
no longer counts. What qualifies and fits a 16 GB card, checked with `glyd-engine fit`/`preflight` on glyd-pc:

| model | released | on the 4080 SUPER |
| :--- | :--- | :--- |
| `google/gemma-4-12B-it` | 2026-05-23 | lossless 14.9 GiB does not fit; **kestrel 9.0 GiB and swift 7.6 GiB run** (the engine only: vLLM has no lossy levels) |
| `google/gemma-4-12B-it-qat-w4a16-ct` (vLLM's 4-bit checkpoint of it) | 2026-06-05 | vLLM 0.30 runs it, 8.28 GiB: the comparison |
| `Qwen/Qwen3.8-27B` | 2026-08-05 | does not fit at any level (wren 14.6 GiB + KV against 13.2 GiB free) |
| `Qwen/Qwen3.8-Flash-Next` | 2026-08-24 | 180B parameters; "does not run with glyd run yet" (Condor) |
| `Qwen/Qwen3.5-9B`, Gemma 4 E4B, the rest of the Gemma 4 family | 2026-02/03 | older than the cutoff (and vLLM's on this GPU: the engine does not take Qwen3.5 lossless) |

**No model of the Qwen3 dense family (Qwen3ForCausalLM, the one family `glyd run` hands to the engine by default) is newer than 2025.** Gemma 4 12B is the one model that can be measured, at the engine's lossy levels.

## Gemma 4 12B against vLLM, one run each (spread of the 3 rounds of a load under 0.1%)

`glyd-engine serve google/gemma-4-12B-it:LEVEL --context 8192` (main 2026-11-20: Gemma 4 runs **one request at a time**, `work_gen`) and `vllm serve google/gemma-4-12B-it-qat-w4a16-ct --max-model-len 8192
--max-num-seqs 8` (vLLM 0.30.0, `VLLM_USE_FLASHINFER_SAMPLER=0` as before), the loads of the earlier records (1, 4 and 8 requests at once with short, ~1,900 and ~3,750-token prompts, 128 new tokens; a run with a
container on the GPU is thrown away and run again); `scripts/run_gemma.sh`. Weights on the GPU: kestrel 9.1 GiB (6.5 bits a weight), swift 7.7 GiB (5.5), vLLM 8.28 GiB (4 bits and the 16-bit parts). Peak GPU
memory: kestrel 10,766 MiB, swift 9,286, vLLM 15,334 (it takes 92% of the card for its cache). Each cell: kestrel / swift / vLLM.

| load | decode tok/s a user | total tok/s | first token s |
| :--- | :--- | :--- | :--- |
| short (~37 tokens), 1 at once | 66.0 / 76.5 / 74.6 | 66.0 / 76.5 / 74.6 | 0.027 / 0.027 / 0.021 |
| short (~37 tokens), 4 at once | 66.0 / 76.5 / 72.8 | 65.3 / 75.6 / 286.9 | 2.953 / 2.557 / 0.058 |
| short (~37 tokens), 8 at once | 66.0 / 76.5 / 70.4 | 65.2 / 75.5 / 544.7 | 6.856 / 5.929 / 0.095 |
| ~1,900 tokens, 1 at once | 62.7 / 72.2 / 71.6 | 62.7 / 72.2 / 71.6 | 0.535 / 0.536 / 0.496 |
| ~1,900 tokens, 4 at once | 62.7 / 72.0 / 51.8 | 52.4 / 58.7 / 146.6 | 4.372 / 3.978 / 1.452 |
| ~1,900 tokens, 8 at once | 62.7 / 72.2 / 37.3 | 51.0 / 57.1 / 177.8 | 9.486 / 8.560 / 2.747 |
| ~3,750 tokens, 1 at once | 61.4 / 70.4 / 70.6 | 61.4 / 70.4 / 70.6 | 1.117 / 1.120 / 1.018 |
| ~3,750 tokens, 4 at once | 61.4 / 70.4 / 42.2 | 43.8 / 48.2 / 106.6 | 5.886 / 5.489 / 3.005 |
| ~3,750 tokens, 8 at once | 61.4 / 70.4 / 23.7 | 41.7 / 45.7 / 111.3 | 12.256 / 11.319 / 5.002 |

- **With one user** (the load a one-at-a-time engine serves): swift decodes 3% faster than vLLM's 4-bit checkpoint (76.5 against 74.6 tokens/s, 7.7 GiB against 8.28) and kestrel, at 6.5 bits, 12% slower (66.0:
  9.1 GiB, 88% of its 75 tokens/s ceiling); the first token is **behind at both levels, 21% on a short prompt (27 against 21 ms), 7% at ~1,900 tokens, 9% to 10% at ~3,750**. By the bar's 3% neither passes
  even this one load.
- **With 4 and 8 users** the engine answers them one after another: a user's decode rate is the same, the 8th waits for seven replies (first token 5.9 to 12.3 s against vLLM's 0.09 to 5.0 s), and the tokens/s in all stays at
  one user's (41 to 76) where vLLM's grows with the users (111 to 545): 55% to 88% behind. This is the loss the owner expected; it is not a tuning gap, it is what one at a time means.

So **Gemma 4 is not on the list on the 4080 SUPER** at either level, and `glyd run google/gemma-4-12B-it` stays vLLM's by default (it does not fit lossless anyway; the lossy levels remain the engine's by asking for them:
`glyd run google/gemma-4-12B-it:swift`).

## What changed in the code and the docs

- `GPUS` is per (GPU, family): `("RTX 4080 SUPER", (8, 9), &["Qwen3"])`; `gpu_refusal(.., family)` says "a GPU the engine is measured at least as fast as vLLM on for Qwen3 models only" for another family; `glyd run`'s default
  path (`preflight` of a lossless model) is only the Qwen3 dense family's, so that is the one family consulted today, and the lossy levels and `GLYD_ENGINE=1` look at no list.
- The speed bar (`glyd-engine bench --bar`, which `scripts/gpu_bar.py` hands its arguments) takes the family (`--family`) and prints the line with it, reads the model's release date (`--released`, or the result file's own: these
  measurement files carry their model, date and family) and exits 2 for a model older than 6 months unless `--stale-ok`; docs/adding-a-gpu.md and docs/bench.md say so.
- **For the owner:** the 4080 SUPER's Qwen3 line rests on Qwen3-8B (572 days old), for there is no newer model of that family. the bar passes it only with `--stale-ok`. If the rule is to stand without
  exceptions, the line goes (`GPUS` empty again and the Qwen3 dense models back to vLLM by default on this card); it is one line in `machine.rs`.

## Files

`engine-kestrel.json`, `engine-swift.json`, `vllm-w4a16.json`: the measurement files (`python3 scripts/gpu_bar.py engine-swift.json vllm-w4a16.json "NVIDIA GeForce RTX 4080 SUPER" 8.9 --family Gemma4`: exit 1, the loads it fails);
`scripts/run_gemma.sh`, `scripts/load.py` (the earlier record's). Raw outputs and the servers' logs are not in the tree.
