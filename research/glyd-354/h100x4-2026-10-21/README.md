# 4x H100 SXM5, 2026-10-21 (GLY-354, GLY-351, GLY-318)

Lambda gpu_4x_h100_sxm5, us-south-3, 9:48-10:59 AM Eastern, about $19. `job.sh` on the host (downloads into the Hub cache, the CUDA 13
container), `inner.sh` in the container, `retest.sh` and `retest2.sh` with the fixed builds. The engine and GPU library were private main
at #207/#208 (`out/` has every output).

## DeepSeek-V4-Flash (FP8/FP4 as released)

- `glyd run deepseek-ai/DeepSeek-V4-Flash`: placed itself over 4 GPUs (145.3 GiB), ready in 3m15s, "The capital of France is Paris."
- Decode, 256 tokens after a 6-token prompt: 13.63 ms a token on 2 GPUs (73.4 tok/s), 13.95 on 4 (71.7).
- GLY-351: the 517-token score twice on 2 GPUs and once on 4: the same per-token file (md5 df50d614...), ppl 2.228. Coded FP8 packs
  (GLYD_F8H_PLAIN=0): 2.226, from their one-token kernel's sum order. No run-to-run or split difference.
- `serve`: chat and stream OK. Tools failed on a 256-row prompt chunk (fixed in #210, retested: a 301-token prompt runs, `tool_choice`
  required gives `get_weather {"city": "Paris"}`). With plain `tools` the model writes the call as JSON text, not DSML (chat and thinking
  mode). Our prompt is the reference encoder's for an empty system message. Open.

## GLM-5.3-Flash (FP8)

- `glyd run`: the FP8 matrices held coded to fit (#208), 266.0 GiB over 4 GPUs, ready in 2m29s, the right answer.
- `serve`: chat OK. Tools failed on `tojson(ensure_ascii=False)` (fixed in #212). Retested: the model calls in its own
  `<tool_call>NAME<arg_key>..` format, read since #213.

## Nemotron 3.5 Lightning (bf16, penguin)

- `check --layout mma`, old engine against the dynamic one: identical tokens and logits (3 prompts and one of 1,501 tokens). Decode
  5.07 ms a token on one H100.
- `run` and `serve` OK; plain tools a correct call; `tool_choice` required named the call `functions.get_weather` (fixed in #211).

## Nemotron 3 Super (hummingbird)

- `run`, the dynamic engine against the old one (GLYD_NH_DYNAMIC=0): the same answer with every expert on the GPU (45.3 GiB) and with
  `--expert-cache 8` (104 of 512 experts a layer on the GPU, misses on 102 CPU threads, 14.1 GiB).
- `run` prints the reasoning into the answer with no separator.

## Notes

- The host's `hf download` of glyd/nemotron-3-hummingbird failed once on a brotli mismatch (`process() takes no keyword arguments`);
  `pip install -U brotli brotlicffi` and a second download fixed it.
