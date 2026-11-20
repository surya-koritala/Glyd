# The engine against vLLM on an RTX 4080 SUPER, Qwen3-8B (2026-11-20)

**The engine does not pass the speed bar on this GPU, so the RTX 4080 SUPER is not in `GPUS`.** It matches vLLM's decode speed at every load (0.99x to 1.04x per user) and
beats it from 4,000-token prompts on, with less memory; it fails on time to first token, where it is slower than vLLM at 4 and 8 short prompts at once (1.7x and 2.4x:
80 and 162 ms against 48 and 68 ms), at 2,000-token prompts (1.04x to 1.08x: 0.37 s against 0.357 s one user, 1.72 s against 1.58 s eight users), and on total tokens/s
with 8 short prompts at once (431.5 against 452.1, 4.5% behind).

Machine: glyd-pc, RTX 4080 SUPER 16 GB (736 GB/s, 3105 MHz max, 320 W), driver 595.91.07 (CUDA 13.2), 60 GB RAM, Ubuntu 26.04, nothing else on the GPU (the CI container's
presence was looked for every second of every run; no run had one). Model: `Qwen/Qwen3-8B` (bf16, 16.4 GB, which does not fit this card), lossless in both.

| | engine | vLLM |
| :--- | :--- | :--- |
| program | `glyd-engine serve` 0.32.1 as `gpu.yml` builds it for sm_89 (main 444e2dcf), `--context 8192 --users 8` | `vllm serve --quantization glyd --max-model-len 8192 --max-num-seqs 8`: vLLM 0.30.0 and glyd-gpu 0.32.1 (the pin of `glyd-gpu-py`), the path `glyd serve` starts |
| weights on the GPU | 10.3 GiB (mma layout, 12.4 GB packed) | 10.38 GiB (`glyd@1`) |
| ceiling, one user | 736 GB/s / 11.06 GB = 66.5 tokens/s | 736 GB/s / 11.15 GB = 66.0 tokens/s |
| peak GPU memory in the load | **14,796 MiB** | 15,188 MiB (392 MiB, 2.6% more) |

The ceiling is the weights read once for each token; the KV cache's reads (which grow with the prompt) are not in it, so the long-prompt rows are below it by that much.
Each row is a load: N requests sent at once, each a different prompt from its first token on (no shared prefix), streamed, 256 tokens at most, temperature 0, thinking off.
Per request, time to first token is from sending it, and the decode rate is the tokens after the first over the time from the first to the last. "total" is all the requests'
tokens after their first over the time from the earliest first token to the last token. Three rounds a row; the table is the median of the requests and rounds of two runs
of each program (`engine1`, `engine2`, `vllm1`, `vllm2`, alternated: vllm1, engine2, vllm2 after engine1), which agree to 0.1% in decode rate and to 1.1% in first token time, except the engine's short prompts at 4 and 8 at once (9.5% and 5.5%: 0.084 and 0.076 s, 0.158 and 0.166 s).

| load | decode tok/s a user: engine (% of ceiling) | vLLM (%) | engine / vLLM | total tok/s: engine | vLLM | first token s: engine | vLLM | engine / vLLM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short (~35 tokens), 1 at once | 59.9 (90%) | 60.0 (91%) | 1.00x | 59.9 | 60.0 | 0.021 | 0.024 | 0.88x |
| short (~35 tokens), 4 at once | 58.3 (88%) | 58.7 (89%) | 0.99x | 228.9 | 233.6 | 0.080 | 0.048 | 1.67x |
| short (~35 tokens), 8 at once | 56.3 (85%) | 57.0 (86%) | 0.99x | 431.5 | 452.1 | 0.162 | 0.068 | 2.38x |
| ~2000-token, 1 at once | 58.7 (88%) | 58.7 (89%) | 1.00x | 58.7 | 58.7 | 0.370 | 0.357 | 1.04x |
| ~2000-token, 4 at once | 48.8 (73%) | 48.6 (74%) | 1.00x | 174.8 | 176.3 | 0.949 | 0.881 | 1.08x |
| ~2000-token, 8 at once | 39.9 (60%) | 39.5 (60%) | 1.01x | 261.2 | 263.6 | 1.718 | 1.584 | 1.08x |
| ~4000-token, 1 at once | 57.7 (87%) | 56.8 (86%) | 1.02x | 57.7 | 56.8 | 0.696 | 0.734 | 0.95x |
| ~4000-token, 4 at once | 42.4 (64%) | 41.4 (63%) | 1.02x | 142.9 | 139.0 | 1.761 | 1.825 | 0.97x |
| ~4000-token, 8 at once | 32.6 (49%) | 31.4 (47%) | 1.04x | 136.6 | 130.5 | 3.188 | 3.300 | 0.97x |

Decode: the engine is at 85% to 90% of the ceiling with short prompts at 1 to 8 users and with one 2,000-token prompt, vLLM at 86% to 91%; the one-user figure of the earlier record
(59.7 tokens/s, 90%, `rtx4080s-qwen3-8b-engine-2026-11-20`) is this run's 59.9. At 4,000-token prompts the engine decodes 2% to 4% faster than vLLM.

## Where it fails the bar, by how much

(`scripts/gpu_bar.py`, which allows 3% for the spread of two runs; both pairs of runs fail in the same six places.)

| load | what | engine | vLLM | behind |
| :--- | :--- | ---: | ---: | ---: |
| short prompts, 4 at once | first token | 0.080 s | 0.048 s | 67% slower |
| short prompts, 8 at once | first token | 0.162 s | 0.068 s | 138% slower |
| short prompts, 8 at once | total tokens/s | 431.5 | 452.1 | 4.5% |
| 2,000 tokens, 1 at once | first token | 0.370 s | 0.357 s | 4% |
| 2,000 tokens, 4 at once | first token | 0.949 s | 0.881 s | 8% |
| 2,000 tokens, 8 at once | first token | 1.718 s | 1.584 s | 8% |

The engine's first-token time for short prompts grows with the number of requests (21, 80 and 162 ms for 1, 4 and 8: about 20 ms each, one prompt after another),
vLLM's does not (24, 48 and 68 ms: it prefills them together). That, and a prompt kernel 4% to 8% slower than vLLM's at 2,000 tokens (this machine has no cuBLAS, so the
engine runs its own products: its 4,000-token prompts are 3% to 5% faster), is what stands between the engine and the list on this GPU. No change to `GPUS`.

## Things to know

- vLLM 0.30's FlashInfer sampler compiles with nvcc the first time it samples, and glyd-pc has none: the vLLM runs use `VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM's own sampler). It
  is, if anything, slower per step than FlashInfer's, so the flag favours the engine in these tables, not vLLM.
- Both programs fill the card by design (the KV cache takes what the weights leave): the memory row is what each reserves, not what the weights need. vLLM's KV cache held
  24,528 tokens (3.37 GiB); the engine's holds 8 users of 8,192.
- One model, one GPU, two runs of each program. 8 users with 4,000-token prompts is more than either holds comfortably at once (8 x 4,300 tokens): both queue some of them, and
  the rows are as the servers answered them (no request failed).
- The prompts are random words (about 1.09 tokens a word) with an essay to write; the token counts in the rows are the servers' own `prompt_tokens`.

## Files

`scripts/load.py` (the load), `scripts/run.sh` (one program, one load, the GPU's lock and the CI check), `scripts/until_clean.sh`, `scripts/pair.sh` (the order of the runs),
`scripts/summarize.py` (the table above: `python3 summarize.py 10.3 10.38 engine1.json engine2.json -- vllm1.json vllm2.json`). The raw outputs (`load.json`, the servers' logs)
are not in the tree; the program that applies the bar to them is `scripts/gpu_bar.py` in the repository, and docs/adding-a-gpu.md says how a GPU gets on the list.

```bash
# on the GPU machine (the engine built for it, and vLLM with glyd-gpu in an environment):
bash scripts/pair.sh out        # needs ~/glyd-scratch/gpu.lock's directory, BIN= and VENV= (see run.sh)
python3 scripts/summarize.py 10.3 10.38 out/engine1/load.json out/engine2/load.json -- out/vllm1/load.json out/vllm2/load.json
python3 ../../../scripts/gpu_bar.py out/engine1/load.json out/vllm1/load.json "NVIDIA GeForce RTX 4080 SUPER" 8.9
```
