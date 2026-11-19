# glyd-engine, a first engine: Qwen3 on an L4 from Glyd-packed weights (2026-10-03)

Qwen3-8B (and 0.6B, 1.7B, 4B) on an NVIDIA L4 (22 GiB, Ada, compute capability 8.9; AWS g6.4xlarge, driver 595.91), `glyd-engine` (Rust, branch `engine-mvp`, binary built from a8d97a88) against vLLM 0.30.0 (torch 2.13, CUDA 13.0) in its default mode (torch.compile, FULL decode graphs): `glyd@1` is Glyd's weights with vLLM's own cache, `kv+glyd` Glyd's weights with the lossless KV cache, `bf16` the checkpoint as it is.
The engine holds every matrix of the model compressed in GPU memory (the embedding and the LM head too), keeps its KV cache lossless (12 bits a value, read back bit for bit) and decodes greedily for one user. It is a spike: one user at a time, text in and text out, no server, Qwen3 dense only.

## What to know

- **Tokens, against vLLM+Glyd (`glyd@1`), 8 prompts of up to 31 tokens, 64 greedy tokens each**: the first token is the same for 8 of 8, 6 of 8 prompts are the same in all 64 tokens, and 461 of the 512 tokens are equal from the start of their prompts up to the first difference (455 with vLLM run one prompt at a time; the logprob of those tokens differs by 0.0103 on average, 0.13 at most). The control, vLLM against itself with only its batching or its step size changed: 5 of 8 prompts the same and 482 of 512 tokens (a batch against one at a time), 6 of 8 and 465 (in a batch, 1,024 tokens a step against 8,192), 8 of 8 and 512 (one at a time, the same). Every difference is at a near tie: the two runs' logprobs of their own choices where they part are within 0.12 of each other, the long prompts' too. Teacher-forced over 1,535 positions of a text, the largest-logit token is the same at 98.4% of them (vLLM against itself with another step size: 98.6 to 99.0%). One position of the text, 1500, a newline before an indentation, amplifies tiny differences: vLLM's own logprob of the next token there is -0.97 at 8,192 tokens a step, -4.02 at 1,024, -3.29 at 512 and -2.58 at 256 (bf16: -1.31, -0.25 at 512, -6.25 at 256); the engine's is -6.75. The engine's outputs are the same bits in every run and between its two builds.
- **Memory**: the weights take 10.37 GiB, 68.0% of bf16's 15.26 GiB (-32.0%): every matrix at 10.8 to 10.9 bits a weight, the embedding (10.86) and the LM head (10.77) included. vLLM+Glyd holds 11.38 GiB. At 0.9 of the GPU the KV cache holds **84,080 tokens** (110.5 KiB a token), against 50,368 with vLLM's cache in `glyd@1` and 65,568 with the lossless one in `kv+glyd` at vLLM's default 8,192 tokens a step (1.67x and 1.28x), 55,168 and 72,176 at 1,024 a step (1.52x and 1.16x): 20 requests of 4,096 tokens against vLLM+Glyd's 12. The cache's check after every run, that no value was lost, passes in every run.
- **Speed, one user, 1,024 tokens in and 256 out**: the engine decodes at 45.7 to 47.1 ms a token (21.2 to 21.9 tokens a second; medians of the runs), vLLM+Glyd at 47.0 to 47.6 ms (21.0 to 21.3), vLLM bf16 at 60.4 ms (16.6): **1.018x the tokens a second of vLLM+Glyd** (1.009x, 1.027x and 1.018x in three rounds that alternate the two in one hold of the box) and 1.28x to 1.32x of bf16's. The L4 holds its 72 W cap while it decodes Glyd's weights (the SM clock 1,215 to 1,305 MHz of 2,040 at 64 to 69 C in `engine/clock.csv`), so a token's time follows the GPU's state, by about 2% between a cool run and a hot one, for both programs: in the round whose temperature samples read 64 to 70 C the engine takes 45.7 ms and vLLM+Glyd 47.0, in the first round (80 to 81 C) 47.1 and 47.5 ms. 95% of a step is the weights' products (the library's kernels, the same ones vLLM+Glyd runs), which read the weights at 232 GB/s of the 259 GB/s this L4 streams.
- **Start and size**: from a shell to the first token in 3.1 s with the weights in the operating system's file cache (the weights 2.3 s of it, 4.8 GB/s); vLLM+Glyd builds its `LLM()` in 51 s warm. The program is 5.5 MB with the kernels for Ampere, Ada and Hopper (5.1 MB with those for the L4 alone, the build the runs used) and the library 28.5 MB, 34 MB together (no CUDA runtime library: the driver is all they open). A build of the branch head from scratch with the kernels for four architectures, the model packed again, gives the same tokens, logprobs and scores (`engine/release-build.txt`).
- **Long prompts are slow**: prompt attention is the unfinished part of the spike and grows with the square of the prompt. A 1,024-token prompt takes 435 to 458 ms (vLLM+Glyd 404 to 435), 3,500 tokens 2.04 s (vLLM+Glyd 1.14 s) and 7,500 tokens 6.2 s. Decoding after them takes 47.4 ms a token after 3,500 tokens (vLLM+Glyd 48.6) and 48.9 ms after 7,500. A kernel for the prompt's attention is the first item after the spike.
- **Other sizes**: Qwen3-0.6B: 0.75 GiB (67.6% of bf16), 5.16 ms a token (194 tokens a second); 1.7B: 2.16 GiB (67.3%), 11.98 ms (83.5); 4B: 5.08 GiB (67.8%), 26.01 ms (38.5), each with a 1,024-token prompt. Against vLLM+Glyd of the same size the first token is the same for 8, 7 and 8 of 8 prompts (vLLM against itself, batch against one at a time: 8, 7 and 8) and 4, 5 and 6 of 8 prompts are the same in all 64 tokens (vLLM against itself: 5, 5 and 6); of the long prompts 1, 0 and 3 of 3. Every difference at every size is at a near tie, the two runs' logprobs of their own choices within 0.16 of each other. Teacher-forced top-1 agreement: 97.8%, 98.8% and 98.6%.
- **The engine's own kernels give vLLM's bits**: its norms, the residual add, the SiLU product and the query/key norm with rotary give 100.000%, 100.000%, 100.000% and 99.999% of the bits of vLLM's compiled (default mode) operations on 4,096 random rows (`kcheck`), and the rotary tables 100.000% of torch's on the CPU.
- **Not in the spike**: batching and a server, a kernel for long prompts, Llama, Mistral and Gemma, GPUs other than the L4, the KV tables made by the engine itself (they come from the plugin's calibration, `python -m glyd_gpu kvcal`).

## Tokens: where the engine and vLLM part (Qwen3-8B, tokens equal from the start of the prompt, of 64)

| prompt | engine, vLLM in a batch | engine, vLLM one at a time | vLLM in a batch, one at a time | vLLM at 1,024 tokens a step against 8,192, in a batch | vLLM at 1,024 against 8,192, one at a time |
| :--- | ---: | ---: | ---: | ---: | ---: |
| The history of data compression began | 20 | 20 | 54 | 30 | 64 |
| def fibonacci(n): ... | 64 | 64 | 64 | 64 | 64 |
| Q: A train leaves at 3:40 pm ... | 64 | 64 | 64 | 64 | 64 |
| The capital of Australia is | 64 | 64 | 64 | 64 | 64 |
| Translate to French: ... | 64 | 64 | 64 | 64 | 64 |
| In quantum mechanics, ... | 64 | 64 | 64 | 64 | 64 |
| SELECT name, COUNT(*) FROM orders ... | 57 | 64 | 57 | 64 | 64 |
| Once upon a time, ... | 64 | 51 | 51 | 51 | 64 |
| **tokens of 512** | **461** | **455** | **482** | **465** | **512** |
| **prompts the same in all 64** | **6** | **6** | **5** | **6** | **8** |

The long prompts (the first 1,030, 2,100 and 3,500 tokens of a text, one at a time, 64 tokens each):

| prompt tokens | engine, vLLM at 8,192 tokens a step | engine, vLLM at 1,024 | vLLM at 1,024, at 8,192 |
| :--- | ---: | ---: | ---: |
| 1,030 | 7 | 7 | 64 |
| 2,100 | 64 | 64 | 64 |
| 3,500 | 38 | 50 | 38 |

Where they part, the two runs' logprobs of their own choices (a near tie where they are close): prompt 0 (batch) token 20: -0.950 and -0.884; prompt 6 (batch) token 57: -0.633 and -0.694; prompt 0 (single) token 20: -0.950 and -0.900; prompt 7 (single) token 51: -0.576 and -0.693; 1,030-token prompt token 7: -2.626 and -2.612; 3,500-token prompt token 38: -2.079 and -1.971.

## Memory

| | weights | KV cache tokens at 0.9 of the GPU, 8,192 a step | 1,024 a step |
| :--- | ---: | ---: | ---: |
| vLLM, bf16 | 15.27 GiB | 22,928 | 27,504 |
| vLLM + Glyd (`glyd@1`) | 11.38 GiB | 50,368 | 55,168 |
| vLLM + Glyd + the lossless KV cache (`kv+glyd`) | 11.51 GiB | 65,568 | 72,176 |
| **glyd-engine** | **10.374 GiB** (68.0% of bf16's 15.256) | | **84,080** (110.5 KiB a token) |

The engine's bits a weight: qkv 10.80, o 10.76, gate_up 10.93, down 10.85, embed 10.86, lm_head 10.77. The cache's 8.857 GiB is what is left of 0.9 of the GPU's 22.04 GiB after the weights, 356 MiB of work buffers and the driver's own use; vLLM's lines are in `vllm/cache/`.

## Speed

| one user, 1,024 tokens in, 256 out | decode ms a token | tokens a second | prefill ms |
| :--- | ---: | ---: | ---: |
| **glyd-engine** (the five runs of the final proofs, GPU hot) | **46.80** (46.51 to 46.94) | **21.37** | 452 |
| glyd-engine, eager (no graph) | 46.94 | 21.30 | 456 |
| glyd-engine, runs that start cool (six runs, 14:50 and 15:57 UTC) | 45.71 (45.61 to 45.82) | 21.88 | 437 |
| vLLM + Glyd (`glyd@1`), earlier | 47.30 (47.15 to 47.41) | 21.14 | 420 |
| vLLM + Glyd (`glyd@1`), after the engine's runs | 47.60 (47.32 to 48.00) | 21.01 | 432 |
| vLLM + Glyd + lossless KV cache (`kv+glyd`), earlier | 47.01 (46.82 to 47.20) | 21.27 | 414 |
| vLLM + Glyd + lossless KV cache (`kv+glyd`), after the engine's runs | 47.46 (47.07 to 47.69) | 21.07 | 429 |
| vLLM, bf16 | 60.43 (60.41 to 60.45) | 16.55 | 335 |

| round | vLLM + Glyd (`glyd@1`), three runs | glyd-engine, three runs | engine / vLLM tokens a second |
| :--- | ---: | ---: | ---: |
| 1 | 47.48 ms (47.46 to 47.61) | 47.07 ms (47.06 to 47.34) | 1.009x |
| 2 | 46.98 ms (46.94 to 46.98) | 45.72 ms (45.66 to 45.80) | 1.027x |
| 3 | 47.55 ms (47.52 to 47.81) | 46.69 ms (46.49 to 46.85) | 1.018x |

Median of the rounds: 1.018x.

| shape | prefill ms | decode ms a token |
| :--- | ---: | ---: |
| glyd-engine, 128 tokens in, 256 out | 89 | 46.39 |
| glyd-engine, 3,500 in, 128 out | 2,043 | 47.37 |
| vLLM + Glyd, 3,500 in, 128 out | 1,136 | 48.59 |
| vLLM + Glyd + lossless KV cache, 3,500 in, 128 out | 1,190 | 48.20 |
| glyd-engine, 7,500 in, 128 out (8,192 positions) | 6,181 | 48.88 |

One decode step's GPU time (eager, mean of 20 steps), by kind of work:

| work | ms | share |
| :--- | ---: | ---: |
| gate and up product | 20.79 | 44.4% |
| down product | 10.72 | 22.9% |
| qkv product | 5.49 | 11.7% |
| o product | 3.90 | 8.3% |
| LM head product | 3.49 | 7.4% |
| attention over the cache | 1.01 | 2.2% |
| norm (input, with the residual) | 0.33 | 0.7% |
| norm (after attention, with the residual) | 0.33 | 0.7% |
| KV write | 0.32 | 0.7% |
| SiLU product | 0.21 | 0.4% |
| query/key norm and rotary | 0.20 | 0.4% |
| total | 46.83 | 100.0% |


## Files

`engine/`: `proofs.out` (the final proofs' log: memory, start, the benches, the step profile, the token and teacher-forced runs, a text, the kernel check), `proofs-cool.out` (the same script's first run, GPU cool, 14:50 UTC, built from 929c19a1's tree), `eng-8b.json` (the engine's 64 tokens and logprobs for the 8 prompts and the three long ones), `eng-8b-score.json` (teacher-forced), `pair.out` (the engine and vLLM alternating), `ab.out` (that first build against the final one, alternating), `release-build.txt` (the build of the head, checked), `clock.csv` (the GPU's clock, power and temperature once a second while the engine decodes), `sizes.txt`. The engine's lines about the cache's storage are shortened to "KV cache: nothing lost" in these logs.
`vllm/`: `ref-8b-glyd1.json`, `ref-8b-kvglyd.json` and `ref-8b-bf16.json` (vLLM's tokens and logprobs for the same prompts, and its teacher-forced scores), `ref-8b-glyd1-b1024.json` (`glyd@1` with 1,024 tokens a step), `speed-8b-*.json`, `bracket-speed-8b-*.json` (the speeds again after the engine's runs), `pair-vllm-*.json`, `speed3500-*.json`, `score-*.json` (teacher-forced at other step sizes), `cache/` (vLLM's lines on its KV cache).
`small/`: the same for Qwen3-0.6B, 1.7B and 4B. `ctx8k/`: the 7,500-token prompt with tables for 8,192 positions. `machine.txt`. `numbers.txt`: every number above, computed from these files by [`glyd-engine/tools/record.py`](../../../glyd-engine/tools/record.py); the other tools (`ref_vllm.py`, `ref_speed.py`, `ref_cache.py`, `ref_score.py`, `compare.py`, `kcheck.py`) are beside it.

