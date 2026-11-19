# Heavy requests wait, they do not fail: floods, the reserves and the KV cache against stock's

vLLM 0.30.0, the dev L4 (24 GB), 2026-10-02. Ungated mirrors. "Heavy" prompts are prompts of the model's rare tokens (tokens whose embeddings were never trained: Llama-3.1-8B has 289, Mistral-7B-v0.3 902, the Qwen models none), which the cache holds in more than 12 bits, in a second reserve it keeps for them.

- No request that stock vLLM serves ends with an error. 40 prompts of 1,400 heavy tokens at once with 8 ordinary ones: all 40 served on every one of the five models, and 24 concurrent over `vllm serve` on Llama-3.1-8B: all 24 served. Where the cache's second reserve has room for two or three of them at a time the others wait, as vLLM's own requests do when KV blocks are short.
- **Known limit:** prompts made mostly of rare tokens are served more slowly when many arrive at once: the flood ran at 0.88x (Llama-3.1-8B) and 0.77x (Mistral-7B) of stock's prompt tokens a second, and at 1.00x to 1.01x on the Qwen models, which have no such token.
- A prompt that could never fit the second reserve even alone is refused when it is sent, with a 400 and why, as one longer than the model's context is: 12 of 12 on Llama-3.1-8B and Mistral-7B at reserves made too small, and over HTTP (a heavy prompt of 1,000 tokens, each of 8 concurrent ones, one of 4,000: all 400; the ordinary requests among them served, the server went on). Every API server process decides this on its own, from the model's tables in the cache directory and the size of the KV cache vLLM reports, and nothing is handed over between processes: with `--api-server-count 2`, and with the API servers given a `TMPDIR` of their own, 40 of 40 concurrent 4,000-token prompts were refused with a 400 between the processes and the 1,400-token ones served (below).
- The 8 ordinary requests sent with a flood are served and are not delayed behind the held ones: they finish in 8.1 s (Mistral-7B) and 10.5 s (Llama-3.1-8B) where stock's, queued behind the flood, take 19.0 s and 19.8 s (alone 3.8 s and 4.2 s); under `VLLM_BATCH_INVARIANT=1` their tokens and logprobs are the same, bit for bit, as without the flood (8 of 8 on every model measured; stock's own, the control: 8 of 8). Over `vllm serve` the 6 ordinary requests took 4.0 s in the flood against 0.95 s alone and stock's 9.4 s, with the same text; in a steady stream with bursts (510 requests) the ordinary requests' latency was stock's.
- Where the reserve runs out in flight (made so here: no waiting for room, a reserve that holds one such prompt and not all) the requests that need it are preempted and run again: 12 of 12 served on every model checked (27 to 29 preemptions), with `verify` on (0 values differ), with async scheduling on and off, with prefix caching, and under `VLLM_BATCH_INVARIANT=1`, where each request comes back to exactly the tokens it had when it ran alone (12 of 12 on three models; in the default mode 12 of 12 on Mistral-7B and Llama-3.1-8B and 8 of 12 on Qwen3-8B, whose prompt is one token repeated, as vLLM's own preemption does).
- The reserves are as large as the model's vocabulary sweep says: the KV cache is 1.25x to 1.30x stock's, and ordinary prompts take none of the second reserve.
- Every value read back is the value written: 0 differ in every run below (real prompts through the write kernel and back, `verify` against a bf16 shadow of each layer, the floods).

## KV cache per model

| Model | layers x KV heads | reserves: first + second, % of the KV cache | KV tokens, stock -> kv (8,704 tokens a request and a step) | kv / stock | the earlier version's (its reserves) | bits a value |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 36 x 8 | 1.64% + 0.23% | 83,008 -> 107,760 | 1.2982x | 1.2922x (2.13% + 0.23%) | 12.32 |
| Qwen3-8B | 36 x 8 | 1.59% + 0.23% | 25,232 -> 32,384 | 1.2834x | 1.2771x (2.07% + 0.23%) | 12.47 |
| Qwen2.5-7B | 28 x 4 | 1.83% + 0.29% | 77,152 -> 98,544 | 1.2773x | 1.2698x (2.38% + 0.29%) | 12.53 |
| Mistral-7B-v0.3 | 32 x 8 | 1.40% + 1.55% | 43,520 -> 55,456 | 1.2743x | 1.2688x (1.82% + 1.55%) | 12.56 |
| Llama-3.1-8B | 32 x 8 | 2.45% + 1.55% | 30,464 -> 38,160 | 1.2526x | 1.2426x (3.19% + 1.55%) | 12.77 |

## Values

| Model | real prompts through the write kernel and back: values / differ | verify: values / differ | the model's checks |
| :--- | ---: | ---: | ---: |
| Qwen3-4B | 1,980,334,080 / 0 | 6,329,327,616 / 0 | 13 of 13 |
| Qwen3-8B | 1,980,334,080 / 0 | 6,329,327,616 / 0 | 21 of 21 |
| Qwen2.5-7B | 770,129,920 / 0 | 2,461,405,184 / 0 | 13 of 13 |
| Mistral-7B-v0.3 | 1,845,166,080 / 0 | 5,982,126,080 / 0 | 21 of 21 |
| Llama-3.1-8B | 1,726,021,632 / 0 | 5,591,007,232 / 0 | 21 of 21 |

None of those real prompts needed the second reserve.

## Prompts of the heavy tokens, one at a time

| Model | prompts | second reserve taken at most | verify: values / differ | prompt logprobs bit for bit stock's | first token bit for bit stock's |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 3 | 0% | 274,931,712 / 0 | 3 of 3 | 3 of 3 |
| Qwen3-8B | 3 | 0% | 274,931,712 / 0 | 3 of 3 | 3 of 3 |
| Qwen2.5-7B | 3 | 0% | 106,917,888 / 0 | 3 of 3 | 3 of 3 |
| Mistral-7B-v0.3 | 8 | 32% | 556,793,856 / 0 | 8 of 8 | 8 of 8 |
| Llama-3.1-8B | 13 | 22% | 872,677,376 / 0 | 13 of 13 | 13 of 13 |

## 40 prompts of 1,400 heavy tokens at once with 8 ordinary ones, kv / stock

| Model | served | ended with an error or a 400 | wall seconds | prompt tokens a second | kv / stock throughput | the 8 ordinary prompts' latency alone, s (median) | in the flood, s | requests held / preempted | first reserve taken at most | second reserve taken at most |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 40 of 40 / 40 of 40 | 0 / 0 | 11.25 / 11.3 | 4,985 / 4,964 | 1.00x | 2.41 / 2.40 | 11.25 / 11.30 | 0 / 0 | 33.2% | 0% |
| Qwen3-8B | 40 of 40 / 40 of 40 | 0 / 0 | 19.97 / 20.17 | 2,810 / 2,782 | 1.01x | 4.26 / 4.25 | 19.97 / 20.17 | 0 / 0 | 56.9% | 0% |
| Qwen2.5-7B | 40 of 40 / 40 of 40 | 0 / 0 | 17.34 / 17.42 | 3,236 / 3,220 | 1.00x | 3.89 / 3.88 | 17.34 / 17.42 | 0 / 0 | 38.5% | 0% |
| Mistral-7B-v0.3 | 40 of 40 / 40 of 40 | 0 / 0 | 24.55 / 19.03 | 2,286 / 2,950 | 0.77x | 3.82 / 3.80 | 8.06 / 19.03 | 38 / 0 | 0.3% | 90% |
| Llama-3.1-8B | 40 of 40 / 40 of 40 | 0 / 0 | 22.4 / 19.76 | 2,506 / 2,839 | 0.88x | 4.21 / 4.22 | 10.52 / 19.76 | 37 / 0 | 0.4% | 90% |

(Each cell is kv / stock.)

## The ordinary prompts' bits in the flood (VLLM_BATCH_INVARIANT=1)

| Model | flood served | kv: the 8 ordinary prompts' 64 greedy tokens and logprobs the same bits as alone | stock (the control) | ordinary latency alone / in the flood, s | flood wall seconds kv / stock |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 40 of 40 | 8 of 8 | 8 of 8 | 5.10 / 22.72 | 22.72 / 23.43 |
| Mistral-7B-v0.3 | 40 of 40 | 8 of 8 | 8 of 8 | 4.58 / 9.37 | 28.58 / 22.09 |
| Llama-3.1-8B | 40 of 40 | 8 of 8 | 8 of 8 | 4.97 / 12.33 | 26.24 / 22.72 |

## The reserve runs out in flight: 12 prompts of 900 heavy tokens at once, no waiting for room, reserves that hold one of them and not all (verify on)

| Model | mode | served | ended with an error | ordinary failed | held / preempted | values compared / differ | the 24 tokens after each prompt the same as alone | wall seconds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | async scheduling on | 12 of 12 | 0 | 0 of 8 | 14 / 29 | 3,793,502,208 / 0 | 8 of 12 | 45.1 |
| Qwen3-8B | async scheduling off | 12 of 12 | 0 | 0 of 8 | 14 / 29 | 3,793,420,288 / 0 | 8 of 12 | 45.83 |
| Qwen3-8B | prefix caching, chunked prefill on | 12 of 12 | 0 | 0 of 8 | 7 / 10 | 3,245,662,208 / 0 | 12 of 12 | 18.51 |
| Qwen3-8B | VLLM_BATCH_INVARIANT=1 | 12 of 12 | 0 | 0 of 8 | 13 / 28 | 3,851,198,464 / 0 | 12 of 12 | 48.08 |
| Mistral-7B-v0.3 | async scheduling on | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 3,207,217,152 / 0 | 12 of 12 | 38.26 |
| Mistral-7B-v0.3 | async scheduling off | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 3,207,217,152 / 0 | 12 of 12 | 39.19 |
| Mistral-7B-v0.3 | prefix caching, chunked prefill on | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 3,204,071,424 / 0 | 12 of 12 | 38.86 |
| Mistral-7B-v0.3 | VLLM_BATCH_INVARIANT=1 | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 3,207,217,152 / 0 | 12 of 12 | 44.62 |
| Llama-3.1-8B | async scheduling on | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 2,871,828,480 / 0 | 12 of 12 | 40.48 |
| Llama-3.1-8B | async scheduling off | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 2,870,452,224 / 0 | 12 of 12 | 40.94 |
| Llama-3.1-8B | prefix caching, chunked prefill on | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 2,870,779,904 / 0 | 12 of 12 | 40.42 |
| Llama-3.1-8B | VLLM_BATCH_INVARIANT=1 | 12 of 12 | 0 | 0 of 8 | 12 / 27 | 2,871,828,480 / 0 | 12 of 12 | 45.79 |

## Reserves forced too small for any of them to fit, 12 prompts of 900 heavy tokens at once (verify on)

| Model |  | served | refused (400) | ended with an error | ordinary failed | preempted | values compared / differ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | with the API server's check (models that have heavy tokens) | 0 of 12 | 0 (400) | 12 | 0 of 8 | 57 | 1,584,777,216 / 0 |
| Qwen3-8B | with the API server's check off (the scheduler ends them) | 0 of 12 | 0 (400) | 12 | 0 of 8 | 57 | 1,584,777,216 / 0 |
| Mistral-7B-v0.3 | with the API server's check (models that have heavy tokens) | 0 of 12 | 12 (400) | 0 | 0 of 8 | 0 | 871,366,656 / 0 |
| Mistral-7B-v0.3 | with the API server's check off (the scheduler ends them) | 0 of 12 | 0 (400) | 12 | 0 of 8 | 0 | 871,366,656 / 0 |
| Llama-3.1-8B | with the API server's check (models that have heavy tokens) | 0 of 12 | 12 (400) | 0 | 0 of 8 | 0 | 825,950,208 / 0 |
| Llama-3.1-8B | with the API server's check off (the scheduler ends them) | 0 of 12 | 0 (400) | 12 | 0 of 8 | 0 | 825,950,208 / 0 |

The 400's text: "glyd kv: this prompt's tokens need more of the lossless KV cache's reserve for tokens it cannot store at 12 bits than the cache holds (N against M): send a shorter prompt, or serve with kv off". Where the API server's check is off (or the model has no heavy token, as Qwen3-8B), the scheduler ends the request with an error finish and: "glyd kv: the cache cannot hold this request, even alone: it needs more of its reserve for tokens that do not store at 12 bits than the cache has: send a shorter prompt, or serve with kv off".

## `vllm serve` (defaults, an empty tables cache, `--max-model-len 4096 --max-num-seqs 32`), kv against stock

24 concurrent prompts of 1,400 tokens of the model's heavy tokens (Llama-3.1-8B: 85 of its 289; Qwen3-8B, which has none: its heaviest token repeated, behind a token of each prompt's own), with 6 ordinary requests (16 tokens) sent 0.3 to 0.8 s after the flood started (`scripts/serve_flood.py`).

| Model | | start to serving, s (kv: the calibration included) | the 24 heavy prompts | the flood's seconds | heavy prompt tokens a second | the ordinary requests (HTTP) | their latency in the flood, median, s | their text the same as alone | a prompt of 700 heavy tokens after it | a prompt of 4,000 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | kv lossless | 258 | 24 of 24 (HTTP 200) | 11.43 | 2,940 | 200 | 4.024 (0.953 alone) | yes | 200 | 200 (1.29 s) |
| Llama-3.1-8B | stock | 54 | 24 of 24 (HTTP 200) | 9.99 | 3,363 | 200 | 9.428 (0.952 alone) | yes | 200 | 200 (1.28 s) |
| Qwen3-8B | kv lossless | 159 | 24 of 24 (HTTP 200) | 10.17 | 3,304 | 200 | 9.603 (0.965 alone) | yes | 200 | 200 (1.36 s) |
| Qwen3-8B | stock | 66 | 24 of 24 (HTTP 200) | 10.86 | 3,095 | 200 | 10.291 (0.964 alone) | NO | 200 | 200 (1.36 s) |

The same server on Llama-3.1-8B with the second reserve made too small for any of the heavy prompts: a heavy prompt of 1,000 tokens, each of 8 concurrent ones of 1,400, a prompt of 700 after them and one of 4,000 all got HTTP 400 with the text above (the second reserve's room 300: the prompt of 1,000 tokens needing 819, the one of 700 572, the one of 4,000 3,250); the 6 ordinary requests sent among them
were served (median 1.073 s) with the same text; the server was alive after.

### Several API server processes (`serve-apis/`; `scripts/serve_apis.py`)

Llama-3.1-8B over `vllm serve` at its defaults with the tables made, the second reserve made small enough that a 4,000-token heavy prompt could never fit (0.55% of the KV cache: the API server's figure for it 1,457, the engine's 1,449), `--max-num-seqs 32`, `--max-model-len 4096`. One heavy prompt of 700 tokens and one of 1,400, an ordinary request, a heavy prompt of 4,000, then 40 concurrent heavy prompts of 4,000 tokens, then 12 concurrent of 1,400.
Nothing is handed over between the processes (no file, no environment variable): each API server process reads the model's tables from the cache directory and the KV cache's size from vLLM.

| API server processes | the API servers' TMPDIR | start to serving, s | 700 and 1,400 tokens | the 4,000-token prompt | 40 concurrent of 4,000 | 400s by process | 12 concurrent of 1,400 | server alive after | files in the temporary directories |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | the engine's | 36 | 200, 200 | 400 (3,250 against 1,457) | 40 of 40 refused (400) | one process: 41 | 12 of 12 served (200) | yes | none |
| 2 | the engine's | 39 | 200, 200 | 400 (3,250 against 1,457) | 40 of 40 refused (400) | process 0: 21, process 1: 20 | 12 of 12 served (200) | yes | none |
| 2 | a directory of its own | 39 | 200, 200 | 400 (3,250 against 1,457) | 40 of 40 refused (400) | process 0: 32, process 1: 9 | 12 of 12 served (200) | yes | none |

(The 400s by process count the 40 and the single 4,000-token prompt: 41.)

### A steady stream with bursts: Llama-3.1-8B over `vllm serve`, 3 ordinary requests a second (32 tokens) for 150 s, a burst of 12 prompts of 1,400 heavy tokens every 30 s (`scripts/serve_mixed.py`)

|  | ordinary requests | ordinary latency out of a burst, s (median / p95 / max) | during a burst | heavy requests | heavy latency, s (median / max) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| kv lossless | 450 (200: 450) | 2.074 / 4.674 / 6.758 | 4.356 / 6.565 / 6.83 | 60 (200: 60) | 3.659 / 6.406 |
| stock | 450 (200: 450) | 2.068 / 6.605 / 6.851 | 4.456 / 6.775 / 6.943 | 60 (200: 60) | 4.908 / 5.428 |

## Kernels, sanitizer, unit tests

- Kernel suite (`check_vllm.py --kv --kernels-only`): 64 checks, all passed.
- compute-sanitizer over the kernels' quick checks (`sanitizer/`): memcheck: 0 errors; initcheck: 0 errors; racecheck: 0 hazards displayed (0 errors, 0 warnings); synccheck: 0 errors.
- `bindings/python/test_vllm.py`: 19 tests pass (the scheduler's waiting, preempting and ending of requests against a fake of vLLM's, the prediction and the API servers' check, the check in a process that reads its own tables); the Rust crate's 15 tests pass.
- `kv_stress.py` (Qwen3-8B, quick): capture: 25,683,296,256 values written through the kernel and read back: 0 differ, none lost. A cache filled entirely with each of the 12 heaviest repeated tokens in turn, with the first reserve at 1.0% of the cache (the plugin sizes this model's at 1.59%): where it ran out (6 of the 12) the second reserve took the rest, none lost, 0 differ. Engine: with `verify`, 12,986,523,648 and 13,988,044,800 values compared, none differ; with prefix caching and chunked prefill on: no engine stop.

## Llama-3.1-8B: the full run (34 of 35 checks pass)

The model's checks, the floods, the compiled and FULL-graph runs, batch invariance, prefix caching, chunked prefill and `exact`, against vLLM's own.

| | check |
| :--- | :--- |
| PASS | calibration (32,768 tokens, 96 s; the vocabulary's sweep 56 s): 289 of the 128,256 tokens need the second reserve (21 that the tokenizer makes from text, 268 only by token ids) |
| PASS | vLLM's own keys and values of 14 real prompts (26,337 tokens, 1,726,021,632 values over 32 layers) through the write kernel and back: 0 differ; none of them needed the second reserve |
| PASS | `verify`, 14 prompts of 6 to 8,192 tokens and 512 generated each: 5,591,007,232 values, none differ |
| PASS | KV cache 38,160 tokens against stock's 30,464 at the standard flags (1.2526x, 12.77 bits a value; eager, `--max-model-len 8704`, 0.9); 27,232 against 21,872 (1.2451x) at `--max-model-len 4096`, 0.85 |
| PASS | prompt logprobs of a 1,543-token prompt bit for bit stock's; the 8 prompts' first generated token's logprob bit for bit stock's, their 64 greedy tokens the same from the start for 64, 41, 64, 19, 64, 57, 64 and 64 tokens |
| PASS | vLLM's default mode (compiled, piecewise graphs and FULL decode graphs, 1,718 captured): the first generated tokens stock's, their logprobs within 0.000 of stock's; the FULL decode graphs alone (35 captured) replay eager's tokens and logprobs bit for bit (8 of 8) |
| PASS | `VLLM_BATCH_INVARIANT=1`, eager and FULL graphs: each of 8 prompts' 64 tokens and logprobs the same bits alone and in a batch of 8 (8 of 8 each; vLLM's own the control: 8 of 8); compiled in inductor's deterministic mode, both: the first token's logprob bit for bit (8 of 8) and a 1,543-token prompt's logprobs |
| PASS | chunked prefill (512 and 2,048 tokens, 14 prompts of up to 4,096): prompt logprobs bit for bit stock's (14 of 14) and the first token's logprob (14 of 14); prefix hits with chunked prefill 14 of 14; four requests sharing 1,600 tokens 4 of 4; eight sharing a prefix 8 of 8; requests with prefixes of their own bit for bit; metadata built under torch's synchronization errors: none raised |
| FAIL | prefix hits (14 prompts of up to 3,564 tokens, one request at a time): the first token's logprob bit for bit stock's in 13 of 14. The prompt that differs reads a cached prefix that holds 4 tokens an earlier request generated: their keys and values were made by the decode kernel, which sums in another order than FlashAttention (the same prompt and scenario on Qwen3-8B, compared layer by layer in [`l4-vllm-kv-prefix-2026-10-02`](../l4-vllm-kv-prefix-2026-10-02): the layers of every prefill step that reads a cached prefix written by prefill steps are bit for bit stock's, the differing prompt parts from layer 1's attention output, vLLM's own FlashAttention with its decode steps' keys in one split moves it the same way, and with `exact` all 14 prompts are bit for bit); in a fresh engine all 32 layers of its prefill are bit for bit stock's; `verify` over the 14 prompts: no value differs |
| PASS | `exact`: 8 prompts' 64 tokens and logprobs bit for bit stock's (8 of 8) |
| PASS | 13 prompts of the heavy tokens: every write compared with a bf16 shadow (872,677,376 values) none differ; prompt logprobs bit for bit stock's (13 of 13), the first token (13 of 13) |
| PASS | the floods of this record, in the same run (tables above): 40 of 40 served, with and without prefix caching, under `VLLM_BATCH_INVARIANT=1`; the reserve run out in flight, async scheduling on and off and with prefix caching: 12 of 12 served, 0 differ; reserves too small: 12 of 12 refused with a 400, or ended by the scheduler |

## Files

`models/<model>/` (each model's result files: the checks, the floods, the heavy-token runs; JSON, stock beside kv), `serve-*` (each `vllm serve` run's result file), `serve-apis/` (the multi-process runs' result files and the launcher that gives the API servers their own TMPDIR), `scripts/` (`serve_flood.py`, `serve_mixed.py`, `serve_apis.py`), `sanitizer/`. The checks' tool is [`gpu/vllm/check_vllm.py`](../../../gpu/vllm/check_vllm.py). Result files are the harness's own JSON with its internal counters removed; logs that carry the plugin's internal diagnostics are not part of this record.
