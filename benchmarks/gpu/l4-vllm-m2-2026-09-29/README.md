# vLLM with `--quantization glyd`: M2 on an L4 (2026-09-29)

The plugin (`bindings/python/glyd/gpu/vllm_plugin.py`) against vLLM's own bf16 on one GPU: its checks on three Qwen3
models, and `vllm bench serve` on Qwen3-8B.

## Setup

- **Machine:** AWS g6.4xlarge: NVIDIA L4 (24 GB, 72 W cap), 16 vCPUs.
- **Software:** vLLM 0.30.0 with torch 2.13.0+cu130 and transformers 5.17.0, in a venv of its own. vLLM's defaults
  throughout: torch.compile, CUDA graphs (FULL_AND_PIECEWISE, to 512 tokens) and chunked prefill, except where a run
  says eager.
- **Library:** built from main at v0.25.0 (the branch's `gpu/`).
- **Plugin:** the vllm-plugin branch at 6adc96e for the checks and the bench at 1-∞ requests a second; at 49ce3b5,
  with 563da8e's `bench_serve.sh`, for the rest. The commits between take the GPU's code from the library (the same
  code on an L4 with this library) and add the scripts' options. `determinism/` ran minutes before 6adc96e, on its code
  but for the exact refusal's wording and a save's placeholders.
- **Models:** Qwen3-1.7B, Qwen3-4B-Instruct-2507 and Qwen3-8B, from the box's cache. Llama-3.1-8B is gated (its
  weights need a Hugging Face token, which is not handled here), so the Llama architecture was checked on
  `01-ai/Yi-1.5-6B-Chat` after M2 (Apache-2.0, ungated; its config names `LlamaForCausalLM`, so vLLM runs it with its
  Llama model code; hidden 4096, 32 layers, grouped-query attention, an LM head of its own), on v0.25.1's library.
- Every run held the box's lock, one GPU job at a time.

## Files

- `gpu/vllm/check_vllm.py`, `bench_serve.sh` and `bench_summary.py` (in the repo) made the checks and the benches.
- The runs, in order, each step under the box's lock (`run.log` is their console):
  - `m2_run.sh`: the Qwen3-1.7B check and the first (cold) bench; `m2_run2.sh`: the Qwen3-8B and Qwen3-4B checks (the
    same steps, the lock taken a step at a time);
  - `m2_extra.sh`: the bench at 0.25 requests a second, then `repro.sh`;
  - `m2_extra2.sh`: the bench again at 1 request a second and at once, warm;
  - `m2_extra3.sh`: `repeat.py`, then the l4-routes library: its build, Qwen3-8B's check run and the bench;
  - `m2_extra4.sh`: `opcheck.py` (now `diag/opcheck-m2.py`) under compute-sanitizer (m2_extra3.sh's run of it stopped
    before its first product: it had not loaded the library).
  - after M2, on v0.25.1's library (main at 617c027 with the plugin): `nondeterminism/` (`dbg1.sh` to `dbg4.sh`,
    `detcost.sh`), then `m3_prep.sh` (the library for sm_89, `dbg4.sh`, Yi-1.5-6B-Chat's download and check, the
    sanitizer runs), `chain2.sh` (M3's job's dry run, `detcost.sh`, `m3_prep.sh` again for the Yi check with the LM
    head fix), then `sanitizer/midrace.sh`.
- `check/`: each model's console output (`check-MODEL.txt`), and a JSON and a vLLM log a run (`check-MODEL/`).
- `bench/`: `bench-Qwen3-8B/` (cold), `bench-Qwen3-8B-low/` and `bench-Qwen3-8B-warm/` (warm),
  `bench-Qwen3-8B-l4routes/`: vLLM's result JSONs and console output, each server's log and KV cache, nvidia-smi's
  samples (from the low-load bench on), and `summary.txt`.
- `determinism/` (`determinism.sh`), `repro/` (`repro.sh`) and `diag/` (`repeat.py`, `opcheck-m2.py`): the same model
  and options in fresh processes and in one, and v0.25.0's kernels under compute-sanitizer's initcheck and memcheck.
- `nondeterminism/`: where two compiled processes part, and inductor's deterministic mode; `yi/`: Yi-1.5-6B-Chat's
  check; `sanitizer/`: v0.25.1's kernels under compute-sanitizer's racecheck, synccheck, initcheck and memcheck.
- `l4routes/`: Qwen3-8B on the l4-routes branch's library, and its build's output.

## Checks (`check_vllm.py`): all passed

Each run is a vLLM of its own at `gpu_memory_utilization=0.85` and `max_model_len=4096`: 8 prompts, 64 greedy tokens
each, and bf16's own continuation of the first prompt (1,536 tokens) fed back through `prompt_logprobs`. Glyd ran with
`GLYD_VERIFY=1`. Each first run compiled on a cache without its graph, so its KV cache is a cold start's (see
Serving). "Top-1" is the share of the continuation's tokens that each run also ranks first; "|Δ|" is the mean
|logprob difference| of those tokens from bf16's, CUDA graphs against CUDA graphs.

| | Qwen3-1.7B | Qwen3-4B-Instruct-2507 | Qwen3-8B | Yi-1.5-6B-Chat (after M2) |
| :--- | ---: | ---: | ---: | ---: |
| Weights: bf16 / tiered / 12-bit (GiB) | 3.22 / 2.48 / 2.90 | 7.64 / 5.52 / 6.44 | 15.27 / 11.64 / 12.34 | 11.29 / 8.63 / 9.05 |
| KV cache: bf16 / tiered / 12-bit (tokens) | 128,064 / 139,760 / 134,480 | 70,192 / 84,368 / 78,672 | 11,056 / 36,144 / 31,872 | 101,568 / 143,488 / 137,664 |
| Packs unpacked to their weights, bit for bit | 112 of 112 | 144 of 144 | 144 of 144 | 128 of 128 |
| Worst layer against F.linear, 1-4,096 tokens (relative) | 3.70e-3 | 3.72e-3 | 3.66e-3 | 3.61e-3 |
| bf16 eager against bf16: top-1 / \|Δ\| (the floor) | 0.9870 / 1.24e-2 | 0.9929 / 9.56e-3 | 0.9922 / 9.08e-3 | 0.9942 / 5.93e-3 |
| Glyd tiered against bf16: top-1 / \|Δ\| | 0.9916 / 1.16e-2 | 0.9935 / 7.87e-3 | 0.9942 / 8.40e-3 | 0.9948 / 4.97e-3 |
| Glyd 12-bit against bf16: top-1 / \|Δ\| | 0.9890 / 1.25e-2 | 0.9935 / 7.87e-3 | 0.9942 / 8.40e-3 | 0.9929 / 5.34e-3 |
| Exact, eager: bf16 eager's tokens, logprobs and continuation, bit for bit | 8 of 8, yes | 8 of 8, yes | 8 of 8, yes | 8 of 8, yes |
| Exact under torch.compile | refused | refused | refused | refused; with inductor deterministic, compiled bf16's bit for bit (8 of 8, yes) |
| Default mode, compiled, inductor deterministic, a run and a restart on its graphs | | | | the same bits (8 of 8, yes) |
| Saves, tiered and 12-bit, each loaded in both layouts | bit for bit | bit for bit | not run | bit for bit (after the LM head fix) |

- **Every layer's product** matched F.linear on its unpacked matrix within 1e-2 at 13 batch sizes, and gave the same
  bits on a second call.
- **Glyd's tokens in its default mode are within bf16's own noise** on all three models, in both layouts: Glyd's top-1
  agreement with bf16 is at or above bf16 eager's, and its |Δ| at most 1.003x bf16 eager's (the check allows 2x).
- **The compile cache** kept a graph for each of bf16, tiered and 12-bit on one `VLLM_CACHE_ROOT` (3 of 3, each
  model). Each layout started again on it loaded its own graph ("Directly load AOT compilation") and stood against
  bf16 as before.
- **Saves:** `glyd.save_pretrained` of each layout, loaded as saved and in the other layout (unpacked and packed again),
  gave the tokens, logprobs and continuation of the bf16 checkpoint packed at load in that layout, bit for bit (eager).
- **bf16 compiled again on an empty cache,** against its first compile: bit for bit on Qwen3-1.7B and Qwen3-8B; on
  Qwen3-4B-Instruct-2507 and Yi-1.5-6B-Chat, 0 of 8 prompts bit for bit and the continuation not.
- **Yi-1.5-6B-Chat** (`yi/`, `m3_prep.sh`): all 19 checks passed with v0.25.1's library. Its first run found that a glyd
  save of a model with its own LM head did not load: vLLM stopped at "There is no module or parameter named
  'lm_head.glyd_...' in LlamaForCausalLM" (Qwen3-1.7B and 4B tie theirs; Llama-3.1-8B and Qwen3-8B would have met it).
  The plugin now loads that pack as saved; `yi/` is the run with the fix.

## Serving Qwen3-8B (`bench_serve.sh`)

`vllm serve Qwen/Qwen3-8B [--quantization glyd] --max-model-len 4096 --gpu-memory-utilization 0.9`, then `vllm bench
serve` on the random dataset: 1,024 tokens in, 256 out (`--ignore-eos`). Glyd is the tiered layout, the L4's default.
Every request completed in every run.

**Two starts.** vLLM keeps each server's compiled graph in its compile cache. The first bench started each server on
an empty cache (cold): compiling there raised vLLM's profiled peak by about 1 GiB (1.61 against 0.62 GiB for bf16, 1.67
against 0.66 for Glyd), and the KV cache is sized after it. The later benches started on the filled cache (warm), as a
server started again does. The saving is the same 3.63 GiB of weights, about 25,450 tokens of KV cache, either way:

| | Weights | KV cache, warm | Requests of 1,280 tokens at once | KV cache, cold | At once |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 27,024 tokens | 21.1 | 19,760 tokens | 15.4 |
| Glyd tiered | 11.64 GiB | 52,496 tokens (1.94x) | 41.0 | 45,200 tokens (2.29x) | 35.3 |

**Warm** (`bench-Qwen3-8B-low/`: 32 prompts at 0.25 requests a second; `bench-Qwen3-8B-warm/`: 64 at 1, 256 at once;
each mode from about the GPU's idle temperature; the clock is nvidia-smi's median over the rate's samples while the GPU
worked, the temperature its highest):

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | bf16 | 0.22 | 56.5 | 458 / 903 | 69.9 / 73.2 | 66.7 / 322.1 | 1605 MHz, 74 C |
| 0.25 | glyd | 0.23 | 58.1 | 594 / 1,263 | 54.8 / 59.0 | 49.9 / 467.4 | 1245 MHz, 72 C |
| 1 | bf16 | 0.67 | 170.5 | 3,213 / 12,672 | 99.3 / 111.0 | 83.6 / 355.1 | 1500 MHz, 71 C |
| 1 | glyd | 0.77 | 196.5 | 870 / 1,779 | 100.1 / 132.1 | 67.9 / 829.8 | 1230 MHz, 70 C |
| inf | bf16 | 0.77 | 196.4 | 148,852 / 305,366 | 109.6 / 181.2 | 84.6 / 603.2 | 1425 MHz, 79 C |
| inf | glyd | 0.99 | 252.5 | 115,115 / 238,703 | 162.1 / 260.4 | 94.3 / 866.9 | 1215 MHz, 80 C |

**Cold** (`bench-Qwen3-8B/`: 64 prompts at 1 request a second, 128 at 4, 256 at 16 and at once; bf16 first, then Glyd
on a GPU still warm from it, no clock logged):

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.58 | 149.7 | 8,390 / 26,561 | 92.3 / 101.7 | 75.7 / 379.5 |
| 1 | glyd | 0.75 | 192.6 | 945 / 3,206 | 102.8 / 133.8 | 69.6 / 829.0 |
| 4 | bf16 | 0.64 | 164.5 | 71,070 / 149,222 | 97.2 / 156.8 | 76.8 / 595.9 |
| 4 | glyd | 0.90 | 229.3 | 39,242 / 96,190 | 144.0 / 231.2 | 89.1 / 859.6 |
| 16 | bf16 | 0.64 | 164.8 | 176,493 / 364,098 | 96.5 / 157.5 | 80.0 / 595.6 |
| 16 | glyd | 0.94 | 240.3 | 114,925 / 238,527 | 146.1 / 238.1 | 89.1 / 860.5 |
| inf | bf16 | 0.64 | 164.6 | 184,519 / 380,442 | 96.6 / 157.8 | 80.0 / 597.9 |
| inf | glyd | 0.94 | 240.9 | 122,695 / 253,569 | 145.7 / 236.7 | 88.9 / 859.7 |

With requests waiting (the servers' logs, every 10 s), bf16 ran a median of 22 requests at once warm and 16 cold, Glyd
42 and 36, each with its KV cache 97-98% full.

**Where Glyd wins and where it loses, on this L4 with this model** (warm; cold in brackets):

- **Capacity:** 1.94x the KV cache [2.29x]: 41 requests of 1,280 tokens at once against 21 [35 against 15].
- **Low load, 0.25 requests a second:** each request finished in 20% less time (14.6 against 18.3 s): a token every
  54.8 ms against 69.9 (-22%). Its first token came 30% later (594 against 458 ms).
- **1 request a second,** past bf16's saturation: Glyd served 15% more requests a second [29%], its first token in
  0.87 s against 3.2 s [0.95 against 8.4], a token every 100.1 ms against 99.3 [102.8 against 92.3].
- **Saturated:** Glyd served 1.29x the requests and output tokens a second [1.39-1.46x from 4 requests a second up],
  with a 23% lower mean time to first token [33-45%]. Its time per output token was 48% higher [48-51%]: each step
  carried about twice the requests.
- **Steps with a prompt in them** (the p99 inter-token latency) took 44-45% longer than bf16's at 0.25 requests a
  second and saturated, 2.3x at 1 a second. With v0.25.0's library an L4's prompts took +27.8% and +31.0% over bf16's
  time at 1,024 and 2,048 tokens (the l4-routes branch measured it); that branch takes a faster path for an L4's
  prompts from 896 tokens.
- **Steps without a prompt** (the median inter-token latency) were 25% shorter at 0.25 requests a second, 19% at 1,
  and 11% longer saturated, for about twice the requests a step.
- **Clock:** at the L4's 72 W cap Glyd's kernels ran at a median 1,215-1,245 MHz, bf16's at 1,425-1,605.

**Host overhead.** The op costs the host 44-54 µs a product, against F.linear's 15-22 µs (`host_us_*` in each check's
layers). It shows only where a step is not a CUDA graph, over 512 tokens. For Qwen3-8B that is 144 products, under 4 ms
more host time for a step whose GPU time runs to hundreds of milliseconds (the inter-token p99 above). The gap in the
prompt steps is the GPU's time, not the host's.

## The same run in a fresh process, and exact mode

Qwen3-1.7B, tiered, the check's 8 prompts and continuation. `determinism.sh` and `repro.sh` start each run in a
process of its own; `repeat.py` generates three times in one process (prefix caching off).

| Runs compared | Glyd: prompts bit for bit, continuation | bf16 |
| :--- | :--- | :--- |
| Eager, two processes | 8 of 8, yes | |
| Compiled, two processes on two empty caches | 0 of 8, no | 5 of 8, no |
| Compiled with CUDA graphs, a second process on the first's cache (its graph loaded) | 0 of 8, no (twice) | 8 of 8, yes |
| Compiled without CUDA graphs, likewise | 0 of 8, no | 8 of 8, yes |
| Compiled, three times in one process (the prompts alone) | 8 of 8, twice | 8 of 8, twice |

bf16 compiled again on an empty cache, in the checks: bit for bit on Qwen3-1.7B and Qwen3-8B, 0 of 8 on
Qwen3-4B-Instruct-2507. So vLLM's own compiled bf16 is not always the same from one compile to the next.

### The cause (`nondeterminism/`, after M2)

Not Glyd's kernels. `dbg_hash.py` hashes, in call order, every product's input and output and every FlashAttention
call's query, key, value and output over the continuation's one pass (1,542 tokens; Qwen3-1.7B, tiered, compiled with
CUDA graphs off so every call runs its Python), each run a process of its own, the first compiling and the others
loading its graphs (`dbg1.sh`, `dbg2.sh`: v0.25.0's library).

- Across 12 pairs of processes, the library's product never gave another output for the same input; nor did
  FlashAttention, in the 9 pairs where it was hooked too (`dbg2.sh`).
- The first tensor to differ was always attention's query or key, its value the same, after a qkv product that
  matched. Between the two run the q and k RMSNorm and the rotary embedding, which vLLM leaves to inductor (its custom
  ops off) and has it group into combo kernels, benchmarked (`combo_kernels` and `benchmark_combo_kernel`, vLLM's
  defaults, for this fusion).
- With combo kernels off, or on without their benchmark: the processes still parted, at the same place.
- With inductor's deterministic mode (`"deterministic": true`: no on-device benchmarking that moves numerics; it
  refuses the combo kernels' benchmark, which vLLM turns on, so `"benchmark_combo_kernel": false` with it), three
  processes, one compiling and two loading, gave the same bits: every product, every attention call and all 1,541
  logprobs, Glyd's and bf16's (`dbg3.sh`).
- In that mode, compiled with CUDA graphs as vLLM runs by default, each run on an empty compile cache (`dbg4.sh`,
  v0.25.1's library): bf16 against bf16 8 of 8 prompts and the continuation bit for bit; Glyd's default mode against
  itself the same; and exact mode, its refusal lifted for the test, against bf16: 8 of 8 and the continuation bit for
  bit. Glyd's default-mode continuation (1,542 tokens) was bf16's bit for bit too (an L4's long prompts in v0.25.1).

Inductor's deterministic mode costs nothing measurable here (`detcost.py`: Qwen3-8B, CUDA graphs on, each on an empty
compile cache; tokens/s at 1, 8 and 32 sequences, then 8 prompts of 1,024 tokens):

| | Default | Deterministic |
| :--- | :--- | :--- |
| bf16 | 16.7 / 126.3 / 449.5 tokens/s; 2.179 s | 16.7 / 126.1 / 448.6 tokens/s; 2.183 s |
| Glyd tiered | 21.7 / 168.0 / 584.9 tokens/s; 2.284 s | 21.6 / 167.5 / 589.2 tokens/s; 2.240 s |

Its compiles were faster, having no combo kernels to benchmark: Glyd's cold start took 118 s against 153.

Why vLLM's compiled bf16 agreed on one cache in these pairs while Glyd's did not is not shown here.

So exact mode now runs compiled where inductor is deterministic, `--compilation-config '{"inductor_compile_config":
{"deterministic": true, "combo_kernels": true, "benchmark_combo_kernel": false}}'`, and is refused compiled without
it, the message giving both ways; check_vllm.py checks both (Yi-1.5-6B-Chat below). Exact mode eager is bf16 eager's
bit for bit on all four models.

## The l4-routes library (`l4routes/`, `bench-Qwen3-8B-l4routes/`)

The l4-routes branch (1f4343b, not merged) takes a faster path for an L4's prompts from 896 tokens in the tiered
layout. The plugin takes the library's choices, so with that library built and loaded in place of v0.25.0's:

- **Qwen3-8B's check run** (tiered, compiled): every pack verified (144), every layer within 4.67e-3 of F.linear with
  the same bits on a second call. The continuation's 1,543-token pass took the new path, and its logits were compiled
  bf16's bit for bit (top-1 0.9935, as bf16's own). The 8 prompts' steps, in the default mode, were as before (0 of 8
  bit for bit, the same tokens as v0.25.0's library).
- The new path's host time is 85 µs a product at 1,024 tokens.

**The bench with that library** (`bench-Qwen3-8B-l4routes/`, Glyd alone, warm, against the warm bf16 above):

| Rate (req/s) | Library | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | l4-routes | 0.23 | 58.1 | 537 / 1,128 | 54.1 / 57.6 | 49.9 / 408.1 | 1230 MHz, 76 C |
| 1 | l4-routes | 0.78 | 198.5 | 504 / 1,535 | 79.9 / 113.6 | 64.1 / 461.8 | 1155 MHz, 79 C |
| inf | l4-routes | 1.07 | 272.8 | 106,481 / 217,471 | 148.7 / 235.0 | 93.4 / 727.9 | 1140 MHz, 80 C |

- Its scratch buffer is 0.19 GiB: the weights' line reads 11.83 GiB, and the KV cache holds 51,040 tokens (52,496
  without it).
- Against bf16: the first token at low load +17% (537 against 458 ms; v0.25.0's library, +30%), and the steps with a
  prompt +27% at inter-token p99 (+45%); at 1 request a second a token every 79.9 ms against 99.3 (-20%); saturated,
  1.39x the requests a second (1.29x) and a token every 148.7 ms against 109.6 (+36%; +48%).
- It ran as hot as v0.25.0's library's bench or hotter (76-80 C at most, against 70-80 C) and at a lower median clock
  (1,140-1,230 MHz, against 1,215-1,245).

## compute-sanitizer on v0.25.1's kernels (`sanitizer/`)

`opcheck.py` runs the library's linear (route -1: its own choice of kernel) and its whole-matrix unpack on random
packs, both layouts, shapes 2048x2048 and 1024x4096, at 1, 16, 17, 33, 64, 65, 129, 300 and 1,024 tokens, each product
checked against the unpacked matrix's.

- **synccheck, initcheck, memcheck:** 0 errors.
- **racecheck:** reported potential hazards in one kernel alone, the 12-bit layout's kernel for 17-64 tokens on Ampere
  and Ada (to 128 on an A100); the other kernels were clean. The hazards are "potential" ones
  (`mid-racecheck-lineinfo.txt`).
  - Called 2,000 times a shape (2048x2048, 6144x2048, 2048x6144, 12288x2048) and M (17, 33, 64), that kernel gave the
    same bits on every call, within 1.9-3.1e-3 of the unpacked matrix's product (`mid-stress.txt`); the tiered layout's
    kernel beside it likewise. So these read as false positives, not a race; the tiered layout, whose products never
    differed for the same inputs across processes, does not take this path.
