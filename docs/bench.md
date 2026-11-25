# glyd bench

One harness for the speed of a server, one JSON format for the result, and a gate that holds a change to the last result on the same GPU. The code is `glyd-engine/src/bench.rs`; the command is
`glyd-engine bench` (and `glyd bench` where the `glyd` program passes it on). The earlier `glyd-engine bench DIR` (one sequence, each decode step timed on the host, `--profile`) is `glyd-engine microbench DIR`.

## Measure

```bash
glyd-engine bench google/gemma-4-12B-it:wren --suite gate --runs 5 --no-draft --out result.json   # starts `glyd-engine serve` of this program, measures it, stops it
glyd-engine bench --against http://127.0.0.1:8001 --suite bar --out vllm.json   # any OpenAI-compatible server: vLLM, SGLang, llama.cpp
```

For each load (a prompt kind and a number of simultaneous requests) it runs a warm-up round and `--runs` rounds. Every prompt is different from its first token on, so no prefix is shared; the prompts
are fixed (`bench::prompt`: a question of about 35 tokens, or random words of a fixed seed to 1,000 to 32,000 tokens and a question that ignores them), so two runs of two programs read the same text.
Each request is a streamed chat completion at temperature 0 with `--max-tokens` (128) and thinking off.

| Suite | Loads |
| :--- | :--- |
| `gate` (default) | short x1, short x8, 1k x1, 4k x1: what a pull request is held to |
| `bar` | short, 2k and 4k prompts, each at 1, 4 and 8 at once: the speed bar's matrix (docs/adding-a-gpu.md) |
| `full` | short at 1, 8, 64 and 256 at once; 1k, 4k and 32k at 1: everything R8 lists (64 and 256 at once need the server to take them: R11) |
| `arrive` | 8 requests one after another: short prompts 100 ms apart, 1k 250 ms apart, 4k 500 ms apart. The later prompts come in while the earlier replies are being written |
| `chat` | conversations of 4 turns that begin with a 1k and a 4k prompt, 1 and 8 of them at once: each later turn resends the conversation and adds a question |
| `shared` | 8 requests 250 ms apart whose 1k or 4k of notes are the same and whose questions differ (`same1k`, `same4k`), and 8 conversations of 4 turns that begin with the same 4k (`same4k*4`): what a server shares of a start it has run |
| `occupancy` | short at 1, 8, 32 and 64 at once: one-token and batched decode as shares of the ceiling, the datacenter occupancy check of P20 (benchmarks/consumer/README.md) |

`--loads short:1,4k:8` names loads by hand. `4k+250ms:8` is 8 requests of 4k prompts, each begun 250 ms after the one before, not all at once: what a server does when a prompt comes in
while it is writing other replies shows in that load's first-token time and in the 99th percentile of the gaps between tokens (a reply that stops while a prompt runs has a long gap). Such a load has
no prefill rate, and its name in the result is the kind as written (`4k+250ms`).

`1k*4:8` is 8 conversations of 4 turns, each begun with a 1k prompt. A later turn sends the conversation so far (the prompts and the replies it got) and a short question more, as a chat client
does. The load's `ttft_s` is the first turns' first token and `ttft_later_s` the later turns'; `prefix_hit_rate` is the share of the prompt tokens of the measured rounds that a glyd server found
in its cache (its `/stats`; the server the bench starts is told its callers are one user, `GLYD_ENGINE_CACHED_USAGE=1`, so it says; null for a server given with
`--against` that does not). A server that keeps nothing of a prompt it has seen runs every turn from its first token: its later turns' first token grows with the
conversation and its hit rate is 0. `1k*4+250ms:8` begins the conversations 250 ms apart. The server `bench` starts is asked for as many users as the heaviest load sends (up to the scheduler's 64; `--users N` says otherwise): `serve` alone takes 4, and 8 requests at once would be 4 and 4 waiting. Where the server's plan holds fewer (its `/v1/glyd/info` `limits.concurrency`), the bench says so before it measures and the result keeps it as `server.concurrency` beside `server.users`. A load that fails is counted (`errors`), not hidden.

Per load and run: time to first token, decode tokens/s a user (tokens after the first over the time from first to last), the inter-token latency (its mean and the 99th percentile of every gap), decode tokens/s
in all (every user's tokens after the first over the span from the earliest first token to the last token), and prefill tokens/s (the prompts' tokens over the slowest first token). A user's figure
is the median over the requests of the round; each file keeps, for each figure, the runs' **mean, standard deviation, median, smallest, largest and the runs themselves**.

The peak GPU memory is sampled every 100 ms from `nvidia-smi` while everything runs. Before the server starts the harness looks at the GPU five times 0.3 s apart: if its median utilization is past 10% or
more than 2,500 MiB are in use, the result says `contention.busy_before: true` and the gate will not decide on it.

## Engines side by side: `--report`

A result says which engine served it and what a comparison groups it by: `--engine NAME` (`glyd` for the server the bench starts), `--source REPO` (the
Hub repo whose weights the server runs; the bench's own model without its tag by default), `--bits B` (bits a weight; a Glyd level's by default) and
`--params B` (parameters, in billions), kept under `server.engine` and `model.source`, `model.bits`, `model.params_b`.

```bash
glyd-engine bench --report DIR [--title TEXT] > DIR/README.md
```

reads every result under DIR and writes the comparison page as Markdown: per GPU and its tier (8, 12, 16, 24 or 32 GB), each engine's largest model
(by `params_b`) whose decode one user at a time on a short prompt is 20 tokens/s or more with no failed request, then each model's results side by
side (the most bits first) with the files beside each (`NAME.cmd`, `NAME.log`) and the sides that did not run (`NAME.skip`). `scripts/consumer_bench.sh`
makes such a directory for a release (benchmarks/consumer/README.md).

## Where the time goes: `GLYD_PROF=1`

A development build (the default features) set `GLYD_PROF=1` times the description executor's passes by operation on the GPU itself, with no external profiler
(`glyd-engine/src/dynamic/prof.rs`): a CUDA event after each operation of every pass, inside the step's graph where a decode step replays one (each sequence's graph
holds its own events). `bench` starts its server with the variable, reads the server's figures of each load's measured rounds (`GET /v1/glyd/prof?reset=1`), prints them
under the load's line and keeps them in the load's cell (`gpu_profile`); `glyd-engine run` prints them when it ends.

```bash
GLYD_PROF=1 glyd-engine bench google/gemma-4-12B-it:wren --loads short:1,short:8,1k:1,4k:1 --runs 3 --no-draft --out prof.json
```

For each kind of pass (decode rows alone, a prompt's chunk alone, or both in one pass; prompts grouped by the power of two above their tokens) it gives the passes, how many
were a graph's replay, the GPU time a pass and each operation's share (`norms`, `attention: qkv product`, `attention: preparation (norms, rotary, cache write)`,
`attention (global)`, `attention: o product`, `MLP: gate and up product, SiLU`, `MLP: down product`, `head`, ...). An eager pass's operation is timed from the event of the
one before it, so a host slower than the GPU shows as GPU time: `queued_ms`, the host's time to queue the pass, close to its GPU time says the pass waits on the host.
For each kind of step (a call of the scheduler's step) it gives the wall time, the GPU time of its passes, the rest (the host's work, the copies and the waits around the
passes) and the time from one call's end to the next one's start (the scheduler, the sampler, the server; for prompts it holds the time no request was waiting). The
events cost a little (about 2 µs each in a graph on an RTX 4080 SUPER, docs/ceiling.md): read the speeds from a run without the variable. Operations an executor family runs with no event of their own are
counted in the next operation that has one.

## The ceiling

`pct_of_ceiling` of a load is its decode tokens/s a user over the ceiling: the memory's bandwidth over the weights' bytes a token. The bandwidth is the spec sheet's for the GPU's name (`bench::spec_bandwidth`:
736 GB/s for an RTX 4080 SUPER, 3,352 for an H100 SXM; `--bandwidth-gbs` for any other) and the bytes are what the server says it holds (`/v1/glyd/info`, `memory.weights_bytes`) or `--weights-gib` (Gemma 4 12B at the wren
level in the engine: 6.3 GiB, a ceiling of 108.8 tokens/s on the RTX 4080 SUPER). KV reads are not counted, so a long prompt's share is lower than its ceiling would make it. The engine drafts for
Gemma 4 (speculative decoding, 8 tokens a round), which can pass the ceiling: `--no-draft` starts the server without it, and the gate's results are taken that way.

## The format: `glyd-bench/1`

```text
schema, label, recorded_unix
machine      gpu, driver, memory_mib, max_sm_mhz, max_mem_mhz, power_limit_w, compute_capability, cpu, cpu_threads, os, arch
model        id, argument, level, weights_bytes, weights_source, source, bits, params_b
bandwidth    gbs, source
versions     glyd_engine, git, server
server       against (the URL, or null), engine, started_by_bench, users (asked of a server the bench starts), concurrency (what the server says it runs at once)
config       suite, loads, runs, warmup, max_tokens
contention   busy_before, utilization_median_pct, before, after (memory, utilization, clocks, the computer's load)
idle_mib, peak_mib
cells[]      prompt, conc, prompt_tokens, errors, ceiling_tps_per_user, pct_of_ceiling,
             ttft_s, decode_tps_per_user, itl_ms, itl_p99_ms, agg_tps, prefill_tps: each {mean, sd, median, min, max, runs[]}
             gpu_profile (GLYD_PROF=1 only): passes[] {kind, rows, prompt_tokens, passes, graphs, gpu_ms, queued_ms, ops[] {op, ms, share}},
                                             steps[] {kind, calls, wall_ms, gpu_ms, host_ms, between_ms}
```

## The gate

```bash
glyd-engine bench --compare BASE.json CAND.json [--history DIR] [--z 3] [--floor 0.003]
```

Each of four metrics (decode tokens/s a user, in all, prefill tokens/s, time to first token) of each load of CAND against BASE's medians. A change is a **regression where it is worse by more than the noise band**,
and the band is measured, not a percentage chosen:

- the spread of the runs: `z` (3) times 1.25 times the square root of the sum of each file's (standard deviation over its mean)^2 over its runs: the standard error of the two medians;
- where `--history` holds three or more earlier results on the same GPU and model, the spread of their medians from one session to another (`z` times its relative standard deviation times the square root of 1 + 1/k): runs
  within a session do not see the GPU's clocks, the host's load or the server's restart, sessions do; the band is the larger of the two;
- a floor of 0.3% (`--floor`), the resolution of the figures.

Exit 0: within the band. Exit 1: a regression (or failed requests). Exit 2: not decided, the candidate was measured on a busy GPU. Results of different GPUs or models are refused: a gate compares a GPU with itself.
`--synthetic F` scales CAND's speeds by F (its times by 1/F) first: the gate's own test (`--synthetic 0.8` must exit 1; CI does it).

### In CI

`.github/workflows/gpu-bench.yml` runs on the self-hosted GPU machine (the repository variable `GPU_RUNNER`): on a merge to main it measures the gate's loads on Gemma 4 12B at the wren level (released 2026-05-23; the model is a recent one, as every model we test or benchmark is) and keeps the result in the runner's volume as
main's last on that GPU (`/cache/bench/main-GPU.json`, with the last 20 in `history-GPU/`); on a pull request labeled `gpu` it measures the same and runs `bench --compare` against main's last, then the synthetic
slowdown. A result taken while the GPU was busy (other jobs of the runner do not wait for it) is taken again, up to 20 times 45 s apart.

### Where the noise band is

From the runs of the workflow itself: glyd-pc runs GitHub Actions jobs only (a model is measured in CI, or on a rented GPU, never by hand on it), so main's results keep in `history-GPU/` as the merges come, and
from the third on the gate's band takes in the spread from one session to another. Until then it is the spread of the runs of the two files with a floor of 0.3%. An earlier measurement of five sessions
on Qwen3-8B (not a recent model, so not kept in the tree) had decode within 0.03-0.12% from session to session, the prefill of a long prompt and a single request's first token within 0.15-0.4%.

## The speed bar

`glyd-engine bench --bar ENGINE.json OTHER.json` (and `scripts/gpu_bar.py`, which hands it its arguments): at every load the first server's decode tokens/s, in all and for a user, and its time to first token at
least the second's within 3%, its peak memory lower; a pass prints the line for `GPUS`, per (GPU, family) (`--family`), for a model released within 6 months (`--released`; exit 2 for an older one, `--stale-ok`). The same program measured both: docs/adding-a-gpu.md.

## Not yet

- Quality: the lossy levels' quality bar is its own command, `glyd-engine quality` (docs/quality.md: KL on four fixed texts and small task sets against the lossless level, a card per pack, publish or refuse). In the bench, `--quality` takes the next-token distribution (the top 20 log-probabilities of a raw completion of one token) after 48 prefixes of one fixed text and keeps them in the result; `--quality-ref RESULT.json` gives the KL against a reference result's (mean, largest, and at how many prefixes the best token is the same), and `bench --compare` holds a candidate to its base's (a mean past `--kl-max`, 0.005, or the best token changed at more than 5% of the prefixes is a regression). It needs a server that returns `logprobs` on completions (the engine's, vLLM, llama.cpp). Not yet: the logits fixtures' references of `glyd-engine/docs/logits-tests.md` in the result, and a reference result on a recent model (needs a GPU run: a rental).
- 64 and 256 at once: the `full` suite asks for them; the server answers as many as it takes.
