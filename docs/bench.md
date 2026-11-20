# glyd bench

One harness for the speed of a server, one JSON format for the result, and a gate that holds a change to the last result on the same GPU. The code is `glyd-engine/src/bench.rs`; the command is
`glyd-engine bench` (and `glyd bench` where the `glyd` program passes it on). The earlier `glyd-engine bench DIR` (one sequence, each decode step timed on the host, `--profile`) is `glyd-engine microbench DIR`.

## Measure

```bash
glyd-engine bench Qwen/Qwen3-8B --suite gate --runs 5 --out result.json        # starts `glyd-engine serve` of this program, measures it, stops it
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

`--loads short:1,4k:8` names loads by hand. The server `bench` starts takes as many users as the heaviest load asks for (at most 8; `--users N` says): `serve` alone takes 4, and 8 requests at once would be 4 and 4 waiting. A load that fails is counted (`errors`), not hidden.

Per load and run: time to first token, decode tokens/s a user (tokens after the first over the time from first to last), the inter-token latency (its mean and the 99th percentile of every gap), decode tokens/s
in all (every user's tokens after the first over the span from the earliest first token to the last token), and prefill tokens/s (the prompts' tokens over the slowest first token). A user's figure
is the median over the requests of the round; each file keeps, for each figure, the runs' **mean, standard deviation, median, smallest, largest and the runs themselves**.

The peak GPU memory is sampled every 100 ms from `nvidia-smi` while everything runs. Before the server starts the harness looks at the GPU five times 0.3 s apart: if its median utilization is past 10% or
more than 2,500 MiB are in use, the result says `contention.busy_before: true` and the gate will not decide on it.

## The ceiling

`pct_of_ceiling` of a load is its decode tokens/s a user over the ceiling: the memory's bandwidth over the weights' bytes a token. The bandwidth is the spec sheet's for the GPU's name (`bench::spec_bandwidth`:
736 GB/s for an RTX 4080 SUPER; `--bandwidth-gbs` for any other) and the bytes are what the server says it holds (`/v1/glyd/info`, `memory.weights_bytes`) or `--weights-gib` (Qwen3-8B lossless in
the engine: 10.3 GiB, a ceiling of 66.5 tokens/s on the RTX 4080 SUPER). KV reads are not counted, so a long prompt's share is lower than its ceiling would make it.

## The format: `glyd-bench/1`

```text
schema, label, recorded_unix
machine      gpu, driver, memory_mib, max_sm_mhz, max_mem_mhz, power_limit_w, compute_capability, cpu, cpu_threads, os, arch
model        id, argument, level, weights_bytes, weights_source
bandwidth    gbs, source
versions     glyd_engine, git, server
server       against (the URL, or null), started_by_bench
config       suite, loads, runs, warmup, max_tokens
contention   busy_before, utilization_median_pct, before, after (memory, utilization, clocks, the computer's load)
idle_mib, peak_mib
cells[]      prompt, conc, prompt_tokens, errors, ceiling_tps_per_user, pct_of_ceiling,
             ttft_s, decode_tps_per_user, itl_ms, itl_p99_ms, agg_tps, prefill_tps: each {mean, sd, median, min, max, runs[]}
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

`.github/workflows/gpu-bench.yml` runs on the self-hosted GPU machine (the repository variable `GPU_RUNNER`): on a merge to main it measures the gate's loads on Qwen3-8B and keeps the result in the runner's volume as
main's last on that GPU (`/cache/bench/main-GPU.json`, with the last 20 in `history-GPU/`); on a pull request labeled `gpu` it measures the same and runs `bench --compare` against main's last, then the synthetic
slowdown. A result taken while the GPU was busy is taken again, up to three times a minute apart.

### Where the noise band is

Measured on the RTX 4080 SUPER of glyd-pc, five sessions of the gate's loads (a server started for each): see
[benchmarks/gpu/rtx4080s-bench-2026-11-20](../benchmarks/gpu/rtx4080s-bench-2026-11-20).

## The speed bar

`glyd-engine bench --bar ENGINE.json OTHER.json` (and `scripts/gpu_bar.py`, which hands it its arguments): at every load the first server's decode tokens/s, in all and for a user, and its time to first token at
least the second's within 3%, its peak memory lower; a pass prints the line for `GPUS`, per (GPU, family) (`--family`), for a model released within 6 months (`--released`; exit 2 for an older one, `--stale-ok`). The same program measured both: docs/adding-a-gpu.md.

## Not yet

- Quality (KL against a reference on a fixed text set, a small eval): the logits fixtures of `glyd-engine/docs/logits-tests.md` are the fixed text and the references; the harness does not run them into its result yet.
- 64 and 256 at once: the `full` suite asks for them; the server answers as many as it takes.
