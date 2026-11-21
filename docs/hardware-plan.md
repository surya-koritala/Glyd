# Hardware plan for CI and release gates

Which machine runs each milestone gate in [ROADMAP.md](../ROADMAP.md), how a run is started, how long it takes and what it costs
at public on-demand prices. Plan only: no instance has been started for this page and nothing has been spent.

Prices were read on 2026-11-20 (see [Prices](#prices)). They change; read them again before a run. Hours are estimates
unless the row says where they come from. The tables count a pass of each gate once.

## What we have

| Machine | What it is | Used for |
| :--- | :--- | :--- |
| glyd-pc | RTX 4080 SUPER 16 GB (sm_89), 60 GB RAM, Ubuntu, driver reporting CUDA 13.2 ([record](../benchmarks/gpu/rtx4080s-qwen3-8b-engine-2026-11-20/README.md)). The self-hosted runner `vars.GPU_RUNNER` names (`tools/ci/setup-gpu-runner.sh`) | `gpu.yml`; the 16 GB consumer tier |
| The owner's Mac | MacBook Pro, Apple M1 Max | Metal and macOS checks, by hand; covers the M1 row of M5a |
| Lambda | Account with credits. 8x A100, 8x H100 and 1x H100 runs were made through its API (`scripts/gpu_lambda.sh`, `research/*/launch.sh`, `go.sh`) | Datacenter GPUs, 8-GPU nodes |
| Vast.ai | Account with credits. Consumer cards, rented by offer | RTX 3000, 4000 and 5000 series; Windows VM offers were not checked |
| AWS | Account with credits. `gpu-cloud.yml` starts a GPU runner, but its role and variables are not set up ([#128](https://github.com/surya-koritala/Glyd/issues/128)) | L4, A10G, L40S, T4; Windows GPU instances |
| GitHub-hosted runners | Free for standard runners in a public repository ([GitHub docs](https://docs.github.com/en/billing/concepts/product-billing/github-actions)) | Everything without a GPU: Windows, macOS, arm64 Linux, installers |

Owner for every row below: Surya Koritala, the only maintainer ([GOVERNANCE.md](../GOVERNANCE.md)). A row gets a second name when a
maintainer takes the area.

## Prices

All in USD per hour, on demand, before tax. Read on 2026-11-20 between 00:24 and 00:30 US Eastern.

- **Lambda**, instances table of [lambda.ai/pricing](https://lambda.ai/pricing): 1x H100 SXM $4.89, 2x H100 SXM $4.79 per GPU,
  8x H100 $4.59 per GPU ($36.72 for the node), 8x A100 80 GB $3.19 per GPU ($25.52), 1x A100 SXM 40 GB $1.99, 1x B200 $7.99,
  8x B200 $7.69 per GPU, 1x GH200 $2.59, 1x A10 $1.29. `research/kimi-k3-a100-2026-10-22b/go.sh` says $22.32 for the 8x A100 box:
  the price has gone up since 2026-10-22. Lambda lists no H200, L4, L40S, T4 or AMD part. Storage and egress are priced apart and are not in
  this plan.
- **AWS**, us-east-1, Linux, the price list behind [aws.amazon.com/ec2/pricing/on-demand](https://aws.amazon.com/ec2/pricing/on-demand/)
  (file published 2026-11-20 00:50 UTC): g4dn.xlarge (T4) $0.526, g6.xlarge (L4) $0.8048, g5.xlarge (A10G) $1.006, g6e.xlarge (L40S)
  $1.861, g6e.12xlarge (4x L40S) $10.4926, p5.4xlarge (1x H100) $6.88, p5.48xlarge (8x H100) $55.04, p4de.24xlarge (8x A100 80 GB)
  $27.447, p5en.48xlarge (8x H200) $63.296, p6-b200.48xlarge (8x B200) $113.9328. Windows: g4dn.xlarge $0.71, g6.xlarge $0.9888,
  g5.xlarge $1.19, g6e.xlarge $2.045. Mac instances are bare metal on Dedicated Hosts with a 24-hour minimum
  ([AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-mac-instances.html)); their price is not in this plan.
  Storage, snapshots and data transfer are extra.
- **Vast.ai** has no fixed price list: [vast.ai/pricing](https://vast.ai/pricing) points to a marketplace. The figures are the median
  of the 1-GPU on-demand offers that its public search (`console.vast.ai/api/v0/bundles`, by `gpu_name`) returned at the time:
  RTX 3060 $0.074, RTX 3070 $0.115, RTX 3090 $0.206, RTX 4090 $0.469 (cheapest $0.298), RTX 5080 $0.278, RTX 5090 $0.607 (cheapest $0.269),
  L4 $0.350, T4 $0.148, A100 PCIe $0.803, H100 SXM $2.407, B200 $7.503. The same search returned no offers for MI300X, RX 7900 XTX,
  RX 9070 XT or GH200. Offers differ in driver version, host quality and disk and bandwidth charges; the tables use the median and
  ignore those charges.

## The gates

Each table lists the runs of one gate. "Started by" is the script or workflow that exists today, or says by hand. Where a gate needs a
script that does not exist, the nearest one is named.

### M0 (v0.33.0)

Gate: A tag on main produces every release artifact; on a clean Linux machine with an NVIDIA driver, `install.sh` gives a first token from the engine without Python; Linux, Windows and macOS compile CI and GPU CI are green.

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| GPU CI on every main push that touches the engine or GPU code | RTX 4080 SUPER (sm_89) | glyd-pc | `gpu.yml`, automatic | 0.5 | owned | $0 |
| GPU CI on an L4 | L4 24 GB (sm_89) | AWS g6.xlarge | `gpu-cloud.yml`, by hand | 1 | AWS $0.8048/h | $0.80 |
| GPU CI on an A10G | A10G 24 GB (sm_86) | AWS g5.xlarge | `gpu-cloud.yml`, by hand | 1 | AWS $1.006/h | $1.01 |
| GPU CI on an L40S | L40S 48 GB (sm_89) | AWS g6e.xlarge | `gpu-cloud.yml`, by hand | 1 | AWS $1.861/h | $1.86 |
| T4 is refused with a clear message (Turing, until V5) | T4 16 GB (sm_75) | AWS g4dn.xlarge | `gpu-cloud.yml`, by hand | 0.5 | AWS $0.526/h | $0.26 |
| Clean machine: `install.sh`, first token, no Python | an NVIDIA GPU with driver 580+, fresh Ubuntu 24.04 | AWS g6.xlarge | by hand (no script yet) | 0.75 | AWS $0.8048/h | $0.60 |
| Windows and macOS compile and unit tests | none (no GPU) | GitHub-hosted | `ci.yml` matrix (S9, #213), automatic | 0.25 | free in a public repository | $0 |

Total for one pass of M0: **$4.54**.

### M1 (v0.34.0)

Gate: On clean Ubuntu 24.04 and Windows 11 machines with only the NVIDIA driver (an 8 GB laptop GPU, a 16 GB desktop GPU, and one on the oldest supported driver): install to first token in under 2 minutes for 10 pinned models; the plan is what loads; Open WebUI, Continue and an OpenAI SDK client connect; JSON-schema output is valid on every pinned model that runs.

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| Ubuntu 24.04, 8 GB GPU: 10 models, install to first token, tools connect | 8 GB GPU (RTX 3070 8 GB stands in for a laptop part) | Vast.ai 1x RTX 3070 | by hand | 2 | Vast median $0.115/h | $0.23 |
| Ubuntu 24.04, 16 GB GPU: the same | RTX 4080 SUPER 16 GB | glyd-pc | by hand, in a fresh container or user | 2 | owned | $0 |
| Ubuntu 24.04 on the oldest supported driver: the same | any supported GPU, driver chosen by the offer | Vast.ai 1x RTX 3090 | by hand; pick the offer by `driver_version` | 2 | Vast median $0.206/h | $0.41 |
| Windows 11, 24 GB GPU: the same | L4 24 GB | AWS g6.xlarge, Windows | by hand | 2 | AWS Windows $0.9888/h | $1.98 |
| Windows 11 on the oldest supported driver: the same | A10G 24 GB | AWS g5.xlarge, Windows | by hand | 2 | AWS Windows $1.19/h | $2.38 |
| Windows 11 with an 8 GB or 16 GB GPU | 8 GB and 16 GB GeForce | owner's machine (none rentable) | by hand | 2 | gap: see below | $0 |
| An Apple or AMD machine gets a one-line answer | a Mac; a Radeon | owner's machine | by hand | 0.25 | owned | $0 |

Total for one pass of M1: **$5.00**.

### M2 (v0.35.0)

Gate: 64 concurrent requests through one scheduler on an RTX 4080 SUPER and an L4 with logits matching the CPU reference; Kimi-K3, DeepSeek-V4 and GLM-5.3 keep single-user parity on their reference hardware; no `glyd bench` regression; every pack since v0.29 loads; 1x H100 occupancy canary recorded (`P20`).

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| 64 concurrent requests, logits against the CPU reference | RTX 4080 SUPER | glyd-pc | `gpu.yml` by hand, then `glyd bench` (R8) | 2 | owned | $0 |
| The same on an L4 | L4 24 GB | AWS g6.xlarge | `gpu-cloud.yml`, by hand | 2 | AWS $0.8048/h | $1.61 |
| 1x H100 single-token and batched occupancy canary (`P20`, `V8a`) | 1x H100 80 GB | Lambda 1x H100 SXM | by hand | 1 | Lambda $4.89/h | $4.89 |
| Kimi-K3 single-user parity | 8x A100 80 GB | Lambda 8x A100 80 GB | a copy of `research/kimi-k3-a100-2026-10-22b/go.sh` | 2.5 | Lambda $3.19 per GPU-h x 8 | $63.80 |
| DeepSeek-V4 single-user parity | 8x A100 80 GB (8x H100 if it does not fit) | Lambda 8x A100 80 GB | the same script, another bundle | 2.5 | Lambda $3.19 per GPU-h x 8 | $63.80 |
| GLM-5.3 single-user parity | 8x A100 80 GB | Lambda 8x A100 80 GB | the same script, another bundle | 2.5 | Lambda $3.19 per GPU-h x 8 | $63.80 |
| Every published pack since v0.29 still loads; `glyd bench` against the floor | RTX 4080 SUPER | glyd-pc | by hand | 1 | owned | $0 |

Total for one pass of M2: **$197.90**.

### M3 (v0.36.0)

Gate: A dated snapshot of the top 50 text-generation models runs; the 10 most-pulled local models run (including canonical GGUF quants); top-15 curated pre-packed repos hit first token with zero local pack wait (`P25`); Claude Code, Continue and Open WebUI pass tool-call checks; every installer channel passes; consumer benchmarks per VRAM tier.

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| Top-50 models, the ones up to about 32B | 1x H100 80 GB | Lambda 1x H100 SXM | by hand, a script per batch of models | 6 | Lambda $4.89/h | $29.34 |
| Top-50 models, 70B to 400B and the largest MoE | 8x H100 80 GB | Lambda 8x H100 SXM | by hand, a script like `go.sh` | 8 | Lambda $4.59 per GPU-h x 8 | $293.76 |
| 10 most-pulled local models, including GGUF, and consumer benchmarks per VRAM tier: 8 GB | RTX 3070 8 GB | Vast.ai 1x RTX 3070 | by hand | 3 | Vast median $0.115/h | $0.35 |
| The same, 12 GB | RTX 3060 12 GB | Vast.ai 1x RTX 3060 | by hand | 3 | Vast median $0.074/h | $0.22 |
| The same, 16 GB | RTX 4080 SUPER | glyd-pc | by hand | 3 | owned | $0 |
| The same, 24 GB | RTX 4090 24 GB | Vast.ai 1x RTX 4090 | by hand | 3 | Vast median $0.469/h | $1.41 |
| The same, 32 GB, and consumer Blackwell FP4 and FP8 (V7a) | RTX 5090 32 GB | Vast.ai 1x RTX 5090 | by hand | 3 | Vast median $0.607/h | $1.82 |
| Consumer Blackwell on 16 GB (V7a) | RTX 5080 16 GB | Vast.ai 1x RTX 5080 | by hand | 1.5 | Vast median $0.278/h | $0.42 |
| Installer channels: install.sh, winget, Homebrew, Docker | Linux, Windows, macOS, no GPU needed for the install itself | GitHub-hosted | workflow on release, automatic once written (P7c) | 0.5 | free in a public repository | $0 |

Total for one pass of M3: **$327.31**.

### M4 (v0.37.0 — Many GPUs, many machines)

Gate: Two consumer machines run a model neither holds alone at the predicted speed (+-10%); a 600B-class MoE on one 8x H100 node with experts in host RAM (EP decode within 15% of SGLang); tensor parallel on 2 GPUs at least 1.7x one GPU; one 8-GPU node serves the OpenAI API with DP replicas, metrics, compressed streaming cold start (`C7`), and <5 s sleep-to-RAM wake (`C8`).

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| Two consumer machines run a model neither holds alone, at the speed `glyd fit` predicts | 2 machines with one GPU each, a LAN between them | AWS 2x g6.xlarge in one placement group | `glyd cluster` on both, by hand | 3 | AWS 2 x $0.8048/h | $4.83 |
| GeForce multi-GPU without peer access (C2e) | 2x RTX 4090 in one box | Vast.ai 2x RTX 4090 offer | by hand | 2 | Vast 2 x median $0.469/h | $1.88 |
| Tensor parallel on 2 GPUs at least 1.7x one GPU | 2x H100 SXM (NVLink) | Lambda 2x H100 SXM | by hand | 1.5 | Lambda $4.79 per GPU-h x 2 | $14.37 |
| 600B-class MoE on 8x H100 with experts in host RAM; 8-GPU OpenAI server, streaming load (`C7`) and sleep-to-RAM (`C8`) | 8x H100 80 GB | Lambda 8x H100 SXM | by hand, a script like `go.sh` | 7 | Lambda $4.59 per GPU-h x 8 | $257.04 |

Total for one pass of M4: **$278.11**.

### M5a (v0.38.0 — Apple Silicon, CPU and desktop extras)

Gate: Apple M1 and M4 (Metal via `mlx-c` + Glyd Metal kernels) and CPU x86_64/arm64 pass KL thresholds against CUDA on the fitting M3 subset; macOS arm64 builds pass install checks; Vision (`F2a`), Embeddings (`F2b`), Batched LoRA (`F4`) and Speculative Decoding (`R6`) pass logits and `glyd bench` gates.

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| Apple M1 and M4: fitting M3 subset, KL against CUDA | Apple silicon | owner's machine | by hand | 4 | owned | $0 |
| CPU x86_64 and arm64 | no GPU | GitHub-hosted (x86_64 and arm64 Linux) | `ci.yml`, automatic | 0.5 | free in a public repository | $0 |
| Vision, Embeddings, Batched LoRA and Spec Decode gates | RTX 4080 SUPER + L4 24 GB | glyd-pc + AWS g6.xlarge | `gpu.yml` / `gpu-cloud.yml` | 2 | AWS $0.8048/h | $1.61 |

Total for one pass of M5a: **$1.61**.

### M5b (v0.39.0 — AMD ROCm and older NVIDIA)

Gate (runs in parallel as hardware/contributors become available; does not block M4, M5a or NVIDIA 1.0): Per device, the subset of the M3 set that fits passes KL thresholds against CUDA: T4 (`sm_75`), L4, RTX 4080 SUPER, A100, H100, DGX Spark; RX 7900 XTX, RX 9070 XT, Strix Halo, MI300X; `P7b` downloads signed backend bundles cleanly.

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| Per device: the M3 subset that fits, KL against CUDA: T4 (`V5`) | T4 16 GB | AWS g4dn.xlarge | `gpu-cloud.yml` or by hand | 2 | AWS $0.526/h | $1.05 |
| L4 (regression baseline) | L4 24 GB | AWS g6.xlarge | `gpu-cloud.yml` or by hand | 2 | AWS $0.8048/h | $1.61 |
| RTX 4080 SUPER | RTX 4080 SUPER | glyd-pc | `gpu.yml` | 2 | owned | $0 |
| A100 | A100 40 GB | Lambda 1x A100 SXM 40 GB | by hand | 2 | Lambda $1.99/h | $3.98 |
| H100 | H100 80 GB | Lambda 1x H100 SXM | by hand | 2 | Lambda $4.89/h | $9.78 |
| DGX Spark | GB10, aarch64 | owner's machine (none rentable) | by hand | 2 | gap: see below | $0 |
| RX 7900 XTX, RX 9070 XT, Strix Halo (`V1`) | RDNA 3, RDNA 4, RDNA 3.5 APU | owner's machine (none rentable) | by hand | 6 | gap: see below | $0 |
| MI300X (`V2`) | MI300X 192 GB | owner's machine or a provider outside these four | by hand | 2 | not priced: none of the four lists it | not priced |

Total for one pass of M5b: **$16.42**.

### M6 (v1.0.0)

Gate: Stable surface with CI compatibility checks; llm-d routes to Glyd pods by prefix-cache hits; a Helm install passes readiness on a GPU cluster; SBOM and provenance; a published comparison with vLLM and SGLang per quality tier on H100 (and B200 once `V8a` covers `sm_100` standard dense/FP8/NVFP4 GEMMs — never benchmark `mma.sync` against `tcgen05`).

| Run | Devices | Covered by | Started by | Hours | Rate | Cost |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: |
| Comparison with vLLM and SGLang, per quality tier, on H100 | 1x H100 80 GB | Lambda 1x H100 SXM | by hand, configs and logs kept | 8 | Lambda $4.89/h | $39.12 |
| The same on B200 (requires `V8a` `sm_100` kernels) | 1x B200 180 GB | Lambda 1x B200 | by hand, configs and logs kept | 8 | Lambda $7.99/h | $63.92 |
| Helm install passes readiness; llm-d routes by prefix-cache hit | 2 GPU nodes in one cluster | AWS 2x g6.xlarge (cluster control plane not priced here) | by hand | 4 | AWS 2 x $0.8048/h | $6.44 |
| Compatibility checks, SBOM and provenance | none (no GPU) | GitHub-hosted | `ci.yml` and release workflows, automatic | 0.5 | free in a public repository | $0 |

Total for one pass of M6: **$109.48**.

## Total per release gate

The two-machine rows of M4 and M6 use AWS L4 instances as a proxy for two consumer machines on a LAN.

| Gate | One pass | Budget (1.5x) | Largest row |
| :--- | ---: | ---: | :--- |
| M0, v0.33.0 | $4.54 | $6.81 | Three AWS GPU CI runs, $3.67 |
| M1, v0.34.0 | $5.00 | $7.50 | Two Windows runs on AWS, $4.36 |
| M2, v0.35.0 | $197.90 | $296.85 | Three 8x A100 runs, $191.40 |
| M3, v0.36.0 | $327.31 | $490.97 | 8x H100 for the large models, $293.76 |
| M4, v0.37.0 | $278.11 | $417.17 | One 8x H100 session, $257.04 |
| M5a, v0.38.0 | $1.61 | $2.42 | L4 Vision/LoRA/SpecDecode gate, $1.61 |
| M5b, v0.39.0 | $16.42 | $24.63 | One H100 parity baseline, $9.78 |
| M6, v1.0.0 | $109.48 | $164.22 | B200 comparison, $63.92 |
| All eight | $940.37 | $1,410.56 | |

The budget is one and a half passes, for a retry. Past runs lost whole attempts to causes unrelated to the code: a PyPI index
that was ten minutes stale, a run stopped to save credits
([benchmarks/gpu/big-moe-2026-10-06](../benchmarks/gpu/big-moe-2026-10-06/README.md)).

The large costs are the 8-GPU rows (M2, M3, M4). A download dominates them: on 2026-10-06 a 250 GB checkpoint took 874 s on a rented
host, about 0.29 GB/s ([same record](../benchmarks/gpu/big-moe-2026-10-06/README.md)). At that rate a 1.2 TB bf16 checkpoint takes about
70 minutes, which at $36.72 an hour is $43 before the first token. A persistent filesystem on the provider keeps the weights between runs and costs
less than the re-download; its price is not in this plan. Compressed streaming and peer-to-peer weight distribution (`C7` in M4) addresses this directly.

## Devices no rental covers

These need a machine the owner buys or borrows. The issue ([#219](https://github.com/surya-koritala/Glyd/issues/219)) lists them as
"own". This hardware gap is why AMD ROCm was split into M5b so it never blocks M4 (NVIDIA multi-GPU), M5a (Apple/CPU) or M6 (1.0).

| Device | Gate | Why no rental |
| :--- | :--- | :--- |
| DGX Spark | M5b | Not on Lambda, AWS or Vast.ai |
| RX 7900 XTX, RX 9070 XT | M5b | No Vast.ai offer on 2026-11-20; Lambda and AWS have none |
| Strix Halo | M5b | Not on any of the four |
| MI300X | M5b | No offer on Lambda, AWS or Vast.ai; needs a provider outside these four, not priced here |
| Windows 11 with an 8 GB or 16 GB GeForce | M1 | AWS Windows GPU instances have 16 GB (T4, refused until V5) or 24 GB; Windows VM offers on Vast.ai were not checked |
| A laptop GPU | M1 | Only desktop parts are rented; a desktop card with the same memory stands in |
| Apple M1 and M4, Metal | M5a | The owner's Mac covers one chip. AWS Mac Dedicated Hosts have a 24-hour minimum |
| GH200, GB200 | M1 (P18) | GH200 is on Lambda at $2.59 per hour; GB200 is not in the four |

## Automatic and manual

**Automatic CI** (nothing here starts a rented machine):

- `ci.yml` on every pull request and push to main: build, tests, lints, the engine's host tests. Windows and macOS compile and unit
  tests join it with S9 ([#213](https://github.com/surya-koritala/Glyd/issues/213)).
- `gpu.yml` on glyd-pc: pushes to main that touch `glyd-engine/`, `gpu/`, `glyd-gpu/` or the lockfile, and pull requests from this
  repository labeled `gpu` (never a fork). One GPU architecture (sm_89), one small model.
- `kernels.yml` compiles the kernels for sm_80 to sm_120 with no GPU; `starlings-ci.yml`, `tree.yml`, `dco.yml` and `install-check.yml`
  (weekly, no GPU).

**Manual release gates** (a person starts them, on the release candidate, before the tag):

- `gpu-cloud.yml`, by hand, for L4, A10G, L40S and T4.
- Every Lambda and Vast.ai run, and the 8-GPU runs above. No script runs them today beyond the launchers in `research/` and `scripts/`.
- The clean-machine install checks (M0, M1) and the Windows, Apple and AMD checks.

A manual gate becomes automatic only when it costs nothing (glyd-pc, GitHub-hosted runners) or when it is cheap and the account has a
cap (the L4 run, about $0.80). Everything over $5 stays manual and is started with the issue's go-ahead from the owner.

## Spend guards

Already in the repository:

- `gpu-cloud.yml`: the instance runs `shutdown -h +180` at start (a three-hour cap), a `stop` job runs `always()` after the tests,
  and every instance carries the tags `Project=glyd-ci` and `Run=<run id>`. The tests have a 90-minute timeout (`gpu-tests.yml`).
- `scripts/gpu_lambda.sh` terminates its instance when it exits and after `MAX_MIN` (75), then lists the account's instances.
  The `research/*/launch.sh` and `go.sh` scripts terminate at their cap (60 minutes; 2 h 25 min for the 8x A100 Kimi run) and start
  a second terminator that outlives the script (`go.sh`).
- `scripts/gpu_aws.sh` tags everything it creates `glyd-bench` and deletes it on exit.
- `tools/runpod/rp_guard.sh` and `rp_budget.sh` (Runpod: a per-pod age cap and a $100 daily cap that deletes `glyd-*` pods). Runpod
  is not one of the four providers here; the scripts are the model for the others.

To add before the first rented run of a gate:

1. **A per-run cap.** Every run sets `MAX_MIN` to 1.25 times the hours in its row, so its cost cannot pass 1.25 times the row's cost.
   A run whose row is over $50 starts only after the owner says yes in the gate's issue, naming the row and the cap.
2. **Terminate, not stop, on AWS.** `shutdown -h` stops an EBS-backed instance and leaves its volume billing unless the instance was
   launched with "terminate on shutdown". Check that the runner's launch setting is `terminate` before relying on the three-hour cap.
3. **A tag sweeper.** A scheduled job (`schedule`, every 30 minutes) that lists AWS instances tagged `Project=glyd-ci` older than four hours and
   terminates them, and lists Lambda instances named `glyd-*` older than the longest cap and reports them. This is the cap when a
   launcher dies.
4. **A name and a tag on everything:** `glyd-<gate>-<run>`, as `rp_guard.sh` already assumes.
5. **Account limits.** An AWS Budget with an alert at 50% and 100% of the month's plan (it needs the `Project` tag activated as a cost
   allocation tag to filter on it) and the vCPU quota for G and P instances set to what one gate needs, so a mistake cannot start more. Prepaid or credit balances on Lambda and Vast.ai
   kept at the next run's cap, with auto top-up off.
6. **One cheap dry run first.** A script new to a gate runs on glyd-pc or a $0.10 to $1 instance until it reaches its first number; only
   then is the expensive instance started. A launcher that finds no capacity polls and starts nothing until it has an instance
   (`go.sh` polls for up to 8 hours).
7. **Results leave the instance.** A run writes `DONE` and copies its results back before it terminates, so a retry does not repeat the run.
