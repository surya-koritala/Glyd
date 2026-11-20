# Labels

How issues and pull requests are labeled. The maintainer applies them when triaging; a new issue from a form starts with
`needs-triage`.

## What kind

| Label | Meaning |
| :--- | :--- |
| `bug` | Something fails or gives wrong output |
| `enhancement` | A new feature or a change to one |
| `performance` | Speed, memory or size is worse than it should be |
| `documentation` | The docs are wrong or missing something |
| `question` | A question that is not a bug or a request |
| `epic` | A tracking issue for a milestone (see [ROADMAP.md](../ROADMAP.md)) |

## Where in the code

| Label | Meaning |
| :--- | :--- |
| `area: api` | HTTP APIs: OpenAI, Anthropic, Ollama, MCP |
| `area: ci` | CI, runners, releases |
| `area: cli` | The `glyd` command line |
| `area: cluster` | Multi-GPU and multi-machine (Starlings) |
| `area: docs` | Documentation |
| `area: formats` | Weight and pack formats |
| `area: install` | Installers, packages, platforms |
| `area: kernels` | GPU and CPU kernels |
| `area: models` | Model families and descriptions |
| `area: observability` | Metrics, traces, health |
| `area: planner` | Adapting to model, hardware and platform |
| `area: python-port` | Porting or removing Python |
| `area: runtime` | Engine runtime |
| `area: scheduler` | Batching, KV cache, prefix cache |
| `area: security` | Security, supply chain, safe defaults |
| `condor` | Condor: a big MoE on one consumer GPU (`condor/`) |

## Which hardware

| Label | Meaning |
| :--- | :--- |
| `backend: cpu` | CPU backend and kernels |
| `backend: cuda` | NVIDIA CUDA |
| `backend: metal` | Apple Metal / MLX |
| `backend: rocm` | AMD ROCm/HIP |
| `backend: vulkan` | Portable fallback backend |
| `needs-gpu` | Needs real GPU hardware to build or test |
| `gpu` | On a pull request from this repository: run the GPU tests on the self-hosted runner (see [CONTRIBUTING.md](../CONTRIBUTING.md)) |

## State and who can help

| Label | Meaning |
| :--- | :--- |
| `needs-triage` | New; the maintainer has not looked at it yet |
| `needs-info` | Waiting for the reporter: a version, `glyd-engine doctor` output, a command |
| `good first issue` | Small, with a clear scope and the files to start in; fine for a first pull request |
| `help wanted` | The maintainer wants a contribution on this one |
| `duplicate`, `invalid`, `wontfix` | Closed with a reason in the comment |
| `dependencies`, `rust`, `python` | Pull requests that update a dependency file, Rust code or Python code |
| `accessibility` | A barrier for people with disabilities |

## Triage

The maintainer goes through `needs-triage` issues about once a week. Each one gets a kind, an area, and either a question
(`needs-info`), a closing label, or an answer. The maintainer closes an issue that has had `needs-info` for 30 days with no reply; it can be
reopened.
