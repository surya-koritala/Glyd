<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/glyd-mark-white.svg">
    <img src="docs/img/glyd-mark.svg" alt="Glyd" width="56">
  </picture>
</p>

<h1 align="center">Glyd</h1>

<p align="center"><strong>Fast, compressed inference for open LLMs, from a laptop GPU to multi-GPU servers.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/glyd/"><img alt="PyPI" src="https://img.shields.io/pypi/v/glyd?label=PyPI&color=orange"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a>
  <a href="https://pypistats.org/packages/glyd"><img alt="Downloads" src="https://img.shields.io/pypi/dm/glyd?label=Downloads&color=blue"></a>
  <a href="https://huggingface.co/glyd"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/Models-Hugging_Face-yellow"></a>
  <a href="https://github.com/sponsors/surya-koritala"><img alt="Sponsor" src="https://img.shields.io/badge/Sponsor-GitHub_Sponsors-ea4aaa?logo=githubsponsors&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://surya-koritala.github.io/Glyd/guide/">Docs</a> ·
  <a href="https://huggingface.co/glyd">Models</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="https://github.com/surya-koritala/Glyd/discussions">Discussions</a> ·
  <a href="https://github.com/surya-koritala/Glyd/releases">Releases</a> ·
  <a href="https://github.com/sponsors/surya-koritala">Sponsor</a>
</p>

Glyd is an open-source inference engine for large language models, written in Rust with its own CUDA kernels. It
compresses a model's weights, losslessly or at a lossy level you choose, plans the model to the GPUs, RAM
and disk you have, and serves it with OpenAI-, Anthropic- and Ollama-compatible APIs. One program, no Python.

## Get started

Linux (x86_64 or aarch64) with an NVIDIA GPU and its driver:

```bash
curl -LsSf https://getglyd.com/install.sh | sh
```

Windows x86_64 (preview), in PowerShell:

```powershell
irm https://getglyd.com/install.ps1 | iex
```

Then:

```bash
glyd fit google/gemma-4-12B-it      # the plan for this machine: level, context, where each part lives
glyd run google/gemma-4-12B-it      # chat in the terminal and at http://localhost:8000
glyd serve google/gemma-4-12B-it    # the API for your tools; glyd connect says how to point them at it
```

`glyd run` loads the plan `glyd fit` prints. Pre-packed models download from [huggingface.co/glyd](https://huggingface.co/glyd);
any other model is packed on your machine once. The [guide](https://surya-koritala.github.io/Glyd/guide/) covers installs, levels,
serving, tools and troubleshooting.

## Compression levels

| Level | Bits per weight | Lossless | What it is |
| :--- | :--- | :--- | :--- |
| **penguin** | about 10.8 | yes | the release's weights, bit for bit (an FP8 or FP4 release is kept as released) |
| **kestrel** | 6.5 | no | 6-bit codes and a scale for every 32 weights |
| **swift** | 5.5 | no | 5-bit codes |
| **wren** | 4.5 | no | 4-bit codes; exact on a quantization-aware checkpoint (Gemma 4 QAT) |
| **hummingbird** | 2 to 3 on experts | no | mixture-of-experts layers: the busiest experts at 3 bits, the rest at 2 |

The planner picks the best level that fits: penguin where the model fits, else the highest lossy level that does. Ask for
one with a tag: `glyd run google/gemma-4-12B-it:wren`. The KV cache is bf16 pages; `GLYD_KV=lossless` asks a server for 12-bit lossless pages where the plan finds them no bigger.

## Supported hardware

| Vendor | System | GPUs | Status |
| :--- | :--- | :--- | :--- |
| NVIDIA | Linux&nbsp;x86_64 | Ampere to Blackwell: RTX 30, 40 and 50 series, RTX A-series and Ada, A100, L4, L40S, H100, H200, B200 | Supported |
| NVIDIA | Linux&nbsp;aarch64 | GH200, GB200, DGX Spark, Jetson Thor | Built, not&nbsp;measured |
| NVIDIA | Windows&nbsp;x86_64 | The same GPUs as Linux x86_64 | Preview |
| AMD | Linux&nbsp;x86_64 | RDNA 3 and 4 | Development&nbsp;build |
| Apple | macOS | M-series Macs | [Planned](ROADMAP.md) |
| None (CPU&nbsp;only) | x86_64, arm64 | | [Planned](ROADMAP.md) |

NVIDIA GPUs need driver R535 or newer (R570 for Blackwell). The AMD build is made from source with `GLYD_HIP=1`. `glyd doctor` checks a
machine and says what to fix.

| Setup | What runs | Status |
| :--- | :--- | :--- |
| Several GPUs, one machine | Layers split or shared across the GPUs; mixture-of-experts experts in RAM and on NVMe | Supported |
| Several machines | One model across computers with [Starlings](starlings/), becoming `glyd cluster` | Beta |

## Supported model families

Glyd supports model families, not individual models: any size or fine-tune of these families runs, from the Hub or from 16-bit GGUF files.

| Family | Levels |
| :--- | :--- |
| Gemma | penguin, kestrel, swift, wren |
| Qwen | penguin, kestrel, swift, wren, hummingbird |
| Llama | penguin, kestrel, swift, wren |
| Mistral | penguin, kestrel, swift, wren |
| Nemotron | penguin, kestrel, swift, wren, hummingbird |
| Kimi | penguin, hummingbird |
| GLM | penguin, hummingbird |
| DeepSeek | penguin, hummingbird |
| Muse | penguin, hummingbird |
| Domyn Edge | penguin, kestrel, swift, wren |
| Granite | penguin, kestrel, swift, wren |
| Mellum | kestrel, swift, wren |
| Aion | penguin, kestrel, swift, wren |

The engine tells a family by a model's `config.json`, not by its name.

**Planned families** ([roadmap](ROADMAP.md)): AliceAI, Beam, Cosmos, dots, ERNIE, Flint, Hunyuan, Inkling, LFM, Ling, MiMo, MiniMax, Mistral Large,
Unlimited OCR. Glyd follows the families released in the last six months. A family joins the table when the
engine runs it.

## Serving and tools

| Area | What Glyd has |
| :--- | :--- |
| APIs | OpenAI (Chat Completions, Responses), Anthropic Messages, Ollama |
| Structured output | JSON schema, JSON mode and forced tool calls, constrained on the GPU |
| Tools | Open WebUI, Continue, Claude Code, the OpenAI, Anthropic and Ollama SDKs, LangChain (`glyd connect`) |
| Background service | loads a model on the first request, unloads it when idle (`glyd service`) |
| Operations | health and readiness, Prometheus metrics, OpenTelemetry traces, draining on shutdown |
| Security | listens on localhost by default, API keys, Host and Origin checks; signed releases |

## File compression

Glyd began as a lossless compressor for the files that fill object storage, and still ships it: `glyd compress FILE -o OUT`,
`glyd decompress`, a store that compresses across objects (`glyd-store`), and the `glyd` crate. It is separate from the
engine: [docs/codec/README.md](docs/codec/README.md).

## Development and contributing

Contributions are welcome, from bug fixes and docs to model support and kernels.

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd
cargo build --release -p glyd --features engine   # glyd with the engine linked; its kernels need the CUDA toolkit (12.8 or newer)
cargo test -p glyd-engine                          # host-side tests, no GPU needed
```

1. Fork the repository and branch from `main`. For larger changes, open an issue or a
   [discussion](https://github.com/surya-koritala/Glyd/discussions) first.
2. Add tests with the change. Commits carry a DCO sign-off (`git commit -s`).
3. Open a pull request saying what changed and how you tested it.

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/surya-koritala/Glyd/labels/good%20first%20issue). Design notes:
[docs/architecture.md](docs/architecture.md).

## Community

- **Questions and ideas:** [GitHub Discussions](https://github.com/surya-koritala/Glyd/discussions).
- **Plans:** the [roadmap](ROADMAP.md), with one GitHub milestone per release.
- **Security reports:** [SECURITY.md](SECURITY.md).
- **Governance and the name:** [GOVERNANCE.md](GOVERNANCE.md).
- **Support the work:** [GitHub Sponsors](https://github.com/sponsors/surya-koritala) funds the GPUs Glyd is measured and tested on.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Releases v0.29.0 to v0.32.1 were published under BUSL-1.1.
