# About this guide

Glyd packs LLM weights and the KV cache losslessly and decodes them inside the GPU kernels, and it has an inference engine
written in Rust that runs on them. You name a model and Glyd fits it to your GPU. This guide is for people who want to run
it.

It describes what is on the `main` branch of [surya-koritala/Glyd](https://github.com/surya-koritala/Glyd), the version
in `glyd-engine/Cargo.toml` and what has been merged since. Every command here was checked against the code on main.
Where a feature is still in a pull request, the page says what main does today and no more.

## What runs today

- NVIDIA Ampere or newer (RTX 30 series and up, A10, A100, L4, L40S, H100), Linux x86_64, driver 580 or newer.
- Lossless: Qwen3 dense, Gemma 4 12B and 31B, Nemotron-H, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark.
  Lossy only: Qwen3.5, 3.6 and 3.8 dense. The [Models](models.md) page has the full table and is generated from the code.
- A 180B mixture of experts on one 12-16 GB GPU: [Condor](condor.md).

Not yet: Windows, macOS, aarch64 and CPU-only engines. AMD is a development build. The engine serves one request at a time
except for Qwen3, which batches up to 8.

## Pages

1. [Install](install.md): what each machine gets today.
2. [First run](first-run.md): check the machine, see the plan, run a model.
3. [Choosing a level](levels.md): penguin, kestrel, swift, wren, hummingbird, with what each costs in quality and memory.
4. [Serving and the APIs](serve.md): the OpenAI and Anthropic endpoints, the key, the chat page.
5. [Connecting tools](tools.md): the settings, and which tools have been tried.
6. [Several GPUs and Starlings](multi-gpu.md).
7. [Condor](condor.md).
8. [Troubleshooting](troubleshooting.md), by what `glyd-engine doctor` prints.
9. [Models](models.md), generated from `glyd-engine/src/support.rs`.

## The command is `glyd-engine`

The engine's program is `glyd-engine`. Its messages say `glyd`, because the front-door program that will hold these
commands under that name is not on main yet. The commands on these pages work with `glyd-engine`. Where `glyd` (the file
compressor) is installed beside `glyd-engine` and the Python tool is not, `glyd run`, `serve`, `doctor`, `login`, `fetch`,
`preflight`, `prepare`, `supports`, `costs` and `rates` go to the engine. `fit` is `glyd-engine fit` only.

## Numbers

Every number has its machine and a link to the log. The records in `benchmarks/` carry the date and the build they were
made on. A number from an older build is marked as one.

## Versions

This guide at `/guide/` follows main. Each release tag that has the guide also gets its own copy at `/guide/vX.Y.Z/`, built from that tag's text.
