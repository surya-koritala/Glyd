<h1 align="center">Glyd</h1>
<h3 align="center">Lossless AI compression: 33% less GPU memory, bit for bit.</h3>

<p align="center">
<a href="https://github.com/surya-koritala/Glyd/actions"><img alt="CI" src="https://github.com/surya-koritala/Glyd/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/surya-koritala/Glyd/releases"><img alt="Release" src="https://img.shields.io/github/v/release/surya-koritala/Glyd?label=release"></a>
<a href="#license"><img alt="License: BUSL-1.1" src="https://img.shields.io/badge/license-BUSL--1.1-blue.svg"></a>
</p>

Glyd keeps a bf16 model's weights and KV cache in fewer bits in GPU memory and rebuilds the exact values
as the model runs. It is the same model in about a third less memory, so it fits a smaller GPU or serves
more requests on the same one.

## Run it

```bash
curl -LsSf https://getglyd.com/install.sh | sh
glyd run Qwen/Qwen3.5-9B                       # settings for your GPU, then a chat in the terminal
glyd run Qwen/Qwen3.8-27B:swift                # fewer bits, not lossless: :kestrel 6.5 bits per weight, :swift 5.5
```

With vLLM, or in Python:

```bash
pip install "glyd[vllm]"
vllm serve Qwen/Qwen3.5-9B --quantization glyd
```
```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3.5-9B")
```

Linux (x86_64 or aarch64) with an NVIDIA GPU: Ampere, Ada, Hopper or Blackwell, driver 580 or newer.

## Results

| Measured | bf16 | Glyd |
| :--- | ---: | ---: |
| Qwen3.5-9B under vLLM on an RTX 4090: weights | 17.66 GiB | **12.68 GiB** (−28.2%) |
| Qwen3.5-9B under vLLM on an RTX 5090: weights | 17.66 GiB | **13.89 GiB** (−21.3%) |
| Qwen3.5-9B on an RTX 4090 (24 GB): context | out of memory at start (even at 4,096 tokens) | **189,440 tokens** |
| Qwen3.8-27B under vLLM on an H100: weights | 50.22 GiB | **38.77 GiB** (−22.8%) |
| Qwen3.8-27B: its layers' matrices, read back bit for bit | 49.52 GB | **33.28 GB** (−32.8%) |

The freed memory becomes KV cache. Logs: [rtx4090-5090-qwen35-9b-v029-2026-10-06](benchmarks/gpu/rtx4090-5090-qwen35-9b-v029-2026-10-06)
(Qwen3.5-9B, vLLM 0.30.0; [CHANGELOG v0.29.0](CHANGELOG.md#v0290--2026-10-06)),
[h100-qwen38-v028-2026-10-03](benchmarks/gpu/h100-qwen38-v028-2026-10-03) and
[open-models-a10-2026-09-27](benchmarks/gpu/open-models-a10-2026-09-27) (`sizes.txt`). Every result:
[getglyd.com/benchmarks](https://getglyd.com/benchmarks/).

**What lossless means here:** every weight, and every key and value in the KV cache, reads back as the
exact bf16 value. Outputs can still differ from bf16's in the last bits, because Glyd's kernels add the
same products in a different order, as any two GPU kernels do. `exact=True` gives bf16's logits exactly.

## The codec

Underneath the GPU work, Glyd is a lossless compression library and CLI in Rust with a C ABI: a drop-in
alternative to LZ4, Snappy and zstd that also turns logs and table dumps into typed columns (`-r`) and
compresses a new version of a file against the old one (`--base`).

```bash
brew install surya-koritala/glyd/glyd          # or a release's tarball
glyd --max events.json -o events.glyd          # fewer bytes than zstd -3, faster reads
glyd -d events.glyd -o events.json
```

## Docs

- [getglyd.com/docs](https://getglyd.com/docs/): getting started, vLLM, the GPU package, the codec.
- [DETAILS.md](DETAILS.md): everything on one page, with the results and their logs
  ([as text for LLMs](https://getglyd.com/llms-full.txt)).
- [CHANGELOG.md](CHANGELOG.md), [the roadmap](https://getglyd.com/roadmap/), [SECURITY.md](SECURITY.md).

## License

From v0.29.0 all of Glyd is under the [Business Source License 1.1](LICENSE): the `glyd` crate and CLI, the C ABI, the
Python and Go bindings, the store (`glyd-store`) and the GPU package (`glyd-gpu`).

Free forever for personal and non-commercial use (including education, research and nonprofits) on computers you own or rent for yourself, and for anyone to try, test and develop with. Commercial use needs a license: suryakoritala@getglyd.com. Each version becomes Apache-2.0 four years after its release.

Releases up to v0.28 keep the licenses they shipped with, in those releases' tags: the codec (the `glyd` crate, CLI, C ABI
and bindings) under BSD-3-Clause OR GPL-2.0, the store and the GPU package under BUSL-1.1. Code in
[third_party/](third_party/) keeps its own license.
