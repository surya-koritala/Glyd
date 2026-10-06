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
glyd run Qwen/Qwen3-8B                         # settings for your GPU, then a chat in the terminal
```

With vLLM, or in Python:

```bash
pip install "glyd[vllm]"
vllm serve Qwen/Qwen3-8B --quantization glyd
```
```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3-8B")
```

Linux (x86_64 or aarch64) with an NVIDIA GPU: Ampere, Ada or Hopper, driver 580 or newer.

## Results

| Measured | bf16 | Glyd |
| :--- | ---: | ---: |
| Qwen3-8B under vLLM on an L4: weights | 15.27 GiB | **10.38 GiB** (−32.0%) |
| Qwen3-32B: 48 GB GPUs it needs | 2 | **1** |
| Qwen2.5-72B on two H100s: requests a second, at full load | 0.88 | **3.59** (4.1x) |
| Qwen3-30B-A3B on an H100: requests a second, at full load | 12.32 | **17.10** (1.39x) |

The freed memory becomes KV cache, which is where the extra requests come from. Logs:
[vllm-v028-l4-2026-10-04](benchmarks/gpu/vllm-v028-l4-2026-10-04),
[lambda-gpu_4x_a6000-20260926-084757](benchmarks/gpu/lambda-gpu_4x_a6000-20260926-084757),
[h100x2-qwen2.5-72b-2026-10-03](benchmarks/gpu/h100x2-qwen2.5-72b-2026-10-03),
[h100-moe-v028-2026-10-03](benchmarks/gpu/h100-moe-v028-2026-10-03). Every result, and where Glyd is
still slower than bf16: [getglyd.com/benchmarks](https://getglyd.com/benchmarks/).

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
- [DETAILS.md](DETAILS.md): everything on one page, with every result and its log
  ([as text for LLMs](https://getglyd.com/llms-full.txt)).
- [CHANGELOG.md](CHANGELOG.md), [the roadmap](https://getglyd.com/roadmap/), [SECURITY.md](SECURITY.md).

## License

From v0.29.0 all of Glyd is under the [Business Source License 1.1](LICENSE): the `glyd` crate and CLI, the C ABI, the
Python and Go bindings, the store (`glyd-store`) and the GPU package (`glyd-gpu`).

Free forever for personal and non-commercial use (including education, research and nonprofits) on computers you own or rent for yourself, and for anyone to try, test and develop with. Commercial use needs a license: suryakoritala@getglyd.com. Each version becomes Apache-2.0 four years after its release.

Releases up to v0.28 keep the licenses they shipped with, in those releases' tags: the codec (the `glyd` crate, CLI, C ABI
and bindings) under BSD-3-Clause OR GPL-2.0, the store and the GPU package under BUSL-1.1. Code in
[third_party/](third_party/) keeps its own license.
