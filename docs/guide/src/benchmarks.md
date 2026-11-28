# Benchmarks against other engines

Each release is measured against llama.cpp, Ollama and LM Studio on the same GPU, per VRAM tier (8, 12, 16, 24 and 32 GB): the largest model each
engine runs at 20 tokens a second or more for one user, and the same model at about the same bits on each. One harness measures every engine
(`glyd bench`, with the same prompts and loads), and every result keeps its command lines, versions and server log.

- The method, the models of each tier and how to run it on your own GPU:
  [benchmarks/consumer](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/consumer).
- A release's results and its comparison page: `benchmarks/consumer/vX.Y.Z/` there. No release has been measured this way yet.
- Quality at equal bits (Glyd's levels against GGUF, AWQ and FP8 of the same model, as KL from the lossless level and small task sets):
  [docs/quality.md](https://github.com/surya-koritala/Glyd/blob/main/docs/quality.md).

To measure your own GPU:

```bash
GGUF_DIR=~/gguf bash scripts/consumer_bench.sh results/
```

It runs the models of your GPU's tier on Glyd and on every other engine it finds, and writes `results/README.md`.
