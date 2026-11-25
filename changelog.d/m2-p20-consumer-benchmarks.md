- Consumer benchmarks for every release (P20): `scripts/consumer_bench.sh` measures the models of a GPU's memory tier (8, 12, 16, 24 or 32 GB,
  `benchmarks/consumer/matrix.tsv`) on Glyd, llama.cpp, Ollama and LM Studio on the same GPU with `glyd bench`: the same weights at about the same
  bits (llama.cpp's GGUF made from the same checkpoint, the same file for Ollama and LM Studio), no draft model on either side, each run's command
  lines, versions and server log kept beside its result. `glyd bench --report DIR` writes the comparison page: per GPU each engine's largest model at
  20 tokens/s or more, and each model's runs side by side. Results record `--engine`, `--source`, `--bits` and `--params`. The `Consumer bench`
  workflow runs it on the project's GPU machine when a release is published. `glyd bench --suite occupancy` (1, 8, 32 and 64 at once) is the 1x H100
  occupancy check; the H100, H200 and the tiers' consumer cards are in the bench's spec-sheet bandwidths. No release has been measured this way yet. Part of #242.
