# Consumer benchmarks, NVIDIA GeForce RTX 4080 SUPER (16 GB)

Every figure is the median of the runs of `glyd-engine bench` ([docs/bench.md](https://github.com/surya-koritala/Glyd/blob/main/docs/bench.md)) against each engine's server on the same GPU, greedy, thinking off, no draft model; the command lines, versions and server logs are in the files named beside each row. Nothing here is estimated.

## NVIDIA GeForce RTX 4080 SUPER (16 GB tier)

Driver 595.99.02, 32 threads of AMD Ryzen 9 7950X3D 16-Core Processor; loads short:1, short:4, 4k:1, 3 runs each, at most 256 tokens a reply.

### The largest model at 20 tokens/s or more (one user, short prompt)

| engine | model | parameters | level or quant | decode tokens/s |
| :-- | :-- | --: | :-- | --: |
| glyd | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 32.9B | wren | 126.1 |
| llama.cpp | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 32.9B | Q4_K_M | 85.1 |

### google/gemma-4-12B-it

| engine | level or quant | bits a weight | decode tokens/s, one user | in all, most at once | first token, 4K prompt (s) | peak GPU memory (MiB) | failed requests | files |
| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |
| glyd | kestrel | 6.53 | 65.6 | 250.0 (x4) | 1.077 | 14631 | 0 | glyd-gemma-4-12B-it-kestrel.json, glyd-gemma-4-12B-it-kestrel.cmd, glyd-gemma-4-12B-it-kestrel.log |
| llama.cpp | Q6_K | 6.52 | 61.7 | 212.5 (x4) | 1.120 | 12501 | 0 | llama.cpp-gemma-4-12B-it-Q6_K.json, llama.cpp-gemma-4-12B-it-Q6_K.cmd, llama.cpp-gemma-4-12B-it-Q6_K.log |
| llama.cpp | Q4_0 | 4.65 | 82.1 | 287.7 (x4) | 0.999 | 9821 | 0 | llama.cpp-gemma-4-12B-it-Q4_0.json, llama.cpp-gemma-4-12B-it-Q4_0.cmd, llama.cpp-gemma-4-12B-it-Q4_0.log |
| glyd | wren | 4.53 | 90.5 | 341.1 (x4) | 1.085 | 11767 | 0 | glyd-gemma-4-12B-it-wren.json, glyd-gemma-4-12B-it-wren.cmd, glyd-gemma-4-12B-it-wren.log |

### nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

| engine | level or quant | bits a weight | decode tokens/s, one user | in all, most at once | first token, 4K prompt (s) | peak GPU memory (MiB) | failed requests | files |
| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |
| llama.cpp | Q4_K_M | 6.18 | 85.1 | 109.5 (x4) | 3.110 | 14683 | 0 | llama.cpp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-Q4_K_M.json, llama.cpp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-Q4_K_M.cmd, llama.cpp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-Q4_K_M.log |
| glyd | wren | 4.53 | 126.1 | 195.4 (x4) | 0.747 | 14475 | 0 | glyd-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-wren.json, glyd-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-wren.cmd, glyd-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-wren.log |
