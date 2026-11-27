# Consumer benchmarks (P20)

Per release and per VRAM tier (8, 12, 16, 24 and 32 GB): the largest model each engine runs at 20 tokens/s or more for one user, and the same model at
about the same bits on Glyd, llama.cpp, Ollama and LM Studio, measured by one harness on one GPU. A release's results go in `vX.Y.Z/GPU/`: each run's
result (`NAME.json`, `glyd-bench/1`), its command lines and versions (`NAME.cmd`), its server's log (`NAME.log`, or with the release's assets where it is
over the tree's 64 KB), the sides that could not run (`NAME.skip`, with the reason) and the page made from them (`README.md`). The first release measured
this way is v0.35.0, on the 16 GB tier: [v0.35.0/rtx4080s](v0.35.0/rtx4080s/README.md) (the release's engine with the harness as it is here: the
release's own run stopped on two rows and had no llama.cpp side).

## How

```bash
GLYD_ENGINE=target/release/glyd-engine GGUF_DIR=~/gguf bash scripts/consumer_bench.sh benchmarks/consumer/v0.35.0/rtx4080s
```

- **The rows:** [matrix.tsv](matrix.tsv), the models of each tier (released within the last six months), the level Glyd runs each at and the llama.cpp
  quant of the same weights at about the same bits (wren 4.53 against Q4_K_M or, for Gemma 4's quantization-aware checkpoint, Q4_0 of that checkpoint;
  swift 5.53 against Q5_K_M; kestrel 6.53 against Q6_K). The script runs the rows of the GPU's own tier (`TIER` to choose). A row is a level
  `glyd fit` loads on a card of that tier; Glyd's server takes the steps its plan marks (`GLYD_DEGRADE=yes`: a mixture's experts in the computer's
  memory, as llama.cpp's `--fit on` puts experts on the CPU), and a run whose server loaded another level than the row's is not a result.
- **The same weights:** the GGUF is `GGUF_DIR/MODELNAME-QUANT.gguf`; with `CONVERT=1` the script makes a missing one from the Hub checkpoint with
  llama.cpp's own `convert_hf_to_gguf.py` and `llama-quantize` (llama.cpp at `LLAMA_CPP_REF`, 2026-11-22 by default). Ollama (`ollama create` from that
  file) and LM Studio (the file loaded by hand, `LMSTUDIO_URL`) run the same file. Bits a weight of a GGUF are its bytes over the parameters; Glyd's
  are its level's.
- **The measurement:** `glyd-engine bench` ([docs/bench.md](../../docs/bench.md)) against each server, the same prompts and loads (`short:1`, `short:4`,
  `4k:1`, 3 runs, 256 tokens a reply), greedy, thinking off. Glyd runs with `--no-draft` and llama.cpp with no draft model, so neither drafts. llama.cpp
  is started with `--fit on` (it sets its own GPU layers and experts on the CPU to fit the card) and 4 slots of 8,192 tokens each (`-c 32768 -np 4`: its
  context is shared by its slots); Ollama with 4 parallel requests of 8,192 tokens and flash attention. Peak GPU memory is sampled from `nvidia-smi` while each runs.
- **The page:** `glyd-engine bench --report DIR` (the script runs it last): per GPU, each engine's largest model at 20 tokens/s or more, then every
  model's runs side by side with the files that hold them.
- **Automated:** `.github/workflows/consumer-bench.yml` runs the script on the project's GPU machine when a release is published (or by hand) and keeps
  the results as the run's artifact; a pull request puts them here. The machine keeps the GGUF files between runs; a run by hand with `convert`
  makes the ones the rows lack (once a model and quant), so that a release's run has every side to measure.

## The datacenter occupancy check (1x H100)

`docs/ceiling.md` measured Glyd's one-token kernels at 92-96% of the RTX 4080 SUPER's memory rate and 57% on A100/H100. Each milestone, on one rented
H100, the engine's decode is measured at 1, 8, 32 and 64 at once and its share of the ceiling kept, and held to the previous check:

```bash
glyd-engine bench google/gemma-4-12B-it:wren --suite occupancy --users 64 --runs 3 --no-draft --out h100-$(date +%F).json
glyd-engine bench --compare benchmarks/consumer/occupancy/LAST.json h100-$(date +%F).json --history benchmarks/consumer/occupancy
```

`pct_of_ceiling` of each load is its decode tokens/s a user over the GPU's memory rate (the H100 SXM's spec sheet, 3,352 GB/s) divided by the weights'
bytes: at 1 at once the one-token kernels' share, at 64 the batched step's. The results go in `occupancy/` with the machine's description. None has
been measured yet.
