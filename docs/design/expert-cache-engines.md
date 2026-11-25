# The expert cache beside llama.cpp's and KTransformers' (R12, #232)

What each engine does with the experts of a mixture of experts that the GPU does not hold, read from their sources (llama.cpp at commit
0b51bb2f, 2026-11-22; KTransformers through SGLang's integration of it at de41ccfa, 2026-11-23), and what a same-bits, same-GPU comparison has to hold
fixed. The comparison itself is a measurement on the issue's GPU (phase 3); this note is what it compares.

## The three designs

| | Glyd (`offload.rs`, `devplan.rs`) | llama.cpp (`--cpu-moe`, `--n-cpu-moe N`, `--moe-cache-mib N`) | KTransformers (SGLang's `kt_ep_wrapper`) |
|---|---|---|---|
| Where every expert is | the computer's memory (pinned), or the disk past a budget | host buffers for the layers the flags name | the CPU's own quantized copy (`--kt-weight-path`, AMX int4 by default) for the experts past `kt_num_gpu_experts` |
| What the GPU holds | places a layer (LRU), filled as calls miss | with `--moe-cache-mib`: slots in banks shared by a group of layers | experts `0 .. kt_num_gpu_experts - 1` of each layer, fixed at load |
| Policy | least recently used; a call's experts asked for busiest last (`by_rows`); a miss the host computed twice within a window promoted behind the call | frequency classes (log2 of a use count decayed with a half-life of N tokens), LRU in each class, the victim the oldest by age / 2^class | none: the split is by expert id |
| A miss | copied, or computed on the host from the release's own bytes, split by measured times (`plan`: the longer of copies and host pairs shortest) | uploaded, or computed on the CPU: the misses of lowest use count up to `cpu_share` of them, which read a zero slot on the GPU | always the CPU's, for the whole batch |
| More misses than places | waves of copies (`waves_quicker`), or the host's | refused for that batch (no slots); large batches read the cache and do not fill it | n/a |
| Host waits a layer | one (the router's choices read back), none on the plan on the GPU (`--device-plan`: FP4 experts, the step one graph) | the scheduler splits the graph at the slot map and copies it with a callback that plans and uploads: one host step a layer | `submit_forward` then `sync_forward`: one wait a layer |
| Bytes read a step | each missed expert once for all its rows (`LayerStats`) | each uploaded expert once; consecutive experts uploaded together | each CPU expert once a batch |

## Same bits

- Glyd's copy mode multiplies over places with the resident layer's kernels: the same logits to the bit as every expert on the GPU
  (`tiercheck --mode copy`, DeepSeek-V4's and Kimi-K3's experts; GLM-5.3-Flash's FP8 and Muse Spark's lossless experts for a short
  prompt). With the host's half, the host's pairs are its own arithmetic over the same bytes (`tiercheck --mode auto|cpu` prints the
  largest difference).
- llama.cpp's cache uploads the GGUF's own blocks, so its uploaded experts are the all-GPU path's bits; its CPU misses are the CPU
  backend's products of the same blocks.
- KTransformers' CPU experts are a second, requantized copy: not the release's bits by construction.

A same-bits comparison therefore holds the weights fixed as GGUF blocks both engines read (a Glyd level whose experts are a GGUF
type, `gguf.rs`), compares copy-only runs first (all misses uploaded or copied: logits bit for bit against each engine's all-GPU
run), then each engine's host share, and only then times (phase 3: the step at 1, 8 and 32 rows on the 4080 SUPER, with each
engine's cache the same bytes).
