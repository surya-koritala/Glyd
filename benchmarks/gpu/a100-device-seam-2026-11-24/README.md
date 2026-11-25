# The device and library traits before and after, A100 (2026-11-24)

R1a (#134) and R17 (#238) at c492163c: every driver call through `device::Device`, the GPU library and cuBLAS through
`gpu_ops::GpuOps` and `Gemm`, and the packed products' ways, the GPU's class and the KV cache's sizes computed in Rust (`fold.rs`), against
its base (63bdcdcb) on the same machine.

Machine: a rented Lambda NVIDIA A100-SXM4-40GB (driver 570.148.08, 30 vCPUs). Both trees built there by the GPU job's recipe (kernels for sm_80),
each in its own target directory.

**Result: no change from the traits and the fold. Every check gives the same verdicts on both trees, foldcheck finds no difference from the
library, and `bench --compare` says "within the noise band" for every pair.**

## foldcheck (the folded host side against the library's own)

- the GPU's code: 1 compared, 0 differ
- the packed products' ways (11 compute capabilities by 8 classes, with and without SPLIT, both layouts, 13 shapes, 185 token counts): 846,560 compared, 0 differ
- the KV cache's sizes and the decode's workspace: 33,503 compared, 0 differ

## The full regression, base and change

The GPU job's model-free checks (kvshapes, kcheck35, attncheck, kindcheck, seamcheck, bufcheck, kv12check, usercheck of six families (prompts one a pass and
together), roundcheck, keepcheck, logitscheck, levelcheck, twomodels, moecheck, tiercheck of four random models) on each tree: no verdict differs.
The lines that differ are kvshapes' statistics ("extra chunks at most", random per run) and tiercheck's model paths. kvshapes, kindcheck and
roundcheck muse exit non-zero on both trees alike (not from this change).

## The bench gate (`bench MODEL --suite gate --runs 3 --no-draft`), base and change alternated

| model | load | figure | base 1 | change 1 | base 2 | change 2 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| gemma-4-12B wren (bf16 KV) | 1k, 1 user | decode / prefill | 84.6 / 5,084 | 84.6 / 5,080 | 84.7 / 5,083 | 84.6 / 5,079 |
| | 4k, 1 user | decode / prefill | 83.5 / 4,439 | 83.5 / 4,435 | 83.5 / 4,439 | 83.5 / 4,439 |
| gemma-4-12B wren (12-bit KV) | 1k, 1 user | decode / prefill | 81.6 / 5,090 | 81.6 / 5,096 | | |
| | 4k, 1 user | decode / prefill | 81.6 / 4,741 | 81.6 / 4,737 | | |

The eight users' first token moves between runs (gemma: 0.105 and 0.083 s) for both builds alike: it is when the requests reach the round.
JSON: `base*-*.json`, `x*-*.json`.
