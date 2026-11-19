# 0.29.4's GPU check: Swift, the level tags and best-fit scales on an A100 (2026-10-09)

Private record. One A100-SXM4-80GB (Runpod pod `gq5lunwnwtm262`, driver 595.91.07, $1.79/hour, 15:19-15:39 UTC: about $0.60), started
by `tools/runpod/rp_run.sh`; branch `int5-swift` at e7aa3d89 (Int5, the level tags, best-fit scales for Int6 and Int5). Job: `job.sh`
(library by `gpu/build_lib.sh` with CUDA 12.8, the engine's release build, the crate's tests, examples/int5 and int6, then Qwen/Qwen3.8-27B
at 1d4bf0f2: bench and runs). Every step in `steps.txt` ended as it should (`tag-wren`'s status 1 is the refusal asked for).

## Builds and exactness

* `build.log`: the engine and the examples build; `cargo test --release -p glyd-gpu`: 32 passed, 0 failed.
* `int5-check.log`, `int6-check.log`: the GPU's decode of four matrices is `unpack_int5`'s / `unpack_int6`'s bit for bit, with the packers'
  best-fit scales; products from 33 to 512 tokens bit for bit against the tiered layout's of the same weights. "all checks passed".

## Qwen3.8-27B, one user (`bench --prompt 1024 --new 256`, CUDA graphs)

| On this pod | decode, ms a token | tokens a second | prompt of 1,024, ms | weights, GiB | in use, GiB | loaded in, s |
| :-- | --: | --: | --: | --: | --: | --: |
| Swift (`--layout i5`) | 21.039 | 47.53 | 417.7 | 17.351 | 19.504 | 157.3 |
| Kestrel (`--layout i6`) | 21.578 | 46.34 | 420.1 | 20.479 | 22.551 | 126.6 |

Decode and memory as on the earlier A100 pod (benchmarks/gpu/int5-a100-2026-10-09: 47.78 and 46.62 tokens a second). Loading takes
longer: 157 s and 127 s here against 51 s and 49 s with the old scale rule. The best-fit search tries 16 scales for every run of 32 as
the model loads (the bf16 checkpoint is packed at every start): 3.1 and 2.6 times the old rule's load times (those were on another
A100-SXM4 pod, whose CPU may differ). A pre-packed download (0.30.0) removes the packing from the start.

## The level tags (`run-*.log`, `tag-wren.log`)

* `run ~/v/bf16:swift`: "about 17.3 GiB on the GPU at the swift level (about 5.5 bits per weight, not lossless)", ready in 2m37s; asked
  for the capital of Australia, its reasoning has Canberra, chosen as a compromise between Sydney and Melbourne (the reply ends at the
  run's token limit while still reasoning: the run had no `--no-think`).
* `run ~/v/bf16:kestrel --no-think`: about 20.5 GiB at the kestrel level, ready in 2m05s; "101, 103, 107".
* `run ~/v/bf16 --lossy --no-think`: the same as kestrel; 17 x 23 = 391.
* `run ~/v/bf16:wren`: status 1, "glyd: :wren is not built yet. The levels available now: penguin, kestrel, swift".
