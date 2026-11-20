# Benchmarks

The results the docs and the code cite: one directory per run, with its summary (`README.md`) and the scripts that made it.

| Path | What |
| :--- | :--- |
| [gpu/](gpu) | Engine and GPU runs, e.g. [h100x2-qwen2.5-72b-2026-10-03](gpu/h100x2-qwen2.5-72b-2026-10-03) |
| [weights/](weights) | Packing and quantization studies |
| [cpu/](cpu) | CPU expert kernels |
| [cards/](cards) | The numbers on the model cards |
| [moe-skew-2026-10-09/](moe-skew-2026-10-09) | Expert routing traces; the offload tests read them |

Raw outputs (logs, per-tensor JSON, CSV) are not kept in the tree. Those up to 2026-11-20, including the codec's
per-machine runs, are at commit [7d9e9be6](https://github.com/surya-koritala/Glyd/tree/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks); a path a
summary or a code comment names that is not here is there. New raw outputs go with the pull request that adds the summary
(as an attachment or release asset), not into git. CI refuses data files over 64 KB here and in `research/`
(`scripts/check_tree.sh`).

Reproduce the codec's per-machine runs:

```bash
AWS_PROFILE=you REGION=us-east-1 TYPES="c7g.2xlarge c7i.2xlarge" scripts/bench_aws.sh main
```
