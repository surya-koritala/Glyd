# Benchmarks

Per-machine results produced by `scripts/bench_aws.sh` (one directory per
instance type: `machine.txt`, `quick3.txt`, `v7_bench.txt`, `ultra_bench.txt`,
`field_survey.txt`, `multicore.txt`), plus the local Apple M1 Max runs
quoted in the top-level README (`m1-max/`: the ultra-level run; the other
M1 numbers are in `CHANGELOG-BENCH.md`). Every file is the harness's own output;
every Glyd number is paired with the reference library measured in the
same process.

Reproduce on your own account:

```bash
AWS_PROFILE=you REGION=us-east-1 TYPES="c7g.2xlarge c7i.2xlarge" scripts/bench_aws.sh main
```

GPU results are in [gpu/](gpu), a directory for each run with its README and
logs; for example [h100x2-qwen2.5-72b-2026-10-03](gpu/h100x2-qwen2.5-72b-2026-10-03),
Qwen2.5-72B over two H100s in vLLM, bf16 against Glyd.
