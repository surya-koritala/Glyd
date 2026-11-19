# autotune-2026-10-06: the route-table tool; the sweep was not run

The library's routes (which kernel and plan multiplies a matrix for M tokens on a GPU) are hand-set per GPU class. This record holds
the tool that would derive route tables from a measured sweep (`tools/autotune/`) and a dry run of it on earlier, partial timings.

## Result

**Not measured: the sweep did not run.** Its four pods (RTX 4090, RTX 5090, H100, L40S) were stopped within about two minutes of
start, before any timing; no sweep result exists. Whether today's routes are the fastest is not known.

**Nothing changes.** `gpu/route_tables.h` is empty and the library does not include it (`glyd_gpu.cu` is unchanged); every route
stays as it is. Including the header is a later change, built with nvcc and checked on a GPU, once a sweep has rows that qualify.

The dry run: `tables.py` on the base rows of earlier `gpu/layouts.py` runs (below) gave no row. Those runs timed only today's route,
gemm (to 64 tokens), big with its own plan and decode, for four models on two GPUs; the routes, plans and GPUs they did not time are
listed below. They do not show that today's routes are the fastest. `summary.txt` is that dry run's output (a step's products by
today's routes; "after the table" equals "today" because the table is empty).

## The dry run's inputs, and what was not measured

- The dry run's inputs (earlier runs, not this sweep): the base rows (no development variable) of `gpu/layouts.py`'s runs at commit 2933f7ed (RTX 5090) and 3ed531a9 (H100
  80GB HBM3), whose routes and kernels are main's (`glyd_gpu.cu`'s routes and dense kernels unchanged since; main added the FP8
  code only). Routes timed there: today's, gemm (to 64 tokens where it is not today's), big with the prompt kernel's own plan (past
  64 tokens, not on Hopper), decode (past 64). 1018 and 922 timings, every product checked against fp32 and for the same bits twice,
  every pack unpacked bit for bit. Converted into `sweep-1120.json` and `sweep-7090.json` (`tables.py --record` derives this record
  again from them).
- Not measured: the RTX 4090 and the L40S; on the RTX 5090 and the H100 the mid route for the 12-bit layout, the wg route outside its
  range, the prompt kernel's plans 1, 2 and 3 apart, big at 1-64 tokens, the gemm route's variables (the 12-bit prefetch and batched
  exception loads, the tiered 8-warp kernel); granite-4.2-8b, Qwen3.5-35B-A3B's dense matrices and Qwen3.8-27B. `tools/autotune/job.sh`
  measures all of these (one pod a GPU, about 25 minutes; checks: the self-test, check_step.py, each config against fp32, twice for
  the same bits, the decode route bit for bit against bf16). Its four pods (RTX 4090, RTX 5090, H100, L40S) were started on
  2026-10-06 at 05:17 UTC and stopped within two minutes by the night's Runpod budget block; no sweep result came back.
- Out of this record's scope: the tiered layout's products of up to 64 tokens on compute capability 12 and on Hopper (another
  change, tiered-speed, measures them: the tables never change them); the experts' grouped products (moe); FP8's fused product
  (fp8-speed rewrites its plan); the route SPLIT (glyd.gpu's GLinear only).

## The rules (`tools/autotune/tables.py`)

Per GPU class, layout, size class of a matrix (its weights O K: `BINS`) and token count measured, each config's time summed over the
class's matrices. The fastest config the library can be told to take (a route, the prompt kernel's plan; never a development
variable) replaces today's route only where it takes at least 3% less time over the class and no matrix of the class takes more than
1% over today's time; a row spans the measured counts in a run with the same config, never past the last one. Kept: today's routes
that decode a prompt ahead (glyd.gpu overlaps that decode, which the sweep does not measure). The rows are written to `gpu/route_tables.h`
(not included by the library yet, see Result); a release reads no variable for them.

```sh
# on a pod (tools/runpod): the sweep, one GPU
tar -cf at_src.tar gpu glyd-gpu-py tools/autotune COMMIT; GPU="NVIDIA GeForce RTX 4090" MAX_MIN=55 NAME=at4090 UPLOAD="at_src.tar" bash tools/runpod/rp_run.sh tools/autotune/job.sh
# then, here: the tables, the header, this record
python3 tools/autotune/tables.py results-at4090/sweep.json [...] --record benchmarks/gpu/autotune-2026-10-06
python3 tools/autotune/test_tables.py
```
