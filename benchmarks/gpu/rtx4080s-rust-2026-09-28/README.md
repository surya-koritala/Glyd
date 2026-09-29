# rust-gpu on an RTX 4080 SUPER, 2026-09-28

The measurements and checks the rust-gpu branch's docs quote: `glyd pack` and `glyd verify` (the glyd-gpu command),
the 12-bit layout's load, generate() against main, check_capi. The box: an RTX 4080 SUPER and a Ryzen 9 7950X3D (16
cores, 32 threads), CUDA 13, shared with other jobs. Every step ran under the box's lock once its GPU was idle, one at
a time, on at most 8 threads.

`run.sh` ran the branch's tree (70fa411 but for three changes made before that commit and after the run began, none
on the paths timed: verify's check of a merged pack's member names, which reads names alone; model.groups taking its
groups from format.GROUPS, the same two; tests and text) against main's (origin/main 151d146: its package and its
library, built in the same run). Each step's log starts with a line of the box's state as it began: the time, the load
averages, the three busiest processes by ps's %CPU (a process's average over its life). Every round is kept.

| log | what |
| :--- | :--- |
| build.txt | the branch's and main's libraries (gpu/build_lib.sh), the crate and CLI, the crate's tests |
| selftest.txt, check_capi.txt, test_gpu.txt, check_api-dense.txt | the checks: the self-test, check_capi (the JIT build against the library, bit for bit), test_gpu.py, check_api on Qwen3-0.6B and 1.7B |
| pack-python.txt | `python -m glyd.gpu pack` of each model in each layout (on the GPU): the reference |
| pack.txt | `glyd-gpu pack`, three rounds a model and layout on 8 threads: time, GB/s of bf16, peak RSS, every file's sha256 against Python's |
| verify.txt | `glyd-gpu verify` of each save: three rounds on the CPU (8 threads), one on the GPU |
| load.txt | `from_pretrained(layout="mma12")` (load_time.py) from the bf16 checkpoint, the tiered save and the 12-bit save; three rounds, fresh processes, warm page cache |
| unpack_c.txt | gpu/examples/unpack.c against the branch's library, on the saved Qwen3-0.6B |
| generate.txt | generate() eager (gen.py: the best of three runs a process, 128 tokens), main's package and library and the branch's in turn, the order swapped each round; four rounds |
| tiny.txt | tiny random checkpoints of each family packed by Python and by glyd-gpu (built from 70fa411), both layouts, every file's sha256; each save verified by both (tiny.py, tiny_job.sh) |
| test_gpu-70fa411.txt | test_gpu.py on 70fa411's package |
| main-reads-new-save.txt | main's package (0.23.0) loading and verifying a save that carries glyd.json's "tensors" (main-reads-new-save.sh) |

## Rounds with another process busy

A step is marked where another process showed more than 20% CPU as it began. Nothing is left out of the ranges the
docs quote. The process was GNOME's file indexer, `localsearch-ext`, each time (once with another job's CUDA compiler
beside it):

- pack.txt, 1 of 30 steps: round 1 of Qwen3-4B-Instruct-2507 in the 12-bit layout (2.88 GB/s; rounds 2-3: 2.91,
  2.87).
- verify.txt, 3 of 40: round 1 of Qwen3-4B-Instruct-2507 12-bit on the CPU (2.82 s; 2.83, 2.84), of Qwen3-8B tiered
  (9.14 s; 9.14, 9.19) and of Qwen3-8B 12-bit (7.34 s; 7.35, 7.46).
- load.txt, 2 of 27: round 3 of Qwen3-4B-Instruct-2507 from the 12-bit save, with cicc (99.8%) too (0.84 s; 0.85,
  0.84); round 1 of Qwen3-8B from the bf16 checkpoint (3.61 s; 3.54, 3.55).
- generate.txt, 30 of 32 steps, main's and the branch's alike: all but round 1's main Qwen3-1.7B tiered and round 3's
  main Qwen3-4B-Instruct-2507 12-bit.
- pack-python.txt (the reference, not quoted for speed), 1 of 10: Qwen3-8B in the 12-bit layout; test_gpu.txt (a
  check).

## merge-v0.24.0

rust-gpu merged with v0.24.0 (main 44393a9) and its routes ported, checked on the same box as the merge commit has it
but for text (CHANGELOG.md, a doc comment of lib.rs): `run.sh` built the library and the crate, then ran the crate's
tests with the library and the GPU, the self-test, check_capi (6975 calls through both hosts bit for bit and 220044 routes
as 0.24.0's rule, which its log sums as 227019 calls; check_capi prints them apart since), test_gpu.py, and check_api on
Qwen3-0.6B and Qwen3-1.7B and on granite-3.1-3b-a800m-instruct, one at a time under the box's lock; each exits 0.

## review-3

Review 3's fixes, checked on the same box: `run.sh` built the library, then ran check_capi (6975 calls through both
hosts bit for bit, 220044 routes as 0.24.0's rule) and test_gpu.py (test_route_env and the rest), one at a time under
the box's lock, each exiting 0; `route_env.txt`: glyd.gpu.model's import refusing GLYD_WG_MAX set to 1e3, nothing and
2k, and taking 512.

