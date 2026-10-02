# glyd pack against Python's, on the AWS dev machine, 2026-09-29

v0.25.0's `glyd pack` (the glyd-gpu crate) against `python -m glyd.gpu pack` in both layouts, and the release tree's
self-test and xcheck. The machine: the AWS dev machine, a g6.4xlarge with an NVIDIA
L4 (Ada, sm_89, 23 GB, 72 W, driver 595.91.07), 16 vCPUs (8 cores of an AMD EPYC 7R13) and 60 GB, CUDA 13,
PyTorch 2.14.0, transformers 5.17.0. The tree: rust-splitbyte at 089102e (C API 5), with main at db8e7b0 beside it
for xcheck (the 12-bit layout before v0.25.0, C API 4).

`run.sh` ran every step under the machine's lock, one at a time, once the GPU was idle (another job shares the
machine). Each log's steps start with a line of the machine's state: the time, the load averages, and the three
busiest processes by ps's %CPU. No other process was busy as a step began. `summary.txt` counts each log's steps by
their exit and its byte-identical and differing saves.

| log | what |
| :--- | :--- |
| build.txt, build-*.txt | the release library (gpu/build_lib.sh: every architecture), main's and two patched libraries for sm_89 alone, the crate (its binary, examples and tests) |
| selftest.txt | the self-test (gpu/glyd_gpu.py) with the release library: its products against fp32, every bf16 bit pattern unpacked in both layouts |
| xcheck.txt | xcheck.py, synthetic: the 12-bit layout against main's (before v0.25.0) in every entry point the L4 takes, 3205 calls the same bits and 486 refused by both |
| xcheck-geforce.txt | the same through both trees built to treat the L4 as a GeForce Ada card (its code 1089): GeForce Ada's prompt paths, which no other GPU takes, 3205 calls the same bits |
| rust-tests.txt | the crate's tests with the library and the GPU |
| test_gpu.txt | test_gpu.py |
| pack-python.txt | `python -m glyd.gpu pack` of each model in each layout (on the GPU): the reference, its files' sha256 |
| pack.txt | `glyd-gpu pack`, three rounds a model and layout on 16 threads: time, GB/s of bf16, peak RSS, every file's sha256 against Python's |
| verify.txt | `glyd-gpu verify` of each Rust save, three rounds on the CPU (16 threads) and one on the GPU (`--device cuda:0`), and `python -m glyd.gpu verify` of it |
| examples.txt | the crate's examples on Qwen3-0.6B's save in the smallest layout (`mma`): unpack.rs (a pack, a merged one; the checkpoint's bits) and linear.rs (a product by each of the library's paths, bit for bit its kernel's) |
| tiny.txt | tiny.py's checkpoints (Llama tied and not, Qwen2, Mistral, Granite, Qwen3 with a hidden size no multiple of 128 and one untied, GraniteMoe) packed by both in both layouts, every file's sha256; each Rust save verified by both |

## Results

Every step exited 0.

- Byte for byte: all 30 rounds of Qwen3-0.6B, 1.7B, 4B-Instruct-2507 and 8B and granite-3.1-3b-a800m-instruct
  (glyd-v3's only mixture of experts here), `mma` and 12-bit, every file's sha256 Python's. The same for all 16
  tiny saves.
- Verified: every Rust save, by `glyd-gpu verify` on the CPU and on the GPU and by `python -m glyd.gpu verify`.

`glyd-gpu pack` on 16 threads, three rounds, GB/s of bf16 (`mma` / 12-bit):

| model | bf16 | `mma` | 12-bit | peak RSS |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1.19 GB | 1.55 | 1.64-1.67 | 1.1 / 1.2 GB |
| Qwen3-1.7B | 3.44 GB | 1.55-1.58 | 1.88-1.91 | 3.1-3.2 / 3.3-3.4 GB |
| Qwen3-4B-Instruct-2507 | 8.04 GB | 1.58-1.60 | 2.01-2.03 | 6.6 / 7.3-7.4 GB |
| Qwen3-8B | 16.38 GB | 0.71-0.88 | 0.67-0.72 | 11.4-11.6 / 13.2-13.6 GB |
| granite-3.1-3b-a800m-instruct | 6.75 GB | 1.44-1.45 | 1.78-1.80 | 5.6-5.7 / 6.0-6.1 GB |

Qwen3-8B's rounds took the disk's pace. Its three shards were written with the kernel's writeback busy (kworker
flush in the state lines), and the pack used 3.2-6.4 of the 16 CPUs (320-635% in pack.txt). The smaller models used
more.

`glyd-gpu verify` of Qwen3-8B (253 packed tensors and 146 saved as they are): 11.6-11.8 s `mma` and 8.8-9.0 s
12-bit on the CPU (16 threads, 2.5-2.7 GB peak RSS), 29.9 s and 26.6 s on the GPU.
