# The M1 gate, 2026-11-21

Three rented machines (Vast.ai), each a clean Ubuntu 24.04 with only the NVIDIA driver, one at a time, destroyed after the
run. The release candidate's Linux x86_64 tarball: the engine of release-gpu.yml run 44942490462 (main with #505: run and serve
load the level fit plans), its CUDA 12 library, and the CUDA 13 library of run 44934561638 (`gpu/` is the same in both; every file
checked against its sha256). Installed with `GLYD_VERSION=0.33.1 GLYD_ENGINE_TARBALL=... sh scripts/install.sh`. The engine reports
0.33.1 because the version is bumped at the release.

| Machine | GPU, driver | Result |
| :--- | :--- | :--- |
| `r535/` | RTX 3060 12 GB, 535.154.05 (CUDA 12.2, the oldest supported) | `p6-driver-check.sh`: PASS (doctor picks the CUDA 12 library, a first answer, serve) |
| `8gb/` | RTX 3060 Ti 8 GB, 595.91.07, 503 GiB RAM | 25 PASS, 1 FAIL (plan-is-load: the script, below) |
| `16gb/` | RTX A4000 16 GB, 580.178.04, 503 GiB RAM | 26 PASS, 2 FAIL (plan-is-load: the script, below) |

What ran, from `results.tsv`:

| Machine | Model | Plan | First token after the download | Download | JSON schema |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 8 GB | Nemotron 3.5 Lightning | kestrel, experts in RAM (3.5 GiB cache on the GPU) | 43 s | 87 s (25.8 GB) | valid |
| 16 GB | Gemma 4 12B-it | kestrel | 11 s | 60 s (10.6 GB) | valid |
| 16 GB | Nemotron 3.5 Lightning | kestrel, experts in RAM (11.3 GiB cache on the GPU) | 42 s | 113 s (25.8 GB) | valid |

Every other pinned model was refused in two lines with a non-zero exit (too big for the card, a family this engine does not run
yet, or a download larger than the disk), and nothing called Python or pip. Installs took 11 to 13 s.

**plan-is-load.** The three FAILs are the gate script's, not the engine's: it asked `glyd fit` for the plan's weights while the
server held the GPU, so fit saw the GPU full and printed no plan to read. Fixed in this pull request (the plan is read before the
server starts). The same check made from the server logs, which print the plan before loading:

| Machine | Model | Planned | Loaded | Difference |
| :--- | :--- | :--- | :--- | :--- |
| 8 GB | Lightning | 5.30 GiB | 5.16 GiB | -2.7% |
| 16 GB | Gemma 4 12B | 9.10 GiB | 9.38 GiB | +3.0% |
| 16 GB | Lightning | 13.10 GiB | 12.97 GiB | -1.0% |

All within the gate's 10%.

**The checks other runs measured** (`ext/`, read by `m1-gate.sh --ext`): driver-r535 PASS (`r535/`); p17-openai-sdk PASS (the client
matrix in the GPU job); the others SKIP, each saying where it was covered. The download is timed apart from the first token
because it is the connection's: the 2-minute limit is on loading and the first token.

## Speed (`bench/`)

`glyd bench --suite gate --runs 3 --no-draft --bandwidth-gbs 448` on the same machines, after the gate, the same release candidate.
Medians of three runs, 128 tokens each, no draft model; the JSON files have every run, the spread and the machine.

| GPU | Model, plan | One user, decode | 8 users, all of them | First token, short prompt | Prefill, 4k prompt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RTX A4000 16 GB | Gemma 4 12B-it, kestrel (9.1 GiB) | 38.0 tok/s (83% of the bandwidth ceiling, 46.1) | 220.7 tok/s (28.3 each) | 0.05 s | 1,688 tok/s |
| RTX A4000 16 GB | Nemotron 3.5 Lightning, kestrel, experts in RAM | 75.3 tok/s | 71.0 tok/s (one at a time) | 0.16 s | 2,199 tok/s |
| RTX 3060 Ti 8 GB | Nemotron 3.5 Lightning, kestrel, experts in RAM | 27.2 tok/s | 27.2 tok/s (one at a time) | 0.73 s | 1,016 tok/s |

The ceiling is the GPU's memory bandwidth over the bytes a token reads. For Lightning it is printed against all the weights on
the GPU rather than the 3B a token uses, so its percentage (above 100%) means nothing; with experts in RAM its speed is set by
the cache's hits and the PCIe link. Lightning serves eight users one after the other: the dynamic families' steps do not batch
across users yet (the 8-user cell's first token, 6 to 17 s, is the queue).

## Drivers (`p6/`)

`p6-driver-check.sh`, the release candidate on two more rented machines:

| Machine | GPU, driver | Result |
| :--- | :--- | :--- |
| `p6/ada-r565/` | RTX 4060 Ti 16 GB, 565.57.01 (CUDA 12.7) | PASS: doctor picks the CUDA 12 library, a first answer, serve |
| `p6/blackwell-r595/` | RTX 5060 Ti 16 GB, 595.112.07 (CUDA 13.2) | FAIL: `cuLaunchKernel: too many resources requested for launch (701)`, with either library |

The Blackwell failure was the engine's: on sm_120 the compiler gave `g4_add_norm` 80 registers a thread (53 on sm_89), and it is
launched with 1,024 threads, so the block needed 81,920 of the SM's 65,536. Fixed in this pull request with `__launch_bounds__(1024)`
(64 registers, 40 bytes spilled; `moe_logits`, at 72, gets its 256 too), and kernels.yml now fails on any kernel past 64 registers
without launch bounds, on all six architectures (`scripts/check_kernel_regs.py`). The release candidate after it is checked on
an RTX 5060 Ti again before v0.34.0.

Windows 11 ships in v0.34.0 as a preview; its gate (`m1-gate.ps1`) runs on a cloud Windows GPU when one is available.
