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

Windows 11 ships in v0.34.0 as a preview; its gate (`m1-gate.ps1`) runs on a cloud Windows GPU when one is available.
