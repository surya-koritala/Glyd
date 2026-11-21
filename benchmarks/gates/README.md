# Milestone gates

`m1-gate.sh` (and `m1-gate.ps1`, its PowerShell twin for Windows, not yet run on one) runs the M1 gate of [ROADMAP.md](../../ROADMAP.md) on one clean machine; `m1-models.txt` is the list of pinned models and what each machine tier is expected to do with it.

```bash
benchmarks/gates/m1-gate.sh [--tier 8gb|16gb|24gb] [--models FILE] [--glyd PATH] [--install-cmd CMD | --skip-install] [--out DIR] [--ext DIR]
benchmarks/gates/m1-gate.sh --plan-only --machine rtx4060-laptop|rtx4080s|rtx4090 [--glyd PATH]
```

One line a step (`PASS`, `FAIL`, `SKIP`), a summary, `results.tsv` and `m1-gate.json` in `--out`, exit status 0 only where nothing failed. The steps: the machine and its tier; the install; for each model the planner's answer against
the file's expectation (`run` or `refuse` for the tier), then either the refusal (a few lines, a non-zero exit) or the first streamed token (answer or reasoning) within 120 s of the model's download ending (the download is timed apart and reported beside it: it is the connection's), the printed plan's level and weights against what
`/v1/glyd/info` says loaded (10%), and JSON-schema output checked with `jq`; that nothing called `python` or `pip` (both are shimmed for the run); on a machine without an NVIDIA driver, the one-line answer. The checks other
runs measured come in through `--ext DIR` as `NAME.result` files holding one line, `PASS|FAIL|SKIP`, a tab, the name, a tab, the detail: `driver-r535`, `driver-r550`, `driver-r570`, `p2a-no-oom`,
`r5a-schema-real-weights`, `r8-reference-gpu-results`, `p17-open-webui`, `p17-continue`, `p17-openai-sdk`. A required one that is missing fails the gate.

`--plan-only` needs no GPU: it asks `glyd fit MODEL --machine PRESET` and compares with the file, which is how the expectations were made and how they are kept true (run it with the release build: the wren level is a development build's).

`p6-driver-check.sh --bundle DIR` is P6's acceptance on one driver (R535, R550, R570: the unpacked engine bundle on a machine with only the driver: `--version`, `doctor` naming the CUDA 12 library, a first answer from `glyd/gemma-4-12B-wren`, the CUDA 12 cross-check, `serve`, and the too-old message below R535); it writes `driver-rNNN.result` for `m1-gate.sh --ext`.

## The model list

Only models released in the last six months are used (the owner's rule: `createdAt` on the Hub). Of the ten pinned, the engine's planner runs 3 on a 16 GB card, 4 on 24 GB and 1 on 8 GB (Nemotron-3.5-Lightning at wren, experts in RAM; release build); the others get its refusal
on every tier, which the gate checks as a result of its own: why a model does not run is part of the product. The list is changed with the engine: a model that starts to run on a tier changes its column and the pull request says so.
