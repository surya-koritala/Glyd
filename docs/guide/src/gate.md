# Checking a machine: the M1 gate

`benchmarks/gates/m1-gate.sh` checks one clean machine end to end: install, first token, the plan, the refusals, the tools.
`m1-gate.ps1` is its PowerShell twin for Windows. This page says what each step checks. A test
(`scripts/check_gate_docs.py`) fails CI if the script has a step this page does not name.

```bash
benchmarks/gates/m1-gate.sh --tier 16gb                       # on the machine: GPU tier 8gb, 16gb or 24gb
benchmarks/gates/m1-gate.sh --plan-only --machine rtx4080s    # no GPU: the planner's answers against the model list
```

It prints one line a step (`PASS`, `FAIL` or `SKIP`), writes `results.tsv` and `m1-gate.json`, and exits 0 only where nothing
failed and every required step has a result.

## The models

`benchmarks/gates/m1-models.txt` lists the pinned models and, for each machine tier, whether the engine runs it (`run`) or
refuses it (`refuse`). A model that is refused still has to be refused well: a few lines and a non-zero exit. Only models
released in the last six months are on the list.

## The steps

| Step | What it checks | Read more |
| :--- | :--- | :--- |
| `machine` | The GPU, its tier, the RAM and the OS, as facts for the record. | [First run](first-run.md) |
| `install` | The install command runs, timed (`curl -LsSf https://getglyd.com/install.sh \| sh`). | [Install](install.md) |
| `vendor-one-line` | On a machine with no NVIDIA driver (Apple, AMD), `run` answers in one or two lines and exits non-zero. | [Install](install.md), [Troubleshooting](troubleshooting.md) |
| `plan:MODEL` | The planner's answer (`run` or `refuse`) is what `m1-models.txt` expects for the tier. | [First run](first-run.md) |
| `refusal:MODEL` | A model the planner refuses gives a refusal of at most three lines and a non-zero exit. | [Troubleshooting](troubleshooting.md) |
| `first-token:MODEL` | `serve` starts and the first streamed token arrives within 120 seconds of starting it, the download and packing included. | [First run](first-run.md), [Serving](serve.md) |
| `plan-is-load:MODEL` | The level the plan printed is the level that loaded, and the weights are within 10% of the plan, read from `/v1/glyd/info`. | [First run](first-run.md), [Serving](serve.md) |
| `json-schema:MODEL` | A request with a JSON schema returns an object that is valid against it. | [Serving](serve.md) |
| `no-python` | Nothing called `python` or `pip` during the run: both are shimmed to fail. | [Install](install.md) |

## Checks other runs measured

These come in as result files (`--ext DIR`, one `NAME.result` each). A required one that is missing fails the gate.

| Step | What it is | Read more |
| :--- | :--- | :--- |
| `driver-r535`, `driver-r550`, `driver-r570` | The engine bundle on a machine with only that NVIDIA driver (the CUDA 12 build): version, `doctor`, a first answer, `serve`, and the message for a driver that is too old. `p6-driver-check.sh` makes them. | [Troubleshooting](troubleshooting.md) |
| `p2a-no-oom` | For each model of the list on the reference GPU, `run` does not run out of memory at the planned context, and the plan matches what loaded (`cargo xtask ci-plan-check`). | [First run](first-run.md) |
| `r5a-schema-real-weights` | JSON-schema output is valid on real weights. | [Serving](serve.md) |
| `r8-reference-gpu-results` | The reference results of `glyd bench` on the reference GPU. | [benchmarks/](https://github.com/surya-koritala/Glyd/tree/main/benchmarks) |
| `p17-open-webui`, `p17-continue`, `p17-openai-sdk` | Open WebUI, Continue and an OpenAI SDK client connect with default settings. | [Connecting tools](tools.md) |

The Windows twin is `m1-gate.ps1`. It has not been run on a Windows machine yet.
