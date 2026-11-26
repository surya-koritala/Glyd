# Checking a machine: the M2 gate

`benchmarks/gates/m2-gate.sh` checks the M2 gate on one GPU machine, an RTX 4080 SUPER or an L4: a pinned list of models serves 64 concurrent requests through one
scheduler, with its logits held to the CPU `Ref` backend, and the large families that no one GPU holds are refused well. This page says what each step checks. A test
(`scripts/check_gate_docs.py`) fails CI if the script has a step this page does not name.

```bash
benchmarks/gates/m2-gate.sh --gpu l4 --ext DIR                 # on the machine: the rows of the L4, the results other runs measured in DIR
benchmarks/gates/m2-gate.sh --plan-only --machine rtx4080s     # no GPU: the planner's answers against the model list
```

It prints one line a step (`PASS`, `FAIL`, `SKIP` or `INFO`), writes `results.tsv` and `m2-gate.json`, and exits 0 only where nothing failed, nothing was skipped
and every required step has a result. `--rows REGEX` runs a subset of the rows; a subset is not the gate. `--engine PATH` names the `glyd-engine` program of a
development build.

## The models

`benchmarks/gates/m2-models.txt` lists, for each GPU, the pinned models (`RUN`), the owner's alternatives (`ALT`) and the models no single GPU holds (`NONE`), each with
the levels and the contexts to check. A model that is refused still has to be refused well: the planner says so (exit 3) and the command that would run it gives a few
lines and a non-zero exit. A level marked `*` is one where the planner does not count 64 users.

## The steps

| Step | What it checks | Read more |
| :--- | :--- | :--- |
| `machine` | The GPU, the list of rows for it, the RAM and the OS, as facts for the record. The run stops if the GPU is not the one the rows are for. | [First run](first-run.md) |
| `plan:MODEL:LEVEL@CONTEXT` | `glyd fit MODEL:LEVEL --users 64 --context N` is a plan for 64 users at the level asked. A `NONE` row (`plan:MODEL`) is refused with exit 3 and the planner's first line is printed. | [Choosing a level](levels.md) |
| `serve-64:MODEL:LEVEL@CONTEXT` | `serve --users 64` comes up, and 64 streaming requests (prompts of half the context, 128 tokens each) sent at once all end with their tokens. `/v1/glyd/info` says 64 at once, `/stats` has 64 running at the peak, and no sequence was sent back for room. | [Serving](serve.md) |
| `plan-is-load:MODEL:LEVEL@CONTEXT` | What `/v1/glyd/info` says loaded is the plan: the level, the weights within 10%, the engine's memory within the plan's, and the memory the GPU lost since the plan was made within what the plan counts (the last only on this computer: not with `--machine`). | [Serving](serve.md) |
| `logits-vs-ref:MODEL:LEVEL` | The model's logits at 64 users against the CPU `Ref` backend. No engine command checks that on real weights yet, so this step is a `SKIP`, which keeps the gate red. | [Troubleshooting](troubleshooting.md) |
| `refusal:MODEL` | A model the list says no GPU of this kind holds is refused by the command: exit 3 and at most three lines. | [Troubleshooting](troubleshooting.md) |
| `alt:MODEL` | An `ALT` row is the owner's choice instead of its family's `RUN` row: it is checked by `--plan-only` and run only where `--rows` selects it. | [Choosing a level](levels.md) |

## Checks other runs measured

These come in as result files (`--ext DIR`, one `NAME.result` each, a line `PASS`, `FAIL` or `SKIP`, a tab, the name, a tab, the detail). A required one that is
missing fails the gate.

| Step | What it is | Read more |
| :--- | :--- | :--- |
| `parity-kimi-k3`, `parity-deepseek-v4`, `parity-glm-5.3` | Single-user parity of each large family on its reference hardware. | [Condor](condor.md) |
| `bench-no-regression` | `glyd bench --compare` of the gate's loads against the previous record on the same GPU: no regression beyond the measured band. | [benchmarks/](https://github.com/surya-koritala/Glyd/tree/main/benchmarks) |
| `packs-since-v0.29` | Every pack Glyd has published since v0.29 still loads to its recorded logits. With `--engine`, the gate runs the suite itself: `glyd-engine packcheck --gpu --engine`. | [Levels](levels.md) |
