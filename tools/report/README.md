# tools/report

The internal results report: per 2026 model and GPU, Glyd as measured on pods against bf16 as **estimated** (bf16 is not run).
Standard library only; reads the pods' results and the records (no GPU). CI runs `--selftest`.

```sh
python3 tools/report/report.py --runs benchmarks/gpu "$GLYD_RUNS" \
  --out research/report-2026-10-06/README.md --json research/report-2026-10-06/measured.json
python3 tools/report/report.py --selftest
# facts.json again from the checkpoint headers research/inventory-2026/fetch.py read:
python3 tools/report/report.py --headers DIR_OF_HEADERS --out /dev/null
```

- **Reader**: every vLLM server log under `--runs` (`Model loading took`, `Available KV cache memory`, `GPU KV cache size`,
  `Maximum concurrency`, the attention block size, the settings from `non-default args` and glyd serve's
  `gpu-memory-utilization=` line, Glyd's layout line), the `vllm bench serve` results beside it (`c<N>.json`, `*-rateinf.json`:
  output tokens/s and TTFT) and `speed1.py` results (`speed-*.json`, one user). The GPU from the nearest `gpu.txt` /
  `machine.txt`; Glyd's version from a `v0NN` directory or the nearest `env.txt` / `versions.txt` / `install.out` / `job.log`
  (the main table takes Glyd 0.29 only). A Hub cache path as the model is mapped to `Org/Name`. 2026 models only (those in
  `facts.json`); the models skipped are printed. `--json` writes every run it read, sources relative to the results root.
- **Where the same pod measured bf16** the report shows that measurement (marked `meas.`) instead of the estimate.
- **bf16 estimator** (at a Glyd run's settings): weights = the bytes vLLM loads (`inventory.json`) + 0.13 GiB of vLLM buffers;
  KV room = Glyd's KV memory - (bf16's weights - Glyd's weights); users at a context = the run's logged concurrency converted
  by one request's KV cost (a hybrid's in vLLM blocks, its recurrent state's blocks counted; else `fit.kv_bytes`), bf16's
  scaled by its KV room; output tokens/s from a step model (weights read once, a MoE's experts as the batch touches them,
  each user's KV and recurrent state, an empirical saturation term, the prompts' compute, the batch capped by the users the
  KV room runs at once and `--max-num-seqs`), four constants fitted on every bf16 point measured on a 2026 model. The report
  prints the in-sample error and the **held-out** error (each run predicted without it) by GPU and kind, the users estimate
  against every bf16 run on a pod with a Glyd 0.29 run, the run-to-run noise of bf16's KV memory, and the weights estimate.
- `research/report-2026-10-06/facts.json`: per model what decoding reads (dense bytes, routed experts, active parameters,
  KV config, recurrent state), from the safetensors headers.

Estimates are for internal reports only; published numbers are measured, bf16 and Glyd from the same run.
