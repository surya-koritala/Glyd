## Notes (by hand; everything above is generated from the runs)

- **35B-A3B MoEs on the H100** save 18.8%, under the inventory's projection for that GPU (23.4%); the gap is not explained yet.
- **Qwen3.5-35B-A3B on the H100 (results-sbh100)**: Glyd's server started (its memory is in the table) but the pod stopped before
  Glyd's bench ran, and gemma-4-26B-A4B was not reached; the tok/s dashes on that row are not a failed bench.
- **Not in the table, from tonight's v0.29 sweep (results-m29a, results-m29b)**: Glyd 0.29.0 did not start on gemma-4-12B,
  gemma-4-E4B, gemma-4-26B-A4B and gemma-4-31B (transformers 5.18's AmbiguousGlobalPerLayerAttributeError on a per-layer
  attribute), and refused NVIDIA-Nemotron-3-Nano-4B on an RTX 4090 with an estimate of 33.6 GiB packed for a model bf16 loads in
  7.47 GiB. The gemma-4-12B and Nemotron-3-Nano rows above come from the branch build of PR #21 (glyd 0.29+runfam).
- **Qwen3.5-122B-A10B on 4 x H100 (results-bigmoe-4xh100)**: the download finished and bf16's check server started; no weights,
  KV or Glyd numbers were written, so the model stays in the projected table.
- Rows marked "pod results ... (not a record yet)" come from tonight's pods; their records are not in benchmarks/gpu yet.
