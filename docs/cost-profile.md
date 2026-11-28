# Cost profiles: glyd-costs/1

What a model at a level holds on each layer, what a decoded token reads, and a prediction of decode speed from the machine's
memory tiers. For `glyd run`'s planner, the fit page and Glyd-Compute's scheduler. The numbers come from the
packed directory's own files, not from a formula: config.json, glyd.json and the safetensors headers, pack by pack, which are the
bytes the loader uploads. A formula is used only where the files are not there (a model not packed yet), and the document says so.

```
glyd-engine costs MODEL[:level] [--ctx N] [--prompt N] [--new N] [--chunk N] [--no-gpu]   # one JSON document on stdout (--no-gpu: the GPU not asked)
glyd-engine costs MODEL[:level] --predict [--rates tiers.json] [--placement NAME] [--gpu-experts F] [--hit F] [--ram-experts F]
glyd-engine costs DIR --load                                                       # a Qwen3.5-family model loaded: its allocation beside the profile
glyd-engine preflight MODEL:level --costs                                          # the same document, where glyd run asks before a download
glyd-engine rates [--dir DIR] [--mib N] [--reps N] [--no-gpu]                      # this machine's tiers, measured
curl -s -H "Authorization: Bearer $GLYD_API_KEY" http://127.0.0.1:8000/v1/glyd/info | jq .costs    # while serving
```

MODEL is a directory, a Hub name with a level tag (`Qwen/Qwen3.5-9B:linnet`: the pre-packed repo `glyd/Qwen3.5-9B-linnet`, from the
cache where it is there, else the Hub by ranges: its small files and each shard's header, nothing more), or a pre-packed repo's own
name. Families described: Qwen3.5 (9B, 27B; Qwen3.6, Qwen3.8) and Nemotron-H (Nemotron 3 Super). Gemma 4 is refused for now.
The record: [benchmarks/gpu/rtx4080s-cost-profile-2026-10-20](../benchmarks/gpu/rtx4080s-cost-profile-2026-10-20/README.md).

## The profile

Bytes are bytes (not GiB); a value that is not known is `null`, never 0. Three kinds of number are kept apart by `basis`: sizes
(`file`: read from the safetensors headers; `estimated`: the shape times the layout's bits a weight, src/plan.rs; `mixed`), the
working memory (`measured`, from the loaded engine's counters, else `unavailable`), and timing (`estimated` from the predictor,
or null).

| field | what | unit |
| :--- | :--- | :--- |
| `object`, `schema` | `glyd.costs`, `glyd-costs/1` | |
| `build.version`, `build.commit`, `build.development` | the engine, its source commit (GLYD_SOURCE_COMMIT; null where the build was given none), whether a development build | |
| `model.id` | MODEL as typed | |
| `model.repo`, `model.revision` | the Hub repo and commit, from the cache's layout (`models--ORG--NAME/snapshots/REV`) or the Hub's listing; null for a plain directory | |
| `model.family`, `model.format`, `model.layout` | `Qwen3.5` or `Nemotron-H`; glyd.json's `format` (`glyd-v7`, `glyd-v4`) and the layout word (`i6`, `i5`, `i4`, `hb`, `mma`) | |
| `model.source`, `model.made_by` | glyd.json's: the model the level was made from, at its revision, and the Glyd that packed it | |
| `model.hidden`, `model.vocab`, `model.max_context`, `model.layers` | the config's | |
| `level.requested`, `level.effective`, `level.lossless` | the tag asked for (null: none), the level the files are at (glyd.json's `level`, else by its layout), whether lossless | |
| `basis.sizes`, `basis.packs_from_files`, `basis.packs_estimated` | where the sizes come from, and how many packs each way | |
| `basis.kv_and_state`, `basis.working_memory`, `basis.note` | the KV and state are the config's formula, which the engine allocates to the byte (checked on the record); the working memory is measured or unavailable | |
| `precision.weights`, `precision.bits_per_weight` | the level and layout, and the packed matrices' bytes times 8 over their weights | bits |
| `precision.activations`, `sums`, `kv`, `state` | bf16, fp32, bf16, and the family's states (Qwen3.5: fp32 delta rule, bf16 convolution; Nemotron-H: fp32 scan, bf16 convolution) | |
| `device` | the GPU the engine would use: `ordinal`, `name`, `compute_capability`, `cuda` (the driver's), `driver` (nvidia-smi's version), `total_bytes`, `free_bytes` (now, every process's use in it); null without a GPU | bytes |
| `host` | `cpu` (the model name), `threads`, `ram_total_bytes`, `ram_available_bytes` (MemAvailable now) | bytes |
| `workload` | `context_tokens` (`--ctx`, 4096), `prompt_tokens` (1024), `output_tokens` (256), `prefill_chunk_tokens` (`--chunk`, 2048), `sessions` (1: the profile is of one sequence) | tokens |
| `observed_at`, `samples` | Unix seconds; the sample count of the timing fields (0: none measured) | s |
| `layers[i].index`, `layers[i].kind` | the layer; `full_attention`, `linear_attention` (Gated DeltaNet), `mamba` (Mamba-2) or `moe` | |
| `layers[i].resident_bytes.projections` | the mixer's matrices: attention's q/k/v and o, linear attention's and Mamba-2's in and out projections, a mixture layer's latent projections (Nemotron's fc1/fc2), the small dense parts (Qwen's b and a) | bytes |
| `layers[i].resident_bytes.norms` | the input and post-attention norms, q and k norms (Nemotron: the layer's one norm) | bytes |
| `layers[i].resident_bytes.mlp` | a dense MLP (gate, up, down) | bytes |
| `layers[i].resident_bytes.experts`, `experts_count`, `expert_mean`, `expert_min`, `expert_max` | every routed expert's two matrices, together and one (the mean, the smallest, the largest: the Hummingbird level holds a layer's experts in two sizes) | bytes |
| `layers[i].resident_bytes.shared_experts`, `router` | the shared expert's matrices; the router's weight and its score-correction bias | bytes |
| `layers[i].resident_bytes.recurrent` | a recurrence's parameters: the convolution's weight and bias, A_log, D, dt_bias, the gated norm | bytes |
| `layers[i].resident_bytes.other`, `total` | a tensor of no class above (the Hummingbird experts' rotation signs); the sum | bytes |
| `layers[i].top_k` | the experts a token takes of the layer (0: dense) | |
| `layers[i].active_bytes_per_token` | what a decoded token reads of the layer: everything dense, plus `top_k` experts of the mean size (a Hummingbird layer's most-used experts are its 3-bit ones, so the true mean of what is read is a little above the plain mean) | bytes |
| `layers[i].kv_bytes_per_token` | a token's keys and values in the layer (full attention: 2 heads x head width x 2 bytes; 0 elsewhere) | bytes |
| `layers[i].state_bytes_per_session` | the state a sequence carries in the layer (a linear-attention or Mamba-2 layer's recurrent state and convolution window; zeroed at each request) | bytes |
| `non_layer.embeddings_bytes`, `final_norm_bytes`, `head_bytes`, `head_tied`, `other_bytes` | outside the layers; `head_tied` says the head is the embedding (then `head_bytes` is 0) | bytes |
| `non_layer.active_bytes_per_token` | the head (the embedding where tied), one row of the embedding, the final norm | bytes |
| `totals.resident_bytes`, `resident_layer_bytes`, `resident_expert_bytes` | the weights on the GPU when the whole model is there; the layers' share; the routed experts' share | bytes |
| `totals.active_bytes_per_token`, `active_expert_bytes_per_token`, `active_dense_bytes_per_token` | what a token reads of the weights; the experts' share of it; the rest | bytes |
| `totals.kv_bytes_per_token`, `kv_bytes_at_context` | all layers; times `workload.context_tokens` (the engine allocates the whole context at load) | bytes |
| `totals.state_bytes_per_session` | all layers (one session) | bytes |
| `totals.working_bytes` | the activations, the products' workspaces, the logits, the rotary tables: measured (`allocated`) or null | bytes |
| `totals.overhead_bytes` | the CUDA context, the kernels' code, the driver's rounding: null (not counted by the engine; 338 MiB in the serve record on the 4080 SUPER) | bytes |
| `totals.engine_bytes_at_context` | resident + KV at the context + state (+ the working memory where measured) | bytes |
| `totals.bf16_bytes`, `totals.weights` | the same tensors in bf16, and the packed matrices' weight count | bytes, weights |
| `allocated` | with `--load` or while serving: the engine's own counters (`weights_bytes`, `kv_bytes`, `kv_bytes_per_token`, `state_bytes`, `working_bytes`, `engine_bytes`, `context_tokens`), the measured side of the profile; null otherwise | bytes |
| `excluded[]` | `what` (vision tower, MTP head, draft model), `bytes_in_files` (their bytes where the files have them; a packed directory has none), `note` | bytes |
| `timing.prefill_tok_s`, `timing.decode_tok_s` | null, or `{value, basis, placement}`: `--predict` fills `decode_tok_s` with the first prediction (`basis: estimated`). A measurement is recorded beside the profile (the record's bench output), not inside it | tokens/s |
| `predictions[]`, `tiers` | with `--predict`: below | |

## Tiers

`glyd-engine rates` measures this machine, each tier the way the engine uses it, and prints one document: `rates.NAME` = `{gbs, bytes,
seconds, samples, how}` (GB/s = 1e9 bytes a second; `samples` passes over a buffer of `buffer_bytes`, 512 MiB unless `--mib`).

| rate | how |
| :--- | :--- |
| `vram_read` | a kernel that reads the buffer and sums it: the ceiling a memory-bound decode step has |
| `vram_copy` | cuMemcpyDtoD: bytes copied a second (read and written both) |
| `vram_product` | with `--dir` a glyd-v7 directory in a block layout: the engine's own product at one row over the first four layers' packs, back to back (no cache hits): the rate a decode step reads weights at. The predictor uses it for the GPU's memory where it is there, else `vram_read` |
| `link_h2d_pinned`, `link_d2h_pinned` | cuMemcpyHtoD from pinned host memory (what the expert cache's copies use), and back |
| `link_h2d_pageable` | the same from pageable memory (the driver stages it) |
| `ram_read`, `ram_read_1_thread` | every thread streaming its share of a buffer of u64 and summing it; one thread alone |
| `ssd_read_16`, `ssd_read_1` | with `--dir`: random 2 MiB spans of the directory's safetensors files, 16 at a time and one at a time, by the SSD tier's reader (src/disk.rs: direct reads where the file system takes them, `how` says which) |

`device` and `host` are the profile's. `--no-gpu` skips the GPU's rows.

## The predictor

A token's time is the sum, over the tiers, of the bytes it reads from each over that tier's rate; tokens a second is its inverse:

```
tok/s = 1 / ( vram_bytes / vram_rate + link_bytes / link_rate + ram_bytes / ram_rate + ssd_bytes / ssd_rate )
```

Nothing overlaps in it, and nothing is computed: a GPU that reads its weights at the product rate and a CPU that reads experts at its
kernel's rate are what the measured tiers say. Each row also gives `tok_s_overlapped`: the GPU's reads and the computer's memory's
taken as overlapped (the expert cache's CPU mode computes the missed experts while the GPU runs the layer's others), the longer of the
two plus the link's and the disk's time, which a layer waits for. `predictions[]` has one row a placement, the first the
`--placement` asked for (`vram` unless given):

| placement | bytes a token by tier |
| :--- | :--- |
| `vram` | every weight on the GPU: vram = the active bytes + the keys and values of `positions_read` tokens + the states read and written |
| `experts_ram_cpu` | the routed experts in the computer's memory, a cache of a share of them on the GPU (`--gpu-experts`, places over experts), a missed expert computed by the CPU where it is: vram = the dense bytes + the hits' bytes; ram = the misses' bytes |
| `experts_ram_copy` | the same, a missed expert copied to the GPU before its layer: vram = the dense bytes + every expert read; link = the misses' bytes |
| `experts_ssd_cpu` | `experts_ram_cpu` with the computer's memory holding a share of the experts (`--ram-experts`), the rest on the disk: of the misses, the share the memory does not hold comes from the disk |
| `stream_ssd` | every weight read from the disk each token (a model larger than the memory): ssd = link = the active bytes; vram as `vram` |

`--hit F` is the share of a token's expert reads the cache has (the measured replay's hit rate where it is known; `--gpu-experts`
itself where not, a floor: the routers' skew makes the real rate higher). `positions_read` is the prompt plus half the output.
`bytes_per_token` and `assumes` in each row say what went in; `tiers` the rates used (GB/s). A tier a placement needs with no rate
gives `tok_s: null`.

Known limits: one session; a decode step's launch gaps, the sampler and the token's read-back are not modelled (they are in the
measured figure); the GPU's rate from `vram_read` over-predicts a dense model by the kernels' efficiency (the record has both rates);
the MoE rows take the mean expert; prefill is not predicted.

## Mapping to Glyd-Compute's proposed ModelCostProfile (model-cost-profile.schema.json)

Compute's names are strings of digits for bytes and a pool per tier; ours are numbers and one profile per model. The adapter's mapping:

| Compute | here |
| :--- | :--- |
| `key.manifest_digest` | glyd.json's SHA-256 (the serve record's `model.files[]` has it); `model.repo` + `model.revision` name the same files |
| `key.execution_profile`, `boundary_profile` | `level.effective`, `model.layout`; one boundary (whole model, one GPU) |
| `key.backend_build_digest` | `build.commit` (and the binary's SHA-256 in the record's build.txt) |
| `key.device_digest`, `pool_inventory_digest` | `device`, `host`; the pools are vram, link, ram, ssd |
| `key.workload` | `workload` (the same five fields) |
| `costs.measurement.basis`, `observed_at`, `samples` | `basis.*`, `observed_at`, `samples`; `valid_for_ms` and `evidence_digest`: not given (sizes do not expire; the record is the evidence) |
| `layers[].layer_type` | `layers[].kind`: full_attention; linear_attention and mamba -> linear_attention; moe -> other |
| `pool_cost.resident_weight_bytes` | `layers[].resident_bytes.total` (one pool: vram, the whole model there); a placement's split is the predictor's `bytes_per_token` |
| `pool_cost.active_weight_bytes_per_token` | `layers[].active_bytes_per_token` |
| `pool_cost.kv_bytes_per_token` | `layers[].kv_bytes_per_token` |
| `pool_cost.recurrent_bytes_per_session` + `convolution_bytes_per_session` | `layers[].state_bytes_per_session` (both together; the split is the config's: Qwen3.5 fp32 heads x 128 x 128 x 4 and conv 3 x conv_dim x 2) |
| `pool_cost.prefill_scratch_bytes`, `decode_scratch_bytes`, `staging_bytes`, `reserve_bytes`, `resident_overhead_bytes` | `totals.working_bytes` (one number for both phases, measured or null), null, null, null, `totals.overhead_bytes` (null) |
| `components[]` (embeddings, head, final norm; vision, MTP as not_in_scope) | `non_layer.*`, `excluded[]` |
| `per_stage` | `totals` |
| `phase_cost.prefill_us`, `decode_us_per_token` | `timing.prefill_tok_s`, `timing.decode_tok_s` (tokens a second, with `basis`); `predictions[].ms_per_token` |

Missing values are null here and absent there; neither is zero.
