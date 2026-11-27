# Logits tests

Every model family the engine runs has a fixture of reference logits, and `glyd-engine logitscheck` compares the engine's logits with it. The GPU job (`.github/workflows/gpu-tests.yml`, started by `gpu.yml`) runs it on merges to main that touch the engine, `gpu/`, `glyd-gpu/`, `tests/clients/`, the lockfile or the job's own scripts and workflows, and on every pull request labeled `gpu`; no checkpoint is downloaded: every case is a random model. The suite took about 25 s on the RTX 4080 SUPER runner when it had nine cases (2026-11-20); it has 18 now and has not been timed since. The code is `src/logits.rs`, the fixtures are `tests/logits/`.

## What is compared

A fixture holds fixed tokens: a prompt of 24, 12 decode steps (each fed the token the fixture names), and a long prompt of 150 that crosses the engine's chunk of 64. For the last position of each pass (14 rows) it keeps the 64 largest logits with their token ids, the log-sum-exp of the whole row, and a hash of every logit's bits. A fixture says where its numbers came from (`provenance.source`) and the bound each level must meet:

| Bound | Meaning |
| :--- | :--- |
| `bits` | every logit of every row equal to the bit (the hash). For an engine's own recorded output on one kernel set (`gpu`: sm89-geforce-80sm, sm89-l4-58sm, sm86-a10-72sm; see below): the lossless path promises the same bits run to run, and a change that moves one is a change to look at. A GPU of another kernel set skips the file; a case with no fixture its GPU can be held to fails. |
| `kl` | KL divergence of the reference's softmax from the engine's, over the reference's 64 largest tokens and the rest as one bin: at most `max` in every row and `mean` on average, and the engine's best token the reference's wherever the reference's lead over its second is past 0.1 (a lead within the roundings is a coin's toss), `top1` of those at least. |

## A reference or a pin

A fixture is one of two things, and `logitscheck` says which on every line and counts them apart in its last lines ("N held to a reference, M pinned, K not held"):

- A reference (`CASE.json.gz`, no `gpu`): another implementation's numbers, transformers or a CPU reference. A line within its bound starts `PASS`: the engine's logits are right to that bound.
- A pin (`CASE.KEY[.LEVEL].json.gz`, with a `gpu`): the engine's own output as a GPU of the kernel set KEY recorded it. A line whose bits are unchanged starts `PIN`: the logits have not moved since the recording. It says nothing of whether they were right when recorded.

A pin's key is everything the engine's kernel choices and launches read of the GPU (`fold::kernel_set`): the compute capability, each word of the name a choice reads (`geforce`: GeForce Ada's stream-K prompt kernels; `l4`, `a10`, `l40s`: their measured routes; `pcie`, `gh200`, `h100`, `nvl`), and the SM count, which the products' blocks and K splits are sized to and so the order of their sums. An RTX 4080 SUPER is `sm89-geforce-80sm`, an L4 `sm89-l4-58sm`, an A10 `sm86-a10-72sm`, an L40 `sm89-142sm`. Two GPUs of one key make the same bits; a GPU of a key with no pin says so ("no pin of this kernel set recorded") and is held to the case's reference alone, or fails where it has none.

A case with pins alone is printed as `pinned only`, never as within bounds: none today, every case has a reference in `tests/logits` (the pins of `dense`, `qwen35`, `gemma4` and `nemotron-h` are beside theirs). `packcheck --engine` reads no pin: it holds each saved directory to the engine's own run of the same model packed as it loads in the same layout, made in the same run, bit for bit, and to the case's reference where one bounds the level.

## The cases

| Case | Family | Reference | Levels (bound) |
| :--- | :--- | :--- | :--- |
| `dense` | The plain dense kind (one attention and one gated MLP a layer, on the executor; the kind's case, no release's) | transformers, bf16 on the same random weights; pins `dense.sm89-geforce-80sm`, `dense.sm89-l4-58sm` | penguin (KL); pin: penguin (bits) |
| `qwen35` | Qwen3.5 family (Gated DeltaNet and gated attention: Qwen3.8-27B's layout) | transformers, bf16 on the same random weights; pins `qwen35.sm89-geforce-80sm`, `qwen35.sm89-l4-58sm` | penguin, kestrel, swift, wren (KL); pin: penguin (bits) |
| `qwen35-wide` | Qwen3.5 family, the wide shape (three value heads to a key head, a head of its own, the text model of a checkpoint with a vision tower) | the layer kinds' CPU reference, `src/runtime/reference` | penguin (KL) |
| `qwen35moe` | Qwen3.5 family with experts (softmax-routed, a gated shared expert) | the layer kinds' CPU reference | penguin, kestrel, swift, wren (KL, the bounds of `qwen35`'s), on the engine's own choices |
| `muse` | Muse Spark | the layer kinds' CPU reference | penguin (KL), on its choices |
| `kimi` | Kimi-K3 (delta attention, latent attention, MoE) | the layer kinds' CPU reference | penguin (KL), on its choices |
| `kimi-dsa` | Kimi-K3 family, the newer config's sparse attention (an indexer of a key a token) | the layer kinds' CPU reference | penguin (KL), on its choices |
| `glm` | GLM-5.3-Flash (delta attention, sparse attention, MoE) | the layer kinds' CPU reference, the head's logits in bf16 as the release has them | penguin (KL), on its choices |
| `glm-dsa` | GLM-5.3's layout (the latent attention with its rotary on every layer, indexers of their own and shared, the plain residual) | the layer kinds' CPU reference | penguin (KL), on its choices |
| `glm-dsa-f8` | The same layout as released in FP8 (E4M3 matrices with fp32 block scales, held as released) | the layer kinds' CPU reference, reading `weight_scale_inv` exactly | penguin (KL), on its choices |
| `deepseek-v4` | DeepSeek-V4 | `src/dsv4_ref.rs`, without the release's simulated quantizations; held in three parts (below) | penguin (KL), on its choices |
| `deepseek-v41` | DeepSeek-V4.1 | the layer kinds' CPU reference, rounding where the release's `model.py` does; held in three parts as `deepseek-v4` is (below), with its release's quantizations of FP8 blocks of 32 | penguin (KL), on its choices |
| `gemma4` | Gemma 4 (sliding, global K = V, soft-capped head) | the layer kinds' CPU reference, rounding where the description says the release does; pin `gemma4.sm86-a10-72sm` | penguin (KL); pin: penguin (bits) |
| `gemma4-kv` | Gemma 4's E2B/E4B layers (KV sharing across layers, the double-wide MLP) | the layer kinds' CPU reference | penguin (KL) |
| `nemotron-h` | Nemotron-H (Mamba-2, attention, MoE) | the layer kinds' CPU reference; pins `nemotron-h.KEY` and `nemotron-h.KEY.kestrel` of `sm89-geforce-80sm` and `sm89-l4-58sm` | penguin (KL); pins: penguin, kestrel (bits) |
| `qwen4x` | Qwen3.8-Flash-Next (Gated DeltaNet, indexed attention, hyper-connections, per-layer embeddings, MoE) | the layer kinds' CPU reference | penguin (KL), on its choices |
| `qwen4x-s2` | The same family with other counts (2 streams, index heads of 64 over pools of 2, no n-gram embedding, experts off the six-bit grid) | the layer kinds' CPU reference | penguin (KL), on its choices |
| `qwen4x-cb` | The same family through the expert cache (hidden 2,560, two-bit codebook experts of 640 neurons from a pack, 3 places of 8 a layer) | the layer kinds' CPU reference | penguin (KL), on its choices |

A case with a CPU reference has the kind's bound (KL mean 1e-4, max 2.5e-4, best token 85%), except the transformers fixtures of the lossy levels, whose bounds are wider (`qwen35`: kestrel 6e-3 / 1.5e-2, swift 2.5e-2 / 5e-2, wren 9e-2 / 1.4e-1, with `top1` 65%, 20% and 0) and `dense` (3e-5 / 4e-5, set from its measured run). Each fixture's `provenance.note` says where its bound came from. A case held on the engine's own choices is held to its CPU reference with the engine's routers' experts and its indexers' picks given to the reference (`Case::on_choices`), since a router's near ties flip with the rounding of its input; its rows against the reference's own are printed all the same. Every case a CPU reference holds but `nemotron-h` has its draw's noise floor (the reference against itself with the release's roundings) held to at most a tenth of its bound by `logits::tests::every_case_held_to_a_reference_is_well_conditioned`; a case past it would be printed as open (`NOISE_FLOOR_OPEN`: none now).

`deepseek-v4` is held in three parts (`deepseek-v41` in the same three, with its release's form of the quantizations): (1) its engine and its reference both without the release's simulated quantizations, on the engine's own choices at the kind's bound; (2) each operation that quantizes as the release does (`dsv4_ref::qcheck_rows`), fed the reference's own input, bit for bit; (3) the whole model with them, its figures printed and never a pass or a fail (the reference's own noise floor with the simulation is eight times the kind's bound: logits.rs `dsv4_also` says why).

Only models released in the last six months are tested, and the big ones do not fit a 16 GB card (Muse, Kimi, DeepSeek, GLM, Qwen3.8-Flash-Next, Gemma 4 12B and 31B), so every case is a random model of the family's layout, with a fixed seed: the layer kinds, the routers and the heads are the real ones, the weights are not. A random model's logits are close to flat and its expert choices have near ties, so the CPU references agree with the engine on the best token less often than a trained model's do (`musecheck`, `kimicheck`, `dsv4check` print the same): a case with routers or indexers is held on the engine's own choices, at the kind's bound, and the bits fixtures carry the sensitivity of the engine against itself. A pin (`dense`, `qwen35`, `gemma4`, `nemotron-h`) is the engine's own output at the commit that recorded it, and says so: it guards a rewrite against changing the numbers, it does not say the numbers are right. The kernel checks against the layer kinds' CPU reference (`kindcheck`, kind by kind and chained) and the families' own checks (`kcheck35`, `g4check`, `nhtoy`, `musecheck`, `kimicheck`, `dsv4check`) say that.

## Commands

    glyd-engine logitscheck [CASE...] [--fixtures DIR] [--level L] [--list-cases]    the engine against the fixtures (exit 1 where one is outside its bound)
    glyd-engine logitscheck muse kimi deepseek-v4 --cpu-ref                     the CPU references against their own fixtures (no GPU): has the reference moved
    glyd-engine logitsrecord CASE --from cpu|engine --out DIR [--level L] [--bound bits|kl:MEAN:MAX:TOP1] [--tokens FILE] [--note TEXT] [--date D]
    glyd-engine logitsbound FIXTURE LEVEL=BOUND...                              a fixture's bounds replaced
    python3 glyd-engine/tools/logits_ref_hf.py CASE MODEL_ID OUT.json.gz [--bound LEVEL=kl:...]    a fixture from transformers (on the project's GPU machine)

Record on the project's GPU machine (with no CI job running), in the CI image, with the same build as the job. To record the engine's own bits for a new kernel set, run `logitsrecord CASE --from engine --bound bits --out tests/logits` on a GPU of it (with `--level LEVEL` for a lossy level's); the file is named `CASE.KEY.json.gz` (`CASE.KEY.LEVEL.json.gz`), and `packcheck --engine --record` writes the set's Hummingbird digest `tests/packs/nemotron-h-hb.KEY.json.gz`. Such a file is a pin: record it only from an engine whose logits a reference or the layer kinds' check already holds, and never to make a failing line pass.

## Adding a model

1. A case in `cases()`: a Hub id or a generator of a random model, the engine's loader at a level, the CPU reference where there is one.
2. A reference: `logits_ref_hf.py` for a Hub model, `logitsrecord --from cpu` where the case has a CPU reference (a `*_ref.rs`, or the layer kinds' `src/runtime/reference`), and `logitsrecord --from engine` for the bits.
3. Bounds: for a case a CPU reference holds, the kind's (KL mean 1e-4, max 2.5e-4, best token 85%), set before any run is judged by it (`logitsrecord --bound`, `logitsbound`); a case whose draw's noise floor passes a tenth of the bound gets a better draw, never a wider bound (`logits::tests::every_case_held_to_a_reference_is_well_conditioned`).
4. `cargo test -p glyd-engine --lib logits::` reads every fixture and checks its shape.

A kernel change that moves the lossless bits fails the `bits` fixtures on purpose: record them again with the change in the same pull request, and say in it why the numbers moved. In the same way the gate holds every case with a CPU reference to its reference fixture (`logits::tests::every_reference_holds_its_fixture`, a `run` line of `tests/ignored-in-gate.txt`): a fixture the layer kinds' reference recorded, every row to the bit, since that reference is the same bits on every computer (its exp, log, tanh, pow, sin and cos are the `libm` crate's); any other within its bound. A change of the reference that moves a fixture records it again (`logitsrecord CASE --from cpu`) in the same change, and says which rows moved and why: through the releases' bf16 rounding points a last place can flip a router's near tie and move a row by 1e-3 of KL.
