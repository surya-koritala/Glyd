# Logits tests

Every model family the engine runs has a fixture of reference logits, and `glyd-engine logitscheck` compares the engine's logits with it. The GPU job (`.github/workflows/gpu-tests.yml`) runs it on every merge to main and on every pull request labeled `gpu`; on the RTX 4080 SUPER runner the whole suite takes about 25 s (no checkpoint is downloaded: every case is a random model). The code is `src/logits.rs`, the fixtures are `tests/logits/`.

## What is compared

A fixture holds fixed tokens: a prompt of 24, 12 decode steps (each fed the token the fixture names), and a long prompt of 150 that crosses the engine's chunk of 64. For the last position of each pass (14 rows) it keeps the 64 largest logits with their token ids, the log-sum-exp of the whole row, and a hash of every logit's bits. A fixture says where its numbers came from (`provenance.source`) and the bound each level must meet:

| Bound | Meaning |
| :--- | :--- |
| `bits` | every logit of every row equal to the bit (the hash). For an engine's own recorded output on one GPU generation (`gpu`: sm89): the lossless path promises the same bits run to run, and a change that moves one is a change to look at. Other GPU generations skip these. |
| `kl` | KL divergence of the reference's softmax from the engine's, over the reference's 64 largest tokens and the rest as one bin: at most `max` in every row and `mean` on average, and the engine's best token the reference's wherever the reference's lead over its second is past 0.1 (a lead within the roundings is a coin's toss), `top1` of those at least. |

## The cases

| Case | Family | Model | Reference | Levels |
| :--- | :--- | :--- | :--- | :--- |
| `qwen3` | Qwen3 dense (the dense engine's architecture; no recent release uses it) | random, Qwen3-0.6B's layout, 4 layers | transformers, bf16 on the same random weights | penguin (KL) |
| `qwen35` | Qwen3.5 family (Gated DeltaNet and gated attention: Qwen3.8-27B's layout) | random, 0.8B's widths, 4 layers | transformers, bf16 on the same random weights | penguin, kestrel, swift, wren (KL) |
| `muse` | Muse Spark | random, 4 layers, 16 experts | the CPU reference, `src/muse_ref.rs` | penguin (KL) |
| `kimi` | Kimi-K3 (delta attention, latent attention, MoE) | random, 8 layers | `src/kimi_ref.rs` | penguin (KL) |
| `deepseek-v4` | DeepSeek-V4 | random, 5 layers | `src/dsv4_ref.rs` | penguin (KL) |
| `gemma4` | Gemma 4 (sliding, global K = V, soft-capped head) | random, 4 layers, heads 256 and 512 | the engine's own bits | penguin (bits) |
| `nemotron-h` | Nemotron-H (Mamba-2, attention, MoE) | random, 4 layers | the engine's own bits | penguin, kestrel (bits) |
| `glm` | GLM-5.3-Flash | random toy, experts at kestrel | the engine's own bits | kestrel (bits) |
| `qwen4x` | Qwen3.8-Flash-Next | random tiny of `tools/qwen4x_ref.py`'s layout, experts at kestrel | the engine's own bits | kestrel (bits) |

Only models released in the last six months are tested, and those are all larger than a 16 GB card holds (Qwen3.8-27B, Qwen3.8-Flash-Next, GLM-5.3-Flash, Muse-Spark-Open, Kimi-K3, DeepSeek-V4, Gemma 4 12B), so every case is a random model of the layout. The big releases do not fit a 16 GB card (Muse, Kimi, DeepSeek, GLM, Nemotron 3 Super, Gemma 4 12B and 31B), so a random model of the family's layout stands in, with a fixed seed: the layer kinds, the routers and the heads are the real ones, the weights are not. A random model's logits are close to flat and its expert choices have near ties, so the CPU references agree with the engine on the best token less often than a trained model's do (the existing `musecheck`, `kimicheck`, `dsv4check` print the same); the KL bounds for those are wide and the bits fixtures carry the sensitivity. Where the case has no reference of another implementation (gemma4, nemotron-h, glm, qwen4x) the fixture is the engine's own output at the commit that recorded it, and says so: it guards a rewrite against changing the numbers, it does not say the numbers are right. The families' kernel checks against transformers (`kcheck*`, `qxtoy`, `g4check`, `nhtoy`, `glmtoy`) say that.

## Commands

    glyd-engine logitscheck [CASE...] [--fixtures DIR] [--level L] [--list-cases]    the engine against the fixtures (exit 1 where one is outside its bound)
    glyd-engine logitscheck muse kimi deepseek-v4 --cpu-ref                     the CPU references against their own fixtures (no GPU): has the reference moved
    glyd-engine logitsrecord CASE --from cpu|engine --out DIR [--level L] [--bound bits|kl:MEAN:MAX:TOP1] [--tokens FILE] [--note TEXT] [--date D]
    glyd-engine logitsbound FIXTURE LEVEL=BOUND...                              a fixture's bounds replaced
    python3 glyd-engine/tools/logits_ref_hf.py CASE MODEL_ID OUT.json.gz [--bound LEVEL=kl:...]    a fixture from transformers (GPU lock on glyd-pc)

Record on glyd-pc (take `~/glyd-scratch/gpu.lock`, no CI container running), in the CI image, with the same build as the job. To record the engine's own bits for a new GPU generation, run `logitsrecord CASE --from engine --out tests/logits` there; the file is named `CASE.smNN.json.gz`.

## Adding a model

1. A case in `cases()`: a Hub id or a generator of a random model, the engine's loader at a level, the CPU reference where there is one.
2. A reference: `logits_ref_hf.py` for a Hub model, `logitsrecord --from cpu` where the family has a `*_ref.rs`, and `logitsrecord --from engine` for the bits.
3. Bounds from a measured run (`logitscheck` prints each row set's KL and best-token counts), about twice the measurement, with `logitsbound`.
4. `cargo test -p glyd-engine --lib logits::` reads every fixture and checks its shape.

A kernel change that moves the lossless bits fails the `bits` fixtures on purpose: record them again with the change in the same pull request, and say in it why the numbers moved.
