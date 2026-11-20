# The model catalog

What is listed, who owns it, what a model needs before it is listed, and how a change is reviewed. Short on purpose.

## What the catalog is

- The families the engine runs and the levels each runs at: `glyd-engine/src/support.rs`, shown on the [Models](models.md) page.
- Each level's definition: `glyd-engine/src/levels.rs`.
- The cards: `cards/NAME-LEVEL.json`, the measured numbers a model's Hugging Face card is made from
  (`huggingface/model_card.py`), and the pre-packed repos `glyd/NAME-LEVEL` on the Hub.
- The manifest of short names to repos, when P4 lands ([#153](https://github.com/surya-koritala/Glyd/issues/153)). Its path goes
  into `.github/CODEOWNERS` in the same change.

The engine tells a family apart by its `config.json`, so a new size of a listed family runs without a catalog entry. An entry
is for a new family, a new level of a family, or a pre-packed repo.

## What we test and showcase

Models released in the last six months. The guide's examples, new measured runs and new cards use them; the cutoff moves with
the date (on 2026-11-20 it is 2026-05-20). An older model that already runs keeps running and keeps its entry. It gets no new
showcase.

## Who owns an entry

- The catalog owner is the maintainer who covers the engine in [MAINTAINERS](https://github.com/surya-koritala/Glyd/blob/main/MAINTAINERS).
  Today that is `@surya-koritala`, who covers all areas.
- Each entry has an entry owner, named in the pull request that adds it. The entry owner can re-run its checks on a machine,
  re-measures it when the engine changes in a way that moves its numbers, and answers issues about it. Until a second maintainer
  joins, the catalog owner is the entry owner.
- `.github/CODEOWNERS` lists the catalog files, so a change to one asks the owner for review.
- An entry whose owner has left goes back to the catalog owner, who finds a new owner or removes the entry in a pull request.

## Before a model is listed

Every item is a file or a command in the pull request.

| Needs | What counts |
| :--- | :--- |
| The kind runs | `glyd-engine supports MODEL` exits 0, and the family has a row in `support.rs` with its test (`every_family_is_told_apart` in the same file reads a fixture `config.json`). |
| A logits test | The engine's top-k logits and KL against a stored reference on a fixed prompt, for a small checkpoint of the family. The per-family harness is R7 ([#146](https://github.com/surya-koritala/Glyd/issues/146)) and is not merged. Until it is, attach the output of the family's check command (`glyd-engine kcheck`, `check`; [CONTRIBUTING.md](https://github.com/surya-koritala/Glyd/blob/main/CONTRIBUTING.md)) and say so. |
| A measured run | `glyd-engine run` or `serve` on named hardware, in `benchmarks/gpu/MACHINE-MODEL-DATE/` with a README: the machine, the driver, the engine's commit, the command, the tokens/s, the memory in use, and the log. `benchmarks/gpu/rtx4080s-qwen3-8b-engine-2026-11-20` is the form to copy; the model in a new run is one from the last six months. |
| Quality, for a lossy level | KL from bf16 and top-1 agreement on the pinned WikiText-2 text, by `research/lossy-2026/kld_study.py` as in `benchmarks/weights/kld-q56-2026-10-09`, with the FP8 checkpoint beside it where one exists. |
| License and gating | The model's license, its link, and whether the Hub gates it, recorded in the pull request. If the license allows redistributing derived weights, a pre-packed repo may be published and its card carries the base license (as `cards/nemotron-3-hummingbird.LICENSE` does). If it does not, the model is listed as run-from-the-original-repo only. A gated model's message names the license page and `glyd-engine login`. |
| A card | `cards/NAME-LEVEL.json` with `size_gb`, the speed on each GPU measured, `kl_bf16`, the date and the Glyd version and commit. A card has only what was measured. |

The Models page needs no edit: it is generated from `support.rs`.

The manifest's entries must carry the license and a gated flag, the level, the size at each memory tier, and a link to the
measured run. A schema check that refuses an entry without them is part of P4.

## How a review goes

1. The pull request is labeled `area: models` and uses the pull request template. It names the entry owner.
2. `.github/CODEOWNERS` asks the owner for review. The `Build & Test` check must pass.
3. The reviewer goes down the table above and confirms each item: the command was run (re-run `supports` and `fit`), every
   number on the card has a log in the pull request, the license line is right.
4. Where there is a second maintainer, a maintainer does not merge their own catalog change without that review.
5. A number that cannot be repeated is removed from the card, not softened.

## Asking for a model

Open a [Model request](https://github.com/surya-koritala/Glyd/issues/new?template=model_request.yml). Run
`glyd-engine supports MODEL` first and paste what it says; the form also asks for the architecture, the license, whether the
Hub gates it, the size you would run and the hardware.

- If `supports` says the kind runs, the reply is the command to run it. A catalog entry is only needed for a pre-packed repo
  or a card.
- If it does not, the request becomes an engine issue labeled `area: models`. A maintainer decides if it joins the
  [ROADMAP](https://github.com/surya-koritala/Glyd/blob/main/ROADMAP.md), and says why if not.
- Someone who can test on the hardware, or who sends the logits fixture and the measured run, moves a request fastest.
