# Pre-packed 2026 models for Hugging Face (glyd/<model>)

Private tooling. A first run of a `glyd/<model>` repo loads the packed weights as saved: no download of the bf16
checkpoint, no packing. No weights are kept in git; the saves are made on a pod from the recipe below.

| File | What it does |
| :-- | :-- |
| `job.sh` | On one 80 GB GPU pod: for each model, download, `python -m glyd_gpu pack` (glyd-v4), `check.py`, `glyd verify` (the release CLI, on the CPU); with `UPLOAD_ORG`, `card.py` + `upload.py` for each save that passed |
| `check.py` | Loads the save as saved with `verify=True`, gets one greedy chat answer, measures the save's bytes against the bf16 bytes of the same tensors and the weights' GPU memory |
| `card.py` | Writes the model card from `check.py`'s JSON (`--serve` adds the `glyd run` / `vllm serve` lines) |
| `upload.py` | Uploads one save as a private repo with README, LICENSE and NOTICE, checks the Hub's file names and sizes and the card against the local ones, and makes the repo public only with `--public` |
| `publish.sh` | Runs `job.sh` through `rp_run.sh` with `UPLOAD_ORG`, and passes the Hugging Face token to the job over ssh into a FIFO (never into a file on the pod, never printed) |
| `cards/` | The seven cards, from the 2026-10-06 run, without `--serve`: vLLM refuses these saves today |
| `results/` | What the 2026-10-06 run measured, per model (`results/README.md`) |

## Uploading

The organization `glyd` exists (huggingface.co/glyd). `ORG=glyd bash tools/hf-prepacked/publish.sh` uploads the repos as private and checks them; it needs a write token
for `glyd`'s repos (`pip install huggingface_hub`, then `hf auth login`) and uses `tools/runpod/rp_run.sh`. The seven models take about 45 minutes, so a
running `rp_guard.sh` must allow that (`MAX_AGE`). Add `PUBLIC=1` to make each repo public once its files and card have been checked. Add `SERVE=1` only once vLLM loads these families' saves.
A lossy pack of a recent release is published by `glyd-engine ship` (`glyd-engine/src/ship.rs`), with its quality card.

## Loading in glyd 0.29.0 (2026-10-06, read from the code then; not read again since)

- **`glyd.from_pretrained("glyd/<model>")`** loads a save of format glyd-v1 to v4, in either layout, for any
  family that transformers loads. On 2026-10-06 all seven saves loaded this way, passed verify bit for bit and answered
  a chat prompt (`results/`). The saves are 18% to 31% smaller than the bf16 bytes of the same tensors.
- **`vllm serve glyd/<model> --quantization glyd`** loads saves of `Qwen3ForCausalLM` and `LlamaForCausalLM`
  only (`SAVES` in `glyd_gpu/vllm_plugin.py`). A save of Qwen3.5 or Qwen3.8 (`Qwen3_5ForConditionalGeneration`) or of
  Gemma 4 (`Gemma4ForConditionalGeneration`) is refused, and the message says to serve the bf16 checkpoint. `glyd run` and
  `glyd serve` start vLLM, so they refuse these saves the same way. Saves that hold mixture-of-experts packs are refused in vLLM too.
- **`glyd pack` and `glyd verify`, as `pip install glyd` installs them**, run the Python pack and verify on the GPU,
  the same as `python -m glyd_gpu pack|verify`. On the pod, `glyd pack` packed Qwen3.5-4B and gemma-4-12B-it, and
  `glyd verify` passed on every save. The Rust `glyd-gpu` command packs and verifies on the CPU, but neither the pip
  wheel nor the release tarball ships it, and it packs only Qwen3, Qwen2, Llama, Mistral, Granite and GraniteMoe
  (`glyd-gpu/src/save.rs`, `FAMILIES`).
- **Qwen3.5 and Qwen3.8 saves contain only the text model.** `from_pretrained` loads them as
  `AutoModelForCausalLM`, which transformers maps to `Qwen3_5ForCausalLM`, and that class drops `model.visual.*` and
  `mtp.*`. The `config.json` in the save still names `Qwen3_5ForConditionalGeneration`. Gemma 4 loads as
  `Gemma4ForConditionalGeneration`, so its save holds the whole model.
- **Layout:** publish the tiered layout (glyd-v4, `pack`'s default), which is the smallest. On GPUs where the 12-bit layout
  is faster, it is decoded and packed again as it loads. A 12-bit variant (glyd-v3, `--layout mma12`) would only skip
  that repacking on the A10, A100, H100 and Blackwell. It is not needed until vLLM loads these saves.

## Format guarantee

Each save names its format in `glyd.json`. Every glyd release reads every earlier format: today glyd-v1 to glyd-v4,
which covers saves from glyd 0.20 on (`FORMATS` in `glyd_gpu/format.py`). Older tiered packs are packed again as they
load. A key in `glyd.json` that a newer release added is skipped with a warning and everything else is still checked.
An older release refuses a newer format by its name and never misreads it. A change to the format must keep this
property: add a format name and keep reading the earlier ones.

## Models (2026, Apache-2.0, not gated; licenses checked on the Hub 2026-10-06)

Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B and Qwen/Qwen3.8-27B ship with a LICENSE file. This repo uploads
that file with the save. google/gemma-4-E4B-it, -12B-it and -31B-it carry the apache-2.0 tag but no LICENSE file,
so `job.sh` uploads the Apache-2.0 text from apache.org with them. None of the seven has a NOTICE file. `upload.py`
writes a NOTICE that says the weights are stored in Glyd's format (Apache-2.0 section 4(b)), and each card credits the original model and its maker.
