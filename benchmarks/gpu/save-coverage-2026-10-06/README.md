# Saves with their tables packed (glyd-v5), 2026-10-06

Packing the tables in a save is opt-in (`python -m glyd.gpu pack MODEL OUT --tables`, `save_pretrained(..., tables=True)`).
Saves without it are unchanged (glyd-v4). This record was made while the tables were packed by default, with the code
described below; every "after" row is what `--tables` gives that code.

One Runpod pod, NVIDIA L40S (Ada, sm 8.9, 46 GB), driver 580.159.03, torch 2.8.0+cu128, transformers 5.18.0. 2026 models only.
"Before" is glyd_gpu 0.29.0 (private main 63625822 and its libraries). "After" is the same package with `format.py`, `hf.py`
and `_cli.py` from the branch `save-coverage-2026`; those three files differ from main only by this change. The libraries
are the same in both runs.

Steps for each model (`scripts/job.sh`):
1. `python -m glyd_gpu pack MODEL OUT` with each code, tiered layout, verify on (`scripts/measure.py` records the time and
   PyTorch's peak).
2. `scripts/check.py` on each save: loaded as saved with verify (every packed tensor decoded against glyd.json's sha256, and
   every tensor saved as it is checked by its own), then one greedy answer, the GPU's bytes after the load, and the save's
   sizes.
3. With the branch: `python -m glyd_gpu verify` on its save; a load of the v0.29.0 save; and the bf16 checkpoint loaded by
   `from_pretrained` directly.
4. v0.29.0 reading the branch's glyd.json.

The L40S stands in for the RTX 4090. It is the same GPU generation, and the layout is forced to tiered in every step. The
4090's 24 GB is too small: gemma-4-E4B-it's pack and verified load peak at 27.2 GB with either code.

## Results (GB = 10^9 bytes)

Smaller is measured against the bf16 bytes of the same tensors.

| Model | Save before (smaller) | Save after (smaller) | Kept as they are, before → after | GPU after a load, both | Peak, both |
| :-- | --: | --: | :-- | --: | --: |
| google/gemma-4-E2B-it | 8.66 GB (15.1%) | 6.85 GB (32.9%) | 5.54 GB → 0.03 GB | 7.84 GB | 20.1 GB |
| google/gemma-4-E4B-it | 12.98 GB (18.3%) | 10.68 GB (32.8%) | 7.01 GB → 0.03 GB | 12.08 GB | 27.2 GB |
| Qwen/Qwen3.5-9B | 12.72 GB (29.0%) | 12.05 GB (32.7%) | 2.05 GB → 0.01 GB | 12.52 GB | 17.2 GB |
| google/gemma-4-12B-it | 16.78 GB (29.9%) | 16.11 GB (32.6%) | 2.03 GB → 0.02 GB | 18.00 GB | 22.6 GB |

- Packed per save: 2 more packs on each Gemma 4 E model (the embedding and the per-layer embedding) and 1 more on Qwen3.5-9B
  and gemma-4-12B-it (the embedding). What stays as it is: norms, small vectors, conv weights, and Gemma 4's 3-D vision
  position table (31.5 MB).
- Exactness: `verify` on every branch save gave E2B 530 tensors decoded to glyd.json's sha256 and 1,424 saved as they are
  matching theirs; E4B 597 and 1,482; Qwen3.5-9B 203 and 225; 12B 334 and 345. In this run each table's pack was checked
  by that `verify` and by the verified load, not as it was made; the code now also checks it as it is made (not re-run).
- Same answer: for each model, the greedy answer is identical across the before save, the after save, the branch loading
  the v0.29.0 save, and the bf16 checkpoint loaded directly.
- Compatibility: the branch loads v0.29.0's glyd-v4 saves, with the same GPU bytes and answer. v0.29.0 refuses a glyd-v5
  save by its format: `format 'glyd-v5'; this glyd reads glyd-v1 and glyd-v2 and glyd-v3 and glyd-v4`.
- GPU bytes and peaks are the same before and after. `from_pretrained` already packed the tables on the GPU as it loaded
  them; only the save kept them bf16. Loading the bf16 checkpoint directly: E2B 7.87, E4B 12.12, Qwen3.5-9B 12.56,
  12B 18.04 GB.
- Load time, loaded with verify (`load_s` in `results/*-check.json`), v0.29.0's save with v0.29.0's code / the same save
  with the branch / the branch's glyd-v5 save / the bf16 checkpoint packed as it loads (branch): E2B 12.3 / 12.1 / 14.8 /
  7.4 s, E4B 20.5 / 21.2 / 23.6 / 12.2 s, Qwen3.5-9B 23.2 / 24.8 / 24.4 / 12.6 s, 12B 29.4 / 31.8 / 33.5 / 18.1 s. Load
  times without verify were not measured.
- Pack time, before → after: E2B 23.4 → 28.8 s, E4B 44.4 → 44.2 s, Qwen3.5-9B 50.7 → 53.3 s, 12B 60.2 → 61.8 s.
- GPU tests of the branch on the pod (`results/tests.txt`): test_manifest_and_names, test_check_files,
  test_tables_saved_packed, test_cli_pack_and_verify and test_packed_weight_view pass.

## Not changed here

- Qwen3.5's vision tower and MTP head (1.40 GB of Qwen3.5-9B's 19.31 GB) are not in its saves at all. `from_pretrained`
  loads the text model only.
- GPU bytes: a tied output layer (all Gemma 4) is packed a second time beside the embedding. That is about 0.9 GB on E4B
  and 1.4 GB on 12B, and is most of E4B's gap between 24.5% and 32%.
- Peak: gemma-4-E4B-it's pack and verified load reach 27.2 GB, which does not fit a 24 GB card.
