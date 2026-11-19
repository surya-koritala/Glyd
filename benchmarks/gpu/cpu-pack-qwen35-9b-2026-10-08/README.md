# Qwen3.5-9B packed on the CPU, checked against the GPU's save

Qwen/Qwen3.5-9B (commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`) was packed on a GitHub runner with no GPU (`glyd-gpu pack --plan`, the plan from `huggingface/plan.py`) and, separately, by this repository's glyd_gpu on one L40 (`from_pretrained` then `save_pretrained`, both in 1 GB shards). The two saves' weights are byte-identical.

| | CPU save (GitHub runner) | GPU save (L40) |
| :-- | :-- | :-- |
| Weight shards | 12 files, 12,718,661,901 bytes | the same 12 files, byte for byte |
| `model.safetensors.index.json`, config, tokenizer and template files | | identical |
| `glyd.json` | | identical but for `source.revision`: the CPU save records the commit, the GPU save `null` |
| Tensors (order, file, name, dtype, shape, sha256) | 685 | identical |
| Packs verified | 201 tensors decode to `glyd.json`'s sha256; 226 saved as they are match | 202 packed tensors checked as made (the embedding's pack among them) |
| Size | 11.85 GiB against 16.68 GiB of bf16 (−29.0%) | |
| Time, peak memory | pack 207 s and verify 107 s on 2 cores, 3.75 GB peak | pack 89 s, save 33 s |

The GPU save, loaded back with `verify=True` (202 packed tensors and 226 others checked against `glyd.json`, none unchecked), generates the same 24 greedy tokens as the model packed from the bf16 checkpoint: "The three primary colours of light are red, green, and blue. When these colours are combined in equal intensities, they produce white light." The CPU save's weights are those bytes.

The plan builds the model transformers' from_pretrained builds for this checkpoint (`Qwen3_5ForCausalLM`, transformers 5.18.0): 153 packs, 226 tensors saved as they are, checkpoint names renamed as transformers renames them, the linear-attention layers' f32 `A_log` and norm weights saved as bf16 as a bf16 model holds them. The vision tower (333 tensors) and the multi-token-prediction head (15) are not in it, as from_pretrained does not load them for a causal LM.

## Run

CPU: [CPU pack run 37831492949](https://github.com/Sigbound/glyd-gpu-private/actions/runs/37831492949) on main `a0ee691b`; `cpu/` holds its plan, `glyd.json`, index, file and tensor listings and the pack and verify output.

GPU: [bundle run 37832201735](https://github.com/Sigbound/glyd-gpu-private/actions/runs/37832201735) at merge commit `0eb2e481d8bf173749171b01fafb973ad1f6ab76` (its `glyd-gpu-py/glyd_gpu`, `huggingface` and `gpu` trees equal #111's head and merged main `907d74ed`): glyd_gpu's sources, its CUDA library built by `gpu/build_lib.sh` in release.yml's container, and the job; artifact zip SHA256 `2ca5555a302243435871ebebacfd9752524be54c3e5adf90ae1399eb2bc35b58`, bundle SHA256 `bae153374ecde04364ebcba68e47be9a82eec75c263659eacdf4895823290733`, all 33 members checked before upload and on the pod. Pod `clcyoub6b1cj3d`, an L40, was created at 19:54 UTC on 2026-10-08; the job ran 19:55:06–19:58:19 (`gpu/steps.txt`) and the launcher deleted the pod at 19:59; an authenticated listing at 19:59:59 UTC returned no pods (`gpu/deletion-proof.json`). The guard accounts 325 seconds at $0.82/hour, a conservative $0.074 GPU estimate excluding storage, within the $0.37 cap (`gpu/cost.json`, which also lists the two earlier attempts: one deleted by the price check before running, one stopped in its first minute). `RECORD-SHA256SUMS` covers this record.
