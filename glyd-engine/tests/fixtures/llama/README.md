Config files and tensor-name lists of the released Llama 3.x checkpoints, for `tests/llama_describe.rs` (F1a). Files read from the public mirror
`unsloth/Llama-3.*` (the meta-llama repositories are gated): config.json as released, less the mirror's own `unsloth_fixed` and `pad_token_id`
keys; the tensor names of the 8B's and the 70B's `model.safetensors.index.json` (names only). No weights, no checkpoint.
