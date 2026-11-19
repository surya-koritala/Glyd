# Complete Candidate A weight inventory

Qwen/Qwen3.8-27B: **24.43534 GiB encoded versus 51.74694 GiB of original BF16 tensor values (52.77916% smaller)**. All 1,199 tensors and 27,781,427,952 values were encoded and decoded. Every decoded value matched the declared Candidate A rounded value exactly. Candidate A is lossy relative to the original BF16 weights.

This measures reference-format bytes, including tensor headers, block headers, dictionaries and exceptions. It is not GPU-resident memory, a saved deployable model, inference performance or a quality result. Each encoded block was materialized and checked, then discarded. No full encoded checkpoint or original weights are included in this record.

## Evidence

[Complete CI run and raw artifact](https://github.com/surya-koritala/glyd-gpu-private/actions/runs/37557984503), source `64fa083191943b2d0526fa6ec2219311a4a535a6`. The pinned BF16 checkpoint is `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. All 18 downloaded shard sizes and SHA256 hashes matched the pinned Hub metadata. Coverage matched the checkpoint index and the independent 1,199-tensor inventory.

`summary.json` is the downloaded inventory artifact; `ci.json` is the corresponding CI status response. `tensors.jsonl` contains all original per-tensor result objects in name order. `source.json` preserves checkpoint identity, shard hashes and hashes of the original individual tensor artifacts. `SHA256SUMS` covers the files in this record. No model or test ran on a workstation.

| Tensor group | Tensors | BF16 GiB | Candidate A GiB | Reduction |
| :-- | --: | --: | --: | --: |
| embedding | 1 | 2.36816 | 1.11940 | 52.731% |
| full attention | 96 | 3.12502 | 1.47695 | 52.738% |
| head | 1 | 2.36816 | 1.12134 | 52.649% |
| linear attention matrix | 240 | 10.35645 | 4.88679 | 52.814% |
| linear attention other | 192 | 0.00368 | 0.00189 | 48.671% |
| mlp | 192 | 31.87500 | 15.04609 | 52.797% |
| mtp | 15 | 0.79106 | 0.37494 | 52.603% |
| other | 129 | 0.00123 | 0.00057 | 53.680% |
| vision | 333 | 0.85818 | 0.40736 | 52.532% |

## FP8 storage reference

The official `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` checkpoint has 30,866,866,928 bytes (28.74701 GiB) across its 66 indexed safetensors files. Candidate A's counted reference-format bytes are 14.99867% smaller. FP8's count includes file headers and weights that remain BF16; this is not a comparison with a hypothetical eight bits for every parameter. The pinned metadata and shard hashes are in `fp8-reference.json`. Neither GPU memory nor speed can be inferred from these file counts.

## Fixed-value representation headroom

Current representation: 7.55533418 bits per value. The measured empirical independent-symbol entropy, conditioned on tensor, is 6.50005082 bits per value. This is a statistical estimate under that probability model, not an absolute compression ceiling or a demonstrated GPU-decodable format. Coding overhead, random access and execution cost remain separate constraints. Any subsequent representation must reconstruct these same rounded values exactly.

## Reproduce

Run the `Lossy 27B reference study` workflow at the source commit with `full_inventory=true`. Its prerequisite exercises exact reconstruction, corruption checks, encoder-byte compatibility, complete byte accounting and the pinned sample study before the full inventory. The full job downloads and verifies one shard at a time, records all tensor groups including vision and MTP, and deletes checkpoint shards as it finishes them.

The earlier 24.88120 GiB sample-based projection is superseded by this complete measurement. Matched quality evaluation and compact-format GPU execution remain separate gates.
