# Quantized integration and layer validation

Source e3f91e3d, one RTX 5090, vLLM 0.30.0. GPU scale round trips pass. The whole-model performance harness ran with `scales=false`: its results measure BF16-parts-only compression, not compressed scales. Separate layer checks used scale verification. Original jobs, options, logs and summaries are retained.

| Checkpoint | Stock weight GiB | Glyd weight GiB | Layer/weight evidence |
| :-- | --: | --: | :-- |
| Qwen3.5-35B-A3B GPTQ INT4 | 21.12 | 19.46 | 1,471/1,471 layer calls; 271/271 packs and 837/837 other tensors exact. End-to-end equality passes in the recorded batch-invariant mode. |
| Qwen3.6-35B-A3B NVFP4 | 20.37 | 19.84 | 671/671 layer calls; 111/111 packs and 1,319/1,319 other tensors exact. End-to-end equality remains unestablished. |

Ordinary stock execution differs on repeated MoE calls: 96/200 for GPTQ and 86/200 for NVFP4. Raw exact-mode disagreements remain in the record; the layer checks do not replace end-to-end acceptance. The performance inputs request checkpoint `main`, so checkpoint revision pinning remains a reproducibility gap.

Pod k9slfciopzp2vb completed and was deleted at 20:05 UTC; launcher.log records an empty pod list. The guard recorded 4,968 seconds at $0.99/hour, an estimated $1.3662 GPU charge excluding storage and other fees. This record establishes no scales-enabled whole-model speed/memory result, all-load speed claim or current-head release acceptance.
