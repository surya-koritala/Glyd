# FP4 checkpoint validation

One RTX 5090 using source 6222f68d85, API 18. The full source hash is in COMMIT; model-meta.json pins the five 2026 checkpoint revisions. Kernel/reference/graph checks and tensor scans completed. Whole-model outcomes differ by checkpoint.

| Checkpoint | Recorded outcome |
| :-- | :-- |
| Qwen3.8-27B NVFP4 | Stock 19.92 GiB, default Glyd 18.45 GiB, exact-mode Glyd 18.59 GiB. Exact text, prompt-logprob and image comparisons match stock eager; 111 BF16 packs and 128 forced code packs verify. Default code admission found no qualifying code compression; normal savings come from BF16 weights and scales. |
| Gemma 4 31B NVFP4 | Normal compilation runs out of memory. Exact comparison matches the Glyd BF16-parts-only exact reference, not stock eager. Normal serving is not validated. |
| Qwen3.6-35B-A3B NVFP4 | Stock 20.37 GiB, normal Glyd 19.25 GiB, exact-mode load 19.39 GiB. All 160 forced code and 160 scale packs verify at load. No stock-eager output reference was configured, so end-to-end exactness is unmeasured. One admitted expert-code pack increases net memory after its scratch buffer; admission needs correction. |
| Nemotron 3.5 Lightning NVFP4 | Stock loads at 17.82 GiB; all three Glyd starts fail with an unhashable-list warning error. A later host-tested fix requires GPU validation. |
| AMD Qwen3.8-27B MXFP4 | Tensor/code-bit checks only; no serving claim. |

This is an incomplete coverage result, not release acceptance. No six-load speed claim follows. Exact-mode comparisons and normal-mode measurements are separate; forced code checks can increase memory and do not describe ordinary admission.

Pod nv1e8pz64eec5u was deleted at 22:29 UTC; launcher.log records deletion and an empty pod list. Ledger duration 4,260 seconds at $0.99/hour gives an estimated GPU charge of $1.1715, excluding storage and other fees. Including the previous failed setup, the FP4 GPU estimate is $1.231725. The source was isolated FP4 code, not the subsequent combined integration.
