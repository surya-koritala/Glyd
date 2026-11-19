# Initial native lossy quality comparison

Historical Qwen3.8-27B language-likelihood screen on one A100 80 GB: eight fixed windows, 2,048 scored tokens per arm. Source, pinned checkpoint, input hashes and individual target scores are retained in attempt-2. Attempt-1 failed dependency setup before downloading the model or scoring tokens; both pods were deleted.

| Arm | Perplexity | Change from native BF16 control | Next-token agreement |
| :-- | --: | --: | --: |
| Native BF16 control | 6.9308 | — | 100% |
| Candidate A | 6.9223 | -0.1212% | 95.7031% |
| Candidate B | 7.0196 | +1.2819% | 93.2129% |

Candidate A's window-bootstrap 95% perplexity-change interval is -0.5487% to +0.3103%; this does not establish an improvement. Candidate B's interval is +0.5231% to +2.1230%. These are language-likelihood and token-agreement measurements, not task-accuracy percentages. The native BF16 control passed its predetermined comparison with the independent BF16 reference; it was not bit-exact across runtimes.

The 51.9175% / 58.1665% storage savings in outcome.json were measured on 51 windows from 23 tensors, not the complete checkpoint. All native arms still used the existing GPU representation, around 37.70 GiB. No compact-format resident VRAM, FP8 comparison or serving-speed result was measured here. Vision and MTP were excluded from this language pass. Candidate A's later complete weight inventory and matched FP8 / broader task records supersede the sample extrapolation and extend this initial screen.

Guard estimates: attempt-1 $0.0538833 (122 seconds) and attempt-2 $0.3038667 (688 seconds), both at $1.59/hour; combined $0.35775, excluding storage and other fees. Source-provenance and deletion receipts are retained. BUNDLE-SHA256SUMS are original input receipts and include binaries that are not stored in this record; the top-level SHA256SUMS covers the retained files. Original device outputs are unchanged. Local status prose and its local-only manifest are omitted.
