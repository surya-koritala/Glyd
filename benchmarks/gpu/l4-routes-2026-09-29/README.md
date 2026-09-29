# A prompt's routes on an L4, 2026-09-29

The AWS dev machine's NVIDIA L4 (g6.4xlarge: 24 GB, 72 W cap, 2040 MHz at most; AMD EPYC 7R13, 16 vCPUs; driver
595.91.07, torch 2.14.0+cu130, transformers 5.17.0), Qwen3-8B and Qwen3-4B-Instruct-2507 from its cache.
`route_e2e.py` times one forward pass over a prompt (logits_to_keep=1: the pass before the first token). It runs Glyd
as `glyd.from_pretrained` loads it (q, k, v and gate, up merged, eager) with each route forced on every GLinear, and
bf16 in its own process. The routes:

- **fused**: the prompt kernel.
- **decoded**: each matrix decoded whole, then cuBLAS, on the current stream.
- **ahead**: each matrix decoded ahead on a second stream beside the products before it (model.Ahead).
- **default**: the library's own routes.

Each timing is the median of 3 (5 in main-fine/) after 2 untimed passes. nvidia-smi samples every 100 ms beside each
run, and each cell gives its window's SM clock and power.

- `main/` (`run1.sh`): v0.25.0's tree (origin/main f62a9d3) at 128-8192 tokens, fused, decoded and ahead, both layouts.
- `main-fine/` (`run2.sh`): fused against decoded at 896-3072 tokens, where the two cross.
- `l4-routes/` (`run3.sh`): this branch (72a46e2), the L4 its own class (3089). The library's routes (default)
  against bf16; before them, on the L4, check_capi (7043 calls bit for bit, 240048 routes as the rule, the L4's and
  the L40S's GLinear routing), test_gpu.py and the glyd-gpu crate's tests, all passing.

The L4 ran every route at its 72 W cap from about 512 tokens up. The fused kernel ran at 1200-1360 MHz; cuBLAS behind
a decode ran at 1050-1155 MHz; bf16's cuBLAS at 880-1110 MHz. The fused kernel still lost to the decode from 896
tokens in the tiered layout and from 2560 in the 12-bit one, and lost more the longer the prompt. It decodes each weight
again for every 256 tokens. Decoding ahead, beside cuBLAS, was no faster than decoding first on the current stream
(Qwen3-8B tiered: 1.08-1.49x its time at 2048-8192 tokens). So the L4 takes the route DECODE: from 896 tokens tiered
and 2560 12-bit, not exact's.

Qwen3-8B, a prompt's pass over bf16's time in the same run (main/, then l4-routes/ for "now"):

| Prompt | 128 | 512 | 1024 | 2048 | 4096 | 8192 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiered, fused (was) | +5.6% | +17.6% | +27.8% | +31.0% | +37.9% | +98.4% |
| tiered, decoded ahead | +48.7% | +23.7% | +26.8% | +20.4% | +20.1% | +56.5% |
| tiered, decoded (now from 896) | +105.4% | +37.5% | +25.5% | +11.3% | +8.4% | +4.7% |
| **tiered, now** | +4.6% | +15.7% | +20.5% | +6.8% | +4.1% | -0.0% |
| 12-bit, fused (was, and now to 2559) | -9.0% | -0.8% | +7.2% | +11.0% | +19.8% | +105.0% |
| **12-bit, now** | -10.5% | -6.6% | +1.9% | +4.3% | +5.8% | +2.4% |

Qwen3-4B-Instruct-2507, tiered, now: +3.1 / +23.1 / +10.7 / +10.8 / +5.8 / +2.3%. Before: fused +4.6 / +25.8 / +9.5
/ +30.9 / +33.2 / +43.0%. In the 12-bit layout, now: -14.0 / +10.2 / -11.6 / +10.5 / +6.3 / +1.9%.

Where the fused and decoded routes cross (main-fine/, fused against decoded):
- **Tiered:** decoded was 4-5% slower at 768 tokens (main/), then 9-11% faster at 896. At 1024 Qwen3-4B's two were
  even; at 1280-3072 decoded was 7-21% faster.
- **12-bit:** decoded was 0.4-21% slower to 2304 tokens, 1-3% faster at 2560, 3-5% at 3072 and 7-49% at 4096-8192.

The 12-bit layout's own fused kernel takes the L4's prompts in less time than the tiered layout's, at every length to
2304 tokens (6-18% less, both models).
