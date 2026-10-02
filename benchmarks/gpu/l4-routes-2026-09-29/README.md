# A prompt's time on an L4 and an L40S, 2026-09-29

The AWS dev machine's NVIDIA L4 (g6.4xlarge: 24 GB, 72 W cap, 2040 MHz at most; AMD EPYC 7R13, 16 vCPUs; driver
595.91.07, torch 2.14.0+cu130, transformers 5.17.0), Qwen3-8B and Qwen3-4B-Instruct-2507 from its cache.
`route_e2e.py` times one forward pass over a prompt (logits_to_keep=1: the pass before the first token). It runs Glyd
as `glyd.from_pretrained` loads it (q, k, v and gate, up merged, eager), and bf16 in its own process. Each timing is
the median of 3 (5 in main-fine/) after 2 untimed passes. nvidia-smi samples every 100 ms beside each run, and each cell
gives its window's SM clock and power.

- `main/` (`run1.sh`): v0.25.0's tree (origin/main f62a9d3) at 128-8192 tokens, both layouts.
- `main-fine/` (`run2.sh`): the same at 896-3072 tokens, in finer steps.
- `l4-routes/` (`run3.sh`): this branch (72a46e2), the L4 its own class (3089). The library's own choices (default)
  against bf16; before them, on the L4, check_capi (7043 calls bit for bit, 240048 cases against the rule), test_gpu.py
  and the glyd-gpu crate's tests, all passing.
- `gap/` (`run4.sh`, 910091c): `gap.py`, a prompt's time to its first token through generate() against a plain forward
  pass in one process (Glyd compiled and not, bf16); then the two ways again for Qwen3-8B, both layouts.
- `l40s/` (`l40s_job.sh`, 213ca87): the same measurement on an L40S (AWS g6e.xlarge: 46 GB, 350 W, 4 vCPUs; Deep
  Learning AMI, driver 595.91.07, AMD EPYC 7R13; the AMI's nvcc 13.0), Qwen3-8B, both layouts, 512-8192 tokens, each run
  from about the GPU's idle temperature (14 minutes).
- `l40s-class/` (1f4343b, the L40S's class): on the L4, the library built by build_lib.sh, then check_capi,
  test_gpu.py and the crate's tests, all passing. The release's head is checked in
  benchmarks/gpu/decode-fix-2026-09-29/l4-head.

The L4 ran at its 72 W cap from about 512 tokens up. For the smallest layout (`mma`) from 896 tokens, and for 12-bit
from 2560, the library now takes a faster path for prompts on the L4 (not for exact mode), by more the longer the
prompt.

Qwen3-8B, a prompt's pass over bf16's time, each row from main/ (one run: "now" is the new path where the library
changed, the earlier one where it did not):

| Prompt | 128 | 512 | 1024 | 2048 | 4096 | 8192 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mma`, v0.25.0 | +5.6% | +17.6% | +27.8% | +31.0% | +37.9% | +98.4% |
| **`mma`, now** | +5.6% | +17.6% | +25.5% | +11.3% | +8.4% | +4.7% |
| 12-bit, v0.25.0 | -9.0% | -0.8% | +7.2% | +11.0% | +19.8% | +105.0% |
| **12-bit, now** | -9.0% | -0.8% | +7.2% | +11.0% | +9.3% | +4.4% |

Qwen3-4B-Instruct-2507 in main/, `mma`, now: +4.6 / +25.8 / +12.9 / +14.0 / +10.1 / +4.3%; before: +4.6 / +25.8 /
+9.5 / +30.9 / +33.2 / +43.0% (at 1024 tokens the earlier pass the faster in this run, even in main-fine/). 12-bit,
now: -13.3 / +7.4 / -8.0 / +9.9 / +6.8 / +5.0%.

l4-routes/ (the library as built, against bf16 in the same run) ran about 10 C cooler than main/ (the Glyd runs' medians
66-72 C against 78-81 C), so its times are lower throughout, the unchanged ones' too: Qwen3-8B `mma` +4.6 / +15.7 /
+20.5 / +6.8 / +4.1 / -0.0%, 12-bit -10.5 / -6.6 / +1.9 / +4.3 / +5.8 / +2.4%. Its rows are for the library as built,
not for "was" against "now".

Each layout as the library runs it, the 12-bit layout's prompts took 5-20% less time than `mma`'s to 1536
tokens and about the same from 1792 (2.4% more to 3.6% less; both models, main/, main-fine/ and l4-routes/).

## generate() against a forward pass, and the L4's temperature (gap/)

Within one process a prompt's time to its first token through generate() was its forward pass's. Qwen3-8B `mma`,
gen1 over forward: 1.005x at 2048 and 8192 tokens compiled, 1.001x and 1.008x not; bf16 1.000x and 0.983x. A
forward pass right after a generate() of 16 tokens took the same time.

The 5-9% seen between the respond job's time to first token and route_e2e.py's pass came from comparing runs at
different temperatures. At its 72 W cap the L4's clock falls as it heats. The same pass of Qwen3-8B (`mma`, 2048
tokens) took:

| run | ms | SM clock | temperature |
| :-- | --: | --: | --: |
| l4-routes/ | 739 | 1148 MHz | 66 C |
| main-fine/ | 748 | 1125 MHz | 71 C |
| main/ | 765 | 1095 MHz | 75 C |
| gap/ | 794 | 1035 MHz | 82 C |

The respond job's Glyd process ran after 20 minutes of bf16's.

## An L40S (l40s/)

The L40S has the L4's bandwidth per FLOP: 350 W for 864 GB/s, where the L4 has 72 W for 300. From 2048 tokens the new
path ran at the GPU's 350 W cap (medians 342-352 W and 1718-1935 MHz, the software power cap set in every sample). From
1024 tokens (`mma`) and 2048 (12-bit) the library now takes it on an L40S, not for exact mode. It is a class of its
own (`GLYD_GPU_L40S`, 4000: an L40S's code is 4089): the L40 and RTX 6000 Ada were not measured. Qwen3-8B, over bf16's
time, before and with the new path (forced on at every length here; the library takes it from the lengths above):

| Prompt | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mma`, before | +20.6% | +27.9% | +38.1% | +34.1% | +41.2% | +37.3% | +39.4% | +35.0% |
| `mma`, new path | +53.5% | +33.7% | +30.3% | +13.9% | +11.9% | +8.0% | +10.9% | +3.8% |
| 12-bit, before | -0.3% | +4.8% | +13.6% | +9.4% | +15.5% | +14.9% | +20.0% | +18.2% |
| 12-bit, new path | +56.3% | +36.5% | +35.0% | +17.8% | +12.9% | +7.7% | +11.3% | +3.9% |

At 512 tokens the 12-bit layout takes bf16's time (-0.3%), where `mma` is +20.6%. Each layout as the library
runs it, the 12-bit layout's prompts took 17.3 / 18.1 / 12.8 / 4.0% less time than `mma`'s at 512 / 768 /
1024 / 1536 tokens and were within 0.9% of them from 2048. The layout stays `mma` (33% less memory); `layout="mma12"`
(25%) for the fastest short prompts.
