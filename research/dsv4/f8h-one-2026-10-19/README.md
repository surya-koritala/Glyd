# The layout H one-token FP8 product: where its instructions go, and a leaner kernel (2026-10-19)

The one-token FP8 products are the biggest part of a DeepSeek-V4-Flash layer on an H100 (123 of ~290 us, at 27% of the
memory bandwidth) and all of GLM-5.3-Flash's experts. On the A100 (`../a100-stalls-2026-10-19`) they ran at ~25 lane
instructions a byte of weights in layout C; layout H brought that down; this is the per-line profile of layout H and the
first kernel changes from it. RTX 4080 SUPER, Nsight Compute 2025.4.1 (`-lineinfo` build), DeepSeek dense dims with
random weights whose exponents are spread like trained ones (`dsv4random --real`), one token.

## Where a warp-step goes (`f8_linear_kernel`, layout H, M = 1)

A warp-step is one 16 x 128 row block (2,048 values, ~1.8 KB in layout H). 854 warp instructions a step =
15 lane instructions a byte; an H100 SXM issues ~10 a byte at its bandwidth, a B200 ~4.6.

| Part | Instructions a step | Share |
|---|---|---|
| Decode: 3-bit codes to nibbles (`f8_spread3`) 72, escape marks 36, the table permutes 25, the literals into the escapes' bytes ~97, loads and scans ~40 | ~430 | 50% |
| Loop structure: three 64-bit cursors, `mine`, the CTA barrier and the copies' bookkeeping, re-reading lane ids under register pressure | ~206 | 24% |
| e4m3 bytes to bf16 pairs for the tensor cores (`f8_bf_lo/hi`: prmt, shift, mask) | ~89 | 10% |
| The products: X loads, `f8_x64`, 8 MMAs, the block scale | ~68 | 8% |
| The copies in (cp.async of core, literals, X, heads) | ~56 | 7% |

`f8_linear_kernel-lines.txt` has the top lines. The decode is per value (6.7 lane instructions a value), the structure
per step; the escape path (2.9% of values on DeepSeek's dense matrices) is taken on nearly every warp-step because some
lane of the 32 has one.

## The changes (branch f8h-one), the same digits everywhere

1. `f8h_lin1_kernel`: the one-token product on `f8_linear_kernel`'s plan (the CTA's patches, the sums of its first and
   last row group in `parts`, the last CTA's add), so its results are the same to the bit. Each warp copies its own
   head (two steps ahead, to registers), core, literals and X (a step ahead, to its own slots): no CTA barrier a step,
   32-bit cursors. 78-80 registers with 3 CTAs an SM (`GLYD_F8H_CTAS=2`: 95/114 registers, 2 an SM).
2. `f8h_granule`: an escape's mark by one add (`(y + 0x11111111) & 0x88888888`: code 7 + 1 carries into bit 3), the
   table's escape byte 0 and the literals ORed in (an unmarked byte's selector 8 more: prmt's sign mode on a literal
   gives 0), so no mask and no select per register.

| One token on the 4080 SUPER | `f8_linear_kernel` | `f8h_lin1_kernel` | |
|---|---|---|---|
| DeepSeek dense 8192 x 4096: warp instructions | 14.00 M | 12.85 M | -8% |
| the same, time | 51.6 us | 50.5 us | the 4080 reads 29.7 MB at ~590 GB/s either way: memory-bound |
| GLM experts' product (4 layers, 32 experts, 8 slots): instructions | 53.1 M | 50.0 M (2 CTAs/SM), 53.1 M (3) | the register cap costs the experts' kernel 6% |
| DeepSeek 3-layer step (graph) | 2.365 ms | 2.366 ms | |
| GLM 4-layer step (graph) | 3.961 ms | 3.963 ms | |

Checks: `dsv4check` 11 of 24, 2.4437; `f8moecheck --peaked` 0.489 and 0.446 bf16 steps; both kernels, both occupancy
settings. The 4080 cannot show the gain: it is at its bandwidth, and it holds 2 CTAs of either kernel an SM (38 KB of
shared memory a CTA; an H100 holds 6, an A100 4).

## What is left in the decode

- The code spread (72 a step) is the price of 3-bit codes read by 4-bit permute selectors; a cheaper spread would need
  the codes stored differently.
- The literal insertion (~97 a step) could be a sparse correction instead (the escapes' values as scalar FMAs on the
  accumulators: ~20 a step), at the cost of bit identity with the old kernel (fp32 order).
- The conversion (89 a step): the format could hold the sign and mantissa nibble pre-shifted, so a 16-bit lane is two
  bytes assembled by one permute: ~0.5 instructions a value less.
- The loop structure (~200 a step) is mostly the compiler's: the cursors and the heads' rotation as register moves.

## H100 SXM (Lambda gpu_1x_h100_sxm5, us-south-2)

The Lambda images there now carry driver 570 (CUDA 12.8) on Ubuntu 22.04; the PC's build needs glibc 2.39 and CUDA 13, so the
session runs in the `nvidia/cuda:13.0.3-devel-ubuntu24.04` container with its forward-compatibility libcuda (copied aside first:
installing Nsight Compute replaces it with one the 570 kernel module refuses, cuInit error 100, which ended the third session
after its first runs). `sudo nvidia-modprobe -u -c=0` on the host first, or the container has no /dev/nvidia-uvm.

The third session's runs before the break, `f8_linear_kernel` with the new `f8h_granule` (the escape marks and the literal OR),
layout H, graphs, 32 steps after 256 tokens: DeepSeek 3-layer step 1.383 / 1.389 ms (1.400 the day before: within noise),
GLM 4-layer step 2.016 ms (2.074 before: -2.8% from the decode change alone, all of it FP8 products).

The fourth and fifth sessions (one H100, one A100 80 GB) died the same way at the same point: ~4 minutes after boot the
container loses the GPU (cuInit error 100) while the host's nvidia-smi is fine, no Xid, no module reload. That is the
known docker + systemd failure: a `daemon-reload` on the host (cloud-init's last steps here) re-applies the device
cgroups and drops the GPU from running containers. `--privileged` on `docker run` fixes it; the fifth session was
restarted that way on its A100 80 GB and ran through. Five launches, about $3 (`a100/launch.sh`, `job.sh`, `inner.sh`).

## A100 80 GB (Lambda gpu_1x_a100_sxm4, us-east-1), Nsight Compute 2025.3.1, clocks locked, three repeats each

Every check the same digits with both kernels and both occupancy settings (`a100/check-*.txt`, `f8moe*.txt`).

| One token, layout H | `f8_linear_kernel` | `f8h_lin1_kernel`, 2 CTAs/SM | `f8h_lin1_kernel`, 3 CTAs/SM | |
|---|---|---|---|---|
| DeepSeek dense 8192 x 4096 (29.7 MB) | 58.2 us, 510 GB/s | 54.9, 541 | **52.9, 561** | -9% |
| DeepSeek dense, the second (29.8 MB) | 56.2-57.6 | 54.0-54.2 | **51.3-51.6, 580** | -10% |
| DeepSeek experts' product (8 slots, 29.8 MB) | 58.1-58.5 | 55.3-56.9 | **54.4-54.8** | -6.5% |
| GLM experts gate/up (118.8 MB) | 189.2 us, 628 GB/s | 176.2, 674 | **171.3-173.9, 690** | -9% |
| GLM experts down (59.5 MB) | 98.8 | 94.7 | **93.8** | -5% |
| 14.9 MB and 7.5 MB dense | 34.5, 25.3 | 32.6, 24.0 | 32.5, 23.4 | -6% |
| registers / warps active / issue slots used | 107-127 / 22.6% / 55-58% | 95-114 / 22.8% / 55-61% | 78-79 / 31% / 57-64% | |

(`a100/ncu-*.txt`; `a100/l1-details.txt` is the 3-CTA kernel's full report: 5.0 warps a scheduler, 1.6 eligible, 8.4
cycles an issued instruction.) The step-level timings on that host were too noisy to read (12% swings between repeats
of the same build); the per-kernel figures above are the record.

So on HBM the leaner kernel is 5-10% faster, the 3-CTA setting best (now the default; `GLYD_F8H_CTAS=2` the other). At
560-690 GB/s the A100 80 GB is at 28-34% of its bandwidth with issue slots 60% used: the kernel waits on dependencies
and on its own pipeline's fill (a CTA has ~6 steps of work after 2 steps of latency at these grids) more than on issue.
Next on this kernel: a deeper prefetch (copies two steps ahead) and fewer, longer-lived CTAs, then the decode's
instruction count (the literal path and the code spread).

## Session 6, A100 40 GB (Lambda gpu_1x_a100_sxm4 us-east-1 gave the 40 GB card this time; Nsight Compute 2025.3.1; `a100-40gb-session6/`)

`f8h_lin2_kernel` (GLYD_F8H_ONE=2): the copies two steps ahead in the same two core slots (a step's 14 core words a lane go to registers first, then its
slot takes step j + 2's core), three small slots for a step's literals, X and head (the heads through shared memory: two heads a warp in registers,
not three); 43.3 KB a CTA (A100 3 an SM, H100 5, 4080 2). The same digits as the others (every check). It is not faster:

| One token, layout H, A100 40 GB | `f8_linear_kernel` | `f8h_lin1_kernel` | `f8h_lin2_kernel`, 3 CTAs/SM | 2 CTAs/SM |
|---|---|---|---|---|
| DeepSeek dense 8192 x 4096 (29.7 MB) | 60.1 us | **56.2** | 57.8 | 58.0 |
| GLM experts gate/up (118.8 MB) | 150.1 | **140.7** | 142.1 | 141.6 |
| GLM experts down (59.5 MB) | 33.8 | 33.2 | 34.4 | **32.5** |
| warp instructions, dense | 13.61 M | 12.92 M | 13.32 M | 13.19 M |

Where the warps wait (`ncu-ds-*.txt`, stalls as warps per issue-active cycle; pipes as % of each pipe's peak), the dense product:

| | issue active | warps active | ALU | FMA | XU | LSU | not selected | math pipe throttle | wait | long scoreboard | short scoreboard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `f8h_lin1_kernel`, 3/SM | 57.1% | 31.9% | 60.1% | 27.9% | 14.8% | 3.8% | 1.76 | 1.39 | 1.38 | 1.62 | 0.38 |
| `f8h_lin2_kernel`, 3/SM | 57.1% | 31.9% | 59.9% | 26.5% | 14.4% | 4.1% | 1.73 | 1.38 | 1.47 | 1.18 | 0.40 |
| `f8h_lin2_kernel`, 2/SM | 56.5% | 22.7% | 59.6% | 26.4% | 14.0% | 4.1% | 1.06 | 0.83 | 1.43 | 0.55 | 0.38 |
| `f8_linear_kernel` | 55.6% | 22.9% | 59.1% | 26.2% | 13.3% | 3.5% | 1.03 | 0.81 | 1.41 | 0.54 | 0.55 |

The deeper prefetch does what it should to the memory wait (long scoreboard 1.62 -> 1.18 at 3 CTAs, 0.55 at 2) and the time does not move: the
issue side is the limit. The issue slot is used 57% of the cycles with 5-6 warps a scheduler, the ALU pipe 60% of its peak, and the stalls that
grow are the ones of warps that are ready but not picked (not selected, math pipe throttle) and of dependent instructions (wait). So on the A100
the one-token product of layout H is instruction-bound: ~13 M warp instructions for 29.7 MB, 0.44 a byte; the step timings of the engine agree
(DeepSeek 2.50 ms with either kernel, GLM 3.2-3.7 ms: noise). What is left is fewer instructions a value (the decode is 6.7 a value, ~5 at
best for this format) or more warps (the decode's registers cap it at 3 CTAs of 8 warps).

The raw bytes as the measure of the ceiling: a patch stored plain (every value its e4m3 byte, 8 bits a value, the release's size) needs no
decode: a step is the copies, the bf16 conversion, X and 8 MMAs, ~360 warp instructions for 2,048 bytes against ~815 for 1,792. Session 7
measures it: `GLYD_F8H_PLAIN=1` packs every patch plain, and `f8h_lin2_kernel` copies a plain row block through its slot (4 x 16 bytes a lane,
two steps ahead) instead of 16 loads a lane from global memory at the decode.

## Session 7, A100 40 GB: the same kernel fed plain bytes (`a100-40gb-session7-plain/`, ~$1.30)

`GLYD_F8H_PLAIN=1` packs every patch plain (each value its e4m3 byte: 8 bits, the release's size) and `f8h_lin2_kernel` copies a plain row block
through its slot like a coded one (4 x 16 bytes a lane, two steps ahead); the digits are the same for every variant (`check-*`, `f8moe*`). No
decode at all, 45% fewer instructions, and the time does not move:

| One token, A100 40 GB, ncu | layout H, `f8h_lin1` | layout H, `f8h_lin2` | plain bytes, `f8h_lin2` | plain, `f8h_lin1` (16 loads a lane at the decode) | plain, `f8_linear_kernel` |
|---|---|---|---|---|---|
| DeepSeek dense 8192 x 4096 | **56.1 us** (29.7 MB, 530 GB/s, 12.9 M instr., issue 57%) | 59.9 (13.9 M) | **56.4 us** (33.7 MB, 598 GB/s, 7.1 M instr., issue 37%) | 61.6 (616 GB/s) | 69.3 |
| GLM experts gate/up | **140.4 us** (89 MB, 635 GB/s, 38.1 M) | 155.8 | **138.6 us** (102 MB, 736 GB/s, 21.1 M, issue 38%) | 156.6 | 167.4 |
| GLM experts down | 32.7 | 34.8 | 30.5 | 36.1 | 40.4 |

The plain product's stalls (`ncu-ds-2-3-1.txt`): long scoreboard 2.5, MIO throttle 2.35, LG throttle 1.6, short scoreboard 1.6, wait 1.9; DRAM
42% of peak, L2 55%, "Mem Busy" 55%, issue 37%. With the decode gone the memory instruction queues and the data wait are what is left, at 600-740
GB/s of the card's 1,555. So on this GPU the kernel's structure caps the product at 40-45% of the bandwidth whatever the bytes are: the decode of
layout H is hidden under that cap today (its 11% fewer bytes cost nothing), and the structure is the thing to fix, for coded and plain alike. The
step timings of the engine (`ds-*`, `glm-*`) are the usual noise (2.38-2.58 ms, 3.40-3.70 ms).

Session 8 measures the structure alone (`stream.cu`, compiled on the instance): the same bytes read by a grid-stride stream, by the engine's order
with plain loads one and two steps ahead, by the engine's cp.async mechanism, with the two halves adjacent, interleaved across CTAs, 16 warps a
CTA, 6 CTAs an SM. On the 4080 SUPER every pattern reads at 89-93% of its 736 GB/s (the card is not the problem); the A100's numbers are the
decision.

## Session 8, A100 40 GB: the byte stream alone (`a100-40gb-session8-stream/`, `stream.cu`, ~$0.40)

The same bytes read by different orders and mechanisms, no products: a grid-stride stream (4 or 8 loads of 16 bytes a thread in flight); the
engine's order (a CTA's contiguous range of units, the two halves 256 KB apart) with plain loads a step ahead; the engine's order and mechanism
(cp.async into two shared-memory slots a warp, wait_group 1, 47 KB a CTA, 3 CTAs an SM); the halves adjacent; the units interleaved across CTAs;
16 warps a CTA; 6 CTAs an SM. Reps rotate over 8 x L2 of copies (the first run on the 4080 read 2.3 TB/s from L2). Medians of two runs:

| A100 40 GB (1,555 GB/s) | 121.6 MB (GLM gate/up's size) | 33.6 MB (DeepSeek dense) | 29.4 MB |
|---|---|---|---|
| grid-stride stream | **1,291-1,305 GB/s** (83%) | 886 | 896 |
| engine's order, plain loads a step ahead (`ours1`) | 958-982 | | 755 |
| engine's order and mechanism, cp.async (`cpasync`) | **1,291** | | **1,024** |
| halves adjacent, plain loads | 966-990 | | 755 |
| interleaved units, plain loads | 1,291 | | 1,024 |
| 16 warps a CTA, interleaved | 1,277 | | 989 |
| 6 CTAs an SM, engine's order, plain loads | 848-861 | | 683 |
| 6 CTAs an SM, interleaved | 1,291 | | 1,024 |

(The "two steps ahead in registers" rows are not valid: a register array indexed by the step went to local memory.) On the 4080 SUPER every
pattern reads at 89-93% of its 736 GB/s.

So the engine's order with its cp.async mechanism streams at the card's rate (83% of peak on 121 MB; 66% on 30 MB, where ~10 us of launch
and tail are a third of the time) when nothing else is in the step. The real kernel reads the same bytes at 600-740 GB/s with the same order and
mechanism: the difference is the step's work. Its copies for the step after next are issued before the step's products, and session 7's stalls
(LG and MIO throttle) say the warps queue on those copy instructions instead of computing: memory time and compute time add instead of
overlapping. Session 9 measures the one-line change: the copies issued after the products (`f8h_lin1_kernel` and `f8h_lin2_kernel`, coded and plain).

## Session 9, A100 40 GB: the copies after the products (`a100-40gb-session9-late/`, ~$1.30)

Both kernels issue the step after next's copies after the step's products. No gain: dense `f8h_lin1` coded 55.4 us (56.1 before), `f8h_lin2`
plain 53.7 (56.4), `f8h_lin2` coded 65.1 (59.9); GLM gate/up 141.8 / 141.4 / 168.7. The same digits. So the copy issue was not what held the
products back.

What does: the SASS of the plain `f8h_lin2_kernel` (`sass-ds-2-3-1.csv.gz`, Nsight's per-instruction samples), 7.48 M warp instructions for the
dense product = 456 a warp-step, by opcode: IMAD 29.4%, LOP3 11.8%, PRMT 7.0%, IADD3 6.3%, ISETP 6.2%, BRA 4.5%, LDGSTS 4.3%, LDS 3.9%, HFMA2 3.5%,
SHF 3.5%, BSYNC 2.5%, LEA 2.1%, BSSY 2.0%, HMMA 1.8%; local-memory stores and loads 1.2% of the instructions but 9% of the stall samples. The data
path (8 MMAs, the 16 X conversions, the 32 permutes and ~40 logic ops of the bf16 conversion, 8 shared loads, 6 copies) is ~110 of the 456; the
rest is address arithmetic, cursors, predicates, branches and convergence barriers around divergent lambdas, and a few spills under the 80-register
cap: the step's structure, which is the same with the coded patches. The stall samples sit on that integer work (IMAD 18%, IADD3, ISETP, BRA) and
on the spills, not on the MMAs.

Session 10 measures a kernel written for the plain pack alone (`f8b_lin1_kernel`, GLYD_F8H_ONE=3): the same plan and sums, two core slots and two
X slots a warp, a step's words to registers then its slots refilled two steps ahead, pointers and cursors and nothing else: ~110 instructions a
step if the compiler keeps it so. Its time against the stream's 1,024 GB/s at this size says what the structure can reach; the coded decode then has to
fit in what is left.

## Session 10, A100 40 GB: the lean kernel on plain bytes (`a100-40gb-session10-lean/`, ~$1.30)

`f8b_lin1_kernel` (GLYD_F8H_ONE=3, plain packs): f8h_lin1_kernel's plan and sums, two core slots and two X slots a warp, a step's words to
registers, the slots refilled two steps ahead, the copies coalesced (512 contiguous bytes an instruction) and put where the lane's reads hit eight
bank groups; 64 registers, no spills (67 at 2 CTAs an SM). The same digits as every other variant (`check-*`, `f8moe*`, and the 4080's
`pc-check-4080.txt`).

| One token, A100 40 GB, ncu | layout H, `f8h_lin1` (today's) | plain, `f8h_lin2` | **plain, `f8b_lin1`, 3 CTAs/SM** | 2 CTAs/SM | the stream alone (session 8) |
|---|---|---|---|---|---|
| DeepSeek dense 8192 x 4096 | 55.3 us, 537 GB/s, 12.9 M instr. | 52.4, 645 | **41.5 us, 813 GB/s, 6.3 M** | **39.6, 851** | 28.7-32.8 us (1,024-896) |
| GLM experts gate/up | 141.0 us, 633 GB/s | 140.5, 728 | **102.5 us, 988 GB/s** | **101.4, 999** | ~94 (1,291) |
| GLM experts down | 32.3 | 29.9 | 25.6 | 25.5 | |
| `f8moecheck` experts, free-running clocks | 52.3 us (567 GB/s) | 55.6 (682) | **33.9 (1,118)** | **33.4 (1,133)** | |
| engine step, DeepSeek 3 layers (graph) | 2.50-2.57 ms | 2.50 | 2.29-2.32 | 2.31 | |
| engine step, GLM 4 layers (graph) | 3.42-3.57 ms | 3.55 | 3.06-3.17 | 2.96-3.20 | |

The lean step is 368 warp instructions (the data path ~110; the compiler's rest), issue 42-44%, and its stalls are the data wait (long scoreboard
4.8-5.3) and the row-group barrier (1.7): the memory side is now what it waits on, 80-83% of what the stream alone reads at these sizes. So the
step's structure was worth 28% on the products and 8-10% on the engine's step on this card, with raw bytes.

What this decides. On the A100 the products of the 7.1-bit layout run at 55% of the raw bytes' speed and the gap is instructions (the decode is
~430 a step; the lean step has ~110 to spare at 42% issue). The choice for datacenter GPUs is therefore: layout H on disk and in the download (11%
smaller than the release), and in VRAM either raw bytes where they fit (the FP8 parts at the release's size; the products at 80-90% of the stream) or
layout H where they do not (GLM-5.3-Flash's FP8 experts: ~270 GB coded, 321 GB raw: 4 x 80 GB holds the coded form only). The engine can choose
at load from the memory it has. Next for the kernels: the lean structure under the coded decode (`f8h_lin1`'s 206 instructions of structure a
step are the same waste), then the FP4 experts' one-token kernel, which has the same structure; the H100's numbers when it has capacity.

## Session 11, A100 40 GB: the lean step under the coded decode (`a100-40gb-session11-lin3/`, ~$1.30)

`f8h_lin3_kernel` (GLYD_F8H_ONE=4): f8h_lin2's work with f8b's step (two core slots, two small slots filled a step ahead, three heads a warp in
registers, X copied by all 32 lanes). The same digits everywhere, plain patches included. No gain for the coded layout:

| One token, A100 40 GB, ncu | `f8h_lin1` (today's) | `f8h_lin3`, 2 CTAs/SM | `f8h_lin3`, 3 CTAs/SM (80 registers) | `f8h_lin3` on plain packs | `f8b_lin1` on plain packs |
|---|---|---|---|---|---|
| DeepSeek dense 8192 x 4096 | 55.5 us, 12.9 M instr., issue 58% | 56.5 us, 13.2 M, 122 regs | 62.6 us, 14.7 M | 43.5 us (776 GB/s) | **41.1 us (820 GB/s)** |
| GLM experts gate/up | 142.2 us | 141.0 | 161.7 | 107.9 | **102.4** |
| `f8moecheck` experts, free-running | 51.9 us | 55.5 | 65.3 | 42.7 | **33.7** |

With the coded decode in the step, the lean structure buys nothing: the product is instruction-bound at 57-62% issue and the decode's ~430
instructions a step are the body; the structure's saving is small beside it, and under the 80-register cap the compiler's spills give it
back. On plain packs `f8h_lin3` is within 6% of `f8b_lin1`, so one kernel can serve both pack kinds.

So the question GLY-349's first step asked is answered: on the A100, layout H's one-token product cannot come within 15% of the raw bytes'
without a cheaper decode (fewer instructions a value: a format change). The speed path for datacenter GPUs is raw e4m3 bytes in VRAM where they
fit (layout H on disk and in the download; layout H in VRAM where the raw bytes do not fit); next is the FP4 experts' kernel, where the decode is
lighter a byte (12-13 lane instructions a byte against layout H's ~25) and the step's structure may matter more.

## Session 12, 2x H100 SXM (Lambda gpu_2x_h100_sxm5 us-southeast-1, driver 580.105, the container route; `h100x2-session12/`, ~$4.40)

The same digits for every variant. The kernels (ncu, medians; the H100's peak 3,350 GB/s):

| One token, H100 | `f8h_lin1` (today's, layout H) | `f8b_lin1` (lean, raw bytes) | `f8h_lin3` on plain packs | FP4 experts `mma_moe_kernel<Fp4R>` (today's) |
|---|---|---|---|---|
| DeepSeek dense 8192 x 4096 | 33.8 us, 879 GB/s, 13.4 M instr., issue 64% | **21.6 us, 1,559 GB/s**, 6.8 M, issue 53% | 27.0 us, 1,249 GB/s | 42.8 us, 38.4 MB, **898 GB/s**, 14.7 M, 114 regs, warps 20%, issue 51% |
| GLM experts gate/up | 85.6 us, 1,041 GB/s | **52.4 us, 1,934 GB/s** | 59.3 us, 1,708 | |
| engine step, DeepSeek 3 layers / GLM 4 layers (graph) | 1.37 / 1.99 ms | **1.28 / 1.79 ms** | 1.34 / 1.90 | |

The real DeepSeek-V4-Flash on the two GPUs (149 GB download, split 22 | 21 layers, load ~300 s), the FP8 parts packed at load:

| | layout H in VRAM | raw FP8 bytes in VRAM (`GLYD_F8H_PLAIN=1`, `GLYD_F8H_ONE=3`) |
|---|---|---|
| weights on GPU 0 | 73.36 GiB (74.16 of 79.18 in use) | 73.97 GiB (74.98 in use): +0.6 GiB |
| decode | 14.80 ms a token, **67.6 a second** | 13.31 ms a token, **75.2 a second** (+11%) |
| score, the 517 tokens | perplexity 2.226 (mean log-probability -0.8002) | 2.228 (-0.8010) |

The two scores differ in the third digit, and the session of 2026-10-19 (the older build, layout H) gave 2.242 on the same tokens: the whole-model
run is not bit-stable across builds or runs, while every kernel check on random weights is exact to the digit. The products are not the cause (the
same sums in the same order); the candidates are the CPU side of the routing and the attention path. To be found before the release claims
"the same digits every run" for this model.

What this settles: on the H100 the one-token FP8 products with raw bytes read at 1.5-1.9 TB/s (45-58% of peak) against 0.9-1.0 TB/s with the
coded layout, and the real model gains 11% tokens a second for 0.6 GiB a GPU. The FP4 experts' product is now the slowest part of a DeepSeek
layer (898 GB/s, 27% of peak, 20% warps active at 114 registers): the four-steps-a-stage variant of its ring is the next measurement.

## Session 13, A100 40 GB: the new defaults, and the FP4 experts' ring (`a100-40gb-session13-defaults/`, ~$1.30)

With no environment set, an HBM GPU now packs layout H plain and runs `f8b_lin1_kernel` (`GLYD_F8H_PLAIN=0` keeps the coded layout in VRAM; a GDDR GPU
keeps it by default). Same digits everywhere. DeepSeek dense product 41.5 us (coded 56.3), experts' check 33.8 us (coded 52.3), the 3-layer step
2.325 ms (coded 2.495). The FP4 experts' one-token product through the shared-memory ring is slower than without it: 69.2 us unstaged, 76.3 with
the 8-slot ring, 78.9 and 79.2 with four steps a stage (104 -> 114-119 registers); four steps a stage is no better than one. The FP4 kernel's
limit is its step's structure and its 104 registers (22% of the warps active), as the FP8 kernel's was: the next step is a lean FP4 kernel on
`f8b_lin1_kernel`'s pattern. Known gap of the new default: it does not look at memory, so a model whose raw FP8 bytes do not fit (GLM-5.3-Flash's
FP8 experts on 4 x 80 GB) needs `GLYD_F8H_PLAIN=0` or the engine's choice at load (GLY-349 step 3).
