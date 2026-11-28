# A decode step's ceiling: the formula

`glyd-engine ceiling MODEL --box B` (a development build, glyd-engine/src/ceiling.rs) computes it. MODEL is a directory or a Hub repo; for a repo, only the
safetensors headers are read, by ranges. The numbers below come from that command.

## Inputs

- **The model:** every tensor's name, dtype, shape and bytes from its checkpoint, and from config.json the layers, experts, top-k and attention
  layout.
- **The engine:**
  - **Packed sizes:** a bf16 matrix packs to 68.9% of its bytes tiered and 75.2% in the 12-bit layout, measured on Kimi-K3's layer 0. The planner's
    72% and 78% are bounds for memory, not for time. FP8 is held at 113% plain or 90% coded; DeepSeek-V4's at 100%, exact. FP4 experts are held as
    released.
  - **Slicing:** a matrix of at least 16 MiB whose rows divide into 64s per GPU is cut into one slice per GPU.
  - **Exchanges:** one per sliced product. q, k and v count as one product, gate and up as one, and each MoE layer's experts add two.
- **The box:**
  - each GPU's memory read rate (measured where we have it);
  - PCIe per GPU when every GPU copies at once;
  - the host's memory read rate.
- **Our code on the box:**
  - each product class's share of the read rate;
  - the fixed time;
  - a small exchange's time;
  - a host pair's time.

## The terms of one token, per GPU

| term | formula |
|---|---|
| weights | Σ class bytes ÷ (read rate × our share for that class). Dense bytes are divided by n where sliced and whole where not; experts are top_k × MoE layers × expert bytes × hit ÷ n. |
| attention | (the cache read at the position + linear attention's states read and written) ÷ read rate. Every GPU does this in full. |
| missing experts | top_k × MoE layers × expert bytes × (1 − hit) ÷ (copy rate + host pair rate). The plan shares the misses so both end together. At the box's rates the host's memory read rate bounds it. |
| exchanges | exchanges × our small exchange's time (2 µs at the box's rate: one NVLink write and a kernel) |
| fixed | small kernels and the expert cache's plan per MoE layer |

- **Ceiling:** the terms at the box's rates.
- **"Ours":** the terms at our measured rates.
- **Unexplained:** measured minus "ours". It is a missing input to measure, not a reason to rent.

## Calibration: each input measured once per box, not guessed

The figures of this table and of the sections that use them were measured on 2026-10-22 (the dates are beside each constant in
`glyd-engine/src/ceiling.rs`), on the two boxes its columns name.

| input | how | 4080 SUPER (PC) | 8x A100 80 GB |
|---|---|---|---|
| a product class's share of the read rate | `tpbench MODEL --one --prof` on real layers: a stage's bytes ÷ its GPU time | bf16 tiered 92.2%, 12-bit ~96%; DeepSeek's FP8 + FP4 95.1% | FP8 and FP4 57%, fitted to two runs (below); bf16 is H100's 80%, not measured |
| fixed time | measured step with graphs − its products' GPU time | DeepSeek ~0.08 ms a layer, Kimi's KDA layer ~0.075 | 9.0 ms for DeepSeek-V4-Flash (0.21 a layer), fitted |
| a product's start (`--t0-us`) | stages in the step graph (`GLYD_PROF_GRAPH=1`), time against bytes | 5.5 µs | ×2.3 assumed (~13 µs) |
| a small exchange | `tpgather` (1 x 896 a GPU) | — | 8.9 / 12.9 / 19.5 µs at 2 / 4 / 8 GPUs |
| PCIe, every GPU at once | the run's copy statistics | — | 12.3 GB/s (one GPU alone 23.3) |

**The A100 fit:** two runs, split on 2 and across 2, give two equations in the two unknowns. The products come to 57% and the fixed time to 9.0 ms.
Both are physical:
- **57%:** our one-token kernels do not fill the big GPUs, as Nsight showed on H100.
- **9.0 ms:** the 4080's ~4.5 ms at about half the clock.

The other two runs are predictions, made with no further fitting.

## Products at a slice's size: bytes plus a fixed cost a layer (2026-10-22, the PC)

GLYD_TP_EMULATE=N with `tpbench --one` loads and runs one GPU as the first of N would. Each sliced product is its slice's alone and the gathers are left
out, so the outputs are not the model's, but every kernel runs at the size a GPU of N runs it. The model used was Kimi-K3's first 4 layers, real widths,
32 experts a layer (`hubslice --experts 32`): a dense KDA layer, two KDA layers with MoE, and an MLA layer with MoE. Steps had graphs on and the 12-bit
layout.

| as GPU 0 of | measured | bytes / (702 GB/s x 96.5%) + 0.62 ms | error |
|---|---|---|---|
| 1 | 10.78 ms | 10.84 | +0.6% (fitted) |
| 2 | 5.75 | 5.78 | +0.5% |
| 4 | 3.23 | 3.25 | +0.6% |
| 8 | 1.94 | 1.98 | +2% (fitted) |

- **Bytes:** products read at 96.5% of the copy rate at every slice size.
- **Fixed cost:** what a slice loses is a cost per step that does not shrink with it. Each product's start and last wave, plus the small kernels, come
  to ~0.14 ms a layer on the 4080.
- **A100:** DeepSeek-V4-Flash's runs fit the same form, its fixed cost 0.21 ms a layer against ~0.09 on the 4080 (×2.3, about the clocks' ratio). By
  that ratio, Kimi-K3's is ~0.33 ms a layer on A100, ~31 ms a token.
- **Per stage at GPU 0 of 8:** `tpbench --one --prof` gives each stage's time. Kimi's MoE and MLA blocks are timed inside too.

## The fixed cost, inside the step graph (2026-10-22, the PC)

`GLYD_PROF_GRAPH=1 tpbench --one --prof` records the stage marks as event nodes of the step graph, so each stage is timed as the graph runs it
(~2 µs an event). What it showed on the 4080 SUPER, Kimi-K3's 4 real layers as GPU 0 of 8:

- **A launch in a graph is 0.5 µs** (an empty kernel, graph replays). The fixed cost is inside the kernels.
- **A product is ~5.5 µs + bytes / 643 GB/s** (the fit of that run; 96.5% of the 702 GB/s copy rate the consistency table above uses would be 677, so the two figures do not agree, and the run's raw data is not kept to say which is off), from 2 MB to 220 MB. Each product a layer fewer saves ~5.5 µs, and
  across GPUs its gather.
- **One-SM kernels are latency-bound.** The old depth mix read its sources one at a time: 1.4 µs a source, 17 µs at the 10 sources real Kimi-K3 reaches.

Done (PR #239 and the MoE pack): add, mix, push and norm one kernel; the delta attention's six products one; MLA's q_a, gate and kv_a one; the
latent down projection and the shared experts' gate and up one. 1.94 -> 1.82 ms a step. The formula now counts Kimi's packs as the engine makes them
(`product_of`): bytes plus 5.5 µs a product give 1.60 ms; the remaining 0.22 ms is the small kernels (router, plan, activations, norms, mix, conv).

Kimi-K3 on 8x A100 after it: 699 exchanges a token (882 before). With the 4080's costs scaled by 2.3: ~0.23 ms a layer fixed (~21 ms a token, from
~29) and ~3.6 ms fewer exchanges. Device-cache prediction: ~88 ms a token with today's gathers, ~78 with the LL gather (#230); not yet run.

## Consistency (2026-10-22)

| case | ceiling | ours | measured | error |
|---|---|---|---|---|
| DeepSeek-V4-Flash, 3 real layers, 4080, one GPU (measured inputs) | 2.01 ms | 2.39 | 2.36 | +1.3% |
| Kimi-K3 layer 0 + head, 4080, tiered (measured inputs) | 4.62 | 5.11 | 5.05 | +1.2% |
| Kimi-K3 layer 0 + head, 4080, 12-bit (the tiered share) | 5.04 | 5.57 | 5.39 | +3.3% |
| DeepSeek-V4-Flash, 8x A100, split by layers on 2 (fitted) | 5.30 | 17.94 | 18.32 | −2.1% |
| the same, across 2 (fitted) | 3.54 | 16.69 | 16.96 | −1.6% |
| the same, across 4 (predicted) | 2.33 | 15.91 | 15.92 | −0.1% |
| the same, across 8 (predicted) | 2.07 | 17.42 | 17.17 | +1.5% |
| Kimi-K3, 8x A100, across 8 | 19.4 | 95.0 | 120 | −21% |
| Kimi-K3, 8x A100, split by layers | 69.3 | 251.5 | 204 | +23% |

**Exchange counts against the engine** (tpbench's gathers in a prompt pass, two places on the PC):
- DeepSeek-V4 real layers: 23 = 23.
- Kimi-K3 layer 0: 6 = 6.
- Random Kimi-K3 at 1 to 7 layers, every matrix sliced: equal at each. At 8 layers it is 85 against 83: in a prompt pass, the last layer (MLA) makes 2
  fewer gathers.

**Kimi-K3's rows are not calibrated.** Their inputs are the expert cache's run statistics:
- the hit rate (48-88% across the windows of a run);
- how the plan shares the misses between copies and the host;
- the plan's cost per MoE layer on that box (0.06 ms on the PC, about 0.4 implied on the A100).

Those vary within a run and were logged per 64 steps, not per layer. The next step is per-layer statistics in the engine, so that one run gives the
inputs and the next is predicted.

## Experts in the expert cache, many rows: the per-layer statistics (2026-11-23, the PC)

`storebench MODEL --places N --rows 1,8,32 --out FILE` runs batched decode through the expert cache and writes each count of rows' per-layer
statistics (`LayerStats`: experts asked for and found, copies, the host's experts and pairs, the bytes read for the misses, the copy stream's,
the host's and each call's own time). `ceiling MODEL --box 4080s --stats FILE [--calibrate MODEL:FILE2] [--fixed-us US]` predicts each step from them. The first formula (the one the table below was made with) took:

- the dense weights once a step, each expert a layer runs on the GPU once, the attention once a row;
- a layer's misses: the longer of its own copies' measured time and the host's measured time, plus the call's own time on the host; the step is
  at least the copy stream's whole time (promotions too: one queue);
- a fixed time a MoE layer and a row, fitted to FILE2's lines' unexplained (without it, FILE's first line's, and no time a row).

The formula of `ceiling.rs`'s `cached_step` now is the one of the second table below (the layers in order); this table's errors are the first formula's.

Random models at the release's widths, larger than the GPU's 16 GB (18 GB each), 8 steps after 3, prompts of 16 tokens; the bytes read for a step's
misses equal the distinct missed experts' bytes in every line (+0.00%: each copied once, or read once by the host for all its rows):

| model, places | mode | rows | measured | ceiling --stats (ours), the first formula | calibrated on |
|---|---|---|---|---|---|
| DeepSeek-V4, 16 layers of 64 experts, 32 places | copy | 1 / 8 / 32 | 30.75 / 146.06 / 280.21 ms | (fitted) / −2.3% / −7.8% | its 1-row line |
| | auto | 1 / 8 / 32 | 19.58 / 95.69 / 286.45 ms | −5.5% / −13.4% / −16.3% | the copy run |
| Kimi-K3, 4 layers (3 of 256 experts), 128 places | copy | 1 / 8 / 32 | 23.47 / 111.60 / 237.68 ms | (fitted) / −0.0% / −1.8% | its 1-row line |
| | auto | 1 / 8 / 32 | 17.00 / 51.51 / 201.33 ms | −10.1% / +1.0% / +5.2% | the copy run |

- **Auto against copy:** the host's half takes 36% (DeepSeek-V4) and 28% (Kimi-K3) off a step at 1 row and 34% and 54% at 8. At 32 rows a miss has
  several rows; DeepSeek-V4's misses outnumber the places its hits leave, and a call is copied whole in waves where the host's share would take
  longer (`waves_quicker`): 286 ms against copy's 280 (334 ms median before that choice).
- **The time a row:** 45 µs a row and a MoE layer on DeepSeek-V4 and 49 on Kimi-K3, fitted. What it is (the launches of the operations a pass runs
  sequence by sequence, no graph capturing a step that reads the router's choices on the host) is not measured.
- **DeepSeek-V4 in auto mode is under-predicted at 8 and 32 rows.** Its per-layer statistics show why: after each layer's host pairs the host waits
  about as long as that layer's copies took, so its copies do not run while the host computes, as the formula assumes and as Kimi-K3's do (its
  wait after a layer is 2 ms beside 9-11 ms of copies). Why DeepSeek-V4's do not is not measured (a trace of the copy stream against the host's
  pairs is the next measurement).

### Why: the copy stream's order, and calls of two kinds (2026-11-23, from the same runs)

The per-layer statistics of those runs say what the formula missed; it does not need a trace.

- **8 rows.** Each DeepSeek-V4 layer copies 6 experts for its call (2.9 ms) and promotes 4 behind it (1.9 ms), the host computes 10
  (3.0 ms). The promotions are queued on the one copy stream after the layer's products (they wait for them), so the next layer's
  copies wait behind them: the copy stream's period a layer is its copies, the products and its promotions (5.3 ms), not the longer
  of the copies and the host's 3 ms. Kimi-K3 at 8 rows promotes none and was predicted within 1%.
- **32 rows.** A layer's hits leave 2 places for 30 misses, so `waves_quicker` copies some calls whole in waves (31 copies, no host
  half) and gives the host the others (2 copies, 28 experts): layer 4's 13 copies and 17 host experts a step are 3 calls of the
  one kind and 5 of the other, and the longer of 13 copies and 17 host experts is not the mean of the calls' times. Rebuilt from
  that split, the step of the formula below is 289.2 ms against 286.5 measured (+1.0%); the lines now carry the calls copied whole
  (`whole`, `whole_bytes`), so the next run measures the split instead.

`ceiling --stats` now follows the layers in order on the host, the copy stream and the GPU, at the run's measured rates: a call
plans, its copies start when the copy stream has run what is queued before them, its products after its copies, its promotions
after its products; the host's pairs from the plan on; the next layer after the products, the host's pairs and the rest of the
layer. A call copied whole has the host wait for its copies and products. The fixed time a layer goes with the products (the
copies behind them wait for it), the time a row with the rest of the layer. On the lines above, calibrated on each model's copy run
(`ceiling::tests::the_cached_step_predicts_the_measured_auto_runs`, with the GPU's dense reads left to the fixed time; the test holds
each line to 10% but DeepSeek-V4's at 32 rows, which it prints):

| model | rows | measured | the formula before | now |
|---|---|---|---|---|
| DeepSeek-V4, auto | 1 / 8 / 32 | 19.58 / 95.69 / 286.45 ms | -5.5% / -13.4% / -16.3% | +6.3% / -6.8% / -21.1% (no `whole` in these lines) |
| Kimi-K3, auto | 1 / 8 / 32 | 17.00 / 51.51 / 201.33 ms | -10.1% / +1.0% / +5.2% | +2.2% / +0.5% / -2.2% |

`storebench --out` writes each layer's totals over the timed steps (`"totals": true`), so a share of calls is kept as it is.

## What it says

- **One GPU:** our code is near the ceiling: 85-94% of it on the 4080.
- **DeepSeek-V4-Flash on 8x A100:** at 12-21% of the ceiling.
  - Fixed time (9.0 ms): the small kernels' fusion.
  - Exchanges (2.9-6.3 ms): the low-latency gather and fewer exchanges.
  - Products at 57%: the one-token kernels' occupancy.
- **Kimi-K3 on 8x A100:** missing experts, the fixed time and the exchanges, in that order. The per-layer statistics come first.

## Limits

- The A100's bf16 share is H100's, not measured. The A100's FP8 and FP4 share is one fitted number for both.
- H100's PCIe rate when every GPU copies is assumed.
- Attention is modelled for Kimi-K3 (KDA states, MLA cache) and DeepSeek-V4 (window plus compressed, approximate) only.
- A packed size is the measured average share. Single matrices vary by a few percent; the head is the largest.
- Decode only, at one user; `--stats` also predicts steps of 1, 8 and 32 rows through the expert cache on one GPU. The prompt is not modelled.
