# GOAL 2: Best in class, where "class" is measured, not assumed

Supersedes GOAL.md. That file asked for a combination of numbers no codec on
earth achieves. This one targets real artifacts with measured coordinates.

Work continues until every gate in Tier 1 passes on one commit, then Tier 2.
Never change a threshold. Never delete a floor.

---

## 0. The measured field (Phase 0, Silesia, this machine)

Single core, liblz4 = reference C LZ4 as baseline.

| Codec | Ratio | vs LZ4 | Comp GB/s | vs LZ4 | Decomp GB/s | vs LZ4 |
|---|---:|---:|---:|---:|---:|---:|
| zstd-3 | 3.2045 | 1.525x | 0.322 | 0.381x | 1.575 | 0.308x |
| zstd-1 | 2.8942 | 1.378x | 0.533 | 0.630x | 1.660 | 0.325x |
| LZAV-hi | 2.8028 | 1.334x | 0.063 | 0.074x | 3.031 | 0.592x |
| LZAV | 2.4500 | 1.166x | 0.489 | 0.578x | 3.130 | 0.612x |
| zstd--1 | 2.4380 | 1.160x | 0.535 | 0.633x | 2.174 | 0.425x |
| zstd--3 | 2.2399 | 1.066x | 0.635 | 0.751x | 2.301 | 0.450x |
| **Alatirok** | **2.1950** | 1.045x | **0.387** | 0.458x | **3.529** | 0.690x |
| liblz4 | 2.1009 | 1.000x | 0.846 | 1.000x | 5.116 | 1.000x |
| lz4_flex | 2.0971 | 0.998x | 0.670 | 0.793x | 3.698 | 0.723x |
| snappy | 2.0761 | 0.988x | 0.778 | 0.920x | 2.126 | 0.415x |
| zstd--5 | 2.0570 | 0.979x | 0.671 | 0.794x | 2.377 | 0.465x |

**No codec dominates liblz4 on all three axes.** LZ4 holds the fastest
compression and the fastest decompression in the entire field. Every codec that
beats it on ratio pays on both speeds. This is why GOAL.md's G2+G3+G4 was
unreachable: it demanded a point no artifact occupies.

**The class** is the codecs that beat liblz4 on ratio: zstd-3, zstd-1, LZAV-hi,
LZAV, zstd--1, zstd--3, Alatirok.

**Where we already stand.** Alatirok holds the highest decompression speed in
that class, 3.529 GB/s against LZAV's 3.130 and zstd-1's 1.660. No class member
dominates us, because each one that beats our ratio loses our decode. We are
already Pareto-optimal. We are also the weakest on ratio and compression speed.

---

## 1. Tiered targets, each with a feasibility proof

Every threshold below is a coordinate of a codec that exists and was measured
above. That is the rule GOAL.md violated.

### Tier 1: dominate LZAV (nearest neighbour)

| Gate | Requirement | Current | Feasibility proof |
|---|---|---:|---|
| **T1.1** Correctness | All tests green; Silesia + enwik8 round-trip every path; 1,000,000-mutation fuzz, zero panics, ASAN clean | PASS | already held |
| **T1.2** Ratio | Silesia total >= **2.4500** | 2.1950 | LZAV achieves 2.4500 |
| **T1.3** Compression 1C | Silesia total >= **0.489 GB/s** | 0.387 | LZAV achieves 0.489 |
| **T1.4** Decompression 1C | Silesia total >= **3.130 GB/s** | 3.529 | LZAV achieves 3.130; we already exceed it |
| **T1.5** Holdout | Same three on enwik8 vs LZAV measured there | not run | LZAV achieves them |
| **T1.6** Multi-core floor | 16C decomp >= 10 GB/s on mozilla, nci, webster, samba | PASS | already held |
| **T1.7** Memory floor | Compressor <= 1 MB working set; decompressor 0 alloc beyond output | PASS | already held |

LZAV occupies (2.4500, 0.489, 3.130) simultaneously, so T1.2+T1.3+T1.4 are
jointly satisfiable by a real artifact. Passing Tier 1 means Alatirok strictly
dominates LZAV: equal-or-better ratio and compression, strictly better decode.

### Tier 2: dominate the fast-zstd points

| Gate | Requirement | Feasibility proof |
|---|---|---|
| **T2.1** | Ratio >= 2.4380 and comp >= 0.535 and decomp >= 2.174 | zstd--1 achieves all three |
| **T2.2** | Ratio >= 2.2399 and comp >= 0.635 and decomp >= 2.301 | zstd--3 achieves all three |

### Tier 3: dominate the whole class (NOT yet proven feasible)

Requires ratio >= 3.2045 (zstd-3) while holding decode above 3.13 GB/s. **No
codec in the field does this.** zstd-3 reaches that ratio through entropy
coding, which costs decode speed. Tier 3 therefore may be infeasible for an
LZ77-only design and must not be attempted until a feasibility study either
finds an artifact occupying that point or demonstrates headroom. Treat as
research, not as a gate.

---

## 2. Rules that prevent gates from fighting each other

1. **Feasibility before freezing.** No threshold is frozen without a named
   codec that achieves that exact combination, measured on this machine.
2. **One attack, the rest are floors.** Attack one axis at a time. Every other
   axis carries a floor at its current measured value.
3. **Floors ratchet.** A verified gain becomes the new floor. Never trade a win
   away silently.
4. **Coupled axes are named.** Ratio and compression speed are the same dial in
   an LZ77 match finder: ratio comes from more search, speed from less. Any
   plan that moves both against each other in one configuration is invalid on
   its face. Escaping the coupling requires a better algorithm, not a constant.

---

## 3. Measurement procedure

Unchanged from GOAL.md Section 1 and still binding: reference C liblz4 via the
`lz4` crate as baseline, single-core runs pinned internally, median of five
measurements each >= 1.0 s, corpus totals as sum-of-bytes over sum-of-time,
enwik8 as an untuned holdout, `RUSTFLAGS="-C target-cpu=native"`, machine idle.

Add LZAV (the `lzav` crate, real C LZAV 4.3) as a second gated baseline, since
Tier 1 is defined against it.

---

## 4. Crash resilience (this machine crashes)

The host has had 10 unexpected shutdowns since July, including two on
2026-09-17, with no WHEA entries. Sustained all-core load raises the odds. Work
must therefore survive an abrupt power loss at any instant.

- **Durable output only.** Results are written under the repo on /mnt/c, never
  to /tmp or a session scratchpad. Those do not survive.
- **Incremental writes.** Every per-file measurement is appended and flushed
  immediately, so a crash costs at most one file's work.
- **Resumable runs.** A benchmark re-run skips (file, codec) pairs already
  present in the partial CSV.
- **Commit early.** Commit after each verified step, not at the end of a batch.
- **Detach long jobs.** `setsid nohup <cmd> > log 2>&1 < /dev/null &` so a
  session ending does not kill them. This survives teardown, not a power cut.

---

## 5. Known levers, ranked, for Tier 1

Ratio needs +11.6% and compression +26.4%; decode already passes with 12.7% to
spare. Both ratio levers below are believed to also reduce token count, which
helps decode, so they do not fight T1.4.

1. ~~**Match window beyond 64 KB.**~~ **REFUTED 2026-09-17.** Implemented as
   format v4 (zero marker in the offset stream, true offset in extras, so no
   token bits spent) and swept over window sizes 1/4/16 MB and far-match
   thresholds 12..48. Every configuration lost to the 64 KB baseline:
   2.19502 baseline against a best far-offset result of 2.18781. The cause is
   that the hash table keeps only the most recent position per hash, so a far
   candidate only arises when a pattern has not recurred recently; taking it
   costs 7 bytes against 3 and consumes positions that would otherwise yield
   better near matches. **A larger window is worthless without a match finder
   that can choose among several candidates.** Reverted.
2. **Repeat-offset codes.** Cost nothing to decode, shrink the offset stream
   (31.9% of output), and reduce per-token bytes.
3. **Optimal or wider parsing**, only if T1.3 still holds afterwards.
4. **Cheaper match finding**, needed for T1.3. Study LZAV's finder: it reaches
   0.489 GB/s at ratio 2.45, which our current design cannot do at any point on
   its measured frontier.

Evidence that a plain re-tune will not work: the v3 frontier sweep over table
size and lazy matching tops out at ratio 1.045x liblz4 and compression 0.698x,
at opposite ends. Tier 1 needs both moved together, which requires adopting a
better match finder, not a different constant.
