# GOAL 3: Speed. Industry standard means beating LZ4 where it lives.

Supersedes GOAL2's Tier 1 as the primary target, by decision on 2026-09-18.
GOAL2's rules (measure first, feasibility before freezing, one attack axis with
floors on the rest, floors ratchet, results under the repo, commit per verified
step) remain binding. GOAL2's Tier 1 finding stands as the reason for the
pivot: strictly dominating LZAV on all three axes is infeasible for this design
because ratio and decode speed trade against each other (see CHANGELOG-BENCH:
entropy coding refuted twice, bit-packed offsets refuted, window saturates).

## 0. The physics, measured on this machine (single core, pinned)

| Working set | memset GB/s | memcpy GB/s |
|---|---:|---:|
| 16 KB (L1) | 124 | 64 |
| 512 KB (L2) | 153 | 67 |
| 8 MB (L3) | 29 | 48 |
| 212 MB (DRAM) | 29 | 19 |

A decoder must write every output byte, so **memcpy of the output is the hard
ceiling: 22.9 GB/s over Silesia** (small files cache-resident, large ones
DRAM-bound). No codec approaches it because decode cost is per token, not per
byte: Silesia is 14-20 million tokens, and DRAM speed would allow ~0.1 ns each.
The wall is reachable only on highly redundant data (few tokens).

## 1. The field against the wall (Silesia, this machine, single core)

| Codec | Decode GB/s | % of wall | Ratio | Comp GB/s |
|---|---:|---:|---:|---:|
| memcpy | 22.9 | 100 | - | - |
| liblz4 | 5.8 | 25 | 2.101 | 0.85 |
| lz4_flex | 3.7 | 16 | 2.097 | 0.67 |
| Alatirok (v5) | 3.1 | 14 | 2.388 | 0.45 |
| LZAV | 3.1 | 14 | 2.450 | 0.49 |
| zstd -3 / -1 / 1 / 3 | 2.3 / 2.2 / 1.7 / 1.6 | 7-10 | 2.24-3.20 | 0.3-0.6 |
| snappy | 2.1 | 9 | 2.076 | 0.78 |

liblz4 is the open-source speed champion; every measured codec decodes slower.
The only thing above it is RAD Oodle (Selkie/Mermaid), commercial and closed,
not measurable here. It is the unmeasured bar, believed ~1.5-2x LZ4 on decode.

Alatirok's asset: its parse emits ~14.4M tokens against LZ4's ~20M (30% fewer)
at a higher ratio. Its liability: ~4.6 ns per token against LZ4's ~1.8.

## 2. Tiers

### Tier S1: dominate liblz4 on decode and ratio (proven point)

**PASSED 2026-09-18 at commit 54f0a24** (format v6, AVX2 32-token pre-pass,
minimum match 7). Numbers are the same-run quick3 result; per-file table in
README.md.

| Gate | Requirement | Start | Now | Status |
|---|---|---:|---:|---|
| **S1.1** Correctness | all tests green, 1M-mutation fuzz, Silesia + enwik8 round-trip | PASS | PASS | held |
| **S1.2** Decode 1C | >= liblz4 measured in the same run, same protocol | 3.1 (55%) | 6.05 vs 5.54 (109%) | **PASS** |
| **S1.3** Ratio floor | Silesia >= liblz4's 2.1009 | 2.388 | 2.192 | held (spent 0.20 on S1.2) |
| **S1.4** Multi-core | 16C decode >= 10 GB/s on mozilla, nci, webster, samba | 13.4 | 13+ | held |
| **S1.5** Memory | decoder allocates nothing beyond output | PASS | PASS | held |

Cost of passing: compression 0.45 -> 0.35 GB/s, which is S3's job.

Decode is the attack axis. Ratio and multi-core are floors. Compression speed
is NOT a Tier S1 gate: LZ4's 0.85 GB/s comes from a finder that cannot produce
our ratio, so it is met by a separate level (S3), the way every industry codec
does it.

### Tier S2: toward the wall (research, not a gate)

Vectorized token decode: decode a 32-byte chunk of the token stream with AVX2
(lengths, escape flags, offset widths via shuffles and prefix sums), then run
the copies. This is what the separated-stream format was built for and what an
inline format like LZ4's cannot do. Plausible ceiling ~9-10 GB/s on Silesia
(~40% of the wall, ~1.7x LZ4). Becomes a gate only when a measured prototype
shows the number.

Status 2026-09-18: the vectorized pre-pass is built and is what carried S1.
What remains is the copy loop (~9 cycles/token; the pre-pass is ~3.4) and the
two files still behind liblz4, nci and webster. Finding the next cut needs
hardware counters, which WSL2 does not expose: continue on bare-metal Linux
or macOS.

### Tier S3: levels

| Level | Finder | Target | Purpose |
|---|---|---|---|
| fast | 4-byte hash, 1-way, 64 KB window, no lazy, skip | comp >= 0.85 GB/s, ratio >= 2.10 | LZ4-class compression speed; also handles x-ray-like data cheaply. **Not built; next.** |
| default | LZAV-port finder, minimum match 7 | ratio 2.19, decode > liblz4 | the S1 point (current default) |
| dense | LZAV-port finder, minimum match 5 (`ALATIROK_DENSE=1`) | ratio 2.39 | the ratio point; decode ~55% of liblz4 |

## 3. Floors carried from GOAL2, and one consciously retired

Kept: correctness (1M fuzz), multi-core, memory, Silesia total ratio >= 2.1009
(raised from the old 1.85 regression floor: it ratchets up).

Retired: the per-file x-ray regression floor of 1.01 is no longer enforced
through the dense retry. Reason: liblz4 itself gets 1.010 on x-ray and does so
at 18.9 GB/s decode and near-memcpy compress; our dense retry got 1.08 at a
cost of 23 ms compression (0.34 GB/s) and 2.7 ms decode. Under GOAL3 the fast
level (S3) is the correct way to meet it; until that exists x-ray is stored
raw. This is a user decision, recorded here, not a silent trade.

## 4. Measurement

GOAL2 section 3 procedure. Every report of an Alatirok decode number carries
liblz4's number from the same run so the gate cannot be met by drift.
