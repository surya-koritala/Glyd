# Tier 1 frontier map (measured 2026-09-17)

Silesia, single core, core-pinned, median of 3. Targets from Phase 0 (LZAV):
**ratio 2.4500, compression 0.489 GB/s, decompression 3.130 GB/s.**

Session start: ratio 2.19502, comp 0.387, decomp 3.529.

| Match finder | Ratio | Comp GB/s | Decomp GB/s | T1.2 | T1.3 | T1.4 |
|---|---:|---:|---:|:--:|:--:|:--:|
| 1-way, word-tag, h14 | 2.1543 | **0.4996** | 3.899 | no | **PASS** | PASS |
| 1-way, word-tag, h15 (HEAD) | 2.1797 | 0.4747 | 3.902 | no | no | PASS |
| 1-way, word-tag, h16 | 2.1950 | 0.4409 | 3.818 | no | no | PASS |
| 1-way, word-tag, h17 | 2.2029 | 0.4210 | 3.853 | no | no | PASS |
| 2-way, h15, probe32 | 2.2513 | 0.3754 | 3.875 | no | no | PASS |
| 2-way, h16, probe32 | 2.2595 | 0.3429 | 3.785 | no | no | PASS |
| 2-way, h17, probe32 | **2.2637** | 0.3285 | 3.823 | no | no | PASS |
| 6-byte hash, 1-way, b14 | 2.1085 | **0.5318** | 4.746 | no | **PASS** | PASS |
| 6-byte hash, 1-way, b15 | 2.1448 | 0.4864 | 4.745 | no | ~ | PASS |
| 6-byte hash, 1-way, b16 | 2.1647 | 0.4438 | 4.768 | no | no | PASS |
| 6-byte hash, 2-way, b16 | 2.2051 | 0.3620 | 4.643 | no | no | PASS |

## What this says

**T1.3 is solved.** 0.5318 GB/s against a 0.489 target, from storing the match
word in hash entries so a failed probe costs a register compare instead of a
cache-missing read into the source.

**T1.4 is solved with room to spare.** Every configuration passes; the 6-byte
hash reaches 4.75 GB/s, 52% above the floor. Longer matches mean fewer tokens,
and decode cost tracks token count.

**T1.2 is the whole remaining problem.** The best ratio found is 2.2637 against
2.4500 needed, a further +8.2%. Ratio and compression speed remain on one dial:
the best-ratio point has the worst compression and vice versa.

## Where the bytes are

Measured on the compressed output:

| Stream | Share |
|---|---:|
| Offsets | 37.8% |
| Tokens | ~26% |
| Literals | ~27% |
| Extras | rest |

Offsets are the largest component, so ratio work should target them.

Measured offset magnitudes: 17.4% below 256, 29.9% below 1K, 48.0% below 4K,
72.4% below 16K, 27.6% at or above 16K.

## Ratio options, with measured value

| Option | Measured or modelled gain | Status |
|---|---|---|
| Larger match window | **negative**, 2.19502 -> 2.18781 best | refuted, reverted |
| Repeat-offset codes | +1.23% of output | measured: only 3.3% of matches repeat, not worth a format change |
| 1-byte offsets when < 256 | +3.28% of output, ratio -> ~2.269 | not implemented, needs a token bit |
| 2-way buckets | +2.6% ratio | implemented, costs 22% compression |
| Entropy coding the offset stream | unknown, largest single target at 37.8% | not attempted; decode headroom exists to pay for it |

Stacking every positive option above lands near 2.35, short of 2.4500. Reaching
T1.2 likely needs entropy coding, which is the technique zstd uses to reach
2.89 and which the 52% decode headroom could fund.
