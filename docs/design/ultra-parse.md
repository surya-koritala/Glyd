# The ultra level's parse

`--ultra` writes the same v7 blocks as `--max` (see [format-v7.md](format-v7.md));
only the choice of sequences changes. Where `--max` takes the first good
match it finds (a double-fast lazy parse, ~300 MB/s), `--ultra` prices every
way of coding a block and takes the cheapest (~5 MB/s). The decoder does
not know the difference, and the denser output decodes slightly faster:
fewer sequences per byte.

## Match finder

A binary tree per 4-byte hash bucket over the 2 MB window (`tree`, 2 × 2²¹
slots; `hash4`, 2²⁰ heads), as in zstd's `btlazy2`/`btopt` and LZMA's
`bt4`. Every position is inserted at its bucket's root; the walk down
compares the new suffix against the nodes on its path, threading each onto
the smaller or larger side, so the tree stays sorted by suffix. Because a
node's children are always older positions, a search walks from the newest
match to older ones, and each match it lists is longer than the last: the
list is exactly "for each length, the nearest position achieving it", which
is what the parse wants. Comparisons resume at the common length already
proven by the two bounding nodes (`cls`/`cll`), so a walk of `DEPTH` = 64
nodes costs far less than 64 full comparisons.

Suffixes are ordered over the whole input, not the block, so trees built in
one block stay valid in the next. Positions older than the window are never
followed: their slots belong to newer positions (the tree is a ring). A
node at the window's edge is linked but its children are cut. Positions
under a long match found while inserting are skipped (their twins are in
the tree already), which keeps runs of one byte linear.

A separate 3-byte head (`hash3`, one position per bucket) offers the
nearest 3-byte match, and the three repeat offsets are tried at every
position at any length.

## Pricing

Prices are in 1/256 bit, from symbol counts: literal bytes, the ll/ml/off
codes, and the extra bits each code carries (`Prices`). A code's price is
−log2 of its frequency, add-one smoothed, floored at one bit (no prefix
code spends less). Counts come from the recent blocks, each block weighing
half the next (`Stats::decay_into`), plus a fixed prior of 4096 per table
taken from the `--max` parse over Silesia with weight moved onto the
shortest matches. The prior is what keeps the parse from a bad fixed point:
a block parsed on prices from a sparse block would price every code it did
not use at log2(total) bits, get sparser, and never recover (x-ray went
1.35 with priors absent, 1.62 with them, against 1.46 for `--max`).

The first block of a call has no history: it is parsed on the prior and its
own byte frequencies, then again on the counts of that parse. The first
pass's table insertions are logged and undone in between, which is exact
there (nothing older is in the trees).

## Dynamic program

`opt[i]` holds the cheapest known way to arrive at block offset `i`: its
price, how it arrived (a literal, or a match of `mlen` at `off`), the
literal run length so far, and the repeat-offset state on that path. For
each `cur` in order:

- one more literal reaches `cur + 1` at the byte's price plus the change
  in the literal-length code's price (`ll(n) − ll(n − 1)`, so a run's
  code is paid incrementally and a match then pays `ll(0)`);
- every candidate at `cur` reaches `cur + l` for each length `l` it
  covers (a repeat offset from `MIN_MATCH`, a tree match only for the
  lengths past the previous, nearer candidate) at the match's price:
  `ll(0)` + `ml(l)` + the offset's code, which is a rep code when the
  path's rep state holds it.

A match of `SUFFICIENT_LEN` = 256 or more is taken whole and no position
inside it is searched (zstd's `targetLength`). The back-trace from
`opt[block_len]` yields the sequences; trailing literals form the
literal-only last sequence the format requires.

## Measured (Silesia, one run, Apple M1 Max)

| | Ratio | Compress | Decode |
| :--- | ---: | ---: | ---: |
| Glyd `--max` | 3.218 | 330 MB/s | 2,090 MB/s |
| Glyd `--ultra` | 3.801 | 4.8 MB/s | 2,186 MB/s |
| zstd -16 (default window) | 3.834 | 8.0 MB/s | 1,781 MB/s |
| zstd -16 (2 MB window) | 3.802 | 8.3 MB/s | 1,777 MB/s |
| zstd -19 (default window) | 4.006 | 4.0 MB/s | 1,636 MB/s |
| zstd -19 (2 MB window) | 3.910 | 4.9 MB/s | 1,625 MB/s |

What the sweeps said: tree depth saturates at 32–64 (3.804 at 32, 3.806
at 512); `SUFFICIENT_LEN` 64/128/256/1024 gives 3.792/3.803/3.806/3.807; a
second pass per block on the block's own counts gains 0.05%; a 3-byte
tree instead of the 3-byte head gains nothing; zstd-style finer length
codes would save 0.6% of the sequence section. The coder itself spends
1.45% over the order-0 estimate of its parse, ~950 bytes per 256 KB
block (sub-stream size tables and padding, entropy tables, headers).

## Block splitting (v0.3.1)

After the parse of a 256 KB block, `split_points` looks for a cut where
the two parts coded on their own statistics (order-0 cost of the literal
bytes and the ll/ml/off codes; extra bits are the same either way) are
cheaper than the whole by more than four times a block's overhead
(~600 bytes of framing and tables), with parts of at least 48 KB, and
tries each part again. The parts become blocks of their own: the repeat
offsets restart, the tables may be reused across them, the decoder sees
ordinary blocks. Silesia: 3.925 -> 3.931, decode 2,150 -> 2,142 MB/s;
mozilla and samba, whose sections differ, gain most. The margin is the
decode price: every block costs the decoder its table builds and stream
tails (~2.6 us), so at twice the overhead the same corpus gains 0.33%
for 2.5% decode, at four times 0.15% for 0.4%.
