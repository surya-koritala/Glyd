# JPEG recoding, Glyd's own

A JPEG is DCT coefficients under a Huffman code. Recoding it losslessly
means taking the coefficients out, coding them with a model that knows
what neighbouring blocks look like, and putting the identical JPEG back.
Since v0.13.0 that has been Lepton's road (`lepton_jpeg`, the Rust port
of Dropbox's). This is the design for doing it in `src/jpg/`, with
nothing outside this repository, the way `src/reflate/` does deflate.

## What a JPEG is, for this purpose

Markers, then scans. Everything up to the first scan (APP segments,
quantization and Huffman tables, the frame header, restart interval,
comments) and every marker between scans is kept as it is. A baseline
scan is 8×8 blocks in MCU order, each block a DC difference and up to 63
AC coefficients in zigzag order under the scan's Huffman codes, restart
markers every `n` MCUs, the stream padded with 1 bits to a byte before
each marker. A progressive JPEG spreads the same coefficients over
several scans (a band of positions, then refinement bits), with
end-of-band runs whose grouping is the encoder's choice.

## Parts

1. **Parser and writer** (`parse.rs`, `write.rs`): the headers kept
   verbatim; a baseline scan decoded to coefficients (per component, a
   block grid) and written back from them with the same tables, the
   same restart markers, the same padding — bit for bit, or the JPEG
   stays closed. Padding bits that are not 1s and bytes after EOI are
   kept explicitly. Arithmetic-coded, 12-bit, lossless and hierarchical
   JPEGs stay closed. Progressive comes second (its parser needs the
   refinement logic; its writer needs the end-of-band choices coded as
   corrections, as `reflate` codes an encoder's choices).
2. **Model** (`model.rs`): coefficients coded with the binary
   arithmetic coder of `src/reflate/coder.rs`, block by block, under
   contexts from the block above and to the left: for each zigzag
   position whether it is zero (context: the position, how many
   coefficients are left, the neighbours' values there), its magnitude
   in exponent-and-mantissa bins (context: the position and the
   neighbours' magnitudes), its sign (context: the neighbours' signs
   for the low positions); the DC predicted from the neighbours through
   the pixel edge they share, its residual coded. Chroma under the luma
   block's shape. The whole thing measured against `lepton_jpeg` on the
   same photos at every step, and not switched to until it wins.
3. **Envelope**: `GLYDJPEG` is Lepton's; the new one is `GLYDJPG2` with
   the kept bytes, the model's stream, and the corrections. Inside
   containers (`REFLATE_JPEG`, a JPEG stored in a zip), the same
   recoder.

## Order of work

1. Parser and writer, bit-exact on the fixtures in `tests/data/jpeg/`
   (baseline at several qualities and samplings, grayscale, restart
   intervals, optimized tables) and on the photos on this machine.
2. A first model: zigzag position and neighbour contexts, DC by
   neighbour average. Measured against Lepton.
3. The DC edge predictor and the chroma contexts; the model tuned
   until it beats Lepton on every photo.
4. Progressive JPEGs.
5. The switch, Lepton staying only to read `GLYDJPEG`.

## Where it stands (v0.14.0)

Steps 1, 2, 3 and 5 are in; progressive JPEGs (step 4) are next. The
envelope stayed `GLYDJPEG`, the stream inside it self-identifying
(`GJPG`, spec 2c), so nothing else changed. What the model turned out
to need, each step measured on five photos (13.4, 6.4, 2.1, 1.8 and
1.9 MB; two at quality ~100, 4:2:0 and 4:2:2; three at 75–95):

- The interior 7×7 first, its nonzero count under the neighbours'
  counts; each coefficient's zero flag under the position, what is
  left of the count and the neighbours' magnitudes there (left and
  above weighted 13:13:6 with the corner, plus how far left and above
  disagree); the exponent in unary under the position, the
  neighbours' bucket and what is left; the top three mantissa bits
  under a tree per exponent, the rest raw; the sign under the
  neighbours' signs.
- Then each edge (the first row, the first column): its nonzero count
  first, then each coefficient under a prediction from pixel
  continuity — with dequantized coefficients the boundary between two
  blocks is Σ C(v)·(−1)^v·F[v] from the neighbour's side and Σ C(v)·F[v]
  from ours, so the one unknown is solved for. The weights of true
  pixel continuity (cos((2·7+1)vπ/16) against cos(vπ/16)) came out
  worse than the boundary midpoint's ±1; coding the residual against
  the prediction instead of the value under its bucket came out much
  worse.
- The DC last, from both edges by the same relation, weighted towards
  the side whose eight boundary pixels agree more, the residual under
  how far the two sides disagree.
- Probabilities that count their bits, adapting at 1/(n + 1.5) down to
  1/256 (0.6% smaller than a fixed 1/128; a floor of 1/1024 or 1/60,
  or a mix of two rates, were worse); a range coder with a 32-bit
  range and 16-bit probabilities (0.5% smaller than the corrections
  coder's carry-less one).
- The kept bytes (EXIF, an embedded preview) compressed at the max
  level: 98 KB → 75 KB on the 13.4 MB photo. Lepton compresses its
  header too.
- Four stripes of block rows after a prefix, each coded on its own
  core: 0.1–0.2% per stripe, which is the contexts not following the
  picture from the rows before (continuing a stripe from the previous
  one's final state costs 0.05%; a warm start from the prefix helps
  the 13 MB photo and not the 6 MB one). Lepton's format is eight
  partitions, 0.3% over its one.

Against v0.13.4 (Lepton's stream, single-threaded) on this Mac, every
decode byte-exact: smaller on all five photos (9,934,880 vs 9,971,627;
4,997,778 vs 5,001,278; 1,665,845 vs 1,679,213; 1,455,649 vs 1,470,578;
1,448,903 vs 1,455,039 bytes), 1.6–1.9× faster to write and 1.6–1.7×
faster to read on all cores (13.4 MB: 1.02 and 0.54 s against 1.77 and
0.91). On one core it was slower: 2.31 and 1.24 s against 1.76 and
0.91. The coder alone ran at 5 ns a decision, 160 million of them for
the 13.4 MB photo, most of the time; the Huffman parse and write were
0.27 and 0.15 s of it.

v0.14.1 took the speed on: the range coder's decision is branch-free
(5.2 → 3.6 ns on random bits: the mispredicted branch on the bit was
the cost, not the chain of dependent operations — two coders in
lockstep gained nothing); the scan is written in bands on every core
(bits unstuffed per band, joined with the stuffing and the markers
in one pass) and, when it has restart intervals, parsed in bands at
its markers; a file of 10 MB and up takes eight stripes (a stripe
costs a few KB whatever the size). All cores, against v0.13.4: the
13.4 MB photo written in 0.62 s and read in 0.29 (1.77 and 0.91); the
6.4 MB one 0.48 and 0.23 (0.97 and 0.50); the three of ~2 MB 0.16–0.21
and 0.07–0.10 (0.37–0.43 and 0.19–0.22); bytes as above but the 13.4
MB photo at eight stripes, 9,943,767. One core: 1.99 and 1.06 s against
1.77 and 0.91 for the 13.4 MB photo — 12–16% slower still, the model's
decisions (160 million at ~3.6 ns plus their branches) being the rest.
