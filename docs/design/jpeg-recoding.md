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
