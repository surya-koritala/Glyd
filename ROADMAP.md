# Roadmap

Every item has a measured gate, taken against the reference codec on the
same machine and thread count, on named public data. Nothing ships on a
number that was not reproduced.

## Where it stands (v0.16.0, 2026-09-25)

The 8.7 GB real-data corpus on AWS Graviton3, 8 threads
([report](docs/benchmarks/suite-2026-09-21.md)): `--max` 3.94 (zstd -3
3.85) at 0.77× its write speed and 7× its read speed; `--ultra` 4.66
(zstd -19 4.66); `--max -r` 4.71 at 643 MB/s; `--ultra -r` 5.22.
Telemetry with `-r`: 2.5–3.5× fewer bytes than zstd -3; application and
system logs (templates): 1.4–3.3× fewer than zstd -3 and 1.1–2.1× fewer
than zstd -19; the cold level: 1.5–2.6× fewer than zstd -19 at 1.2–1.5
MB/s per core. Versions with
`--base`: 1.1–2.1× fewer bytes than `zstd -3 --patch-from` at 1.8–3× its
speed; `--ultra --base` 5–21% fewer than zstd -19's patch. Model
weights (safetensors, v0.14.9): 12–13% under zstd -19 at 25× its
write speed; a checkpoint against the last, 24% under zstd -19's
patch. PyTorch checkpoints with optimizer state (v0.15.0): 83% of their
size alone, 77% against the one before (zstd -19 92%). Weights compressed
in GPU memory (`gpu/`, v0.16.0): Qwen2.5-7B in 11.05 GB on a 16 GB
card, where bf16 takes 15.25 GB, 1.23-1.33x bf16's tokens/s at 1 to 48
sequences at once; prompts to 128 tokens faster, to 4096 within 5-9%. A
terabyte-year in S3 read monthly: `--max -r` $61.7, zstd -3 $73.3.

Measured floors, not to be retried: JSON API events and crawl indexes
are 20–65% hashes and random ids once compressed (typed columns gain
0.1%); Parquet's pages open now (snappy and zstd reproduced, the
values modeled: 31–44% under the file); generic text and binaries sit on the same
entropy floor for every codec (zstd -19, xz, Glyd `--ultra` within 2%).
Details: [experiments/structure/README.md](experiments/structure/README.md).

The store (v0.8.0, at a terabyte v0.12.0, v0.14.7, v0.14.8 and v0.14.9): objects compressed
across a bucket, 3.5× fewer bytes than zstd -3 per object on 1.18 TB of
releases, dumps and events (4.6× on a 39 GB bucket) — the largest
lever measured, because the redundancy of object storage is between
objects. Gzip objects opened (v0.12.0): 23–69% under the gzip.

## Next, in order of what moves the bill

0. **The store at scale.** Done: the fingerprint table on disk
   (mapped, 12 bytes per 4 KB stored), packs for small puts, delete
   and compaction, verification, levels, rebase, a second candidate
   tried on a sample, and S3 spoken directly (SigV4 over HTTPS, the
   standard credential chain, any S3-compatible endpoint; v0.11.0).
   Multipart upload (v0.11.1): 64 MB parts on 8 connections, aborted
   whole on any failure. Rebuild (v0.11.2): index lines beside every
   object, the directory remade from the bucket alone. The gate
   (v0.12.0, [report](docs/benchmarks/store-gate-2026-09-22.md)): 1.18
   TB, 1,192 objects, 49.0 GB stored against zstd -3's 153.5 GB,
   every object back byte-exact, rebuilt from the bucket and verified;
   put 150 MB/s, get 186 MB/s on one im4gn.4xlarge. Left: overlapping
   one object's upload with the next one's compression (the put rate
   is one process, one object at a time).

1. **Write speed of `--max`** (v0.14.5: 1.01–1.05× zstd -3 on one
   Graviton3 core on three of five files, 0.89× on the log and the
   table dump; 1.07–1.26× on a Ryzen 9; the parse is zstd -3's
   double-fast without a lazy step; what is left on match-dense data
   is the per-sequence cost of the eight-stream sections).
   Done: `--dense`, a unit's blocks parsed in stripes on all cores, so
   medium inputs keep 128 MB units and the same bytes as one core (ten
   cores had cost 7–9%); opt-in, since reads then scale only with the
   units. Left: the far pass is one core per unit (half of it the
   anchor gather, which could split across cores), so a file of few
   units idles cores at its start; and a decoder that decodes a unit's
   stripes in parallel, holes for the far matches filled after, which
   would make dense the default. Gate: 0.9× zstd -3 on one core with
   the corpus ratio kept; at equal bytes, one core is already 3–10×
   faster than the zstd level that reaches them.
2. **Small objects**: a single-pass decoder for compact blocks and a
   cheaper per-object encoder (zstd is 1.4–2× faster per object); a
   dictionary that carries a record schema, so `-r` ratios reach
   one-record objects. Gate: within 1.2× of zstd per object.
3. **The x86-64 decoder** (0.79–0.97× zstd -3 on one Sapphire Rapids
   core against 1.15–1.32× on Graviton3 and 1.00–1.30× on a Ryzen 9,
   v0.14.7). Gate: 1.0× in the published run.
4. **Streaming for v9** in `GlydReader`/`GlydWriter` (v6 levels only
   today); the CLI already streams batches of units.
5. **Already-compressed objects opened** (research section J). Done:
   gzip (v0.12.0), zip, tar, Office documents, jars, PDF, zlib and PNG,
   nested four deep (next release): 5–82% under the object at the
   level that opens it, 16–74% where zstd -19 gets 3–53%; JPEG
   recoded by Glyd's own model (v0.14.0), 22–26%, inside the others too.
   Parquet with snappy or zstd pages (v0.14.8: every page written back
   byte for byte by a port of the compressor that wrote it, the values
   modeled, 31–44% under the file where zstd -19 gets 1%), and zstd
   objects the same way (zstd 1.5.2 to 1.5.7, levels 1 and 3, as the
   library and the command line write). Next: gzip, lz4 and brotli
   pages and zstd 1.4, then
   the recipe itself (preflate's corrections are a fifth of a pdfTeX
   stream; a better predictor of zlib's choices would halve what
   opening costs), then zip entries and PDF streams that are
   themselves JPEG or PNG (a .docx's pictures). Gate: the measured
   number on real objects, every one restored byte for byte.
6. **Video: measure the headroom, decide nothing before.** The
   residual an H.264 file keeps is what its block predictor could not
   guess; how much a far stronger predictor would guess of the same
   stream is unknown, not a limit (the 1–3% in the literature is from
   weak attempts). The experiment, two weeks: pull the symbols the
   codec stores (modes, motion vectors, coefficients) from real files,
   predict them with the cold level's mixing conditioned on the
   decoded frames before, count bits. Under 5%: closed for good. Over
   15%: the largest byte class there is, and it moves to the top of
   this list.

Done since v0.5.0: record-mode reads (column-at-a-time rebuild, 1.4–1.7×
faster; `--max -r` now within 1% of zstd -3's S3 row at ten CPU-billed
reads a month); the template shape for logs; base mode's region chosen
from a coarse map of the base (content found wherever it moved) and
`--ultra --base` at the plain ultra speed (was 1–4 MB/s); the kernel
point-release chain measured (15 versions in 228 MB, a step 1.8 MB, no
rebasing needed inside a series). Not reached: the kernel pair under
zstd -19's patch size at `--max` speed (`--max --base` 3.03 MB against
2.58; `--ultra --base` 2.04 MB at 3 MB/s). The levers were sized
against each other first
([experiments/research/README.md](experiments/research/README.md)):
version chains 5.7× over per-version compression, template logs 1.2–1.6×
over zstd -19 (measured 1.1–2.1× once built), context mixing 1.2–1.5×
at ~1 MB/s for cold data only, float columns nothing.

The cold level (`--cold`, v0.7.0): context mixing at 1.2–1.3 MB/s
per core, 1.5–2× fewer bytes than zstd -19 on logs, dumps, JSON and
text, the zpaq -m5 class. Left there: a second match model for long
repeats, an indirect (byte-history) context, SIMD for the mixer; each
worth 1–2%. Gate for more: within 1% of zpaq -m5 on every class.
