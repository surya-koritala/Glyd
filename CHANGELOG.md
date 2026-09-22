# Changelog

All notable user-facing changes. Measurement history, floors and refuted
ideas live in [CHANGELOG-BENCH.md](CHANGELOG-BENCH.md).
Versioning follows [SemVer](https://semver.org); the on-disk format has its
own version in every block header (v6, v7) and every release decodes
every earlier format.

## Unreleased

### Deflate containers opened

- Zip (and so .docx, .xlsx, .pptx, .jar, .apk, .odt), zlib streams and
  PNG join gzip: `src/deflate.rs`, envelope `GLYDDEFL` (replacing
  v0.12.0's `GLYDGZIP`). Every deflate stream inside is decoded with
  what it takes to re-encode it bit for bit; headers, directories,
  stored entries and non-image chunks are kept; a PNG's image stream is
  cut back into its IDAT chunks. Entries preflate cannot reproduce stay
  as they are, and so does an object that would not shrink. The CLI's
  `--max` and `--ultra` take record mode on an opened container where
  it pays. PDF too: every stream whose data is zlib and that ends
  before an `endstream`, found by scanning. The recipe goes into the
  envelope compressed (a PDF's duplicate fonts and a jar's thousand
  entries repeat their corrections), and the object is also compressed
  closed at the same level, the smaller kept. Measured, decodes
  compared: a Guava jar 3.05 → 1.76 MB at `--max` (zstd -19 on the
  jar: 2.70), 1.14 MB cold; a GitHub source zip 2.73 → 2.28, 1.64
  cold; a 60-slide .pptx 88 → 24 KB; a 30,000-row .xlsx 1.34 → 0.47
  MB cold; a pdfTeX paper 2.22 → 0.73 MB (zstd -19: 1.04), a paper
  with figures 6.77 → 4.23 (5.54); a PNG photo 1.83 → 1.64, 1.19 cold.

### JPEG transcoded

- A JPEG is recoded losslessly by Lepton (`lepton_jpeg`, the Rust port
  of Dropbox's, behind the default feature `jpeg`): its DCT
  coefficients under an arithmetic coder with a predictor across
  blocks, envelope `GLYDJPEG`, the identical JPEG back. Six photos,
  35.9 MB: 27.3 MB, 24% fewer bytes (a JPEG XL transcode: 20%), every
  one restored byte for byte; 5–7 MB/s in, 12–14 MB/s out, one core.
  A JPEG Lepton cannot take, or that does not shrink, stays as it is.

### Speed, same bytes

- The store's put, where its time went (`GLYD_STORE_TIMING=1` prints
  it): the fingerprint scan now runs on every core; a 4 GB cache of
  decoded objects serves the next base and every delta on a chain (its
  root decoded once, where each object over 1 GB fetched and decoded
  its base again); a 32 MB sample decides a delta before the whole is
  tried, and its alone size is the estimate (an hour of events shares
  half its fingerprints with the hour before and gained nothing from a
  full delta). Kernels put at 800–1,600 MB/s on this Mac (300–500
  before), events at 1,300–1,900; every stored byte identical.
- The max level on all cores at one core's ratio (`compress_max_stream`):
  units stay at the far matcher's 128 MB instead of shrinking to give
  every core one; a unit's far matches are found by one core and its
  blocks parsed in 16 MB stripes by all of them, tables seeded with
  the 2 MB before each stripe. Ten cores gave 7–9% more bytes than one
  on 512 MB files; now the same bytes at any core count. The parallel
  max level, record mode's plain fallback and the CLI's `--max` use it.

## v0.12.0 — 2026-09-22

### Gzip objects opened

- A gzip object's deflate streams are decoded to their plain text
  with what it takes to re-encode each bit for bit (`preflate-rs`,
  the crate's one dependency, behind the default feature `deflate`);
  the plain text then takes whatever was asked — a level, record
  mode, the cold level, a base — so a gzipped log costs what the log
  costs. Every encode entry point opens gzip input, every decode
  entry point closes it, and an opened object that would cost more
  than the gzip stays as it is. A gzip -6 NASA log: 20.7 MB → 8.2 MB
  at `--max -r`, 6.5 MB cold; a gzipped 512 MB kernel tree: 72.7 MB →
  56.1 MB at `--max`, 42.0 MB ultra; all back byte-exact. Through the
  store too.

### The store at a terabyte

- The gate run ([report](docs/benchmarks/store-gate-2026-09-22.md)):
  1,192 objects, 1.18 TB, put into S3 from one instance, 49.0 GB
  stored against zstd -3's 153.5 GB (3.13× fewer bytes, 24× against
  raw), every object read back byte-exact, the metadata directory
  rebuilt from the bucket and verified.
- The first attempt died of memory on 20 GB objects: put now maps its
  files instead of reading them, and the last object is kept as the
  likeliest next base only up to 1 GB.
- `glyd-store --version`.

## v0.11.2 — 2026-09-21

- A lost metadata directory is rebuilt from the objects: every
  object's index lines now ride beside it in the backend as
  `<id>.index` (a pack's carry its members'), and `--rebuild` (or
  `Store::rebuild_with`) remakes the index from those sidecars and the
  fingerprint table by reading every object back. Checked live: two
  kernel tarballs put to S3, the metadata directory deleted, rebuilt
  in 15 s, verified, read back byte-exact, and a third version then
  stored as a 1.8 MB delta against the second.
- `Backend::list` on both backends.

## v0.11.1 — 2026-09-21

- Multipart upload: objects over 64 MB go to S3 as 64 MB parts on up
  to 8 connections and one completion, so objects up to 640 GB store
  and fat links fill; a failed part or completion aborts the upload,
  leaving no parts behind to be billed. Checked live: a 17 MB object
  in four parts read back whole, undersized parts refused and aborted
  cleanly, the 201 MB kernel object in three parts byte-exact. On a
  home uplink the two-kernel put took 12.2 s against 11.6 s single-put:
  the link, not the client, is the limit there.

## v0.11.0 — 2026-09-21

### The store speaks S3 itself

- `S3Backend` replaces `S3Cli`: PUT, GET, HEAD, DELETE and
  ListObjectsV2 over HTTPS with Signature V4, no AWS CLI on the
  machine. Credentials from the environment, `~/.aws/credentials`
  (`AWS_PROFILE`) or the instance/container role, refreshed before
  they expire; the region from the environment, the profile, or the
  bucket's answer; `AWS_ENDPOINT_URL` for MinIO, R2, B2, Ceph and other
  S3-compatible services. Retries with backoff on 5xx and throttling.
  Two 1.5 GB kernel tarballs put through S3, one as a 3.4 MB delta,
  read back byte-exact and verified; the two-object put took the same
  11.6 s as through the CLI on this link. Single puts up to 5 GB;
  multipart upload is next.
- `--audit s3://...` lists and reads through the same client.
- The `glyd-store` crate takes its first dependencies for this:
  `ureq` (HTTPS through rustls), `sha2`, `hmac`. The `glyd` crate
  stays at zero.

## v0.10.2 — 2026-09-21

- The codec's licenses are now exactly zstd's: BSD 3-Clause (`LICENSE`)
  or GPL version 2 (`COPYING`), at the user's option, replacing
  Apache-2.0 OR GPL-2.0. Whatever may ship zstd may ship Glyd. The store
  stays under BUSL 1.1.
- `glyd.h` and the CLI banner no longer claim GPU or AVX-512 kernels;
  there are none (AVX2 and NEON only).

## v0.10.1 — 2026-09-21

- The codec is dual-licensed, Apache-2.0 or GPL-2.0 at the user's
  option (`LICENSE`, `LICENSE-GPL2`), the choice zstd offers, so GPLv2
  projects can carry it too. The store stays under BUSL 1.1.

## v0.10.0 — 2026-09-21

### Installing
- `brew install surya-koritala/glyd/glyd` (the tap
  github.com/surya-koritala/homebrew-glyd; installs both CLIs and
  `glyd.h`); every release carries prebuilt CLIs, libraries and Python
  wheels for Linux x86_64 / aarch64 and macOS arm64
  (`.github/workflows/release.yml`), and publishes to crates.io and
  PyPI once those tokens are repository secrets.

### The codec under Apache-2.0; the store its own crate under BUSL
- The `glyd` crate (the codec, every level and mode, base mode, packs,
  shape dictionaries, the C ABI, the `glyd` CLI) is licensed under the
  Apache License 2.0. The store moved to the `glyd-store` crate
  (`glyd-store/`, a workspace member) under the Business Source License
  1.1, with its own CLI (`glyd-store DIR --put ...`, `--audit`) and its
  C ABI in `libglyd_store`, which carries the codec's ABI too. The
  bindings load `libglyd_store` when present and `libglyd` otherwise
  (everything but `Store`). `Mapping` moved to `glyd::mmap`.

## v0.9.3 — 2026-09-21

### Adoption: bindings, the audit, the spec, lzbench, packaging
- C ABI, allocating form (`include/glyd.h`): `glyd_compress2` (every
  level, record mode, thread count), `glyd_decompress2` (any stream),
  `glyd_decompressed_len`, base mode, packs (`glyd_pack`,
  `glyd_unpack_object`, `glyd_pack_len`), the store (`glyd_store_*`:
  open with a directory or an S3 url, put, get, id_of, delete,
  compact, flush, rebase, verify, stats, set_level, count), `glyd_free`;
  `glyd_compress_fast` / `_turbo` in the buffer form; `glyd_version`
  reports the crate's version. Every decoder now reads a pack (its
  objects back to back).
- Python (`bindings/python`, ctypes, `pip install`) and Go
  (`bindings/go`, cgo) bindings over that ABI, each with a test that
  round-trips every level, record mode, base mode, packs and the store.
- `glyd --audit DIR|s3://bucket/prefix [--sample N]`: a sample of the
  objects (runs of consecutive names at eight places in the listing)
  through a store, against zstd -3 per object when the CLI is there,
  scaled to the listing with the yearly cost at S3 Standard's list
  price.
- `docs/spec.md`: the formats as a map for decoder writers (block
  framings and flags, every envelope's layout, the store's files).
- `contrib/lzbench`: Glyd in lzbench (`setup.sh <checkout>` builds it
  in; levels 1 default, 2 fast, 3 turbo, 4 max, 5 ultra); run and
  checked on an lzbench checkout.
- Packaging: crates.io metadata (the crate excludes the corpus, the
  benchmarks and the bindings), a Homebrew formula (`Formula/glyd.rb`).

## v0.9.2 — 2026-09-21

### Write speed, and the corpus rerun on AWS
- The suite (`scripts/bench_aws_suite.sh`, both machines, every decode
  checked; [docs/benchmarks/suite-2026-09-21.md](docs/benchmarks/suite-2026-09-21.md)):
  `--max` 3.939 over the 8.7 GB corpus (zstd -3 3.851) at 1,512 MB/s
  on 8 Graviton3 cores against zstd -3's 1,969 — 0.77× (0.61× in the
  previous report), decoding at 10,413 against 1,422; one core 233
  against 313 (0.74×), decode 1,571 against 1,425. Sapphire Rapids:
  1,122 against 1,513 at 8 threads, 266 against 377 on one. `--max -r`
  4.712 at 643 MB/s; `--ultra` 4.663 (zstd -19 4.664); `--ultra -r`
  5.224. In the S3 workflow `--max`'s reads now cost less CPU than
  zstd -3's on Graviton3 (9.7 against 10.7 s over the corpus), so it
  is the cheapest row at every read rate: a terabyte-year at 1 / 10 /
  100 reads a month $71.6 / $81.3 / $178 against zstd -3's $73.3 /
  $84.1 / $191; `--max -r` $61.7 / $82.4 / $289.
- The max level's finder tables are zstd -3's size (17/16 bits, 768
  KB) instead of 2 MB: on Graviton3 and Sapphire Rapids the 2 MB
  tables ran 10–25% slower for 0.5–1.5% fewer bytes (on an M1 the
  difference is 2–7%). One core, `benchmarks/max`: JSON events
  330/463 MB/s against zstd -3's 440/633 (Graviton3 / Sapphire Rapids),
  a table dump 301/420 against 331/462, a root filesystem 178/248
  against 115/294. Where the time goes: the parse 45–80%, the
  long-distance pass 7–45% (JSON), the entropy coder 10–18%.
- The CLI maps its input file instead of reading it first, and at
  `--max` and `--ultra` writes each unit as it finishes from a writer
  thread (`compress_stream`), so the output never sits whole in memory
  and the write overlaps the compressing: a 512 MB file at `--max` on
  ten cores 1,060 → 1,500 MB/s (a root filesystem) and 2,090 → 2,940
  (JSON events); in-process the level runs at 1,760 and 4,200.
- Measured and not taken: probing only the first repeat offset (no
  faster, 2–3% more bytes); 16/15 tables (5–12% faster again, 1–2%
  more bytes).

## v0.9.1 — 2026-09-21

### The store: S3, rebase, a second candidate
- `Backend` trait for the objects' bytes: `LocalBackend` (a directory)
  and `S3Cli` (an S3 bucket through the AWS CLI, `aws s3 cp` per
  object; the library carries no HTTP client); `Store::open_with`.
  CLI `--s3 s3://bucket/prefix`. Metadata (index, table) stays local.
  Round trip through a real bucket verified.
- `rebase(id)` / `--rebase ID`: an object stored alone again, one
  decode to read; a later index line for an id replaces the earlier.
- Two base candidates when the second scores at least half the first:
  both tried on the first 32 MB, the smaller delta wins the object. On
  the bucket: 1,313.6 -> 1,312.1 MB (the single candidate was already
  within a few percent of ideal), put 63 -> 70 s.
- The per-object fingerprint files are gone (the table on disk is the
  record).

## v0.9.0 — 2026-09-21

### The store, complete
- `delete(id)` (a tombstone in the index; the bytes stay while a live
  object's chain runs through them), `compact()` (removes what no live
  object needs: deleted objects off every live chain, packs whose
  members are all deleted; returns the bytes freed), `verify()` (every
  live object read back and checked), `id_of(name)`, `set_level`
  (`Max`, `Ultra`, `Cold` for objects stored alone and packs; deltas at
  the ultra level under `Ultra`). Deleted objects are never chosen as
  bases. CLI: `--find NAME`, `--delete ID`, `--compact`, `--verify`;
  `--ultra` / `--cold` with `--store` set the level.
- Put keeps the last large object in memory as the likeliest next base,
  and judges a delta against the object alone estimated from its first
  64 MB when that settles it either way (a version's delta is a few
  percent of the estimate, an unrelated object's about all of it),
  compressing the whole object alone only in between. The bucket put
  79 -> 63 s (620 MB/s end to end), same bytes; a version of the last
  object put runs at 900 MB/s. An estimate used alone, without the
  exact check in between, chose bases that were not worth it (the
  bucket 1,314 -> 1,890 MB) — measured, and not shipped.

## v0.8.1 — 2026-09-21

### The store at scale
- The fingerprint table lives on disk: an open-addressing hash table
  (12-byte slots, the fingerprint and the object id, linear probing, a
  fingerprint's last eight holders kept) mapped into memory through
  libc's `mmap`, at most half full, doubled in a fresh file when it
  fills. The store's memory no longer grows with what it holds: the
  39 GB bucket's table is 100 MB on disk and the put's memory is the
  object's own working set. Same bytes (1,314 MB, 29.9x), 480 MB/s end
  to end.
- Small objects (under 256 KB) go into packs of about 2 MB, one stored
  object each (`flush` writes the open pack; `Drop` flushes); `get`
  decodes the pack and slices, keeping the last pack decoded. 2,000
  GitHub events put one by one: 9.0x against zstd -3's 3.6x per event.
- Objects stored alone go through record mode where it pays
  (`compress_records_into_max`), so a log or a dump put into a store
  gets its columns.

## v0.8.0 — 2026-09-21

### The store: compression across objects
- `Store::open(dir)`, `put(name, data)`, `get(id)`, `entries`, `stats`;
  CLI `--store DIR --put FILE...`, `--get ID -o`, `--stats`. Each
  object's fingerprints (one sparse anchor in 4 KB) are looked up in
  the store's table (the last eight holders of each); the stored object
  sharing the most is its base, taken when the delta (`--base`) saves a
  fifth or more of the object alone; chains at most four long, the
  chain's root past that. A 39 GB bucket (six Ubuntu image builds, the
  fifteen Linux 6.10 releases, two months of three Wikipedia tables,
  twelve hours of GitHub events; `scripts/download_bucket.sh`): 1,334
  MB against zstd -3's 6,132 MB per object, 4.6x, put at 500 MB/s end
  to end and read back at 270 MB/s with the file written, every object
  byte-exact; by family 13.9x, 5.2x, 2.0x, 1.3x. The
  research pass that led here: experiments/research/README.md, H.

### Packs: small objects as one stream
- `compress_pack(objects, out, level)`, `decompress_pack`,
  `decompress_pack_object(pack, i)`, `pack_len`; CLI `--pack files...`,
  `--unpack dir`. Envelope `GLYDPACK`: the count, the lengths as
  zigzag deltas compressed at the max level, then the concatenation in
  record mode where it pays. 1 MB packs of 1 KB objects at `--max`:
  JSON events 42.4x (zstd -3 + dict per object 10.6x), NASA log 15.2x
  (5.3x), HDFS 16.2x (6.0x), CSV telemetry 8.9x (3.8x), taxi CSV 7.6x
  (3.9x); 40-130 MB/s to pack, an object read back in 0.5-1.7 ms.
  Record mode's pay decision now samples an eighth of a small input
  (at least 256 KB) instead of the whole of it.

### Shape dictionaries: record mode for small objects
- `ShapeDict::train(sample)`, `compress`, `decompress`, `to_bytes`,
  `from_bytes`; CLI `--shape-train`, `--shape`. The dictionary carries
  the shape (delimited, JSON lines, or a log's skeletons: punctuation
  between runs of letters and digits), the frames lines take with a
  column per hole, the columns' types (integers with a recency ring
  and deltas, times, decimals, dictionaries seeded with the sample's
  values by frequency, constants, text) and an LZ `Dict` trained on
  such objects' images. An object is a compact image: a byte per row,
  then every column's values, each column self-delimiting; unknown
  lines stay raw. 1–4 KB objects cut from real files: JSON lines 14.6×
  and 24.3× (zstd -3 + dict 10.6× and 13.0×), CSV telemetry 5.8× and
  7.9× (3.8×, 4.4×), HDFS log 6.7× and 9.7× (6.0×, 7.4×), NASA log
  5.1× and 7.3× (5.3×, 6.4×). 60–150 MB/s to code, 55–430 MB/s to
  decode, one core. The same objects packed into one record-mode
  stream cost 2–4× less than zstd + dict per object.

## v0.7.0 — 2026-09-20

### The cold level: context mixing
- `glyd --cold` (`compress_into_cold`, `compress_parallel_into_cold`,
  `compress_records_into_cold` with `-r`): every bit predicted from
  eleven contexts (byte orders 1-4, 6, 8; the word and the one before;
  the column and the byte above; the JSON key; the longest earlier
  match) through paq-style bit histories, mixed by two networks, two
  SSE stages, a binary arithmetic coder; 32 MB units coded from an
  empty model, in parallel, each with a checksum in the envelope
  (`GLYDCOLD`); every decoder reads it. 64 MB slices, two threads: JSON
  events 22.5x (zstd -19 14.6x, `--ultra` 15.9x, zpaq -m5 22.8x), NASA
  log `-r` 31.1x (15.7x, 26.4x, 31.7x), page_props dump `-r` 11.6x
  (6.2x, 8.6x, 11.1x), webster 7.1x (4.8x, 4.8x, 7.3x); the HDFS and
  Spark logs (128 MB, `-r`) 33.7x and 65.2x against zstd -19's 16.0x
  and 25.2x; 1.2-1.5 MB/s per core each way, 400 MB per thread. Record mode decides whether
  its transform pays at the max level whatever the level.

## v0.6.0 — 2026-09-20

### Base mode: content found wherever it moved, ultra at full speed
- Each unit's region of the base is chosen from a coarse map of the
  base (its sparse anchors, one per KB, found with the matcher's vector
  scan at 3-4 GB/s and sorted by the hash of the 32 bytes at each): the
  96 MB window holding the most of the unit's own anchors, or the base
  around the unit's position when its content is new. A version with
  48 MB inserted before the kernel tree costs 18.4 MB against 33.5 MB
  with the fixed window (zstd -3 --patch-from: 18.7 MB). The consecutive
  pairs are unchanged within 0.5%; `--max --base` runs at 720-1,700
  MB/s on ten M1 cores (was 860-2,020: the map's cost).
- `--ultra --base` inserted each unit's whole 96 MB region into the
  tree finder, which reaches 8 MB back: it now starts at the window's
  edge and runs at the plain `--ultra` speed, 3-11 MB/s on ten M1 cores
  against 1-4 before (the kernel pair 483 s, the new version alone at
  `--ultra` 555 s), the bytes the same.
- Chains measured (`scripts/download_chain.sh`, `scripts/bench_chain.sh`):
  the 15 Linux 6.10 point releases cost 228 MB each against the one
  before (zstd -3 --patch-from 260 MB; stored one by one 3.0-3.2 GB) or
  246 MB each against 6.10 (zstd 265 MB), a step 1.8 MB and the delta
  against a base 14 releases old 3.6 MB.

### Record mode: templates
- Logs whose lines vary in shape (application and system logs) take a
  fourth shape: each line's template (its text with a hole where every
  token holding a digit was) goes into a dictionary, and the tokens
  become typed columns keyed by template and slot; lines past the
  column budget stay raw. loghub 2.0, 128 MB of each, 10 cores: HDFS
  `--max -r` 22.0 against zstd -3's 10.5 and zstd -19's 16.0 (`--ultra
  -r` 27.5), Spark 47.0 against 14.5 and 25.2 (53.6), BGL 15.9 against
  11.0 and 22.4 (28.9), Android 17.9 against 12.9 and 23.0 (25.4);
  writes at 260-460 MB/s, reads at 1,200-1,400 MB/s. The levers were
  sized first (experiments/research/README.md): version chains, these
  log shapes, context mixing for cold data, float columns.

### Reads
- The record-mode rebuild decodes a column at a time into tables and
  assembles the rows by copy (reserved space, no length branch per
  value; integers through a digit-pair table; the minute's prefix of a
  time column kept and copied as a block; a ring for the recency list;
  16-byte padded dict8 entries; varints of up to three bytes from one
  load). One core, M1 Max: int columns 489 -> 810 MB/s, dictionaries
  344 -> 1,076, times 477 -> 1,815; the NASA log 509 -> 734, JSON
  lines 734 -> 1,117, the taxi CSV 260 -> 409. Record images are
  unchanged.
- The S3 workflow rerun on both AWS machines: the CLI's decompress
  CPU per 8.7 GB fell from 14.5 to 11.7 s on Graviton3 for `--max`
  (zstd -3: 10.9) and from 19.2 to 14.2 on Sapphire Rapids (zstd -3:
  10.4); `--ultra` 11.1 against zstd -19's 12.5. A terabyte-year at
  ten reads a month: `--max` $83.1, `--max -r` $83.6, zstd -3 $84.2.
- The reference codecs (zstd, LZ4, Snappy, LZAV) are dev-dependencies:
  the benchmarks link them, the library and CLI carry none.

## v0.5.0 — 2026-09-20

### Base mode
- `glyd --base old new` / `compress_with_base`: a new version of an
  object compressed against the old one, decodable with it (`glyd -d
  --base old`, `decompress_with_base`). Units of 32 MB are parsed with
  the base around their own position as history (32 MB of slack each
  way), the long-distance matcher reaching all of it; the decoder reads
  the base in place. Against zstd 1.5.7 `--patch-from` on the same
  machine, byte-exact: Wikipedia page-table dumps a month apart
  `--max` 1.79 MB at 863 MB/s (zstd -3 patch 3.84 MB at 409, zstd -19
  patch 1.30 MB at 2), `--ultra` 1.23 MB; Ubuntu cloud root filesystems
  16 days apart 5.31 MB at 2,018 MB/s (zstd -3 8.82 MB at 654, zstd -19
  5.61 MB at 39), `--ultra` 4.59 MB; Linux 6.10 -> 6.10.1 3.04 MB at
  1,980 MB/s (zstd -3 3.26 MB at 560, zstd -19 2.58 MB at 30), `--ultra`
  2.04 MB. Plain `--max` on those
  files: 33, 287 and 200 MB. Design notes in docs/design/format-v7.md;
  the measurement that led here in experiments/structure/README.md.
- A far match's cap of 130 bytes per sequence applied to the
  repeat-offset continuation as well; a repeat carries no offset bits,
  so the rest of the match is now one sequence. Plain `--max` gains 1%
  on JSON events.
- The long-distance matcher's table grows with the input (a slot per
  16 bytes, up to 2^25 entries).
- The CLI decodes a batch of units at a time into one reused buffer
  and writes as it goes (`decompress_stream`): the memory is a batch,
  not the file, and no output page is touched for the first time after
  the first batch; the library's parallel paths run on scoped worker
  threads (`set_threads`) instead of rayon.

## v0.4.0 — 2026-09-20

### Long-distance matching
- The max and ultra levels find repeats of 32 bytes or more up to 128 MB
  back (`src/ldm.rs`): one pass over the input before the parse, with
  content-defined anchors (one position in 16, found 16 at a time with
  NEON or AVX2) hashing the 32 bytes after them into a 16 MB table whose
  entries carry a hash check; matches are verified, extended both ways
  and handed to the parse, which takes one wherever it beats the local
  finder. Format v9 offsets grow to 27 bits (30 offset codes; v8 blocks
  keep 26, and a table with fewer symbols than its version allows still
  decodes). A far match is capped at 130 bytes per sequence so the
  decoder's one-load walk holds its extra bits, the rest following as a
  repeat-offset sequence. Parallel units grow to one per core, up to
  128 MB (the matcher's reach is the unit).
- The pass runs at 1.2-2 GB/s on one core. The max level gates it:
  after 4 MB and 16 MB (or half the input) it stops on data whose
  repeats are too few or too near to pay (media, Parquet, most SQL
  dumps), which keep 92-96% of their speed; the ultra level runs it
  whole. Where it stays on, the max level compresses at 63-85% of its
  former speed for 3-16% fewer bytes. One core, M1 Max, 128 MB of
  GitHub Archive JSON: `--max` 11.63 -> 9.73 MB at 577 MB/s (zstd -3
  12.90 MB at 895; `zstd -3 --long=27` 10.20 MB at 433), `--ultra`
  7.78 MB (zstd -19 8.96, `zstd -19 --long=27` 7.80); NASA access
  log 13.22 -> 11.92 MB at 435 MB/s (`zstd -3 --long=27` 13.63 MB at
  383); Silesia `--max` 3.259 -> 3.302 at 247 MB/s (290 before; zstd
  -3 3.205 at 335). The 8.7 GB corpus on 10 cores: `--max` 3.89 ->
  3.96 at 2,000 MB/s (2,400 before; zstd -3 3.85 at 4,000), JSON
  events 11.49 -> 13.26 (zstd -3 10.46); `--ultra` 4.65 (zstd -19
  4.66), JSON events 16.40 (zstd -19 15.07).

### Record mode
- `glyd -r` / `compress_records_with`: delimited lines, SQL dumps and
  JSON lines become typed column streams (integer, decimal and
  date-time deltas, dictionaries with recency ranks, text) before the
  level, in parallel 32 MB units, rebuilt byte for byte; other data is
  left as it is; input the transform does not pay on (API events with
  hashes and free text, binaries) takes the plain parallel path. JSON
  lines: a column per key path, typed values leave holes in a frame of
  the structure, keys and text. Telemetry as rows, 128 MB slices, 10
  cores: a cluster trace as CSV `--max -r` 12.6 against zstd -3's 4.5
  and zstd -19's 6.9, as JSON lines 54.6 against 15.7 and 28.7; daily
  weather 19.6 / 47.0 against 7.0 / 18.7 and 12.0 / 31.6; taxi trips
  exported to CSV 8.9 against 5.6 and 8.4 (`scripts/download_ext_corpus.sh`). Whole 8.7 GB corpus, 10 cores: `--ultra -r`
  5.20 against zstd -19's 4.66 (SQL dumps 1.43x smaller, access logs
  1.54x, JSON 1.09x through the plain level's matcher); `--max -r`
  4.75 at 1,100 MB/s. Design notes in
  docs/design/format-v7.md; the prototypes and measurements that led
  here in experiments/structure/.

### Small objects and dictionaries
- `Dict`: a prepared dictionary (trained content plus entropy tables)
  for small objects; `Dict::train` (cover selection as zstd's fastcover,
  scoring each distinct string once), `to_bytes`/`from_bytes`,
  `compress_with_dict`, `compress_with_dict_ultra`,
  `decompress_with_dict`. The object is parsed in place against the
  dictionary's own seeded tables; the decoder copies from the content
  and borrows the dictionary's built tables.
- Format v9: compact framing for blocks of at most 32 KB (one marker
  byte, varint lengths, a 5-10 byte sub-header, single-stream sections
  under 1,024 symbols, no padding on disk): 207 -> 21 bytes of framing
  on a 4 KB object. Every earlier format decodes unchanged
  (tests/format_compat.rs holds v0.2.0, v0.3.0 and v0.4.0 output).
- The ultra level with a dictionary prices its parse from the
  dictionary's tables.
- Per-object work cut: entropy tables and codes built once per
  dictionary, table costs from a lookup, buffers kept across calls,
  single-stream decode paths (three code chains at once, one-load
  batches).

GitHub Archive JSON objects, Apple M1 Max, one core, 110 KB
dictionaries trained on other objects (zstd's numbers without a
checksum; Glyd writes 4 bytes per object): 1 KB objects `--max` + Dict
ratio 4.75 (zstd -3 + dict 4.96), compress 245 MB/s (454), decode 920
MB/s (1,117); 4 KB 6.28 (6.42), 321 (588), 1,209 (1,440); 16 KB 7.65
(7.66), 394 (635), 1,674 (1,890). `--ultra` + Dict: 5.25 / 7.13 / 8.84
(zstd -19 + dict 5.47 / 7.41 / 8.95). Before this work the same objects
compressed to 2.15 / 3.87 / - with a window-only dictionary at 6 MB/s
and decoded at 170 MB/s.

### Verification and benchmarks
- `scripts/verify_roundtrip.sh` (every level and core mode through the
  CLI, byte-compared; corrupted copies rejected or decoded exactly),
  `scripts/download_bench_corpus.sh` (~9 GB of logs, JSON, SQL dumps
  and Parquet with a separate training set), `examples/bench_suite.rs`
  (Glyd against zstd -3, zstd -19 and LZ4 at a stated thread count,
  every decode checked, peak memory, small-object latencies),
  `scripts/s3_workflow.sh` (compress, upload, download, decompress,
  verify, monthly cost), `scripts/bench_aws_suite.sh` (all of it on
  Graviton3 and Sapphire Rapids), `scripts/report_suite.py`.
- The format-compatibility fixtures are now committed (they were
  ignored by the `*.glyd` rule; CI failed on every push since they were
  added).
- Results of the program on Graviton3 and Sapphire Rapids:
  docs/benchmarks/suite-2026-09.md (raw rows in benchmarks/suite/).

### Parallel paths
- The parallel compressors cut the input into units of at least 2 MB
  (v6 levels), 8 MB (`--max`) and 16 MB (`--ultra`) instead of 256 KB,
  growing with the input (two units per core, up to 64 MB): JSON events
  lost 4.7% to 8 MB units against the sequential ratio, 1% at 64 MB. Each unit is
  still a chain of its own (parallel decode, random access), and the
  ratio now stays within 0.5-0.7% of the sequential path; at 256 KB the
  CLI's multi-core default was giving up 3% (default level), 6.6%
  (`--max`) and 13% (`--ultra`). Multi-core decode of files with few
  units is correspondingly less parallel (Silesia on 10 cores: 31,900
  MB/s against 42,900).

### Ultra level
- Blocks are split where the parse's statistics change
  (`v7_ultra::split_points`); prices carry the parse's own prior at
  weight 2 and half a bit per literal. Silesia 3.925 → 3.946 (zstd -19:
  4.006), decode 2,150 → 2,090 MB/s.
- x86-64: the walk's and the tANS batch's per-stream state through
  memory: +6% max-level decode on Sapphire Rapids.

## v0.3.0 — 2026-09-19

### Format v8
Every level of the entropy-coded family now writes format v8; v7 (and
v6) files from earlier releases decode unchanged (tests/format_compat.rs
holds v0.2.0 output as fixtures).
- 8 MB window (26 offset codes).
- Section layout: 24-bit sub-stream sizes and one padding per section
  instead of per stream; tANS counts as width + mantissa; literal tables
  as nibbles with unused-symbol runs. Per-block overhead 950 → 421
  bytes.
- Length codes with direct codes to 15 (runs) and 34 (matches) and
  short buckets before the log2 ones.

Silesia, M1 Max, same run: `--ultra` 3.80 → **3.93** (zstd -16 3.83,
zstd -19 4.01), decode 2,150 MB/s (1.3× zstd -19); `--max` 3.22 →
**3.25** (zstd -3 3.20), decode 1,890 MB/s (1.27× zstd -3). Design
notes: docs/design/format-v7.md, "Format v8".

### Library
- The ultra finder sizes its tables to the input per call: 2 MB for a
  256 KB chunk, 64 MB for an input that fills the window.
- `bits::Stream` (a sub-stream with its own length and the bytes to the
  section's end) replaces bare slices in the decoders' signatures.

## v0.2.0 — 2026-09-19

### Levels
- **`--ultra` / `-19`** (format v7, same decoder): optimal parse on a
  binary-tree match finder, every position priced in the coder's own
  bits ([design](docs/design/ultra-parse.md)). Silesia ratio 3.80 vs
  `--max`'s 3.22; zstd -16 3.83, zstd -19 4.01 (3.91 inside Glyd's 2 MB
  window). Compresses at 4.8 MB/s; its output decodes at 2,190 MB/s,
  1.3× zstd -19's. `compress_into_ultra`, `compress_parallel_into_ultra`,
  `compress_with_dict_ultra`; C `glyd_compress_ultra`,
  `glyd_compress_ultra_parallel`.

### Platforms
- x86-64 `--max` decoder: an AVX2+BMI2 entry point and loop shapes for
  16 registers (stream-major entropy batches, the NEON copy structure,
  split walk tables). Sapphire Rapids decode 858 → 1,300 MB/s in the
  published run (zstd -3: 1,260 MB/s); default builds gain the same.
- Cross-platform benchmarks re-run; `ultra_bench` added to the script.

### CLI
- `-19` / `--ultra`; `--single-core` is now `-s` (`-1` was `--fast`).

### Fixed
- Nothing user-visible; see CHANGELOG-BENCH.md for the measurement trail.

## v0.1.0 — 2026-09-19

First public release.

### Levels
- **default** (format v6): LZAV-class parse, minimum match 7. Silesia
  ratio 2.19, decode 6,900 MB/s (1.6× liblz4), compress 340 MB/s.
- **`--fast` / `-1`**: LZ4-class finder, minimum match 5. Ratio 2.18,
  decode 4,900 MB/s, compress 550 MB/s.
- **`--turbo` / `-t`**: minimum match 10. Ratio 1.88, decode 9,200 MB/s
  (2.1× liblz4).
- **`--max` / `-9`** (format v7): 8-way interleaved Huffman literals,
  tANS-coded sequences with repeat offsets, 2 MB window, double-fast
  lazy parse; three-pass decoder. Ratio 3.22 vs zstd -3's 3.20, decode
  1,860 MB/s (1.3× zstd -3), compress 300 MB/s.

### Platforms
- aarch64 NEON decoders for v6 and v7; x86-64 AVX2 decoder for v6 and a
  portable scalar path everywhere else.
- Multi-core compression and decompression over independent 256 KB blocks.

### APIs
- Rust: `compress_into{,_fast,_turbo,_max}`, `compress_parallel_into*`,
  `decompress`, `decompress_into`, `decompress_parallel*`,
  `compress_with_dict` / `decompress_with_dict` (v7), `GlydReader` /
  `GlydWriter` (v6 streaming).
- C ABI (`include/glyd.h`, `libglyd`): `glyd_compress*`,
  `glyd_decompress*`, `glyd_max_compressed_len`, `glyd_version`.
- CLI `glyd`: compress/decompress, level flags, pipes, `-b` benchmark.

### Safety
- Every decoder fuzzed with 1,000,000 random mutations per run into
  exact-size buffers with sentinel guards; no per-call allocation in the
  decoder (1.5 MB thread-local scratch for v7).

### Known gaps
- `--max` compresses at ~90% of zstd -3's speed and decodes 1.3× (not 2×).
- `--max` beats zstd -3 on 3 of 5 extended-corpus files (loses 0.6% on
  repetitive JSON).
- No AVX2 v7 decoder yet (scalar on x86); streaming adapters are v6-only.
