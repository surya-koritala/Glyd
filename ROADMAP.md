# Roadmap

Every item has a measured gate, taken against the reference codec on the
same machine and thread count, on named public data. Nothing ships on a
number that was not reproduced.

## Where it stands (v0.5.0, 2026-09-20)

The 8.7 GB real-data corpus on AWS Graviton3, 8 threads
([report](docs/benchmarks/suite-2026-09-20.md)): `--max` 3.96 (zstd -3
3.85) at 0.61× its write speed and 6× its read speed; `--ultra` 4.66
(zstd -19 4.66); `--max -r` 4.75 at 675 MB/s; `--ultra -r` 5.21.
Telemetry with `-r`: 2.5–3.5× fewer bytes than zstd -3. Versions with
`--base`: 1.1–2.1× fewer bytes than `zstd -3 --patch-from` at 2–3.5× its
speed; `--ultra --base` 5–21% fewer than zstd -19's patch. A
terabyte-year in S3 read monthly: `--max -r` $61.7, zstd -3 $73.3.

Measured floors, not to be retried: JSON API events and crawl indexes
are 20–65% hashes and random ids once compressed (typed columns gain
0.1%); Parquet is zstd inside; generic text and binaries sit on the same
entropy floor for every codec (zstd -19, xz, Glyd `--ultra` within 2%).
Details: [experiments/structure/README.md](experiments/structure/README.md).

## Next, in order of what moves the bill

1. **Record-mode reads.** The column rebuild runs at 24–35 ns per value
   and makes `-r` reads cost 2–3× zstd's CPU, which is what decides the
   bill at high read rates. Reserved-capacity writes, a digit-pair
   integer formatter, the dictionary decoder taking its rank without a
   second search, an unchecked varint fast path. Gate: rebuild at
   1.5–2 GB/s per core; `--max -r` the cheapest S3 row at a hundred
   CPU-billed reads a month.
2. **Base mode, the rest of the leap.** A base index built once and
   shared by the units, so `--ultra --base` runs at the plain ultra
   speed instead of 1–4 MB/s; a coarse map of the base so content that
   moved farther than 32 MB is still matched; chains of versions with a
   measured answer to how long a chain before re-basing. Gate: kernel
   pair under zstd -19's patch size at `--max` speed.
3. **Write speed of `--max`** (0.58–0.66× zstd -3 on one server core):
   the finder's cache footprint and the matcher pass. Gate: 0.8× zstd -3
   with the corpus ratio kept.
4. **Small objects**: a single-pass decoder for compact blocks and a
   cheaper per-object encoder (zstd is 1.4–2× faster per object); a
   dictionary that carries a record schema, so `-r` ratios reach
   one-record objects. Gate: within 1.2× of zstd per object.
5. **The x86-64 decoder** (0.80× zstd -3 on one Sapphire Rapids core
   against 1.03× on Graviton3). Gate: 1.0× in the published run.
6. **Streaming for v9** in `GlydReader`/`GlydWriter` (v6 levels only
   today); the CLI already streams batches of units.

Later, if the CPU is acceptable where it applies: context-mixing literal
models for the free text inside logs and JSON (the 30% of a compressed
GitHub event that is prose), at 10–50× the CPU for 6–9%.
