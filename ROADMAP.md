# Roadmap

Every item has a measured gate, taken against the reference codec on the
same machine and thread count, on named public data. Nothing ships on a
number that was not reproduced.

## Where it stands (v0.5.0, 2026-09-20)

The 8.7 GB real-data corpus on AWS Graviton3, 8 threads
([report](docs/benchmarks/suite-2026-09-20.md)): `--max` 3.96 (zstd -3
3.85) at 0.61× its write speed and 6× its read speed; `--ultra` 4.66
(zstd -19 4.66); `--max -r` 4.75 at 675 MB/s; `--ultra -r` 5.21.
Telemetry with `-r`: 2.5–3.5× fewer bytes than zstd -3; application and
system logs (templates, unreleased): 1.4–3.3× fewer than zstd -3 and
1.1–2.1× fewer than zstd -19. Versions with
`--base`: 1.1–2.1× fewer bytes than `zstd -3 --patch-from` at 1.8–3× its
speed; `--ultra --base` 5–21% fewer than zstd -19's patch. A
terabyte-year in S3 read monthly: `--max -r` $61.7, zstd -3 $73.3.

Measured floors, not to be retried: JSON API events and crawl indexes
are 20–65% hashes and random ids once compressed (typed columns gain
0.1%); Parquet is zstd inside; generic text and binaries sit on the same
entropy floor for every codec (zstd -19, xz, Glyd `--ultra` within 2%).
Details: [experiments/structure/README.md](experiments/structure/README.md).

## Next, in order of what moves the bill

1. **Write speed of `--max`** (0.58–0.66× zstd -3 on one server core):
   the finder's cache footprint and the matcher pass. Gate: 0.8× zstd -3
   with the corpus ratio kept.
2. **Small objects**: a single-pass decoder for compact blocks and a
   cheaper per-object encoder (zstd is 1.4–2× faster per object); a
   dictionary that carries a record schema, so `-r` ratios reach
   one-record objects. Gate: within 1.2× of zstd per object.
3. **The x86-64 decoder** (0.80× zstd -3 on one Sapphire Rapids core
   against 1.03× on Graviton3). Gate: 1.0× in the published run.
4. **Streaming for v9** in `GlydReader`/`GlydWriter` (v6 levels only
   today); the CLI already streams batches of units.

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

Later, if the CPU is acceptable where it applies: context-mixing literal
models for the free text inside logs and JSON (the 30% of a compressed
GitHub event that is prose), at 10–50× the CPU for 6–9%.
