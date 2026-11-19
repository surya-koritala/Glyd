# Beyond byte-level matching: what a structure-aware front-end is worth

Python prototypes (2026-09-19) that transform real data into typed
columns or templates, compress every stream with `zstd -19`, and compare
the total with `zstd -19` on the untouched file. Each script rebuilds
the input from its streams and reports whether the bytes match. zstd is
the second stage only to isolate the transform; the streams are what
Glyd's multi-stream coder would take.

| Data (slice) | Script | Transform | Size vs zstd -19 | Exact |
| :--- | :--- | :--- | ---: | :--- |
| NASA access log, 50 MB | logcols.py | typed columns by the log's format (host dictionary, timestamp deltas, request split, status, size) | 1.39x smaller | yes |
| NASA access log, 50 MB | logcols2.py | + path dictionary, move-to-front ids, size "same as last time for this path" | 2.27x smaller | invertible transforms, encode side measured |
| NASA / ClarkNet / pageviews, 50 MB | fieldcols.py | generic: split on the delimiter, one column per field, integers as deltas, few-valued columns as dictionary + MTF | 1.34x / 1.23x / 1.08x smaller | yes |
| NASA / pageviews | generic_templates.py | CLP-style: tokens with digits are variables, the rest a template | 1.10x smaller / 1.28x larger | yes |
| simplewiki pagelinks, 30 MB | sqlcols.py | INSERT tuples as typed columns | 1.52x smaller | yes (the slice cut a tuple in half; whole statements rebuild exactly) |
| enwiki page_props, 50 MB | sqlcols.py | same | 1.41x smaller | yes |
| GitHub Archive events, 50 MB | jsoncols.py | objects shredded to one column per key path, schemas dictionary-coded | 1.06x larger | 15,505 of 15,506 lines exact (one `` escape case) |
| GitHub Archive, one hour (814 MB) | `zstd -19 --long=27` | a 128 MB window instead of 8 MB | 1.23x smaller | n/a |

What it says:
- Record-shaped text (access logs, tab-separated dumps, SQL `INSERT`
  tuples) carries most of its redundancy across records in the same
  field, which byte-level matching only partly reaches. Typed columns
  win 23-127% over the best generic level, with the byte-exact rebuild
  kept by a frame stream.
- JSON events carry their redundancy inside a record (the same repo and
  actor strings in several fields) and across the whole hour; the first
  is already found by matching, the second needs a window far beyond
  8 MB (128 MB: 23%). Shredding into columns loses the within-record
  matches and costs more than it saves.
- A generic "template + variables" tokenizer is not the lever on this
  data; knowing the field structure is.

Run: `python3 experiments/structure/<script>.py <file>` (needs `zstd` on PATH).

## Versions of an object (2026-09-20): where the leap is

`examples/versions.rs` (and `versions.py`, the same in Python) on
pairs of consecutive versions of real objects, every rebuild byte-exact.
"Alone" is the new version compressed by itself with zstd -19; "chunk
dedup" stores only the content-defined chunks (gear hash, 8 KB mean)
that the old version lacks; "delta" is `zstd -19 --patch-from old`,
byte-level matching with the old version as the window.

| Old -> new | New version | Alone | Chunk dedup | Delta vs old | Delta is |
| :--- | ---: | ---: | ---: | ---: | ---: |
| simplewiki `page` table, dumps a month apart | 108 MB | 24.9 MB (4.3x) | 24.5 MB (98% of chunks touched) | **1.30 MB (83x)** | **19x smaller than alone** |
| Linux 6.10 -> 6.10.1 source tar | 1.50 GB | 149 MB (10x) | 79.6 MB (8 KB chunks) | **2.58 MB (580x)** | **58x smaller** |
| Ubuntu 24.04 cloud root filesystem, builds 16 days apart | 1.11 GB | 245 MB (4.5x) | 61.9 MB | **5.61 MB (197x)** | **44x smaller** |

Chunk-level dedup is what backup systems do and gains 1-4x here; the
byte-level delta against the previous version gains 19-58x. That is
the leap: not a better codec for one object, but compressing each
version against the last. It fits Glyd's parts (the long-distance
matcher's anchors index the old version; units decode in parallel;
record mode still applies to the new bytes) and needs a reference
match type whose source is the old version, an envelope naming it, and
`glyd --base old new` / `glyd -d --base old`.
