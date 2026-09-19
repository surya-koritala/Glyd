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
