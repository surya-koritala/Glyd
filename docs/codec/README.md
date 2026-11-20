# The Glyd file codec

Glyd's first product, still shipped and still compatible: a compressor for the files that fill object storage (logs and
dumps as typed columns, packs of many small objects, versions against the last one) with the fastest reads at every ratio.
It is separate from the inference engine (`glyd-engine`, the GPU library and the model commands), which is what the rest of
the repository is about ([README.md](../../README.md), [docs/architecture.md](../architecture.md)). Nothing here changes
what `glyd run` or `glyd serve` do, and nothing there changes a byte the codec writes.

## Where it lives

| What | Where | Name |
|---|---|---|
| The codec library (formats, levels, record mode, packs, shape dictionaries, a C ABI) | `src/` | crate `glyd` on crates.io (the name existed before the engine; it stays so that `cargo add glyd` and every existing user keep working) |
| The command | `src/bin/glyd.rs` | the one `glyd` program: `glyd compress FILE -o OUT`, `glyd decompress FILE -o OUT` (the words are `-c` and `-d`; `glyd FILE -o OUT.glyd` is the same as ever). The engine's commands (`run`, `serve`, `pull`, `fit`, ...) come first in `glyd --help` |
| The store (compresses across objects; a directory or S3) | `glyd-store/` | crate and command `glyd-store` ([README](../../glyd-store/README.md)) |
| Bindings | `bindings/` | Python and C |
| The formats | [spec.md](spec.md), [design/](design/) | a map for decoder writers; the v7 container, DEFLATE reconstruction, JPEG recoding, the ultra parse |

The crate names are decided: `glyd` (the codec) and `glyd-store` keep theirs, `glyd-engine` and `glyd-gpu` are the engine's,
and the engine is linked into the release `glyd` program (`glyd --version` says so), so one download has both. A build without
the `engine` feature (`cargo build -p glyd`) is the codec alone and has no GPU code.

## Use

```bash
glyd compress logs.tar -o logs.tar.glyd         # the default level; -1 fast, -t turbo, -9 max, -19 ultra, -C cold
glyd decompress logs.tar.glyd -o logs.tar
glyd --pack events/*.json -o events.glyd        # many small objects, any one read back alone
glyd --base dump-monday.sql dump-tuesday.sql -o tuesday.glyd
```

`glyd --help` has every option and example.

## Measurements and benchmarks

- [DETAILS.md](../../DETAILS.md): the measured results, each with its machine and a link to its log.
- [benchmarks/](benchmarks/README.md): the verification and benchmark program, and the tables of each run
  (`suite-*`, `landscape-*`, `store-gate-*`, `weights-*`).
- [engineering-notes.md](engineering-notes.md): the measurement history behind the numbers; `research/codec/CHANGELOG-BENCH.md` has every step.
- Glyd is in [TurboBench](https://github.com/powturbo/TurboBench), which benchmarks it beside other compressors on the reader's own files ([#117](https://github.com/surya-koritala/Glyd/issues/117)).

## Compatibility

A file written by any release decodes with every later one; the on-disk formats are not changed by anything in the engine's work.
The tests that hold this, all run by `cargo test -p glyd` in CI:

- `tests/format_compat.rs`: files written by v0.2.0 (formats v6, v7), v0.3.0 (v8) and v0.4.0 (v9), decoded; a fixture is never regenerated.
- `tests/roundtrip.rs`, `v7_roundtrip.rs`, `v7_codecs.rs`, `ultra.rs`, `cold.rs`, `pack.rs`, `records.rs`, `base.rs`, `streaming_test.rs`, `exact_dst.rs`: every level and mode round trips.
- `tests/fuzz_safety.rs`, `tests/v7_fuzz.rs`: a million mutations of every decoder into guarded buffers.
- `tests/regression_floors.rs`: the ratio floors.
- `tests/cli_codec.rs`: the command words write the bytes the flags write.
